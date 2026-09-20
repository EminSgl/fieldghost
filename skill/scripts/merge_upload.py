"""
Stage 5: merge every clip's refined tracks into the CVAT task's global frame
numbering, split at each job's frame boundaries, and (optionally) upload.

This script is the ONLY place that is allowed to touch what ends up on the
CVAT server, and it always rebuilds its output from the untouched per-clip
`*_tracks.json` files (the ones refine.py wrote) rather than editing a
previous merge in place. That matters for exactly one reason, learned the
hard way: a static-object filter (see find_static_objects.py) is a heuristic
and heuristics are sometimes wrong at scale. If `--exclude-boxes` removes
too much or too little, you fix the box list and rerun this script -- you
are never patching a already-damaged file, because there isn't one; the
per-clip track files are the only source of truth and this script is a pure
function of them plus your config.

Frame numbering: your CVAT task samples every `frame_step`'th native video
frame (its `frame_filter` setting). A clip's own frame N in its detections/
tracks file is native frame `N * frame_step` *within that clip*. To place it
on the task's timeline you need `clip.source_start_frame`, the native frame
number in the FULL source video where this clip begins -- e.g. if you split
one long video into 15-minute clips at 30fps, clip 2 might start at native
frame 27000. Task frame = (clip.source_start_frame + N * frame_step) / frame_step
= clip.source_start_frame // frame_step + N.

Overlapping clips: if your clips overlap in time (common so nothing gets cut
off at a clip boundary mid-action), each clip should only contribute the
half of the overlap closest to it, or you'll get duplicate boxes for the
same subject in the overlap region. Set an explicit `window: [lo, hi]`
(task-frame, half-open) per clip in the config for precise control; if you
leave it out, this script defaults to the midpoint between each clip's start
and its neighbours' starts, which is a reasonable starting point but you
should spot-check the seams in the CVAT UI and adjust by hand if a subject
visibly jumps or duplicates at a boundary -- exactly like tuning a crossfade.

Usage:
    python merge_upload.py config.yaml --clips-dir ./clips_out --out-dir ./merged
    python merge_upload.py config.yaml --clips-dir ./clips_out --out-dir ./merged --upload --token TOKEN
    python merge_upload.py config.yaml --clips-dir ./clips_out --out-dir ./merged \\
        --upload --token TOKEN --exclude-boxes confirmed_static_objects.json

`--exclude-boxes` takes a JSON file: a list of {"x_lo":..,"x_hi":..,"y_lo":..,"y_hi":..}
pixel boxes (in the source video's coordinate space). Any track whose
AVERAGE shape-centre falls inside one of these boxes is dropped before
upload. Only put a box in this file after visually confirming it in the
CVAT UI at the sample frame find_static_objects.py reports -- see that
script's docstring and the skill's SKILL.md for why blind, unconfirmed
removal is a bad idea.
"""
import argparse
import glob
import json
import os
import sys

import yaml

from cvat_client import login, put_annotations, TokenExpired

VIS_SPEC_ID_KEY = "spec_id"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def ghost_attr(shape, ghost_cfg):
    if not ghost_cfg:
        return []
    value = ghost_cfg["unresolved_value"] if shape["occluded"] else ghost_cfg["resolved_value"]
    return [{"spec_id": ghost_cfg["spec_id"], "value": value}]


def fix_track_attrs(track, ghost_cfg):
    if not ghost_cfg:
        return track
    track["attributes"] = [a for a in track["attributes"] if a["spec_id"] != ghost_cfg["spec_id"]]
    track["shapes"] = [{**s, "attributes": [a for a in s["attributes"] if a["spec_id"] != ghost_cfg["spec_id"]]
                         + ghost_attr(s, ghost_cfg)} for s in track["shapes"]]
    return track


def default_windows(clips, frame_step):
    """Midpoint-between-neighbours default when a clip has no explicit `window`."""
    starts = [c["source_start_frame"] // frame_step for c in clips]
    windows = []
    for i, c in enumerate(clips):
        if "window" in c:
            windows.append(tuple(c["window"]))
            continue
        lo = starts[i] if i == 0 else (starts[i - 1] + starts[i]) // 2
        hi = starts[i] + 10**9 if i == len(clips) - 1 else (starts[i] + starts[i + 1]) // 2
        windows.append((lo, hi))
    return windows


def clip_to_task_frames(clip_cfg, tracks_path, window, frame_step, ghost_cfg):
    off = clip_cfg["source_start_frame"] // frame_step
    lo, hi = window
    d = json.load(open(tracks_path, encoding="utf-8"))
    shapes = []
    for s in d["shapes"]:
        k = s["frame"] + off
        if lo <= k < hi:
            shapes.append({**s, "frame": k,
                            "attributes": [a for a in s["attributes"] if not ghost_cfg or a["spec_id"] != ghost_cfg["spec_id"]]
                            + ghost_attr(s, ghost_cfg)})
    tracks = []
    for tr in d["tracks"]:
        sh = [{**s, "frame": s["frame"] + off} for s in tr["shapes"]]
        keep = [s for s in sh if lo <= s["frame"] < hi]
        if not keep:
            continue
        tracks.append(fix_track_attrs({**tr, "frame": keep[0]["frame"], "shapes": keep}, ghost_cfg))
    return shapes, tracks


def split_for_job(shapes, tracks, start, stop):
    js = [s for s in shapes if start <= s["frame"] <= stop]
    jt = []
    for tr in tracks:
        sh = [s for s in tr["shapes"] if start <= s["frame"] <= stop]
        if not sh:
            continue
        if sh[-1]["frame"] < stop and not sh[-1]["outside"]:
            nxt = [s for s in tr["shapes"] if s["frame"] == sh[-1]["frame"] + 1]
            if not nxt:
                sh.append({**sh[-1], "frame": sh[-1]["frame"] + 1, "outside": True})
        jt.append({**tr, "frame": sh[0]["frame"], "shapes": sh})
    return js, jt


def track_center(track):
    meas = [s for s in track["shapes"] if not s["outside"]]
    if not meas:
        return None
    cx = sum((s["points"][0] + s["points"][2]) / 2 for s in meas) / len(meas)
    cy = sum((s["points"][1] + s["points"][3]) / 2 for s in meas) / len(meas)
    return cx, cy


def apply_exclusions(tracks, boxes):
    if not boxes:
        return tracks, 0
    kept, removed = [], 0
    for tr in tracks:
        c = track_center(tr)
        if c is not None:
            cx, cy = c
            if any(b["x_lo"] <= cx <= b["x_hi"] and b["y_lo"] <= cy <= b["y_hi"] for b in boxes):
                removed += 1
                continue
        kept.append(tr)
    return kept, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--clips-dir", required=True, help="directory containing <clip.name>_tracks.json for every clip")
    ap.add_argument("--out-dir", required=True, help="where to write job_<id>.json")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--token", help="CVAT auth token. If omitted and --upload is set, use --username/--password to log in.")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--exclude-boxes", help="JSON file of confirmed static-object exclusion boxes (see docstring)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cvat = cfg["cvat"]
    frame_step = cvat["frame_step"]
    ghost_cfg = cvat.get("ghost_attribute")
    clips = cfg["clips"]
    jobs = cfg["jobs"]

    exclude_boxes = json.load(open(args.exclude_boxes)) if args.exclude_boxes else []

    os.makedirs(args.out_dir, exist_ok=True)
    windows = default_windows(clips, frame_step)

    all_shapes, all_tracks = [], []
    for clip_cfg, window in zip(clips, windows):
        tracks_path = os.path.join(args.clips_dir, f"{clip_cfg['name']}_tracks.json")
        if not os.path.exists(tracks_path):
            print(f"clip {clip_cfg['name']}: {tracks_path} not found, skipping", file=sys.stderr)
            continue
        s, t = clip_to_task_frames(clip_cfg, tracks_path, window, frame_step, ghost_cfg)
        print(f"clip {clip_cfg['name']}: window {window} shapes={len(s)} tracks={len(t)}")
        all_shapes += s
        all_tracks += t

    all_tracks, removed = apply_exclusions(all_tracks, exclude_boxes)
    if exclude_boxes:
        print(f"exclusion boxes: removed {removed} confirmed static-object tracks")

    token = args.token
    if args.upload and not token:
        if not (args.username and args.password):
            print("no --token and no --username/--password given; cannot upload", file=sys.stderr)
            sys.exit(1)
        token = login(cvat["host"], args.username, args.password)

    for job in jobs:
        js, jt = split_for_job(all_shapes, all_tracks, job["start"], job["end"])
        payload = {"version": 0, "tags": [], "shapes": js, "tracks": jt}
        out_path = os.path.join(args.out_dir, f"job_{job['job_id']}.json")
        json.dump(payload, open(out_path, "w"))
        n_track_shapes = sum(len(t["shapes"]) for t in jt)
        print(f"job {job['job_id']} [{job['start']},{job['end']}]: "
              f"shapes={len(js)} tracks={len(jt)} track_shapes={n_track_shapes} -> {out_path}")
        if args.upload and (js or jt):
            try:
                status = put_annotations(cvat["host"], token, job["job_id"], payload)
                print(f"  PUT {status}")
            except TokenExpired:
                print(
                    f"  PUT failed: token expired/invalid for job {job['job_id']}. "
                    f"Re-run with --username/--password to mint a fresh token, or pass a new --token.",
                    file=sys.stderr,
                )
                sys.exit(2)


if __name__ == "__main__":
    main()
