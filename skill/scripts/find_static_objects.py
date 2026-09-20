"""
Stage 6a: find CANDIDATE static-object false positives after upload.

Read this before you run it: this heuristic is not a classifier you can
trust unattended. Static field furniture (a corner flag, a touchline pole,
an advertising board on a post) gets flagged for the same reason a real,
distant, mostly-still subject does -- both produce a small, thin box that
barely moves for a long stretch. In testing, roughly a third of "confident"
candidates from this exact heuristic turned out to be real, distant,
temporarily-stationary subjects (players warming up, standing during a
stoppage). This script only PRINTS candidates with a sample frame number for
each. It never deletes anything. The workflow is:

  1. Run this script, get a short list of candidates per job.
  2. Open each candidate's sample frame in the CVAT UI (use whatever browser
     automation you have -- Playwright, Puppeteer, a `/browse`-style skill,
     or just tell the human where to look) at a decent zoom and LOOK at it.
  3. Only for the ones you visually confirm are static furniture, record a
     tight pixel bounding box around their position (in the source video's
     coordinate space) in an exclusion-boxes JSON file.
  4. Re-run merge_upload.py with --exclude-boxes pointing at that file. It
     rebuilds the merge from scratch (the untouched per-clip track files),
     so a box you got wrong costs you nothing but a second run.

The signal this script actually looks for: bin every measured (non-ghost)
shape's box centre into a small pixel grid, per job. A cell that accumulates
detections from many DISTINCT tracks (not one long track -- many short,
separate ones) at nearly the same spot over a long span, with a small/thin
average box, is the fingerprint of one physical object being repeatedly
re-detected and re-tracked from scratch every time occlusion or a confidence
dip breaks the track -- which is exactly what happens to something that
never moves and therefore never gets a stitchable trajectory. A real subject
passing through the same spot many times over 80 minutes of a match (e.g. a
kickoff restart point) will usually show up as far fewer, longer tracks, but
this is a tendency, not a guarantee -- hence step 2 above being mandatory,
not optional.

Usage:
    python find_static_objects.py ./merged/job_102.json ./merged/job_103.json ... \\
        --cell 10 --min-tracks 15 --max-width 20
"""
import argparse
import json
from collections import defaultdict


def scan_job(path, cell, min_tracks, max_width, min_frames):
    d = json.load(open(path, encoding="utf-8"))
    cells = defaultdict(lambda: {"tracks": set(), "widths": [], "sample": None})
    for i, t in enumerate(d["tracks"]):
        for s in t["shapes"]:
            if s["outside"]:
                continue
            cx = (s["points"][0] + s["points"][2]) / 2
            cy = (s["points"][1] + s["points"][3]) / 2
            key = (round(cx / cell) * cell, round(cy / cell) * cell)
            c = cells[key]
            c["tracks"].add(i)
            c["widths"].append(s["points"][2] - s["points"][0])
            if c["sample"] is None:
                c["sample"] = (s["frame"], [round(v, 1) for v in s["points"]])

    hits = []
    for (x, y), c in cells.items():
        avg_w = sum(c["widths"]) / len(c["widths"])
        if len(c["tracks"]) >= min_tracks and avg_w <= max_width:
            hits.append({"x": x, "y": y, "n_tracks": len(c["tracks"]), "avg_width": round(avg_w, 1),
                         "sample_frame": c["sample"][0], "sample_box": c["sample"][1]})
    hits.sort(key=lambda h: -h["n_tracks"])

    # de-duplicate neighbouring grid cells belonging to the same physical object
    picked = []
    for h in hits:
        if any(abs(h["x"] - p["x"]) < cell * 4 and abs(h["y"] - p["y"]) < cell * 4 for p in picked):
            continue
        picked.append(h)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_files", nargs="+", help="job_<id>.json files from merge_upload.py's --out-dir")
    ap.add_argument("--cell", type=int, default=10, help="grid cell size in pixels")
    ap.add_argument("--min-tracks", type=int, default=15,
                     help="minimum distinct tracks re-appearing in the same cell to flag it")
    ap.add_argument("--max-width", type=float, default=20,
                     help="only flag cells whose average box width is at or below this (thin/small objects)")
    ap.add_argument("--min-frames", type=int, default=0, help="reserved for future use")
    args = ap.parse_args()

    print(
        "NOTE: these are CANDIDATES only. Visually confirm each one in the CVAT UI "
        "before excluding it -- see this script's docstring.\n"
    )
    any_hits = False
    for path in args.job_files:
        hits = scan_job(path, args.cell, args.min_tracks, args.max_width, args.min_frames)
        print(f"=== {path}: {len(hits)} candidate(s) ===")
        for h in hits:
            any_hits = True
            print(f"  pos=({h['x']},{h['y']}) n_tracks={h['n_tracks']} avg_width={h['avg_width']} "
                  f"-> check frame {h['sample_frame']}, box {h['sample_box']}")
    if not any_hits:
        print("no candidates found with the current thresholds.")


if __name__ == "__main__":
    main()
