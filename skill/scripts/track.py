"""
Stage-3 standalone tracker: turn one clip's raw detections into CVAT tracks,
with NO gap-recovery (see refine.py for the full pipeline including recovery
-- that's what you want for a real run). This script exists for a quick look
at raw tracking quality, or for debugging track_lib.py in isolation.

Usage:
    python track.py <clip.mp4> <detections.json.part> <out_tracks.json> \\
        --label-id 8 --frame-step 5 [--max-frames N]
"""
import argparse
import json

import track_lib as T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("detections_part")
    ap.add_argument("out")
    ap.add_argument("--label-id", type=int, required=True)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--attrs", default="[]")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    det = [json.loads(l) for l in open(args.detections_part, encoding="utf-8") if l.strip()]
    if args.max_frames:
        det = det[:args.max_frames]
    per_frame_boxes = [[s["points"] for s in f] for f in det]

    tracks, boxes = T.build_tracks(per_frame_boxes, args.clip, args.frame_step)

    attrs = json.loads(args.attrs)
    cvat_tracks = T.to_cvat(tracks, len(boxes), args.label_id, attrs)
    json.dump({"version": 0, "tags": [], "shapes": [], "tracks": cvat_tracks},
              open(args.out, "w", encoding="utf-8"))

    lens = sorted((len(t) for t in tracks), reverse=True)
    ghosts = sum(1 for t in tracks for s in t if s[2])
    print(f"frames={len(boxes)} detections={sum(map(len, boxes))} tracks={len(tracks)} "
          f"ghost_boxes={ghosts} longest={lens[:5]} tracks>=30f={sum(1 for l in lens if l >= 30)}")


if __name__ == "__main__":
    main()
