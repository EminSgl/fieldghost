"""
Stage 4 -- the signature pass: build tracks (like track.py), then go back and
try to *resolve* every ghost box stitching had to invent for an occlusion gap.

For each ghost (ESTIMATED, interpolated) box in a track:
  1. Crop a tightly zoomed region of the original frame around the estimate.
  2. Re-run the detector on just that crop, at a lower confidence threshold
     than the main detection pass -- a small, partially-occluded, or
     motion-blurred subject that the full-frame/tiled pass missed at normal
     confidence is often findable once you're looking at nothing else.
  3. Accept the local detection as a real ("measured", occluded=False) box
     only if it plausibly IS the estimate: close enough by IoU or centre
     distance (both relative to the estimate's own height, so it scales with
     subject size), and a similar size (not a knee-high fragment or a
     different, larger subject that wandered into the crop). Reject it if a
     box that's already accounted for (measured, or already accepted this
     frame) covers most of it -- that's a duplicate, not a recovery.

Every ghost that could NOT be resolved this way stays a ghost -- CVAT's
`occluded=True` on that shape -- so a human reviewer can find exactly the
frames that still need a manual look. That list, plus frames where the raw
box count jumps sharply against its neighbours (often a sign of a missed or
spurious detection), is written to `<out_prefix>_checklist.json`.

This step also re-runs the stitcher with a *tighter* MAX_GAP than track.py's
default: once recovery is available, you generally want to stitch only short
real gaps and leave longer ones as separate tracks rather than bridge them
with a long run of un-recoverable ghosts.

Usage:
    python refine.py <clip.mp4> <detections.json.part> <out_prefix> \\
        --model model.onnx --label-id 8 --frame-step 5 [options]
Writes <out_prefix>_tracks.json and <out_prefix>_checklist.json.
"""
import argparse
import json

import numpy as np
import cv2

import track_lib as T
import detect as D


def crop_around(frame, box, min_side=160, zoom=2.5):
    h, w = frame.shape[:2]
    bh, bw = box[3] - box[1], box[2] - box[0]
    side = max(zoom * max(bh, bw), min_side)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return x1, y1, x2, y2


def find_in_crop(frame, est, min_side, zoom):
    x1, y1, x2, y2 = crop_around(frame, est, min_side, zoom)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    dets = D.infer_tile(frame[y1:y2, x1:x2])
    return [(a + x1, b + y1, c + x1, d + y1, s) for a, b, c, d, s in dets]


def accept(det, est, min_iou, max_dist, size_ratio):
    b = np.array(det[:4])
    e = np.array(est)
    iou = T.iou_matrix([e], [b])[0, 0]
    eh, dh = e[3] - e[1], b[3] - b[1]
    dist = np.linalg.norm((e[:2] + e[2:]) / 2 - (b[:2] + b[2:]) / 2) / max(eh, 8)
    ratio = dh / max(eh, 1e-6)
    return (iou >= min_iou or dist <= max_dist) and size_ratio[0] <= ratio <= size_ratio[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("detections_part")
    ap.add_argument("out_prefix")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label-id", type=int, required=True)
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--attrs", default="[]", help="default attrs for real (non-ghost) tracks, as a JSON list")
    ap.add_argument("--recover-threshold", type=float, default=0.10,
                     help="confidence threshold used only for the zoomed-crop recovery re-detection")
    ap.add_argument("--min-iou", type=float, default=0.30)
    ap.add_argument("--max-dist", type=float, default=0.6)
    ap.add_argument("--size-ratio", type=float, nargs=2, default=(0.55, 1.8))
    ap.add_argument("--dup-iou", type=float, default=0.6)
    ap.add_argument("--max-gap", type=int, default=6,
                     help="max sampled-frame gap the stitcher may bridge in this pass (default: 6 ~= 1s at frame-step 5, 30fps)")
    ap.add_argument("--jump-threshold", type=int, default=5,
                     help="box-count jump vs the median of +-5 neighbouring frames that flags a frame for the checklist")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    T.MAX_GAP = args.max_gap
    D.CLASS_ID = args.class_id
    D.load_model(args.model)

    det = [json.loads(l) for l in open(args.detections_part, encoding="utf-8") if l.strip()]
    if args.max_frames:
        det = det[:args.max_frames]
    n = len(det)
    per_frame_boxes = [[s["points"] for s in f] for f in det]

    tracks, boxes = T.build_tracks(per_frame_boxes, args.clip, args.frame_step)

    # ---- recovery of ghost (estimated) boxes ----
    want = {}
    for ti, tr in enumerate(tracks):
        for si, (t, b, ghost) in enumerate(tr):
            if ghost:
                want.setdefault(t, []).append((ti, si))
    measured = {}
    for tr in tracks:
        for t, b, ghost in tr:
            if not ghost:
                measured.setdefault(t, []).append(np.array(b))

    cap = cv2.VideoCapture(args.clip)
    D.THRESHOLD = args.recover_threshold
    recovered = tried = 0
    for t in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t * args.frame_step)
        ok, fr = cap.read()
        if not ok:
            continue
        used = []
        for ti, si in want[t]:
            _, est, _ = tracks[ti][si]
            tried += 1
            cands = [d for d in find_in_crop(fr, est, 160, 2.5)
                     if accept(d, est, args.min_iou, args.max_dist, args.size_ratio)]
            best = None
            for d in sorted(cands, key=lambda d: -d[4]):
                b = np.array(d[:4])
                if any(T.iou_matrix([b], [m])[0, 0] > args.dup_iou for m in measured.get(t, []) + used):
                    continue
                best = b
                break
            if best is not None:
                tracks[ti][si] = (t, best, False)
                used.append(best)
                recovered += 1
    cap.release()

    # ---- checklist for human QA ----
    counts = [len(b) for b in boxes]
    jumps = []
    for t in range(n):
        lo, hi = max(0, t - 5), min(n, t + 6)
        nb = counts[lo:t] + counts[t + 1:hi]
        if nb and abs(counts[t] - int(np.median(nb))) >= args.jump_threshold:
            jumps.append({"sampled_frame": t, "reason": f"box count {counts[t]} vs neighbours ~{int(np.median(nb))}"})
    still_ghost = {}
    for tid, tr in enumerate(tracks):
        for t, b, ghost in tr:
            if ghost:
                still_ghost.setdefault(t, []).append(tid)
    gaps = [{"sampled_frame": t, "reason": f"unresolved ghost box for track(s) {ids[:6]}"}
            for t, ids in sorted(still_ghost.items())]
    json.dump({"box_count_jumps": jumps, "unresolved_ghost_frames": gaps},
              open(args.out_prefix + "_checklist.json", "w"), indent=1)

    # tracks shorter than 3 boxes are too unreliable as an ID -> emit as plain shapes instead
    attrs = json.loads(args.attrs)
    long_tr = [tr for tr in tracks if len(tr) >= 3]
    single = [s for tr in tracks if len(tr) < 3 for s in tr]
    shapes = [{"type": "rectangle", "occluded": bool(ghost), "z_order": 0, "rotation": 0, "outside": False,
               "frame": int(t), "label_id": args.label_id, "group": 0, "source": "auto",
               "points": [round(float(v), 1) for v in b], "attributes": attrs}
              for t, b, ghost in single]
    json.dump({"version": 0, "tags": [], "shapes": shapes,
               "tracks": T.to_cvat(long_tr, n, args.label_id, attrs)},
              open(args.out_prefix + "_tracks.json", "w"))

    print(f"frames={n} tracks={len(long_tr)} single_boxes={len(single)} "
          f"ghosts_tried={tried} recovered={recovered} "
          f"still_ghost={sum(len(v) for v in still_ghost.values())} "
          f"count_jumps={len(jumps)} unresolved_ghost_frames={len(gaps)}")


if __name__ == "__main__":
    main()
