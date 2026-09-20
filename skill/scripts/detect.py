"""
GPU detection pass: run an ONNX object detector over one video clip, sampling
frames at a fixed step, and write raw per-frame boxes as CVAT-shaped JSON.

Checkpointed by design: every sampled frame is appended as one JSON line to
`<out>.part` and fsync'd immediately. If the process crashes, the machine
reboots, or you just Ctrl-C it, rerunning the exact same command resumes from
the last complete line instead of starting over. `<out>` (the non-.part file)
is only written once, at the very end, as a convenience full snapshot; the
tracker and refine pass both read `<out>.part` directly, so the run is fully
resumable at any point without it.

The model is expected to be a YOLO-style ONNX export with NMS baked in,
producing rows of (batch_idx, x1, y1, x2, y2, class_id, score) in the
letterboxed 640x640 input space -- this is what `yolov7-nms-<size>.onnx`
exports and most YOLOv5/v7/v8 "end2end" ONNX exports look like. If your
model's output layout differs, adjust `infer_tile` accordingly; everything
else (checkpointing, tiling, resume logic) is model-agnostic.

Multi-scale tiling: besides the full frame, the frame is re-scanned in a
2x2, 3x3 and 4x4 grid (each tile enlarged by `overlap` so neighbours overlap
and nothing gets cut in half at a tile edge). This catches small/distant
subjects a single 640x640 pass on a 1080p+ frame would miss. Skip tiling
(`--no-tiling`) for close-up footage where subjects already fill the frame.

Usage:
    python detect.py <clip.mp4> <out.json> --model model.onnx [options]

Options:
    --class-id N        COCO class to keep (default 0 = person)
    --threshold F        confidence threshold (default 0.25)
    --frame-step N       sample every Nth native frame (default 5 -- match
                          your CVAT task's frame_filter step)
    --label-id N         CVAT label_id to stamp on every shape (required)
    --attrs JSON         extra default attributes as a JSON list, e.g.
                          '[{"spec_id": 12, "value": "unreviewed"}]'
    --max-frames N       stop after N sampled frames (for a quick test run)
    --no-tiling          only run the full-frame pass, skip the zoom grid
"""
import argparse
import glob
import json
import os
import site
import sys
import time

# onnxruntime-gpu on Windows needs the NVIDIA pip packages' DLL dirs on PATH;
# without this it silently falls back to CPU with no error.
for _sp in site.getsitepackages():
    for _d in glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np
import onnxruntime as ort

GRIDS = [(2, 2), (3, 3), (4, 4)]
OVERLAP = 0.20

SESS = None
INP = None
OUT_NAMES = None
CLASS_ID = 0
THRESHOLD = 0.25


def load_model(model_path):
    global SESS, INP, OUT_NAMES
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    SESS = ort.InferenceSession(model_path, providers=providers, sess_options=so)
    active = SESS.get_providers()
    if "CUDAExecutionProvider" not in active:
        print(
            "WARNING: onnxruntime is running on CPU, not GPU "
            "(CUDAExecutionProvider unavailable). Detection will be much "
            "slower. Check `pip show onnxruntime-gpu` and your CUDA/cuDNN "
            "install if you expected GPU.",
            file=sys.stderr,
        )
    INP = SESS.get_inputs()[0].name
    OUT_NAMES = [o.name for o in SESS.get_outputs()]


def letterbox(im, new_shape=(640, 640)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)


def infer_tile(bgr_tile):
    """Run the model on one BGR image tile. Returns [(x1,y1,x2,y2,score), ...]
    in the tile's own pixel coordinates."""
    x, r, (dw, dh) = letterbox(bgr_tile)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    batch = np.ascontiguousarray(np.expand_dims(chw, 0).astype(np.float32) / 255.0)
    det = SESS.run(OUT_NAMES, {INP: batch})[0]
    h, w = bgr_tile.shape[:2]
    results = []
    for row in det:
        _, x1, y1, x2, y2, label, score = row
        if int(label) != CLASS_ID or score < THRESHOLD:
            continue
        x1 = max(0, min(w, (x1 - dw) / r))
        y1 = max(0, min(h, (y1 - dh) / r))
        x2 = max(0, min(w, (x2 - dw) / r))
        y2 = max(0, min(h, (y2 - dh) / r))
        if x2 > x1 and y2 > y1:
            results.append((float(x1), float(y1), float(x2), float(y2), float(score)))
    return results


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


def nms_merge(dets, iou_thresh=0.5):
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    for d in dets:
        if all(_iou(d, k) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def tile_grid(w, h, cols, rows, overlap=OVERLAP):
    tw, th = w / cols, h / rows
    ow, oh = tw * overlap, th * overlap
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw - ow / 2))
            y1 = max(0, int(r * th - oh / 2))
            x2 = min(w, int((c + 1) * tw + ow / 2))
            y2 = min(h, int((r + 1) * th + oh / 2))
            yield x1, y1, x2, y2


def detect_frame(frame, use_tiling):
    h, w = frame.shape[:2]
    dets = list(infer_tile(frame))
    if use_tiling:
        for cols, rows in GRIDS:
            for (ox1, oy1, ox2, oy2) in tile_grid(w, h, cols, rows):
                crop = frame[oy1:oy2, ox1:ox2]
                for (x1, y1, x2, y2, s) in infer_tile(crop):
                    dets.append((x1 + ox1, y1 + oy1, x2 + ox1, y2 + oy1, s))
    return nms_merge(dets)


def main():
    global CLASS_ID, THRESHOLD

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label-id", type=int, required=True)
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--attrs", default="[]")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-tiling", action="store_true")
    args = ap.parse_args()

    CLASS_ID = args.class_id
    THRESHOLD = args.threshold
    label_id = args.label_id
    attrs = json.loads(args.attrs)
    use_tiling = not args.no_tiling

    load_model(args.model)

    NL = "\n"
    part_path = args.out + ".part"
    done = 0
    if os.path.exists(part_path):
        with open(part_path, encoding="utf-8") as f:
            good = [l for l in f.read().split(NL) if l.strip()]
        try:
            json.loads(good[-1])
        except Exception:
            good = good[:-1]
        with open(part_path, "w", encoding="utf-8") as f:
            f.write("".join(l + NL for l in good))
        done = len(good)
        print(f"RESUME {done} sampled frames already done", flush=True)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.clip}")

    frame_idx = 0
    sampled = 0
    t0 = time.time()
    part = open(part_path, "a", encoding="utf-8")
    while True:
        if sampled < done and frame_idx % args.frame_step == 0:
            if not cap.grab():
                break
            sampled += 1
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_step == 0:
            dets = detect_frame(frame, use_tiling)
            fshapes = [{
                "type": "rectangle", "occluded": False, "z_order": 0,
                "points": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "rotation": 0, "outside": False, "frame": sampled,
                "label_id": label_id, "group": 0, "source": "auto",
                "attributes": attrs,
            } for (x1, y1, x2, y2, score) in dets]
            part.write(json.dumps(fshapes) + NL)
            part.flush()
            os.fsync(part.fileno())
            sampled += 1
            if sampled % 10 == 0:
                print(f"[{args.clip}] sampled={sampled} native_frame={frame_idx} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
            if args.max_frames and sampled >= args.max_frames:
                break
        frame_idx += 1
    cap.release()
    part.close()

    shapes = []
    with open(part_path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                shapes.extend(json.loads(l))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"version": 0, "tags": [], "shapes": shapes, "tracks": []}, f)
    print(f"DONE {args.clip}: sampled_frames={sampled} total_shapes={len(shapes)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
