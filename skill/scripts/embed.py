"""
Stage 4.5 -- appearance embedding for long-term re-identification.

Ordinary tracking (track.py / refine.py) only bridges SHORT gaps -- a
constant-velocity + IoU model has nothing useful to say about a subject who
was invisible for 15 seconds behind a ruck, or who left frame and came back.
That's fine for a smooth, uninterrupted clip, but it means every longer
occlusion silently starts a brand new track ID for the same real person.
This stage is what turns "785 tracks for 30 players" back into "30 players,
each with one persistent ID across the whole match" -- see reid_merge.py for
the matching step that actually does that; this script only computes the
per-track appearance fingerprint reid_merge.py matches on.

For each track, sample a handful of its MEASURED (non-ghost) boxes spread
across its lifetime, crop them from the source video, and run each crop
through an image embedding model -- averaged and L2-normalized into one
vector per track. Two subjects that are actually the same person tend to
produce similar vectors even minutes apart and from a different track;
two different subjects, even in similar kit, tend not to.

The model can be ANYTHING that takes an image and returns a single fixed-size
vector: a real person-re-ID network (e.g. an OSNet export) if you have one,
or, if you don't, a generic ImageNet-pretrained classifier's output layer
works surprisingly well as a fallback appearance fingerprint -- it wasn't
trained to distinguish people, but two crops of the same person in the same
kit still tend to activate it similarly. A real re-ID model will out-perform
this fallback, especially for telling apart players on the same team; use one
if you have it.

Two backends are supported, picked automatically from the model file's
extension:

- `.onnx` -> onnxruntime. Default preprocessing matches a generic ImageNet
  classifier (resize to a square, RGB, /255, ImageNet mean/std) -- override
  with --scale/--mean/--std/--bgr for a different model.
- `.xml` -> OpenVINO IR (needs a sibling `.bin` file with the same base
  name, which is how OpenVINO ships every model, including a real person
  re-ID network from Intel's Open Model Zoo -- e.g.
  `person-reidentification-retail-0288`, input 128x256 (w x h), RGB, raw
  0-255 float32, no mean/std subtraction: run this script with
  `--width 128 --height 256 --scale 1 --mean 0 0 0 --std 1 1 1` for that
  model specifically). A dedicated re-ID model like this one is a real
  improvement over a generic classifier's output layer -- it was actually
  trained to tell people apart, not just to name what's in the picture, and
  the difference shows up directly in how well the matching in
  reid_merge.py separates real identities.

Usage:
    python embed.py <clip.mp4> <clip_tracks.json> <out_embeddings.json> \\
        --model model.onnx --input-name data --width 224 --height 224 [--samples 5]

Writes a JSON object: {"<track_index>": [float, ...], ...} -- one L2-normalized
vector per track that had at least one measured shape.
"""
import argparse
import glob
import json
import os
import site
import sys

# same fix as detect.py: onnxruntime-gpu on Windows needs the NVIDIA pip
# packages' DLL dirs on PATH, or CUDAExecutionProvider silently fails to
# load and every inference call here falls back to (much slower) CPU.
for _sp in site.getsitepackages():
    for _d in glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OnnxBackend:
    def __init__(self, model_path, input_name):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = input_name
        self.output_name = self.sess.get_outputs()[0].name
        self.on_gpu = "CUDAExecutionProvider" in self.sess.get_providers()

    def run(self, batch):
        return self.sess.run([self.output_name], {self.input_name: batch})[0].reshape(-1)


class OpenVinoBackend:
    def __init__(self, model_path, input_name):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled = core.compile_model(model, "CPU")
        self.output = self.compiled.output(0)
        self.on_gpu = False  # OpenVINO CPU plugin; GPU plugin needs an Intel GPU + separate setup

    def run(self, batch):
        return np.asarray(self.compiled(batch)[self.output]).reshape(-1)


def load_backend(model_path, input_name):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".xml":
        return OpenVinoBackend(model_path, input_name)
    return OnnxBackend(model_path, input_name)


def preprocess(bgr_crop, width, height, bgr, scale, mean, std):
    img = bgr_crop if bgr else cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / scale
    normed = (resized - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    chw = normed.transpose(2, 0, 1)
    return np.ascontiguousarray(np.expand_dims(chw, 0))


def pick_sample_frames(track, n_samples):
    measured = [s for s in track["shapes"] if not s["outside"] and not s["occluded"]]
    if not measured:
        return []
    if len(measured) <= n_samples:
        return measured
    idxs = np.linspace(0, len(measured) - 1, n_samples).round().astype(int)
    return [measured[i] for i in sorted(set(idxs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("tracks_json")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, help="a .onnx file (onnxruntime) or a .xml file (OpenVINO IR, needs a sibling .bin)")
    ap.add_argument("--input-name", default="data")
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--bgr", action="store_true", help="skip BGR->RGB conversion (some models, e.g. some OpenVINO IR exports, expect BGR)")
    ap.add_argument("--scale", type=float, default=255.0, help="pixel values are divided by this before mean/std (default: 255, i.e. normalize to 0-1)")
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--samples", type=int, default=5, help="measured boxes sampled per track, spread across its lifetime")
    ap.add_argument("--mean", type=float, nargs=3, default=None, help="override normalization mean (default: ImageNet)")
    ap.add_argument("--std", type=float, nargs=3, default=None, help="override normalization std (default: ImageNet)")
    args = ap.parse_args()

    mean = args.mean if args.mean else IMAGENET_MEAN
    std = args.std if args.std else IMAGENET_STD

    backend = load_backend(args.model, args.input_name)
    if not backend.on_gpu:
        print(
            "WARNING: running on CPU, not GPU -- this stage reads one video frame per "
            "sampled box, so CPU-only will be slow on a long clip.", file=sys.stderr,
        )

    tracks = json.load(open(args.tracks_json, encoding="utf-8"))["tracks"]

    # Gather every (native_frame, track_index, box) this run needs, and visit
    # them in ascending frame order with a single sequential decode pass
    # (repeated cap.grab() to skip, one cap.read() to capture) rather than
    # cap.set(CAP_PROP_POS_FRAMES, ...) per sample. Random-access seeking can
    # be extremely slow on some encodes (sparse keyframes, VFR, a "repaired"
    # transcode) -- sequential decoding is the one access pattern every
    # container/codec handles well, and it turns this into a single pass
    # over the clip no matter how many tracks/samples there are.
    wanted = []  # (native_frame, track_idx, points)
    for ti, tr in enumerate(tracks):
        for s in pick_sample_frames(tr, args.samples):
            wanted.append((s["frame"] * args.frame_step, ti, s["points"]))
    wanted.sort(key=lambda w: w[0])

    vecs_by_track = {}
    cap = cv2.VideoCapture(args.clip)
    pos = 0
    for i, (native_frame, ti, points) in enumerate(wanted):
        while pos < native_frame:
            if not cap.grab():
                break
            pos += 1
        if pos != native_frame:
            continue
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        x1, y1, x2, y2 = [max(0, int(round(v))) for v in points]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        batch = preprocess(crop, args.width, args.height, args.bgr, args.scale, mean, std)
        vec = backend.run(batch)
        vecs_by_track.setdefault(ti, []).append(vec)
        if (i + 1) % 200 == 0:
            print(f"embedded sample {i + 1}/{len(wanted)} ({len(vecs_by_track)} tracks so far, "
                  f"native frame {native_frame})", flush=True)
    cap.release()

    embeddings = {}
    for ti, vecs in vecs_by_track.items():
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        embeddings[str(ti)] = (avg / norm if norm > 0 else avg).tolist()

    json.dump(embeddings, open(args.out, "w"))
    print(f"DONE {args.clip}: {len(embeddings)}/{len(tracks)} tracks embedded -> {args.out}")


if __name__ == "__main__":
    main()
