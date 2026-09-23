"""
Stage 4.5b -- kit-color feature extraction, for team/role classification.

Deliberately separate from embed.py's re-ID embedding: that embedding is
trained to recognize the same PERSON despite appearance variation, which
makes it a poor tool for the opposite job -- telling two DIFFERENT people
apart by what they're wearing. This script computes a plain color
descriptor instead: crop the jersey region (upper portion of the box, where
a rugby/football kit's primary color lives, not shorts/socks/skin), mask
out the pitch's green, and summarize what's left as a hue/saturation/value
feature. Cheap, deterministic, no model needed -- exactly what team-color
clustering in published sports-analytics pipelines uses.

Usage:
    python colorfeat.py <clip.mp4> <clip_tracks.json> <out_colorfeat.json> \\
        --frame-step 5 [--samples 6]

Writes {"<track_index>": {"hue_cos": f, "hue_sin": f, "sat": f, "val": f, "n": int}, ...}
-- hue as (cos, sin) of 2x the angle so a hue clustering distance behaves
correctly across the 0/360-degree wraparound (and doesn't confuse two kits
180 degrees apart, e.g. red vs cyan, which --- unlikely in practice but the
right way to encode a circular quantity regardless).
"""
import argparse
import json

import cv2
import numpy as np


def pick_sample_frames(track, n_samples):
    measured = [s for s in track["shapes"] if not s["outside"] and not s["occluded"]]
    if not measured:
        return []
    if len(measured) <= n_samples:
        return measured
    idxs = np.linspace(0, len(measured) - 1, n_samples).round().astype(int)
    return [measured[i] for i in sorted(set(idxs))]


def jersey_color(bgr_crop):
    """Mean HSV of the crop's upper torso region, with pitch-green and
    near-black/near-white (shadow/highlight blowout) pixels masked out so
    the grass background and lighting extremes don't dominate the mean."""
    h, w = bgr_crop.shape[:2]
    if h < 6 or w < 4:
        return None
    torso = bgr_crop[int(h * 0.10):int(h * 0.55), :]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    is_green = (hue > 35) & (hue < 85) & (sat > 60)
    is_extreme = (val < 25) | (val > 240) | (sat < 20)
    keep = ~(is_green | is_extreme)
    if keep.sum() < max(6, 0.05 * len(hue)):
        return None
    return float(np.mean(hue[keep])), float(np.mean(sat[keep])), float(np.mean(val[keep]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("tracks_json")
    ap.add_argument("out")
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    tracks = json.load(open(args.tracks_json, encoding="utf-8"))["tracks"]

    wanted = []
    for ti, tr in enumerate(tracks):
        for s in pick_sample_frames(tr, args.samples):
            wanted.append((s["frame"] * args.frame_step, ti, s["points"]))
    wanted.sort(key=lambda w: w[0])

    readings = {}
    cap = cv2.VideoCapture(args.clip)
    pos = 0
    for native_frame, ti, points in wanted:
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
        color = jersey_color(crop)
        if color is not None:
            readings.setdefault(ti, []).append(color)
    cap.release()

    out = {}
    for ti, readings_list in readings.items():
        hues = np.array([r[0] for r in readings_list])
        sats = np.array([r[1] for r in readings_list])
        vals = np.array([r[2] for r in readings_list])
        theta = np.deg2rad(hues * 2)  # OpenCV hue is 0-179 (= 0-358 degrees); double it to full circle
        out[str(ti)] = {
            "hue_cos": float(np.mean(np.cos(theta))),
            "hue_sin": float(np.mean(np.sin(theta))),
            "sat": float(np.mean(sats)) / 255.0,
            "val": float(np.mean(vals)) / 255.0,
            "n": len(readings_list),
        }

    json.dump(out, open(args.out, "w"))
    print(f"DONE {args.clip}: {len(out)}/{len(tracks)} tracks got a color reading -> {args.out}")


if __name__ == "__main__":
    main()
