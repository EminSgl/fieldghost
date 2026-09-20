"""
Shared tracking library: turns a list of per-frame detection boxes into CVAT
tracks. Used directly by track.py (pass 1) and imported by refine.py (pass 2,
which reruns this same stitching after recovering gap boxes).

Two stages:

Stage 1 (`stage1`) -- forward frame-to-frame association. Each active
tracklet predicts its next box with a constant-velocity model; predictions
are matched to this frame's detections with the Hungarian algorithm on a
cost that blends IoU with jersey/appearance similarity (an HSV histogram of
the upper ~55% of the box, i.e. roughly where a kit/jersey is instead of legs
or ground). A tracklet that goes unmatched for more than `max_lost` frames is
retired; short gaps within that window survive untouched.

Stage 2 (`stitch`) -- tracklet stitching across longer gaps (occlusion,
a pile-up, a subject leaving and re-entering frame). The end of an earlier
tracklet is extrapolated forward and the start of a later one extrapolated
backward; if the two predictions land close enough (in a box-height-relative
sense, with tolerance growing with the gap length) and the appearance
histograms are close enough, they're linked and the gap is filled with
linearly interpolated boxes marked `occluded=True` -- CVAT's own field for
"this shape is an estimate, not a fresh detection." Nothing here is
species/sport-specific: it works on any roughly-rigid tracked object as long
as detections come from the same fixed-position camera.

`drop_fragments` removes boxes that are >85% contained inside a clearly
larger box in the same frame -- a common artifact of tiled multi-scale
detection, where a zoomed-in tile re-detects a leg or torso as its own
"person" on top of the correct full-body box.
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Tunables, exposed as module attributes so callers (refine.py) can override
# them per run, e.g. a tighter MAX_GAP for the second pass.
MAX_LOST = 4      # stage 1: frames a tracklet may go unmatched before retiring
MAX_GAP = 60       # stage 2: max sampled-frame gap a stitch may bridge
MIN_LEN = 3        # tracklets shorter than this aren't used as stitch anchors
W_APP = 0.35       # weight of appearance distance vs (1 - IoU) in stage 1 cost


def iou_matrix(a, b):
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    ub = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(ua + ub - inter, 1e-9)


def drop_fragments(boxes):
    keep = []
    for i, b in enumerate(boxes):
        ab = (b[2] - b[0]) * (b[3] - b[1])
        frag = False
        for j, B in enumerate(boxes):
            if i == j:
                continue
            aB = (B[2] - B[0]) * (B[3] - B[1])
            if aB <= ab / 0.6:
                continue
            ix = max(0, min(b[2], B[2]) - max(b[0], B[0]))
            iy = max(0, min(b[3], B[3]) - max(b[1], B[1]))
            if ix * iy / max(ab, 1e-9) > 0.85:
                frag = True
                break
        if not frag:
            keep.append(i)
    return keep


def hist(frame, box):
    """HSV colour histogram of the upper body region of `box` in `frame`,
    used as a lightweight appearance cue (kit colour) for association."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h = y2 - y1
    y2 = y1 + max(2, int(h * 0.55))
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hg = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    cv2.normalize(hg, hg, 1, 0, cv2.NORM_L1)
    return hg.flatten().astype(np.float32)


def app_dist(h1, h2):
    if h1 is None or h2 is None:
        return 0.5
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(h1 * h2)))))


class Tracklet:
    def __init__(self, t, box, h):
        self.t, self.boxes, self.hists = [t], [np.array(box, float)], [h]
        self.hist = h
        self.lost = 0

    def vel(self):
        if len(self.t) < 2:
            return np.zeros(4)
        dt = self.t[-1] - self.t[-2]
        return (self.boxes[-1] - self.boxes[-2]) / dt

    def predict(self, t):
        return self.boxes[-1] + self.vel() * (t - self.t[-1])

    def add(self, t, box, h):
        self.t.append(t)
        self.boxes.append(np.array(box, float))
        self.hists.append(h)
        if h is not None:
            self.hist = h if self.hist is None else 0.8 * self.hist + 0.2 * h
        self.lost = 0


def stage1(dets, hists):
    active, done = [], []
    for t, (boxes, hs) in enumerate(zip(dets, hists)):
        matched_t, matched_d = set(), set()
        if active and boxes:
            pred = [tr.predict(t) for tr in active]
            iou = iou_matrix(pred, boxes)
            ap = np.array([[app_dist(tr.hist, h) for h in hs] for tr in active])
            cost = (1 - iou) + W_APP * ap
            cost[iou < 0.1] = 1e6
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 1e5:
                    active[i].add(t, boxes[j], hs[j])
                    matched_t.add(i)
                    matched_d.add(j)
        for i, tr in enumerate(active):
            if i not in matched_t:
                tr.lost += 1
        for j, b in enumerate(boxes):
            if j not in matched_d:
                active.append(Tracklet(t, b, hs[j]))
        keep = []
        for tr in active:
            (done if tr.lost > MAX_LOST else keep).append(tr)
        active = keep
    return done + active


def stitch(tracklets):
    """Greedy-optimal linking of tracklet ends to later tracklet starts."""
    ts = sorted(tracklets, key=lambda x: x.t[0])
    n = len(ts)
    cost = np.full((n, n), 1e6)
    for i, a in enumerate(ts):
        if len(a.t) < MIN_LEN:
            continue
        for j, b in enumerate(ts):
            if i == j or len(b.t) < MIN_LEN:
                continue
            gap = b.t[0] - a.t[-1]
            if gap < 1 or gap > MAX_GAP:
                continue
            damp = min(gap, 8)
            pa = a.boxes[-1] + a.vel() * damp * (gap / damp if gap <= 8 else 1)
            vb = (b.boxes[1] - b.boxes[0]) / (b.t[1] - b.t[0]) if len(b.t) > 1 else np.zeros(4)
            pb = b.boxes[0] - vb * min(gap, 8)
            ca = np.array([(pa[0] + pa[2]) / 2, (pa[1] + pa[3]) / 2])
            cb = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            h = max(a.boxes[-1][3] - a.boxes[-1][1], 8)
            dist = np.linalg.norm(ca - cb) / (h * (1 + 0.35 * gap))
            direct = np.linalg.norm(
                np.array([(a.boxes[-1][0] + a.boxes[-1][2]) / 2, (a.boxes[-1][1] + a.boxes[-1][3]) / 2]) -
                np.array([(b.boxes[0][0] + b.boxes[0][2]) / 2, (b.boxes[0][1] + b.boxes[0][3]) / 2])
            ) / (h * (1 + 0.35 * gap))
            d = min(dist, direct)
            ad = app_dist(a.hist, b.hist)
            if d > 1.0 or ad > 0.55:
                continue
            cost[i, j] = d + 0.8 * ad
    r, c = linear_sum_assignment(cost)
    nxt = {i: j for i, j in zip(r, c) if cost[i, j] < 1e5}
    prv = {j: i for i, j in nxt.items()}
    chains, seen = [], set()
    for i in range(n):
        if i in prv or i in seen:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        seen.update(chain)
        chains.append([ts[k] for k in chain])
    return chains


def chain_to_track(chain):
    """Flatten a chain of stitched tracklets into [(frame, box, is_ghost), ...],
    filling stitched gaps with linear interpolation flagged as ghost boxes."""
    shapes = []
    for k, tr in enumerate(chain):
        for t, b in zip(tr.t, tr.boxes):
            shapes.append((t, b, False))
        if k + 1 < len(chain):
            nx = chain[k + 1]
            t0, b0, t1, b1 = tr.t[-1], tr.boxes[-1], nx.t[0], nx.boxes[0]
            for t in range(t0 + 1, t1):
                a = (t - t0) / (t1 - t0)
                shapes.append((t, b0 * (1 - a) + b1 * a, True))
    shapes.sort(key=lambda x: x[0])
    return shapes


def to_cvat(tracks, n_frames, label_id, attrs):
    """[(frame, box, is_ghost), ...] per track -> CVAT track dicts.
    `is_ghost` maps to CVAT's `occluded` flag on each shape."""
    out = []
    for shapes in tracks:
        sh = []
        for t, b, ghost in shapes:
            sh.append({"type": "rectangle", "occluded": bool(ghost), "outside": False, "z_order": 0,
                       "rotation": 0, "frame": int(t), "source": "auto",
                       "points": [round(float(v), 1) for v in b], "attributes": []})
        last = shapes[-1][0]
        if last + 1 < n_frames:
            sh.append({**sh[-1], "frame": int(last + 1), "outside": True})
        out.append({"frame": int(shapes[0][0]), "label_id": label_id, "group": 0, "source": "auto",
                    "attributes": attrs, "shapes": sh})
    return out


def build_tracks(detections_per_frame, video_path, frame_step):
    """End-to-end: raw per-frame boxes + the source video -> stitched tracks.
    `detections_per_frame` is a list (one entry per sampled frame) of box
    lists [x1,y1,x2,y2]. Returns [(frame, box, is_ghost), ...] per track."""
    boxes = [[pts[i] for i in drop_fragments(pts)] for pts in detections_per_frame]
    cap = cv2.VideoCapture(video_path)
    hists = []
    for t in range(len(boxes)):
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("video shorter than detections -- wrong file, or frame-step mismatch")
        hists.append([hist(fr, b) for b in boxes[t]])
        for _ in range(frame_step - 1):
            cap.grab()
    cap.release()
    chains = stitch(stage1(boxes, hists))
    return [chain_to_track(c) for c in chains], boxes
