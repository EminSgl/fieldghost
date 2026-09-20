---
name: fieldghost
description: Runs a full sports-video-to-CVAT annotation pipeline -- GPU person/object detection, occlusion-aware multi-object tracking, gap recovery for occluded subjects, and upload to CVAT (Computer Vision Annotation Tool) task jobs via its REST API -- then helps QA and clean up false-positive "ghost" labels (static field furniture like corner flags or poles mislabeled as people) through a visually-verified removal workflow. Use this skill whenever the user wants to auto-annotate a sports video (or any fixed-camera video with a CVAT task already set up) with tracked bounding boxes: phrases like "label this match", "track the players", "detect and track in CVAT", "upload annotations to my CVAT task/job", "auto-track this video", or "these boxes/labels look wrong, can we clean them up" (for a task this skill or a similar pipeline produced) should all trigger it, even if the user doesn't say "fieldghost" or "pipeline" explicitly. Also use it if the user is debugging a CVAT upload getting 403'd, a Cloudflare block on a CVAT API call, or ghost/occluded box handling in CVAT tracks.
---

# FieldGhost

An end-to-end pipeline that takes a video already split into clips and
registered as jobs on a CVAT task, and produces tracked, occlusion-aware
bounding box annotations uploaded straight into those jobs -- built around
one core idea: **when a subject is briefly hidden, don't just drop the
track or freeze a box on top of them -- guess where they are, then go back
and try to actually find them there before giving up and leaving a ghost.**

You (the agent running this skill) are expected to drive the whole thing:
ask the user for what's missing, run the scripts, report progress in plain
numbers, and stop for explicit confirmation at the two points that touch
shared state (uploading, deleting). Nothing here needs the user to know
Python or the CVAT API -- translate stage output into "clip 2: 785 tracks,
53 still need a manual look" for them, not raw JSON.

## Before you start: gather what you need

Ask the user (don't guess these -- a wrong task id or label id fails loud,
but a wrong frame_step or clip start silently misaligns every box):

1. **CVAT host** (e.g. `https://cvat.example.com`) and how they'll
   authenticate -- an existing auth token, or a username+password you'll
   exchange for one via `POST /api/auth/login`.
2. **The task**: is it already created with jobs (one per clip, or however
   they split it)? If not, that setup (uploading video, creating the task,
   choosing a `frame_filter` step) happens in the CVAT UI or via
   `POST /api/tasks` first -- this skill starts from an existing task.
   You need each job's id and its `[start_frame, stop_frame]`, and the
   task's `frame_filter` step. See `references/cvat-api-notes.md` for the
   exact API calls to look these up if the user doesn't have them handy.
3. **The label id** to stamp on detections (`GET /api/labels?task_id=`),
   and, if they want ghost-vs-detected marked on each box, a spec_id for a
   *mutable* attribute to hold that.
4. **The video clip files** on disk, and for each one, its
   `source_start_frame` -- the native frame number in the full source video
   where that clip begins. If clips overlap in time, flag that now; you'll
   set an explicit `window` per clip in the config instead of relying on
   the automatic midpoint split.
5. **A detector model**: a YOLO-family ONNX export with NMS included (ask
   if they have one; `yolov7-nms-640.onnx`-style exports from the official
   YOLOv7 repo work directly with `scripts/detect.py` as-is). Confirm which
   class id is the subject they care about (COCO class 0 = person, if
   tracking players/staff/officials).
6. **GPU availability.** Ask, or just try importing `onnxruntime` and
   checking `onnxruntime.get_available_providers()` for
   `CUDAExecutionProvider`. Detection is by far the slowest stage; running
   it on CPU works but expect an order of magnitude slower, and say so
   before committing to it on a long video.

Fill these into a copy of `config.example.yaml` (see that file's comments --
it's meant to be self-explanatory) before running anything.

## Before doing anything: check for a run already in progress

Long detection/refine runs on real match footage take hours, not minutes,
and get interrupted by reboots, closed laptops, and crashed terminals more
often than they run to completion uninterrupted. Before starting fresh,
look for evidence of a prior attempt in whatever working directory the user
points you at (or ask where they were running it before):

- `<clip>.json.part` files -- a detection pass in progress or done but not
  finalized. `detect.py` resumes from these automatically; just re-run the
  same command.
- `<clip>_tracks.json` and `<clip>_checklist.json` -- refine.py already
  completed for that clip. Don't redo it.
- `job_<id>.json` files in a merge output directory -- merge already ran;
  check whether it was also uploaded (tail any logs, or just check the job
  in the CVAT UI / `GET /api/jobs/<id>/annotations`) before re-uploading.

Report what you find in plain terms ("clips 1 and 2 are fully done, clip 3
finished detection but not refine, clip 4 hasn't started") and only run the
missing pieces. Never restart a stage that's already checkpointed as done --
besides wasting GPU time, `detect.py`'s resume logic is there specifically
so you don't have to.

## The pipeline, stage by stage

Run stages 1-4 once per clip; stage 5 once for the whole task.

### 1. Detect (`scripts/detect.py`)

```
python detect.py <clip.mp4> <out.json> --model <model.onnx> --label-id <id> \
    --frame-step <step> [--class-id 0] [--threshold 0.25]
```

GPU, one clip at a time or several in parallel if the GPU has headroom --
more parallel jobs means more wall-clock time per job from contention, so
ask the user how many they want running at once rather than assuming
"as many as possible" is best. Checkpointed to `<out>.json.part`; safe to
kill and resume at any time with the exact same command.

### 2 & 3. Track + refine (`scripts/refine.py`)

Skip plain `track.py` for a real run -- `refine.py` does stitching AND the
gap-recovery pass in one go, which is the point of this whole pipeline:

```
python refine.py <clip.mp4> <out.json.part> <clip_prefix> --model <model.onnx> \
    --label-id <id> --frame-step <step>
```

Writes `<clip_prefix>_tracks.json` (the final per-clip tracks) and
`<clip_prefix>_checklist.json` (frames worth a human glance: sudden box-count
jumps, and any occlusion gap that recovery could NOT resolve, still marked
as a ghost box). Report the summary line it prints -- recovered count vs
still-ghost count is the single most useful health metric for a clip.

### 4. Merge + upload (`scripts/merge_upload.py`)

```
python merge_upload.py config.yaml --clips-dir <dir_with_*_tracks.json> --out-dir <merged_dir>
```

Always run this WITHOUT `--upload` first and look at the printed per-job
shape/track counts -- do they look plausible for the footage? Then, **stop
and confirm with the user before adding `--upload`**: this is the first
point where the pipeline touches shared state on the CVAT server, and a
`PUT` there is a full replace of whatever's already in that job (see
`references/cvat-api-notes.md`). Once confirmed:

```
python merge_upload.py config.yaml --clips-dir <dir> --out-dir <merged_dir> \
    --upload --username <user> --password <pass>
```

(or `--token <token>` if they already have one). If you get a `TokenExpired`
error on upload, that's expected sometimes -- just re-run with fresh
credentials, don't treat it as a pipeline failure.

### 5. QA: find and remove false-positive "ghost object" labels

Read this section in full before touching anything here -- it's the part
most likely to go wrong if rushed.

After upload, run:

```
python find_static_objects.py <merged_dir>/job_*.json --min-tracks 15 --max-width 20
```

This flags pixel locations where many separate short tracks pile up at
nearly the same spot with a small/thin average box -- the signature of one
static object (a corner flag, a touchline pole, a sponsor board on a post)
being re-detected and re-tracked from scratch every time occlusion or a
confidence dip breaks its "track." **Treat every result as a hypothesis,
not a fact.** In practice, roughly a third of candidates from this exact
heuristic turned out to be real, distant, temporarily-still subjects instead
(players warming up standing around, or standing during a stoppage in
play) -- there is no purely statistical feature that reliably tells those
apart from actual static furniture, because both are small, thin, and
barely move.

So, for every candidate:

1. Open the CVAT job at the sample frame the script printed, using whatever
   browser automation is available to you (this project was built using
   `gstack`'s `/browse` skill, but any Playwright/Puppeteer-style tool or
   even just asking the human to look works the same way) -- CVAT job URLs
   are `<host>/tasks/<task_id>/jobs/<job_id>?frame=<n>`.
2. Zoom in enough to actually tell people from poles. A wide establishing
   screenshot is not enough resolution to make this call reliably -- crop
   tightly around the reported box, or increase the browser viewport size
   before screenshotting, or use the CVAT UI's own zoom.
3. Only if you can see with your own eyes that it's a static object, note a
   tight pixel bounding box around it (in source-video coordinates) into an
   exclusion file:
   ```json
   [{"x_lo": 185, "x_hi": 235, "y_lo": 170, "y_hi": 210}]
   ```
4. Re-run `merge_upload.py` with `--exclude-boxes that_file.json --upload`.
   Because the merge always rebuilds from the untouched per-clip track
   files, a box you got even slightly wrong (too wide, wrong spot) costs
   nothing to fix -- adjust the box and re-run, you're never patching
   already-damaged data.

**Never skip step 1-2 and remove candidates in bulk based on the heuristic
alone.** If you're tempted to because there are a lot of candidates: that's
exactly the situation where a bad blanket filter does the most damage (in
testing, one first-pass attempt at "just remove anything that looks static
enough" wiped over half the real tracks in one job before anyone looked at
a single frame). Confirm each one, or leave it for the user to review
directly in the CVAT UI -- both are fine outcomes; silently deleting real
annotations is not.

## Reporting progress

Translate every stage's output into plain sentences with the actual
numbers, not raw stdout dumps: "Clip 3 done: 1945 tracks, 3700 frames still
need review out of 5439 total." Stop and ask before `--upload` and before
writing to any `--exclude-boxes` file that will delete tracks. Everything
else (detection, refine, merge without upload, running the candidate
finder) is read-only or local-file-only and fine to just do.
