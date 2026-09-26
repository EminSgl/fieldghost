---
name: fieldghost
description: Runs a full sports-video-to-CVAT annotation pipeline -- GPU person/object detection, occlusion-aware multi-object tracking, gap recovery for occluded subjects, appearance-based re-identification so each real player keeps ONE persistent track ID across the whole match instead of a new one every time they're re-detected, and upload to CVAT (Computer Vision Annotation Tool) task jobs via its REST API -- then helps QA and clean up false-positive "ghost" labels (static field furniture like corner flags or poles mislabeled as people) through a visually-verified removal workflow. Use this skill whenever the user wants to auto-annotate a sports video (or any fixed-camera video with a CVAT task already set up) with tracked bounding boxes: phrases like "label this match", "track the players", "detect and track in CVAT", "upload annotations to my CVAT task/job", "auto-track this video", "why does this player keep getting a new ID/label", "keep the same number/ID for the same player", or "these boxes/labels look wrong, can we clean them up" (for a task this skill or a similar pipeline produced) should all trigger it, even if the user doesn't say "fieldghost" or "pipeline" explicitly. Also use it if the user is debugging a CVAT upload getting 403'd, a Cloudflare block on a CVAT API call, or ghost/occluded box handling in CVAT tracks.
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

## A killed background run may not actually be dead -- verify before you resume

If you're driving `detect.py`/`refine.py` as a background process (an agent
harness's own "kill this task, it's using too much memory" reaper, a crashed
terminal, a closed laptop lid) and you're told the task was stopped: **do
not trust that and just relaunch.** On at least one real run, an agent's
background-task manager repeatedly killed a detection run for running the
system out of memory, reported it as killed, and the agent just relaunched
it each time on the assumption the slate was clean. It wasn't: on Windows in
particular, killing the top-level shell wrapper does not reliably kill the
`python.exe` child (or even the inner `bash.exe` that spawned it) -- it can
keep running, orphaned, invisible to the harness's own tracking. Every
"just restart" stacked ANOTHER copy on top of the surviving ones. Six of
them ended up running simultaneously, all fighting over one small laptop
GPU's ~4GB of VRAM (which is exactly the kind of contention that produces
more out-of-memory pressure, not less -- a vicious cycle), AND, worse, all
independently `open(part_path, "a")`-appending to the *same* `.json.part`
checkpoint file with no coordination between them. The result: a checkpoint
file with far more lines than the clip could possibly have real sampled
frames (7994 lines for a clip whose true budget was 5400) -- silently
corrupted, not obviously broken, the kind of thing that would have polluted
a "finished" detection pass with duplicate/conflicting frame data.

Before resuming ANY interrupted detect/refine run, actually check, don't
assume:

1. List every process that could be a leftover worker, filtering by command
   line (not just name) so you catch the whole tree -- e.g. on Windows,
   `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match
   'detect.py|refine.py|run_pipeline' }`. Do this even (especially) right
   after a harness reports the task "killed" -- that status describes what
   the harness *tried* to do, not a verified fact about the OS process
   table.
2. If anything turns up, force-kill the whole tree, not just the PID you
   were told about -- a plain `Stop-Process` on one PID can leave its
   children running. On Windows, `taskkill /F /T /PID <pid>` (the `/T`
   kills the tree) is far more reliable than killing PIDs one at a time.
   Re-run the same query and confirm it comes back empty before moving on.
3. For a shared GPU workload specifically, cross-check with the GPU itself,
   not just the OS process list -- `nvidia-smi --query-compute-apps=pid,
   used_memory,process_name --format=csv` tells you exactly how many
   processes are actually holding the GPU right now, which is a much less
   ambiguous signal than parsing a process tree (background-task wrapper
   shells nest in ways that can look like duplicates when they aren't).
   Likewise, a Python venv on Windows shows TWO `python.exe` per worker:
   the venv's `Scripts\python.exe` launcher plus the real interpreter as
   its child (check `ParentProcessId`) -- that's one run, not a duplicate.
4. If a checkpoint file (`<clip>.json.part`) could have been written to by
   more than one process -- i.e. you skipped step 1-2 even once during that
   run's lifetime, or you're not sure -- don't trust it. A quick sanity
   check: line count should never exceed the clip's expected sampled-frame
   budget (`native_frames // frame_step`, plus or minus a handful for an
   in-flight last write). If it does, the file is corrupted; delete it and
   the derived `_tracks.json`/`_checklist.json` and redo that clip's
   detection from scratch. Re-running costs GPU time; trusting corrupted
   detections costs you finding out much later, after re-ID and upload,
   when a clip's numbers look wrong for no visible reason.

## The pipeline, stage by stage

Run stages 1-4 once per clip; stages 5-6 once for the whole task.

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

### 4. Re-identify players across the whole match (`scripts/embed.py` + `scripts/reid_merge.py`)

Optional but usually what the user actually wants when they ask for
"tracking": without this stage, ordinary tracking (stages 2-3) only bridges
SHORT gaps, so every longer occlusion -- a ruck, a player leaving frame for
a minute -- silently starts a brand-new track ID for the same real person.
On real match footage this is not a rare edge case: expect on the order of
10-30x more raw tracks than there are actual players on the pitch. This
stage is what turns that back into one persistent ID per player.

For each clip, compute one appearance embedding per track:

```
python embed.py <clip.mp4> <clip_prefix>_tracks.json <clip_prefix>_embeddings.json \
    --model <embedding_model.onnx> --input-name <input_tensor_name> [--samples 5]
```

Any model that maps an image to a fixed-size vector works -- a real
person-re-ID network (e.g. an OSNet export) gives the best results,
especially telling apart players on the same team in identical kit; a
generic ImageNet classifier's output layer is a usable fallback if that's
all you have (ask the user which they have before picking one; don't
silently assume). This step reads video frames one at a time per sampled
box, so it's noticeably slower than the earlier stages on a long clip --
tell the user roughly how long before starting, and it's fine to run it
after everything else (order relative to merge doesn't matter, only that
it happens before the `--reid` merge step below).

Then, as part of the merge (see stage 5), pass `--reid` and it merges
tracks across clips into persistent identities automatically. Report the
one-line summary `reid_merge.match_and_merge` prints -- fragment count
before vs. persistent identity count after is the number the user actually
cares about.

**Be upfront about what this does and doesn't get you.** This is an
appearance-similarity heuristic, not ground truth, and on real match
footage tested during development it did NOT collapse cleanly down to one
ID per real player -- it took 785 raw fragment tracks down to roughly
500-550 persistent identities at a conservative threshold (~0.85-0.90
cosine), a real improvement but nowhere near the ~30 actual players on the
pitch. Push the threshold lower and the fragment count drops further, but a
new failure mode replaces it: a handful of identities start silently
absorbing dozens of genuinely different people (same-kit teammates
especially) into one ID. `reid_merge.py`'s complete-linkage matching (a
candidate must resemble every existing exemplar of an identity, not just a
blurred running average) blunts this but does not eliminate it -- there is
no threshold in between that gets you both a small fragment count AND
correct identities with a similarity heuristic alone, at least not with a
retail-domain re-ID model looking at outdoor sports footage it wasn't
trained on.

Tell the user this plainly: this stage meaningfully reduces track clutter
and gets some real re-appearances right, but a same-kit team's individual
players will still fragment across multiple IDs, and it needs a human
spot-checking merged identities in the CVAT UI, not blind trust. Getting
all the way to one ID per real player reliably needs either a re-ID model
actually fine-tuned on this sport/footage, or a jersey-number OCR pass
(read the real printed number instead of inferring identity from
appearance) -- both are real, larger follow-on projects, not something to
attempt silently as part of this stage.

### 4.5 Team/role classification (`scripts/colorfeat.py` + `scripts/classify_roles.py`)

Optional, and separate from re-ID (stage 4) on purpose: a re-ID embedding is
trained to recognize the same PERSON despite appearance changes, which makes
it a poor tool for the opposite job of telling two DIFFERENT people apart by
what they're wearing. For each clip, extract a jersey-color feature instead:

```
python colorfeat.py <clip.mp4> <clip_prefix>_tracks.json <clip_prefix>_colorfeat.json [--samples 6]
```

Cheap, deterministic, no model -- crops the jersey region, masks out pitch
green and lighting extremes, and summarizes hue/saturation as a clustering
feature. Then, as part of the merge (stage 5), pass `--classify-roles
--role-attr-spec <id>` and it clusters every track (or re-id-merged identity)
into `team_a` / `team_b` / `uncertain` using k=3 k-means on color -- k=3, not
k=2, because a third genuinely distinct kit color (e.g. a referee) otherwise
has nowhere to go but whichever team centroid is nearest and silently joins a
team. The two largest clusters become the teams; the smallest, plus any track
too far from its own cluster's centroid, becomes `uncertain`.

**This does not attempt a further `referee`/`staff` split.** Tested on real
rugby footage: referees are commonly in all-black or another muted color,
not reliably distinguishable from a team in a similar dark kit using color
alone, and the `uncertain` bucket in practice is dominated by ambiguous crops
(motion blur, tackle pile-ups, misdetected static objects) rather than clean
shots of match officials. Report `uncertain` counts to the user as "needs a
human look," not as "these are the referees."

`team_a`/`team_b` are arbitrary labels with no color hint given -- nothing in
a single match tells you which cluster is the "home" side. Tell the user this
before they read anything into which label a given cluster got.

### 5. Merge + upload (`scripts/merge_upload.py`)

```
python merge_upload.py config.yaml --clips-dir <dir_with_*_tracks.json> --out-dir <merged_dir>
```

Add `--reid` (see stage 4) to merge fragmented tracks into persistent player
identities as part of this step -- it needs `<clip_name>_embeddings.json`
next to each clip's tracks file in `--clips-dir`. Add `--player-id-spec <id>`
if the user wants the persistent identity number stamped as a visible CVAT
attribute on each track, not just reflected in there being fewer tracks. Add
`--classify-roles --role-attr-spec <id>` (see stage 4.5) to stamp a
team_a/team_b/uncertain role attribute too -- it needs
`<clip_name>_colorfeat.json` next to each clip's tracks file in
`--clips-dir`.

Always run this WITHOUT `--upload` first and look at the printed per-job
shape/track counts -- do they look plausible for the footage? Then, **stop
and confirm with the user before adding `--upload`**: this is the first
point where the pipeline touches shared state on the CVAT server, and a
`PUT` there is a full replace of whatever's already in that job (see
`references/cvat-api-notes.md`). Once confirmed:

```
python merge_upload.py config.yaml --clips-dir <dir> --out-dir <merged_dir> \
    --reid --upload --username <user> --password <pass>
```

(or `--token <token>` if they already have one). If you get a `TokenExpired`
error on upload, that's expected sometimes -- just re-run with fresh
credentials, don't treat it as a pipeline failure.

### 6. QA: find and remove false-positive "ghost object" labels

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

A box only ever excludes a track that stays inside it for its ENTIRE
measured lifetime, never just a track whose *average* position happens to
land inside it. A confirmed box marks a pole's fixed screen position, and a
real pole fragment never leaves it; a moving player's trajectory average can
still land inside the same box just because their path happened to cross it
(most visibly: someone running in from a frame edge, straight through where
a pole sits on screen). Averaging used to wrongly drop exactly that case --
if a legitimately-tracked, clearly-moving subject goes missing from the
upload after adding an exclusion box, that mismatch is the first thing to
check, not the detector.

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
