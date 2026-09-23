<div align="center">

<img src="assets/banner.svg" alt="FieldGhost" width="100%" />

### Point it at a video and a CVAT task. Get back tracked players, not empty gaps.

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-38bdf8.svg)](LICENSE)
[![Works with](https://img.shields.io/badge/works%20with-CVAT-a78bfa)](https://github.com/cvat-ai/cvat)
[![Skill for](https://img.shields.io/badge/skill%20for-Claude%20Code%20%7C%20Codex-34d399)](#-install)
[![Made for](https://img.shields.io/badge/built%20on-a%20real%2080%20minute%20match-f8fafc)](#-the-honest-part)

</div>

---

## The problem this exists to fix

You run an off-the-shelf detector on sports footage and it does fine — right
up until a ruck forms, a tackle happens, or two players cross paths. For a
few frames, your subject is gone. Most auto-tracking pipelines handle this
one of two ways: **drop the track** (now it's two different "people" before
and after the pile-up) or **freeze a box** on the last known position and
call it done (now it's confidently wrong for half a second).

FieldGhost does neither. When a track hits a gap, it makes an estimate —
call it a **ghost box** — and then it goes back and actually looks: crops in
tight on where the subject *should* be, reruns detection at a lower
threshold on just that crop, and only if it finds something that plausibly
matches does the ghost become real again. What's left unresolved stays
flagged as a ghost, so a human reviewer knows exactly which handful of
frames, out of thousands, actually need their eyes.

Then it uploads straight into your CVAT task via the REST API, and helps you
hunt down the other kind of ghost — the corner flag or touchline pole that
got mislabeled as a person — without ever deleting real annotations by
accident.

## What it actually does

```mermaid
flowchart LR
    A[GPU detect<br/><sub>ONNX / YOLO, checkpointed</sub>] --> B[Track<br/><sub>IoU + jersey-colour, Hungarian match</sub>]
    B --> C[Recover ghosts<br/><sub>zoom crop + re-detect on every gap</sub>]
    C --> I[Re-identify<br/><sub>appearance embedding, fewer duplicate IDs</sub>]
    I --> D[Merge clips<br/><sub>→ CVAT task frame numbers</sub>]
    D --> E[Upload<br/><sub>PUT /jobs/&lt;id&gt;/annotations</sub>]
    E --> F{QA pass}
    F -->|candidate found| G[Visually confirm<br/>in the CVAT UI]
    G -->|confirmed static object| H[Exclude + re-merge<br/><sub>from source, never in-place</sub>]
    G -->|real subject| F
    H --> E

    style C fill:#2e1065,stroke:#a78bfa,color:#f8fafc
    style I fill:#0c2f4a,stroke:#38bdf8,color:#f8fafc
    style F fill:#1e293b,stroke:#38bdf8,color:#f8fafc
```

Six stages, one video → one CVAT task:

1. **Detect** — GPU pass with an ONNX YOLO model, multi-scale tiling for
   small/distant subjects, checkpointed to disk every frame so a crash or a
   reboot costs you nothing.
2. **Track** — constant-velocity + IoU + jersey-colour association, stitched
   across occlusion gaps with the Hungarian algorithm.
3. **Recover** — the part that gives this its name. Every gap gets one more
   real shot at being found before it's allowed to stay a ghost.
4. **Re-identify** — short-range tracking alone still gives every longer
   occlusion a brand-new ID. This stage fingerprints each track by
   appearance and merges the ones that are almost certainly the same real
   player, cutting fragment count substantially — see
   [below](#cutting-down-duplicate-ids) for what this does and doesn't
   solve.
5. **Merge & upload** — clip-local frame numbers → your CVAT task's global
   numbering, split at job boundaries, `PUT` straight to the API.
6. **QA** — a heuristic flags *candidates* for static-object false positives
   (never auto-deletes), you visually confirm in the CVAT UI, and only
   confirmed objects get excluded — from a fresh rebuild, not an in-place
   edit.

## 🚀 Install

FieldGhost ships as a **Claude Code skill** (works the same way for any
agent that reads `SKILL.md`-style instructions, Codex included).

```bash
git clone https://github.com/EminSgl/fieldghost.git
cp -r fieldghost/skill ~/.claude/skills/fieldghost
pip install -r ~/.claude/skills/fieldghost/requirements.txt
```

Then just talk to your agent:

> "Track the players in this rugby match and upload it to my CVAT task."

It'll ask you for what it needs (CVAT host, task/job ids, label id, your
clips, a detector model) — see [`skill/SKILL.md`](skill/SKILL.md) for the
full checklist it works from, or fill in
[`skill/config.example.yaml`](skill/config.example.yaml) yourself and hand
it over directly.

No Claude/Codex, just Python? All five stages are plain, dependency-light
scripts under `skill/scripts/` — run them by hand, same flags, same config
file. `SKILL.md` doubles as the operator's manual.

## The ghost, made visible

Every recovered box remembers what it used to be. Set a mutable attribute
in your config and every shape gets tagged `detected` or
`unresolved_detection_gap` — filter for the second one in the CVAT UI and
you get a punch list of exactly the handful of frames worth a human glance,
out of however many thousand your match had.

```yaml
ghost_attribute:
  spec_id: 26
  resolved_value: "detected"
  unresolved_value: "unresolved_detection_gap"
```

## Cutting down duplicate IDs

Short-range tracking alone gives you a smooth track — right up until
someone disappears into a ruck for eight seconds. Come back out the other
side and, to the tracker, that's a brand-new person: on real match footage,
expect on the order of 10-30x more raw tracks than there are actual players
on the pitch.

`embed.py` fingerprints every track by appearance (any image-embedding
model works — a real person-re-ID network gives noticeably better results
than a generic ImageNet classifier's output layer), and `merge_upload.py
--reid` merges the ones that are almost certainly the same real player —
provided their time ranges don't overlap, since nothing can be in two
places on the pitch at once — into a single persistent track:

```bash
# a real person-re-ID model (recommended) -- e.g. Intel Open Model Zoo's
# person-reidentification-retail-0288, an OpenVINO IR (.xml + .bin)
python embed.py clip.mp4 clip_tracks.json clip_embeddings.json \
    --model person-reidentification-retail-0288.xml \
    --width 128 --height 256 --scale 1 --mean 0 0 0 --std 1 1 1

python merge_upload.py config.yaml --clips-dir . --out-dir merged --reid --player-id-spec 27
```

**What this actually gets you, measured on real match footage:** 785 raw
fragment tracks came down to somewhere in the 500s at a conservative
threshold — a real reduction, not the ~30 real players on the pitch. Push
the threshold looser and the count keeps dropping, but a new problem shows
up instead: individual identities start silently absorbing dozens of
different people, worst with teammates in identical kit. There's no
threshold that gives you both a small count *and* correct identities from
appearance alone — this is an honest limit of similarity-based re-ID on
footage a retail-trained model was never tuned for, not a bug to tune away.

Treat the output as "meaningfully fewer, cleaner tracks that still want a
human glance in the CVAT UI," not "solved." Getting the rest of the way to
one ID per real player reliably needs a re-ID model actually trained on
this sport, or reading the printed jersey number instead of inferring
identity from appearance — both real follow-on projects, not something
this stage fakes silently. Full writeup, including why the matching uses
complete-linkage instead of a running average (chaining is a real trap
here), in [`skill/SKILL.md`](skill/SKILL.md#4-re-identify-players-across-the-whole-match-scriptsembedpy--scriptsreid_mergepy).

## Team classification

Separate from re-ID on purpose: telling the same person apart across gaps
and telling two *different* people apart by kit color are opposite jobs.
`colorfeat.py` extracts a cheap hue/saturation feature from the jersey
region (pitch green and lighting extremes masked out), and
`merge_upload.py --classify-roles` k-means clusters it into `team_a` /
`team_b` / `uncertain` — k=3, not k=2, so a third genuinely distinct kit
color (a referee, most often) has somewhere to go instead of silently
joining whichever team centroid is nearest.

It deliberately stops at `uncertain` rather than guessing `referee` vs
`staff`: tested on real rugby footage, referees are commonly in all-black or
another muted color that isn't reliably distinguishable from a team in a
similar dark kit by color alone, and `uncertain` in practice is dominated by
ambiguous crops (motion blur, tackle pile-ups), not clean official shots.

## The honest part

The QA pass includes a heuristic that flags *candidate* false positives —
static things like corner flags or touchline poles that get mislabeled as
people because, to a detector, a thin motionless object and a real subject
standing still for a few seconds look the same. We tested it on real match
footage. About a third of its "confident" candidates were actually real
players standing around during a stoppage.

So it doesn't auto-delete. It prints a list, a sample frame for each, and
tells you exactly where to look. You (or your agent, with a browser)
confirm each one visually before anything gets removed — and removal always
rebuilds from the untouched source tracks, never edits a file in place. Get
a box wrong, fix it, rerun; nothing was ever at risk.

We think a pipeline that's upfront about where it can be wrong is more
useful than one that quietly can't be — see
[`skill/SKILL.md`](skill/SKILL.md#5-qa-find-and-remove-false-positive-ghost-object-labels)
for the full writeup.

## Gotchas this skill already knows about

So your agent doesn't have to rediscover them:

- CVAT is usually behind **Cloudflare** — no User-Agent, no response.
  Handled in `scripts/cvat_client.py`.
- CVAT tokens **go stale** between sessions. Upload raises a clear
  `TokenExpired` instead of a confusing 403.
- Mutable attributes belong on the **shape**, not the track, or CVAT
  silently drops them.
- `PUT /annotations` is a **full replace**, not a merge — which is exactly
  why every stage here is rebuild-from-source, not edit-in-place.
- Long GPU runs get **checkpointed**, not restarted, after a crash or reboot.
- An exclusion box only drops a track that stays inside it for its **entire**
  measured lifetime, never just a track whose *average* position lands
  inside it — otherwise a moving subject whose path merely crosses a pole's
  screen position gets wrongly excluded whole.

Full detail in [`skill/references/cvat-api-notes.md`](skill/references/cvat-api-notes.md).

## Requirements

- [CVAT](https://github.com/cvat-ai/cvat) task already created, video
  uploaded, split into jobs
- A YOLO-family ONNX model with NMS baked in (e.g. a `yolov7-nms-*.onnx`
  export)
- Python 3.10+, ideally with `onnxruntime-gpu` + a CUDA GPU (CPU works, just
  slower)

See [`skill/requirements.txt`](skill/requirements.txt).

## Contributing

Issues and PRs welcome — especially other detector backends, other
tracking association strategies, or a smarter static-object classifier than
"ask a human to look." If you use FieldGhost on a sport this wasn't tested
on, we'd love to hear how it went.

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free to use, modify, and share for
any noncommercial purpose (personal projects, research, education, hobby
use). Commercial use requires a separate license from the copyright holder —
open an issue or reach out if that's what you need.
