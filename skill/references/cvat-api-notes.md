# CVAT API notes and gotchas

Load this file when you hit an auth error, an upload rejection, or need to
look up an id (label, job, task) before running a script.

## Cloudflare (403 with no CVAT error body)

Most hosted CVAT instances sit behind Cloudflare. Any HTTP client with a
default or missing `User-Agent` gets a 403 (Cloudflare error code 1010)
before the request ever reaches CVAT's own auth layer -- the response body
won't look like a CVAT error at all, it's Cloudflare's own block page/JSON.
`scripts/cvat_client.py` always sends a real browser User-Agent for this
reason. If you write ad hoc requests yourself (e.g. to look something up
with `curl`), add `-A "Mozilla/5.0 ..."` or you'll hit the same wall.

A non-standard `Accept` header can also get a 406 from CVAT itself (not
Cloudflare) -- don't set `Accept: application/json` explicitly on the login
call; let the client's default apply.

## Auth tokens go stale

A token that worked in a previous session can 403 on `PUT
/api/jobs/<id>/annotations` today with no other symptom -- CVAT sessions/
tokens don't last forever. `put_annotations` in `cvat_client.py` raises
`TokenExpired` on any 403 specifically so you can catch it and re-authenticate
rather than treating it as a generic failure. Get a fresh token with:

```
POST /api/auth/login
{"username": "...", "password": "..."}
-> {"key": "<40-char token>"}
```

Then use `Authorization: Token <key>` on subsequent requests.

## Finding ids you need before running the pipeline

- **Label id**: `GET /api/labels?task_id=<id>` (or open the task in the UI,
  Actions > Labels, and inspect the network request).
- **Attribute spec ids**: same response, under each label's `attributes`
  list, each with its own `id`.
- **Job ids and frame windows**: `GET /api/tasks/<id>/jobs` lists jobs; each
  job's own `GET /api/jobs/<id>` response has `start_frame`/`stop_frame`.
- **Task frame_filter step**: `GET /api/tasks/<id>` -> look at how the task
  was created, or infer it from `stop_frame - start_frame` vs the source
  video's actual frame count if unsure.

## Mutable attributes must live on the shape, not the track

CVAT tracks have two places attributes can go: the track's own top-level
`attributes` list (for values that don't change over the track's lifetime,
e.g. jersey number) and each individual shape's `attributes` list (for
values that CAN change per-frame, e.g. "is this box currently an estimate or
a real detection"). A **mutable** attribute (marked as such on the label in
CVAT) put on the track instead of the shape gets silently dropped or
ignored on upload -- no error, it just won't show up. `merge_upload.py`'s
`ghost_attribute` handling puts the value on every shape for exactly this
reason; if you add your own per-frame-varying attribute, do the same.

## PUT is a full replace

`PUT /api/jobs/<id>/annotations` replaces the ENTIRE annotation set for that
job with whatever you send -- it is not a merge or a patch. If you want to
add to what's already there, `GET` first, merge in your code, then `PUT` the
combined result. This is also why `merge_upload.py` always rebuilds its
payload from the per-clip track files rather than trying to diff against
what's already on the server: a full-replace endpoint punishes drift between
your local state and the server's state, so don't let there be any -- always
regenerate, never patch blind.
