"""
Global re-identification matcher: takes every track from every clip (already
placed on the CVAT task's shared frame timeline) plus the appearance
embedding `embed.py` computed for each one, and merges the tracks that are
almost certainly the same real subject into a single CVAT track with one
persistent identity -- so a player who gets re-tracked five times across a
match ends up as one track object in CVAT, not five.

The matching rule is simple on purpose: two tracks can only be the same
subject if their time ranges DON'T overlap (nothing can be in two places on
the pitch at once), and among all non-overlapping candidates, matched to
whichever the appearance embedding is most similar to -- above a similarity
threshold, otherwise treated as a new identity. This is a greedy nearest-
neighbour match processed in time order, which is simpler and much cheaper
than a globally-optimal assignment, and works well because in practice each
track has few plausible candidates (they must not overlap it in time) and
the similarity gap between "this is the same person" and "this is someone
else" is usually large enough that greedy is fine.

This is NOT foolproof, especially for players on the same team in identical
kit -- appearance embeddings from a generic model (see embed.py) are good at
"looks like the same subject" but can be fooled by "looks like the same
team." A dedicated person-re-ID model does meaningfully better here; a
jersey-number OCR pass would do better still. Treat `player_id` as a strong
hint you should spot-check in the CVAT UI on a new dataset, not an
infallible ground truth -- the same honesty principle as the static-object
QA pass in find_static_objects.py: a heuristic that admits its failure mode
is more useful than one that hides it.

Used as a library by merge_upload.py; see that script for how the pieces fit
together (clip-local track index + clip name -> embedding lookup -> this).
"""
import numpy as np


def track_time_range(track):
    frames = [s["frame"] for s in track["shapes"] if not s["outside"]]
    return (min(frames), max(frames)) if frames else (0, -1)


def _overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def _cosine(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom > 0 else 0.0


def _sorted_shapes(track):
    return sorted(track["shapes"], key=lambda s: s["frame"])


def build_merged_track(members, player_id, player_id_spec=None):
    """Concatenate several fragment tracks (same identity, non-overlapping
    time ranges) into one CVAT track, closing each fragment's visibility
    with an `outside` shape so CVAT knows the subject disappears between
    fragments instead of interpolating a straight line across the gap."""
    members = sorted(members, key=lambda m: track_time_range(m)[0])
    shapes = []
    for i, m in enumerate(members):
        msh = _sorted_shapes(m)
        if msh and msh[-1]["outside"]:
            msh = msh[:-1]
        if not msh:
            continue
        shapes.extend(msh)
        if i + 1 < len(members):
            shapes.append({**msh[-1], "frame": msh[-1]["frame"] + 1, "outside": True})
    attrs = list(members[0].get("attributes", []))
    if player_id_spec is not None:
        attrs = [a for a in attrs if a["spec_id"] != player_id_spec]
        attrs.append({"spec_id": player_id_spec, "value": str(player_id)})
    return {
        "frame": shapes[0]["frame"], "label_id": members[0]["label_id"], "group": 0,
        "source": "auto", "attributes": attrs, "shapes": shapes,
    }


def match_and_merge(tracks, embedding_of, threshold=0.75, player_id_spec=None, max_exemplars=12):
    """
    tracks: list of CVAT track dicts (already on the shared timeline).
    embedding_of: callable(track) -> vector or None.
    Returns (merged_tracks, stats) where stats reports how much merging happened.

    Matching is complete-linkage, not average-linkage: a candidate track must
    be similar to EVERY exemplar already in an identity (the minimum
    pairwise similarity, not the similarity to a running-average embedding),
    or it doesn't join. This costs a bit more compute but avoids the classic
    failure mode of incremental nearest-neighbour clustering: a chain of
    individually-plausible matches (A~B, B~C, C~D...) can walk an identity's
    running average steadily away from where it started, until it's
    absorbing tracks that don't actually resemble the original member at
    all -- exactly the "one identity ate 100 unrelated fragments" bug this
    guards against. `max_exemplars` caps memory/compute per identity by
    keeping the most recent N members once an identity grows large; losing
    a bit of long-history precision is a fine trade for staying bounded.
    """
    entries = []
    for tr in tracks:
        rng = track_time_range(tr)
        if rng[1] < rng[0]:
            continue
        entries.append({"track": tr, "emb": embedding_of(tr), "range": rng})
    entries.sort(key=lambda e: e["range"][0])

    gallery = []  # [{"embs": [vector, ...], "ranges": [...], "members": [...]}]
    for e in entries:
        best_i, best_sim = None, -1.0
        if e["emb"] is not None:
            for i, g in enumerate(gallery):
                if not g["embs"]:
                    continue
                if any(_overlaps(e["range"], r) for r in g["ranges"]):
                    continue
                sim = min(_cosine(e["emb"], m) for m in g["embs"])
                if sim > best_sim:
                    best_sim, best_i = sim, i
        if best_i is not None and best_sim >= threshold:
            g = gallery[best_i]
            g["members"].append(e["track"])
            g["ranges"].append(e["range"])
            if e["emb"] is not None:
                g["embs"].append(e["emb"])
                if len(g["embs"]) > max_exemplars:
                    g["embs"].pop(0)
        else:
            gallery.append({
                "embs": [e["emb"]] if e["emb"] is not None else [],
                "ranges": [e["range"]], "members": [e["track"]],
            })

    merged = [build_merged_track(g["members"], pid, player_id_spec) for pid, g in enumerate(gallery)]
    groups = [g["members"] for g in gallery]  # merged[i] was built from groups[i] -- same order, same length
    sizes = [len(g["members"]) for g in gallery]
    stats = {
        "input_tracks": len(tracks),
        "output_identities": len(gallery),
        "tracks_without_embedding": sum(1 for e in entries if e["emb"] is None),
        "largest_identity_fragment_count": max(sizes, default=0),
        "identities_with_multiple_fragments": sum(1 for n in sizes if n > 1),
    }
    return merged, groups, stats
