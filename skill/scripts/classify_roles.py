"""
Team classification: assigns each track `team_a`, `team_b`, or `uncertain`
from the kit-color feature colorfeat.py computed, using the standard
sports-analytics approach -- unsupervised clustering on jersey color -- that
published team-sport tracking pipelines report ~90%+ accuracy with.

k=3 on ALL tracks, not k=2: with only two clusters, a third genuinely
distinct color (a referee in a kit that matches neither team -- tested on
real footage with a blue-kit referee alongside a dark and a light team) has
nowhere to go but whichever team centroid is nearest, so it silently joins a
team instead of standing out. A third cluster gives a real third color
somewhere to land. The two LARGEST of the three clusters become `team_a`/
`team_b` (on a normal match, most tracks are players, split roughly evenly
between two sides, so the two biggest clusters are essentially guaranteed to
be the teams); the smallest becomes `uncertain` rather than guessed at
further -- see below for why this doesn't try to call it `referee`.

On top of the 3-way split, anything far from its OWN assigned centroid --
further than a robust "how far is genuinely unusual" cutoff (median + k*MAD,
not a fixed percentile, which would carve off the same fraction of tracks
regardless of whether the real non-player count is 2 or 20) -- is also
pulled into `uncertain`, catching outliers within a nominally-team cluster
too (a misdetected static object, a motion-blurred tackle pile-up).

Tried and deliberately NOT doing here: splitting `uncertain` further into
`referee` vs `staff`. The obvious idea -- referees wear a bright,
high-visibility color, staff wear plain sportswear -- does not hold up on
real rugby footage tested during development: rugby referees are commonly
in all-black or another muted color, which is not reliably distinguishable
from a team playing in a similar dark kit using color alone, and on
inspection the `uncertain` bucket was dominated by ambiguous crops (motion
blur, tackle pile-ups, a misdetected static object) rather than genuine
match officials. Rather than ship a `referee` label that is usually wrong,
this stops at `uncertain` and leaves that bucket for a human to sort in the
CVAT UI -- consistent with this project's stance elsewhere (ghost boxes,
static-object QA) that an honest "not sure" beats a confident guess.

`team_a` / `team_b` are arbitrary labels when run with no color hint --
nothing in a single match's footage tells you which cluster is the "home"
side. If the user knows and cares which cluster is which real team, remap
the two labels once, after looking at a few examples of each.
"""
import numpy as np
from scipy.cluster.vq import kmeans2


def classify(items, feature_of, outlier_k=3.5, seed=0):
    """
    items: list of anything (tracks, identities, ...).
    feature_of: callable(item) -> {"hue_cos","hue_sin","sat","val","n"} or None.
    Returns (roles, stats) where roles[i] is "team_a" / "team_b" / "uncertain"
    / "unknown" (no usable color reading) for items[i].
    """
    feats, idxs = [], []
    for i, it in enumerate(items):
        f = feature_of(it)
        if f is None:
            continue
        feats.append([f["hue_cos"], f["hue_sin"], f["sat"]])
        idxs.append(i)

    roles = ["unknown"] * len(items)
    if len(feats) < 4:
        return roles, {"classified": 0, "unknown": len(items), "note": "too few tracks with a color reading to cluster"}

    feats = np.asarray(feats, dtype=float)  # hue_cos, hue_sin, sat -- val (brightness) left out, too lighting-dependent
    centroids, labels = kmeans2(feats, k=3, seed=seed, minit="++")

    counts = np.bincount(labels, minlength=3)
    order = np.argsort(-counts)  # largest first
    cluster_role = {order[0]: "team_a", order[1]: "team_b", order[2]: "uncertain"}

    dist_to_own = np.linalg.norm(feats - centroids[labels], axis=1)
    median = np.median(dist_to_own)
    mad = np.median(np.abs(dist_to_own - median)) or 1e-6
    threshold = median + outlier_k * mad * 1.4826  # 1.4826 makes MAD ~= std for a normal distribution
    is_far_outlier = dist_to_own > threshold

    n_team_a = n_team_b = n_uncertain = 0
    for row, i in enumerate(idxs):
        role = "uncertain" if is_far_outlier[row] else cluster_role[labels[row]]
        roles[i] = role
        if role == "team_a":
            n_team_a += 1
        elif role == "team_b":
            n_team_b += 1
        else:
            n_uncertain += 1

    stats = {
        "classified": len(idxs), "unknown": len(items) - len(idxs),
        "team_a": n_team_a, "team_b": n_team_b, "uncertain": n_uncertain,
        "outlier_threshold": float(threshold),
    }
    return roles, stats
