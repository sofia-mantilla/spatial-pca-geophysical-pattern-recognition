"""Is the multivariate result an artefact of its smaller test set? (R2.4)

Case 2 is scored against the five deposits of the multivariate shapefile,
which are the ones with usable radiometric coverage, while Case 1 is scored
against all twelve. Reviewer 2 asks whether that smaller set flatters the
multivariate ranking.

The ranking is not touched. The same published model -- reference Alemao,
k_TMI=2, k_U=8, alpha_TMI=0.503 -- is scored a second time against every
deposit of the twelve-deposit shapefile, so the only thing that changes is
which footprints count as a success. Alemao is deposit 3 in both shapefiles,
with identical outlines, so it is the reference and excluded either way.

Run from inside paper/:

    python run_all_twelve_case2.py

Writes results/all_twelve_case2.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import case1_uni_repro as C1
import null_expectation as NE
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
REF = 3                       # Alemao, in both shapefiles
K_TMI, K_U, ALPHA = 2, 8, 0.503
T_MAX = 6000
PUBLISHED = {"end": 63.7, "hits": 3}


def score(order, idx, wshape, origin, dep, deposits_fn, label):
    """Recovery and the equal-area random expectation against one test set."""
    old = R.deposits_wgs84
    R.deposits_wgs84 = deposits_fn
    try:
        res = R.footprint_recovery(order, idx, wshape, origin, dep, REF)
        deps = deposits_fn()
        tmi, _u, ox, oy = R.load_cropped()
        rects = NE.rects_from_idx(idx, wshape, origin)
        cells, areas = NE.build_cells(deps, REF - 1, rects)
        curve = NE.exact_random_curve(cells, areas, len(idx), t_max=T_MAX)
        union_rand = NE.exact_random_union(idx, wshape, len(idx), tmi.shape, t_max=T_MAX)
        union = NE.union_area_curve(order, idx, wshape, dep, tmi.shape)
    finally:
        R.deposits_wgs84 = old

    ranks = np.arange(1, T_MAX + 1)
    t_star = float(np.interp(union[-1], union_rand, ranks))
    expected = float(np.interp(t_star, ranks, curve))
    m = R.metrics(res)
    return {
        "test_set": label,
        "n_test_deposits": len(deps) - 1,
        "end": round(float(m["end"]), 1),
        "auc": round(float(m["auc"]), 1),
        "hits": int(m["hits"]),
        "hit_ranks": {str(k): v for k, v in m["hit_ranks"].items()},
        "a250": round(float(union[-1] / np.isfinite(tmi).sum()), 4),
        "random_matched": round(100 * expected, 1),
        "ratio": round(float(m["end"] / 100.0 / expected), 2) if expected > 0 else None,
    }


def main() -> int:
    order, idx, wshape, origin, dep = R.rank_concat(REF, K_TMI, K_U, ALPHA)

    rows = [
        score(order, idx, wshape, origin, dep, R.deposits_wgs84,
              "the five of the multivariate shapefile (as published)"),
        score(order, idx, wshape, origin, dep, C1.deposits_case1_wgs84,
              "all twelve known deposits"),
    ]

    # The same reference and the same windows on one variable only, so that
    # the test-set size cancels: alpha=1 is TMI alone, alpha=0 uranium alone.
    for label, alpha in (("TMI alone", 1.0), ("radiometric U alone", 0.0)):
        o, i, ws, org, d = R.rank_concat(REF, K_TMI, K_U, alpha)
        rows.append(score(o, i, ws, org, d, R.deposits_wgs84,
                          "%s, scored on the five" % label))
        rows.append(score(o, i, ws, org, d, C1.deposits_case1_wgs84,
                          "%s, scored on all twelve" % label))

    published = rows[0]
    if (abs(published["end"] - PUBLISHED["end"]) > 0.15
            or published["hits"] != PUBLISHED["hits"]):
        raise SystemExit("the published Case-2 result did not reproduce: %r" % published)
    print("published Case 2 reproduced: %.1f%%, %d hits\n" % (
        published["end"], published["hits"]))

    width = max(len(r["test_set"]) for r in rows)
    print("%-*s %5s %8s %6s %8s %8s" % (width, "scored against", "n", "c(250)",
                                        "hits", "random", "ratio"))
    for r in rows:
        print("%-*s %5d %7.1f%% %6d %7.1f%% %7.2f"
              % (width, r["test_set"], r["n_test_deposits"], r["end"],
                 r["hits"], r["random_matched"], r["ratio"]))
    for r in rows:
        print("\n%s\n  hits at ranks %s" % (r["test_set"], r["hit_ranks"]))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "all_twelve_case2.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote %s" % (OUT_DIR / "all_twelve_case2.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
