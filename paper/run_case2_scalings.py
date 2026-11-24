"""How the Case-2 result depends on the scale each block's distance is divided by.

The two blocks' distances are not commensurable, so a weighted average of the
raw values is governed by whichever block returns the larger numbers. This
script scores the combined ranking under each of the usual scale estimators,
and under no scaling at all, on the five Case-2 test deposits.

The median is adopted in the paper as the standard robust scale for a skewed
distribution. It is not the highest scorer here, and that is the point: the
choice is made on the estimator and not on the result.

Run from inside paper/:

    python run_case2_scalings.py

Writes results/case2_scalings.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import case1_uni_repro as C1
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
REF, K_TMI, K_U = 3, 2, 8

R.deposits_wgs84 = C1.deposits_case2_wgs84


def main() -> int:
    X, idx, wshape, origin = R.window_matrix(REF)
    dep = X.shape[0] - 1
    npix = wshape[0] * wshape[1]
    Z1 = R.block_scores(X[:, :npix], K_TMI)
    Z2 = R.block_scores(X[:, npix:2 * npix], K_U)
    dT = R.block_distance(Z1, dep)
    dU = R.block_distance(Z2, dep)
    a = R.ALPHA_TMI

    def top250(order):
        return set(int(w) for w in order[order != dep][:R.N_TOP])

    topT, topU = top250(np.argsort(dT)), top250(np.argsort(dU))

    def score(order):
        res = R.footprint_recovery(order, idx, wshape, origin, dep, REF)
        m = R.metrics(res)
        t = top250(order)
        return {"c250": round(float(m["end"]), 1), "auc": round(float(m["auc"]), 1),
                "hits": int(m["hits"]),
                "hit_ranks": {str(k): [int(t_) + 1 for t_ in v]
                              for k, v in m["hit_ranks"].items()},
                "shared_with_TMI_alone": len(t & topT),
                "shared_with_U_alone": len(t & topU)}

    iqr = lambda d: float(np.subtract(*np.percentile(d, [75, 25])))
    scalings = {
        "median": (np.median(dT), np.median(dU)),
        "interquartile range": (iqr(dT), iqr(dU)),
        "mean": (dT.mean(), dU.mean()),
        "standard deviation": (dT.std(), dU.std()),
        "maximum": (dT.max(), dU.max()),
    }
    out = {"alpha_TMI": a, "n_test_deposits": len(R.deposits_wgs84()) - 1,
           "block_distances": {
               v: {"mean": round(float(d.mean()), 1),
                   "median": round(float(np.median(d)), 2),
                   "iqr": round(iqr(d), 2), "max": round(float(d.max()), 1)}
               for v, d in (("TMI", dT), ("radiometric U", dU))},
           "scalings": {}}

    for name, (sT, sU) in scalings.items():
        out["scalings"][name] = score(np.argsort(a * dT / sT + (1 - a) * dU / sU))

    # a rank transform in place of a scale factor
    out["scalings"]["percentile rank"] = score(
        np.argsort(a * R.block_prank(dT) + (1 - a) * R.block_prank(dU)))

    # the two unscaled references
    out["unscaled"] = {
        "weighted average of the raw distances": score(
            np.argsort(a * dT + (1 - a) * dU)),
        "weighted sum of the squared distances": score(
            R.rank_concat(REF, K_TMI, K_U, a, scale=None)[0]),
    }
    out["single_variables"] = {"TMI alone": score(np.argsort(dT)),
                               "radiometric U alone": score(np.argsort(dU))}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "case2_scalings.json").write_text(json.dumps(out, indent=1))

    print("block distances (before any scaling):")
    for v, d in out["block_distances"].items():
        print("  %-14s mean %6.1f  median %6.2f  IQR %6.2f  max %6.1f"
              % (v, d["mean"], d["median"], d["iqr"], d["max"]))
    print("\nalpha_TMI = %.3f, %d test deposits" % (a, out["n_test_deposits"]))
    hdr = ("scale applied to each block", "c(250)", "AUC", "hits", "w/ TMI", "w/ U")
    print("\n  %-44s %8s %7s %5s %7s %6s" % hdr)
    print("  %s" % ("(the last two columns are how many of the top 250 the ranking "
                    "shares with each single variable)"))

    def line(name, m, mark=""):
        print("  %-44s %7.1f%% %7.1f %5d %7d %6d%s"
              % (name, m["c250"], m["auc"], m["hits"],
                 m["shared_with_TMI_alone"], m["shared_with_U_alone"], mark))

    for name, m in out["scalings"].items():
        line(name, m, "  <- adopted" if name == "median" else "")
    for name, m in out["unscaled"].items():
        line("none: " + name, m)
    print()
    for name, m in out["single_variables"].items():
        line(name, m)
    print("\nwrote %s" % (OUT_DIR / "case2_scalings.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
