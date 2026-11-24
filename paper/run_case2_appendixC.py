"""Appendix C Case-2 tables: weight ablation, classical baselines, joint PCA.

Every Case-2 row of Tables "weights", "baselines" and the joint-decomposition
paragraph, each with c(250), AUC, hits, a(250) and the ratio to a random
selection covering the same area. Also the per-deposit overlap counts the
deposit-by-deposit narrative quotes.

Run from inside paper/:

    python run_case2_appendixC.py

Writes results/case2_appendixC.json.
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
REF, K_TMI, K_U = 3, 2, 8
T = 6000

R.deposits_wgs84 = C1.deposits_case2_wgs84


def scores_full(Xb, k):
    sd = Xb.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xb - Xb.mean(axis=0)) / sd
    Zc = Z - Z.mean(axis=0)
    U_, S, _ = np.linalg.svd(Zc, full_matrices=False)
    return (U_ * S)[:, :k], (S ** 2)[:k]


def demeaned(Xb):
    return Xb - Xb.mean(axis=1, keepdims=True)


def ncc(Xb, d):
    Xc = demeaned(Xb)
    nrm = np.linalg.norm(Xc, axis=1)
    nrm = np.where(nrm == 0, 1.0, nrm)
    Xn = Xc / nrm[:, None]
    return Xn @ Xn[d]


def main() -> int:
    tmi, _u, _ox, _oy = R.load_cropped()
    grid, valid = tmi.shape, int(np.isfinite(tmi).sum())
    deposits = R.deposits_wgs84()

    X, idx, wshape, origin = R.window_matrix(REF)
    dep = X.shape[0] - 1
    npix = wshape[0] * wshape[1]
    n = len(idx)

    rects = NE.rects_from_idx(idx, wshape, origin)
    cells, areas = NE.build_cells(deposits, REF - 1, rects)
    rc = NE.exact_random_curve(cells, areas, n, t_max=T)
    ru = NE.exact_random_union(idx, wshape, n, grid, t_max=T)
    ranks = np.arange(1, T + 1)

    def row(order):
        res = R.footprint_recovery(order, idx, wshape, origin, dep, REF)
        m = R.metrics(res)
        ua = NE.union_area_curve(order, idx, wshape, dep, grid)
        t_star = float(np.interp(ua[-1], ru, ranks))
        em = float(np.interp(t_star, ranks, rc))
        n_win = {}
        for _r, ids in res["overlap_by_rank"].items():
            for i in ids:
                n_win[str(i + 1)] = n_win.get(str(i + 1), 0) + 1
        return {
            "c250": round(float(m["end"]), 1),
            "auc": round(float(m["auc"]), 1),
            "hits": int(m["hits"]),
            "a250": round(100 * float(ua[-1]) / valid, 1),
            "ratio_to_random_equal_area": round(float(m["end"]) / 100.0 / em, 2) if em > 0 else None,
            "coverage_pct": {str(i + 1): round(100 * v, 1)
                             for i, v in sorted(res["coverage_by_deposit"].items())},
            "windows_overlapping_each_deposit": n_win,
        }

    out = {"alpha_TMI": R.ALPHA_TMI, "n_test_deposits": len(deposits) - 1,
           "random_at_rank250_pct": round(100 * float(rc[249]), 1)}

    # ---- weight ablation
    Zt, lamt = scores_full(X[:, :npix], K_TMI)
    Zu, lamu = scores_full(X[:, npix:2 * npix], K_U)
    St, Su = Zt / np.sqrt(lamt), Zu / np.sqrt(lamu)

    def combine(A, B, wt, wu, alpha=R.ALPHA_TMI):
        dt = np.sqrt(((A - A[dep]) ** 2) @ wt)
        du = np.sqrt(((B - B[dep]) ** 2) @ wu)
        return np.argsort(alpha * dt / np.median(dt) + (1 - alpha) * du / np.median(du))

    out["weights"] = {
        "deposit-based z_dm^2": row(combine(Zt, Zu, R.blk_w(Zt[dep]), R.blk_w(Zu[dep]))),
        "uniform": row(combine(Zt, Zu, np.ones(K_TMI) / K_TMI, np.ones(K_U) / K_U)),
        "standardized scores": row(combine(St, Su, R.blk_w(St[dep]), R.blk_w(Su[dep]))),
        "whitened 1/lambda": row(combine(Zt, Zu, (1 / lamt) / (1 / lamt).sum(),
                                         (1 / lamu) / (1 / lamu).sum())),
    }

    # ---- classical template baselines, combined with the alpha_v rule
    a = R.ALPHA_TMI
    nT, nU = ncc(X[:, :npix], dep), ncc(X[:, npix:2 * npix], dep)
    dTc = np.linalg.norm(demeaned(X[:, :npix]) - demeaned(X[:, :npix])[dep], axis=1)
    dUc = np.linalg.norm(demeaned(X[:, npix:2 * npix])
                         - demeaned(X[:, npix:2 * npix])[dep], axis=1)
    out["baselines"] = {
        "wPCA combined": out["weights"]["deposit-based z_dm^2"],
        "correlation match, combined": row(np.argsort(-(a * nT + (1 - a) * nU))),
        "correlation match, U alone": row(np.argsort(-nU)),
        "correlation match, TMI alone": row(np.argsort(-nT)),
        "demeaned match, combined": row(np.argsort(a * dTc / dTc.std()
                                                   + (1 - a) * dUc / dUc.std())),
        "raw multi": row(R.rank_raw(REF)[0]),
    }

    # ---- joint decomposition of the concatenated pixel vector
    joint = {}
    for k in (2, 4, 8, 10, 16, 24, 40):
        Z, _lam = scores_full(X, k)
        w = R.blk_w(Z[dep])
        joint[str(k)] = row(np.argsort(np.sqrt(((Z - Z[dep]) ** 2) @ w)))
    out["joint_decomposition"] = joint

    # ---- the single variables, for the per-deposit table
    out["single_variables"] = {
        "TMI alone": row(R.rank_concat(REF, K_TMI, K_U, 1.0)[0]),
        "radiometric U alone": row(R.rank_concat(REF, K_TMI, K_U, 0.0)[0]),
    }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "case2_appendixC.json").write_text(json.dumps(out, indent=1))

    def show(title, d):
        print("\n%s   (random at rank 250: %.1f%%)" % (title, out["random_at_rank250_pct"]))
        print("  %-30s %8s %7s %5s %8s %7s" % ("", "c(250)", "AUC", "hits", "a(250)", "ratio"))
        for k, v in d.items():
            print("  %-30s %7.1f%% %7.1f %5d %7.1f%% %7.2f"
                  % (k, v["c250"], v["auc"], v["hits"], v["a250"],
                     v["ratio_to_random_equal_area"]))

    show("Weight ablation, Case 2", out["weights"])
    show("Classical template baselines, Case 2", out["baselines"])
    show("Joint decomposition of the concatenated variables", joint)
    show("Single variables", out["single_variables"])
    print("\nPer-deposit coverage and overlapping windows:")
    for lab, v in (("combined", out["weights"]["deposit-based z_dm^2"]),
                   ("TMI alone", out["single_variables"]["TMI alone"]),
                   ("U alone", out["single_variables"]["radiometric U alone"])):
        print("  %-12s coverage %s" % (lab, v["coverage_pct"]))
        print("  %-12s windows  %s" % ("", v["windows_overlapping_each_deposit"]))
    print("\nwrote %s" % (OUT_DIR / "case2_appendixC.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
