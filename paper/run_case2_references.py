"""Every area, random and grouping reference quoted for Case 2 in Section 5.2.

One script, one JSON, so that no number in the Case-2 text is held only in
prose. Covers:

  a(250) for the combined ranking and each baseline
  the random selection's expected area, expected recovery and mean hits at
    rank 250, and the share of draws that match the combined ranking
  the budget windows that overlap the reference deposit, with their ranks
  the spatially distinct groups among the 250, and how many touch no deposit
  per-deposit recovery of the combined ranking and of each single variable

Run from inside paper/:

    python run_case2_references.py

Writes results/case2_references.json.
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
N_DRAW = 2000
SEED = 20261123

R.deposits_wgs84 = C1.deposits_case2_wgs84


def clusters(order, idx, wshape, dep, deposits, origin, n_top=R.N_TOP):
    """Distinct groups among the top windows, and those touching no deposit.

    Two windows join when their rectangles overlap, which on a stride grid is
    |dr| < window height and |dc| < window width.
    """
    wh, ww = wshape
    sel = [int(i) for i in order[order != dep][:n_top]]
    rc = np.array([[idx[i][0], idx[i][1]] for i in sel])
    parent = list(range(len(rc)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(rc)):
        near = np.nonzero((np.abs(rc[:, 0] - rc[i, 0]) < wh)
                          & (np.abs(rc[:, 1] - rc[i, 1]) < ww))[0]
        for j in near[near > i]:
            ra, rb = find(i), find(int(j))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for i in range(len(rc)):
        groups.setdefault(find(i), []).append(i)

    rects = NE.rects_from_idx(idx, wshape, origin)
    clear = 0
    for members in groups.values():
        touches = any(R.rings_clip_area(rings, *rects[sel[i]]) > 1e-9
                      for i in members for rings in deposits)
        if not touches:
            clear += 1
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    return len(sizes), sizes, clear


def main() -> int:
    tmi, _u, _ox, _oy = R.load_cropped()
    grid, valid = tmi.shape, int(np.isfinite(tmi).sum())
    deposits = R.deposits_wgs84()

    order, idx, wshape, origin, dep = R.rank_concat(REF, K_TMI, K_U, R.ALPHA_TMI)
    n = len(idx)
    rects = NE.rects_from_idx(idx, wshape, origin)
    cells, areas = NE.build_cells(deposits, REF - 1, rects)

    rankings = {
        "combined": order,
        "TMI alone": R.rank_concat(REF, K_TMI, K_U, 1.0)[0],
        "radiometric U alone": R.rank_concat(REF, K_TMI, K_U, 0.0)[0],
        "raw multi": R.rank_raw(REF)[0],
    }

    out = {"alpha_TMI": R.ALPHA_TMI, "k_TMI": K_TMI, "k_U": K_U,
           "n_windows": int(n - 1), "n_test_deposits": len(deposits) - 1,
           "study_area_valid_px": valid, "draws": N_DRAW}

    # The two blocks' distance distributions, which the scaling of step 5.1
    # answers to. Reported before any scaling is applied.
    X = R.window_matrix(REF)[0]
    npix = wshape[0] * wshape[1]
    dT = R.block_distance(R.block_scores(X[:, :npix], K_TMI), dep)
    dU = R.block_distance(R.block_scores(X[:, npix:2 * npix], K_U), dep)
    out["block_distances"] = {
        v: {"mean": round(float(d.mean()), 1), "median": round(float(np.median(d)), 2),
            "max": round(float(d.max()), 1)}
        for v, d in (("TMI", dT), ("radiometric U", dU))}

    out["rankings"] = {}
    for name, o in rankings.items():
        res = R.footprint_recovery(o, idx, wshape, origin, dep, REF)
        m = R.metrics(res)
        ua = NE.union_area_curve(o, idx, wshape, dep, grid)
        out["rankings"][name] = {
            "c250": round(float(m["end"]), 1),
            "auc": round(float(m["auc"]), 1),
            "hits": int(m["hits"]),
            "hit_ranks": {str(k): [int(t) + 1 for t in v] for k, v in m["hit_ranks"].items()},
            "a250": round(100 * float(ua[-1]) / valid, 1),
            "coverage_pct": {str(i + 1): round(100 * v, 1)
                             for i, v in sorted(res["coverage_by_deposit"].items())},
        }

    # --- random selection of 250 windows, at equal rank
    rand_curve = NE.exact_random_curve(cells, areas, n, t_max=R.N_TOP)
    rand_union = NE.exact_random_union(idx, wshape, n, grid, t_max=R.N_TOP)
    out["random_at_rank250"] = {
        "expected_recovery_pct": round(100 * float(rand_curve[-1]), 1),
        "expected_a250_pct": round(100 * float(rand_union[-1]) / valid, 1),
    }

    # --- Monte-Carlo: mean hits, and the share of draws matching the ranking
    rng = np.random.default_rng(SEED)
    pool = np.arange(n)
    pool = pool[pool != dep]
    target_c = out["rankings"]["combined"]["c250"] / 100.0
    target_h = out["rankings"]["combined"]["hits"]
    hits_d, reach_c, reach_h = [], 0, 0
    for _ in range(N_DRAW):
        pick = rng.choice(pool, size=R.N_TOP, replace=False)
        res = R.footprint_recovery(np.concatenate([pick, [dep]]), idx, wshape,
                                   origin, dep, REF)
        cov = res["coverage_by_deposit"]
        cbar = float(np.mean([cov.get(i, 0.0) for i in range(len(deposits))
                              if i != REF - 1]))
        h = sum(1 for i, v in cov.items() if v >= R.MIN_COVER)
        hits_d.append(h)
        reach_c += cbar >= target_c - 1e-12
        reach_h += h >= target_h
    out["random_monte_carlo"] = {
        "mean_hits": round(float(np.mean(hits_d)), 1),
        "share_reaching_c250_pct": round(100.0 * reach_c / N_DRAW, 2),
        "share_reaching_hits_pct": round(100.0 * reach_h / N_DRAW, 2),
    }

    # --- the reference deposit's own ground inside the budget
    ref_ranks = NE.budget_ranks_overlapping(order, idx, wshape, origin, dep,
                                            deposits[REF - 1])
    out["budget_windows_overlapping_reference"] = {
        "n": len(ref_ranks), "ranks": [int(r) for r in ref_ranks]}

    ng, sizes, clear = clusters(order, idx, wshape, dep, deposits, origin)
    out["distinct_groups"] = {"n": ng, "largest": sizes[0],
                              "touching_no_known_deposit": clear,
                              "sizes": sizes[:8]}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "case2_references.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nwrote %s" % (OUT_DIR / "case2_references.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
