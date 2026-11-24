"""Appendix A: the held-out protocol for the Case-2 choices.

Each of the deposits that pass the uranium threshold serves in turn as the
reference, with the window its own outline sets. Every other deposit of that
set is a held-out test deposit, and for each reference and test pair the
balance weight is computed from the remaining deposits only, so no deposit
influences its own result.

Also runs the 15 x 15 grid of component counts that Appendix B's panel (b) and
the k statements of Section 5.2 read from.

Run from inside paper/:

    python run_case2_heldout.py [--grid]

Writes results/case2_heldout.json and, with --grid,
results/case2_heldout_grid.json and appendix_b/lodo_grid_by_k.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

import case1_uni_repro as C1
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
REFS = [1, 2, 3, 4, 5, 6]         # the deposits passing the uranium threshold
TESTS = [1, 2, 3, 4, 5, 6]
KMAX = 40
N_BOOT = 20000
SEED = 20260715

R.deposits_wgs84 = C1.deposits_case2_wgs84
log = lambda *a: (print(time.strftime("[%H:%M:%S]"), *a), sys.stdout.flush())
CACHE: dict = {}
MEMO: dict = {}


def prep(dep):
    if dep in CACHE:
        return CACHE[dep]
    X, idx, wshape, origin = R.window_matrix(dep)
    npix = wshape[0] * wshape[1]

    def sc(Xb):
        sd = Xb.std(axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = (Xb - Xb.mean(axis=0)) / sd
        Zc = Z - Z.mean(axis=0)
        U_, S, _ = np.linalg.svd(Zc, full_matrices=False)
        return np.ascontiguousarray((U_ * S)[:, :KMAX])

    CACHE[dep] = dict(Z1=sc(X[:, :npix]), Z2=sc(X[:, npix:2 * npix]), idx=idx,
                      wshape=wshape, origin=origin, dep_index=X.shape[0] - 1)
    R.window_matrix.cache_clear()
    return CACHE[dep]


def coverage(dep, k1, k2, alpha, scale=R.BLOCK_SCALE):
    key = (dep, k1, k2, round(float(alpha), 6), scale)
    if key in MEMO:
        return MEMO[key]
    c = prep(dep)
    Z1, Z2, di = c["Z1"][:, :k1], c["Z2"][:, :k2], c["dep_index"]
    if scale is None:
        F = np.hstack([Z1, Z2])
        w = np.concatenate([R.blk_w(Z1[di]) * alpha, R.blk_w(Z2[di]) * (1 - alpha)])
        d = np.sqrt(((F - F[di]) ** 2) @ w)
    else:
        d1 = R.block_distance(Z1, di)
        d2 = R.block_distance(Z2, di)
        d = (alpha * d1 / R.block_scale(d1, scale)
             + (1 - alpha) * d2 / R.block_scale(d2, scale))
    res = R.footprint_recovery(np.argsort(d), c["idx"], c["wshape"], c["origin"],
                               di, dep)
    MEMO[key] = res["coverage_by_deposit"]
    return MEMO[key]


def protocol(k1, k2, scale=R.BLOCK_SCALE, refs=REFS, tests=TESTS):
    """Held-out coverage of every reference and test pair."""
    rows = []
    for h in refs:
        cT = coverage(h, k1, k2, 1.0, scale)
        cU = coverage(h, k1, k2, 0.0, scale)
        c5 = coverage(h, k1, k2, 0.5, scale)
        for t in tests:
            if t == h:
                continue
            others = [s for s in tests if s not in (h, t)]
            pT = float(np.mean([cT.get(s - 1, 0.0) for s in others]))
            pU = float(np.mean([cU.get(s - 1, 0.0) for s in others]))
            al = pT / (pT + pU) if pT + pU > 0 else 0.5
            rows.append(dict(
                h=h, t=t, alpha=round(al, 4),
                a_uni=float(coverage(h, k1, k2, al, scale).get(t - 1, 0.0)),
                fixed=float(c5.get(t - 1, 0.0)),
                U=float(cU.get(t - 1, 0.0)),
                TMI=float(cT.get(t - 1, 0.0))))
    return rows


def summarise(rows, tests=TESTS, n_boot=N_BOOT, seed=SEED):
    means = {k: float(np.mean([r[k] for r in rows]))
             for k in ("a_uni", "fixed", "U", "TMI")}
    best = "U" if means["U"] >= means["TMI"] else "TMI"
    d = np.array([r["a_uni"] - r[best] for r in rows])
    t_of = np.array([r["t"] for r in rows])
    present = [t for t in tests if (t_of == t).any()]
    cl = {t: d[t_of == t] for t in present}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([cl[t] for t in rng.choice(present, size=len(present),
                                                       replace=True)]).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cm = np.array([cl[t].mean() for t in present])
    perm = [abs((cm * np.array(sg)).mean())
            for sg in itertools.product([1, -1], repeat=len(present))]
    return dict(
        n_pairs=len(rows), n_clusters=len(present),
        means={k: round(v, 4) for k, v in means.items()},
        best_single=best, advantage=round(float(d.mean()), 4),
        ci95=[round(float(lo), 4), round(float(hi), 4)],
        p_signflip=round(float(np.mean([pp >= abs(cm.mean()) - 1e-12 for pp in perm])), 4),
        p_min_attainable=round(2.0 / 2 ** len(present), 4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="store_true", help="also run the 15 x 15 grid")
    ap.add_argument("--kmax", type=int, default=15)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    out = {"references": REFS, "test_deposits": TESTS,
           "k_TMI": 2, "k_U": 8, "block_scale": R.BLOCK_SCALE}
    rows = protocol(2, 8)
    out["heldout"] = summarise(rows)
    out["pairs"] = [{k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in r.items()} for r in rows]
    s = out["heldout"]
    print("\n(2, 8), %d pairs over %d held-out deposits" % (s["n_pairs"], s["n_clusters"]))
    for lab, key in (("combined, alpha_v rule", "a_uni"), ("equal balance weights", "fixed"),
                     ("radiometric U alone", "U"), ("TMI alone", "TMI")):
        print("  %-24s %.3f" % (lab, s["means"][key]))
    print("  advantage over %-9s %+.3f   95%% CI [%+.3f, %+.3f]   sign-flip p = %.4f "
          "(smallest attainable %.4f)"
          % (s["best_single"], s["advantage"], s["ci95"][0], s["ci95"][1],
             s["p_signflip"], s["p_min_attainable"]))

    if args.grid:
        cells = []
        for k1 in range(1, args.kmax + 1):
            for k2 in range(1, args.kmax + 1):
                rr = protocol(k1, k2)
                ss = summarise(rr, n_boot=4000)
                cells.append(dict(k1=k1, k2=k2, **ss["means"],
                                  adv=ss["advantage"], ci_lo=ss["ci95"][0],
                                  ci_hi=ss["ci95"][1]))
            log("k_TMI = %d of %d done" % (k1, args.kmax))
        (OUT_DIR / "case2_heldout_grid.json").write_text(
            json.dumps({"kmax": args.kmax, "cells": cells}, indent=1))
        with open(HERE / "appendix_b" / "lodo_grid_by_k.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["k1", "k2", "a_uni", "U", "TMI",
                                              "adv_over_U"])
            w.writeheader()
            for c in cells:
                w.writerow({"k1": c["k1"], "k2": c["k2"], "a_uni": c["a_uni"],
                            "U": c["U"], "TMI": c["TMI"],
                            "adv_over_U": round(c["a_uni"] - c["U"], 4)})
        best = max(cells, key=lambda c: c["a_uni"])
        bestadv = max(cells, key=lambda c: c["adv"])
        excl = [c for c in cells if c["ci_lo"] > 0]
        top = [c for c in cells if abs(c["a_uni"] - best["a_uni"]) < 5e-4]
        c28 = next(c for c in cells if (c["k1"], c["k2"]) == (2, 8))
        c38 = next(c for c in cells if (c["k1"], c["k2"]) == (3, 8))
        row2 = [c["a_uni"] for c in cells if c["k1"] == 2 and 6 <= c["k2"] <= 13]
        out["grid"] = {
            "n_cells": len(cells),
            "highest_a_uni": best["a_uni"], "highest_at": [best["k1"], best["k2"]],
            "cells_reaching_highest": [[c["k1"], c["k2"]] for c in top],
            "all_highest_have_kTMI_2": all(c["k1"] == 2 for c in top),
            "largest_advantage": bestadv["adv"],
            "largest_advantage_at": [bestadv["k1"], bestadv["k2"]],
            "cells_with_ci_excluding_zero": [[c["k1"], c["k2"]] for c in excl],
            "k28": c28, "k38": c38,
            "kTMI2_kU_6_to_13_range": [min(row2), max(row2)],
        }
        print("\ngrid: highest %.3f at (%d, %d); %d cells reach it, all k_TMI=2: %s"
              % (best["a_uni"], best["k1"], best["k2"], len(top),
                 out["grid"]["all_highest_have_kTMI_2"]))
        print("  largest advantage %+.3f at (%d, %d)"
              % (bestadv["adv"], bestadv["k1"], bestadv["k2"]))
        print("  CI excludes zero at %d of %d cells: %s"
              % (len(excl), len(cells), out["grid"]["cells_with_ci_excluding_zero"]))
        print("  (2,8) %.3f   (3,8) %.3f   U alone at (2,8) %.3f"
              % (c28["a_uni"], c38["a_uni"], c28["U"]))
        print("  k_TMI=2, k_U 6..13: %.3f to %.3f" % (min(row2), max(row2)))

    # --- which k_TMI each reference wants, at the retained k_U ---
    KS = [2, 3, 4, 6, 8, 10, 12, 15]
    prof = {}
    for k1 in KS:
        rr = protocol(k1, 8)
        for h in REFS:
            prof.setdefault(h, {})[k1] = round(
                float(np.mean([r["a_uni"] for r in rr if r["h"] == h])), 4)
    out["k_TMI_by_reference"] = {
        "k_TMI_tested": KS, "k_U": 8,
        "by_reference": {str(h): prof[h] for h in REFS},
        "best_k_TMI": {str(h): max(prof[h], key=lambda k: prof[h][k]) for h in REFS},
        "mean_all_six": {str(k): round(float(np.mean([prof[h][k] for h in REFS])), 4)
                         for k in KS},
        "mean_without_paulo_afonso": {
            str(k): round(float(np.mean([prof[h][k] for h in REFS if h != 6])), 4)
            for k in KS},
    }
    print("\nheld-out recovery by reference against k_TMI, at the retained k_U = 8")
    print("  %-14s %s" % ("reference", "".join("%7d" % k for k in KS)))
    for h in REFS:
        print("  %-14s %s   best %d"
              % (h, "".join("%7.3f" % prof[h][k] for k in KS),
                 out["k_TMI_by_reference"]["best_k_TMI"][str(h)]))
    for lab, key in (("mean, all six", "mean_all_six"),
                     ("mean, no dep 6", "mean_without_paulo_afonso")):
        d = out["k_TMI_by_reference"][key]
        print("  %-14s %s   best %s"
              % (lab, "".join("%7.3f" % d[str(k)] for k in KS),
                 max(KS, key=lambda k: d[str(k)])))

    (OUT_DIR / "case2_heldout.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (OUT_DIR / "case2_heldout.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
