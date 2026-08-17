"""All calculations needed to correct the Case-2 briefing/paper numbers.

Uses numpy_repro_concat (validated against cached pipeline pickles: identical
hit ranks/overlap events, AUC within +/-0.02 across 8 cached configs).

Outputs (case2_experiments/results/):
  corrected_dep3_table.csv   - demo table rows, both metrics, hits
  alpha_sweep_dep3.csv       - fine alpha sweep at (3,16) and (2,6)
  lodo_nested_by_k.csv       - nested LODO alpha_uni vs baselines per (k1,k2)
  lodo_pairs.csv             - per-(h,t) held-out coverages (for significance)
  greenfield_alpha.csv       - LODO-selected fixed alpha per (k1,k2)
  significance.csv           - bootstrap CIs + permutation p per (k1,k2)

Run from the worktree root:  python case2_experiments/run_corrections.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy_repro_concat as R  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DEPS = [1, 2, 3, 4, 5]
KMAX = 40
K_COMBOS = [(2, 6), (3, 16), (2, 12), (2, 16), (2, 34), (3, 6), (3, 34), (10, 12)]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]
FINE = [round(0.05 * i, 2) for i in range(21)]
RNG = np.random.default_rng(20260715)
N_BOOT = 20000

log = lambda *a: (print(time.strftime("[%H:%M:%S]"), *a), sys.stdout.flush())

# ---------------------------------------------------------------- score cache
CACHE: dict[int, dict] = {}


def prep(dep: int):
    if dep in CACHE:
        return CACHE[dep]
    X, idx, wshape, origin = R.window_matrix(dep)
    npix = wshape[0] * wshape[1]

    def scores(Xb):
        mu = Xb.mean(axis=0)
        sd = np.where(Xb.std(axis=0, ddof=0) == 0, 1.0, Xb.std(axis=0, ddof=0))
        Z = (Xb - mu) / sd
        Zc = Z - Z.mean(axis=0)
        U_, S, _ = np.linalg.svd(Zc, full_matrices=False)
        return np.ascontiguousarray((U_ * S)[:, :KMAX])

    Z1, Z2 = scores(X[:, :npix]), scores(X[:, npix:2 * npix])
    CACHE[dep] = dict(Z1=Z1, Z2=Z2, idx=idx, wshape=wshape, origin=origin,
                      dep_index=X.shape[0] - 1)
    R.window_matrix.cache_clear()   # free X
    log(f"prep dep {dep}: {Z1.shape[0]} rows, window {wshape}")
    return CACHE[dep]


_MEMO: dict[tuple, dict] = {}


def concat_run(dep, k1, k2, alpha):
    key = (dep, k1, k2, round(float(alpha), 6))
    if key in _MEMO:
        return _MEMO[key]
    c = prep(dep)
    Z1, Z2, di = c["Z1"][:, :k1], c["Z2"][:, :k2], c["dep_index"]
    F = np.hstack([Z1, Z2])
    w = np.concatenate([R.blk_w(Z1[di]) * alpha, R.blk_w(Z2[di]) * (1 - alpha)])
    order = np.argsort(np.sqrt(((F - F[di]) ** 2) @ w))
    out = R.footprint_recovery(order, c["idx"], c["wshape"], c["origin"], di, dep)
    _MEMO[key] = out
    return out


def m_auc(d):
    mf = d["cum_mean_recovered_frac"]
    return float(mf.sum() * (250 / len(mf)))


def m_row(name, d):
    mf = d["cum_mean_recovered_frac"]
    hb = d["hit_by_rank"]
    return dict(
        method=name,
        auc_perdep=round(m_auc(d), 1),
        end_perdep_pct=round(float(mf[-1] * 100), 1),
        end_total_pct=round(float(d["cum_recovered_frac_total"][-1] * 100), 1),
        hits=len(hb),
        hit_detail="; ".join(f"dep{'+'.join(str(t + 1) for t in ids)}@{r}" for r, ids in sorted(hb.items())),
    )


def alpha_uni(dep, k1, k2, exclude=()):
    """Brownfield rule: alpha from univariate coverage of the other targets."""
    covT = concat_run(dep, k1, k2, 1.0)["coverage_by_deposit"]
    covU = concat_run(dep, k1, k2, 0.0)["coverage_by_deposit"]
    others = [t for t in DEPS if t != dep and t not in exclude]
    pT = float(np.mean([covT.get(t - 1, 0.0) for t in others]))
    pU = float(np.mean([covU.get(t - 1, 0.0) for t in others]))
    return (pT / (pT + pU) if pT + pU > 0 else 0.5), covT, covU


# ================================================================ A. dep3 table
log("A: corrected Deposit-3 table (k3/16 and k2/6)")
rows = []
raw = R.run(3, 3, 16, 0.0, mode="raw")
rows.append(m_row("Raw multivariate (no PCA)", raw))
for (k1, k2) in [(3, 16), (2, 6)]:
    dT = concat_run(3, k1, k2, 1.0)
    dU = concat_run(3, k1, k2, 0.0)
    a_uni, _, _ = alpha_uni(3, k1, k2)
    dA = concat_run(3, k1, k2, a_uni)
    d9 = concat_run(3, k1, k2, 0.9)
    rows += [
        m_row(f"Univariate TMI (k={k1})", dT),
        m_row(f"Univariate U (k={k2})", dU),
        m_row(f"Concat TMI{k1}/U{k2}, alpha_uni={a_uni:.3f} (honest rule)", dA),
        m_row(f"Concat TMI{k1}/U{k2}, alpha=0.9 (tuned/oracle)", d9),
    ]
with open(OUT / "corrected_dep3_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
log("A done")

# ================================================================ B. fine sweep
log("B: fine alpha sweep, dep3")
with open(OUT / "alpha_sweep_dep3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k1", "k2", "alpha", "auc_perdep", "end_perdep_pct", "hits"])
    for (k1, k2) in [(3, 16), (2, 6)]:
        a_uni, _, _ = alpha_uni(3, k1, k2)
        for a in sorted(set(FINE + [round(a_uni, 3)])):
            d = concat_run(3, k1, k2, a)
            w.writerow([k1, k2, a, round(m_auc(d), 2),
                        round(float(d["cum_mean_recovered_frac"][-1] * 100), 2),
                        len(d["hit_by_rank"])])
log("B done")

# ================================================================ C. LODO nested per k
log("C: nested LODO per (k1,k2)")
pairs_out = []
nested_summary = []
for (k1, k2) in K_COMBOS:
    pairs = {"a_uni": [], "fixed": [], "U": [], "TMI": []}
    for h in DEPS:
        covT = concat_run(h, k1, k2, 1.0)["coverage_by_deposit"]
        covU = concat_run(h, k1, k2, 0.0)["coverage_by_deposit"]
        cov05 = concat_run(h, k1, k2, 0.5)["coverage_by_deposit"]
        for t in DEPS:
            if t == h:
                continue
            others = [s for s in DEPS if s not in (h, t)]
            pT = float(np.mean([covT.get(s - 1, 0.0) for s in others]))
            pU = float(np.mean([covU.get(s - 1, 0.0) for s in others]))
            a = pT / (pT + pU) if pT + pU > 0 else 0.5
            c_ht = concat_run(h, k1, k2, a)["coverage_by_deposit"].get(t - 1, 0.0)
            vals = dict(a_uni=float(c_ht), fixed=float(cov05.get(t - 1, 0.0)),
                        U=float(covU.get(t - 1, 0.0)), TMI=float(covT.get(t - 1, 0.0)))
            for k, v in vals.items():
                pairs[k].append(v)
            pairs_out.append(dict(k1=k1, k2=k2, h=h, t=t, alpha=round(a, 4), **{k: round(v, 5) for k, v in vals.items()}))
    s = {k: float(np.mean(v)) for k, v in pairs.items()}
    nested_summary.append(dict(k1=k1, k2=k2, **{k: round(v, 4) for k, v in s.items()},
                               adv_over_U=round(s["a_uni"] - s["U"], 4)))
    log(f"  k{k1}/{k2}: a_uni {s['a_uni']:.3f} fixed {s['fixed']:.3f} U {s['U']:.3f} TMI {s['TMI']:.3f}")
with open(OUT / "lodo_pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pairs_out[0])); w.writeheader(); w.writerows(pairs_out)
with open(OUT / "lodo_nested_by_k.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(nested_summary[0])); w.writeheader(); w.writerows(nested_summary)
log("C done")

# ================================================================ D. significance
log("D: bootstrap + permutation significance")
sig = []
import itertools
prs = {(r["k1"], r["k2"]): [p for p in pairs_out if (p["k1"], p["k2"]) == (r["k1"], r["k2"])]
       for r in nested_summary}
for (k1, k2), pl in prs.items():
    d = np.array([p["a_uni"] - p["U"] for p in pl])            # 20 paired diffs
    t_of = np.array([p["t"] for p in pl])
    mean = d.mean()
    # cluster bootstrap over held-out deposit t (5 clusters)
    cl = {t: d[t_of == t] for t in DEPS}
    boots = []
    for _ in range(N_BOOT):
        ts = RNG.choice(DEPS, size=len(DEPS), replace=True)
        boots.append(np.concatenate([cl[t] for t in ts]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # sign-flip permutation on cluster means (2^5 exact)
    cm = np.array([cl[t].mean() for t in DEPS])
    perm = [np.abs((cm * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=5)]
    p_exact = float(np.mean([pp >= abs(cm.mean()) - 1e-12 for pp in perm]))
    sig.append(dict(k1=k1, k2=k2, mean_adv=round(mean, 4),
                    ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4),
                    p_signflip=round(p_exact, 4)))
with open(OUT / "significance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sig[0])); w.writeheader(); w.writerows(sig)
log("D done")

# ================================================================ E. greenfield alpha
log("E: greenfield fixed-alpha (LODO-selected)")
gf = []
for (k1, k2) in K_COMBOS:
    aucs = {}    # (h, alpha) -> per-deposit AUC of ranking from h
    for h in DEPS:
        for a in ALPHA_GRID:
            aucs[(h, a)] = m_auc(concat_run(h, k1, k2, a))
    rows_ = []
    for h in DEPS:   # hold out h as the 'unknown district'
        train = [x for x in DEPS if x != h]
        a_star = max(ALPHA_GRID, key=lambda a: np.mean([aucs[(x, a)] for x in train]))
        rows_.append((h, a_star, aucs[(h, a_star)], aucs[(h, 0.0)], aucs[(h, 0.5)]))
    sel = [r[1] for r in rows_]
    gf.append(dict(k1=k1, k2=k2,
                   selected_alphas=";".join(f"{a:.1f}" for a in sel),
                   heldout_auc_selected=round(float(np.mean([r[2] for r in rows_])), 2),
                   heldout_auc_Uonly=round(float(np.mean([r[3] for r in rows_])), 2),
                   heldout_auc_fixed05=round(float(np.mean([r[4] for r in rows_])), 2)))
    log(f"  k{k1}/{k2}: selected {sel} heldout {gf[-1]['heldout_auc_selected']}")
with open(OUT / "greenfield_alpha.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gf[0])); w.writeheader(); w.writerows(gf)
log("E done")

log("ALL DONE")
