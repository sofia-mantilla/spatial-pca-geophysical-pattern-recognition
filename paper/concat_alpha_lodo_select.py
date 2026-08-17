"""Experiment 2: scientifically select alpha by leave-one-deposit-out CV,
with a random-ranking permutation-null gate.

For each candidate alpha and each reference deposit d in 1..5:
  - rank with concat+alpha fusion (whitened per-variable scores, no 2nd PCA)
  - measure footprint recovery of the OTHER deposits
Aggregate AUC across the 5 reference folds -> pick alpha maximizing the aggregate.
Alemao (Deposit 3) is then just one fold, never individually optimized.

Permutation null: for each deposit, draw R random top-250 window sets and
recompute recovery -> a null distribution. The selected alpha only "counts" if
its aggregate AUC beats the aggregate null (p-value reported).

Run from inside the worktree (takes ~20-30 min; lower ALPHAS/R/DEPOSITS to trim):
    python concat_alpha_lodo_select.py
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import matplotlib.pyplot as plt

import spatial_pca.pipeline as P
from spatial_pca.spca.ranking import RankingResult, _fit_block_scores

CONFIG = ROOT / "configs" / "carajas_multi_tmi_u_square_tmi2_u34.yaml"
DEPOSITS = [1, 2, 3, 4, 5]
K1, K2 = 2, 34
WHITEN_STATES = [True, False]          # cross-validate whitening jointly with alpha
ALPHAS = np.round(np.linspace(0.0, 1.0, 11), 2)
R_NULL = 300
RNG = np.random.default_rng(0)
OUT = ROOT / "outputs" / "concat_lodo"
OUT.mkdir(parents=True, exist_ok=True)

STATE = {"alpha": 0.5, "whiten": True}
CAP: dict = {}

# fixed per-variable k; silence per-deposit CSV lookup and all diagnostic plots
P._resolve_multivariate_best_kpcs = lambda *a, **k: (K1, K2)
for _n in ("plot_cumulative_recovery", "plot_top_windows_overlay",
           "plot_pc_score_map", "plot_score_pairs"):
    setattr(P, _n, lambda *a, **k: None)
P._build_diagnostic_paths = lambda *a, **k: {}

# keep originals for null draws, then wrap to capture geometry/targets
_ORIG_BUILD = P.build_top_windows_gdf
_ORIG_VALID = P.validate_footprint_recovery
_ORIG_WM = P.build_multivariate_window_matrix


def _wm_wrap(*a, **k):
    wm = _ORIG_WM(*a, **k)
    CAP["wm"] = wm
    return wm


def _build_wrap(*a, **k):
    CAP["transform"] = k.get("transform")
    CAP["crs"] = k.get("crs")
    return _ORIG_BUILD(*a, **k)


def _valid_wrap(*a, **k):
    CAP["dep_gdf"] = k.get("deposits_gdf")
    CAP["ref_idx"] = k.get("reference_deposit_index")
    CAP["min_cover"] = k.get("min_cover")
    return _ORIG_VALID(*a, **k)


P.build_multivariate_window_matrix = _wm_wrap
P.build_top_windows_gdf = _build_wrap
P.validate_footprint_recovery = _valid_wrap


def _blk_w(zdep):
    w = zdep ** 2
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / max(len(w), 1)


def _ranking(**kw):
    X = np.asarray(kw["X_multi"], float)
    dep = int(kw["deposit_index"])
    win_h, win_w = int(kw["window_shape"][0]), int(kw["window_shape"][1])
    fpv = kw.get("features_per_variable")
    n_pix = int(fpv) if fpv else win_h * win_w
    a = float(STATE["alpha"])
    Z1, _, _, _, _ = _fit_block_scores(X[:, :n_pix], K1, svd_solver="full")
    Z2, _, _, _, _ = _fit_block_scores(X[:, n_pix:2 * n_pix], K2, svd_solver="full")
    if STATE["whiten"]:
        Z1 = Z1 / np.where(Z1.std(0, ddof=0) == 0, 1.0, Z1.std(0, ddof=0))
        Z2 = Z2 / np.where(Z2.std(0, ddof=0) == 0, 1.0, Z2.std(0, ddof=0))
    F = np.hstack([Z1, Z2])
    w = np.concatenate([_blk_w(Z1[dep]) * a, _blk_w(Z2[dep]) * (1 - a)])
    d = np.sqrt(((F - F[dep]) ** 2) @ w)
    order = np.argsort(d)
    res = RankingResult(order, d[order], w, F, K1 + K2, False, True, False,
                        "square", "selected_pcs", "concat_scores_alpha", {})
    return res.to_pipeline_dict()


P.run_spca_ranking_pipeline = _ranking


def _auc_end(mf):
    mf = np.asarray(mf, float)
    return float(mf.sum()) * (250 / len(mf)), float(mf[-1])


def _read(base):
    pkl = glob.glob(str(base / "**" / "validation_topk_results.pkl"), recursive=True)
    d = pickle.load(open(pkl[0], "rb"))
    return _auc_end(d["cum_mean_recovered_frac"])


def _null_for_deposit(dval):
    """Random-ranking null (R draws) using the captured geometry + targets."""
    wm = CAP["wm"]
    wi = np.asarray(wm.window_indices_for_mapping, int)   # (n, >=3): row,col,id
    n = wi.shape[0]
    aucs, ends = [], []
    for _ in range(R_NULL):
        sel = RNG.choice(n, size=min(250, n), replace=False)
        twi = wi[sel]
        gdf = _ORIG_BUILD(
            window_indices=twi[:, :2], window_shape=wm.window_shape,
            transform=CAP["transform"], crs=CAP["crs"],
            ranks=np.arange(1, len(sel) + 1), scores=np.zeros(len(sel)),
            window_ids=twi[:, 2],
        )
        rec = _ORIG_VALID(top_windows_gdf=gdf, deposits_gdf=CAP["dep_gdf"],
                          reference_deposit_index=CAP["ref_idx"],
                          min_cover=CAP["min_cover"])
        a, e = _auc_end(rec.cum_mean_recovered_frac)
        aucs.append(a); ends.append(e)
    return np.array(aucs), np.array(ends)


# ---- sweep: whiten x deposit x alpha, plus one (whiten-independent) null ------
auc = {w: {d: [] for d in DEPOSITS} for w in WHITEN_STATES}
null_auc = {}
for d in DEPOSITS:
    CAP.clear()
    for w in WHITEN_STATES:
        STATE["whiten"] = w
        for a in ALPHAS:
            STATE["alpha"] = float(a)
            base = OUT / f"d{d}_{'wh' if w else 'raw'}_a{a:.2f}"
            P.run_spca_from_config(CONFIG, deposit_1based=d, k_pcs=K2,
                                   output_dir_override=str(base))
            u, _ = _read(base)
            auc[w][d].append(u)
    null_auc[d], _ = _null_for_deposit(d)   # geometry identical across whiten/alpha
    best_w = max(WHITEN_STATES, key=lambda w: max(auc[w][d]))
    print(f"deposit {d}: best={'wh' if best_w else 'raw'} "
          f"alpha={ALPHAS[int(np.argmax(auc[best_w][d]))]:.2f} "
          f"(AUC {max(auc[best_w][d]):.1f})  null95={np.percentile(null_auc[d],95):.1f}")

NULL = np.array([null_auc[d] for d in DEPOSITS])     # (n_dep, R)
agg_null = NULL.mean(axis=0)
agg = {w: np.array([auc[w][d] for d in DEPOSITS]).mean(axis=0) for w in WHITEN_STATES}

# joint best over (whiten, alpha)
best = max(((w, i) for w in WHITEN_STATES for i in range(len(ALPHAS))),
           key=lambda wi: agg[wi[0]][wi[1]])
bw, bi = best
astar, best_auc = ALPHAS[bi], agg[bw][bi]
p_val = float((agg_null >= best_auc).mean())

print("\n=== AGGREGATE (LODO over deposits 1-5) ===")
for w in WHITEN_STATES:
    tag = "whitened" if w else "unwhitened"
    for a, g in zip(ALPHAS, agg[w]):
        print(f"  {tag:10s} alpha={a:.2f}  aggregate AUC={g:.2f}")
print(f"\nSelected: {'whitened' if bw else 'unwhitened'}, alpha*={astar:.2f}, "
      f"aggregate AUC={best_auc:.2f}")
print(f"Aggregate null AUC: mean={agg_null.mean():.2f}  95th={np.percentile(agg_null,95):.2f}")
print(f"Permutation p-value (selected vs null) = {p_val:.3f}")

# ---- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.6))
colors = {True: "#1f5fb4", False: "#c0392b"}
for w in WHITEN_STATES:
    tag = "whitened" if w else "unwhitened"
    for d in DEPOSITS:
        ax.plot(ALPHAS, auc[w][d], "-", color=colors[w], alpha=0.18, lw=1)
    ax.plot(ALPHAS, agg[w], "-o", color=colors[w], lw=2.5, label=f"aggregate ({tag})")
ax.axvline(astar, color="black", ls="--",
           label=f"selected: {'wh' if bw else 'raw'}, $\\alpha^*$={astar:.2f}")
ax.axhspan(agg_null.mean(), np.percentile(agg_null, 95), color="grey", alpha=0.2,
           label="random-null band")
ax.set_xlabel(r"$\alpha$  (TMI weight; $1-\alpha$ on U)")
ax.set_ylabel("AUC (mean-frac x250)")
ax.set_title(f"Concat fusion: joint LODO selection of (whitening, alpha)   p={p_val:.3f}")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
p = OUT / "alpha_lodo_selection.png"
fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- table ------------------------------------------------------------------
with open(OUT / "alpha_lodo.csv", "w") as f:
    f.write("whiten,alpha," + ",".join(f"dep{d}_auc" for d in DEPOSITS) + ",aggregate_auc\n")
    for w in WHITEN_STATES:
        for i, a in enumerate(ALPHAS):
            f.write(f"{int(w)},{a:.2f}," + ",".join(f"{auc[w][d][i]:.2f}" for d in DEPOSITS)
                    + f",{agg[w][i]:.2f}\n")
print("\nSaved:", p, "and", OUT / "alpha_lodo.csv")
