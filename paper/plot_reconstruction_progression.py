"""Reconstruction progression to k=34 for BOTH TMI and Radiometric-U (Alemao/Dep3).

recon_k = sum_{j<k} z_dep,j * component_j (standardized space). Shows how many PCs
each variable actually needs before only noise is added. Error-vs-k curves (to
k=34) mark the elbows -> justifies k_TMI=2, k_U=6.

Run from inside the worktree:
    python plot_reconstruction_progression.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spatial_pca.pipeline as P
from spatial_pca.spca.ranking import RankingResult, _fit_block_scores
from spatial_pca.colormaps import resolve_colormap

CONFIG = ROOT / "configs" / "carajas_multi_tmi_u_square_tmi2_u34.yaml"
N_MAX = 34
SHOW_KS = [1, 2, 4, 6, 10, 17, 25, 34]
KEEP = {"TMI": 2, "U": 6}
CMAP = resolve_colormap("spatial_pca_paper")
OUT = ROOT / "outputs" / "recon_progression"; OUT.mkdir(parents=True, exist_ok=True)

P._resolve_multivariate_best_kpcs = lambda *a, **k: (2, 6)
for _n in ("plot_cumulative_recovery", "plot_top_windows_overlay",
           "plot_pc_score_map", "plot_score_pairs"):
    setattr(P, _n, lambda *a, **k: None)
P._build_diagnostic_paths = lambda *a, **k: {}
CAP = {}


def _fit(block, dep):
    mean = block.mean(0); std = block.std(0); std = np.where(std == 0, 1, std)
    Xs = (block - mean) / std
    pca = PCA(n_components=min(N_MAX + 2, Xs.shape[1]), svd_solver="full").fit(Xs)
    W = pca.components_
    return Xs[dep], W, Xs[dep] @ W.T, pca.explained_variance_ratio_


def _cap(**kw):
    X = np.asarray(kw["X_multi"], float); dep = int(kw["deposit_index"])
    wh, ww = int(kw["window_shape"][0]), int(kw["window_shape"][1])
    fpv = kw.get("features_per_variable"); n_pix = int(fpv) if fpv else wh * ww
    CAP["win"] = (wh, ww)
    CAP["TMI"] = _fit(X[:, :n_pix], dep)
    CAP["U"] = _fit(X[:, n_pix:2 * n_pix], dep)
    o = np.argsort(((X - X[dep]) ** 2).sum(1))
    return RankingResult(o, np.zeros(len(o)), np.ones(X.shape[1]), X, X.shape[1],
                         False, True, False, "square", "selected_pcs", "cap", {}).to_pipeline_dict()


P.run_spca_ranking_pipeline = _cap
P.run_spca_from_config(CONFIG, deposit_1based=3, k_pcs=6, output_dir_override=str(OUT / "run"))

wh, ww = CAP["win"]
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, len(SHOW_KS) + 1, height_ratios=[1, 1, 0.9],
                      hspace=0.3, wspace=0.12)


def show(ax, m, title, keep=False):
    v = np.abs(m).max() or 1
    ax.imshow(m.reshape(wh, ww), origin="upper", cmap=CMAP, vmin=-v, vmax=v)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color=("#166016" if keep else "#333"))
    for s in ax.spines.values():
        s.set_edgecolor("#1a9641" if keep else "#ccc"); s.set_linewidth(2.4 if keep else 1)


for row, var in enumerate(("TMI", "U")):
    true, W, z, evr = CAP[var]
    cum = np.cumsum(evr)
    vt = np.abs(true).max() or 1
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(true.reshape(wh, ww), origin="upper", cmap=CMAP, vmin=-vt, vmax=vt)
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_title(f"{var}\nTRUE", fontsize=10, fontweight="bold")
    for j, k in enumerate(SHOW_KS):
        show(fig.add_subplot(gs[row, j + 1]), z[:k] @ W[:k],
             f"k={k} ({100*cum[k-1]:.0f}%)", keep=(k == KEEP[var]))

axc = fig.add_subplot(gs[2, 1:])
for var, col in (("TMI", "#e08a00"), ("U", "#7a1fb4")):
    true, W, z, evr = CAP[var]
    err = [np.linalg.norm(true - z[:k] @ W[:k]) / np.linalg.norm(true) for k in range(1, N_MAX + 1)]
    axc.plot(range(1, N_MAX + 1), err, "-o", ms=4, color=col, label=f"{var} recon error")
    axc.axvline(KEEP[var], color=col, ls="--", lw=1.5)
axc.set_xlabel("number of PCs (k)"); axc.set_ylabel("reconstruction error")
axc.set_title("Error vs k to 34 — dashed = chosen k (elbow)"); axc.legend(); axc.grid(alpha=.3)
fig.suptitle("Alemao reconstruction progression to k=34 — TMI done by ~2, U by ~6, rest is noise", fontsize=14)
p = OUT / "recon_progression_tmi_u.png"; fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", p)
