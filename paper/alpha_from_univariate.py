"""Set alpha from univariate performance and see what it delivers.

Unwhitened concat, TMI k=2, U k=6. For each deposit:
  AUC_TMI = univariate TMI recovery (alpha=1)
  AUC_U   = univariate U   recovery (alpha=0)
  alpha_uni = AUC_TMI / (AUC_TMI + AUC_U)
then evaluate the fused ranking at alpha_uni, at fixed 0.5, and on a coarse grid
for context. (In-sample per deposit: the reference deposit ranks the others both
to set alpha and to score it -- fine for 'what does the rule say', nesting later.)

Run from inside the worktree (~7 min):
    python alpha_from_univariate.py
"""
from __future__ import annotations
import glob, os, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import spatial_pca.pipeline as P
from spatial_pca.spca.ranking import RankingResult, _fit_block_scores

CONFIG = ROOT / "configs" / "carajas_multi_tmi_u_square_tmi2_u34.yaml"
DEPOSITS = [1, 2, 3, 4, 5]
K1, K2 = 2, 6                     # TMI k=2, U k=6 (noise shelf dropped)
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
names = {1: "Tucuma", 2: "Pedra Branca", 3: "Alemao", 4: "Furnas", 5: "Salobo"}

P._resolve_multivariate_best_kpcs = lambda *a, **k: (K1, K2)
for _n in ("plot_cumulative_recovery", "plot_top_windows_overlay",
           "plot_pc_score_map", "plot_score_pairs"):
    setattr(P, _n, lambda *a, **k: None)
P._build_diagnostic_paths = lambda *a, **k: {}
STATE = {"alpha": 0.5}


def _blk_w(z):
    w = z ** 2
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / max(len(w), 1)


def _rank(**kw):
    X = np.asarray(kw["X_multi"], float); dep = int(kw["deposit_index"])
    wh, ww = int(kw["window_shape"][0]), int(kw["window_shape"][1])
    fpv = kw.get("features_per_variable"); n_pix = int(fpv) if fpv else wh * ww
    a = float(STATE["alpha"])
    Z1, _, _, _, _ = _fit_block_scores(X[:, :n_pix], K1, svd_solver="full")
    Z2, _, _, _, _ = _fit_block_scores(X[:, n_pix:2 * n_pix], K2, svd_solver="full")
    F = np.hstack([Z1, Z2])
    w = np.concatenate([_blk_w(Z1[dep]) * a, _blk_w(Z2[dep]) * (1 - a)])
    d = np.sqrt(((F - F[dep]) ** 2) @ w); order = np.argsort(d)
    return RankingResult(order, d[order], w, F, K1 + K2, False, True, False,
                         "square", "selected_pcs", "concat", {}).to_pipeline_dict()


P.run_spca_ranking_pipeline = _rank


def _auc(base):
    pk = glob.glob(str(base / "**" / "validation_topk_results.pkl"), recursive=True)
    mf = np.asarray(pickle.load(open(pk[0], "rb"))["cum_mean_recovered_frac"], float)
    return float(mf.sum()) * (250 / len(mf))


def run(d, a):
    STATE["alpha"] = float(a)
    base = ROOT / "outputs" / "alpha_uni" / f"d{d}_a{a:.3f}"
    P.run_spca_from_config(CONFIG, deposit_1based=d, k_pcs=K2, output_dir_override=str(base))
    return _auc(base)


rows = []
for d in DEPOSITS:
    g = {a: run(d, a) for a in GRID}
    auc_U, auc_T = g[0.0], g[1.0]
    a_uni = auc_T / (auc_T + auc_U) if (auc_T + auc_U) > 0 else 0.5
    auc_uni = run(d, a_uni)
    best_a = max(GRID, key=lambda a: g[a]); best = g[best_a]
    rows.append((d, auc_T, auc_U, a_uni, auc_uni, g[0.5], best_a, best))

print(f"\n{'deposit':15s} {'AUC_TMI':>8s} {'AUC_U':>7s} {'a_uni':>6s} "
      f"{'AUC@a_uni':>10s} {'AUC@0.5':>8s} | {'gridbest_a':>10s} {'AUC':>6s}")
for d, aT, aU, au, auu, a05, ba, bv in rows:
    print(f"{names[d]+' ('+str(d)+')':15s} {aT:8.1f} {aU:7.1f} {au:6.2f} "
          f"{auu:10.1f} {a05:8.1f} | {ba:10.2f} {bv:6.1f}")

A = np.array([r for r in rows])
print("\nAGGREGATE (mean over deposits):")
print(f"  U-only        : {A[:,2].mean():.1f}")
print(f"  fixed a=0.5   : {A[:,5].mean():.1f}")
print(f"  a_uni rule    : {A[:,4].mean():.1f}   (mean a_uni = {A[:,3].mean():.2f})")
print(f"  grid-best/dep : {A[:,7].mean():.1f}   (in-sample ceiling, ref only)")
