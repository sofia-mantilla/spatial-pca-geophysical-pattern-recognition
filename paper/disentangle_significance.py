"""Disentangle which variable's extra PCs hurt, + bootstrap the k=2/6 advantage.

Nested alpha_uni test (held-out target coverage, 20 pairs) for:
  (2,6)  baseline         (2,12) U extended        (10,6) TMI extended
Known: (2,6)=0.483 helps, (10,12)=0.434 fusion gone.
Bootstrap the paired (alpha_uni - U-only) difference for (2,6): 95% CI + p.

Run from inside the worktree (~25 min; drop a config to trim):
    python disentangle_significance.py
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
CONFIGS = [(2, 6), (2, 12), (10, 6)]
CFG = {"K1": 2, "K2": 6}

P._resolve_multivariate_best_kpcs = lambda *a, **k: (CFG["K1"], CFG["K2"])
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
    a = float(STATE["alpha"]); K1, K2 = CFG["K1"], CFG["K2"]
    Z1, _, _, _, _ = _fit_block_scores(X[:, :n_pix], K1, svd_solver="full")
    Z2, _, _, _, _ = _fit_block_scores(X[:, n_pix:2 * n_pix], K2, svd_solver="full")
    F = np.hstack([Z1, Z2])
    w = np.concatenate([_blk_w(Z1[dep]) * a, _blk_w(Z2[dep]) * (1 - a)])
    d = np.sqrt(((F - F[dep]) ** 2) @ w); order = np.argsort(d)
    return RankingResult(order, d[order], w, F, K1 + K2, False, True, False,
                         "square", "selected_pcs", "concat", {}).to_pipeline_dict()


P.run_spca_ranking_pipeline = _rank


def _cov(h, a):
    STATE["alpha"] = float(a)
    base = ROOT / "outputs" / "disentangle" / f"k{CFG['K1']}_{CFG['K2']}_d{h}_a{a:.3f}"
    P.run_spca_from_config(CONFIG, deposit_1based=h, k_pcs=CFG["K2"], output_dir_override=str(base))
    pk = glob.glob(str(base / "**" / "validation_topk_results.pkl"), recursive=True)
    return pickle.load(open(pk[0], "rb")).get("coverage_by_deposit", {})


def cg(cov, t):
    return float(cov.get(t - 1, 0.0))


perpair = {}
for K1, K2 in CONFIGS:
    CFG["K1"], CFG["K2"] = K1, K2
    covTMI = {h: _cov(h, 1.0) for h in DEPOSITS}
    covU = {h: _cov(h, 0.0) for h in DEPOSITS}
    cov05 = {h: _cov(h, 0.5) for h in DEPOSITS}
    au, fx, uu = [], [], []
    for h in DEPOSITS:
        for t in DEPOSITS:
            if t == h:
                continue
            others = [s for s in DEPOSITS if s not in (h, t)]
            pT = np.mean([cg(covTMI[h], s) for s in others])
            pU = np.mean([cg(covU[h], s) for s in others])
            a = pT / (pT + pU) if (pT + pU) > 0 else 0.5
            au.append(cg(_cov(h, a), t)); fx.append(cg(cov05[h], t)); uu.append(cg(covU[h], t))
    perpair[(K1, K2)] = (np.array(au), np.array(fx), np.array(uu))
    print(f"TMI {K1:2d} / U {K2:2d} :  a_uni {np.mean(au):.3f} | fixed {np.mean(fx):.3f} | U {np.mean(uu):.3f}")

print("\n--- disentangle ---")
print("  (2,6)=0.483 helps  ->  (2,12) isolates U extend, (10,6) isolates TMI extend, (10,12)=0.434 dead")

# bootstrap the (2,6) paired advantage a_uni - U
au, fx, uu = perpair[(2, 6)]
diff = au - uu
rng = np.random.default_rng(0)
boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
p = float((boot <= 0).mean())
print(f"\n--- bootstrap: k=2/6 fusion advantage (alpha_uni - U-only) ---")
print(f"  observed mean diff = {diff.mean():+.3f}")
print(f"  95% CI = [{lo:+.3f}, {hi:+.3f}]")
print(f"  P(diff <= 0) = {p:.3f}   ->  {'significant' if p < 0.05 else 'NOT significant'} at 0.05")
