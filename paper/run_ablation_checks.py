"""Appendix C ablations: weight variants (both cases) + joint concat-PCA (Case 2).

Reproduces Table "weights ablation" and the joint-decomposition paragraph.
Run from the worktree root: python run_ablation_checks.py  (~1 min)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy_repro_concat as R
import case1_uni_repro as C1

def scores_full(Xb, k):
    mu = Xb.mean(axis=0); sd = Xb.std(axis=0, ddof=0); sd = np.where(sd==0,1.0,sd)
    Z = (Xb-mu)/sd; Zc = Z - Z.mean(axis=0)
    U_,S,_ = np.linalg.svd(Zc, full_matrices=False)
    return (U_*S)[:,:k], (S**2)[:k]

def recov(order, idx, wshape, origin, dep, ref, case1=False):
    if case1:
        old = R.deposits_wgs84; R.deposits_wgs84 = C1.deposits_case1_wgs84
        try: return R.footprint_recovery(order, idx, wshape, origin, dep, ref)
        finally: R.deposits_wgs84 = old
    return R.footprint_recovery(order, idx, wshape, origin, dep, ref)

# joint PCA on concatenated pixels, Case 2
X, idx, wshape, origin = R.window_matrix(3); dep = X.shape[0]-1
print('Joint PCA on concatenated TMI+U pixels (dep 3):')
for k in (2, 4, 8, 10, 16, 24, 40):
    Z, lam = scores_full(X, k)
    w = R.blk_w(Z[dep])
    m = R.metrics(recov(np.argsort(np.sqrt(((Z-Z[dep])**2) @ w)), idx, wshape, origin, dep, 3))
    print(f'  k={k:2d}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}')

# weight ablation
print('\nWeight ablation, Case 1 (k=17):')
X1, idx1, w1, o1 = C1.window_matrix_uni(); dep1 = X1.shape[0]-1
Z1, lam1 = scores_full(X1, 17)
for mode in ('z2','unif','white'):
    if mode=='z2':    w = R.blk_w(Z1[dep1])
    elif mode=='unif': w = np.ones(Z1.shape[1])/Z1.shape[1]
    else:             w = (1/lam1)/(1/lam1).sum()
    m = R.metrics(recov(np.argsort(np.sqrt(((Z1-Z1[dep1])**2) @ w)), idx1, w1, o1, dep1, 6, case1=True))
    print(f'  {mode:6s}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}')

print('\nWeight ablation, Case 2 (k_TMI=2, k_U=8, alpha=0.503):')
npix = wshape[0]*wshape[1]
Zt, lamt = scores_full(X[:, :npix], 2)
Zu, lamu = scores_full(X[:, npix:2*npix], 8)
for mode in ('z2','unif','white'):
    if mode=='z2':    wt, wu = R.blk_w(Zt[dep]), R.blk_w(Zu[dep])
    elif mode=='unif': wt, wu = np.ones(2)/2, np.ones(8)/8
    else:             wt, wu = (1/lamt)/(1/lamt).sum(), (1/lamu)/(1/lamu).sum()
    F = np.hstack([Zt, Zu]); w = np.concatenate([wt*0.503, wu*0.497])
    m = R.metrics(recov(np.argsort(np.sqrt(((F-F[dep])**2) @ w)), idx, wshape, origin, dep, 3))
    print(f'  {mode:6s}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}')

# ---- Paulo Afonso as a fifth Case-2 test deposit (Appendix C last paragraph) ----
print('\nPaulo Afonso added to the Case-2 test set {1,2,4,5,6}:')
deps6 = C1.deposits_case1_wgs84()[:6]
def recov6(order):
    old = R.deposits_wgs84; R.deposits_wgs84 = lambda: deps6
    try: return R.footprint_recovery(order, idx, wshape, origin, dep, 3)
    finally: R.deposits_wgs84 = old
for lab, order in [('Combined a=0.503', R.rank_concat(3,2,8,0.503)[0]),
                   ('TMI alone', R.rank_concat(3,2,8,1.0)[0]),
                   ('U alone', R.rank_concat(3,2,8,0.0)[0]),
                   ('Raw multi', R.rank_raw(3)[0])]:
    d6 = recov6(order); m = R.metrics(d6)
    print(f'  {lab:18s}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}  PA coverage {100*d6["coverage_by_deposit"].get(5,0.0):5.1f}%')

# ---- Classical template baselines (Appendix C, Table "baselines") ----
def _demeaned(Xb):
    return Xb - Xb.mean(axis=1, keepdims=True)

def rank_demeaned(Xb, d):
    """Euclidean distance after removing each window's own mean (ascending)."""
    Xc = _demeaned(Xb)
    return np.argsort(np.linalg.norm(Xc - Xc[d], axis=1))

def ncc_scores(Xb, d):
    """Normalized cross-correlation against the deposit template (higher = closer)."""
    Xc = _demeaned(Xb)
    nrm = np.linalg.norm(Xc, axis=1)
    nrm = np.where(nrm == 0, 1.0, nrm)
    Xn = Xc / nrm[:, None]
    return Xn @ Xn[d]

print('\nClassical template baselines, Case 1 (reference Paulo Afonso):')
for lab, order in [('demeaned match', rank_demeaned(X1, dep1)),
                   ('correlation match', np.argsort(-ncc_scores(X1, dep1)))]:
    m = R.metrics(recov(order, idx1, w1, o1, dep1, 6, case1=True))
    print(f'  {lab:18s}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}  ranks {m["hit_ranks"]}')

print('\nClassical template baselines, Case 2 (reference Alemao, alpha=0.503):')
npix2 = wshape[0] * wshape[1]
nT, nU = ncc_scores(X[:, :npix2], dep), ncc_scores(X[:, npix2:2 * npix2], dep)
dT, dU = (np.linalg.norm(_demeaned(X[:, :npix2]) - _demeaned(X[:, :npix2])[dep], axis=1),
          np.linalg.norm(_demeaned(X[:, npix2:2 * npix2]) - _demeaned(X[:, npix2:2 * npix2])[dep], axis=1))
for lab, order in [('correlation, TMI', np.argsort(-nT)),
                   ('correlation, U', np.argsort(-nU)),
                   ('correlation, combined', np.argsort(-(0.503 * nT + 0.497 * nU))),
                   ('demeaned, combined', np.argsort(0.503 * dT / dT.std() + 0.497 * dU / dU.std()))]:
    m = R.metrics(recov(order, idx, wshape, origin, dep, 3))
    print(f'  {lab:22s}: end {m["end"]:5.1f}%  AUC {m["auc"]:6.1f}  hits {m["hits"]}')
