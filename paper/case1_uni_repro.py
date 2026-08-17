"""Case-1 (univariate TMI, reference Paulo Afonso = deposit 6) replication.

Reuses numpy_repro_concat's validated primitives (ERS read, crop, shapefile,
Helmert reprojection, exact-geometry footprint recovery). Univariate chain:
TMI-only sliding windows on TMI's own NaN mask, standardized PCA scores,
z^2 deposit weights, weighted distance, appended deposit row.

Verification gates (published / pipeline numbers):
  k=17: cbar(250)=0.4704, AUC 57.9, 5 hits, first hit rank 5 (deposit 11)
  k=8 : 0.383, 4 hits     k=2 : 0.242, 2 hits      (k_sweep_case1.csv)
  raw : 20.7, AUC 33.5, 1 hit (deposit 1, rank 30)
  n windows = 13,031; window = 28 x 46 px (N-S x E-W)
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, '/home/claude/b3')
import numpy_repro_concat as R

CASE1_SHP = R.DATA / "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp"
REF_DEP = 6
STRIDE = R.STRIDE


@lru_cache(maxsize=1)
def window_matrix_uni():
    """Univariate TMI window matrix with the reference-deposit window appended."""
    tmi, _u, ox, oy = R.load_cropped()
    deps = R.read_shp_polygons(CASE1_SHP)
    rings = deps[REF_DEP - 1]
    xs = [p[0] for r in rings for p in r]; ys = [p[1] for r in rings for p in r]
    cs = math.floor((min(xs) - ox) / R.PX); ce = math.ceil((max(xs) - ox) / R.PX)
    rs = math.floor((oy - max(ys)) / R.PX); re = math.ceil((oy - min(ys)) / R.PX)
    H, W = tmi.shape
    cs, ce = max(cs, 0), min(ce, W); rs, re = max(rs, 0), min(re, H)
    tpl = tmi[rs:re, cs:ce]
    wh, ww = tpl.shape
    if np.isnan(tpl).any():
        raise ValueError("template contains NaNs")
    pad_y = (STRIDE - (tmi.shape[0] - wh) % STRIDE) % STRIDE
    pad_x = (STRIDE - (tmi.shape[1] - ww) % STRIDE) % STRIDE
    pt = np.pad(tmi, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
    ncols = (pt.shape[1] - ww) // STRIDE + 1
    sw = np.lib.stride_tricks.sliding_window_view
    vt = sw(pt, (wh, ww))[::STRIDE, ::STRIDE]
    keep = ~np.isnan(vt).any(axis=(2, 3))
    rows, cols = np.nonzero(keep)
    ids = rows * ncols + cols
    Xt = vt[rows, cols].reshape(len(rows), -1)
    X = np.vstack([Xt, tpl.ravel()])
    idx = np.stack([rows * STRIDE, cols * STRIDE, ids], axis=1)
    return X, idx, (wh, ww), (ox, oy)


def rank_spca(k):
    X, idx, wshape, origin = window_matrix_uni()
    dep = X.shape[0] - 1
    Z = R.block_scores(X, k)
    w = R.blk_w(Z[dep])
    d = np.sqrt(((Z - Z[dep]) ** 2) @ w)
    return np.argsort(d), idx, wshape, origin, dep


def rank_raw_uni(standardize="ddof1"):
    X, idx, wshape, origin = window_matrix_uni()
    dep = X.shape[0] - 1
    if standardize == "none":
        Xs = X
    else:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1 if standardize == "ddof1" else 0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        Xs = (X - mu) / sd
    d = np.sqrt(((Xs - Xs[dep]) ** 2).sum(axis=1))
    return np.argsort(d), idx, wshape, origin, dep


@lru_cache(maxsize=1)
def deposits_case1_wgs84():
    out = []
    for rings in R.read_shp_polygons(CASE1_SHP):
        out.append([[R.sad69_to_wgs84_utm(x, y) for (x, y) in ring] for ring in rings])
    return out


def recovery(order, idx, wshape, origin, dep_index, n_top=R.N_TOP):
    """footprint_recovery against the 12-deposit Case-1 shapefile."""
    old = R.deposits_wgs84
    R.deposits_wgs84 = deposits_case1_wgs84
    try:
        return R.footprint_recovery(order, idx, wshape, origin, dep_index, REF_DEP, n_top=n_top)
    finally:
        R.deposits_wgs84 = old


def run_uni(k=None, raw=None):
    if raw is not None:
        order, idx, wshape, origin, dep = rank_raw_uni(raw)
    else:
        order, idx, wshape, origin, dep = rank_spca(k)
    return recovery(order, idx, wshape, origin, dep)


if __name__ == "__main__":
    X, idx, wshape, origin = window_matrix_uni()
    print(f"n windows = {X.shape[0]-1}, window = {wshape} (N-S x E-W)")
    for k in (17, 8, 2):
        m = R.metrics(run_uni(k=k))
        print(f"k={k:2d}: end {m['end']:.2f}  AUC {m['auc']:.1f}  hits {m['hits']}  ranks {m['hit_ranks']}")
    for mode in ("ddof1", "ddof0", "none"):
        m = R.metrics(run_uni(raw=mode))
        print(f"raw[{mode}]: end {m['end']:.2f}  AUC {m['auc']:.1f}  hits {m['hits']}  ranks {m['hit_ranks']}")
