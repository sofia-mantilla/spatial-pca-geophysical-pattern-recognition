"""B5: Case-1 reference-swap — every deposit as reference, k=17, TMI.

For each reference: c̄(250), AUC, hits, a(250) (union px), exact random E[c̄(t)]
to t=6000, exact random union to t=6000, matched-area random expectation.
Writes refswap_results.json.
"""
import json, math, sys
import numpy as np

import numpy_repro_concat as R
import case1_uni_repro as C1
import null_expectation as NE

KM2 = 0.04
tmi, _u, ox, oy = R.load_cropped()
GRID = tmi.shape
VALID = int(np.isfinite(tmi).sum())
deps_shp = R.read_shp_polygons(C1.CASE1_SHP)
deps_wgs = C1.deposits_case1_wgs84()

def window_matrix_ref(ref):
    rings = deps_shp[ref - 1]
    xs = [p[0] for r in rings for p in r]; ys = [p[1] for r in rings for p in r]
    cs = math.floor((min(xs) - ox) / R.PX); ce = math.ceil((max(xs) - ox) / R.PX)
    rs = math.floor((oy - max(ys)) / R.PX); re = math.ceil((oy - min(ys)) / R.PX)
    H, W = tmi.shape
    cs, ce = max(cs, 0), min(ce, W); rs, re = max(rs, 0), min(re, H)
    tpl = tmi[rs:re, cs:ce]
    wh, ww = tpl.shape
    if np.isnan(tpl).any():
        raise ValueError('template NaN')
    pad_y = (R.STRIDE - (tmi.shape[0] - wh) % R.STRIDE) % R.STRIDE
    pad_x = (R.STRIDE - (tmi.shape[1] - ww) % R.STRIDE) % R.STRIDE
    pt = np.pad(tmi, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
    ncols = (pt.shape[1] - ww) // R.STRIDE + 1
    sw = np.lib.stride_tricks.sliding_window_view
    vt = sw(pt, (wh, ww))[::R.STRIDE, ::R.STRIDE]
    keep = ~np.isnan(vt).any(axis=(2, 3))
    rows, cols = np.nonzero(keep)
    ids = rows * ncols + cols
    Xt = vt[rows, cols].reshape(len(rows), -1)
    X = np.vstack([Xt, tpl.ravel()])
    idx = np.stack([rows * R.STRIDE, cols * R.STRIDE, ids], axis=1)
    return X, idx, (wh, ww), (ox, oy)

out = {}
T = 6000
for ref in range(1, 13):
    try:
        X, idx, wshape, origin = window_matrix_ref(ref)
    except ValueError as e:
        out[str(ref)] = dict(error=str(e))
        print(ref, 'SKIP:', e, flush=True)
        continue
    n = X.shape[0] - 1
    dep = n
    Z = R.block_scores(X, 17)
    w = R.blk_w(Z[dep])
    d = np.sqrt(((Z - Z[dep]) ** 2) @ w)
    order = np.argsort(d)
    old = R.deposits_wgs84
    R.deposits_wgs84 = C1.deposits_case1_wgs84
    try:
        res = R.footprint_recovery(order, idx, wshape, origin, dep, ref)
    finally:
        R.deposits_wgs84 = old
    m = R.metrics(res)
    rects = NE.rects_from_idx(idx, wshape, origin)
    cells, areas = NE.build_cells(deps_wgs, ref - 1, rects)
    rc = NE.exact_random_curve(cells, areas, n, t_max=T)
    ru = NE.exact_random_union(idx, wshape, n, GRID, t_max=T)
    ua = NE.union_area_curve(order, idx, wshape, dep, GRID)
    t_star = float(np.interp(ua[-1], ru, np.arange(1, T + 1)))
    e_matched = float(np.interp(t_star, np.arange(1, T + 1), rc))
    out[str(ref)] = dict(
        n=n, wshape=list(wshape), end=m['end'], auc=m['auc'], hits=m['hits'],
        hit_ranks={str(k): v for k, v in m['hit_ranks'].items()},
        union250_px=float(ua[-1]), union250_frac=float(ua[-1] / VALID),
        rand250=float(rc[249]), rand_matched_area=e_matched,
        ratio_matched=float(m['end'] / 100.0 / e_matched) if e_matched > 0 else None,
    )
    print(ref, 'end %.1f%% hits %d union %.1f%% rand250 %.1f%% matched %.1f%% ratio %.2f'
          % (m['end'], m['hits'], 100 * ua[-1] / VALID, 100 * rc[249],
             100 * e_matched, m['end'] / 100 / e_matched), flush=True)

json.dump(out, open('refswap_results.json', 'w'), indent=1)
print('DONE')
