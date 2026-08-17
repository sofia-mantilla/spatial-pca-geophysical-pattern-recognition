"""Extract the twelve Case-1 deposit TMI windows and centred south-north profiles.

Case 1 is univariate TMI over the twelve known deposits, whose outlines live in
`Prospect_in Carajas_v2.shp` (the Case-2 file holds only the five multivariate
deposits). Window geometry follows the same rule as the pipeline: the deposit
outline's bounding box, snapped to raster cells.
"""
from __future__ import annotations

import math

import numpy as np
import numpy_repro_concat as R

PX_KM = R.PX / 1000.0
CASE1_SHP = R.DATA / "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp"


def deposit_window(tmi, ox, oy, rings):
    """TMI window under one deposit outline (bounding box snapped to cells)."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    cs = math.floor((min(xs) - ox) / R.PX); ce = math.ceil((max(xs) - ox) / R.PX)
    rs = math.floor((oy - max(ys)) / R.PX); re = math.ceil((oy - min(ys)) / R.PX)
    H, W = tmi.shape
    cs, ce = max(cs, 0), min(ce, W)
    rs, re = max(rs, 0), min(re, H)
    return tmi[rs:re, cs:ce]


def sn_profile(M):
    """Centred south-north profile through the window centre.

    Rows run north -> south (raster origin is upper-left), so a south-north line is a
    column read bottom-to-top. Distance increases NORTHWARD from the window centre,
    which is what "distance along the south-north line" means; values are centred on
    their mean.
    """
    col = M[:, M.shape[1] // 2]
    d = (np.arange(M.shape[0]) - (M.shape[0] - 1) / 2.0) * PX_KM
    return d[::-1], col - np.nanmean(col)


def collect(n_deposits: int = 12):
    tmi, _u, ox, oy = R.load_cropped()
    deps = R.read_shp_polygons(CASE1_SHP)
    out = {}
    for i in range(min(n_deposits, len(deps))):
        w = deposit_window(tmi, ox, oy, deps[i])
        d, p = sn_profile(w)
        out[i + 1] = dict(tmi=w, wshape=w.shape, prof_d=d, prof_tmi=p,
                          nan=int(np.isnan(w).sum()))
    return out


if __name__ == "__main__":
    data = collect()
    print(f"{len(data)} deposits read from {CASE1_SHP.name}")
    for k, v in data.items():
        print(f"  Dep {k:2d}: window {str(v['wshape']):9s} "
              f"TMI [{np.nanmin(v['tmi']):8.1f}, {np.nanmax(v['tmi']):8.1f}]  "
              f"absP98 {np.percentile(np.abs(v['tmi']), 98):7.1f}  "
              f"span +/-{v['prof_d'].max():.1f} km  NaN {v['nan']}")
