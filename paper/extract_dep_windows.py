"""Extract Deposits 1-5 TMI / radiometric-U windows and centered S-N profiles.

Uses numpy_repro_concat.window_matrix(dep), whose returned X has the deposit's own
window as its last row (TMI half then U half), reshaped to wshape.
"""
from __future__ import annotations

import numpy as np
import numpy_repro_concat as R

PX_KM = R.PX / 1000.0
DEPS = (1, 2, 3, 4, 5)


def deposit_windows(dep: int):
    X, _idx, wshape, _origin = R.window_matrix(dep)
    wh, ww = wshape
    npix = wh * ww
    row = X[-1]
    tmi = row[:npix].reshape(wshape)
    u = row[npix:2 * npix].reshape(wshape)
    return tmi, u, wshape


def sn_profile(M):
    """Centered south-north profile through the window centre.

    Rows run north -> south (raster origin is upper-left), so a south-north line is a
    column. Distance is centred on the window centre; values are centred on their mean.
    """
    col = M[:, M.shape[1] // 2]
    d = (np.arange(M.shape[0]) - (M.shape[0] - 1) / 2.0) * PX_KM
    return d[::-1], col - np.nanmean(col)


def collect():
    out = {}
    for d in DEPS:
        tmi, u, wshape = deposit_windows(d)
        dt, pt = sn_profile(tmi)
        du, pu = sn_profile(u)
        out[d] = dict(tmi=tmi, u=u, wshape=wshape,
                      prof_d=dt, prof_tmi=pt, prof_u=pu)
    return out


if __name__ == "__main__":
    data = collect()
    for d, v in data.items():
        print(f"Dep {d}: window {v['wshape']}  "
              f"TMI [{np.nanmin(v['tmi']):8.1f}, {np.nanmax(v['tmi']):8.1f}]  "
              f"U [{np.nanmin(v['u']):6.2f}, {np.nanmax(v['u']):6.2f}]  "
              f"profile span {v['prof_d'].min():.2f}..{v['prof_d'].max():.2f} km")
