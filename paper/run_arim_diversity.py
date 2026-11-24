"""Pablo's reading, measured: how the selection spreads over mapped units.

A co-author's reading of the high-ranking areas is that the IOCG deposits of
this province occupy diverse lithologies, so a prediction confined to one host
lithology would be the suspect outcome. That is a statement about spread across
mapped units, and it can be measured.

The map is the ARIM Carajas preliminary integration map at 1:250,000, used by
its own unit codes (SIGLA_UNID). No keyword rule and no lithological grouping
of ours enters this, which is the point: the only judgement is the survey's.

Four quantities, for the twelve known deposits, for the top 250 windows of each
case that touch no known deposit, and for 2,000 random selections of 250
windows put through the same filter:

  units        distinct mapped units the windows touch
  effective    exp(Shannon entropy) of the area shares. A selection lying
               wholly in one unit scores 1; one spread evenly over ten scores 10
  largest      share of the selected area in its single largest unit, which is
               the "confined to one lithology" index the reading is about
  on hosts     share of the selected area on units that host at least one of
               the twelve known deposits. The host list comes from where the
               deposits are, so no opinion enters it either

Every population is restricted to the ARIM footprint, since the map stops short
of the survey's eastern edge and a window outside it has no unit at all.

Run from inside paper/:

    python run_arim_diversity.py [--draws 2000] [--seed 20261123]

Writes results/arim_diversity.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

import case1_uni_repro as C1
import numpy_repro_concat as R
from run_era_shares import clear_of_deposits
from run_geology_vs_baselines import (_demeaned, deposit_boxes, ncc_scores,
                                      rank_demeaned)

HERE = Path(__file__).resolve().parent
LITO = (HERE.parent / "data" / "Carajas_Brazil_Structures"
        / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas"
        / "Litologia" / "arim_carajas_lito.shp")
OUT_DIR = HERE / "results"
N_TOP = 250


def unit_grid():
    """Integer unit code per cell, 0 where the ARIM map does not reach."""
    tmi, _u, ox, oy = R.load_cropped()
    shape = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - shape[0] * R.PX, ox + shape[1] * R.PX, oy)
    g = gpd.clip(gpd.read_file(LITO).to_crs("EPSG:32722"), area)
    g = g[~g.geometry.is_empty]
    names = sorted(set(g["SIGLA_UNID"].fillna("?")))
    code = {s: i + 1 for i, s in enumerate(names)}
    shapes = [(geom, code[s]) for geom, s in zip(g.geometry, g["SIGLA_UNID"].fillna("?"))]
    grid = rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                     dtype="int32")
    print("%d polygons, %d distinct units, map covers %.1f%% of the grid"
          % (len(g), len(names), 100 * (grid > 0).mean()))
    return grid, {v: k for k, v in code.items()}, (ox, oy)


def tally(grid, cells, wshape):
    """Area in cells per unit code, over a set of windows."""
    wh, ww = wshape
    H, W = grid.shape
    out = defaultdict(int)
    for r, c in cells:
        r0, c0 = min(r, H - wh), min(c, W - ww)
        patch = grid[r0:r0 + wh, c0:c0 + ww]
        vals, counts = np.unique(patch, return_counts=True)
        for v, n in zip(vals, counts):
            if v:
                out[int(v)] += int(n)
    return out


def measures(t, hosts):
    total = sum(t.values())
    if not total:
        return None
    p = np.array(sorted(t.values(), reverse=True), dtype=float) / total
    ent = -(p * np.log(p)).sum()
    on_host = sum(n for u, n in t.items() if u in hosts) / total
    return {"units": len(t), "effective": round(float(np.exp(ent)), 2),
            "largest": round(float(p[0]), 3), "on_hosts": round(float(on_host), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    grid, names, (ox, oy) = unit_grid()
    H, W = grid.shape
    deposits = deposit_boxes()

    # units that host at least one known deposit
    hosts, dep_cells = set(), []
    for d in deposits:
        x0, y0, x1, y1 = d.bounds
        r0 = max(int((oy - y1) / R.PX), 0); r1 = min(int((oy - y0) / R.PX) + 1, H)
        c0 = max(int((x0 - ox) / R.PX), 0); c1 = min(int((x1 - ox) / R.PX) + 1, W)
        patch = grid[r0:r1, c0:c1]
        hosts |= {int(v) for v in np.unique(patch) if v}
        dep_cells.append((r0, c0, r1, c1))
    print("units hosting at least one known deposit: %d (%s)"
          % (len(hosts), ", ".join(sorted(names[u] for u in hosts)[:8]) + " ..."))

    dep_t = defaultdict(int)
    for r0, c0, r1, c1 in dep_cells:
        vals, counts = np.unique(grid[r0:r1, c0:c1], return_counts=True)
        for v, n in zip(vals, counts):
            if v:
                dep_t[int(v)] += int(n)

    rng = np.random.default_rng(args.seed)
    out = {"draws": args.draws, "seed": args.seed,
           "n_host_units": len(hosts),
           "host_units": sorted(names[u] for u in hosts),
           "the twelve known deposits": measures(dep_t, hosts)}

    for case in (1, 2):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
            Xb, _i, _w, _o = C1.window_matrix_uni()
            d = Xb.shape[0] - 1
            ranks = {"wPCA": order, "demeaned match": rank_demeaned(Xb, d),
                     "correlation match": np.argsort(-ncc_scores(Xb, d))}
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, R.ALPHA_TMI)
            Xb, _i, _w, _o = R.window_matrix(3)
            d = Xb.shape[0] - 1
            npix = wshape[0] * wshape[1]
            nT, nU = ncc_scores(Xb[:, :npix], d), ncc_scores(Xb[:, npix:2*npix], d)
            dT = np.linalg.norm(_demeaned(Xb[:, :npix]) - _demeaned(Xb[:, :npix])[d], axis=1)
            dU = np.linalg.norm(_demeaned(Xb[:, npix:2*npix])
                                - _demeaned(Xb[:, npix:2*npix])[d], axis=1)
            ranks = {"wPCA": order,
                     "demeaned match": np.argsort(R.ALPHA_TMI*dT/dT.std() + (1 - R.ALPHA_TMI)*dU/dU.std()),
                     "correlation match": np.argsort(-(R.ALPHA_TMI*nT + (1 - R.ALPHA_TMI)*nU))}
        allb = [(r, c) for r, c, _ in idx]
        base = measures(tally(grid, allb, wshape), hosts)
        others = {}
        for lab, rk in ranks.items():
            others[lab] = measures(tally(grid, clear_of_deposits(rk, idx, wshape,
                                                                 origin, dep, deposits),
                                          wshape), hosts)
        sel = others["wPCA"]
        pool = np.arange(len(idx)); pool = pool[pool != dep]
        null = []
        for _ in range(args.draws):
            pick = rng.choice(pool, size=N_TOP, replace=False)
            m = measures(tally(grid, clear_of_deposits(pick, idx, wshape, origin,
                                                       dep, deposits), wshape), hosts)
            if m:
                null.append(m)
        rnd = {}
        for k in ("units", "effective", "largest", "on_hosts"):
            v = np.array([x[k] for x in null], dtype=float)
            rnd[k] = {"mean": round(float(v.mean()), 3),
                      "p5": round(float(np.percentile(v, 5)), 3),
                      "p95": round(float(np.percentile(v, 95)), 3),
                      "pct_of_draws_below": round(float((v < sel[k]).mean()), 3)}
        out["case%d" % case] = {"selection": sel, "rankings": others,
                                "all windows": base, "random": rnd}

        print("\n=== Case %d, top %d clear of known deposits" % (case, N_TOP))
        print("  %-10s %9s %9s %9s %9s %9s %16s"
              % ("", "wPCA", "demeaned", "corr.", "deposits", "all win",
                 "random (5-95%)"))
        for k in ("units", "effective", "largest", "on_hosts"):
            print("  %-10s %9s %9s %9s %9s %9s   %s"
                  % (k, others["wPCA"][k], others["demeaned match"][k],
                     others["correlation match"][k],
                     out["the twelve known deposits"][k], base[k],
                     "%.2f (%.2f-%.2f)" % (rnd[k]["mean"], rnd[k]["p5"], rnd[k]["p95"])))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "arim_diversity.json").write_text(json.dumps(out, indent=1,
                                                            ensure_ascii=False))
    print("\nwrote %s" % (OUT_DIR / "arim_diversity.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
