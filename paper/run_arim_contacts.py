"""Do the high-ranking windows sit on lithological contacts? (R2.6)

Table 1 describes the Carajas IOCG deposits as breccia and vein systems at
lithological contacts. If the ranking is finding that setting rather than any
magnetic pattern, its windows should lie nearer mapped contacts than ordinary
ground, and nearer than a template matcher that is only matching anomaly shape.

Distances are computed from the polygon geometry. A rasterised distance
transform at 200 m places boundaries up to one cell too close and shortens
every distance by of order 100 m.

run_contact_proximity.py asked this of the Para 1:1,000,000 sheet, which has
230 polygons over the survey and puts its contacts in coarse places. This asks
it of the ARIM Carajas 1:250,000 map, which has 728, using its own unit codes.
A contact here is a boundary between two different SIGLA_UNID, not an artefact
of one unit being drawn as several polygons.

Populations: the twelve known deposits, the top 250 of each ranking that touch
no known deposit, every candidate window, and 2,000 random selections of 250.
The three rankings are wPCA and the two classical template baselines of the
Appendix C table, so the question "does this separate the methods" can be
answered rather than assumed. All populations are restricted to the ARIM
footprint.

Run from inside paper/:

    python run_arim_contacts.py [--draws 2000] [--buffer 500]

Writes results/arim_contacts.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from shapely.strtree import STRtree

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


def grids():
    tmi, _u, ox, oy = R.load_cropped()
    shape = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - shape[0] * R.PX, ox + shape[1] * R.PX, oy)
    g = gpd.clip(gpd.read_file(LITO).to_crs("EPSG:32722"), area)
    g = g[~g.geometry.is_empty]
    # dissolve by unit first: a contact separates different units
    diss = g.dissolve("SIGLA_UNID").reset_index()
    lines = [geom.boundary for geom in diss.geometry if not geom.is_empty]
    tree = STRtree(lines)
    footprint = rasterize(list(g.geometry), out_shape=shape, transform=transform,
                          fill=0, default_value=1).astype(bool)
    print("%d polygons, %d units, footprint %.1f%% of the grid"
          % (len(g), len(diss), 100 * footprint.mean()))
    return (tree, lines), footprint, (ox, oy), shape


def summarise(values, buf):
    v = np.asarray(values, float)
    return {"n": int(v.size), "median_m": round(float(np.median(v))),
            "mean_m": round(float(v.mean())),
            "within_buf": round(float((v <= buf).mean()), 3),
            "within_1km": round(float((v <= 1000).mean()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--buffer", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    (tree, lines), foot, (ox, oy), (H, W) = grids()

    def dist_at(x, y):
        """Exact distance from a point to the nearest mapped unit boundary.

        An earlier version read this off a rasterised distance transform. At
        200 m with all_touched rasterisation that places every boundary up to
        one cell too close, which shortened every distance here by of order
        100 m. The geometry is used directly instead.
        """
        p = Point(x, y)
        return p.distance(lines[tree.nearest(p)])
    deposits = deposit_boxes()
    rng = np.random.default_rng(args.seed)
    out = {"draws": args.draws, "buffer_m": args.buffer}

    dep_d = []
    for d in deposits:
        r = int(round((oy - d.centroid.y) / R.PX)); c = int(round((d.centroid.x - ox) / R.PX))
        r = min(max(r, 0), H - 1); c = min(max(c, 0), W - 1)
        if foot[r, c]:
            dep_d.append(dist_at(d.centroid.x, d.centroid.y))
    out["the twelve known deposits"] = summarise(dep_d, args.buffer)
    print("deposits inside the ARIM footprint: %d of 12" % len(dep_d))

    for case in (1, 2):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
            Xb, _i, _w, _o = C1.window_matrix_uni(); d = Xb.shape[0] - 1
            ranks = {"wPCA": order, "demeaned match": rank_demeaned(Xb, d),
                     "correlation match": np.argsort(-ncc_scores(Xb, d))}
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
            Xb, _i, _w, _o = R.window_matrix(3); d = Xb.shape[0] - 1
            npix = wshape[0] * wshape[1]
            nT, nU = ncc_scores(Xb[:, :npix], d), ncc_scores(Xb[:, npix:2*npix], d)
            dT = np.linalg.norm(_demeaned(Xb[:, :npix]) - _demeaned(Xb[:, :npix])[d], axis=1)
            dU = np.linalg.norm(_demeaned(Xb[:, npix:2*npix])
                                - _demeaned(Xb[:, npix:2*npix])[d], axis=1)
            ranks = {"wPCA": order,
                     "demeaned match": np.argsort(0.503*dT/dT.std() + 0.497*dU/dU.std()),
                     "correlation match": np.argsort(-(0.503*nT + 0.497*nU))}
        wh, ww = wshape

        def centres(cells):
            v = []
            for r, c in cells:
                rr = min(r + wh // 2, H - 1); cc = min(c + ww // 2, W - 1)
                if foot[rr, cc]:
                    v.append(dist_at(ox + (c + ww / 2) * R.PX,
                                     oy - (r + wh / 2) * R.PX))
            return v

        rows = {"all candidate windows":
                summarise(centres([(r, c) for r, c, _ in idx]), args.buffer)}
        for lab, rk in ranks.items():
            rows[lab] = summarise(centres(clear_of_deposits(rk, idx, wshape,
                                                            origin, dep, deposits)),
                                  args.buffer)
        pool = np.arange(len(idx)); pool = pool[pool != dep]
        null = []
        for _ in range(args.draws):
            pick = rng.choice(pool, size=N_TOP, replace=False)
            null.append(summarise(centres(clear_of_deposits(pick, idx, wshape,
                                                            origin, dep, deposits)),
                                  args.buffer))
        rnd = {}
        for k in ("median_m", "within_buf", "within_1km"):
            v = np.array([x[k] for x in null], float)
            rnd[k] = {"mean": round(float(v.mean()), 3),
                      "p5": round(float(np.percentile(v, 5)), 3),
                      "p95": round(float(np.percentile(v, 95)), 3)}
            for lab in rows:
                rows[lab].setdefault("percentile", {})[k] = round(
                    float((v < rows[lab][k]).mean()), 3)
        out["case%d" % case] = {"rankings": rows, "random": rnd}

        print("\n=== Case %d: distance from the window centre to a mapped contact" % case)
        print("  %-24s %9s %9s %11s %11s"
              % ("population", "n", "median", "<=%dm" % args.buffer, "<=1 km"))
        for lab in ("wPCA", "demeaned match", "correlation match",
                    "all candidate windows"):
            r = rows[lab]
            print("  %-24s %9d %8dm %10.1f%% %10.1f%%"
                  % (lab, r["n"], r["median_m"], 100*r["within_buf"], 100*r["within_1km"]))
        d12 = out["the twelve known deposits"]
        print("  %-24s %9d %8dm %10.1f%% %10.1f%%"
              % ("the twelve deposits", d12["n"], d12["median_m"],
                 100*d12["within_buf"], 100*d12["within_1km"]))
        print("  %-24s %9s %8.0fm %10.1f%% %10.1f%%   (5-95%%: %.0f-%.0fm, %.1f-%.1f%%)"
              % ("random, mean", "", rnd["median_m"]["mean"],
                 100*rnd["within_buf"]["mean"], 100*rnd["within_1km"]["mean"],
                 rnd["median_m"]["p5"], rnd["median_m"]["p95"],
                 100*rnd["within_buf"]["p5"], 100*rnd["within_buf"]["p95"]))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "arim_contacts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote %s" % (OUT_DIR / "arim_contacts.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
