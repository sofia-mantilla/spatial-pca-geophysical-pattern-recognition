"""Are the high-ranking windows near mapped copper and gold occurrences? (R2.6)

Reviewer 2 asks whether the high-ranking areas away from the known deposits
are prospective settings or similar magnetic patterns over unrelated geology.
The mapped comparisons of Appendix C bound one alternative but cannot speak to
mineralization. The mineral-occurrence register can, as far as a register goes.

The occurrences are those of the ARIM Carajas preliminary integration map,
1:250,000, Servico Geologico do Brasil (2019), themselves drawn from the
GeoSGB Recmin base. Nothing about the ranking was sent to obtain them.

Three things keep the comparison honest.

Occurrences that fall inside any of the twelve known deposit outlines are
dropped. They are the deposits the validation already counts, so leaving them
in would credit the ranking twice for the same ground.

Copper and gold occurrences are compared against an iron and manganese
control. Iron and manganese are the province's unrelated magnetic sources, so
a ranking that merely finds magnetic rock should sit as near those as it does
to copper and gold. That contrast, not the copper distance alone, is the
result.

Every population is restricted to window centres inside the ARIM mapped
footprint, because the map stops short of the survey's eastern edge; the footprint
comes from make_arim_footprint.py.

The two layers used here, data/Carajas_Brazil_Structures/arim_estrutura.* and
arim_recmin.*, are the Estrutura and Ocorrencia_Mineral shapefiles of that
archive, copied out unchanged so a clone reproduces this without the 33 MB
download. make_arim_footprint.py derives the coverage outline from its
lithology layer.

Run from inside paper/:

    python run_occurrence_proximity.py

Writes results/occurrence_proximity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

import case1_uni_repro as C1
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "Carajas_Brazil_Structures"
OCCURRENCES = DATA / "arim_recmin.shp"
FOOTPRINT = DATA / "arim_footprint.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250

TARGET = ("Cobre (Cu)", "Ouro (Au)")
CONTROL = ("Ferro (Fe)", "Manganês (Mn)")


def grids(deposits):
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - height * R.PX, ox + width * R.PX, oy)

    pts = gpd.read_file(OCCURRENCES).to_crs("EPSG:32722")
    pts = gpd.clip(pts, area)
    pts = pts[~pts.geometry.is_empty]

    on_known = pts.geometry.apply(lambda g: any(g.within(d) for d in deposits))
    print("occurrences inside the survey: %d, of which %d inside a known "
          "deposit outline (dropped)" % (len(pts), int(on_known.sum())))
    pts = pts[~on_known]

    def distance_to(sel, label):
        geoms = list(pts.geometry[sel])
        if not geoms:
            raise SystemExit("run_occurrence_proximity.py: none for %s" % label)
        burn = rasterize(geoms, out_shape=(height, width), transform=transform,
                         fill=0, default_value=1, all_touched=True).astype(bool)
        print("  %-26s %3d occurrences" % (label, len(geoms)))
        return distance_transform_edt(~burn) * R.PX

    sub = pts["SUBS_PRINC"]
    dists = {
        "Cu and Au occurrences": distance_to(sub.isin(TARGET), "Cu and Au"),
        "Fe and Mn occurrences (control)": distance_to(sub.isin(CONTROL), "Fe and Mn"),
    }

    foot = gpd.read_file(FOOTPRINT).to_crs("EPSG:32722")
    foot = gpd.clip(foot, area)
    foot = foot[~foot.geometry.is_empty]
    coverage = rasterize(list(foot.geometry), out_shape=(height, width),
                         transform=transform, fill=0, default_value=1).astype(bool)
    return dists, coverage, (ox, oy)


def summarise(name, values, dropped):
    v = np.asarray(values, dtype=float)
    return {
        "population": name,
        "n": int(v.size),
        "dropped_outside_arim": int(dropped),
        "median_m": round(float(np.median(v)), 0),
        "mean_m": round(float(v.mean()), 0),
        "within_5km": round(float((v <= 5000).mean()), 3),
        "within_10km": round(float((v <= 10000).mean()), 3),
    }


def main() -> int:
    deposits = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                    max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
                for rings in C1.deposits_case1_wgs84()]
    dists, coverage, (ox, oy) = grids(deposits)
    nrow, ncol = coverage.shape

    def clamp(r, c):
        return min(max(r, 0), nrow - 1), min(max(c, 0), ncol - 1)

    populations = []
    _o, idx1, wshape1, _oo, _d = C1.rank_spca(17)
    wh, ww = wshape1
    rr = np.clip(idx1[:, 0] + wh // 2, 0, nrow - 1)
    cc = np.clip(idx1[:, 1] + ww // 2, 0, ncol - 1)
    keep = coverage[rr, cc]
    populations.append(("all candidate windows", rr[keep], cc[keep],
                        int((~keep).sum())))

    for case, label in ((1, "Case 1 top 250, no known deposit"),
                        (2, "Case 2 top 250, no known deposit")):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
        oxx, oyy = origin
        wh, ww = wshape
        rs, cs, out = [], [], 0
        for i in [w for w in order if w != dep][:N_TOP]:
            r, c, _ = idx[i]
            x1, y2 = oxx + c * R.PX, oyy - r * R.PX
            rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
            if any(rect.intersects(d) for d in deposits):
                continue
            rr_, cc_ = clamp(r + wh // 2, c + ww // 2)
            if coverage[rr_, cc_]:
                rs.append(rr_); cs.append(cc_)
            else:
                out += 1
        populations.append((label, np.array(rs), np.array(cs), out))

    out = {}
    for defname, grid in dists.items():
        rows = [summarise(name, grid[r, c], dropped)
                for name, r, c, dropped in populations]
        out[defname] = rows
        print("\n=== distance to %s" % defname)
        w = max(len(r["population"]) for r in rows)
        print(f"{'population':{w}}  {'n':>6} {'out':>4}  {'median':>9}  "
              f"{'mean':>9}  {'<=5km':>6} {'<=10km':>7}")
        for r in rows:
            print(f"{r['population']:{w}}  {r['n']:6d} {r['dropped_outside_arim']:4d}  "
                  f"{r['median_m']:7.0f} m  {r['mean_m']:7.0f} m  "
                  f"{100*r['within_5km']:5.1f}% {100*r['within_10km']:6.1f}%")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "occurrence_proximity.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'occurrence_proximity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
