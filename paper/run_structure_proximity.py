"""Do the high-ranking windows follow the mapped shear zones? (R2.6)

The Carajas IOCG deposits are described as breccia and vein systems along
shear zones (Table 1), and a co-author's reading of the high-ranking areas is
that many of them follow the WNW-trending Cinzento, Carajas and Canaa systems.
This turns that reading into a number.

The structures are the Mapa de integracao geologico-geofisica of the ARIM
Carajas project, 1:250,000, preliminary version, Servico Geologico do Brasil
(2019), SIRGAS 2000. Nothing about the ranking was sent to obtain it.

Two cautions decide what is counted.

The layer's 150 "Lineamento Magnetico" features are interpretations of the
same aeromagnetic survey the method ranks on. Measuring the distance from a
magnetically selected window to a magnetically interpreted lineament would
be circular, so they are dropped, and the two definitions used here are
mapped shear zones and all mapped structures other than those lineaments.

The ARIM map does not cover the whole survey: its eastern limit falls about
24 km short of the survey's, in the Neoproterozoic cover the ranking already
avoids. Every population is therefore restricted to window centres inside
the ARIM mapped footprint, the outline of its lithology polygons as
make_arim_footprint.py derives it, and the counts
dropped are reported. Comparing a selection against a baseline measured over
different ground is the error this avoids.

The two layers used here, data/Carajas_Brazil_Structures/arim_estrutura.* and
arim_recmin.*, are the Estrutura and Ocorrencia_Mineral shapefiles of that
archive, copied out unchanged so a clone reproduces this without the 33 MB
download. make_arim_footprint.py derives the coverage outline from its
lithology layer.

Run from inside paper/:

    python run_structure_proximity.py

Writes results/structure_proximity.json.
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
STRUCTURES = DATA / "arim_estrutura.shp"
FOOTPRINT = DATA / "arim_footprint.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250

# Interpreted from the same magnetic data the ranking uses: excluded.
CIRCULAR = "Lineamento Magnético"
# The named systems the co-author's reading points at. Canaa is not a named
# feature of this layer, which the write-up has to say.
NAMED = ["Lineamento Cinzento", "Falha Carajás", "Zona de Cisalhamento Pojuca"]


def grids():
    """Distance grids in metres, plus the ARIM coverage mask, on the TMI grid."""
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - height * R.PX, ox + width * R.PX, oy)

    st = gpd.read_file(STRUCTURES).to_crs("EPSG:32722")
    st = gpd.clip(st, area)
    st = st[~st.geometry.is_empty]

    magnetic = st["TIPO_ESTRU"] == CIRCULAR
    shear = st["TIPO_ESTRU"].str.contains("cisalhamento", case=False, na=False)
    named = st["NOME_ESTRU"].isin(NAMED)

    def distance_to(sel, label):
        geoms = list(st.geometry[sel])
        if not geoms:
            raise SystemExit("run_structure_proximity.py: no features for %s" % label)
        burn = rasterize(geoms, out_shape=(height, width), transform=transform,
                         fill=0, default_value=1, all_touched=True).astype(bool)
        print("  %-34s %4d features, %6d cells" % (label, len(geoms), burn.sum()))
        return distance_transform_edt(~burn) * R.PX

    print("structures inside the survey: %d, of which %d magnetic lineaments "
          "(dropped)" % (len(st), int(magnetic.sum())))
    dists = {
        "shear zones": distance_to(shear, "shear zones"),
        "all structures but magnetic": distance_to(~magnetic, "all but magnetic"),
        "named systems": distance_to(named, "Cinzento, Carajas, Pojuca"),
    }

    foot = gpd.read_file(FOOTPRINT).to_crs("EPSG:32722")
    foot = gpd.clip(foot, area)
    foot = foot[~foot.geometry.is_empty]
    coverage = rasterize(list(foot.geometry), out_shape=(height, width),
                         transform=transform, fill=0, default_value=1).astype(bool)
    print("  ARIM footprint covers %.1f%% of the survey grid"
          % (100 * coverage.mean()))
    return tmi, dists, coverage, (ox, oy)


def summarise(name, values, dropped):
    v = np.asarray(values, dtype=float)
    return {
        "population": name,
        "n": int(v.size),
        "dropped_outside_arim": int(dropped),
        "median_m": round(float(np.median(v)), 0),
        "mean_m": round(float(v.mean()), 0),
        "within_1km": round(float((v <= 1000).mean()), 3),
        "within_2km": round(float((v <= 2000).mean()), 3),
        "within_5km": round(float((v <= 5000).mean()), 3),
    }


def main() -> int:
    tmi, dists, coverage, (ox, oy) = grids()
    nrow, ncol = coverage.shape
    deposits = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                    max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
                for rings in C1.deposits_case1_wgs84()]

    def clamp(r, c):
        return min(max(r, 0), nrow - 1), min(max(c, 0), ncol - 1)

    populations = []

    order1, idx1, wshape1, _o1, _d1 = C1.rank_spca(17)
    wh, ww = wshape1
    rr = np.clip(idx1[:, 0] + wh // 2, 0, nrow - 1)
    cc = np.clip(idx1[:, 1] + ww // 2, 0, ncol - 1)
    keep = coverage[rr, cc]
    populations.append(("all candidate windows", rr[keep], cc[keep],
                        int((~keep).sum())))

    dep_r, dep_c, dep_out = [], [], 0
    for geom in deposits:
        r, c = clamp(int(round((oy - geom.centroid.y) / R.PX)),
                     int(round((geom.centroid.x - ox) / R.PX)))
        if coverage[r, c]:
            dep_r.append(r); dep_c.append(c)
        else:
            dep_out += 1
    populations.append(("the twelve known deposits", np.array(dep_r),
                        np.array(dep_c), dep_out))

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
              f"{'mean':>9}  {'<=1km':>6} {'<=2km':>6} {'<=5km':>6}")
        for r in rows:
            print(f"{r['population']:{w}}  {r['n']:6d} {r['dropped_outside_arim']:4d}  "
                  f"{r['median_m']:7.0f} m  {r['mean_m']:7.0f} m  "
                  f"{100*r['within_1km']:5.1f}% {100*r['within_2km']:5.1f}% "
                  f"{100*r['within_5km']:5.1f}%")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "structure_proximity.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'structure_proximity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
