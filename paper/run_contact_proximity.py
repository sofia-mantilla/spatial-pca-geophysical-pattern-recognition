"""Do the high-ranking windows sit on geological contacts? (R2.6)

The Carajas IOCG deposits are described as breccia and vein systems at
lithological contacts and along shear zones (Table 1). If the ranking is
finding something geological rather than any magnetic pattern, its windows
should sit nearer mapped contacts than the survey does on average.

This rasterises the unit boundaries of the Para 1:1,000,000 sheet (CPRM,
2008) onto the TMI grid, takes a distance transform, and reads the distance
to the nearest contact at the centre of every window. Three populations are
compared: all candidate windows, the top 250 of each case, and the twelve
known deposits.

Run from inside paper/:

    python run_contact_proximity.py

Writes results/contact_proximity.json.
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
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250


def contact_distance_grid():
    """Metres to the nearest mapped unit contact, on the TMI grid."""
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - height * R.PX, ox + width * R.PX, oy)

    units = gpd.read_file(GEOLOGY).to_crs("EPSG:32722")
    units = gpd.clip(units, area)
    units = units[~units.geometry.is_empty]
    # Dissolve by unit first, so a contact is a boundary between different
    # units and not an artefact of one unit being mapped as several polygons.
    units = units.dissolve("SIGLA").reset_index()
    lines = [geom.boundary for geom in units.geometry if not geom.is_empty]

    contacts = rasterize(lines, out_shape=(height, width), transform=transform,
                         fill=0, default_value=1, all_touched=True).astype(bool)
    distance = distance_transform_edt(~contacts) * R.PX
    return tmi, distance, (ox, oy)


def summarise(name, values):
    values = np.asarray(values, dtype=float)
    return {
        "population": name,
        "n": int(values.size),
        "median_m": round(float(np.median(values)), 0),
        "mean_m": round(float(values.mean()), 0),
        "within_1km": round(float((values <= 1000).mean()), 3),
        "within_2km": round(float((values <= 2000).mean()), 3),
    }


def window_centres(idx, wshape):
    wh, ww = wshape
    return idx[:, 0] + wh // 2, idx[:, 1] + ww // 2


def main() -> int:
    tmi, distance, (ox, oy) = contact_distance_grid()
    deposits = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                    max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
                for rings in C1.deposits_case1_wgs84()]

    rows = []

    # Every window the search considers, as the baseline.
    order1, idx1, wshape1, origin1, dep1 = C1.rank_spca(17)
    rr, cc = window_centres(idx1, wshape1)
    rows.append(summarise("all candidate windows (Case 1 grid)", distance[rr, cc]))

    # The known deposits themselves.
    dep_d = []
    for geom in deposits:
        x, y = geom.centroid.x, geom.centroid.y
        r = int(round((oy - y) / R.PX)); c = int(round((x - ox) / R.PX))
        r = min(max(r, 0), distance.shape[0] - 1); c = min(max(c, 0), distance.shape[1] - 1)
        dep_d.append(distance[r, c])
    rows.append(summarise("the twelve known deposits", dep_d))

    for case, label in ((1, "Case 1 top 250"), (2, "Case 2 top 250")):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
        oxx, oyy = origin
        wh, ww = wshape
        clear = []
        for i in [w for w in order if w != dep][:N_TOP]:
            r, c, _ = idx[i]
            x1, y2 = oxx + c * R.PX, oyy - r * R.PX
            rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
            if not any(rect.intersects(d) for d in deposits):
                rr = min(r + wh // 2, distance.shape[0] - 1)
                cc = min(c + ww // 2, distance.shape[1] - 1)
                clear.append(distance[rr, cc])
        rows.append(summarise(f"{label}, no known deposit", clear))

    width = max(len(r["population"]) for r in rows)
    print(f"{'population':{width}}  {'n':>6}  {'median':>8}  {'mean':>8}  "
          f"{'<=1 km':>7}  {'<=2 km':>7}")
    for r in rows:
        print(f"{r['population']:{width}}  {r['n']:6d}  {r['median_m']:7.0f} m  "
              f"{r['mean_m']:7.0f} m  {100*r['within_1km']:6.1f}%  {100*r['within_2km']:6.1f}%")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "contact_proximity.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_DIR / 'contact_proximity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
