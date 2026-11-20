"""Geology of the high-ranking areas that hold no known deposit (R2.6).

Reviewer 2 asks whether the high-ranking windows away from known deposits are
prospective geological settings or similar magnetic patterns over unrelated
geology. This takes the published Case-1 ranking (univariate TMI, reference
Paulo Afonso, k=17), drops every one of the top 250 windows that touches a
known deposit, groups what is left into contiguous areas, and reads the
mapped unit under each one.

The geology is the Para 1:1,000,000 sheet of the Servico Geologico do Brasil
(CPRM, 2008), held in the GeoSGB database and retrieved for the study-area
bounding box only. Nothing about the ranking was sent to obtain it. The
lookup itself runs offline against data/Carajas_Brazil_Geology/.

The same map places all twelve known deposits in Archean units consistent
with Table 1, which is the check that it is being read correctly.

Run from inside paper/:

    python run_highrank_geology.py

Writes results/highrank_geology.csv and results/highrank_geology.json.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, shape
from shapely.prepared import prep
from skimage.measure import label

import case1_uni_repro as C1
import numpy_repro_concat as R

K_PCS = 17
N_TOP = 250
GEOLOGY = Path(__file__).resolve().parent.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
OUT_DIR = Path(__file__).resolve().parent / "results"
WGS84 = (6378137.0, 1 / 298.257223563)

# Units that could host the deposit type, by their mapped lithology: Archean
# supracrustal and greenstone sequences, iron formation, and the mafic to
# ultramafic and metavolcanic rocks the Carajas IOCG systems sit in or beside.
HOST_WORDS = ("anfibolito", "actinolita", "metabasalto", "metavulc", "greenstone",
              "formação ferrífera", "formacao ferrifera", "bif", "serpentinito",
              "metagabro", "metabásica", "metabasica", "xisto")


def to_lonlat(x, y):
    X, Y = R.sad69_to_wgs84_utm(x, y)
    lat, lon = R.tm_inverse(X, Y, *WGS84)
    return math.degrees(lon), math.degrees(lat)


def load_geology():
    data = json.loads(GEOLOGY.read_text())
    units = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if geom.is_valid and not geom.is_empty:
            units.append((prep(geom), feature["properties"]))
    return units


def unit_at(units, lon, lat):
    point = Point(lon, lat)
    for prepared, props in units:
        if prepared.contains(point):
            return props
    return None


def is_host(props):
    text = " ".join(str(props.get(k, "")) for k in ("LITOTIPOS", "LEGENDA", "NOME")).lower()
    return any(word in text for word in HOST_WORDS)


def main() -> int:
    order, idx, wshape, origin, dep = C1.rank_spca(K_PCS)
    ox, oy = origin
    wh, ww = wshape
    tmi, _u, _ox, _oy = R.load_cropped()

    deposits = [shape({"type": "Polygon", "coordinates": [r for r in rings]})
                for rings in C1.deposits_case1_wgs84()]
    deposits = [g.buffer(0) for g in deposits]

    top = [i for i in order if i != dep][:N_TOP]
    keep, dropped = [], 0
    for i in top:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
        if any(rect.intersects(d) for d in deposits):
            dropped += 1
        else:
            keep.append((r, c))
    print(f"top {N_TOP}: {dropped} windows touch a known deposit, {len(keep)} do not",
          flush=True)

    mask = np.zeros(tmi.shape, dtype=bool)
    for r, c in keep:
        mask[r:r + wh, c:c + ww] = True
    labels = label(mask, connectivity=2)
    n_areas = labels.max()
    cell_km2 = (R.PX / 1000.0) ** 2
    print(f"grouped into {n_areas} contiguous areas", flush=True)

    units = load_geology()
    rows = []
    for a in range(1, n_areas + 1):
        rr, cc = np.nonzero(labels == a)
        area_km2 = len(rr) * cell_km2
        # Sample the area on a coarse lattice rather than at one point, so a
        # large area that straddles two units reports both.
        step = max(1, int(len(rr) ** 0.5 / 4))
        tally, seen = Counter(), {}
        for r, c in zip(rr[::step], cc[::step]):
            lon, lat = to_lonlat(ox + (c + 0.5) * R.PX, oy - (r + 0.5) * R.PX)
            props = unit_at(units, lon, lat)
            key = "unmapped" if props is None else str(props.get("SIGLA", "?"))
            tally[key] += 1
            if props is not None:
                seen[key] = props
        if not tally:
            continue
        main_key, main_n = tally.most_common(1)[0]
        props = seen.get(main_key, {})
        lon, lat = to_lonlat(ox + (cc.mean() + 0.5) * R.PX, oy - (rr.mean() + 0.5) * R.PX)
        rows.append({
            "area_id": a,
            "area_km2": round(area_km2, 1),
            "lon": round(lon, 4),
            "lat": round(lat, 4),
            "unit": main_key,
            "unit_share": round(main_n / sum(tally.values()), 2),
            "name": str(props.get("NOME", ""))[:60],
            "lithology": str(props.get("LITOTIPOS", ""))[:110],
            "era": str(props.get("ERA_MAX", "")),
            "plausible_host": bool(props) and is_host(props),
            "units_present": ";".join(f"{k}:{v}" for k, v in tally.most_common()),
        })

    rows.sort(key=lambda r: -r["area_km2"])
    hosts = sum(1 for r in rows if r["plausible_host"])
    total = sum(r["area_km2"] for r in rows)
    host_area = sum(r["area_km2"] for r in rows if r["plausible_host"])
    print(f"\n{hosts} of {len(rows)} areas sit in a unit whose mapped lithology could host "
          f"the deposit type\n{host_area:.0f} km2 of {total:.0f} km2 by area\n", flush=True)
    for r in rows[:12]:
        flag = "host " if r["plausible_host"] else "other"
        print(f"  {r['area_id']:>3}  {r['area_km2']:6.1f} km2  {flag}  {r['unit']:12} "
              f"{r['lithology'][:58]}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "highrank_geology.json").write_text(json.dumps(
        {"dropped_touching_deposit": dropped, "areas": rows}, indent=1, ensure_ascii=False))
    with (OUT_DIR / "highrank_geology.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_DIR / 'highrank_geology.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
