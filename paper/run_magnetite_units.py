"""Does the ranking find mapped magnetite rock rather than mineralisation? (R2.6)

This is the direct form of Reviewer 2's suspicion. The method matches a
magnetic dipole, and magnetite-bearing units produce dipoles whether or not
they carry copper. The Para 1:100,000 sheets (CPRM, GeoSGB) map lithotypes,
including magnetitite and banded iron formation, so the question can be
asked of the map rather than argued.

For every window the search considers, and for the top 250 of each case, this
measures how much of the window falls on a unit whose mapped lithotypes
include magnetitite (strict) or magnetitite or banded iron formation (broad).
The twelve known deposits are measured the same way, since the useful
comparison is not against zero but against what a known deposit looks like.

Run from inside paper/:

    python run_magnetite_units.py

Writes results/magnetite_units.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

import case1_uni_repro as C1
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_100k_carajas.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250
STRICT = ("magnetitito",)
BROAD = ("magnetitito", "formação ferrífera", "formacao ferrifera", "jaspelito")


def masks():
    """Boolean grids of magnetite-bearing units, on the TMI grid."""
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    units = gpd.read_file(GEOLOGY).to_crs("EPSG:32722")
    units = gpd.clip(units, box(ox, oy - height * R.PX, ox + width * R.PX, oy))
    units = units[~units.geometry.is_empty]

    text = (units["LITOTIPOS"].fillna("") + " " + units["NOME"].fillna("")).str.lower()
    out = {}
    for name, words in (("strict", STRICT), ("broad", BROAD)):
        hit = text.apply(lambda t: any(w in t for w in words))
        geoms = list(units.geometry[hit])
        grid = (rasterize(geoms, out_shape=(height, width), transform=transform,
                          fill=0, default_value=1, all_touched=False).astype(bool)
                if geoms else np.zeros((height, width), bool))
        out[name] = grid
        print(f"{name}: {int(hit.sum())} polygons, "
              f"{100 * grid.mean():.1f}% of the grid", flush=True)
    return tmi, out


def coverage(mask, boxes, wshape):
    """Fraction of each window footprint falling on the mask."""
    wh, ww = wshape
    integral = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    height, width = mask.shape
    out = []
    for r, c in boxes:
        r0, c0 = min(r, height - wh), min(c, width - ww)
        total = (integral[r0 + wh, c0 + ww] - integral[r0, c0 + ww]
                 - integral[r0 + wh, c0] + integral[r0, c0])
        out.append(total / float(wh * ww))
    return np.asarray(out)


def summarise(name, values, baseline=None):
    row = {
        "population": name,
        "n": int(values.size),
        "touching": round(float((values > 0).mean()), 3),
        "mean_fraction": round(float(values.mean()), 4),
    }
    if baseline is not None and baseline["mean_fraction"] > 0:
        row["enrichment"] = round(row["mean_fraction"] / baseline["mean_fraction"], 2)
    return row


def main() -> int:
    tmi, grids = masks()
    deposits = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                    max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
                for rings in C1.deposits_case1_wgs84()]

    runs = {}
    for case in (1, 2):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
        ox, oy = origin
        wh, ww = wshape
        clear = []
        for i in [w for w in order if w != dep][:N_TOP]:
            r, c, _ = idx[i]
            x1, y2 = ox + c * R.PX, oy - r * R.PX
            rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
            if not any(rect.intersects(d) for d in deposits):
                clear.append((r, c))
        runs[case] = {"all": [(r, c) for r, c, _ in idx], "clear": clear,
                      "wshape": wshape, "origin": origin}

    # Deposit windows, cut the way the method cuts them.
    dep_boxes = {}
    for case in (1, 2):
        wh, ww = runs[case]["wshape"]
        ox, oy = runs[case]["origin"]
        boxes = []
        for geom in deposits:
            x, y = geom.centroid.x, geom.centroid.y
            boxes.append((max(0, int((oy - y) / R.PX) - wh // 2),
                          max(0, int((x - ox) / R.PX) - ww // 2)))
        dep_boxes[case] = boxes

    out = {}
    for kind, mask in grids.items():
        rows = []
        base = None
        for case in (1, 2):
            wshape = runs[case]["wshape"]
            allv = coverage(mask, runs[case]["all"], wshape)
            if base is None:
                base = summarise(f"all candidate windows (Case {case} grid)", allv)
                rows.append(base)
            elif case == 2:
                rows.append(summarise(f"all candidate windows (Case {case} grid)", allv, base))
            rows.append(summarise(f"the twelve known deposits (Case {case} window)",
                                  coverage(mask, dep_boxes[case], wshape), base))
            rows.append(summarise(f"Case {case} top {N_TOP}, no known deposit",
                                  coverage(mask, runs[case]["clear"], wshape), base))
        out[kind] = rows

        print(f"\n--- {kind} ---")
        width = max(len(r["population"]) for r in rows)
        for r in rows:
            extra = "" if "enrichment" not in r else f"  enrichment {r['enrichment']:.2f}x"
            print(f"{r['population']:{width}}  n {r['n']:6d}"
                  f"  touching {100 * r['touching']:5.1f}%"
                  f"  mean cover {100 * r['mean_fraction']:5.2f}%{extra}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "magnetite_units.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_DIR / 'magnetite_units.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
