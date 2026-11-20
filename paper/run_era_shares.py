"""Age of the ground the ranking selects, against the survey as a whole (R2.6).

Areal shares by era, for the top 250 windows of each case that touch no known
deposit, against every window the search considers. The map is the Para
1:1,000,000 sheet (CPRM, 2008), which is the only one covering the whole
survey. The comparison is area-weighted and uses window footprints rather
than centres, so a window straddling two eras contributes to both.

Run from inside paper/:

    python run_era_shares.py

Writes results/era_shares.json.
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
from run_magnetite_units import coverage

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250
ORDER = ["Paleoarqueano", "Mesoarqueano", "Neoarqueano", "Paleoproterozóico",
         "Mesoproterozóico", "Neoproterozóico", "Paleozóico", "Mesozóico", "Cenozóico"]


def era_masks():
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    units = gpd.read_file(GEOLOGY).to_crs("EPSG:32722")
    units = gpd.clip(units, box(ox, oy - height * R.PX, ox + width * R.PX, oy))
    units = units[~units.geometry.is_empty]
    out = {}
    for era in ORDER:
        geoms = list(units.geometry[units["ERA_MAX"] == era])
        out[era] = (rasterize(geoms, out_shape=(height, width), transform=transform,
                              fill=0, default_value=1).astype(bool)
                    if geoms else np.zeros((height, width), bool))
    return out


def clear_of_deposits(order, idx, wshape, origin, dep, deposits):
    ox, oy = origin
    wh, ww = wshape
    out = []
    for i in [w for w in order if w != dep][:N_TOP]:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
        if not any(rect.intersects(d) for d in deposits):
            out.append((r, c))
    return out


def main() -> int:
    masks = era_masks()
    deposits = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                    max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
                for rings in C1.deposits_case1_wgs84()]

    out = {}
    for case in (1, 2):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
        allb = [(r, c) for r, c, _ in idx]
        clear = clear_of_deposits(order, idx, wshape, origin, dep, deposits)
        rows = {}
        for era, mask in masks.items():
            base = float(coverage(mask, allb, wshape).mean())
            top = float(coverage(mask, clear, wshape).mean())
            if base > 0 or top > 0:
                rows[era] = {"survey": round(100 * base, 1), "top250": round(100 * top, 1),
                             "ratio": round(top / base, 2) if base > 0 else None}
        arch = [e for e in ORDER if e.endswith("arqueano")]
        cover = ["Neoproterozóico", "Paleozóico", "Mesozóico", "Cenozóico"]
        agg = {}
        for name, eras in (("Archean", arch), ("Neoproterozoic and younger", cover)):
            base = sum(rows[e]["survey"] for e in eras if e in rows)
            top = sum(rows[e]["top250"] for e in eras if e in rows)
            agg[name] = {"survey": round(base, 1), "top250": round(top, 1),
                         "ratio": round(top / base, 2) if base else None}
        out[f"case{case}"] = {"n_clear": len(clear), "by_era": rows, "aggregate": agg}

        print(f"\n=== Case {case}, top {N_TOP} clear of known deposits (n={len(clear)})")
        print(f"{'era':24} {'survey':>8} {'top 250':>8} {'ratio':>7}")
        for era, r in rows.items():
            print(f"{era:24} {r['survey']:7.1f}% {r['top250']:7.1f}% {str(r['ratio']):>7}")
        for name, r in agg.items():
            print(f"{name:24} {r['survey']:7.1f}% {r['top250']:7.1f}% {str(r['ratio']):>7}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "era_shares.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'era_shares.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
