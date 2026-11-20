"""Which Palaeoproterozoic units carry the Case-2 enrichment (R2.6).

run_era_shares.py reports that the Case-2 selection favours Palaeoproterozoic
ground at 1.94 times its share of the survey while Case 1 avoids it at 0.41.
A co-author's reading of that shift is that the Palaeoproterozoic granites of
the province are naturally uranium-rich, and that the ~1.88 Ga magmatism they
belong to is itself associated with Cu-Au mineralization. Both halves of that
reading are about particular units, so this breaks the era down by unit.

Same geology, windows and area weighting as run_era_shares.py: the Para
1:1,000,000 sheet (CPRM, 2008), the top 250 windows of each case that touch
no known deposit, window footprints rather than centres.

Run from inside paper/:

    python run_palaeo_units.py

Writes results/palaeo_units.json.
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
from run_era_shares import clear_of_deposits, GEOLOGY, N_TOP
from run_magnetite_units import coverage

OUT_DIR = Path(__file__).resolve().parent / "results"
ERA = "Paleoproterozóico"


def unit_masks():
    """One raster mask per Palaeoproterozoic unit, plus the era as a whole."""
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    units = gpd.read_file(GEOLOGY).to_crs("EPSG:32722")
    units = gpd.clip(units, box(ox, oy - height * R.PX, ox + width * R.PX, oy))
    units = units[~units.geometry.is_empty]
    pp = units[units["ERA_MAX"] == ERA]

    def burn(geoms):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((height, width), bool)
        return rasterize(geoms, out_shape=(height, width), transform=transform,
                         fill=0, default_value=1).astype(bool)

    masks = {}
    for sigla, grp in pp.groupby("SIGLA"):
        name = str(grp["NOME"].iloc[0])
        lo, hi = float(grp["IDADE_MIN"].iloc[0]), float(grp["IDADE_MAX"].iloc[0])
        masks[str(sigla)] = (name, lo, hi, burn(list(grp.geometry)))
    masks["__era__"] = (ERA, float("nan"), float("nan"), burn(list(pp.geometry)))
    return masks


def main() -> int:
    masks = unit_masks()
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
        for sigla, (name, lo, hi, mask) in masks.items():
            base = float(coverage(mask, allb, wshape).mean())
            top = float(coverage(mask, clear, wshape).mean())
            if base <= 0 and top <= 0:
                continue
            rows[sigla] = {
                "name": name,
                "age_ma": None if lo != lo else [lo, hi],
                "survey": round(100 * base, 2),
                "top250": round(100 * top, 2),
                "ratio": round(top / base, 2) if base > 0 else None,
            }
        out[f"case{case}"] = {"n_clear": len(clear), "by_unit": rows}

        era = rows.pop("__era__", None)
        ranked = sorted(rows.items(), key=lambda kv: -kv[1]["top250"])
        print(f"\n=== Case {case}: Palaeoproterozoic units, top {N_TOP} clear "
              f"of known deposits (n={len(clear)})")
        print(f"{'unit':34} {'age (Ma)':>14} {'survey':>8} {'top250':>8} {'ratio':>7}")
        for sigla, r in ranked:
            age = "-" if r["age_ma"] is None else "%.0f-%.0f" % (r["age_ma"][1], r["age_ma"][0])
            print(f"{r['name'][:34]:34} {age:>14} {r['survey']:7.2f}% "
                  f"{r['top250']:7.2f}% {str(r['ratio']):>7}")
        if era:
            print(f"{'ERA TOTAL':34} {'':>14} {era['survey']:7.2f}% "
                  f"{era['top250']:7.2f}% {str(era['ratio']):>7}")
            out[f"case{case}"]["era_total"] = era

        # The share of the era's selected ground that sits in the 1.8-2.05 Ga
        # granite suites, which is the magmatism the reading names.
        granite = [r for _s, r in ranked
                   if r["age_ma"] and r["age_ma"][0] <= 1810
                   and ("ranit" in r["name"] or "Suíte" in r["name"])]
        if era and era["top250"] > 0:
            g = sum(r["top250"] for r in granite)
            print(f"  of the era's {era['top250']:.2f}% selected, {g:.2f}% is "
                  f"~1.88 Ga granite ({100 * g / era['top250']:.0f}% of it)")
            out[f"case{case}"]["granite_1p88ga"] = {
                "top250": round(g, 2),
                "survey": round(sum(r["survey"] for r in granite), 2),
                "fraction_of_era_selected": round(g / era["top250"], 3),
            }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "palaeo_units.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'palaeo_units.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
