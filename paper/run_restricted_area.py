"""Does the search area flatter the ratio to random? (R2.6)

The eastern third of the study area is Neoproterozoic and Phanerozoic cover.
No known deposit lies there, the ranking avoids it, and yet it counts in the
denominator: the random expectation is computed over every valid window,
including ground where nothing was ever findable. If that is doing the work,
the method's advantage over a random selection is overstated.

This rescores the published Case-1 ranking with the candidate pool cut back
to basement, and recomputes the random expectation over the smaller pool.
The ranking itself is untouched, so any change comes from the baseline and
not from re-tuning.

Two cuts are reported:

  basement        drops windows centred on Neoproterozoic or Phanerozoic
                  cover, which holds none of the twelve deposits
  Archean only    also drops Palaeoproterozoic, which does host Estrela,
                  so this one removes a test deposit's ground and is the
                  more aggressive reading

Run from inside paper/:

    python run_restricted_area.py
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
import null_expectation as NE
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
OUT_DIR = HERE / "results"
T_MAX = 6000
COVER = {"Neoproterozóico", "Paleozóico", "Mesozóico", "Cenozóico"}
ARCHEAN = {"Paleoarqueano", "Mesoarqueano", "Neoarqueano"}


def era_grid():
    """Per-cell era label on the TMI grid, as an integer code."""
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    units = gpd.read_file(GEOLOGY).to_crs("EPSG:32722")
    units = gpd.clip(units, box(ox, oy - height * R.PX, ox + width * R.PX, oy))
    units = units[~units.geometry.is_empty]

    eras = sorted({str(e) for e in units["ERA_MAX"]})
    code = {e: i + 1 for i, e in enumerate(eras)}
    shapes = [(geom, code[str(era)]) for geom, era in zip(units.geometry, units["ERA_MAX"])]
    grid = rasterize(shapes, out_shape=(height, width), transform=transform,
                     fill=0, all_touched=False).astype(np.int16)
    return tmi, grid, code


def score(order, idx, wshape, origin, dep, keep, grid_shape, valid_cells, deps):
    """Recovery and the random expectation over the kept candidate pool."""
    kept_order = [i for i in order if i != dep and keep[i]]
    idx_kept = idx[keep]
    n = int(keep.sum())

    rects = NE.rects_from_idx(idx_kept, wshape, origin)
    cells, areas = NE.build_cells(deps, C1.REF_DEP - 1, rects)
    curve = NE.exact_random_curve(cells, areas, n, t_max=T_MAX)
    union_rand = NE.exact_random_union(idx_kept, wshape, n, grid_shape, t_max=T_MAX)
    ranks = np.arange(1, T_MAX + 1)

    res = C1.recovery(np.asarray(kept_order), idx, wshape, origin, dep)
    m = R.metrics(res)
    union = NE.union_area_curve(np.asarray(kept_order), idx, wshape, dep, grid_shape)
    t_star = float(np.interp(union[-1], union_rand, ranks))
    expected = float(np.interp(t_star, ranks, curve))
    return {
        "n_windows": n,
        "search_area_km2": round(valid_cells * (R.PX / 1000) ** 2, 0),
        "end": round(float(m["end"]), 1),
        "hits": int(m["hits"]),
        "a250": round(float(union[-1] / valid_cells), 4),
        "random_matched": round(expected, 4),
        "ratio": round(float(m["end"] / 100.0 / expected), 2) if expected > 0 else None,
    }


def main() -> int:
    tmi, eras, code = era_grid()
    order, idx, wshape, origin, dep = C1.rank_spca(17)
    deps = C1.deposits_case1_wgs84()
    wh, ww = wshape
    finite = np.isfinite(tmi)

    cover_codes = {code[e] for e in COVER if e in code}
    arch_codes = {code[e] for e in ARCHEAN if e in code}
    centre_era = eras[np.clip(idx[:, 0] + wh // 2, 0, eras.shape[0] - 1),
                      np.clip(idx[:, 1] + ww // 2, 0, eras.shape[1] - 1)]

    cases = {
        "published (whole survey)": (np.ones(len(idx), bool), finite),
        "basement (no Neoproterozoic or younger)": (
            ~np.isin(centre_era, list(cover_codes)),
            finite & ~np.isin(eras, list(cover_codes))),
        "Archean only": (
            np.isin(centre_era, list(arch_codes)),
            finite & np.isin(eras, list(arch_codes))),
    }

    out = {}
    print(f"{'search area':42} {'n':>6} {'km2':>7} {'c(250)':>7} {'hits':>5} "
          f"{'a(250)':>7} {'random':>7} {'ratio':>6}")
    for name, (keep, area_mask) in cases.items():
        row = score(order, idx, wshape, origin, dep, keep,
                    tmi.shape, int(area_mask.sum()), deps)
        out[name] = row
        print(f"{name:42} {row['n_windows']:6d} {row['search_area_km2']:7.0f} "
              f"{row['end']:6.1f}% {row['hits']:5d} {100*row['a250']:6.1f}% "
              f"{100*row['random_matched']:6.1f}% {row['ratio']:6.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "restricted_area.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT_DIR / 'restricted_area.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
