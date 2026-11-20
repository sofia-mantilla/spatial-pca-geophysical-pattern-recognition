"""Do the three geological maps of Appendix C agree with each other?

The appendix does not rest on one map. The magnetitite comparison uses the
twelve 1:100,000 sheets, the era shares and contact distances use the Para
1:1,000,000 sheet, and the structures come from the ARIM Carajas preliminary
integration map at 1:250,000 -- the map a co-author sent as a GeoTIFF, whose
vector lithology ships in the same archive and was used here only to find the
ARIM footprint.

Mixing scales is defensible only if they agree about the ground. This checks
that, three ways:

  area        how much magnetitite, and how much of the wider iron-formation
              set, each map puts inside the survey
  overlap     where they put it: intersection over union, and each map's
              share of the other, on the ground both of them cover
  era         cell-by-cell agreement between the ARIM and 1:1,000,000 ages

It then re-reads the headline quantity of Table 10 -- magnetitite cover under
each ranking -- off the ARIM map instead of the 1:100,000 sheets, so the
conclusion can be seen to survive, or not survive, a change of base map.

The same keyword rule is applied to each map's own lithotype fields.

Run from inside paper/:

    python run_map_agreement.py

Writes results/map_agreement.json.
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
from run_era_shares import clear_of_deposits
from run_geology_vs_baselines import (_demeaned, deposit_boxes, ncc_scores,
                                      rank_demeaned)
from run_magnetite_units import BROAD, STRICT, coverage

HERE = Path(__file__).resolve().parent
GEO = HERE.parent / "data" / "Carajas_Brazil_Geology"
SHEETS_100K = GEO / "para_100k_carajas.geojson"
SHEET_1M = GEO / "para_1m_carajas.geojson"
ARIM_LITO = (HERE.parent / "data" / "Carajas_Brazil_Structures"
             / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas"
             / "Litologia" / "arim_carajas_lito.shp")
OUT_DIR = HERE / "results"
N_TOP = 250


def frame():
    tmi, _u, ox, oy = R.load_cropped()
    h, w = tmi.shape
    return (h, w), from_origin(ox, oy, R.PX, R.PX), \
        box(ox, oy - h * R.PX, ox + w * R.PX, oy), (ox, oy)


def burn(geoms, shape, transform):
    if len(geoms) == 0:
        return np.zeros(shape, bool)
    return rasterize(list(geoms), out_shape=shape, transform=transform,
                     fill=0, default_value=1).astype(bool)


def load(path, text_cols, shape, transform, area):
    g = gpd.read_file(path).to_crs("EPSG:32722")
    g = gpd.clip(g, area)
    g = g[~g.geometry.is_empty]
    text = g[text_cols[0]].fillna("")
    for c in text_cols[1:]:
        text = text + " " + g[c].fillna("")
    text = text.str.lower()
    out = {"coverage": burn(g.geometry, shape, transform)}
    for name, words in (("strict", STRICT), ("broad", BROAD)):
        hit = text.apply(lambda s: any(w in s for w in words))
        out[name] = burn(g.geometry[hit], shape, transform)
        out["n_" + name] = int(hit.sum())
    out["frame"] = g
    return out


def pair(a, b, both):
    """Agreement between two boolean grids, on the ground both maps cover."""
    A, B = a & both, b & both
    inter, union = (A & B).sum(), (A | B).sum()
    return {
        "a_pct_of_survey": round(100 * float(A.mean()), 2),
        "b_pct_of_survey": round(100 * float(B.mean()), 2),
        "iou": round(float(inter / union), 3) if union else None,
        "a_captured_by_b": round(float(inter / A.sum()), 3) if A.sum() else None,
        "b_captured_by_a": round(float(inter / B.sum()), 3) if B.sum() else None,
    }


def main() -> int:
    shape, transform, area, (ox, oy) = frame()
    print("loading the three maps", flush=True)
    sheets = load(SHEETS_100K, ["LITOTIPOS", "NOME"], shape, transform, area)
    onem = load(SHEET_1M, ["LITOTIPOS", "NOME"], shape, transform, area)
    arim = load(ARIM_LITO, ["LITOTIPO1", "LITOTIPO2", "NOME_UNIDA"],
                shape, transform, area)
    for lab, m in (("1:100,000 sheets", sheets), ("1:1,000,000 sheet", onem),
                   ("ARIM 1:250,000", arim)):
        print("  %-20s covers %5.1f%% of the survey, %4d magnetitite and "
              "%4d iron-formation polygons"
              % (lab, 100 * m["coverage"].mean(), m["n_strict"], m["n_broad"]))

    out = {"coverage_pct": {lab: round(100 * float(m["coverage"].mean()), 1)
                            for lab, m in (("sheets_100k", sheets),
                                           ("sheet_1m", onem),
                                           ("arim_250k", arim))}}

    print("\n=== where each map puts magnetite-bearing rock")
    out["overlap"] = {}
    for key in ("strict", "broad"):
        for la, a, lb, b in (("1:100,000", sheets, "ARIM", arim),
                             ("1:1,000,000", onem, "ARIM", arim),
                             ("1:100,000", sheets, "1:1,000,000", onem)):
            both = a["coverage"] & b["coverage"]
            r = pair(a[key], b[key], both)
            out["overlap"]["%s|%s|%s" % (key, la, lb)] = r
            print("  %-6s %-12s vs %-12s: %4.2f%% vs %4.2f%% of survey, "
                  "IoU %s, %s of the first is inside the second"
                  % (key, la, lb, r["a_pct_of_survey"], r["b_pct_of_survey"],
                     r["iou"], "n/a" if r["a_captured_by_b"] is None
                     else "%.0f%%" % (100 * r["a_captured_by_b"])))

    print("\n=== era, ARIM against the 1:1,000,000 sheet")
    a_era = gpd.clip(gpd.read_file(ARIM_LITO).to_crs("EPSG:32722"), area)
    o_era = gpd.clip(gpd.read_file(SHEET_1M).to_crs("EPSG:32722"), area)
    eras = sorted(set(a_era["ERA_MAXIMA"].dropna())
                  | set(o_era["ERA_MAX"].dropna()))
    code_a = np.zeros(shape, np.uint8)
    code_o = np.zeros(shape, np.uint8)
    for i, e in enumerate(eras, start=1):
        code_a[burn(a_era.geometry[a_era["ERA_MAXIMA"] == e], shape, transform)] = i
        code_o[burn(o_era.geometry[o_era["ERA_MAX"] == e], shape, transform)] = i
    both = (code_a > 0) & (code_o > 0)
    agree = float((code_a[both] == code_o[both]).mean())
    print("  %.1f%% of the ground both map agree on the era (%d cells)"
          % (100 * agree, int(both.sum())))
    out["era_agreement"] = round(agree, 3)
    conf = {}
    for i, e in enumerate(eras, start=1):
        sel = both & (code_o == i)
        if sel.sum() < 50:
            continue
        vals, counts = np.unique(code_a[sel], return_counts=True)
        top = sorted(zip(counts, vals), reverse=True)[:2]
        conf[e] = [(eras[v - 1], round(float(c / sel.sum()), 3)) for c, v in top]
        print("    1:1M %-20s -> ARIM %s" % (e, ", ".join(
            "%s %.0f%%" % (n, 100 * f) for n, f in conf[e])))
    out["era_confusion"] = conf

    print("\n=== Table 10's magnetitite column, read off the ARIM map instead")
    deposits = deposit_boxes()
    rows = {}
    for case in (1, 2):
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
            Xb, _i, _w, _o = C1.window_matrix_uni()
            d = Xb.shape[0] - 1
            ranks = {"wPCA": order, "demeaned match": rank_demeaned(Xb, d),
                     "correlation match": np.argsort(-ncc_scores(Xb, d))}
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
            Xb, _i, _w, _o = R.window_matrix(3)
            d = Xb.shape[0] - 1
            npix = wshape[0] * wshape[1]
            nT, nU = ncc_scores(Xb[:, :npix], d), ncc_scores(Xb[:, npix:2 * npix], d)
            dT = np.linalg.norm(_demeaned(Xb[:, :npix]) - _demeaned(Xb[:, :npix])[d], axis=1)
            dU = np.linalg.norm(_demeaned(Xb[:, npix:2 * npix])
                                - _demeaned(Xb[:, npix:2 * npix])[d], axis=1)
            ranks = {"wPCA": order,
                     "demeaned match": np.argsort(0.503 * dT / dT.std()
                                                  + 0.497 * dU / dU.std()),
                     "correlation match": np.argsort(-(0.503 * nT + 0.497 * nU))}
        allb = [(r, c) for r, c, _ in idx]
        row = {}
        for maplab, m in (("1:100,000 sheets", sheets), ("ARIM 1:250,000", arim)):
            base = float(coverage(m["strict"], allb, wshape).mean())
            for lab, rank in ranks.items():
                cells = clear_of_deposits(rank, idx, wshape, origin, dep, deposits)
                v = float(coverage(m["strict"], cells, wshape).mean())
                row.setdefault(lab, {})[maplab] = (round(v / base, 2) if base
                                                   else None)
        rows["case%d" % case] = row
        print("  Case %d          %18s %18s" % (case, "1:100,000 sheets", "ARIM 1:250,000"))
        for lab in ranks:
            print("    %-18s %18s %18s"
                  % (lab, row[lab]["1:100,000 sheets"], row[lab]["ARIM 1:250,000"]))
    out["magnetitite_by_map"] = rows

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "map_agreement.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote %s" % (OUT_DIR / "map_agreement.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
