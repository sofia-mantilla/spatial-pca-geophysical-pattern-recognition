"""The geology test again, on the ARIM 1:250,000 map instead of the SGB sheets.

Appendix C reads magnetitite and era off the Para 1:100,000 sheets and the
1:1,000,000 sheet. A co-author supplied the ARIM Carajas preliminary
integration map at 1:250,000, which is the compilation his own figure is
drawn from, so the same test can be put to it. If the conclusion depends on
which map is used, that should be visible rather than argued.

What is kept identical to run_geology_vs_baselines.py: the rankings (wPCA,
the demeaned match, the normalized cross-correlation match), the 2,000
random draws of 250 windows, the known-deposit filter, and the keyword rule
-- "magnetitito" for the strict set, plus banded iron formation and
jaspelito for the broad one.

What changes: the keywords are matched against ARIM's own LITOTIPO1 and
LITOTIPO2 fields rather than the sheets' LITOTIPOS, and ages come from
ERA_MAXIMA. The nine-class grouping of litho_classes.py is not used here at
all; it colours a figure and no number has ever passed through it.

Two restrictions, both applied to every population including the random
draws, so that nothing is compared across different ground:

  footprint    the ARIM map covers about 90% of the survey, stopping short
               of the eastern edge
  attributed   228 of its 893 polygons carry no lithotype at all, and 226 no
               era. Ground with no attribute cannot be found to be
               magnetitite, so it is excluded rather than counted as absence

Run from inside paper/ (a few minutes):

    python run_geology_arim.py [--draws 2000] [--seed 20261120]

Writes results/geology_arim.json.
"""
from __future__ import annotations

import argparse
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
ARIM = (HERE.parent / "data" / "Carajas_Brazil_Structures"
        / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas"
        / "Litologia" / "arim_carajas_lito.shp")
SHEETS = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_100k_carajas.geojson"
OUT_DIR = HERE / "results"
N_TOP = 250
ERAS = ("Mesoarqueano", "Neoarqueano", "Paleoproterozóico", "Cenozóico")


def grids():
    tmi, _u, ox, oy = R.load_cropped()
    shape = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - shape[0] * R.PX, ox + shape[1] * R.PX, oy)

    def burn(geoms):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(shape, bool)
        return rasterize(geoms, out_shape=shape, transform=transform,
                         fill=0, default_value=1).astype(bool)

    g = gpd.clip(gpd.read_file(ARIM).to_crs("EPSG:32722"), area)
    g = g[~g.geometry.is_empty]
    text = (g["LITOTIPO1"].fillna("") + ", " + g["LITOTIPO2"].fillna("")).str.lower()
    attributed = text.str.strip(", ") != ""

    out = {}
    for name, words in (("magnetitite", STRICT), ("iron formation set", BROAD)):
        hit = attributed & text.apply(lambda s: any(w in s for w in words))
        out[name] = burn(g.geometry[hit])
        print("  %-20s %3d ARIM polygons" % (name, int(hit.sum())))
    for era in ERAS:
        out[era] = burn(g.geometry[g["ERA_MAXIMA"] == era])

    footprint = burn(g.geometry)
    attr_mask = burn(g.geometry[attributed])
    print("  ARIM footprint %.1f%% of the grid; with a lithotype %.1f%%"
          % (100 * footprint.mean(), 100 * attr_mask.mean()))

    # the 1:100,000 sheets, for the side-by-side
    s = gpd.clip(gpd.read_file(SHEETS).to_crs("EPSG:32722"), area)
    s = s[~s.geometry.is_empty]
    st = (s["LITOTIPOS"].fillna("") + " " + s["NOME"].fillna("")).str.lower()
    for name, words in (("magnetitite", STRICT), ("iron formation set", BROAD)):
        out["sheets " + name] = burn(s.geometry[st.apply(
            lambda x: any(w in x for w in words))])
    out["sheets coverage"] = burn(s.geometry)
    return shape, out, attr_mask


def quantities(cells, wshape, g, keys, base):
    out = {}
    for k in keys:
        v = float(coverage(g[k], cells, wshape).mean())
        out[k] = round(v / base[k], 3) if base[k] else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    print("building the ARIM grids", flush=True)
    shape, g, attr = grids()
    nrow, ncol = shape
    deposits = deposit_boxes()
    rng = np.random.default_rng(args.seed)
    keys = ["magnetitite", "iron formation set"] + list(ERAS)
    sheet_keys = ["sheets magnetitite", "sheets iron formation set"]
    out = {"draws": args.draws, "seed": args.seed}

    for case in (1, 2):
        print("\n=== Case %d" % case, flush=True)
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
            Xb, _i, _w, _o = C1.window_matrix_uni()
            d = Xb.shape[0] - 1
            ranks = {"wPCA": order, "demeaned match": rank_demeaned(Xb, d),
                     "correlation match": np.argsort(-ncc_scores(Xb, d))}
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, R.ALPHA_TMI)
            Xb, _i, _w, _o = R.window_matrix(3)
            d = Xb.shape[0] - 1
            npix = wshape[0] * wshape[1]
            nT, nU = ncc_scores(Xb[:, :npix], d), ncc_scores(Xb[:, npix:2 * npix], d)
            dT = np.linalg.norm(_demeaned(Xb[:, :npix]) - _demeaned(Xb[:, :npix])[d], axis=1)
            dU = np.linalg.norm(_demeaned(Xb[:, npix:2 * npix])
                                - _demeaned(Xb[:, npix:2 * npix])[d], axis=1)
            ranks = {"wPCA": order,
                     "demeaned match": np.argsort(R.ALPHA_TMI * dT / dT.std()
                                                  + (1 - R.ALPHA_TMI) * dU / dU.std()),
                     "correlation match": np.argsort(-(R.ALPHA_TMI * nT + (1 - R.ALPHA_TMI) * nU))}

        wh, ww = wshape

        def inside(cells):
            """Keep only window centres on attributed ARIM ground."""
            keep = []
            for r, c in cells:
                rr = min(r + wh // 2, nrow - 1)
                cc = min(c + ww // 2, ncol - 1)
                if attr[rr, cc]:
                    keep.append((r, c))
            return keep

        allb = inside([(r, c) for r, c, _ in idx])
        base = {k: float(coverage(g[k], allb, wshape).mean())
                for k in keys + sheet_keys}
        print("  baseline over %d of %d candidate windows on attributed ground"
              % (len(allb), len(idx)))

        rows = {}
        for lab, rank in ranks.items():
            cells = inside(clear_of_deposits(rank, idx, wshape, origin, dep, deposits))
            rows[lab] = quantities(cells, wshape, g, keys + sheet_keys, base)
            rows[lab]["n"] = len(cells)

        print("  %d random draws" % args.draws, flush=True)
        pool = np.arange(len(idx))
        pool = pool[pool != dep]
        null = []
        for _ in range(args.draws):
            pick = rng.choice(pool, size=N_TOP, replace=False)
            cells = inside(clear_of_deposits(pick, idx, wshape, origin, dep, deposits))
            null.append(quantities(cells, wshape, g, keys + sheet_keys, base))
        rnd = {}
        for k in keys + sheet_keys:
            v = np.array([x[k] for x in null if x[k] is not None], dtype=float)
            rnd[k] = {"mean": round(float(v.mean()), 3),
                      "p5": round(float(np.percentile(v, 5)), 3),
                      "p95": round(float(np.percentile(v, 95)), 3)}
            for lab in rows:
                x = rows[lab][k]
                rows[lab].setdefault("percentile", {})[k] = (
                    round(float((v < x).mean()), 3) if x is not None else None)
        out["case%d" % case] = {"rankings": rows, "random": rnd}

        print("\n  %-22s %8s %9s %9s %9s %16s"
              % ("quantity (ARIM)", "wPCA", "demeaned", "corr.", "random", "5-95% of draws"))
        for k in keys:
            print("  %-22s %8.2f %9.2f %9.2f %9.2f     %.2f-%.2f"
                  % (k, rows["wPCA"][k], rows["demeaned match"][k],
                     rows["correlation match"][k], rnd[k]["mean"],
                     rnd[k]["p5"], rnd[k]["p95"]))
        print("  -- the same windows read off the 1:100,000 sheets --")
        for k in sheet_keys:
            print("  %-22s %8.2f %9.2f %9.2f %9.2f     %.2f-%.2f"
                  % (k.replace("sheets ", ""), rows["wPCA"][k],
                     rows["demeaned match"][k], rows["correlation match"][k],
                     rnd[k]["mean"], rnd[k]["p5"], rnd[k]["p95"]))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "geology_arim.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote %s" % (OUT_DIR / "geology_arim.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
