"""Which mapped units does each selection occupy, against chance?

The ranking uses no geological information, so the units it lands on are an
independent check on the ground it selects. Units are the ARIM Carajas
1:250,000 map's own (SIGLA_UNID): no grouping or keyword rule of ours enters
this, which matters because the enrichment of a class we defined would partly
measure how we drew the class.

For each selection, the share of its area falling in each unit is compared
with that unit's share of the surveyed area, and with the same quantity for
2,000 random selections of the same size. Windows touching a known deposit
are dropped from every population, the real ones and the random draws alike,
so a unit cannot be enriched merely because the ranking found a deposit in it.

Geometry is taken on the 200 m raster grid, the grid the ranking itself runs
on, so a window's composition is a count of its cells.

Run from inside paper/:

    python run_case2_unit_enrichment.py [--draws 2000]

Writes results/case2_unit_enrichment.json.
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
from make_fig_litho_targets import deposit_frames

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
LITO = (HERE.parent / "data" / "Carajas_Brazil_Structures"
        / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas"
        / "Litologia" / "arim_carajas_lito.shp")
UTM22S = "EPSG:32722"
N_TOP = 250
MIN_SURVEY_SHARE = 1.0       # below this a ratio is noise; reported but flagged


def unit_raster():
    """ARIM units burned onto the ranking's own grid, plus the code table."""
    tmi, _u, ox, oy = R.load_cropped()
    shape = tmi.shape
    transform = from_origin(ox, oy, R.PX, R.PX)
    area = box(ox, oy - shape[0] * R.PX, ox + shape[1] * R.PX, oy)
    g = gpd.clip(gpd.read_file(LITO).to_crs(UTM22S), area)
    g = g[~g.geometry.is_empty]
    sig = sorted(set(g["SIGLA_UNID"].astype(str)))
    code = {s: i + 1 for i, s in enumerate(sig)}
    name = {}
    for s, n in zip(g["SIGLA_UNID"].astype(str), g["NOME_UNIDA"].astype(str)):
        name.setdefault(s, n)
    burned = rasterize(
        ((geom, code[str(s)]) for geom, s in zip(g.geometry, g["SIGLA_UNID"])),
        out_shape=shape, transform=transform, fill=0, dtype="int32")
    valid = np.isfinite(tmi) & (burned > 0)
    print("%d polygons, %d units, ARIM covers %.1f%% of the valid grid"
          % (len(g), len(sig), 100 * valid.sum() / np.isfinite(tmi).sum()))
    return burned, valid, {v: k for k, v in code.items()}, name, shape


def shares(mask, burned, valid, ncode):
    """Percentage of the masked area falling in each unit code."""
    sel = burned[mask & valid]
    if sel.size == 0:
        return np.zeros(ncode + 1)
    out = np.bincount(sel, minlength=ncode + 1).astype(float)
    return 100.0 * out / out.sum()


def window_mask(rects_rc, wshape, shape):
    """Boolean grid of the cells any of these windows covers."""
    wh, ww = wshape
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in rects_rc:
        m[r:min(r + wh, H), c:min(c + ww, W)] = True
    return m


def selection(case, scale=R.BLOCK_SCALE, alpha=None):
    """Top-250 window positions that touch no known deposit."""
    if case == 1:
        order, idx, wshape, origin, dep = C1.rank_spca(17)
    else:
        a = R.ALPHA_TMI if alpha is None else alpha
        order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, a, scale=scale)
    return _clear(order, idx, wshape, origin, dep), wshape, (order, idx, wshape, origin, dep)


def _clear(order, idx, wshape, origin, dep, pick=None):
    ox, oy = origin
    wh, ww = wshape
    deps = deposit_frames().geometry
    out = []
    budget = pick if pick is not None else [w for w in order if w != dep][:N_TOP]
    for i in budget:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        if not deps.intersects(box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)).any():
            out.append((r, c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args()

    burned, valid, decode, name, shape = unit_raster()
    ncode = max(decode)
    survey = shares(valid, burned, valid, ncode)

    out = {"draws": a.draws, "n_units": ncode,
           "note": "ARIM 1:250,000 SIGLA_UNID; no grouping of ours",
           "units": {}}
    rng = np.random.default_rng(a.seed)
    cases = {}

    for label, kw in (("case1", dict(case=1)),
                      ("case2", dict(case=2)),
                      ("case2_unscaled", dict(case=2, scale=None, alpha=0.503))):
        rc, wshape, (order, idx, ws, origin, dep) = selection(**kw)
        sel = shares(window_mask(rc, wshape, shape), burned, valid, ncode)
        cases[label] = {"n_windows_clear": len(rc), "share": sel}
        if label == "case2_unscaled":
            continue
        pool = np.arange(len(idx))
        pool = pool[pool != dep]
        draws = np.zeros((a.draws, ncode + 1))
        for d in range(a.draws):
            pick = rng.choice(pool, size=N_TOP, replace=False)
            rc_r = _clear(order, idx, ws, origin, dep, pick=pick)
            draws[d] = shares(window_mask(rc_r, wshape, shape), burned, valid, ncode)
        cases[label]["draws"] = draws
        print("%s: %d windows clear of deposits, %d random draws done"
              % (label, len(rc), a.draws))

    for code in range(1, ncode + 1):
        sg = decode[code]
        rec = {"name": name.get(sg, sg), "survey_share_pct": round(survey[code], 3)}
        for label in ("case1", "case2", "case2_unscaled"):
            c = cases[label]
            s = c["share"][code]
            rec[label] = {"selected_share_pct": round(s, 3),
                          "enrichment": round(s / survey[code], 3) if survey[code] else None}
            if "draws" in c:
                d = c["draws"][:, code]
                lo, hi = np.percentile(d, [5, 95])
                rec[label]["random_share_p5_p95"] = [round(lo, 3), round(hi, 3)]
                rec[label]["above_random"] = bool(s > hi)
                rec[label]["below_random"] = bool(s < lo)
        out["units"][sg] = rec

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "case2_unit_enrichment.json").write_text(json.dumps(out, indent=1))

    big = [(sg, r) for sg, r in out["units"].items()
           if r["survey_share_pct"] >= MIN_SURVEY_SHARE]
    big.sort(key=lambda kv: -kv[1]["case2"]["enrichment"])
    print("\nARIM units covering at least %.0f%% of the survey (%d of %d)"
          % (MIN_SURVEY_SHARE, len(big), ncode))
    print("  %-32s %7s | %6s %6s %-9s | %6s %6s"
          % ("unit", "survey", "C2 %", "x", "vs rand", "C1 %", "x"))
    for sg, r in big:
        c2, c1 = r["case2"], r["case1"]
        flag = "above" if c2["above_random"] else ("below" if c2["below_random"] else "")
        print("  %-32s %6.1f | %6.1f %6.2f %-9s | %6.1f %6.2f"
              % (r["name"][:32], r["survey_share_pct"],
                 c2["selected_share_pct"], c2["enrichment"], flag,
                 c1["selected_share_pct"], c1["enrichment"]))
    print("\nwrote %s" % (OUT_DIR / "case2_unit_enrichment.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
