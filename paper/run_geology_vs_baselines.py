"""The geology of the selection, against random and against the standard methods.

Appendix C compares the geology under the top 250 windows with the geology
under every window the search considers. That says whether the ranking favours
a kind of ground, but not whether the difference is larger than a draw of 250
windows would give by chance, and not whether a classical template matcher on
the same data would have selected the same geology. The validation of
Sections 4 and 5 asks both of those questions of the recovery metrics. This
asks them of the maps, on the same footing.

Three populations, for each case:

  wPCA            the published ranking
  standard        the two classical template baselines of Appendix C, a
                  demeaned Euclidean match and a normalized cross-correlation
                  match, ranked and cut at the same 250
  random          draws of 250 windows from the same candidate pool, passed
                  through the same known-deposit filter

Every quantity is read exactly as the script that publishes it reads it:
magnetite cover from run_magnetite_units, era shares from run_era_shares,
contact distance from run_contact_proximity, shear-zone distance from
run_structure_proximity, including its restriction to the ARIM footprint.
What is new is the random distribution behind each number and the classical
rankings to put beside it.

The classical scores are the ones defined in run_ablation_checks.py, repeated
here because that file is a script and importing it would run it.

Run from inside paper/ (a few minutes, most of it the two rankings):

    python run_geology_vs_baselines.py [--draws 2000] [--seed 20261120]

Writes results/geology_vs_baselines.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box

import case1_uni_repro as C1
import numpy_repro_concat as R
from run_contact_proximity import contact_distance_grid
from run_era_shares import era_masks
from run_magnetite_units import coverage, masks as magnetite_masks
from run_structure_proximity import grids as structure_grids

OUT_DIR = Path(__file__).resolve().parent / "results"
N_TOP = 250


# -- the classical template baselines, as run_ablation_checks.py defines them --
def _demeaned(Xb):
    return Xb - Xb.mean(axis=1, keepdims=True)


def rank_demeaned(Xb, d):
    Xc = _demeaned(Xb)
    return np.argsort(np.linalg.norm(Xc - Xc[d], axis=1))


def ncc_scores(Xb, d):
    Xc = _demeaned(Xb)
    nrm = np.linalg.norm(Xc, axis=1)
    nrm = np.where(nrm == 0, 1.0, nrm)
    return (Xc / nrm[:, None]) @ (Xc / nrm[:, None])[d]


def deposit_boxes():
    return [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
            for rings in C1.deposits_case1_wgs84()]


def clear_cells(windows, idx, wshape, origin, deposits):
    """(row, col) of those windows whose footprint touches no known deposit."""
    ox, oy = origin
    wh, ww = wshape
    out = []
    for i in windows:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
        if not any(rect.intersects(d) for d in deposits):
            out.append((r, c))
    return out


def quantities(cells, wshape, grids, baseline):
    """Every published geological quantity, for one set of windows."""
    wh, ww = wshape
    nrow, ncol = grids["contact"].shape
    rr = np.clip(np.array([r for r, _ in cells]) + wh // 2, 0, nrow - 1)
    cc = np.clip(np.array([c for _, c in cells]) + ww // 2, 0, ncol - 1)
    out = {}
    for key in ("magnetite strict", "magnetite broad"):
        v = float(coverage(grids[key], cells, wshape).mean())
        out[key] = v / baseline[key] if baseline[key] else float("nan")
    for key in ("Archean", "Palaeoproterozoic", "Neoproterozoic and younger"):
        v = float(coverage(grids[key], cells, wshape).mean())
        out[key] = v / baseline[key] if baseline[key] else float("nan")
    out["contact distance m"] = float(np.median(grids["contact"][rr, cc]))
    inside = grids["arim"][rr, cc]
    out["shear distance m"] = (float(np.median(grids["shear"][rr[inside], cc[inside]]))
                               if inside.any() else float("nan"))
    return out


def baseline_means(all_cells, wshape, grids):
    base = {}
    for key in ("magnetite strict", "magnetite broad", "Archean",
                "Palaeoproterozoic", "Neoproterozoic and younger"):
        base[key] = float(coverage(grids[key], all_cells, wshape).mean())
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    print("building the map grids", flush=True)
    _tmi, mag = magnetite_masks()
    eras = era_masks()
    _t2, contact, _o = contact_distance_grid()
    _t3, sdists, arim, _o2 = structure_grids()

    grids = {"magnetite strict": mag["strict"], "magnetite broad": mag["broad"],
             "Archean": np.zeros_like(mag["strict"]),
             "Palaeoproterozoic": eras["Paleoproterozóico"],
             "Neoproterozoic and younger": np.zeros_like(mag["strict"]),
             "contact": contact, "shear": sdists["shear zones"], "arim": arim}
    for era in ("Paleoarqueano", "Mesoarqueano", "Neoarqueano"):
        grids["Archean"] |= eras[era]
    for era in ("Neoproterozóico", "Paleozóico", "Mesozóico", "Cenozóico"):
        grids["Neoproterozoic and younger"] |= eras[era]

    deposits = deposit_boxes()
    rng = np.random.default_rng(args.seed)
    out = {"draws": args.draws, "seed": args.seed}

    for case in (1, 2):
        print("\n=== Case %d" % case, flush=True)
        if case == 1:
            order, idx, wshape, origin, dep = C1.rank_spca(17)
            Xb, _i, _w, _o = C1.window_matrix_uni()
            rankings = {
                "wPCA (published)": order,
                "demeaned match": rank_demeaned(Xb, Xb.shape[0] - 1),
                "correlation match": np.argsort(-ncc_scores(Xb, Xb.shape[0] - 1)),
            }
        else:
            order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, R.ALPHA_TMI)
            Xb, _i, _w, _o = R.window_matrix(3)
            d = Xb.shape[0] - 1
            npix = wshape[0] * wshape[1]
            nT = ncc_scores(Xb[:, :npix], d)
            nU = ncc_scores(Xb[:, npix:2 * npix], d)
            dT = np.linalg.norm(_demeaned(Xb[:, :npix])
                                - _demeaned(Xb[:, :npix])[d], axis=1)
            dU = np.linalg.norm(_demeaned(Xb[:, npix:2 * npix])
                                - _demeaned(Xb[:, npix:2 * npix])[d], axis=1)
            rankings = {
                "wPCA (published)": order,
                "demeaned match": np.argsort(R.ALPHA_TMI * dT / dT.std()
                                             + (1 - R.ALPHA_TMI) * dU / dU.std()),
                "correlation match": np.argsort(-(R.ALPHA_TMI * nT + (1 - R.ALPHA_TMI) * nU)),
            }

        all_cells = [(r, c) for r, c, _ in idx]
        base = baseline_means(all_cells, wshape, grids)

        rows = {}
        for label, rank in rankings.items():
            cells = clear_cells([w for w in rank if w != dep][:N_TOP],
                                idx, wshape, origin, deposits)
            rows[label] = quantities(cells, wshape, grids, base)
            rows[label]["n"] = len(cells)

        print("  %d random draws of %d windows" % (args.draws, N_TOP), flush=True)
        pool = np.arange(len(idx))
        pool = pool[pool != dep]
        null = []
        for _ in range(args.draws):
            pick = rng.choice(pool, size=N_TOP, replace=False)
            cells = clear_cells(pick, idx, wshape, origin, deposits)
            null.append(quantities(cells, wshape, grids, base))

        keys = [k for k in rows["wPCA (published)"] if k != "n"]
        rnd = {}
        for k in keys:
            v = np.array([d[k] for d in null], dtype=float)
            v = v[np.isfinite(v)]
            rnd[k] = {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)),
                      "p95": float(np.percentile(v, 95))}
            for label in rows:
                x = rows[label][k]
                rows[label].setdefault("percentile", {})[k] = (
                    float((v < x).mean()) if np.isfinite(x) else None)
        out["case%d" % case] = {"rankings": rows, "random": rnd}

        unit = {"contact distance m": "m", "shear distance m": "m"}
        print("\n  %-28s %9s %9s %9s %9s %19s"
              % ("quantity", "wPCA", "demeaned", "corr.", "random", "wPCA percentile"))
        for k in keys:
            f = "%9.0f" if k in unit else "%9.2f"
            print("  %-28s" % k
                  + f % rows["wPCA (published)"][k]
                  + f % rows["demeaned match"][k]
                  + f % rows["correlation match"][k]
                  + f % rnd[k]["mean"]
                  + "   %5.1f%% (5-95%%: " % (100 * rows["wPCA (published)"]["percentile"][k])
                  + (f % rnd[k]["p5"]).strip() + ", " + (f % rnd[k]["p95"]).strip() + ")")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "geology_vs_baselines.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("\nwrote %s" % (OUT_DIR / "geology_vs_baselines.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
