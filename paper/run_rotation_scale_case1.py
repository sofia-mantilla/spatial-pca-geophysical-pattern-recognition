"""Case-1 rotation and scale sensitivity of the reference window (R2.3).

Reviewer 2 asks how robust the ranking is to a reference deposit of a
different orientation or size. This repeats Case 1 (univariate TMI, reference
Paulo Afonso = deposit 6, k=17) with the reference re-sampled over the range
of orientations and sizes the province actually shows, and scores each variant
the way Appendix C scores a reference swap: c(250), AUC, hits, a(250) and the
ratio to a random selection of equal area.

The two variants are not the same kind of change, because the method takes the
window dimensions from the reference deposit's own extent (Section 3.2).

  Rotation turns the body without changing how much ground it covers, so the
  window keeps its 28 x 46 pixels and only its contents turn. The sliding
  windows, n and the random expectation are shared across angles.

  Scale changes how much ground the body covers, so it changes the window.
  A body at 0.5 spreads the same dipole over half the distance, which at 200 m
  cells is a 14 x 23 window holding the same shape in fewer pixels. The
  sliding windows are re-extracted at that shape, n changes with it, and the
  random expectation is recomputed for each scale. Nothing is zoomed out and
  no extra background enters the window.

Angles and scales are measured, not chosen. reference_geometry_ranges.py reads
the long-axis azimuth and the area of all twelve outlines and the dipole axis
of all twelve TMI windows. The nine IOCG deposits strike within -2 to +14
degrees of the reference, along the Cinzento trend, while their dipole axes
hold at 126 to 175 degrees whatever the body does, which is the field-fixed
axis the no-reduction-to-pole processing implies. Only Guepardo and Estrela,
the two deposits that are not IOCG, strike elsewhere, at -40 and -53 degrees.
Every other deposit is between 0.33 and 0.95 of the reference's linear size;
none is larger. 90 degrees and 1.25 are controls: neither occurs here.

k is held at 17 throughout, so each variant answers what happens to someone
who keeps the published settings and brings a different reference deposit.

Verification gate: the (0 degrees, 1.0) variant must return the published
Case-1 numbers, c(250)=47.0, AUC 57.9, 5 hits, n=13,031.

Run from inside paper/:

    python run_rotation_scale_case1.py

Writes results/rotation_scale_case1.csv and results/rotation_scale_case1.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from skimage.transform import resize, warp

import case1_uni_repro as C1
import null_expectation as NE
import numpy_repro_concat as R

K_PCS = 17
T_MAX = 6000
ANGLES = (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, -40.0, -55.0, 90.0)
SCALES = (1.0, 0.95, 0.8, 0.65, 0.5, 0.35, 1.25)
OUT_DIR = Path(__file__).resolve().parent / "results"
ORIGIN = (0.0, 0.0)


def reference_bbox():
    """Pixel bounds of the Case-1 reference window, as window_matrix_uni cuts it."""
    tmi, _u, ox, oy = R.load_cropped()
    rings = R.read_shp_polygons(C1.CASE1_SHP)[C1.REF_DEP - 1]
    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    cs = math.floor((min(xs) - ox) / R.PX)
    ce = math.ceil((max(xs) - ox) / R.PX)
    rs = math.floor((oy - max(ys)) / R.PX)
    re = math.ceil((oy - min(ys)) / R.PX)
    height, width = tmi.shape
    return tmi, max(rs, 0), min(re, height), max(cs, 0), min(ce, width)


def rotated_reference(tmi, rs, re, cs, ce, angle):
    """The reference window with its contents turned by angle, same pixel shape.

    Samples come from the raster itself, so the corners that swing in from
    outside the deposit bounds are real data and nothing is padded. Returns
    None if the turned window reaches past the grid or into no-data.
    """
    native = tmi[rs:re, cs:ce]
    if angle == 0.0:
        return native

    wh, ww = native.shape
    row_centre = (rs + re - 1) / 2.0
    col_centre = (cs + ce - 1) / 2.0
    d_row, d_col = np.meshgrid(
        np.arange(wh, dtype=float) - (wh - 1) / 2.0,
        np.arange(ww, dtype=float) - (ww - 1) / 2.0,
        indexing="ij",
    )
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Inverse map, output pixel to source pixel. Rows increase southward, which
    # is why the row term carries the sign opposite to a counter-clockwise
    # rotation in map coordinates.
    coords = np.stack([
        row_centre + d_row * cos_t + d_col * sin_t,
        col_centre - d_row * sin_t + d_col * cos_t,
    ])
    support = warp(np.isfinite(tmi).astype(float), coords, order=1,
                   mode="constant", cval=0.0, preserve_range=True)
    if not np.allclose(support, 1.0):
        return None
    patch = warp(tmi, coords, order=1, mode="constant", cval=np.nan,
                 preserve_range=True)
    return None if np.isnan(patch).any() else patch


def scaled_reference(native, scale):
    """The same dipole over a smaller or larger window, at the same cell size.

    A body 0.5 the linear size spreads the same picture over half the distance,
    so the window that contains it is half as many pixels across and holds the
    same shape at fewer samples. Downsampling is anti-aliased.
    """
    if scale == 1.0:
        return native
    wh, ww = native.shape
    shape = (max(2, int(round(wh * scale))), max(2, int(round(ww * scale))))
    return resize(native, shape, order=1, anti_aliasing=scale < 1.0,
                  preserve_range=True, mode="edge")


def window_matrix_for(tmi, template):
    """Sliding windows at this template's shape, with the template appended.

    Follows case1_uni_repro.window_matrix_uni exactly, for an arbitrary shape.
    """
    wh, ww = template.shape
    stride = R.STRIDE
    pad_y = (stride - (tmi.shape[0] - wh) % stride) % stride
    pad_x = (stride - (tmi.shape[1] - ww) % stride) % stride
    padded = np.pad(tmi, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
    ncols = (padded.shape[1] - ww) // stride + 1
    view = np.lib.stride_tricks.sliding_window_view(padded, (wh, ww))[::stride, ::stride]
    keep = ~np.isnan(view).any(axis=(2, 3))
    rows, cols = np.nonzero(keep)
    windows = view[rows, cols].reshape(len(rows), -1)
    X = np.vstack([windows, template.ravel()])
    idx = np.stack([rows * stride, cols * stride, rows * ncols + cols], axis=1)
    return X, idx, (wh, ww)


def null_for(idx, wshape, n, grid, deps):
    """Random-selection expectation and union curve for one window shape."""
    rects = NE.rects_from_idx(idx, wshape, ORIGIN)
    cells, areas = NE.build_cells(deps, C1.REF_DEP - 1, rects)
    return {
        "ranks": np.arange(1, T_MAX + 1),
        "curve": NE.exact_random_curve(cells, areas, n, t_max=T_MAX),
        "union": NE.exact_random_union(idx, wshape, n, grid, t_max=T_MAX),
    }


def score(X, idx, wshape, null, grid, valid):
    """Rank with the appended reference row, then score recovery against it."""
    dep = X.shape[0] - 1
    Z = R.block_scores(X, K_PCS)
    dist = np.sqrt(((Z - Z[dep]) ** 2) @ R.blk_w(Z[dep]))
    order = np.argsort(dist)

    res = C1.recovery(order, idx, wshape, ORIGIN, dep)
    m = R.metrics(res)
    union = NE.union_area_curve(order, idx, wshape, dep, grid)
    t_star = float(np.interp(union[-1], null["union"], null["ranks"]))
    expected = float(np.interp(t_star, null["ranks"], null["curve"]))
    return {
        "n": int(dep),
        "window": f"{wshape[0]}x{wshape[1]}",
        "end": float(m["end"]),
        "auc": float(m["auc"]),
        "hits": int(m["hits"]),
        "hit_ranks": {str(k): v for k, v in m["hit_ranks"].items()},
        "union250_frac": float(union[-1] / valid),
        "rand_matched_area": expected,
        "ratio_matched": float(m["end"] / 100.0 / expected) if expected > 0 else None,
    }


def report(key, row):
    print(f"{key:10s} {row['window']:>7}  n {row['n']:6d}  end {row['end']:5.1f}%"
          f"  auc {row['auc']:6.1f}  hits {row['hits']:d}"
          f"  a(250) {100 * row['union250_frac']:4.1f}%"
          f"  ratio {row['ratio_matched']:.2f}", flush=True)


def main() -> int:
    global ORIGIN
    tmi, rs, re, cs, ce = reference_bbox()
    _tmi, _u, ox, oy = R.load_cropped()
    ORIGIN = (ox, oy)
    grid = tmi.shape
    valid = int(np.isfinite(tmi).sum())
    native = tmi[rs:re, cs:ce]
    deps = C1.deposits_case1_wgs84()
    out: dict[str, dict] = {}

    # Rotation: the window shape never changes, so one null serves every angle.
    X0, idx0, wshape0 = window_matrix_for(tmi, native)
    null0 = null_for(idx0, wshape0, X0.shape[0] - 1, grid, deps)
    for angle in ANGLES:
        key = f"rot{angle:+g}"
        template = rotated_reference(tmi, rs, re, cs, ce, angle)
        if template is None:
            out[key] = {"angle_deg": angle, "scale": 1.0, "error": "reaches no-data"}
            print(f"{key:10s} SKIP: reaches outside the grid or into no-data", flush=True)
            continue
        row = {"angle_deg": angle, "scale": 1.0}
        row.update(score(np.vstack([X0[:-1], template.ravel()]),
                         idx0, wshape0, null0, grid, valid))
        out[key] = row
        report(key, row)

    # Scale: the window shape changes with the body, so everything is rebuilt.
    for scale in SCALES:
        key = f"scale{scale:g}"
        X, idx, wshape = window_matrix_for(tmi, scaled_reference(native, scale))
        null = null_for(idx, wshape, X.shape[0] - 1, grid, deps)
        row = {"angle_deg": 0.0, "scale": scale}
        row.update(score(X, idx, wshape, null, grid, valid))
        out[key] = row
        report(key, row)

    base = out.get("rot+0")
    if base and "end" in base:
        assert abs(base["end"] - 47.04) < 0.05, f"gate failed: end={base['end']}"
        assert base["hits"] == 5, f"gate failed: hits={base['hits']}"
        assert base["n"] == 13031, f"gate failed: n={base['n']}"
        assert out["scale1"]["end"] == base["end"], "scale 1.0 must equal rotation 0"
        print("gate passed: the unrotated, unscaled reference reproduces Case 1", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "rotation_scale_case1.json").open("w") as handle:
        json.dump(out, handle, indent=1)
    columns = ["angle_deg", "scale", "window", "n", "end", "auc", "hits",
               "union250_frac", "rand_matched_area", "ratio_matched"]
    with (OUT_DIR / "rotation_scale_case1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in out.values():
            writer.writerow(row)
    print(f"wrote {OUT_DIR / 'rotation_scale_case1.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
