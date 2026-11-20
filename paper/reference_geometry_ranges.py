"""What rotations and scales the Carajas deposits themselves justify (R2.3).

The rotation and scale sensitivity test needs a range that the province
supports, not a round number. This measures three things from the data that
Case 1 already uses, and prints the ranges they imply.

1. Body orientation. The long-axis azimuth of each deposit outline, from the
   principal axis of its vertices. The spread of these about the reference
   deposit is how much a body's strike actually varies here, and so how far
   the reference window has to turn to stand for another deposit.

2. Dipole orientation. The azimuth of the low-to-high axis of each deposit's
   TMI window, taken between the centroids of its coldest and hottest tenth.
   The grids are used as flown, without reduction to the pole, so this axis is
   set by the inducing field and is the same everywhere in the survey. It is
   the part of the window geometry that cannot turn with the body, and the
   spread of these azimuths measures how nearly fixed it is.

3. Body size. The square root of each outline's area, relative to the
   reference. This is the factor by which a body of this province differs in
   linear size from Paulo Afonso, and so the scale range the test should span.

Run from inside paper/:

    python reference_geometry_ranges.py
"""
from __future__ import annotations

import math

import numpy as np

import case1_uni_repro as C1
import numpy_repro_concat as R

NAMES = {
    1: "Tucuma", 2: "Pedra Branca", 3: "Alemao", 4: "Furnas",
    5: "Salobo", 6: "Paulo Afonso", 7: "Cabano", 8: "AQW1 and AQW2",
    9: "Pantera", 10: "Guepardo", 11: "Sossego", 12: "Estrela",
}
COLD_HOT_FRACTION = 0.10


def _vertices(rings):
    return np.array([p for ring in rings for p in ring], dtype=float)


def long_axis_azimuth(rings):
    """Azimuth of the outline's principal axis, degrees east of north, 0-180."""
    pts = _vertices(rings)
    pts = pts - pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts, full_matrices=False)
    east, north = vt[0]
    return math.degrees(math.atan2(east, north)) % 180.0


def outline_size(rings):
    """Square root of the outline area, in metres."""
    area = abs(R.poly_area(rings[0]))
    for ring in rings[1:]:
        area -= abs(R.poly_area(ring))
    return math.sqrt(max(area, 0.0))


def dipole_azimuth(window):
    """Azimuth of the low-to-high axis of a TMI window, degrees east of north."""
    finite = np.isfinite(window)
    values = window[finite]
    if values.size < 20:
        return None
    lo = np.quantile(values, COLD_HOT_FRACTION)
    hi = np.quantile(values, 1.0 - COLD_HOT_FRACTION)
    rows, cols = np.nonzero(finite)
    cold = window[rows, cols] <= lo
    hot = window[rows, cols] >= hi
    if not cold.any() or not hot.any():
        return None
    # Rows increase southward, so a northward step is a decrease in row.
    north = rows[cold].mean() - rows[hot].mean()
    east = cols[hot].mean() - cols[cold].mean()
    return math.degrees(math.atan2(east, north)) % 360.0


def signed_difference(angle, reference, period=180.0):
    """Smallest signed difference angle - reference, within +/- period/2."""
    return (angle - reference + period / 2.0) % period - period / 2.0


def main() -> int:
    tmi, _u, ox, oy = R.load_cropped()
    deposits = R.read_shp_polygons(C1.CASE1_SHP)
    height, width = tmi.shape

    rows = []
    for dep in range(1, 13):
        rings = deposits[dep - 1]
        xs = [p[0] for ring in rings for p in ring]
        ys = [p[1] for ring in rings for p in ring]
        cs = max(math.floor((min(xs) - ox) / R.PX), 0)
        ce = min(math.ceil((max(xs) - ox) / R.PX), width)
        rs = max(math.floor((oy - max(ys)) / R.PX), 0)
        re = min(math.ceil((oy - min(ys)) / R.PX), height)
        rows.append({
            "id": dep,
            "name": NAMES[dep],
            "strike": long_axis_azimuth(rings),
            "size_m": outline_size(rings),
            "dipole": dipole_azimuth(tmi[rs:re, cs:ce]),
        })

    ref = next(r for r in rows if r["id"] == C1.REF_DEP)
    print(f"reference: deposit {ref['id']} ({ref['name']})\n")
    print(f"{'ID':>3} {'Deposit':16} {'strike':>7} {'d strike':>9} "
          f"{'size (km)':>10} {'size ratio':>11} {'dipole':>7} {'d dipole':>9}")
    for r in rows:
        d_strike = signed_difference(r["strike"], ref["strike"])
        ratio = r["size_m"] / ref["size_m"]
        dip = "   n/a" if r["dipole"] is None else f"{r['dipole']:6.1f}"
        d_dip = ("      n/a" if r["dipole"] is None or ref["dipole"] is None
                 else f"{signed_difference(r['dipole'], ref['dipole'], 360.0):+8.1f}")
        print(f"{r['id']:>3} {r['name']:16} {r['strike']:6.1f}  {d_strike:+8.1f} "
              f"{r['size_m'] / 1000:9.2f} {ratio:10.2f}  {dip} {d_dip}")

    d_strikes = [signed_difference(r["strike"], ref["strike"])
                 for r in rows if r["id"] != ref["id"]]
    ratios = [r["size_m"] / ref["size_m"] for r in rows if r["id"] != ref["id"]]
    dipoles = [r["dipole"] for r in rows if r["dipole"] is not None]

    print("\nranges the province supports")
    print(f"  body strike about the reference: {min(d_strikes):+.0f} to "
          f"{max(d_strikes):+.0f} deg, half-range {max(abs(min(d_strikes)), abs(max(d_strikes))):.0f}")
    print(f"  size relative to the reference : {min(ratios):.2f} to {max(ratios):.2f}")
    if len(dipoles) > 1:
        ref_dip = dipoles[0]
        spread = [signed_difference(d, ref_dip, 360.0) for d in dipoles]
        print(f"  dipole axis across the survey  : {min(spread):+.0f} to {max(spread):+.0f} deg "
              f"about deposit {[r['id'] for r in rows if r['dipole'] is not None][0]}, "
              f"which the field fixes and the body cannot turn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
