"""Pure-numpy replication of the concat-fusion pipeline (no sklearn/rasterio/geopandas).

Replicates, bit-compatibly, the exact chain used by the concat experiment scripts:
  ERS raster read -> crop to demo polygon (rasterio.mask semantics) -> pad_raster ->
  stride-8 sliding windows (shared NaN mask, both variables) -> per-block standardized
  PCA scores (full SVD) -> concat ranking with alpha + z_dep^2 block weights ->
  top-250 windows -> footprint recovery (exact geometry: rect x quad clipping, union
  via cell decomposition; deposits reprojected SAD69->WGS84 UTM22S by Helmert).

Written to run in an environment without the geo stack. VERIFY against cached
pickles before trusting new numbers (see verify_against_cache()).
"""
from __future__ import annotations

import math
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
for _cand in (DATA, HERE.parent / "data", HERE.parent / "Spatial_PCA_Multivariate_Repo" / "data"):
    if (_cand / "Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp").exists():
        DATA = _cand
        break
TMI_BIN = DATA / "Carajas_Brazil_Univariate_TMI/1097_1125_1129_TMI_merged"
U_BIN = DATA / "Carajas_Brazil_Multivariate_TMI_U/1097_1125_1129_RAD_eU_merged"
DEMO_SHP = DATA / "Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp"
DEP_SHP = DATA / "Carajas_Brazil_Multivariate_TMI_U/Prospect_in Carajas_multi.shp"

NL, NC = 3527, 2485          # full raster lines/cells
PX = 200.0                    # cell size (m)
X0, Y0 = 279900.0, 9656100.0  # full-raster top-left (registration coord)
NODATA = -99999.0
STRIDE = 8
N_TOP = 250
MIN_COVER = 0.5

# GRS67 (SAD69) / WGS84 ellipsoids, UTM zone 22S
GRS67 = (6378160.0, 1 / 298.247167427)
WGS84 = (6378137.0, 1 / 298.257223563)
UTM = dict(k0=0.9996, lon0=math.radians(-51.0), fe=500000.0, fn=10000000.0)
# SAD69 -> WGS84 geocentric translation. pyproj/PROJ picked the IBGE
# (-66.87, 4.37, -38.52) transformation; verified against cached
# dep_area_by_deposit to 0.65 m^2 out of 43 km^2 (1.5e-8 relative).
HELMERT = (-66.87, 4.37, -38.52)


# ---------------------------------------------------------------- shapefile
def read_shp_polygons(path: Path):
    b = Path(path).read_bytes()
    recs, i = [], 100
    while i < len(b):
        _, clen = struct.unpack(">ii", b[i:i + 8]); i += 8
        c = b[i:i + clen * 2]; i += clen * 2
        if struct.unpack("<i", c[:4])[0] != 5:
            continue
        nparts, npts = struct.unpack("<ii", c[36:44])
        parts = struct.unpack("<%di" % nparts, c[44:44 + 4 * nparts])
        pts = struct.unpack("<%dd" % (2 * npts), c[44 + 4 * nparts:44 + 4 * nparts + 16 * npts])
        rings = []
        bounds = list(parts) + [npts]
        for p0, p1 in zip(bounds[:-1], bounds[1:]):
            rings.append([(pts[2 * k], pts[2 * k + 1]) for k in range(p0, p1)])
        recs.append(rings)
    return recs


# ------------------------------------------------- transverse mercator (Krüger)
def _tm_consts(a, f):
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64 + n**6 / 256)
    alpha = [
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180 - 127 * n**5 / 288 + 7891 * n**6 / 37800,
        13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440 + 281 * n**5 / 630 - 1983433 * n**6 / 1935360,
        61 * n**3 / 240 - 103 * n**4 / 140 + 15061 * n**5 / 26880 + 167603 * n**6 / 181440,
        49561 * n**4 / 161280 - 179 * n**5 / 168 + 6601661 * n**6 / 7257600,
        34729 * n**5 / 80640 - 3418889 * n**6 / 1995840,
        212378941 * n**6 / 319334400,
    ]
    beta = [
        n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360 - 81 * n**5 / 512 + 96199 * n**6 / 604800,
        n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440 + 46 * n**5 / 105 - 1118711 * n**6 / 3870720,
        17 * n**3 / 480 - 37 * n**4 / 840 - 209 * n**5 / 4480 + 5569 * n**6 / 90720,
        4397 * n**4 / 161280 - 11 * n**5 / 504 - 830251 * n**6 / 7257600,
        4583 * n**5 / 161280 - 108847 * n**6 / 3991680,
        20648693 * n**6 / 638668800,
    ]
    return n, A, alpha, beta


def tm_inverse(x, y, a, f):
    """UTM 22S easting/northing -> (lat, lon) radians on ellipsoid (a, f)."""
    n, A, _, beta = _tm_consts(a, f)
    e = math.sqrt(f * (2 - f))
    xi = (y - UTM["fn"]) / (UTM["k0"] * A)
    eta = (x - UTM["fe"]) / (UTM["k0"] * A)
    xi_p, eta_p = xi, eta
    for j, b in enumerate(beta, start=1):
        xi_p -= b * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
        eta_p -= b * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi
    for _ in range(8):
        s = math.sinh(e * math.atanh(e * math.sin(lat)))
        lat = math.asin(math.tanh(math.asinh(math.tan(chi)) + math.asinh(s)) )
    lon = UTM["lon0"] + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return lat, lon


def tm_forward(lat, lon, a, f):
    n, A, alpha, _ = _tm_consts(a, f)
    e = math.sqrt(f * (2 - f))
    t = math.sinh(math.atanh(math.sin(lat)) - e * math.atanh(e * math.sin(lat)))
    xi_p = math.atan2(t, math.cos(lon - UTM["lon0"]))
    eta_p = math.atanh(math.sin(lon - UTM["lon0"]) / math.sqrt(1 + t * t))
    xi, eta = xi_p, eta_p
    for j, al in enumerate(alpha, start=1):
        xi += al * math.sin(2 * j * xi_p) * math.cosh(2 * j * eta_p)
        eta += al * math.cos(2 * j * xi_p) * math.sinh(2 * j * eta_p)
    x = UTM["fe"] + UTM["k0"] * A * eta
    y = UTM["fn"] + UTM["k0"] * A * xi
    return x, y


def geodetic_to_geocentric(lat, lon, a, f):
    e2 = f * (2 - f)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return (N * math.cos(lat) * math.cos(lon),
            N * math.cos(lat) * math.sin(lon),
            N * (1 - e2) * math.sin(lat))


def geocentric_to_geodetic(X, Y, Z, a, f):
    e2 = f * (2 - f)
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    lat = math.atan2(Z, p * (1 - e2))
    for _ in range(10):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(Z + e2 * N * math.sin(lat), p)
    return lat, lon


def sad69_to_wgs84_utm(x, y):
    lat, lon = tm_inverse(x, y, *GRS67)
    X, Y, Z = geodetic_to_geocentric(lat, lon, *GRS67)
    X, Y, Z = X + HELMERT[0], Y + HELMERT[1], Z + HELMERT[2]
    lat, lon = geocentric_to_geodetic(X, Y, Z, *WGS84)
    return tm_forward(lat, lon, *WGS84)


# ---------------------------------------------------------------- geometry
def poly_area(pts):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def clip_rect(pts, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman clip of polygon pts (list of (x,y)) by an axis box."""
    def clip_edge(poly, inside, intersect):
        out = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            ci, pi = inside(cur), inside(prev)
            if ci:
                if not pi:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif pi:
                out.append(intersect(prev, cur))
        return out
    def ix_x(x0):
        def f(p, q):
            t = (x0 - p[0]) / (q[0] - p[0])
            return (x0, p[1] + t * (q[1] - p[1]))
        return f
    def ix_y(y0):
        def f(p, q):
            t = (y0 - p[1]) / (q[1] - p[1])
            return (p[0] + t * (q[0] - p[0]), y0)
        return f
    poly = list(pts)
    for inside, ix in (
        (lambda p: p[0] >= xmin, ix_x(xmin)),
        (lambda p: p[0] <= xmax, ix_x(xmax)),
        (lambda p: p[1] >= ymin, ix_y(ymin)),
        (lambda p: p[1] <= ymax, ix_y(ymax)),
    ):
        if not poly:
            return []
        poly = clip_edge(poly, inside, ix)
    return poly


def rings_clip_area(rings, xmin, xmax, ymin, ymax):
    """Area of polygon-with-holes clipped to a box.

    Shapefile convention: outer rings clockwise (negative shoelace), holes
    counter-clockwise (positive shoelace). Clipped outer areas add, holes subtract.
    """
    tot = 0.0
    for ring in rings:
        r = ring[:-1] if ring[0] == ring[-1] else ring
        a_full = poly_area(r)
        c = clip_rect(r, xmin, xmax, ymin, ymax)
        if len(c) >= 3:
            a_clip = abs(poly_area(c))
            tot += a_clip if a_full < 0 else -a_clip
    return max(tot, 0.0)


def point_in_rings(x, y, rings):
    inside = False
    for ring in rings:
        r = ring[:-1] if ring[0] == ring[-1] else ring
        j = len(r) - 1
        for i in range(len(r)):
            xi, yi = r[i]; xj, yj = r[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
    return inside


# ---------------------------------------------------------------- raster + windows
@lru_cache(maxsize=1)
def load_cropped():
    """Return (tmi, u) cropped/masked float64 arrays and crop transform origin."""
    demo = read_shp_polygons(DEMO_SHP)[0]  # WGS84 UTM22S already (= raster CRS)
    xs = [p[0] for p in demo[0]]; ys = [p[1] for p in demo[0]]
    c0 = math.floor((min(xs) - X0) / PX); c1 = math.ceil((max(xs) - X0) / PX)
    r0 = math.floor((Y0 - max(ys)) / PX); r1 = math.ceil((Y0 - min(ys)) / PX)
    def rd(p):
        a = np.fromfile(p, dtype="<f4").reshape(NL, NC)[r0:r1, c0:c1].astype(np.float64)
        a[a == NODATA] = np.nan
        return a
    tmi, u = rd(TMI_BIN), rd(U_BIN)
    H, W = tmi.shape
    ox, oy = X0 + c0 * PX, Y0 - r0 * PX
    ccx = ox + (np.arange(W) + 0.5) * PX
    ccy = oy - (np.arange(H) + 0.5) * PX
    mask = np.zeros((H, W), dtype=bool)
    for i, y in enumerate(ccy):
        row = mask[i]
        for j, x in enumerate(ccx):
            row[j] = point_in_rings(x, y, demo)
    tmi[~mask] = np.nan
    u[~mask] = np.nan
    return tmi, u, ox, oy


@lru_cache(maxsize=8)
def window_matrix(dep_1based: int):
    """Replicate build_multivariate_window_matrix for reference deposit."""
    tmi, u, ox, oy = load_cropped()
    deps = read_shp_polygons(DEP_SHP)          # SAD69 coords used AS raster CRS (assume_raster)
    rings = deps[dep_1based - 1]
    xs = [p[0] for r in rings for p in r]; ys = [p[1] for r in rings for p in r]
    cs = math.floor((min(xs) - ox) / PX); ce = math.ceil((max(xs) - ox) / PX)
    rs = math.floor((oy - max(ys)) / PX); re = math.ceil((oy - min(ys)) / PX)
    H, W = tmi.shape
    cs, ce = max(cs, 0), min(ce, W); rs, re = max(rs, 0), min(re, H)
    tpl_t, tpl_u = tmi[rs:re, cs:ce], u[rs:re, cs:ce]
    wh, ww = tpl_t.shape
    if np.isnan(tpl_t).any() or np.isnan(tpl_u).any():
        raise ValueError("template contains NaNs")

    def padded(a):
        pad_y = (STRIDE - (a.shape[0] - wh) % STRIDE) % STRIDE
        pad_x = (STRIDE - (a.shape[1] - ww) % STRIDE) % STRIDE
        return np.pad(a, ((0, pad_y), (0, pad_x)), constant_values=np.nan)

    pt, pu = padded(tmi), padded(u)
    shared = np.isnan(pt) | np.isnan(pu)
    pt, pu = pt.copy(), pu.copy()
    pt[shared] = np.nan; pu[shared] = np.nan
    nrows = (pt.shape[0] - wh) // STRIDE + 1
    ncols = (pt.shape[1] - ww) // STRIDE + 1
    sw = np.lib.stride_tricks.sliding_window_view
    vt = sw(pt, (wh, ww))[::STRIDE, ::STRIDE]
    vu = sw(pu, (wh, ww))[::STRIDE, ::STRIDE]
    keep = ~np.isnan(vt).any(axis=(2, 3))       # same for vu (shared mask)
    rows, cols = np.nonzero(keep)
    ids = rows * ncols + cols
    Xt = vt[rows, cols].reshape(len(rows), -1)
    Xu = vu[rows, cols].reshape(len(rows), -1)
    dep_row = np.concatenate([tpl_t.ravel(), tpl_u.ravel()])
    X = np.vstack([np.hstack([Xt, Xu]), dep_row])
    idx = np.stack([rows * STRIDE, cols * STRIDE, ids], axis=1)
    return X, idx, (wh, ww), (ox, oy)


# ---------------------------------------------------------------- ranking
def block_scores(Xb, k):
    mu = Xb.mean(axis=0)
    sd = Xb.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xb - mu) / sd
    Zc = Z - Z.mean(axis=0)
    U_, S, _ = np.linalg.svd(Zc, full_matrices=False)
    return (U_ * S)[:, :k]


def blk_w(z):
    w = z ** 2
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / max(len(w), 1)


def rank_concat(dep_1based, k1, k2, alpha):
    X, idx, wshape, origin = window_matrix(dep_1based)
    npix = wshape[0] * wshape[1]
    dep = X.shape[0] - 1
    Z1 = block_scores(X[:, :npix], k1)
    Z2 = block_scores(X[:, npix:2 * npix], k2)
    F = np.hstack([Z1, Z2])
    w = np.concatenate([blk_w(Z1[dep]) * alpha, blk_w(Z2[dep]) * (1 - alpha)])
    d = np.sqrt(((F - F[dep]) ** 2) @ w)
    order = np.argsort(d)
    return order, idx, wshape, origin, dep


def rank_raw(dep_1based):
    X, idx, wshape, origin = window_matrix(dep_1based)
    dep = X.shape[0] - 1
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    d = np.sqrt(((Xs - Xs[dep]) ** 2).sum(axis=1))
    return np.argsort(d), idx, wshape, origin, dep


# ---------------------------------------------------------------- validation
@lru_cache(maxsize=1)
def deposits_wgs84():
    out = []
    for rings in read_shp_polygons(DEP_SHP):
        out.append([[sad69_to_wgs84_utm(x, y) for (x, y) in ring] for ring in rings])
    return out


def footprint_recovery(order, idx, wshape, origin, dep_index, ref_dep_1based, n_top=N_TOP):
    ox, oy = origin
    wh, ww = wshape
    valid = order[(order != dep_index)][:n_top]
    pos = {int(w[2]): (w[0], w[1]) for w in idx}
    rects = []
    for widx in valid:
        r, c, _ = idx[widx]
        x1, y2 = ox + c * PX, oy - r * PX
        rects.append((x1, x1 + ww * PX, y2 - wh * PX, y2))  # xmin,xmax,ymin,ymax

    deps = deposits_wgs84()
    test = {i: d for i, d in enumerate(deps) if i != ref_dep_1based - 1}
    dep_area = {}
    for i, rings in test.items():
        a = 0.0
        for ring in rings:
            r = ring[:-1] if ring[0] == ring[-1] else ring
            s = poly_area(r)
            a += abs(s) if s < 0 else -abs(s)   # outer CW positive, holes CCW negative
        dep_area[i] = abs(a) if a != 0 else sum(abs(poly_area(r[:-1] if r[0]==r[-1] else r)) for r in rings)
    # deposit quads here are simple single-ring polys; use abs area
    dep_area = {i: abs(poly_area(rings[0][:-1] if rings[0][0]==rings[0][-1] else rings[0])) for i, rings in test.items()}

    # cell decomposition per test deposit
    cover_cells = {}
    for i, rings in test.items():
        ring = rings[0][:-1] if rings[0][0] == rings[0][-1] else rings[0]
        bx = [p[0] for p in ring]; by = [p[1] for p in ring]
        cand = [j for j, (a, b, c, d) in enumerate(rects)
                if not (b <= min(bx) or a >= max(bx) or d <= min(by) or c >= max(by))]
        xs = sorted({v for j in cand for v in rects[j][:2]} | {min(bx), max(bx)})
        ys = sorted({v for j in cand for v in rects[j][2:]} | {min(by), max(by)})
        cells = []
        for xa, xb in zip(xs[:-1], xs[1:]):
            for ya, yb in zip(ys[:-1], ys[1:]):
                ar = rings_clip_area(rings, xa, xb, ya, yb)
                if ar > 1e-9:
                    cells.append([xa, xb, ya, yb, ar, False])
        cover_cells[i] = (cand, cells)

    covered_area = {}
    coverage = {}
    hit_by_rank, overlap_by_rank, first_hit = {}, {}, {}
    hits = set()
    cum_total, cum_mean = [], []
    total_area = sum(dep_area.values())
    rect_rank = {j: rk for rk, j in enumerate(range(len(rects)), start=1)}

    for rank, (xa, xb, ya, yb) in enumerate(rects, start=1):
        overlapped, newly = [], []
        for i, rings in test.items():
            cand, cells = cover_cells[i]
            inter = rings_clip_area(rings, xa, xb, ya, yb)
            if inter <= 1e-9:
                continue
            overlapped.append(i)
            add = 0.0
            for cell in cells:
                cxa, cxb, cya, cyb, ar, done = cell
                if done:
                    continue
                if cxa >= xa - 1e-9 and cxb <= xb + 1e-9 and cya >= ya - 1e-9 and cyb <= yb + 1e-9:
                    cell[5] = True
                    add += ar
            if add > 0 or i not in covered_area:
                covered_area[i] = covered_area.get(i, 0.0) + add
                coverage[i] = covered_area[i] / dep_area[i]
            if coverage.get(i, 0.0) >= MIN_COVER and i not in hits:
                hits.add(i); first_hit[i] = rank; newly.append(i)
        if overlapped:
            overlap_by_rank[rank] = sorted(overlapped)
        if newly:
            hit_by_rank[rank] = sorted(newly)
        tot = sum(covered_area.values())
        cum_total.append(tot / total_area)
        cum_mean.append(float(np.mean([coverage.get(i, 0.0) for i in dep_area])))

    return {
        "cum_recovered_frac_total": np.asarray(cum_total),
        "cum_mean_recovered_frac": np.asarray(cum_mean),
        "coverage_by_deposit": coverage,
        "dep_area_by_deposit": dep_area,
        "overlap_by_rank": overlap_by_rank,
        "hit_by_rank": hit_by_rank,
        "first_hit_rank_by_deposit": first_hit,
    }


def run(dep_1based, k1, k2, alpha, mode="fusion"):
    if mode == "raw":
        order, idx, wshape, origin, dep = rank_raw(dep_1based)
    else:
        order, idx, wshape, origin, dep = rank_concat(dep_1based, k1, k2, alpha)
    return footprint_recovery(order, idx, wshape, origin, dep, dep_1based)


def metrics(d):
    mf = d["cum_mean_recovered_frac"]
    return dict(
        auc=float(mf.sum() * (N_TOP / len(mf))),
        end=float(mf[-1] * 100),
        end_total=float(d["cum_recovered_frac_total"][-1] * 100),
        hits=len(d["hit_by_rank"]),
        hit_ranks={r: [int(t) for t in ids] for r, ids in sorted(d["hit_by_rank"].items())},
    )


if __name__ == "__main__":
    import json, sys
    dep = int(sys.argv[1]); k1 = int(sys.argv[2]); k2 = int(sys.argv[3])
    alpha = float(sys.argv[4]); mode = sys.argv[5] if len(sys.argv) > 5 else "fusion"
    print(json.dumps(metrics(run(dep, k1, k2, alpha, mode)), indent=1))
