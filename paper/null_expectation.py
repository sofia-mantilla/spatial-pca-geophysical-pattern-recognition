"""Null-expectation machinery for the top-t window budget (paper B3 fix).

Given a set of candidate windows (rects) and test-deposit outlines, computes:

  * exact_random_curve  - E[cbar(t)], the mean recovered fraction of the test
    deposits when t windows are drawn uniformly WITHOUT replacement from the n
    valid windows. Exact via the hypergeometric identity: a cell contained by
    m windows is uncovered after t draws with probability
    C(n-m, t)/C(n, t) = prod_{j<t} (n-m-j)/(n-j).
    Cells are induced by the edges of every candidate window intersecting the
    deposit, exactly as in numpy_repro_concat.footprint_recovery, so the
    expectation uses the same geometry as the published curves.

  * mc_band             - Monte-Carlo 5-95% envelope of cbar(t) (figure band).

  * union_area_curve    - exact union pixel area of the top-t budget windows of
    an actual ranking (pixel grid; windows are cell-aligned).

  * exact_random_union  - E[union pixel area of t random windows], exact via
    per-pixel cover counts and the same hypergeometric identity.

Depends only on numpy + numpy_repro_concat (pure-numpy replication).
"""
from __future__ import annotations

import numpy as np

import numpy_repro_concat as R


def rects_from_idx(idx, wshape, origin):
    ox, oy = origin
    wh, ww = wshape
    out = []
    for r, c, _ in idx:
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        out.append((x1, x1 + ww * R.PX, y2 - wh * R.PX, y2))
    return out


def deposit_cells(rings, rects):
    """Cells induced by all candidate-window edges over one deposit.

    Returns (cells, dep_area) where cells is a list of (area, owner window ids).
    """
    ring = rings[0][:-1] if rings[0][0] == rings[0][-1] else rings[0]
    bx = [p[0] for p in ring]; by = [p[1] for p in ring]
    dep_area = abs(R.poly_area(ring))
    cand = [j for j, (a, b, c, d) in enumerate(rects)
            if not (b <= min(bx) or a >= max(bx) or d <= min(by) or c >= max(by))]
    xs = sorted({v for j in cand for v in (rects[j][0], rects[j][1])} | {min(bx), max(bx)})
    ys = sorted({v for j in cand for v in (rects[j][2], rects[j][3])} | {min(by), max(by)})
    cells = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        for ya, yb in zip(ys[:-1], ys[1:]):
            ar = R.rings_clip_area(rings, xa, xb, ya, yb)
            if ar > 1e-9:
                owners = [j for j in cand
                          if rects[j][0] <= xa + 1e-9 and rects[j][1] >= xb - 1e-9
                          and rects[j][2] <= ya + 1e-9 and rects[j][3] >= yb - 1e-9]
                cells.append((ar, owners))
    return cells, dep_area


def build_cells(deposits_rings, ref_index0, rects):
    """deposit_cells for every test deposit (reference excluded)."""
    cells_by_dep, dep_areas = {}, {}
    for i, rings in enumerate(deposits_rings):
        if i == ref_index0:
            continue
        cells, dep_area = deposit_cells(rings, rects)
        cells_by_dep[i] = cells
        dep_areas[i] = dep_area
    return cells_by_dep, dep_areas


def exact_random_curve(cells_by_dep, dep_areas, n, t_max=R.N_TOP):
    ms = sorted({len(o) for cells in cells_by_dep.values() for (_, o) in cells if o})
    q = {m: np.ones(t_max + 1) for m in ms}
    for m in ms:
        for t in range(1, t_max + 1):
            q[m][t] = q[m][t - 1] * (n - m - (t - 1)) / (n - (t - 1))
    curve = np.zeros(t_max)
    J = len(cells_by_dep)
    for i, cells in cells_by_dep.items():
        cov = np.zeros(t_max)
        for ar, owners in cells:
            m = len(owners)
            if m:
                cov += ar * (1.0 - q[m][1:t_max + 1])
        curve += cov / dep_areas[i] / J
    return curve


def mc_band(cells_by_dep, dep_areas, n, t_max=R.N_TOP, nperm=2000, seed=20260812):
    rng = np.random.default_rng(seed)
    win2cells = {}
    for i, cells in cells_by_dep.items():
        for ci, (ar, owners) in enumerate(cells):
            for j in owners:
                win2cells.setdefault(j, []).append((i, ci))
    inter_wins = np.array(sorted(win2cells))
    J = len(cells_by_dep)
    curves = np.zeros((nperm, t_max))
    for p in range(nperm):
        ranks = rng.choice(n, size=len(inter_wins), replace=False)
        order = np.argsort(ranks)
        covered = {i: np.zeros(len(cells_by_dep[i]), dtype=bool) for i in cells_by_dep}
        cov_area = {i: 0.0 for i in cells_by_dep}
        curve = np.zeros(t_max)
        prev = 0
        cbar = 0.0
        for k in order:
            t_at = int(ranks[k]) + 1
            if t_at > t_max:
                break
            curve[prev:t_at - 1] = cbar
            prev = t_at - 1
            for (i, ci) in win2cells[int(inter_wins[k])]:
                if not covered[i][ci]:
                    covered[i][ci] = True
                    cov_area[i] += cells_by_dep[i][ci][0]
            cbar = sum(cov_area[i] / dep_areas[i] for i in cells_by_dep) / J
            curve[t_at - 1] = cbar
        curve[prev:] = cbar
        curves[p] = curve
    return (np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0),
            curves.mean(axis=0))


def union_area_curve(order, idx, wshape, dep_index, grid_shape, n_top=R.N_TOP):
    wh, ww = wshape
    Hp = grid_shape[0] + (R.STRIDE - (grid_shape[0] - wh) % R.STRIDE) % R.STRIDE
    Wp = grid_shape[1] + (R.STRIDE - (grid_shape[1] - ww) % R.STRIDE) % R.STRIDE
    grid = np.zeros((Hp, Wp), dtype=bool)
    budget = [w for w in order if w != dep_index][:n_top]
    out = np.zeros(n_top)
    tot = 0
    for t, widx in enumerate(budget):
        r, c, _ = idx[widx]
        sub = grid[r:r + wh, c:c + ww]
        tot += int((~sub).sum())
        sub[:] = True
        out[t] = tot
    return out


def exact_random_union(idx, wshape, n, grid_shape, t_max=R.N_TOP):
    wh, ww = wshape
    Hp = grid_shape[0] + (R.STRIDE - (grid_shape[0] - wh) % R.STRIDE) % R.STRIDE
    Wp = grid_shape[1] + (R.STRIDE - (grid_shape[1] - ww) % R.STRIDE) % R.STRIDE
    cnt = np.zeros((Hp, Wp), dtype=np.int32)
    for r, c, _ in idx:
        cnt[r:r + wh, c:c + ww] += 1
    vals, freq = np.unique(cnt[cnt > 0], return_counts=True)
    q = np.ones((len(vals), t_max + 1))
    for vi, m in enumerate(vals):
        for t in range(1, t_max + 1):
            q[vi, t] = q[vi, t - 1] * (n - m - (t - 1)) / (n - (t - 1))
    return ((1.0 - q[:, 1:]) * freq[:, None]).sum(axis=0)


def budget_ranks_overlapping(order, idx, wshape, origin, dep_index, ref_rings,
                             n_top=R.N_TOP):
    """Budget ranks (1-based) whose window intersects the reference outline."""
    rects = rects_from_idx(idx, wshape, origin)
    budget = [w for w in order if w != dep_index][:n_top]
    return [t for t, widx in enumerate(budget, start=1)
            if R.rings_clip_area(ref_rings, *rects[widx]) > 1e-9]
