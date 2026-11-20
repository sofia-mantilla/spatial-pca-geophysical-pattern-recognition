"""Weight ablation with the area budget, and the distinct-cluster counts.

Produces the Appendix C table "Ablation of the component weighting at rank 250"
and the two cluster counts quoted in Sections 4.1 and 4.2.

Four weightings, all other settings as published:
  z2       deposit-based, w_m = z_dm^2 / sum_u z_du^2          (the published rule)
  unif     uniform over the retained components
  white    inverse-eigenvalue, w_m ~ 1/lambda_m
  zstd     the deposit rule on scores standardized by sqrt(lambda_m)

For each, c(250), AUC, hits, a(250) and the ratio to a random selection
covering the same area -- the variants examine different amounts of ground,
so the ratio is the comparable column.

Run from inside paper/:  python run_weight_ablation_area.py   (~3 min)
"""
import numpy as np

import numpy_repro_concat as R
import case1_uni_repro as C1
import null_expectation as NE

T = 6000
tmi, _u, _ox, _oy = R.load_cropped()
GRID = tmi.shape
VALID = int(np.isfinite(tmi).sum())


def scores_full(Xb, k):
    """Scores and eigenvalues, column-standardized with ddof=0 as published."""
    mu = Xb.mean(axis=0)
    sd = Xb.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xb - mu) / sd
    Zc = Z - Z.mean(axis=0)
    U_, S, _ = np.linalg.svd(Zc, full_matrices=False)
    return (U_ * S)[:, :k], (S ** 2)[:k]


def table(case, variants, idx, wshape, origin, dep, ref, deps_wgs, case1):
    rects = NE.rects_from_idx(idx, wshape, origin)
    cells, areas = NE.build_cells(deps_wgs, ref - 1, rects)
    rc = NE.exact_random_curve(cells, areas, dep, t_max=T)
    ru = NE.exact_random_union(idx, wshape, dep, GRID, t_max=T)
    print("\n%s   (random at rank 250: %.1f%%)" % (case, 100 * rc[249]))
    print("  %-20s %8s %7s %5s %8s %13s %6s"
          % ("weights", "c(250)", "AUC", "hits", "a(250)", "E[rand|area]", "ratio"))
    for name, order in variants:
        old = R.deposits_wgs84
        if case1:
            R.deposits_wgs84 = C1.deposits_case1_wgs84
        try:
            res = R.footprint_recovery(order, idx, wshape, origin, dep, ref)
        finally:
            R.deposits_wgs84 = old
        m = R.metrics(res)
        ua = NE.union_area_curve(order, idx, wshape, dep, GRID)
        t_star = float(np.interp(ua[-1], ru, np.arange(1, T + 1)))
        em = float(np.interp(t_star, np.arange(1, T + 1), rc))
        print("  %-20s %7.1f%% %7.1f %5d %7.1f%% %12.1f%% %6.2f"
              % (name, m["end"], m["auc"], m["hits"],
                 100 * ua[-1] / VALID, 100 * em, m["end"] / 100 / em))


def clusters(order, idx, wshape, dep, n_top=250):
    """Spatially distinct groups among the top windows.

    Two windows join when their rectangles overlap, which on a stride grid is
    |dr| < window height and |dc| < window width. Returns the group count and
    the group sizes, largest first.
    """
    wh, ww = wshape
    sel = order[order != dep][:n_top]
    rc = np.array([[idx[i][0], idx[i][1]] for i in sel])
    parent = list(range(len(rc)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(rc)):
        near = np.nonzero((np.abs(rc[:, 0] - rc[i, 0]) < wh)
                          & (np.abs(rc[:, 1] - rc[i, 1]) < ww))[0]
        for j in near[near > i]:
            ra, rb = find(i), find(int(j))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for i in range(len(rc)):
        groups.setdefault(find(i), []).append(i)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    return len(sizes), sizes


# ------------------------------------------------------------------ Case 1
X1, idx1, ws1, o1 = C1.window_matrix_uni()
dep1 = X1.shape[0] - 1
Z1, lam1 = scores_full(X1, 17)
Zs1 = Z1 / np.sqrt(lam1)


def order1(Z, w):
    return np.argsort(np.sqrt(((Z - Z[dep1]) ** 2) @ w))


v1 = [("Deposit-based z_dm^2", order1(Z1, R.blk_w(Z1[dep1]))),
      ("Standardized scores", order1(Zs1, R.blk_w(Zs1[dep1]))),
      ("Uniform", order1(Z1, np.ones(17) / 17)),
      ("Whitened 1/lambda", order1(Z1, (1 / lam1) / (1 / lam1).sum()))]
table("Case 1: Paulo Afonso, univariate TMI, k=17", v1,
      idx1, ws1, o1, dep1, 6, C1.deposits_case1_wgs84(), True)

# ------------------------------------------------------------------ Case 2
X, idx, wshape, origin = R.window_matrix(3)
dep = X.shape[0] - 1
npix = wshape[0] * wshape[1]
Zt, lamt = scores_full(X[:, :npix], 2)
Zu, lamu = scores_full(X[:, npix:2 * npix], 8)
St, Su = Zt / np.sqrt(lamt), Zu / np.sqrt(lamu)


def order2(A, B, wt, wu, alpha=0.503):
    F = np.hstack([A, B])
    w = np.concatenate([wt * alpha, wu * (1 - alpha)])
    return np.argsort(np.sqrt(((F - F[dep]) ** 2) @ w))


v2 = [("Deposit-based z_dm^2", order2(Zt, Zu, R.blk_w(Zt[dep]), R.blk_w(Zu[dep]))),
      ("Uniform", order2(Zt, Zu, np.ones(2) / 2, np.ones(8) / 8)),
      ("Standardized scores", order2(St, Su, R.blk_w(St[dep]), R.blk_w(Su[dep]))),
      ("Whitened 1/lambda", order2(Zt, Zu, (1 / lamt) / (1 / lamt).sum(),
                                   (1 / lamu) / (1 / lamu).sum()))]
table("Case 2: Alemao, TMI + radiometric U, k=(2,8), alpha=0.503", v2,
      idx, wshape, origin, dep, 3, R.deposits_wgs84(), False)

# ----------------------------------------- how variance-led the raw rule is
wr, ws = R.blk_w(Z1[dep1]), R.blk_w(Zs1[dep1])
ev = lam1 / lam1.sum()
print("\nCase 1 weights, the six the published rule ranks highest:")
for m_ in np.argsort(-wr)[:6]:
    print("  PC%-3d variance %5.1f%%   raw %5.1f%%   standardized %5.1f%%"
          % (m_ + 1, 100 * ev[m_], 100 * wr[m_], 100 * ws[m_]))
print("  weight order vs eigenvalue order, Spearman: %+.2f"
      % np.corrcoef(np.argsort(np.argsort(-wr)), np.arange(len(wr)))[0, 1])

# ------------------------------------------------- distinct groups (B6)
print("\nDistinct groups among the top 250:")
nc1, sz1 = clusters(v1[0][1], idx1, ws1, dep1)
nc2, sz2 = clusters(v2[0][1], idx, wshape, dep)
print("  Case 1: %d groups, largest %d windows, sizes %s" % (nc1, sz1[0], sz1[:8]))
print("  Case 2: %d groups, largest %d windows, sizes %s" % (nc2, sz2[0], sz2[:8]))
