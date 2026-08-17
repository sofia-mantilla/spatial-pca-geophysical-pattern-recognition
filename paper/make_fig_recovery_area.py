"""Recovery versus area examined (prediction-area reading), both cases.

x-axis: union area of the top-t windows as a fraction of the study area.
y-axis: mean recovered fraction cbar(t).
The random reference is (E[union(t)], E[cbar(t)]) -- close to the diagonal
"recover what you cover". Rank-250 endpoints are marked.

Pure numpy + matplotlib. Run from the worktree root:
    python make_fig_recovery_area.py [--outdir DIR] [--dpi 300]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib.pyplot as plt

import numpy_repro_concat as R
import case1_uni_repro as C1
import null_expectation as NE

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--outdir", default=str(HERE / "figures_out"))
ap.add_argument("--dpi", type=int, default=400)
args = ap.parse_args()
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

GRAY = "#5c5c5c"

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})
T_RAND = 3000

tmi, _u, _ox, _oy = R.load_cropped()
GRID = tmi.shape
VALID = float(np.isfinite(tmi).sum())

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))

# ---------------------------------------------------------------- panel (a)
X1, idx1, w1, o1 = C1.window_matrix_uni(); n1 = X1.shape[0] - 1
deps1 = C1.deposits_case1_wgs84()
rects1 = NE.rects_from_idx(idx1, w1, o1)
cells1, areas1 = NE.build_cells(deps1, C1.REF_DEP - 1, rects1)
rand_c1 = NE.exact_random_curve(cells1, areas1, n1, t_max=T_RAND)
rand_u1 = NE.exact_random_union(idx1, w1, n1, GRID, t_max=T_RAND) / VALID

o_spca = C1.rank_spca(17)
o_raw = C1.rank_raw_uni("ddof1")
curves1 = []
for lab, (order, *_rest), col in [("wPCA (k=17)", o_spca, "#1f4fd8"),
                                  ("Raw", o_raw, "#e0a800")]:
    res = C1.recovery(order, *_rest)
    ua = NE.union_area_curve(order, idx1, w1, n1, GRID) / VALID
    curves1.append((lab, ua, np.asarray(res["cum_mean_recovered_frac"]), col))

ax = axes[0]
xmax1 = 0.50
m = rand_u1 <= xmax1
ax.plot(rand_u1[m], rand_c1[m], color=GRAY, ls="--", lw=2.0,
        label="Random selection (expected)")
for lab, ua, cc, col in curves1:
    ax.plot(ua, cc, color=col, lw=2.4, label=lab)
    ax.scatter([ua[-1]], [cc[-1]], s=55, color=col, edgecolors="k", lw=.7, zorder=6)
# equal-recovery guide: from wPCA endpoint horizontally to the random curve
end_a, end_c = curves1[0][1][-1], curves1[0][2][-1]
a_eq = float(np.interp(end_c, rand_c1, rand_u1))
ax.annotate("", xy=(a_eq, end_c), xytext=(end_a, end_c),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4, ls=":"))
ax.annotate("same recovery,\n3.1$\\times$ the area", xy=((end_a + a_eq) / 2, end_c),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=14,
            color=GRAY)
ax.set_xlim(0, xmax1); ax.set_ylim(0, 0.6)
ax.set_xlabel("Fraction of study area examined")
ax.set_ylabel("Mean recovered fraction")
ax.set_title("(a) Case 1 — Reference Deposit 6")
ax.legend(fontsize=14, loc="lower right")

# ---------------------------------------------------------------- panel (b)
DEP, K1, K2, ALPHA = 3, 2, 8, 0.503
X2, idx2, w2, o2 = R.window_matrix(DEP); n2 = X2.shape[0] - 1
deps2 = R.deposits_wgs84()
rects2 = NE.rects_from_idx(idx2, w2, o2)
cells2, areas2 = NE.build_cells(deps2, DEP - 1, rects2)
rand_c2 = NE.exact_random_curve(cells2, areas2, n2, t_max=T_RAND)
rand_u2 = NE.exact_random_union(idx2, w2, n2, GRID, t_max=T_RAND) / VALID

specs2 = [
    (rf"wPCA Multi ($\alpha_{{\mathrm{{TMI}}}}$={ALPHA:.3f})", R.rank_concat(DEP, K1, K2, ALPHA)[0], "#2e9e2e"),
    (f"wPCA TMI (k={K1})", R.rank_concat(DEP, K1, K2, 1.0)[0], "#1f4fd8"),
    (f"wPCA U (k={K2})", R.rank_concat(DEP, K1, K2, 0.0)[0], "#8e24aa"),
    ("Raw Multi", R.rank_raw(DEP)[0], "#e0a800"),
]
ax = axes[1]
xmax2 = 0.70
m = rand_u2 <= xmax2
ax.plot(rand_u2[m], rand_c2[m], color=GRAY, ls="--", lw=2.0,
        label="Random selection (expected)")
end_ab = end_cb = None
for lab, order, col in specs2:
    res = R.footprint_recovery(order, idx2, w2, o2, n2, DEP)
    ua = NE.union_area_curve(order, idx2, w2, n2, GRID) / VALID
    cc = np.asarray(res["cum_mean_recovered_frac"])
    ax.plot(ua, cc, color=col, lw=2.4, label=lab)
    ax.scatter([ua[-1]], [cc[-1]], s=55, color=col, edgecolors="k", lw=.7, zorder=6)
    if end_ab is None:
        end_ab, end_cb = ua[-1], cc[-1]
a_eq2 = float(np.interp(end_cb, rand_c2, rand_u2))
ax.annotate("", xy=(a_eq2, end_cb), xytext=(end_ab, end_cb),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4, ls=":"))
# the random diagonal passes just under the right end of the guide arrow, so the
# label sits below the arrow where the panel is empty
ax.annotate("same recovery,\n7.2$\\times$ the area", xy=(end_ab + 0.34 * (a_eq2 - end_ab), end_cb),
            textcoords="offset points", xytext=(0, -46), ha="center", fontsize=14,
            color=GRAY)
ax.set_xlim(0, xmax2); ax.set_ylim(0, 0.7)
ax.set_xlabel("Fraction of study area examined")
ax.set_title("(b) Case 2 — Reference Deposit 3")
ax.legend(fontsize=14, loc="lower right")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"recovery_vs_area.{ext}", dpi=args.dpi, facecolor="white")
print("recovery_vs_area.png/.pdf written to", OUT)
print("case1: wPCA endpoint a=%.3f c=%.3f ; random needs a=%.3f" % (end_a, end_c, a_eq))
print("case2: comb endpoint a=%.3f c=%.3f ; random needs a=%.3f" % (end_ab, end_cb, a_eq2))
