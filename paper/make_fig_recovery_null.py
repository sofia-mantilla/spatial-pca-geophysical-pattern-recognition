"""Recovery-comparison figures for both cases, WITH the random-ranking null.

Regenerates (paper figure names):
  case1_recovery_comparison.png / .pdf  - Case 1 (Deposit 6): wPCA k=17 vs raw
  multi_recovery_comparison.png / .pdf  - Case 2 (Deposit 3): wPCA TMI/U/comb vs raw multi
each with the exact random-ranking expectation E[cbar(t)] (dashed) and a
Monte-Carlo 5-95% envelope (shaded).

Also prints the union-area fractions and budget facts quoted in the paper text.

Pure numpy + matplotlib. Run from the worktree root:
    python make_fig_recovery_null.py [--outdir DIR] [--dpi 300]
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
ap.add_argument("--nperm", type=int, default=2000)
args = ap.parse_args()
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

GRAY = "#5c5c5c"

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})


def draw(ax, runs, rand_curve, rand_lo, rand_hi):
    x = np.arange(1, R.N_TOP + 1)
    ax.fill_between(x, rand_lo, rand_hi, color=GRAY, alpha=0.14, lw=0, zorder=1)
    ax.plot(x, rand_curve, color=GRAY, lw=2.0, ls="--", zorder=2,
            label="Random selection (expected; shaded 5–95%)")
    for lab, d, col in runs:
        y = np.asarray(d["cum_mean_recovered_frac"], float)
        ax.step(x, y, where="post", color=col, lw=2.4, label=lab, zorder=3)
        for r in d["overlap_by_rank"]:
            if r in d["hit_by_rank"]:
                continue
            ax.scatter([r], [y[min(r - 1, len(y) - 1)]], s=42, facecolors="#ffd21e",
                       edgecolors="k", lw=.6, zorder=5)
        for r, ids in d["hit_by_rank"].items():
            yy = y[min(r - 1, len(y) - 1)]
            ax.scatter([r], [yy], s=70, facecolors="#e03131", edgecolors="k", lw=.7, zorder=6)
            ax.annotate("+".join(str(t + 1) for t in ids), (r, yy),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=14, fontweight="bold", color="#7a1010")
    ax.scatter([], [], s=42, facecolors="#ffd21e", edgecolors="k", label="Overlap event (yellow)")
    ax.scatter([], [], s=70, facecolors="#e03131", edgecolors="k", label="Hit event (red)")
    ax.set_xlim(0, R.N_TOP)
    ax.set_xlabel("Prediction rank")
    ax.set_ylabel("Cumulative recovered fraction")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"{stem}.png/.pdf written to {OUT}")


tmi, _u, _ox, _oy = R.load_cropped()
GRID = tmi.shape
VALID_PX = int(np.isfinite(tmi).sum())

# ================================ CASE 1 ================================
X1, idx1, wshape1, origin1 = C1.window_matrix_uni()
n1 = X1.shape[0] - 1
deps1 = C1.deposits_case1_wgs84()
rects1 = NE.rects_from_idx(idx1, wshape1, origin1)
cells1, areas1 = NE.build_cells(deps1, C1.REF_DEP - 1, rects1)
rand1 = NE.exact_random_curve(cells1, areas1, n1)
lo1, hi1, _ = NE.mc_band(cells1, areas1, n1, nperm=args.nperm)

o_spca = C1.rank_spca(17)
o_raw = C1.rank_raw_uni("ddof1")
runs1 = [
    ("wPCA Uni (TMI) (k=17)", C1.recovery(*o_spca), "#1f4fd8"),
    ("Raw Uni (TMI)", C1.recovery(*o_raw), "#e0a800"),
]
fig, ax = plt.subplots(figsize=(10.5, 7.4))
draw(ax, runs1, rand1, lo1, hi1)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=14, loc="upper left")
fig.tight_layout()
save(fig, "case1_recovery_comparison")

ua_s = NE.union_area_curve(o_spca[0], idx1, wshape1, n1, GRID)
ua_r = NE.union_area_curve(o_raw[0], idx1, wshape1, n1, GRID)
ua_rand1 = NE.exact_random_union(idx1, wshape1, n1, GRID)
ov1 = NE.budget_ranks_overlapping(o_spca[0], idx1, wshape1, origin1, n1,
                                  deps1[C1.REF_DEP - 1])
print("CASE1: E[cbar(250)]=%.4f AUC=%.1f | union250/valid: wPCA %.3f raw %.3f rand %.3f"
      % (rand1[-1], rand1.sum(), ua_s[-1] / VALID_PX, ua_r[-1] / VALID_PX,
         ua_rand1[-1] / VALID_PX))
print("CASE1 budget ranks overlapping reference:", ov1)

# ================================ CASE 2 ================================
DEP, K1, K2, ALPHA = 3, 2, 8, 0.503
X2, idx2, wshape2, origin2 = R.window_matrix(DEP)
n2 = X2.shape[0] - 1
deps2 = R.deposits_wgs84()
rects2 = NE.rects_from_idx(idx2, wshape2, origin2)
cells2, areas2 = NE.build_cells(deps2, DEP - 1, rects2)
rand2 = NE.exact_random_curve(cells2, areas2, n2)
lo2, hi2, _ = NE.mc_band(cells2, areas2, n2, nperm=args.nperm)

runs2 = [
    (f"wPCA Uni (TMI, k={K1})", R.run(DEP, K1, K2, 1.0), "#1f4fd8"),
    (f"wPCA Uni (Radiometric_U, k={K2})", R.run(DEP, K1, K2, 0.0), "#8e24aa"),
    ("Raw Multi (TMI+U)", R.run(DEP, K1, K2, 0.0, mode="raw"), "#e0a800"),
    (rf"wPCA Multi (TMI+U, $\alpha_{{\mathrm{{TMI}}}}$={ALPHA:.3f})",
     R.run(DEP, K1, K2, ALPHA), "#2e9e2e"),
]
fig, ax = plt.subplots(figsize=(10.5, 7.4))
draw(ax, runs2, rand2, lo2, hi2)
ax.set_ylim(0, 0.7)
ax.legend(fontsize=14, loc="upper left")
fig.tight_layout()
save(fig, "multi_recovery_comparison")

orders2 = dict(
    comb=R.rank_concat(DEP, K1, K2, ALPHA)[0],
    tmi=R.rank_concat(DEP, K1, K2, 1.0)[0],
    u=R.rank_concat(DEP, K1, K2, 0.0)[0],
    raw=R.rank_raw(DEP)[0],
)
ua_rand2 = NE.exact_random_union(idx2, wshape2, n2, GRID)
fr = {nm: NE.union_area_curve(o, idx2, wshape2, n2, GRID)[-1] / VALID_PX
      for nm, o in orders2.items()}
ov2 = NE.budget_ranks_overlapping(orders2["comb"], idx2, wshape2, origin2, n2,
                                  deps2[DEP - 1])
print("CASE2: E[cbar(250)]=%.4f AUC=%.1f | union250/valid: comb %.3f tmi %.3f u %.3f raw %.3f rand %.3f"
      % (rand2[-1], rand2.sum(), fr["comb"], fr["tmi"], fr["u"], fr["raw"],
         ua_rand2[-1] / VALID_PX))
print("CASE2 budget ranks overlapping reference:", ov2)
