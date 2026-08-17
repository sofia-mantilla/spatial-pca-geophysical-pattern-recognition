"""Case-2 combined-ranking top-windows 3D figure (paper: analog of Case 1's
top_similar_windows_3d). Concat method TMI k=2 / U k=8, alpha_uni.

Excludes the reference deposit and its stride-overlapping copies; shows the
first four independent windows next to the deposit geometry, one row per
variable, common z-range per variable. Run from the worktree root:
    python case2_experiments/make_multi_top_windows_3d.py
"""
import os, math
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy_repro_concat as R

PAPER_CMAP = LinearSegmentedColormap.from_list("spatial_pca_paper", (
    (0.00, "#0101ff"), (0.167, "#2dc4ff"), (0.334, "#26ff00"), (0.501, "#ffe101"),
    (0.668, "#ff0109"), (0.835, "#ef00ff"), (1.00, "#de9eff")))

DEP, K1, K2, ALPHA = 3, 2, 8, 0.5033
order, idx, wshape, origin, dep_index = R.rank_concat(DEP, K1, K2, ALPHA)
X, _, _, _ = R.window_matrix(DEP)
tmi, u, ox, oy = R.load_cropped()
wh, ww = wshape; npix = wh * ww
depT = X[-1][:npix].reshape(wshape); depU = X[-1][npix:2 * npix].reshape(wshape)
deps = R.read_shp_polygons(R.DEP_SHP); rings = deps[DEP - 1]
xs = [p[0] for r in rings for p in r]; ys = [p[1] for r in rings for p in r]
dc = max(math.floor((min(xs) - ox) / R.PX), 0)
dr = max(math.floor((oy - max(ys)) / R.PX), 0)

def win(arr, j):
    r0, c0 = idx[j][:2]; return arr[r0:r0 + wh, c0:c0 + ww]

def overlaps_dep(j):
    r0, c0 = idx[j][:2]; return abs(r0 - dr) < wh and abs(c0 - dc) < ww

ranked = [j for j in order if j != dep_index]
top = [j for j in ranked if not overlaps_dep(j)][:4]
panels = [("Reference deposit 3", None)] + [(f"Rank {ranked.index(j)+1}", j) for j in top]
fig = plt.figure(figsize=(19, 8.2))
for row, (arr, dep_w, name) in enumerate([(tmi, depT, "TMI"), (u, depU, "Radiometric U")]):
    ws = [dep_w] + [win(arr, j) for _, j in panels[1:]]
    vmin = min(np.nanmin(w) for w in ws); vmax = max(np.nanmax(w) for w in ws)
    for col, ((lab, _), W) in enumerate(zip(panels, ws)):
        ax = fig.add_subplot(2, len(panels), row * len(panels) + col + 1, projection="3d")
        Y, Xg = np.mgrid[0:W.shape[0], 0:W.shape[1]]
        ax.plot_surface(Xg, Y, W, cmap=PAPER_CMAP, vmin=vmin, vmax=vmax,
                        rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.set_zlim(vmin, vmax); ax.view_init(elev=38, azim=-60)
        ax.set_title(f"{lab}\n({name})", fontsize=11, pad=0)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(labelsize=7)
fig.suptitle("Case 2: reference-deposit geometry and top independent windows of the combined ranking (common z-range per variable)", fontsize=13.5)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.06, hspace=0.12)
out = Path(__file__).resolve().parent / "figures" / "paper_corrected" / "multi_top_similar_windows_3d.png"
fig.savefig(out, dpi=250, facecolor="white")
print("saved", out)
