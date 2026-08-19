"""Appendix B figure: sensitivity of both rankings to the number of retained components.

Panel (a)  Case 1, Paulo Afonso. Mean recovered fraction at rank 250 against k, from
           paper/appendix_b/k_sweep_case1.csv
           (produced by scripts/run_case1_k_sweep.py).
Panel (b)  Case 2, Alemao. Held-out mean recovered fraction against k_U at k_TMI = 2,
           from the 15x15 nested-LODO grid
           paper/appendix_b/lodo_grid_by_k.csv
           (produced by case2_experiments/lodo_k_grid.py).

Writes a vector PDF and a 600-dpi PNG. Pure pandas + matplotlib, ~2 s.

    python paper/make_fig_appendixB_k_sensitivity.py
    python paper/make_fig_appendixB_k_sensitivity.py --outdir /path/to/figures --dpi 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CASE1_CSV = REPO / "paper" / "appendix_b" / "k_sweep_case1.csv"
CASE2_CSV = REPO / "paper" / "appendix_b" / "lodo_grid_by_k.csv"
OUTDIR = REPO / "paper" / "appendix_b"
STEM = "appendixB_k_sensitivity"

K_RETAINED_CASE1 = 17
PLATEAU = (12, 60)
K_TMI = 2
K_U_RETAINED = 8
TMI_ALONE_HELDOUT = 0.3226  # held-out mean recovered fraction, TMI alone at k_TMI = 2

INK, MUTED, BLUE, RUST, GREEN = "#1a1a1a", "#8a8f98", "#2563eb", "#c2410c", "#16a34a"
ANNOTATE_K = {2, 3, 5, 9, 12, 17, 30, 60}


def build(case1: pd.DataFrame, case2: pd.DataFrame):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    k = case1["k_pcs"].to_numpy()
    c = case1["mean_recovered_frac_250"].to_numpy()
    hits = case1["n_hits"].to_numpy()
    ax.axvspan(*PLATEAU, color=GREEN, alpha=0.10, zorder=1)
    ax.plot(k, c, lw=2.4, color=BLUE, marker="o", ms=4.5, zorder=5)
    ax.axvline(K_RETAINED_CASE1, lw=1.2, color=MUTED, zorder=3)
    ax.annotate(f"retained\n$k={K_RETAINED_CASE1}$", xy=(K_RETAINED_CASE1, 0.10),
                fontsize=8, color=INK, ha="center",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    plateau = case1[case1.k_pcs.between(*PLATEAU)]
    ax.text(26, 0.505,
            f"stable range $k={PLATEAU[0]}$–${PLATEAU[1]}$\n"
            f"$\\bar{{c}}={plateau.mean_recovered_frac_250.iloc[0]:.3f}$, "
            f"{int(plateau.n_hits.iloc[0])} hits",
            fontsize=8, color=INK, ha="center")
    for kk, cc, hh in zip(k, c, hits):
        if kk in ANNOTATE_K:
            ax.annotate(str(int(hh)), xy=(kk, cc), xytext=(0, 7),
                        textcoords="offset points", fontsize=7, color=MUTED, ha="center")
    ticks = [2, 3, 5, 8, 12, 17, 25, 40, 60]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlabel("Retained components $k$")
    ax.set_ylabel("Mean recovered fraction $\\bar{c}(250)$")
    ax.set_title("(a) Case 1 — Paulo Afonso, TMI", fontsize=10, loc="left", color=INK)
    ax.set_ylim(0, 0.60)

    g = case2[case2.k1 == K_TMI].sort_values("k2")
    bx.plot(g.k2, g.a_uni, lw=2.4, color=BLUE, marker="o", ms=4.5,
            label="Combined, $\\alpha_v$ rule", zorder=5)
    bx.plot(g.k2, g.U, lw=2.0, color=RUST, marker="s", ms=4,
            label="Radiometric U alone", zorder=4)
    bx.axhline(TMI_ALONE_HELDOUT, lw=1.8, ls="--", color=MUTED,
               label=f"TMI alone ($k_{{TMI}}={K_TMI}$)", zorder=3)
    bx.axvline(K_U_RETAINED, lw=1.2, color=MUTED, zorder=2)
    bx.annotate(f"retained\n$k_U={K_U_RETAINED}$", xy=(K_U_RETAINED, 0.26),
                fontsize=8, color=INK, ha="center",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    bx.set_xlabel(f"Retained components $k_U$   (at $k_{{TMI}}={K_TMI}$)")
    bx.set_ylabel("Held-out mean recovered fraction")
    bx.set_title("(b) Case 2 — Alemão, leave-one-deposit-out", fontsize=10,
                 loc="left", color=INK)
    bx.set_ylim(0.24, 0.56)
    bx.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    bx.legend(fontsize=7.8, frameon=False, loc="lower right", labelcolor=INK)

    for a in (ax, bx):
        a.grid(alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case1-csv", default=str(CASE1_CSV))
    ap.add_argument("--case2-csv", default=str(CASE2_CSV))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args(argv)

    for path, what in ((args.case1_csv, "Case 1 sweep"), (args.case2_csv, "Case 2 LODO grid")):
        if not Path(path).exists():
            print(f"missing {what}: {path}", file=sys.stderr)
            return 1

    fig = build(pd.read_csv(args.case1_csv), pd.read_csv(args.case2_csv))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # TrueType, no Type 3 — journals reject Type 3
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    for ext, kw in (("pdf", {}), ("png", {"dpi": args.dpi})):
        p = outdir / f"{args.stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        print(f"Wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
