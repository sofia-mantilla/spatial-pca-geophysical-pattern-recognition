"""Case 1 sensitivity sweep: recompute the Paulo Afonso ranking for a range of k.

Produces the evidence behind Appendix B, panel (a). For each k the full univariate
TMI pipeline is rerun and the validation result is summarised into one CSV row.

Run from the repo root (~20 min for the default grid):

    python paper/run_case1_k_sweep.py

Options:
    --config   run config (default configs/carajas_uni_tmi.yaml)
    --kpcs     space-separated k values to sweep
    --out      CSV path (default paper/appendix_b/k_sweep_case1.csv)
    --reuse    skip k values whose output folder already has a validation pickle
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

DEFAULT_CONFIG = "configs/carajas_uni_tmi.yaml"
DEFAULT_KPCS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 20, 25, 30, 40, 60]
OUT_SUBDIR = "Output_Carajas_Brazil_Univariate_TMI"
FIELDS = [
    "k_pcs",
    "mean_recovered_frac_250",
    "auc_recovery",
    "n_hits",
    "first_hit_rank",
    "deposits_hit_1based",
]


def summarise(pkl_path: Path, k: int) -> dict:
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    c = np.asarray(d["cum_mean_recovered_frac"], dtype=float)
    hits = d["first_hit_rank_by_deposit"]
    return {
        "k_pcs": k,
        "mean_recovered_frac_250": round(float(c[-1]), 6),
        "auc_recovery": round(float(np.sum(c)), 4),
        "n_hits": len(hits),
        "first_hit_rank": min(hits.values()) if hits else "",
        # pickle keys are 0-based deposit indices; the paper numbers deposits from 1
        "deposits_hit_1based": "|".join(str(i + 1) for i in sorted(hits)),
    }


def find_pickle(out_dir: Path, k: int) -> Path | None:
    hits = sorted(out_dir.glob(f"Deposit_*_kpcs_{k}/validation_topk_results.pkl"))
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--kpcs", type=int, nargs="+", default=DEFAULT_KPCS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args(argv)

    out_dir = REPO / "outputs" / OUT_SUBDIR
    out_csv = (Path(args.out) if args.out
               else REPO / "paper" / "appendix_b" / "k_sweep_case1.csv")
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    rows = []
    for k in args.kpcs:
        pkl = find_pickle(out_dir, k) if args.reuse else None
        if pkl is None:
            print(f"[k={k}] running pipeline ...", flush=True)
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "run_project_from_config.py"),
                 "--config", args.config, "--kpcs", str(k)],
                cwd=REPO, env=env, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            pkl = find_pickle(out_dir, k)
        if pkl is None:
            raise RuntimeError(f"no validation pickle produced for k={k}")
        row = summarise(pkl, k)
        rows.append(row)
        print(f"[k={k}] c-bar {row['mean_recovered_frac_250']:.4f}  "
              f"hits {row['n_hits']}  AUC {row['auc_recovery']}", flush=True)

    rows.sort(key=lambda r: r["k_pcs"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
