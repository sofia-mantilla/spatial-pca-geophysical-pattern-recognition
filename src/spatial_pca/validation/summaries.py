"""Summary-table helpers for SPCA sweep outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_COLUMNS = [
    "deposit_1based",
    "method_name",
    "analysis_type",
    "variables",
    "k_pcs",
    "auc_recovery",
    "recovery_end",
    "red_points_count",
    "first_red_rank",
    "mean_red_rank",
    "hit_earliness_score",
    "k_eval",
    "output_dir",
    "validation_pkl",
]


def write_sweep_summary_tables(
    results: list[Any],
    *,
    output_dir: str | Path,
    method_tag: str,
    method_name: str,
    analysis_type: str,
    variables: list[str],
) -> list[Path]:
    """Write run-group CSV summaries from saved SPCA output objects."""

    if not results:
        return []

    rows = [
        summarize_result(
            result,
            method_name=method_name,
            analysis_type=analysis_type,
            variables=variables,
        )
        for result in results
    ]
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    all_path = base_dir / f"kpcs_sweep_auc_by_deposit_{method_tag}.csv"
    argmax_path = base_dir / f"kpcs_min_argmax_auc_by_deposit_{method_tag}.csv"
    best_path = base_dir / f"kpcs_best_multicriteria_by_deposit_{method_tag}.csv"

    _write_csv(all_path, rows)
    best_rows = _best_auc_rows(rows)
    _write_csv(argmax_path, best_rows)
    _write_csv(best_path, best_rows)
    return [all_path, argmax_path, best_path]


def summarize_result(
    result: Any,
    *,
    method_name: str,
    analysis_type: str,
    variables: list[str],
) -> dict[str, Any]:
    """Build one summary row from an SPCAOutput-like object."""

    recovery = result.recovery_result
    curve = np.asarray(recovery.cum_recovered_frac_total, dtype=float)
    k_eval = int(curve.size)
    auc = _normalized_auc(curve)
    hit_ranks = sorted(int(rank) for rank in recovery.hit_by_rank)

    if hit_ranks:
        first_red_rank = float(hit_ranks[0])
        mean_red_rank = float(np.mean(hit_ranks))
        hit_earliness_score = float(np.sum(1.0 / np.asarray(hit_ranks, dtype=float)))
    else:
        first_red_rank = np.nan
        mean_red_rank = np.nan
        hit_earliness_score = 0.0

    return {
        "deposit_1based": int(result.deposit_1based),
        "method_name": method_name,
        "analysis_type": analysis_type,
        "variables": "+".join(variables),
        "k_pcs": int(result.k_pcs),
        "auc_recovery": auc,
        "recovery_end": float(curve[-1]) if k_eval else 0.0,
        "red_points_count": int(len(hit_ranks)),
        "first_red_rank": first_red_rank,
        "mean_red_rank": mean_red_rank,
        "hit_earliness_score": hit_earliness_score,
        "k_eval": k_eval,
        "output_dir": str(result.case_output_dir),
        "validation_pkl": str(result.validation_path),
    }


def _normalized_auc(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    if curve.size == 1:
        return float(curve[0])
    x = np.linspace(0.0, 1.0, curve.size)
    return float(np.trapezoid(curve, x))


def _best_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_deposit: dict[int, dict[str, Any]] = {}
    for row in rows:
        dep_id = int(row["deposit_1based"])
        current = best_by_deposit.get(dep_id)
        if current is None:
            best_by_deposit[dep_id] = row
            continue
        row_key = (float(row["auc_recovery"]), -int(row["k_pcs"]))
        current_key = (float(current["auc_recovery"]), -int(current["k_pcs"]))
        if row_key > current_key:
            best_by_deposit[dep_id] = row
    return [best_by_deposit[dep_id] for dep_id in sorted(best_by_deposit)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
