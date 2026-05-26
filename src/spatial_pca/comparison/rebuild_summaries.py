"""Rebuild comparison summary CSVs from saved validation pickles."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from spatial_pca.comparison.comparison_cases import (
    CaseInfo,
    find_all_validation_pkls,
    load_validation_dict,
)
from spatial_pca.validation.summaries import SUMMARY_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild kpcs sweep/best CSVs from validation_topk_results.pkl files."
    )
    parser.add_argument("--output-root", required=True, help="Output case root to scan.")
    parser.add_argument("--method-tag", required=True, help="CSV filename tag to write.")
    parser.add_argument("--method-name", default="Spatial_PCA", help="Method name to include.")
    parser.add_argument("--analysis-type", default="Multi", choices=("Uni", "Multi"))
    parser.add_argument("--selected-variable", help="Optional univariate selected variable filter.")
    parser.add_argument("--min-cover", type=float, default=0.5, help="Optional min-cover filter.")
    parser.add_argument("--deposits", nargs="*", type=int, help="Optional 1-based deposits to include.")
    return parser


def rebuild_summary_tables(
    *,
    output_root: str | Path,
    method_tag: str,
    method_name: str = "Spatial_PCA",
    analysis_type: str = "Multi",
    selected_variable: str | None = None,
    min_cover: float | None = 0.5,
    deposits: list[int] | None = None,
) -> list[Path]:
    output_root_path = Path(output_root).expanduser().resolve()
    cases = find_all_validation_pkls([str(output_root_path)], deposits)
    rows = [
        summarize_case(case)
        for case in cases
        if _matches_case(
            case,
            method_name=method_name,
            analysis_type=analysis_type,
            selected_variable=selected_variable,
            min_cover=min_cover,
        )
    ]
    rows = sorted(rows, key=lambda r: (int(r["deposit_1based"]), int(r["k_pcs"])))
    if not rows:
        raise RuntimeError(
            "No matching validation pickles found for "
            f"{method_name} {analysis_type} under {output_root_path}"
        )

    all_path = output_root_path / f"kpcs_sweep_auc_by_deposit_{method_tag}.csv"
    argmax_path = output_root_path / f"kpcs_min_argmax_auc_by_deposit_{method_tag}.csv"
    best_path = output_root_path / f"kpcs_best_multicriteria_by_deposit_{method_tag}.csv"

    _write_csv(all_path, rows)
    best_rows = _best_auc_rows(rows)
    _write_csv(argmax_path, best_rows)
    _write_csv(best_path, best_rows)
    return [all_path, argmax_path, best_path]


def summarize_case(case: CaseInfo) -> dict[str, Any]:
    payload = load_validation_dict(str(case.pkl_path))
    curve = np.asarray(payload.get("cum_recovered_frac_total", []), dtype=float)
    hit_ranks = sorted(int(rank) for rank in (payload.get("hit_by_rank", {}) or {}))
    variables = _case_variables(case, payload)

    return {
        "deposit_1based": int(case.deposit_1based),
        "method_name": case.method_name,
        "analysis_type": case.analysis_type,
        "variables": variables,
        "k_pcs": int(_case_k_pcs(case, payload)),
        "auc_recovery": _normalized_auc(curve),
        "recovery_end": float(curve[-1]) if curve.size else 0.0,
        "red_points_count": int(len(hit_ranks)),
        "first_red_rank": float(hit_ranks[0]) if hit_ranks else math.nan,
        "mean_red_rank": float(np.mean(hit_ranks)) if hit_ranks else math.nan,
        "hit_earliness_score": float(np.sum(1.0 / np.asarray(hit_ranks, dtype=float))) if hit_ranks else 0.0,
        "k_eval": int(curve.size),
        "output_dir": str(case.case_dir),
        "validation_pkl": str(case.pkl_path),
    }


def _matches_case(
    case: CaseInfo,
    *,
    method_name: str,
    analysis_type: str,
    selected_variable: str | None,
    min_cover: float | None,
) -> bool:
    if case.method_name != method_name:
        return False
    if case.analysis_type != analysis_type:
        return False
    if min_cover is not None and case.min_cover is not None and abs(case.min_cover - min_cover) > 1e-9:
        return False
    if selected_variable is not None and case.selected_variable != selected_variable:
        return False
    return True


def _case_variables(case: CaseInfo, payload: dict[str, Any]) -> str:
    payload_vars = payload.get("variables")
    if isinstance(payload_vars, (list, tuple)):
        return "+".join(str(v) for v in payload_vars)
    if isinstance(payload_vars, str):
        return payload_vars
    if case.analysis_type == "Uni":
        return str(case.selected_variable or payload.get("selected_variable") or "")
    return "+".join(v for v in [case.variable_1, case.variable_2] if v)


def _case_k_pcs(case: CaseInfo, payload: dict[str, Any]) -> int:
    if case.k_pcs is not None:
        return int(case.k_pcs)
    for key in ("k_pcs", "k_pcs_fused"):
        value = payload.get(key)
        if value is not None:
            return int(value)
    raise KeyError(f"Could not determine k_pcs for {case.pkl_path}")


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
        row_key = (float(row["auc_recovery"]), -int(row["k_pcs"]))
        current_key = (
            (float(current["auc_recovery"]), -int(current["k_pcs"]))
            if current is not None
            else None
        )
        if current is None or row_key > current_key:
            best_by_deposit[dep_id] = row
    return [best_by_deposit[dep_id] for dep_id in sorted(best_by_deposit)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = rebuild_summary_tables(
        output_root=args.output_root,
        method_tag=args.method_tag,
        method_name=args.method_name,
        analysis_type=args.analysis_type,
        selected_variable=args.selected_variable,
        min_cover=args.min_cover,
        deposits=args.deposits,
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
