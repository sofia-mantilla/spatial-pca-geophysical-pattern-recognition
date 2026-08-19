"""Debug Carajas Deposit 3 multivariate TMI + U SPCA reproducibility.

This script rebuilds the Deposit 3, K=17 two-stage fused PCA ranking inputs
and saves the diagnostics that matter for comparing paper-era and current
workflow behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

DEFAULT_CONFIG = "configs/carajas_multi_tmi_u_square_tmi2_u34.yaml"
DEFAULT_OLD_OUTPUT = (
    "/Users/sofiamantillasalas/Library/CloudStorage/OneDrive-Stanford/"
    "Research_Stanford/Research_files/MineralX_research/EroCopper_project/"
    "Carajas_maps_and_data/outputs_v5/"
    "Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17"
)
DEFAULT_DEBUG_DIR = "docs/repro_debug_deposit3_multivariate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save reproducibility diagnostics for Carajas Deposit 3 multivariate SPCA."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML/JSON config to use.")
    parser.add_argument("--deposit", type=int, default=3, help="1-based deposit number.")
    parser.add_argument("--kpcs", type=int, default=17, help="Fused PC count.")
    parser.add_argument("--old-output-dir", default=DEFAULT_OLD_OUTPUT, help="Paper-era case folder.")
    parser.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR, help="Directory for diagnostics.")
    parser.add_argument(
        "--weight-mode",
        choices=("square", "abs"),
        default="square",
        help="Deposit score transform used for component weights.",
    )
    parser.add_argument(
        "--normalize-weights-over",
        choices=("selected_pcs", "all_pcs"),
        default="selected_pcs",
        help="Denominator used when normalizing component weights.",
    )
    parser.add_argument(
        "--use-whitening",
        action="store_true",
        help="Rank in eigenvalue-whitened fused PCA space.",
    )
    parser.add_argument(
        "--no-use-weights",
        action="store_true",
        help="Rank by unweighted L2 distances in the selected comparison space.",
    )
    parser.add_argument(
        "--stage1-pca-svd-solver",
        default="auto",
        help="sklearn PCA svd_solver for the per-variable PCA fits.",
    )
    parser.add_argument(
        "--fused-pca-svd-solver",
        default="auto",
        help="sklearn PCA svd_solver for the fused PCA fit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve_repo_path(args.config)
    debug_dir = _resolve_repo_path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = build_diagnostics(args, config_path=config_path, debug_dir=debug_dir)
    json_path = debug_dir / "deposit3_multivariate_repro_diagnostics.json"
    json_path.write_text(json.dumps(_jsonable(diagnostics), indent=2) + "\n", encoding="utf-8")

    print(json.dumps(_jsonable(_short_console_summary(diagnostics)), indent=2))
    print(f"Wrote diagnostics JSON: {json_path}")
    print(f"Wrote component-weight plot: {diagnostics['current_reproduction']['component_weights_plot']}")
    return 0


def build_diagnostics(
    args: argparse.Namespace,
    *,
    config_path: Path,
    debug_dir: Path,
) -> dict[str, Any]:
    from spatial_pca.config import load_run_config
    from spatial_pca.geodata.deposits import get_deposit_template
    from spatial_pca.pipeline import (
        _get_deposits_path,
        _get_summary_variables,
        _load_case_deposits,
        _resolve_multivariate_best_kpcs,
        _validate_multivariate_rasters,
        load_multivariate_rasters,
    )
    from spatial_pca.spca.pca import fit_spca
    from spatial_pca.spca.ranking import rank_multi_two_stage_pca_fusion
    from spatial_pca.spca.windows import build_multivariate_window_matrix
    from spatial_pca.validation.diagnostics import plot_deposit_scores_and_weights

    config = load_run_config(config_path)
    variable_names = _get_summary_variables(config)
    raster_data = load_multivariate_rasters(config, variable_names)
    _validate_multivariate_rasters(raster_data, reference_variable=variable_names[0])
    deposits_path = _get_deposits_path(config)
    deposits_gdf = _load_case_deposits(config, deposits_path, raster_data[variable_names[0]].crs)

    templates = {
        var: get_deposit_template(deposits_gdf, int(args.deposit) - 1, raster_data[var])
        for var in variable_names
    }
    window_matrix = build_multivariate_window_matrix(
        rasters={var: raster_data[var].array for var in variable_names},
        deposit_templates={var: templates[var].array for var in variable_names},
        variable_names=variable_names,
        stride_y=int(config["analysis_defaults"]["stride_y"]),
        stride_x=int(config["analysis_defaults"]["stride_x"]),
    )
    pca_result = fit_spca(
        window_matrix.data_for_pca,
        var_name="Combined",
        patch_size=window_matrix.window_shape,
    )
    k_var1, k_var2 = _resolve_multivariate_best_kpcs(config, int(args.deposit))
    ranking_result = rank_multi_two_stage_pca_fusion(
        X_multi=window_matrix.data_for_pca,
        deposit_index=window_matrix.deposit_index,
        window_shape=window_matrix.window_shape,
        k_pcs_var1=k_var1,
        k_pcs_var2=k_var2,
        k_pcs_fused=int(args.kpcs),
        use_whitening=bool(args.use_whitening),
        use_weights=not bool(args.no_use_weights),
        weight_mode=str(args.weight_mode),
        normalize_weights_over=str(args.normalize_weights_over),
        stage1_pca_svd_solver=str(args.stage1_pca_svd_solver),
        fused_pca_svd_solver=str(args.fused_pca_svd_solver),
        standardize_fused_input=True,
    )
    fusion = ranking_result.fusion_details or {}

    component_plot = plot_deposit_scores_and_weights(
        scores=fusion["Zf_scores"],
        explained_variance_ratio=fusion["explained_variance_ratio_fused"],
        weights=ranking_result.weights,
        deposit_index=window_matrix.deposit_index,
        k_used=ranking_result.k_used,
        output_path=debug_dir / "component_weights_debug.png",
        k_display=min(int(ranking_result.k_used), 12),
        weight_ylim=(0.0, 1.0),
    )

    n_windows = int(window_matrix.window_indices_for_mapping.shape[0])
    valid_rank_mask = (
        (ranking_result.ranked_idx != window_matrix.deposit_index)
        & (ranking_result.ranked_idx < n_windows)
    )
    valid_ranked_idx = ranking_result.ranked_idx[valid_rank_mask]
    valid_ranked_dists = ranking_result.ranked_dists[valid_rank_mask]
    first_window_rows = valid_ranked_idx[:20]
    first_window_grid = window_matrix.window_indices_for_mapping[first_window_rows]

    current = {
        "label": "current code with paper-era Deposit 3 two-stage settings",
        "config_path": str(config_path),
        "variables": list(variable_names),
        "raster_paths": {
            var: str(_resolve_repo_path(config["paths"][f"variable_{idx + 1}_file_path"]))
            for idx, var in enumerate(variable_names)
        },
        "crop_polygon_path": str(_resolve_repo_path(config["paths"]["polygon_path"])),
        "deposits_path": str(deposits_path),
        "deposit_1based": int(args.deposit),
        "deposit_index": int(window_matrix.deposit_index),
        "window_shape": list(window_matrix.window_shape),
        "data_for_pca_shape": list(window_matrix.data_for_pca.shape),
        "number_valid_windows": n_windows,
        "stride_y": int(config["analysis_defaults"]["stride_y"]),
        "stride_x": int(config["analysis_defaults"]["stride_x"]),
        "k_pcs_var1": int(k_var1),
        "k_pcs_var2": int(k_var2),
        "k_pcs_fused": int(ranking_result.k_used),
        "ranking_mode": ranking_result.ranking_mode,
        "use_whitening": bool(ranking_result.use_whitening),
        "use_weights": bool(ranking_result.use_weights),
        "weight_mode": ranking_result.weight_mode,
        "normalize_weights_over": ranking_result.normalize_weights_over,
        "stage1_pca_svd_solver": str(fusion["stage1_pca_svd_solver"]),
        "fused_pca_svd_solver": str(fusion["fused_pca_svd_solver"]),
        "standardize_fused_input": bool(fusion["standardize_fused_input"]),
        "first20_raw_deposit_pca_scores": _float_list(fusion["zf_dep_full"], 20),
        "first20_eigenvalues": _float_list(fusion["eigvals_fused"], 20),
        "first20_weights": _float_list(ranking_result.weights, 20),
        "sum_weights": float(np.sum(ranking_result.weights)),
        "top_weighted_pcs_1based": _int_list(np.argsort(ranking_result.weights)[::-1] + 1, 20),
        "first20_ranked_window_rows": _int_list(first_window_rows, 20),
        "first20_ranked_window_grid_row_col_id": first_window_grid[:20].astype(int).tolist(),
        "first20_ranked_distances": _float_list(valid_ranked_dists, 20),
        "component_weights_plot": str(component_plot),
    }

    return {
        "current_reproduction": current,
        "old_saved_artifact": _load_old_artifact(Path(str(args.old_output_dir)).expanduser()),
    }


def _load_old_artifact(old_output_dir: Path) -> dict[str, Any]:
    validation_path = old_output_dir / "validation_topk_results.pkl"
    image_path = old_output_dir / "Fused_dep_3_rot_0_deg_weights_usedK.png"
    artifact: dict[str, Any] = {
        "case_dir": str(old_output_dir),
        "validation_pickle": str(validation_path),
        "weights_figure": str(image_path),
        "available": validation_path.exists(),
        "weights_figure_exists": image_path.exists(),
    }
    if not validation_path.exists():
        return artifact

    with validation_path.open("rb") as infile:
        payload = pickle.load(infile)
    artifact["payload_keys"] = sorted(str(key) for key in payload.keys())
    artifact["ranking_mode"] = payload.get("ranking_mode")
    artifact["spca_diagnostics"] = payload.get("spca_diagnostics", {})
    for key in ("ranked_pred_rows", "ranked_idx", "top_pred_indices"):
        if key in payload:
            artifact[f"first20_{key}"] = _int_list(payload[key], 20)
    for key in ("ranked_pred_distances", "ranked_dists", "top_pred_distances"):
        if key in payload:
            artifact[f"first20_{key}"] = _float_list(payload[key], 20)
    return artifact


def _short_console_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    current = diagnostics["current_reproduction"]
    old = diagnostics["old_saved_artifact"]
    return {
        "current_reproduction": {
            key: current[key]
            for key in (
                "data_for_pca_shape",
                "deposit_index",
                "window_shape",
                "number_valid_windows",
                "first20_raw_deposit_pca_scores",
                "first20_eigenvalues",
                "first20_weights",
                "sum_weights",
                "top_weighted_pcs_1based",
                "weight_mode",
                "normalize_weights_over",
                "use_whitening",
                "use_weights",
                "first20_ranked_window_rows",
                "first20_ranked_distances",
            )
        },
        "old_saved_artifact": {
            "available": old["available"],
            "ranking_mode": old.get("ranking_mode"),
            "spca_diagnostics": old.get("spca_diagnostics", {}),
            "first20_ranked_pred_rows": old.get("first20_ranked_pred_rows"),
        },
    }


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _float_list(values: Any, n: int) -> list[float]:
    arr = np.asarray(values, dtype=float).ravel()[:n]
    return [float(value) for value in arr]


def _int_list(values: Any, n: int) -> list[int]:
    arr = np.asarray(values, dtype=int).ravel()[:n]
    return [int(value) for value in arr]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
