"""Replay the old paper-code Deposit 3 multivariate SPCA case.

This is a forensic helper: it imports the legacy paper code from the local
GitHub_Sofia project, rebuilds the matrix with the restored v5 config, and
compares the resulting two-stage ranking to the restored outputs_v5 ranking.
Run it with the old environment because the legacy module imports osgeo:

    /Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/my_env/bin/python scripts/replay_old_paper_code_dep3.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

OLD_PROJECT_ROOT = Path("/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project")
OLD_CODE_DIR = OLD_PROJECT_ROOT / "Code" / "paper_figures_and_code"
DEFAULT_CONFIG = OLD_CODE_DIR / "uni_multi_rotation_spatial_PCA_config_dep3_spca_multi_k17_v5.json"
DEFAULT_OLD_OUTPUT = Path(
    "/Users/sofiamantillasalas/Library/CloudStorage/OneDrive-Stanford/"
    "Research_Stanford/Research_files/MineralX_research/EroCopper_project/"
    "Carajas_maps_and_data/outputs_v5/"
    "Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17"
)
DEFAULT_DEBUG_JSON = (
    REPO_ROOT
    / "docs"
    / "repro_debug_deposit3_multivariate"
    / "old_paper_code_replay_diagnostics.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay legacy paper-code Deposit 3 SPCA ranking.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Legacy JSON config.")
    parser.add_argument("--old-output-dir", default=str(DEFAULT_OLD_OUTPUT), help="Restored v5 case dir.")
    parser.add_argument("--output-json", default=str(DEFAULT_DEBUG_JSON), help="Where to save diagnostics.")
    parser.add_argument(
        "--svd-solver",
        default="auto",
        choices=("auto", "full", "covariance_eigh", "randomized"),
        help="PCA svd_solver to force inside legacy functions.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random state for randomized PCA.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diagnostics = replay_case(args)
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(diagnostics), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(_console_summary(diagnostics)), indent=2))
    print(f"Wrote diagnostics: {output_path}")
    return 0


def replay_case(args: argparse.Namespace) -> dict[str, Any]:
    if str(OLD_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(OLD_CODE_DIR))

    import spca_paper_functions as spf

    if args.svd_solver != "auto":
        from sklearn.decomposition import PCA as SklearnPCA

        class ForcedPCA(SklearnPCA):
            def __init__(self, n_components=None):
                kwargs: dict[str, Any] = {
                    "n_components": n_components,
                    "svd_solver": args.svd_solver,
                }
                if args.svd_solver == "randomized":
                    kwargs["random_state"] = int(args.random_state)
                super().__init__(**kwargs)

        spf.PCA = ForcedPCA

    with Path(args.config).expanduser().open("r", encoding="utf-8") as infile:
        run_json = json.load(infile)

    old_payload_path = Path(args.old_output_dir).expanduser() / "validation_topk_results.pkl"
    with old_payload_path.open("rb") as infile:
        old_payload = pickle.load(infile)
    target_rows = np.asarray(old_payload["ranked_pred_rows"], dtype=int)

    run = run_json["run"]
    sweep = run_json["sweep"]
    analysis = run_json["analysis_defaults"]
    paths = run_json["paths"]
    targets = run_json["targets"]
    best_files = run_json["best_kpcs_files"]

    dep_number = int(sweep["deposits_1based"][0])
    deposit_id = dep_number - 1
    k_fused = int(sweep["kpcs"][0])
    variable_1 = str(analysis["variable_1"])
    variable_2 = str(analysis["variable_2"])
    outputs_dir = Path(str(run["base_output_dir"])).expanduser() / str(run["outputs_subdir"])

    k1_path = spf.resolve_existing_csv(
        str(outputs_dir / best_files["var1_filename"]),
        str(outputs_dir / best_files["var1_legacy_filename"]),
    )
    k2_path = spf.resolve_existing_csv(
        str(outputs_dir / best_files["var2_filename"]),
        str(outputs_dir / best_files["var2_legacy_filename"]),
    )
    k_var1 = _read_k_for_deposit(k1_path, dep_number)
    k_var2 = _read_k_for_deposit(k2_path, dep_number)

    variables_data, variables_meta, variables_transform, variables_extent, _, _, _ = (
        spf.load_merge_crop_prepare_two_vars(
            tif_directory=paths["tif_directory"],
            mask_files=list(paths["mask_files"]),
            var1_path=paths["tmi_file_path"],
            var2_path=paths["rad_file_path"],
            polygon_path=paths["polygon_path"],
            var1_name=variable_1,
            var2_name=variable_2,
            grayscale_weights=tuple(paths["grayscale_weights"]),
            force_crs=analysis["force_crs"],
            nodata_to_nan=paths["nodata_to_nan"],
        )
    )
    del variables_meta, variables_extent

    deposits_path = targets["deposits_shp_paths"][sweep["targets_shp_mode"]]
    _, _, deposit_data, deposit_extent = spf.load_deposit_and_extract_arrays(
        deposits_shp_path=deposits_path,
        deposit_id=deposit_id,
        variables_data=variables_data,
        variables_transform=variables_transform,
        variable_1=variable_1,
        variable_2=variable_2,
        extract_raster_from_polygon_fn=spf.extract_raster_from_polygon,
        verbose=False,
    )

    rot = spf.rotate_deposit_data(
        analysis_type=run["analysis_type"],
        selected_variable=run["uni_selected_variable"],
        variable_1=variable_1,
        variable_2=variable_2,
        deposit_data=deposit_data,
        deposit_extent=deposit_extent,
        rotation_angle=analysis["rotation_angle"],
        reshape=True,
        cval=0.0,
        vmin={variable_1: -150, variable_2: analysis["vmin_var2"]},
        vmax={variable_1: 150, variable_2: analysis["vmax_var2"]},
        order_uni=0,
        order_multi=1,
        verbose=False,
    )

    out_sw = spf.build_sliding_windows_and_pca_matrix(
        analysis_type=run["analysis_type"],
        variables_data=variables_data,
        variables_to_process=[variable_1, variable_2],
        window_shape=rot["window_shape"],
        stride_y=int(analysis["stride_y"]),
        stride_x=int(analysis["stride_x"]),
        deposit_rotated_data=rot["deposit_rotated_data"],
        selected_variable=run["uni_selected_variable"],
        pad_raster_fn=spf.pad_raster,
        get_padded_and_windows_fn=spf.get_padded_and_windows,
        verbose=False,
    )

    pca = spf.apply_pca_and_plot(
        data=out_sw["data_for_pca"],
        var_name=out_sw["pca_var_name"],
        patch_size=rot["window_shape"],
    )
    z = pca[f"{out_sw['pca_var_name']}_score"]
    eigvals = pca[f"{out_sw['pca_var_name']}_eigvals"]

    ranking = spf.run_spca_ranking_pipeline(
        X_multi=out_sw["data_for_pca"],
        Z=z,
        eigvals=eigvals,
        deposit_index=out_sw["deposit_index"],
        window_shape=rot["window_shape"],
        analysis_type=run["analysis_type"],
        multi_ranking_mode=run["multi_ranking_mode"],
        k_pcs_rank=k_fused,
        k_pcs_rank_var1=k_var1,
        k_pcs_rank_var2=k_var2,
        n_top_windows=int(analysis["n_top_windows"]),
        fusion_weight_var1=float(sweep["fusion_weight_var1"]),
        fusion_weight_var2=float(sweep["fusion_weight_var2"]),
    )

    ranked_idx = np.asarray(ranking["ranked_idx"], dtype=int)
    ranked_dists = np.asarray(ranking["ranked_dists"], dtype=float)
    n_windows = int(out_sw["window_indices_for_mapping"].shape[0])
    valid_mask = (ranked_idx != int(out_sw["deposit_index"])) & (ranked_idx < n_windows)
    replay_rows = ranked_idx[valid_mask]
    replay_dists = ranked_dists[valid_mask]
    fusion = ranking["fusion_details"]

    prefix = _prefix_match_count(target_rows, replay_rows)
    top20_overlap = len(set(target_rows[:20].tolist()).intersection(replay_rows[:20].tolist()))

    return {
        "legacy_code_dir": str(OLD_CODE_DIR),
        "config": str(Path(args.config).expanduser()),
        "old_payload": str(old_payload_path),
        "svd_solver": args.svd_solver,
        "random_state": int(args.random_state) if args.svd_solver == "randomized" else None,
        "data_for_pca_shape": list(np.asarray(out_sw["data_for_pca"]).shape),
        "deposit_index": int(out_sw["deposit_index"]),
        "window_shape": list(rot["window_shape"]),
        "number_valid_windows": n_windows,
        "k_pcs_var1": int(k_var1),
        "k_pcs_var2": int(k_var2),
        "k_pcs_fused": int(k_fused),
        "prefix_match_count": int(prefix),
        "top20_overlap": int(top20_overlap),
        "target_first20_ranked_rows": _int_list(target_rows, 20),
        "replay_first20_ranked_rows": _int_list(replay_rows, 20),
        "replay_first20_ranked_distances": _float_list(replay_dists, 20),
        "first20_raw_deposit_pca_scores": _float_list(fusion["zf_dep_full"], 20),
        "first20_eigenvalues": _float_list(fusion["eigvals_fused"], 20),
        "first20_weights": _float_list(ranking["weights"], 20),
        "sum_weights": float(np.sum(ranking["weights"])),
        "top_weighted_pcs_1based": _int_list(np.argsort(ranking["weights"])[::-1] + 1, 20),
    }


def _read_k_for_deposit(csv_path: str, dep_number: int) -> int:
    df = pd.read_csv(csv_path)
    match = df.loc[df["deposit_1based"].astype(int) == int(dep_number), "k_pcs"]
    if match.empty:
        raise ValueError(f"No row for deposit {dep_number} in {csv_path}")
    return int(match.iloc[0])


def _prefix_match_count(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.size, b.size)
    count = 0
    for idx in range(n):
        if int(a[idx]) != int(b[idx]):
            break
        count += 1
    return count


def _console_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "svd_solver",
        "data_for_pca_shape",
        "deposit_index",
        "window_shape",
        "number_valid_windows",
        "k_pcs_var1",
        "k_pcs_var2",
        "k_pcs_fused",
        "prefix_match_count",
        "top20_overlap",
        "target_first20_ranked_rows",
        "replay_first20_ranked_rows",
        "first20_raw_deposit_pca_scores",
        "first20_weights",
        "top_weighted_pcs_1based",
    ]
    return {key: diagnostics[key] for key in keys}


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
