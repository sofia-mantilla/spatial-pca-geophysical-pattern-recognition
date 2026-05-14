"""Pipeline wiring for SPCA workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatial_pca.config import load_run_config, resolve_variable_raster_path
from spatial_pca.geodata.deposits import (
    TemplateData,
    get_circle_patch_template,
    get_deposit_template,
    load_deposits,
)
from spatial_pca.geodata.exports import (
    build_circle_top_window_centers_gdf,
    build_circle_top_windows_gdf,
    build_top_windows_gdf,
    save_geopackage,
)
from spatial_pca.geodata.rasters import RasterData, load_raster
from spatial_pca.provenance import build_provenance, write_provenance
from spatial_pca.spca.pca import PCAResult, fit_spca
from spatial_pca.spca.ranking import RankingResult, rank_spca_windows, run_spca_ranking_pipeline
from spatial_pca.spca.windows import (
    WindowMatrix,
    build_multivariate_circle_window_matrix,
    build_multivariate_window_matrix,
    build_univariate_circle_window_matrix,
    build_univariate_window_matrix,
)
from spatial_pca.validation.diagnostics import (
    plot_deposit_scores_and_weights,
    plot_loading_maps,
    plot_multivariate_rotated_deposit,
    plot_multivariate_top_similar_windows,
    plot_pc_score_map,
    plot_reconstruction_progression,
    plot_rotated_deposit,
    plot_score_pairs,
    plot_top_similar_windows,
    plot_two_stage_multivariate_reconstruction_progression,
)
from spatial_pca.validation.footprint_recovery import (
    FootprintRecoveryResult,
    build_validation_payload,
    plot_cumulative_recovery,
    plot_top_windows_overlay,
    validate_footprint_recovery,
    write_validation_payload,
)
from spatial_pca.validation.summaries import write_sweep_summary_tables


@dataclass(frozen=True)
class SPCAOutput:
    """Container for the outputs of a single SPCA case run."""

    deposit_1based: int
    k_pcs: int
    case_output_dir: Path
    raster_data: RasterData | dict[str, RasterData]
    deposit_template: TemplateData | dict[str, TemplateData]
    window_matrix: WindowMatrix
    pca_result: PCAResult
    ranking_result: RankingResult
    recovery_result: FootprintRecoveryResult
    top_windows_path: Path
    validation_path: Path
    recovery_plot_path: Path
    top_windows_plot_path: Path
    pc_score_map_path: Path
    diagnostic_paths: dict[str, Path]
    resolved_config_path: Path
    provenance_path: Path


def run_spca_from_config(
    config_source: str | Path | dict[str, Any],
    *,
    deposit_1based: int | None = None,
    k_pcs: int | None = None,
    output_dir_override: str | Path | None = None,
    top_k: int | None = None,
) -> list[SPCAOutput]:
    """Run SPCA for one or more deposits/kpcs from a Spatial PCA config."""

    config = _load_config(config_source)
    if output_dir_override is not None:
        config["resolved"]["output_dir"] = str(Path(output_dir_override).expanduser().resolve())

    plan = _build_run_plan(config, deposit_1based=deposit_1based, k_pcs=k_pcs)
    results: list[SPCAOutput] = []

    for deposit_value, k_value in plan:
        result = run_single_case(
            config=config,
            deposit_1based=deposit_value,
            k_pcs=k_value,
            top_k=top_k,
        )
        results.append(result)

    write_sweep_summary_tables(
        results,
        output_dir=config["resolved"]["output_dir"],
        method_tag=_build_method_tag(config),
        method_name=config["run"]["method_name"],
        analysis_type=config["run"]["analysis_type"],
        variables=_get_summary_variables(config),
    )

    return results


def run_single_case(
    *,
    config: dict[str, Any],
    deposit_1based: int,
    k_pcs: int,
    top_k: int | None = None,
) -> SPCAOutput:
    """Run a single SPCA case and write case outputs to disk."""

    output_dir = build_case_output_dir(config, deposit_1based, k_pcs)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_type = str(config["run"]["analysis_type"])
    deposits_path = _get_deposits_path(config)
    n_top_windows = top_k or int(config["analysis_defaults"]["n_top_windows"])

    if analysis_type == "Uni":
        variable_name = _get_variable_name(config)
        raster_data = load_variable_raster(config, variable_name)
        reference_raster = raster_data
        deposits_gdf = _load_case_deposits(config, deposits_path, reference_raster.crs)
        patch_config = config.get("patch")
        if patch_config is None:
            template = get_deposit_template(deposits_gdf, deposit_1based - 1, raster_data)
            window_matrix = build_univariate_window_matrix(
                raster=raster_data.array,
                deposit_template=template.array,
                variable_name=variable_name,
                stride_y=int(config["analysis_defaults"]["stride_y"]),
                stride_x=int(config["analysis_defaults"]["stride_x"]),
            )
        else:
            _require_circle_patch(patch_config)
            template = get_circle_patch_template(
                patch_config=patch_config,
                deposits_gdf=deposits_gdf,
                deposit_index=deposit_1based - 1,
                raster_data=raster_data,
            )
            window_matrix = build_univariate_circle_window_matrix(
                raster=raster_data.array,
                circle_template=template.array,
                variable_name=variable_name,
                stride_y=int(config["analysis_defaults"]["stride_y"]),
                stride_x=int(config["analysis_defaults"]["stride_x"]),
                radius_m=float(template.radius_m),
                feature_mask=template.feature_mask,
            )
        pca_result = fit_spca(
            window_matrix.data_for_pca,
            var_name=variable_name,
            patch_size=window_matrix.window_shape,
        )
        ranking_result = rank_spca_windows(
            scores=pca_result.scores,
            eigvals=pca_result.eigvals,
            deposit_index=window_matrix.deposit_index,
            k_pcs=k_pcs,
        )
        summary_variables = [variable_name]
        validation_extras = {
            "ranking_mode": ranking_result.ranking_mode,
            "k_pcs_var1": np.nan,
            "k_pcs_var2": np.nan,
            "k_pcs_fused": np.nan,
        }
    elif analysis_type == "Multi":
        summary_variables = _get_summary_variables(config)
        raster_data = load_multivariate_rasters(config, summary_variables)
        reference_raster = raster_data[summary_variables[0]]
        _validate_multivariate_rasters(raster_data, reference_variable=summary_variables[0])
        deposits_gdf = _load_case_deposits(config, deposits_path, reference_raster.crs)
        patch_config = config.get("patch")
        if patch_config is None:
            template = {
                var: get_deposit_template(deposits_gdf, deposit_1based - 1, raster_data[var])
                for var in summary_variables
            }
            window_matrix = build_multivariate_window_matrix(
                rasters={var: raster_data[var].array for var in summary_variables},
                deposit_templates={var: template[var].array for var in summary_variables},
                variable_names=summary_variables,
                stride_y=int(config["analysis_defaults"]["stride_y"]),
                stride_x=int(config["analysis_defaults"]["stride_x"]),
            )
        else:
            _require_circle_patch(patch_config)
            template = {
                var: get_circle_patch_template(
                    patch_config=patch_config,
                    deposits_gdf=deposits_gdf,
                    deposit_index=deposit_1based - 1,
                    raster_data=raster_data[var],
                )
                for var in summary_variables
            }
            shared_radius_m = float(template[summary_variables[0]].radius_m)
            window_matrix = build_multivariate_circle_window_matrix(
                rasters={var: raster_data[var].array for var in summary_variables},
                circle_templates={var: template[var].array for var in summary_variables},
                variable_names=summary_variables,
                stride_y=int(config["analysis_defaults"]["stride_y"]),
                stride_x=int(config["analysis_defaults"]["stride_x"]),
                radius_m=shared_radius_m,
                feature_mask=template[summary_variables[0]].feature_mask,
            )
        pca_result = fit_spca(
            window_matrix.data_for_pca,
            var_name="Combined",
            patch_size=window_matrix.window_shape,
        )
        k_pcs_var1, k_pcs_var2 = _resolve_multivariate_best_kpcs(config, deposit_1based)
        ranking_out = run_spca_ranking_pipeline(
            X_multi=window_matrix.data_for_pca,
            Z=pca_result.scores,
            eigvals=pca_result.eigvals,
            deposit_index=window_matrix.deposit_index,
            window_shape=window_matrix.window_shape,
            analysis_type=analysis_type,
            multi_ranking_mode=str(config["run"].get("multi_ranking_mode", "two_stage_pca_fusion")),
            k_pcs_rank=k_pcs,
            k_pcs_rank_var1=k_pcs_var1,
            k_pcs_rank_var2=k_pcs_var2,
            n_top_windows=n_top_windows,
            features_per_variable=(
                int(window_matrix.feature_mask.sum())
                if window_matrix.patch_geometry_type == "circle" and window_matrix.feature_mask is not None
                else None
            ),
        )
        ranking_result = ranking_out["ranking_result"]
        variable_name = "Combined"
        validation_extras = {
            "ranking_mode": ranking_result.ranking_mode,
            "k_pcs_var1": int(k_pcs_var1),
            "k_pcs_var2": int(k_pcs_var2),
            "k_pcs_fused": int(k_pcs),
        }
    else:
        raise NotImplementedError(f"Unsupported analysis_type '{analysis_type}'.")

    n_windows = window_matrix.window_indices_for_mapping.shape[0]
    valid_rank_mask = (
        (ranking_result.ranked_idx != window_matrix.deposit_index)
        & (ranking_result.ranked_idx < n_windows)
    )
    valid_ranked_idx = ranking_result.ranked_idx[valid_rank_mask][:n_top_windows]
    valid_ranked_dists = ranking_result.ranked_dists[valid_rank_mask][:n_top_windows]
    top_window_indices = window_matrix.window_indices_for_mapping[valid_ranked_idx]
    extra_columns = {"deposit_1based": np.full(len(valid_ranked_idx), deposit_1based, dtype=int)}
    top_windows_path = output_dir / "top_windows.gpkg"
    if window_matrix.patch_geometry_type == "circle":
        top_windows_gdf = build_circle_top_windows_gdf(
            window_indices=top_window_indices,
            window_shape=window_matrix.window_shape,
            transform=reference_raster.transform,
            radius_m=float(window_matrix.patch_radius_m),
            crs=reference_raster.crs,
            ranks=np.arange(1, len(valid_ranked_idx) + 1, dtype=int),
            scores=valid_ranked_dists,
            window_ids=top_window_indices[:, 2],
            extra_columns=extra_columns,
        )
        save_geopackage(top_windows_gdf, top_windows_path, layer_name="top_windows")
        export_geometry = str(config.get("patch", {}).get("export_geometry", "polygon"))
        if export_geometry in {"point", "both"}:
            centers_gdf = build_circle_top_window_centers_gdf(top_windows_gdf)
            save_geopackage(centers_gdf, top_windows_path, layer_name="top_window_centers")
    else:
        top_windows_gdf = build_top_windows_gdf(
            window_indices=top_window_indices[:, :2],
            window_shape=window_matrix.window_shape,
            transform=reference_raster.transform,
            crs=reference_raster.crs,
            ranks=np.arange(1, len(valid_ranked_idx) + 1, dtype=int),
            scores=valid_ranked_dists,
            window_ids=top_window_indices[:, 2],
            extra_columns=extra_columns,
        )
        save_geopackage(top_windows_gdf, top_windows_path)

    resolved_config_path = write_resolved_config(config, output_dir / "run_config_resolved.json")
    provenance = build_provenance(config)
    provenance_path = write_provenance(provenance, output_dir / "run_provenance.json")
    validation_deposits_gdf = _load_case_deposits(
        config,
        deposits_path,
        reference_raster.crs,
        policy_key="validation_deposit_crs_policy",
    )
    recovery_result = validate_footprint_recovery(
        top_windows_gdf=top_windows_gdf,
        deposits_gdf=validation_deposits_gdf,
        reference_deposit_index=deposit_1based - 1,
        min_cover=float(config["analysis_defaults"]["min_cover"]),
    )
    spca_diagnostics = {
        "k_used": int(ranking_result.k_used),
        "use_whitening": bool(ranking_result.use_whitening),
        "use_weights": bool(ranking_result.use_weights),
    }
    if ranking_result.ranking_mode == "two_stage_pca_fusion":
        fusion_details = ranking_result.fusion_details or {}
        spca_diagnostics.update(
            {
                "K_var1": int(fusion_details["K_var1"]),
                "K_var2": int(fusion_details["K_var2"]),
                "K_fused": int(fusion_details["K_fused"]),
                "standardize_fused_input": bool(fusion_details["standardize_fused_input"]),
            }
        )
    validation_payload = build_validation_payload(
        recovery=recovery_result,
        method_name=config["run"]["method_name"],
        analysis_type=config["run"]["analysis_type"],
        deposit_id=deposit_1based,
        variables=summary_variables,
        k_pcs=k_pcs,
        min_cover=float(config["analysis_defaults"]["min_cover"]),
        top_windows_path=top_windows_path,
        run_config_path=resolved_config_path,
        provenance_path=provenance_path,
        spca_diagnostics=spca_diagnostics,
        deposit_metrics={},
        ranking_mode=validation_extras["ranking_mode"],
        k_pcs_var1=validation_extras["k_pcs_var1"],
        k_pcs_var2=validation_extras["k_pcs_var2"],
        k_pcs_fused=validation_extras["k_pcs_fused"],
    )
    validation_path = write_validation_payload(
        validation_payload,
        output_dir / "validation_topk_results.pkl",
    )
    recovery_plot_path = plot_cumulative_recovery(
        recovery_result,
        output_dir / "cumulative_footprint_recovery_fraction.png",
        deposit_1based=deposit_1based,
        min_cover=float(config["analysis_defaults"]["min_cover"]),
    )
    top_windows_plot_filename = _build_top_windows_plot_filename(summary_variables, n_top_windows)
    top_windows_title = (
        f"{' + '.join(summary_variables)}: Top {n_top_windows} Prediction Windows"
    )
    top_windows_plot_path = plot_top_windows_overlay(
        top_windows_gdf=top_windows_gdf,
        deposits_gdf=validation_deposits_gdf,
        reference_deposit_index=deposit_1based - 1,
        background_layers=_build_top_windows_background_layers(
            config=config,
            raster_data=raster_data,
            deposit_1based=deposit_1based,
        ),
        transform=reference_raster.transform,
        output_path=output_dir / top_windows_plot_filename,
        title=top_windows_title,
        image_cmap=_get_image_colormap(config),
    )
    pc_score_map_path = plot_pc_score_map(
        scores=pca_result.scores,
        window_indices=window_matrix.window_indices_for_mapping,
        window_shape=window_matrix.window_shape,
        transform=reference_raster.transform,
        background=reference_raster.array,
        background_extent=reference_raster.extent,
        deposit_index=window_matrix.deposit_index,
        deposit_polygon=_get_reference_deposit_polygon(template, summary_variables[0]),
        variable_name=variable_name,
        output_path=output_dir / "pc_score_map.png",
    )
    diagnostic_paths = _build_diagnostic_paths(
        config=config,
        deposit_1based=deposit_1based,
        k_pcs=k_pcs,
        output_dir=output_dir,
        variable_name=variable_name,
        summary_variables=summary_variables,
        template=template,
        window_matrix=window_matrix,
        pca_result=pca_result,
        ranking_result=ranking_result,
        valid_ranked_idx=valid_ranked_idx,
        valid_ranked_dists=valid_ranked_dists,
        top_window_indices=top_window_indices,
        image_cmap=_get_image_colormap(config),
    )
    score_pairs_path = plot_score_pairs(
        comparison_space=ranking_result.comparison_space,
        weights=ranking_result.weights,
        deposit_index=window_matrix.deposit_index,
        output_path=output_dir / "score_pairs.png",
        ranked_idx=valid_ranked_idx,
        top_n_to_plot=int(config.get("visualization", {}).get("score_pairs_top_n_to_plot", 3)),
    )
    if score_pairs_path is not None:
        diagnostic_paths["score_pairs"] = score_pairs_path

    return SPCAOutput(
        deposit_1based=deposit_1based,
        k_pcs=k_pcs,
        case_output_dir=output_dir,
        raster_data=raster_data,
        deposit_template=template,
        window_matrix=window_matrix,
        pca_result=pca_result,
        ranking_result=ranking_result,
        recovery_result=recovery_result,
        top_windows_path=top_windows_path,
        validation_path=validation_path,
        recovery_plot_path=recovery_plot_path,
        top_windows_plot_path=top_windows_plot_path,
        pc_score_map_path=pc_score_map_path,
        diagnostic_paths=diagnostic_paths,
        resolved_config_path=resolved_config_path,
        provenance_path=provenance_path,
    )


def _load_config(config_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_source, dict):
        return config_source
    return load_run_config(config_source)


def _build_run_plan(
    config: dict[str, Any],
    *,
    deposit_1based: int | None = None,
    k_pcs: int | None = None,
) -> list[tuple[int, int]]:
    if deposit_1based is not None and k_pcs is not None:
        return [(deposit_1based, k_pcs)]

    deposits = [int(v) for v in config["sweep"]["deposits_1based"]]
    kpcs_list = [int(v) for v in config["sweep"]["kpcs"]]

    if deposit_1based is not None:
        return [(deposit_1based, int(k)) for k in kpcs_list]
    if k_pcs is not None:
        return [(int(dep), k_pcs) for dep in deposits]

    if config["run"]["run_mode"] == "sweep_kpcs":
        return [(int(dep), int(k)) for dep in deposits for k in kpcs_list]

    raise NotImplementedError(
        f"Run mode '{config['run']['run_mode']}' is not supported by this pipeline yet."
    )


def _get_variable_name(config: dict[str, Any]) -> str:
    if config["run"]["analysis_type"] == "Uni":
        return config["run"]["uni_selected_variable"]
    return config["analysis_defaults"]["variable_1"]


def _build_method_tag(config: dict[str, Any]) -> str:
    method = "spca" if config["run"]["method_name"] == "Spatial_PCA" else "raw"
    analysis = str(config["run"]["analysis_type"]).lower()
    if analysis == "multi":
        return f"{method}_{analysis}_{config['run']['multi_ranking_mode']}"
    variable = _get_variable_name(config).lower()
    return f"{method}_{analysis}_{variable}"


def _get_variable_limits(
    config: dict[str, Any],
    variable_name: str,
    deposit_1based: int,
) -> tuple[float | None, float | None]:
    variable_name_upper = variable_name.upper()
    if variable_name_upper in {"MAG", "TMI"}:
        limits = config.get("visualization", {}).get("deposit_limits_tmi", {})
        value = limits.get(str(deposit_1based))
        if isinstance(value, list) and len(value) == 2:
            return float(value[0]), float(value[1])
    variable_2 = config["analysis_defaults"].get("variable_2")
    if variable_2 is not None and variable_name_upper == str(variable_2).upper():
        return (
            float(config["analysis_defaults"]["vmin_var2"]),
            float(config["analysis_defaults"]["vmax_var2"]),
        )
    return None, None


def _get_image_colormap(config: dict[str, Any]) -> str | None:
    return config.get("visualization", {}).get("image_colormap")


def _get_summary_variables(config: dict[str, Any]) -> list[str]:
    if config["run"]["analysis_type"] == "Uni":
        return [_get_variable_name(config)]
    return _get_multivariate_variable_names(config)


def _require_circle_patch(patch_config: Any) -> None:
    if not isinstance(patch_config, dict):
        raise ValueError("Config section 'patch' must be an object when provided.")
    geometry = str(patch_config.get("geometry", patch_config.get("shape", ""))).strip().lower()
    if geometry != "circle":
        raise ValueError("Only patch.geometry='circle' is currently supported.")


def _get_reconstruction_max_k(config: dict[str, Any], k_pcs: int, num_pcs: int) -> int:
    ncols_recon = 5
    target_pcs = int(k_pcs) + int(config["reconstruction"]["extra_pcs"])
    target_rounded = int(np.ceil(target_pcs / ncols_recon) * ncols_recon)
    return min(int(num_pcs), target_rounded)


def _get_deposits_path(config: dict[str, Any]) -> Path:
    target_mode = config["sweep"]["targets_shp_mode"]
    target_paths = config["targets"]["deposits_shp_paths"]
    if target_mode not in target_paths:
        raise ValueError(
            f"targets_shp_mode '{target_mode}' is not defined in targets.deposits_shp_paths."
        )
    return Path(str(target_paths[target_mode])).expanduser()


def _load_case_deposits(
    config: dict[str, Any],
    deposits_path: Path,
    reference_crs: Any,
    *,
    policy_key: str = "deposit_crs_policy",
):
    """Load deposit targets using the configured CRS policy."""

    targets = config.get("targets", {})
    policy = str(targets.get(policy_key, targets.get("deposit_crs_policy", "reproject_to_raster")))
    deposits = load_deposits(deposits_path, target_crs=None)

    if reference_crs is None:
        return deposits

    if policy == "reproject_to_raster":
        return deposits.to_crs(reference_crs)
    if policy == "assume_raster":
        return deposits.set_crs(reference_crs, allow_override=True)

    raise ValueError(
        "targets.deposit_crs_policy must be 'reproject_to_raster' or 'assume_raster'."
    )


def _get_raster_path(config: dict[str, Any], variable_name: str) -> Path:
    return resolve_variable_raster_path(config, variable_name)


def load_variable_raster(config: dict[str, Any], variable_name: str) -> RasterData:
    return load_raster(
        _get_raster_path(config, variable_name),
        force_crs=config["analysis_defaults"]["force_crs"],
        nodata_to_nan=float(config["paths"]["nodata_to_nan"]),
        crop_polygon_path=config["paths"]["polygon_path"],
    )


def load_multivariate_rasters(config: dict[str, Any], variable_names: list[str]) -> dict[str, RasterData]:
    return {var: load_variable_raster(config, var) for var in variable_names}


def build_case_output_dir(
    config: dict[str, Any],
    deposit_1based: int,
    k_pcs: int,
) -> Path:
    base_dir = Path(str(config["resolved"]["output_dir"]))
    run = config["run"]
    analysis = config["analysis_defaults"]
    if run["analysis_type"] == "Uni":
        variable = run["uni_selected_variable"]
    else:
        variable = f"{analysis['variable_1']}_{analysis['variable_2']}"

    case_name = (
        f"Deposit_{deposit_1based}_{run['method_name']}_{run['analysis_type']}_"
        f"{variable}_{analysis['rotation_angle']}_deg_{analysis['min_cover']}_minCov_kpcs_{k_pcs}"
    )
    return base_dir / case_name


def write_resolved_config(config: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    return output_path


def _build_top_windows_plot_filename(summary_variables: list[str], n_top_windows: int) -> str:
    variable_token = "_".join(_filename_token(variable) for variable in summary_variables)
    return f"{variable_token}_Top_{int(n_top_windows)}_Predicted_Windows.png"


def _filename_token(value: Any) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "variable"


def _get_multivariate_variable_names(config: dict[str, Any]) -> list[str]:
    analysis = config["analysis_defaults"]
    return [str(analysis["variable_1"]), str(analysis["variable_2"])]


def _validate_multivariate_rasters(
    rasters: dict[str, RasterData],
    *,
    reference_variable: str,
) -> None:
    reference = rasters[reference_variable]
    reference_shape = np.asarray(reference.array).shape
    for variable_name, raster in rasters.items():
        if raster.crs != reference.crs:
            raise ValueError(
                f"Raster CRS mismatch: '{variable_name}' does not match '{reference_variable}'."
            )
        if np.asarray(raster.array).shape != reference_shape:
            raise ValueError(
                f"Raster shape mismatch: '{variable_name}' has shape {np.asarray(raster.array).shape}, expected {reference_shape}."
            )
        if raster.transform != reference.transform:
            raise ValueError(
                f"Raster transform mismatch: '{variable_name}' does not match '{reference_variable}'."
            )


def _resolve_multivariate_best_kpcs(config: dict[str, Any], deposit_1based: int) -> tuple[int, int]:
    output_dir = Path(str(config["resolved"]["output_dir"]))
    files = config["best_kpcs_files"]
    project_root = Path(str(config["resolved"]["project_root"]))
    var1_path = _resolve_support_csv_path(
        files=files,
        project_root=project_root,
        output_dir=output_dir,
        direct_path_key="var1_path",
        filename_key="var1_filename",
        legacy_filename_key="var1_legacy_filename",
    )
    var2_path = _resolve_support_csv_path(
        files=files,
        project_root=project_root,
        output_dir=output_dir,
        direct_path_key="var2_path",
        filename_key="var2_filename",
        legacy_filename_key="var2_legacy_filename",
    )
    if not var1_path.exists():
        raise FileNotFoundError(f"Missing multivariate var1 best-k CSV: {var1_path}")
    if not var2_path.exists():
        raise FileNotFoundError(f"Missing multivariate var2 best-k CSV: {var2_path}")
    return (
        _read_best_k_for_deposit(var1_path, deposit_1based),
        _read_best_k_for_deposit(var2_path, deposit_1based),
    )


def _read_best_k_for_deposit(csv_path: Path, deposit_1based: int) -> int:
    df = pd.read_csv(csv_path)
    if not {"deposit_1based", "k_pcs"}.issubset(df.columns):
        raise ValueError(f"CSV '{csv_path}' must contain columns: deposit_1based and k_pcs.")
    matches = df.loc[df["deposit_1based"].astype(int) == int(deposit_1based), "k_pcs"]
    if matches.empty:
        raise ValueError(f"CSV '{csv_path}' has no row for deposit {deposit_1based}.")
    return int(matches.iloc[0])


def _resolve_support_csv_path(
    *,
    files: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    direct_path_key: str,
    filename_key: str,
    legacy_filename_key: str,
) -> Path:
    direct_path = files.get(direct_path_key)
    if direct_path:
        candidate = Path(str(direct_path)).expanduser()
        return candidate if candidate.is_absolute() else (project_root / candidate)

    candidate = output_dir / str(files[filename_key])
    if candidate.exists():
        return candidate

    legacy_candidate = output_dir / str(files[legacy_filename_key])
    return legacy_candidate if legacy_candidate.exists() else candidate


def _get_reference_deposit_polygon(
    template: TemplateData | dict[str, TemplateData],
    reference_variable: str,
):
    if isinstance(template, TemplateData):
        return template.polygon
    return template[reference_variable].polygon


def _build_diagnostic_paths(
    *,
    config: dict[str, Any],
    deposit_1based: int,
    k_pcs: int,
    output_dir: Path,
    variable_name: str,
    summary_variables: list[str],
    template: TemplateData | dict[str, TemplateData],
    window_matrix: WindowMatrix,
    pca_result: PCAResult,
    ranking_result: RankingResult,
    valid_ranked_idx: np.ndarray,
    valid_ranked_dists: np.ndarray,
    top_window_indices: np.ndarray,
    image_cmap: str | None,
) -> dict[str, Path]:
    if config["run"]["analysis_type"] == "Uni":
        vmin, vmax = _get_variable_limits(config, variable_name, deposit_1based)
        return {
            "rotated_deposit": plot_rotated_deposit(
                deposit_array=template.array,
                deposit_extent=template.extent,
                deposit_1based=deposit_1based,
                variable_name=variable_name,
                rotation_angle=float(config["analysis_defaults"]["rotation_angle"]),
                output_path=output_dir / "rotated_deposit.png",
                vmin=vmin,
                vmax=vmax,
                image_cmap=image_cmap,
            ),
            "top_similar_windows": plot_top_similar_windows(
                flattened_windows=window_matrix.display_sliding_windows
                if window_matrix.display_sliding_windows is not None
                else window_matrix.combined_sliding_windows,
                ranked_window_rows=valid_ranked_idx,
                ranked_distances=valid_ranked_dists,
                window_ids=top_window_indices[:, 2],
                window_shape=window_matrix.window_shape,
                variable_name=variable_name,
                output_path=output_dir / "top_similar_windows.png",
                n_rows=int(config["analysis_defaults"]["top_windows_plot_n_rows"]),
                n_cols=int(config["analysis_defaults"]["top_windows_plot_n_cols"]),
                vmin=vmin,
                vmax=vmax,
                image_cmap=image_cmap,
                feature_mask=(
                    None if window_matrix.display_sliding_windows is not None else window_matrix.feature_mask
                ),
            ),
            "component_weights": plot_deposit_scores_and_weights(
                scores=pca_result.scores,
                explained_variance_ratio=pca_result.explained_variance_ratio,
                weights=ranking_result.weights,
                deposit_index=window_matrix.deposit_index,
                k_used=ranking_result.k_used,
                output_path=output_dir / "component_weights.png",
            ),
            "loading_maps": plot_loading_maps(
                loadings=pca_result.loadings,
                scores=pca_result.scores,
                weights=ranking_result.weights,
                deposit_index=window_matrix.deposit_index,
                window_shape=window_matrix.window_shape,
                output_path=output_dir / "loading_maps.png",
                max_pcs=min(4, int(ranking_result.k_used)),
                image_cmap=image_cmap,
                feature_mask=window_matrix.feature_mask,
            ),
            "reconstruction_progression": plot_reconstruction_progression(
                scores=pca_result.scores,
                loadings=pca_result.loadings,
                mean=pca_result.mean,
                std_safe=pca_result.std_safe,
                deposit_index=window_matrix.deposit_index,
                deposit_1based=deposit_1based,
                window_shape=window_matrix.window_shape,
                optimal_k=k_pcs,
                output_path=output_dir / "reconstruction_progression.png",
                variable_name=variable_name,
                vmin=vmin,
                vmax=vmax,
                max_k=_get_reconstruction_max_k(config, k_pcs, pca_result.num_pcs),
                image_cmap=image_cmap,
                feature_mask=window_matrix.feature_mask,
            ),
        }

    variable_names = tuple(summary_variables)
    vmin_by_var = {var: _get_variable_limits(config, var, deposit_1based)[0] for var in variable_names}
    vmax_by_var = {var: _get_variable_limits(config, var, deposit_1based)[1] for var in variable_names}
    n_show = int(config["analysis_defaults"]["top_windows_plot_n_rows"]) * int(
        config["analysis_defaults"]["top_windows_plot_n_cols"]
    )
    fusion_details = ranking_result.fusion_details or {}
    diagnostic_paths = {
        "rotated_deposit": plot_multivariate_rotated_deposit(
            deposit_arrays={var: template[var].array for var in variable_names},
            deposit_1based=deposit_1based,
            rotation_angle=float(config["analysis_defaults"]["rotation_angle"]),
            output_path=output_dir / "rotated_deposit.png",
            vmin_by_var=vmin_by_var,
            vmax_by_var=vmax_by_var,
            image_cmap=image_cmap,
        ),
        "top_similar_windows": plot_multivariate_top_similar_windows(
            per_variable_windows=window_matrix.per_variable_display_windows
            if window_matrix.per_variable_display_windows is not None
            else (window_matrix.per_variable_windows or {}),
            ranked_window_rows=valid_ranked_idx,
            ranked_distances=valid_ranked_dists,
            window_ids=top_window_indices[:, 2],
            window_shape=window_matrix.window_shape,
            output_path=output_dir / "top_similar_windows.png",
            n_show=n_show,
            vmin_by_var=vmin_by_var,
            vmax_by_var=vmax_by_var,
            image_cmap=image_cmap,
            feature_mask=(
                None if window_matrix.per_variable_display_windows is not None else window_matrix.feature_mask
            ),
        ),
    }

    if ranking_result.ranking_mode == "two_stage_pca_fusion":
        diagnostic_paths["component_weights"] = plot_deposit_scores_and_weights(
            scores=fusion_details["Zf_space"],
            explained_variance_ratio=fusion_details["explained_variance_ratio_fused"],
            weights=ranking_result.weights,
            deposit_index=window_matrix.deposit_index,
            k_used=ranking_result.k_used,
            output_path=output_dir / "component_weights.png",
            k_display=min(int(ranking_result.k_used), 12),
            recompute_weights_from_scores=True,
            weight_ylim=(0.0, 1.0),
        )
        diagnostic_paths["reconstruction_progression"] = plot_two_stage_multivariate_reconstruction_progression(
            fusion_details=fusion_details,
            window_shape=window_matrix.window_shape,
            variable_names=(variable_names[0], variable_names[1]),
            output_path=output_dir / "reconstruction_progression.png",
            max_k_fused=_get_reconstruction_max_k(config, k_pcs, int(fusion_details["M_fused"])),
            image_cmap=image_cmap,
            vmin_by_var=vmin_by_var,
            vmax_by_var=vmax_by_var,
            feature_mask=window_matrix.feature_mask,
            title=(
                f"Deposit {deposit_1based} two-stage fused reconstruction progression "
                f"(optimal fused k={k_pcs})"
            ),
        )
        return diagnostic_paths

    diagnostic_paths["component_weights"] = plot_deposit_scores_and_weights(
        scores=pca_result.scores,
        explained_variance_ratio=pca_result.explained_variance_ratio,
        weights=ranking_result.weights,
        deposit_index=window_matrix.deposit_index,
        k_used=ranking_result.k_used,
        output_path=output_dir / "component_weights.png",
        k_display=min(int(ranking_result.k_used), 12),
    )
    return diagnostic_paths


def _build_top_windows_background_layers(
    *,
    config: dict[str, Any],
    raster_data: RasterData | dict[str, RasterData],
    deposit_1based: int,
) -> dict[str, dict[str, Any]]:
    if config["run"]["analysis_type"] == "Uni":
        variable_name = _get_variable_name(config)
        raster = raster_data if isinstance(raster_data, RasterData) else raster_data[variable_name]
        vmin, vmax = _get_variable_limits(config, variable_name, deposit_1based)
        return {
            variable_name: {
                "array": raster.array,
                "extent": raster.extent,
                "vmin": vmin,
                "vmax": vmax,
            }
        }

    layers: dict[str, dict[str, Any]] = {}
    for variable_name in _get_summary_variables(config):
        raster = raster_data[variable_name] if isinstance(raster_data, dict) else raster_data
        vmin, vmax = _get_variable_limits(config, variable_name, deposit_1based)
        layers[variable_name] = {
            "array": raster.array,
            "extent": raster.extent,
            "vmin": vmin,
            "vmax": vmax,
        }
    return layers


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run a Spatial PCA workflow from config.")
    parser.add_argument("config_path", help="Path to a Spatial PCA JSON or YAML run config.")
    parser.add_argument(
        "--deposit",
        type=int,
        help="Optional 1-based deposit number to run. Defaults to all deposits in the config.",
    )
    parser.add_argument(
        "--kpcs",
        type=int,
        help="Optional number of PCA components to run. Defaults to config sweep kpcs.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for the resolved run output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results = run_spca_from_config(
        args.config_path,
        deposit_1based=args.deposit,
        k_pcs=args.kpcs,
        output_dir_override=args.output_dir,
    )
    for result in results:
        print(f"Wrote top windows: {result.top_windows_path}")
        print(f"Wrote config: {result.resolved_config_path}")
        print(f"Wrote provenance: {result.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
