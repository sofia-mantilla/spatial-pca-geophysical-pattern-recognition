"""3D RBF modeling helpers for Cu/Au drillhole mineralization validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spatial_pca.validation.drillhole.mineralization import classify_predicted_grid


@dataclass(frozen=True)
class RBFGridConfig:
    """Regular 3D grid settings for v1 RBF validation."""

    xy_spacing_m: float = 50.0
    depth_spacing_m: float = 25.0
    depth_min_m: float = 0.0
    depth_max_m: float = 600.0
    xy_padding_m: float = 100.0


@dataclass(frozen=True)
class RBFModelConfig:
    """RBF interpolation settings.

    ``xy_scale_m`` and ``depth_scale_m`` put horizontal and vertical coordinates
    on comparable scales before interpolation.
    """

    kernel: str = "thin_plate_spline"
    smoothing: float = 0.0
    neighbors: int | None = 64
    xy_scale_m: float = 1000.0
    depth_scale_m: float = 250.0


@dataclass(frozen=True)
class DrillSupportConfig:
    """Distance thresholds for treating RBF predictions as drill-supported."""

    max_xy_distance_m: float = 500.0
    max_3d_distance_m: float = 350.0


def build_regular_3d_grid(assays: pd.DataFrame, config: RBFGridConfig) -> pd.DataFrame:
    """Create a regular x/y/depth grid over the drilled assay footprint."""

    required = ["X", "Y"]
    missing = [col for col in required if col not in assays.columns]
    if missing:
        raise ValueError(f"Missing required grid extent columns: {missing}")
    if config.xy_spacing_m <= 0 or config.depth_spacing_m <= 0:
        raise ValueError("Grid spacing values must be positive.")
    if config.depth_max_m <= config.depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m.")

    x = assays["X"].dropna().astype(float)
    y = assays["Y"].dropna().astype(float)
    if x.empty or y.empty:
        raise ValueError("Cannot build grid without valid X/Y assay coordinates.")

    x_values = _regular_axis(x.min() - config.xy_padding_m, x.max() + config.xy_padding_m, config.xy_spacing_m)
    y_values = _regular_axis(y.min() - config.xy_padding_m, y.max() + config.xy_padding_m, config.xy_spacing_m)
    depth_values = _regular_axis(config.depth_min_m, config.depth_max_m, config.depth_spacing_m)

    xx, yy, dd = np.meshgrid(x_values, y_values, depth_values, indexing="xy")
    return pd.DataFrame(
        {
            "X": xx.ravel(),
            "Y": yy.ravel(),
            "depth_m": dd.ravel(),
        }
    )


def predict_cu_au_rbf_grid(
    assays: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    model_config: RBFModelConfig,
    support_config: DrillSupportConfig | None = None,
    mineralization_rule: dict | None = None,
) -> pd.DataFrame:
    """Predict Cu/Au grades on a 3D grid and classify modeled mineralization."""

    out = grid.copy()
    out["log1p_Cu_pct_pred"] = _fit_predict_rbf(
        assays,
        grid,
        value_col="log1p_Cu_pct",
        model_config=model_config,
    )
    out["log1p_Au_ppm_pred"] = _fit_predict_rbf(
        assays,
        grid,
        value_col="log1p_Au_ppm",
        model_config=model_config,
    )
    out["Cu_pct_pred"] = np.expm1(out["log1p_Cu_pct_pred"]).clip(lower=0)
    out["Au_ppm_pred"] = np.expm1(out["log1p_Au_ppm_pred"]).clip(lower=0)
    out = add_drill_support_distances(out, assays)
    support_mask = _support_mask(out, support_config)
    raw_mineralized = classify_predicted_grid(
        out["Cu_pct_pred"],
        out["Au_ppm_pred"],
        **(mineralization_rule or {}),
    )
    out["raw_modeled_mineralized"] = raw_mineralized
    out["drill_supported"] = support_mask
    out["modeled_mineralized"] = raw_mineralized & support_mask
    return out


def add_drill_support_distances(grid: pd.DataFrame, assays: pd.DataFrame) -> pd.DataFrame:
    """Add nearest assay support distances in XY and XYZ space."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required to calculate drill-support distances.") from exc

    required_grid = ["X", "Y", "depth_m"]
    required_assays = ["X", "Y", "depth_mid_m"]
    missing_grid = [col for col in required_grid if col not in grid.columns]
    missing_assays = [col for col in required_assays if col not in assays.columns]
    if missing_grid or missing_assays:
        raise ValueError(f"Missing support-distance columns: grid={missing_grid}, assays={missing_assays}")

    out = grid.copy()
    training = assays.loc[
        assays["X"].notna() & assays["Y"].notna() & assays["depth_mid_m"].notna()
    ].copy()
    if training.empty:
        raise ValueError("Cannot calculate support distances without valid assay coordinates.")

    xy_tree = cKDTree(training[["X", "Y"]].to_numpy(dtype=float))
    xyz_tree = cKDTree(training[["X", "Y", "depth_mid_m"]].to_numpy(dtype=float))
    out["nearest_assay_xy_distance_m"] = xy_tree.query(out[["X", "Y"]].to_numpy(dtype=float), k=1)[0]
    out["nearest_assay_3d_distance_m"] = xyz_tree.query(out[["X", "Y", "depth_m"]].to_numpy(dtype=float), k=1)[0]
    return out


def _fit_predict_rbf(
    assays: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    value_col: str,
    model_config: RBFModelConfig,
) -> np.ndarray:
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as exc:
        raise ImportError(
            "scipy is required for RBF interpolation. Install project dependencies "
            "or add scipy explicitly to the environment."
        ) from exc

    required = ["X", "Y", "depth_mid_m", value_col]
    missing = [col for col in required if col not in assays.columns]
    if missing:
        raise ValueError(f"Missing required RBF training columns: {missing}")

    training = assays.loc[
        assays["X"].notna()
        & assays["Y"].notna()
        & assays["depth_mid_m"].notna()
        & assays[value_col].notna()
    ].copy()
    if training.empty:
        raise ValueError(f"No valid training rows for {value_col}.")

    train_xyz = _scale_xyz(
        training["X"].to_numpy(dtype=float),
        training["Y"].to_numpy(dtype=float),
        training["depth_mid_m"].to_numpy(dtype=float),
        model_config=model_config,
    )
    pred_xyz = _scale_xyz(
        grid["X"].to_numpy(dtype=float),
        grid["Y"].to_numpy(dtype=float),
        grid["depth_m"].to_numpy(dtype=float),
        model_config=model_config,
    )

    rbf = RBFInterpolator(
        train_xyz,
        training[value_col].to_numpy(dtype=float),
        kernel=model_config.kernel,
        smoothing=float(model_config.smoothing),
        neighbors=model_config.neighbors,
    )
    return np.asarray(rbf(pred_xyz), dtype=float)


def _scale_xyz(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    *,
    model_config: RBFModelConfig,
) -> np.ndarray:
    if model_config.xy_scale_m <= 0 or model_config.depth_scale_m <= 0:
        raise ValueError("RBF coordinate scale values must be positive.")
    return np.column_stack(
        [
            x / model_config.xy_scale_m,
            y / model_config.xy_scale_m,
            depth / model_config.depth_scale_m,
        ]
    )


def _regular_axis(start: float, stop: float, spacing: float) -> np.ndarray:
    first = np.floor(start / spacing) * spacing
    last = np.ceil(stop / spacing) * spacing
    return np.arange(first, last + spacing * 0.5, spacing, dtype=float)


def _support_mask(grid: pd.DataFrame, support_config: DrillSupportConfig | None) -> np.ndarray:
    if support_config is None:
        return np.ones(len(grid), dtype=bool)
    if support_config.max_xy_distance_m <= 0 or support_config.max_3d_distance_m <= 0:
        raise ValueError("Drill-support distance thresholds must be positive.")
    return (
        (grid["nearest_assay_xy_distance_m"].to_numpy(dtype=float) <= support_config.max_xy_distance_m)
        & (grid["nearest_assay_3d_distance_m"].to_numpy(dtype=float) <= support_config.max_3d_distance_m)
    )
