"""Window-overlap validation against RBF-derived mineralization layers."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


CLASS_ORDER = ["TP", "FP", "FN", "TN"]


def build_slice_cell_polygons(
    slice_layer: pd.DataFrame,
    *,
    xy_spacing_m: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """Convert x/y slice-layer cells to square polygons for overlap analysis."""

    required = ["X", "Y", "slice_id", "modeled_footprint", "mineralized_thickness_m", "max_Cu_pct", "max_Au_ppm"]
    missing = [col for col in required if col not in slice_layer.columns]
    if missing:
        raise ValueError(f"Missing required slice-layer columns: {missing}")
    if xy_spacing_m <= 0:
        raise ValueError("xy_spacing_m must be positive.")

    half = xy_spacing_m / 2.0
    geometries = [
        box(float(row.X) - half, float(row.Y) - half, float(row.X) + half, float(row.Y) + half)
        for row in slice_layer.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(slice_layer.copy(), geometry=geometries, crs=crs)


def validate_ranked_windows_against_slice(
    windows: gpd.GeoDataFrame,
    slice_cells: gpd.GeoDataFrame,
    *,
    k: int,
    actual_positive_fraction: float = 0.1,
    rank_col: str = "rank",
    window_id_col: str = "window_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify ranked windows against one RBF depth-slice layer."""

    if windows.empty:
        raise ValueError("windows GeoDataFrame is empty.")
    if slice_cells.empty:
        raise ValueError("slice_cells GeoDataFrame is empty.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if not 0 <= actual_positive_fraction <= 1:
        raise ValueError("actual_positive_fraction must be between 0 and 1.")
    if rank_col not in windows.columns:
        raise ValueError(f"Rank column not found in windows: {rank_col}")

    work_windows = windows.copy()
    if work_windows.crs != slice_cells.crs:
        work_windows = work_windows.to_crs(slice_cells.crs)
    if window_id_col not in work_windows.columns:
        work_windows[window_id_col] = np.arange(1, len(work_windows) + 1, dtype=int)

    mineralized_cells = slice_cells.loc[slice_cells["modeled_footprint"].astype(bool)].copy()
    rows = []
    for window in work_windows.itertuples(index=False):
        geom = window.geometry
        rank = int(getattr(window, rank_col))
        window_area = float(geom.area)
        overlapping = mineralized_cells.loc[mineralized_cells.geometry.intersects(geom)].copy()

        mineralized_area = 0.0
        area_weighted_thickness = 0.0
        max_cu = np.nan
        max_au = np.nan
        if not overlapping.empty:
            intersections = overlapping.geometry.intersection(geom)
            areas = intersections.area.to_numpy(dtype=float)
            mineralized_area = float(areas.sum())
            area_weighted_thickness = float(
                np.sum(areas * overlapping["mineralized_thickness_m"].to_numpy(dtype=float))
            )
            max_cu = float(overlapping["max_Cu_pct"].max())
            max_au = float(overlapping["max_Au_ppm"].max())

        mineralized_fraction = mineralized_area / window_area if window_area else np.nan
        mean_thickness = area_weighted_thickness / mineralized_area if mineralized_area else 0.0
        predicted_positive = rank <= k
        actual_positive = bool(mineralized_fraction >= actual_positive_fraction)
        rows.append(
            {
                "slice_id": str(slice_cells["slice_id"].iloc[0]),
                "k": int(k),
                "window_id": getattr(window, window_id_col),
                "rank": rank,
                "predicted_positive": predicted_positive,
                "mineralized_area_m2": mineralized_area,
                "window_area_m2": window_area,
                "mineralized_fraction": mineralized_fraction,
                "mean_thickness_m": mean_thickness,
                "max_Cu_pct": max_cu,
                "max_Au_ppm": max_au,
                "rbf_supported_positive": actual_positive,
                "class": _confusion_class(predicted_positive, actual_positive),
            }
        )

    table = pd.DataFrame(rows)
    summary = summarize_window_validation(table)
    return table, summary


def summarize_window_validation(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize TP/FP/FN/TN and standard metrics for one slice/K table."""

    if table.empty:
        raise ValueError("Cannot summarize empty window-validation table.")

    counts = {klass: int((table["class"] == klass).sum()) for klass in CLASS_ORDER}
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    false_positive_rate = _safe_divide(fp, fp + tn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return pd.DataFrame(
        [
            {
                "slice_id": str(table["slice_id"].iloc[0]),
                "k": int(table["k"].iloc[0]),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "f1": f1,
            }
        ]
    )


def _confusion_class(predicted_positive: bool, actual_positive: bool) -> str:
    if predicted_positive and actual_positive:
        return "TP"
    if predicted_positive and not actual_positive:
        return "FP"
    if not predicted_positive and actual_positive:
        return "FN"
    return "TN"


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")
