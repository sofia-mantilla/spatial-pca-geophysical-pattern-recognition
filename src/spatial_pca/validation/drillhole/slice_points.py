#!/usr/bin/env python3
"""Point-based SPCA validation against borehole-slice validation points."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def load_ranked_windows(shapefile_path: Path, *, k: int | None = None) -> gpd.GeoDataFrame:
    windows = gpd.read_file(shapefile_path)
    if windows.empty:
        raise ValueError(f"Predicted windows shapefile is empty: {shapefile_path}")

    windows = windows.reset_index(drop=True).copy()
    windows["window_rank"] = np.arange(1, len(windows) + 1, dtype=int)
    if k is not None:
        if k <= 0:
            raise ValueError("k must be positive when provided.")
        windows = windows.iloc[: min(k, len(windows))].copy()
    return windows


def load_borehole_slice_points(points_csv: Path, *, slice_id: str | None = None) -> gpd.GeoDataFrame:
    points_df = pd.read_csv(points_csv, low_memory=False)
    required = {"hole_id", "slice_id", "X", "Y", "validation_label"}
    missing = sorted(required - set(points_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in borehole-slice points CSV: {missing}")

    if slice_id is not None:
        points_df = points_df.loc[points_df["slice_id"] == slice_id].copy()
    if points_df.empty:
        raise ValueError(f"No borehole-slice points found for slice_id={slice_id!r}.")

    points = gpd.GeoDataFrame(
        points_df.copy(),
        geometry=gpd.points_from_xy(points_df["X"], points_df["Y"]),
        crs="EPSG:26909",
    )
    points["point_id"] = points["hole_id"].astype(str) + "|" + points["slice_id"].astype(str)
    return points


def _coerce_to_match_crs(frame: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if frame.crs is None:
        return frame.set_crs(target_crs, allow_override=True)
    if frame.crs != target_crs:
        return frame.to_crs(target_crs)
    return frame


def _compute_confusion_class(validation_label: str | None, predicted_positive: bool, eligible: bool) -> str | None:
    if not eligible or validation_label not in {"positive", "negative"}:
        return None
    if validation_label == "positive" and predicted_positive:
        return "TP"
    if validation_label == "positive" and not predicted_positive:
        return "FN"
    if validation_label == "negative" and predicted_positive:
        return "FP"
    return "TN"


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def validate_slice_points_against_ranked_windows(
    *,
    windows_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    evaluation_domain_gdf: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    if windows_gdf.empty:
        raise ValueError("windows_gdf is empty.")
    if points_gdf.empty:
        raise ValueError("points_gdf is empty.")
    if evaluation_domain_gdf.empty:
        raise ValueError("evaluation_domain_gdf is empty.")

    windows = windows_gdf.copy()
    points = points_gdf.copy()
    domain = evaluation_domain_gdf.copy()

    windows = _coerce_to_match_crs(windows, windows.crs)
    points = _coerce_to_match_crs(points, windows.crs)
    domain = _coerce_to_match_crs(domain, windows.crs)

    domain_geom = domain.geometry.unary_union
    points["inside_evaluation_domain"] = points.geometry.apply(domain_geom.covers)
    points["eligible_for_scoring"] = (
        points["inside_evaluation_domain"]
        & points["validation_label"].isin(["positive", "negative"])
    )
    points["exclusion_reason"] = np.where(
        ~points["inside_evaluation_domain"],
        "outside_evaluation_domain",
        np.where(
            points["validation_label"].eq("insufficient_data"),
            "insufficient_data",
            np.where(points["validation_label"].isin(["positive", "negative"]), "", "unknown_label"),
        ),
    )
    points["first_hit_rank"] = pd.array([pd.NA] * len(points), dtype="Int64")
    points["cover_count"] = 0

    eligible_points = points.loc[points["eligible_for_scoring"]].copy()
    positive_ids = set(eligible_points.loc[eligible_points["validation_label"] == "positive", "point_id"])
    negative_ids = set(eligible_points.loc[eligible_points["validation_label"] == "negative", "point_id"])
    hit_positive: set[str] = set()
    hit_negative: set[str] = set()

    recovery_rows: list[dict] = []

    for row in windows.itertuples(index=False):
        rank = int(row.window_rank)
        geom = row.geometry
        covered_mask = eligible_points.geometry.apply(geom.covers)
        covered = eligible_points.loc[covered_mask, ["point_id", "validation_label"]].copy()

        if not covered.empty:
            covered_ids = covered["point_id"].tolist()
            points.loc[points["point_id"].isin(covered_ids), "cover_count"] += 1
            first_hit_mask = points["point_id"].isin(covered_ids) & points["first_hit_rank"].isna()
            points.loc[first_hit_mask, "first_hit_rank"] = rank

        new_positive_ids = [pid for pid in covered.loc[covered["validation_label"] == "positive", "point_id"] if pid not in hit_positive]
        new_negative_ids = [pid for pid in covered.loc[covered["validation_label"] == "negative", "point_id"] if pid not in hit_negative]
        hit_positive.update(new_positive_ids)
        hit_negative.update(new_negative_ids)

        recovery_rows.append(
            {
                "slice_id": str(points["slice_id"].iloc[0]),
                "rank": rank,
                "new_positive_hits": int(len(new_positive_ids)),
                "new_negative_hits": int(len(new_negative_ids)),
                "cum_positive_hits": int(len(hit_positive)),
                "cum_negative_hits": int(len(hit_negative)),
                "cum_positive_recovery_frac": _safe_divide(len(hit_positive), len(positive_ids)),
                "cum_negative_capture_frac": _safe_divide(len(hit_negative), len(negative_ids)),
                "window_id": getattr(row, "id", rank),
            }
        )

    points["inside_predicted_union"] = points["cover_count"] > 0
    points["predicted_label"] = np.where(
        ~points["eligible_for_scoring"],
        pd.NA,
        np.where(points["inside_predicted_union"], "predicted_positive", "predicted_negative"),
    )
    points["confusion_class"] = [
        _compute_confusion_class(vlabel, bool(pred_pos), bool(eligible))
        for vlabel, pred_pos, eligible in zip(
            points["validation_label"],
            points["inside_predicted_union"],
            points["eligible_for_scoring"],
        )
    ]

    eligible_scored = points.loc[points["eligible_for_scoring"]].copy()
    tp = int((eligible_scored["confusion_class"] == "TP").sum())
    fn = int((eligible_scored["confusion_class"] == "FN").sum())
    fp = int((eligible_scored["confusion_class"] == "FP").sum())
    tn = int((eligible_scored["confusion_class"] == "TN").sum())

    confusion_summary = pd.DataFrame(
        [
            {
                "slice_id": str(points["slice_id"].iloc[0]),
                "n_eligible_positive": int(len(positive_ids)),
                "n_eligible_negative": int(len(negative_ids)),
                "n_excluded": int((~points["eligible_for_scoring"]).sum()),
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "recall": _safe_divide(tp, tp + fn),
                "specificity": _safe_divide(tn, tn + fp),
                "precision": _safe_divide(tp, tp + fp),
                "negative_predictive_value": _safe_divide(tn, tn + fn),
                "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
                "balanced_accuracy": np.nanmean([
                    _safe_divide(tp, tp + fn),
                    _safe_divide(tn, tn + fp),
                ]),
                "positive_recovery_final": _safe_divide(tp, len(positive_ids)),
                "negative_capture_final": _safe_divide(fp, len(negative_ids)),
                "k_eval": int(len(windows)),
            }
        ]
    )

    return {
        "point_status": pd.DataFrame(points.drop(columns="geometry")),
        "recovery_by_rank": pd.DataFrame(recovery_rows),
        "confusion_summary": confusion_summary,
    }
