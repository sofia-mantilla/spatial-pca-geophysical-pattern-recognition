"""Loading and cleaning helpers for desurveyed Spatial PCA drillhole assays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_pca.validation.drillhole.mineralization import apply_mineralization_columns
from spatial_pca.validation.drillhole.tables import read_xlsx_sheet


SENTINEL_MISSING_VALUE = -99999
DEFAULT_ASSAY_COLUMNS = [
    "hole_id",
    "X",
    "Y",
    "Z",
    "From",
    "To",
    "interval_length",
    "depth_mid_m",
    "Cu_pct",
    "Au_ppm",
    "log1p_Cu_pct",
    "log1p_Au_ppm",
    "mineralized_state",
    "mineralized_label",
    "mineralized",
]


def load_clean_assays(
    path: Path,
    *,
    sheet_name: str = "in",
    hole_id_column: str = "HoleID",
    mineralization_rule: dict | None = None,
) -> pd.DataFrame:
    """Load the desurveyed workbook and return cleaned interval assays."""

    df = read_xlsx_sheet(path, sheet_name=sheet_name)
    required = [hole_id_column, "X", "Y", "Z", "From", "To", "interval_length", "Cu_pct", "Au_ppm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required assay columns in {path}: {missing}")

    df = df.replace(SENTINEL_MISSING_VALUE, np.nan)
    numeric_cols = ["X", "Y", "Z", "From", "To", "interval_length", "Cu_pct", "Au_ppm"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].replace(SENTINEL_MISSING_VALUE, np.nan)

    for grade_col in ("Cu_pct", "Au_ppm"):
        df.loc[df[grade_col] < 0, grade_col] = np.nan

    hole_id = df[hole_id_column].astype(str).str.strip()
    hole_id = hole_id.where(hole_id.ne(""), pd.NA)
    hole_id = hole_id.where(~hole_id.str.lower().eq("nan"), pd.NA)
    df["hole_id"] = hole_id
    df["depth_mid_m"] = (df["From"] + df["To"]) / 2.0
    df["log1p_Cu_pct"] = np.log1p(df["Cu_pct"])
    df["log1p_Au_ppm"] = np.log1p(df["Au_ppm"])

    df = apply_mineralization_columns(df, **(mineralization_rule or {}))
    return df


def filter_modeling_assays(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep assay rows usable as 3D RBF training points."""

    required = ["hole_id", "X", "Y", "depth_mid_m", "Cu_pct", "Au_ppm"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required modeling assay columns: {missing}")

    valid = frame.loc[
        frame["hole_id"].notna()
        & frame["X"].notna()
        & frame["Y"].notna()
        & frame["depth_mid_m"].notna()
        & (frame["depth_mid_m"] >= 0)
        & (frame["Cu_pct"].notna() | frame["Au_ppm"].notna())
    ].copy()
    return valid.reset_index(drop=True)


def select_assay_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, readable subset of columns for generated CSV outputs."""

    columns = [col for col in DEFAULT_ASSAY_COLUMNS if col in frame.columns]
    extras = [col for col in ("OP_AREA", "Lith1", "Alt1", "zm") if col in frame.columns]
    return frame[columns + extras].copy()
