#!/usr/bin/env python3
"""Table, slice, and GeoPackage helpers for drillhole validation."""

from __future__ import annotations

import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def load_json_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared = []
    for si in root.findall("a:si", XLSX_NS):
        shared.append("".join(t.text or "" for t in si.iterfind(".//a:t", XLSX_NS)))
    return shared


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None:
        inline_node = cell.find("a:is", XLSX_NS)
        if inline_node is None:
            return ""
        return "".join(t.text or "" for t in inline_node.iterfind(".//a:t", XLSX_NS))
    raw_value = value_node.text or ""
    if cell_type == "s":
        try:
            return shared[int(raw_value)]
        except (ValueError, IndexError):
            return raw_value
    return raw_value


def _col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for char in col_letters:
        idx = idx * 26 + ord(char) - 64
    return idx - 1


def read_xlsx_sheet(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        shared = _load_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("p:Relationship", XLSX_NS)
        }
        sheets = workbook.find("a:sheets", XLSX_NS)
        if sheets is None or len(sheets) == 0:
            raise ValueError(f"No sheets found in workbook: {path}")

        sheet_info = None
        for sheet in sheets:
            name = sheet.attrib["name"]
            if sheet_name is None or name == sheet_name:
                sheet_info = (
                    name,
                    rid_to_target[
                        sheet.attrib[
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                        ]
                    ],
                )
                break
        if sheet_info is None:
            available = [sheet.attrib["name"] for sheet in sheets]
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available}")

        _, target = sheet_info
        xml_root = ET.fromstring(zf.read(f"xl/{target}"))
        sheet_rows = xml_root.find("a:sheetData", XLSX_NS)
        if sheet_rows is None:
            raise ValueError(f"Sheet has no row data: {path}")

        rows = sheet_rows.findall("a:row", XLSX_NS)
        if not rows:
            raise ValueError(f"Sheet is empty: {path}")

        header_cells = rows[0].findall("a:c", XLSX_NS)
        header = [_cell_value(cell, shared) for cell in header_cells]

        records: list[list[str]] = []
        for row in rows[1:]:
            record = [""] * len(header)
            for cell in row.findall("a:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                col_letters = re.sub(r"\d", "", ref)
                if not col_letters:
                    continue
                col_idx = _col_letters_to_index(col_letters)
                if col_idx < len(header):
                    record[col_idx] = _cell_value(cell, shared)
            records.append(record)

    return pd.DataFrame(records, columns=header)


def _evaluate_threshold_tristate(
    *,
    primary: pd.Series,
    secondary: pd.Series,
    primary_threshold: float,
    secondary_threshold: float,
    joint_threshold: float | None = None,
) -> pd.Series:
    state = pd.Series(pd.NA, index=primary.index, dtype="Int8")

    primary = pd.to_numeric(primary, errors="coerce")
    secondary = pd.to_numeric(secondary, errors="coerce")

    positive = (primary >= primary_threshold) | (secondary >= secondary_threshold)
    if joint_threshold is not None:
        positive |= (primary >= joint_threshold) & (secondary >= joint_threshold)

    if joint_threshold is None:
        known_zero = primary.notna() & secondary.notna() & ~positive
    else:
        known_zero = primary.notna() & secondary.notna() & ~positive

    state.loc[positive] = 1
    state.loc[known_zero] = 0
    return state


def build_drill_validation_table(
    path: Path,
    *,
    sheet_name: str = "in",
    hole_id_column: str = "HoleID",
) -> pd.DataFrame:
    df = read_xlsx_sheet(path, sheet_name=sheet_name)

    required_columns = [hole_id_column, "X", "Y", "Z", "From", "To", "interval_length", "Cu_pct", "Au_ppm"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path.name}")

    numeric_columns = ["X", "Y", "Z", "From", "To", "interval_length", "Cu_pct", "Au_ppm"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace(-99999, np.nan, inplace=True)

    hole_id = df[hole_id_column].astype(str).str.strip()
    hole_id = hole_id.where(hole_id.ne(""), pd.NA)
    hole_id = hole_id.where(~hole_id.str.lower().eq("nan"), pd.NA)
    df["hole_id"] = hole_id

    df["depth_mid_m"] = (df["From"] + df["To"]) / 2.0
    df["mineralized_state"] = _evaluate_threshold_tristate(
        primary=df["Cu_pct"],
        secondary=df["Au_ppm"],
        primary_threshold=0.1,
        secondary_threshold=0.1,
        joint_threshold=0.08,
    )
    df["high_grade_state"] = _evaluate_threshold_tristate(
        primary=df["Cu_pct"],
        secondary=df["Au_ppm"],
        primary_threshold=1.0,
        secondary_threshold=1.0,
    )

    df["mineralized_label"] = df["mineralized_state"].map(
        {
            0: "not_mineralized",
            1: "mineralized",
        }
    ).astype("string")
    df["mineralized_label"] = df["mineralized_label"].fillna("unknown")

    df["high_grade_label"] = df["high_grade_state"].map(
        {
            0: "not_high_grade",
            1: "high_grade",
        }
    ).astype("string")
    df["high_grade_label"] = df["high_grade_label"].fillna("unknown")

    # Preserve the original boolean fields for compatibility with older scripts.
    df["mineralized"] = df["mineralized_state"].eq(1)
    df["high_grade"] = df["high_grade_state"].eq(1)
    return df


def _projection_unit_vector(azimuth_deg: float) -> tuple[float, float]:
    radians = math.radians(azimuth_deg)
    return math.sin(radians), math.cos(radians)


def build_borehole_order_table(
    intervals: pd.DataFrame,
    *,
    point_method: str = "shallowest",
    azimuth_deg: float = 135.0,
) -> pd.DataFrame:
    required = ["hole_id", "X", "Y", "From", "To"]
    missing = [col for col in required if col not in intervals.columns]
    if missing:
        raise ValueError(f"Missing required interval columns for ordering: {missing}")

    valid = intervals.loc[
        intervals["hole_id"].notna()
        & intervals["X"].notna()
        & intervals["Y"].notna()
        & intervals["From"].notna()
    ].copy()
    if valid.empty:
        raise ValueError("No valid intervals found for borehole ordering.")

    point_method = point_method.lower()
    if point_method == "mean":
        points = (
            valid.groupby("hole_id", as_index=False)
            .agg(
                X=("X", "mean"),
                Y=("Y", "mean"),
                min_from_m=("From", "min"),
                max_to_m=("To", "max"),
                interval_count=("hole_id", "size"),
            )
        )
    elif point_method == "shallowest":
        shallow_idx = valid.groupby("hole_id")["From"].idxmin()
        points = valid.loc[shallow_idx, ["hole_id", "X", "Y", "From", "To"]].copy()
        summary = (
            valid.groupby("hole_id", as_index=False)
            .agg(
                min_from_m=("From", "min"),
                max_to_m=("To", "max"),
                interval_count=("hole_id", "size"),
            )
        )
        points = points.rename(columns={"From": "shallowest_from_m", "To": "shallowest_to_m"})
        points = points.merge(summary, on="hole_id", how="left")
    else:
        raise ValueError("point_method must be 'shallowest' or 'mean'.")

    ux, uy = _projection_unit_vector(azimuth_deg)
    points["section_distance_m"] = points["X"] * ux + points["Y"] * uy
    points = points.sort_values(["section_distance_m", "hole_id"], ascending=[True, True]).reset_index(drop=True)
    points["borehole_rank"] = np.arange(1, len(points) + 1, dtype=int)
    points["section_azimuth_deg"] = azimuth_deg
    points["point_method"] = point_method
    return points


def make_points_gdf(
    df: pd.DataFrame,
    *,
    x_col: str = "X",
    y_col: str = "Y",
    crs: str = "EPSG:26909",
) -> gpd.GeoDataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Point dataframe must contain '{x_col}' and '{y_col}' columns.")
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs,
    )


def _sanitize_gpkg_layer_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return cleaned[:60] if cleaned else "layer"


def write_partitioned_point_gpkg(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    *,
    base_layer_name: str,
    partition_specs: list[tuple[str, ...]] | None = None,
) -> None:
    if gdf.empty:
        raise ValueError(f"Cannot write empty GeoPackage layer set: {gpkg_path}")

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()

    gdf.to_file(gpkg_path, layer=_sanitize_gpkg_layer_name(base_layer_name), driver="GPKG")

    for spec in partition_specs or []:
        missing = [col for col in spec if col not in gdf.columns]
        if missing:
            raise ValueError(f"Cannot partition GeoPackage by missing columns: {missing}")

        subset = gdf.loc[gdf[list(spec)].notna().all(axis=1)].copy()
        if subset.empty:
            continue

        grouped = subset.groupby(list(spec), dropna=True)
        for keys, frame in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            suffix = "__".join(_sanitize_gpkg_layer_name(value) for value in keys)
            layer_name = _sanitize_gpkg_layer_name(f"{base_layer_name}__{suffix}")
            frame.to_file(gpkg_path, layer=layer_name, driver="GPKG")


def filter_intervals_to_max_depth(
    intervals: pd.DataFrame,
    *,
    max_depth_m: float,
    clip_partial_intervals: bool = True,
) -> pd.DataFrame:
    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive.")

    required = ["From", "To"]
    missing = [col for col in required if col not in intervals.columns]
    if missing:
        raise ValueError(f"Missing required interval columns for depth filtering: {missing}")

    subset = intervals.loc[
        intervals["From"].notna()
        & intervals["To"].notna()
        & (intervals["From"] < max_depth_m)
    ].copy()
    if subset.empty:
        return subset

    if clip_partial_intervals:
        subset.loc[:, "To"] = subset["To"].clip(upper=max_depth_m)
        if "interval_length" in subset.columns:
            subset.loc[:, "interval_length"] = subset["To"] - subset["From"]
        if "depth_mid_m" in subset.columns:
            subset.loc[:, "depth_mid_m"] = (subset["From"] + subset["To"]) / 2.0
        subset = subset.loc[subset["To"] > subset["From"]].copy()
    else:
        subset = subset.loc[subset["To"] <= max_depth_m].copy()

    return subset


def summarize_drill_validation(intervals: pd.DataFrame, order: pd.DataFrame) -> dict:
    valid_depth = intervals.loc[intervals["From"].notna() & intervals["To"].notna()]
    return {
        "interval_rows": int(len(intervals)),
        "borehole_count": int(order["hole_id"].nunique()),
        "rows_with_hole_id": int(intervals["hole_id"].notna().sum()),
        "rows_with_xy": int((intervals["X"].notna() & intervals["Y"].notna()).sum()),
        "rows_with_depth": int((intervals["From"].notna() & intervals["To"].notna()).sum()),
        "mineralized_count": int(intervals["mineralized_state"].eq(1).sum()),
        "not_mineralized_count": int(intervals["mineralized_state"].eq(0).sum()),
        "unknown_mineralized_count": int(intervals["mineralized_state"].isna().sum()),
        "max_depth_m": None if valid_depth.empty else float(valid_depth["To"].max()),
    }


def _format_slice_id(slice_from_m: float, slice_to_m: float) -> str:
    left = int(slice_from_m) if float(slice_from_m).is_integer() else str(slice_from_m).replace(".", "p")
    right = int(slice_to_m) if float(slice_to_m).is_integer() else str(slice_to_m).replace(".", "p")
    return f"{left}_{right}m"


def split_intervals_by_slices(
    intervals: pd.DataFrame,
    *,
    slice_edges_m: list[float],
) -> pd.DataFrame:
    if len(slice_edges_m) < 2:
        raise ValueError("slice_edges_m must contain at least two values.")
    if sorted(slice_edges_m) != list(slice_edges_m):
        raise ValueError("slice_edges_m must be sorted in ascending order.")

    required = ["From", "To", "hole_id"]
    missing = [col for col in required if col not in intervals.columns]
    if missing:
        raise ValueError(f"Missing required interval columns for slicing: {missing}")

    base = intervals.loc[
        intervals["hole_id"].notna()
        & intervals["From"].notna()
        & intervals["To"].notna()
        & (intervals["To"] > intervals["From"])
    ].copy()
    if base.empty:
        return base

    segments: list[dict] = []
    for row in base.itertuples(index=False):
        row_dict = row._asdict()
        interval_from = float(row_dict["From"])
        interval_to = float(row_dict["To"])

        for slice_from_m, slice_to_m in zip(slice_edges_m[:-1], slice_edges_m[1:]):
            overlap_from = max(interval_from, float(slice_from_m))
            overlap_to = min(interval_to, float(slice_to_m))
            if overlap_to <= overlap_from:
                continue

            segment = dict(row_dict)
            segment["slice_from_m"] = float(slice_from_m)
            segment["slice_to_m"] = float(slice_to_m)
            segment["slice_id"] = _format_slice_id(float(slice_from_m), float(slice_to_m))
            segment["interval_from_m"] = overlap_from
            segment["interval_to_m"] = overlap_to
            segment["thickness_in_slice_m"] = overlap_to - overlap_from
            segment["depth_mid_m"] = (overlap_from + overlap_to) / 2.0
            segments.append(segment)

    return pd.DataFrame(segments)


def summarize_borehole_slices(
    intervals_by_slice: pd.DataFrame,
    *,
    minimum_classified_thickness_m: float,
    mineralized_fraction_threshold: float,
) -> pd.DataFrame:
    if minimum_classified_thickness_m <= 0:
        raise ValueError("minimum_classified_thickness_m must be positive.")
    if not 0 <= mineralized_fraction_threshold <= 1:
        raise ValueError("mineralized_fraction_threshold must be between 0 and 1.")

    required = ["hole_id", "slice_id", "slice_from_m", "slice_to_m", "thickness_in_slice_m", "mineralized_state"]
    missing = [col for col in required if col not in intervals_by_slice.columns]
    if missing:
        raise ValueError(f"Missing required interval-by-slice columns: {missing}")

    frame = intervals_by_slice.copy()
    frame["mineralized_state"] = pd.array(frame["mineralized_state"], dtype="Int8")
    frame["is_classified"] = frame["mineralized_state"].notna()
    frame["classified_thickness_component_m"] = frame["thickness_in_slice_m"].where(frame["is_classified"], 0.0)
    frame["mineralized_thickness_component_m"] = frame["thickness_in_slice_m"].where(frame["mineralized_state"].eq(1), 0.0)
    frame["not_mineralized_thickness_component_m"] = frame["thickness_in_slice_m"].where(frame["mineralized_state"].eq(0), 0.0)
    frame["unknown_thickness_component_m"] = frame["thickness_in_slice_m"].where(frame["mineralized_state"].isna(), 0.0)

    summary = (
        frame.groupby(["hole_id", "slice_id", "slice_from_m", "slice_to_m"], as_index=False)
        .agg(
            interval_segment_count=("hole_id", "size"),
            sampled_thickness_m=("thickness_in_slice_m", "sum"),
            classified_thickness_m=("classified_thickness_component_m", "sum"),
            mineralized_thickness_m=("mineralized_thickness_component_m", "sum"),
            not_mineralized_thickness_m=("not_mineralized_thickness_component_m", "sum"),
            unknown_thickness_m=("unknown_thickness_component_m", "sum"),
        )
    )

    summary["mineralized_fraction"] = np.where(
        summary["classified_thickness_m"] > 0,
        summary["mineralized_thickness_m"] / summary["classified_thickness_m"],
        np.nan,
    )
    summary["minimum_classified_thickness_m"] = float(minimum_classified_thickness_m)
    summary["mineralized_fraction_threshold"] = float(mineralized_fraction_threshold)
    summary["support_flag"] = np.where(
        summary["classified_thickness_m"] >= minimum_classified_thickness_m,
        "enough_data",
        "insufficient_data",
    )
    summary["validation_label"] = np.where(
        summary["classified_thickness_m"] < minimum_classified_thickness_m,
        "insufficient_data",
        np.where(
            summary["mineralized_fraction"] >= mineralized_fraction_threshold,
            "positive",
            "negative",
        ),
    )
    summary["validation_binary"] = pd.array(
        np.where(
            summary["validation_label"] == "positive",
            1,
            np.where(summary["validation_label"] == "negative", 0, pd.NA),
        ),
        dtype="Int8",
    )
    return summary


def summarize_slice_level(borehole_slice_summary: pd.DataFrame) -> pd.DataFrame:
    required = [
        "slice_id",
        "slice_from_m",
        "slice_to_m",
        "classified_thickness_m",
        "sampled_thickness_m",
        "mineralized_thickness_m",
        "not_mineralized_thickness_m",
        "unknown_thickness_m",
        "mineralized_fraction",
        "validation_label",
    ]
    missing = [col for col in required if col not in borehole_slice_summary.columns]
    if missing:
        raise ValueError(f"Missing required borehole-slice summary columns: {missing}")

    frame = borehole_slice_summary.copy()
    frame["is_positive"] = frame["validation_label"].eq("positive").astype(int)
    frame["is_negative"] = frame["validation_label"].eq("negative").astype(int)
    frame["is_insufficient"] = frame["validation_label"].eq("insufficient_data").astype(int)
    frame["supported_fraction_component"] = frame["mineralized_fraction"].where(
        frame["validation_label"].isin(["positive", "negative"]),
        np.nan,
    )

    summary = (
        frame.groupby(["slice_id", "slice_from_m", "slice_to_m"], as_index=False)
        .agg(
            borehole_slice_rows=("slice_id", "size"),
            positive_count=("is_positive", "sum"),
            negative_count=("is_negative", "sum"),
            insufficient_data_count=("is_insufficient", "sum"),
            total_sampled_thickness_m=("sampled_thickness_m", "sum"),
            total_classified_thickness_m=("classified_thickness_m", "sum"),
            total_mineralized_thickness_m=("mineralized_thickness_m", "sum"),
            total_not_mineralized_thickness_m=("not_mineralized_thickness_m", "sum"),
            total_unknown_thickness_m=("unknown_thickness_m", "sum"),
            mean_mineralized_fraction_supported=("supported_fraction_component", "mean"),
            median_mineralized_fraction_supported=("supported_fraction_component", "median"),
        )
    )
    summary["supported_count"] = summary["positive_count"] + summary["negative_count"]
    summary["positive_rate_supported"] = np.where(
        summary["supported_count"] > 0,
        summary["positive_count"] / summary["supported_count"],
        np.nan,
    )
    return summary
