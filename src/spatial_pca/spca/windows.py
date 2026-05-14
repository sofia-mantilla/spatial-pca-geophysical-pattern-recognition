"""Sliding-window construction and window-index mapping utilities.

This module adapts the univariate sliding-window behavior used by the
external SPCA reference workflow. CRS and map transforms are intentionally
handled outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.util.shape import view_as_windows


@dataclass(frozen=True)
class WindowMatrix:
    """Container for SPCA window matrix inputs."""

    data_for_pca: np.ndarray
    deposit_index: int
    pca_var_name: str
    window_shape: tuple[int, int]
    combined_sliding_windows: np.ndarray
    combined_deposit: np.ndarray
    window_indices_for_mapping: np.ndarray
    variable_names: tuple[str, ...] = ()
    per_variable_windows: dict[str, np.ndarray] | None = None
    window_indices_by_var: dict[str, np.ndarray] | None = None
    feature_mask: np.ndarray | None = None
    display_sliding_windows: np.ndarray | None = None
    display_deposit: np.ndarray | None = None
    per_variable_display_windows: dict[str, np.ndarray] | None = None
    patch_geometry_type: str = "rectangle"
    patch_radius_m: float | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return keys matching the old window-builder result contract."""

        if self.per_variable_windows is not None and self.window_indices_by_var is not None:
            variables_sliding_windows = {
                var: {
                    "flattened_windows": self.per_variable_windows[var],
                    "window_indices": self.window_indices_by_var[var],
                }
                for var in self.variable_names
            }
        else:
            variables_sliding_windows = {
                self.pca_var_name: {
                    "flattened_windows": self.combined_sliding_windows,
                    "window_indices": self.window_indices_for_mapping,
                }
            }
        return {
            "variables_sliding_windows": variables_sliding_windows,
            "combined_sliding_windows": self.combined_sliding_windows,
            "combined_deposit": self.combined_deposit,
            "data_for_pca": self.data_for_pca,
            "deposit_index": self.deposit_index,
            "pca_var_name": self.pca_var_name,
            "window_indices_for_mapping": self.window_indices_for_mapping,
            "window_indices_by_var": self.window_indices_by_var,
        }


def pad_raster(data: Any, window_shape: tuple[int, int], stride_y: int, stride_x: int) -> np.ndarray:
    """Pad a 2D raster with NaNs so sliding windows cover the edge consistently."""

    arr = _validate_2d_array(data, name="data")
    win_h, win_w = _validate_window_shape(window_shape)
    stride_y = _validate_positive_int(stride_y, "stride_y")
    stride_x = _validate_positive_int(stride_x, "stride_x")

    pad_y = (stride_y - (arr.shape[0] - win_h) % stride_y) % stride_y
    pad_x = (stride_x - (arr.shape[1] - win_w) % stride_x) % stride_x
    extra_pad_y = max(0, win_h - (arr.shape[0] + pad_y))
    extra_pad_x = max(0, win_w - (arr.shape[1] + pad_x))
    return np.pad(
        arr,
        ((0, pad_y + extra_pad_y), (0, pad_x + extra_pad_x)),
        constant_values=np.nan,
    )


def get_padded_and_windows(
    data: Any,
    window_shape: tuple[int, int],
    stride_y: int,
    stride_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded raster and a 4D sliding-window view."""

    padded = pad_raster(data, window_shape, stride_y, stride_x)
    windows = view_as_windows(padded, window_shape, step=(stride_y, stride_x))
    return padded, windows


def build_univariate_window_matrix(
    *,
    raster: Any,
    deposit_template: Any,
    variable_name: str,
    stride_y: int,
    stride_x: int,
) -> WindowMatrix:
    """Build flattened raster windows and append the deposit template row."""

    raster_arr = _validate_2d_array(raster, name="raster")
    deposit_arr = _validate_2d_array(deposit_template, name="deposit_template")
    window_shape = _validate_window_shape(deposit_arr.shape)
    stride_y = _validate_positive_int(stride_y, "stride_y")
    stride_x = _validate_positive_int(stride_x, "stride_x")

    _, windows = get_padded_and_windows(raster_arr, window_shape, stride_y, stride_x)
    flattened_windows: list[np.ndarray] = []
    window_indices: list[list[int]] = []
    original_window_number = 0

    for row_idx, row in enumerate(windows):
        for col_idx, window in enumerate(row):
            if not np.isnan(window).any():
                flattened_windows.append(window.flatten())
                top_left_y = row_idx * stride_y
                top_left_x = col_idx * stride_x
                window_indices.append([top_left_y, top_left_x, original_window_number])
            original_window_number += 1

    if not flattened_windows:
        raise ValueError("No finite sliding windows were found for the raster and template shape.")

    combined_sliding_windows = np.asarray(flattened_windows, dtype=float)
    window_indices_for_mapping = np.asarray(window_indices, dtype=int)
    combined_deposit = deposit_arr.flatten().astype(float)
    if np.isnan(combined_deposit).any():
        n_missing = int(np.isnan(combined_deposit).sum())
        raise ValueError(
            f"Deposit template contains {n_missing} NaNs. Mask or impute before PCA."
        )

    data_for_pca = np.vstack([combined_sliding_windows, combined_deposit])
    deposit_index = combined_sliding_windows.shape[0]

    return WindowMatrix(
        data_for_pca=data_for_pca,
        deposit_index=deposit_index,
        pca_var_name=variable_name,
        window_shape=window_shape,
        combined_sliding_windows=combined_sliding_windows,
        combined_deposit=combined_deposit,
        window_indices_for_mapping=window_indices_for_mapping,
        variable_names=(variable_name,),
        per_variable_windows={variable_name: combined_sliding_windows},
        window_indices_by_var={variable_name: window_indices_for_mapping},
    )


def build_multivariate_window_matrix(
    *,
    rasters: dict[str, Any],
    deposit_templates: dict[str, Any],
    variable_names: list[str] | tuple[str, ...],
    stride_y: int,
    stride_x: int,
) -> WindowMatrix:
    """Build a combined multivariate window matrix using a shared NaN mask."""

    ordered_vars = tuple(str(v) for v in variable_names)
    if len(ordered_vars) < 2:
        raise ValueError("Multivariate window building requires at least two variables.")

    first_var = ordered_vars[0]
    if first_var not in rasters or first_var not in deposit_templates:
        raise ValueError(f"Missing raster or template for variable '{first_var}'.")

    first_template = _validate_2d_array(deposit_templates[first_var], name=f"deposit_templates['{first_var}']")
    window_shape = _validate_window_shape(first_template.shape)
    stride_y = _validate_positive_int(stride_y, "stride_y")
    stride_x = _validate_positive_int(stride_x, "stride_x")

    shared_mask = None
    for var in ordered_vars:
        if var not in rasters:
            raise ValueError(f"Missing raster for variable '{var}'.")
        if var not in deposit_templates:
            raise ValueError(f"Missing deposit template for variable '{var}'.")
        template = _validate_2d_array(deposit_templates[var], name=f"deposit_templates['{var}']")
        if template.shape != window_shape:
            raise ValueError(
                f"Deposit template for variable '{var}' has shape {template.shape}, expected {window_shape}."
            )
        padded = pad_raster(rasters[var], window_shape, stride_y, stride_x)
        var_mask = np.isnan(padded)
        shared_mask = var_mask if shared_mask is None else (shared_mask | var_mask)

    per_variable_windows: dict[str, np.ndarray] = {}
    window_indices_by_var: dict[str, np.ndarray] = {}
    common_ids: set[int] | None = None

    for var in ordered_vars:
        raster_arr = _validate_2d_array(rasters[var], name=f"rasters['{var}']")
        padded = pad_raster(raster_arr, window_shape, stride_y, stride_x).copy()
        padded[shared_mask] = np.nan
        windows = view_as_windows(padded, window_shape, step=(stride_y, stride_x))

        flattened_windows: list[np.ndarray] = []
        window_indices: list[list[int]] = []
        original_window_number = 0
        for row_idx, row in enumerate(windows):
            for col_idx, window in enumerate(row):
                if not np.isnan(window).any():
                    flattened_windows.append(window.flatten())
                    top_left_y = row_idx * stride_y
                    top_left_x = col_idx * stride_x
                    window_indices.append([top_left_y, top_left_x, original_window_number])
                original_window_number += 1

        if not flattened_windows:
            raise ValueError(f"No finite sliding windows were found for variable '{var}'.")

        idx_arr = np.asarray(window_indices, dtype=int)
        win_arr = np.asarray(flattened_windows, dtype=float)
        per_variable_windows[var] = win_arr
        window_indices_by_var[var] = idx_arr

        var_ids = set(idx_arr[:, 2].tolist())
        common_ids = var_ids if common_ids is None else common_ids.intersection(var_ids)

    common_id_values = np.asarray(sorted(common_ids or []), dtype=int)
    if common_id_values.size == 0:
        raise ValueError("No common finite sliding windows were found across multivariate inputs.")

    filtered_windows_list: list[np.ndarray] = []
    filtered_indices_ref: np.ndarray | None = None
    filtered_windows_by_var: dict[str, np.ndarray] = {}
    filtered_indices_by_var: dict[str, np.ndarray] = {}

    for var in ordered_vars:
        idx_arr = window_indices_by_var[var]
        fw = per_variable_windows[var]
        mask_common = np.isin(idx_arr[:, 2], common_id_values)
        fw_filtered = fw[mask_common]
        idx_filtered = idx_arr[mask_common]
        filtered_windows_by_var[var] = fw_filtered
        filtered_indices_by_var[var] = idx_filtered
        filtered_windows_list.append(fw_filtered)
        if filtered_indices_ref is None:
            filtered_indices_ref = idx_filtered

    combined_sliding_windows = np.hstack(filtered_windows_list)
    combined_deposit = np.concatenate(
        [
            _validate_2d_array(
                deposit_templates[var],
                name=f"deposit_templates['{var}']",
            ).flatten().astype(float)
            for var in ordered_vars
        ]
    )
    if np.isnan(combined_deposit).any():
        n_missing = int(np.isnan(combined_deposit).sum())
        raise ValueError(
            f"Combined multivariate deposit template contains {n_missing} NaNs. Mask or impute before PCA."
        )

    data_for_pca = np.vstack([combined_sliding_windows, combined_deposit])
    deposit_index = combined_sliding_windows.shape[0]

    return WindowMatrix(
        data_for_pca=data_for_pca,
        deposit_index=deposit_index,
        pca_var_name="Combined",
        window_shape=window_shape,
        combined_sliding_windows=combined_sliding_windows,
        combined_deposit=combined_deposit,
        window_indices_for_mapping=filtered_indices_ref,
        variable_names=ordered_vars,
        per_variable_windows=filtered_windows_by_var,
        window_indices_by_var=filtered_indices_by_var,
    )


def build_univariate_circle_window_matrix(
    *,
    raster: Any,
    circle_template: Any,
    variable_name: str,
    stride_y: int,
    stride_x: int,
    radius_m: float,
    feature_mask: Any | None = None,
) -> WindowMatrix:
    """Build circle-masked univariate windows using a bbox-plus-mask template."""

    raster_arr = _validate_2d_array(raster, name="raster")
    template_arr = _validate_2d_array(circle_template, name="circle_template")
    mask = _resolve_feature_mask(feature_mask, template_arr, name="feature_mask")
    if not np.any(mask):
        raise ValueError("circle_template has no finite in-circle pixels.")
    if not np.isfinite(template_arr[mask]).all():
        raise ValueError("circle_template contains non-finite values inside the circle mask.")

    window_shape = _validate_window_shape(template_arr.shape)
    stride_y = _validate_positive_int(stride_y, "stride_y")
    stride_x = _validate_positive_int(stride_x, "stride_x")

    _, windows = get_padded_and_windows(raster_arr, window_shape, stride_y, stride_x)
    feature_windows: list[np.ndarray] = []
    display_windows: list[np.ndarray] = []
    window_indices: list[list[int]] = []
    original_window_number = 0
    half_h = window_shape[0] // 2
    half_w = window_shape[1] // 2

    for row_idx, row in enumerate(windows):
        for col_idx, window in enumerate(row):
            selected = np.asarray(window, dtype=float)[mask]
            if np.isfinite(selected).all():
                top_left_y = row_idx * stride_y
                top_left_x = col_idx * stride_x
                center_row = top_left_y + half_h
                center_col = top_left_x + half_w
                feature_windows.append(selected)
                display_windows.append(np.where(mask, window, np.nan).flatten())
                window_indices.append(
                    [top_left_y, top_left_x, original_window_number, center_row, center_col]
                )
            original_window_number += 1

    if not feature_windows:
        raise ValueError("No finite circle-masked windows were found for the raster and template shape.")

    combined_sliding_windows = np.asarray(feature_windows, dtype=float)
    display_sliding_windows = np.asarray(display_windows, dtype=float)
    window_indices_for_mapping = np.asarray(window_indices, dtype=int)
    combined_deposit = template_arr[mask].astype(float)
    display_deposit = np.where(mask, template_arr, np.nan).flatten().astype(float)
    data_for_pca = np.vstack([combined_sliding_windows, combined_deposit])
    _require_finite_matrix(data_for_pca, name="circle data_for_pca")
    deposit_index = combined_sliding_windows.shape[0]

    return WindowMatrix(
        data_for_pca=data_for_pca,
        deposit_index=deposit_index,
        pca_var_name=variable_name,
        window_shape=window_shape,
        combined_sliding_windows=combined_sliding_windows,
        combined_deposit=combined_deposit,
        window_indices_for_mapping=window_indices_for_mapping,
        variable_names=(variable_name,),
        per_variable_windows={variable_name: combined_sliding_windows},
        window_indices_by_var={variable_name: window_indices_for_mapping},
        feature_mask=mask,
        display_sliding_windows=display_sliding_windows,
        display_deposit=display_deposit,
        per_variable_display_windows={variable_name: display_sliding_windows},
        patch_geometry_type="circle",
        patch_radius_m=float(radius_m),
    )


def build_multivariate_circle_window_matrix(
    *,
    rasters: dict[str, Any],
    circle_templates: dict[str, Any],
    variable_names: list[str] | tuple[str, ...],
    stride_y: int,
    stride_x: int,
    radius_m: float,
    feature_mask: Any | None = None,
) -> WindowMatrix:
    """Build a combined multivariate window matrix using a shared circle mask."""

    ordered_vars = tuple(str(v) for v in variable_names)
    if len(ordered_vars) < 2:
        raise ValueError("Multivariate circle window building requires at least two variables.")

    first_template = _validate_2d_array(circle_templates[ordered_vars[0]], name="circle_template")
    window_shape = _validate_window_shape(first_template.shape)
    mask = _resolve_feature_mask(feature_mask, first_template, name="feature_mask")
    if not np.any(mask):
        raise ValueError("Circle template has no finite in-circle pixels.")
    if not np.isfinite(first_template[mask]).all():
        raise ValueError("Circle template contains non-finite values inside the circle mask.")

    for var in ordered_vars[1:]:
        template = _validate_2d_array(circle_templates[var], name=f"circle_templates['{var}']")
        if template.shape != window_shape:
            raise ValueError(
                f"Circle template for variable '{var}' has shape {template.shape}, expected {window_shape}."
            )
        if feature_mask is None and not np.array_equal(np.isfinite(template), mask):
            raise ValueError(f"Circle feature mask mismatch for variable '{var}'.")
        if not np.isfinite(template[mask]).all():
            raise ValueError(f"Circle template for variable '{var}' contains non-finite in-circle values.")

    stride_y = _validate_positive_int(stride_y, "stride_y")
    stride_x = _validate_positive_int(stride_x, "stride_x")
    half_h = window_shape[0] // 2
    half_w = window_shape[1] // 2

    per_variable_windows: dict[str, np.ndarray] = {}
    per_variable_display_windows: dict[str, np.ndarray] = {}
    window_indices_by_var: dict[str, np.ndarray] = {}
    common_ids: set[int] | None = None

    for var in ordered_vars:
        raster_arr = _validate_2d_array(rasters[var], name=f"rasters['{var}']")
        _, windows = get_padded_and_windows(raster_arr, window_shape, stride_y, stride_x)

        feature_windows: list[np.ndarray] = []
        display_windows: list[np.ndarray] = []
        window_indices: list[list[int]] = []
        original_window_number = 0

        for row_idx, row in enumerate(windows):
            for col_idx, window in enumerate(row):
                selected = np.asarray(window, dtype=float)[mask]
                if np.isfinite(selected).all():
                    top_left_y = row_idx * stride_y
                    top_left_x = col_idx * stride_x
                    center_row = top_left_y + half_h
                    center_col = top_left_x + half_w
                    feature_windows.append(selected)
                    display_windows.append(np.where(mask, window, np.nan).flatten())
                    window_indices.append(
                        [top_left_y, top_left_x, original_window_number, center_row, center_col]
                    )
                original_window_number += 1

        if not feature_windows:
            raise ValueError(f"No finite circle-masked windows were found for variable '{var}'.")

        idx_arr = np.asarray(window_indices, dtype=int)
        win_arr = np.asarray(feature_windows, dtype=float)
        display_arr = np.asarray(display_windows, dtype=float)
        per_variable_windows[var] = win_arr
        per_variable_display_windows[var] = display_arr
        window_indices_by_var[var] = idx_arr
        var_ids = set(idx_arr[:, 2].tolist())
        common_ids = var_ids if common_ids is None else common_ids.intersection(var_ids)

    common_id_values = np.asarray(sorted(common_ids or []), dtype=int)
    if common_id_values.size == 0:
        raise ValueError("No common finite circle-masked windows were found across multivariate inputs.")

    filtered_windows_list: list[np.ndarray] = []
    filtered_indices_ref: np.ndarray | None = None
    filtered_windows_by_var: dict[str, np.ndarray] = {}
    filtered_display_by_var: dict[str, np.ndarray] = {}

    for var in ordered_vars:
        idx_arr = window_indices_by_var[var]
        mask_common = np.isin(idx_arr[:, 2], common_id_values)
        filtered_windows_by_var[var] = per_variable_windows[var][mask_common]
        filtered_display_by_var[var] = per_variable_display_windows[var][mask_common]
        filtered_windows_list.append(filtered_windows_by_var[var])
        if filtered_indices_ref is None:
            filtered_indices_ref = idx_arr[mask_common]

    combined_sliding_windows = np.hstack(filtered_windows_list)
    display_sliding_windows = np.hstack([filtered_display_by_var[var] for var in ordered_vars])
    combined_deposit = np.concatenate(
        [np.asarray(circle_templates[var], dtype=float)[mask] for var in ordered_vars]
    )
    display_deposit = np.concatenate(
        [np.where(mask, np.asarray(circle_templates[var], dtype=float), np.nan).flatten() for var in ordered_vars]
    )
    data_for_pca = np.vstack([combined_sliding_windows, combined_deposit])
    _require_finite_matrix(data_for_pca, name="multivariate circle data_for_pca")
    deposit_index = combined_sliding_windows.shape[0]

    return WindowMatrix(
        data_for_pca=data_for_pca,
        deposit_index=deposit_index,
        pca_var_name="Combined",
        window_shape=window_shape,
        combined_sliding_windows=combined_sliding_windows,
        combined_deposit=combined_deposit,
        window_indices_for_mapping=filtered_indices_ref,
        variable_names=ordered_vars,
        per_variable_windows=filtered_windows_by_var,
        window_indices_by_var={var: window_indices_by_var[var][np.isin(window_indices_by_var[var][:, 2], common_id_values)] for var in ordered_vars},
        feature_mask=mask,
        display_sliding_windows=display_sliding_windows,
        display_deposit=display_deposit,
        per_variable_display_windows=filtered_display_by_var,
        patch_geometry_type="circle",
        patch_radius_m=float(radius_m),
    )


def build_univariate_window_matrix_legacy_dict(
    *,
    raster: Any,
    deposit_template: Any,
    variable_name: str,
    stride_y: int,
    stride_x: int,
) -> dict[str, Any]:
    """Build the univariate window matrix and return old dictionary-shaped output."""

    return build_univariate_window_matrix(
        raster=raster,
        deposit_template=deposit_template,
        variable_name=variable_name,
        stride_y=stride_y,
        stride_x=stride_x,
    ).to_legacy_dict()


def _validate_2d_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}.")
    return arr


def _resolve_feature_mask(mask: Any | None, template_arr: np.ndarray, *, name: str) -> np.ndarray:
    if mask is None:
        return np.isfinite(template_arr)
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != template_arr.shape:
        raise ValueError(f"{name} shape {mask_arr.shape} does not match template shape {template_arr.shape}.")
    return mask_arr


def _require_finite_matrix(matrix: np.ndarray, *, name: str) -> None:
    if np.isfinite(matrix).all():
        return
    n_bad = int((~np.isfinite(matrix)).sum())
    raise ValueError(f"{name} contains {n_bad} non-finite values.")


def _validate_window_shape(window_shape: tuple[int, int]) -> tuple[int, int]:
    if len(window_shape) != 2:
        raise ValueError("window_shape must be a two-item tuple: (height, width).")
    win_h = _validate_positive_int(window_shape[0], "window_shape[0]")
    win_w = _validate_positive_int(window_shape[1], "window_shape[1]")
    return win_h, win_w


def _validate_positive_int(value: Any, name: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int_value
