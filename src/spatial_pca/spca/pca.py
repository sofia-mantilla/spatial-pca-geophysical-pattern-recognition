"""PCA fitting utilities for spatial-window matrices.

This module adapts the PCA behavior used by the external SPCA reference
workflow while keeping the implementation local, importable, and free of
plotting side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    """Container for a fitted spatial PCA result."""

    var_name: str
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    eigvals: np.ndarray
    loadings: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    std_safe: np.ndarray
    standardized: np.ndarray
    patch_size: tuple[int, int]
    multi_mode: bool
    n_vars: int
    size_per_var: int
    num_pcs: int
    model: PCA

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return keys matching the old `apply_pca_and_plot` result contract."""

        prefix = self.var_name
        return {
            f"{prefix}_score": self.scores,
            f"{prefix}_explained": self.explained_variance_ratio,
            f"{prefix}_loadings": self.loadings,
            f"{prefix}_eigvals": self.eigvals,
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_multi_mode": self.multi_mode,
            f"{prefix}_n_vars": self.n_vars,
            f"{prefix}_size_per_var": self.size_per_var,
            f"{prefix}_num_pcs": self.num_pcs,
            f"{prefix}_X_stdzd": self.standardized,
            f"{prefix}_std_safe": self.std_safe,
        }


def fit_spca(data: Any, *, var_name: str, patch_size: tuple[int, int]) -> PCAResult:
    """Fit PCA to a spatial-window matrix using the external workflow behavior.

    Parameters
    ----------
    data:
        Matrix with rows as windows/templates and columns as flattened pixels
        or features.
    var_name:
        Name used for compatibility with legacy result keys.
    patch_size:
        Window shape as ``(height, width)``. Used to detect univariate versus
        multivariate feature blocks.
    """

    X = np.asarray(data, dtype=np.float64)
    _validate_matrix(X)
    win_h, win_w = _validate_patch_size(patch_size)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, ddof=1, keepdims=True)
    X_std_safe = X_std.copy()
    X_std_safe[X_std_safe == 0] = 1.0
    X_stdzd = (X - X_mean) / X_std_safe

    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features)
    model = PCA(n_components=n_components)
    scores = model.fit_transform(X_stdzd)
    loadings = model.components_
    eigvals = model.explained_variance_
    explained = model.explained_variance_ratio_

    size_per_var = win_h * win_w
    if n_features % size_per_var == 0 and n_features // size_per_var > 1:
        multi_mode = True
        n_vars = n_features // size_per_var
    else:
        multi_mode = False
        n_vars = 1

    return PCAResult(
        var_name=var_name,
        scores=scores,
        explained_variance_ratio=explained,
        eigvals=eigvals,
        loadings=loadings,
        mean=X_mean,
        std=X_std,
        std_safe=X_std_safe,
        standardized=X_stdzd,
        patch_size=(win_h, win_w),
        multi_mode=multi_mode,
        n_vars=n_vars,
        size_per_var=size_per_var,
        num_pcs=n_components,
        model=model,
    )


def fit_spca_legacy_dict(data: Any, *, var_name: str, patch_size: tuple[int, int]) -> dict[str, Any]:
    """Fit SPCA and return the old dictionary-shaped result."""

    return fit_spca(data, var_name=var_name, patch_size=patch_size).to_legacy_dict()


def _validate_matrix(X: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"PCA input must be a 2D matrix, got shape {X.shape}.")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"PCA input must be non-empty, got shape {X.shape}.")
    if not np.isfinite(X).all():
        bad = ~np.isfinite(X)
        n_bad = int(bad.sum())
        n_bad_rows = int(bad.any(axis=1).sum())
        n_bad_cols = int(bad.any(axis=0).sum())
        raise ValueError(
            "[fit_spca] Non-finite values in PCA input: "
            f"{n_bad} cells | bad rows={n_bad_rows}/{X.shape[0]} | "
            f"bad cols={n_bad_cols}/{X.shape[1]}. "
            "Fix by masking or imputing before PCA."
        )


def _validate_patch_size(patch_size: tuple[int, int]) -> tuple[int, int]:
    if len(patch_size) != 2:
        raise ValueError("patch_size must be a two-item tuple: (height, width).")
    win_h = int(patch_size[0])
    win_w = int(patch_size[1])
    if win_h <= 0 or win_w <= 0:
        raise ValueError("patch_size values must be positive.")
    return win_h, win_w
