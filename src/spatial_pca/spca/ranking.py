"""Window-ranking utilities for SPCA and baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class RankingResult:
    """Container for ranked SPCA window distances."""

    ranked_idx: np.ndarray
    ranked_dists: np.ndarray
    weights: np.ndarray
    comparison_space: np.ndarray
    k_used: int
    use_whitening: bool
    use_weights: bool
    return_squared: bool
    ranking_mode: str = "shared_weighted_l2"
    fusion_details: dict[str, Any] | None = None

    def to_legacy_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the old tuple contract: ranked_idx, ranked_dists, weights, Z_space."""

        return self.ranked_idx, self.ranked_dists, self.weights, self.comparison_space

    def to_pipeline_dict(self) -> dict[str, Any]:
        """Return keys matching the old ranking pipeline output for shared SPCA mode."""

        return {
            "ranked_idx": self.ranked_idx,
            "ranked_dists": self.ranked_dists,
            "weights": self.weights,
            "Z_space": self.comparison_space,
            "fusion_details": self.fusion_details or {},
            "use_separate_multi_fusion": False,
            "use_two_stage_multi_fusion": self.ranking_mode == "two_stage_pca_fusion",
        }


def rank_spca_windows(
    *,
    scores: Any,
    eigvals: Any,
    deposit_index: int,
    k_pcs: int | None = None,
    top_k: int | None = None,
) -> RankingResult:
    """Rank windows using the default shared SPCA behavior from the reference workflow."""

    return rank_by_weighted_l2(
        scores=scores,
        eigvals=eigvals,
        deposit_index=deposit_index,
        k_pcs=k_pcs,
        top_k=top_k,
        return_squared=False,
        use_whitening=False,
        use_weights=True,
    )


def rank_by_weighted_l2(
    *,
    scores: Any,
    eigvals: Any,
    deposit_index: int,
    k_pcs: int | None = None,
    top_k: int | None = None,
    eps: float = 1e-12,
    return_squared: bool = False,
    use_whitening: bool = True,
    use_weights: bool = True,
) -> RankingResult:
    """Compute weighted L2 distances in raw or whitened PCA score space.

    This is a cleaned local version of the external
    `rank_by_weighted_L2_prewhiten_weights` function.
    """

    Z = np.asarray(scores, dtype=float)
    lam_all = np.asarray(eigvals, dtype=float)
    _validate_scores_and_eigvals(Z, lam_all, deposit_index)

    n_samples, n_components = Z.shape
    if k_pcs is None:
        k_used = n_components
    else:
        k_used = max(1, min(int(k_pcs), n_components))

    lam = lam_all[:k_used].copy()
    lam[lam < eps] = eps

    if use_whitening:
        sqrt_lam = np.sqrt(lam)
        comparison_space = Z[:, :k_used] / sqrt_lam
        deposit_space = Z[deposit_index, :k_used] / sqrt_lam
    else:
        comparison_space = Z[:, :k_used].copy()
        deposit_space = Z[deposit_index, :k_used].copy()

    deposit_unwhitened = Z[deposit_index, :k_used]
    raw_weights = deposit_unwhitened**2
    weight_sum = raw_weights.sum()
    if weight_sum > 0:
        weights = raw_weights / weight_sum
    else:
        weights = np.ones(k_used, dtype=float) / float(k_used)

    diff = comparison_space - deposit_space
    if use_weights:
        dists_sq = (diff**2) @ weights
    else:
        dists_sq = np.sum(diff**2, axis=1)

    dists = dists_sq if return_squared else np.sqrt(dists_sq)
    order = np.argsort(dists)
    ranked_idx = order
    ranked_dists = dists[order]

    if top_k is not None:
        top_k = max(1, min(int(top_k), ranked_idx.shape[0]))
        ranked_idx = ranked_idx[:top_k]
        ranked_dists = ranked_dists[:top_k]

    return RankingResult(
        ranked_idx=ranked_idx,
        ranked_dists=ranked_dists,
        weights=weights,
        comparison_space=comparison_space,
        k_used=k_used,
        use_whitening=use_whitening,
        use_weights=use_weights,
        return_squared=return_squared,
        ranking_mode="shared_weighted_l2",
        fusion_details={},
    )


def rank_multi_two_stage_pca_fusion(
    *,
    X_multi: Any,
    deposit_index: int,
    window_shape: tuple[int, int],
    n_features_var1: int | None = None,
    k_pcs_var1: int,
    k_pcs_var2: int,
    k_pcs_fused: int | None = None,
    top_k: int | None = None,
    eps: float = 1e-12,
    return_squared: bool = False,
    use_whitening: bool = False,
    use_weights: bool = True,
    standardize_fused_input: bool = True,
) -> RankingResult:
    """Rank multivariate windows using the paper two-stage fused PCA method."""

    X = np.asarray(X_multi, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_multi must be 2D.")
    if not np.isfinite(X).all():
        raise ValueError("X_multi contains non-finite values.")
    if not (0 <= int(deposit_index) < X.shape[0]):
        raise IndexError("deposit_index is out of bounds for X_multi.")

    win_h = int(window_shape[0])
    win_w = int(window_shape[1])
    n_pix = int(n_features_var1) if n_features_var1 is not None else (win_h * win_w)
    if X.shape[1] < 2 * n_pix:
        raise ValueError(
            f"Expected at least {2 * n_pix} multivariate features for two-stage fusion, got {X.shape[1]}."
        )

    X1 = X[:, :n_pix]
    X2 = X[:, n_pix : 2 * n_pix]
    Z1k, K1_eff, pca1, X1_mean, X1_std_safe = _fit_block_scores(X1, k_pcs_var1)
    Z2k, K2_eff, pca2, X2_mean, X2_std_safe = _fit_block_scores(X2, k_pcs_var2)

    F = np.hstack([Z1k, Z2k])
    if standardize_fused_input:
        F_mu = F.mean(axis=0)
        F_sd = F.std(axis=0, ddof=0)
        F_sd_safe = np.where(F_sd == 0, 1.0, F_sd)
        F_in = (F - F_mu) / F_sd_safe
    else:
        F_mu = None
        F_sd_safe = None
        F_in = F

    M_fused = min(F_in.shape[0], F_in.shape[1])
    pca_fused = PCA(n_components=M_fused)
    Zf = pca_fused.fit_transform(F_in)
    eig_fused = np.asarray(pca_fused.explained_variance_, dtype=float)

    Kf_eff = M_fused if k_pcs_fused is None else max(1, min(int(k_pcs_fused), M_fused))
    lam = eig_fused[:Kf_eff].copy()
    lam[lam < eps] = eps

    if use_whitening:
        sqrt_lam = np.sqrt(lam)
        comparison_space = Zf[:, :Kf_eff] / sqrt_lam
        deposit_space = Zf[int(deposit_index), :Kf_eff] / sqrt_lam
    else:
        comparison_space = Zf[:, :Kf_eff].copy()
        deposit_space = Zf[int(deposit_index), :Kf_eff].copy()

    deposit_unwhitened = Zf[int(deposit_index), :Kf_eff]
    raw_weights = deposit_unwhitened**2
    weight_sum = raw_weights.sum()
    if weight_sum > 0:
        weights = raw_weights / weight_sum
    else:
        weights = np.ones(Kf_eff, dtype=float) / float(Kf_eff)

    diff = comparison_space - deposit_space
    if use_weights:
        dists_sq = (diff**2) @ weights
    else:
        dists_sq = np.sum(diff**2, axis=1)

    dists = dists_sq if return_squared else np.sqrt(dists_sq)
    order = np.argsort(dists)
    ranked_idx = order
    ranked_dists = dists[order]
    if top_k is not None:
        top_k = max(1, min(int(top_k), ranked_idx.shape[0]))
        ranked_idx = ranked_idx[:top_k]
        ranked_dists = ranked_dists[:top_k]

    fusion_details = {
        "mode": "two_stage_pca_fusion",
        "K_var1": int(K1_eff),
        "K_var2": int(K2_eff),
        "n_features_var1": int(n_pix),
        "n_features_var2": int(n_pix),
        "K_fused": int(Kf_eff),
        "M_fused": int(M_fused),
        "Zf_full": Zf,
        "Zf_space": comparison_space,
        "weights_fused": weights,
        "eigvals_fused": eig_fused,
        "explained_variance_ratio_fused": np.asarray(
            pca_fused.explained_variance_ratio_,
            dtype=float,
        ),
        "standardize_fused_input": bool(standardize_fused_input),
        "zf_dep_full": np.asarray(Zf[int(deposit_index), :], dtype=float),
        "pca_var1": pca1,
        "pca_var2": pca2,
        "pca_fused": pca_fused,
        "X1_mean": np.asarray(X1_mean, dtype=float),
        "X1_std_safe": np.asarray(X1_std_safe, dtype=float),
        "X2_mean": np.asarray(X2_mean, dtype=float),
        "X2_std_safe": np.asarray(X2_std_safe, dtype=float),
        "F_mu": np.asarray(F_mu, dtype=float) if F_mu is not None else None,
        "F_std_safe": np.asarray(F_sd_safe, dtype=float) if F_sd_safe is not None else None,
    }

    return RankingResult(
        ranked_idx=ranked_idx,
        ranked_dists=ranked_dists,
        weights=weights,
        comparison_space=comparison_space,
        k_used=int(Kf_eff),
        use_whitening=use_whitening,
        use_weights=use_weights,
        return_squared=return_squared,
        ranking_mode="two_stage_pca_fusion",
        fusion_details=fusion_details,
    )


def _fit_block_scores(
    X_block: np.ndarray,
    k_req: int,
) -> tuple[np.ndarray, int, PCA, np.ndarray, np.ndarray]:
    X_mean = X_block.mean(axis=0)
    X_std = X_block.std(axis=0, ddof=0)
    X_std_safe = np.where(X_std == 0, 1.0, X_std)
    X_stdzd = (X_block - X_mean) / X_std_safe

    n_samples, n_features = X_stdzd.shape
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_stdzd)
    k_eff = max(1, min(int(k_req), n_components))
    return scores[:, :k_eff], int(k_eff), pca, X_mean, X_std_safe


def rank_by_weighted_l2_legacy_tuple(
    *,
    Z: Any,
    eigvals: Any,
    deposit_index: int,
    K: int | None = None,
    top_k: int | None = None,
    eps: float = 1e-12,
    return_squared: bool = False,
    use_whitening: bool = True,
    use_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy adapter using the old argument names and tuple return shape."""

    return rank_by_weighted_l2(
        scores=Z,
        eigvals=eigvals,
        deposit_index=deposit_index,
        k_pcs=K,
        top_k=top_k,
        eps=eps,
        return_squared=return_squared,
        use_whitening=use_whitening,
        use_weights=use_weights,
    ).to_legacy_tuple()


def _validate_scores_and_eigvals(Z: np.ndarray, eigvals: np.ndarray, deposit_index: int) -> None:
    if Z.ndim != 2:
        raise ValueError(f"scores must be a 2D matrix, got shape {Z.shape}.")
    if Z.shape[0] < 1 or Z.shape[1] < 1:
        raise ValueError(f"scores must be non-empty, got shape {Z.shape}.")
    if not np.isfinite(Z).all():
        raise ValueError("scores contain non-finite values.")
    if eigvals.ndim != 1:
        raise ValueError("eigvals must be a 1D array.")
    if eigvals.shape[0] != Z.shape[1]:
        raise ValueError("eigvals must have length equal to scores.shape[1].")
    if not np.isfinite(eigvals).all():
        raise ValueError("eigvals contain non-finite values.")
    if not (0 <= int(deposit_index) < Z.shape[0]):
        raise IndexError("deposit_index is out of bounds for scores.")
