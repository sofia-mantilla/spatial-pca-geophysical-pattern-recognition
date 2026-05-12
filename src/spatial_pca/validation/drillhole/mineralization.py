"""Cu/Au mineralization rules for drillhole validation workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CU_THRESHOLD_PCT = 0.1
DEFAULT_AU_THRESHOLD_PPM = 0.1
DEFAULT_JOINT_CU_THRESHOLD_PCT = 0.08
DEFAULT_JOINT_AU_THRESHOLD_PPM = 0.08


def classify_cu_au_mineralization(
    frame: pd.DataFrame,
    *,
    cu_col: str = "Cu_pct",
    au_col: str = "Au_ppm",
    cu_threshold_pct: float = DEFAULT_CU_THRESHOLD_PCT,
    au_threshold_ppm: float = DEFAULT_AU_THRESHOLD_PPM,
    joint_cu_threshold_pct: float = DEFAULT_JOINT_CU_THRESHOLD_PCT,
    joint_au_threshold_ppm: float = DEFAULT_JOINT_AU_THRESHOLD_PPM,
) -> pd.Series:
    """Return 1/0/<NA> mineralization labels from the project Cu/Au rule.

    A row is mineralized if Cu or Au passes the single-element threshold, or if
    both pass the lower joint threshold. Rows with neither Cu nor Au available
    remain unknown.
    """

    if cu_col not in frame.columns or au_col not in frame.columns:
        missing = [col for col in (cu_col, au_col) if col not in frame.columns]
        raise ValueError(f"Missing required grade columns: {missing}")

    cu = pd.to_numeric(frame[cu_col], errors="coerce")
    au = pd.to_numeric(frame[au_col], errors="coerce")
    known = cu.notna() | au.notna()
    positive = (
        (cu >= cu_threshold_pct)
        | (au >= au_threshold_ppm)
        | ((cu >= joint_cu_threshold_pct) & (au >= joint_au_threshold_ppm))
    )

    state = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    state.loc[known & positive] = 1
    state.loc[known & ~positive] = 0
    return state


def apply_mineralization_columns(
    frame: pd.DataFrame,
    *,
    state_col: str = "mineralized_state",
    label_col: str = "mineralized_label",
    bool_col: str = "mineralized",
    **rule_kwargs,
) -> pd.DataFrame:
    """Return a copy with project-standard mineralization columns added."""

    out = frame.copy()
    out[state_col] = classify_cu_au_mineralization(out, **rule_kwargs)
    out[label_col] = out[state_col].map({1: "mineralized", 0: "not_mineralized"}).astype("string")
    out[label_col] = out[label_col].fillna("unknown")
    out[bool_col] = out[state_col].eq(1)
    return out


def classify_predicted_grid(
    cu_pred: pd.Series | np.ndarray,
    au_pred: pd.Series | np.ndarray,
    *,
    cu_threshold_pct: float = DEFAULT_CU_THRESHOLD_PCT,
    au_threshold_ppm: float = DEFAULT_AU_THRESHOLD_PPM,
    joint_cu_threshold_pct: float = DEFAULT_JOINT_CU_THRESHOLD_PCT,
    joint_au_threshold_ppm: float = DEFAULT_JOINT_AU_THRESHOLD_PPM,
) -> np.ndarray:
    """Classify predicted Cu/Au arrays using the same mineralization rule."""

    cu = np.asarray(cu_pred, dtype=float)
    au = np.asarray(au_pred, dtype=float)
    return (
        (cu >= cu_threshold_pct)
        | (au >= au_threshold_ppm)
        | ((cu >= joint_cu_threshold_pct) & (au >= joint_au_threshold_ppm))
    )
