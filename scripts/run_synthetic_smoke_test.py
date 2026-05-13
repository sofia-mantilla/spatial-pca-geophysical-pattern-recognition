"""Run a lightweight Spatial PCA smoke test on bundled synthetic data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def main() -> int:
    """Run the synthetic example through PCA and ranking without notebooks."""

    from spatial_pca.examples.illustrative import prepare_univariate_window_context
    from spatial_pca.spca.pca import fit_spca
    from spatial_pca.spca.ranking import rank_spca_windows

    data_dir = REPO_ROOT / "data" / "Illustrative Example Input Data"
    field = np.loadtxt(data_dir / "synthetic_field.csv", delimiter=",", skiprows=1)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    known_windows = pd.read_csv(data_dir / "known_deposit_windows.csv")

    context = prepare_univariate_window_context(
        field=field,
        window_shape=(int(metadata["window_height"]), int(metadata["window_width"])),
        stride_y=int(metadata["stride_y"]),
        stride_x=int(metadata["stride_x"]),
        training_window_index=int(metadata["training_window_index"]),
        variable_name=str(metadata["variable_name"]),
    )

    pca_result = fit_spca(
        context.pca_input,
        var_name=str(metadata["variable_name"]),
        patch_size=context.window_matrix.window_shape,
    )
    ranking = rank_spca_windows(
        scores=pca_result.scores,
        eigvals=pca_result.eigvals,
        deposit_index=context.deposit_index,
        k_pcs=None,
    )

    n_windows = context.window_matrix.window_indices_for_mapping.shape[0]
    valid_rank_mask = (ranking.ranked_idx != context.deposit_index) & (ranking.ranked_idx < n_windows)
    top_5 = ranking.ranked_idx[valid_rank_mask][:5].astype(int).tolist()

    print("smoke_test=PASS")
    print(f"field_shape={tuple(field.shape)}")
    print(f"window_matrix_shape={tuple(context.pca_input.shape)}")
    print(f"known_windows={known_windows['window_index'].astype(int).tolist()}")
    print(f"top_5_window_indices={top_5}")
    print(f"k_used={int(ranking.k_used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
