"""Shared script bootstrap for running from a source checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Reproducibility: pin BLAS/LAPACK to a single thread BEFORE numpy is imported.
# Multithreaded reductions change floating-point summation order between runs;
# with (near-)degenerate PCA eigenvalues this jitter rotates the retained basis
# and changes rankings (docs/reproducibility_debug_deposit3_multivariate.md).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")


def add_src_to_path() -> Path:
    """Add the repository ``src`` directory to ``sys.path`` and return repo root."""

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root
