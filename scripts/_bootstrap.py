"""Shared script bootstrap for running from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    """Add the repository ``src`` directory to ``sys.path`` and return repo root."""

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root
