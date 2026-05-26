"""Run validation comparison plots from a JSON or YAML config."""

from __future__ import annotations

import os

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def main(argv: list[str] | None = None) -> int:
    from spatial_pca.comparison.comparison_cases import main as comparison_main

    comparison_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
