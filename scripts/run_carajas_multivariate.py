"""Run the Carajas Deposit 3 multivariate TMI + U Spatial PCA case."""

from __future__ import annotations

from run_project_from_config import build_parser, run_config


DEFAULT_CONFIG = "configs/carajas_multi_tmi_u.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Run the Carajas multivariate TMI + Radiometric U Spatial PCA demo."
    for action in parser._actions:
        if action.dest == "config":
            action.required = False
            action.default = DEFAULT_CONFIG
    args = parser.parse_args(argv)
    run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
