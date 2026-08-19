"""Run the Carajas Deposit 3 multivariate TMI + U Spatial PCA case."""

from __future__ import annotations

from run_project_from_config import build_parser, run_config


# Case 2 demo: score-concatenation multivariate run, reference deposit Alemao (3).
# (configs/carajas_multi_tmi_u.yaml is the slow Appendix B k-selection sweep, not this demo.)
DEFAULT_CONFIG = "configs/carajas_multi_tmi_u_concat_scores_tmi17_u25.yaml"
DEFAULT_DEPOSIT = 3


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Run the Carajas multivariate TMI + Radiometric U wPCA demo (Case 2, reference Alemao)."
    for action in parser._actions:
        if action.dest == "config":
            action.required = False
            action.default = DEFAULT_CONFIG
        elif action.dest == "deposit":
            action.default = DEFAULT_DEPOSIT
    args = parser.parse_args(argv)
    run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
