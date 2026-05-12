"""Configuration loading and validation for Spatial PCA workflows."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent.
    yaml = None


class ConfigError(ValueError):
    """Raised when a workflow config is missing required structure."""


REQUIRED_SECTIONS: tuple[str, ...] = (
    "run",
    "sweep",
    "reconstruction",
    "analysis_defaults",
    "visualization",
    "paths",
    "targets",
    "best_kpcs_files",
)

PATCH_ALLOWED_SHAPES: frozenset[str] = frozenset({"circle"})
PATCH_ALLOWED_SOURCES: frozenset[str] = frozenset({"manual", "deposit_bounds"})
PATCH_ALLOWED_EXPORT_GEOMETRIES: frozenset[str] = frozenset({"polygon", "point", "both"})
PATCH_ALLOWED_RADIUS_RULES: frozenset[str] = frozenset({"half_max_extent"})

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "run": (
        "base_output_dir",
        "run_mode",
        "analysis_type",
        "method_name",
        "uni_selected_variable",
        "multi_ranking_mode",
    ),
    "sweep": (
        "deposits_1based",
        "kpcs",
        "score_w_auc",
        "score_w_red_points",
        "score_w_hit_early",
        "targets_shp_mode",
        "best_kpcs_source_tag",
        "fusion_weight_var1",
        "fusion_weight_var2",
        "top_windows_color_scale_mode",
    ),
    "reconstruction": ("extra_pcs", "enable"),
    "analysis_defaults": (
        "variable_1",
        "variable_2",
        "vmin_var2",
        "vmax_var2",
        "rotation_angle",
        "stride_x",
        "stride_y",
        "n_top_windows",
        "force_crs",
        "show_plots",
        "min_cover",
    ),
    "visualization": ("deposit_limits_tmi",),
    "paths": (
        "mask_files",
        "polygon_path",
        "nodata_to_nan",
        "grayscale_weights",
    ),
    "targets": ("deposits_shp_paths",),
    "best_kpcs_files": (
        "multicriteria_filename_template",
        "multicriteria_legacy_filename_template",
        "var1_filename",
        "var1_legacy_filename",
        "var2_filename",
        "var2_legacy_filename",
    ),
}


def load_run_config(config_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Load, validate, and annotate a Spatial PCA run config.

    The returned dictionary includes a ``resolved`` section with derived paths
    and portability diagnostics. This function does not create output folders.
    """

    config_file = Path(config_path).expanduser().resolve()
    if project_root is None:
        project_root_path = _discover_project_root(config_file)
    else:
        project_root_path = Path(project_root).expanduser().resolve()

    config = _read_config_file(config_file)

    validate_run_config(config)
    output_dir = resolve_output_dir(config, project_root=project_root_path)
    absolute_paths = find_absolute_paths(config)

    if absolute_paths:
        warnings.warn(
            "Config contains absolute paths; this is reproducible on this machine "
            "but not portable without local path updates.",
            stacklevel=2,
        )

    config["resolved"] = {
        "config_path": str(config_file),
        "project_root": str(project_root_path),
        "output_dir": str(output_dir),
        "absolute_paths": absolute_paths,
    }
    return config


def _read_config_file(config_file: Path) -> dict[str, Any]:
    suffix = config_file.suffix.lower()
    with config_file.open("r", encoding="utf-8") as infile:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError("YAML configs require PyYAML. Install dependencies from requirements.txt.")
            loaded = yaml.safe_load(infile)
        else:
            loaded = json.load(infile)
    if not isinstance(loaded, dict):
        raise ConfigError("Config root must be a mapping/object.")
    return loaded


def validate_run_config(config: dict[str, Any]) -> None:
    """Validate required sections, required keys, and basic value constraints."""

    if not isinstance(config, dict):
        raise ConfigError("Config root must be a JSON object.")

    _require_keys("root", config, REQUIRED_SECTIONS)
    for section, keys in REQUIRED_KEYS.items():
        section_data = config[section]
        if not isinstance(section_data, dict):
            raise ConfigError(f"Config section '{section}' must be an object.")
        _require_keys(section, section_data, keys)

    run = config["run"]
    sweep = config["sweep"]
    analysis = config["analysis_defaults"]
    targets = config["targets"]
    visualization = config["visualization"]

    if run["analysis_type"] not in {"Uni", "Multi"}:
        raise ConfigError("run.analysis_type must be 'Uni' or 'Multi'.")
    if run["method_name"] not in {"Spatial_PCA", "Raw_comparison"}:
        raise ConfigError("run.method_name must be 'Spatial_PCA' or 'Raw_comparison'.")
    if run["run_mode"] not in {"sweep_kpcs", "optimal_k_recon"}:
        raise ConfigError("run.run_mode must be 'sweep_kpcs' or 'optimal_k_recon'.")

    _require_positive_ints("sweep.deposits_1based", sweep["deposits_1based"])
    _require_positive_ints("sweep.kpcs", sweep["kpcs"])
    _require_positive_int("analysis_defaults.stride_x", analysis["stride_x"])
    _require_positive_int("analysis_defaults.stride_y", analysis["stride_y"])
    _require_positive_int("analysis_defaults.n_top_windows", analysis["n_top_windows"])

    min_cover = float(analysis["min_cover"])
    if not 0.0 < min_cover <= 1.0:
        raise ConfigError("analysis_defaults.min_cover must be in the interval (0, 1].")

    deposits_shp_paths = targets["deposits_shp_paths"]
    if not isinstance(deposits_shp_paths, dict) or not deposits_shp_paths:
        raise ConfigError("targets.deposits_shp_paths must be a non-empty object.")
    if sweep["targets_shp_mode"] not in deposits_shp_paths:
        raise ConfigError(
            "sweep.targets_shp_mode must match one key in targets.deposits_shp_paths."
        )

    _validate_raster_path_config(config)

    image_colormap = visualization.get("image_colormap")
    if image_colormap is not None:
        try:
            from spatial_pca.colormaps import resolve_colormap

            resolve_colormap(image_colormap)
        except ImportError:
            warnings.warn(
                "Skipping colormap validation because matplotlib is not installed.",
                stacklevel=2,
            )
        except ValueError as exc:
            raise ConfigError(
                "visualization.image_colormap must be 'paper', 'spatial_pca_paper', "
                "or a valid matplotlib colormap name."
            ) from exc

    _validate_optional_patch_config(config.get("patch"))


def resolve_output_dir(config: dict[str, Any], project_root: str | Path | None = None) -> Path:
    """Resolve the configured output directory for a run group."""

    run = config["run"]
    base_output_dir = Path(str(run["base_output_dir"])).expanduser()
    if not base_output_dir.is_absolute() and project_root is not None:
        base_output_dir = Path(project_root).expanduser() / base_output_dir
    outputs_subdir_value = run.get("outputs_subdir")
    if outputs_subdir_value is None or str(outputs_subdir_value).strip() == "":
        outputs_subdir = Path(build_output_subdir_from_config(config))
    else:
        outputs_subdir = Path(str(outputs_subdir_value))
    if outputs_subdir.is_absolute():
        return outputs_subdir
    return base_output_dir / outputs_subdir


def build_output_subdir_from_config(config: dict[str, Any]) -> str:
    """Build a descriptive output-group folder name from existing config fields."""

    run = config["run"]
    sweep = config["sweep"]
    analysis = config["analysis_defaults"]
    patch = config.get("patch")

    tokens: list[str] = [
        _slugify_method(run["method_name"]),
        str(run["analysis_type"]).lower(),
        _slugify_variables(run, analysis),
    ]

    if patch is not None:
        tokens.extend(
            [
                str(patch["shape"]).lower(),
                _compact_patch_source_token(patch["source"]),
            ]
        )
        if str(patch["source"]) == "manual":
            manual = patch["manual"]
            tokens.extend(
                [
                    f"cx{_format_coord_token(manual['center_x'])}",
                    f"cy{_format_coord_token(manual['center_y'])}",
                    f"r{_format_metric_token(manual['radius_m'])}",
                ]
            )
        elif str(patch["source"]) == "deposit_bounds":
            tokens.append(_compact_radius_rule_token(patch["deposit_bounds"]["radius_rule"]))

    tokens.extend(
        [
            _format_int_list_token("dep", sweep["deposits_1based"]),
            _format_int_list_token("k", sweep["kpcs"]),
            f"s{int(analysis['stride_x'])}x{int(analysis['stride_y'])}",
            f"mc{_format_decimal_token(analysis['min_cover'])}",
        ]
    )

    if str(run["analysis_type"]) == "Multi":
        tokens.append(_compact_ranking_mode_token(run["multi_ranking_mode"]))

    return "_".join(token for token in tokens if token)


def find_absolute_paths(value: Any, prefix: str = "") -> list[str]:
    """Return config locations whose string values look like absolute paths."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_absolute_paths(nested_value, nested_prefix))
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            nested_prefix = f"{prefix}[{index}]"
            found.extend(find_absolute_paths(nested_value, nested_prefix))
    elif isinstance(value, str) and Path(value).expanduser().is_absolute():
        found.append(prefix)
    return found


def resolve_variable_raster_path(config: dict[str, Any], variable_name: str) -> Path:
    """Resolve the raster path for a configured variable name.

    Preferred portable config forms are:

    - ``paths.variable_1_file_path`` / ``paths.variable_2_file_path``
    - ``paths.variable_paths.<variable_name>``

    The older Carajas configs may still use ``tmi_file_path`` and
    ``rad_file_path`` as legacy aliases for variable 1 and variable 2.
    """

    path_value = _get_variable_raster_path_value(config, variable_name)
    if path_value is None:
        raise ConfigError(
            f"Could not resolve a raster path for variable '{variable_name}'. "
            "Use paths.variable_paths, paths.variable_1_file_path, or "
            "paths.variable_2_file_path."
        )
    return Path(str(path_value)).expanduser()


def _validate_raster_path_config(config: dict[str, Any]) -> None:
    paths = config["paths"]
    variable_paths = paths.get("variable_paths")
    if variable_paths is not None and not isinstance(variable_paths, dict):
        raise ConfigError("paths.variable_paths must be an object when provided.")

    for variable_name in _required_raster_variables(config):
        if _get_variable_raster_path_value(config, variable_name) is None:
            raise ConfigError(
                f"Missing raster path for variable '{variable_name}'. "
                "Use paths.variable_paths, paths.variable_1_file_path, or "
                "paths.variable_2_file_path."
            )


def _required_raster_variables(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    analysis = config["analysis_defaults"]
    if run["analysis_type"] == "Uni":
        return [str(run["uni_selected_variable"])]
    return [str(analysis["variable_1"]), str(analysis["variable_2"])]


def _get_variable_raster_path_value(config: dict[str, Any], variable_name: str) -> Any | None:
    paths = config["paths"]
    analysis = config["analysis_defaults"]
    variable_name_text = str(variable_name)
    variable_name_upper = variable_name_text.upper()

    variable_paths = paths.get("variable_paths")
    if isinstance(variable_paths, dict):
        if variable_name_text in variable_paths:
            return variable_paths[variable_name_text]
        for key, value in variable_paths.items():
            if str(key).upper() == variable_name_upper:
                return value

    if variable_name_upper == str(analysis["variable_1"]).upper():
        if "variable_1_file_path" in paths:
            return paths["variable_1_file_path"]
        if "tmi_file_path" in paths:
            return paths["tmi_file_path"]

    if variable_name_upper == str(analysis["variable_2"]).upper():
        if "variable_2_file_path" in paths:
            return paths["variable_2_file_path"]
        if "rad_file_path" in paths:
            return paths["rad_file_path"]

    return None


def _slugify_method(method_name: str) -> str:
    method = str(method_name).strip().lower()
    if method == "spatial_pca":
        return "spca"
    return method.replace(" ", "_")


def _slugify_variables(run: dict[str, Any], analysis: dict[str, Any]) -> str:
    if str(run["analysis_type"]) == "Uni":
        return str(run["uni_selected_variable"]).lower()
    return f"{str(analysis['variable_1']).lower()}_{str(analysis['variable_2']).lower()}"


def _format_int_list_token(prefix: str, values: list[Any]) -> str:
    ints = [int(v) for v in values]
    if len(ints) == 1:
        return f"{prefix}{ints[0]}"
    unique_sorted = sorted(set(ints))
    if unique_sorted == list(range(unique_sorted[0], unique_sorted[-1] + 1)):
        return f"{prefix}{unique_sorted[0]}-{unique_sorted[-1]}"
    return f"{prefix}{'-'.join(str(v) for v in unique_sorted)}"


def _format_decimal_token(value: Any) -> str:
    number = float(value)
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _format_metric_token(value: Any) -> str:
    return _format_decimal_token(value)


def _format_coord_token(value: Any) -> str:
    return str(int(round(float(value))))


def _compact_patch_source_token(value: Any) -> str:
    source = str(value).strip().lower()
    mapping = {
        "deposit_bounds": "bounds",
        "manual": "manual",
    }
    return mapping.get(source, source)


def _compact_radius_rule_token(value: Any) -> str:
    rule = str(value).strip().lower()
    mapping = {
        "half_max_extent": "hmax",
    }
    return mapping.get(rule, rule)


def _compact_ranking_mode_token(value: Any) -> str:
    mode = str(value).strip().lower()
    mapping = {
        "two_stage_pca_fusion": "2stage",
    }
    return mapping.get(mode, mode)


def _validate_optional_patch_config(patch: Any) -> None:
    if patch is None:
        return
    if not isinstance(patch, dict):
        raise ConfigError("Config section 'patch' must be an object when provided.")

    _require_keys("patch", patch, ("shape", "source", "rotation_deg", "stride_units", "export_geometry"))

    shape = str(patch["shape"])
    source = str(patch["source"])
    export_geometry = str(patch["export_geometry"])
    stride_units = str(patch["stride_units"])
    rotation_deg = float(patch["rotation_deg"])

    if shape not in PATCH_ALLOWED_SHAPES:
        raise ConfigError(f"patch.shape must be one of {sorted(PATCH_ALLOWED_SHAPES)}.")
    if source not in PATCH_ALLOWED_SOURCES:
        raise ConfigError(f"patch.source must be one of {sorted(PATCH_ALLOWED_SOURCES)}.")
    if export_geometry not in PATCH_ALLOWED_EXPORT_GEOMETRIES:
        raise ConfigError(
            f"patch.export_geometry must be one of {sorted(PATCH_ALLOWED_EXPORT_GEOMETRIES)}."
        )
    if stride_units != "pixels":
        raise ConfigError("patch.stride_units must be 'pixels' for the current implementation plan.")
    if rotation_deg != 0.0:
        raise ConfigError("patch.rotation_deg must be 0.0 for the current circle-only phase.")

    if source == "manual":
        manual = patch.get("manual")
        if not isinstance(manual, dict):
            raise ConfigError("patch.manual must be an object when patch.source is 'manual'.")
        _require_keys("patch.manual", manual, ("center_x", "center_y", "radius_m"))
        _require_real("patch.manual.center_x", manual["center_x"])
        _require_real("patch.manual.center_y", manual["center_y"])
        _require_positive_real("patch.manual.radius_m", manual["radius_m"])
    elif source == "deposit_bounds":
        deposit_bounds = patch.get("deposit_bounds")
        if not isinstance(deposit_bounds, dict):
            raise ConfigError(
                "patch.deposit_bounds must be an object when patch.source is 'deposit_bounds'."
            )
        _require_keys("patch.deposit_bounds", deposit_bounds, ("radius_rule",))
        radius_rule = str(deposit_bounds["radius_rule"])
        if radius_rule not in PATCH_ALLOWED_RADIUS_RULES:
            raise ConfigError(
                f"patch.deposit_bounds.radius_rule must be one of {sorted(PATCH_ALLOWED_RADIUS_RULES)}."
            )


def _require_keys(section_name: str, section_data: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section_data]
    if missing:
        raise ConfigError(f"Missing keys in '{section_name}' config: {missing}")


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")


def _require_positive_ints(name: str, values: Any) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{name} must be a non-empty list of positive integers.")
    for value in values:
        _require_positive_int(name, value)


def _require_real(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a real number.")


def _require_positive_real(name: str, value: Any) -> None:
    _require_real(name, value)
    if float(value) <= 0.0:
        raise ConfigError(f"{name} must be greater than 0.")


def _discover_project_root(config_file: Path) -> Path:
    for parent in (config_file.parent, *config_file.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Spatial PCA run config.")
    parser.add_argument("config_path", help="Path to a JSON or YAML run config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_run_config(args.config_path)
    resolved = config["resolved"]
    run = config["run"]
    sweep = config["sweep"]
    print(f"Config: {resolved['config_path']}")
    print(f"Method: {run['method_name']} | Analysis: {run['analysis_type']}")
    print(f"Deposits: {sweep['deposits_1based']} | k_pcs: {sweep['kpcs']}")
    if "patch" in config:
        patch = config["patch"]
        print(f"Patch: {patch['shape']} | Source: {patch['source']} | Export: {patch['export_geometry']}")
    print(f"Output dir: {resolved['output_dir']}")
    print(f"Absolute path fields: {len(resolved['absolute_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
