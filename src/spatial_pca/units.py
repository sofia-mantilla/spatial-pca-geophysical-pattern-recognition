"""Physical units for variable display labels on figure axes and colorbars.

Colorbars and value axes that show a physical quantity carry its unit
("TMI (nT)", "eU (ppm)"); dimensionless quantities (PCA loadings, PC scores,
distances) stay bare. Variable names not listed here are returned unchanged.
"""

from __future__ import annotations


def variable_display_label(variable_name: str) -> str:
    """Return the display label (with unit) for a geophysical variable name."""

    key = str(variable_name).strip().upper().replace(" ", "_").replace("-", "_")
    if key in {"TMI", "MAG"}:
        return f"{variable_name} (nT)"
    if key in {"RADIOMETRIC_U", "RAD_U", "RAD_EU", "EU", "U"}:
        return "eU (ppm)"
    return str(variable_name)
