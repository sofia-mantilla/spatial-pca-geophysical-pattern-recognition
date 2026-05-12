"""Reusable colormaps for Spatial PCA plotting."""

from __future__ import annotations

from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import Colormap, LinearSegmentedColormap


PAPER_COLORMAP_STOPS = (
    (0.00, "#0101ff"),
    (0.167 * 1, "#2dc4ff"),
    (0.167 * 2, "#26ff00"),
    (0.167 * 3, "#ffe101"),
    (0.167 * 4, "#ff0109"),
    (0.167 * 5, "#ef00ff"),
    (1.00, "#de9eff"),
)


def build_paper_colormap(name: str = "spatial_pca_paper") -> LinearSegmentedColormap:
    """Return the legacy SPCA paper colormap used in the old workflow."""

    return LinearSegmentedColormap.from_list(name, PAPER_COLORMAP_STOPS)


DEFAULT_PAPER_CMAP = build_paper_colormap()


def resolve_colormap(value: str | Colormap | None) -> Colormap:
    """Resolve a config value to a matplotlib colormap.

    Supported string aliases:
    - ``paper`` or ``spatial_pca_paper`` for the SPCA paper colormap
    - any matplotlib colormap name accepted by ``plt.get_cmap``
    """

    if value is None:
        return DEFAULT_PAPER_CMAP
    if isinstance(value, Colormap):
        return value

    name = str(value).strip()
    if name.lower() in {"paper", "spatial_pca_paper", "northisle_paper"}:
        return DEFAULT_PAPER_CMAP
    return mpl_colormaps[name]
