"""Reduce the ARIM Carajas lithology layer to its outline, once.

run_structure_proximity.py and run_occurrence_proximity.py need to know where
the ARIM map has coverage, because it stops short of the survey's eastern
edge and a distance measured outside it is meaningless. Only the outline is
needed for that, not the 13 MB of polygons and attributes, so the outline is
derived here and committed, and the full ARIM download stays out of the repo.

Run from inside paper/, with the ARIM archive extracted under
data/Carajas_Brazil_Structures/:

    python make_arim_footprint.py

Writes data/Carajas_Brazil_Structures/arim_footprint.geojson in EPSG:4674,
as the layer itself is.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
ARIM = (HERE.parent / "data" / "Carajas_Brazil_Structures"
        / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas")
LITHOLOGY = ARIM / "Litologia" / "arim_carajas_lito.shp"
OUT = HERE.parent / "data" / "Carajas_Brazil_Structures" / "arim_footprint.geojson"


def main() -> int:
    if not LITHOLOGY.exists():
        raise SystemExit("make_arim_footprint.py: %s not found. Extract the ARIM "
                         "archive there first." % LITHOLOGY)
    lito = gpd.read_file(LITHOLOGY)
    print("read %d polygons, %s" % (len(lito), lito.crs))
    hull = lito.geometry.union_all()
    # Drop slivers and interior gaps: the mask is about coverage, not detail.
    hull = hull.buffer(0.01).buffer(-0.01)
    out = gpd.GeoDataFrame({"name": ["ARIM Carajas mapped footprint"]},
                           geometry=[hull], crs=lito.crs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, driver="GeoJSON")
    print("wrote %s (%.1f kB), bounds %s"
          % (OUT, OUT.stat().st_size / 1024,
             tuple(round(v, 3) for v in out.total_bounds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
