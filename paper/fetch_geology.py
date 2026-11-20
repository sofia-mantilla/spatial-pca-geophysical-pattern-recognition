"""Fetch the geological maps Appendix C uses, from GeoSGB.

The maps are the Servico Geologico do Brasil's own, served from the GeoSGB
ArcGIS REST endpoint. They are not kept in this repository: they are third
party data of some size, and the licence for redistributing them is not ours
to assume. This script retrieves them, so the appendix stays reproducible.

Two layers are fetched, over the study-area bounding box and nothing wider.
That bounding box is the one already published with the paper, so nothing
about the ranking, the deposits or the targets is sent anywhere. It is padded
slightly so that units crossing the edge come back whole; every script clips
to the study area itself, which leaves 210 polygons of the 1:1,000,000 sheet
and 1,206 of the 1:100,000 sheets, the figures the appendix reports.

    para_1m_carajas.geojson     Para 1:1,000,000 (CPRM, 2008). Covers the
                                whole survey; used for the figures and for
                                the age comparison.
    para_100k_carajas.geojson   The twelve 1:100,000 sheets published over
                                the survey, covering 76.5 per cent of it;
                                used for the magnetite-unit comparison,
                                which needs the finer lithotype lists.

Run from inside paper/:

    python fetch_geology.py              # skips files already present
    python fetch_geology.py --force      # fetch again anyway

Writes into data/Carajas_Brazil_Geology/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "Carajas_Brazil_Geology"
BASE = "https://geoportal.sgb.gov.br/server/rest/services/geologia"

# The study area as published: SAD69 UTM 22S grid, given here in WGS84
# degrees, padded by a twentieth of a degree so that units crossing the edge
# come back whole.
PAD = 0.05
BBOX = (-51.6761 - PAD, -7.0549 - PAD, -49.2278 + PAD, -5.7008 + PAD)

LAYERS = {
    "para_1m_carajas.geojson": {
        "service": "litoestratigrafia_estados",
        "layer": 9,                      # Para 1:1,000,000 [2008]
        "expect": 267,
        "what": "Para 1:1,000,000 (CPRM, 2008)",
    },
    "para_100k_carajas.geojson": {
        "service": "litoestratigrafia_100000",
        "layer": 0,
        "expect": 1365,
        "what": "Para 1:100,000, the twelve published sheets",
    },
}
TIMEOUT = 180
MAX_DEPTH = 4


def query(service, layer, box):
    """Every feature meeting one box, and whether the server truncated it.

    The endpoint does not support pagination, so a box the server truncates
    has to be split rather than paged through.
    """
    params = {
        "geometry": ",".join("%.6f" % v for v in box),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    url = "%s/%s/MapServer/%d/query" % (BASE, service, layer)
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(params).encode(),
        headers={"User-Agent": "spatial-pca-paper/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise SystemExit("GeoSGB refused the query: %s" % payload["error"])
    return payload.get("features", []), bool(payload.get("exceededTransferLimit"))


def collect(service, layer, box, seen, depth=0):
    """Features of one box, splitting into quarters if the server truncates."""
    features, truncated = query(service, layer, box)
    if truncated and depth < MAX_DEPTH:
        x0, y0, x1, y1 = box
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        out = []
        for quarter in ((x0, y0, xm, ym), (xm, y0, x1, ym),
                        (x0, ym, xm, y1), (xm, ym, x1, y1)):
            time.sleep(0.4)              # the service is a public one
            out.extend(collect(service, layer, quarter, seen, depth + 1))
        return out
    if truncated:
        print("  the server truncated a box even at the smallest split;"
              " some units may be missing", file=sys.stderr)

    kept = []
    for feature in features:
        key = feature.get("id", feature.get("properties", {}).get("OBJECTID"))
        if key is None or key not in seen:
            if key is not None:
                seen.add(key)
            kept.append(feature)
    return kept


def fetch(name, spec, force):
    target = OUT_DIR / name
    if target.exists() and not force:
        print("%-28s already here, skipping" % name)
        return True

    features = collect(spec["service"], spec["layer"], BBOX, set())

    if not features:
        print("%-28s nothing returned" % name)
        return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False))
    note = "" if len(features) == spec["expect"] else \
        "  (%d when the paper was written; the survey revises its maps)" % spec["expect"]
    print("%-28s %d features, %.1f MB%s"
          % (name, len(features), target.stat().st_size / 1e6, note))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="fetch again even if the file is already there")
    args = parser.parse_args()

    print("GeoSGB, study-area bounding box %.4f %.4f %.4f %.4f" % BBOX)
    ok = True
    for name, spec in LAYERS.items():
        print("%s:" % spec["what"])
        ok = fetch(name, spec, args.force) and ok
    print("\ninto %s" % OUT_DIR)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
