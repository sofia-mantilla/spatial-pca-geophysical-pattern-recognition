# wPCA Explorer — interactive web app

**Live app:** https://sofia-mantilla.github.io/spatial-pca-geophysical-pattern-recognition/

A single-file, browser-based implementation of the windowed PCA (wPCA) workflow
from:

> Mantilla Salas, S., Mejia-Herrera, P., Kloeckner, J., Asadi, A., Yin, D. Z.,
> & Caers, J. (2026). *Geometry-based targeting from a single known deposit
> using windowed principal component analysis: an IOCG case study in Carajás,
> Brazil.* Natural Resources Research.

wPCA learns the reference deposit's geophysical geometry and ranks all other
areas by how closely they reproduce that geometry.

## What's in this folder

| File | Purpose |
|---|---|
| `index.html` | The complete app — Carajás demo (full 200 m SGB/CPRM grids embedded) + upload mode for your own GeoTIFFs. No build step, no dependencies, no server. |
| `wpca_core.js` | The JavaScript port of the wPCA pipeline (window extraction, per-block standardized PCA, z² deposit weights, balance-weighted concatenation, exact-geometry footprint recovery, SAD69→WGS84 Helmert). Inlined into `index.html`; kept here as the canonical source. |
| `validate_gates.cjs` | Replication-gate harness. Runs the JS port on the full 200 m data and asserts the paper's numbers. |
| `wpca_test_data.zip` | Synthetic test dataset (MAG/RAD/GRAV GeoTIFFs + deposit outlines + README) for trying the upload mode. |

## Reproducibility

The JavaScript implementation reproduces the paper's replication gates
**exactly** (same numbers as `paper/case1_uni_repro.py` and
`paper/run_ablation_checks.py`):

- **Case 1** (univariate TMI, ref. Paulo Afonso, k=17): recovery 47.04%,
  AUC 57.9, 5 hits at ranks 5/103/126/166/220, n = 13,031 windows.
- **Case 2** (TMI+U concat, ref. Alemão, k=2/8, α=0.503): recovery 63.75%,
  AUC 106.1, 3 hits at ranks 10/142/246, n = 13,197 windows.

To verify (requires the Carajás data folders from the repo README under
`../data/`):

```bash
node --max-old-space-size=6144 validate_gates.cjs      # both cases, ~75 s
```

In the app itself, switching the map to **Paper 200 m (exact)** and pressing
Run with the default settings reproduces Case 1 to the digit.

## Data security

The app is one HTML file with no server, no uploads, and no telemetry. All
files are parsed and analysed locally by the browser; it works offline.
Confidential exploration data never leaves the user's computer.

## Local development

Open `index.html` in any modern browser — that's it. The Carajás grids are
embedded (quantized uint16, base64); the 400 m preview is derived in-page.
