# Spatial PCA for Geophysical Pattern Recognition in Mineral Exploration

**A reproducible method repository with an application to Carajas IOCG deposits**

This repository contains a reusable Spatial Principal Component Analysis
(Spatial PCA) workflow for geophysical pattern recognition in mineral
exploration. The main application is the Carajas, Brazil IOCG case study from
the paper:

> Multiphysics Geophysical Pattern Recognition for Mineral Exploration using
> Spatial Principal Component Analysis: Case Study of Carajas, Brazil Using IOCG
> Deposits

The repository is organized as a method repository. The Carajas case study is
the primary reproducible application, but the workflow is designed so it can be
adapted to other mineral exploration datasets by editing configuration files.

## What the Method Does

Spatial PCA learns the geophysical geometry of a known deposit and ranks other
locations in the study area by how closely their local geophysical patterns
match that deposit. The workflow:

1. Samples the study area using sliding windows.
2. Extracts a training window from a selected known deposit.
3. Fits spatial PCA to windowed geophysical data.
4. Uses deposit-specific PC weights to emphasize PCs important to the training
   deposit.
5. Ranks candidate windows by deposit-weighted similarity.
6. Validates rankings against other known deposits using footprint recovery and
   hit metrics.

This is useful when the target signature is not just a high or low anomaly, but
a spatial arrangement of geophysical responses around a known mineralized
system.

## Repository Layout

```text
Spatial_PCA_Multivariate_Repo/
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── carajas_uni_tmi.yaml
│   ├── carajas_multi_tmi_u.yaml
│   └── template_project.yaml
├── src/spatial_pca/
│   ├── config.py
│   ├── pipeline.py
│   ├── spca/
│   ├── geodata/
│   └── validation/
├── scripts/
│   ├── run_carajas_univariate.py
│   ├── run_carajas_multivariate.py
│   └── run_project_from_config.py
├── data/
│   └── Illustrative Example Input Data/
├── notebooks/
│   ├── 00_illustrative_synthetic_demo.ipynb
│   ├── 01_carajas_univariate_demo.ipynb
│   └── 02_carajas_multivariate_demo.ipynb
├── outputs/
└── docs/figures/
```

Large rasters, shapefiles, and private project data should live outside this
GitHub repository. Point the config files to those external files.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate spatial-pca
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Files

Each run config should provide:

- Geophysical raster paths, for example TMI and Radiometric U grids.
- A study area polygon for clipping the raster analysis domain.
- A deposits shapefile containing the training deposit and known validation
  deposits.
- CRS information, if raster CRS metadata needs to be repaired.
- Analysis settings such as stride, retained PCs, rotation angle, minimum
  footprint overlap, and output directory.

The reusable template uses variable-based raster paths:

```yaml
paths:
  variable_1_file_path: "/path/to/new_project/variable_1_grid.ers"
  variable_2_file_path: "/path/to/new_project/variable_2_grid.ers"
```

For local work, copy a config to `configs/local_*.yaml` and edit the paths
there. Local configs are ignored by Git. The Carajas application configs keep
the paper variable names (`TMI`, `Radiometric_U`) for readability.

## Case 1: Paulo Afonso / Deposit 6, Univariate TMI

This demo trains on Deposit 6 using the TMI grid and retained Spatial PCA
components from the paper application.

```bash
python scripts/run_carajas_univariate.py --config configs/carajas_uni_tmi.yaml
```

Configuration: `configs/carajas_uni_tmi.yaml`

Key settings:

- Training deposit: Paulo Afonso / Deposit 6
- Analysis type: `Uni`
- Variable: `TMI`
- Retained PCs: `17`
- Validation: footprint recovery against other known Carajas deposits

## Case 2: Alemao / Deposit 3, Multivariate TMI + Radiometric U

This demo trains on Deposit 3 using a multivariate TMI + Radiometric U workflow.
It uses the two-stage fused PCA ranking path and validates candidate windows
against known deposits with relevant magnetic and radiometric geometry.

```bash
python scripts/run_carajas_multivariate.py --config configs/carajas_multi_tmi_u.yaml
```

Configuration: `configs/carajas_multi_tmi_u.yaml`

Key settings:

- Training deposit: Alemao / Deposit 3
- Analysis type: `Multi`
- Variables: `TMI`, `Radiometric_U`
- Fused retained PCs: `17`
- Per-variable retained PCs: `TMI = 2`, `Radiometric_U = 34`
- Validation: footprint recovery against known Carajas deposits

## Synthetic Illustrative Example

For a lightweight demo that does not need external geophysical files, open:

```text
notebooks/00_illustrative_synthetic_demo.ipynb
```

The notebook uses tracked input files in
`data/Illustrative Example Input Data/` and calls the repository Spatial PCA
functions directly. Outputs are written to `outputs/Illustrative_Example/`.

## Run Any Project From Config

Start from the template:

```bash
cp configs/template_project.yaml configs/local_my_project.yaml
```

Edit `analysis_defaults.variable_1`, `analysis_defaults.variable_2`,
`paths.variable_1_file_path`, `paths.variable_2_file_path`, deposit paths, CRS,
training deposit ID, retained PCs, stride, rotation angle, and output directory.
Then run:

```bash
python scripts/run_project_from_config.py --config configs/local_my_project.yaml
```

For a multivariate project, set `run.analysis_type` to `Multi`, set
`analysis_defaults.variable_1` and `analysis_defaults.variable_2`, and either
provide direct `best_kpcs_files.var1_kpcs` / `best_kpcs_files.var2_kpcs` values
or point the config to supporting best-k CSV files.

## Outputs

Each case writes a run folder under `outputs/`, including:

- `top_windows.gpkg`: ranked sliding-window geometries.
- `<variables>_Top_<N>_Predicted_Windows.png`: map of top-ranked windows.
- `top_similar_windows.png`: image chips for top-ranked windows.
- `pc_score_map.png`: PCA score diagnostic map.
- `component_weights.png`: deposit-specific PC weights.
- `cumulative_footprint_recovery_fraction.png`: validation curve.
- `validation_topk_results.pkl`: validation payload.
- `run_config_resolved.json` and `run_provenance.json`: reproducibility records.

Generated outputs are ignored by Git except for the placeholder
`outputs/.gitkeep`.

## Example Figure

![Example top-ranked windows](docs/figures/top_windows.png)

Example output from the Carajas univariate TMI application. The map shows the
top-ranked sliding windows whose geophysical geometry most closely matches the
training deposit geometry.

## GitHub Repository

Remote repository name:

```text
Spatial_PCA_Multivariate_Repo
```

Repository URL:

```text
https://github.com/sofia-mantilla/Spatial_PCA_Multivariate_Repo.git
```
