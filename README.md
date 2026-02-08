# PredictiveSurveillanceRisk

Code and reproducibility assets for ALPR-based predictive surveillance risk analysis.

## Included in this repository

- Core implementation: `src/`
- Experiment and batch scripts: `scripts/`
- Tests: `tests/`
- Metro configs and data source references: `data/external/metro_batch/`
- Baseline manuscript camera inputs: `data/raw/*_cameras.geojson`
- Baseline simulation outputs used for reproducibility:
  - `results/traffic_weighted/`
  - `results/robustness/`
- Paper metrics source file: `data/source_of_truth.json`

## Not included (download separately)

Large external datasets are intentionally not committed:

- State/National AADT shapefiles (US32) under `data/raw/aadt_us32/`
- Legacy AADT folder under `data/raw/aadt/`
- Chicago taxi CSV

Use:

- State AADT source index: `data/external/metro_batch/aadt_sources_us32.json`
- Chicago Taxi dataset portal:
  - https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick checks

```powershell
python -m pytest -q
python scripts/check_data_freshness.py --strict-missing
```

## Notes

- Manuscript/paper text and figure assets are intentionally excluded from this repo.
- For publication metro expansion workflow, see `PublicationGrade_TODO.md`.
