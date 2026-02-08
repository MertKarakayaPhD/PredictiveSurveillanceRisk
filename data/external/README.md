# External Data (Not Paper Source-of-Truth)

This folder stores externally sourced reference datasets used for exploratory and operational workflows (for example, nationwide camera catalogs for fast ROI selection).

These files are **not** the manuscript source-of-truth and must not be used directly to regenerate paper tables/figures.

Paper inputs remain under:
- `data/raw/*_cameras.geojson`
- `data/raw/aadt/*`
- `data/source_of_truth.json`

If you populate `data/external/camera_catalog/`, it should contain a marker file:
- `NOT_FOR_PAPER_DO_NOT_USE_FOR_MANUSCRIPT.txt`

and a provenance summary:
- `camera_catalog_summary.json`
