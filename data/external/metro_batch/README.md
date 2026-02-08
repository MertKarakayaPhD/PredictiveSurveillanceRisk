# Metro Batch Configuration

This folder contains metro ROI batch configs for large-scale simulation runs.

- `metros_us_32.json`: 32 US metros with center coordinates and radius-based boundary proxies.
- `aadt_sources_us32.json`: per-state AADT source links for the 32-metro set.

Notes:
- Boundaries are configured as circular ROI proxies (`circle_proxy`) for scalable batch execution.
- For publication-grade boundary analysis, build official polygon boundaries (CBSA) and rerun.
  - Use `scripts/build_publication_boundaries.py` to generate `metros_us_32_publication.json` and boundary GeoJSON files.
- These configs are operational inputs for expansion studies, not manuscript source-of-truth data.
