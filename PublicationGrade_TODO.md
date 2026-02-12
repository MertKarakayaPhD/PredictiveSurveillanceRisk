# Publication-Grade Expansion TODO

## 0) Execution Split by PC (Follow This Order)

### PC roles

- **Ryzen PC (simulation machine):** run camera generation + long simulations only.
- **Current PC (paper machine):** aggregate outputs, run paper calculations, update figures/tables/manuscript text.

### Ordered workflow

1. **Ryzen PC:** pull/sync latest repo snapshot.
2. **Ryzen PC:** generate manuscript-eligible camera GeoJSON for target metros.
3. **Ryzen PC:** run `run_roi_analysis.py` for those metros with `--require-aadt`.
4. **Ryzen PC -> Current PC:** copy back outputs and logs.
5. **Current PC:** aggregate metrics CSV + paper-side analyses + manuscript edits.

### Ryzen PC commands (new-metro pipeline)

```powershell
cd C:\Projects\alpr_model
python -m pip install -e ".[dev]"
```

Generate camera files (examples):

```powershell
python scripts/generate_metro_camera_geojson.py --metro-id philadelphia_pa
python scripts/generate_metro_camera_geojson.py --metro-id pittsburgh_pa
```

Run simulations (examples; PA AADT already available in `data/raw/aadt_us32/PA/`):

```powershell
python scripts/run_roi_analysis.py --name philadelphia_pa --center-lat 39.9526 --center-lon -75.1652 --radius-km 40 --camera-geojson data/raw/philadelphia_pa_cameras.geojson --state PA --aadt-path data/raw/aadt_us32/PA/pennsylvania_aadt.shp --require-aadt --n-vehicles 5000 --n-trips 10 --workers 6 --seed 42 --output-root results/metro_batch
python scripts/run_roi_analysis.py --name pittsburgh_pa --center-lat 40.4406 --center-lon -79.9959 --radius-km 30 --camera-geojson data/raw/pittsburgh_pa_cameras.geojson --state PA --aadt-path data/raw/aadt_us32/PA/pennsylvania_aadt.shp --require-aadt --n-vehicles 5000 --n-trips 10 --workers 6 --seed 42 --output-root results/metro_batch
```

Ryzen worker setting:
- Start at `--workers 6`.
- Drop to `4` if RAM pressure rises.
- Only test `8` after at least one stable completed metro.

### Copy-back from Ryzen PC to Current PC

Copy these folders/files into the same paths on Current PC:

- `results/metro_batch/*/summary.json`
- `results/metro_batch/*/psr.json`
- `results/metro_batch/*/ring_metrics.json`
- `results/metro_batch/*/road_trajectories.pkl` (optional but recommended)
- `data/raw/*_cameras.geojson` (new metros only)
- `logs/metro_batch/<RUN_ID>/`

### Current PC commands (paper-side only)

Build combined metro metrics table:

```powershell
$rows = Get-ChildItem results\metro_batch -Directory | ForEach-Object {
  $s = Join-Path $_.FullName "summary.json"
  if (Test-Path $s) {
    $j = Get-Content $s -Raw | ConvertFrom-Json
    [PSCustomObject]@{
      metro=$j.name; state=$j.state; cameras=$j.n_cameras_roi;
      density=$j.camera_density_per_km2; p1=$j.observation_stats.p_at_least_1;
      p2=$j.observation_stats.p_at_least_2; u2=$j.u2_random.uniqueness;
      acc5=$j.predictability.acc5_markov_order1; psr=$j.psr.score
    }
  }
}
$rows | Sort-Object psr -Descending | Export-Csv results\metro_batch\metrics_summary.csv -NoTypeInformation
```

Then update manuscript tables/appendix text from:
- `results/metro_batch/metrics_summary.csv`
- per-metro `summary.json`, `ring_metrics.json`, and `psr.json`.

## 1) Final 32 Metros in Batch

1. New York-Newark-Jersey City (NY)
2. Los Angeles-Long Beach-Anaheim (CA)
3. Chicago-Naperville-Elgin (IL)
4. Dallas-Fort Worth-Arlington (TX)
5. Houston-The Woodlands-Sugar Land (TX)
6. Washington-Arlington-Alexandria (DC)
7. Miami-Fort Lauderdale-West Palm Beach (FL)
8. Philadelphia-Camden-Wilmington (PA)
9. Atlanta-Sandy Springs-Roswell (GA)
10. Phoenix-Mesa-Chandler (AZ)
11. Boston-Cambridge-Newton (MA)
12. San Francisco-Oakland-Berkeley (CA)
13. Riverside-San Bernardino-Ontario (CA)
14. Detroit-Warren-Dearborn (MI)
15. Seattle-Tacoma-Bellevue (WA)
16. Minneapolis-St. Paul-Bloomington (MN)
17. San Diego-Chula Vista-Carlsbad (CA)
18. Tampa-St. Petersburg-Clearwater (FL)
19. Denver-Aurora-Lakewood (CO)
20. Baltimore-Columbia-Towson (MD)
21. St. Louis (MO)
22. Charlotte-Concord-Gastonia (NC)
23. Orlando-Kissimmee-Sanford (FL)
24. San Antonio-New Braunfels (TX)
25. Portland-Vancouver-Hillsboro (OR)
26. Sacramento-Roseville-Folsom (CA)
27. Pittsburgh (PA)
28. Las Vegas-Henderson-Paradise (NV)
29. Austin-Round Rock-Georgetown (TX)
30. Cincinnati (OH)
31. Kansas City (MO)
32. Columbus (OH)

## 2) Publication-Grade Boundaries (CBSA Polygons)

- [ ] Download Census TIGER/Line CBSA shapefile index: https://www2.census.gov/geo/tiger/TIGER2024/CBSA/
- [ ] Download `tl_2024_us_cbsa.zip`: https://www2.census.gov/geo/tiger/TIGER2024/CBSA/tl_2024_us_cbsa.zip
- [ ] Extract to local folder (example: `data/external/census/tiger2024_cbsa/`)
- [ ] Build per-metro polygon boundaries + publication config:

```powershell
python scripts/build_publication_boundaries.py `
  --cbsa-shapefile data/external/census/tiger2024_cbsa/tl_2024_us_cbsa.shp `
  --metro-config data/external/metro_batch/metros_us_32.json `
  --out-dir data/external/metro_batch/boundaries_cbsa `
  --output-config data/external/metro_batch/metros_us_32_publication.json `
  --min-match-score 0.70 `
  --max-radius-km 80
```

Notes:
- Output config is `data/external/metro_batch/metros_us_32_publication.json`.
- Each metro gets `boundary_geojson` set for polygon clipping in ROI runs.

## 3) AADT Downloads by State (for the 32 metros)

Storage recommendation:
- Put each state dataset in `data/raw/aadt_us32/<STATE>/`.
- Use the shapefile path in each metro config as `aadt_path`.

Already staged in `data/raw/aadt_us32/`:
- [x] GA: `data/raw/aadt_us32/GA/georgia_aadt.shp`
- [x] NC: `data/raw/aadt_us32/NC/north_carolina_aadt.shp`
- [x] PA: `data/raw/aadt_us32/PA/pennsylvania_aadt.shp`
- [x] TN: `data/raw/aadt_us32/TN/tennessee_aadt.shp`
- [x] VA: `data/raw/aadt_us32/VA/virginia_aadt.shp`
- [x] ME: `data/raw/aadt_us32/ME/maine_aadt.shp`

Compatibility note:
- Legacy copies remain in `data/raw/aadt/` for old scripts.
- New publication runs should reference `data/raw/aadt_us32/<STATE>/...` paths.

Still needed for 32-metro publication run:

- [ ] AZ (Phoenix): https://azdot.gov/planning/transportation-data/traffic-monitoring-program/traffic-data
- [ ] CA (LA, SF Bay, Riverside, San Diego, Sacramento): https://gis.data.ca.gov/datasets/Caltrans::annual-average-daily-traffic-aadt/about
- [ ] CO (Denver): https://dtdapps.coloradodot.info/otis/
- [ ] DC (Washington): https://opendata.dc.gov/datasets/DDOT::traffic-volume/about
- [ ] FL (Miami, Tampa, Orlando): https://gis-fdot.opendata.arcgis.com/
- [ ] IL (Chicago): https://data-idot.opendata.arcgis.com/
- [ ] MA (Boston): https://mhd.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=
- [ ] MD (Baltimore): https://maryland.maps.arcgis.com/apps/webappviewer/index.html?id=146e46f6511943f994ed16992decf4cc
- [ ] MI (Detroit): https://mdot.maps.arcgis.com/home/item.html?id=8d82f7f7f5c748f8ae7f786f186de34d
- [ ] MN (Minneapolis-St. Paul): https://www.dot.state.mn.us/traffic/data/data-products.html
- [ ] MO (St. Louis, Kansas City): https://www.modot.org/missouri-traffic-counts-map
- [ ] NV (Las Vegas): https://gis-ndot.opendata.arcgis.com/
- [ ] NY (New York): https://gis.ny.gov/gisdata/inventories/details.cfm?DSID=1301
- [ ] OH (Cincinnati, Columbus): https://services6.arcgis.com/ZIyYf7f1dA2byKzS/arcgis/rest/services/ODOTTrafficAADT/FeatureServer
- [ ] OR (Portland): https://www.oregon.gov/odot/Data/Pages/Traffic-Counting.aspx
- [ ] TX (DFW, Houston, San Antonio, Austin): https://gis-txdot.opendata.arcgis.com/search?collection=Dataset&tags=aadt
- [ ] WA (Seattle): https://wsdot.wa.gov/data/tools/traffic-data-geospatial-portal

Fallback national source (if state portals are blocked or incomplete):
- FHWA HPMS shapefiles: https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm

Additional states needed for cross-state CBSAs:
- [ ] NJ (for New York, Philadelphia): https://www.njdotlocateme.com/trafficdata/
- [ ] DE (for Philadelphia): https://deldot.gov/Publications/manuals/traffic_counts/index.shtml
- [ ] VA (for Washington): https://vdot.maps.arcgis.com/home/index.html
- [ ] WI (for Minneapolis-St. Paul): https://wisconsindot.gov/Pages/projects/data-plan/traf-counts/default.aspx

## 4) Configure AADT Paths in Publication Config

- [ ] Open `data/external/metro_batch/metros_us_32_publication.json`
- [ ] For each metro, set:
  - `aadt_path`: full or repo-relative path to the downloaded state AADT shapefile

For cross-state CBSAs, prefer `aadt_paths` (list) instead of a single `aadt_path`.
This is now supported by the batch runner and ROI script.

Examples where multi-state AADT is strongly recommended:
- `new_york_ny`: NY + NJ + PA
- `washington_dc`: DC + MD + VA
- `philadelphia_pa`: PA + NJ + DE
- `portland_or`: OR + WA
- `minneapolis_st_paul_mn`: MN + WI

Example:

```json
{
  "id": "washington_dc",
  "aadt_paths": [
    "data/raw/aadt_us32/DC/dc_aadt.shp",
    "data/raw/aadt_us32/MD/maryland_aadt.shp",
    "data/raw/aadt_us32/VA/virginia_aadt.shp"
  ]
}
```

## 5) Run Publication Batch

- [ ] Dry-run first:

```powershell
python scripts/run_metro_batch.py `
  --config data/external/metro_batch/metros_us_32_publication.json `
  --camera-catalog-csv data/external/camera_catalog/cameras_us_active.csv.gz `
  --dry-run --skip-preflight --workers 6 --blas-threads 1
```

- [ ] Full run:

```powershell
python scripts/run_metro_batch.py `
  --config data/external/metro_batch/metros_us_32_publication.json `
  --camera-catalog-csv data/external/camera_catalog/cameras_us_active.csv.gz `
  --n-vehicles 5000 --n-trips 10 --workers 6 --seed 42 `
  --ring-breaks-km 0,5,10,20,40,80 `
  --keep-last-runs 8 --blas-threads 1
```

Resume behavior:
- Re-run the same command; completed metros are skipped if signature-matched outputs already exist.

## 5.1) Unattended 5-Day Queue (Current PC)

Recommended for long unattended execution on the current workstation:
- CPU: Intel i9-14900F (32 logical threads)
- RAM: 32 GB
- GPU: RTX 4070 SUPER (12 GB VRAM)

Important:
- Current simulation path is CPU/network-graph dominated; GPU offload is not the primary bottleneck.
- Keep `--blas-threads 1` when using many workers.

Run queue (Chicago excluded by default):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_us32_unattended.ps1 `
  -Workers 14 -MinWorkers 10 -WorkerStepDown 2 `
  -MaxRetriesPerMetro 3 -CooldownMinutes 5 `
  -BlasThreads 1 -MpChunksize 2
```

Resume after interruption/error:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_us32_unattended.ps1 `
  -Workers 14 -MinWorkers 10 -WorkerStepDown 2 `
  -MaxRetriesPerMetro 3 -CooldownMinutes 5 `
  -BlasThreads 1 -MpChunksize 2
```

Notes:
- The queue is safe to rerun; per-metro runs use `run_metro_batch.py`, which skips completed outputs.
- Run logs/state are written under `logs/us32_unattended/<RUN_ID>/`.
- Include Chicago by adding `-IncludeChicago`.

## 6) Manuscript-Eligible New Metro Runs (Per-Metro, Not External Catalog)

Important:
- For manuscript-eligible runs, generate and use metro camera GeoJSON files under `data/raw/`.
- Do **not** use `data/external/camera_catalog/*` for final manuscript tables/figures.
- Use `--require-aadt` so runs fail fast if state AADT is missing.

### 6.1 Immediate runs available now (PA already present)

- [ ] Generate manuscript-eligible camera files (Philadelphia, Pittsburgh):

```powershell
python scripts/generate_metro_camera_geojson.py --metro-id philadelphia_pa
python scripts/generate_metro_camera_geojson.py --metro-id pittsburgh_pa
```

- [ ] Run manuscript-eligible PA simulations (AADT required):

```powershell
python scripts/run_roi_analysis.py --name philadelphia_pa --center-lat 39.9526 --center-lon -75.1652 --radius-km 40 --camera-geojson data/raw/philadelphia_pa_cameras.geojson --state PA --aadt-path data/raw/aadt_us32/PA/pennsylvania_aadt.shp --require-aadt --n-vehicles 5000 --n-trips 10 --workers 4 --seed 42 --output-root results/metro_batch
python scripts/run_roi_analysis.py --name pittsburgh_pa --center-lat 40.4406 --center-lon -79.9959 --radius-km 30 --camera-geojson data/raw/pittsburgh_pa_cameras.geojson --state PA --aadt-path data/raw/aadt_us32/PA/pennsylvania_aadt.shp --require-aadt --n-vehicles 5000 --n-trips 10 --workers 4 --seed 42 --output-root results/metro_batch
```

### 6.2 Template for any additional metro

- [ ] Download state AADT first (unless already present in `data/raw/aadt_us32/<STATE>/` or configured in `aadt_paths`).
- [ ] Generate manuscript-eligible camera file:

```powershell
python scripts/generate_metro_camera_geojson.py --metro-id METRO_ID
```

- [ ] Run simulation with AADT enforcement:

```powershell
python scripts/run_roi_analysis.py --name METRO_ID --center-lat LAT --center-lon LON --radius-km RADIUS_KM --camera-geojson data/raw/METRO_ID_cameras.geojson --state ST --aadt-path data/raw/aadt_us32/ST/STATE_AADT_FILE.shp --require-aadt --n-vehicles 5000 --n-trips 10 --workers 4 --seed 42 --output-root results/metro_batch
```

Notes:
- If metro spans multiple states, use `--aadt-paths` with comma-separated shapefile paths.
- Start with `--workers 4` on Ryzen 7 2700X/32GB; increase to `6` only after stable runs.
- Custom metro (not in config) camera generation example:

```powershell
python scripts/generate_metro_camera_geojson.py --name custom_roi --center-lat LAT --center-lon LON --radius-km RADIUS_KM
```
