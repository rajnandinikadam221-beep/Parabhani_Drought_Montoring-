ROLE: You are a senior geospatial data engineer. Build, run and verify an end-to-end
"Real-Time Drought Monitoring for Parbhani District" pipeline. Execute everything
automatically, fix errors yourself, and do not stop to ask questions unless
authentication or the repo URL is missing.

0. INPUTS
- Shapefile: {PATH_TO_PARBHANI_TEHSIL_SHP} (tehsil-wise). Reproject to EPSG:4326 for GEE and EPSG:32643 for area calculations. Validate and repair geometries.
- Repository: {REPO_URL}. Clone it, create a virtual environment, install requirements, run an initial smoke test.
- Study period: 2016-01-01 to 2026-10-06. ALL data is clipped strictly to the shapefile; nothing outside Parbhani is downloaded or processed.
- GEE project ID: {GEE_PROJECT_ID}

1. PROJECT STRUCTURE (create first)
project/
  config/config.yaml        (dates, paths, scale, index parameters, thresholds)
  data/{raw,interim,processed}/  outputs/{csv,rasters,figures,maps,reports}/  logs/
  src/
    processing/             (index calculation, SPI, SSMI, zonal stats, classification)
    gee/                    (auth, Sentinel download)
    weather/                (Open-Meteo client)
    ml/                     (sampling, random forest)
    viz/                    (plotly, seaborn, maps)
    api/                    (index and drought API)
    main.py                 (runs the whole pipeline end to end)
Use pathlib, type hints, logging, and a requirements.txt. Every step must be idempotent and resumable (skip files that already exist).

2. GEE SETUP (src/gee/auth.py)
- Use geemap and earthengine-api.
- Authenticate ONCE with ee.Authenticate(); afterwards reuse saved credentials via ee.Initialize(project=GEE_PROJECT_ID) inside a try/except that falls back to ee.Authenticate() only when credentials are missing or expired. The user must never log in repeatedly.

3. SENTINEL-2 DOWNLOAD (src/gee/download.py)
- Collection: COPERNICUS/S2_SR_HARMONIZED. For dates before it has coverage over Parbhani, use COPERNICUS/S2_HARMONIZED (TOA) and log which months used which collection.
- Filter by the district geometry and by date; apply the Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, cs_cdf >= 0.6) mask, falling back to the SCL band.
- Build monthly median composites for 2016-2026 and download one GeoTIFF per month clipped to the district. Default scale is 20 m (configurable to 10 m). Bands: B2, B3, B4, B5, B8, B11, B12. Split into tiles with geemap.download_ee_image_tiles or use Drive export if the size limit is hit; retry with backoff.
- Log the number of images found per month and flag months with no valid data.

4. OPEN-METEO DATA AND SPI (src/weather/openmeteo.py)
- Endpoint: https://archive-api.open-meteo.com/v1/archive. Use daily precipitation_sum, plus soil_moisture variables from the hourly/daily endpoint as the soil moisture fallback. Fetch for each tehsil centroid plus a 0.1 degree grid across the district. Use timezone Asia/Kolkata, honour rate limits (cache responses, sleep, retry).
- Fetch from 1991 onwards (to give SPI a proper reference period), but report the 2016-2026 window.
- Compute SPI-1, SPI-3, SPI-6, SPI-12 using a gamma distribution fit (scipy) per calendar month on the 1991-2020 reference period, with the zero-rain probability correction. Save as CSV and expose through an API function.
- Classify SPI: >=2 extremely wet; 1.5-1.99 very wet; 1-1.49 moderately wet; -0.99 to 0.99 near normal; -1 to -1.49 moderate drought; -1.5 to -1.99 severe drought; <=-2 extreme drought.

5. INDICES (src/processing/indices.py), using numpy and rasterio only
Compute per monthly composite, as float32 GeoTIFF with nodata handled:
- NDVI = (B8-B4)/(B8+B4)
- EVI = 2.5*(B8-B4)/(B8+6*B4-7.5*B2+1)
- PCI (Chlorophyll Index Red-Edge) = B8/B5 - 1
- SSMI: take monthly soil moisture (ERA5-Land volumetric_soil_water_layer_1 from GEE, or SMAP SPL4SMGP, resampled to the same grid; Open-Meteo as fallback), then compute the standardized anomaly per calendar month: (x - mean) / std, giving SSMI.
Clip valid ranges, mask clouds and water, and save each index as its own raster and as its own CSV (tehsil, date, mean, median, std, min, max, valid pixel count).

6. RANDOM POINTS AND ZONAL STATISTICS (src/ml/sampling.py)
- Generate stratified random points inside each tehsil (default 500 per tehsil, fixed seed, min spacing), proportional to tehsil area.
- Stack the indices (NDVI, EVI, PCI, SSMI, SPI) as a multi-layer dataset; extract values at the points and run zonal statistics per tehsil with rasterstats or numpy/rasterio masks. Output one combined, tidy table keyed by tehsil and month.
- Random forest (scikit-learn): train/test split by year or tehsil (not random rows) to avoid leakage; use SPI-based labels plus a composite-index label as the target; report accuracy, F1, a confusion matrix and feature importance; save the model with joblib. Make clear in the report what is a label and what is a prediction.

7. DROUGHT CLASSIFICATION ALGORITHM (src/processing/classify.py)
- Baseline: monthly climatology (mean and std) per tehsil for each index. Compute anomalies/z-scores and VCI = (NDVI-min)/(max-min).
- Combined Drought Index = weighted sum (default weights in config: SPI 0.35, SSMI 0.25, NDVI/VCI 0.25, EVI 0.10, PCI 0.05), standardized.
- Stages: 0 No drought, 1 Mild, 2 Moderate, 3 Severe, 4 Extreme (thresholds in config). Also map to the 3 dashboard classes: Drought / Normal / Non-drought.
- Output a per-tehsil monthly stage CSV, classified rasters, and the area in km2 and % of each stage per tehsil. Also export a JSON file suited to a Next.js dashboard.

8. API (src/api/)
Small FastAPI app with endpoints: /spi, /indices/{ndvi|evi|pci|ssmi}, /drought-stage, /baseline?year=YYYY, each filterable by tehsil and date range, returning JSON.

9. VISUALIZATION (src/viz/), saved to outputs/figures and outputs/maps
- Plotly (interactive HTML): index time series per tehsil, SPI bars, drought-stage timeline heatmap, baseline vs selected-year comparison.
- Seaborn/Matplotlib (300 dpi, publication style): correlation heatmap of the indices, distributions, boxplots by tehsil, RF confusion matrix and feature importance.
- Maps with geopandas/matplotlib (title, legend, north arrow, scale bar, CRS label): mean NDVI, EVI, PCI, SSMI, SPI per tehsil and final drought-stage maps for the driest, wettest and latest months.

10. RUN AND REPORT
- Run main.py end to end automatically. After each stage, verify the outputs (file counts, no all-NaN rasters, CRS check, shapefile bounds check) and fix problems.
- Write outputs/reports/summary.md and summary.json containing: number of files and total GB downloaded (Sentinel, Open-Meteo); number of files and GB processed; rows in each CSV; time taken per stage and in total; months with missing or low-quality data; versions of libraries; a list of assumptions and limitations (Sentinel-2 availability before 2018, SSMI source, 10-year baseline).
- Final answer: a short table of results, the output folder tree, and commands to rerun.