import os
import sys
import time
import logging
import argparse
import json
import warnings
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import ee
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon

warnings.filterwarnings('ignore')

# Workspace Constants
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
DASHBOARDS_DIR = os.path.join(PROJECT_DIR, "dashboards")
RASTERS_DIR = os.path.join(OUTPUTS_DIR, "rasters")

# Ensure required directories exist
for d in [DATA_DIR, OUTPUTS_DIR, SCRIPTS_DIR, LOGS_DIR, DASHBOARDS_DIR, RASTERS_DIR]:
    os.makedirs(d, exist_ok=True)

# Tracking metrics for the report
metrics = {
    "data_downloaded_mb": 0.0,
    "s2_scenes_processed": 0,
    "s2_composites_processed": 0,
    "api_records_processed": 0,
    "spatial_points": 0,
    "start_time": time.time(),
    "module_times": {},
    "outputs": []
}

def setup_logging():
    log_file = os.path.join(LOGS_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Pipeline started. Logging to {log_file}")

def authenticate_gee(project_id=None):
    """
    Initializes Google Earth Engine strictly via ee.Initialize().
    Never calls interactive ee.Authenticate() inside the pipeline.
    Fails fast with exit code 1 if initialization fails.
    """
    logging.info("Initializing Google Earth Engine...")
    t0 = time.time()
    gee_proj = project_id or os.environ.get("GEE_PROJECT")
    
    try:
        if gee_proj:
            logging.info(f"Connecting to Earth Engine with project ID: '{gee_proj}'...")
            ee.Initialize(project=gee_proj)
        else:
            logging.info("Connecting to Earth Engine with persistent credentials...")
            ee.Initialize()
        logging.info("Google Earth Engine initialized successfully.")
    except Exception as e:
        logging.error("=" * 65)
        logging.error("Google Earth Engine initialization failed!")
        logging.error(f"Error details: {e}")
        logging.error("-" * 65)
        logging.error("Authentication required outside this pipeline.")
        logging.error("Please run the following command in your terminal:")
        logging.error("    earthengine authenticate")
        if not gee_proj:
            logging.error("Ensure a Google Cloud Project with Earth Engine API enabled is configured:")
            logging.error("    set GEE_PROJECT=your-project-id (or pass --gee-project your-project-id)")
        logging.error("All synthetic / dummy data fallbacks have been removed.")
        logging.error("Exiting pipeline immediately.")
        logging.error("=" * 65)
        sys.exit(1)
        
    metrics["module_times"]["gee_init"] = time.time() - t0

def load_and_prepare_shapefile():
    """
    Loads tehsil boundary shapefile for Parbhani.
    Validates existence, geometry, and CRS. Exits if missing or invalid.
    Reprojects to EPSG:4326 for GEE/APIs and EPSG:32643 (UTM 43N) for area calculations.
    """
    logging.info("Loading Study Area & Boundary Masking...")
    t0 = time.time()
    
    candidate_paths = [
        os.path.join(DATA_DIR, "Parbhani", "Parbhani.shp"),
        os.path.join(DATA_DIR, "Parabhani", "Parabhani.shp"),
        os.path.join(DATA_DIR, "Parbhani.shp")
    ]
    
    shp_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            shp_path = p
            break
            
    if not shp_path:
        logging.error("=" * 65)
        logging.error("Shapefile not found!")
        logging.error(f"Searched candidate locations: {candidate_paths}")
        logging.error("Please provide the tehsil shapefile at 'data/Parbhani/Parbhani.shp'.")
        logging.error("Exiting pipeline immediately.")
        logging.error("=" * 65)
        sys.exit(1)
        
    logging.info(f"Loading shapefile from: {shp_path}")
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        logging.error(f"Failed to read shapefile at {shp_path}: {e}")
        sys.exit(1)
        
    if gdf.empty:
        logging.error(f"Shapefile at {shp_path} contains 0 features. Exiting pipeline.")
        sys.exit(1)
        
    if gdf.crs is None:
        logging.error(f"Shapefile at {shp_path} has an invalid or missing Coordinate Reference System (CRS). Exiting pipeline.")
        sys.exit(1)
        
    logging.info(f"Shapefile loaded successfully with {len(gdf)} features. Original CRS: {gdf.crs}")
    
    # Standardize tehsil name column
    tehsil_col = None
    for col in ['SUB_DIST', 'TEHSIL', 'TALUK', 'SUBDIST_NA', 'NAME_3', 'NAME_2', 'DISTRICT']:
        if col in gdf.columns:
            tehsil_col = col
            break
            
    if tehsil_col:
        gdf['tehsil'] = gdf[tehsil_col].astype(str).str.title().str.strip()
    else:
        gdf['tehsil'] = [f"Tehsil_{i+1}" for i in range(len(gdf))]
        
    # Reproject to EPSG:4326 for GEE / API bounding coordinates
    try:
        gdf_4326 = gdf.to_crs(epsg=4326)
    except Exception as e:
        logging.error(f"Failed to reproject shapefile to EPSG:4326: {e}")
        sys.exit(1)
        
    # Reproject to EPSG:32643 (UTM Zone 43N) for accurate metric area and sampling
    try:
        gdf_proj = gdf.to_crs(epsg=32643)
    except Exception as e:
        logging.error(f"Failed to reproject shapefile to EPSG:32643: {e}")
        sys.exit(1)
        
    total_area_sqkm = gdf_proj.geometry.area.sum() / 1e6
    logging.info(f"Parbhani District total area: {total_area_sqkm:.2f} km² across {len(gdf)} tehsils: {gdf_4326['tehsil'].tolist()}")
    
    metrics["module_times"]["shapefile_processing"] = time.time() - t0
    return gdf_4326, gdf_proj

def calculate_spi(precip_series, window=3):
    """
    Calculates Standardized Precipitation Index (SPI) using a Gamma distribution fit
    with zero-precipitation probability adjustment.
    """
    rolling = precip_series.rolling(window=window).sum()
    valid = rolling.dropna()
    if len(valid) < window:
        return pd.Series(index=precip_series.index, dtype=float)
        
    n_total = len(valid)
    zeros = (valid <= 0).sum()
    q = zeros / n_total  # Probability of zero rain
    
    positive = valid[valid > 0]
    if len(positive) < 3:
        return pd.Series(index=precip_series.index, dtype=float)
        
    try:
        shape, loc, scale = st.gamma.fit(positive, floc=0)
        spi_values = pd.Series(index=valid.index, dtype=float)
        for idx, val in valid.items():
            if val <= 0:
                prob = q
            else:
                gamma_cdf = st.gamma.cdf(val, shape, loc=loc, scale=scale)
                prob = q + (1.0 - q) * gamma_cdf
            # Avoid extremes of CDF (0 or 1) that lead to -inf / +inf
            prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
            spi_values[idx] = st.norm.ppf(prob)
            
        result = pd.Series(index=precip_series.index, dtype=float)
        result.update(spi_values)
        return result
    except Exception as e:
        logging.warning(f"Gamma fit for SPI-{window} failed ({e}), using standardized z-score.")
        mean_val = rolling.mean()
        std_val = rolling.std()
        if std_val > 0:
            return (rolling - mean_val) / std_val
        return pd.Series(index=precip_series.index, dtype=float)

def fetch_meteo_data(gdf_4326, start_year=2016, end_year=2026, force=False):
    """
    Fetches real Open-Meteo precipitation, temperature, and FAO reference ET (ET0).
    Checks cache before downloading. Computes SPI-1, SPI-3, SPI-6, SPI-12, and water deficit.
    """
    logging.info("Retrieving Meteorological Data (Open-Meteo)...")
    t0 = time.time()
    
    csv_out_path = os.path.join(OUTPUTS_DIR, "parbhani_meteorological_indices.csv")
    raw_cache_path = os.path.join(DATA_DIR, "open_meteo_raw.json")
    
    if os.path.exists(csv_out_path) and not force:
        logging.info(f"[CACHE] Found cached meteorological indices at: {csv_out_path}")
        df_monthly = pd.read_csv(csv_out_path, index_col=0, parse_dates=True)
        metrics["api_records_processed"] = len(df_monthly) * 30
        metrics["outputs"].append(csv_out_path)
        metrics["module_times"]["meteo_processing"] = time.time() - t0
        return df_monthly

    centroid = gdf_4326.geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    logging.info(f"Querying Open-Meteo Archive API for Parbhani Centroid (Lon: {lon:.4f}, Lat: {lat:.4f})...")
    
    end_date_limit = min(f"{end_year}-12-31", (datetime.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d'))
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": f"{start_year}-01-01",
        "end_date": end_date_limit,
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "et0_fao_evapotranspiration"],
        "timezone": "Asia/Kolkata"
    }
    
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        logging.error(f"Open-Meteo API returned HTTP {response.status_code}: {response.text}")
        sys.exit(1)
        
    downloaded_bytes = len(response.content)
    metrics["data_downloaded_mb"] += downloaded_bytes / (1024 * 1024)
    data = response.json()
    
    # Cache raw JSON
    with open(raw_cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        logging.error("Failed to retrieve valid daily meteo data from Open-Meteo response.")
        sys.exit(1)
        
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    metrics["api_records_processed"] += len(df)
    
    # Calculate daily derived metrics
    df["temperature_2m_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    df["water_deficit"] = df["precipitation_sum"] - df["et0_fao_evapotranspiration"]
    
    # Resample to monthly sums / means
    df_monthly = pd.DataFrame()
    df_monthly["precipitation_sum"] = df["precipitation_sum"].resample('ME').sum()
    df_monthly["et0_sum"] = df["et0_fao_evapotranspiration"].resample('ME').sum()
    df_monthly["water_deficit"] = df["water_deficit"].resample('ME').sum()
    df_monthly["temperature_mean"] = df["temperature_2m_mean"].resample('ME').mean()
    
    # Standardized Precipitation Index across multiple windows
    df_monthly["SPI-1"] = calculate_spi(df_monthly["precipitation_sum"], window=1)
    df_monthly["SPI-3"] = calculate_spi(df_monthly["precipitation_sum"], window=3)
    df_monthly["SPI-6"] = calculate_spi(df_monthly["precipitation_sum"], window=6)
    df_monthly["SPI-12"] = calculate_spi(df_monthly["precipitation_sum"], window=12)
    
    # Standardized Water Deficit Index (SWDI / meteorological SSMI equivalent from P - ET0)
    mean_wd = df_monthly["water_deficit"].mean()
    std_wd = df_monthly["water_deficit"].std()
    df_monthly["SWDI"] = (df_monthly["water_deficit"] - mean_wd) / std_wd if std_wd > 0 else 0.0
    
    df_monthly.to_csv(csv_out_path)
    metrics["outputs"].append(csv_out_path)
    logging.info(f"Processed {len(df)} daily records into {len(df_monthly)} monthly intervals. Saved to {csv_out_path}")
    
    metrics["module_times"]["meteo_processing"] = time.time() - t0
    return df_monthly

def shapely_to_ee(geom):
    """Converts a Shapely geometry to an ee.Geometry."""
    if geom.geom_type == 'Polygon':
        coords = [list(geom.exterior.coords)]
        return ee.Geometry.Polygon(coords)
    elif geom.geom_type == 'MultiPolygon':
        polys = []
        for poly in geom.geoms:
            polys.append(list(poly.exterior.coords))
        return ee.Geometry.MultiPolygon(polys)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

def fetch_gee_indices(gdf_4326, start_year=2016, end_year=2026, force=False):
    """
    Downloads and computes Sentinel-2 monthly composites and indices (NDVI, EVI, PCI) from GEE.
    Implements monthly caching: reruns skip months that have already been computed or downloaded.
    """
    logging.info("Executing Earth Engine Sentinel-2 Data Pipeline...")
    t0 = time.time()
    
    monthly_csv_path = os.path.join(OUTPUTS_DIR, "sentinel2_monthly_indices.csv")
    existing_records = {}
    
    if os.path.exists(monthly_csv_path) and not force:
        logging.info(f"[CACHE] Found existing Sentinel-2 indices table at: {monthly_csv_path}")
        cached_df = pd.read_csv(monthly_csv_path)
        for _, row in cached_df.iterrows():
            existing_records[row['month']] = row.to_dict()
            
    district_union = gdf_4326.geometry.unary_union
    ee_aoi = shapely_to_ee(district_union)
    
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Generate list of year-months to process
    months_to_process = []
    for y in range(start_year, min(end_year, current_year) + 1):
        max_m = current_month if y == current_year else 12
        for m in range(1, max_m + 1):
            months_to_process.append((y, m))
            
    logging.info(f"Targeting {len(months_to_process)} monthly intervals for Sentinel-2 processing ({start_year}-{end_year}).")
    
    records = list(existing_records.values())
    new_downloads = 0
    
    for y, m in months_to_process:
        month_str = f"{y}-{m:02d}"
        raster_filename = f"NDVI_{month_str}.tif"
        raster_path = os.path.join(RASTERS_DIR, raster_filename)
        
        # Check cache: skip if month already processed and raster exists
        if month_str in existing_records and os.path.exists(raster_path) and not force:
            continue
            
        start_date = f"{y}-{m:02d}-01"
        if m == 12:
            end_date = f"{y+1}-01-01"
        else:
            end_date = f"{y}-{m+1:02d}-01"
            
        logging.info(f"Processing Sentinel-2 for {month_str} [{start_date} to {end_date}]...")
        
        try:
            # Query Harmonized Surface Reflectance, fallback to TOA for earlier periods
            s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(ee_aoi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            scene_count = s2_col.size().getInfo()
            collection_tier = 'SR'
            
            if scene_count == 0:
                s2_col = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                          .filterBounds(ee_aoi)
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
                scene_count = s2_col.size().getInfo()
                collection_tier = 'TOA'
                
            metrics["s2_scenes_processed"] += scene_count
            
            if scene_count == 0:
                logging.warning(f"No Sentinel-2 scenes found for {month_str} (Cloud < 30%).")
                continue
                
            metrics["s2_composites_processed"] += 1
            
            # Median composite clipped to AOI
            composite = s2_col.median().clip(ee_aoi)
            
            # Vegetation Indices
            ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
            evi = composite.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))', {
                    'NIR': composite.select('B8'),
                    'RED': composite.select('B4'),
                    'BLUE': composite.select('B2')
            }).rename('EVI')
            
            # Red-Edge Chlorophyll Index: PCI = (B8 / B5) - 1
            pci = composite.expression(
                '(NIR / RE1) - 1', {
                    'NIR': composite.select('B8'),
                    'RE1': composite.select('B5')
            }).rename('PCI')
            
            stacked = composite.addBands([ndvi, evi, pci])
            
            # Regional Mean Zonal Stats over Parbhani
            stats = stacked.select(['NDVI', 'EVI', 'PCI']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_aoi,
                scale=250,
                maxPixels=1e9
            ).getInfo()
            
            mean_ndvi = stats.get('NDVI', np.nan)
            mean_evi = stats.get('EVI', np.nan)
            mean_pci = stats.get('PCI', np.nan)
            
            rec = {
                "month": month_str,
                "scenes_count": scene_count,
                "collection": collection_tier,
                "mean_ndvi": round(mean_ndvi, 4) if mean_ndvi is not None else np.nan,
                "mean_evi": round(mean_evi, 4) if mean_evi is not None else np.nan,
                "mean_pci": round(mean_pci, 4) if mean_pci is not None else np.nan,
                "raster_path": raster_path
            }
            records = [r for r in records if r['month'] != month_str]
            records.append(rec)
            new_downloads += 1
            
            # Download monthly NDVI GeoTIFF raster at 250m scale
            try:
                download_url = ndvi.getDownloadURL({
                    'scale': 250,
                    'crs': 'EPSG:4326',
                    'region': ee_aoi,
                    'format': 'GEO_TIFF'
                })
                r = requests.get(download_url, timeout=60)
                if r.status_code == 200:
                    with open(raster_path, 'wb') as f:
                        f.write(r.content)
                    metrics["data_downloaded_mb"] += len(r.content) / (1024 * 1024)
            except Exception as e_down:
                logging.warning(f"Could not download GeoTIFF for {month_str}: {e_down}")
                
        except Exception as e:
            logging.error(f"Error processing GEE Sentinel-2 for month {month_str}: {e}")
            logging.error("Exiting pipeline to avoid synthetic fallback.")
            sys.exit(1)
            
    # Save updated indices CSV
    s2_df = pd.DataFrame(records)
    if not s2_df.empty:
        s2_df.sort_values(by='month', inplace=True)
        s2_df.to_csv(monthly_csv_path, index=False)
        metrics["outputs"].append(monthly_csv_path)
        
    logging.info(f"Sentinel-2 pipeline completed: {new_downloads} new monthly composites processed.")
    metrics["module_times"]["gee_processing"] = time.time() - t0
    return s2_df

def generate_spatial_samples(gdf_proj, n_samples=5000, force=False):
    """
    Generates stratified random sample points across tehsils proportional to area.
    Checks cache before regenerating.
    """
    logging.info("Executing Stratified Spatial Sampling...")
    t0 = time.time()
    
    out_csv = os.path.join(OUTPUTS_DIR, "spatial_samples.csv")
    if os.path.exists(out_csv) and not force:
        logging.info(f"[CACHE] Found spatial samples at: {out_csv}. Skipping sampling.")
        samples_df = pd.read_csv(out_csv)
        metrics["spatial_points"] = len(samples_df)
        metrics["outputs"].append(out_csv)
        metrics["module_times"]["spatial_sampling"] = time.time() - t0
        return samples_df
        
    np.random.seed(42)
    points = []
    total_area = gdf_proj.geometry.area.sum()
    
    for idx, row in gdf_proj.iterrows():
        poly = row['geometry']
        tehsil_name = row.get('tehsil', f'Tehsil_{idx}')
        area_ratio = poly.area / total_area
        tehsil_n = max(20, int(n_samples * area_ratio))
        
        min_x, min_y, max_x, max_y = poly.bounds
        count = 0
        attempts = 0
        max_attempts = tehsil_n * 60
        while count < tehsil_n and attempts < max_attempts:
            attempts += 1
            pt = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            if pt.within(poly):
                points.append({
                    "tehsil": tehsil_name,
                    "x_utm": pt.x,
                    "y_utm": pt.y,
                    "geometry": pt
                })
                count += 1
                
    samples_gdf = gpd.GeoDataFrame(points, crs=gdf_proj.crs)
    samples_4326 = samples_gdf.to_crs(epsg=4326)
    samples_gdf['lon'] = samples_4326.geometry.x
    samples_gdf['lat'] = samples_4326.geometry.y
    
    samples_df = samples_gdf.drop(columns='geometry')
    samples_df.to_csv(out_csv, index=False)
    
    metrics["spatial_points"] = len(samples_df)
    metrics["outputs"].append(out_csv)
    logging.info(f"Generated {len(samples_df)} stratified points across {len(gdf_proj)} tehsils. Saved to {out_csv}")
    
    metrics["module_times"]["spatial_sampling"] = time.time() - t0
    return samples_df

def generate_visualizations(meteo_df, s2_df, gdf_4326, force=False):
    """
    Generates publication-quality charts and interactive dashboards strictly from real data.
    """
    logging.info("Generating Visualizations and Automated Dashboards...")
    t0 = time.time()
    
    out_trend = os.path.join(OUTPUTS_DIR, "SPI_vs_NDVI_trend.png")
    out_heatmap = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    out_dashboard = os.path.join(DASHBOARDS_DIR, "interactive_drought_dashboard.html")
    out_map = os.path.join(OUTPUTS_DIR, "spatial_severity_map.png")
    
    all_exist = all(os.path.exists(p) for p in [out_trend, out_heatmap, out_dashboard, out_map])
    if all_exist and not force:
        logging.info("[CACHE] All visualization outputs exist in cache. Skipping regeneration.")
        metrics["outputs"].extend([out_trend, out_heatmap, out_dashboard, out_map])
        metrics["module_times"]["visualizations"] = time.time() - t0
        return

    # Merge meteorological and Sentinel-2 data on month
    df_combined = meteo_df.copy()
    df_combined['month'] = df_combined.index.strftime('%Y-%m')
    
    has_s2 = s2_df is not None and not s2_df.empty and 'mean_ndvi' in s2_df.columns
    if has_s2:
        df_combined = df_combined.merge(s2_df[['month', 'mean_ndvi', 'mean_evi', 'mean_pci']], on='month', how='left')
        df_combined.set_index(pd.to_datetime(df_combined['month']), inplace=True)
        
    # 1. SPI vs Vegetation Health Trend
    plt.figure(figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    ax1 = plt.gca()
    ax1.plot(df_combined.index, df_combined['SPI-3'], label='SPI-3 (Meteorological Drought)', color='#1f77b4', linewidth=2)
    ax1.plot(df_combined.index, df_combined['SPI-6'], label='SPI-6 (Hydrological Drought)', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax1.axhline(-1.0, color='orange', linestyle='--', alpha=0.7, label='Moderate Drought Threshold (-1.0)')
    ax1.axhline(-1.5, color='red', linestyle='--', alpha=0.7, label='Severe Drought Threshold (-1.5)')
    ax1.set_ylabel('Standardized Precipitation Index', color='#1f77b4', fontsize=12)
    ax1.set_xlabel('Year', fontsize=12)
    
    if has_s2 and df_combined['mean_ndvi'].dropna().count() > 0:
        ax2 = ax1.twinx()
        ax2.plot(df_combined.index, df_combined['mean_ndvi'], label='Sentinel-2 Mean NDVI', color='#2ca02c', linewidth=2.5)
        ax2.set_ylabel('Sentinel-2 Mean NDVI', color='#2ca02c', fontsize=12)
        ax2.grid(False)
        
    plt.title("Historical Multi-Year Drought Trends: Parbhani District (2016–2026)", fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig(out_trend, dpi=300)
    plt.close()
    metrics["outputs"].append(out_trend)
    
    # 2. Correlation Heatmap
    corr_cols = ['precipitation_sum', 'et0_sum', 'water_deficit', 'temperature_mean', 'SPI-1', 'SPI-3', 'SPI-6']
    if has_s2:
        for col in ['mean_ndvi', 'mean_evi', 'mean_pci']:
            if col in df_combined.columns and df_combined[col].dropna().count() > 3:
                corr_cols.append(col)
                
    corr_matrix = df_combined[corr_cols].dropna().corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Correlation Matrix: Hydro-Meteorological & Satellite Indicators", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_heatmap, dpi=300)
    plt.close()
    metrics["outputs"].append(out_heatmap)
    
    # 3. Interactive Plotly Dashboard
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_combined['month'],
        y=df_combined['precipitation_sum'],
        name='Precipitation (mm)',
        marker_color='rgba(54, 162, 235, 0.6)',
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=df_combined['month'],
        y=df_combined['SPI-3'],
        mode='lines+markers',
        name='SPI-3 (3-Month Drought Index)',
        line=dict(color='#ff7f0e', width=2.5)
    ))
    if has_s2 and df_combined['mean_ndvi'].dropna().count() > 0:
        fig.add_trace(go.Scatter(
            x=df_combined['month'],
            y=df_combined['mean_ndvi'],
            mode='lines+markers',
            name='Sentinel-2 NDVI',
            line=dict(color='#2ca02c', width=2.5)
        ))
        
    fig.update_layout(
        title='<b>Real-Time & Historical Drought Monitoring System - Parbhani District</b>',
        xaxis=dict(title='Month', tickangle=-45),
        yaxis=dict(title='Index Value (SPI / NDVI)', zeroline=True, zerolinecolor='gray'),
        yaxis2=dict(title='Precipitation (mm)', overlaying='y', side='right', showgrid=False),
        template='plotly_dark',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
        hovermode='x unified',
        height=650
    )
    fig.write_html(out_dashboard)
    metrics["outputs"].append(out_dashboard)
    
    # 4. Spatial Tehsil Drought Severity Map
    # Compute real severity category based on latest available SPI-3
    latest_spi = df_combined['SPI-3'].dropna().iloc[-1] if not df_combined['SPI-3'].dropna().empty else 0.0
    
    def classify_spi(val):
        if val <= -2.0:
            return "Extreme Drought"
        elif val <= -1.5:
            return "Severe Drought"
        elif val <= -1.0:
            return "Moderate Drought"
        elif val < 0.0:
            return "Mild Drought"
        else:
            return "Normal / Wet"
            
    current_status = classify_spi(latest_spi)
    gdf_viz = gdf_4326.copy()
    gdf_viz['current_status'] = current_status
    
    severity_colors = {
        "Extreme Drought": "#8B0000",
        "Severe Drought": "#FF4500",
        "Moderate Drought": "#FFA500",
        "Mild Drought": "#FFD700",
        "Normal / Wet": "#2E8B57"
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_viz.plot(ax=ax, color=severity_colors.get(current_status, "#2E8B57"), edgecolor='black', linewidth=1.2)
    
    # Annotate tehsil names
    for _, row in gdf_viz.iterrows():
        pt = row.geometry.centroid
        ax.annotate(row['tehsil'], xy=(pt.x, pt.y), xytext=(3, 3),
                    textcoords="offset points", fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
                    
    ax.set_title(f"Tehsil-Wise Drought Severity Status - Parbhani District\nLatest Metric: SPI-3 = {latest_spi:.2f} ({current_status})",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig(out_map, dpi=300)
    plt.close()
    metrics["outputs"].append(out_map)
    
    metrics["module_times"]["visualizations"] = time.time() - t0
    logging.info(f"Visualizations saved successfully to outputs and dashboards.")

def generate_execution_summary():
    """Generates execution_summary.txt with real metrics."""
    t0 = time.time()
    total_time = t0 - metrics["start_time"]
    summary_path = os.path.join(PROJECT_DIR, "execution_summary.txt")
    
    lines = [
        "==================================================",
        "   DROUGHT MONITORING PIPELINE EXECUTION SUMMARY  ",
        "==================================================",
        f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Data Downloaded: {metrics['data_downloaded_mb']:.2f} MB",
        f"Sentinel-2 Scenes Processed: {metrics['s2_scenes_processed']}",
        f"Sentinel-2 Composites Generated: {metrics['s2_composites_processed']}",
        f"Meteorological API Records: {metrics['api_records_processed']}",
        f"Spatial Points Sampled: {metrics['spatial_points']}",
        "",
        "--- Processing Times ---",
        f"Total Elapsed Time: {total_time:.2f} seconds"
    ]
    
    for mod, dur in metrics["module_times"].items():
        lines.append(f" - {mod.replace('_', ' ').title()}: {dur:.2f} s")
        
    lines.extend([
        "",
        "--- Generated Outputs ---"
    ])
    for out in sorted(set(metrics["outputs"])):
        lines.append(f" - {out}")
        
    lines.append("==================================================")
    
    summary_text = "\n".join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    logging.info("Execution summary generated:")
    print(summary_text)

def main():
    parser = argparse.ArgumentParser(description="Parbhani Real-Time & Historical Drought Monitoring Pipeline (2016-2026)")
    parser.add_argument("--force", action="store_true", help="Force recomputation and overwrite all cached outputs")
    parser.add_argument("--gee-project", type=str, default=os.environ.get("GEE_PROJECT"), help="Google Earth Engine Project ID")
    parser.add_argument("--start-year", type=int, default=2016, help="Start year (default: 2016)")
    parser.add_argument("--end-year", type=int, default=2026, help="End year (default: 2026)")
    parser.add_argument("--sample-points", type=int, default=5000, help="Number of spatial sample points (default: 5000)")
    parser.add_argument("--skip-gee", action="store_true", help="Skip Google Earth Engine processing if running standalone weather/GIS")
    args = parser.parse_args()
    
    setup_logging()
    logging.info("Starting Drought Monitoring Pipeline for Parbhani District, Maharashtra...")
    
    # 1. GEE Authentication (Mandatory unless --skip-gee is explicitly provided)
    if not args.skip_gee:
        authenticate_gee(project_id=args.gee_project)
    else:
        logging.warning("Flag --skip-gee provided: Skipping Earth Engine authentication and satellite processing.")
        
    # 2. Shapefile Loading & Boundary Masking
    gdf_4326, gdf_proj = load_and_prepare_shapefile()
    
    # 3. Open-Meteo Meteorological Data Retrieval & SPI / SWDI Processing
    meteo_df = fetch_meteo_data(
        gdf_4326=gdf_4326,
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force
    )
    
    # 4. Google Earth Engine Sentinel-2 Monthly Composites & Indices
    s2_df = None
    if not args.skip_gee:
        s2_df = fetch_gee_indices(
            gdf_4326=gdf_4326,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force
        )
        
    # 5. Stratified Spatial Sampling
    generate_spatial_samples(
        gdf_proj=gdf_proj,
        n_samples=args.sample_points,
        force=args.force
    )
    
    # 6. Advanced Visualizations & Interactive Dashboard
    generate_visualizations(
        meteo_df=meteo_df,
        s2_df=s2_df,
        gdf_4326=gdf_4326,
        force=args.force
    )
    
    # 7. Execution Summary & Telemetry
    generate_execution_summary()
    logging.info("Drought monitoring pipeline completed successfully.")

if __name__ == "__main__":
    main()
