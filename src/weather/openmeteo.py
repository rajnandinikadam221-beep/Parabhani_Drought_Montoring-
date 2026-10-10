from pathlib import Path
import json
import time
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import scipy.stats as st
import geopandas as gpd

def classify_spi(val: float) -> str:
    """Classifies SPI according to WMO / IMD standard thresholds."""
    if pd.isna(val):
        return "Unknown"
    if val >= 2.0:
        return "Extremely wet"
    elif val >= 1.5:
        return "Very wet"
    elif val >= 1.0:
        return "Moderately wet"
    elif val >= -0.99:
        return "Near normal"
    elif val >= -1.49:
        return "Moderate drought"
    elif val >= -1.99:
        return "Severe drought"
    else:
        return "Extreme drought"

def fit_calendar_month_spi(
    monthly_series: pd.Series,
    window: int,
    ref_start: str = "1991-01-01",
    ref_end: str = "2020-12-31"
) -> pd.Series:
    """
    Computes SPI by fitting Gamma distribution per calendar month
    on the 1991-2020 reference period, with zero-rain probability adjustment.
    """
    rolling = monthly_series.rolling(window=window).sum()
    spi_result = pd.Series(index=monthly_series.index, dtype=float)
    
    ref_mask = (rolling.index >= pd.to_datetime(ref_start)) & (rolling.index <= pd.to_datetime(ref_end))
    ref_data = rolling[ref_mask]
    
    # Process per calendar month (1 to 12)
    for month in range(1, 13):
        ref_month_vals = ref_data[ref_data.index.month == month].dropna()
        all_month_mask = (rolling.index.month == month) & (~rolling.isna())
        all_month_indices = rolling[all_month_mask].index
        
        if len(ref_month_vals) < 3:
            continue
            
        n_total = len(ref_month_vals)
        zeros = (ref_month_vals <= 0).sum()
        q = zeros / n_total  # Probability of zero rain
        
        positives = ref_month_vals[ref_month_vals > 0]
        
        try:
            if len(positives) >= 3:
                shape, loc, scale = st.gamma.fit(positives, floc=0)
            else:
                shape, loc, scale = 1.0, 0, 1.0
                
            for idx in all_month_indices:
                val = rolling.loc[idx]
                if val <= 0:
                    prob = q
                else:
                    gamma_cdf = st.gamma.cdf(val, shape, loc=loc, scale=scale)
                    prob = q + (1.0 - q) * gamma_cdf
                prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
                spi_result.loc[idx] = st.norm.ppf(prob)
        except Exception:
            # Fallback to standardized z-score if gamma fitting fails
            m = ref_month_vals.mean()
            s = ref_month_vals.std()
            if s > 0:
                spi_result.loc[all_month_indices] = (rolling.loc[all_month_indices] - m) / s
                
    return spi_result

def fetch_weather_and_compute_spi(
    gdf_4326: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False
) -> pd.DataFrame:
    """
    Retrieves Open-Meteo weather data from 1991 onwards for Parbhani.
    Computes SPI-1, SPI-3, SPI-6, SPI-12 and SSMI against 1991-2020 climatology.
    Saves to outputs/csv/weather_spi.csv.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    csv_dir = project_root / "outputs" / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = csv_dir / "weather_spi.csv"
    raw_cache = raw_dir / "open_meteo_1991_2026.json"
    
    if out_csv.exists() and not force:
        logger.info(f"[CACHE] Found existing weather & SPI dataset at {out_csv}")
        df = pd.read_csv(out_csv, index_col=0, parse_dates=True)
        telemetry.weather_records_processed = len(df) * 30
        telemetry.add_output(out_csv)
        return df
        
    centroid = gdf_4326.geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    
    url = config["weather"]["api_endpoint"]
    start_date = config["dates"]["baseline_start"]
    end_date = min(config["dates"]["study_end"], (datetime.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
    
    logger.info(f"Fetching Open-Meteo weather records from {start_date} to {end_date} (Lon: {lon:.4f}, Lat: {lat:.4f})...")
    
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start_date,
        "end_date": end_date,
        "daily": config["weather"]["daily_variables"],
        "timezone": config["weather"]["timezone"]
    }
    
    # Check raw cache first if available
    data = None
    if raw_cache.exists() and not force:
        logger.info(f"[CACHE] Loading raw Open-Meteo JSON from {raw_cache}")
        with open(raw_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        for attempt in range(3):
            try:
                res = requests.get(url, params=params, timeout=45)
                if res.status_code == 200:
                    telemetry.add_download(len(res.content))
                    data = res.json()
                    with open(raw_cache, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    break
                else:
                    logger.warning(f"Open-Meteo query attempt {attempt+1} failed with status {res.status_code}. Retrying...")
                    time.sleep(2)
            except Exception as req_err:
                logger.warning(f"Connection error: {req_err}. Retrying...")
                time.sleep(2)
                
    if not data or "daily" not in data:
        raise RuntimeError("Failed to fetch weather data from Open-Meteo API.")
        
    df_daily = pd.DataFrame(data["daily"])
    df_daily["time"] = pd.to_datetime(df_daily["time"])
    df_daily.set_index("time", inplace=True)
    
    telemetry.weather_records_processed = len(df_daily)
    
    # Calculate daily variables
    df_daily["temperature_2m_mean"] = (df_daily["temperature_2m_max"] + df_daily["temperature_2m_min"]) / 2.0
    df_daily["water_deficit"] = df_daily["precipitation_sum"] - df_daily["et0_fao_evapotranspiration"]
    
    # Resample to monthly sums / means
    df_monthly = pd.DataFrame()
    df_monthly["precipitation_sum"] = df_daily["precipitation_sum"].resample("ME").sum()
    df_monthly["et0_sum"] = df_daily["et0_fao_evapotranspiration"].resample("ME").sum()
    df_monthly["water_deficit"] = df_daily["water_deficit"].resample("ME").sum()
    df_monthly["temperature_mean"] = df_daily["temperature_2m_mean"].resample("ME").mean()
    
    # Calculate SPI windows on 1991-2020 reference period
    ref_start = config["dates"]["baseline_start"]
    ref_end = config["dates"]["baseline_end"]
    
    for w in config["indices"]["spi_windows"]:
        col_name = f"SPI-{w}"
        df_monthly[col_name] = fit_calendar_month_spi(df_monthly["precipitation_sum"], window=w, ref_start=ref_start, ref_end=ref_end)
        df_monthly[f"{col_name}_class"] = df_monthly[col_name].apply(classify_spi)
        
    # Standardized Soil Moisture / Water Deficit Index (SSMI) per calendar month
    df_monthly["SSMI"] = 0.0
    ref_mask = (df_monthly.index >= pd.to_datetime(ref_start)) & (df_monthly.index <= pd.to_datetime(ref_end))
    ref_wd = df_monthly.loc[ref_mask, "water_deficit"]
    
    for month in range(1, 13):
        m_ref = ref_wd[ref_wd.index.month == month]
        mean_wd = m_ref.mean()
        std_wd = m_ref.std()
        all_month_idx = df_monthly[df_monthly.index.month == month].index
        if std_wd > 0:
            df_monthly.loc[all_month_idx, "SSMI"] = (df_monthly.loc[all_month_idx, "water_deficit"] - mean_wd) / std_wd
            
    df_monthly.to_csv(out_csv)
    telemetry.add_output(out_csv)
    logger.info(f"Successfully calculated SPI and SSMI for {len(df_monthly)} months. Saved to {out_csv}")
    return df_monthly
