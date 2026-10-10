from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

INDEX_NAMES = ["NDVI", "EVI", "PCI"]
STAT_COLS = ["mean", "median", "std", "min", "max", "valid_pixel_count"]
NODATA_SENTINEL = -9999.0


def _load_raster_means_for_tehsil(src: rasterio.DatasetReader, geom) -> Optional[Dict[str, float]]:
    """Zonal statistics of one raster over one polygon (numpy/rasterio only)."""
    try:
        out_img, _ = rio_mask(src, [geom.__geo_interface__], crop=True,
                              filled=False, all_touched=False)
    except ValueError:
        # Polygon does not overlap raster bounds
        return None
    arr = out_img[0]
    data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
    data = data[np.isfinite(data) & (data != NODATA_SENTINEL)]
    if data.size == 0:
        return None
    return {
        "mean": round(float(data.mean()), 4),
        "median": round(float(np.median(data)), 4),
        "std": round(float(data.std(ddof=0)), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "valid_pixel_count": int(data.size),
    }


def _compute_index_zonal_tables(
    gdf_4326: gpd.GeoDataFrame,
    rasters_dir: Path,
    months: List[str],
    logger: logging.Logger,
) -> Dict[str, pd.DataFrame]:
    """Returns {INDEX: DataFrame(tehsil, month, mean, median, std, min, max, count)}."""
    tehsils = sorted(gdf_4326["tehsil"].unique())
    geoms = {row["tehsil"]: row.geometry for _, row in gdf_4326.iterrows()}

    tables: Dict[str, List[dict]] = {idx: [] for idx in INDEX_NAMES}
    for i, month_str in enumerate(months, start=1):
        for idx in INDEX_NAMES:
            path = rasters_dir / f"S2_{idx}_{month_str}.tif"
            if not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    for tehsil in tehsils:
                        stats = _load_raster_means_for_tehsil(src, geoms[tehsil])
                        row = {"tehsil": tehsil, "month": month_str}
                        if stats is None:
                            row.update({c: np.nan for c in STAT_COLS})
                        else:
                            row.update(stats)
                        tables[idx].append(row)
            except Exception:  # noqa: BLE001 - one bad raster must not kill the stage
                logger.exception(f"Zonal statistics failed for {path.name}")
        if i % 25 == 0:
            logger.info(f"  zonal statistics progress: {i}/{len(months)} months")

    return {idx: pd.DataFrame(rows) for idx, rows in tables.items()}


def _merge_ssmi(gdf_4326: gpd.GeoDataFrame, weather_study: pd.DataFrame,
                logger: logging.Logger, project_root: Path) -> pd.DataFrame:
    """
    Per-tehsil SSMI from SMAP when available, else district weather-based SSMI.
    Returns DataFrame(tehsil, month, SSMI, ssmi_source).
    """
    months = weather_study["month"].tolist()
    tehsils = sorted(gdf_4326["tehsil"].unique())
    panel = pd.MultiIndex.from_product([tehsils, months],
                                       names=["tehsil", "month"]).to_frame(index=False)

    district = weather_study.set_index("month")["SSMI"]
    panel["SSMI"] = panel["month"].map(district).astype(float)
    panel["ssmi_source"] = np.where(panel["SSMI"].notna(), "water_deficit", "none")

    smap_csv = project_root / "outputs" / "csv" / "soil_moisture_ssmi.csv"
    if smap_csv.exists():
        try:
            smap = pd.read_csv(smap_csv)
            panel = panel.merge(
                smap[["tehsil", "month", "SSMI"]].rename(columns={"SSMI": "SSMI_smap"}),
                on=["tehsil", "month"], how="left")
            has_smap = panel["SSMI_smap"].notna()
            panel.loc[has_smap, "SSMI"] = panel.loc[has_smap, "SSMI_smap"]
            panel.loc[has_smap, "ssmi_source"] = "smap"
            panel.drop(columns=["SSMI_smap"], inplace=True)
            filled = int((~has_smap).sum())
            if filled:
                logger.info(f"SMAP unavailable for {filled} tehsil-months — "
                            f"filled with weather-based SSMI.")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read SMAP soil moisture table; "
                             "using weather-based SSMI only.")
    else:
        logger.warning("No SMAP soil moisture table found — using weather-based SSMI.")

    panel["SSMI"] = panel["SSMI"].round(3)
    return panel


def process_monthly_indices(
    gdf_4326: gpd.GeoDataFrame,
    df_weather: pd.DataFrame,
    df_s2: Optional[pd.DataFrame],
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False,
) -> pd.DataFrame:
    """
    Per-tehsil zonal statistics of NDVI/EVI/PCI from the downloaded monthly
    rasters, merged with district weather (SPI, precipitation, temperature)
    and per-tehsil SSMI into one tidy table keyed by tehsil and month.

    Outputs:
      outputs/csv/zonal_ndvi.csv | zonal_evi.csv | zonal_pci.csv
          (tehsil, month, mean, median, std, min, max, valid_pixel_count)
      outputs/csv/tehsil_zonal_indices.csv  (wide, one row per tehsil x month)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_dir = project_root / "outputs" / "csv"
    rasters_dir = project_root / "outputs" / "rasters"
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / "tehsil_zonal_indices.csv"
    index_csvs = {idx: csv_dir / f"zonal_{idx.lower()}.csv" for idx in INDEX_NAMES}

    # --- staleness check: any input newer than the outputs forces recompute ---
    raster_files = sorted(rasters_dir.glob("S2_NDVI_*.tif"))
    if not force and out_csv.exists() and all(p.exists() for p in index_csvs.values()):
        if not raster_files:
            logger.info("[CACHE] No Sentinel-2 rasters found; keeping existing zonal table.")
            telemetry.add_output(out_csv)
            return pd.read_csv(out_csv)
        input_files = list(raster_files)
        for extra in (csv_dir / "soil_moisture_ssmi.csv", csv_dir / "weather_spi.csv"):
            if extra.exists():
                input_files.append(extra)
        newest_input = max(p.stat().st_mtime for p in input_files)
        oldest_output = min(p.stat().st_mtime for p in [out_csv, *index_csvs.values()])
        if newest_input <= oldest_output:
            logger.info(f"[CACHE] Zonal index tables are newer than all inputs: {out_csv}")
            telemetry.add_output(out_csv)
            for p in index_csvs.values():
                telemetry.add_output(p)
            return pd.read_csv(out_csv)
        logger.info("Inputs changed since the last zonal run — recomputing.")

    if not raster_files:
        logger.warning("No Sentinel-2 rasters available — zonal satellite columns "
                       "will be NaN. Run the GEE stage first.")
    months = sorted({p.name[len("S2_NDVI_"):-len(".tif")] for p in raster_files})
    logger.info(f"Computing per-tehsil zonal statistics for {len(months)} months x "
                f"{len(gdf_4326)} tehsils ({', '.join(INDEX_NAMES)})...")

    tables = _compute_index_zonal_tables(gdf_4326, rasters_dir, months, logger)

    for idx, df_idx in tables.items():
        path = index_csvs[idx]
        df_idx.to_csv(path, index=False)
        telemetry.add_output(path)
        valid = int(df_idx["valid_pixel_count"].notna().sum())
        logger.info(f"  {idx}: {len(df_idx)} rows ({valid} valid tehsil-months) → {path.name}")

    # --- wide tidy table: weather + SSMI + satellite indices ------------------
    # Window on year-month strings (not timestamps): the last weather month is
    # labelled with its month-end date, which would otherwise fall outside a
    # study_end like 2026-10-06 and silently drop the latest partial month.
    start_key = pd.to_datetime(config["dates"]["study_start"]).strftime("%Y-%m")
    end_key = pd.to_datetime(config["dates"]["study_end"]).strftime("%Y-%m")
    weather = df_weather.copy()
    weather_keys = weather.index.strftime("%Y-%m")
    weather = weather[(weather_keys >= start_key) & (weather_keys <= end_key)]
    weather["month"] = weather.index.strftime("%Y-%m")
    weather = weather.reset_index(drop=True)

    tehsils = sorted(gdf_4326["tehsil"].unique())
    panel = pd.MultiIndex.from_product([tehsils, weather["month"]],
                                       names=["tehsil", "month"]).to_frame(index=False)

    weather_cols = ["precipitation_sum", "et0_sum", "temperature_mean",
                    "SPI-1", "SPI-3", "SPI-6", "SPI-12"]
    panel = panel.merge(weather[["month"] + [c for c in weather_cols if c in weather.columns]],
                        on="month", how="left")

    ssmi_df = _merge_ssmi(gdf_4326, weather, logger, project_root)
    panel = panel.merge(ssmi_df, on=["tehsil", "month"], how="left")

    for idx in INDEX_NAMES:
        df_idx = tables[idx]
        if df_idx.empty:
            panel[idx] = np.nan
            continue
        keep = df_idx[["tehsil", "month", "mean"]].rename(columns={"mean": idx})
        panel = panel.merge(keep, on=["tehsil", "month"], how="left")

    for col in ["SPI-1", "SPI-3", "SPI-6", "SPI-12"]:
        if col in panel.columns:
            panel[col] = panel[col].round(3)

    panel.sort_values(by=["tehsil", "month"], inplace=True)
    panel.to_csv(out_csv, index=False)
    telemetry.add_output(out_csv)

    coverage = {idx: int(panel[idx].notna().sum()) for idx in INDEX_NAMES}
    logger.info(f"Generated {len(panel)} zonal records across {len(tehsils)} tehsils "
                f"(satellite coverage: {coverage}). Saved to {out_csv}")
    return panel
