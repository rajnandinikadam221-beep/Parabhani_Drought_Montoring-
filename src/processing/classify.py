from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

STAGE_NAMES = {0: "No Drought", 1: "Mild Drought", 2: "Moderate Drought",
               3: "Severe Drought", 4: "Extreme Drought"}
PIXEL_RES_M = 100.0
PIXEL_AREA_KM2 = (PIXEL_RES_M ** 2) / 1e6
NODATA_SENTINEL = -9999.0
RASTER_NODATA = 255


def _cdi_from_components(values: Dict[str, Optional[float]],
                         weights: Dict[str, float]) -> float:
    """Weighted mean of the available standardized components."""
    parts, w_sum = [], 0.0
    mapping = {
        "spi": values.get("SPI-3"),
        "ssmi": values.get("SSMI"),
        "ndvi_vci": values.get("VCI_z"),
        "evi": values.get("EVI_z"),
        "pci": values.get("PCI_z"),
    }
    for key, val in mapping.items():
        if val is not None and not pd.isna(val):
            parts.append(float(val) * weights[key])
            w_sum += weights[key]
    return float(sum(parts) / w_sum) if w_sum > 0 else np.nan


def _stage_from_cdi(cdi: float, thresholds: Dict[str, float]) -> int:
    if pd.isna(cdi):
        return -1
    if cdi <= thresholds["extreme_drought"]:
        return 4
    if cdi <= thresholds["severe_drought"]:
        return 3
    if cdi <= thresholds["moderate_drought"]:
        return 2
    if cdi < thresholds["mild_drought"]:
        return 1
    return 0


def _stage_name(stage: int) -> str:
    return STAGE_NAMES.get(stage, "No Data")


def _dashboard_class(stage: int) -> str:
    if stage >= 2:
        return "Drought"
    if stage == 1:
        return "Normal"
    if stage == 0:
        return "Non-drought"
    return "No Data"


# ---------------------------------------------------------------------------
# Analysis grid and raster helpers
# ---------------------------------------------------------------------------

def _build_grid(gdf_proj: gpd.GeoDataFrame) -> Tuple[int, int, Any]:
    """EPSG:32643 grid, 100 m cells, snapped to whole multiples of the resolution."""
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    res = PIXEL_RES_M
    minx = np.floor(minx / res) * res
    miny = np.floor(miny / res) * res
    maxx = np.ceil(maxx / res) * res
    maxy = np.ceil(maxy / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    return width, height, from_origin(minx, maxy, res, res)


def _reproject_index(path: Path, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Reprojects one downloaded index raster onto the analysis grid (float32, -9999 nodata)."""
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype("float32")
        src_valid = (np.isfinite(src_arr) & (src_arr != NODATA_SENTINEL)).astype("float32")
        dst_arr = np.full(dst_shape, NODATA_SENTINEL, dtype="float32")
        dst_valid = np.zeros(dst_shape, dtype="float32")
        reproject(source=src_arr, destination=dst_arr,
                  src_transform=src.transform, src_crs=src.crs,
                  src_nodata=NODATA_SENTINEL,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  dst_nodata=NODATA_SENTINEL, resampling=Resampling.bilinear)
        reproject(source=src_valid, destination=dst_valid,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    dst_arr[dst_valid < 0.5] = NODATA_SENTINEL
    return dst_arr


def _write_classified(stage_grid: np.ndarray, gdf_proj, transform,
                      classified_dir: Path, month_str: Path) -> None:
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "width": stage_grid.shape[1], "height": stage_grid.shape[0],
        "crs": gdf_proj.crs, "transform": transform, "nodata": RASTER_NODATA,
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(classified_dir / f"DROUGHT_STAGE_{month_str}.tif", "w", **profile) as dst:
        dst.write(stage_grid, 1)


# ---------------------------------------------------------------------------
# Pixel-level CDI / stage computation for one month
# ---------------------------------------------------------------------------

def _compute_month_cdi(
    df_month: pd.DataFrame,
    gdf_proj: gpd.GeoDataFrame,
    grid,
    tehsil_ids: np.ndarray,
    tehsil_index: Dict[str, int],
    veg_arrays: Optional[Dict[str, np.ndarray]],
    clim_lookup: Dict[Tuple[str, int], Dict[str, float]],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cdi_grid float32 with NaN outside/No Data, stage_grid uint8).
    Components: SPI-3 (0.35), SSMI (0.25), VCI_z (0.25), EVI_z (0.10), PCI_z (0.05),
    each standardised and renormalised over whatever is available per pixel.
    """
    height, width = tehsil_ids.shape
    n_par = len(gdf_proj) + 1

    spi_v = np.full(n_par, np.nan, dtype="float64")
    ssmi_v = np.full(n_par, np.nan, dtype="float64")
    evi_z_v = np.full(n_par, np.nan, dtype="float64")
    pci_z_v = np.full(n_par, np.nan, dtype="float64")
    ndvi_min_v = np.full(n_par, np.nan, dtype="float64")
    ndvi_max_v = np.full(n_par, np.nan, dtype="float64")

    for _, row in df_month.iterrows():
        tid = tehsil_index.get(row["tehsil"])
        if tid is None:
            continue
        spi_v[tid] = row.get("SPI-3", np.nan)
        ssmi_v[tid] = row.get("SSMI", np.nan)
        evi_z_v[tid] = row.get("EVI_z", np.nan)
        pci_z_v[tid] = row.get("PCI_z", np.nan)
        ndvi_min_v[tid] = row.get("ndvi_min", np.nan)
        ndvi_max_v[tid] = row.get("ndvi_max", np.nan)

    spi_p = spi_v[tehsil_ids]
    ssmi_p = ssmi_v[tehsil_ids]
    evi_z_p = evi_z_v[tehsil_ids]
    pci_z_p = pci_z_v[tehsil_ids]

    num = np.zeros(tehsil_ids.shape, dtype="float64")
    den = np.zeros(tehsil_ids.shape, dtype="float64")

    def _accum(component: np.ndarray, w: float) -> None:
        mask = np.isfinite(component)
        num[mask] += component[mask] * w
        den[mask] += w

    _accum(spi_p, weights["spi"])
    _accum(ssmi_p, weights["ssmi"])

    if veg_arrays is not None:
        ndvi_a = veg_arrays["NDVI"]
        evi_a = veg_arrays["EVI"]
        pci_a = veg_arrays["PCI"]
        # -9999 nodata must never enter the anomaly maths
        ndvi_valid = ndvi_a != NODATA_SENTINEL
        evi_a = np.where(evi_a == NODATA_SENTINEL, np.nan, evi_a)
        pci_a = np.where(pci_a == NODATA_SENTINEL, np.nan, pci_a)

        # pixel VCI: (NDVI-min)/(max-min)*100, then standardised to VCI_z
        span = ndvi_max_v[tehsil_ids] - ndvi_min_v[tehsil_ids]
        vci = np.full(tehsil_ids.shape, np.nan, dtype="float64")
        np.divide((ndvi_a.astype("float64") - ndvi_min_v[tehsil_ids]) * 100.0,
                  span, out=vci, where=(span > 1e-6) & ndvi_valid)
        vci_z = (vci - 50.0) / 25.0
        _accum(vci_z, weights["ndvi_vci"])

        month_num = int(str(df_month["month"].iloc[0])[5:7])
        evi_m = np.full(n_par, np.nan); evi_s = np.full(n_par, np.nan)
        pci_m = np.full(n_par, np.nan); pci_s = np.full(n_par, np.nan)
        for t_name, tid in tehsil_index.items():
            c = clim_lookup.get((t_name, month_num), {})
            evi_m[tid] = c.get("EVI_m", np.nan); evi_s[tid] = c.get("EVI_s", np.nan)
            pci_m[tid] = c.get("PCI_m", np.nan); pci_s[tid] = c.get("PCI_s", np.nan)

        def _pixel_z(arr, mean_v, std_v):
            mean_p = mean_v[tehsil_ids]
            std_p = std_v[tehsil_ids]
            return (arr.astype("float64") - mean_p) / np.where(std_p > 0, std_p, np.nan)

        # nodata / zero-variance pixels stay NaN and drop out of the weighted mean
        evi_z_px = _pixel_z(evi_a, evi_m, evi_s)
        pci_z_px = _pixel_z(pci_a, pci_m, pci_s)
        _accum(evi_z_px, weights["evi"])
        _accum(pci_z_px, weights["pci"])

    with np.errstate(invalid="ignore", divide="ignore"):
        cdi = np.where(den > 0, num / den, np.nan)
    cdi[tehsil_ids == 0] = np.nan

    stage = np.select(
        [cdi <= -2.0, cdi <= -1.5, cdi <= -1.0, cdi < 0.0, np.isfinite(cdi)],
        [4, 3, 2, 1, 0], default=RASTER_NODATA).astype("uint8")
    stage[tehsil_ids == 0] = RASTER_NODATA
    return cdi.astype("float32"), stage


# ---------------------------------------------------------------------------
# Area accounting
# ---------------------------------------------------------------------------

def _stage_areas_from_raster(stage_grid: np.ndarray, gdf_proj: gpd.GeoDataFrame,
                             transform) -> pd.DataFrame:
    """Per-tehsil stage area in km² and % counted from the classified raster."""
    rows = []
    for _, row in gdf_proj.iterrows():
        try:
            tehsil_mask = geometry_mask([row.geometry.__geo_interface__],
                                        out_shape=stage_grid.shape,
                                        transform=transform,
                                        invert=True, all_touched=False)
        except ValueError:
            tehsil_mask = np.zeros(stage_grid.shape, dtype=bool)
        vals = stage_grid[tehsil_mask]
        total_px = int((vals != RASTER_NODATA).sum())
        total_area = total_px * PIXEL_AREA_KM2
        for st in range(5):
            px = int((vals == st).sum())
            area = px * PIXEL_AREA_KM2
            pct = (area / total_area * 100.0) if total_area > 0 else 0.0
            rows.append({"tehsil": row["tehsil"], "stage": st,
                         "stage_name": _stage_name(st),
                         "area_km2": round(area, 3),
                         "pct_area": round(pct, 2)})
    return pd.DataFrame(rows)


def _stage_areas_scalar(df_month: pd.DataFrame,
                        gdf_proj: gpd.GeoDataFrame) -> pd.DataFrame:
    """Whole-tehsil fallback used when a classified raster is missing."""
    rows = []
    areas = {row["tehsil"]: row.geometry.area / 1e6 for _, row in gdf_proj.iterrows()}
    for _, r in df_month.iterrows():
        area = areas.get(r["tehsil"], np.nan)
        for st in range(5):
            match = (r["stage"] == st)
            rows.append({"tehsil": r["tehsil"], "stage": st,
                         "stage_name": _stage_name(st),
                         "area_km2": round(area, 3) if match else 0.0,
                         "pct_area": 100.0 if match else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------

def compute_drought_stages_and_cdi(
    df_zonal: pd.DataFrame,
    gdf_proj: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False,
) -> pd.DataFrame:
    """
    Monthly-climatology anomalies per tehsil, VCI, pixel-level Combined
    Drought Index, drought stages 0-4, dashboard classes, classified stage
    rasters (EPSG:32643, 100 m) and per-stage affected area per tehsil.

    Outputs:
      outputs/csv/tehsil_drought_stages.csv
      outputs/csv/stage_area_summary.csv
      outputs/rasters/classified/DROUGHT_STAGE_YYYY-MM.tif
      outputs/reports/drought_dashboard_data.json
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_dir = project_root / "outputs" / "csv"
    rasters_dir = project_root / "outputs" / "rasters"
    classified_dir = rasters_dir / "classified"
    reports_dir = project_root / "outputs" / "reports"
    for d in (csv_dir, classified_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / "tehsil_drought_stages.csv"
    area_csv = csv_dir / "stage_area_summary.csv"
    out_json = reports_dir / "drought_dashboard_data.json"
    zonal_csv = csv_dir / "tehsil_zonal_indices.csv"

    if out_csv.exists() and out_json.exists() and area_csv.exists() and not force:
        newest_input = max(
            (p.stat().st_mtime for p in [zonal_csv, *rasters_dir.glob("S2_NDVI_*.tif")]
             if p.exists()), default=0)
        if newest_input <= out_csv.stat().st_mtime:
            logger.info(f"[CACHE] Drought stages are up to date: {out_csv}")
            telemetry.add_output(out_csv)
            telemetry.add_output(area_csv)
            telemetry.add_output(out_json)
            return pd.read_csv(out_csv)
        logger.info("Inputs changed since the last classification — recomputing.")

    logger.info("Computing climatology anomalies, CDI and classification stages...")
    weights = config["indices"]["cdi_weights"]
    thresholds = config["indices"]["thresholds"]

    df = df_zonal.copy()
    df["month"] = df["month"].astype(str)
    df["month_num"] = pd.to_datetime(df["month"]).dt.month

    # --- 1. VCI: per-tehsil study-period NDVI min/max -----------------------
    if "NDVI" in df.columns and df["NDVI"].notna().any():
        gmin = df.groupby("tehsil")["NDVI"].transform("min")
        gmax = df.groupby("tehsil")["NDVI"].transform("max")
        span = (gmax - gmin).replace(0, np.nan)
        df["ndvi_min"] = gmin
        df["ndvi_max"] = gmax
        df["VCI"] = ((df["NDVI"] - gmin) / span * 100.0).round(2)
    else:
        df["ndvi_min"] = np.nan
        df["ndvi_max"] = np.nan
        df["VCI"] = np.nan

    # --- 2. Monthly-climatology z-scores per tehsil -------------------------
    for col in ["EVI", "PCI", "NDVI", "SSMI", "SPI-3"]:
        if col in df.columns:
            mean = df.groupby(["tehsil", "month_num"])[col].transform("mean")
            std = df.groupby(["tehsil", "month_num"])[col].transform(lambda s: s.std(ddof=0))
            df[f"{col}_z"] = ((df[col] - mean) / std.replace(0, np.nan)).round(3)
        else:
            df[f"{col}_z"] = np.nan
    df["VCI_z"] = ((df["VCI"] - 50.0) / 25.0).round(3)

    # --- 3. Tehsil-level scalar CDI (baseline + cloud-gap fallback) ---------
    df["CDI_scalar"] = df.apply(
        lambda r: _cdi_from_components(
            {"SPI-3": r.get("SPI-3"), "SSMI": r.get("SSMI"), "VCI_z": r.get("VCI_z"),
             "EVI_z": r.get("EVI_z"), "PCI_z": r.get("PCI_z")}, weights),
        axis=1).round(3)

    # --- 4. Pixel-level CDI + classified rasters ----------------------------
    grid = _build_grid(gdf_proj)
    width, height, transform = grid
    tehsil_ids = rasterize(
        ((row.geometry, i + 1) for i, (_, row) in enumerate(gdf_proj.iterrows())),
        out_shape=(height, width), transform=transform, fill=0, dtype="int16")
    tehsil_index = {name: i + 1 for i, name in enumerate(gdf_proj["tehsil"])}
    logger.info(f"Analysis grid: {width}x{height} cells @ {PIXEL_RES_M:.0f} m "
                f"({gdf_proj.crs}), {int((tehsil_ids > 0).sum())} in-district cells")

    clim_lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
    for (t_name, mn), grp in df.groupby(["tehsil", "month_num"]):
        clim_lookup[(t_name, int(mn))] = {
            "EVI_m": grp["EVI"].mean() if "EVI" in grp else np.nan,
            "EVI_s": grp["EVI"].std(ddof=0) if "EVI" in grp else np.nan,
            "PCI_m": grp["PCI"].mean() if "PCI" in grp else np.nan,
            "PCI_s": grp["PCI"].std(ddof=0) if "PCI" in grp else np.nan,
        }

    months = sorted(df["month"].unique())
    veg_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    cdi_means: Dict[Tuple[str, str], float] = {}
    rasters_written = 0

    for month_str in months:
        df_month = df[df["month"] == month_str]
        paths = {b: rasters_dir / f"S2_{b}_{month_str}.tif"
                 for b in ("NDVI", "EVI", "PCI")}
        if all(p.exists() for p in paths.values()):
            if month_str not in veg_cache:
                veg_cache[month_str] = {
                    b: _reproject_index(paths[b], (height, width), transform, gdf_proj.crs)
                    for b in ("NDVI", "EVI", "PCI")}
            veg_arrays = veg_cache[month_str]
        else:
            veg_arrays = None

        cdi_grid, stage_grid = _compute_month_cdi(
            df_month, gdf_proj, grid, tehsil_ids, tehsil_index,
            veg_arrays, clim_lookup, weights, )
        _write_classified(stage_grid, gdf_proj, transform, classified_dir, month_str)
        rasters_written += 1

        for _, row in df_month.iterrows():
            tid = tehsil_index.get(row["tehsil"])
            px = cdi_grid[(tehsil_ids == tid) & np.isfinite(cdi_grid)] if tid else np.array([])
            cdi_means[(row["tehsil"], month_str)] = (
                float(np.mean(px)) if px.size >= 10 else float(row["CDI_scalar"]))

    df["CDI"] = [
        round(cdi_means.get((t, m), c), 3)
        for t, m, c in zip(df["tehsil"], df["month"], df["CDI_scalar"])]
    df["stage"] = df["CDI"].apply(lambda c: _stage_from_cdi(c, thresholds))
    df["stage_name"] = df["stage"].apply(_stage_name)
    df["dashboard_class"] = df["stage"].apply(_dashboard_class)

    tehsil_areas = {row["tehsil"]: row.geometry.area / 1e6
                    for _, row in gdf_proj.iterrows()}
    total_district_area = gdf_proj.geometry.area.sum() / 1e6
    df["tehsil_area_km2"] = df["tehsil"].map(tehsil_areas).round(2)

    df.drop(columns=[c for c in ("month_num", "CDI_scalar") if c in df.columns],
            inplace=True)
    df.to_csv(out_csv, index=False)
    telemetry.add_output(out_csv)
    logger.info(f"Drought stages saved: {len(df)} rows "
                f"(stage distribution: {df['stage_name'].value_counts().to_dict()})")

    # --- 5. Per-stage affected area ----------------------------------------
    area_frames = []
    for month_str in months:
        tif = classified_dir / f"DROUGHT_STAGE_{month_str}.tif"
        df_month = df[df["month"] == month_str]
        if tif.exists():
            with rasterio.open(tif) as src:
                frame = _stage_areas_from_raster(src.read(1), gdf_proj, src.transform)
        else:
            frame = _stage_areas_scalar(df_month, gdf_proj)
        frame.insert(1, "month", month_str)
        area_frames.append(frame)
    area_df = pd.concat(area_frames, ignore_index=True) if area_frames else pd.DataFrame()
    if not area_df.empty:
        area_df.to_csv(area_csv, index=False)
        telemetry.add_output(area_csv)
        logger.info(f"Stage area summary saved: {len(area_df)} rows → {area_csv}")

    # --- 6. Dashboard JSON ---------------------------------------------------
    latest = df["month"].max()
    json_data = {
        "metadata": {
            "region": "Parbhani District, Maharashtra",
            "total_area_km2": round(total_district_area, 2),
            "tehsils_count": len(tehsil_areas),
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "tehsils": list(tehsil_areas.keys()),
        "timeline": months,
        "latest_status": {
            "month": latest,
            "records": df[df["month"] == latest][
                ["tehsil", "CDI", "stage", "stage_name", "dashboard_class",
                 "tehsil_area_km2"]].to_dict(orient="records"),
        },
        "latest_stage_area_km2": (
            area_df[area_df["month"] == latest]
            .groupby("stage_name")["area_km2"].sum().round(2).to_dict()
            if not area_df.empty else {}),
        "time_series": df.groupby("month").agg({
            "precipitation_sum": "mean",
            "SPI-3": "mean",
            "CDI": "mean",
        }).round(2).reset_index().to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    telemetry.add_output(out_json)
    logger.info(f"Dashboard JSON saved to {out_json} "
                f"({rasters_written} classified rasters in {classified_dir})")
    return df
