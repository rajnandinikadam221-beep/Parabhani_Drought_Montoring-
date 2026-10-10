from pathlib import Path
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def generate_stratified_samples(
    gdf_proj: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False
) -> pd.DataFrame:
    """
    Generates spatially stratified random points across each tehsil,
    proportional to area with fixed seed for reproducibility.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_dir = project_root / "outputs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "stratified_sample_points.csv"

    if out_csv.exists() and not force:
        logger.info(f"[CACHE] Found existing sample points at {out_csv}")
        df = pd.read_csv(out_csv)
        telemetry.spatial_points_sampled = len(df)
        telemetry.add_output(out_csv)
        ensure_point_index_values(project_root, telemetry, logger, force=force)
        return df

    n_per_tehsil = config["sampling"]["points_per_tehsil"]
    seed = config["sampling"]["random_seed"]
    min_spacing = float(config["sampling"].get("min_spacing_meters", 100))
    np.random.seed(seed)

    logger.info(f"Generating stratified sample points ({n_per_tehsil}/tehsil, "
                f"seed={seed}, min spacing {min_spacing:.0f} m)...")
    total_area = gdf_proj.geometry.area.sum()
    points = []

    for _, row in gdf_proj.iterrows():
        poly = row["geometry"]
        tehsil = row["tehsil"]
        area_ratio = poly.area / total_area
        n = max(20, int(n_per_tehsil * area_ratio * len(gdf_proj)))

        minx, miny, maxx, maxy = poly.bounds
        accepted = np.empty((0, 2), dtype="float64")
        attempts, max_attempts = 0, n * 120
        while len(accepted) < n and attempts < max_attempts:
            pt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            attempts += 1
            if not pt.within(poly):
                continue
            if accepted.size:
                d = np.hypot(accepted[:, 0] - pt.x, accepted[:, 1] - pt.y)
                if d.min() < min_spacing:
                    continue
            accepted = np.vstack([accepted, [pt.x, pt.y]])
            points.append({"tehsil": tehsil, "x_utm": pt.x, "y_utm": pt.y, "geometry": pt})
        if len(accepted) < n:
            logger.warning(f"{tehsil}: only {len(accepted)}/{n} points after "
                           f"{attempts} attempts (min spacing {min_spacing:.0f} m).")

    samples_gdf = gpd.GeoDataFrame(points, crs=gdf_proj.crs)
    samples_4326 = samples_gdf.to_crs(epsg=4326)
    samples_gdf["lon"] = samples_4326.geometry.x
    samples_gdf["lat"] = samples_4326.geometry.y

    df_out = samples_gdf.drop(columns="geometry")
    df_out.to_csv(out_csv, index=False)
    telemetry.spatial_points_sampled = len(df_out)
    telemetry.add_output(out_csv)

    logger.info(f"Generated {len(df_out)} sample points across {len(gdf_proj)} tehsils → {out_csv}")
    ensure_point_index_values(project_root, telemetry, logger, force=force)
    return df_out


def ensure_point_index_values(
    project_root: Path,
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """
    §6: stacks NDVI/EVI/PCI rasters and extracts their values at every sample
    point for every month, then joins SPI and SSMI from the tehsil zonal table.

    Output: outputs/csv/point_index_values.csv
        (tehsil, month, lon, lat, NDVI, EVI, PCI, SPI-3, SSMI)
    """
    import rasterio

    rasters_dir = project_root / "outputs" / "rasters"
    csv_dir = project_root / "outputs" / "csv"
    points_csv = csv_dir / "stratified_sample_points.csv"
    out_csv = csv_dir / "point_index_values.csv"

    ndvi_files = sorted(rasters_dir.glob("S2_NDVI_*.tif"))
    if not points_csv.exists() or not ndvi_files:
        logger.warning("Point extraction skipped: missing sample points or NDVI rasters.")
        return None

    if out_csv.exists() and not force:
        cache_inputs = list(ndvi_files) + [csv_dir / "tehsil_zonal_indices.csv"]
        newest_input = max(p.stat().st_mtime for p in cache_inputs if p.exists())
        if out_csv.stat().st_mtime >= newest_input:
            logger.info(f"[CACHE] Point index values are up to date: {out_csv.name}")
            telemetry.add_output(out_csv)
            return None

    pts = pd.read_csv(points_csv)
    coords = list(zip(pts["lon"].to_numpy(), pts["lat"].to_numpy()))
    n_pts = len(pts)
    months = [p.name[len("S2_NDVI_"):-len(".tif")] for p in ndvi_files]
    logger.info(f"Extracting stacked index values at {n_pts} points x "
                f"{len(months)} months...")

    base = np.repeat(pts[["tehsil", "lon", "lat"]].to_numpy(), len(months), axis=0)
    month_col = np.tile(np.array(months, dtype=object), n_pts)
    stack = {idx: np.full(n_pts * len(months), np.nan, dtype="float32")
             for idx in ("NDVI", "EVI", "PCI")}

    for i, month_str in enumerate(months):
        slice_idx = slice(i * n_pts, (i + 1) * n_pts)
        for idx in stack:
            path = rasters_dir / f"S2_{idx}_{month_str}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                vals = np.array([v[0] for v in src.sample(coords)], dtype="float32")
            vals[~np.isfinite(vals) | (vals == -9999.0)] = np.nan
            stack[idx][slice_idx] = vals
        if (i + 1) % 40 == 0:
            logger.info(f"  point extraction progress: {i + 1}/{len(months)} months")

    df = pd.DataFrame({
        "tehsil": base[:, 0],
        "month": month_col,
        "lon": base[:, 1].astype("float64"),
        "lat": base[:, 2].astype("float64"),
        **stack,
    })

    zonal_csv = csv_dir / "tehsil_zonal_indices.csv"
    if zonal_csv.exists():
        zonal = pd.read_csv(zonal_csv)
        keep = [c for c in ("tehsil", "month", "SPI-3", "SPI-6", "SSMI") if c in zonal.columns]
        df = df.merge(zonal[keep], on=["tehsil", "month"], how="left")

    df.to_csv(out_csv, index=False)
    telemetry.add_output(out_csv)
    logger.info(f"Point index values saved: {len(df)} rows → {out_csv.name} "
                f"(NDVI valid: {int(df['NDVI'].notna().sum())})")
    return df
