from pathlib import Path
import logging
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import ee
import rasterio
from rasterio.mask import mask as rio_mask

CS_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
INDEX_BANDS = ["NDVI", "EVI", "PCI"]
NODATA_VALUE = -9999.0
MAX_DOWNLOAD_ATTEMPTS = 4


def shapely_to_ee(geom) -> ee.Geometry:
    """Converts a Shapely geometry into an Earth Engine geometry."""
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon([list(geom.exterior.coords)])
    elif geom.geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon([list(p.exterior.coords) for p in geom.geoms])
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def _district_geometry(gdf_4326: gpd.GeoDataFrame):
    """Union of all tehsil geometries (handles geopandas version differences)."""
    try:
        return gdf_4326.geometry.union_all()
    except AttributeError:
        return gdf_4326.geometry.unary_union


def _apply_scene_mask(col: ee.ImageCollection, tier: str, use_cs: bool,
                      cs_threshold: float, start_date: str, end_date: str) -> ee.ImageCollection:
    """
    Masks a scene collection.
    Priority: Cloud Score+ (cs_cdf >= threshold) joined by system:index;
    fallback: SCL band for SR scenes (mask clouds/shadows/cirrus/snow/water),
    QA60 bits for TOA scenes.
    The CS+ side is filtered by the month window strings — deriving the window
    from aggregate_min/max of system:time_start collapses for months whose
    scenes all fall on one day and can empty the join.
    """
    if use_cs:
        cs_col = (ee.ImageCollection(CS_PLUS_COLLECTION)
                  .filterBounds(col.geometry())
                  .filterDate(start_date, end_date))
        joined = ee.Join.saveFirst("cs").apply(
            col, cs_col, ee.Filter.equals(leftField="system:index", rightField="system:index"))

        def _mask_cs(img: ee.Image) -> ee.Image:
            cs = ee.Image(img.get("cs")).select("cs_cdf")
            out = img.updateMask(cs.gte(cs_threshold))
            if tier == "SR":
                # also drop standing water (SCL class 6) from vegetation indices
                out = out.updateMask(img.select("SCL").neq(6))
            return out

        return ee.ImageCollection(joined).map(_mask_cs)

    if tier == "SR":
        def _mask_scl(img: ee.Image) -> ee.Image:
            scl = img.select("SCL")
            # keep: 2 dark, 4 vegetation, 5 bare soil, 7 unclassified
            keep = scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5)).Or(scl.eq(7))
            return img.updateMask(keep)

        return col.map(_mask_scl)

    def _mask_qa60(img: ee.Image) -> ee.Image:
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).neq(0)
        cirrus = qa.bitwiseAnd(1 << 11).neq(0)
        return img.updateMask(cloud.Not().And(cirrus.Not()))

    return col.map(_mask_qa60)


def _build_month_composite(aoi: ee.Geometry, start_date: str, end_date: str,
                           config: dict) -> Tuple[ee.Image, int, str, str]:
    """
    Returns (3-band NDVI/EVI/PCI composite image, scene count, tier, mask_method).
    Raises ValueError when the month holds no usable scenes.
    """
    max_cloud = config["gee"].get("max_cloud_percentage", 60)
    fallback_cloud = config["gee"].get("max_cloud_fallback", 80)
    cs_threshold = config["gee"].get("cloud_score_threshold", 0.6)

    # Scene selection: SR first, then TOA; each tier first at the configured
    # cloud threshold, then at the relaxed fallback (pixel-level CS+ masking
    # still removes clouds afterwards).
    col = None
    scene_count, tier, used_cloud = 0, "SR", max_cloud
    for cloud_max in dict.fromkeys([max_cloud, fallback_cloud]):
        for cid, tname in ((config["gee"]["s2_sr_collection"], "SR"),
                           (config["gee"]["s2_toa_collection"], "TOA")):
            cand = (ee.ImageCollection(cid)
                    .filterBounds(aoi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max)))
            n = int(cand.size().getInfo())
            if n > 0:
                col, scene_count, tier, used_cloud = cand, n, tname, cloud_max
                break
        if col is not None:
            break

    if col is None:
        raise ValueError("no_scenes")

    # Cloud Score+ availability for this month (covers S2 from 2015 onwards).
    # Only trusted when CS+ covers at least every scene we selected; otherwise
    # some joined images would lack the cs_cdf band and break the mask.
    cs_count = int(ee.ImageCollection(CS_PLUS_COLLECTION)
                   .filterBounds(aoi).filterDate(start_date, end_date).size().getInfo())
    use_cs = cs_count >= scene_count
    mask_method = "cloud_score_plus" if use_cs else ("SCL" if tier == "SR" else "QA60")
    if cs_count and not use_cs:
        mask_method += "_fallback"
    if used_cloud != max_cloud:
        mask_method += f"_cloud{used_cloud}"

    col = _apply_scene_mask(col, tier, use_cs, cs_threshold, start_date, end_date)

    # ee.Join.saveFirst is an inner join: a CS+ mismatch can empty the
    # collection, and EE would only surface that at download time as a
    # band-less composite. Catch it here instead.
    if int(col.size().getInfo()) == 0:
        raise ValueError("unusable")

    composite = col.median().clip(aoi)
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = composite.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))",
        {"NIR": composite.select("B8"), "RED": composite.select("B4"),
         "BLUE": composite.select("B2")}).rename("EVI")
    pci = composite.expression("(NIR / RE1) - 1",
                               {"NIR": composite.select("B8"),
                                "RE1": composite.select("B5")}).rename("PCI")
    stack = ee.Image.cat([ndvi, evi, pci]).toFloat()
    return stack, scene_count, tier, mask_method


def _download_stack(stack: ee.Image, aoi: ee.Geometry, scale: int,
                    logger: logging.Logger, month_str: str) -> bytes:
    """Downloads the 3-band index stack with retry + exponential backoff."""
    url = stack.getDownloadURL({
        "scale": scale,
        "crs": "EPSG:4326",
        "region": aoi,
        "format": "GEO_TIFF",
    })
    last_err = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            res = requests.get(url, timeout=180)
            if res.status_code == 200 and len(res.content) > 1000:
                return res.content
            last_err = f"HTTP {res.status_code}, {len(res.content)} bytes"
        except Exception as exc:  # noqa: BLE001 - transient network failures
            last_err = str(exc)
        wait = 5 * (2 ** (attempt - 1))
        logger.warning(f"Download attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed for "
                       f"{month_str}: {last_err}. Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError(f"GeoTIFF download failed for {month_str}: {last_err}")


def _split_stack_to_index_rasters(content: bytes, stack_path: Path,
                                  rasters_dir: Path, month_str: str) -> Dict[str, Path]:
    """
    Splits the downloaded 3-band stack into one GeoTIFF per index.
    EE writes masked pixels as -inf; normalises them to -9999 nodata.
    """
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.write_bytes(content)

    written: Dict[str, Path] = {}
    with rasterio.open(stack_path) as src:
        if src.count < len(INDEX_BANDS):
            raise RuntimeError(
                f"Stack for {month_str} has {src.count} bands, expected {len(INDEX_BANDS)}")
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=NODATA_VALUE,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        for i, band_name in enumerate(INDEX_BANDS, start=1):
            arr = src.read(i).astype("float32")
            arr[~np.isfinite(arr)] = NODATA_VALUE
            out_path = rasters_dir / f"S2_{band_name}_{month_str}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)
            written[band_name] = out_path
    return written


def _raster_valid_mask(arr: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    return valid


def _district_means_from_rasters(raster_paths: Dict[str, Path],
                                 district_geom, logger: logging.Logger) -> Dict[str, float]:
    """District-wide mean of each index computed locally from the downloaded rasters."""
    stats: Dict[str, float] = {}
    geoms = [district_geom.__geo_interface__]
    for band_name, path in raster_paths.items():
        try:
            with rasterio.open(path) as src:
                out_img, _ = rio_mask(src, geoms, crop=True, filled=False, all_touched=True)
                arr = out_img[0]
                data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
                data = data[np.isfinite(data) & (data != NODATA_VALUE)]
                stats[f"mean_{band_name.lower()}"] = (
                    round(float(data.mean()), 4) if data.size else np.nan)
        except Exception as exc:  # noqa: BLE001 - stats must never abort the pipeline
            logger.warning(f"Local zonal mean failed for {band_name} @ {path.name}: {exc}")
            stats[f"mean_{band_name.lower()}"] = np.nan
    return stats


def download_sentinel2_composites(
    gdf_4326: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False,
) -> pd.DataFrame:
    """
    Builds monthly median Sentinel-2 composites (SR with Cloud Score+ masking,
    TOA fallback), exports NDVI/EVI/PCI, splits them into one GeoTIFF per index
    clipped to Parbhani, and records district means in a CSV.
    Months whose 3 index rasters already exist are skipped unless force=True.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    rasters_dir = project_root / "outputs" / "rasters"
    stack_dir = rasters_dir / "stack"
    csv_dir = project_root / "outputs" / "csv"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / "sentinel2_indices.csv"
    existing_records: Dict[str, dict] = {}
    if out_csv.exists() and not force:
        logger.info(f"[CACHE] Found existing Sentinel-2 indices CSV at {out_csv}")
        df_old = pd.read_csv(out_csv)
        for _, r in df_old.iterrows():
            existing_records[str(r["month"])] = r.to_dict()

    district_geom = _district_geometry(gdf_4326)
    ee_aoi = shapely_to_ee(district_geom)

    study_start = datetime.strptime(config["dates"]["study_start"], "%Y-%m-%d")
    study_end = datetime.strptime(config["dates"]["study_end"], "%Y-%m-%d")

    months: List[Tuple[int, int]] = []
    curr = study_start.replace(day=1)
    while curr <= study_end:
        months.append((curr.year, curr.month))
        curr = curr.replace(year=curr.year + 1, month=1) if curr.month == 12 \
            else curr.replace(month=curr.month + 1)

    logger.info(f"Sentinel-2 processing pipeline targeted for {len(months)} months "
                f"(Cloud Score+ threshold {config['gee'].get('cloud_score_threshold', 0.6)}, "
                f"scene cloud filter < {config['gee'].get('max_cloud_percentage', 60)}%).")

    records = list(existing_records.values())
    new_composites = 0
    scale = config["gee"].get("default_scale", 100)
    mask_methods_used: Dict[str, int] = {}

    for y, m in months:
        month_str = f"{y}-{m:02d}"
        raster_paths = {b: rasters_dir / f"S2_{b}_{month_str}.tif" for b in INDEX_BANDS}
        all_exist = all(p.exists() for p in raster_paths.values())

        if month_str in existing_records and all_exist and not force:
            continue

        start_date = f"{y}-{m:02d}-01"
        end_date = f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01"

        logger.info(f"Querying GEE Sentinel-2 for {month_str}...")
        try:
            stack, scene_count, tier, mask_method = _build_month_composite(
                ee_aoi, start_date, end_date, config)
        except ValueError as reason:
            if str(reason) == "no_scenes":
                logger.warning(f"No valid Sentinel-2 scenes for {month_str} "
                               f"(Cloud < {config['gee'].get('max_cloud_percentage', 60)}%). "
                               f"Flagging month.")
                telemetry.warnings_and_flags.append(
                    f"Month {month_str}: No valid Sentinel-2 scenes")
            else:
                logger.warning(f"Month {month_str}: scenes present but unusable "
                               f"after cloud screening. Flagging month.")
                telemetry.warnings_and_flags.append(
                    f"Month {month_str}: unusable after cloud screening")
            continue
        except Exception:  # noqa: BLE001 - one bad month must not kill a long run
            logger.exception(f"GEE query failed for {month_str}; flagging month.")
            telemetry.warnings_and_flags.append(f"Month {month_str}: GEE query failed")
            continue

        telemetry.s2_scenes_processed += scene_count
        telemetry.s2_composites_processed += 1
        mask_methods_used[mask_method] = mask_methods_used.get(mask_method, 0) + 1

        try:
            content = _download_stack(stack, ee_aoi, scale, logger, month_str)
        except RuntimeError as exc:
            logger.error(str(exc))
            telemetry.warnings_and_flags.append(f"Month {month_str}: {exc}")
            continue

        telemetry.add_download(len(content))
        try:
            written = _split_stack_to_index_rasters(
                content, stack_dir / f"S2_stack_{month_str}.tif", rasters_dir, month_str)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to split stack for {month_str}: {exc}")
            telemetry.warnings_and_flags.append(f"Month {month_str}: stack split failed")
            continue
        for p in written.values():
            telemetry.add_output(p)

        # Stack kept only when explicitly requested (raw 3-index composite)
        if not config["gee"].get("keep_stack", False):
            try:
                (stack_dir / f"S2_stack_{month_str}.tif").unlink(missing_ok=True)
            except OSError:
                pass

        stats = _district_means_from_rasters(written, district_geom, logger)
        rec = {"month": month_str, "scenes_count": scene_count, "tier": tier,
               "mask_method": mask_method, **stats}
        records = [r for r in records if r.get("month") != month_str]
        records.append(rec)
        new_composites += 1
        logger.info(f"  {month_str}: {scene_count} scenes ({tier}, {mask_method}) → "
                    f"NDVI={stats.get('mean_ndvi')} EVI={stats.get('mean_evi')} "
                    f"PCI={stats.get('mean_pci')}")

    df_s2 = pd.DataFrame(records)
    if not df_s2.empty:
        df_s2.sort_values(by="month", inplace=True)
        df_s2.to_csv(out_csv, index=False)
        telemetry.add_output(out_csv)

    if mask_methods_used:
        logger.info(f"Cloud masking methods used: {mask_methods_used}")
    flagged = [r["month"] for r in records
               if pd.isna(r.get("mean_ndvi"))]
    if flagged:
        logger.warning(f"{len(flagged)} months have no valid NDVI: {flagged}")

    logger.info(f"Sentinel-2 processing completed: {new_composites} new composites created. "
                f"Saved to {out_csv}")
    return df_s2
