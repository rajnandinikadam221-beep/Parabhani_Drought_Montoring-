from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import ee


def _month_windows(study_start: str, study_end: str) -> List[Tuple[str, str, int, int]]:
    """(month_label, start_date, year, month_number) for every month in the window."""
    start = datetime.strptime(study_start, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(study_end, "%Y-%m-%d")
    out = []
    curr = start
    while curr <= end:
        label = f"{curr.year}-{curr.month:02d}"
        if curr.month == 12:
            next_m = f"{curr.year + 1}-01-01"
        else:
            next_m = f"{curr.year}-{curr.month + 1:02d}-01"
        out.append((label, f"{curr.year}-{curr.month:02d}-01", next_m, curr.year, curr.month))
        curr = curr.replace(year=curr.year + 1, month=1) if curr.month == 12 \
            else curr.replace(month=curr.month + 1)
    return [(label, s, e, y, m) for label, s, e, y, m in out]


def _standardize_ssmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    SSMI = (sm - calendar-month mean) / calendar-month std per tehsil.
    Uses every available year of SMAP (2015+); 1991-2020 is impossible for
    passive-microwave soil moisture and is documented as a limitation.
    """
    df = df.copy()
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    df["SSMI"] = np.nan
    for tehsil, grp in df.groupby("tehsil"):
        for m, g in grp.groupby("month_num"):
            vals = g["sm_mean"].dropna()
            if len(vals) < 5:
                continue
            mu, sigma = vals.mean(), vals.std(ddof=0)
            if sigma > 0:
                idx = g.index[g["sm_mean"].notna()]
                df.loc[idx, "SSMI"] = ((df.loc[idx, "sm_mean"] - mu) / sigma).round(3)
    return df.drop(columns=["month_num"])


def fetch_soil_moisture_ssmi(
    gdf_4326: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Per-tehsil monthly soil moisture (SMAP L4 surface moisture) and SSMI.
    One batched Earth Engine round trip: monthly-mean images -> toBands() ->
    reduceRegions() over the tehsils.

    Returns a tidy DataFrame (tehsil, month, sm_mean, SSMI) or None when the
    source is unavailable (callers then fall back to the weather-based SSMI).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_dir = project_root / "outputs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "soil_moisture_ssmi.csv"

    if out_csv.exists() and not force:
        logger.info(f"[CACHE] Found existing SMAP soil moisture table at {out_csv}")
        telemetry.add_output(out_csv)
        return pd.read_csv(out_csv)

    sm_config = config.get("soil_moisture", {})
    collection_id = sm_config.get("source", "NASA/SMAP/SPL4SMGP/008")
    band = sm_config.get("band", "sm_surface")
    scale = sm_config.get("scale", 11000)

    try:
        smap = (ee.ImageCollection(collection_id)
                .filterDate(config["dates"]["study_start"], config["dates"]["study_end"])
                .select([band]))
        scene_count = int(smap.size().getInfo())
        if scene_count == 0:
            raise RuntimeError(f"{collection_id} has no scenes in the study window")
        logger.info(f"SMAP soil moisture: {scene_count} granules in {collection_id}. "
                    f"Building per-tehsil monthly means...")

        features = [
            ee.Feature(_shapely_to_ee(row.geometry), {"tehsil": row["tehsil"]})
            for _, row in gdf_4326.iterrows()
        ]
        tehsil_fc = ee.FeatureCollection(features)

        months = _month_windows(config["dates"]["study_start"], config["dates"]["study_end"])
        monthly_images = []
        for label, start_date, end_date, y, m in months:
            sub = smap.filterDate(start_date, end_date)
            monthly_images.append(sub.mean())

        stack = ee.ImageCollection.fromImages(monthly_images).toBands()
        band_names = stack.bandNames().getInfo()          # round trip 1
        fc = stack.reduceRegions(
            reducer=ee.Reducer.mean(),
            collection=tehsil_fc,
            scale=scale,
            tileScale=4,
        ).getInfo()                                        # round trip 2
    except Exception:  # noqa: BLE001 - soil moisture must never kill the pipeline
        logger.exception("SMAP soil moisture fetch failed; falling back to "
                         "weather-based SSMI for this run.")
        return None

    if len(band_names) != len(months):
        logger.error(f"SMAP band/month mismatch: {len(band_names)} bands vs "
                     f"{len(months)} months - skipping SSMI.")
        return None

    rows = []
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        tehsil = props.get("tehsil")
        for i, (label, *_rest) in enumerate(months):
            val = props.get(band_names[i])
            rows.append({"tehsil": tehsil, "month": label,
                         "sm_mean": float(val) if val is not None else np.nan})

    if not rows:
        logger.warning("SMAP reduceRegions returned no features; skipping SSMI.")
        return None

    df = pd.DataFrame(rows)
    df = _standardize_ssmi(df)
    valid = df["sm_mean"].notna().sum()
    ssmi_valid = df["SSMI"].notna().sum()
    if valid == 0:
        logger.warning("SMAP returned no valid soil moisture values; skipping SSMI.")
        return None

    df.to_csv(out_csv, index=False)
    telemetry.add_output(out_csv)
    logger.info(f"SMAP soil moisture saved: {valid} valid tehsil-months, "
                f"{ssmi_valid} SSMI values → {out_csv}")
    return df


def _shapely_to_ee(geom):
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon([list(geom.exterior.coords)])
    elif geom.geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon([list(p.exterior.coords) for p in geom.geoms])
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
