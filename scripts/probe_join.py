"""Probe: why 2016-01 composite came out band-less (CS+ join issue)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
from src.gee.auth import initialize_gee
from src.utils import load_config, setup_logging, load_and_validate_shapefile
from src.gee.download import shapely_to_ee, _district_geometry, CS_PLUS_COLLECTION

logger = setup_logging()
config = load_config()
initialize_gee(project_id=config["gee"]["project_id"], logger=logger, fail_fast=True)
gdf, _ = load_and_validate_shapefile(config, logger)
aoi = shapely_to_ee(_district_geometry(gdf))

for month_str, start, end in [
    ("2016-01", "2016-01-01", "2016-02-01"),
    ("2016-07", "2016-07-01", "2016-08-01"),
    ("2024-06", "2024-06-01", "2024-07-01"),
]:
    print(f"\n=== {month_str} ===")
    for name, cid in [("SR", config["gee"]["s2_sr_collection"]),
                      ("TOA", config["gee"]["s2_toa_collection"])]:
        col = (ee.ImageCollection(cid).filterBounds(aoi).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
        n = col.size().getInfo()
        idx = col.aggregate_array("system:index").slice(0, 3).getInfo() if n else []
        print(f"{name} (cloud<60): {n} scenes, sample idx={idx}")

    for cs_cid in [CS_PLUS_COLLECTION, "GOOGLE/CLOUD_SCORE_PLUS/V1/S2"]:
        cs = (ee.ImageCollection(cs_cid).filterBounds(aoi).filterDate(start, end))
        n = cs.size().getInfo()
        idx = cs.aggregate_array("system:index").slice(0, 3).getInfo() if n else []
        print(f"{cs_cid}: {n} scenes, sample idx={idx}")

    # join test against SR and TOA
    for name, cid in [("SR", config["gee"]["s2_sr_collection"]),
                      ("TOA", config["gee"]["s2_toa_collection"])]:
        col = (ee.ImageCollection(cid).filterBounds(aoi).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
        cs = ee.ImageCollection(CS_PLUS_COLLECTION).filterBounds(aoi).filterDate(start, end)
        import ee as _ee
        joined = _ee.Join.saveFirst("cs").apply(
            col, cs, _ee.Filter.equals(leftField="system:index", rightField="system:index"))
        print(f"join {name} x CS+_HARMONIZED: {_ee.ImageCollection(joined).size().getInfo()}")
        cs2 = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2").filterBounds(aoi).filterDate(start, end)
        joined2 = _ee.Join.saveFirst("cs").apply(
            col, cs2, _ee.Filter.equals(leftField="system:index", rightField="system:index"))
        print(f"join {name} x CS+_S2: {_ee.ImageCollection(joined2).size().getInfo()}")
