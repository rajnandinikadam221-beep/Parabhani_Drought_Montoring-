"""Background runner: Sentinel-2 stage only (Cloud Score+ + per-index rasters)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging, TelemetryTracker, load_and_validate_shapefile
from src.gee.auth import initialize_gee
from src.gee.download import download_sentinel2_composites

config = load_config()
logger = setup_logging()
telemetry = TelemetryTracker()

gdf_4326, _ = load_and_validate_shapefile(config, logger)
assert initialize_gee(project_id=config["gee"]["project_id"], logger=logger, fail_fast=True)
df = download_sentinel2_composites(gdf_4326, config, telemetry, logger, force=False)

logger.info(f"Background Sentinel-2 stage finished: {len(df)} months in CSV, "
            f"{telemetry.s2_composites_processed} composites this run, "
            f"{telemetry.data_downloaded_mb:.1f} MB downloaded.")
