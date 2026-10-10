"""Smoke test: new Cloud Score+ downloader on a 2-month window."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging, TelemetryTracker, load_and_validate_shapefile
from src.gee.auth import initialize_gee
from src.gee.download import download_sentinel2_composites

config = load_config()
config["dates"]["study_start"] = "2024-01-01"
config["dates"]["study_end"] = "2024-02-29"

logger = setup_logging()
telemetry = TelemetryTracker()
gdf_4326, _ = load_and_validate_shapefile(config, logger)
assert initialize_gee(project_id=config["gee"]["project_id"], logger=logger, fail_fast=True)
df = download_sentinel2_composites(gdf_4326, config, telemetry, logger, force=True)

import pandas as pd, numpy as np, rasterio
print("\n=== SMOKE RESULT ===")
print(df.to_string(index=False))
root = Path(__file__).resolve().parent.parent
for m in ["2024-01", "2024-02"]:
    for b in ["NDVI", "EVI", "PCI"]:
        p = root / "outputs" / "rasters" / f"S2_{b}_{m}.tif"
        if not p.exists():
            print(f"MISSING {p.name}")
            continue
        with rasterio.open(p) as ds:
            a = ds.read(1, masked=True)
            valid = a.count()
            mean = float(a.mean()) if valid else float("nan")
            print(f"{p.name}: nodata={ds.nodata} valid={valid} mean={mean:.4f}")
        assert ds.nodata == -9999.0, "nodata not normalised"
        assert valid > 1000, "too few valid pixels"
        assert np.isfinite(mean), "all-invalid raster"
print("SMOKE TEST PASSED")
