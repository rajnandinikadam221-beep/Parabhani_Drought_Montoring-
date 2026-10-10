from pathlib import Path
import os
import sys
import time
import logging
from typing import Dict, Any, Tuple
import yaml
import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import Point, Polygon, MultiPolygon

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Sets up unified logging to file and stdout."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logger = logging.getLogger("ParbhaniDrought")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    # Force UTF-8 on Windows to allow Unicode symbols (e.g. ✓) in log output
    if hasattr(sh.stream, "reconfigure"):
        try:
            sh.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    logger.info(f"Logging initialized. Output file: {log_file}")
    return logger

class TelemetryTracker:
    """Tracks execution telemetry, file sizes, processing counts, and stage timings."""
    def __init__(self):
        self.start_time = time.time()
        self.data_downloaded_mb = 0.0
        self.s2_scenes_processed = 0
        self.s2_composites_processed = 0
        self.weather_records_processed = 0
        self.spatial_points_sampled = 0
        self.stage_timings: Dict[str, float] = {}
        self.output_files = []
        self.warnings_and_flags = []

    def start_stage(self, stage_name: str) -> float:
        return time.time()

    def end_stage(self, stage_name: str, t0: float):
        self.stage_timings[stage_name] = round(time.time() - t0, 2)

    def add_download(self, bytes_count: int):
        self.data_downloaded_mb += bytes_count / (1024 * 1024)

    def add_output(self, path: Path):
        self.output_files.append(str(path))

def load_and_validate_shapefile(config: Dict[str, Any], logger: logging.Logger) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads, validates, repairs, and reprojects the Parbhani Tehsil shapefile.
    Returns (gdf_4326, gdf_proj_32643).
    """
    project_root = Path(__file__).resolve().parent.parent
    shp_path = project_root / config["study_area"]["shapefile"]
    fallback_path = project_root / config["study_area"].get("fallback_shapefile", "")
    
    target_path = None
    if shp_path.exists():
        target_path = shp_path
    elif fallback_path.exists():
        target_path = fallback_path
    else:
        logger.error(f"Shapefile not found at {shp_path} or fallback {fallback_path}!")
        sys.exit(1)
        
    logger.info(f"Loading boundary shapefile from: {target_path}")
    gdf = gpd.read_file(target_path)
    
    if gdf.empty:
        logger.error("Shapefile contains no geometries! Exiting.")
        sys.exit(1)
        
    if gdf.crs is None:
        logger.error(f"Shapefile at {target_path} lacks a valid CRS! Exiting.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
    
    # Repair geometries if needed
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    
    # Extract tehsil name
    tehsil_col = None
    for col in ['SUB_DIST', 'TEHSIL', 'TALUK', 'SUBDIST_NA', 'NAME_3', 'NAME_2', 'DISTRICT']:
        if col in gdf.columns:
            tehsil_col = col
            break
            
    if tehsil_col:
        gdf["tehsil"] = gdf[tehsil_col].astype(str).str.title().str.strip()
    else:
        gdf["tehsil"] = [f"Tehsil_{i+1}" for i in range(len(gdf))]
        
    # Reproject to EPSG:4326 and EPSG:32643
    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf_proj = gdf.to_crs(epsg=32643)
    
    total_area_km2 = gdf_proj.geometry.area.sum() / 1e6
    logger.info(f"Parbhani District: {total_area_km2:.2f} km² across {len(gdf)} tehsils: {gdf_4326['tehsil'].tolist()}")
    
    return gdf_4326, gdf_proj
