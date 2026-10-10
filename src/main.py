#!/usr/bin/env python3
"""
Main pipeline entry point — Parbhani Real-Time & Historical Drought Monitoring (2016–2026).

Usage:
    python src/main.py                        # Full run (GEE + Weather + ML + Viz)
    python src/main.py --skip-gee             # Weather, indices, ML, and Viz only
    python src/main.py --force                # Recompute all, ignore cache
    python src/main.py --api                  # Start FastAPI server
    python src/main.py --gee-project <id>     # Override GEE project ID
"""

from pathlib import Path
import sys
import os
import argparse
import json
import time
from datetime import datetime
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, TelemetryTracker, load_and_validate_shapefile
from src.gee.auth import initialize_gee
from src.gee.download import download_sentinel2_composites
from src.weather.openmeteo import fetch_weather_and_compute_spi
from src.gee.soil_moisture import fetch_soil_moisture_ssmi
from src.processing.indices import process_monthly_indices
from src.processing.classify import compute_drought_stages_and_cdi
from src.ml.sampling import generate_stratified_samples
from src.ml.random_forest import train_drought_classifier
from src.viz.charts import generate_all_visualizations


def verify_outputs(stage: str, logger, telemetry, gdf_4326=None) -> list:
    """
    §10: after each stage, verify outputs (file counts, no all-NaN rasters,
    CRS checks, shapefile bounds check). Problems are logged, flagged in the
    telemetry and returned.
    """
    problems = []
    outputs = PROJECT_ROOT / "outputs"
    csv_dir = outputs / "csv"
    rasters_dir = outputs / "rasters"

    def _flag(msg: str):
        problems.append(msg)
        logger.warning(f"[VERIFY/{stage}] {msg}")
        telemetry.warnings_and_flags.append(f"verify:{stage}: {msg}")

    if stage == "shapefile" and gdf_4326 is not None:
        if gdf_4326.crs is None:
            _flag("shapefile has no CRS")
        minx, miny, maxx, maxy = gdf_4326.total_bounds
        if not (75.0 < minx < 77.0 and 76.0 < maxx < 78.0
                and 18.0 < miny < 19.5 and 19.0 < maxy < 21.0):
            _flag(f"shapefile bounds outside expected Parbhani range: "
                  f"({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
        if len(gdf_4326) != 9:
            _flag(f"expected 9 tehsils, found {len(gdf_4326)}")

    elif stage == "sentinel2":
        import rasterio
        months_expected = 130
        for idx in ("NDVI", "EVI", "PCI"):
            files = sorted(rasters_dir.glob(f"S2_{idx}_*.tif"))
            if len(files) < months_expected - 3:  # allow a few flagged months
                _flag(f"{idx}: only {len(files)}/{months_expected} rasters")
            empty, bad_crs = [], []
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        if src.crs is None or src.crs.to_epsg() != 4326:
                            bad_crs.append(f.name)
                        arr = src.read(1)
                        if not np.isfinite(arr[arr != -9999]).any():
                            empty.append(f.name)
                except Exception as exc:  # noqa: BLE001
                    _flag(f"unreadable raster {f.name}: {exc}")
            if empty:
                _flag(f"{len(empty)} all-NaN {idx} rasters: {empty[:6]}")
            if bad_crs:
                _flag(f"{len(bad_crs)} {idx} rasters not in EPSG:4326")

    elif stage == "zonal":
        import pandas as pd
        zonal = csv_dir / "tehsil_zonal_indices.csv"
        if not zonal.exists():
            _flag("tehsil_zonal_indices.csv missing")
        else:
            df = pd.read_csv(zonal)
            for col in ("NDVI", "EVI", "PCI", "SPI-3", "SSMI"):
                if col in df.columns and df[col].notna().sum() == 0:
                    _flag(f"zonal column {col} is entirely NaN")
            dupes = df.duplicated(subset=["tehsil", "month"]).sum()
            if dupes:
                _flag(f"{dupes} duplicate tehsil-month rows in zonal table")

    elif stage == "classification":
        import pandas as pd
        import rasterio
        stages_csv = csv_dir / "tehsil_drought_stages.csv"
        area_csv = csv_dir / "stage_area_summary.csv"
        if not stages_csv.exists() or not area_csv.exists():
            _flag("stage CSVs missing")
        else:
            df = pd.read_csv(stages_csv)
            bad = set(df["stage"].dropna().unique()) - {-1, 0, 1, 2, 3, 4}
            if bad:
                _flag(f"invalid stage codes: {bad}")
            if df["CDI"].abs().max() > 10:
                _flag(f"CDI out of range: max |CDI| = {df['CDI'].abs().max():.1f}")
            area = pd.read_csv(area_csv)
            totals = area.groupby("month")["area_km2"].sum()
            if (totals - 6307.5).abs().max() > 1.0:
                _flag(f"stage area sums deviate from 6307.5 km2: "
                      f"min {totals.min():.1f}, max {totals.max():.1f}")
        class_dir = rasters_dir / "classified"
        files = sorted(class_dir.glob("DROUGHT_STAGE_*.tif"))
        if len(files) < 127:
            _flag(f"only {len(files)} classified rasters")
        for f in files:
            with rasterio.open(f) as src:
                if src.crs is None or src.crs.to_epsg() != 32643:
                    _flag(f"{f.name} CRS is {src.crs}, expected EPSG:32643")
                    break
                if src.nodata != 255:
                    _flag(f"{f.name} nodata is {src.nodata}, expected 255")
                    break

    elif stage == "visualizations":
        figs_dir = outputs / "figures"
        maps_dir = outputs / "maps"
        dashboards = PROJECT_ROOT / "dashboards"
        expected = [figs_dir / n for n in
                    ("SPI_NDVI_trend.png", "SPI3_bar_drought_classification.png",
                     "correlation_heatmap.png", "tehsil_boxplots.png",
                     "drought_stage_heatmap.png", "index_distributions.png",
                     "rf_confusion_matrix.png", "rf_feature_importance.png")]
        expected += [maps_dir / n for n in
                     ("map_index_ndvi.png", "map_index_evi.png",
                      "map_index_pci.png", "map_index_ssmi.png",
                      "map_index_spi3.png", "contact_sheet.png",
                      "data_audit.md", "qc_report.md",
                      "map_rf_drought_classification.png",
                      "map_ssmi_category.png", "fig_study_area.png",
                      "fig_methodology_flowchart.png",
                      "fig_multi_index_assessment.png",
                      "fig_rf_confusion_matrix.png",
                      "fig_rf_feature_importance.png", "fig_rf_features.png",
                      "fig_model_validation.png", "fig_index_distribution.png",
                      "fig_tehsil_boxplots.png", "fig_rainfall_distribution.png",
                      "fig_spi3_classification.png", "fig_drought_trend.png")]
        expected += [dashboards / "interactive_drought_dashboard.html",
                     dashboards / "drought_stage_timeline.html",
                     dashboards / "drought_maps_dashboard.html"]
        missing = [p.name for p in expected if not p.exists()]
        if missing:
            _flag(f"missing visualization outputs: {missing}")
        if len(list(maps_dir.glob("map_stage_*.png"))) < 5:
            _flag("fewer than 5 drought-stage maps (expect 2019-05, "
                  "2025-09, 2017-01, 2026-09, 2026-10 provisional)")
        if len(list(maps_dir.glob("map_*.png"))) < 20:
            _flag("fewer than 20 publication maps")
        if len(list(maps_dir.glob("map_*.pdf"))) < 20:
            _flag("fewer than 20 map PDFs")
        if len(list((outputs / "dashboard_check").glob("*.png"))) < 6:
            _flag("fewer than 6 dashboard section screenshots")
        qc = maps_dir / "qc_report.md"
        if qc.exists() and "ALL PASS" not in qc.read_text(encoding="utf-8"):
            _flag("qc_report.md does not report ALL PASS")

    if not problems:
        logger.info(f"[VERIFY/{stage}] OK — all checks passed.")
    else:
        logger.warning(f"[VERIFY/{stage}] {len(problems)} problem(s) found.")
    return problems


def write_summary_report(telemetry: TelemetryTracker, config: dict, logger):
    """Writes outputs/reports/summary.md and summary.json with full metrics."""
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_time = time.time() - telemetry.start_time
    now = datetime.now().isoformat(timespec="seconds")

    # Collect output file tree and sizes
    outputs_dir = PROJECT_ROOT / "outputs"
    file_tree = []
    total_bytes = 0
    for p in sorted(outputs_dir.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            file_tree.append({"path": str(p.relative_to(PROJECT_ROOT)), "size_kb": round(sz / 1024, 1)})

    # ── Actual data volumes and data-quality accounting (file-based, so a
    #    cached re-run still reports the true download sizes) ────────────────
    rasters_dir = outputs_dir / "rasters"
    csv_dir = outputs_dir / "csv"
    s2_files = sorted(rasters_dir.glob("S2_*.tif"))
    s2_bytes = sum(p.stat().st_size for p in s2_files)
    weather_csv = csv_dir / "weather_spi.csv"
    weather_bytes = weather_csv.stat().st_size if weather_csv.exists() else 0
    s2_csv = csv_dir / "sentinel2_indices.csv"

    s2_scenes_total, s2_composites, tier_counts, mask_counts = 0, 0, {}, {}
    low_quality_months = []
    if s2_csv.exists():
        import pandas as _pd
        _s2 = _pd.read_csv(s2_csv)
        s2_composites = len(_s2)
        if "scenes_count" in _s2.columns:
            s2_scenes_total = int(_s2["scenes_count"].fillna(0).sum())
        for col, bucket in (("tier", tier_counts), ("mask_method", mask_counts)):
            if col in _s2.columns:
                for k, v in _s2[col].value_counts().items():
                    bucket[str(k)] = int(v)
        if "mean_ndvi" in _s2.columns:
            low_quality_months = _s2.loc[_s2["mean_ndvi"].isna(), "month"].tolist()

    ndvi_rasters = sorted(rasters_dir.glob("S2_NDVI_*.tif"))
    months_with_raster = {p.name[len("S2_NDVI_"):-len(".tif")] for p in ndvi_rasters}
    study_months = [f"{y}-{m:02d}" for y in range(2016, 2027) for m in range(1, 13)
                    if f"{y}-{m:02d}" <= "2026-10"]
    months_without_raster = [m for m in study_months if m not in months_with_raster]

    csv_row_counts = {}
    for p in sorted(csv_dir.glob("*.csv")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                csv_row_counts[p.name] = max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            csv_row_counts[p.name] = -1

    classified_count = len(list((rasters_dir / "classified").glob("DROUGHT_STAGE_*.tif")))

    # Markdown summary
    md_lines = [
        "# Parbhani Drought Monitoring Pipeline — Execution Report",
        f"\n**Generated:** {now}",
        f"\n## Study Area",
        f"- **Region:** {config['study_area']['name']} District, Maharashtra, India",
        f"- **Tehsils:** 9 (Gangakhed, Jintur, Manwath, Palam, Parbhani, Pathri, Purna, Selu, Sonpeth)",
        f"- **Study Period:** {config['dates']['study_start']} → {config['dates']['study_end']}",
        f"- **Baseline Reference:** {config['dates']['baseline_start']} → {config['dates']['baseline_end']}",
        "\n## Data Downloaded & Derived",
        f"- **Sentinel-2 rasters on disk:** {len(s2_files)} files "
        f"({s2_bytes / (1024**3):.2f} GB; NDVI/EVI/PCI × {len(ndvi_rasters)} months)",
        f"- **Sentinel-2 scenes queried (Cloud Score+ join):** {s2_scenes_total} "
        f"across {s2_composites} monthly composites",
        f"- **Mask strategy:** {json.dumps(mask_counts) if mask_counts else 'n/a'}",
        f"- **Filter tiers used:** {json.dumps(tier_counts) if tier_counts else 'n/a'}",
        f"- **Weather (Open-Meteo):** {weather_bytes / (1024**2):.2f} MB "
        f"(weather_spi.csv, {csv_row_counts.get('weather_spi.csv', 0)} monthly rows)",
        f"- **Spatial Sample Points:** {telemetry.spatial_points_sampled or csv_row_counts.get('stratified_sample_points.csv', 0)}",
        f"- **Classified rasters:** {classified_count} (EPSG:32643, 100 m, uint8)",
        f"- **Total outputs:** {len(file_tree)} files, {total_bytes / (1024**3):.2f} GB on disk",
        "\n## Data Availability",
    ]
    if months_without_raster:
        md_lines.append(
            f"- **Months with no Sentinel-2 raster** ({len(months_without_raster)}): "
            + ", ".join(months_without_raster)
            + " — no usable overpasses in the join window.")
    else:
        md_lines.append("- All study months have Sentinel-2 rasters.")
    if low_quality_months:
        md_lines.append(
            f"- **Months whose composite is fully cloud-masked (NaN mean NDVI)** "
            f"({len(low_quality_months)}): " + ", ".join(low_quality_months))
    md_lines.append(
        "- SPI fits a Gamma distribution on the "
        f"{config['dates']['baseline_start']}–{config['dates']['baseline_end']} baseline.")

    md_lines += ["\n## Processing Times",]
    for stage, dur in telemetry.stage_timings.items():
        md_lines.append(f"- **{stage.replace('_', ' ').title()}:** {dur:.2f} s")
    md_lines.append(f"- **Total Elapsed:** {total_time:.2f} s")

    md_lines += ["\n## Rows per CSV (outputs/csv)"]
    for name, n in sorted(csv_row_counts.items()):
        md_lines.append(f"- `{name}`: {n} rows")

    md_lines += [
        "\n## Assumptions & Limitations",
        "- Sentinel-2: S2_SR_HARMONIZED (surface reflectance) preferred; "
        "S2_TOA_HARMONIZED only for months with no usable SR scenes "
        f"(see tier counts above). Scene filter: <{config['gee']['max_cloud_percentage']}% "
        f"scene cloud, falling back to <{config['gee'].get('max_cloud_fallback', 80)}%, "
        "plus pixel-level Cloud Score+ (cs_cdf ≥ 0.6) where available.",
        "- SSMI: primary source is NASA SMAP SPL4SMGP (10 km soil moisture, 2015+) "
        "standardised per tehsil per calendar month; months without SMAP fall back to "
        "a precipitation − ET₀ water-deficit SSMI (flagged in ssmi_source column).",
        "- CDI weights: SPI .35, SSMI .25, VCI(z) .25, EVI .10, PCI .05; thresholds "
        "extreme ≤ −2.0, severe ≤ −1.5, moderate ≤ −1.0, mild ≤ 0.0.",
        "- CDI is computed per pixel (100 m) over the district and averaged per tehsil; "
        "per-tehsil zonal statistics are exact means over district-intersecting pixels.",
        "- Random Forest: two labels (composite CDI stage and SPI-3-only stage); "
        "features are SPI/SSMI/NDVI/EVI/PCI only — CDI is never fed back as a feature.",
        "- Weather API/archive (Open-Meteo) provides the precipitation, ET₀ and "
        "temperature series; no station data is used.",
        "\n## Library Versions",
    ]
    import importlib
    for lib in ["geopandas", "rasterio", "shapely", "pandas", "numpy", "scipy", "sklearn", "ee", "plotly", "fastapi"]:
        try:
            mod = importlib.import_module(lib if lib != "sklearn" else "sklearn")
            version = getattr(mod, "__version__", "?")
            md_lines.append(f"- `{lib}` {version}")
        except Exception:
            md_lines.append(f"- `{lib}` (unavailable)")

    md_lines += [
        "\n## Output Files",
        f"*Total: {len(file_tree)} files, {total_bytes / (1024*1024):.2f} MB on disk*",
        "```"
    ]
    for f in file_tree:
        md_lines.append(f"  {f['path']} ({f['size_kb']} KB)")
    md_lines.append("```")

    if telemetry.warnings_and_flags:
        md_lines += ["\n## ⚠️ Flags & Warnings"]
        for w in telemetry.warnings_and_flags:
            md_lines.append(f"- {w}")

    md_lines += [
        "\n## How to Re-run",
        "```bash",
        "# Full pipeline with GEE:",
        "python src/main.py --gee-project dark-influence-491307-n3",
        "",
        "# Force recompute all outputs:",
        "python src/main.py --gee-project dark-influence-491307-n3 --force",
        "",
        "# Weather + ML + Viz only (no GEE):",
        "python src/main.py --skip-gee",
        "",
        "# Launch REST API:",
        "python src/main.py --api",
        "```",
    ]

    md_text = "\n".join(md_lines)
    md_path = reports_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.info(f"Summary report written to {md_path}")

    # JSON summary
    json_summary = {
        "generated_at": now,
        "study_area": config["study_area"]["name"],
        "total_elapsed_seconds": round(total_time, 2),
        "sentinel_raster_files": len(s2_files),
        "sentinel_raster_gb": round(s2_bytes / (1024 ** 3), 3),
        "sentinel_scenes_queried": s2_scenes_total,
        "sentinel_composites": s2_composites,
        "weather_csv_mb": round(weather_bytes / (1024 ** 2), 3),
        "months_without_raster": months_without_raster,
        "months_fully_masked": low_quality_months,
        "mask_method_counts": mask_counts,
        "filter_tier_counts": tier_counts,
        "csv_row_counts": csv_row_counts,
        "classified_rasters": classified_count,
        "spatial_points": telemetry.spatial_points_sampled or csv_row_counts.get("stratified_sample_points.csv", 0),
        "stage_timings_s": telemetry.stage_timings,
        "output_files_count": len(file_tree),
        "output_total_gb": round(total_bytes / (1024 ** 3), 3),
        "warnings": telemetry.warnings_and_flags,
    }
    json_path = reports_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    logger.info(f"JSON summary written to {json_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("   DROUGHT MONITORING PIPELINE — EXECUTION SUMMARY")
    print("=" * 65)
    for k, v in json_summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parbhani Real-Time & Historical Drought Monitoring Pipeline (2016–2026)"
    )
    parser.add_argument("--gee-project", type=str, default=os.environ.get("GEE_PROJECT"), help="GEE Cloud Project ID")
    parser.add_argument("--skip-gee", action="store_true", help="Skip GEE authentication and Sentinel-2 download")
    parser.add_argument("--force", action="store_true", help="Force recomputation and overwrite all cached outputs")
    parser.add_argument("--api", action="store_true", help="Launch the FastAPI server instead of running the pipeline")
    args = parser.parse_args()

    # ── FastAPI mode ──────────────────────────────────────────────────────
    if args.api:
        import uvicorn
        from src.api.endpoints import create_app
        print("Starting Drought Monitoring REST API on http://127.0.0.1:8000")
        uvicorn.run(create_app(), host="127.0.0.1", port=8000)
        return

    # ── Pipeline mode ─────────────────────────────────────────────────────
    config = load_config()
    logger = setup_logging()
    telemetry = TelemetryTracker()

    logger.info("=" * 65)
    logger.info("  Parbhani Drought Monitoring Pipeline — Starting")
    logger.info(f"  GEE Project: {args.gee_project or config['gee']['project_id'] or 'from env'}")
    logger.info(f"  Force recompute: {args.force}")
    logger.info(f"  Skip GEE: {args.skip_gee}")
    logger.info("=" * 65)

    # ── Stage 1: Shapefile ────────────────────────────────────────────────
    t0 = telemetry.start_stage("shapefile")
    gdf_4326, gdf_proj = load_and_validate_shapefile(config, logger)
    telemetry.end_stage("shapefile", t0)
    verify_outputs("shapefile", logger, telemetry, gdf_4326=gdf_4326)

    # ── Stage 2: GEE Authentication ───────────────────────────────────────
    gee_enabled = False
    if not args.skip_gee:
        t0 = telemetry.start_stage("gee_auth")
        project_id = args.gee_project or config["gee"].get("project_id")
        gee_enabled = initialize_gee(project_id=project_id, logger=logger, fail_fast=True)
        telemetry.end_stage("gee_auth", t0)

    # ── Stage 3: Open-Meteo Weather + SPI (1991–2026) ────────────────────
    t0 = telemetry.start_stage("weather_spi")
    df_weather = fetch_weather_and_compute_spi(gdf_4326, config, telemetry, logger, force=args.force)
    telemetry.end_stage("weather_spi", t0)

    # ── Stage 4: Sentinel-2 GEE Download ─────────────────────────────────
    df_s2 = None
    if gee_enabled:
        t0 = telemetry.start_stage("gee_sentinel2")
        df_s2 = download_sentinel2_composites(gdf_4326, config, telemetry, logger, force=args.force)
        telemetry.end_stage("gee_sentinel2", t0)
        verify_outputs("sentinel2", logger, telemetry)
    else:
        logger.warning("GEE skipped — Sentinel-2 NDVI/EVI/PCI will not be available.")

    # ── Stage 4b: SMAP soil moisture → per-tehsil SSMI ──────────────────────
    if gee_enabled:
        t0 = telemetry.start_stage("soil_moisture")
        try:
            fetch_soil_moisture_ssmi(gdf_4326, config, telemetry, logger, force=args.force)
        except Exception:  # noqa: BLE001 - weather-based SSMI remains as fallback
            logger.exception("SMAP soil moisture stage failed — "
                             "falling back to weather-based SSMI.")
        telemetry.end_stage("soil_moisture", t0)

    # ── Stage 5: Zonal Statistics (all indices per tehsil per month) ──────
    t0 = telemetry.start_stage("zonal_indices")
    df_zonal = process_monthly_indices(gdf_4326, df_weather, df_s2, config, telemetry, logger, force=args.force)
    telemetry.end_stage("zonal_indices", t0)
    verify_outputs("zonal", logger, telemetry)

    # ── Stage 6: CDI + Drought Stage Classification ───────────────────────
    t0 = telemetry.start_stage("drought_classification")
    df_stages = compute_drought_stages_and_cdi(df_zonal, gdf_proj, config, telemetry, logger, force=args.force)
    telemetry.end_stage("drought_classification", t0)
    verify_outputs("classification", logger, telemetry)

    # ── Stage 7: Stratified Spatial Sampling ─────────────────────────────
    t0 = telemetry.start_stage("spatial_sampling")
    generate_stratified_samples(gdf_proj, config, telemetry, logger, force=args.force)
    telemetry.end_stage("spatial_sampling", t0)

    # ── Stage 8: Random Forest Classification ────────────────────────────
    t0 = telemetry.start_stage("ml_random_forest")
    train_drought_classifier(df_stages, config, telemetry, logger, force=args.force)
    telemetry.end_stage("ml_random_forest", t0)

    # ── Stage 9: All Visualizations ──────────────────────────────────────
    t0 = telemetry.start_stage("visualizations")
    generate_all_visualizations(df_weather, df_stages, df_s2, gdf_4326, config, telemetry, logger, force=args.force)
    telemetry.end_stage("visualizations", t0)
    verify_outputs("visualizations", logger, telemetry)

    # ── Stage 10: Summary Report ──────────────────────────────────────────
    write_summary_report(telemetry, config, logger)

    problems = [w for w in telemetry.warnings_and_flags if w.startswith("verify:")]
    if problems:
        logger.warning(f"Pipeline finished with {len(problems)} verification issue(s) - see summary.md.")
    logger.info("Pipeline completed successfully. All outputs in outputs/ and dashboards/.")


if __name__ == "__main__":
    main()
