#!/usr/bin/env python3
"""run_all_maps.py - single entry point for the publication figure set.

Renders every map/figure (outputs/maps, PNG + vector PDF), builds the maps
dashboard, captures section screenshots, rebuilds the contact sheet and runs
the 8 quality checks. Read-only with respect to all data: the CDI formula,
thresholds, trained RF model and every CSV in outputs/csv are never touched
(data_snapshot_before == data_snapshot_after proves it).

Usage:  python run_all_maps.py
Exit:   0 if all 8 QC checks pass, 1 otherwise.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.utils import (load_config, setup_logging,  # noqa: E402
                       TelemetryTracker, load_and_validate_shapefile)
from src.processing.maps import (render_all_maps, build_contact_sheet,  # noqa: E402
                                 write_qc_report, build_snapshot)
from src.viz.charts import (build_maps_dashboard,  # noqa: E402
                            capture_dashboard_screenshots)

CSV_DIR = ROOT / "outputs" / "csv"
MAPS_DIR = ROOT / "outputs" / "maps"
DASH_DIR = ROOT / "dashboards"
SHOT_DIR = ROOT / "outputs" / "dashboard_check"


def main() -> int:
    config = load_config()
    logger = setup_logging()
    tel = TelemetryTracker()

    df_stages = pd.read_csv(CSV_DIR / "tehsil_drought_stages.csv")
    df_weather = pd.read_csv(CSV_DIR / "weather_spi.csv",
                             parse_dates=["time"]).set_index("time")
    df_s2 = None
    s2_path = CSV_DIR / "sentinel2_indices.csv"
    if s2_path.exists():
        df_s2 = pd.read_csv(s2_path)
    gdf_4326, _ = load_and_validate_shapefile(config, logger)
    data_asof = config["dates"]["study_end"]

    logger.info("=== run_all_maps: rendering publication figure set ===")
    result = render_all_maps(df_stages, df_weather, gdf_4326, MAPS_DIR,
                             data_asof, tel, logger, config, df_s2)
    html = build_maps_dashboard(DASH_DIR, MAPS_DIR, tel, logger)
    capture_dashboard_screenshots(html, SHOT_DIR, logger)
    build_contact_sheet(MAPS_DIR, tel, logger)
    qc = write_qc_report(result, MAPS_DIR, gdf_4326, df_stages, config,
                         logger)

    out = ["=== run_all_maps QC ==="]
    for k, v in qc.items():
        out.append(f"{'PASS' if v else 'FAIL'}  {k}")
    out.append(f"months: {result.get('months')}")
    out.append(f"event rules: {result.get('event_rules')}")
    stats = result.get("stats", [])
    out.append(f"figures: {len(stats)}")
    for s in stats:
        probs = s.get("problems", [])
        out.append(f"  {'ok ' if not probs else 'PROB'} {s['file']} "
                   f"mode={s.get('label_mode', '?')} problems={len(probs)}"
                   + (f" {probs}" if probs else ""))
    out.append(f"qc_report.md -> {MAPS_DIR / 'qc_report.md'}")
    audit_md = MAPS_DIR / "data_audit.md"
    if audit_md.exists():
        out.append("")
        out.append("=== DATA AUDIT (outputs/maps/data_audit.md) ===")
        out.append(audit_md.read_text(encoding="utf-8"))

    # Phase 1/3: data snapshot proof (unchanged data)
    snap_after = build_snapshot(CSV_DIR)
    snap_before_path = MAPS_DIR / "data_snapshot_before.json"
    if snap_before_path.exists():
        import json
        before = json.loads(snap_before_path.read_text(encoding="utf-8"))
        same = before == snap_after
        out.append("")
        out.append("=== DATA SNAPSHOT (before == after) ===")
        out.append("IDENTICAL" if same else "*** CHANGED - REVERT! ***")
        (MAPS_DIR / "data_snapshot_after.json").write_text(
            json.dumps(snap_after, indent=1), encoding="utf-8")
        if not same:
            for k in set(before) | set(snap_after):
                if before.get(k) != snap_after.get(k):
                    out.append(f"  DIFFERS: {k}")
            qc["snapshot"] = False
    else:
        out.append("data_snapshot_before.json missing - run "
                   "p1_backup_snapshot.py first")
        qc["snapshot"] = False

    text = "\n".join(out)
    print(text)
    (ROOT / "logs" / "map_render_report.txt").write_text(text,
                                                         encoding="utf-8")

    if all(qc.values()):
        print("RESULT: ALL 8 QC CHECKS PASS")
        return 0
    print("RESULT: QC FAILURES - see qc_report.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
