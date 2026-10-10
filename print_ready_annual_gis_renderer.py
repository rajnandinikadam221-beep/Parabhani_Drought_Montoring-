#!/usr/bin/env python3
"""print_ready_annual_gis_renderer.py

Standalone, publication-quality annual GIS renderer for Parbhani district
(outputs/reports/prompt.md, system component: print_ready_annual_gis_renderer).

For every year 2016-2026 it exports ONE independent, single-canvas map per
indicator (no multi-panel subplots, no summary tables, no locator frames):

    reference_map_administrative.png   1x boundaries + tehsil names only
    {year}_map_ndvi.png                NDVI   (annual mean of monthly values)
    {year}_map_evi.png                 EVI    (annual mean of monthly values)
    {year}_map_pci.png                 PCI    (annual mean of monthly values)
    {year}_map_ssmi.png                SSMI   (annual mean of monthly values)
    {year}_map_cdi.png                 CDI    (peak severity = annual MINIMUM)

Cartography: EPSG:32643, charcoal tehsil outlines (#2B2B2B, 0.85), desaturated
colorblind-safe ramps, 8.5 pt centred labels at polygon representative points
(inside + white halo, so text never crosses a line), graphic km scale bar,
minimalist north arrow outside the polygons, WGS-1984 lat/lon graticule ticks,
a thin neatline, and a 300 dpi bbox_inches='tight' PNG per figure.

Dashboard integration: immediately after map export the project dashboard
engine is refreshed (Plotly HTML dashboards + FastAPI-served
outputs/reports/drought_dashboard_data.json) and a companion
dashboards/annual_gis_dashboard.html is written that embeds the exact same
PNGs and the exact same annual data matrices / color breaks.

Run:  python print_ready_annual_gis_renderer.py [--skip-dashboard]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Paths / project wiring
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
OUT_DIR = PROJECT_ROOT / "maps_output"
DASHBOARDS_DIR = PROJECT_ROOT / "dashboards"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
SHAPEFILE_CANDIDATES = [
    PROJECT_ROOT / "data" / "Parbhani" / "Parbhani.shp",
    PROJECT_ROOT / "data" / "Parabhani" / "Parabhani.shp",
]
STAGES_CSV = CSV_DIR / "tehsil_drought_stages.csv"

YEAR_START, YEAR_END = 2016, 2026
TEHSIL_COL = "SUB_DIST"          # shapefile attribute carrying tehsil names
CRS_UTM = "EPSG:32643"
CRS_WGS = "EPSG:4326"
EDGE_COLOR = "#2B2B2B"
EDGE_LW = 0.85
DPI = 300
FIGSIZE = (8.5, 10.0)
YEAR_ORDER = ["ndvi", "evi", "pci", "ssmi", "cdi"]
INDICATOR_COL = {"ndvi": "NDVI", "evi": "EVI", "pci": "PCI",
                 "ssmi": "SSMI", "cdi": "CDI"}
INDICATOR_LABEL = {"ndvi": "NDVI", "evi": "EVI", "pci": "PCI",
                   "ssmi": "SSMI", "cdi": "CDI"}

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Desaturated, colorblind-safe ramps (low value -> high value)
# ---------------------------------------------------------------------------
VEG_RAMP = ["#2F5D3A", "#5C8A4B", "#93B06A", "#C4C98D", "#D9C287", "#B4884C",
            "#7D4F23"]                                      # green -> brown
PCI_RAMP = ["#F2EAD3", "#D8C79A", "#B7A876", "#8F8558", "#5F5A3B"]
SSMI_RAMP = ["#7A4A2B", "#A9764E", "#D3B98C", "#E9E7DA", "#A9C6D4", "#5E8FA8",
             "#2F6076"]                                      # dry -> wet
CDI_RAMP = ["#8E2216", "#C34E2B", "#DE7F3E", "#EDB257", "#F7DA7C",
            "#FFF4C2"]                                       # dark red -> yellow


def _cmap(name: List[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("r", name)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_annual_matrices() -> Tuple[Dict[int, Dict[str, Dict[str, float]]],
                                    Dict[int, str]]:
    """Annual representative layer per tehsil per indicator.

    Vegetation / soil / PCI indices use the annual mean of available monthly
    values; CDI uses the annual minimum (peak drought severity). Missing
    (cloud-masked) months are skipped by pandas' NaN-aware mean.
    """
    df = pd.read_csv(STAGES_CSV)
    df["month"] = df["month"].astype(str)
    df["year"] = df["month"].str[:4].astype(int)
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)]

    matrices: Dict[int, Dict[str, Dict[str, float]]] = {}
    cdi_month: Dict[int, str] = {}
    for year, sub in df.groupby("year"):
        matrices[int(year)] = {}
        for key in YEAR_ORDER:
            col = INDICATOR_COL[key]
            if key == "cdi":
                # peak severity: the tehsil value is the minimum CDI month
                agg = sub.groupby("tehsil")[col].min()
                idx = sub.groupby("month")[col].mean().idxmin()
                cdi_month[int(year)] = str(idx)
            else:
                agg = sub.groupby("tehsil")[col].mean()
            matrices[int(year)][key] = {
                str(t): (float(v) if pd.notna(v) else float("nan"))
                for t, v in agg.items()}
    return matrices, cdi_month


def global_norm(matrices: Dict[int, Dict[str, Dict[str, float]]], key: str):
    """One colour scale per indicator across ALL years, so a colour means the
    same value in every annual map of that indicator (dashboard-identical)."""
    vals = [v for y in matrices.values()
            for v in y[key].values() if pd.notna(v)]
    vmin, vmax = float(min(vals)), float(max(vals))
    if key == "ssmi":
        m = max(abs(vmin), abs(vmax)) or 1.0
        return TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m), _cmap(SSMI_RAMP), \
            f"{vmin:.2f} to {vmax:+.2f}"
    if key == "cdi":
        return Normalize(vmin=vmin, vmax=vmax), _cmap(CDI_RAMP), \
            f"{vmin:.2f} to {vmax:+.2f}"
    cmap = _cmap(VEG_RAMP if key in ("ndvi", "evi") else PCI_RAMP)
    return Normalize(vmin=vmin, vmax=vmax), cmap, f"{vmin:.3f} to {vmax:.3f}"


# ---------------------------------------------------------------------------
# Map decorations
# ---------------------------------------------------------------------------
def _nice_step(span: float, target: int = 4) -> float:
    raw = span / max(target, 1)
    for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if raw <= s:
            return s
    return 5.0


def add_graticule(ax) -> None:
    """WGS-1984 lat/lon graticule ticks on the outer frame (data are
    EPSG:32643). Deterministic - labels are placed just outside the neatline."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xticks([])
    ax.set_yticks([])
    xmid, ymid = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    to_wgs = Transformer.from_crs(CRS_UTM, CRS_WGS, always_xy=True)
    to_utm = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)

    lons = [to_wgs.transform(x, y)[0] for x in (x0, x1) for y in (y0, y1)]
    lats = [to_wgs.transform(x, y)[1] for x in (x0, x1) for y in (y0, y1)]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    step_lon = _nice_step(lon_max - lon_min)
    step_lat = _nice_step(lat_max - lat_min)
    fmt_lon = "{:.1f}" if step_lon >= 0.25 else "{:.2f}"
    fmt_lat = "{:.1f}" if step_lat >= 0.25 else "{:.2f}"
    ticky = 0.022 * (y1 - y0)
    tickx = 0.022 * (x1 - x0)

    lon = np.ceil(lon_min / step_lon) * step_lon
    while lon < lon_max:
        X, _ = to_utm.transform(lon, ymid)
        if x0 < X < x1:
            for yy, off, va in ((y1, 3, "bottom"), (y0, -3, "top")):
                sgn = -1 if va == "bottom" else 1
                ax.plot([X, X], [yy, yy + sgn * ticky], color=EDGE_COLOR,
                        lw=0.8, clip_on=False, zorder=9)
                ax.annotate(fmt_lon.format(lon) + "\u00b0E", (X, yy),
                            xytext=(0, off), textcoords="offset points",
                            ha="center", va=va, fontsize=6.5, color="#333333",
                            clip_on=False)
        lon = round(lon + step_lon, 8)

    lat = np.ceil(lat_min / step_lat) * step_lat
    while lat < lat_max:
        _, Y = to_utm.transform(xmid, lat)
        if y0 < Y < y1:
            for xx, off, ha in ((x0, -3, "right"), (x1, 3, "left")):
                sgn = 1 if ha == "right" else -1
                ax.plot([xx, xx + sgn * tickx], [Y, Y], color=EDGE_COLOR,
                        lw=0.8, clip_on=False, zorder=9)
                ax.annotate(fmt_lat.format(lat) + "\u00b0N", (xx, Y),
                            xytext=(off, 0), textcoords="offset points",
                            ha=ha, va="center", fontsize=6.5, color="#333333",
                            clip_on=False)
        lat = round(lat + step_lat, 8)


def add_scalebar(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width, height = x1 - x0, y1 - y0
    bar_km = min([1, 2, 5, 10, 20, 25, 50, 100],
                 key=lambda c: abs(c - width * 0.28))
    bar_m = bar_km * 1000.0
    xl, yb = x0 + 0.06 * width, y0 + 0.045 * height
    h = 0.013 * height
    for i in range(2):
        c = "#2B2B2B" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rectangle((xl + i * bar_m / 2, yb), bar_m / 2, h,
                               facecolor=c, edgecolor=EDGE_COLOR, lw=0.6,
                               zorder=6))
    for frac, lab in ((0.0, "0"), (0.5, f"{bar_km / 2:g}"),
                      (1.0, f"{bar_km:g} km")):
        ax.text(xl + frac * bar_m, yb + h * 1.7, lab, ha="center",
                va="bottom", fontsize=7, color="#2B2B2B", zorder=6)


def add_north_arrow(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width, height = x1 - x0, y1 - y0
    cx, cy = x1 - 0.065 * width, y1 - 0.13 * height
    s = 0.030 * height
    ax.add_patch(MplPolygon([(cx, cy + s), (cx - 0.5 * s, cy - 0.65 * s),
                             (cx, cy - 0.2 * s), (cx + 0.5 * s, cy - 0.65 * s)],
                            closed=True, facecolor=EDGE_COLOR,
                            edgecolor=EDGE_COLOR, zorder=7))
    ax.text(cx, cy + 1.3 * s, "N", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=EDGE_COLOR, zorder=7)


def add_neatline(fig) -> None:
    fig.add_artist(Rectangle((0.008, 0.008), 0.984, 0.984,
                             transform=fig.transFigure, fill=False,
                             edgecolor=EDGE_COLOR, lw=1.0, zorder=20))


# ---------------------------------------------------------------------------
# Single-canvas renderer
# ---------------------------------------------------------------------------
def _frame(fig, title: str, subtitle: str, caption: str) -> None:
    fig.text(0.5, 0.958, title, ha="center", va="top", fontsize=13.5,
             fontweight="bold", color="#141414")
    fig.text(0.5, 0.928, subtitle, ha="center", va="top", fontsize=9,
             color="#555555")
    wrapped = "\n".join(textwrap.wrap(caption, 150))
    fig.text(0.5, 0.062, wrapped, ha="center", va="bottom", fontsize=8,
             color="#333333", linespacing=1.45)
    fig.text(0.07, 0.018, "Data: Sentinel-2 (GEE) \u00b7 Open-Meteo \u00b7 "
             "NASA SMAP L4 | Boundaries: Parbhani.shp | CRS: EPSG:32643 | "
             "WGS 1984 graticule", ha="left", va="bottom", fontsize=6.2,
             color="#808080")
    fig.text(0.93, 0.018, f"Prepared: {pd.Timestamp.now():%Y-%m-%d} \u00b7 "
             f"{DPI} dpi", ha="right", va="bottom", fontsize=6.2,
             color="#808080")


def _plot_polygons(ax, gdf: gpd.GeoDataFrame, column: Optional[str],
                   cmap, norm, cbar_label: str, show_cbar: bool) -> None:
    if show_cbar and column is not None:
        gdf.plot(ax=ax, column=column, cmap=cmap, norm=norm,
                 edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                 missing_kwds=dict(color="#D9D9D9", hatch="//",
                                   edgecolor=EDGE_COLOR),
                 zorder=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = ax.figure.add_axes([0.885, 0.30, 0.020, 0.42])
        cb = ax.figure.colorbar(sm, cax=cax)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_edgecolor(EDGE_COLOR)
        cb.outline.set_linewidth(0.5)
    else:
        gdf.plot(ax=ax, facecolor="#DCDCD2", edgecolor=EDGE_COLOR,
                 linewidth=EDGE_LW, zorder=2)


def _label_tehsils(ax, gdf: gpd.GeoDataFrame) -> None:
    import matplotlib.patheffects as pe
    pts = gdf.geometry.representative_point()
    halo = [pe.withStroke(linewidth=2.2, foreground="#FFFFFF")]
    for name, pt in zip(gdf[TEHSIL_COL], pts):
        ax.text(pt.x, pt.y, str(name), ha="center", va="center",
                fontsize=8.5, color="#141414", zorder=8,
                path_effects=halo)


def render_map(gdf_proj: gpd.GeoDataFrame, *, column: Optional[str],
               title: str, subtitle: str, caption: str, out_path: Path,
               cmap=None, norm=None, cbar_label: str = "",
               show_cbar: bool = True) -> Path:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.075, 0.115, 0.775, 0.765])
    _plot_polygons(ax, gdf_proj, column, cmap, norm, cbar_label, show_cbar)
    _label_tehsils(ax, gdf_proj)

    x0, y0, x1, y1 = gdf_proj.total_bounds
    padx, pady = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    ax.set_aspect("equal")
    add_graticule(ax)
    add_scalebar(ax)
    add_north_arrow(ax)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_edgecolor(EDGE_COLOR)
        ax.spines[side].set_linewidth(0.8)
    ax.set_facecolor("#FFFFFF")

    _frame(fig, title, subtitle, caption)
    add_neatline(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Dashboard integration
# ---------------------------------------------------------------------------
def build_annual_dashboard(matrices, cdi_month, index: List[dict]) -> Path:
    """Companion dashboard that embeds the EXACT same PNGs and annual data
    matrices / color breaks used for the static maps."""
    out = DASHBOARDS_DIR / "annual_gis_dashboard.html"
    DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
    chips = [
        ("NDVI / EVI", "green \u2192 brown (desaturated, colorblind-safe)"),
        ("PCI", "light sand \u2192 dark slate"),
        ("SSMI", "dry brown \u2192 wet teal (0-centred)"),
        ("CDI", "yellow \u2192 dark red (peak severity)"),
    ]
    sections = []
    for key in YEAR_ORDER:
        label = INDICATOR_LABEL[key]
        figs = [it for it in index if it["indicator"] == key]
        cells = "".join(
            f'<figure><img loading="lazy" src="../maps_output/{Path(it["file"]).name}"'
            f' alt="{it["file"]}"><figcaption>Figure {it["figure"]}: '
            f'{label} {it["year"]}</figcaption></figure>' for it in figs)
        sections.append(f'<section id="s_{key}"><h2>{label} \u2014 annual '
                        f'maps {YEAR_START}\u2013{YEAR_END}</h2>'
                        f'<div class="grid">{cells}</div></section>')
    matrix_json = json.dumps(
        {str(y): matrices[y] for y in sorted(matrices)}, ensure_ascii=False,
        indent=1)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parbhani Annual GIS Dashboard ({YEAR_START}\u2013{YEAR_END})</title>
<style>
 body{{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:#fafafa;color:#1a1a1a}}
 header{{position:sticky;top:0;background:#fff;border-bottom:2px solid #d9d9d9;padding:10px 18px;z-index:5}}
 header h1{{margin:0 0 4px;font-size:19px}}
 .chips{{display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#555}}
 nav a{{margin-right:14px;font-size:13px;color:#1a6faf;text-decoration:none;font-weight:600}}
 main{{padding:0 18px 30px;max-width:1750px;margin:0 auto}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
 figure{{margin:0;background:#fff;border:1px solid #d9d9d9;border-radius:6px;padding:8px}}
 figure img{{width:100%;height:auto;display:block}}
 figcaption{{font-size:12px;color:#555;padding:6px 2px 0}}
 section h2{{font-size:17px;border-bottom:1px solid #d9d9d9;padding-bottom:5px}}
 pre{{max-height:320px;overflow:auto;background:#fff;border:1px solid #d9d9d9;border-radius:6px;padding:10px;font-size:11px}}
</style></head><body>
<header><h1>Parbhani District \u2014 Annual GIS Dashboard</h1>
<div class="chips">{''.join(f'<span><b>{a}</b>: {b}</span>' for a,b in chips)}</div>
<nav>{''.join(f'<a href="#s_{k}">{INDICATOR_LABEL[k]}</a>' for k in YEAR_ORDER)}</nav></header>
<main>
<p>Static print maps and this dashboard share the identical source
 <code>tehsil_drought_stages.csv</code>, the identical annual aggregation
 rules, and the identical per-indicator colour breaks (see matrix below).
 CDI maps show the annual <b>minimum</b> monthly CDI (peak severity); all
 other indicators show the annual <b>mean</b> of available months.</p>
{''.join(sections)}
<section id="matrix"><h2>Annual data matrices (identical to map values)</h2>
<pre id="mtx">{matrix_json}</pre></section>
</main>
<footer style="border-top:2px solid #d9d9d9;background:#fff;padding:12px 18px;font-size:12px;color:#555">
 FastAPI engine serves the same data at <code>GET /dashboard</code> \u00b7
 generated {pd.Timestamp.now():%Y-%m-%d %H:%M}</footer>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


def trigger_dashboard_update(df_stages, gdf_4326, logger) -> List[str]:
    """Refresh the project dashboard engine so it reflects the current data
    state, then emit the annual companion dashboard."""
    logs: List[str] = []
    try:
        from src.utils import TelemetryTracker, load_config
        from src.viz.charts import (build_plotly_dashboard,
                                    build_plotly_timeline_dashboard)
        tel = TelemetryTracker()
        config = load_config()
        if "tehsil" not in gdf_4326.columns and TEHSIL_COL in gdf_4326.columns:
            gdf_4326 = gdf_4326.rename(columns={TEHSIL_COL: "tehsil"})
        df_weather = pd.read_csv(CSV_DIR / "weather_spi.csv",
                                 parse_dates=["time"]).set_index("time")
        s2 = CSV_DIR / "sentinel2_indices.csv"
        df_s2 = pd.read_csv(s2) if s2.exists() else None
        build_plotly_dashboard(df_weather, df_stages, df_s2, gdf_4326,
                               DASHBOARDS_DIR, tel, logger)
        build_plotly_timeline_dashboard(df_stages, DASHBOARDS_DIR, tel, logger)
        logs.append("[OK] interactive_drought_dashboard.html refreshed "
                    "(Plotly data-state update)")
        logs.append("[OK] drought_stage_timeline.html refreshed "
                    "(Plotly data-state update)")
    except Exception as exc:  # noqa: BLE001
        logs.append(f"[WARN] project Plotly dashboard refresh failed: {exc}")
    json_path = REPORTS_DIR / "drought_dashboard_data.json"
    if json_path.exists():
        logs.append(f"[OK] FastAPI GET /dashboard data-state valid -> "
                    f"{json_path.relative_to(PROJECT_ROOT)} "
                    f"({json_path.stat().st_size:,} bytes)")
    else:
        logs.append("[WARN] drought_dashboard_data.json missing - run the "
                    "pipeline to populate the FastAPI dashboard")
    return logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-dashboard", action="store_true",
                    help="export maps only, do not refresh dashboards")
    args = ap.parse_args()

    print("=" * 72)
    print(" PRINT-READY ANNUAL GIS RENDERER - Parbhani district "
          f"({YEAR_START}-{YEAR_END})")
    print("=" * 72)

    shp = next((p for p in SHAPEFILE_CANDIDATES if p.exists()), None)
    if shp is None:
        raise FileNotFoundError("Parbhani shapefile not found in "
                                + ", ".join(str(p) for p in SHAPEFILE_CANDIDATES))
    print(f"[input] shapefile : {shp.relative_to(PROJECT_ROOT)}")
    print(f"[input] data      : {STAGES_CSV.relative_to(PROJECT_ROOT)}")
    gdf_proj = gpd.read_file(shp).to_crs(CRS_UTM)
    gdf_4326 = gdf_proj.to_crs(CRS_WGS)

    matrices, cdi_month = load_annual_matrices()
    years = sorted(matrices)
    print(f"[info]  tehsils   : {len(gdf_proj)}  |  years: {years}")

    index: List[dict] = []
    figure_no = 1

    # 1) single administrative reference map --------------------------------
    ref_caption = ("Figure 1: Administrative reference map of the nine tehsils "
                   "of Parbhani district, Maharashtra, India. Coordinates in "
                   "WGS 1984 datum.")
    ref_path = render_map(
        gdf_proj, column=None, show_cbar=False,
        title="Baseline Reference Map \u2014 Parbhani District Tehsils",
        subtitle="Administrative boundaries and tehsil names only "
                 "(9 sub-districts)",
        caption=ref_caption,
        out_path=OUT_DIR / "reference_map_administrative.png")
    index.append({"file": str(ref_path.relative_to(PROJECT_ROOT)),
                  "indicator": "reference", "year": None,
                  "figure": figure_no, "caption": ref_caption})
    print(f"[{figure_no:>3}] {ref_path.name}")
    figure_no += 1

    # 2) per-year, per-indicator maps ---------------------------------------
    for year in years:
        for key in YEAR_ORDER:
            norm, cmap, rng = global_norm(matrices, key)
            label = INDICATOR_LABEL[key]
            values = matrices[year][key]
            work = gdf_proj[[TEHSIL_COL, "geometry"]].copy()
            work["_v"] = work[TEHSIL_COL].map(values)
            if key == "cdi":
                subtitle = (f"Peak drought severity \u2014 annual minimum "
                            f"monthly CDI \u00b7 worst district month: "
                            f"{cdi_month[year]} \u00b7 range {rng}")
                cbar_label = "CDI (annual minimum)"
                title = f"CDI Drought Severity \u2014 Parbhani District {year}"
            elif key == "ssmi":
                subtitle = (f"Annual mean soil-moisture anomaly \u00b7 range "
                            f"{rng} (0-centred)")
                cbar_label = "SSMI (annual mean)"
                title = f"SSMI Soil Moisture \u2014 Parbhani District {year}"
            else:
                subtitle = (f"Annual mean of available monthly tehsil values "
                            f"\u00b7 range {rng}")
                cbar_label = f"{label} (annual mean)"
                title = f"{label} \u2014 Parbhani District {year}"
            caption = (f"Figure {figure_no}: Spatial distribution of {label} "
                       f"across the tehsils of Parbhani district for the year "
                       f"{year}. Coordinates in WGS 1984 datum.")
            out = OUT_DIR / f"{year}_map_{key}.png"
            render_map(work, column="_v", cmap=cmap, norm=norm,
                       cbar_label=cbar_label, title=title, subtitle=subtitle,
                       caption=caption, out_path=out)
            index.append({"file": str(out.relative_to(PROJECT_ROOT)),
                          "indicator": key, "year": int(year),
                          "figure": figure_no, "caption": caption})
            print(f"[{figure_no:>3}] {out.name}")
            figure_no += 1

    n_maps = len(index)
    print("-" * 72)
    print(f"[done]  {n_maps} maps exported -> "
          f"{OUT_DIR.relative_to(PROJECT_ROOT)}")

    # 3) dashboard data-state update ----------------------------------------
    dash_logs: List[str] = []
    if not args.skip_dashboard:
        df_stages = pd.read_csv(STAGES_CSV)
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger("annual_gis")
        dash_logs = trigger_dashboard_update(df_stages, gdf_4326, logger)
        annual_html = build_annual_dashboard(matrices, cdi_month, index)
        dash_logs.append("[OK] annual_gis_dashboard.html written "
                         f"({n_maps} maps embedded, identical matrices/"
                         "ramps) -> "
                         f"{annual_html.relative_to(PROJECT_ROOT)}")
        manifest = OUT_DIR / "annual_layers_manifest.json"
        manifest.write_text(json.dumps(
            {"years": years, "indicators": YEAR_ORDER, "aggregation":
             {"NDVI": "annual mean", "EVI": "annual mean",
              "PCI": "annual mean", "SSMI": "annual mean",
              "CDI": "annual minimum (peak severity)"},
             "cdi_worst_month": {str(k): v for k, v in cdi_month.items()},
             "ramps": {"NDVI/EVI": VEG_RAMP, "PCI": PCI_RAMP,
                       "SSMI": SSMI_RAMP, "CDI": CDI_RAMP},
             "cdi_thresholds": {"extreme": -2.0, "severe": -1.5,
                                "moderate": -1.0, "mild": 0.0},
             "matrices": {str(y): matrices[y] for y in years},
             "figures": index},
            ensure_ascii=False, indent=1), encoding="utf-8")
        dash_logs.append("[OK] annual_layers_manifest.json written "
                         "(matrices + ramps + thresholds)")

    # 4) structured console report ------------------------------------------
    print("-" * 72)
    print(" ACADEMIC CAPTIONS & FILE LIST")
    print("-" * 72)
    for it in index:
        print(f"  {it['file']}")
        print(f"      {it['caption']}")
    print("-" * 72)
    print(" DASHBOARD COMPONENT UPDATE LOGS")
    print("-" * 72)
    if args.skip_dashboard:
        print("  [SKIP] dashboard refresh disabled (--skip-dashboard)")
    else:
        for line in dash_logs:
            print(f"  {line}")
    print("=" * 72)
    print(f" COMPLETE - {n_maps} standalone 300 dpi figures in "
          f"{OUT_DIR.name}/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
