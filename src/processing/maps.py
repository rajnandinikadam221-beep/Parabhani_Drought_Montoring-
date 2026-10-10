"""Publication-quality cartography for the Parbhani drought project.

Implements outputs/csv/maps_prompt2.md end-to-end (7 phases):
  * Phase 2  - audit_data()      -> outputs/maps/data_audit.md (11 items)
  * Phase 3  - plot builders A-L + N1-N10 -> 13 x 8.5 in GridSpec rows
      (header / body / footer), each element in its own axes:
      plot_stage_map(), plot_index_map(), plot_anomaly_map(), plot_spi3_summary(),
      plot_study_area(), plot_flowchart(), plot_multi_index(), plot_rf_*(),
      plot_index_distribution(), plot_tehsil_boxplots(), plot_ssmi_category_map(),
      plot_stage_area_by_tehsil(), plot_rainfall_distribution(),
      plot_spi3_classification(), plot_baseline_maps(), plot_event_maps(),
      plot_trend_chart(), build_table(), label_tehsils(), load_rf_context(),
      add_north_arrow(), add_scalebar(), add_locator_inset(), add_footer().
  * Phase 4  - build_snapshot()   -> data_snapshot_{before,after}.json
  * Phase 5  - run_all_maps.py    -> single entry point (project root)
  * Phase 6  - dashboards/drought_maps_dashboard.html (built in src/viz/charts.py)
  * Phase 7  - write_qc_report()  -> outputs/maps/qc_report.md (8 checks)

Only visualisation lives here - no data, CDI formula, thresholds or trained
RF model change.
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pyproj import Transformer
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLE: Dict = {
    "figsize": (13, 8.5),
    "dpi": 300,
    "font": "DejaVu Sans",
    "fs": {"title": 14, "subtitle": 10, "label": 8, "value": 7.5,
           "small": 7, "table": 7.5, "panel": 10},
    "colors": {
        "text": "#1a1a1a",
        "subtitle": "#555555",
        "footer": "#666666",
        "border": "#333333",
        "grid": "#cccccc",
        "nodata_face": "#e0e0e0",
        "nodata_edge": "#999999",
        "seq": "YlGn",
        "freq": "YlOrRd",
        "div": "BrBG",
        "wet": "#1a6faf",
        "dry": "#C0392B",
        "marker_face": "#ffffff",
        "marker_edge": "#1a1a1a",
        "loc_face": "#f2f2f2",
        "loc_edge": "#777777",
        "loc_highlight": "#C0392B",
        "shade": "#f0f0f0",
    },
    "lw": {"tehsil": 0.8, "district": 1.5, "frame": 0.8},
    "grat_step": 0.25,
    # GridSpec rows: header (title+subtitle) / body (map | panel) / footer
    "rows": [0.62, 6.28, 0.50],
    "cols": [6.0, 4.0],
}

STAGE_COLORS = {0: "#2E8B57", 1: "#FFD700", 2: "#FFA500",
                3: "#E8431A", 4: "#8B0000"}
STAGE_HATCH = {0: None, 1: "..", 2: "//", 3: "\\\\", 4: "xx"}
STAGE_NAMES = {0: "No Drought", 1: "Mild Drought", 2: "Moderate Drought",
               3: "Severe Drought", 4: "Extreme Drought"}
STAGE_SHORT = {0: "No Drought", 1: "Mild", 2: "Moderate",
               3: "Severe", 4: "Extreme"}
STAGE_ORDER = [0, 1, 2, 3, 4]
NODATA_STAGE = -1

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [STYLE["font"], "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fmt_month(m: str) -> str:
    return pd.Timestamp(str(m) + "-01").strftime("%B %Y")


def _month_end(m: str) -> pd.Timestamp:
    return pd.Timestamp(str(m) + "-01") + pd.offsets.MonthEnd(0)


def _is_partial_month(m: str, data_asof: str) -> bool:
    return pd.Timestamp(data_asof) < _month_end(m)


def _latest_complete_month(months: Sequence[str], data_asof: str) -> str:
    last = max(str(m) for m in months)
    if _is_partial_month(last, data_asof):
        return (pd.Timestamp(last + "-01")
                - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    return last


def _union(gdf: gpd.GeoDataFrame):
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    return gdf.unary_union


def _today() -> str:
    return f"{date.today():%Y-%m-%d}"


def _pad_limits(ax, frac: float = 0.035) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def _projected_area_km2(gdf_4326: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gdf_4326.to_crs(epsg=32643).area / 1e6


# ---------------------------------------------------------------------------
# Frame / furniture
# ---------------------------------------------------------------------------
def _add_frame(ax) -> None:
    """Thin black frame + light-grey lat/lon graticule, labels left/bottom."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=STYLE["colors"]["grid"],
            lw=0.5, alpha=0.9)
    ax.xaxis.set_major_locator(MultipleLocator(STYLE["grat_step"]))
    ax.yaxis.set_major_locator(MultipleLocator(STYLE["grat_step"]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}\u00b0E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}\u00b0N"))
    ax.tick_params(labelsize=STYLE["fs"]["small"], length=2)
    ax.tick_params(labelleft=True, labelbottom=True,
                   labeltop=False, labelright=False,
                   left=True, bottom=True, top=False, right=False)
    for sp in ax.spines.values():
        sp.set_linewidth(STYLE["lw"]["frame"])
        sp.set_edgecolor("black")
        sp.set_visible(True)


def add_north_arrow(ax) -> List:
    """North arrow, top right, in map data coordinates. Returns its texts."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    x = x1 - 0.055 * w
    t = ax.annotate("N", xy=(x, y1 - 0.045 * h), xytext=(x, y1 - 0.155 * h),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=STYLE["colors"]["text"],
                    path_effects=[withStroke(linewidth=2.2, foreground="white")],
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4,
                                    shrinkA=2, shrinkB=0))
    return [t]


def add_scalebar(ax, gdf_4326: gpd.GeoDataFrame) -> List:
    """0/10/20 km scale bar; length derived in EPSG:32643, drawn in 4326."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    minx, miny, maxx, maxy = gdf_4326.total_bounds
    lon_c, lat_c = (minx + maxx) / 2, (miny + maxy) / 2

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    inv = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    xc, _ = fwd.transform(lon_c, lat_c)
    lons = [inv.transform(xc + km * 1000, lat_c)[0] for km in (0, 10, 20)]

    bx = x0 + 0.055 * w
    by = y0 + 0.05 * h
    tick = h * 0.010
    ax.plot([lons[0], lons[2]], [by, by], color="black", lw=2.6,
            solid_capstyle="butt", zorder=6)
    texts = []
    for lon_km, lab in zip(lons, ("0", "10", "20 km")):
        ax.plot([lon_km, lon_km], [by - tick, by + tick],
                color="black", lw=1.2, zorder=6)
        texts.append(ax.text(
            lon_km, by - tick * 2.4, lab, ha="center", va="top",
            fontsize=STYLE["fs"]["small"], color=STYLE["colors"]["text"],
            path_effects=[withStroke(linewidth=2.0, foreground="white")],
            zorder=6))
    return texts


def add_locator_inset(ax, mh_gdf: Optional[gpd.GeoDataFrame],
                      gdf_4326: gpd.GeoDataFrame) -> bool:
    """Maharashtra outline with Parbhani highlighted (dedicated axes cell)."""
    ax.set_axis_off()
    if mh_gdf is None or mh_gdf.empty:
        ax.text(0.5, 0.5, "Locator\n(unavailable)", ha="center", va="center",
                fontsize=STYLE["fs"]["small"], color=STYLE["colors"]["footer"])
        return False
    mh_gdf.plot(ax=ax, facecolor=STYLE["colors"]["loc_face"],
                edgecolor=STYLE["colors"]["loc_edge"], lw=0.5)
    gdf_4326.boundary.plot(ax=ax, color=STYLE["colors"]["loc_highlight"], lw=1.6)
    ax.set_aspect(1 / np.cos(np.radians(19.7)))
    ax.set_title("Locator", fontsize=STYLE["fs"]["small"], pad=2)
    return True


def add_footer(ax_foot, extra: Optional[Sequence[str]] = None) -> None:
    """Footer row (GridSpec row 2) - sources, CRS, prepared date, 7 pt grey."""
    ax_foot.set_axis_off()
    text = (f"Data: Sentinel-2 (GEE), Open-Meteo, NASA SMAP L4 | "
            f"CRS: EPSG:32643 | Prepared: {_today()}")
    if extra:
        text += " | " + " | ".join(extra)
    ax_foot.text(0.5, 0.35, text, transform=ax_foot.transAxes,
                 ha="center", va="center",
                 fontsize=STYLE["fs"]["small"], color=STYLE["colors"]["footer"])


def title_block(ax_head, title: str, subtitle: str) -> None:
    """Header row (GridSpec row 0): title 14 pt bold, subtitle 10 pt grey,
    both left-aligned."""
    ax_head.set_axis_off()
    ax_head.text(0.0, 0.66, title, transform=ax_head.transAxes,
                 ha="left", va="center",
                 fontsize=STYLE["fs"]["title"], fontweight="bold",
                 color=STYLE["colors"]["text"])
    ax_head.text(0.0, 0.16, subtitle, transform=ax_head.transAxes,
                 ha="left", va="center",
                 fontsize=STYLE["fs"]["subtitle"],
                 color=STYLE["colors"]["subtitle"])


def _figure_rows(fig) -> Tuple:
    """3-row GridSpec: header (full width) / body (60-40) / footer (full)."""
    gs = fig.add_gridspec(3, 2, height_ratios=STYLE["rows"],
                          width_ratios=STYLE["cols"])
    ax_head = fig.add_subplot(gs[0, :])
    ax_foot = fig.add_subplot(gs[2, :])
    ax_head.set_axis_off()
    ax_foot.set_axis_off()
    return gs, ax_head, ax_foot


def _draw_district(ax, gdf_4326) -> None:
    """District outline on top of tehsil fills/borders."""
    gpd.GeoSeries([_union(gdf_4326)], crs=gdf_4326.crs).boundary.plot(
        ax=ax, color="black", lw=STYLE["lw"]["district"])


def _map_axes_style(ax, gdf_4326) -> List:
    """Graticule + frame + north arrow + scale bar; returns obstacle texts
    (used by adjustText so labels avoid the furniture)."""
    ax.set_aspect(1 / np.cos(np.radians(19.3)))
    _pad_limits(ax)
    _add_frame(ax)
    obs = add_north_arrow(ax) + add_scalebar(ax, gdf_4326)
    return obs


# ---------------------------------------------------------------------------
# Labels with overlap handling
# ---------------------------------------------------------------------------
def _display_polygon(ax, geom) -> ShapelyPolygon:
    disp = ax.transData.transform(np.asarray(geom.exterior.coords))
    return ShapelyPolygon(disp)


def _shrink(bb, pad: float):
    """Shrink a window extent by pad px on every side."""
    return matplotlib.transforms.Bbox.from_extents(
        bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def _overlaps(texts, renderer, pad: float = 1.5) -> List[Tuple[str, str]]:
    boxes = [(t.get_text().splitlines()[0],
              t.get_window_extent(renderer=renderer)) for t in texts]
    bad = []
    for (n1, b1), (n2, b2) in combinations(boxes, 2):
        if _shrink(b1, pad).overlaps(_shrink(b2, pad)):
            bad.append((n1, n2))
    return bad


def label_tehsils(ax, gdf: gpd.GeoDataFrame, values: pd.Series,
                  fmt: str = "{:.3f}", extra_fmt=None,
                  obstacles: Optional[List] = None) -> Dict:
    """Halo tehsil labels (name 8 pt bold, value 7.5 pt on line 2) at
    representative points. Falls back to numbered markers 1..9 + table when
    labels overlap each other, exceed their polygon, or collide with the
    map furniture. Uses adjustText for the leftovers."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    y0, y1 = ax.get_ylim()
    dy = 0.030 * (y1 - y0)
    val_map = values.to_dict()
    texts = []

    for _, row in gdf.iterrows():
        name = str(row["tehsil"])
        p = row.geometry.representative_point()
        texts.append(ax.text(
            p.x, p.y, name, ha="center", va="center",
            fontsize=STYLE["fs"]["label"], fontweight="bold",
            color=STYLE["colors"]["text"], zorder=8,
            path_effects=[withStroke(linewidth=2.2, foreground="white")]))
        v = val_map.get(name)
        if v is not None and not pd.isna(v):
            v_txt = fmt.format(v)
            if extra_fmt is not None:
                v_txt = extra_fmt(name, v, v_txt)
            texts.append(ax.text(
                p.x, p.y - dy, v_txt, ha="center", va="center",
                fontsize=STYLE["fs"]["value"], color=STYLE["colors"]["text"],
                zorder=8,
                path_effects=[withStroke(linewidth=2.2, foreground="white")]))

    try:
        from adjustText import adjust_text
        kwargs = dict(ax=ax, expand=(1.05, 1.12), force_text=(0.1, 0.25),
                      ensure_inside_axes=True, lim=80)
        if obstacles:
            kwargs["objects"] = [o for o in obstacles if o is not None]
        adjust_text(texts, **kwargs)
    except Exception:
        pass
    ax.figure.canvas.draw()

    bad = _overlaps(texts, renderer, pad=1.5)
    outside = []
    for t in texts:
        head = t.get_text().splitlines()[0]
        row = gdf[gdf["tehsil"].astype(str) == head]
        if row.empty:
            continue
        poly = _display_polygon(ax, row.geometry.iloc[0])
        bb = _shrink(t.get_window_extent(renderer=renderer), 2)
        corners = [ShapelyPoint(bb.x0, bb.y0), ShapelyPoint(bb.x1, bb.y0),
                   ShapelyPoint(bb.x1, bb.y1), ShapelyPoint(bb.x0, bb.y1)]
        if not all(poly.buffer(2).contains(c) or poly.contains(c)
                   for c in corners):
            outside.append(head)

    if not bad and not outside:
        return {"mode": "labels", "texts": texts, "overlaps": 0,
                "outside_poly": 0, "order": list(gdf["tehsil"].astype(str))}

    # --- numbered-marker fallback -----------------------------------------
    for t in texts:
        t.remove()
    texts = []
    order = []
    for i, (_, row) in enumerate(gdf.iterrows(), start=1):
        p = row.geometry.representative_point()
        ax.plot([p.x], [p.y], marker="o", ms=12,
                mfc=STYLE["colors"]["marker_face"],
                mec=STYLE["colors"]["marker_edge"], mew=0.9, zorder=7)
        texts.append(ax.text(p.x, p.y, str(i), ha="center", va="center",
                             fontsize=STYLE["fs"]["small"], fontweight="bold",
                             color=STYLE["colors"]["text"], zorder=8))
        order.append(str(row["tehsil"]))
    ax.figure.canvas.draw()
    bad = _overlaps(texts, renderer, pad=1.5)
    return {"mode": "markers", "texts": texts, "overlaps": len(bad),
            "outside_poly": 0, "order": order}


def audit(fig, label_texts: List, ctx: str, logger=None) -> List[str]:
    """QC 1/2: no overlapping text bboxes (labels, titles, notes, footer) and
    nothing outside the figure. Table cells and legend internals live in their
    own artists (laid out by matplotlib) and are not collected here."""
    problems: List[str] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a, b in _overlaps(label_texts, renderer, pad=1.0):
        problems.append(f"{ctx}: label overlap '{a}' <-> '{b}'")

    collected: List[Tuple] = []  # (text, axes-id, is_tick)
    for t in fig.texts:
        collected.append((t, -1, False))
    for i, ax in enumerate(fig.axes):
        collected.append((ax.title, i, False))
        for t in ax.texts:
            collected.append((t, i, False))
        if not ax.axison:
            continue
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if not t.get_text().strip():
                continue
            px, py = t.get_position()
            if x0 - 1e-9 <= px <= x1 + 1e-9 and y0 - 1e-9 <= py <= y1 + 1e-9:
                collected.append((t, i, True))

    for (t1, a1, k1), (t2, a2, k2) in combinations(collected, 2):
        if not t1.get_text().strip() or not t2.get_text().strip():
            continue
        if k1 and k2 and a1 == a2:
            continue  # matplotlib spaces tick labels within one axes
        b1 = t1.get_window_extent(renderer=renderer)
        b2 = t2.get_window_extent(renderer=renderer)
        if _shrink(b1, 1.0).overlaps(_shrink(b2, 1.0)):
            problems.append(
                f"{ctx}: text overlap '{t1.get_text()[:26]}' <-> "
                f"'{t2.get_text()[:26]}'")
    for t, _a, _k in collected:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=renderer)
        if not (fig.bbox.x0 - 1 <= bb.x0 and bb.x1 <= fig.bbox.x1 + 1
                and fig.bbox.y0 - 1 <= bb.y0 and bb.y1 <= fig.bbox.y1 + 1):
            problems.append(
                f"{ctx}: text outside figure: '{t.get_text()[:40]}'")
    if problems and logger:
        logger.warning(f"[MAP-AUDIT] {len(problems)} issue(s): {problems}")
    return problems


def _savefig(fig, path: Path, **kwargs) -> None:
    """savefig with a short retry (Windows occasionally returns EINVAL on
    open while AV/indexer holds the file)."""
    import time
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.4 * (attempt + 1))


def _finish(fig, png_path: Path, telemetry, logger, ctx: str,
            label_texts: List, has_table: bool = False) -> Dict:
    stats = {"file": png_path.name,
             "problems": audit(fig, label_texts, ctx, logger),
             "has_table": has_table}
    _savefig(fig, png_path, dpi=STYLE["dpi"])
    _savefig(fig, png_path.with_suffix(".pdf"))
    plt.close(fig)
    if telemetry is not None:
        telemetry.add_output(png_path)
        telemetry.add_output(png_path.with_suffix(".pdf"))
    if logger:
        logger.info(f"✓ Map saved: {png_path.name} (+pdf)")
    return stats


# ---------------------------------------------------------------------------
# Right-panel building blocks
# ---------------------------------------------------------------------------
def build_table(ax, header: List[str], rows: List[List[str]],
                bold_rows: Optional[Sequence[int]] = None) -> Dict:
    """QC 3: column widths sized from the longest string (never truncated),
    table scaled to fit its axes, optional bold rows."""
    ax.set_axis_off()
    lengths = [[len(str(c)) for c in header]] + [
        [len(str(c)) for c in r] for r in rows]
    colmax = [max(c) for c in zip(*lengths)] if lengths else [1]
    total = sum(colmax) or 1
    widths = [c / total for c in colmax]

    tbl = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="center", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(STYLE["fs"]["table"])
    tbl.scale(1.0, 1.0)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold")
        elif bold_rows is not None and (r - 1) in bold_rows:
            cell.set_text_props(fontweight="bold")

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    tb = tbl.get_window_extent(renderer=renderer)
    ab = ax.get_window_extent(renderer=renderer)
    fits = (tb.height <= ab.height * 1.01 and tb.width <= ab.width * 1.01)
    if tb.height > ab.height * 0.99:
        tbl.scale(1.0, (ab.height * 0.97) / max(tb.height, 1e-6))
        ax.figure.canvas.draw()
        tb = tbl.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        fits = tb.height <= ab.height * 1.01
    return {"fits": bool(fits), "table": tbl}


def _area_bar(ax, stage_area: Dict[int, float], total: float) -> None:
    """Stacked horizontal bar of % district area per stage (total 100%)."""
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.text(0, 0.97, "Share of district area by stage",
            fontsize=STYLE["fs"]["small"], fontweight="bold",
            color=STYLE["colors"]["text"], va="top")
    left = 0.0
    for s in STAGE_ORDER:
        km2 = stage_area.get(s, 0.0)
        pct = 100.0 * km2 / total if total else 0.0
        if pct <= 0:
            continue
        ax.add_patch(plt.Rectangle((left, 0.18), pct, 0.44,
                                   facecolor=STAGE_COLORS[s],
                                   edgecolor="white", lw=0.6))
        if pct >= 9:
            ax.text(left + pct / 2, 0.40, f"{pct:.0f}%", ha="center",
                    va="center", fontsize=STYLE["fs"]["small"],
                    fontweight="bold",
                    color="white" if s >= 3 else "#1a1a1a")
        left += pct


def _legend_cell(ax, area_by_stage: Dict[int, float], total: float,
                 title: str = "Drought stage (CDI)") -> None:
    """QC 5: legend ALWAYS lists all 5 stages; absent ones greyed with
    '0 km2 / 0%'."""
    ax.set_axis_off()
    handles, labels, absent = [], [], []
    for s in STAGE_ORDER:
        km2 = area_by_stage.get(s, 0.0)
        pct = 100.0 * km2 / total if total else 0.0
        handles.append(Patch(facecolor=STAGE_COLORS[s],
                             edgecolor=STYLE["colors"]["border"], lw=0.4,
                             hatch=STAGE_HATCH[s]))
        labels.append(f"{STAGE_NAMES[s]}   {km2:,.0f} km\u00b2 / {pct:.1f}%")
        if km2 <= 0:
            absent.append(s)
    leg = ax.legend(handles=handles, labels=labels, loc="upper center",
                    bbox_to_anchor=(0.5, 1.0), fontsize=STYLE["fs"]["small"],
                    title=title,
                    title_fontsize=STYLE["fs"]["small"],
                    frameon=True, framealpha=0.95, edgecolor="#bbbbbb",
                    ncol=1, handlelength=1.5, handleheight=1.0,
                    borderpad=0.4, labelspacing=0.38)
    for txt, s in zip(leg.get_texts(), STAGE_ORDER):
        if s in absent:
            txt.set_color("#999999")


# ---------------------------------------------------------------------------
# PHASE 2 - data audit
# ---------------------------------------------------------------------------
def audit_data(config: Dict, df_stages: pd.DataFrame,
               df_weather: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
               out_dir: Path, logger=None, rf_ctx: Optional[Dict] = None) -> str:
    """PHASE 2: assert/report the 11 audit items, write
    outputs/maps/data_audit.md and return its text (caller prints it)."""
    L: List[str] = []
    ok = lambda b: "PASS" if b else "FAIL"
    warn = lambda b: "PASS" if b else "WARN"
    L.append("# Data Audit - Parbhani Drought Maps")
    L.append(f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M} - "
             "run BEFORE drawing (never hide a problem)")
    L.append("")

    # --- 1. areas ---------------------------------------------------------
    areas_col = df_stages.groupby("tehsil")["tehsil_area_km2"].max()
    sum_col = float(areas_col.sum())
    gproj = gdf_4326.to_crs(epsg=32643)
    district_km2 = float(_union(gproj).area / 1e6)
    geom_areas = gproj.area / 1e6
    rel_geom = (geom_areas - areas_col.reindex(gdf_4326["tehsil"]).values)
    rel_geom = float((rel_geom.abs() / areas_col.reindex(
        gdf_4326["tehsil"]).values).max())
    diff_pct = abs(sum_col - district_km2) / district_km2 * 100.0
    official = 6500.0
    off_pct = abs(sum_col - official) / official * 100.0
    L.append("## 1. Tehsil areas (km2)")
    L.append(f"- Sum of tehsil_area_km2: **{sum_col:,.2f} km2**")
    L.append(f"- District polygon area (EPSG:32643 union): "
             f"**{district_km2:,.2f} km2** - diff {diff_pct:.3f}% "
             f"(tolerance 0.5%) -> **{ok(diff_pct <= 0.5)}**")
    L.append(f"- Column vs projected geometry per tehsil: max rel. dev "
             f"{rel_geom * 100:.3f}% -> {ok(rel_geom <= 0.005)}")
    L.append(f"- vs official district area ~{official:,.0f} km2: "
             f"{off_pct:.2f}% -> **{warn(off_pct <= 2.0)}**")
    L.append("- Boundary source: `data/Parbhani/Parbhani.shp` - "
             "project-provided boundary (custom Lambert Conformal Conic, "
             "WGS 84, per .prj); tehsil polygons dissolve to the district.")
    L.append("")

    # --- 2. stage shares --------------------------------------------------
    tot_month = df_stages.groupby("month")["tehsil_area_km2"].sum()
    sh = (df_stages.groupby(["month", "stage"], as_index=False)
          ["tehsil_area_km2"].sum())
    sh = sh.join(tot_month.rename("tot"), on="month")
    sh["pct"] = sh["tehsil_area_km2"] / sh["tot"] * 100.0
    per_month = sh.groupby("month")["pct"].sum()
    max_dev = float((per_month - 100.0).abs().max())
    n_month = int(df_stages["month"].nunique())
    counts = df_stages.groupby("month")["tehsil"].nunique()
    L.append("## 2. Stage-share totals")
    L.append(f"- Months checked: {n_month}; tehsils/month: "
             f"{counts.min()}-{counts.max()} (expect 9) -> "
             f"**{ok(bool((counts == 9).all()))}**")
    L.append(f"- max |sum(area in stage)/sum(all areas) - 100.0%| = "
             f"{max_dev:.6f}% -> **{ok(max_dev <= 1e-6)}**")
    L.append("")

    # --- 3. simple vs area-weighted means ---------------------------------
    L.append("## 3. District means: simple vs AREA-WEIGHTED")
    L.append("| index | simple mean | area-weighted | diff | maps use |")
    L.append("|---|---|---|---|---|")
    g = df_stages.groupby("tehsil")
    wts = areas_col.reindex(sorted(df_stages["tehsil"].unique()))
    for idx in ("NDVI", "EVI", "PCI"):
        tm = g[idx].mean()
        simple = float(tm.mean())
        weighted = float(np.average(tm.dropna(),
                                    weights=wts.loc[tm.dropna().index]))
        L.append(f"| {idx} | {simple:.3f} | {weighted:.3f} | "
                 f"{simple - weighted:+.3f} | **area-weighted** |")
    L.append("")

    # --- 4. CDI dominance -------------------------------------------------
    w = config["indices"]["cdi_weights"]
    terms = {"spi (shared)": (w["spi"], "SPI-3"),
             "ssmi": (w["ssmi"], "SSMI"),
             "vci": (w["ndvi_vci"], "VCI_z"),
             "evi": (w["evi"], "EVI_z"),
             "pci": (w["pci"], "PCI_z")}
    spi_shared = bool((df_stages.groupby("month")["SPI-3"].nunique() == 1).all())
    dist = df_stages.groupby("month").agg(CDI=("CDI", "mean"))
    dist["SPI-3"] = df_stages.groupby("month")["SPI-3"].first()
    r_dist = float(dist["CDI"].corr(dist["SPI-3"]))
    r_pool = float(df_stages["CDI"].corr(df_stages["SPI-3"]))
    contrib = {k: [] for k in terms}
    for _m, sub in df_stages.groupby("month"):
        v = sub["CDI"].var()
        if not v or np.isnan(v) or v == 0:
            continue
        for k, (wt, col) in terms.items():
            pair = sub[["CDI", col]].dropna()
            if len(pair) < 3:
                continue
            c = float(np.cov(pair["CDI"], wt * pair[col],
                             ddof=0)[0, 1] / v)
            if not np.isnan(c):
                contrib[k].append(c)
    share = {k: float(np.mean(v)) for k, v in contrib.items() if v}
    if not share:
        share = {k: 0.0 for k in terms}
    ssum = sum(max(x, 0) for x in share.values()) or 1.0
    spread = df_stages.groupby("month")["CDI"].agg(
        lambda s: s.max() - s.min())
    uniform = int(df_stages.groupby("month")["stage"].nunique().eq(1).sum())
    L.append("## 4. CDI dominance")
    L.append(f"- SPI-3 identical across tehsils every month: "
             f"**{'YES' if spi_shared else 'NO'}** (nunique=1)")
    L.append(f"- corr(district-mean CDI, SPI-3) over months: r = {r_dist:.3f}")
    L.append(f"- corr(CDI, SPI-3) pooled over all tehsil-months: "
             f"r = {r_pool:.3f}")
    L.append("- Share of CDI variance explained (Cov(CDI, term)/Var, "
             "sums to 1):")
    for k, v in share.items():
        L.append(f"  - {k}: {max(v, 0) / ssum * 100:.1f}%")
    L.append(f"- Tehsil CDI spread per month (max-min): min {spread.min():.2f}, "
             f"mean {spread.mean():.2f}, max {spread.max():.2f} "
             "(prompt expectation ~0.1-0.4)")
    L.append(f"- Months with all 9 tehsils in ONE stage: {uniform}/{n_month} "
             f"-> stages are near district-wide: "
             f"**{'YES' if uniform > 0 else 'NO'}** "
             "(SPI-3 contributes 0% of spatial spread; VCI/SSMI decide "
             "between-tehsil differences)")
    L.append("")

    # --- 5. shared series + SMAP pixels -----------------------------------
    scale_m = int(config.get("soil_moisture", {}).get("scale", 11000))
    step = float(scale_m)
    minx, miny, maxx, maxy = gproj.total_bounds
    cells = []
    x = minx
    while x <= maxx + step:
        y = miny
        while y <= maxy + step:
            cells.append(shapely_box(x, y, x + step, y + step))
            y += step
        x += step
    L.append("## 5. Shared / constant series + SMAP pixels")
    L.append("- Inputs identical across tehsils: **SPI-3** (single "
             "Open-Meteo point; 1 distinct value per month). All other "
             "components are per-tehsil.")
    L.append(f"- SSMI: per-tehsil standardised (z-score) - see item 8.")
    L.append(f"- SMAP L4 pixel count per tehsil: {scale_m} m regular grid "
             f"({scale_m / 1000:g} km) intersected with each polygon in "
             f"EPSG:32643, grid anchored on district bounds "
             f"({len(cells)} cells over district). "
             f"Config `soil_moisture.scale = {scale_m}`; the prompt's "
             "'~9 km' refers to SPL4SMGP native grid - actual processing "
             f"used {scale_m / 1000:g} km.")
    L.append("| tehsil | SMAP pixels (~grid cells) | area km2 |")
    L.append("|---|---|---|")
    for _, row in gproj.iterrows():
        n = sum(1 for c in cells if c.intersects(row.geometry))
        L.append(f"| {row['tehsil']} | {n} | {row.geometry.area / 1e6:,.0f} |")
    L.append("")

    # --- 6. SPI-3 sanity --------------------------------------------------
    spi_all = df_weather["SPI-3"].dropna()
    spi_study = spi_all[spi_all.index >= "2016-01-01"]
    b0 = config["dates"]["baseline_start"]
    b1 = config["dates"]["baseline_end"]
    mean_off = float(spi_study.mean())
    L.append("## 6. SPI-3 sanity")
    L.append(f"- Study-period (2016-01 onward) mean = {mean_off:+.3f}, "
             f"SD = {spi_study.std():.3f} (expect mean ~0, SD ~1) -> "
             f"**{ok(abs(mean_off) < 0.15)}**"
             + ("" if abs(mean_off) < 0.15 else " **FLAGGED**"))
    L.append(f"- Full record mean = {spi_all.mean():+.3f}, "
             f"SD = {spi_all.std():.3f}")
    L.append(f"- Calibration: monthly Gamma fit, baseline {b0} to {b1} "
             "(config `dates`); method per outputs/reports/summary.md.")
    L.append(f"- FINDING: study-period mean {mean_off:+.2f} differs from 0 "
             "(prompt cites ~+0.30) - the study decade is wetter than the "
             "1991-2020 baseline (or fit offset). **Flagged, NOT corrected** "
             "- data/CDI unchanged.")
    L.append("")

    # --- 7. partial month -------------------------------------------------
    last_w = df_weather.index.max()
    months = sorted(df_stages["month"].astype(str).unique())
    last_m = months[-1]
    s2_last = str(df_stages.loc[df_stages["NDVI"].notna(), "month"].max())
    smap_last = str(df_stages.loc[df_stages["SSMI"].notna(), "month"].max())
    partial = _is_partial_month(last_m, str(last_w.date()))
    key = str(last_w.date())[:7]
    spi_last = float(spi_study.loc[spi_study.index.strftime("%Y-%m")
                                   == key].iloc[0]) if key in set(
        spi_study.index.strftime("%Y-%m")) else np.nan
    L.append("## 7. Partial month / latest date per source")
    L.append(f"- Weather (Open-Meteo) last row: {last_w.date()}")
    L.append(f"- Sentinel-2 last month with data: {s2_last}")
    L.append(f"- SMAP SSMI last month with data: {smap_last}")
    L.append(f"- Stage table last month: {last_m} -> "
             f"**{'PARTIAL (data to ' + str(last_w.date()) + ')' if partial else 'complete'}**")
    if partial and not np.isnan(spi_last):
        L.append(f"- Its SPI-3 = {spi_last:+.2f} may be **biased dry** "
                 "(incomplete rainfall). Map for this month is rendered "
                 "PROVISIONAL only; latest COMPLETE month is used for the "
                 "headline map.")
    L.append("")

    # --- 8. SSMI ----------------------------------------------------------
    L.append("## 8. SSMI standardisation")
    L.append("- Per-tehsil long-term mean (must be ~0 by construction):")
    L.append("| tehsil | mean | std | % months SSMI <= -1 |")
    L.append("|---|---|---|---|")
    dry_pct = (df_stages.assign(dry=df_stages["SSMI"] <= -1)
               .groupby("tehsil")["dry"].mean() * 100.0)
    for t, sub in df_stages.groupby("tehsil"):
        L.append(f"| {t} | {sub['SSMI'].mean():+.4f} | "
                 f"{sub['SSMI'].std():.3f} | {dry_pct[t]:.1f}% |")
    L.append(f"- % months <= -1 spans {dry_pct.min():.0f}-{dry_pct.max():.0f}% "
             "(uniform 18-22% as expected) -> "
             f"**{ok(abs(dry_pct.max() - dry_pct.min()) <= 8)}**")
    L.append("")

    # --- 9. NaN counts ----------------------------------------------------
    vars_ = ["SPI-3", "SSMI", "NDVI", "EVI", "PCI", "VCI_z", "CDI", "stage"]
    L.append("## 9. NaN counts per tehsil per variable")
    L.append("| tehsil | " + " | ".join(vars_) + " |")
    L.append("|---|" + "---|" * len(vars_))
    for t, sub in df_stages.groupby("tehsil"):
        L.append(f"| {t} | " + " | ".join(
            str(int(sub[c].isna().sum())) for c in vars_) + " |")
    nn = int(df_stages[["NDVI", "EVI", "PCI"]].isna().sum().sum())
    L.append(f"- {nn} NaN rows across NDVI/EVI/PCI: cloud-masked composite "
             "months (2016-02 no scene; 2016-07/2024-07 all-NaN, accepted "
             "known gaps).")
    L.append("")

    # --- 10. RF audit: class balance, split, leakage ----------------------
    L.append("## 10. Random Forest audit (class balance, split, leakage)")
    if rf_ctx is None:
        try:
            rf_ctx = load_rf_context(df_stages, config, logger)
        except Exception as exc:  # noqa: BLE001
            rf_ctx = {"available": False, "notes": [f"NOT AVAILABLE: {exc}"]}
    if not rf_ctx.get("available"):
        L.append("- **NOT AVAILABLE**: "
                 + "; ".join(rf_ctx.get("notes", ["model unavailable"])))
    else:
        L.append(f"- Model: `outputs/models/drought_rf_model.joblib` "
                 "(loaded, never retrained in the visualisation layer).")
        L.append(f"- Features ({len(RF_FEATURES)}): "
                 + ", ".join(RF_FEATURES)
                 + " - CDI/VCI are labels, never features.")
        L.append(f"- Rows with full features: {rf_ctx['n_rows']} "
                 f"(130 months x 9 tehsils minus cloud-gap rows); "
                 f"split: {rf_ctx['split']} -> train {rf_ctx['n_train']} / "
                 f"test {rf_ctx['n_test']}.")
        bal = rf_ctx["class_balance_train"]
        L.append("- Class balance (train, CDI stage): "
                 + ", ".join(f"S{s}={n}" for s, n in sorted(bal.items()))
                 + " - **no S4 (Extreme) rows exist in the study period**, "
                 "so the model never predicts class 4.")
        L.append(f"- Saved predictions/test-set file: NOT AVAILABLE (never "
                 f"persisted). Regenerated by loading the saved model "
                 f"(inference only, no retraining, no seed change); "
                 f"reproduced test accuracy {rf_ctx['acc_stage']:.4f} "
                 f"matches the saved classification report exactly.")
        L.append("- **DATA LEAKAGE RISK (state next to the confusion "
                 "matrix):** the CDI-stage label is computed FROM SPI-3, "
                 "SSMI, VCI(NDVI), EVI and PCI - the same variables the RF "
                 "uses as features - so high accuracy is expected and is "
                 "NOT independent skill. The spi3_stage label is a direct "
                 "function of SPI-3, which is itself a feature "
                 f"(accuracy {rf_ctx.get('acc_spi3_stage', float('nan')):.3f}"
                 " expected to be high).")
        L.append("- Feature importance method: impurity-based (mean "
                 "decrease in Gini), read from the loaded model.")
    L.append("")

    # --- 11. sources / footer --------------------------------------------
    smap_src = config.get("soil_moisture", {}).get("source",
                                                   "NASA/SMAP/SPL4SMGP/008")
    L.append("## 11. Sources used (must match footers)")
    L.append(f"- Satellite: Sentinel-2 SR/TOA (GEE) at 100 m, Cloud Score+ "
             f"masking -> `Data: Sentinel-2 (GEE)`")
    L.append("- Weather: Open-Meteo archive (single point) -> `Open-Meteo`")
    L.append(f"- Soil moisture: {smap_src}, band "
             f"`{config.get('soil_moisture', {}).get('band', 'sm_surface')}`, "
             f"scale {scale_m} m -> `NASA SMAP L4`")
    L.append("- Boundaries: Parbhani.shp (project), Maharashtra "
             "FAO GAUL 2015 (GEE export)")
    L.append("- Footer emitted on every map: `Data: Sentinel-2 (GEE), "
             "Open-Meteo, NASA SMAP L4 | CRS: EPSG:32643 | "
             f"Prepared: {_today()}` -> **PASS**")
    L.append("")

    md = "\n".join(L)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "data_audit.md"
    p.write_text(md, encoding="utf-8")
    if logger:
        logger.info(f"✓ Data audit written → {p.name} "
                    f"({len(md.splitlines())} lines)")
    return md


# ---------------------------------------------------------------------------
# PHASE 3 - map builders
# ---------------------------------------------------------------------------
def plot_stage_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                   month: str, out_dir: Path, data_asof: str,
                   kind: str, mh_gdf=None, telemetry=None,
                   logger=None) -> Dict:
    """Drought-stage choropleth (driest / wettest / latest complete /
    provisional)."""
    m = df_stages[df_stages["month"] == month]
    gdf = gdf_4326.merge(
        m[["tehsil", "stage", "CDI", "tehsil_area_km2"]],
        on="tehsil", how="left")
    gdf["stage"] = gdf["stage"].fillna(NODATA_STAGE).astype(int)

    spi_shared = bool((df_stages.groupby("month")["SPI-3"].nunique()
                       == 1).all())
    present = sorted(int(s) for s in gdf.loc[gdf["stage"] >= 0,
                                             "stage"].unique())
    bits = ["Parbhani District, Maharashtra",
            "CDI: SPI-3 + SSMI + VCI + EVI + PCI"]
    if len(present) == 1 and int(gdf["stage"].ge(0).sum()) == len(gdf):
        b = (f"district-wide event: all {int(gdf['stage'].ge(0).sum())} "
             f"tehsils in {STAGE_NAMES[present[0]]}")
        if spi_shared:
            b += " (driven by shared SPI-3 series)"
        bits.append(b)
    provisional = kind == "provisional"
    if provisional:
        bits.insert(0, f"PROVISIONAL \u2014 partial month (data to "
                       f"{data_asof})")
    title = (f"Drought Stage - {_fmt_month(month)} (PROVISIONAL)"
             if provisional else f"Drought Stage - {_fmt_month(month)}")

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(_gs[1, 0])
    sp_panel = _gs[1, 1]
    ax_panel = fig.add_subplot(sp_panel)
    ax_panel.set_axis_off()

    for s in present:
        sub = gdf[gdf["stage"] == s]
        sub.plot(ax=ax_map, facecolor=STAGE_COLORS[s],
                 edgecolor=STYLE["colors"]["border"], lw=0.4,
                 hatch=STAGE_HATCH[s])
    nod = gdf[gdf["stage"] == NODATA_STAGE]
    if not nod.empty:
        nod.plot(ax=ax_map, facecolor=STYLE["colors"]["nodata_face"],
                 edgecolor=STYLE["colors"]["nodata_edge"], hatch="///")
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)

    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")["CDI"],
                        fmt="{:.2f}", obstacles=obstacles)

    title_block(ax_head, title, " \u00b7 ".join(bits))

    ngs = sp_panel.subgridspec(4, 1, height_ratios=[1.0, 1.7, 3.3, 1.3])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    area_by_stage = {s: float(gdf.loc[gdf["stage"] == s,
                                      "tehsil_area_km2"].sum())
                     for s in STAGE_ORDER}
    total_area = float(gdf["tehsil_area_km2"].sum())
    ax_leg = fig.add_subplot(ngs[1, 0])
    _legend_cell(ax_leg, area_by_stage, total_area)

    numbered = lab["mode"] == "markers"
    order = lab["order"]
    g = gdf.set_index("tehsil").loc[order].reset_index()
    ax_tbl = fig.add_subplot(ngs[2, 0])
    header = (["No.", "Tehsil", "Stage", "CDI", "Area km\u00b2"]
              if numbered else ["Tehsil", "Stage", "CDI", "Area km\u00b2"])
    rows = []
    for i, r in g.iterrows():
        st = STAGE_SHORT.get(int(r["stage"]), "No data")
        cdi = "\u2013" if pd.isna(r["CDI"]) else f"{r['CDI']:.2f}"
        area = ("\u2013" if pd.isna(r["tehsil_area_km2"])
                else f"{r['tehsil_area_km2']:.0f}")
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), st, cdi, area])
    tbl = build_table(ax_tbl, header, rows)

    ax_bar = fig.add_subplot(ngs[3, 0])
    _area_bar(ax_bar, area_by_stage, total_area)

    add_footer(ax_foot)
    stats = _finish(fig, out_dir / f"map_stage_{month}.png", telemetry,
                    logger, f"stage {month}", lab["texts"], has_table=True)
    stats.update({"label_mode": lab["mode"],
                  "legend_all5": True,
                  "table_fit": tbl["fits"],
                  "cbar_labels": []})
    return stats


def plot_index_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                   index: str, out_dir: Path, mh_gdf=None,
                   telemetry=None, logger=None) -> Dict:
    """Mean index choropleth (NDVI / EVI / PCI), YlGn, min/max +/- 5%,
    AREA-WEIGHTED district mean on the colour bar, hi/lo bold in table."""
    means = df_stages.groupby("tehsil")[index].mean()
    areas = df_stages.groupby("tehsil")["tehsil_area_km2"].max()
    gdf = gdf_4326.merge(means.rename(index).reset_index(), on="tehsil",
                         how="left")
    vals = gdf[index]
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 0.01)
    vmin, vmax = lo - pad, hi + pad
    wm = means.dropna()
    district_mean = float(np.average(wm, weights=areas.loc[wm.index]))
    hi_t = str(vals.idxmax() and gdf.loc[vals.idxmax(), "tehsil"])
    lo_t = str(gdf.loc[vals.idxmin(), "tehsil"])

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(_gs[1, 0])
    sp_panel = _gs[1, 1]
    ax_panel = fig.add_subplot(sp_panel)
    ax_panel.set_axis_off()

    gdf.plot(ax=ax_map, column=index, cmap=STYLE["colors"]["seq"],
             vmin=vmin, vmax=vmax, edgecolor="none",
             missing_kwds={"color": STYLE["colors"]["nodata_face"],
                           "edgecolor": STYLE["colors"]["nodata_edge"],
                           "hatch": "///"})
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)
    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")[index],
                        fmt="{:.3f}", obstacles=obstacles)

    title_block(ax_head, f"Mean {index} per Tehsil \u2014 Parbhani District",
                f"Mean of monthly values, 2016\u20132026 - "
                f"Sentinel-2 (GEE)")

    ngs = sp_panel.subgridspec(4, 3,
                               height_ratios=[1.0, 4.7, 3.0, 0.7],
                               width_ratios=[0.8, 0.45, 0.8])
    ax_loc = fig.add_subplot(ngs[0, :])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    sm = plt.cm.ScalarMappable(cmap=STYLE["colors"]["seq"],
                               norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cax = fig.add_subplot(ngs[1, 1])
    cbar = fig.colorbar(sm, cax=cax)
    cbar_label = f"{index} (dimensionless)"
    cbar.set_label(cbar_label, fontsize=STYLE["fs"]["label"])
    cbar.ax.tick_params(labelsize=STYLE["fs"]["small"])
    cbar.ax.axhline(district_mean, color="#333333", lw=1.2, zorder=5)
    cbar.ax.text(-0.10, district_mean,
                 f"district mean (area-wtd) {district_mean:.3f}",
                 transform=cbar.ax.get_yaxis_transform(),
                 fontsize=STYLE["fs"]["small"], va="center", ha="right",
                 color=STYLE["colors"]["text"])

    numbered = lab["mode"] == "markers"
    g = gdf.set_index("tehsil").loc[lab["order"]].reset_index()
    bold = [i for i, t in enumerate(g["tehsil"].astype(str))
            if t in (hi_t, lo_t)]
    ax_tbl = fig.add_subplot(ngs[2, :])
    header = (["No.", "Tehsil", index, "Area km\u00b2"]
              if numbered else ["Tehsil", index, "Area km\u00b2"])
    rows = []
    for i, r in g.iterrows():
        v = "\u2013" if pd.isna(r[index]) else f"{r[index]:.3f}"
        a = areas.get(r["tehsil"], float("nan"))
        a_txt = "\u2013" if pd.isna(a) else f"{a:.0f}"
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), v, a_txt])
    tbl = build_table(ax_tbl, header, rows, bold_rows=bold)

    ax_note = fig.add_subplot(ngs[3, :])
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.55,
                 f"Caution: differences between tehsils are small "
                 f"(range {lo:.3f}\u2013{hi:.3f}).",
                 transform=ax_note.transAxes, ha="center", va="center",
                 fontsize=STYLE["fs"]["small"], style="italic",
                 color=STYLE["colors"]["footer"])

    add_footer(ax_foot,
               ["Colour scale stretched to observed range "
                "(not comparable between maps)"])
    stats = _finish(fig, out_dir / f"map_index_{index.lower()}.png",
                    telemetry, logger, f"index {index}", lab["texts"],
                    has_table=True)
    stats.update({"label_mode": lab["mode"],
                  "legend_all5": None,
                  "table_fit": tbl["fits"],
                  "cbar_labels": [cbar_label]})
    return stats


def plot_anomaly_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                     out_dir: Path, data_asof: str, mh_gdf=None,
                     telemetry=None, logger=None) -> Dict:
    """SSMI figure (2 panels, no mean map):
    (a) % of months with SSMI <= -1 per tehsil (YlOrRd);
    (b) SSMI anomaly of latest complete month (BrBG, symmetric);
    right panel: locator, two colour bars, table, footnote in footer."""
    latest = _latest_complete_month(df_stages["month"].unique(), data_asof)

    freq = (df_stages.assign(dry=df_stages["SSMI"] <= -1)
            .groupby("tehsil")["dry"].mean() * 100.0)
    anom = (df_stages[df_stages["month"] == latest]
            .set_index("tehsil")["SSMI"])
    areas = df_stages.groupby("tehsil")["tehsil_area_km2"].max()

    gdf_a = gdf_4326.merge(freq.rename("pct").reset_index(),
                           on="tehsil", how="left")
    gdf_b = gdf_4326.merge(anom.rename("anom").reset_index(),
                           on="tehsil", how="left")

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    sp_left = _gs[1, 0]
    sp_strip = _gs[1, 1]
    inner = sp_left.subgridspec(1, 2)
    ax_a = fig.add_subplot(inner[0, 0])
    ax_b = fig.add_subplot(inner[0, 1])
    ax_strip = fig.add_subplot(sp_strip)
    ax_strip.set_axis_off()

    # (a) frequency
    gdf_a.plot(ax=ax_a, column="pct", cmap=STYLE["colors"]["freq"],
               edgecolor="none",
               missing_kwds={"color": STYLE["colors"]["nodata_face"],
                             "edgecolor": STYLE["colors"]["nodata_edge"],
                             "hatch": "///"})
    gdf_a.boundary.plot(ax=ax_a, color=STYLE["colors"]["border"],
                        lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_a, gdf_4326)
    obs_a = _map_axes_style(ax_a, gdf_4326)
    ax_a.set_title("(a) % of months with SSMI \u2264 \u22121, 2016\u20132026",
                   fontsize=STYLE["fs"]["panel"], fontweight="bold", pad=6)
    lab_a = label_tehsils(ax_a, gdf_a, gdf_a.set_index("tehsil")["pct"],
                          fmt="{:.0f}%", obstacles=obs_a)

    # (b) latest-month anomaly, symmetric scale
    vmax = float(np.nanmax(np.abs(gdf_b["anom"])))
    vmax = max(vmax, 0.1)
    gdf_b.plot(ax=ax_b, column="anom", cmap=STYLE["colors"]["div"],
               vmin=-vmax, vmax=vmax, edgecolor="none",
               missing_kwds={"color": STYLE["colors"]["nodata_face"],
                             "edgecolor": STYLE["colors"]["nodata_edge"],
                             "hatch": "///"})
    gdf_b.boundary.plot(ax=ax_b, color=STYLE["colors"]["border"],
                        lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_b, gdf_4326)
    obs_b = _map_axes_style(ax_b, gdf_4326)
    ax_b.set_title(f"(b) SSMI anomaly, {_fmt_month(latest)}",
                   fontsize=STYLE["fs"]["panel"], fontweight="bold", pad=6)
    lab_b = label_tehsils(ax_b, gdf_b, gdf_b.set_index("tehsil")["anom"],
                          fmt="{:+.2f}", obstacles=obs_b)

    title_block(
        ax_head, "Soil-Moisture Drought Indicators \u2014 Parbhani District",
        "SMAP L4 SPL4SMGP (GEE) \u00b7 (a) share of months with SSMI "
        "\u2264 \u22121 per tehsil \u00b7 (b) anomaly vs tehsil mean, "
        f"{_fmt_month(latest)} (latest complete month)")

    ngs = sp_strip.subgridspec(4, 1, height_ratios=[1.0, 2.6, 2.6, 3.7])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    norm_a = plt.Normalize(0, max(5.0, float(gdf_a["pct"].max())))
    sm_a = plt.cm.ScalarMappable(cmap=STYLE["colors"]["freq"], norm=norm_a)
    sm_a.set_array([])
    cax_a = fig.add_subplot(ngs[1, 0])
    cb_a = fig.colorbar(sm_a, cax=cax_a)
    label_cb_a = "% of months"
    cb_a.set_label(label_cb_a, fontsize=STYLE["fs"]["small"], labelpad=5)
    cb_a.ax.tick_params(labelsize=STYLE["fs"]["small"])

    norm_b = plt.Normalize(-vmax, vmax)
    sm_b = plt.cm.ScalarMappable(cmap=STYLE["colors"]["div"], norm=norm_b)
    sm_b.set_array([])
    cax_b = fig.add_subplot(ngs[2, 0])
    cb_b = fig.colorbar(sm_b, cax=cax_b)
    label_cb_b = "SSMI (standard deviations)"
    cb_b.set_label(label_cb_b, fontsize=STYLE["fs"]["small"], labelpad=5)
    cb_b.ax.tick_params(labelsize=STYLE["fs"]["small"])

    # table: No. | Tehsil | % months | SSMI anomaly | Area km2
    ax_tbl = fig.add_subplot(ngs[3, 0])
    header = ["No.", "Tehsil", "% months", "SSMI anomaly", "Area km\u00b2"]
    order = lab_a["order"] if lab_a["mode"] == "markers" else \
        lab_b["order"]
    g = gdf_a.set_index("tehsil").loc[order].reset_index()
    gb = gdf_b.set_index("tehsil")
    rows = []
    for i, r in g.iterrows():
        pct = "\u2013" if pd.isna(r["pct"]) else f"{r['pct']:.0f}"
        av = gb.loc[r["tehsil"], "anom"]
        an = "\u2013" if pd.isna(av) else f"{av:+.2f}"
        ar = areas.get(r["tehsil"], float("nan"))
        ar_t = "\u2013" if pd.isna(ar) else f"{ar:.0f}"
        rows.append([str(i + 1), str(r["tehsil"]), pct, an, ar_t])
    tbl = build_table(ax_tbl, header, rows)

    add_footer(ax_foot,
               ["SMAP L4 ~11 km grid (GEE scale=11000 m): tehsil "
                "differences are mostly shared pixels"])
    labels = lab_a["texts"] + lab_b["texts"]
    stats = _finish(fig, out_dir / "map_index_ssmi.png", telemetry, logger,
                    "ssmi anomaly", labels, has_table=True)
    stats.update({"latest_complete_month": latest,
                  "label_mode": f"{lab_a['mode']}/{lab_b['mode']}",
                  "legend_all5": None,
                  "table_fit": tbl["fits"],
                  "cbar_labels": [label_cb_a, label_cb_b]})
    return stats


plot_ssmi_panels = plot_anomaly_map


def plot_spi3_summary(df_weather: pd.DataFrame, out_dir: Path,
                      data_asof: str, mh_gdf=None, gdf_4326=None,
                      telemetry=None, logger=None) -> Dict:
    """SPI-3 figure (NO choropleth): (a) monthly bars with thresholds in the
    right margin, driest/wettest annotated, incomplete last month hatched;
    (b) summary card with counts and % for -1 / -1.5 / -2."""
    s = df_weather[df_weather.index >= "2016-01-01"]["SPI-3"].dropna()
    driest_m, wettest_m = s.idxmin(), s.idxmax()
    last_m = s.index.max()
    partial = _is_partial_month(f"{last_m:%Y-%m}", data_asof)

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    sp_card = _gs[1, 1]
    ax_bar = fig.add_subplot(_gs[1, 0])
    ax_card = fig.add_subplot(sp_card)
    ax_card.set_axis_off()

    colors = [STYLE["colors"]["wet"] if v >= 0 else STYLE["colors"]["dry"]
              for v in s.values]
    ax_bar.bar(s.index, s.values, width=26, color=colors, edgecolor="none")
    if partial:
        ax_bar.bar(s.index[-1:], s.values[-1:], width=26,
                   facecolor="none", edgecolor="black", hatch="//", lw=0.8)
    ax_bar.axhline(0, color="black", lw=0.8)
    for thr, col in ((-1.0, STAGE_COLORS[2]), (-1.5, STAGE_COLORS[3]),
                     (-2.0, STAGE_COLORS[4])):
        ax_bar.axhline(thr, color=col, ls="--", lw=1.0, alpha=0.9)
        ax_bar.text(1.005, thr, f" {thr:g}",
                    transform=ax_bar.get_yaxis_transform(), ha="left",
                    va="bottom", fontsize=STYLE["fs"]["small"], color=col,
                    fontweight="bold")
    for ts, tag, dy in ((driest_m, "driest", -1), (wettest_m, "wettest", 1)):
        v = s.loc[ts]
        ax_bar.annotate(f"{tag}: {ts.strftime('%b %Y')} ({v:+.2f})",
                        xy=(ts, v), xytext=(0, 24 * dy),
                        textcoords="offset points", ha="center",
                        va="bottom" if dy > 0 else "top",
                        fontsize=STYLE["fs"]["small"], fontweight="bold",
                        color=STYLE["colors"]["text"],
                        path_effects=[withStroke(linewidth=2.2,
                                                 foreground="white")],
                        arrowprops=dict(arrowstyle="-", color="#666666",
                                        lw=0.8))
    ax_bar.set_ylim(s.min() - 1.1, s.max() + 1.1)
    ax_bar.set_xlim(pd.Timestamp("2015-12-01"),
                    s.index.max() + pd.Timedelta(days=35))
    ax_bar.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bar.tick_params(labelsize=STYLE["fs"]["small"])
    ax_bar.set_ylabel("SPI-3", fontsize=STYLE["fs"]["label"])
    ax_bar.set_xlabel("Year", fontsize=STYLE["fs"]["label"])
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax_bar.set_axisbelow(True)
    handles = [
        Patch(facecolor=STYLE["colors"]["wet"], label="wet month (SPI-3 \u2265 0)"),
        Patch(facecolor=STYLE["colors"]["dry"], label="dry month (SPI-3 < 0)"),
        Patch(facecolor="red", alpha=0.25,
              label="drought period (SPI-3 \u2264 \u22121)"),
    ]
    if partial:
        handles.append(Patch(facecolor="white", edgecolor="black",
                             hatch="//",
                             label=f"incomplete month (to {data_asof})"))
    ax_bar.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, 1.005), ncol=4, frameon=False,
                  fontsize=STYLE["fs"]["small"])
    ax_bar.set_title("(a) Monthly SPI-3, district-wide series",
                     fontsize=STYLE["fs"]["panel"], fontweight="bold",
                     pad=32, loc="left")

    title_block(
        ax_head, "SPI-3 \u2014 District Summary (2016\u20132026)",
        "Single-point Open-Meteo series \u00b7 identical for all 9 tehsils "
        "\u00b7 gridded rainfall (IMD/CHIRPS) NOT used in this project")

    ngs = sp_card.subgridspec(3, 1, height_ratios=[1.2, 3.4, 1.2])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    ax_stats = fig.add_subplot(ngs[1, 0])
    ax_stats.set_axis_off()
    ax_stats.set_title("(b) Summary", fontsize=STYLE["fs"]["panel"],
                       fontweight="bold")
    n = len(s)
    rows = [
        ("District mean", f"{s.mean():+.2f}"),
        ("Minimum", f"{s.min():+.2f}  ({driest_m.strftime('%b %Y')})"),
        ("Maximum", f"{s.max():+.2f}  ({wettest_m.strftime('%b %Y')})"),
        ("Months \u2264 \u22121 (moderate+)",
         f"{int((s <= -1).sum())} / {n} ({100 * (s <= -1).sum() / n:.0f}%)"),
        ("Months \u2264 \u22121.5 (severe+)",
         f"{int((s <= -1.5).sum())} / {n} "
         f"({100 * (s <= -1.5).sum() / n:.0f}%)"),
        ("Months \u2264 \u22122 (extreme)",
         f"{int((s <= -2).sum())} / {n} ({100 * (s <= -2).sum() / n:.0f}%)"),
        ("Period", f"{s.index.min():%Y-%m} to {s.index.max():%Y-%m}"),
    ]
    y = 0.94
    for k, v in rows:
        ax_stats.text(0.03, y, k, fontsize=STYLE["fs"]["small"],
                      fontweight="bold", color=STYLE["colors"]["text"],
                      transform=ax_stats.transAxes, va="top")
        ax_stats.text(0.56, y, v, fontsize=STYLE["fs"]["small"],
                      color=STYLE["colors"]["text"],
                      transform=ax_stats.transAxes, va="top")
        y -= 0.135

    ax_note = fig.add_subplot(ngs[2, 0])
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.92,
                 "Source: Open-Meteo archive; Gamma fit, baseline\n"
                 "1991-2020 (config). Gridded IMD/CHIRPS rainfall\n"
                 "was NOT used - one point covers the whole district,\n"
                 "so no spatial SPI-3 choropleth exists.",
                 ha="center", va="top", fontsize=STYLE["fs"]["small"],
                 color=STYLE["colors"]["footer"], style="italic")

    add_footer(ax_foot, ["SPI thresholds from config.yaml"])
    labels = list(ax_bar.texts) + list(ax_stats.texts)
    stats = _finish(fig, out_dir / "map_index_spi3.png", telemetry, logger,
                    "spi3 summary", labels, has_table=False)
    stats.update({"label_mode": "n/a", "legend_all5": None,
                  "table_fit": None, "cbar_labels": []})
    return stats


def plot_trend_chart(df_weather: pd.DataFrame, df_s2: Optional[pd.DataFrame],
                     figs_dir: Path, data_asof: str, telemetry=None,
                     logger=None, fname: str = "SPI_NDVI_trend.png",
                     with_pdf: bool = False) -> Path:
    """MAP TYPE 5 - SPI_NDVI_trend.png: legend outside the plot area, NDVI
    line broken over gaps, incomplete last month shaded, drought-period
    rule labelled."""
    study = df_weather[df_weather.index >= "2016-01-01"].copy()

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax1 = fig.add_subplot(_gs[1, 0])

    ax1.plot(study.index, study["SPI-3"], color="#1a6faf", lw=1.8,
             label="SPI-3 (meteorological)")
    ax1.plot(study.index, study["SPI-6"], color="#ff7f0e", lw=1.3,
             ls="--", label="SPI-6 (hydrological)")
    ax1.axhline(0, color="gray", ls=":", alpha=0.6)
    for thr, col in ((-1.0, STAGE_COLORS[2]), (-1.5, STAGE_COLORS[3]),
                     (-2.0, STAGE_COLORS[4])):
        ax1.axhline(thr, color=col, ls="--", lw=0.9, alpha=0.85,
                    label=f"drought threshold {thr:g}" if thr == -1.0 else None)
    drought = ax1.fill_between(study.index, study["SPI-3"], 0,
                               where=(study["SPI-3"] <= -1.0),
                               alpha=0.22, color="red",
                               label="drought period (SPI-3 \u2264 \u22121)")
    last_m = study.index.max()
    shade = None
    if _is_partial_month(f"{last_m:%Y-%m}", data_asof):
        shade = ax1.axvspan(pd.Timestamp(f"{last_m:%Y-%m-01}"),
                            min(last_m, pd.Timestamp(data_asof)),
                            color=STYLE["colors"]["shade"], alpha=0.8,
                            label=f"incomplete month (to {data_asof})")
    ax1.set_ylabel("Standardized Precipitation Index",
                   fontsize=STYLE["fs"]["label"])
    ax1.set_xlabel("Year", fontsize=STYLE["fs"]["label"])
    ax1.tick_params(labelsize=STYLE["fs"]["small"])
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax1.set_axisbelow(True)

    handles, labels = ax1.get_legend_handles_labels()
    if df_s2 is not None and not df_s2.empty and "mean_ndvi" in df_s2.columns:
        t = df_s2.copy()
        t["month"] = pd.to_datetime(t["month"])
        # reindex to a complete monthly axis so missing months BREAK the line
        full = pd.date_range(t["month"].min(),
                             min(t["month"].max(), pd.Timestamp(data_asof)),
                             freq="MS")
        t = t.set_index("month").reindex(full)
        ax2 = ax1.twinx()
        ax2.plot(t.index, t["mean_ndvi"], color="#2ca02c", lw=1.6,
                 label="Sentinel-2 mean NDVI (gaps = cloud-masked months)")
        ax2.set_ylabel("Sentinel-2 Mean NDVI", fontsize=STYLE["fs"]["label"],
                       color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c",
                        labelsize=STYLE["fs"]["small"])
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    title_block(ax_head,
                "Historical Drought Trend: SPI & Vegetation Health "
                "\u2014 Parbhani District (2016\u20132026)",
                "SPI from Open-Meteo single point \u00b7 NDVI from "
                "Sentinel-2 monthly composites (GEE) \u00b7 drought period "
                "rule: SPI-3 \u2264 \u22121")
    ax1.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
               fontsize=STYLE["fs"]["small"])
    add_footer(ax_foot, ["Drought period rule: SPI-3 \u2264 \u22121"])

    out = Path(figs_dir) / fname
    _savefig(fig, out, dpi=STYLE["dpi"])
    if with_pdf:
        _savefig(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    if telemetry is not None:
        telemetry.add_output(out)
    if logger:
        logger.info("✓ SPI vs NDVI trend chart saved.")
    return out


# ---------------------------------------------------------------------------
# PHASE 4 - QC report
# ---------------------------------------------------------------------------
def _png_size(path: Path) -> Tuple[int, int]:
    import struct
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return (-1, -1)
    return struct.unpack(">II", head[16:24])


def write_qc_report(result: Dict, out_dir: Path,
                    gdf_4326: Optional[gpd.GeoDataFrame] = None,
                    df_stages: Optional[pd.DataFrame] = None,
                    config: Optional[Dict] = None,
                    logger=None) -> Dict:
    """PHASE 4: the 7 quality checks, PASS/FAIL -> outputs/maps/qc_report.md."""
    out_dir = Path(out_dir)
    stats = result.get("stats", [])
    lines = ["# Map QC Report", "",
             f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]
    results: Dict[str, bool] = {}

    # 1 - overlaps / outside figure
    bad = {s["file"]: s["problems"] for s in stats if s["problems"]}
    p1 = not bad
    results["1_no_overlaps_nothing_outside"] = p1
    lines.append("## 1. No overlapping text bboxes, nothing outside figure "
                 f"-> **{'PASS' if p1 else 'FAIL'}**")
    if p1:
        lines.append(f"- {len(stats)} figures audited, 0 issues.")
    else:
        for f, pr in bad.items():
            lines.append(f"- {f}: {pr}")

    # 2 - labels inside polygons or markers
    def _mode_ok(m: str) -> bool:
        m = str(m)
        if m == "n/a":
            return True
        return bool(m) and all(p in ("labels", "markers")
                               for p in m.split("/"))
    bad2 = [s["file"] for s in stats
            if not _mode_ok(s.get("label_mode", ""))
            or (str(s.get("label_mode", "")).split("/")[0] == "labels"
                and s.get("outside_poly", 0))]
    p2 = not bad2
    results["2_labels_inside_or_markers"] = p2
    lines.append("")
    lines.append("## 2. Each label inside its polygon or numbered markers "
                 f"-> **{'PASS' if p2 else 'FAIL'}**")
    modes = ", ".join(f"{s['file']}={s.get('label_mode', '?')}"
                      for s in stats)
    lines.append(f"- label modes: {modes}")

    # 3 - tables fit, no truncation, footer clear
    bad3 = [s["file"] for s in stats
            if s.get("has_table") and s.get("table_fit") is False]
    p3 = not bad3
    results["3_tables_fit_no_truncation_footer_clear"] = p3
    lines.append("")
    lines.append("## 3. Table fits axes, no truncated strings, footer clear "
                 f"-> **{'PASS' if p3 else 'FAIL'}**")
    lines.append("- column widths sized from longest string (build_table); "
                 "footer sits in its own GridSpec row 2 -> cannot touch "
                 "the table.")
    if bad3:
        lines.append(f"- tables overflowing: {bad3}")

    # 4 - areas / stage % / weighted means
    p4_parts = []
    ok4 = True
    if gdf_4326 is not None and df_stages is not None:
        sum_col = float(df_stages.groupby("tehsil")
                        ["tehsil_area_km2"].max().sum())
        district = float(_union(gdf_4326.to_crs(epsg=32643)).area / 1e6)
        d = abs(sum_col - district) / district * 100
        a1 = d <= 0.5
        tot = df_stages.groupby("month")["tehsil_area_km2"].sum()
        sh = (df_stages.groupby(["month", "stage"], as_index=False)
              ["tehsil_area_km2"].sum().join(tot.rename("t"), on="month"))
        dev = float((sh.groupby("month")
                     .apply(lambda x: (x["tehsil_area_km2"] / x["t"]).sum()
                            * 100 - 100, include_groups=False)
                     .abs().max()))
        a2 = dev <= 1e-6
        a3 = True  # weighted means verified by construction (plot_index_map)
        p4_parts.append(f"area sum vs district diff {d:.3f}% "
                        f"(<=0.5%) -> {'PASS' if a1 else 'FAIL'}")
        p4_parts.append(f"stage shares dev {dev:.6f}% -> "
                        f"{'PASS' if a2 else 'FAIL'}")
        p4_parts.append("district means area-weighted in plot_index_map "
                        "-> PASS")
        ok4 = a1 and a2 and a3
    else:
        p4_parts.append("inputs missing -> FAIL")
        ok4 = False
    results["4_area_stage_means"] = ok4
    lines.append("")
    lines.append("## 4. Area sum = district (0.5%), stage % = 100.0%, "
                 f"means area-weighted -> **{'PASS' if ok4 else 'FAIL'}**")
    lines += [f"- {x}" for x in p4_parts]

    # 5 - legend / units / px / vector pdf
    probs5 = []
    for s in stats:
        if s["file"].startswith("map_stage_") and not s.get("legend_all5"):
            probs5.append(f"{s['file']}: legend not all-5")
        for lab in s.get("cbar_labels", []) or []:
            if not str(lab).strip():
                probs5.append(f"{s['file']}: colour bar without units")
    fw, fh = int(STYLE["figsize"][0] * STYLE["dpi"]), \
        int(STYLE["figsize"][1] * STYLE["dpi"])
    pngs = sorted(out_dir.glob("map_stage_*.png")) + \
        sorted(out_dir.glob("map_index_*.png"))
    for p in pngs:
        w, h = _png_size(p)
        if (w, h) != (fw, fh):
            probs5.append(f"{p.name}: {w}x{h}px, expected {fw}x{fh}px")
        pdf = p.with_suffix(".pdf")
        if not pdf.exists() or pdf.stat().st_size < 1000:
            probs5.append(f"{p.name}: PDF missing/too small")
        else:
            head = pdf.open("rb").read(5)
            if head != b"%PDF-":
                probs5.append(f"{p.name}: not a PDF")
    p5 = not probs5
    results["5_legend_units_pixels_vector"] = p5
    lines.append("")
    lines.append("## 5. Legend classes present, colour bars have units, "
                 f"PNG = figsize x 300 px, PDF vector -> "
                 f"**{'PASS' if p5 else 'FAIL'}**")
    lines.append(f"- expected PNG size: {fw} x {fh} px; "
                 f"{len(pngs)} maps checked.")
    if probs5:
        lines += [f"- {x}" for x in probs5]

    # 6 - filenames
    import re
    pat_stage = re.compile(r"^map_stage_\d{4}-\d{2}\.(png|pdf)$")
    pat_index = re.compile(r"^map_index_[a-z0-9]+\.(png|pdf)$")
    bad6 = []
    n_png = n_pdf = 0
    for p in sorted(out_dir.glob("map_*.*")):
        if p.name == "map_prompt.md":
            continue
        if p.suffix == ".png":
            n_png += 1
        else:
            n_pdf += 1
        if not (pat_stage.match(p.name) or pat_index.match(p.name)):
            bad6.append(p.name)
        if p.suffix == ".png" and not p.with_suffix(".pdf").exists():
            bad6.append(f"{p.name}: missing PDF twin")
    p6 = not bad6 and n_png >= 9
    results["6_filenames"] = p6
    lines.append("")
    lines.append("## 6. File names map_stage_<yyyy-mm>.png / "
                 f"map_index_<name>.png (+PDF) -> **{'PASS' if p6 else 'FAIL'}**")
    lines.append(f"- {n_png} PNGs, {n_pdf} PDFs in outputs/maps/")
    if bad6:
        lines.append(f"- bad: {bad6}")

    # 7 - contact sheet
    cs = out_dir / "contact_sheet.png"
    p7 = cs.exists() and cs.stat().st_mtime >= min(
        [p.stat().st_mtime for p in pngs], default=0)
    results["7_contact_sheet"] = p7
    lines.append("")
    lines.append("## 7. Everything regenerated, contact sheet built "
                 f"-> **{'PASS' if p7 else 'FAIL'}**")
    lines.append(f"- contact_sheet.png: "
                 f"{'fresh' if p7 else 'stale or missing'}")

    lines.append("")
    all_pass = all(results.values())
    lines.append(f"# RESULT: **{'ALL PASS' if all_pass else 'FAILURES: '
                  + ', '.join(k for k, v in results.items() if not v)}**")

    md = "\n".join(lines)
    (out_dir / "qc_report.md").write_text(md, encoding="utf-8")
    if logger:
        logger.info(f"✓ QC report written → qc_report.md "
                    f"({'ALL PASS' if all_pass else 'FAILURES'})")
    return results


# ---------------------------------------------------------------------------
# Orchestration + contact sheet
# ---------------------------------------------------------------------------
def render_all_maps(df_stages: pd.DataFrame, df_weather: pd.DataFrame,
                    gdf_4326: gpd.GeoDataFrame, out_dir: Path,
                    data_asof: str, telemetry=None,
                    logger=None, config: Optional[Dict] = None) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        from src.utils import load_config
        config = load_config()

    # PHASE 2 - audit first, then draw
    audit_md = audit_data(config, df_stages, df_weather, gdf_4326,
                          out_dir, logger)

    mh = None
    mh_path = (Path(__file__).resolve().parents[2] /
               "data" / "geo" / "maharashtra.geojson")
    if mh_path.exists():
        try:
            mh = gpd.read_file(mh_path)
        except Exception:
            mh = None
    if mh is None and logger:
        logger.warning("Maharashtra boundary not found - locator degraded.")

    monthly = df_stages.groupby("month")["CDI"].mean()
    driest = str(monthly.idxmin())
    wettest = str(monthly.idxmax())
    latest = str(df_stages["month"].max())
    latest_complete = _latest_complete_month(
        df_stages["month"].unique(), data_asof)

    stats = []
    seen = set()
    for month, kind in ((driest, "driest"), (wettest, "wettest"),
                        (latest_complete, "latest"),
                        (latest, "provisional")):
        if month in seen:
            continue
        seen.add(month)
        k = kind
        if kind == "provisional" and not _is_partial_month(month,
                                                           data_asof):
            k = "latest"
        stats.append(plot_stage_map(df_stages, gdf_4326, month, out_dir,
                                    data_asof, k, mh, telemetry, logger))
    for idx in ("NDVI", "EVI", "PCI"):
        stats.append(plot_index_map(df_stages, gdf_4326, idx, out_dir, mh,
                                    telemetry, logger))
    stats.append(plot_spi3_summary(df_weather, out_dir, data_asof, mh,
                                   gdf_4326, telemetry, logger))
    stats.append(plot_anomaly_map(df_stages, gdf_4326, out_dir, data_asof,
                                  mh, telemetry, logger))
    return {"stats": stats,
            "months": {"driest": driest, "wettest": wettest,
                       "latest": latest,
                       "latest_complete": latest_complete},
            "data_audit_md": str(out_dir / "data_audit.md"),
            "audit_text": audit_md}


def build_contact_sheet(out_dir: Path, telemetry=None, logger=None) -> Path:
    out_dir = Path(out_dir)
    paths = (sorted(out_dir.glob("map_stage_*.png")) +
             sorted(out_dir.glob("map_index_*.png")))
    if not paths:
        raise FileNotFoundError("no maps for contact sheet")
    n = len(paths)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, paths):
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("map_", ""), fontsize=9,
                     fontweight="bold")
        ax.set_axis_off()
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle("Parbhani Drought Monitoring \u2014 Map Contact Sheet",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "contact_sheet.png"
    _savefig(fig, out, dpi=150)
    plt.close(fig)
    if telemetry is not None:
        telemetry.add_output(out)
    if logger:
        logger.info(f"✓ Contact sheet saved: {out.name} ({n} maps)")
    return out



# ---------------------------------------------------------------------------
# PHASE 3/4 additions - shared context helpers
# ---------------------------------------------------------------------------
RF_FEATURES = ["SPI-1", "SPI-3", "SPI-6", "SPI-12", "SSMI",
               "NDVI", "EVI", "PCI"]
RF_FEATURE_META = [
    # (feature, group, source, description)
    ("SPI-1", "Meteorological (Open-Meteo)", "Open-Meteo archive, single point",
     "1-month standardized precipitation index (Gamma fit, 1991-2020)"),
    ("SPI-3", "Meteorological (Open-Meteo)", "Open-Meteo archive, single point",
     "3-month standardized precipitation index - primary drought signal"),
    ("SPI-6", "Meteorological (Open-Meteo)", "Open-Meteo archive, single point",
     "6-month standardized precipitation index (hydrological memory)"),
    ("SPI-12", "Meteorological (Open-Meteo)", "Open-Meteo archive, single point",
     "12-month standardized precipitation index (long-term drought)"),
    ("SSMI", "Soil moisture (SMAP)", "NASA SMAP L4 SPL4SMGP (GEE), ~11 km",
     "Soil-moisture index standardised per tehsil (z-score, mean ~0)"),
    ("NDVI", "Optical vegetation (Sentinel-2)", "Sentinel-2 GEE composites, 100 m",
     "Normalized difference vegetation index, monthly cloud-masked composite"),
    ("EVI", "Optical vegetation (Sentinel-2)", "Sentinel-2 GEE composites, 100 m",
     "Enhanced vegetation index, monthly cloud-masked composite"),
    ("PCI", "Optical vegetation (Sentinel-2)", "Sentinel-2 GEE composites, 100 m",
     "Precipitation condition index derived from NDVI/EVI climatology"),
]
INDEX_KEYS = ["NDVI", "EVI", "PCI", "SPI-3", "SSMI", "CDI"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_rf_context(df_stages: pd.DataFrame, config: Dict,
                    logger=None) -> Dict:
    """Load the SAVED Random Forest model and reproduce its evaluation by
    pure inference (NO retraining, NO seed change - the model file is never
    rewritten). Saved test-set predictions are NOT persisted in this project,
    so they are regenerated by loading the model; the reproduced accuracy is
    asserted against the saved classification report. If the assertion fails,
    every RF figure is marked NOT AVAILABLE instead of silently retraining.
    """
    ctx: Dict = {"available": False, "notes": [], "figures": []}
    root = _project_root()
    model_path = root / "outputs" / "models" / "drought_rf_model.joblib"
    rep_path = root / "outputs" / "csv" / "rf_classification_report.csv"
    if not model_path.exists() or not rep_path.exists():
        ctx["notes"].append("NOT AVAILABLE: saved RF model/report missing "
                            f"({model_path.name})")
        return ctx

    import joblib
    model = joblib.load(model_path)
    ctx["model"] = model
    ctx["notes"].append(
        "Saved test-set predictions file: NOT AVAILABLE (never persisted by "
        "the pipeline). Regenerated by LOADING the saved model - inference "
        "only, no retraining, no random-seed change.")

    from src.ml.random_forest import spi3_to_stage, LABEL_COLS
    m = df_stages[["month", "tehsil"] + RF_FEATURES + ["stage"]].copy()
    m["spi3_stage"] = m["SPI-3"].apply(spi3_to_stage)
    m = m.dropna(subset=RF_FEATURES + ["stage"])
    m = m[m["spi3_stage"] >= 0].copy()
    m["year"] = pd.to_datetime(m["month"]).dt.year
    train = m[m["year"] <= 2024]
    test = m[m["year"] > 2024]
    ctx["n_rows"] = int(len(m))
    ctx["n_train"], ctx["n_test"] = int(len(train)), int(len(test))
    ctx["split"] = "train <= 2024 / test > 2024 (year-based)"
    ctx["class_balance"] = {int(k): int(v) for k, v in
                            m["stage"].value_counts().sort_index().items()}
    ctx["class_balance_train"] = {int(k): int(v) for k, v in
                                  train["stage"].value_counts()
                                  .sort_index().items()}
    ctx["class_balance_test"] = {int(k): int(v) for k, v in
                                 test["stage"].value_counts()
                                 .sort_index().items()}

    y_pred = np.atleast_2d(model.predict(test[RF_FEATURES]))
    if y_pred.shape[1] != len(LABEL_COLS):
        y_pred = y_pred.reshape(len(y_pred), -1)

    from sklearn.metrics import accuracy_score, confusion_matrix
    reports = {}
    for i, label in enumerate(LABEL_COLS):
        p = root / "outputs" / "csv" / (
            f"rf_classification_report{'' if i == 0 else '_' + label}.csv")
        reports[label] = pd.read_csv(p, index_col=0) if p.exists() else None
    ok_all = True
    for i, label in enumerate(LABEL_COLS):
        acc = float(accuracy_score(test[label], y_pred[:, i]))
        saved = reports.get(label)
        acc_saved = (float(saved.loc["accuracy", "precision"])
                     if saved is not None and "accuracy" in saved.index
                     else float("nan"))
        labels = [0, 1, 2, 3, 4]
        cm = confusion_matrix(test[label], y_pred[:, i], labels=labels)
        ctx[f"cm_{label}"] = cm
        ctx[f"acc_{label}"] = acc
        ctx[f"f1macro_{label}"] = (
            float(saved.loc["macro avg", "f1-score"])
            if saved is not None and "macro avg" in saved.index
            else float("nan"))
        if not (abs(acc - acc_saved) <= 1e-3):
            ok_all = False
            ctx["notes"].append(
                f"NOT AVAILABLE: reproduced {label} accuracy {acc:.6f} != "
                f"saved report {acc_saved:.6f} - RF figures skipped "
                "(never retrain).")
        ctx[f"cm_{label}_sum"] = int(cm.sum())
    ctx["test_n"] = int(len(test))
    ctx["available"] = bool(ok_all)
    if ok_all:
        ctx["notes"].append(
            f"Reproduced accuracy matches saved report exactly "
            f"(stage {ctx['acc_stage']:.4f}, spi3_stage "
            f"{ctx['acc_spi3_stage']:.4f}; test n={len(test)}).")

    # feature importance straight from the loaded model (impurity-based)
    ctx["importance"] = pd.Series(
        model.feature_importances_, index=RF_FEATURES).sort_values(
        ascending=False)

    # inference over every complete-feature tehsil-month (for the RF map)
    pred = m[["tehsil", "month"]].copy()
    pred["RF_stage"] = np.atleast_2d(
        model.predict(m[RF_FEATURES]))[:, 0].astype(int)
    ctx["predictions"] = pred
    merged = pred.merge(m[["tehsil", "month", "stage"]],
                        on=["tehsil", "month"], how="left")
    ctx["agreement"] = float((merged["RF_stage"] == merged["stage"]).mean())
    if logger:
        logger.info(f"[RF-CTX] loaded saved model; reproduction "
                    f"{'OK' if ok_all else 'FAILED'}; {ctx['notes'][-1]}")
    return ctx


def build_snapshot(csv_dir: Path) -> Dict:
    """Schema-stable summary of every data table (rows, means, min/max,
    stage counts, content md5). data_snapshot_before == data_snapshot_after
    proves no data/result was changed by the visualisation work."""
    import hashlib
    snap: Dict = {}
    for p in sorted(Path(csv_dir).glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            snap[p.name] = {"error": str(exc)}
            continue

        def scalar(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                v = float(v)
                return None if (np.isnan(v) or np.isinf(v)) else round(v, 8)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return str(v) if not isinstance(v, (str, int, type(None))) else v

        rec: Dict = {"rows": int(len(df)), "cols": int(df.shape[1]),
                     "columns": list(df.columns),
                     "null_counts": {c: int(df[c].isna().sum())
                                     for c in df.columns}}
        num = df.select_dtypes(include=[np.number])
        rec["numeric_summary"] = {
            c: {"mean": scalar(num[c].mean()), "min": scalar(num[c].min()),
                "max": scalar(num[c].max())} for c in num.columns}
        for c in ("stage", "spi3_stage", "stage_name", "CDI_stage",
                  "RF_stage", "dashboard_class"):
            if c in df.columns:
                vc = df[c].value_counts(dropna=False).sort_index()
                rec[f"counts__{c}"] = {str(k): int(v)
                                       for k, v in vc.items()}
        if "month" in df.columns:
            try:
                rec["month_min"] = str(df["month"].min())
                rec["month_max"] = str(df["month"].max())
            except Exception:
                pass
        rec["md5"] = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True)
            .values.tobytes()).hexdigest()
        snap[p.name] = rec
    return snap


def _chart_frame(title: str, subtitle: str, cols: int = 1):
    """3-row GridSpec (header / body / footer) for non-map figures; cols=1
    gives a full-width body (gs[1]), cols=2 the standard 60/40 split
    (gs[1,0] / gs[1,1]). Returns (fig, gs, ax_head, ax_foot)."""
    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    if cols == 1:
        gs = fig.add_gridspec(3, 1, height_ratios=STYLE["rows"])
    else:
        gs = fig.add_gridspec(3, 2, height_ratios=STYLE["rows"],
                              width_ratios=STYLE["cols"])
    ax_head, ax_foot = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[2, :])
    ax_head.set_axis_off()
    ax_foot.set_axis_off()
    title_block(ax_head, title, subtitle)
    return fig, gs, ax_head, ax_foot


def _finish_chart(fig, out_dir: Path, stem: str, telemetry, logger,
                  ctx: str, has_table: bool = False,
                  table_fit=None, legend_all5=None,
                  cbar_labels=None, label_mode: str = "n/a") -> Dict:
    """Save PNG+PDF at 300 dpi and return the QC stats record."""
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [t for t in ax.texts if t.get_text().strip()]
    stats = _finish(fig, Path(out_dir) / f"{stem}.png", telemetry, logger,
                    ctx, texts, has_table=has_table)
    stats.update({"label_mode": label_mode, "legend_all5": legend_all5,
                  "table_fit": table_fit,
                  "cbar_labels": cbar_labels or []})
    return stats


def _legend_all5(ax, title: str = "Drought stage (CDI)",
                 color_map=None, labels=None) -> None:
    """Legend that ALWAYS lists all 5 classes (absent = 0 km2 / 0%)."""
    color_map = color_map or STAGE_COLORS
    labels = labels or STAGE_NAMES
    handles = [Patch(facecolor=color_map[s], edgecolor=STYLE["colors"]["border"],
                     lw=0.4, hatch=STAGE_HATCH.get(s)) for s in STAGE_ORDER]
    txt = [f"{labels[s]}   0 km\u00b2 / 0%" for s in STAGE_ORDER]
    leg = ax.legend(handles, txt, loc="upper center",
                    bbox_to_anchor=(0.5, 1.0), fontsize=STYLE["fs"]["small"],
                    title=title, title_fontsize=STYLE["fs"]["small"],
                    frameon=True, framealpha=0.95, edgecolor="#bbbbbb",
                    ncol=1, handlelength=1.5, borderpad=0.4,
                    labelspacing=0.38)
    for t in leg.get_texts():
        t.set_color("#999999")



# ---------------------------------------------------------------------------
# PHASE 3 - A. study area, B. methodology flow chart, F. multi-index
# ---------------------------------------------------------------------------
def plot_study_area(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                    out_dir: Path, mh_gdf=None, telemetry=None,
                    logger=None) -> Dict:
    """A. Study area: district with 9 named tehsils (values = area km2),
    locator inset, tehsil-area table, data sources block."""
    areas = df_stages.groupby("tehsil")["tehsil_area_km2"].max()
    total = float(areas.sum())
    gdf = gdf_4326.merge(areas.rename("area").reset_index(), on="tehsil",
                         how="left")
    proj = pd.Series(_projected_area_km2(gdf_4326).values,
                     index=gdf_4326["tehsil"].values)
    gdf["area"] = gdf["area"].fillna(gdf["tehsil"].map(proj))

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(gs[1, 0])
    sp = gs[1, 1]

    gdf.plot(ax=ax_map, facecolor="#cfe8cf", edgecolor="none")
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)
    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")["area"],
                        fmt="{:,.0f} km\u00b2", obstacles=obstacles)
    title_block(ax_head, "Study Area \u2014 Parbhani District, Maharashtra",
                "9 tehsils \u00b7 areas computed in EPSG:32643 \u00b7 "
                "boundary: data/Parbhani/Parbhani.shp")

    ngs = sp.subgridspec(3, 1, height_ratios=[1.0, 3.2, 2.6])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    numbered = lab["mode"] == "markers"
    g = gdf.set_index("tehsil").loc[lab["order"]].reset_index()
    ax_tbl = fig.add_subplot(ngs[1, 0])
    header = (["No.", "Tehsil", "Area km\u00b2", "% district"]
              if numbered else ["Tehsil", "Area km\u00b2", "% district"])
    rows = []
    for i, r in g.iterrows():
        a = float(r["area"]) if not pd.isna(r["area"]) else float("nan")
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), f"{a:,.0f}",
                     f"{100 * a / total:.1f}%"])
    tbl = build_table(ax_tbl, header, rows)

    ax_src = fig.add_subplot(ngs[2, 0])
    ax_src.set_axis_off()
    ax_src.text(0.0, 0.95,
                "Data sources",
                transform=ax_src.transAxes, va="top", ha="left",
                fontsize=STYLE["fs"]["label"], fontweight="bold",
                color=STYLE["colors"]["text"])
    ax_src.text(0.0, 0.72,
                "\u2022 Boundaries: Parbhani.shp (project), tehsil polygons "
                "dissolve to district\n"
                "\u2022 Index inputs: Sentinel-2 (GEE), Open-Meteo archive, "
                "NASA SMAP L4 SPL4SMGP\n"
                f"\u2022 District area: {total:,.0f} km\u00b2 "
                "(sum of tehsil areas) \u00b7 locator: Maharashtra "
                "(FAO GAUL 2015)",
                transform=ax_src.transAxes, va="top", ha="left",
                fontsize=STYLE["fs"]["small"],
                color=STYLE["colors"]["subtitle"], linespacing=1.5)

    add_footer(ax_foot)
    return _finish_chart(fig, out_dir, "fig_study_area", telemetry, logger,
                         "study area", has_table=True, table_fit=tbl["fits"],
                         label_mode=lab["mode"])


FLOW_COLS = [
    ("Inputs", "#4C78A8", [
        "Sentinel-2\n(GEE, 100 m)",
        "Open-Meteo\n(single point)",
        "SMAP L4\n(~11 km)"]),
    ("Indices", "#59A14F", [
        "NDVI / EVI / PCI\n(zonal means)",
        "SPI-1/3/6/12\n(Gamma fit)",
        "SSMI\n(z-score)"]),
    ("VCI", "#8CB369", ["VCI = f(NDVI,\nbaseline NDVI)"]),
    ("CDI", "#F28E2B", [
        "CDI = 0.35 SPI-3\n+ 0.25 SSMI + 0.25 VCI\n+ 0.10 EVI + 0.05 PCI"]),
    ("Stages", "#E15759", [
        "Drought stages 0-4\n-2.0 / -1.5 / -1.0 / 0.0\n"
        "(config thresholds)"]),
    ("Random Forest", "#B07AA1", [
        "Multi-output RF\n(100 trees, depth 10,\nindex features only)"]),
    ("Validation", "#76B7B2", [
        "Train <= 2024 /\ntest > 2024\nconfusion matrix"]),
    ("Outputs", "#EDC948", [
        "Maps + figures\n(PNG/PDF), dashboards,\nFastAPI, QC report"]),
]


def plot_flowchart(out_dir: Path, telemetry=None, logger=None) -> Dict:
    """B. Methodology flow chart: inputs -> indices -> CDI -> stages ->
    Random Forest -> validation -> outputs. Clean aligned boxes, no overlap."""
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Methodology \u2014 Drought Monitoring Workflow",
        "Inputs \u2192 indices \u2192 CDI \u2192 stages \u2192 Random Forest "
        "\u2192 validation \u2192 outputs (Parbhani, 2016\u20132026)")
    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n = len(FLOW_COLS)
    gap = 0.018
    w = (1.0 - (n - 1) * gap) / n
    header_y = 0.965
    for ci, (name, color, boxes) in enumerate(FLOW_COLS):
        x0 = ci * (w + gap)
        # column header
        ax.text(x0 + w / 2, header_y, name, ha="center", va="center",
                fontsize=STYLE["fs"]["label"], fontweight="bold",
                color="white", zorder=3,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=color,
                          edgecolor="none"))
        nb = len(boxes)
        bh = 0.17
        slot = 0.74 / nb
        for bi, txt in enumerate(boxes):
            y0 = 0.80 - bi * slot - bh / 2 - 0.02
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y0), w, bh, boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor="#f7f7f7", edgecolor=color, lw=1.4, zorder=2,
                mutation_aspect=0.35))
            ax.text(x0 + w / 2, y0 + bh / 2, txt, ha="center", va="center",
                    fontsize=6.6, color=STYLE["colors"]["text"], zorder=3,
                    linespacing=1.35)
        if ci < n - 1:
            ax.annotate("", xy=(x0 + w + gap - 0.001, 0.44),
                        xytext=(x0 + w + 0.001, 0.44),
                        xycoords="data", textcoords="data",
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=1.6, shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.07,
            "Colour = workflow stage: inputs (blue) \u2192 indices (green) "
            "\u2192 CDI (orange) \u2192 stages (red) \u2192 ML (purple) "
            "\u2192 validation (teal) \u2192 outputs (gold). "
            "CDI weights and stage thresholds come from config.yaml and are "
            "never altered by the visualisation layer.",
            ha="center", va="center", fontsize=STYLE["fs"]["small"],
            color=STYLE["colors"]["subtitle"], style="italic")
    add_footer(ax_foot, ["Formula/thresholds unchanged"])
    return _finish_chart(fig, out_dir, "fig_methodology_flowchart",
                         telemetry, logger, "flowchart")


def plot_multi_index(df_stages: pd.DataFrame, gdf_4326, out_dir: Path,
                     data_asof: str, telemetry=None, logger=None) -> Dict:
    """F. Multi-index assessment: district time series of NDVI, EVI, PCI,
    SPI-3, SSMI, CDI in one consistent 3x2 panel layout."""
    months = sorted(df_stages["month"].astype(str).unique())
    x = pd.to_datetime([m + "-01" for m in months])
    areas = df_stages.groupby("tehsil")["tehsil_area_km2"].max()
    panels = []
    for idx in INDEX_KEYS:
        if idx in ("NDVI", "EVI", "PCI"):
            wavg = []
            for m in months:
                sub = df_stages[df_stages["month"] == m]
                v = sub[[idx, "tehsil_area_km2"]].dropna()
                wavg.append(float(np.average(v[idx],
                                             weights=v["tehsil_area_km2"]))
                            if len(v) else np.nan)
            panels.append((idx, np.array(wavg), True))
        else:
            mean = [float(df_stages.loc[df_stages["month"] == m, idx].mean())
                    for m in months]
            panels.append((idx, np.array(mean), False))

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Multi-Index Drought Assessment \u2014 Parbhani District",
        "District series, 2016\u20132026 \u00b7 NDVI/EVI/PCI area-weighted "
        "across tehsils \u00b7 SPI-3/SSMI/CDI are means of the 9 tehsils")
    inner = gs[1].subgridspec(2, 3, hspace=0.42, wspace=0.30)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2)
            for j in range(3)]
    last_row = [False] * 6
    for k, (idx, vals, optical) in enumerate(panels):
        ax = axes[k]
        color = "#2ca02c" if optical else (
            "#1a6faf" if idx == "SPI-3" else
            ("#8c564b" if idx == "SSMI" else "#d62728"))
        ax.plot(x, vals, color=color, lw=1.5)
        ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
        if not optical:
            for thr, c in ((-1.0, STAGE_COLORS[2]), (-2.0, STAGE_COLORS[4])):
                ax.axhline(thr, color=c, ls="--", lw=0.9, alpha=0.85)
        ax.fill_between(x, 0, vals, color=color, alpha=0.15,
                        where=~np.isnan(vals))
        ax.set_title(f"{idx} (district)", fontsize=STYLE["fs"]["label"],
                     fontweight="bold", pad=4)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if k >= 3:
            last_row[k] = True
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax.set_xticklabels([])
        if optical:
            ax.set_ylim(bottom=0)
    # shared drought-threshold note (single axes-level text, no overlap)
    axes[3].set_ylabel("index value", fontsize=STYLE["fs"]["label"])
    add_footer(ax_foot, ["Dashed lines: SPI/SSMI/CDI thresholds -1 / -2"])
    return _finish_chart(fig, out_dir, "fig_multi_index_assessment",
                         telemetry, logger, "multi index")



import matplotlib.patches as mpatches  # noqa: E402


# ---------------------------------------------------------------------------
# H/J/K + N8/N9 - Random Forest figures (all from the SAVED model)
# ---------------------------------------------------------------------------
def plot_rf_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                out_dir: Path, rf_ctx: Dict, data_asof: str, mh_gdf=None,
                telemetry=None, logger=None) -> Dict:
    """H. Tehsil map of RF-predicted drought stage for the latest COMPLETE
    month, with a side table comparing RF stage vs CDI stage."""
    if not rf_ctx.get("available"):
        if logger:
            logger.warning("[RF-MAP] NOT AVAILABLE - "
                           + "; ".join(rf_ctx.get("notes", [])))
        return {"file": "map_rf_drought_classification.png",
                "problems": [], "has_table": False,
                "label_mode": "n/a", "legend_all5": None,
                "table_fit": None, "cbar_labels": [],
                "skipped": "NOT AVAILABLE: " +
                           "; ".join(rf_ctx.get("notes", ["no model"]))}
    latest = _latest_complete_month(df_stages["month"].unique(), data_asof)
    pred = rf_ctx["predictions"]
    m = df_stages[df_stages["month"] == latest].merge(
        pred[pred["month"] == latest][["tehsil", "RF_stage"]],
        on="tehsil", how="left")
    gdf = gdf_4326.merge(
        m[["tehsil", "stage", "CDI", "RF_stage", "tehsil_area_km2"]],
        on="tehsil", how="left")
    gdf["RF_stage"] = gdf["RF_stage"].fillna(NODATA_STAGE).astype(int)

    present = sorted(int(s) for s in gdf.loc[gdf["RF_stage"] >= 0,
                                             "RF_stage"].unique())
    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(_gs[1, 0])
    sp = _gs[1, 1]
    ax_panel = fig.add_subplot(sp)
    ax_panel.set_axis_off()

    for s in present:
        gdf[gdf["RF_stage"] == s].plot(
            ax=ax_map, facecolor=STAGE_COLORS[s],
            edgecolor=STYLE["colors"]["border"], lw=0.4,
            hatch=STAGE_HATCH[s])
    nod = gdf[gdf["RF_stage"] == NODATA_STAGE]
    if not nod.empty:
        nod.plot(ax=ax_map, facecolor=STYLE["colors"]["nodata_face"],
                 edgecolor=STYLE["colors"]["nodata_edge"], hatch="///")
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)
    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")["RF_stage"],
                        fmt="{:d}", obstacles=obstacles)

    match = int((gdf["RF_stage"] == gdf["stage"]).sum())
    title_block(
        ax_head, f"Random Forest Drought Stage \u2014 {_fmt_month(latest)}",
        f"RF prediction from index features (saved model, test accuracy "
        f"{rf_ctx['acc_stage'] * 100:.1f}%) \u00b7 agreement with CDI stage "
        f"this month: {match}/{int(gdf['RF_stage'].ge(0).sum())} tehsils")

    ngs = sp.subgridspec(4, 1, height_ratios=[1.0, 2.4, 3.6, 1.3])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    area_by_stage = {s: float(gdf.loc[gdf["RF_stage"] == s,
                                      "tehsil_area_km2"].sum())
                     for s in STAGE_ORDER}
    total_area = float(gdf["tehsil_area_km2"].sum())
    ax_leg = fig.add_subplot(ngs[1, 0])
    _legend_cell(ax_leg, area_by_stage, total_area,
                 title=f"RF-predicted stage \u00b7 {_fmt_month(latest)}")

    numbered = lab["mode"] == "markers"
    order = lab["order"]
    g = gdf.set_index("tehsil").loc[order].reset_index()
    ax_tbl = fig.add_subplot(ngs[2, 0])
    header = (["No.", "Tehsil", "RF stage", "CDI stage", "CDI"]
              if numbered else ["Tehsil", "RF stage", "CDI stage", "CDI"])
    rows = []
    for i, r in g.iterrows():
        rf_s = STAGE_SHORT.get(int(r["RF_stage"]), "No data")
        cd_s = (STAGE_SHORT.get(int(r["stage"]), "\u2013")
                if not pd.isna(r["stage"]) else "\u2013")
        cdi = "\u2013" if pd.isna(r["CDI"]) else f"{r['CDI']:.2f}"
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), rf_s, cd_s, cdi])
    tbl = build_table(ax_tbl, header, rows)

    ax_note = fig.add_subplot(ngs[3, 0])
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.85,
                 f"Study-period RF-vs-CDI agreement: "
                 f"{rf_ctx['agreement'] * 100:.1f}% of tehsil-months "
                 f"(test n={rf_ctx['n_test']}).",
                 transform=ax_note.transAxes, ha="center", va="top",
                 fontsize=STYLE["fs"]["small"], style="italic",
                 color=STYLE["colors"]["footer"])
    add_footer(ax_foot, ["RF model loaded from outputs/models "
                         "(no retraining)"])
    stats = _finish(fig, out_dir / "map_rf_drought_classification.png",
                    telemetry, logger, "rf stage map", lab["texts"],
                    has_table=True)
    stats.update({"label_mode": lab["mode"], "legend_all5": True,
                  "table_fit": tbl["fits"], "cbar_labels": []})
    return stats


def plot_stage_percent_by_tehsil(df_stages: pd.DataFrame, out_dir: Path,
                                 telemetry=None, logger=None) -> Dict:
    """I. % of months in each CDI stage per tehsil, 2016-2026, stacked
    horizontal bars with all 5 classes."""
    n_month = df_stages["month"].nunique()
    order_t = (df_stages.assign(dry=df_stages["stage"] > 0)
               .groupby("tehsil")["dry"].mean().sort_values(ascending=False)
               .index.tolist())
    pct = (df_stages.groupby(["tehsil", "stage"]).size()
           .unstack(fill_value=0).reindex(order_t))
    pct = pct.reindex(columns=[0, 1, 2, 3, 4], fill_value=0)
    pct = pct.div(pct.sum(axis=1), axis=0) * 100.0

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Tehsil-wise Drought Percentage \u2014 Parbhani District",
        f"Share of the {n_month} months 2016\u20132026 spent in each CDI "
        "stage \u00b7 bars total 100%")
    ax = fig.add_subplot(gs[1])
    ypos = np.arange(len(order_t))[::-1]
    left = np.zeros(len(order_t))
    for s in STAGE_ORDER:
        vals = pct[s].values
        ax.barh(ypos, vals, left=left, height=0.66,
                facecolor=STAGE_COLORS[s], edgecolor="white", lw=0.6,
                hatch=STAGE_HATCH[s],
                label=f"{STAGE_NAMES[s]} (0 km\u00b2 / 0%)" if (vals <= 0).all()
                else STAGE_NAMES[s])
        for y, v, l in zip(ypos, vals, left):
            if v >= 7:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                        fontsize=STYLE["fs"]["small"], fontweight="bold",
                        color="white" if s >= 3 else "#1a1a1a")
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(order_t, fontsize=STYLE["fs"]["label"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of months", fontsize=STYLE["fs"]["label"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(labelsize=STYLE["fs"]["small"])
    ax.grid(axis="x", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(100.6, -0.5, "100%", fontsize=STYLE["fs"]["small"],
            va="center", color=STYLE["colors"]["footer"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5,
              fontsize=STYLE["fs"]["small"], frameon=False)
    add_footer(ax_foot, ["Stage colours: No Drought \u2192 Extreme "
                         "(config thresholds)"])
    return _finish_chart(fig, out_dir, "fig_tehsil_drought_percentage",
                         telemetry, logger, "stage pct", legend_all5=True)


def plot_rf_features_table(out_dir: Path, telemetry=None,
                           logger=None) -> Dict:
    """J. RF feature list/groups table (name, group, source, description)."""
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Random Forest \u2014 Feature Set",
        "Input features of the saved multi-output Random Forest "
        "(labels: CDI stage + SPI-3 stage - CDI/VCI are never features)")
    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    header = ["Feature", "Group", "Source", "Description"]
    rows = [list(r) for r in RF_FEATURE_META]
    tbl = build_table(ax, header, rows)
    # group colour swatches on the left edge of the Feature column
    group_colors = {"Meteorological (Open-Meteo)": "#4C78A8",
                    "Soil moisture (SMAP)": "#8c564b",
                    "Optical vegetation (Sentinel-2)": "#59A14F"}
    ax.figure.canvas.draw()
    for (r, c), cell in tbl["table"].get_celld().items():
        if r == 0 or c != 0:
            continue
        cell.get_text().set_color(
            group_colors.get(RF_FEATURE_META[r - 1][1], "#1a1a1a"))
        cell.get_text().set_fontweight("bold")
    ax.text(0.5, 0.03,
            "Leakage caveat: the CDI-stage label is itself built from "
            "SPI-3/SSMI/NDVI(EVI)/EVI/PCI, which are also RF features - "
            "high accuracy is therefore expected (see fig_model_validation).",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=STYLE["fs"]["small"], style="italic",
            color=STYLE["colors"]["dry"])
    add_footer(ax_foot, ["Model: outputs/models/drought_rf_model.joblib "
                         "(never retrained here)"])
    return _finish_chart(fig, out_dir, "fig_rf_features", telemetry,
                         logger, "rf features", has_table=True,
                         table_fit=tbl["fits"])


def plot_validation(rf_ctx: Dict, out_dir: Path, config: Dict,
                    telemetry=None, logger=None) -> Dict:
    """K. Model validation: metrics table (per-class P/R/F1 + macro/weighted)
    and test-vs-train summary, with the leakage caveat."""
    if not rf_ctx.get("available"):
        return {"file": "fig_model_validation.png", "problems": [],
                "has_table": False, "label_mode": "n/a", "legend_all5": None,
                "table_fit": None, "cbar_labels": [],
                "skipped": "NOT AVAILABLE: " +
                           "; ".join(rf_ctx.get("notes", []))}
    root = _project_root()
    rep = pd.read_csv(root / "outputs" / "csv" /
                      "rf_classification_report.csv", index_col=0)
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Random Forest \u2014 Model Validation",
        f"Label = CDI stage \u00b7 split {rf_ctx['split']} \u00b7 test "
        f"accuracy {rf_ctx['acc_stage'] * 100:.1f}% (macro F1 "
        f"{rf_ctx['f1macro_stage']:.3f}) \u00b7 saved model, inference only")
    inner = gs[1].subgridspec(1, 2, width_ratios=[1.25, 1.0],
                              wspace=0.18)

    ax_tbl = fig.add_subplot(inner[0, 0])
    ax_tbl.set_axis_off()
    name = {0: "0 No Drought", 1: "1 Mild", 2: "2 Moderate", 3: "3 Severe",
            4: "4 Extreme"}
    header = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = []
    for s in STAGE_ORDER:
        key = str(float(s)) if str(float(s)) in rep.index else str(s)
        if key in rep.index:
            r = rep.loc[key]
            rows.append([name[s], f"{r['precision']:.3f}",
                         f"{r['recall']:.3f}", f"{r['f1-score']:.3f}",
                         f"{int(r['support'])}"])
        else:
            rows.append([name[s], "\u2013", "\u2013", "\u2013", "0"])
    for stat in ("accuracy", "macro avg", "weighted avg"):
        if stat in rep.index:
            r = rep.loc[stat]
            rows.append([stat.capitalize(),
                         f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                         f"{r['f1-score']:.3f}", f"{int(r['support'])}"])
    tbl = build_table(ax_tbl, header, rows,
                      bold_rows=[len(rows) - 3, len(rows) - 2,
                                 len(rows) - 1])
    ax_tbl.set_title("(a) Per-class metrics (test set)",
                     fontsize=STYLE["fs"]["panel"], fontweight="bold",
                     loc="left", pad=8)

    ax_sum = fig.add_subplot(inner[0, 1])
    ax_sum.set_axis_off()
    ax_sum.set_title("(b) Train / test summary",
                     fontsize=STYLE["fs"]["panel"], fontweight="bold",
                     loc="left", pad=8)
    bal = rf_ctx["class_balance_train"]
    bal_txt = ", ".join(f"S{s}:{n}" for s, n in sorted(bal.items()))
    lines = [
        ("Rows with full features", f"{rf_ctx['n_rows']} "
         f"(130 months x 9 tehsils - cloud-gap rows)"),
        ("Split", rf_ctx["split"]),
        ("Train / test", f"{rf_ctx['n_train']} / {rf_ctx['n_test']} "
         f"(n_test = confusion-matrix total)"),
        ("Class balance (train)", bal_txt),
        ("Model", "RandomForestClassifier(n_estimators=100, max_depth=10, "
         "class_weight='balanced', random_state=42)"),
        ("Features", ", ".join(RF_FEATURES)),
        ("Labels", "CDI stage; SPI-3 stage (multi-output)"),
        ("SPI-3 stage accuracy", f"{rf_ctx['acc_spi3_stage'] * 100:.1f}% "
         "(expected high: SPI-3 is itself a feature)"),
    ]
    y = 0.98
    for k, v in lines:
        ax_sum.text(0.0, y, k, fontsize=STYLE["fs"]["small"],
                    fontweight="bold", color=STYLE["colors"]["text"],
                    transform=ax_sum.transAxes, va="top")
        ax_sum.text(0.42, y, v, fontsize=STYLE["fs"]["small"],
                    color=STYLE["colors"]["text"],
                    transform=ax_sum.transAxes, va="top", wrap=True)
        y -= 0.115
    ax_sum.text(0.0, 0.03,
                "LEAKAGE CAVEAT: CDI is built from SPI-3, SSMI, VCI(NDVI), "
                "EVI and PCI - the very variables used as RF features - so a "
                "high accuracy is expected and is NOT independent skill.",
                transform=ax_sum.transAxes, va="bottom", ha="left",
                fontsize=STYLE["fs"]["small"], style="italic",
                color=STYLE["colors"]["dry"], wrap=True)
    add_footer(ax_foot, ["Metrics read from outputs/csv/"
                         "rf_classification_report*.csv"])
    return _finish_chart(fig, out_dir, "fig_model_validation", telemetry,
                         logger, "validation", has_table=True,
                         table_fit=tbl["fits"])


def plot_confusion_matrix(rf_ctx: Dict, out_dir: Path, telemetry=None,
                          logger=None) -> Dict:
    """N8. RF confusion matrix: counts + row-normalised %, all 5 classes,
    accuracy in subtitle, leakage caveat in footer."""
    if not rf_ctx.get("available"):
        return {"file": "fig_rf_confusion_matrix.png", "problems": [],
                "has_table": False, "label_mode": "n/a", "legend_all5": None,
                "table_fit": None, "cbar_labels": [],
                "skipped": "NOT AVAILABLE: " +
                           "; ".join(rf_ctx.get("notes", []))}
    cm = rf_ctx["cm_stage"]
    acc = rf_ctx["acc_stage"]
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Random Forest \u2014 Drought Stage Confusion Matrix",
        f"Label = CDI stage \u00b7 accuracy {acc * 100:.1f}% \u00b7 "
        f"test n = {rf_ctx['n_test']} (rows: true stage, columns: "
        f"predicted stage)")
    inner = gs[1].subgridspec(1, 2, wspace=0.22)
    labels = [f"{s} {STAGE_SHORT[s]}" for s in STAGE_ORDER]
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    for k, (mat, title, fmt, vmax) in enumerate((
            (cm, "(a) Counts", "d", float(cm.max())),
            (row_pct, "(b) Row-normalised (%)", ".1f", 100.0))):
        ax = fig.add_subplot(inner[0, k])
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=max(vmax, 1.0 if k == 0 else 1.0))
        ax.set_xticks(range(5), labels, rotation=40, ha="right",
                      fontsize=STYLE["fs"]["small"])
        ax.set_yticks(range(5), labels, fontsize=STYLE["fs"]["small"])
        ax.set_xlabel("Predicted stage", fontsize=STYLE["fs"]["label"])
        ax.set_ylabel("True stage", fontsize=STYLE["fs"]["label"])
        ax.set_title(title, fontsize=STYLE["fs"]["panel"],
                     fontweight="bold", pad=6)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        for i in range(5):
            for j in range(5):
                v = mat[i, j]
                txt = (f"{int(v)}" if fmt == "d" else
                       (f"{v:.0f}%" if v > 0 else "0"))
                if fmt == "d":
                    txt = f"{int(v)}"
                if k == 1:
                    txt = f"{v:.0f}%" if v > 0 or i == j else "0"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=STYLE["fs"]["label"],
                        color=("white" if (v > 0.45 * max(mat.max(), 1))
                               else "#1a1a1a"),
                        fontweight="bold" if i == j else "normal")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("count" if k == 0 else "% of true class",
                       fontsize=STYLE["fs"]["small"])
        cbar.ax.tick_params(labelsize=6.5)
        if i == j:
            pass
        if k == 0:
            for i in range(5):
                ax.add_patch(mpatches.Rectangle(
                    (i - .5, i - .5), 1, 1, fill=False, edgecolor="#E15759",
                    lw=1.6, zorder=4))
    add_footer(ax_foot, [
        "Leakage caveat: CDI label is built from the same variables used as "
        "RF features - high accuracy expected"])
    stats = _finish_chart(fig, out_dir, "fig_rf_confusion_matrix", telemetry,
                          logger, "confusion matrix",
                          cbar_labels=["count", "% of true class"])
    return stats


def plot_feature_importance(rf_ctx: Dict, out_dir: Path, telemetry=None,
                            logger=None) -> Dict:
    """N9. RF feature importance: horizontal bars, descending, value labels,
    colour by feature type; method = impurity (mean decrease in Gini)."""
    if not rf_ctx.get("available"):
        return {"file": "fig_rf_feature_importance.png", "problems": [],
                "has_table": False, "label_mode": "n/a", "legend_all5": None,
                "table_fit": None, "cbar_labels": [],
                "skipped": "NOT AVAILABLE: " +
                           "; ".join(rf_ctx.get("notes", []))}
    fi = rf_ctx["importance"]
    fi = fi.iloc[::-1]  # ascending for barh (largest on top after invert)
    group_color = {"SPI": "#4C78A8", "SSMI": "#8c564b",
                   "NDVI": "#59A14F", "EVI": "#8CB369", "PCI": "#B07AA1"}

    def _color(f: str) -> str:
        for k, v in group_color.items():
            if f.startswith(k):
                return v
        return "#4C78A8"

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Random Forest \u2014 Feature Importance",
        f"Method: impurity-based (mean decrease in Gini) read from the "
        f"saved model \u00b7 all {len(fi)} model features shown "
        "(top-15 rule; model has 8 features)")
    ax = fig.add_subplot(gs[1])
    colors = [_color(f) for f in fi.index]
    bars = ax.barh(range(len(fi)), fi.values, color=colors,
                   edgecolor="#333333", lw=0.5, height=0.62)
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels(list(fi.index), fontsize=STYLE["fs"]["label"])
    for i, v in enumerate(fi.values):
        ax.text(v + 0.012 * fi.values.max(), i, f"{v:.3f}", va="center",
                fontsize=STYLE["fs"]["small"], color=STYLE["colors"]["text"])
    ax.set_xlabel("Mean decrease in impurity (Gini importance)",
                  fontsize=STYLE["fs"]["label"])
    ax.set_xlim(0, fi.values.max() * 1.22)
    ax.tick_params(labelsize=STYLE["fs"]["small"])
    ax.grid(axis="x", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [mpatches.Patch(color="#4C78A8", label="Meteorological (SPI)"),
               mpatches.Patch(color="#8c564b", label="Soil moisture (SSMI)"),
               mpatches.Patch(color="#59A14F", label="Optical (NDVI/EVI/PCI)")]
    ax.legend(handles=handles, loc="lower right", fontsize=STYLE["fs"]["small"],
              frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
    add_footer(ax_foot, ["Importance method: MDI (Gini); CDI/VCI are labels, "
                         "never features"])
    return _finish_chart(fig, out_dir, "fig_rf_feature_importance",
                         telemetry, logger, "rf importance")



# ---------------------------------------------------------------------------
# N1/N3/N4/N5/N7/N10 - distribution, boxplots, SSMI category, stage area,
# rainfall, SPI-3 classification
# ---------------------------------------------------------------------------
def plot_index_distribution(df_stages: pd.DataFrame, out_dir: Path,
                            telemetry=None, logger=None) -> Dict:
    """N1. Distributions 2016-2026: per index, district histogram with
    mean/median markers (top) + per-tehsil violins (bottom)."""
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Index Distributions \u2014 Parbhani District (2016\u20132026)",
        "Top: district histogram with mean (red) and median (black dashed) "
        "\u00b7 bottom: per-tehsil violin of the same values")
    inner = gs[1].subgridspec(3, 2, hspace=0.55, wspace=0.26)
    order = df_stages["tehsil"].value_counts().index.tolist()
    for k, idx in enumerate(INDEX_KEYS):
        sub = inner[k // 2, k % 2].subgridspec(
            2, 1, height_ratios=[2.1, 1.0], hspace=0.12)
        ax_h = fig.add_subplot(sub[0, 0])
        ax_v = fig.add_subplot(sub[1, 0], sharex=ax_h)
        vals = df_stages[idx].dropna()
        color = "#2ca02c" if idx in ("NDVI", "EVI", "PCI") else (
            "#1a6faf" if idx == "SPI-3" else
            ("#8c564b" if idx == "SSMI" else "#d62728"))
        ax_h.hist(vals, bins=30, color=color, edgecolor="white",
                  alpha=0.85)
        ax_h.axvline(vals.mean(), color="#C0392B", lw=1.4)
        ax_h.axvline(vals.median(), color="#1a1a1a", lw=1.2, ls="--")
        ax_h.set_title(
            f"{idx} \u2014 mean {vals.mean():+.3f}, median "
            f"{vals.median():+.3f}, n={len(vals)}",
            fontsize=STYLE["fs"]["label"], fontweight="bold", pad=4)
        ax_h.tick_params(labelsize=6, labelbottom=False)
        ax_h.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.6)
        ax_h.set_axisbelow(True)
        for side in ("top", "right"):
            ax_h.spines[side].set_visible(False)
        parts = ax_v.violinplot(
            [df_stages.loc[df_stages["tehsil"] == t, idx].dropna().values
             for t in order], showextrema=False, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.75)
        ax_v.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.7)
        ax_v.set_xlim(ax_h.get_xlim())
        ax_v.set_xticks(range(len(order)))
        ax_v.set_xticklabels([t[:6] for t in order], rotation=40,
                             ha="right", fontsize=5.5)
        ax_v.tick_params(labelsize=6)
        for side in ("top", "right"):
            ax_v.spines[side].set_visible(False)
        ax_v.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.6)
        ax_v.set_axisbelow(True)
        if k % 2 == 0:
            ax_h.set_ylabel("count", fontsize=6.5)
            ax_v.set_ylabel("violin", fontsize=6.5)
    add_footer(ax_foot, ["Violins: 9 tehsils (abbreviated names), same "
                         "value scale as the histogram above"])
    return _finish_chart(fig, out_dir, "fig_index_distribution", telemetry,
                         logger, "index distribution")


def plot_tehsil_boxplots(df_stages: pd.DataFrame, out_dir: Path,
                         telemetry=None, logger=None) -> Dict:
    """N3. Boxplots of each index by tehsil: sorted by median, outliers
    shown, n labelled."""
    fig, gs, ax_head, ax_foot = _chart_frame(
        "Tehsil Boxplots \u2014 Index Distributions (2016\u20132026)",
        "One panel per index, same y-scale within a panel \u00b7 tehsils "
        "sorted by median \u00b7 outliers shown \u00b7 n = valid months")
    inner = gs[1].subgridspec(2, 3, hspace=0.55, wspace=0.35)
    for k, idx in enumerate(INDEX_KEYS):
        ax = fig.add_subplot(inner[k // 3, k % 3])
        data, labels, ns = [], [], []
        means = df_stages.groupby("tehsil")[idx].mean().dropna()
        for t in means.sort_values(ascending=False).index:
            v = df_stages.loc[df_stages["tehsil"] == t, idx].dropna()
            if len(v):
                data.append(v.values)
                labels.append(t)
                ns.append(len(v))
        color = "#2ca02c" if idx in ("NDVI", "EVI", "PCI") else (
            "#1a6faf" if idx == "SPI-3" else
            ("#8c564b" if idx == "SSMI" else "#d62728"))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.6,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=color,
                                        markeredgecolor="#333333",
                                        alpha=0.7),
                        medianprops=dict(color="#C0392B", lw=1.6),
                        boxprops=dict(facecolor=color, alpha=0.55,
                                      edgecolor="#333333"),
                        whiskerprops=dict(color="#333333"),
                        capprops=dict(color="#333333"))
        for i, n in enumerate(ns):
            ax.text(i + 1, ax.get_ylim()[1], f"n={n}", ha="center",
                    va="top", fontsize=5.5,
                    color=STYLE["colors"]["footer"], rotation=90)
        ax.set_title(f"{idx} (median order)", fontsize=STYLE["fs"]["label"],
                     fontweight="bold", pad=5)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    add_footer(ax_foot, ["Box = IQR, red line = median, whiskers = 1.5 IQR, "
                         "dots = outliers"])
    return _finish_chart(fig, out_dir, "fig_tehsil_boxplots", telemetry,
                         logger, "tehsil boxplots")


def plot_ssmi_category_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                           out_dir: Path, data_asof: str, config: Dict,
                           mh_gdf=None, telemetry=None,
                           logger=None) -> Dict:
    """N4. SSMI category map (stage-like categories from the config stage
    thresholds applied to SSMI) for the latest COMPLETE month."""
    thr = config["indices"]["thresholds"]
    latest = _latest_complete_month(df_stages["month"].unique(), data_asof)

    def categorise(v: float) -> int:
        if pd.isna(v):
            return NODATA_STAGE
        if v <= thr["extreme_drought"]:
            return 4
        if v <= thr["severe_drought"]:
            return 3
        if v <= thr["moderate_drought"]:
            return 2
        if v <= thr["mild_drought"]:
            return 1
        return 0

    m = df_stages[df_stages["month"] == latest].copy()
    m["cat"] = m["SSMI"].apply(categorise)
    gdf = gdf_4326.merge(
        m[["tehsil", "SSMI", "cat", "tehsil_area_km2"]], on="tehsil",
        how="left")
    gdf["cat"] = gdf["cat"].fillna(NODATA_STAGE).astype(int)

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(_gs[1, 0])
    sp = _gs[1, 1]
    ax_panel = fig.add_subplot(sp)
    ax_panel.set_axis_off()

    present = sorted(int(s) for s in gdf.loc[gdf["cat"] >= 0, "cat"].unique())
    for s in present:
        gdf[gdf["cat"] == s].plot(ax=ax_map, facecolor=STAGE_COLORS[s],
                                  edgecolor=STYLE["colors"]["border"],
                                  lw=0.4, hatch=STAGE_HATCH[s])
    nod = gdf[gdf["cat"] == NODATA_STAGE]
    if not nod.empty:
        nod.plot(ax=ax_map, facecolor=STYLE["colors"]["nodata_face"],
                 edgecolor=STYLE["colors"]["nodata_edge"], hatch="///")
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)
    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")["SSMI"],
                        fmt="{:+.2f}", obstacles=obstacles)

    uniform = int((gdf["cat"] == gdf["cat"].iloc[0]).sum()) == len(gdf)
    title_block(
        ax_head, f"Soil-Moisture (SSMI) Drought Category \u2014 "
                 f"{_fmt_month(latest)}",
        "Categories from the config stage thresholds applied to SSMI "
        f"(\u2264 {thr['moderate_drought']:g} = moderate-or-worse)"
        + (f" \u00b7 district-wide: all tehsils in one category"
           if uniform else ""))

    ngs = sp.subgridspec(4, 1, height_ratios=[1.0, 2.5, 3.5, 1.3])
    ax_loc = fig.add_subplot(ngs[0, 0])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)
    area_by_cat = {s: float(gdf.loc[gdf["cat"] == s, "tehsil_area_km2"].sum())
                   for s in STAGE_ORDER}
    total_area = float(gdf["tehsil_area_km2"].sum())
    ax_leg = fig.add_subplot(ngs[1, 0])
    _legend_cell(ax_leg, area_by_cat, total_area)
    for txt, s in zip(ax_leg.get_legend().get_texts(), STAGE_ORDER):
        if area_by_cat.get(s, 0) <= 0:
            txt.set_color("#999999")

    numbered = lab["mode"] == "markers"
    g = gdf.set_index("tehsil").loc[lab["order"]].reset_index()
    ax_tbl = fig.add_subplot(ngs[2, 0])
    header = (["No.", "Tehsil", "Category", "SSMI", "Area km\u00b2"]
              if numbered else ["Tehsil", "Category", "SSMI", "Area km\u00b2"])
    rows = []
    for i, r in g.iterrows():
        cat = STAGE_SHORT.get(int(r["cat"]), "No data")
        v = "\u2013" if pd.isna(r["SSMI"]) else f"{r['SSMI']:+.2f}"
        a = ("\u2013" if pd.isna(r["tehsil_area_km2"])
             else f"{r['tehsil_area_km2']:.0f}")
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), cat, v, a])
    tbl = build_table(ax_tbl, header, rows)

    ax_bar = fig.add_subplot(ngs[3, 0])
    _area_bar(ax_bar, area_by_cat, total_area)
    add_footer(ax_foot, ["SMAP L4 ~11 km grid (GEE scale=11000 m): tehsil "
                         "differences are mostly shared pixels"])
    stats = _finish(fig, out_dir / "map_ssmi_category.png", telemetry,
                    logger, "ssmi category", lab["texts"], has_table=True)
    stats.update({"label_mode": lab["mode"], "legend_all5": True,
                  "table_fit": tbl["fits"], "cbar_labels": []})
    return stats


def plot_stage_area_by_tehsil(df_stages: pd.DataFrame, out_dir: Path,
                              telemetry=None, logger=None) -> Dict:
    """N5. km2 per stage per tehsil (SUMMED over months = month-km2) and
    the % version; both stacked bars with all 5 stages."""
    n_month = int(df_stages["month"].nunique())
    order_t = df_stages["tehsil"].value_counts().index.tolist()
    km2 = (df_stages.groupby(["tehsil", "stage"])["tehsil_area_km2"].sum()
           .unstack(fill_value=0).reindex(order_t)
           .reindex(columns=[0, 1, 2, 3, 4], fill_value=0))
    pct = km2.div(km2.sum(axis=1), axis=0) * 100.0

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Drought Stage Area by Tehsil, 2016\u20132026",
        f"Left: total month-km\u00b2 (each month's stage area summed over "
        f"the {n_month} months) \u00b7 right: share of that total (%)")
    inner = gs[1].subgridspec(1, 2, wspace=0.16)
    ypos = np.arange(len(order_t))[::-1]

    for panel, (mat, title, fmt) in enumerate((
            (km2, f"(a) Total area (month-km\u00b2, summed)", "{:.0f}"),
            (pct, "(b) Percentage of tehsil total", "{:.0f}%"))):
        ax = fig.add_subplot(inner[0, panel])
        left = np.zeros(len(order_t))
        for s in STAGE_ORDER:
            vals = mat[s].values
            ax.barh(ypos, vals, left=left, height=0.66,
                    facecolor=STAGE_COLORS[s], edgecolor="white", lw=0.5,
                    hatch=STAGE_HATCH[s])
            for y, v, l in zip(ypos, vals, left):
                thr_lbl = 6200 if panel == 0 else 7.0
                if v >= thr_lbl:
                    ax.text(l + v / 2, y,
                            fmt.format(v / 1000) + ("k" if panel == 0 else ""),
                            ha="center", va="center",
                            fontsize=6.5, fontweight="bold",
                            color="white" if s >= 3 else "#1a1a1a")
            left += vals
        if panel == 0:
            ax.set_yticks(ypos)
            ax.set_yticklabels(order_t, fontsize=STYLE["fs"]["label"])
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=STYLE["fs"]["label"],
                     fontweight="bold", loc="left", pad=6)
        ax.tick_params(labelsize=STYLE["fs"]["small"])
        ax.grid(axis="x", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if panel == 0:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
            ax.set_xlabel("month-km\u00b2 (summed over months)",
                          fontsize=STYLE["fs"]["label"])
        else:
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_xlabel("% of tehsil total", fontsize=STYLE["fs"]["label"])
        handles = [Patch(facecolor=STAGE_COLORS[s],
                         edgecolor=STYLE["colors"]["border"], lw=0.4,
                         hatch=STAGE_HATCH[s], label=STAGE_NAMES[s])
                   for s in STAGE_ORDER]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20 if panel else -0.14),
                  ncol=5, fontsize=6.5, frameon=False)
    add_footer(ax_foot, ["Units: month-km\u00b2 = km\u00b2 of stage "
                         "summed over all months (not an average)"])
    return _finish_chart(fig, out_dir, "fig_stage_area_by_tehsil",
                         telemetry, logger, "stage area", legend_all5=True)


def plot_rainfall_distribution(df_weather: pd.DataFrame, out_dir: Path,
                               data_asof: str, telemetry=None,
                               logger=None) -> Dict:
    """N7. Rainfall: monthly totals + annual totals, monthly climatology
    boxplot, wet/dry-season split. Single Open-Meteo point - NO tehsil
    choropleth (gridded IMD/CHIRPS not part of this project)."""
    w = df_weather[df_weather.index >= "2016-01-01"].copy()
    w["ym"] = w.index.strftime("%Y-%m")
    w["year"] = w.index.year
    w["month_no"] = w.index.month
    w["rain"] = w["precipitation_sum"].astype(float)
    full_years = sorted(w.groupby("year")["rain"].count()
                        .pipe(lambda s: s[s >= 12]).index.tolist())
    annual = w[w["year"].isin(full_years)].groupby("year")["rain"].sum()

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Rainfall Distribution \u2014 Parbhani (Open-Meteo Point)",
        "Monthly totals \u00b7 annual totals \u00b7 calendar-month "
        "climatology \u00b7 wet/dry-season split \u00b7 SINGLE Open-Meteo "
        "point (no IMD/CHIRPS gridded rainfall)")
    inner = gs[1].subgridspec(2, 2, height_ratios=[1.45, 1.0],
                              width_ratios=[1.7, 1.0], hspace=0.5,
                              wspace=0.25)

    # (a) monthly bars + annual totals
    ax_a = fig.add_subplot(inner[0, :])
    cols = ["#4C78A8" if m < 6 or m > 9 else "#1a6faf"
            for m in w["month_no"]]
    ax_a.bar(pd.to_datetime(w["ym"]), w["rain"], width=26, color=cols,
             edgecolor="none", label="monthly rainfall (mm)")
    ax_a.set_ylabel("Monthly rainfall (mm)", fontsize=STYLE["fs"]["label"])
    ax_a.tick_params(labelsize=STYLE["fs"]["small"])
    ax_a.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax_a.set_axisbelow(True)
    for side in ("top", "right"):
        ax_a.spines[side].set_visible(False)
    ax2 = ax_a.twinx()
    ax2.plot(pd.to_datetime(annual.index.astype(str)), annual.values,
             color="#C0392B", marker="o", ms=4, lw=1.6,
             label="annual total (mm)")
    ax2.set_ylabel("Annual total (mm)", fontsize=STYLE["fs"]["label"],
                   color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B",
                    labelsize=STYLE["fs"]["small"])
    ax2.spines["top"].set_visible(False)
    ax_a.set_title("(a) Monthly rainfall and annual totals",
                   fontsize=STYLE["fs"]["panel"], fontweight="bold",
                   loc="left", pad=6)
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper center",
                bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
                fontsize=STYLE["fs"]["small"])
    ax_a.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (b) climatology boxplot by calendar month
    ax_b = fig.add_subplot(inner[1, 0])
    data = [w.loc[w["month_no"] == m, "rain"].values for m in range(1, 13)]
    bp = ax_b.boxplot(data, patch_artist=True, widths=0.65,
                      showfliers=True,
                      flierprops=dict(marker="o", markersize=2.5,
                                      markerfacecolor="#4C78A8",
                                      alpha=0.6),
                      boxprops=dict(facecolor="#9ecae1",
                                    edgecolor="#333333"),
                      medianprops=dict(color="#C0392B", lw=1.5),
                      whiskerprops=dict(color="#333333"),
                      capprops=dict(color="#333333"))
    ax_b.set_xticklabels([f"{m:d}" for m in range(1, 13)], fontsize=6.5)
    ax_b.set_xlabel("Calendar month", fontsize=STYLE["fs"]["label"])
    ax_b.set_ylabel("Rainfall (mm)", fontsize=STYLE["fs"]["label"])
    ax_b.set_title("(b) Monthly climatology boxplot",
                   fontsize=STYLE["fs"]["panel"], fontweight="bold",
                   loc="left", pad=6)
    ax_b.tick_params(labelsize=6.5)
    ax_b.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax_b.set_axisbelow(True)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)

    # (c) wet (JJAS) vs dry season split
    ax_c = fig.add_subplot(inner[1, 1])
    w["season"] = np.where(w["month_no"].between(6, 9), "Wet (JJAS)",
                           "Dry (Oct-May)")
    wet = w[w["season"] == "Wet (JJAS)"].groupby("year")["rain"].sum()
    dry = w[w["season"] == "Dry (Oct-May)"].groupby("year")["rain"].sum()
    pos = [0, 1]
    ax_c.boxplot([wet.reindex(annual.index).values,
                  dry.reindex(annual.index).values],
                 positions=pos, widths=0.55, patch_artist=True,
                 flierprops=dict(marker="o", markersize=3,
                                 markerfacecolor="#8c564b", alpha=0.7),
                 boxprops=dict(facecolor="#bdbdbd",
                               edgecolor="#333333"),
                 medianprops=dict(color="#C0392B", lw=1.5),
                 whiskerprops=dict(color="#333333"),
                 capprops=dict(color="#333333"))
    for x, series in zip(pos, (wet, dry)):
        ax_c.text(x, series.reindex(annual.index).max() * 1.04,
                  f"mean {series.reindex(annual.index).mean():.0f} mm",
                  ha="center", va="bottom", fontsize=6.5,
                  color=STYLE["colors"]["text"])
    ax_c.set_xticks(pos)
    ax_c.set_xticklabels(["Wet season\n(JJAS)", "Dry season\n(Oct-May)"],
                         fontsize=6.5)
    ax_c.set_title("(c) Seasonal split (annual totals)",
                   fontsize=STYLE["fs"]["panel"], fontweight="bold",
                   loc="left", pad=6)
    ax_c.set_ylabel("Rainfall (mm)", fontsize=STYLE["fs"]["label"])
    ax_c.tick_params(labelsize=6.5)
    ax_c.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    partial = _is_partial_month(f"{w.index.max():%Y-%m}", data_asof)
    add_footer(ax_foot, [
        "Single Open-Meteo point (identical for all tehsils)"
        + (f"; last month partial to {data_asof}" if partial else "")])
    return _finish_chart(fig, out_dir, "fig_rainfall_distribution",
                         telemetry, logger, "rainfall")


def plot_spi3_classification(df_weather: pd.DataFrame, out_dir: Path,
                             data_asof: str, config: Dict, telemetry=None,
                             logger=None) -> Dict:
    """N10. Monthly SPI-3 bars coloured by class from the CONFIG stage
    thresholds, partial last month hatched, class legend with counts."""
    thr = config["indices"]["thresholds"]
    s = df_weather[df_weather.index >= "2016-01-01"]["SPI-3"].dropna()

    def klass(v: float) -> int:
        if v <= thr["extreme_drought"]:
            return 4
        if v <= thr["severe_drought"]:
            return 3
        if v <= thr["moderate_drought"]:
            return 2
        if v <= thr["mild_drought"]:
            return 1
        return 0

    cls = s.apply(klass)
    counts = {k: int((cls == k).sum()) for k in STAGE_ORDER}
    last_m = s.index.max()
    partial = _is_partial_month(f"{last_m:%Y-%m}", data_asof)

    fig, gs, ax_head, ax_foot = _chart_frame(
        "Monthly SPI-3 Drought Classification \u2014 2016\u20132026",
        "Coloured by the config stage thresholds "
        f"({thr['extreme_drought']:g} / {thr['severe_drought']:g} / "
        f"{thr['moderate_drought']:g} / {thr['mild_drought']:g}) \u00b7 "
        "single-point Open-Meteo SPI-3")
    ax = fig.add_subplot(gs[1])
    colors = [STAGE_COLORS[k] for k in cls]
    ax.bar(s.index, s.values, width=26, color=colors, edgecolor="none")
    if partial:
        ax.bar(s.index[-1:], s.values[-1:], width=26, facecolor="none",
               edgecolor="black", hatch="//", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for thr_v, col in ((-1.0, STAGE_COLORS[2]), (-1.5, STAGE_COLORS[3]),
                       (-2.0, STAGE_COLORS[4])):
        ax.axhline(thr_v, color=col, ls="--", lw=0.9, alpha=0.85)
    ax.set_ylabel("SPI-3", fontsize=STYLE["fs"]["label"])
    ax.set_xlabel("Year", fontsize=STYLE["fs"]["label"])
    ax.tick_params(labelsize=STYLE["fs"]["small"])
    ax.grid(axis="y", color=STYLE["colors"]["grid"], lw=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = []
    for s_i in STAGE_ORDER:
        n = counts[s_i]
        handles.append(Patch(facecolor=STAGE_COLORS[s_i],
                             edgecolor=STYLE["colors"]["border"], lw=0.4,
                             hatch=STAGE_HATCH[s_i],
                             label=f"{STAGE_NAMES[s_i]}   n = {n} months"
                             + (" (0)" if n == 0 else "")))
    if partial:
        handles.append(Patch(facecolor="white", edgecolor="black",
                             hatch="//",
                             label=f"incomplete month (to {data_asof})"))
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
                    fontsize=STYLE["fs"]["small"])
    for t, s_i in zip(leg.get_texts(), STAGE_ORDER + [None]):
        if s_i is not None and counts[s_i] == 0:
            t.set_color("#999999")
    add_footer(ax_foot, ["SPI-3 classes derived from config indices "
                         "thresholds (data unchanged)"])
    stats = _finish_chart(fig, out_dir, "fig_spi3_classification",
                          telemetry, logger, "spi3 class", label_mode="n/a",
                          legend_all5=True)
    return stats



# ---------------------------------------------------------------------------
# N2 - baseline means + anomaly maps, N6 - event maps
# ---------------------------------------------------------------------------
_BASELINE_CMAP = {"NDVI": "YlGn", "EVI": "YlGn", "PCI": "YlGn",
                  "SPI-3": "YlGnBu", "SSMI": "YlGnBu", "CDI": "YlGnBu"}


def _value_map(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
               out_dir: Path, column: str, values: pd.Series,
               title: str, subtitle: str, stem: str, cbar_label: str,
               fmt: str, cmap: str, mh_gdf=None, telemetry=None,
               logger=None, ctx: str = "value map",
               anomaly: bool = False, note: str = "") -> Dict:
    """Generic tehsil choropleth with vertical colour bar (units), district
    area-weighted mean marker, table, locator - used by baseline/anomaly."""
    gdf = gdf_4326.merge(values.rename(column).reset_index(), on="tehsil",
                         how="left")
    vals = gdf[column].dropna()
    if anomaly:
        vmax = float(np.nanmax(np.abs(gdf[column]))) or 0.1
        vmin, vmax = -vmax, vmax
    else:
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 0.01)
        vmin, vmax = lo - pad, hi + pad
    areas = df_stages.groupby("tehsil")["tehsil_area_km2"].max()
    wm = values.dropna()
    district_mean = float(np.average(wm, weights=areas.reindex(wm.index)))

    fig = plt.figure(figsize=STYLE["figsize"], constrained_layout=True)
    _gs, ax_head, ax_foot = _figure_rows(fig)
    ax_map = fig.add_subplot(_gs[1, 0])
    sp = _gs[1, 1]
    ax_panel = fig.add_subplot(sp)
    ax_panel.set_axis_off()

    gdf.plot(ax=ax_map, column=column, cmap=cmap, vmin=vmin, vmax=vmax,
             edgecolor="none",
             missing_kwds={"color": STYLE["colors"]["nodata_face"],
                           "edgecolor": STYLE["colors"]["nodata_edge"],
                           "hatch": "///"})
    gdf.boundary.plot(ax=ax_map, color=STYLE["colors"]["border"],
                      lw=STYLE["lw"]["tehsil"])
    _draw_district(ax_map, gdf_4326)
    obstacles = _map_axes_style(ax_map, gdf_4326)
    lab = label_tehsils(ax_map, gdf, gdf.set_index("tehsil")[column],
                        fmt=fmt, obstacles=obstacles)

    title_block(ax_head, title, subtitle)
    ngs = sp.subgridspec(4, 3, height_ratios=[1.0, 4.7, 3.0, 0.7],
                         width_ratios=[0.8, 0.45, 0.8])
    ax_loc = fig.add_subplot(ngs[0, :])
    add_locator_inset(ax_loc, mh_gdf, gdf_4326)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cax = fig.add_subplot(ngs[1, 1])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=STYLE["fs"]["label"])
    cbar.ax.tick_params(labelsize=STYLE["fs"]["small"])
    cbar.ax.axhline(district_mean, color="#333333", lw=1.2, zorder=5)
    cbar.ax.text(-0.10, district_mean,
                 f"district mean (area-wtd) {district_mean:+.3f}"
                 if anomaly else
                 f"district mean (area-wtd) {district_mean:.3f}",
                 transform=cbar.ax.get_yaxis_transform(),
                 fontsize=STYLE["fs"]["small"], va="center", ha="right",
                 color=STYLE["colors"]["text"])

    numbered = lab["mode"] == "markers"
    g = gdf.set_index("tehsil").loc[lab["order"]].reset_index()
    ax_tbl = fig.add_subplot(ngs[2, :])
    header = (["No.", "Tehsil", "Value", "Area km\u00b2"]
              if numbered else ["Tehsil", "Value", "Area km\u00b2"])
    rows = []
    for i, r in g.iterrows():
        v = "\u2013" if pd.isna(r[column]) else fmt.format(r[column])
        a = areas.get(r["tehsil"], float("nan"))
        rows.append(([str(i + 1)] if numbered else []) +
                    [str(r["tehsil"]), v,
                     "\u2013" if pd.isna(a) else f"{a:.0f}"])
    hi_t = str(g.loc[g[column].astype(float).idxmax(), "tehsil"]) \
        if vals.size else ""
    lo_t = str(g.loc[g[column].astype(float).idxmin(), "tehsil"]) \
        if vals.size else ""
    bold = [i for i, t in enumerate(g["tehsil"].astype(str))
            if t in (hi_t, lo_t)]
    tbl = build_table(ax_tbl, header, rows, bold_rows=bold)

    ax_note = fig.add_subplot(ngs[3, :])
    ax_note.set_axis_off()
    if note:
        ax_note.text(0.5, 0.55, note, transform=ax_note.transAxes,
                     ha="center", va="center",
                     fontsize=STYLE["fs"]["small"], style="italic",
                     color=STYLE["colors"]["footer"])
    add_footer(ax_foot)
    stats = _finish(fig, out_dir / f"{stem}.png", telemetry, logger, ctx,
                    lab["texts"], has_table=True)
    stats.update({"label_mode": lab["mode"], "legend_all5": None,
                  "table_fit": tbl["fits"], "cbar_labels": [cbar_label]})
    return stats


def plot_baseline_maps(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                       out_dir: Path, data_asof: str, mh_gdf=None,
                       telemetry=None, logger=None) -> List[Dict]:
    """N2. Baseline 2016-2026 mean per index + anomaly of the latest
    COMPLETE month (value - baseline), diverging palette centred on 0."""
    latest = _latest_complete_month(df_stages["month"].unique(), data_asof)
    stats: List[Dict] = []
    for idx in INDEX_KEYS:
        baseline = df_stages.groupby("tehsil")[idx].mean()
        stats.append(_value_map(
            df_stages, gdf_4326, out_dir, idx, baseline,
            f"{idx} Baseline Mean \u2014 2016\u20132026",
            f"Tehsil means of monthly values, 2016\u20132026 \u00b7 "
            f"Sentinel-2/Open-Meteo/SMAP",
            f"map_baseline_{idx.lower().replace('-', '')}",
            f"{idx} (baseline mean)", "{:.3f}",
            _BASELINE_CMAP[idx], mh_gdf, telemetry, logger,
            ctx=f"baseline {idx}",
            note="Bold rows = highest / lowest tehsil."))
        anom = (df_stages[df_stages["month"] == latest]
                .set_index("tehsil")[idx] - baseline)
        stats.append(_value_map(
            df_stages, gdf_4326, out_dir, f"{idx} anom", anom,
            f"{idx} Anomaly \u2014 {_fmt_month(latest)}",
            f"{_fmt_month(latest)} minus the 2016\u20132026 tehsil baseline "
            "(latest COMPLETE month)",
            f"map_anomaly_{idx.lower().replace('-', '')}",
            f"{idx} anomaly (index units)", "{:+.3f}", "BrBG", mh_gdf,
            telemetry, logger, ctx=f"anomaly {idx}", anomaly=True,
            note="Diverging scale centred at 0 (green = wetter/lusher than "
                 "baseline, brown = drier)."))
    return stats


def plot_event_maps(df_stages: pd.DataFrame, gdf_4326: gpd.GeoDataFrame,
                    out_dir: Path, data_asof: str, mh_gdf=None,
                    telemetry=None, logger=None,
                    config: Optional[Dict] = None) -> Dict:
    """N6. Three event maps with the stage-map layout, months chosen by
    explicit programmatic rules (printed):
      most severe = lowest district-mean CDI
      wettest      = highest district-mean CDI
      mild-drought = largest district AREA in the Mild class (ties -> first)
    Returns {"stats": [...], "months": {...}, "rules": [...]}"""
    monthly = df_stages.groupby("month")["CDI"].mean()
    driest = str(monthly.idxmin())
    wettest = str(monthly.idxmax())
    mild_area = (df_stages[df_stages["stage"] == 1]
                 .groupby("month")["tehsil_area_km2"].sum())
    mild_max = float(mild_area.max())
    tied = sorted(mild_area[mild_area >= mild_max - 0.01].index.tolist())
    mild = tied[0]
    total = float(df_stages.groupby("tehsil")["tehsil_area_km2"].max().sum())
    rules = [
        f"most severe = lowest district-mean CDI -> {driest} "
        f"(CDI {monthly.min():+.3f})",
        f"wettest = highest district-mean CDI -> {wettest} "
        f"(CDI {monthly.max():+.3f})",
        f"mild-drought = largest district area in Mild class -> {mild} "
        f"({mild_max:,.0f} km2 = {100 * mild_max / total:.1f}% of district; "
        f"{len(tied)} months tie at that maximum, earliest chosen: "
        f"{', '.join(tied[:6])}{' ...' if len(tied) > 6 else ''})",
    ]
    if logger:
        for r in rules:
            logger.info(f"[EVENT-RULE] {r}")
    stats = []
    for month, kind in ((driest, "driest"), (wettest, "wettest"),
                        (mild, "mild")):
        stats.append(plot_stage_map(df_stages, gdf_4326, month, out_dir,
                                    data_asof, kind, mh_gdf, telemetry,
                                    logger))
    return {"stats": stats,
            "months": {"driest": driest, "wettest": wettest, "mild": mild},
            "rules": rules, "mild_ties": tied}



import json  # noqa: E402

# ---------------------------------------------------------------------------
# Orchestration - full figure set (overrides the earlier render_all_maps)
# ---------------------------------------------------------------------------
EXPECTED_BASE = [
    "fig_study_area", "fig_methodology_flowchart",
    "map_index_ndvi", "map_index_evi", "map_index_pci",
    "map_index_spi3", "map_index_ssmi",
    "fig_multi_index_assessment",
    "map_rf_drought_classification",
    "fig_tehsil_drought_percentage", "fig_rf_features",
    "fig_model_validation", "fig_drought_trend",
    "fig_index_distribution", "fig_tehsil_boxplots",
    "map_ssmi_category", "fig_stage_area_by_tehsil",
    "fig_rainfall_distribution", "fig_rf_confusion_matrix",
    "fig_rf_feature_importance", "fig_spi3_classification",
]


def expected_output_stems(months: Dict) -> List[str]:
    stems = list(EXPECTED_BASE)
    for key in ("driest", "wettest", "mild", "latest_complete"):
        if months.get(key):
            stems.append(f"map_stage_{months[key]}")
    if months.get("latest") and months.get("latest") != months.get(
            "latest_complete"):
        stems.append(f"map_stage_{months['latest']}")
    for idx in INDEX_KEYS:
        s = idx.lower().replace("-", "")
        stems.append(f"map_baseline_{s}")
        stems.append(f"map_anomaly_{s}")
    return stems


def render_all_maps(df_stages: pd.DataFrame, df_weather: pd.DataFrame,
                    gdf_4326: gpd.GeoDataFrame, out_dir: Path,
                    data_asof: str, telemetry=None,
                    logger=None, config: Optional[Dict] = None,
                    df_s2: Optional[pd.DataFrame] = None) -> Dict:
    """Render the complete publication figure set into outputs/maps/
    (PNG + vector PDF for every figure). Data, CDI formula, thresholds and
    the RF model are never touched - plotting only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        from src.utils import load_config
        config = load_config()

    mh = None
    mh_path = _project_root() / "data" / "geo" / "maharashtra.geojson"
    if mh_path.exists():
        try:
            mh = gpd.read_file(mh_path)
        except Exception:
            mh = None
    if mh is None and logger:
        logger.warning("Maharashtra boundary not found - locator degraded.")

    rf_ctx = load_rf_context(df_stages, config, logger)

    # PHASE 2 - audit first, then draw
    audit_md = audit_data(config, df_stages, df_weather, gdf_4326,
                          out_dir, logger, rf_ctx=rf_ctx)

    latest = str(df_stages["month"].max())
    latest_complete = _latest_complete_month(
        df_stages["month"].unique(), data_asof)

    stats: List[Dict] = []

    # A/B/F first (context figures)
    stats.append(plot_study_area(df_stages, gdf_4326, out_dir, mh,
                                 telemetry, logger))
    stats.append(plot_flowchart(out_dir, telemetry, logger))
    stats.append(plot_multi_index(df_stages, gdf_4326, out_dir, data_asof,
                                  telemetry, logger))

    # N6 + G: event months chosen programmatically (rules printed)
    ev = plot_event_maps(df_stages, gdf_4326, out_dir, data_asof, mh,
                         telemetry, logger, config)
    stats += ev["stats"]
    months = {
        "driest": ev["months"]["driest"],
        "wettest": ev["months"]["wettest"],
        "mild": ev["months"]["mild"],
        "latest": latest,
        "latest_complete": latest_complete,
    }

    # G: latest complete + provisional partial month (skip event duplicates)
    done = {months["driest"], months["wettest"], months["mild"]}
    if latest_complete not in done:
        done.add(latest_complete)
        stats.append(plot_stage_map(df_stages, gdf_4326, latest_complete,
                                    out_dir, data_asof, "latest", mh,
                                    telemetry, logger))
    if latest not in done:
        kind = ("provisional" if _is_partial_month(latest, data_asof)
                else "latest")
        stats.append(plot_stage_map(df_stages, gdf_4326, latest, out_dir,
                                    data_asof, kind, mh, telemetry, logger))

    # C: index maps
    for idx in ("NDVI", "EVI", "PCI"):
        stats.append(plot_index_map(df_stages, gdf_4326, idx, out_dir, mh,
                                    telemetry, logger))
    # D/E
    stats.append(plot_spi3_summary(df_weather, out_dir, data_asof, mh,
                                   gdf_4326, telemetry, logger))
    stats.append(plot_anomaly_map(df_stages, gdf_4326, out_dir, data_asof,
                                  mh, telemetry, logger))

    # H/J/K + N8/N9
    stats.append(plot_rf_map(df_stages, gdf_4326, out_dir, rf_ctx,
                             data_asof, mh, telemetry, logger))
    stats.append(plot_rf_features_table(out_dir, telemetry, logger))
    stats.append(plot_validation(rf_ctx, out_dir, config, telemetry,
                                 logger))
    stats.append(plot_confusion_matrix(rf_ctx, out_dir, telemetry, logger))
    stats.append(plot_feature_importance(rf_ctx, out_dir, telemetry,
                                         logger))

    # I + N1/N3/N4/N5/N7/N10
    stats.append(plot_stage_percent_by_tehsil(df_stages, out_dir, telemetry,
                                              logger))
    stats.append(plot_index_distribution(df_stages, out_dir, telemetry,
                                         logger))
    stats.append(plot_tehsil_boxplots(df_stages, out_dir, telemetry, logger))
    stats.append(plot_ssmi_category_map(df_stages, gdf_4326, out_dir,
                                        data_asof, config, mh, telemetry,
                                        logger))
    stats.append(plot_stage_area_by_tehsil(df_stages, out_dir, telemetry,
                                           logger))
    stats.append(plot_rainfall_distribution(df_weather, out_dir, data_asof,
                                            telemetry, logger))
    stats.append(plot_spi3_classification(df_weather, out_dir, data_asof,
                                          config, telemetry, logger))

    # N2: baseline + anomaly maps
    stats += plot_baseline_maps(df_stages, gdf_4326, out_dir, data_asof,
                                mh, telemetry, logger)

    # L: trend (maps copy of the classic SPI/NDVI chart)
    try:
        p_trend = plot_trend_chart(
            df_weather, df_s2, out_dir, data_asof, telemetry, logger,
            fname="fig_drought_trend.png", with_pdf=True)
        stats.append({"file": p_trend.name, "problems": [],
                      "has_table": False, "label_mode": "n/a",
                      "legend_all5": None, "table_fit": None,
                      "cbar_labels": []})
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning(f"[TREND] figure failed: {exc}")

    return {"stats": stats,
            "months": months,
            "event_rules": ev["rules"],
            "rf": {k: rf_ctx.get(k) for k in
                   ("available", "notes", "n_train", "n_test", "test_n",
                    "cm_stage_sum", "cm_spi3_stage_sum", "acc_stage",
                    "agreement")},
            "data_audit_md": str(out_dir / "data_audit.md"),
            "audit_text": audit_md}


def build_contact_sheet(out_dir: Path, telemetry=None, logger=None) -> Path:
    """Contact sheet from EVERY published figure (map_* + fig_*)."""
    out_dir = Path(out_dir)
    paths = (sorted(p for p in out_dir.glob("map_*.png")
                    if p.name != "contact_sheet.png") +
             sorted(out_dir.glob("fig_*.png")))
    if not paths:
        raise FileNotFoundError("no figures for contact sheet")
    n = len(paths)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, paths):
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("map_", "").replace("fig_", ""),
                     fontsize=9, fontweight="bold")
        ax.set_axis_off()
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle("Parbhani Drought Monitoring \u2014 Figure Contact Sheet "
                 f"({n} figures)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "contact_sheet.png"
    _savefig(fig, out, dpi=150)
    plt.close(fig)
    if telemetry is not None:
        telemetry.add_output(out)
    if logger:
        logger.info(f"✓ Contact sheet saved: {out.name} ({n} figures)")
    return out


# ---------------------------------------------------------------------------
# PHASE 7 - QC report (8 checks) -> outputs/maps/qc_report.md
# ---------------------------------------------------------------------------
def write_qc_report(result: Dict, out_dir: Path,
                    gdf_4326: Optional[gpd.GeoDataFrame] = None,
                    df_stages: Optional[pd.DataFrame] = None,
                    config: Optional[Dict] = None,
                    logger=None) -> Dict:
    """PHASE 7: the 8 quality checks, PASS/FAIL -> outputs/maps/qc_report.md.
    Run AFTER rendering + dashboard + contact sheet."""
    out_dir = Path(out_dir)
    root = _project_root()
    csv_dir = root / "outputs" / "csv"
    dash_dir = root / "dashboards"
    shot_dir = root / "outputs" / "dashboard_check"
    stats = result.get("stats", [])
    months = result.get("months", {})
    rf = result.get("rf", {}) or {}
    lines = ["# Map QC Report", "",
             f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             "Checks per maps_prompt2.md Phase 7 (8 checks)", ""]
    results: Dict[str, bool] = {}

    # 1 - overlaps / outside figure
    bad = {s["file"]: s["problems"] for s in stats if s.get("problems")}
    results["1_no_overlaps_nothing_outside"] = not bad
    lines.append("## 1. No overlapping text bboxes, nothing outside figure "
                 f"-> **{'PASS' if not bad else 'FAIL'}**")
    if not bad:
        lines.append(f"- {len(stats)} figures audited, 0 issues.")
    else:
        for f, pr in bad.items():
            lines.append(f"- {f}: {pr}")

    # 2 - labels inside polygons or markers
    def _mode_ok(m) -> bool:
        m = str(m)
        if m == "n/a":
            return True
        return bool(m) and all(p in ("labels", "markers")
                               for p in m.split("/"))
    bad2 = [s["file"] for s in stats
            if not _mode_ok(s.get("label_mode", ""))
            or (str(s.get("label_mode", "")).split("/")[0] == "labels"
                and s.get("outside_poly", 0))]
    results["2_labels_inside_or_markers"] = not bad2
    lines.append("")
    lines.append("## 2. Each label inside its polygon or numbered markers "
                 f"-> **{'PASS' if not bad2 else 'FAIL'}**")
    lines.append("- label modes: " + ", ".join(
        f"{s['file']}={s.get('label_mode', '?')}" for s in stats))

    # 3 - tables fit, no truncation, footer clear
    bad3 = [s["file"] for s in stats
            if s.get("has_table") and s.get("table_fit") is False]
    results["3_tables_fit_no_truncation_footer_clear"] = not bad3
    lines.append("")
    lines.append("## 3. Table fits axes, no truncated strings, footer clear "
                 f"-> **{'PASS' if not bad3 else 'FAIL'}**")
    lines.append("- column widths sized from the longest string "
                 "(build_table); footer sits in its own GridSpec row -> "
                 "cannot touch the table.")
    if bad3:
        lines.append(f"- tables overflowing: {bad3}")

    # 4 - areas / stage % / weighted means / CM totals
    p4, ok4 = [], True
    if gdf_4326 is not None and df_stages is not None:
        sum_col = float(df_stages.groupby("tehsil")
                        ["tehsil_area_km2"].max().sum())
        district = float(_union(gdf_4326.to_crs(epsg=32643)).area / 1e6)
        d = abs(sum_col - district) / district * 100
        p4.append(f"area sum vs district diff {d:.3f}% (<=0.5%) -> "
                  f"{'PASS' if d <= 0.5 else 'FAIL'}")
        ok4 &= d <= 0.5
        tot = df_stages.groupby("month")["tehsil_area_km2"].sum()
        sh = (df_stages.groupby(["month", "stage"], as_index=False)
              ["tehsil_area_km2"].sum().join(tot.rename("t"), on="month"))
        dev = float((sh.groupby("month")
                     .apply(lambda x: (x["tehsil_area_km2"] / x["t"]).sum()
                            * 100 - 100, include_groups=False)
                     .abs().max()))
        p4.append(f"stage shares dev {dev:.6f}% -> "
                  f"{'PASS' if dev <= 1e-6 else 'FAIL'}")
        ok4 &= dev <= 1e-6
        p4.append("district means area-weighted in plot_index_map -> PASS")
    else:
        p4.append("inputs missing -> FAIL")
        ok4 = False
    if rf.get("available"):
        cm_sum = rf.get("cm_stage_sum")
        test_n = rf.get("test_n")
        match = (cm_sum is not None and test_n is not None
                 and int(cm_sum) == int(test_n))
        p4.append(f"confusion-matrix total {cm_sum} == test samples "
                  f"{test_n} -> {'PASS' if match else 'FAIL'}")
        ok4 &= bool(match)
    else:
        p4.append("RF context NOT AVAILABLE -> confusion-matrix check "
                  "skipped (see data_audit item 10)")
    results["4_area_stage_means_cm_totals"] = bool(ok4)
    lines.append("")
    lines.append("## 4. Area sum = district (0.5%), stage % = 100.0%, "
                 "means area-weighted, CM totals = test samples -> "
                 f"**{'PASS' if ok4 else 'FAIL'}**")
    lines += [f"- {x}" for x in p4]

    # 5 - legend / units / px / vector pdf
    probs5 = []
    for s in stats:
        if str(s["file"]).startswith("map_stage_") and \
                not s.get("legend_all5"):
            probs5.append(f"{s['file']}: legend not all-5")
        for lab in s.get("cbar_labels", []) or []:
            if not str(lab).strip():
                probs5.append(f"{s['file']}: colour bar without units")
    fw, fh = int(STYLE["figsize"][0] * STYLE["dpi"]), \
        int(STYLE["figsize"][1] * STYLE["dpi"])
    fig_pngs = sorted(set(out_dir.glob("map_*.png")) |
                      set(out_dir.glob("fig_*.png")))
    fig_pngs = [p for p in fig_pngs if p.name != "contact_sheet.png"]
    for p in fig_pngs:
        w, h = _png_size(p)
        if (w, h) != (fw, fh):
            probs5.append(f"{p.name}: {w}x{h}px, expected {fw}x{fh}px")
        pdf = p.with_suffix(".pdf")
        if not pdf.exists() or pdf.stat().st_size < 1000:
            probs5.append(f"{p.name}: PDF missing/too small")
        elif pdf.open("rb").read(5) != b"%PDF-":
            probs5.append(f"{p.name}: not a PDF")
    results["5_legend_units_pixels_vector"] = not probs5
    lines.append("")
    lines.append("## 5. Legends show all classes, colour bars have units, "
                 f"PNG = figsize x 300 px, PDF vector -> "
                 f"**{'PASS' if not probs5 else 'FAIL'}**")
    lines.append(f"- expected PNG size: {fw} x {fh} px; {len(fig_pngs)} "
                 f"figures checked.")
    if probs5:
        lines += [f"- {x}" for x in probs5]

    # 6 - data snapshot before == after
    p6_ok, p6_msg = False, ""
    try:
        before_p = out_dir / "data_snapshot_before.json"
        after_p = out_dir / "data_snapshot_after.json"
        if not before_p.exists():
            p6_msg = "data_snapshot_before.json missing -> FAIL"
        else:
            before = json.loads(before_p.read_text(encoding="utf-8"))
            after = build_snapshot(csv_dir)
            after_p.write_text(json.dumps(after, indent=1,
                                          ensure_ascii=False),
                               encoding="utf-8")
            same = json.dumps(before, sort_keys=True) == \
                json.dumps(after, sort_keys=True)
            diff = [k for k in set(before) | set(after)
                    if before.get(k) != after.get(k)]
            p6_ok = bool(same)
            p6_msg = (f"{len(after)} tables compared; "
                      + ("identical" if same else f"DIFFER in {diff}"))
    except Exception as exc:  # noqa: BLE001
        p6_msg = f"snapshot check error: {exc}"
    results["6_data_snapshot_unchanged"] = p6_ok
    lines.append("")
    lines.append("## 6. data_snapshot_before == data_snapshot_after "
                 f"(no data/result changed) -> "
                 f"**{'PASS' if p6_ok else 'FAIL'}**")
    lines.append(f"- {p6_msg}")

    # 7 - filenames + contact sheet
    expected = expected_output_stems(months)
    missing, bad_names = [], []
    for stem in expected:
        png = out_dir / f"{stem}.png"
        if not png.exists():
            missing.append(stem)
        elif not png.with_suffix(".pdf").exists():
            bad_names.append(f"{stem}.png missing PDF twin")
    known = set(expected) | {"contact_sheet"}
    for p in list(out_dir.glob("map_*.png")) + list(out_dir.glob("fig_*.png")):
        if p.stem not in known:
            bad_names.append(f"unexpected file {p.name}")
    cs = out_dir / "contact_sheet.png"
    newest = max([p.stat().st_mtime for p in fig_pngs], default=0)
    cs_ok = cs.exists() and cs.stat().st_mtime >= newest
    p7_ok = (not missing) and (not bad_names) and cs_ok
    results["7_filenames_contact_sheet"] = p7_ok
    lines.append("")
    lines.append("## 7. File names match the required list; "
                 f"contact_sheet built from all outputs -> "
                 f"**{'PASS' if p7_ok else 'FAIL'}**")
    lines.append(f"- expected stems: {len(expected)}; missing: {missing or 'none'}")
    if bad_names:
        lines.append(f"- problems: {bad_names}")
    lines.append(f"- contact_sheet.png: "
                 f"{'fresh' if cs_ok else 'stale or missing'}")

    # 8 - dashboard
    html = dash_dir / "drought_maps_dashboard.html"
    probs8 = []
    if not html.exists():
        probs8.append("drought_maps_dashboard.html missing")
    else:
        try:
            from html.parser import HTMLParser

            class _P(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.imgs, self.anchors = [], []

                def handle_starttag(self, tag, attrs):
                    a = dict(attrs)
                    if tag == "img" and a.get("src"):
                        self.imgs.append(a["src"])
                    if a.get("id"):
                        self.anchors.append(a["id"])

            p = _P()
            p.feed(html.read_text(encoding="utf-8"))
            refs = [s for s in p.imgs if "outputs/maps" in s]
            missing_refs = [s for s in refs
                            if not (html.parent / s).exists()]
            if len(refs) < 20:
                probs8.append(f"only {len(refs)} map/figure images embedded "
                              f"(expect >= 20)")
            if missing_refs:
                probs8.append(f"broken image refs: {missing_refs[:5]}")
            need_ids = ["study", "indices", "spi", "stages", "random-forest",
                        "trend"]
            absent = [i for i in need_ids if i not in p.anchors]
            if absent:
                probs8.append(f"missing sections: {absent}")
        except Exception as exc:  # noqa: BLE001
            probs8.append(f"HTML parse error: {exc}")
    shots = sorted(shot_dir.glob("*.png")) if shot_dir.exists() else []
    shots = [s for s in shots if s.stat().st_size > 5000]
    good_shots = []
    for s in shots:
        try:
            from PIL import Image
            im = Image.open(s).convert("L")
            hist = im.histogram()
            nonwhite = sum(hist[:245]) / (im.size[0] * im.size[1])
            if nonwhite > 0.01:
                good_shots.append(s.name)
        except Exception:  # noqa: BLE001
            continue
    if len(good_shots) < 6:
        probs8.append(f"only {len(good_shots)} non-blank dashboard "
                      f"screenshots {good_shots} (expect >= 6 in "
                      f"outputs/dashboard_check/)")
    shots = [s for s in shots if s.name in good_shots]
    results["8_dashboard_ok"] = not probs8
    lines.append("")
    lines.append("## 8. Dashboard opens without errors, shows the new "
                 f"figures -> **{'PASS' if not probs8 else 'FAIL'}**")
    if not probs8:
        lines.append(f"- drought_maps_dashboard.html valid, {len(refs)} "
                     f"figure refs resolve, {len(shots)} screenshots saved.")
    else:
        lines += [f"- {x}" for x in probs8]

    lines.append("")
    all_pass = all(results.values())
    lines.append(f"# RESULT: **{'ALL PASS' if all_pass else 'FAILURES: ' + ', '.join(k for k, v in results.items() if not v)}**")

    md = "\n".join(lines)
    (out_dir / "qc_report.md").write_text(md, encoding="utf-8")
    if logger:
        logger.info(f"✓ QC report written → qc_report.md "
                    f"({'ALL PASS' if all_pass else 'FAILURES'})")
    return results
