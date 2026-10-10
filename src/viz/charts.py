from pathlib import Path
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

STAGE_COLORS = {
    4: "#8B0000",   # Extreme Drought
    3: "#FF4500",   # Severe Drought
    2: "#FFA500",   # Moderate Drought
    1: "#FFD700",   # Mild Drought
    0: "#2E8B57",   # No Drought
}
STAGE_NAMES = {
    4: "Extreme Drought",
    3: "Severe Drought",
    2: "Moderate Drought",
    1: "Mild Drought",
    0: "No Drought",
}

def _save(fig, path: Path, telemetry, dpi=300):
    import time
    for attempt in range(5):
        try:
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.4 * (attempt + 1))
    plt.close(fig)
    telemetry.add_output(path)


def plot_index_distributions(df_stages: pd.DataFrame, figs_dir: Path,
                             telemetry, logger):
    """Histograms + KDE of the key indices and the CDI."""
    cols = [c for c in ["SPI-3", "SSMI", "NDVI", "EVI", "PCI", "CDI"]
            if c in df_stages.columns and df_stages[c].notna().any()]
    if not cols:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    palette = {"SPI-3": "#1a6faf", "SSMI": "#8c564b", "NDVI": "#2ca02c",
               "EVI": "#98df8a", "PCI": "#17becf", "CDI": "#d62728"}
    for i, col in enumerate(cols):
        ax = axes[i]
        sns.histplot(df_stages[col].dropna(), kde=True, ax=ax, bins=30,
                     color=palette.get(col, "steelblue"), edgecolor="white")
        ax.axvline(0, color="gray", ls=":", alpha=0.7)
        if col in ("SPI-3", "SSMI", "CDI"):
            ax.axvline(-1.0, color="#FFA500", ls="--", alpha=0.8)
            ax.axvline(-2.0, color="#8B0000", ls="--", alpha=0.8)
        ax.set_title(f"{col} distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Index Distributions — Parbhani Tehsils (2016–2026)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, figs_dir / "index_distributions.png", telemetry)
    logger.info("✓ Index distributions figure saved.")

def plot_spi_ndvi_trend(df_weather: pd.DataFrame, df_s2: Optional[pd.DataFrame], figs_dir: Path, data_asof: str, telemetry, logger):
    """SPI-3 vs Sentinel-2 NDVI trend (delegates to maps.plot_trend_chart)."""
    from src.processing.maps import plot_trend_chart
    plot_trend_chart(df_weather, df_s2, figs_dir, data_asof, telemetry,
                     logger)

def plot_spi_bar_chart(df_weather: pd.DataFrame, figs_dir: Path, telemetry, logger):
    """Monthly SPI-3 bar chart with colour-coded drought categories."""
    study = df_weather[(df_weather.index >= "2016-01-01")].copy()
    colors = [STAGE_COLORS.get(4 if v <= -2 else 3 if v <= -1.5 else 2 if v <= -1 else 1 if v < 0 else 0, "#2E8B57")
              for v in study["SPI-3"].fillna(0)]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(study.index, study["SPI-3"].fillna(0), color=colors, width=25, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-1.0, color="#FFA500", lw=1, ls="--", alpha=0.8)
    ax.axhline(-1.5, color="#FF4500", lw=1, ls="--", alpha=0.8)
    ax.axhline(-2.0, color="#8B0000", lw=1, ls="--", alpha=0.8)
    ax.set_title("Monthly SPI-3 Drought Classification — Parbhani District", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("SPI-3")

    patches = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_NAMES[k]) for k in sorted(STAGE_COLORS, reverse=True)]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    _save(fig, figs_dir / "SPI3_bar_drought_classification.png", telemetry)
    logger.info("✓ SPI-3 bar chart saved.")

def plot_correlation_heatmap(df_weather: pd.DataFrame, df_s2: Optional[pd.DataFrame], figs_dir: Path, telemetry, logger):
    """Correlation heatmap of all hydro-climatic & satellite indices.

    Joins Sentinel-2 monthly means to the weather series on year-month strings
    (weather timestamps are month-END, S2 months are month-START, so a direct
    DatetimeIndex join would silently produce all-NaN columns). Correlations are
    computed pairwise on complete observations per variable pair."""
    study = df_weather[(df_weather.index >= "2016-01-01")].copy()
    study["ym"] = study.index.strftime("%Y-%m")
    cols = ["precipitation_sum", "et0_sum", "water_deficit", "temperature_mean", "SPI-1", "SPI-3", "SPI-6", "SPI-12", "SSMI"]
    if df_s2 is not None and not df_s2.empty:
        s2 = df_s2.copy()
        s2["ym"] = s2["month"].astype(str).str[:7]
        keep = [c for c in ["mean_ndvi", "mean_evi", "mean_pci"] if c in s2.columns]
        s2 = s2[["ym"] + keep].drop_duplicates(subset="ym")
        study = study.merge(s2, on="ym", how="left")
        cols += keep

    friendly = {
        "precipitation_sum": "Precipitation",
        "et0_sum": "ET₀",
        "water_deficit": "Water deficit",
        "temperature_mean": "Temperature",
        "mean_ndvi": "NDVI",
        "mean_evi": "EVI",
        "mean_pci": "PCI",
    }
    corr_df = study[cols].corr()  # pairwise complete observations
    n_pair = int(study[cols].notna().all(axis=1).sum())
    n_months = len(study)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu",
                vmin=-1, vmax=1, square=True, linewidths=0.5, mask=mask,
                annot_kws={"size": 9})
    ax.set_xticklabels([friendly.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_yticklabels([friendly.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    ax.set_title("Correlation Matrix — Hydro-Meteorological & Satellite Indicators\n"
                 f"Pairwise complete observations · {n_months} months ({study['ym'].min()} to {study['ym'].max()}) · {n_pair} months complete for all",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, figs_dir / "correlation_heatmap.png", telemetry)
    logger.info("✓ Correlation heatmap saved.")

def plot_boxplots_by_tehsil(df_stages: pd.DataFrame, figs_dir: Path, telemetry, logger):
    """Boxplots of SPI-3 and NDVI distributions per tehsil."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, color, label in [
        (axes[0], "SPI-3", "#1a6faf", "SPI-3"),
        (axes[1], "NDVI", "#2ca02c", "Sentinel-2 NDVI"),
    ]:
        if col in df_stages.columns and df_stages[col].dropna().count() > 0:
            df_stages.boxplot(column=col, by="tehsil", ax=ax, grid=False,
                              boxprops=dict(color=color), medianprops=dict(color="red", lw=2))
            ax.set_title(f"{label} Distribution by Tehsil", fontsize=11, fontweight="bold")
            ax.set_xlabel("Tehsil", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis="x", rotation=45)
    plt.suptitle("")
    fig.tight_layout()
    _save(fig, figs_dir / "tehsil_boxplots.png", telemetry)
    logger.info("✓ Tehsil boxplots saved.")

def plot_drought_stage_heatmap(df_stages: pd.DataFrame, figs_dir: Path, telemetry, logger):
    """Timeline heatmap: tehsil vs month, coloured by drought stage."""
    if df_stages.empty:
        return
    pivot = df_stages.pivot_table(index="tehsil", columns="month", values="stage", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) // 4), max(5, len(pivot.index))))
    sns.heatmap(pivot, cmap=["#2E8B57", "#FFD700", "#FFA500", "#FF4500", "#8B0000"],
                vmin=0, vmax=4, linewidths=0.3, linecolor="gray",
                cbar_kws={"label": "Drought Stage (0=None, 4=Extreme)"}, ax=ax)
    ax.set_title("Drought Stage Timeline – Parbhani Tehsils (2016–2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Tehsil")
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    _save(fig, figs_dir / "drought_stage_heatmap.png", telemetry)
    logger.info("✓ Drought stage timeline heatmap saved.")

def build_plotly_dashboard(
    df_weather: pd.DataFrame,
    df_stages: pd.DataFrame,
    df_s2: Optional[pd.DataFrame],
    gdf_4326: gpd.GeoDataFrame,
    dashboards_dir: Path,
    telemetry,
    logger
):
    """Interactive Plotly HTML: SPI/precip, drought timelines, per-tehsil index
    time series (NDVI/EVI/PCI/CDI), SSMI and baseline-vs-latest-year SPI-3."""
    study = df_weather[(df_weather.index >= "2016-01-01")].copy()
    study["month"] = study.index.strftime("%Y-%m")
    tehsils = sorted(gdf_4326["tehsil"].unique())

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "SPI-3 & Monthly Precipitation",
            "SPI-3 vs SPI-12 – Drought Severity Timeline",
            "SSMI – Standardised Soil Moisture Index",
            "CDI – Combined Drought Index by Tehsil",
            "NDVI by Tehsil (Sentinel-2)",
            "EVI by Tehsil (Sentinel-2)",
            "PCI by Tehsil (Sentinel-2)",
            "Baseline 1991–2020 vs Selected Year (SPI-3)",
        ],
        specs=[[{}, {}], [{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.06, horizontal_spacing=0.08
    )

    # R1C1: SPI-3 + precipitation
    fig.add_trace(go.Bar(x=study["month"], y=study["precipitation_sum"],
                         name="Precipitation (mm)", marker_color="rgba(54,162,235,0.5)",
                         yaxis="y2", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=study["month"], y=study["SPI-3"],
                             name="SPI-3", line=dict(color="#1a6faf", width=2.2),
                             mode="lines"), row=1, col=1)

    # R1C2: SPI-3 vs SPI-12
    fig.add_trace(go.Scatter(x=study["month"], y=study["SPI-3"],
                             name="SPI-3", line=dict(color="#1a6faf", width=2), mode="lines",
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=study["month"], y=study["SPI-12"],
                             name="SPI-12", line=dict(color="#d62728", width=2, dash="dot"),
                             mode="lines"), row=1, col=2)

    # R2C1: SSMI
    ssmi_colors = ["#2E8B57" if v >= 0 else "#FF4500" for v in study["SSMI"].fillna(0)]
    fig.add_trace(go.Bar(x=study["month"], y=study["SSMI"],
                         name="SSMI", marker_color=ssmi_colors,
                         showlegend=False), row=2, col=1)

    # R2C2: CDI per tehsil
    if not df_stages.empty and "CDI" in df_stages.columns:
        for teh in tehsils:
            t_data = df_stages[df_stages["tehsil"] == teh]
            fig.add_trace(go.Scatter(
                x=t_data["month"], y=t_data["CDI"],
                name=teh, mode="lines", line=dict(width=1.5),
                legendgroup="tehsil", showlegend=False
            ), row=2, col=2)

    # R3: NDVI / EVI / PCI per tehsil (interactive index time series)
    for col, (row_i, col_i) in zip(["NDVI", "EVI", "PCI"], [(3, 1), (3, 2), (4, 1)]):
        if df_stages.empty or col not in df_stages.columns:
            continue
        for teh in tehsils:
            t_data = df_stages[df_stages["tehsil"] == teh]
            fig.add_trace(go.Scatter(
                x=t_data["month"], y=t_data[col],
                name=f"{teh}", mode="lines", line=dict(width=1.4),
                legendgroup=col, showlegend=(col == "NDVI"),
                hovertemplate=f"{teh} {col}: %{{y:.3f}}<extra></extra>"
            ), row=row_i, col=col_i)
        fig.update_xaxes(title_text="Month", row=row_i, col=col_i)
        fig.update_yaxes(title_text=col, row=row_i, col=col_i)

    # R4C2: baseline climatology vs selected (latest complete) year
    w_all = df_weather[df_weather["SPI-3"].notna()]
    base = w_all[(w_all.index >= "1991-01-01") & (w_all.index <= "2020-12-31")]
    if not base.empty:
        clim = base.groupby(base.index.month)["SPI-3"].mean()
        years = w_all.index.year
        full_years = sorted({y for y in years.unique()
                             if int(((years == y)).sum()) >= 12})
        sel_year = full_years[-1] if full_years else int(years.max())
        sel = w_all[years == sel_year].groupby(w_all[years == sel_year].index.month)["SPI-3"].mean()
        fig.add_trace(go.Bar(x=clim.index, y=clim.values, name="Baseline 1991–2020 mean",
                             marker_color="rgba(150,150,150,0.7)"), row=4, col=2)
        fig.add_trace(go.Scatter(x=sel.index, y=sel.values, name=str(sel_year),
                                 line=dict(color="#d62728", width=2.5),
                                 mode="lines+markers"), row=4, col=2)
        fig.update_xaxes(title_text="Calendar month", row=4, col=2)
        fig.update_yaxes(title_text="SPI-3", row=4, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Real-Time & Historical Drought Monitoring Dashboard — Parbhani District, Maharashtra (2016–2026)</b>",
            font=dict(size=17)
        ),
        template="plotly_dark",
        height=1600,
        barmode="stack",
        hovermode="x unified",
        legend=dict(x=1.01, y=0.95, font=dict(size=9)),
        margin=dict(l=60, r=180, t=80, b=60)
    )

    out_html = dashboards_dir / "interactive_drought_dashboard.html"
    fig.write_html(str(out_html))
    telemetry.add_output(out_html)
    logger.info(f"✓ Interactive Plotly dashboard saved → {out_html}")


def build_plotly_timeline_dashboard(df_stages: pd.DataFrame, dashboards_dir: Path,
                                    telemetry, logger):
    """Plotly drought-stage timeline heatmap (tehsil x month) + stage distribution."""
    if df_stages.empty or "stage" not in df_stages.columns:
        return
    pivot = df_stages.pivot_table(index="tehsil", columns="month",
                                  values="stage", aggfunc="mean")
    stage_names = ["No Drought", "Mild Drought", "Moderate Drought",
                   "Severe Drought", "Extreme Drought"]
    colorscale = [[s / 4, STAGE_COLORS[s]] for s in sorted(STAGE_COLORS)]

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=["Drought Stage Timeline (tehsil x month)",
                        "Monthly Drought Stage Distribution (tehsil count)"],
        specs=[[{"type": "heatmap"}], [{}]],
    )
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=colorscale, zmin=0, zmax=4, showscale=True,
        colorbar=dict(title="Stage", tickvals=list(range(5)),
                      ticktext=stage_names),
        hovertemplate="%{y} %{x}<br>stage %{z}<extra></extra>"
    ), row=1, col=1)

    counts = df_stages.groupby(["month", "stage"]).size().unstack(fill_value=0)
    for stage in sorted(counts.columns, reverse=True):
        fig.add_trace(go.Bar(
            x=counts.index, y=counts[stage],
            name=STAGE_NAMES.get(stage, f"Stage {stage}"),
            marker_color=STAGE_COLORS.get(stage, "#999"),
            legendgroup="stage", showlegend=True
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>Drought Stage Timeline — Parbhani Tehsils (2016–2026)</b>",
                   font=dict(size=16)),
        template="plotly_dark", height=900, barmode="stack",
        margin=dict(l=60, r=60, t=70, b=50)
    )
    out_html = dashboards_dir / "drought_stage_timeline.html"
    fig.write_html(str(out_html))
    telemetry.add_output(out_html)
    logger.info(f"✓ Plotly stage timeline saved → {out_html}")

def generate_all_visualizations(
    df_weather: pd.DataFrame,
    df_stages: pd.DataFrame,
    df_s2: Optional[pd.DataFrame],
    gdf_4326: gpd.GeoDataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False
):
    """Orchestrates all matplotlib/seaborn and plotly visualizations."""
    project_root = Path(__file__).resolve().parent.parent.parent
    figs_dir = project_root / "outputs" / "figures"
    maps_dir = project_root / "outputs" / "maps"
    dashboards_dir = project_root / "dashboards"

    for d in [figs_dir, maps_dir, dashboards_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sentinel_needed = df_s2 is not None and not df_s2.empty

    required = [
        figs_dir / "SPI_NDVI_trend.png",
        figs_dir / "SPI3_bar_drought_classification.png",
        figs_dir / "correlation_heatmap.png",
        figs_dir / "tehsil_boxplots.png",
        figs_dir / "drought_stage_heatmap.png",
        figs_dir / "index_distributions.png",
        maps_dir / "map_index_ndvi.png",
        maps_dir / "map_index_evi.png",
        maps_dir / "map_index_pci.png",
        maps_dir / "map_index_ssmi.png",
        maps_dir / "map_index_spi3.png",
        maps_dir / "map_stage_2019-05.png",
        maps_dir / "map_stage_2025-09.png",
        maps_dir / "map_stage_2017-01.png",
        maps_dir / "map_stage_2026-09.png",
        maps_dir / "map_stage_2026-10.png",
        maps_dir / "map_rf_drought_classification.png",
        maps_dir / "map_ssmi_category.png",
        maps_dir / "fig_study_area.png",
        maps_dir / "fig_methodology_flowchart.png",
        maps_dir / "fig_multi_index_assessment.png",
        maps_dir / "fig_rf_confusion_matrix.png",
        maps_dir / "fig_rf_feature_importance.png",
        maps_dir / "fig_rf_features.png",
        maps_dir / "fig_model_validation.png",
        maps_dir / "fig_index_distribution.png",
        maps_dir / "fig_tehsil_boxplots.png",
        maps_dir / "fig_rainfall_distribution.png",
        maps_dir / "fig_spi3_classification.png",
        maps_dir / "fig_drought_trend.png",
        maps_dir / "contact_sheet.png",
        maps_dir / "data_audit.md",
        maps_dir / "qc_report.md",
        dashboards_dir / "interactive_drought_dashboard.html",
        dashboards_dir / "drought_stage_timeline.html",
        dashboards_dir / "drought_maps_dashboard.html",
    ]
    all_exist = (all(p.exists() for p in required)
                 and len(list(maps_dir.glob("map_stage_*.png"))) >= 5
                 and len(list(maps_dir.glob("map_stage_*.pdf"))) >= 5
                 and len(list(Path(project_root / "outputs" /
                                   "dashboard_check").glob("*.png"))) >= 6)
    if all_exist and not force:
        logger.info("[CACHE] All visualizations exist and are current. "
                    "Skipping regeneration (use --force to override).")
        for p in list(figs_dir.glob("*.png")) + list(maps_dir.glob("*.png")):
            telemetry.add_output(p)
        telemetry.add_output(dashboards_dir / "interactive_drought_dashboard.html")
        telemetry.add_output(dashboards_dir / "drought_stage_timeline.html")
        telemetry.add_output(dashboards_dir / "drought_maps_dashboard.html")
        for p in Path(project_root / "outputs" / "dashboard_check").glob("*.png"):
            telemetry.add_output(p)
        return

    logger.info("Generating all visualizations and dashboards...")
    plot_spi_ndvi_trend(df_weather, df_s2, figs_dir,
                        config["dates"]["study_end"], telemetry, logger)
    plot_spi_bar_chart(df_weather, figs_dir, telemetry, logger)
    plot_correlation_heatmap(df_weather, df_s2, figs_dir, telemetry, logger)

    if not df_stages.empty:
        plot_drought_stage_heatmap(df_stages, figs_dir, telemetry, logger)
        plot_boxplots_by_tehsil(df_stages, figs_dir, telemetry, logger)
        plot_index_distributions(df_stages, figs_dir, telemetry, logger)
        from src.processing.maps import (build_contact_sheet,
                                         render_all_maps, write_qc_report)
        result = render_all_maps(df_stages, df_weather, gdf_4326, maps_dir,
                                 config["dates"]["study_end"], telemetry,
                                 logger, config, df_s2)
        html = build_maps_dashboard(dashboards_dir, maps_dir, telemetry,
                                    logger)
        capture_dashboard_screenshots(html,
                                      maps_dir.parent / "dashboard_check",
                                      logger)
        build_contact_sheet(maps_dir, telemetry, logger)
        write_qc_report(result, maps_dir, gdf_4326, df_stages, config,
                        logger)

    build_plotly_dashboard(df_weather, df_stages, df_s2, gdf_4326, dashboards_dir, telemetry, logger)
    build_plotly_timeline_dashboard(df_stages, dashboards_dir, telemetry, logger)
    logger.info("All visualizations completed.")



# ---------------------------------------------------------------------------
# Phase 6 - publication dashboard (new figures) + headless screenshots
# ---------------------------------------------------------------------------
DASHBOARD_SECTIONS = [
    ("study", "Study area & method",
     ["fig_study_area", "fig_methodology_flowchart"]),
    ("indices", "Indices",
     ["map_index_ndvi", "map_index_evi", "map_index_pci",
      "fig_multi_index_assessment", "fig_tehsil_boxplots",
      "map_baseline_ndvi", "map_anomaly_ndvi",
      "map_baseline_evi", "map_anomaly_evi",
      "map_baseline_pci", "map_anomaly_pci"]),
    ("spi", "SPI / SSMI / Rainfall",
     ["map_index_spi3", "fig_spi3_classification", "map_index_ssmi",
      "map_ssmi_category", "fig_rainfall_distribution",
      "map_baseline_spi3", "map_anomaly_spi3",
      "map_baseline_ssmi", "map_anomaly_ssmi"]),
    ("stages", "Drought stages & events", None),   # filled from disk
    ("random-forest", "Random Forest",
     ["map_rf_drought_classification", "fig_rf_confusion_matrix",
      "fig_rf_feature_importance", "fig_rf_features",
      "fig_model_validation", "map_baseline_cdi", "map_anomaly_cdi"]),
    ("trend", "Trend & distribution",
     ["fig_drought_trend", "fig_index_distribution"]),
]


def build_maps_dashboard(dashboards_dir: Path, maps_dir: Optional[Path] = None,
                         telemetry=None, logger=None) -> Path:
    """Phase 6: single-page dashboard with section navigation, embedding the
    regenerated publication figures (PNG with PDF links), the full 5-class
    stage legend, the partial-month warning and the data caveats."""
    dashboards_dir = Path(dashboards_dir)
    root = dashboards_dir.parent
    maps_dir = Path(maps_dir) if maps_dir else root / "outputs" / "maps"
    out = dashboards_dir / "drought_maps_dashboard.html"

    stage_maps = sorted(p.stem for p in maps_dir.glob("map_stage_*.png"))
    sections = []
    for sid, title, stems in DASHBOARD_SECTIONS:
        stems = list(stems) if stems else stage_maps
        have = [s for s in stems if (maps_dir / f"{s}.png").exists()]
        missing = [s for s in stems if s not in have]
        sections.append((sid, title, have, missing))

    def fig_html(stem: str) -> str:
        label = stem.replace("map_", "").replace("fig_", "").replace("_", " ")
        return (
            f'<figure><a href="../outputs/maps/{stem}.pdf" target="_blank">'
            f'<img loading="lazy" src="../outputs/maps/{stem}.png" '
            f'alt="{stem}"></a>'
            f'<figcaption><span>{label}</span>'
            f'<a class="pdf" href="../outputs/maps/{stem}.pdf" '
            f'target="_blank">PDF</a></figcaption></figure>')

    nav = "".join(f'<a href="#{sid}">{title}</a>'
                  for sid, title, _, _ in sections)
    body = []
    for sid, title, have, missing in sections:
        body.append(f'<section id="{sid}"><div class="secmark"></div>'
                    f'<h2>{title}</h2>')
        if sid == "stages":
            body.append(
                '<div class="banner warn">2026-10 is '
                '<strong>PROVISIONAL</strong> \u2014 partial month (data to '
                '2026-10-06). Headline month = latest COMPLETE month '
                '(2026-09). "district-wide event" notes appear only in the '
                'subtitle when all 9 tehsils share a stage (driven by the '
                'shared single-point SPI-3 series).</div>')
        if sid in ("spi",):
            body.append(
                '<div class="banner info">SPI-3 and rainfall come from a '
                '<strong>single Open-Meteo point</strong> \u2014 identical '
                'for all 9 tehsils, so no spatial SPI/rainfall choropleth '
                'exists (no gridded IMD/CHIRPS in this project).</div>')
        if sid in ("spi", "indices"):
            body.append(
                '<div class="banner info">SMAP L4 SPL4SMGP is an '
                '<strong>\u2248 11 km grid</strong> (GEE scale=11000 m) \u2014 '
                'tehsil-level SSMI differences are mostly shared pixels.</div>')
        body.append('<div class="grid">' + "".join(fig_html(s)
                                                   for s in have) + '</div>')
        if missing:
            body.append(f'<p class="missing">NOT AVAILABLE: {missing}</p>')
        body.append('</section>')

    chips = "".join(
        f'<span class="chip"><i style="background:{c}"></i>{n}</span>'
        for c, n in (("#2E8B57", "No Drought"), ("#FFD700", "Mild"),
                     ("#FFA500", "Moderate"), ("#E8431A", "Severe"),
                     ("#8B0000", "Extreme")))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parbhani Drought Bulletin &mdash; Maps &amp; Figures (2016&ndash;2026)</title>
<style>
  :root {{ --ink:#1a1a1a; --grey:#555; --line:#d9d9d9; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif; color:var(--ink);
         background:#fafafa; }}
  header {{ position:sticky; top:0; z-index:10; background:#fff;
            border-bottom:2px solid var(--line); padding:10px 18px; }}
  header h1 {{ margin:0 0 4px; font-size:19px; }}
  header .sub {{ font-size:12px; color:var(--grey); margin-bottom:6px; }}
  nav {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:6px; }}
  nav a {{ font-size:13px; text-decoration:none; color:#1a6faf;
          font-weight:600; }}
  nav a:hover {{ text-decoration:underline; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:6px;
             font-size:11px; }}
  .chip i {{ display:inline-block; width:11px; height:11px;
             margin-right:4px; border:1px solid #333; border-radius:2px; }}
  .banner {{ font-size:13px; padding:9px 12px; border-radius:5px;
             margin:10px 0; line-height:1.45; }}
  .banner.warn {{ background:#fff3e0; border:1px solid #e8930c; }}
  .banner.info {{ background:#eef4fb; border:1px solid #7fa8d0; }}
  main {{ padding:0 18px 30px; max-width:1750px; margin:0 auto; }}
  section {{ padding-top:14px; }}
  section h2 {{ font-size:17px; border-bottom:1px solid var(--line);
                padding-bottom:5px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,
           minmax(560px,1fr)); gap:16px; }}
  figure {{ margin:0; background:#fff; border:1px solid var(--line);
            border-radius:6px; padding:8px; }}
  figure img {{ width:100%; height:auto; display:block;
                border-radius:3px; }}
  figcaption {{ display:flex; justify-content:space-between; font-size:12px;
                padding:6px 2px 0; color:var(--grey); }}
  figcaption .pdf {{ color:#1a6faf; text-decoration:none; font-weight:600; }}
  .missing {{ color:#a33; font-size:12px; }}
  .secmark {{ height:6px; background:#151515; margin:10px 0 6px; }}
  footer {{ border-top:2px solid var(--line); background:#fff;
            padding:12px 18px; font-size:12px; color:var(--grey); }}
  @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Parbhani District Drought Monitoring &mdash; Maps &amp; Figures</h1>
  <div class="sub">Maharashtra, India &middot; 9 tehsils &middot;
    2016-01 to 2026-10 (2026-10 provisional) &middot; prepared
    {pd.Timestamp.now():%Y-%m-%d}</div>
  <div class="legend">{chips}</div>
  <nav>{nav}</nav>
</header>
<main>
<div class="banner warn"><strong>Partial month:</strong> 2026-10 has data
 only to 2026-10-06 (SPI-3 \u2248 -2.1 may be biased dry) &mdash; rendered as
 PROVISIONAL; use 2026-09 as the last complete month.</div>
<div class="banner info"><strong>Data caveats:</strong> SPI-3/rainfall are a
 single Open-Meteo point (identical for all tehsils) &middot; SMAP L4 soil
 moisture \u2248 11 km grid &middot; CDI = 0.35&middot;SPI-3 + 0.25&middot;SSMI
 + 0.25&middot;VCI + 0.10&middot;EVI + 0.05&middot;PCI
 (thresholds -2.0 / -1.5 / -1.0 / 0.0, unchanged).</div>
{''.join(body)}
</main>
<footer>Data: Sentinel-2 (GEE), Open-Meteo, NASA SMAP L4 SPL4SMGP |
Boundaries: Parbhani.shp (project) | CRS: EPSG:32643 | Figures: outputs/maps
(PNG + vector PDF) | Prepared: {pd.Timestamp.now():%Y-%m-%d}</footer>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    if telemetry is not None:
        telemetry.add_output(out)
    if logger:
        logger.info(f"✓ Maps dashboard saved → {out.name} "
                    f"({sum(len(h) for _, _, h, _ in sections)} figures, "
                    f"{len(sections)} sections)")
    return out


CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def capture_dashboard_screenshots(html_path: Path, shot_dir: Path,
                                  logger=None, per_section: int = 6) -> int:
    """Phase 6.4: open the dashboard headless (Chrome/Edge), capture the full
    page in one paint (URL fragments render blank in headless Chrome) and
    crop one screenshot per section using the .secmark bars in the page."""
    import subprocess
    from PIL import Image
    html_path, shot_dir = Path(html_path), Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)
    browser = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if browser is None:
        import shutil
        for cmd in ("chrome", "msedge"):
            if shutil.which(cmd):
                browser = shutil.which(cmd)
                break
    if browser is None:
        if logger:
            logger.warning("[DASH-SHOT] no Chrome/Edge found - screenshots "
                           "NOT AVAILABLE")
        return 0

    full = shot_dir / "_full_page.png"
    url = html_path.resolve().as_uri()
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--hide-scrollbars", "--allow-file-access-from-files",
           "--window-size=1600,16000", "--virtual-time-budget=45000",
           f"--screenshot={full}", url]
    try:
        subprocess.run(cmd, timeout=120, capture_output=True)
    except Exception:
        cmd[1] = "--headless"
        try:
            subprocess.run(cmd, timeout=120, capture_output=True)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning(f"[DASH-SHOT] capture failed: {exc}")
            return 0
    if not full.exists() or full.stat().st_size < 5000:
        if logger:
            logger.warning("[DASH-SHOT] full-page screenshot missing/empty")
        return 0

    img = Image.open(full).convert("RGB")
    w, h = img.size
    px = img.load()
    # section marker = full-width dark bar (6 px tall) - find its rows
    marker_rows = []
    for y in range(h):
        dark = sum(1 for x in range(120, w - 120, 8) if sum(px[x, y]) < 240)
        if dark > ((w - 240) // 8) * 0.95:
            marker_rows.append(y)
    groups = []
    for y in marker_rows:
        if groups and y - groups[-1][-1] <= 2:
            groups[-1].append(y)
        else:
            groups.append([y])
    starts = [g[0] for g in groups if 4 <= len(g) <= 12]
    # content bottom (last row with any non-white pixel)
    content_bottom = h
    for y in range(h - 1, -1, -1):
        if any(sum(px[x, y]) < 720 for x in range(0, w, 16)):
            content_bottom = y + 1
            break

    anchors = [s[0] for s in DASHBOARD_SECTIONS][:per_section]
    n = 0
    for i, anchor in enumerate(anchors):
        y0 = starts[i] if i < len(starts) else int(h * i / len(anchors))
        y1 = (starts[i + 1] if i + 1 < len(starts) else content_bottom)
        y1 = min(y1, y0 + 1500, content_bottom)
        if y1 - y0 < 200:
            y1 = min(content_bottom, y0 + 200)
        crop = img.crop((0, y0, w, y1))
        out = shot_dir / f"{i + 1:02d}_{anchor}.png"
        crop.save(out)
        if out.exists() and out.stat().st_size > 5000:
            n += 1
    full.unlink(missing_ok=True)
    if logger:
        logger.info(f"[DASH-SHOT] {n}/{len(anchors)} section screenshots "
                    f"saved to {shot_dir} "
                    f"(markers found: {len(starts)})")
    return n
