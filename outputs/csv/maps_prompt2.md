TASK
Update the visualisation code for the Parbhani District drought project. (1) Rewrite the EXISTING maps/figures to publication quality (university report + government drought bulletin). (2) ADD the new maps/figures listed below. (3) Update the dashboard to use the new figures.
Work ONLY on plotting/presentation. DO NOT change input data, the CDI formula, drought-stage thresholds, the trained Random Forest model, or any original result numbers.

PHASE 1 - DISCOVER (report briefly, then continue without asking)
1. Find plotting scripts (likely src/processing/*), outputs folder (outputs/maps/), boundary file, data tables (CDI, SPI-3, SPI-6, SSMI, NDVI, EVI, PCI, VCI), the Random Forest code/model/predictions, and the dashboard code (identify the framework, e.g. Streamlit/Dash/HTML).
2. Open all existing PNG/PDF outputs and the dashboard to see the current state.
3. Safety: copy all current outputs and dashboard files to outputs/_backup_before_update/. Before editing, save a checksum/summary (row counts, means, min/max, stage counts) of every data table to outputs/maps/data_snapshot_before.json. After finishing, regenerate the summary and assert it is identical (data_snapshot_after.json). Any difference = FAIL and revert.

STUDY AREA
Parbhani District, Maharashtra, 9 tehsils: Jintur, Selu, Pathri, Manwath, Parbhani, Purna, Sonpeth, Palam, Gangakhed. Compute areas/statistics in EPSG:32643; display with a lat/lon graticule.

PHASE 2 - DATA AUDIT (before drawing; save outputs/maps/data_audit.md)
Assert or report each item. If an input is unavailable, write "NOT AVAILABLE: <reason>" and continue.
1. Tehsil areas (km2) sum to district polygon area (tolerance 0.5%); compare with official area (~6,500 km2; current sum is 6,308 km2, so verify and name the boundary source).
2. Stage-share % = area in stage / total area; totals 100.0%.
3. District means for NDVI/EVI/PCI must be AREA-WEIGHTED. Print simple vs weighted (current maps show simple means: NDVI 0.421 vs ~0.417, EVI 0.287 vs ~0.285, PCI 0.883 vs ~0.872).
4. CDI dominance: per-month correlation of CDI with SPI-3 and share of CDI variance by component. Tehsil CDI spread is only ~0.2-0.6 per month, so stages are nearly district-wide because SPI-3 is one shared point series.
5. Shared series: SPI-3 and rainfall are single-point (Open-Meteo), identical for all tehsils. SMAP L4 is ~9 km; count pixels per tehsil if the raster exists.
6. SPI-3 sanity: expected mean ~0 and SD ~1; current mean is +0.30. Report calibration period/method.
7. Partial month: 2026-10 has data only to 2026-10-06; flag it. Its SPI-3 (~-2.1) may be biased dry. Use 2026-09 as the last complete month.
8. SSMI is standardised per tehsil (mean ~0 by construction); % months <= -1 is uniform 18-22%.
9. NaN counts per tehsil per variable.
10. RF audit: report class balance, train/test split, and whether CDI-component features are also inputs to the label (data leakage risk: if CDI is built from SPI/SSMI/VCI/EVI/PCI and the same variables are RF features, high accuracy is expected, so state this next to the confusion matrix).
11. Footer sources must match what was used (Sentinel-2 GEE, Open-Meteo, NASA SMAP L4 SPL4SMGP).

GLOBAL DESIGN RULES (all figures)
- 13 x 8.5 in landscape, 300 dpi. Save PNG + vector PDF to outputs/maps/. One font (DejaVu Sans). Title 14 pt bold, subtitle 10 pt grey, labels 8 pt, table 7.5 pt. All colours/fonts in one STYLE dict.
- GridSpec + constrained_layout, each element in its own axes; nothing drawn over anything else. Row 0 = title + subtitle; Row 1 left (~60%) = map/chart; Row 1 right (~40%) stacked = locator inset, legend/colour bar, data table, area-share bar; Row 2 = footer under the frame.
- Maps: remove raw UTM axes and "1e6" offset; light-grey graticule, degree labels (76.5°E, 19.0°N) left and bottom only, 8 pt, no axis titles, thin black frame. Tehsil borders 0.8 pt #333333, district outline 1.5 pt.
- Labels: no boxes. representative_point(), bold 8 pt name, 7.5 pt value, patheffects.withStroke halo. If a label does not fit in its polygon or collides, use numbered markers 1-9 + table; adjustText for leftovers. Known problem: Sonpeth and Palam value labels spill outside polygons. Tables never truncate text (current SSMI table shows "Gangakhe").
- Every map: title, subtitle (period + source), north arrow (top right), scale bar (bottom left, from projected CRS, 0/10/20 km), locator inset in the right panel (Maharashtra outline, Parbhani highlighted), footer "Data: Sentinel-2 (GEE), Open-Meteo, NASA SMAP L4 | CRS: EPSG:32643 | Prepared: <today>" 7 pt grey, never touching the table.
- Colour-blind-safe palettes; vertical colour bars ~60% of map height with units, outside the map; NoData tehsils hatched light grey. Non-map charts follow the same fonts, title/subtitle/footer, and avoid legend overlap with data.
- Stage colours fixed order: No Drought #2E8B57, Mild #FFD700, Moderate #FFA500, Severe #E8431A, Extreme #8B0000. Legends always show all 5 classes (absent ones greyed "0 km2 / 0%").

PHASE 3 - FIGURES TO UPDATE (existing)
A. Study area map: district in Maharashtra, 9 tehsils named, graticule, scale, north arrow, tehsil areas table, data sources.
B. Methodology flow chart: inputs (Sentinel-2/GEE, Open-Meteo, SMAP) -> indices (SPI-3, SSMI, VCI, NDVI, EVI, PCI) -> CDI -> stages -> Random Forest -> validation -> outputs. Clean boxes, aligned arrows, no overlap, colour by stage of the workflow.
C. Mean NDVI per tehsil, Mean EVI per tehsil, Mean PCI per tehsil: YlGn, vmin/vmax = tehsil min/max +/- 5%, footer "Colour scale stretched to observed range (not comparable between maps)", 3-decimal labels, AREA-WEIGHTED district mean marked on the colour bar without overlapping ticks, subtitle "Mean of monthly values, 2016-2026", bold highest/lowest tehsil in table, caution that differences are small.
D. SPI-3 figure (NO choropleth): panel (a) monthly SPI-3 bars 2016-2026 (blue wet, red dry, centred 0, dashed thresholds -1/-1.5/-2 labelled in the right margin, driest and wettest months annotated clear of the title); panel (b) summary card (mean, min with date, max with date, months <= -1, <= -1.5, <= -2 with N and %). Hatch the incomplete 2026-10 bar. State single-point Open-Meteo source; if gridded rainfall (IMD/CHIRPS) exists, add a real spatial SPI-3 map.
E. SSMI figure (NO mean map): two panels with one shared title row: (a) % of months with SSMI <= -1 per tehsil, YlOrRd, "% of months"; (b) SSMI anomaly of 2026-09, BrBG, symmetric vmin/vmax centred 0, "SSMI (standard deviations)". Fix the stretched/distorted colour bars (proper vertical bars). Table: No. | Tehsil | % months | SSMI anomaly | Area km2. Footnote: SMAP L4 ~9 km, so tehsil differences are mostly shared pixels.
F. Multi-index drought assessment: one multi-panel figure (NDVI, EVI, PCI, SPI-3, SSMI, CDI) per tehsil or district with a consistent layout.
G. Drought stage maps: 2019-05 (most severe), 2025-09 (wettest), plus the mild-drought month (see new maps), and the latest COMPLETE month 2026-09. Title "Drought Stage - <Month Year>"; "district-wide event" in the subtitle only when all tehsils share the stage (add "driven by shared SPI-3 series" if audit item 4 confirms). Right panel: Tehsil | Stage | CDI | Area km2 table plus a stacked horizontal bar of % area by stage totalling 100%. Keep 2026-10 only as "PROVISIONAL - Partial month (data to 2026-10-06)".
H. Random Forest drought classification map: tehsil map of RF predicted stage (same stage colours, full legend), with a side table comparing RF stage vs CDI stage.
I. Tehsil-wise drought percentage: per tehsil, % of months in each stage 2016-2026 (stacked horizontal bars, stage colours, % labels inside bars where they fit, totals 100%).
J. RF features: feature list/groups (name, source, description) as a clean table or grouped chart.
K. Model validation: metrics table (accuracy, precision, recall, F1 per class, macro F1) and test-vs-train summary, with the leakage caveat from audit item 10.
L. Drought trend: SPI-3 and NDVI time series (SPI_NDVI_trend). Legend outside the plot area (below), break the NDVI line over gaps, shade the incomplete last month, state the drought-period shading rule, no legend/line overlap.

PHASE 4 - NEW FIGURES TO ADD
N1. Index distribution figure 2016-2026: histograms/violin plots of NDVI, EVI, PCI, SPI-3, SSMI, CDI (district and by tehsil) with mean/median markers.
N2. Historical baseline maps 2016-2026: tehsil maps of the 2016-2026 baseline mean for each index, plus anomaly maps (latest complete month minus baseline) with diverging palette centred at 0.
N3. Tehsil boxplots: boxplots of each index by tehsil 2016-2026 (same y-scale per index, sorted by median, outliers shown, n labelled).
N4. Soil-moisture drought indicator map, tehsil-wise: SSMI category map (stage-like categories from the existing SSMI thresholds in config) for the latest complete month, with table and % area by category.
N5. Drought stages 2016-2026, tehsil-wise area by stage: stacked bars of km2 per stage per tehsil (all months summed or averaged as month-km2; state which), plus % version.
N6. Three event maps with the same layout as stage maps: (i) most severe drought month, (ii) wettest month, (iii) mild-drought month. Determine the months programmatically and print the rule: most severe = lowest district-mean CDI (expected 2019-05); wettest = highest district-mean CDI (expected 2025-09); mild-drought = the month with the largest district area in the Mild class (state the month and area).
N7. Rainfall distribution figure: monthly rainfall and annual totals 2016-2026, monthly climatology boxplot, wet/dry-season split. The rainfall is a single Open-Meteo point, so DO NOT draw a tehsil choropleth; if gridded rainfall (IMD/CHIRPS) exists, draw a real spatial rainfall map and state the source.
N8. Random Forest drought-stage confusion matrix: counts and row-normalised %, stage order No Drought -> Extreme, include all 5 classes even if empty, annotated cells, colour-blind-safe sequential palette, accuracy in subtitle, and the leakage caveat.
N9. Random Forest feature importance: horizontal bars sorted descending, top 15, value labels, grouped colour by feature type; note the importance method (impurity or permutation).
N10. Monthly SPI-3 drought classification: bars/steps per month 2016-2026 coloured by SPI-3 class using the existing config thresholds, hatch partial 2026-10, class legend with counts.
RF rule: LOAD the existing trained model and saved predictions. DO NOT retrain or change random seeds. If predictions/test set are not saved, write "NOT AVAILABLE" in the audit and generate the figure only by reading the saved evaluation outputs.

PHASE 5 - CODE STRUCTURE
src/processing/maps.py (shared helpers + all plotting): plot_study_area(), plot_flowchart(), plot_index_map(), plot_spi3_summary(), plot_ssmi_panels(), plot_multi_index(), plot_stage_map(), plot_rf_map(), plot_stage_percent_by_tehsil(), plot_rf_features_table(), plot_validation(), plot_trend_chart(), plot_index_distribution(), plot_baseline_maps(), plot_tehsil_boxplots(), plot_ssmi_category_map(), plot_stage_area_by_tehsil(), plot_event_maps(), plot_rainfall_distribution(), plot_confusion_matrix(), plot_feature_importance(), plot_spi3_classification(); helpers add_north_arrow(), add_scalebar(), add_locator_inset(), add_footer(), label_tehsils(), build_table(), audit_data(). Add a single run_all_maps.py entry point.

OUTPUT NAMES (PNG + PDF in outputs/maps/)
fig_study_area, fig_methodology_flowchart, map_index_ndvi, map_index_evi, map_index_pci, map_index_spi3, map_index_ssmi, fig_multi_index_assessment, map_stage_2019-05, map_stage_2025-09, map_stage_<mild month>, map_stage_2026-09, (map_stage_2026-10 provisional), map_rf_drought_classification, fig_tehsil_drought_percentage, fig_rf_features, fig_model_validation, fig_drought_trend, fig_index_distribution, map_baseline_<index>, fig_tehsil_boxplots, map_ssmi_category, fig_stage_area_by_tehsil, fig_rainfall_distribution, fig_rf_confusion_matrix, fig_rf_feature_importance, fig_spi3_classification, contact_sheet.

PHASE 6 - DASHBOARD UPDATE
1. Replace the old images/charts in the dashboard with the new figures; add pages/tabs or sections for the new ones (Study area & method, Indices, SPI/SSMI/Rainfall, Drought stages & events, Random Forest (map, confusion matrix, importance, validation), Trend & distribution).
2. Keep the same data sources and all numbers; the dashboard may only read the regenerated files or the unchanged data. Show the stage legend with all 5 classes, the partial-month warning (2026-10), the single-point SPI-3/rainfall limitation, and the SMAP 9 km caveat on the relevant pages.
3. Clean layout: consistent header, tehsil/period selectors only if they already existed, no overlapping elements, readable on laptop and projector. Footer with sources and prepared date.
4. Run the dashboard locally and check each page loads without errors; save screenshots to outputs/dashboard_check/.

PHASE 7 - QUALITY CHECKS (code them; write outputs/maps/qc_report.md with PASS/FAIL)
1. No overlapping text bounding boxes (call fig.canvas.draw() first, then use renderer get_window_extent) and nothing outside the figure.
2. Each tehsil label inside its polygon or numbered markers used.
3. Tables fit their axes, no truncated strings, footer clear of table.
4. Area sum = district area (0.5%); stage % = 100.0%; district means area-weighted; confusion-matrix totals = number of test samples.
5. Legends show all classes; colour bars have units; PNG pixel size = figsize x 300; PDFs are vector.
6. Data snapshot before == after (no data or result changed).
7. File names match the list above; contact_sheet.png built from all outputs with no clipped captions.
8. Dashboard pages open without errors and show the new figures.
If any check fails, fix and re-run until all pass.

FINAL REPORT (concise)
Figures updated, figures added, dashboard changes, all audit findings (area difference, simple vs weighted means, CDI/SPI-3 dependence, SMAP resolution, SPI mean offset, partial month, RF leakage risk), items marked NOT AVAILABLE, and confirmation that data, CDI formula, thresholds and RF model were not changed.