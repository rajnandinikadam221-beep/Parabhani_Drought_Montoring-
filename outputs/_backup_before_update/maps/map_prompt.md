TASK
Rewrite the visualisation code for the Parbhani District drought maps so every map is publication-quality (university report + government drought bulletin). Work only on plotting. DO NOT change input data, the CDI formula, or drought-stage thresholds.

PHASE 1 - DISCOVER (do this first, report briefly before editing)
1. Find the scripts that produce the existing maps (likely src/processing/*) and the output folder (likely outputs/maps/). Find the tehsil/district boundary file and the data tables (CDI, SPI-3, SSMI, NDVI, EVI, PCI, VCI).
2. Open the existing PNGs to see the current state: stage maps (2019-05, 2025-09, 2026-10), index maps (ndvi, evi, pci, spi3, ssmi), SPI_NDVI_trend.png, contact_sheet.png.
3. List the files you will change. Then proceed without asking unless something is missing.

STUDY AREA
Parbhani District, Maharashtra, 9 tehsils: Jintur, Selu, Pathri, Manwath, Parbhani, Purna, Sonpeth, Palam, Gangakhed. Compute areas/statistics in EPSG:32643. Display with a lat/lon graticule.

PHASE 2 - DATA AUDIT (run before drawing; save outputs/maps/data_audit.md and print it)
Assert or report each item. Never hide a problem.
1. Tehsil areas (km2) sum to the district polygon area (tolerance 0.5%). Compare with official district area (~6,500 km2); report if the difference is >2% (currently the 9 tehsils sum to 6,308 km2, so verify and name the boundary source).
2. Stage-share % = sum(area in stage) / sum(all tehsil areas); total must equal 100.0%.
3. District means for NDVI/EVI/PCI must be AREA-WEIGHTED. Print simple vs weighted (current maps wrongly show simple means: NDVI 0.421 vs ~0.417, EVI 0.287 vs ~0.285, PCI 0.883 vs ~0.872).
4. CDI dominance: per month, correlation of CDI with SPI-3 and the share of CDI variance explained by each component. Report whether SPI-3 (one shared point series) makes stages near district-wide (tehsil CDI spread is only ~0.1-0.4).
5. Shared/constant series: list inputs identical across tehsils (SPI-3). Count SMAP L4 (~9 km) pixels per tehsil.
6. SPI-3 sanity: mean should be ~0 and SD ~1; current mean is +0.30. Report calibration period and method and flag it.
7. Partial month: detect latest date per source. 2026-10 has data only to 2026-10-06, so flag it. Its SPI-3 (about -2.1) may be biased dry because rainfall is incomplete.
8. SSMI: confirm per-tehsil standardisation (mean ~0 by construction). Report % months with SSMI <= -1 per tehsil (currently uniform 18-22%).
9. NaN counts per tehsil per variable.
10. Footer sources must match what was used (add NASA SMAP L4 SPL4SMGP).

PHASE 3 - IMPLEMENT in src/processing/maps.py
Functions: plot_stage_map(), plot_index_map(), plot_anomaly_map(), plot_spi3_summary(), plot_ssmi_panels(), plot_trend_chart(), add_north_arrow(), add_scalebar(), add_locator_inset(), add_footer(), label_tehsils(), build_table(), audit_data(). Keep all colours and fonts in one STYLE dict. Update the calling scripts to use them.

GLOBAL RULES
- Figure 13 x 8.5 in, 300 dpi. Save PNG (300 dpi) and PDF (vector) to outputs/maps/. One font (DejaVu Sans). Title 14 pt bold, subtitle 10 pt grey, labels 8 pt, table 7.5 pt.
- Use GridSpec + constrained_layout, each element in its own axes, nothing drawn over anything else:
  row 0 = title + subtitle (full width, left-aligned);
  row 1 left (~60%) = map; row 1 right (~40%) stacked = locator inset, legend or colour bar, data table, area-share bar;
  row 2 = footer below the frame.
- Remove raw UTM axes and the 1e6 offset. Light-grey graticule, degree labels (76.5°E, 19.0°N) on left and bottom only, 8 pt, no axis titles, thin black frame.
- Tehsil borders 0.8 pt #333333, district outline 1.5 pt.
- Labels: no boxes, no semi-transparent rectangles. Place at representative_point(), name bold 8 pt, value 7.5 pt on line 2, patheffects.withStroke halo. Measure each label's rendered bbox; if it does not fit in its polygon or collides, use numbered markers 1-9 + table. Use adjustText for leftovers. (Known problem: Sonpeth and Palam value labels spill outside their polygons.)
- Tables: never truncate text (current SSMI table shows "Gangakhe"); size columns from longest string; table must not touch the footer.
- Every map has: title, subtitle (period + source), north arrow (top right of map), scale bar (bottom left, computed from projected CRS, 0/10/20 km), locator inset (Maharashtra outline, Parbhani highlighted) in the right panel (not over the subtitle), footer "Data: Sentinel-2 (GEE), Open-Meteo, NASA SMAP L4 | CRS: EPSG:32643 | Prepared: <today>" 7 pt grey.
- Colour-blind-safe palettes. Vertical colour bars ~60% of map height with units, outside the map. NoData tehsils hatched light grey.

MAP TYPE 1 - STAGE MAPS (2019-05, 2025-09, latest COMPLETE month)
- Colours in fixed order: No Drought #2E8B57, Mild #FFD700, Moderate #FFA500, Severe #E8431A, Extreme #8B0000, plus hatch patterns for greyscale readability.
- Legend ALWAYS shows all 5 classes (absent ones greyed "0 km2 / 0%").
- Title "Drought Stage - <Month Year>". Put "district-wide event" in the subtitle only if all tehsils share the stage, and add "(driven by shared SPI-3 series)" if audit item 4 confirms it.
- Right panel: table Tehsil | Stage | CDI | Area km2, then a stacked horizontal bar of % area per stage totalling 100%.
- 2026-10 is partial: also generate 2026-09 (last complete month). Keep 2026-10 only as "PROVISIONAL" with "Partial month (data to 2026-10-06)" in the subtitle.

MAP TYPE 2 - NDVI, EVI, PCI
- YlGn, vmin/vmax = tehsil min/max +/- 5%. Footer note: "Colour scale stretched to observed range (not comparable between maps)".
- 3-decimal labels. Mark AREA-WEIGHTED district mean on the colour bar without overlapping tick labels.
- Subtitle "Mean of monthly values, 2016-2026 - Sentinel-2 (GEE)". Bold highest/lowest tehsil in the table. Caution line: differences between tehsils are small.

MAP TYPE 3 - SPI-3 (NO choropleth)
- 2 panels: (a) monthly SPI-3 bars 2016-2026 (blue wet, red dry, centred 0), dashed thresholds -1, -1.5, -2 labelled in the right margin, driest and wettest months annotated without crossing the title; (b) summary card: mean, min (date), max (date), months <= -1, <= -1.5, <= -2 with N and %.
- Legend must not cover bars. Hatch the incomplete last month. State the source and that gridded rainfall (IMD/CHIRPS) was not used. If gridded rainfall exists in the project, make a real spatial SPI-3 map instead.

MAP TYPE 4 - SSMI (NO mean map)
- 2 panels with one shared title row, no overlapping panel titles:
  (a) % of months with SSMI <= -1 per tehsil, YlOrRd, label "% of months";
  (b) SSMI anomaly of the latest complete month, BrBG, symmetric vmin/vmax centred at 0, label "SSMI (standard deviations)".
- Fix the colour bars: proper vertical bars ~60% of map height (currently stretched and distorted).
- Table: No. | Tehsil | % months | SSMI anomaly | Area km2, with numbered markers.
- Footnote: SMAP L4 is ~9 km, so tehsil differences are mostly shared pixels.

MAP TYPE 5 - SPI_NDVI_trend.png
- Legend outside the plot area, break NDVI line over gaps, shade the incomplete last month, label drought-period rule.

PHASE 4 - QUALITY CHECKS (code them; write outputs/maps/qc_report.md with PASS/FAIL)
1. No overlapping text bboxes (labels, titles, legend, table, footer) and nothing outside the figure (use renderer get_window_extent).
2. Each label inside its polygon or numbered markers used.
3. Table fits its axes, no truncated strings, footer clear of table.
4. Area sum = district area (0.5%); stage % = 100.0%; district means area-weighted.
5. All legend classes present; colour bars have units; PNG = figsize x 300 px; PDF is vector.
6. File names: map_stage_<yyyy-mm>.png, map_index_<name>.png (+PDF).
7. Regenerate everything, then build outputs/maps/contact_sheet.png (grid thumbnails, no clipped captions).
If any check fails, fix and re-run until all pass.

FINAL REPORT (concise)
Files changed and regenerated, all audit findings (area difference, simple vs weighted means, CDI/SPI-3 dependence, SMAP resolution, SPI mean offset, partial month), and what was intentionally left unchanged (data, CDI formula, thresholds).