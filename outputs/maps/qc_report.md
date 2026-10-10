# Map QC Report

Generated: 2026-10-08 14:09
Checks per maps_prompt2.md Phase 7 (8 checks)

## 1. No overlapping text bboxes, nothing outside figure -> **PASS**
- 38 figures audited, 0 issues.

## 2. Each label inside its polygon or numbered markers -> **PASS**
- label modes: fig_study_area.png=labels, fig_methodology_flowchart.png=n/a, fig_multi_index_assessment.png=n/a, map_stage_2019-05.png=labels, map_stage_2025-09.png=labels, map_stage_2017-01.png=labels, map_stage_2026-09.png=labels, map_stage_2026-10.png=labels, map_index_ndvi.png=labels, map_index_evi.png=labels, map_index_pci.png=labels, map_index_spi3.png=n/a, map_index_ssmi.png=markers/markers, map_rf_drought_classification.png=labels, fig_rf_features.png=n/a, fig_model_validation.png=n/a, fig_rf_confusion_matrix.png=n/a, fig_rf_feature_importance.png=n/a, fig_tehsil_drought_percentage.png=n/a, fig_index_distribution.png=n/a, fig_tehsil_boxplots.png=n/a, map_ssmi_category.png=labels, fig_stage_area_by_tehsil.png=n/a, fig_rainfall_distribution.png=n/a, fig_spi3_classification.png=n/a, map_baseline_ndvi.png=labels, map_anomaly_ndvi.png=labels, map_baseline_evi.png=labels, map_anomaly_evi.png=labels, map_baseline_pci.png=labels, map_anomaly_pci.png=labels, map_baseline_spi3.png=labels, map_anomaly_spi3.png=labels, map_baseline_ssmi.png=labels, map_anomaly_ssmi.png=labels, map_baseline_cdi.png=labels, map_anomaly_cdi.png=labels, fig_drought_trend.png=n/a

## 3. Table fits axes, no truncated strings, footer clear -> **PASS**
- column widths sized from the longest string (build_table); footer sits in its own GridSpec row -> cannot touch the table.

## 4. Area sum = district (0.5%), stage % = 100.0%, means area-weighted, CM totals = test samples -> **PASS**
- area sum vs district diff 0.000% (<=0.5%) -> PASS
- stage shares dev 0.000000% -> PASS
- district means area-weighted in plot_index_map -> PASS
- confusion-matrix total 183 == test samples 183 -> PASS

## 5. Legends show all classes, colour bars have units, PNG = figsize x 300 px, PDF vector -> **PASS**
- expected PNG size: 3900 x 2550 px; 38 figures checked.

## 6. data_snapshot_before == data_snapshot_after (no data/result changed) -> **PASS**
- 13 tables compared; identical

## 7. File names match the required list; contact_sheet built from all outputs -> **PASS**
- expected stems: 38; missing: none
- contact_sheet.png: fresh

## 8. Dashboard opens without errors, shows the new figures -> **PASS**
- drought_maps_dashboard.html valid, 36 figure refs resolve, 6 screenshots saved.

# RESULT: **ALL PASS**