# Map QC Report

Generated: 2026-10-07 21:53

## 1. No overlapping text bboxes, nothing outside figure -> **PASS**
- 9 figures audited, 0 issues.

## 2. Each label inside its polygon or numbered markers -> **PASS**
- label modes: map_stage_2019-05.png=labels, map_stage_2025-09.png=labels, map_stage_2026-09.png=labels, map_stage_2026-10.png=labels, map_index_ndvi.png=labels, map_index_evi.png=labels, map_index_pci.png=labels, map_index_spi3.png=n/a, map_index_ssmi.png=markers/markers

## 3. Table fits axes, no truncated strings, footer clear -> **PASS**
- column widths sized from longest string (build_table); footer sits in its own GridSpec row 2 -> cannot touch the table.

## 4. Area sum = district (0.5%), stage % = 100.0%, means area-weighted -> **PASS**
- area sum vs district diff 0.000% (<=0.5%) -> PASS
- stage shares dev 0.000000% -> PASS
- district means area-weighted in plot_index_map -> PASS

## 5. Legend classes present, colour bars have units, PNG = figsize x 300 px, PDF vector -> **PASS**
- expected PNG size: 3900 x 2550 px; 9 maps checked.

## 6. File names map_stage_<yyyy-mm>.png / map_index_<name>.png (+PDF) -> **PASS**
- 9 PNGs, 9 PDFs in outputs/maps/

## 7. Everything regenerated, contact sheet built -> **PASS**
- contact_sheet.png: fresh

# RESULT: **ALL PASS**