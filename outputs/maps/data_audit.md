# Data Audit - Parbhani Drought Maps
Generated: 2026-10-08 14:06 - run BEFORE drawing (never hide a problem)

## 1. Tehsil areas (km2)
- Sum of tehsil_area_km2: **6,307.49 km2**
- District polygon area (EPSG:32643 union): **6,307.49 km2** - diff 0.000% (tolerance 0.5%) -> **PASS**
- Column vs projected geometry per tehsil: max rel. dev 0.001% -> PASS
- vs official district area ~6,500 km2: 2.96% -> **WARN**
- Boundary source: `data/Parbhani/Parbhani.shp` - project-provided boundary (custom Lambert Conformal Conic, WGS 84, per .prj); tehsil polygons dissolve to the district.

## 2. Stage-share totals
- Months checked: 130; tehsils/month: 9-9 (expect 9) -> **PASS**
- max |sum(area in stage)/sum(all areas) - 100.0%| = 0.000000% -> **PASS**

## 3. District means: simple vs AREA-WEIGHTED
| index | simple mean | area-weighted | diff | maps use |
|---|---|---|---|---|
| NDVI | 0.421 | 0.417 | +0.004 | **area-weighted** |
| EVI | 0.287 | 0.285 | +0.002 | **area-weighted** |
| PCI | 0.883 | 0.872 | +0.011 | **area-weighted** |

## 4. CDI dominance
- SPI-3 identical across tehsils every month: **YES** (nunique=1)
- corr(district-mean CDI, SPI-3) over months: r = 0.871
- corr(CDI, SPI-3) pooled over all tehsil-months: r = 0.863
- Share of CDI variance explained (Cov(CDI, term)/Var, sums to 1):
  - spi (shared): 0.0%
  - ssmi: 21.4%
  - vci: 40.2%
  - evi: 25.5%
  - pci: 12.9%
- Tehsil CDI spread per month (max-min): min 0.09, mean 0.31, max 0.57 (prompt expectation ~0.1-0.4)
- Months with all 9 tehsils in ONE stage: 99/130 -> stages are near district-wide: **YES** (SPI-3 contributes 0% of spatial spread; VCI/SSMI decide between-tehsil differences)

## 5. Shared / constant series + SMAP pixels
- Inputs identical across tehsils: **SPI-3** (single Open-Meteo point; 1 distinct value per month). All other components are per-tehsil.
- SSMI: per-tehsil standardised (z-score) - see item 8.
- SMAP L4 pixel count per tehsil: 11000 m regular grid (11 km) intersected with each polygon in EPSG:32643, grid anchored on district bounds (132 cells over district). Config `soil_moisture.scale = 11000`; the prompt's '~9 km' refers to SPL4SMGP native grid - actual processing used 11 km.
| tehsil | SMAP pixels (~grid cells) | area km2 |
|---|---|---|
| Gangakhed | 12 | 655 |
| Jintur | 20 | 1,252 |
| Manwath | 12 | 495 |
| Palam | 11 | 486 |
| Parbhani | 20 | 1,143 |
| Pathri | 10 | 554 |
| Purna | 14 | 634 |
| Selu | 15 | 705 |
| Sonpeth | 10 | 384 |

## 6. SPI-3 sanity
- Study-period (2016-01 onward) mean = +0.304, SD = 1.046 (expect mean ~0, SD ~1) -> **FAIL** **FLAGGED**
- Full record mean = +0.082, SD = 1.026
- Calibration: monthly Gamma fit, baseline 1991-01-01 to 2020-12-31 (config `dates`); method per outputs/reports/summary.md.
- FINDING: study-period mean +0.30 differs from 0 (prompt cites ~+0.30) - the study decade is wetter than the 1991-2020 baseline (or fit offset). **Flagged, NOT corrected** - data/CDI unchanged.

## 7. Partial month / latest date per source
- Weather (Open-Meteo) last row: 2026-10-31
- Sentinel-2 last month with data: 2026-10
- SMAP SSMI last month with data: 2026-10
- Stage table last month: 2026-10 -> **complete**

## 8. SSMI standardisation
- Per-tehsil long-term mean (must be ~0 by construction):
| tehsil | mean | std | % months SSMI <= -1 |
|---|---|---|---|
| Gangakhed | -0.0000 | 1.004 | 20.0% |
| Jintur | +0.0000 | 1.004 | 17.7% |
| Manwath | +0.0000 | 1.004 | 20.8% |
| Palam | +0.0000 | 1.004 | 20.0% |
| Parbhani | +0.0000 | 1.004 | 20.8% |
| Pathri | -0.0000 | 1.004 | 21.5% |
| Purna | -0.0000 | 1.004 | 19.2% |
| Selu | -0.0000 | 1.004 | 20.8% |
| Sonpeth | -0.0000 | 1.004 | 20.0% |
- % months <= -1 spans 18-22% (uniform 18-22% as expected) -> **PASS**

## 9. NaN counts per tehsil per variable
| tehsil | SPI-3 | SSMI | NDVI | EVI | PCI | VCI_z | CDI | stage |
|---|---|---|---|---|---|---|---|---|
| Gangakhed | 0 | 0 | 12 | 12 | 12 | 12 | 0 | 0 |
| Jintur | 0 | 0 | 6 | 6 | 6 | 6 | 0 | 0 |
| Manwath | 0 | 0 | 14 | 14 | 14 | 14 | 0 | 0 |
| Palam | 0 | 0 | 7 | 7 | 7 | 7 | 0 | 0 |
| Parbhani | 0 | 0 | 8 | 8 | 8 | 8 | 0 | 0 |
| Pathri | 0 | 0 | 14 | 14 | 14 | 14 | 0 | 0 |
| Purna | 0 | 0 | 9 | 9 | 9 | 9 | 0 | 0 |
| Selu | 0 | 0 | 13 | 13 | 13 | 13 | 0 | 0 |
| Sonpeth | 0 | 0 | 12 | 12 | 12 | 12 | 0 | 0 |
- 285 NaN rows across NDVI/EVI/PCI: cloud-masked composite months (2016-02 no scene; 2016-07/2024-07 all-NaN, accepted known gaps).

## 10. Random Forest audit (class balance, split, leakage)
- Model: `outputs/models/drought_rf_model.joblib` (loaded, never retrained in the visualisation layer).
- Features (8): SPI-1, SPI-3, SPI-6, SPI-12, SSMI, NDVI, EVI, PCI - CDI/VCI are labels, never features.
- Rows with full features: 1075 (130 months x 9 tehsils minus cloud-gap rows); split: train <= 2024 / test > 2024 (year-based) -> train 892 / test 183.
- Class balance (train, CDI stage): S0=513, S1=303, S2=55, S3=21 - **no S4 (Extreme) rows exist in the study period**, so the model never predicts class 4.
- Saved predictions/test-set file: NOT AVAILABLE (never persisted). Regenerated by loading the saved model (inference only, no retraining, no seed change); reproduced test accuracy 0.7760 matches the saved classification report exactly.
- **DATA LEAKAGE RISK (state next to the confusion matrix):** the CDI-stage label is computed FROM SPI-3, SSMI, VCI(NDVI), EVI and PCI - the same variables the RF uses as features - so high accuracy is expected and is NOT independent skill. The spi3_stage label is a direct function of SPI-3, which is itself a feature (accuracy 0.907 expected to be high).
- Feature importance method: impurity-based (mean decrease in Gini), read from the loaded model.

## 11. Sources used (must match footers)
- Satellite: Sentinel-2 SR/TOA (GEE) at 100 m, Cloud Score+ masking -> `Data: Sentinel-2 (GEE)`
- Weather: Open-Meteo archive (single point) -> `Open-Meteo`
- Soil moisture: NASA/SMAP/SPL4SMGP/008, band `sm_surface`, scale 11000 m -> `NASA SMAP L4`
- Boundaries: Parbhani.shp (project), Maharashtra FAO GAUL 2015 (GEE export)
- Footer emitted on every map: `Data: Sentinel-2 (GEE), Open-Meteo, NASA SMAP L4 | CRS: EPSG:32643 | Prepared: 2026-10-08` -> **PASS**
