# SYSTEM COMPONENT AUTONOMOUS PIPELINE: print_ready_annual_gis_renderer.py

/plan
Write and execute a comprehensive Python automation script using Geopandas, Matplotlib, and Pandas to programmatically generate and export standalone, publication-quality figures for the Parbhani district dataset. 

The script must ingest data from 'tehsil_drought_stages.csv' and local shapefile paths ('data/Parbhani/Parbhani.shp') to generate individual map layouts across the entire timeline (2016 to 2026). Each spatial map must occupy its own canvas completely free of multi-panel subplots, data summary tables, or locator index frames.

Strictly enforce this automated temporal-spatial workflow:

1. TEMPORAL LOOPING & EXPORT SEPARATION:
- Establish a nested looping structure that iterates systematically through every unique year (2016 through 2026) present in the dataset.
- For each separate year, isolate or calculate representative spatial layers and export exactly one independent figure file for each of the following indicators separately:
  * Baseline Reference Map (1x map showing administrative boundaries and names only).
  * NDVI Spatial Map (1 map per year, calculated individually).
  * EVI Spatial Map (1 map per year, calculated individually).
  * PCI Spatial Map (1 map per year, calculated individually).
  * SSMI Spatial Map (1 map per year, calculated individually).
  * CDI Spatial Map (1 map per year, calculated individually).
- Do not combine different indicators or multiple years onto a single image asset.

2. CARTOGRAPHIC LAYOUT ELEMENTS & OVERLAP PROTECTION:
- Projection Alignment: Project all spatial data into EPSG:32643 (WGS 84 / UTM Zone 43N). Apply crisp polygon outlines (charcoal edge color: '#2B2B2B', line width: 0.85) to separate the 9 tehsils cleanly.
- Symbology & Gradients: Enforce desaturated, colorblind-safe sequential color ramps matching standard drought literature (e.g., green-to-brown for vegetation indices; yellow-to-dark-red for the final composite CDI severity).
- Labeling & Placement: Centrally place clean sans-serif text labels (Arial or Helvetica, 8.5pt) inside each of the 9 tehsil polygons. Enforce strict logic to ensure text never overlaps with lines or background paths.
- Map Decorations: 
  * Append a highly precise graphic Scale Bar calibrated in whole metric units (kilometers) near the bottom edge.
  * Embed a modern, minimalist North Arrow glyph cleanly outside the polygon bounds.
  * Render outer boundary ticks displaying latitude and longitude graticule points projected in WGS 1984.

3. DASHBOARD INTEGRATION & SPATIAL ALIGNMENT RULE:
- Connect the script pipeline directly to the project's interactive dashboard module (FastAPI / Streamlit engine).
- Programmatically trigger a dashboard data-state update immediately following map image export generation. 
- Ensure that the dashboard visualization engine inherits the exact same data matrices, spatial classifications, color ramps, and categorical breaks used to build the static print maps. The analytical map outputs and the active dashboard views must be completely identical.

4. CONSOLE OUTPUT & DIRECTORY PIPELINE:
- Automatically draw a single thin outer frame (Neatline) around each map file layout.
- Save every generated map using a strict, automated naming protocol that mirrors its composition (e.g., 'maps_output/2016_map_ndvi.png', 'maps_output/2020_map_cdi.png') targeting a sharp academic print quality of 300 DPI with 'bbox_inches=tight'.
- Print a clear, structured list of the verified dashboard component update logs, updated file paths, and generic academic captions to the execution window once complete (e.g., "Figure [X]: Spatial distribution of [Indicator] across the tehsils of Parbhani district for the year [Year]. Coordinates in WGS 1984 datum.").

Orchestrate and execute this entire automated visualization script safely within the local sandboxed project workspace.
