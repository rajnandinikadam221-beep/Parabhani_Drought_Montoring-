"""One-off GEE capability probe: CS+ join, ERA5-Land bands, download nodata."""
import ee, json, datetime

ee.Initialize(project="dark-influence-491307-n3")

# --- 1. Cloud Score+ availability over Parbhani ---
aoi = ee.Geometry.Rectangle([76.2, 18.74, 77.2, 19.84])
cs = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
      .filterBounds(aoi).filterDate("2016-01-01", "2016-03-01"))
print("CS+ 2016-01/02 scenes:", cs.size().getInfo())
cs2 = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
       .filterBounds(aoi).filterDate("2024-01-01", "2024-02-01"))
print("CS+ 2024-01 scenes:", cs2.size().getInfo(), "bands:", cs2.first().bandNames().getInfo())

# join test
sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
      .filterDate("2024-01-01", "2024-02-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
joined = ee.Join.saveFirst("cs").apply(
    sr, cs2, ee.Filter.equals(leftField="system:index", rightField="system:index"))
print("SR scenes:", sr.size().getInfo(), "joined:", ee.ImageCollection(joined).size().getInfo())

# --- 2. ERA5-Land band names ---
for coll in ["ECMWF/ERA5_Land/Daily", "ECMWF/ERA5_Land/Monthly"]:
    try:
        c = ee.ImageCollection(coll).filterBounds(aoi).filterDate("2024-01-01", "2024-01-08")
        bands = c.first().bandNames().getInfo()
        sm = [b for b in bands if "soil" in b.lower()]
        print(f"{coll}: n={c.size().getInfo()} soil bands={sm}")
    except Exception as e:
        print(f"{coll}: ERROR {e}")

# --- 3. download URL with nodata on a 3-band image ---
img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi) \
        .filterDate("2024-01-01", "2024-02-01").median().clip(aoi)
ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
evi = img.expression("2.5*((N-R)/(N+6*R-7.5*B+10000))",
                     {"N": img.select("B8"), "R": img.select("B4"), "B": img.select("B2")}).rename("EVI")
stack = ee.Image.cat([ndvi, evi]).toFloat()
try:
    url = stack.getDownloadURL({"scale": 100, "crs": "EPSG:4326", "region": aoi,
                                "format": "GEO_TIFF", "nodata": -9999})
    import requests
    r = requests.get(url, timeout=120)
    print("download status:", r.status_code, "bytes:", len(r.content))
    if r.status_code == 200:
        open("outputs/_probe.tif", "wb").write(r.content)
        import rasterio, numpy as np
        with rasterio.open("outputs/_probe.tif") as ds:
            print("profile nodata:", ds.nodata, "count:", ds.count, "dtype:", ds.dtypes)
            a = ds.read(1)
            print("min/max:", float(np.nanmin(a)), float(np.nanmax(a)),
                  "frac -9999:", float((a == -9999).mean()), "frac 0:", float((a == 0).mean()))
except Exception as e:
    print("download nodata probe ERROR:", e)
