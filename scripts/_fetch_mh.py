import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Drought_Monitouring Project")
from src.gee.auth import initialize_gee

out = Path(r"D:\Drought_Monitouring Project\data\geo\maharashtra.geojson")
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("cached:", out)
    sys.exit(0)

initialize_gee("dark-influence-491307-n3")
import ee

fc = ee.FeatureCollection("FAO/GAUL/2015/level1")
mh = fc.filter(ee.Filter.eq("ADM1_NAME", "Maharashtra")).first()
geom = mh.geometry().simplify(1500)  # metres, keeps file small
geo = geom.getInfo()
out.write_text(json.dumps({"type": "FeatureCollection",
                           "features": [{"type": "Feature", "properties": {"name": "Maharashtra"},
                                         "geometry": geo}]}), encoding="utf-8")
print("saved:", out, out.stat().st_size // 1024, "KB")
