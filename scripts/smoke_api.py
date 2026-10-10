import json
import subprocess
import sys
import time
import urllib.request

ROOT = r"D:\Drought_Monitouring Project"
proc = subprocess.Popen(
    [sys.executable, "src/main.py", "--api"],
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urls = [
    "/",
    "/spi?tehsil=Parbhani&start=2019-01&end=2019-12&window=3",
    "/spi?window=6&start=2020-01&end=2020-06",
    "/indices/ndvi?tehsil=Jintur&start=2024-01&end=2024-06",
    "/indices/ssmi?start=2025-01&end=2025-03",
    "/drought-stage?month=2019-05",
    "/drought-stage?tehsil=Selu&start=2025-01&end=2025-12",
    "/baseline?year=2019",
    "/dashboard",
]
ok = 0
try:
    for _ in range(40):
        time.sleep(0.5)
        try:
            opener.open("http://127.0.0.1:8000/", timeout=3).read()
            break
        except Exception:  # noqa: BLE001
            continue
    for u in urls:
        try:
            with opener.open("http://127.0.0.1:8000" + u, timeout=120) as r:
                body = r.read().decode("utf-8", "replace")
                status = r.status
            snippet = body[:200].replace("\n", " ")
            print(f"OK {status} {u} -> {snippet}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {u} :: {exc}")
finally:
    proc.kill()
    proc.wait()
print(f"\n{ok}/{len(urls)} endpoints OK")
