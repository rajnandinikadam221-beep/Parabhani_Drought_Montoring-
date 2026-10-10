from pathlib import Path
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

def load_csv(name: str) -> pd.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file '{name}' not found. Run the pipeline first.")
    return pd.read_csv(path)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parbhani Drought Monitoring API",
        description="Real-Time & Historical Drought Monitoring for Parbhani District, Maharashtra (2016–2026)",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["Info"])
    def root():
        return {
            "message": "Parbhani Drought Monitoring API",
            "endpoints": ["/spi", "/indices/{index_name}", "/drought-stage", "/baseline", "/dashboard"]
        }

    @app.get("/spi", tags=["SPI"])
    def get_spi(
        tehsil: Optional[str] = Query(None, description="Filter by tehsil name"),
        start: Optional[str] = Query(None, description="Start month e.g. 2020-01"),
        end: Optional[str] = Query(None, description="End month e.g. 2023-12"),
        window: Optional[int] = Query(3, description="SPI window: 1, 3, 6, or 12")
    ):
        """Returns SPI values, optionally filtered by tehsil and date range."""
        df = load_csv("tehsil_drought_stages.csv")
        spi_col = f"SPI-{window}"
        if spi_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"SPI-{window} not available. Use 1, 3, 6, or 12.")
        if tehsil:
            df = df[df["tehsil"].str.lower() == tehsil.lower()]
        if start:
            df = df[df["month"] >= start]
        if end:
            df = df[df["month"] <= end]
        cols = ["tehsil", "month", spi_col, f"{spi_col}_class"] if f"{spi_col}_class" in df.columns else ["tehsil", "month", spi_col]
        return JSONResponse(content=df[cols].dropna().to_dict(orient="records"))

    @app.get("/indices/{index_name}", tags=["Indices"])
    def get_index(
        index_name: str,
        tehsil: Optional[str] = Query(None),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None)
    ):
        """Returns index values (ndvi, evi, pci, ssmi) filtered by tehsil and date range."""
        valid = {"ndvi": "NDVI", "evi": "EVI", "pci": "PCI", "ssmi": "SSMI"}
        col = valid.get(index_name.lower())
        if not col:
            raise HTTPException(status_code=400, detail=f"Unknown index '{index_name}'. Valid: {list(valid.keys())}")
        df = load_csv("tehsil_drought_stages.csv")
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"'{col}' not available in processed data.")
        if tehsil:
            df = df[df["tehsil"].str.lower() == tehsil.lower()]
        if start:
            df = df[df["month"] >= start]
        if end:
            df = df[df["month"] <= end]
        return JSONResponse(content=df[["tehsil", "month", col]].dropna().to_dict(orient="records"))

    @app.get("/drought-stage", tags=["Drought Stage"])
    def get_drought_stage(
        tehsil: Optional[str] = Query(None),
        month: Optional[str] = Query(None, description="Exact month e.g. 2019-05"),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None)
    ):
        """Returns CDI, drought stage and dashboard class per tehsil and month."""
        df = load_csv("tehsil_drought_stages.csv")
        if tehsil:
            df = df[df["tehsil"].str.lower() == tehsil.lower()]
        if month:
            df = df[df["month"] == month]
        if start:
            df = df[df["month"] >= start]
        if end:
            df = df[df["month"] <= end]
        cols = [c for c in ["tehsil", "month", "CDI", "stage", "stage_name", "dashboard_class", "tehsil_area_km2"] if c in df.columns]
        return JSONResponse(content=df[cols].dropna(subset=["CDI"]).to_dict(orient="records"))

    @app.get("/baseline", tags=["Baseline"])
    def get_baseline(
        year: Optional[int] = Query(None, description="Filter to a specific year e.g. 2022"),
        tehsil: Optional[str] = Query(None)
    ):
        """Monthly 1991-2020 climatology (district-level), plus a year's values for comparison."""
        df = load_csv("weather_spi.csv")
        df["time"] = pd.to_datetime(df["time"])
        value_cols = ["SPI-3", "SPI-6", "SSMI", "precipitation_sum", "temperature_mean"]
        base = df[(df["time"] >= "1991-01-01") & (df["time"] <= "2020-12-31")]
        clim = base.groupby(base["time"].dt.month)[value_cols].mean().round(3)
        clim = clim.reset_index().rename(columns={"time": "month"})
        out = {
            "baseline_start": "1991-01-01",
            "baseline_end": "2020-12-31",
            "monthly_climatology": json.loads(clim.to_json(orient="records")),
        }
        if year:
            sel = df[df["time"].dt.year == year]
            out["year"] = year
            out["year_values"] = json.loads(
                sel[["time"] + value_cols].round(3)
                .to_json(orient="records", date_format="iso"))
        return JSONResponse(content=out)

    @app.get("/dashboard", tags=["Dashboard"])
    def get_dashboard_json():
        """Returns the pre-generated Next.js JSON dashboard data."""
        path = REPORTS_DIR / "drought_dashboard_data.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dashboard JSON not yet generated. Run the pipeline.")
        with open(path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
