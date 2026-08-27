from datetime import UTC, datetime, timedelta

import numpy as np
from sentinelhub import CRS, BBox

from data_pipeline.sentinel_request import fetch_bbox, has_imagery

from .fire_weather import fetch_fire_weather

# evalscript output band order: B02=0, B03=1, B04=2, B08=3, B11=4, B12=5, mask=6
GREEN, RED, NIR, MASK = 1, 2, 3, 6


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    nir = bands[NIR].astype(float)
    red = bands[RED].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi[bands[MASK] == 0] = np.nan
    return ndvi


def compute_ndwi(bands: np.ndarray) -> np.ndarray:
    green = bands[GREEN].astype(float)
    nir = bands[NIR].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (green - nir) / (green + nir)
    ndwi[bands[MASK] == 0] = np.nan
    return ndwi


def mean_metric(index: np.ndarray) -> float | None:
    valid = index[np.isfinite(index)]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def prefire_window(start_date: str) -> tuple[str, str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return (
        (start - timedelta(days=15)).strftime("%Y-%m-%d"),
        (start - timedelta(days=1)).strftime("%Y-%m-%d"),
    )


def analyze_prefire(event, resolution: int = 60) -> dict:
    window = prefire_window(event.start_date)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    if not has_imagery(bbox, window):
        return {
            "available": False,
            "skipped_reason": "no_sentinel_data",
            "window": list(window),
            "fetched_on": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    bands = fetch_bbox(window, bbox, resolution=resolution)

    weather_rows = None
    try:
        weather_rows = fetch_fire_weather(
            event.centroid_lat, event.centroid_lon, *window
        )
    except Exception as exc:
        weather_index = None
        weather_error = str(exc)
    else:
        weather_index = None
        weather_error = None
        if weather_rows:
            weather_index = float(
                np.mean([row["fire_danger_index"] for row in weather_rows])
            )

    return {
        "ndvi": mean_metric(compute_ndvi(bands)),
        "ndwi": mean_metric(compute_ndwi(bands)),
        "weather_index": weather_index,
        "weather_error": weather_error,
        "window": list(window),
        "fetched_on": datetime.now(UTC).isoformat(timespec="seconds"),
    }
