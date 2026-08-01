from datetime import UTC, datetime, timedelta

from sentinelhub import CRS, BBox

from data_pipeline.sentinel_request import fetch_bbox

from .prefire import compute_ndvi, mean_metric

RECOVERY_OFFSETS_MONTHS = [1, 3, 6, 9, 12]


def recovery_window(end_date: str, offset_months: int) -> tuple[str, str]:
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end + timedelta(days=30 * (offset_months - 1) + 5)
    stop = end + timedelta(days=30 * (offset_months - 1) + 25)
    return (start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d"))


def recovery_due(event, offset_months: int, today) -> bool:
    end = datetime.strptime(event.end_date, "%Y-%m-%d").date()
    due_date = end + timedelta(days=30 * offset_months - 15)
    return due_date <= today


def analyze_recovery(event, offset_months: int, resolution: int = 60) -> dict:
    window = recovery_window(event.end_date, offset_months)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    bands = fetch_bbox(window, bbox, resolution=resolution)

    post_ndvi = mean_metric(compute_ndvi(bands))
    baseline = event.prefire_metrics.get("ndvi")
    regrowth = None
    if baseline and post_ndvi is not None:
        regrowth = float(post_ndvi / baseline)

    return {
        "offset_months": offset_months,
        "window": list(window),
        "ndvi": post_ndvi,
        "regrowth_ratio": regrowth,
        "fetched_on": datetime.now(UTC).isoformat(timespec="seconds"),
    }
