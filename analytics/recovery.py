from datetime import UTC, datetime, timedelta

from sentinelhub import CRS, BBox

from data_pipeline.sentinel_request import fetch_bbox, has_imagery

from .prefire import compute_ndvi, mean_metric

RECOVERY_OFFSETS_MONTHS = [1, 3, 6, 9, 12]
RECOVERY_RETRY_DAYS = 7
MAX_RECOVERY_ATTEMPTS = 3


def recovery_window(end_date: str, offset_months: int) -> tuple[str, str]:
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end + timedelta(days=30 * (offset_months - 1) + 5)
    stop = end + timedelta(days=30 * (offset_months - 1) + 25)
    return (start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d"))


def recovery_due(event, offset_months: int, today) -> bool:
    end = datetime.strptime(event.end_date, "%Y-%m-%d").date()
    due_date = end + timedelta(days=30 * offset_months - 15)
    return due_date <= today


def recovery_sample_due(event, offset_months: int, today) -> bool:
    """Whether a recovery sample is due for the offset month.

    A valid sample (ndvi present) is never redone. An invalid sample (no data,
    clouds) is retried after RECOVERY_RETRY_DAYS, and permanently given up after
    MAX_RECOVERY_ATTEMPTS so the event can still complete.
    """
    if not recovery_due(event, offset_months, today):
        return False
    attempts = [
        s for s in event.recovery_samples if s["offset_months"] == offset_months
    ]
    if not attempts:
        return True
    last = attempts[-1]
    if last.get("ndvi") is not None:
        return False
    if len(attempts) >= MAX_RECOVERY_ATTEMPTS:
        return False
    fetched = datetime.fromisoformat(last["fetched_on"]).date()
    return (today - fetched).days >= RECOVERY_RETRY_DAYS


def recovery_complete(event) -> bool:
    """True when every offset month is either validly sampled or has exhausted
    its retry attempts, so missing months can't be silently skipped."""
    for month in RECOVERY_OFFSETS_MONTHS:
        attempts = [s for s in event.recovery_samples if s["offset_months"] == month]
        if not attempts:
            return False
        last = attempts[-1]
        if last.get("ndvi") is None and len(attempts) < MAX_RECOVERY_ATTEMPTS:
            return False
    return True


def analyze_recovery(event, offset_months: int, resolution: int = 60) -> dict:
    window = recovery_window(event.end_date, offset_months)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    if not has_imagery(bbox, window):
        return {
            "offset_months": offset_months,
            "window": list(window),
            "ndvi": None,
            "regrowth_ratio": None,
            "fetched_on": datetime.now(UTC).isoformat(timespec="seconds"),
        }
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
