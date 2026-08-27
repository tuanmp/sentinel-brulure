from datetime import date, timedelta

import pandas as pd

from data_pipeline.firm_request import MAX_DAYS_BACK, fetch_fire_events, pick_source

DEFAULT_SOURCE = "VIIRS_SNPP_NRT"


def _chunk_dates(
    start_date: date, end_date: date, span: int = MAX_DAYS_BACK
) -> list[tuple[date, date]]:
    """Split [start_date, end_date] into inclusive windows of at most `span` days."""
    if start_date > end_date:
        return []
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=span - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _bbox_span(group) -> float:
    lon_span = group["longitude"].max() - group["longitude"].min()
    lat_span = group["latitude"].max() - group["latitude"].min()
    return float(lon_span + lat_span)


def _aggregate(df: pd.DataFrame, bbox: list[float]) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = bbox
    in_bbox = df[
        (df["longitude"] >= min_lon)
        & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat)
        & (df["latitude"] <= max_lat)
    ]

    rows = []
    for date_key, group in in_bbox.groupby("acq_date"):
        rows.append(
            {
                "date": str(date_key),
                "frp_mw": float(group["frp"].sum()),
                "detection_count": int(len(group)),
                "bbox_growth_deg": _bbox_span(group),
            }
        )
    return sorted(rows, key=lambda row: row["date"])


def fetch_daily_observations(
    event,
    end_date: date | None = None,
    source: str = DEFAULT_SOURCE,
) -> list[dict]:
    """Page FIRMS history for the event's country from its start date through
    `end_date` (today by default) in FIRMS-sized chunks, filter to the event
    bbox, and aggregate per-day FRP/count/span rows.

    FIRMS only allows 5-day requests, but accepts a DATE anchor to page back
    through arbitrary history. Older windows (before the NRT archive) fall back
    to the matching Standard Processing source via pick_source().
    """
    end = end_date or date.today()
    windows = _chunk_dates(date.fromisoformat(event.start_date), end)

    frames = []
    for window_start, window_end in windows:
        days = (window_end - window_start).days + 1
        src = pick_source(window_start, window_end, source)
        df = fetch_fire_events(
            region=event.country,
            days_back=days,
            date=window_start.isoformat(),
            source=src,
        )
        frames.append(df)

    if not frames:
        return []
    all_df = pd.concat(frames, ignore_index=True)
    return _aggregate(all_df, event.bbox)
