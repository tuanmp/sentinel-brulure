from data_pipeline.firm_request import fetch_fire_events


def _bbox_span(group) -> float:
    lon_span = group["longitude"].max() - group["longitude"].min()
    lat_span = group["latitude"].max() - group["latitude"].min()
    return float(lon_span + lat_span)


def fetch_daily_observations(event, days_back: int = 10) -> list[dict]:
    """Fetch FIRMS detections for the event's country, filter to its bbox,
    and aggregate per-day FRP/count/span rows."""
    df = fetch_fire_events(region=event.country, days_back=days_back)
    min_lon, min_lat, max_lon, max_lat = event.bbox
    in_bbox = df[
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
    ]

    rows = []
    for date, group in in_bbox.groupby("acq_date"):
        rows.append({
            "date": str(date),
            "frp_mw": float(group["frp"].sum()),
            "detection_count": int(len(group)),
            "bbox_growth_deg": _bbox_span(group),
        })
    return sorted(rows, key=lambda row: row["date"])
