from unittest.mock import patch

import pandas as pd

from analytics import during
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1", country="spain",
        bbox=[-1.3, 38.9, -0.6, 39.5],
        centroid_lat=39.2, centroid_lon=-0.95,
        start_date="2026-07-15", end_date="2026-07-16",
    )


def _df():
    return pd.DataFrame({
        "latitude": [39.1, 39.2, 39.0, 42.0],
        "longitude": [-1.0, -0.9, -1.1, -3.0],
        "frp": [100.0, 200.0, 50.0, 999.0],
        "acq_date": ["2026-07-15", "2026-07-15", "2026-07-16", "2026-07-15"],
        "confidence": ["h", "n", "l", "h"],
    })


def test_fetch_daily_observations_groups_and_filters():
    with patch("analytics.during.fetch_fire_events", return_value=_df()):
        rows = during.fetch_daily_observations(_event(), days_back=5)

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-15"
    assert rows[0]["frp_mw"] == 300.0
    assert rows[0]["detection_count"] == 2
    assert rows[1]["date"] == "2026-07-16"
    assert rows[1]["detection_count"] == 1


def test_fetch_daily_observations_outside_bbox_excluded():
    with patch("analytics.during.fetch_fire_events", return_value=_df()):
        rows = during.fetch_daily_observations(_event(), days_back=5)
    all_dates = [r["date"] for r in rows]
    assert len(all_dates) == 2
