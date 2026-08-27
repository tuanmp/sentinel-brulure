from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from analytics import during
from analytics.event import FireEvent


def _event(start="2026-07-15", end="2026-07-16"):
    return FireEvent(
        event_id="e1",
        country="spain",
        bbox=[-1.3, 38.9, -0.6, 39.5],
        centroid_lat=39.2,
        centroid_lon=-0.95,
        start_date=start,
        end_date=end,
    )


def _df():
    return pd.DataFrame(
        {
            "latitude": [39.1, 39.2, 39.0, 42.0],
            "longitude": [-1.0, -0.9, -1.1, -3.0],
            "frp": [100.0, 200.0, 50.0, 999.0],
            "acq_date": ["2026-07-15", "2026-07-15", "2026-07-16", "2026-07-15"],
            "confidence": ["h", "n", "l", "h"],
        }
    )


def _empty_df():
    return pd.DataFrame(
        columns=["latitude", "longitude", "frp", "acq_date", "confidence"]
    )


def _outside_bbox_df():
    return pd.DataFrame(
        {
            "latitude": [42.0, 45.0],
            "longitude": [-3.0, 2.0],
            "frp": [999.0, 500.0],
            "acq_date": ["2026-07-15", "2026-07-15"],
            "confidence": ["h", "h"],
        }
    )


def _patch_firms(fetch_return):
    return patch.multiple(
        "analytics.during",
        fetch_fire_events=patch.DEFAULT,
        pick_source=patch.DEFAULT,
    ), fetch_return


def test_fetch_daily_observations_groups_and_filters():
    with (
        patch("analytics.during.fetch_fire_events", return_value=_df()),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(_event(), end_date=date(2026, 7, 16))

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-15"
    assert rows[0]["frp_mw"] == 300.0
    assert rows[0]["detection_count"] == 2
    assert rows[1]["date"] == "2026-07-16"
    assert rows[1]["detection_count"] == 1


def test_fetch_daily_observations_excludes_outside_bbox():
    with (
        patch("analytics.during.fetch_fire_events", return_value=_df()),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(_event(), end_date=date(2026, 7, 16))

    frp_values = [r["frp_mw"] for r in rows]
    assert rows[0]["frp_mw"] == 300.0
    assert frp_values == [300.0, 50.0]
    assert 999.0 not in frp_values


def test_fetch_daily_observations_empty_df_returns_empty():
    with (
        patch("analytics.during.fetch_fire_events", return_value=_empty_df()),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(_event(), end_date=date(2026, 7, 16))

    assert rows == []


def test_fetch_daily_observations_no_detections_in_bbox_returns_empty():
    with (
        patch("analytics.during.fetch_fire_events", return_value=_outside_bbox_df()),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(_event(), end_date=date(2026, 7, 16))

    assert rows == []


def test_fetch_daily_observations_bbox_growth():
    with (
        patch("analytics.during.fetch_fire_events", return_value=_df()),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(_event(), end_date=date(2026, 7, 16))

    assert rows[0]["bbox_growth_deg"] == pytest.approx(0.2)


def test_fetch_daily_observations_pages_history_in_chunks():
    event = _event(start="2026-07-10", end="2026-07-16")
    calls = []

    def fake_fetch(region, days_back, date, source):
        calls.append((region, days_back, date, source))
        if date == "2026-07-10":
            return pd.DataFrame(
                {
                    "latitude": [39.1],
                    "longitude": [-1.0],
                    "frp": [100.0],
                    "acq_date": ["2026-07-12"],
                    "confidence": ["h"],
                }
            )
        return pd.DataFrame(
            {
                "latitude": [39.2],
                "longitude": [-0.9],
                "frp": [200.0],
                "acq_date": ["2026-07-15"],
                "confidence": ["h"],
            }
        )

    with (
        patch("analytics.during.fetch_fire_events", side_effect=fake_fetch),
        patch("analytics.during.pick_source", return_value="VIIRS_SNPP_NRT"),
    ):
        rows = during.fetch_daily_observations(event, end_date=date(2026, 7, 16))

    assert len(calls) == 2
    assert calls[0] == ("spain", 5, "2026-07-10", "VIIRS_SNPP_NRT")
    assert calls[1] == ("spain", 2, "2026-07-15", "VIIRS_SNPP_NRT")
    assert [r["date"] for r in rows] == ["2026-07-12", "2026-07-15"]


def test_fetch_daily_observations_uses_archive_source_for_old_windows():
    event = _event(start="2019-09-01", end="2019-09-03")
    calls = []

    def fake_pick(start, end, source):
        calls.append((start, end, source))
        return "VIIRS_SNPP_SP"

    with (
        patch("analytics.during.fetch_fire_events", return_value=_empty_df()),
        patch("analytics.during.pick_source", side_effect=fake_pick),
    ):
        rows = during.fetch_daily_observations(event, end_date=date(2019, 9, 1))

    assert rows == []
    assert len(calls) == 1
    assert calls[0][0] == date(2019, 9, 1)
    assert calls[0][2] == "VIIRS_SNPP_NRT"
