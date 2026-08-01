from unittest.mock import patch

import numpy as np
import pytest

from analytics import prefire
from analytics.event import FireEvent


def _bands():
    bands = np.zeros((7, 2, 2), dtype=np.float32)
    bands[1] = 0.5  # GREEN
    bands[2] = 0.2  # RED
    bands[3] = 0.8  # NIR
    bands[6] = 1  # mask
    return bands


def test_compute_ndvi():
    ndvi = prefire.compute_ndvi(_bands())
    assert np.allclose(ndvi, 0.6)


def test_compute_ndwi():
    ndwi = prefire.compute_ndwi(_bands())
    assert np.allclose(ndwi, -0.23076923076923078)


def test_ndvi_masks_invalid_pixels():
    bands = _bands()
    bands[6][0, 0] = 0
    ndvi = prefire.compute_ndvi(bands)
    assert np.isnan(ndvi[0, 0])
    assert ndvi[1, 1] == 0.6


def test_prefire_window():
    window = prefire.prefire_window("2026-07-12")
    assert window == ("2026-06-27", "2026-07-11")


def test_analyze_prefire():
    event = FireEvent(
        event_id="e1",
        country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55,
        centroid_lon=4.5,
        start_date="2026-07-12",
        end_date="2026-07-18",
    )
    with (
        patch("analytics.prefire.fetch_bbox", return_value=_bands()) as mock_fetch,
        patch("analytics.prefire.fetch_fire_weather") as mock_weather,
    ):
        mock_weather.return_value = [
            {"date": "2026-06-27", "fire_danger_index": 40.0},
            {"date": "2026-06-28", "fire_danger_index": 60.0},
        ]
        metrics = prefire.analyze_prefire(event, resolution=60)

    assert mock_fetch.call_args.args[0] == ("2026-06-27", "2026-07-11")
    assert metrics["ndvi"] == 0.6
    assert metrics["ndwi"] == pytest.approx(-0.23076923076923078)
    assert metrics["weather_index"] == 50.0
    assert "fetched_on" in metrics
