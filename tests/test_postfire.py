from unittest.mock import patch

import numpy as np

from analytics import postfire
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1",
        country="greece",
        bbox=[23.4, 37.8, 24.1, 38.4],
        centroid_lat=38.1,
        centroid_lon=23.75,
        start_date="2026-07-10",
        end_date="2026-07-16",
        cluster_id=9,
    )


def _dnbr():
    dnbr = np.full((10, 10), np.nan, dtype=float)
    dnbr[0, 0] = 0.05  # unburned
    dnbr[0, 1] = 0.2  # low
    dnbr[0, 2] = 0.35  # moderate
    dnbr[0, 3] = 0.5  # high
    dnbr[0, 4] = 0.7  # very high
    return dnbr


def test_classify_severity():
    classes = postfire.classify_severity(_dnbr())
    assert set(classes) == {"unburned", "low", "moderate", "high", "very_high"}
    assert classes["unburned"] == 0.2
    assert classes["very_high"] == 0.2
    assert sum(classes.values()) == 1.0


def test_estimate_burned_area():
    area = postfire.estimate_burned_area(_dnbr(), resolution=60)
    assert area == 1.08  # 3 burned px * 3600 m2 / 10000


def test_analyze_postfire():
    with patch("analytics.postfire.process_fire_event") as mock_process:
        mock_process.return_value = {"dnbr": _dnbr()}
        assessment = postfire.analyze_postfire(_event(), resolution=60)

    assert assessment["burned_area_ha"] == 1.08
    assert assessment["severity_classes"]["low"] == 0.2
    assert "fetched_on" in assessment
