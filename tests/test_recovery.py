from unittest.mock import patch

import numpy as np
import pytest

from analytics import recovery
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1",
        country="italy",
        bbox=[14.8, 37.5, 15.5, 38.2],
        centroid_lat=37.85,
        centroid_lon=15.15,
        start_date="2026-07-08",
        end_date="2026-07-14",
        prefire_metrics={"ndvi": 0.6},
    )


def _bands():
    bands = np.zeros((7, 2, 2), dtype=np.float32)
    bands[2] = 0.1  # RED
    bands[3] = 0.5  # NIR
    bands[6] = 1  # mask
    return bands


def test_recovery_offsets():
    assert recovery.RECOVERY_OFFSETS_MONTHS == [1, 3, 6, 9, 12]


def test_recovery_window():
    start, stop = recovery.recovery_window("2026-07-14", 1)
    assert start == "2026-07-19"
    assert stop == "2026-08-08"


def test_analyze_recovery_computes_regrowth_ratio():
    event = _event()
    with patch("analytics.recovery.fetch_bbox", return_value=_bands()):
        sample = recovery.analyze_recovery(event, 1, resolution=60)

    # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
    assert sample["offset_months"] == 1
    assert sample["ndvi"] == pytest.approx(0.6666666666666666)
    assert sample["regrowth_ratio"] == pytest.approx(1.1111111111111112)
    assert "fetched_on" in sample


def test_analyze_recovery_no_baseline():
    event = _event()
    event.prefire_metrics = {}
    with patch("analytics.recovery.fetch_bbox", return_value=_bands()):
        sample = recovery.analyze_recovery(event, 1, resolution=60)
    assert sample["regrowth_ratio"] is None
