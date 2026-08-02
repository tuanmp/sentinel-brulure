from datetime import date
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
    with (
        patch("analytics.recovery.fetch_bbox", return_value=_bands()),
        patch("analytics.recovery.has_imagery", return_value=True),
    ):
        sample = recovery.analyze_recovery(event, 1, resolution=60)

    # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
    assert sample["offset_months"] == 1
    assert sample["ndvi"] == pytest.approx(0.6666666666666666)
    assert sample["regrowth_ratio"] == pytest.approx(1.1111111111111112)
    assert "fetched_on" in sample


def test_analyze_recovery_no_baseline():
    event = _event()
    event.prefire_metrics = {}
    with (
        patch("analytics.recovery.fetch_bbox", return_value=_bands()),
        patch("analytics.recovery.has_imagery", return_value=True),
    ):
        sample = recovery.analyze_recovery(event, 1, resolution=60)
    assert sample["regrowth_ratio"] is None


def test_analyze_recovery_returns_none_ndvi_when_no_coverage():
    event = _event()
    with (
        patch("analytics.recovery.has_imagery", return_value=False) as mock_coverage,
        patch("analytics.recovery.fetch_bbox") as mock_fetch,
    ):
        sample = recovery.analyze_recovery(event, 1, resolution=60)

    assert sample["offset_months"] == 1
    assert sample["ndvi"] is None
    assert sample["regrowth_ratio"] is None
    mock_fetch.assert_not_called()
    assert mock_coverage.call_args.args[1] == (
        "2026-07-19",
        "2026-08-08",
    )


def test_recovery_due_at_boundary():
    event = _event()
    assert recovery.recovery_due(event, 1, date(2026, 7, 29)) is True


def test_recovery_due_not_yet():
    event = _event()
    assert recovery.recovery_due(event, 1, date(2026, 7, 28)) is False


def test_recovery_due_later_offset():
    event = _event()
    assert recovery.recovery_due(event, 12, date(2027, 6, 23)) is False
    assert recovery.recovery_due(event, 12, date(2027, 6, 24)) is True
    assert recovery.recovery_due(event, 12, date(2027, 6, 25)) is True


def _sample(month, ndvi, fetched="2026-08-01T00:00:00+00:00"):
    return {"offset_months": month, "ndvi": ndvi, "fetched_on": fetched}


def test_recovery_sample_due_not_due_yet():
    event = _event()
    assert recovery.recovery_sample_due(event, 1, date(2026, 7, 28)) is False


def test_recovery_sample_due_when_no_attempts():
    event = _event()
    assert recovery.recovery_sample_due(event, 1, date(2026, 7, 30)) is True


def test_recovery_sample_due_false_when_last_valid():
    event = _event()
    event.recovery_samples = [_sample(1, 0.5)]
    assert recovery.recovery_sample_due(event, 1, date(2026, 7, 30)) is False


def test_recovery_sample_due_retries_invalid_after_cooldown():
    event = _event()
    event.recovery_samples = [_sample(1, None, fetched="2026-07-26T00:00:00+00:00")]
    assert recovery.recovery_sample_due(event, 1, date(2026, 7, 29)) is False
    event.recovery_samples = [_sample(1, None, fetched="2026-07-20T00:00:00+00:00")]
    assert recovery.recovery_sample_due(event, 1, date(2026, 7, 29)) is True


def test_recovery_sample_due_stops_after_max_attempts():
    event = _event()
    event.recovery_samples = [
        _sample(1, None) for _ in range(recovery.MAX_RECOVERY_ATTEMPTS)
    ]
    assert recovery.recovery_sample_due(event, 1, date(2026, 9, 1)) is False


def test_recovery_complete_requires_all_offsets():
    event = _event()
    event.recovery_samples = [_sample(m, 0.5) for m in recovery.RECOVERY_OFFSETS_MONTHS]
    assert recovery.recovery_complete(event) is True

    event.recovery_samples = [
        _sample(m, 0.5) for m in recovery.RECOVERY_OFFSETS_MONTHS if m != 6
    ]
    assert recovery.recovery_complete(event) is False


def test_recovery_complete_allows_exhausted_invalid_month():
    event = _event()
    event.recovery_samples = [
        _sample(m, 0.5) for m in recovery.RECOVERY_OFFSETS_MONTHS if m != 6
    ]
    event.recovery_samples += [
        _sample(6, None) for _ in range(recovery.MAX_RECOVERY_ATTEMPTS)
    ]
    assert recovery.recovery_complete(event) is True


def test_recovery_complete_false_when_invalid_not_exhausted():
    event = _event()
    event.recovery_samples = [
        _sample(m, 0.5) for m in recovery.RECOVERY_OFFSETS_MONTHS if m != 6
    ]
    event.recovery_samples.append(_sample(6, None))
    assert recovery.recovery_complete(event) is False
