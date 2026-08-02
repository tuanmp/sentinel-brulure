from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from analytics.event import FireEvent
from dashboard import imagery


def _event():
    return FireEvent(
        event_id="evt-1",
        country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55,
        centroid_lon=4.5,
        start_date="2026-07-12",
        end_date="2026-07-18",
        cluster_id=7,
        during_observations=[
            {"date": "2026-07-15", "frp_mw": 100.0, "detection_count": 5},
            {"date": "2026-07-16", "frp_mw": 260.0, "detection_count": 12},
            {"date": "2026-07-17", "frp_mw": 120.0, "detection_count": 6},
        ],
    )


def _bands():
    bands = np.zeros((7, 8, 8), dtype=np.float32)
    bands[1] = 0.4  # GREEN
    bands[2] = 0.2  # RED
    bands[3] = 0.7  # NIR
    bands[5] = 0.1  # SWIR2
    bands[6] = 1  # mask
    return bands


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_peak_frp_date_returns_highest_frp_day():
    assert imagery.peak_frp_date(_event()) == "2026-07-16"


def test_peak_frp_date_falls_back_to_midpoint_without_observations():
    event = _event()
    event.during_observations = []
    assert imagery.peak_frp_date(event) == "2026-07-15"


def test_phase_windows_before_uses_prefire_window():
    assert imagery.phase_windows(_event(), "before") == (
        "2026-06-27",
        "2026-07-11",
    )


def test_phase_windows_during_defaults_to_peak_date():
    start, stop = imagery.phase_windows(_event(), "during")
    assert start == "2026-07-11"
    assert stop == "2026-07-21"


def test_phase_windows_during_with_explicit_date():
    start, stop = imagery.phase_windows(_event(), "during", date="2026-07-14")
    assert start == "2026-07-09"
    assert stop == "2026-07-19"


def test_phase_windows_after_uses_post_window():
    assert imagery.phase_windows(_event(), "after") == (
        "2026-07-15",
        "2026-07-30",
    )


def test_phase_windows_unknown_phase_raises():
    with pytest.raises(ValueError):
        imagery.phase_windows(_event(), "bogus")


def test_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "CACHE_ROOT", tmp_path)
    bands = _bands()
    path = imagery.cache_path("evt-1", "before", "2026-06-27", 60)
    imagery.save_bands(path, bands)
    assert np.array_equal(imagery.load_cached_bands(path), bands)


def test_load_cached_bands_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "CACHE_ROOT", tmp_path)
    assert (
        imagery.load_cached_bands(
            imagery.cache_path("nope", "before", "2026-01-01", 60)
        )
        is None
    )


def test_fetch_phase_bands_none_when_no_imagery(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "CACHE_ROOT", tmp_path)
    with (
        patch("dashboard.imagery.has_imagery", return_value=False),
        patch("dashboard.imagery.fetch_bbox") as mock_fetch,
    ):
        result = imagery.fetch_phase_bands(_event(), "before")
    assert result is None
    mock_fetch.assert_not_called()


def test_fetch_phase_bands_fetches_caches_and_reuses(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "CACHE_ROOT", tmp_path)
    bands = _bands()
    calls = {"n": 0}

    def fake_fetch(window, bbox, resolution=60, mosaicking="leastRecent"):
        calls["n"] += 1
        assert mosaicking == "leastCC"
        return bands

    with (
        patch("dashboard.imagery.has_imagery", return_value=True),
        patch("dashboard.imagery.fetch_bbox", side_effect=fake_fetch),
    ):
        first = imagery.fetch_phase_bands(_event(), "before", resolution=60)
        second = imagery.fetch_phase_bands(_event(), "before", resolution=60)

    assert np.array_equal(first, bands)
    assert np.array_equal(second, bands)
    assert calls["n"] == 1
    assert imagery.cache_path("evt-1", "before", "2026-06-27", 60).exists()


def test_fetch_phase_bands_forwards_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(imagery, "CACHE_ROOT", tmp_path)
    with (
        patch("dashboard.imagery.has_imagery", return_value=True),
        patch("dashboard.imagery.fetch_bbox", return_value=_bands()) as mock_fetch,
    ):
        imagery.fetch_phase_bands(_event(), "after", resolution=120)
    assert mock_fetch.call_args.kwargs["resolution"] == 120
    assert mock_fetch.call_args.kwargs["mosaicking"] == "leastCC"


def test_render_before_returns_figure():
    fig = imagery.render_before(_event(), _bands())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(_panels(fig)) == 2


def test_render_during_returns_figure():
    fig = imagery.render_during(_event(), _bands(), "2026-07-16")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(_panels(fig)) == 2


def test_render_after_returns_figure():
    fig = imagery.render_after(_event(), _bands(), _bands())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(_panels(fig)) == 3


def test_renderers_handle_all_nan_data():
    blank = np.zeros((7, 8, 8), dtype=np.float32)
    blank[6] = 1
    fig = imagery.render_before(_event(), blank)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(_panels(fig)) == 2
