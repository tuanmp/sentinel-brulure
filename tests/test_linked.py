import numpy as np
import plotly.graph_objects as go
import pytest

from analytics.event import FireEvent
from dashboard import linked


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
            {"date": "2026-07-16", "frp_mw": 260.0, "detection_count": 12},
        ],
    )


def _bands(nir=0.7):
    bands = np.zeros((7, 8, 8), dtype=np.float32)
    bands[1] = 0.4  # GREEN
    bands[2] = 0.2  # RED
    bands[3] = nir
    bands[5] = 0.1  # SWIR2
    bands[6] = 1  # mask
    return bands


def test_build_linked_figure_returns_figure():
    fig = linked.build_linked_figure(
        _event(), _bands(), _bands(), _bands(), "2026-07-16", ndvi_mode="actual"
    )
    assert isinstance(fig, go.Figure)


def test_all_axes_match_primary_for_linked_pan_zoom():
    fig = linked.build_linked_figure(
        _event(), _bands(), _bands(), _bands(), "2026-07-16", ndvi_mode="actual"
    )
    for i in range(2, 5):
        xaxis = fig.layout[f"xaxis{i}"]
        yaxis = fig.layout[f"yaxis{i}"]
        if xaxis.visible is not False:
            assert xaxis.matches == "x"
            assert yaxis.matches == "y"


def test_trace_counts_by_mode():
    base = {
        "event": _event(),
        "pre_bands": _bands(),
        "dur_bands": _bands(),
        "post_bands": _bands(),
        "during_date": "2026-07-16",
    }
    actual = linked.build_linked_figure(**base, ndvi_mode="actual")
    delta = linked.build_linked_figure(**base, ndvi_mode="delta")
    assert len(actual.data) == 7  # before(2) + during(2) + after(3)
    assert len(delta.data) == 8  # after gains ΔNDVI


def test_rgb_traces_are_images_and_indices_heatmaps():
    fig = linked.build_linked_figure(
        _event(), _bands(), _bands(), _bands(), "2026-07-16", ndvi_mode="actual"
    )
    images = [t for t in fig.data if isinstance(t, go.Image)]
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(images) == 3  # before/during/after RGB
    assert len(heatmaps) == 4  # ndvi, ndvi, dnbr, severity


def test_axis_ranges_equal_bbox_extent():
    event = _event()
    fig = linked.build_linked_figure(
        event, _bands(), _bands(), _bands(), "2026-07-16", ndvi_mode="actual"
    )
    assert fig.layout.xaxis.range == (4.2, 4.8)
    assert fig.layout.yaxis.range == (44.3, 44.8)


def test_severity_trace_masks_no_data_as_nan():
    dnbr = np.full((8, 8), np.nan)
    dnbr[:4, :4] = 0.5  # high
    trace = linked.severity_trace(dnbr, _event())
    z = np.asarray(trace.z)
    assert np.isnan(z[5, 5])
    assert z[1, 1] == 3  # high -> class index 3


def test_severity_trace_colorbar_ticks_are_class_names():
    dnbr = np.full((8, 8), np.nan)
    trace = linked.severity_trace(dnbr, _event())
    ticktext = trace.colorbar.ticktext
    assert set(ticktext) >= {"unburned", "moderate", "very_high"}


def test_index_trace_returns_heatmap_with_range():
    values = np.linspace(-0.2, 0.8, 64).reshape(8, 8)
    trace = linked.index_trace(values, "YlGn", -0.3, 0.9, "NDVI", _event())
    assert isinstance(trace, go.Heatmap)
    assert trace.zmin == -0.3
    assert trace.zmax == 0.9


def test_build_linked_figure_rejects_unknown_mode():
    with pytest.raises(ValueError):
        linked.build_linked_figure(
            _event(), _bands(), _bands(), _bands(), "2026-07-16", ndvi_mode="bogus"
        )
