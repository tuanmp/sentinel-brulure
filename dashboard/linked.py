"""Linked-pan/zoom Plotly view of the before/during/after satellite progression.

All panels share matched x/y axes, so zooming or panning on any image moves the
rest to the same geographic segment.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.postfire import SEVERITY_CLASSES
from analytics.prefire import compute_ndvi
from dashboard.imagery import SEVERITY_COLORS, delta_ndvi, severity_labels
from data_pipeline.image_utils import to_rgb
from data_pipeline.sentinel_utils import compute_nbr

SEVERITY_NAMES = [cls for cls, _, _ in SEVERITY_CLASSES]

NDVI_SCALE = ("YlGn", -0.3, 0.9)
DELTA_SCALE = ("RdYlGn", -0.5, 0.5)
DNBR_SCALE = ("RdYlGn_r", -0.2, 0.8)

# Before/during panels each span 2 grid columns so they render ~2x wider than
# the compact after row. None cells create no axes, so nothing needs hiding.
_BEFORE_DURING_SPEC = [{"colspan": 2}, None, {"colspan": 2}, None]


def _colorbar(title: str):
    return dict(title=title, thickness=8, len=0.65, xpad=2)


def rgb_trace(bands: np.ndarray, event):
    min_lon, min_lat, max_lon, max_lat = event.bbox
    rgb = to_rgb(bands).transpose(1, 2, 0)
    h, w = rgb.shape[:2]
    return go.Image(
        z=(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        x0=min_lon,
        dx=(max_lon - min_lon) / w,
        y0=max_lat,
        dy=-(max_lat - min_lat) / h,
        hoverinfo="skip",
    )


def index_trace(values: np.ndarray, colorscale, vmin, vmax, title: str, event):
    min_lon, min_lat, max_lon, max_lat = event.bbox
    h, w = values.shape
    x = np.linspace(min_lon, max_lon, w)
    y = np.linspace(max_lat, min_lat, h)
    return go.Heatmap(
        z=values,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=_colorbar(title),
        hovertemplate="lon %{x:.3f}, lat %{y:.3f}<br>%{z:.3f}<extra></extra>",
    )


def severity_trace(dnbr: np.ndarray, event):
    min_lon, min_lat, max_lon, max_lat = event.bbox
    labels = severity_labels(dnbr).astype(float)
    labels[labels < 0] = np.nan
    h, w = labels.shape
    x = np.linspace(min_lon, max_lon, w)
    y = np.linspace(max_lat, min_lat, h)
    colorscale = [
        ((i + 0.5) / len(SEVERITY_NAMES), color)
        for i, color in enumerate(SEVERITY_COLORS)
    ]
    return go.Heatmap(
        z=labels,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=-0.5,
        zmax=len(SEVERITY_NAMES) - 0.5,
        colorbar={
            **_colorbar("severity"),
            "tickvals": list(range(len(SEVERITY_NAMES))),
            "ticktext": SEVERITY_NAMES,
        },
        hovertemplate="lon %{x:.3f}, lat %{y:.3f}<extra></extra>",
    )


def build_linked_figure(
    event,
    pre_bands: np.ndarray,
    dur_bands: np.ndarray,
    post_bands: np.ndarray,
    during_date: str,
    ndvi_mode: str = "actual",
    resolution: int = 60,
):
    """Linked multi-panel figure: before/during/after with shared pan/zoom axes."""
    if ndvi_mode not in ("actual", "delta"):
        raise ValueError(f"unknown ndvi_mode {ndvi_mode!r}")

    dur_cs, dur_vmin, dur_vmax = DELTA_SCALE if ndvi_mode == "delta" else NDVI_SCALE
    dur_title = "ΔNDVI" if ndvi_mode == "delta" else "NDVI"
    dur_values = (
        delta_ndvi(pre_bands, dur_bands)
        if ndvi_mode == "delta"
        else compute_ndvi(dur_bands)
    )

    dnbr = compute_nbr(pre_bands) - compute_nbr(post_bands)

    if ndvi_mode == "delta":
        specs = [_BEFORE_DURING_SPEC, _BEFORE_DURING_SPEC, [{}, {}, {}, {}]]
        after_titles = [
            "After — RGB",
            "After — dNBR",
            "After — severity",
            "After — ΔNDVI",
        ]
    else:
        specs = [
            _BEFORE_DURING_SPEC,
            _BEFORE_DURING_SPEC,
            [{}, {}, {"colspan": 2}, None],
        ]
        after_titles = ["After — RGB", "After — dNBR", "After — severity", ""]

    fig = make_subplots(
        rows=3,
        cols=4,
        specs=specs,
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
        subplot_titles=[
            "Before — RGB",
            "",
            "Before — NDVI",
            "",
            "During — RGB",
            "",
            f"During — {dur_title} ({during_date})",
            "",
            *after_titles,
        ],
    )

    # before (RGB spans cols 1-2, NDVI spans cols 3-4)
    fig.add_trace(rgb_trace(pre_bands, event), row=1, col=1)
    fig.add_trace(
        index_trace(compute_ndvi(pre_bands), *NDVI_SCALE, "NDVI", event), row=1, col=3
    )
    # during
    fig.add_trace(rgb_trace(dur_bands, event), row=2, col=1)
    fig.add_trace(
        index_trace(dur_values, dur_cs, dur_vmin, dur_vmax, dur_title, event),
        row=2,
        col=3,
    )
    # after
    fig.add_trace(rgb_trace(post_bands, event), row=3, col=1)
    fig.add_trace(index_trace(dnbr, *DNBR_SCALE, "dNBR", event), row=3, col=2)
    fig.add_trace(severity_trace(dnbr, event), row=3, col=3)
    if ndvi_mode == "delta":
        fig.add_trace(
            index_trace(
                delta_ndvi(pre_bands, post_bands), *DELTA_SCALE, "ΔNDVI", event
            ),
            row=3,
            col=4,
        )

    min_lon, min_lat, max_lon, max_lat = event.bbox
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    fig.update_xaxes(range=[min_lon, max_lon])
    fig.update_yaxes(range=[min_lat, max_lat])

    fig.update_layout(
        height=800,
        margin=dict(r=120),
        title=f"{event.country} — cluster {event.cluster_id} (resolution {resolution} m)",
        modebar=dict(remove=["lasso2d", "select2d"]),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig
