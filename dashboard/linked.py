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

_BEFORE_DURING_SPEC = [
    {"colspan": 1},
    {"colspan": 1},
]


def _pixel_coordinates(event, shape):
    """Return pixel-center lon/lat coordinates for a raster."""
    min_lon, min_lat, max_lon, max_lat = event.bbox
    height, width = shape

    dx = (max_lon - min_lon) / width
    dy = (max_lat - min_lat) / height

    x = np.linspace(
        min_lon + dx / 2,
        max_lon - dx / 2,
        width,
    )
    y = np.linspace(
        max_lat - dy / 2,
        min_lat + dy / 2,
        height,
    )

    return x, y


def rgb_trace(bands, event):
    """Create an RGB image trace."""
    rgb = to_rgb(bands).transpose(1, 2, 0)

    min_lon, min_lat, max_lon, max_lat = event.bbox
    height, width = rgb.shape[:2]

    return go.Image(
        z=(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        x0=min_lon,
        dx=(max_lon - min_lon) / width,
        y0=max_lat,
        dy=-(max_lat - min_lat) / height,
        hoverinfo="skip",
    )


def index_trace(
    values,
    colorscale,
    vmin,
    vmax,
    title,
    event,
):
    """Create an index heatmap with its own independent colorbar."""
    x, y = _pixel_coordinates(event, values.shape)

    return go.Heatmap(
        z=values,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title=dict(text=title),
            thickness=10,
            len=0.5,
            lenmode="fraction",
            xref="paper",
            yref="paper",
        ),
        hovertemplate=(
            "lon %{x:.3f}, "
            "lat %{y:.3f}<br>"
            f"{title} %{{z:.3f}}"
            "<extra></extra>"
        ),
    )


def severity_trace(dnbr, event):
    """Create the categorical severity heatmap."""
    labels = severity_labels(dnbr).astype(float)
    labels[labels < 0] = np.nan

    x, y = _pixel_coordinates(event, labels.shape)

    colorscale = [
        (
            (i + 0.5) / len(SEVERITY_NAMES),
            color,
        )
        for i, color in enumerate(SEVERITY_COLORS)
    ]

    return go.Heatmap(
        z=labels,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=-0.5,
        zmax=len(SEVERITY_NAMES) - 0.5,
        colorbar=dict(
            title=dict(text="Severity"),
            tickvals=list(range(len(SEVERITY_NAMES))),
            ticktext=SEVERITY_NAMES,
            thickness=10,
            len=0.05,
            xpad=0.05,
        ),
        hovertemplate=(
            "lon %{x:.3f}, "
            "lat %{y:.3f}<br>"
            "severity %{z}<extra></extra>"
        ),
    )


def _set_colorbar_positions(fig, trace_subplot_pairs, gap=0.015):
    """
    Put each heatmap's colorbar immediately to the right of its subplot.

    trace_subplot_pairs is an iterable of:
        (trace_index, row, col)
    """
    for trace_index, row, col in trace_subplot_pairs:
        subplot = fig.get_subplot(row, col)

        # The x-axis domain is in normalized figure coordinates.
        x_domain = subplot.xaxis.domain
        y_domain = subplot.yaxis.domain

        print(x_domain, y_domain)
        print(subplot)

        fig.data[trace_index].colorbar.x = x_domain[1] + gap
        fig.data[trace_index].colorbar.len = y_domain[1] - y_domain[0]
        fig.data[trace_index].colorbar.y = np.mean(subplot.yaxis.domain)
        fig.data[trace_index].colorbar.xanchor = "left"


def build_linked_figure(
    event,
    pre_bands,
    dur_bands,
    post_bands,
    during_date,
    ndvi_mode="actual",
    resolution=60,
):
    """Build a linked before/during/after satellite comparison figure."""

    if ndvi_mode not in {"actual", "delta"}:
        raise ValueError(f"unknown ndvi_mode {ndvi_mode!r}")

    # ------------------------------------------------------------------
    # Compute derived products
    # ------------------------------------------------------------------

    pre_ndvi = compute_ndvi(pre_bands)

    if ndvi_mode == "delta":
        during_values = delta_ndvi(pre_bands, dur_bands)
        during_scale = DELTA_SCALE
        during_title = "ΔNDVI"
    else:
        during_values = compute_ndvi(dur_bands)
        during_scale = NDVI_SCALE
        during_title = "NDVI"

    dnbr = compute_nbr(pre_bands) - compute_nbr(post_bands)

    after_delta_ndvi = (
        delta_ndvi(pre_bands, post_bands)
        if ndvi_mode == "delta"
        else None
    )

    # ------------------------------------------------------------------
    # Subplot structure
    # ------------------------------------------------------------------

    if ndvi_mode == "delta":
        specs = [
            _BEFORE_DURING_SPEC,
            _BEFORE_DURING_SPEC,
            [{}, {}],
            [{}, {}]
        ]

        titles = [
            "Before — RGB",
            "Before — NDVI",
            "During — RGB",
            f"During — {during_title} ({during_date})",
            "After — RGB",
            "After — dNBR",
            "After — severity",
            "After — ΔNDVI",
        ]
    else:
        specs = [
            _BEFORE_DURING_SPEC,
            _BEFORE_DURING_SPEC,
            [{}, {}],
            [{}, {}]
        ]

        titles = [
            "Before — RGB",
            "Before — NDVI",
            "During — RGB",
            f"During — NDVI ({during_date})",
            "After — RGB",
            "After — dNBR",
            "After — severity",
            None,
        ]

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=specs,
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
        subplot_titles=titles,
    )

    # ------------------------------------------------------------------
    # Add traces
    # ------------------------------------------------------------------

    colorbar_traces = []

    # Before
    fig.add_trace(
        rgb_trace(pre_bands, event),
        row=1,
        col=1,
    )

    trace_idx = len(fig.data)
    fig.add_trace(
        index_trace(
            pre_ndvi,
            *NDVI_SCALE,
            "NDVI",
            event,
        ),
        row=1,
        col=2,
    )
    colorbar_traces.append((trace_idx, 1, 2))

    # During
    fig.add_trace(
        rgb_trace(dur_bands, event),
        row=2,
        col=1,
    )

    trace_idx = len(fig.data)
    fig.add_trace(
        index_trace(
            during_values,
            *during_scale,
            during_title,
            event,
        ),
        row=2,
        col=2,
    )
    colorbar_traces.append((trace_idx, 2, 2))

    # After
    fig.add_trace(
        rgb_trace(post_bands, event),
        row=3,
        col=1,
    )

    trace_idx = len(fig.data)
    fig.add_trace(
        index_trace(
            dnbr,
            *DNBR_SCALE,
            "dNBR",
            event,
        ),
        row=3,
        col=2,
    )
    colorbar_traces.append((trace_idx, 3, 2))

    trace_idx = len(fig.data)
    fig.add_trace(
        severity_trace(dnbr, event),
        row=4,
        col=1,
    )
    colorbar_traces.append((trace_idx, 4, 1))

    if after_delta_ndvi is not None:
        trace_idx = len(fig.data)
        fig.add_trace(
            index_trace(
                after_delta_ndvi,
                *DELTA_SCALE,
                "ΔNDVI",
                event,
            ),
            row=4,
            col=2,
        )
        colorbar_traces.append((trace_idx, 4, 2))

    # ------------------------------------------------------------------
    # Linked geographic axes
    # ------------------------------------------------------------------

    min_lon, min_lat, max_lon, max_lat = event.bbox

    fig.update_xaxes(
        matches="x",
        range=[min_lon, max_lon],
    )

    fig.update_yaxes(
        matches="y",
        range=[min_lat, max_lat],
    )

    # ------------------------------------------------------------------
    # Colorbars
    # ------------------------------------------------------------------

    _set_colorbar_positions(
        fig,
        colorbar_traces,
        gap=0.0
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.update_layout(
        height=2500,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
        title=(
            f"{event.country} — "
            f"cluster {event.cluster_id} "
            f"(resolution {resolution} m)"
        ),
        modebar=dict(
            remove=["lasso2d", "select2d"],
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    return fig