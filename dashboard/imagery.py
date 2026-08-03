from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from sentinelhub import CRS, BBox

from analytics.postfire import (
    SEVERITY_CLASSES,
    classify_severity,
    estimate_burned_area,
)
from analytics.prefire import compute_ndvi, prefire_window
from data_pipeline.image_utils import to_rgb
from data_pipeline.sentinel_request import (
    compute_post_window,
    fetch_bbox,
    has_imagery,
)
from data_pipeline.sentinel_utils import compute_nbr

CACHE_ROOT = Path("reports") / "imagery"
DURING_RADIUS_DAYS = 5

PHASES = ("before", "during", "after")

SEVERITY_COLORS = ["#d9d9d9", "#ffffb2", "#fd8d3c", "#e31a1c", "#800026"]


def cache_path(event_id: str, phase: str, date: str, resolution: int) -> Path:
    return CACHE_ROOT / event_id / f"{phase}_{date}_r{resolution}.npz"


def load_cached_bands(path: Path):
    try:
        return np.load(path)["bands"]
    except (OSError, KeyError, ValueError):
        return None


def save_bands(path: Path, bands: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bands=bands)


def peak_frp_date(event) -> str:
    """Date with the highest during-observation FRP; event midpoint if none."""
    obs = event.during_observations
    if obs:
        return max(obs, key=lambda o: o["frp_mw"])["date"]
    start = _date.fromisoformat(event.start_date)
    end = _date.fromisoformat(event.end_date)
    return (start + (end - start) / 2).isoformat()


def delta_ndvi(pre_bands: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Per-pixel NDVI change relative to the pre-fire reference."""
    return compute_ndvi(bands) - compute_ndvi(pre_bands)


def valid_fraction(bands: np.ndarray) -> float:
    """Fraction of pixels with valid Sentinel data (dataMask != 0)."""
    mask = bands[6]
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0))


def phase_windows(event, phase: str, date: str | None = None) -> tuple[str, str]:
    """Resolve the imagery window for a phase. During defaults to peak-FRP date."""
    if phase == "before":
        return prefire_window(event.start_date)
    if phase == "during":
        target = (
            _date.fromisoformat(date)
            if date
            else _date.fromisoformat(peak_frp_date(event))
        )
        return (
            (target - timedelta(days=DURING_RADIUS_DAYS)).isoformat(),
            (target + timedelta(days=DURING_RADIUS_DAYS)).isoformat(),
        )
    if phase == "after":
        return compute_post_window(event.start_date, event.end_date)
    raise ValueError(f"unknown phase {phase!r}")


def fetch_phase_bands(event, phase: str, date: str | None = None, resolution: int = 60):
    """Fetch and cache Sentinel-2 bands for a phase. Returns None if no imagery."""
    if phase == "during":
        key_date = date or peak_frp_date(event)
    else:
        key_date = phase_windows(event, phase)[0]
    path = cache_path(event.event_id, phase, key_date, resolution)
    cached = load_cached_bands(path)
    if cached is not None:
        return cached
    window = phase_windows(event, phase, date)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    if not has_imagery(bbox, window):
        return None
    bands = fetch_bbox(window, bbox, resolution=resolution, mosaicking="leastCC")
    save_bands(path, bands)
    return bands


def load_cached_phase(event, phase: str, resolution: int = 60, date: str | None = None):
    """Read cached bands for a phase without triggering a fetch."""
    if phase == "during":
        key_date = date or peak_frp_date(event)
    else:
        key_date = phase_windows(event, phase)[0]
    return load_cached_bands(cache_path(event.event_id, phase, key_date, resolution))


def cached_severity_areas(event, resolution: int = 60):
    """Per-class burned area (ha) computed only from cached bands, else None."""
    pre = load_cached_phase(event, "before", resolution)
    post = load_cached_phase(event, "after", resolution)
    if pre is None or post is None:
        return None
    dnbr = compute_nbr(pre) - compute_nbr(post)
    return severity_areas_ha(dnbr, resolution)


def _rgb(bands: np.ndarray) -> np.ndarray:
    return np.clip(to_rgb(bands).transpose(1, 2, 0), 0, 1)


def _extent(event):
    min_lon, min_lat, max_lon, max_lat = event.bbox
    return (min_lon, max_lon, min_lat, max_lat)


def _draw_bbox(ax, event) -> None:
    min_lon, min_lat, max_lon, max_lat = event.bbox
    ax.plot(
        [min_lon, max_lon, max_lon, min_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat, min_lat],
        color="red",
        lw=1.2,
        ls="--",
        alpha=0.8,
    )


def _style_axes(ax, event) -> None:
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    _draw_bbox(ax, event)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_before(event, bands: np.ndarray):
    pre = event.prefire_metrics or {}
    ndvi = compute_ndvi(bands)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes
    ax0.imshow(_rgb(bands), extent=_extent(event))
    ax0.set_title("Pre-fire RGB")
    im1 = ax1.imshow(ndvi, cmap="YlGn", vmin=-0.3, vmax=0.9, extent=_extent(event))
    ax1.set_title("Pre-fire NDVI")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    for ax in axes:
        _style_axes(ax, event)
    window = pre.get("window") or [None, None]
    fig.suptitle(
        f"Before — {event.country} cluster {event.cluster_id}\n"
        f"window {_fmt(window[0])} .. {_fmt(window[1])} | "
        f"mean NDVI {_fmt(pre.get('ndvi'))} | NDWI {_fmt(pre.get('ndwi'))} | "
        f"weather {_fmt(pre.get('weather_index'))} | valid {valid_fraction(bands):.0%}"
    )
    fig.tight_layout()
    return fig


def render_during(
    event, bands: np.ndarray, date: str, ndvi_mode: str = "actual", pre_bands=None
):
    ndvi = compute_ndvi(bands)
    if ndvi_mode == "delta" and pre_bands is not None:
        panel = delta_ndvi(pre_bands, bands)
        title = f"ΔNDVI vs pre-fire — {date}"
        cmap = "RdYlGn"
        vmin, vmax = -0.5, 0.5
    else:
        panel = ndvi
        title = f"During NDVI — {date}"
        cmap = "YlGn"
        vmin, vmax = -0.3, 0.9
    frp = None
    dets = None
    for obs in event.during_observations:
        if obs["date"] == date:
            frp = obs["frp_mw"]
            dets = obs["detection_count"]
            break
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes
    ax0.imshow(_rgb(bands), extent=_extent(event))
    ax0.set_title(f"During RGB — {date}")
    im1 = ax1.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(event))
    ax1.set_title(title)
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    for ax in axes:
        _style_axes(ax, event)
    frp_txt = f"FRP {frp:.0f} MW" if frp is not None else "FRP n/a"
    det_txt = f"{dets} detections" if dets is not None else "no detections"
    fig.suptitle(
        f"During — {event.country} cluster {event.cluster_id} | {date}\n"
        f"{frp_txt} | {det_txt} | valid {valid_fraction(bands):.0%}"
    )
    fig.tight_layout()
    return fig


def _severity_labels(dnbr: np.ndarray) -> np.ndarray:
    labels = np.zeros(dnbr.shape, dtype=int)
    for i, (_, low, high) in enumerate(SEVERITY_CLASSES):
        if low is None:
            mask = dnbr < high
        elif high is None:
            mask = dnbr >= low
        else:
            mask = (dnbr >= low) & (dnbr < high)
        labels[mask] = i
    labels[np.isnan(dnbr)] = -1
    return labels


def severity_areas_ha(dnbr: np.ndarray, resolution: int = 60) -> dict[str, float]:
    """Hectares per severity class."""
    labels = _severity_labels(dnbr)
    pixel_ha = (resolution * resolution) / 10000.0
    return {
        name: float(np.count_nonzero(labels == i) * pixel_ha)
        for i, (name, _, _) in enumerate(SEVERITY_CLASSES)
    }


def render_after(
    event,
    pre_bands: np.ndarray,
    post_bands: np.ndarray,
    resolution: int = 60,
    ndvi_mode: str = "actual",
):
    dnbr = compute_nbr(pre_bands) - compute_nbr(post_bands)
    labels = _severity_labels(dnbr)
    cmap = mcolors.ListedColormap(SEVERITY_COLORS)

    valid = np.isfinite(dnbr)
    stats = {}
    if valid.any():
        values = dnbr[valid]
        stats["mean"] = float(np.mean(values))
        stats["min"] = float(np.min(values))
        stats["max"] = float(np.max(values))

    n_panels = 4 if ndvi_mode == "delta" else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]
    ax0.imshow(_rgb(post_bands), extent=_extent(event))
    ax0.set_title("Post-fire RGB")
    im1 = ax1.imshow(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=0.8, extent=_extent(event))
    ax1.set_title("dNBR (burn severity)")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    masked = np.ma.masked_where(labels < 0, labels)
    im2 = ax2.imshow(
        masked,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(SEVERITY_CLASSES) - 0.5,
        extent=_extent(event),
        interpolation="nearest",
    )
    ax2.set_title("Severity class")
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, ticks=range(len(SEVERITY_CLASSES)))
    cbar.ax.set_yticklabels([cls for cls, _, _ in SEVERITY_CLASSES])
    for ax in axes:
        _style_axes(ax, event)

    if ndvi_mode == "delta":
        ax3 = axes[3]
        im3 = ax3.imshow(
            delta_ndvi(pre_bands, post_bands),
            cmap="RdYlGn",
            vmin=-0.5,
            vmax=0.5,
            extent=_extent(event),
        )
        ax3.set_title("ΔNDVI (post − pre)")
        fig.colorbar(im3, ax=ax3, fraction=0.046)
        _style_axes(ax3, event)

    burned_ha = estimate_burned_area(dnbr, resolution)
    classes = classify_severity(dnbr)
    frac = ", ".join(f"{name}:{share:.0%}" for name, share in classes.items())
    fig.suptitle(
        f"After — {event.country} cluster {event.cluster_id}\n"
        f"burned {burned_ha:.1f} ha | dNBR mean {_fmt(stats.get('mean'))} "
        f"({_fmt(stats.get('min'))}..{_fmt(stats.get('max'))}) | {frac} | "
        f"valid {valid_fraction(post_bands):.0%}"
    )
    fig.tight_layout()
    return fig
