from datetime import UTC, datetime

import numpy as np

from data_pipeline.sentinel_request import process_fire_event

SEVERITY_CLASSES = [
    ("unburned", None, 0.1),
    ("low", 0.1, 0.27),
    ("moderate", 0.27, 0.44),
    ("high", 0.44, 0.66),
    ("very_high", 0.66, None),
]

BURNED_THRESHOLD = 0.27  # severity >= moderate counts as burned


def classify_severity(dnbr: np.ndarray) -> dict[str, float]:
    valid = dnbr[~np.isnan(dnbr)]
    total = valid.size
    counts = {}
    for name, low, high in SEVERITY_CLASSES:
        if low is None:
            mask = valid < high
        elif high is None:
            mask = valid >= low
        else:
            mask = (valid >= low) & (valid < high)
        counts[name] = float(np.count_nonzero(mask) / total) if total else 0.0
    return counts


def estimate_burned_area(dnbr: np.ndarray, resolution: int = 60) -> float:
    burned = np.count_nonzero(dnbr[~np.isnan(dnbr)] >= BURNED_THRESHOLD)
    return float(burned * resolution * resolution / 10000.0)


def analyze_postfire(event, resolution: int = 60, use_model: bool = False) -> dict:
    event_dict = {
        "cluster_id": event.cluster_id,
        "bbox": event.bbox,
        "start_date": event.start_date,
        "end_date": event.end_date,
    }
    result = process_fire_event(event_dict, resolution=resolution, use_model=use_model)
    dnbr = result["dnbr"]

    return {
        "dnbr_mean": float(np.nanmean(dnbr)),
        "dnbr_min": float(np.nanmin(dnbr)),
        "dnbr_max": float(np.nanmax(dnbr)),
        "severity_classes": classify_severity(dnbr),
        "burned_area_ha": estimate_burned_area(dnbr, resolution),
        "fetched_on": datetime.now(UTC).isoformat(timespec="seconds"),
    }
