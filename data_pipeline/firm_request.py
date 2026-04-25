import io
import logging
import os

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ENV_FIRM_API_KEY = "firm_map_key"
DEFAULT_DAYS_BACK = 7
DEFAULT_MIN_CONFIDENCE = "nominal"

REGIONS = {
    "europe": {"bbox": [-25, 35, 45, 75], "name": "Europe"},
    "north_america": {"bbox": [-170, 15, -50, 75], "name": "North America"},
}

CLUSTER_SETTINGS = {
    "eps_km": 10,
    "min_samples": 5,
    "buffer_deg": 0.1,
}

FILTER_SETTINGS = {
    "min_frp_mw": 100,
    "min_detections": 20,
    "max_bbox_deg": 5.0,
}


def _get_api_key():
    key = os.getenv(ENV_FIRM_API_KEY)
    if not key:
        raise ValueError(f"Missing {ENV_FIRM_API_KEY} in environment variables")
    return key


def fetch_fire_events(
    region: str = "world",
    days_back: int = DEFAULT_DAYS_BACK,
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
    source: str = "VIIRS_SNPP_NRT",
) -> pd.DataFrame:
    """
    Fetch fire detections from FIRMS API for a given region.

    Args:
        region: One of REGIONS keys or "world" for global
        days_back: Number of days to query (max 10)
        min_confidence: "low", "nominal", or "high"
        source: FIRMS data source (default VIIRS_SNPP_NRT)

    Returns:
        DataFrame with fire detections
    """
    api_key = _get_api_key()

    if region == "world":
        bbox_str = "world"
    elif region in REGIONS:
        bbox = REGIONS[region]["bbox"]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    else:
        raise ValueError(f"Unknown region: {region}. Use: {list(REGIONS.keys())} or 'world'")

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox_str}/{days_back}"

    logging.info(f"Fetching FIRMS data for region={region}, days={days_back}")
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_levels = {"l": "low", "n": "nominal", "h": "high"}
    df["confidence_label"] = df["confidence"].map(confidence_levels)

    min_level = confidence_map.get(min_confidence, 1)
    df = df[df["confidence_label"].map(confidence_map).fillna(0) >= min_level]

    logging.info(f"Fetched {len(df)} fire detections")
    return df


def cluster_detections(
    df: pd.DataFrame,
    eps_km: float = CLUSTER_SETTINGS["eps_km"],
    min_samples: int = CLUSTER_SETTINGS["min_samples"],
) -> list[dict]:
    """
    Cluster fire detections into discrete fire events using DBSCAN.

    Args:
        df: DataFrame with latitude, longitude columns
        eps_km: Maximum distance (km) between points in same cluster
        min_samples: Minimum detections to form a cluster

    Returns:
        List of event dictionaries
    """
    if len(df) == 0:
        return []

    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_rad = eps_km / 6371

    labels = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    ).fit_predict(coords)

    df = df.copy()
    df["cluster"] = labels

    events = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue

        cluster = df[df["cluster"] == cluster_id]

        buffer = CLUSTER_SETTINGS["buffer_deg"]
        min_lon = cluster["longitude"].min() - buffer
        max_lon = cluster["longitude"].max() + buffer
        min_lat = cluster["latitude"].min() - buffer
        max_lat = cluster["latitude"].max() + buffer

        events.append({
            "cluster_id": int(cluster_id),
            "detection_count": int(len(cluster)),
            "total_frp_mw": float(cluster["frp"].sum()),
            "start_date": str(cluster["acq_date"].min()),
            "end_date": str(cluster["acq_date"].max()),
            "bbox": [float(min_lon), float(min_lat), float(max_lon), float(max_lat)],
            "centroid_lat": float(cluster["latitude"].mean()),
            "centroid_lon": float(cluster["longitude"].mean()),
        })

    events = sorted(events, key=lambda x: x["total_frp_mw"], reverse=True)
    logging.info(f"Clustered into {len(events)} fire events")
    return events


def filter_events(
    events: list[dict],
    min_frp: float = FILTER_SETTINGS["min_frp_mw"],
    min_detections: int = FILTER_SETTINGS["min_detections"],
    max_bbox_deg: float = FILTER_SETTINGS["max_bbox_deg"],
) -> list[dict]:
    """
    Filter fire events by quality criteria.

    Args:
        events: List of event dictionaries
        min_frp: Minimum total FRP (MW)
        min_detections: Minimum number of detections
        max_bbox_deg: Maximum bounding box size (degrees)

    Returns:
        Filtered list of event dictionaries
    """
    filtered = []
    for e in events:
        bbox = e["bbox"]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        if e["total_frp_mw"] < min_frp:
            continue
        if e["detection_count"] < min_detections:
            continue
        if bbox_width > max_bbox_deg or bbox_height > max_bbox_deg:
            continue

        filtered.append(e)

    logging.info(f"Filtered to {len(filtered)} qualified events")
    return filtered


def fetch_and_process(
    region: str = "world",
    days_back: int = DEFAULT_DAYS_BACK,
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
) -> list[dict]:
    """
    Full pipeline: fetch, cluster, and filter fire events.

    Args:
        region: Regional filter ("europe", "north_america", or "world")
        days_back: Days of historical data to fetch
        min_confidence: Minimum confidence level

    Returns:
        List of fire event dicts compatible with sentinel_request.process_fire_event()
    """
    df = fetch_fire_events(region=region, days_back=days_back, min_confidence=min_confidence)
    events = cluster_detections(df)
    events = filter_events(events)
    return events


def fetch_regions_summary(
    days_back: int = DEFAULT_DAYS_BACK,
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, list[dict]]:
    """
    Fetch and process fire events for multiple regions.

    Returns:
        Dict keyed by region name with event lists
    """
    results = {}
    for region, config in REGIONS.items():
        logging.info(f"Processing {config['name']}...")
        try:
            events = fetch_and_process(region, days_back, min_confidence)
            results[region] = events
            logging.info(f"  {config['name']}: {len(events)} qualified events")
        except Exception as e:
            logging.warning(f"  Failed to fetch {region}: {e}")
            results[region] = []
    return results
