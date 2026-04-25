import os

import numpy as np

from data_pipeline.firm_request import fetch_and_process
from data_pipeline.sentinel_request import (
    ENV_ACCESS_TOKEN,
    fetch_token,
    plot_event_result,
    process_fire_event,
)


def main(use_firms: bool = True, region: str = "north_america", days_back: int = 3):
    token = fetch_token()
    assert token is not None, "Failed to fetch Sentinel token"
    assert os.getenv(ENV_ACCESS_TOKEN), "Missing cached access token in environment"

    if use_firms:
        print(f"Fetching fire events from FIRMS ({region}, last {days_back} days)...")
        events = fetch_and_process(region=region, days_back=days_back)
        print(f"Found {len(events)} qualified events")

        if not events:
            print("No events found. Try different region or increase days_back.")
            return

        event = events[0]
        print(f"Processing top event: cluster_id={event['cluster_id']}")
        print(f"  detections: {event['detection_count']}, FRP: {event['total_frp_mw']:.0f}MW")
    else:
        event = {
            "cluster_id": np.int64(325),
            "detection_count": 435,
            "total_frp_mw": np.float64(10373.33),
            "start_date": "2026-04-11",
            "end_date": "2026-04-11",
            "bbox": [
                np.float64(103.38314000000001),
                np.float64(19.23771),
                np.float64(104.0),
                np.float64(20.0),
            ],
            "centroid_lat": np.float64(19.88453790804597),
            "centroid_lon": np.float64(103.8635648275862),
        }
        print("Using hardcoded event (disabled FIRMS)")

    result = process_fire_event(event, resolution=60)
    print("Result keys:", result.keys())
    print("Pre-bands shape:", result["pre_bands"].shape)
    print("Post-bands shape:", result["post_bands"].shape)
    print("dNBR shape:", result["dnbr"].shape)
    print("Tensor shape:", result["tensor"].shape)

    plot_event_result(result, output_path="firms_dnbr.png")


if __name__ == "__main__":
    main(use_firms=True, region="north_america", days_back=3)
