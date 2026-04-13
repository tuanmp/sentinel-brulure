import os

import numpy as np

from data_pipeline.sentinel_request import (
    ENV_ACCESS_TOKEN,
    fetch_token,
    plot_event_result,
    process_fire_event,
)


def main():
    token = fetch_token()
    assert token is not None, "Failed to fetch token"
    assert os.getenv(ENV_ACCESS_TOKEN), "Missing cached access token in environment"

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

    result = process_fire_event(event, resolution=60)
    print("Result keys:", result.keys())
    print("Pre-bands shape:", result["pre_bands"].shape)
    print("Post-bands shape:", result["post_bands"].shape)
    print("dNBR shape:", result["dnbr"].shape)
    print("Tensor shape:", result["tensor"].shape)

    plot_event_result(result, output_path="rhodes_dnbr_lowres.png")


if __name__ == "__main__":
    main()
