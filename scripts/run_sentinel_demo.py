import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import numpy as np

from data_pipeline.firm_request import fetch_and_process
from data_pipeline.sentinel_request import (
    ENV_ACCESS_TOKEN,
    fetch_token,
    plot_event_result,
    process_fire_event,
)


def main():
    parser = argparse.ArgumentParser(description='Sentinel Sat Burn Scar Detection')
    parser.add_argument('--use-model', action='store_true',
                        help='Use Prithvi model for burn scar inference')
    parser.add_argument('--region', type=str, default='north_america',
                        help='FIRMS region (default: north_america)')
    parser.add_argument('--days-back', type=int, default=3,
                        help='Days back for FIRMS query (default: 3)')
    parser.add_argument('--no-firms', action='store_true',
                        help='Use hardcoded event instead of FIRMS')
    parser.add_argument('--resolution', type=int, default=60,
                        help='Sentinel pixel resolution (default: 60)')
    args = parser.parse_args()

    token = fetch_token()
    assert token is not None, "Failed to fetch Sentinel token"
    assert os.getenv(ENV_ACCESS_TOKEN), "Missing cached access token"

    if not args.no_firms:
        print(f"Fetching fire events from FIRMS ({args.region}, last {args.days_back} days)...")
        events = fetch_and_process(region=args.region, days_back=args.days_back)
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
            "bbox": [103.38314, 19.23771, 104.0, 20.0],
            "centroid_lat": np.float64(19.88453790804597),
            "centroid_lon": np.float64(103.8635648275862),
        }
        print("Using hardcoded event")

    result = process_fire_event(event, resolution=args.resolution, use_model=args.use_model)
    print("\nResults:")
    print(f"  Pre-fire bands: {result['pre_bands'].shape}")
    print(f"  Post-fire bands: {result['post_bands'].shape}")
    print(f"  dNBR: {result['dnbr'].shape}")

    if args.use_model and result.get('burn_scar_probs') is not None:
        probs = result['burn_scar_probs']
        print(f"  Burn scar probs: min={probs.min():.3f}, max={probs.max():.3f}")
        print(f"  Mask saved to: {result.get('burn_scar_mask_path', 'N/A')}")

    plot_event_result(result, output_path="burn_scar_result.png")
    print("\nPlot saved to burn_scar_result.png")


if __name__ == "__main__":
    main()
