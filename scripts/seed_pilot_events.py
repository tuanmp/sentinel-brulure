import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.event import FireEvent
from analytics.store import EventStore

# Replace these placeholders with real FIRMS events for the pilot.
# Format matches firm_request.cluster_detections() output.
PILOT_EVENTS = [
    {"country": "france", "bbox": [4.2, 44.3, 4.8, 44.8], "centroid_lat": 44.55, "centroid_lon": 4.5,
     "start_date": "2026-07-12", "end_date": "2026-07-18"},
    {"country": "france", "bbox": [-1.2, 44.3, -0.6, 44.9], "centroid_lat": 44.6, "centroid_lon": -0.9,
     "start_date": "2026-07-05", "end_date": "2026-07-11"},
    {"country": "spain", "bbox": [-1.3, 38.9, -0.6, 39.5], "centroid_lat": 39.2, "centroid_lon": -0.95,
     "start_date": "2026-07-15", "end_date": "2026-07-20"},
    {"country": "spain", "bbox": [-7.8, 42.3, -7.1, 42.9], "centroid_lat": 42.6, "centroid_lon": -7.45,
     "start_date": "2026-07-20", "end_date": "2026-07-25"},
    {"country": "italy", "bbox": [8.8, 39.7, 9.5, 40.4], "centroid_lat": 40.05, "centroid_lon": 9.15,
     "start_date": "2026-07-08", "end_date": "2026-07-14"},
    {"country": "italy", "bbox": [14.8, 37.5, 15.5, 38.2], "centroid_lat": 37.85, "centroid_lon": 15.15,
     "start_date": "2026-07-22", "end_date": "2026-07-28"},
    {"country": "greece", "bbox": [23.4, 37.8, 24.1, 38.4], "centroid_lat": 38.1, "centroid_lon": 23.75,
     "start_date": "2026-07-10", "end_date": "2026-07-16"},
    {"country": "greece", "bbox": [23.0, 38.4, 23.9, 39.1], "centroid_lat": 38.75, "centroid_lon": 23.45,
     "start_date": "2026-07-25", "end_date": "2026-07-30"},
]


def main():
    parser = argparse.ArgumentParser(description="Seed pilot fire events into the store")
    parser.add_argument("--root", default=None, help="Event store root (default: reports/events)")
    args = parser.parse_args()

    store = EventStore(args.root)
    count = 0
    for spec in PILOT_EVENTS:
        store.save_event(FireEvent(
            event_id=str(uuid4()),
            country=spec["country"],
            bbox=spec["bbox"],
            centroid_lat=spec["centroid_lat"],
            centroid_lon=spec["centroid_lon"],
            start_date=spec["start_date"],
            end_date=spec["end_date"],
        ))
        count += 1
    print(f"Seeded {count} pilot events into {store.root}")


if __name__ == "__main__":
    main()
