import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.store import EventStore
from analytics.tracker import process_event


def main():
    parser = argparse.ArgumentParser(description="Run one tracker tick over all tracked events")
    parser.add_argument("--root", default=None, help="Event store root (default: reports/events)")
    parser.add_argument("--resolution", type=int, default=60, help="Sentinel resolution (m)")
    parser.add_argument("--use-model", action="store_true", help="Run Prithvi inference for post-fire")
    parser.add_argument("--quiet-after", type=int, default=3, help="Quiet days before marking ended")
    args = parser.parse_args()

    store = EventStore(args.root)
    events = store.list_events()
    print(f"Tracking {len(events)} events")

    for event in events:
        before = event.status
        process_event(
            event,
            resolution=args.resolution,
            use_model=args.use_model,
            quiet_after=args.quiet_after,
        )
        store.save_event(event)
        failures = f" ({len(event.failures)} failures)" if event.failures else ""
        print(f"  {event.event_id} [{event.country}]: {before} -> {event.status}{failures}")
        for failure in event.failures:
            print(f"    failure: {failure['error']}")


if __name__ == "__main__":
    main()
