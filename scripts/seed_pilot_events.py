import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.event import FireEvent
from analytics.store import EventStore
from data_pipeline.firm_request import COUNTRY_REGIONS, fetch_and_process

COUNTRIES = list(COUNTRY_REGIONS.keys())


def _build_event(spec: dict) -> FireEvent:
    return FireEvent(
        event_id=str(uuid4()),
        country=spec["country"],
        bbox=spec["bbox"],
        centroid_lat=spec["centroid_lat"],
        centroid_lon=spec["centroid_lon"],
        start_date=spec["start_date"],
        end_date=spec["end_date"],
        cluster_id=spec.get("cluster_id"),
    )


def fetch_real_events(country: str, days_back: int, limit: int) -> list[dict]:
    """Fetch and cluster real FIRMS fire events for a country, best first."""
    events = fetch_and_process(
        region=country, days_back=days_back, country=country
    )
    return events[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Seed pilot fire events into the store from real FIRMS fires"
    )
    parser.add_argument(
        "--root", default=None, help="Event store root (default: reports/events)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=5,
        help="Days of FIRMS history to seed from (max 5)",
    )
    parser.add_argument(
        "--per-country",
        type=int,
        default=2,
        help="Max real fires to seed per country",
    )
    parser.add_argument(
        "--country",
        action="append",
        choices=COUNTRIES,
        help="Restrict to a country (repeatable). Default: all.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear the existing store before seeding",
    )
    args = parser.parse_args()

    if not (1 <= args.days_back <= 5):
        parser.error("days-back must be between 1 and 5 (FIRMS limit)")

    store = EventStore(args.root)
    if args.replace:
        for event in store.list_events():
            store.delete_event(event.event_id)

    countries = args.country or COUNTRIES
    total = 0
    for country in countries:
        print(f"Fetching real fires for {country} (last {args.days_back} days)...")
        try:
            events = fetch_real_events(country, args.days_back, args.per_country)
        except Exception as exc:
            print(f"  SKIP {country}: {exc}")
            continue
        for spec in events:
            store.save_event(_build_event(spec))
            total += 1
            print(
                f"  + {spec['country']} cluster={spec.get('cluster_id')} "
                f"{spec['start_date']}..{spec['end_date']} "
                f"frp={spec['total_frp_mw']:.0f}MW det={spec['detection_count']}"
            )
        if not events:
            print(f"  (no qualifying fires found for {country})")
    print(f"Seeded {total} real pilot events into {store.root}")


if __name__ == "__main__":
    main()
