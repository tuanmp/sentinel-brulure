import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.event import FireEvent
from analytics.store import EventStore
from data_pipeline.firm_request import (
    COUNTRY_REGIONS,
    DEFAULT_DAYS_BACK,
    fetch_and_process_range,
)

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


def _window_dates(since: str | None, window_days: int, today: date) -> tuple[str, str]:
    """Return (start, end) inclusive dates for the seeding window."""
    if since is not None:
        start = date.fromisoformat(since)
    else:
        start = today - timedelta(days=window_days - 1)
    end = start + timedelta(days=window_days - 1)
    return start.isoformat(), end.isoformat()


def fetch_real_events(
    country: str, start_date: str, end_date: str, limit: int
) -> list[dict]:
    """Fetch and cluster real FIRMS fire events for a country over a window."""
    events = fetch_and_process_range(
        region=country,
        start_date=start_date,
        end_date=end_date,
        country=country,
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
        "--since",
        default=None,
        help=(
            "Window start (YYYY-MM-DD). Default: today - window-days + 1. "
            "Windows older than ~3 months use FIRMS *_SP archives."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help="Window size in days (internally paged in 5-day chunks)",
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

    if args.window_days < 1:
        parser.error("window-days must be at least 1")
    if args.since is not None:
        try:
            date.fromisoformat(args.since)
        except ValueError:
            parser.error("since must be a valid date (YYYY-MM-DD)")

    start_date, end_date = _window_dates(args.since, args.window_days, date.today())
    print(f"Seeding window: {start_date} .. {end_date}")

    store = EventStore(args.root)
    if args.replace:
        for event in store.list_events():
            store.delete_event(event.event_id)

    countries = args.country or COUNTRIES
    total = 0
    for country in countries:
        print(f"Fetching real fires for {country} ({start_date}..{end_date})...")
        try:
            events = fetch_real_events(country, start_date, end_date, args.per_country)
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
