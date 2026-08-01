import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.report import build_html_report, export_csv, export_json
from analytics.store import EventStore


def main():
    parser = argparse.ArgumentParser(description="Build reports for all tracked events")
    parser.add_argument("--root", default=None, help="Event store root (default: reports/events)")
    parser.add_argument("--out", default=None, help="Output dir (default: reports/)")
    args = parser.parse_args()

    store = EventStore(args.root)
    out_dir = Path(args.out) if args.out else store.root.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = store.list_events()
    print(f"Building reports for {len(events)} events")

    for event in events:
        json_path = out_dir / f"{event.event_id}.json"
        html_path = out_dir / f"{event.event_id}.html"
        csv_path = out_dir / f"{event.event_id}_observations.csv"

        json_path.write_text(json.dumps(export_json(event), indent=2))
        build_html_report(event, html_path)
        csv_path.write_text(export_csv(event.during_observations))
        print(f"  Wrote {json_path.name}, {html_path.name}, {csv_path.name}")


if __name__ == "__main__":
    main()
