from datetime import UTC, date, datetime

from .during import fetch_daily_observations
from .postfire import analyze_postfire
from .prefire import analyze_prefire
from .recovery import RECOVERY_OFFSETS_MONTHS, analyze_recovery, recovery_due


def _merge_observations(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    by_date = {row["date"]: row for row in existing}
    for row in new_rows:
        by_date[row["date"]] = row
    return sorted(by_date.values(), key=lambda row: row["date"])


def _dispatch(event, resolution, use_model, quiet_after, today):
    if event.status == "detected":
        event.prefire_metrics = analyze_prefire(event, resolution=resolution)
        event.transition("active")
    elif event.status == "active":
        rows = fetch_daily_observations(event)
        event.during_observations = _merge_observations(event.during_observations, rows)
        latest = (
            event.during_observations[-1]["date"] if event.during_observations else None
        )
        if latest:
            last_date = datetime.strptime(latest, "%Y-%m-%d").date()
            event.quiet_days = (today - last_date).days
            event.end_date = max(event.end_date, latest)
        else:
            event.quiet_days += 1
        if event.quiet_days >= quiet_after:
            event.transition("ended")
    elif event.status == "ended":
        event.postfire_assessment = analyze_postfire(
            event, resolution=resolution, use_model=use_model
        )
        event.transition("recovering")
    elif event.status == "recovering":
        sampled = {sample["offset_months"] for sample in event.recovery_samples}
        due = [
            month
            for month in RECOVERY_OFFSETS_MONTHS
            if month not in sampled and recovery_due(event, month, today)
        ]
        for month in due:
            event.recovery_samples.append(
                analyze_recovery(event, month, resolution=resolution)
            )
        sampled = {sample["offset_months"] for sample in event.recovery_samples}
        if RECOVERY_OFFSETS_MONTHS[-1] in sampled:
            event.transition("complete")


def process_event(
    event,
    resolution: int = 60,
    use_model: bool = False,
    quiet_after: int = 3,
    today=None,
):
    """Run one tick for a single event. Returns the (mutated) event."""
    today = today or date.today()
    try:
        _dispatch(event, resolution, use_model, quiet_after, today)
        event.failures = []
    except Exception as exc:  # phase failed; keep state, retry next tick
        event.failures.append(
            {
                "status": event.status,
                "error": str(exc),
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
    return event
