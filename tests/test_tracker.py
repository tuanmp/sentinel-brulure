from datetime import date
from unittest.mock import patch

from analytics import tracker
from analytics.event import FireEvent


def _event(status="detected"):
    return FireEvent(
        event_id="e1",
        country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55,
        centroid_lon=4.5,
        start_date="2026-07-12",
        end_date="2026-07-18",
        status=status,
        quiet_days=2,
    )


def test_detected_runs_prefire_and_activates():
    event = _event()
    with patch("analytics.tracker.analyze_prefire", return_value={"ndvi": 0.5}):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert event.status == "active"
    assert event.prefire_metrics == {"ndvi": 0.5}


def test_active_appends_observations_and_ends_after_quiet():
    event = _event(status="active")
    rows = [
        {
            "date": "2026-07-18",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        },
        {
            "date": "2026-07-19",
            "frp_mw": 60.0,
            "detection_count": 3,
            "bbox_growth_deg": 0.05,
        },
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 22)
        )
    assert event.status == "ended"
    assert len(event.during_observations) == 2
    assert event.end_date == "2026-07-19"


def test_active_dedups_on_rerun():
    event = _event(status="active")
    event.during_observations = [
        {
            "date": "2026-07-18",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        },
    ]
    rows = [
        {
            "date": "2026-07-18",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        },
        {
            "date": "2026-07-19",
            "frp_mw": 60.0,
            "detection_count": 3,
            "bbox_growth_deg": 0.05,
        },
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 19)
        )
    assert [o["date"] for o in event.during_observations] == [
        "2026-07-18",
        "2026-07-19",
    ]


def test_ended_runs_postfire_and_recovers():
    event = _event(status="ended")
    with patch(
        "analytics.tracker.analyze_postfire", return_value={"burned_area_ha": 1.0}
    ):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 25))
    assert event.status == "recovering"
    assert event.postfire_assessment == {"burned_area_ha": 1.0}


def test_recovering_samples_due_months_and_completes():
    event = _event(status="recovering")

    def fake_analyze_recovery(ev, month, resolution=60):
        return {"offset_months": month, "ndvi": 0.5}

    with patch("analytics.tracker.analyze_recovery", side_effect=fake_analyze_recovery):
        tracker.process_event(event, resolution=60, today=date(2027, 7, 30))
    assert event.status == "complete"
    assert len(event.recovery_samples) == 5


def test_failure_marks_failure_and_keeps_state():
    event = _event()
    with patch(
        "analytics.tracker.analyze_prefire", side_effect=RuntimeError("no data")
    ):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert event.status == "detected"
    assert len(event.failures) == 1
