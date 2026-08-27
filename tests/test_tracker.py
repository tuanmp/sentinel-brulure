from datetime import date
from unittest.mock import patch

from analytics import tracker
from analytics.event import FireEvent
from analytics.recovery import MAX_RECOVERY_ATTEMPTS


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


def test_ended_reactivates_when_new_detections_appear():
    event = _event(status="ended")
    rows = [
        {
            "date": "2026-07-20",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        }
    ]
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=rows),
        patch("analytics.tracker.analyze_postfire") as mock_postfire,
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 20)
        )
    assert event.status == "active"
    mock_postfire.assert_not_called()


def test_ended_still_transitions_to_recovering_without_recent_activity():
    event = _event(status="ended")
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=[]),
        patch(
            "analytics.tracker.analyze_postfire", return_value={"burned_area_ha": 1.0}
        ),
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 20)
        )
    assert event.status == "recovering"


def test_recovering_reactivates_on_rekindle():
    event = _event(status="recovering")
    event.recovery_samples = [{"offset_months": 1, "ndvi": 0.5}]
    rows = [
        {
            "date": "2026-07-20",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        }
    ]
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=rows),
        patch("analytics.tracker.analyze_recovery") as mock_recovery,
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 20)
        )
    assert event.status == "active"
    assert event.recovery_samples == []
    mock_recovery.assert_not_called()


def test_recovering_samples_due_months_and_completes():
    event = _event(status="recovering")

    def fake_analyze_recovery(ev, month, resolution=60):
        return {"offset_months": month, "ndvi": 0.5}

    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=[]),
        patch("analytics.tracker.analyze_recovery", side_effect=fake_analyze_recovery),
    ):
        tracker.process_event(event, resolution=60, today=date(2027, 7, 30))
    assert event.status == "complete"
    assert len(event.recovery_samples) == 5


def test_recovering_fetches_during_observations():
    event = _event(status="recovering")
    rows = [
        {
            "date": "2026-07-18",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        }
    ]
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=rows),
        patch("analytics.tracker.analyze_recovery", return_value={"ndvi": 0.5}),
    ):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 25))
    assert event.status == "recovering"
    assert len(event.during_observations) == 1
    assert event.during_observations[0]["date"] == "2026-07-18"


def _recovery_fake(month_ndvi):
    def fake(ev, month, resolution=60):
        return {
            "offset_months": month,
            "ndvi": month_ndvi.get(month),
            "fetched_on": "2027-01-01T00:00:00+00:00",
        }

    return fake


def test_recovering_does_not_complete_with_unresolved_invalid_month():
    event = _event(status="recovering")
    event.quiet_days = 10
    fake = _recovery_fake(dict.fromkeys((1, 3, 6, 9, 12)))
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=[]),
        patch("analytics.tracker.analyze_recovery", side_effect=fake),
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2027, 7, 30)
        )
    assert event.status == "recovering"
    assert len(event.recovery_samples) == 5


def test_recovering_completes_after_invalid_month_exhausted():
    event = _event(status="recovering")
    event.quiet_days = 10
    fake = _recovery_fake({1: 0.5, 3: 0.5, 6: None, 9: 0.5, 12: 0.5})
    with (
        patch("analytics.tracker.fetch_daily_observations", return_value=[]),
        patch("analytics.tracker.analyze_recovery", side_effect=fake),
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2027, 7, 30)
        )
        assert event.status == "recovering"
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2027, 8, 6)
        )
        assert event.status == "recovering"
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2027, 8, 13)
        )
    assert event.status == "complete"
    month_6 = [s for s in event.recovery_samples if s["offset_months"] == 6]
    assert len(month_6) == MAX_RECOVERY_ATTEMPTS


def test_failure_marks_failure_and_keeps_state():
    event = _event()
    with patch(
        "analytics.tracker.analyze_prefire", side_effect=RuntimeError("no data")
    ):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert event.status == "detected"
    assert len(event.failures) == 1


def test_success_clears_prior_failures():
    event = _event()
    event.failures = [{"status": "detected", "error": "x", "at": "t"}]
    with patch("analytics.tracker.analyze_prefire", return_value={"ndvi": 0.5}):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert event.status == "active"
    assert event.failures == []


def test_active_no_observations_increments_quiet_days():
    event = _event(status="active")
    with patch("analytics.tracker.fetch_daily_observations", return_value=[]):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 22)
        )
    assert event.status == "ended"
    assert event.quiet_days == 3


def test_active_keeps_later_end_date():
    event = _event(status="active")
    event.end_date = "2026-07-20"
    rows = [
        {
            "date": "2026-07-18",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        }
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 19)
        )
    assert event.end_date == "2026-07-20"


def test_future_observation_quiet_days_never_negative():
    event = _event(status="active")
    rows = [
        {
            "date": "2026-07-20",
            "frp_mw": 100.0,
            "detection_count": 5,
            "bbox_growth_deg": 0.1,
        }
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2026, 7, 19)
        )
    assert event.quiet_days == 0
    assert event.status == "active"


def test_failures_capped_at_max():
    event = _event()
    event.failures = [
        {"status": "detected", "error": f"e{i}", "at": "t"}
        for i in range(tracker.MAX_FAILURES)
    ]
    with patch(
        "analytics.tracker.analyze_prefire", side_effect=RuntimeError("no data")
    ):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert len(event.failures) == tracker.MAX_FAILURES
    assert event.failures[-1]["error"] == "no data"


def test_no_sentinel_coverage_skips_sentinel_phases():
    event = _event()
    event.start_date = "2003-07-12"
    event.end_date = "2003-07-18"
    with (
        patch("analytics.prefire.has_imagery", return_value=False),
        patch("analytics.postfire.has_imagery", return_value=False),
        patch("analytics.recovery.has_imagery", return_value=False),
        patch("analytics.tracker.fetch_daily_observations", return_value=[]),
    ):
        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2003, 7, 22)
        )
        assert event.status == "active"
        assert event.prefire_metrics["skipped_reason"] == "no_sentinel_data"

        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2003, 7, 22)
        )
        assert event.status == "ended"

        tracker.process_event(
            event, resolution=60, quiet_after=3, today=date(2003, 7, 22)
        )
        assert event.status == "recovering"
        assert event.postfire_assessment["skipped_reason"] == "no_sentinel_data"
