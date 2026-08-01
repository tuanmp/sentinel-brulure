import pytest

from analytics.event import FireEvent, InvalidTransitionError


def _event():
    return FireEvent(
        event_id="evt-1",
        country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55,
        centroid_lon=4.5,
        start_date="2026-07-12",
        end_date="2026-07-18",
        cluster_id=7,
    )


def test_initial_status_is_detected():
    assert _event().status == "detected"


def test_transition_happy_path():
    event = _event()
    for target in ["active", "ended", "recovering", "complete"]:
        event.transition(target)
    assert event.status == "complete"


def test_invalid_transition_raises():
    event = _event()
    event.transition("active")
    with pytest.raises(InvalidTransitionError):
        event.transition("complete")


def test_transition_from_complete_raises():
    event = _event()
    for target in ["active", "ended", "recovering", "complete"]:
        event.transition(target)
    with pytest.raises(InvalidTransitionError):
        event.transition("active")


def test_to_dict_from_dict_round_trip():
    event = _event()
    restored = FireEvent.from_dict(event.to_dict())
    assert restored == event
