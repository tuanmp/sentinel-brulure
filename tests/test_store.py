from analytics.event import FireEvent
from analytics.store import EventStore


def _event(event_id="evt-1"):
    return FireEvent(
        event_id=event_id,
        country="spain",
        bbox=[-1.3, 38.9, -0.6, 39.5],
        centroid_lat=39.2,
        centroid_lon=-0.95,
        start_date="2026-07-15",
        end_date="2026-07-20",
        cluster_id=3,
    )


def test_save_and_load_round_trip(tmp_path):
    store = EventStore(root=tmp_path)
    store.save_event(_event())
    loaded = store.load_event("evt-1")
    assert loaded is not None
    assert loaded.country == "spain"
    assert loaded.status == "detected"


def test_load_missing_returns_none(tmp_path):
    store = EventStore(root=tmp_path)
    assert store.load_event("nope") is None


def test_list_events(tmp_path):
    store = EventStore(root=tmp_path)
    store.save_event(_event("a"))
    store.save_event(_event("b"))
    ids = sorted(e.event_id for e in store.list_events())
    assert ids == ["a", "b"]


def test_update_overwrites(tmp_path):
    store = EventStore(root=tmp_path)
    store.save_event(_event())
    event = store.load_event("evt-1")
    event.status = "active"
    store.update_event(event)
    assert store.load_event("evt-1").status == "active"
