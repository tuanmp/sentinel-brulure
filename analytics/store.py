import json
import logging
from pathlib import Path

from .event import FireEvent

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "reports" / "events"


class EventStore:
    """JSON-backed event store. Thin interface so a DB backend can replace it."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, event_id: str) -> Path:
        path = self.root / f"{event_id}.json"
        if path.resolve().parent != self.root.resolve():
            raise ValueError(f"event_id {event_id!r} escapes the store root")
        return path

    def save_event(self, event: FireEvent) -> None:
        payload = event.to_dict()
        self._path(event.event_id).write_text(json.dumps(payload, indent=2))

    def update_event(self, event: FireEvent) -> None:
        self.save_event(event)

    def load_event(self, event_id: str) -> FireEvent | None:
        path = self._path(event_id)
        if not path.exists():
            return None
        try:
            return FireEvent.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
            logger.warning("Failed to load event %s from %s: %s", event_id, path, exc)
            return None

    def list_events(self) -> list[FireEvent]:
        events = []
        for p in sorted(self.root.glob("*.json")):
            event = self.load_event(p.stem)
            if event is not None:
                events.append(event)
        return events
