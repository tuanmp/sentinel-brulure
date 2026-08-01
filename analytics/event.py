from dataclasses import asdict, dataclass, field


class InvalidTransitionError(Exception):
    pass


VALID_TRANSITIONS = {
    "detected": {"active"},
    "active": {"ended"},
    "ended": {"recovering"},
    "recovering": {"complete"},
}


@dataclass
class FireEvent:
    event_id: str
    country: str
    bbox: list[float]
    centroid_lat: float
    centroid_lon: float
    start_date: str
    end_date: str
    cluster_id: int | None = None
    status: str = "detected"
    quiet_days: int = 0
    prefire_metrics: dict = field(default_factory=dict)
    during_observations: list[dict] = field(default_factory=list)
    postfire_assessment: dict = field(default_factory=dict)
    recovery_samples: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FireEvent":
        return cls(**data)

    def transition(self, new_status: str) -> None:
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{self.status}' to '{new_status}'"
            )
        self.status = new_status
