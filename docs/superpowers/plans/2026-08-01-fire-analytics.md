# Fire Event Analytics (Before/During/After) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing burn-scar pipeline into a per-fire-event analytics system producing near-real-time reports covering pre-fire fuel state, during-fire FIRMS progression, post-fire severity, and a 12-month recovery timeline for fires in France, Spain, Italy, and Greece.

**Architecture:** Event-centric pipeline (Approach A). Each fire is a `FireEvent` dataclass that transitions through lifecycle states (`detected → active → ended → recovering → complete`) driven by a daily `tracker`. A thin JSON `EventStore` is the migration seam toward the future FastAPI/SQLite backend. Reuses existing `data_pipeline` modules for Sentinel-2 fetch, NBR/dNBR, and FIRMS clustering.

**Tech Stack:** Python 3.12, `analytics/` package, `data_pipeline/` (existing), `sentinelhub`, `requests`, `numpy`, `pandas`, `matplotlib`, pytest, ruff.

**Layout deviation from spec (documented):** built reports (`.html`, `.json`, `.csv`) live directly in `reports/`; internal event state lives in `reports/events/`. No `reports/reports/`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `data_pipeline/firm_request.py` (modify) | Add `COUNTRY_REGIONS` (FR/ES/IT/GR bboxes) + `country` tagging |
| `analytics/__init__.py` (create) | Package marker |
| `analytics/event.py` (create) | `FireEvent` dataclass + lifecycle transitions |
| `analytics/store.py` (create) | JSON event store (thin interface) |
| `analytics/fire_weather.py` (create) | Open-Meteo fire-danger proxy fetcher |
| `analytics/prefire.py` (create) | NDVI/NDWI dryness + weather index |
| `analytics/during.py` (create) | FIRMS FRP progression |
| `analytics/postfire.py` (create) | dNBR severity + burned-area estimate |
| `analytics/recovery.py` (create) | monthly NDVI vs pre-fire baseline |
| `analytics/tracker.py` (create) | daily state-machine driver |
| `analytics/report.py` (create) | HTML + JSON/CSV exporters |
| `scripts/seed_pilot_events.py` (create) | Seed ~8 known fires into store |
| `scripts/run_event_tracker.py` (create) | Run one daily tick over all events |
| `scripts/build_reports.py` (create) | Regenerate reports from store |
| `.gitignore` (modify) | Ignore `reports/` |
| `tests/test_*.py` (create) | Per-module pytest tests |

---

### Task 1: Country regions + country tagging in firm_request.py

**Files:**
- Modify: `data_pipeline/firm_request.py`
- Test: `tests/test_firm_countries.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_firm_countries.py`:

```python
from unittest.mock import Mock, patch

from data_pipeline import firm_request as fr


def _csv_text():
    return (
        "latitude,longitude,acq_date,acq_time,confidence,frp,daynight\n"
        "44.5,4.5,2026-07-12,1500,n,350.0,D\n"
        "44.6,4.6,2026-07-12,1600,h,500.0,D\n"
        "48.0,2.0,2026-07-12,1500,l,50.0,D\n"
    )


def test_fetch_fire_events_supports_country_region():
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = _csv_text()
        df = fr.fetch_fire_events(region="france", days_back=3)

    url = mock_get.call_args.args[0]
    assert "france" not in url
    assert "-5.5,41.0,9.5,51.5" in url
    assert len(df) == 3
    assert "confidence_label" in df.columns


def test_fetch_fire_events_unknown_region_raises():
    try:
        fr.fetch_fire_events(region="atlantis", days_back=3)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Unknown region" in str(exc)


def test_fetch_and_process_tags_country_on_events():
    with patch("data_pipeline.firm_request.fetch_fire_events") as mock_fetch, \
         patch("data_pipeline.firm_request.cluster_detections") as mock_cluster, \
         patch("data_pipeline.firm_request.filter_events") as mock_filter:
        mock_fetch.return_value = Mock()
        mock_cluster.return_value = [{"cluster_id": 1}]
        mock_filter.return_value = [{"cluster_id": 1}]
        events = fr.fetch_and_process(region="france", country="france")
    assert events[0]["country"] == "france"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_firm_countries.py -v`
Expected: FAIL — `Unknown region: france` / no `country` key.

- [ ] **Step 3: Implement**

Modify `data_pipeline/firm_request.py`:

Add after the existing `REGIONS` dict (around line 25):

```python
COUNTRY_REGIONS = {
    "france": {"bbox": [-5.5, 41.0, 9.5, 51.5], "name": "France"},
    "spain": {"bbox": [-9.5, 35.5, 3.5, 44.0], "name": "Spain"},
    "italy": {"bbox": [6.5, 36.0, 18.5, 47.5], "name": "Italy"},
    "greece": {"bbox": [19.0, 34.5, 28.5, 41.5], "name": "Greece"},
}
```

Replace the bbox-resolution block in `fetch_fire_events` (currently lines 67-73):

```python
    if region == "world":
        bbox_str = "world"
    elif region in REGIONS:
        bbox = REGIONS[region]["bbox"]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    elif region in COUNTRY_REGIONS:
        bbox = COUNTRY_REGIONS[region]["bbox"]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    else:
        raise ValueError(
            f"Unknown region: {region}. Use: {list(REGIONS.keys())} "
            f"or {list(COUNTRY_REGIONS.keys())} or 'world'"
        )
```

Change the `fetch_and_process` signature and body (currently lines 192-211):

```python
def fetch_and_process(
    region: str = "world",
    days_back: int = DEFAULT_DAYS_BACK,
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
    country: str | None = None,
) -> list[dict]:
    """
    Full pipeline: fetch, cluster, and filter fire events.

    Args:
        region: Regional filter ("europe", "north_america", "world", or a
            COUNTRY_REGIONS key like "france")
        days_back: Days of historical data to fetch
        min_confidence: Minimum confidence level
        country: Optional country tag added to each returned event

    Returns:
        List of fire event dicts compatible with sentinel_request.process_fire_event()
    """
    df = fetch_fire_events(region=region, days_back=days_back, min_confidence=min_confidence)
    events = cluster_detections(df)
    events = filter_events(events)
    if country:
        for event in events:
            event["country"] = country
    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_firm_countries.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/firm_request.py tests/test_firm_countries.py
git commit -m "feat: add country regions and country tagging to FIRMS pipeline"
```

---

### Task 2: Repo scaffolding — .gitignore, analytics package

**Files:**
- Modify: `.gitignore`
- Create: `analytics/__init__.py`

- [ ] **Step 1: Implement**

Append to `.gitignore`:

```
# Generated fire analytics artifacts
reports/
```

Create `analytics/__init__.py` (empty file).

- [ ] **Step 2: Commit**

```bash
git add .gitignore analytics/__init__.py
git commit -m "chore: add analytics package and ignore reports/"
```

---

### Task 3: FireEvent model + lifecycle transitions

**Files:**
- Create: `analytics/event.py`
- Test: `tests/test_event.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_event.py`:

```python
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


def test_to_dict_from_dict_round_trip():
    event = _event()
    restored = FireEvent.from_dict(event.to_dict())
    assert restored.event_id == "evt-1"
    assert restored.bbox == [4.2, 44.3, 4.8, 44.8]
    assert restored.during_observations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_event.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics`.

- [ ] **Step 3: Implement**

Create `analytics/event.py`:

```python
from dataclasses import asdict, dataclass, field


class InvalidTransitionError(Exception):
    pass


VALID_TRANSITIONS = {
    "detected": {"active"},
    "active": {"ended"},
    "ended": {"recovering"},
    "recovering": {"complete"},
}

VALID_STATUSES = set(VALID_TRANSITIONS) | {"complete"}


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
        return cls(**dict(data))

    def transition(self, new_status: str) -> None:
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{self.status}' to '{new_status}'"
            )
        self.status = new_status
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_event.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/event.py tests/test_event.py
git commit -m "feat: add FireEvent model with lifecycle transitions"
```

---

### Task 4: JSON EventStore

**Files:**
- Create: `analytics/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
import pytest

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.store`.

- [ ] **Step 3: Implement**

Create `analytics/store.py`:

```python
import json
from pathlib import Path

from .event import FireEvent

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "reports" / "events"


class EventStore:
    """JSON-backed event store. Thin interface so a DB backend can replace it."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, event_id: str) -> Path:
        return self.root / f"{event_id}.json"

    def save_event(self, event: FireEvent) -> None:
        payload = event.to_dict()
        self._path(event.event_id).write_text(
            json.dumps(payload, indent=2, default=str)
        )

    def update_event(self, event: FireEvent) -> None:
        self.save_event(event)

    def load_event(self, event_id: str) -> FireEvent | None:
        path = self._path(event_id)
        if not path.exists():
            return None
        return FireEvent.from_dict(json.loads(path.read_text()))

    def list_events(self) -> list[FireEvent]:
        return [self.load_event(p.stem) for p in sorted(self.root.glob("*.json"))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_store.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/store.py tests/test_store.py
git commit -m "feat: add JSON event store with migration-seam interface"
```

---

### Task 5: Open-Meteo fire-danger proxy

**Files:**
- Create: `analytics/fire_weather.py`
- Test: `tests/test_fire_weather.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fire_weather.py`:

```python
from unittest.mock import Mock, patch

from analytics import fire_weather as fw


def test_fire_danger_low_for_cool_wet_conditions():
    score = fw.compute_fire_danger(temp_c=15, wind_kmh=5, precip_mm=20, soil_moisture=0.4)
    assert score == 0.0


def test_fire_danger_high_for_hot_dry_windy():
    score = fw.compute_fire_danger(temp_c=38, wind_kmh=45, precip_mm=0, soil_moisture=0.0)
    assert score == 100.0


def test_fetch_fire_weather_parses_daily_rows():
    payload = {
        "daily": {
            "time": ["2026-07-01", "2026-07-02"],
            "temperature_2m_max": [30.0, 31.0],
            "wind_speed_10m_max": [20.0, 15.0],
            "precipitation_sum": [0.0, 1.0],
            "soil_moisture_0_to_10cm_mean": [0.12, 0.15],
        }
    }
    mock_resp = Mock()
    mock_resp.json.return_value = payload
    with patch("analytics.fire_weather.requests.get", return_value=mock_resp) as mock_get:
        rows = fw.fetch_fire_weather(44.5, 4.5, "2026-07-01", "2026-07-02")

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-01"
    assert rows[0]["fire_danger_index"] > rows[1]["fire_danger_index"]
    params = mock_get.call_args.kwargs["params"]
    assert params["latitude"] == 44.5
    assert "temperature_2m_max" in params["daily"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_fire_weather.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.fire_weather`.

- [ ] **Step 3: Implement**

Create `analytics/fire_weather.py`:

```python
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_fire_danger(temp_c: float, wind_kmh: float, precip_mm: float, soil_moisture: float) -> float:
    """Simple 0-100 fire-danger proxy from daily weather fields."""
    temp_score = _clip((temp_c - 20.0) / 15.0, 0.0, 1.0)
    wind_score = _clip((wind_kmh - 5.0) / 30.0, 0.0, 1.0)
    precip_score = 1.0 - _clip(precip_mm / 10.0, 0.0, 1.0)
    moisture_score = 1.0 - _clip(soil_moisture / 0.4, 0.0, 1.0)
    return 100.0 * (temp_score + wind_score + precip_score + moisture_score) / 4.0


def fetch_fire_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> list[dict]:
    """Fetch per-day weather rows for a lat/lon window from the Open-Meteo archive."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_max,wind_speed_10m_max,"
            "precipitation_sum,soil_moisture_0_to_10cm_mean"
        ),
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_URL, params=params)
    response.raise_for_status()
    daily = response.json()["daily"]

    rows = []
    for i, date in enumerate(daily["time"]):
        values = [
            daily["temperature_2m_max"][i],
            daily["wind_speed_10m_max"][i],
            daily["precipitation_sum"][i],
            daily["soil_moisture_0_to_10cm_mean"][i],
        ]
        if any(v is None for v in values):
            continue
        temp, wind, precip, moisture = values
        rows.append({
            "date": date,
            "temperature_max_c": temp,
            "wind_max_kmh": wind,
            "precipitation_mm": precip,
            "soil_moisture": moisture,
            "fire_danger_index": compute_fire_danger(temp, wind, precip, moisture),
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_fire_weather.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/fire_weather.py tests/test_fire_weather.py
git commit -m "feat: add Open-Meteo fire-danger proxy"
```

---

### Task 6: Pre-fire analytics (NDVI/NDWI + weather)

**Files:**
- Create: `analytics/prefire.py`
- Test: `tests/test_prefire.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefire.py`:

```python
from unittest.mock import Mock, patch

import numpy as np

from analytics import prefire
from analytics.event import FireEvent


def _bands():
    bands = np.zeros((7, 2, 2), dtype=np.float32)
    bands[1] = 0.5   # GREEN
    bands[2] = 0.2   # RED
    bands[3] = 0.8   # NIR
    bands[6] = 1     # mask
    return bands


def test_compute_ndvi():
    ndvi = prefire.compute_ndvi(_bands())
    assert np.allclose(ndvi, 0.6)


def test_compute_ndwi():
    ndwi = prefire.compute_ndwi(_bands())
    assert np.allclose(ndwi, -0.23076923076923078)


def test_ndvi_masks_invalid_pixels():
    bands = _bands()
    bands[6][0, 0] = 0
    ndvi = prefire.compute_ndvi(bands)
    assert np.isnan(ndvi[0, 0])
    assert ndvi[1, 1] == 0.6


def test_prefire_window():
    window = prefire.prefire_window("2026-07-12")
    assert window == ("2026-06-27", "2026-07-11")


def test_analyze_prefire():
    event = FireEvent(
        event_id="e1", country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55, centroid_lon=4.5,
        start_date="2026-07-12", end_date="2026-07-18",
    )
    with patch("analytics.prefire.fetch_bbox", return_value=_bands()) as mock_fetch, \
         patch("analytics.prefire.fetch_fire_weather") as mock_weather:
        mock_weather.return_value = [
            {"date": "2026-06-27", "fire_danger_index": 40.0},
            {"date": "2026-06-28", "fire_danger_index": 60.0},
        ]
        metrics = prefire.analyze_prefire(event, resolution=60)

    assert mock_fetch.call_args.args[0] == ("2026-06-27", "2026-07-11")
    assert metrics["ndvi"] == 0.6
    assert metrics["ndwi"] == pytest.approx(-0.23076923076923078)
    assert metrics["weather_index"] == 50.0
    assert "fetched_on" in metrics
```

Add import at top: `import pytest`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_prefire.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.prefire`.

- [ ] **Step 3: Implement**

Create `analytics/prefire.py`:

```python
from datetime import datetime, timedelta

import numpy as np
from sentinelhub import BBox, CRS

from data_pipeline.sentinel_request import fetch_bbox

from .fire_weather import fetch_fire_weather

# evalscript output band order: B02=0, B03=1, B04=2, B08=3, B11=4, B12=5, mask=6
GREEN, RED, NIR, MASK = 1, 2, 3, 6


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    nir = bands[NIR].astype(float)
    red = bands[RED].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi[bands[MASK] == 0] = np.nan
    return ndvi


def compute_ndwi(bands: np.ndarray) -> np.ndarray:
    green = bands[GREEN].astype(float)
    nir = bands[NIR].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (green - nir) / (green + nir)
    ndwi[bands[MASK] == 0] = np.nan
    return ndwi


def mean_metric(index: np.ndarray) -> float | None:
    valid = index[~np.isnan(index)]
    if valid.size == 0:
        return None
    return float(np.nanmean(valid))


def prefire_window(start_date: str) -> tuple[str, str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return (
        (start - timedelta(days=15)).strftime("%Y-%m-%d"),
        (start - timedelta(days=1)).strftime("%Y-%m-%d"),
    )


def analyze_prefire(event, resolution: int = 60) -> dict:
    window = prefire_window(event.start_date)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    bands = fetch_bbox(window, bbox, resolution=resolution)

    weather_rows = fetch_fire_weather(
        event.centroid_lat, event.centroid_lon, *window
    )
    weather_index = None
    if weather_rows:
        weather_index = float(
            np.mean([row["fire_danger_index"] for row in weather_rows])
        )

    return {
        "ndvi": mean_metric(compute_ndvi(bands)),
        "ndwi": mean_metric(compute_ndwi(bands)),
        "weather_index": weather_index,
        "window": list(window),
        "fetched_on": datetime.utcnow().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_prefire.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/prefire.py tests/test_prefire.py
git commit -m "feat: add pre-fire NDVI/NDWI dryness and weather index"
```

---

### Task 7: During-fire FIRMS progression

**Files:**
- Create: `analytics/during.py`
- Test: `tests/test_during.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_during.py`:

```python
from unittest.mock import patch

import pandas as pd

from analytics import during
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1", country="spain",
        bbox=[-1.3, 38.9, -0.6, 39.5],
        centroid_lat=39.2, centroid_lon=-0.95,
        start_date="2026-07-15", end_date="2026-07-16",
    )


def _df():
    return pd.DataFrame({
        "latitude": [39.1, 39.2, 39.0, 42.0],
        "longitude": [-1.0, -0.9, -1.1, -3.0],
        "frp": [100.0, 200.0, 50.0, 999.0],
        "acq_date": ["2026-07-15", "2026-07-15", "2026-07-16", "2026-07-15"],
        "confidence": ["h", "n", "l", "h"],
    })


def test_fetch_daily_observations_groups_and_filters():
    with patch("analytics.during.fetch_fire_events", return_value=_df()):
        rows = during.fetch_daily_observations(_event(), days_back=5)

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-15"
    assert rows[0]["frp_mw"] == 300.0
    assert rows[0]["detection_count"] == 2
    assert rows[1]["date"] == "2026-07-16"
    assert rows[1]["detection_count"] == 1


def test_fetch_daily_observations_outside_bbox_excluded():
    with patch("analytics.during.fetch_fire_events", return_value=_df()):
        rows = during.fetch_daily_observations(_event(), days_back=5)
    all_dates = [r["date"] for r in rows]
    assert len(all_dates) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_during.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.during`.

- [ ] **Step 3: Implement**

Create `analytics/during.py`:

```python
from data_pipeline.firm_request import fetch_fire_events


def _bbox_span(group) -> float:
    lon_span = group["longitude"].max() - group["longitude"].min()
    lat_span = group["latitude"].max() - group["latitude"].min()
    return float(lon_span + lat_span)


def fetch_daily_observations(event, days_back: int = 10) -> list[dict]:
    """Fetch FIRMS detections for the event's country, filter to its bbox,
    and aggregate per-day FRP/count/span rows."""
    df = fetch_fire_events(region=event.country, days_back=days_back)
    min_lon, min_lat, max_lon, max_lat = event.bbox
    in_bbox = df[
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
    ]

    rows = []
    for date, group in in_bbox.groupby("acq_date"):
        rows.append({
            "date": str(date),
            "frp_mw": float(group["frp"].sum()),
            "detection_count": int(len(group)),
            "bbox_growth_deg": _bbox_span(group),
        })
    return sorted(rows, key=lambda row: row["date"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_during.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/during.py tests/test_during.py
git commit -m "feat: add during-fire FIRMS FRP progression aggregation"
```

---

### Task 8: Post-fire severity + burned area

**Files:**
- Create: `analytics/postfire.py`
- Test: `tests/test_postfire.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_postfire.py`:

```python
from unittest.mock import patch

import numpy as np

from analytics import postfire
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1", country="greece",
        bbox=[23.4, 37.8, 24.1, 38.4],
        centroid_lat=38.1, centroid_lon=23.75,
        start_date="2026-07-10", end_date="2026-07-16",
        cluster_id=9,
    )


def _dnbr():
    dnbr = np.full((10, 10), np.nan, dtype=float)
    dnbr[0, 0] = 0.05    # unburned
    dnbr[0, 1] = 0.2     # low
    dnbr[0, 2] = 0.35    # moderate
    dnbr[0, 3] = 0.5     # high
    dnbr[0, 4] = 0.7     # very high
    return dnbr


def test_classify_severity():
    classes = postfire.classify_severity(_dnbr())
    assert set(classes) == {"unburned", "low", "moderate", "high", "very_high"}
    assert classes["unburned"] == 0.2
    assert classes["very_high"] == 0.2
    assert sum(classes.values()) == 1.0


def test_estimate_burned_area():
    area = postfire.estimate_burned_area(_dnbr(), resolution=60)
    assert area == 0.108  # 3 burned px * 3600 m2 / 10000


def test_analyze_postfire():
    with patch("analytics.postfire.process_fire_event") as mock_process:
        mock_process.return_value = {"dnbr": _dnbr()}
        assessment = postfire.analyze_postfire(_event(), resolution=60)

    assert assessment["burned_area_ha"] == 0.108
    assert assessment["severity_classes"]["low"] == 0.2
    assert "fetched_on" in assessment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_postfire.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.postfire`.

- [ ] **Step 3: Implement**

Create `analytics/postfire.py`:

```python
from datetime import datetime

import numpy as np

from data_pipeline.sentinel_request import process_fire_event

SEVERITY_CLASSES = [
    ("unburned", None, 0.1),
    ("low", 0.1, 0.27),
    ("moderate", 0.27, 0.44),
    ("high", 0.44, 0.66),
    ("very_high", 0.66, None),
]

BURNED_THRESHOLD = 0.27  # severity >= moderate counts as burned


def classify_severity(dnbr: np.ndarray) -> dict[str, float]:
    valid = dnbr[~np.isnan(dnbr)]
    total = valid.size
    counts = {}
    for name, low, high in SEVERITY_CLASSES:
        if low is None:
            mask = valid < high
        elif high is None:
            mask = valid >= low
        else:
            mask = (valid >= low) & (valid < high)
        counts[name] = float(np.count_nonzero(mask) / total) if total else 0.0
    return counts


def estimate_burned_area(dnbr: np.ndarray, resolution: int = 60) -> float:
    burned = np.count_nonzero(dnbr[~np.isnan(dnbr)] >= BURNED_THRESHOLD)
    return float(burned * resolution * resolution / 10000.0)


def analyze_postfire(event, resolution: int = 60, use_model: bool = False) -> dict:
    event_dict = {
        "cluster_id": event.cluster_id,
        "bbox": event.bbox,
        "start_date": event.start_date,
        "end_date": event.end_date,
    }
    result = process_fire_event(event_dict, resolution=resolution, use_model=use_model)
    dnbr = result["dnbr"]

    return {
        "dnbr_mean": float(np.nanmean(dnbr)),
        "dnbr_min": float(np.nanmin(dnbr)),
        "dnbr_max": float(np.nanmax(dnbr)),
        "severity_classes": classify_severity(dnbr),
        "burned_area_ha": estimate_burned_area(dnbr, resolution),
        "fetched_on": datetime.utcnow().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_postfire.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/postfire.py tests/test_postfire.py
git commit -m "feat: add post-fire dNBR severity classification and burned-area estimate"
```

---

### Task 9: Recovery monitoring

**Files:**
- Create: `analytics/recovery.py`
- Test: `tests/test_recovery.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recovery.py`:

```python
from unittest.mock import patch

import numpy as np

from analytics import recovery
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1", country="italy",
        bbox=[14.8, 37.5, 15.5, 38.2],
        centroid_lat=37.85, centroid_lon=15.15,
        start_date="2026-07-08", end_date="2026-07-14",
        prefire_metrics={"ndvi": 0.6},
    )


def _bands():
    bands = np.zeros((7, 2, 2), dtype=np.float32)
    bands[2] = 0.1   # RED
    bands[3] = 0.5   # NIR
    bands[6] = 1     # mask
    return bands


def test_recovery_offsets():
    assert recovery.RECOVERY_OFFSETS_MONTHS == [1, 3, 6, 9, 12]


def test_recovery_window():
    start, stop = recovery.recovery_window("2026-07-14", 1)
    assert start == "2026-07-19"
    assert stop == "2026-08-08"


def test_analyze_recovery_computes_regrowth_ratio():
    event = _event()
    with patch("analytics.recovery.fetch_bbox", return_value=_bands()):
        sample = recovery.analyze_recovery(event, 1, resolution=60)

    # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
    assert sample["offset_months"] == 1
    assert sample["ndvi"] == np.float64(0.6666666666666666)
    assert sample["regrowth_ratio"] == pytest.approx(1.1111111111111112)
    assert "fetched_on" in sample


def test_analyze_recovery_no_baseline():
    event = _event()
    event.prefire_metrics = {}
    with patch("analytics.recovery.fetch_bbox", return_value=_bands()):
        sample = recovery.analyze_recovery(event, 1, resolution=60)
    assert sample["regrowth_ratio"] is None
```

Add `import pytest` at top.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_recovery.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.recovery`.

- [ ] **Step 3: Implement**

Create `analytics/recovery.py`:

```python
from datetime import datetime, timedelta

import numpy as np
from sentinelhub import BBox, CRS

from data_pipeline.sentinel_request import fetch_bbox

from .prefire import compute_ndvi, mean_metric

RECOVERY_OFFSETS_MONTHS = [1, 3, 6, 9, 12]


def recovery_window(end_date: str, offset_months: int) -> tuple[str, str]:
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end + timedelta(days=30 * (offset_months - 1) + 5)
    stop = end + timedelta(days=30 * offset_months + 25)
    return (start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d"))


def recovery_due(event, offset_months: int, today) -> bool:
    end = datetime.strptime(event.end_date, "%Y-%m-%d").date()
    due_date = end + timedelta(days=30 * offset_months - 15)
    return due_date <= today


def analyze_recovery(event, offset_months: int, resolution: int = 60) -> dict:
    window = recovery_window(event.end_date, offset_months)
    bbox = BBox(event.bbox, crs=CRS.WGS84)
    bands = fetch_bbox(window, bbox, resolution=resolution)

    post_ndvi = mean_metric(compute_ndvi(bands))
    baseline = event.prefire_metrics.get("ndvi")
    regrowth = None
    if baseline and post_ndvi is not None:
        regrowth = float(post_ndvi / baseline)

    return {
        "offset_months": offset_months,
        "window": list(window),
        "ndvi": post_ndvi,
        "regrowth_ratio": regrowth,
        "fetched_on": datetime.utcnow().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_recovery.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/recovery.py tests/test_recovery.py
git commit -m "feat: add post-fire recovery NDVI sampling"
```

---

### Task 10: Event tracker (state machine driver)

**Files:**
- Create: `analytics/tracker.py`
- Test: `tests/test_tracker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracker.py`:

```python
from datetime import date
from unittest.mock import patch

from analytics.event import FireEvent
from analytics import tracker


def _event(status="detected"):
    return FireEvent(
        event_id="e1", country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55, centroid_lon=4.5,
        start_date="2026-07-12", end_date="2026-07-18",
        status=status, quiet_days=2,
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
        {"date": "2026-07-18", "frp_mw": 100.0, "detection_count": 5, "bbox_growth_deg": 0.1},
        {"date": "2026-07-19", "frp_mw": 60.0, "detection_count": 3, "bbox_growth_deg": 0.05},
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(event, resolution=60, quiet_after=3, today=date(2026, 7, 22))
    assert event.status == "ended"
    assert len(event.during_observations) == 2
    assert event.end_date == "2026-07-19"


def test_active_dedups_on_rerun():
    event = _event(status="active")
    event.during_observations = [
        {"date": "2026-07-18", "frp_mw": 100.0, "detection_count": 5, "bbox_growth_deg": 0.1},
    ]
    rows = [
        {"date": "2026-07-18", "frp_mw": 100.0, "detection_count": 5, "bbox_growth_deg": 0.1},
        {"date": "2026-07-19", "frp_mw": 60.0, "detection_count": 3, "bbox_growth_deg": 0.05},
    ]
    with patch("analytics.tracker.fetch_daily_observations", return_value=rows):
        tracker.process_event(event, resolution=60, quiet_after=3, today=date(2026, 7, 19))
    assert [o["date"] for o in event.during_observations] == ["2026-07-18", "2026-07-19"]


def test_ended_runs_postfire_and_recovers():
    event = _event(status="ended")
    with patch("analytics.tracker.analyze_postfire", return_value={"burned_area_ha": 1.0}):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 25))
    assert event.status == "recovering"
    assert event.postfire_assessment == {"burned_area_ha": 1.0}


def test_recovering_samples_due_months_and_completes():
    event = _event(status="recovering")
    samples = [{"offset_months": m, "ndvi": 0.5} for m in [1, 3, 6, 9, 12]]
    def fake_analyze_recovery(ev, month, resolution=60):
        return {"offset_months": month, "ndvi": 0.5}
    with patch("analytics.tracker.analyze_recovery", side_effect=fake_analyze_recovery):
        tracker.process_event(event, resolution=60, today=date(2027, 7, 30))
    assert event.status == "complete"
    assert len(event.recovery_samples) == 5


def test_failure_marks_failure_and_keeps_state():
    event = _event()
    with patch("analytics.tracker.analyze_prefire", side_effect=RuntimeError("no data")):
        tracker.process_event(event, resolution=60, today=date(2026, 7, 20))
    assert event.status == "detected"
    assert len(event.failures) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.tracker`.

- [ ] **Step 3: Implement**

Create `analytics/tracker.py`:

```python
from datetime import datetime, timedelta

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
        latest = event.during_observations[-1]["date"] if event.during_observations else None
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
            month for month in RECOVERY_OFFSETS_MONTHS
            if month not in sampled and recovery_due(event, month, today)
        ]
        for month in due:
            event.recovery_samples.append(
                analyze_recovery(event, month, resolution=resolution)
            )
        if RECOVERY_OFFSETS_MONTHS[-1] in sampled:
            event.transition("complete")


def process_event(event, resolution: int = 60, use_model: bool = False,
                  quiet_after: int = 3, today=None):
    """Run one tick for a single event. Returns the (mutated) event."""
    today = today or datetime.utcnow().date()
    try:
        _dispatch(event, resolution, use_model, quiet_after, today)
        event.failures = []
    except Exception as exc:  # phase failed; keep state, retry next tick
        event.failures.append({
            "status": event.status,
            "error": str(exc),
            "at": datetime.utcnow().isoformat(timespec="seconds"),
        })
    return event
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_tracker.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/tracker.py tests/test_tracker.py
git commit -m "feat: add event lifecycle tracker"
```

---

### Task 11: Report builder (HTML + JSON/CSV)

**Files:**
- Create: `analytics/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import json

from analytics.event import FireEvent
from analytics import report


def _event():
    return FireEvent(
        event_id="e1", country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55, centroid_lon=4.5,
        start_date="2026-07-12", end_date="2026-07-18",
        status="active",
        prefire_metrics={"ndvi": 0.5, "ndwi": -0.1, "weather_index": 55.0},
        during_observations=[
            {"date": "2026-07-18", "frp_mw": 100.0, "detection_count": 5, "bbox_growth_deg": 0.1},
        ],
    )


def test_export_json_round_trips():
    event = _event()
    restored = FireEvent.from_dict(report.export_json(event))
    assert restored.country == "france"
    assert restored.status == "active"


def test_export_csv():
    csv_text = report.export_csv(_event().during_observations)
    assert "date,frp_mw,detection_count,bbox_growth_deg" in csv_text
    assert "2026-07-18,100.0,5,0.1" in csv_text


def test_export_csv_empty():
    assert report.export_csv([]) == ""


def test_build_html_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_frp_chart", lambda event: "CHART_DATA")
    out = tmp_path / "e1.html"
    report.build_html_report(_event(), out)
    html = out.read_text()
    assert "france" in html
    assert "CHART_DATA" in html
    assert "Fire Radiative Power" in html
    assert "Recovery" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.report`.

- [ ] **Step 3: Implement**

Create `analytics/report.py`:

```python
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .event import FireEvent


def export_json(event: FireEvent) -> dict:
    return event.to_dict()


def export_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in columns))
    return "\n".join(lines)


def _frp_chart(event: FireEvent) -> str:
    obs = event.during_observations
    fig, ax = plt.subplots(figsize=(6, 3))
    dates = [o["date"] for o in obs]
    frp = [o["frp_mw"] for o in obs]
    if dates:
        ax.plot(dates, frp, marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("FRP (MW)")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Fire Radiative Power over time")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _metrics_table(event: FireEvent) -> str:
    pre = event.prefire_metrics or {}
    rows = [
        ("NDVI", pre.get("ndvi")),
        ("NDWI", pre.get("ndwi")),
        ("Weather index (0-100)", pre.get("weather_index")),
        ("Burned area (ha)", (event.postfire_assessment or {}).get("burned_area_ha")),
    ]
    cells = "".join(f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows)
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{cells}</table>"


def _severity_table(event: FireEvent) -> str:
    classes = (event.postfire_assessment or {}).get("severity_classes", {})
    cells = "".join(
        f"<tr><td>{name}</td><td>{fraction:.1%}</td></tr>"
        for name, fraction in classes.items()
    )
    return f"<table><tr><th>Class</th><th>Share</th></tr>{cells}</table>"


def _recovery_table(event: FireEvent) -> str:
    samples = event.recovery_samples
    if not samples:
        return "<p>No recovery samples yet.</p>"
    rows = "".join(
        f"<tr><td>{s['offset_months']}</td><td>{s.get('ndvi')}</td>"
        f"<td>{s.get('regrowth_ratio')}</td></tr>"
        for s in samples
    )
    return (
        "<table><tr><th>Months</th><th>NDVI</th><th>Regrowth ratio</th></tr>"
        f"{rows}</table>"
    )


def build_html_report(event: FireEvent, output_path) -> None:
    pre = event.prefire_metrics or {}
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Fire {event.event_id}</title></head>
<body>
<h1>Fire Report: {event.event_id}</h1>
<p><strong>Country:</strong> {event.country} &nbsp;
<strong>Status:</strong> {event.status} &nbsp;
<strong>Window:</strong> {event.start_date} → {event.end_date}</p>
<h2>Before: Fuel State</h2>
<p>NDVI {pre.get('ndvi')}, NDWI {pre.get('ndwi')}, weather index {pre.get('weather_index')}</p>
<h2>During: Fire Radiative Power</h2>
<img src="data:image/png;base64,{_frp_chart(event)}" alt="FRP over time">
<h2>After: Burn Severity</h2>
{_metrics_table(event)}
{_severity_table(event)}
<h2>Recovery</h2>
{_recovery_table(event)}
</body>
</html>
"""
    Path(output_path).write_text(html)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_report.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/report.py tests/test_report.py
git commit -m "feat: add HTML report builder and JSON/CSV exporters"
```

---

### Task 12: Seed script

**Files:**
- Create: `scripts/seed_pilot_events.py`

- [ ] **Step 1: Implement**

Create `scripts/seed_pilot_events.py`:

```python
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.event import FireEvent
from analytics.store import EventStore

# Replace these placeholders with real FIRMS events for the pilot.
# Format matches firm_request.cluster_detections() output.
PILOT_EVENTS = [
    {"country": "france", "bbox": [4.2, 44.3, 4.8, 44.8], "centroid_lat": 44.55, "centroid_lon": 4.5,
     "start_date": "2026-07-12", "end_date": "2026-07-18"},
    {"country": "france", "bbox": [-1.2, 44.3, -0.6, 44.9], "centroid_lat": 44.6, "centroid_lon": -0.9,
     "start_date": "2026-07-05", "end_date": "2026-07-11"},
    {"country": "spain", "bbox": [-1.3, 38.9, -0.6, 39.5], "centroid_lat": 39.2, "centroid_lon": -0.95,
     "start_date": "2026-07-15", "end_date": "2026-07-20"},
    {"country": "spain", "bbox": [-7.8, 42.3, -7.1, 42.9], "centroid_lat": 42.6, "centroid_lon": -7.45,
     "start_date": "2026-07-20", "end_date": "2026-07-25"},
    {"country": "italy", "bbox": [8.8, 39.7, 9.5, 40.4], "centroid_lat": 40.05, "centroid_lon": 9.15,
     "start_date": "2026-07-08", "end_date": "2026-07-14"},
    {"country": "italy", "bbox": [14.8, 37.5, 15.5, 38.2], "centroid_lat": 37.85, "centroid_lon": 15.15,
     "start_date": "2026-07-22", "end_date": "2026-07-28"},
    {"country": "greece", "bbox": [23.4, 37.8, 24.1, 38.4], "centroid_lat": 38.1, "centroid_lon": 23.75,
     "start_date": "2026-07-10", "end_date": "2026-07-16"},
    {"country": "greece", "bbox": [23.0, 38.4, 23.9, 39.1], "centroid_lat": 38.75, "centroid_lon": 23.45,
     "start_date": "2026-07-25", "end_date": "2026-07-30"},
]


def main():
    parser = argparse.ArgumentParser(description="Seed pilot fire events into the store")
    parser.add_argument("--root", default=None, help="Event store root (default: reports/events)")
    args = parser.parse_args()

    store = EventStore(args.root)
    count = 0
    for spec in PILOT_EVENTS:
        store.save_event(FireEvent(
            event_id=str(uuid4()),
            country=spec["country"],
            bbox=spec["bbox"],
            centroid_lat=spec["centroid_lat"],
            centroid_lon=spec["centroid_lon"],
            start_date=spec["start_date"],
            end_date=spec["end_date"],
        ))
        count += 1
    print(f"Seeded {count} pilot events into {store.root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs without network calls**

Run: `PYTHONPATH=. uv run python scripts/seed_pilot_events.py --root /tmp/sentinel_test_store`
Expected: `Seeded 8 pilot events into /tmp/sentinel_test_store` and 8 `.json` files created.

- [ ] **Step 3: Clean up the test store**

```bash
rm -rf /tmp/sentinel_test_store
```

- [ ] **Step 4: Commit**

```bash
git add scripts/seed_pilot_events.py
git commit -m "feat: add pilot event seed script"
```

---

### Task 13: Tracker runner script

**Files:**
- Create: `scripts/run_event_tracker.py`

- [ ] **Step 1: Implement**

Create `scripts/run_event_tracker.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from analytics.store import EventStore
from analytics.tracker import process_event


def main():
    parser = argparse.ArgumentParser(description="Run one tracker tick over all tracked events")
    parser.add_argument("--root", default=None, help="Event store root (default: reports/events)")
    parser.add_argument("--resolution", type=int, default=60, help="Sentinel resolution (m)")
    parser.add_argument("--use-model", action="store_true", help="Run Prithvi inference for post-fire")
    parser.add_argument("--quiet-after", type=int, default=3, help="Quiet days before marking ended")
    args = parser.parse_args()

    store = EventStore(args.root)
    events = store.list_events()
    print(f"Tracking {len(events)} events")

    for event in events:
        before = event.status
        process_event(
            event,
            resolution=args.resolution,
            use_model=args.use_model,
            quiet_after=args.quiet_after,
        )
        store.save_event(event)
        failures = f" ({len(event.failures)} failures)" if event.failures else ""
        print(f"  {event.event_id} [{event.country}]: {before} -> {event.status}{failures}")
        for failure in event.failures:
            print(f"    failure: {failure['error']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify help runs**

Run: `PYTHONPATH=. uv run python scripts/run_event_tracker.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_event_tracker.py
git commit -m "feat: add event tracker runner script"
```

---

### Task 14: Report build script

**Files:**
- Create: `scripts/build_reports.py`

- [ ] **Step 1: Implement**

Create `scripts/build_reports.py`:

```python
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
```

- [ ] **Step 2: Verify help runs**

Run: `PYTHONPATH=. uv run python scripts/build_reports.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_reports.py
git commit -m "feat: add report build script"
```

---

### Task 15: Full suite verification

**Files:** none (verification)

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: all existing + new tests PASS.

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3: Run format check**

Run: `PYTHONPATH=. uv run ruff format --check . --exclude notebooks`
Expected: no changes needed (run `make format` first if it reports diffs).

- [ ] **Step 4: Final commit (if lint/format changed anything)**

```bash
git add -A
git commit -m "chore: apply lint and format fixes"
```

---

## Self-Review

**Spec coverage:**
- Event model + lifecycle states (detected→active→ended→recovering→complete) → Task 3, Task 10 ✓
- JSON event store with thin migration seam → Task 4 ✓
- Pre-fire NDVI/NDWI + weather index → Task 5, Task 6 ✓
- During FIRMS FRP progression + growth → Task 7 ✓
- Post-fire dNBR severity + burned area → Task 8 ✓
- 12-month recovery NDVI sampling → Task 9 ✓
- HTML report + JSON/CSV exporters → Task 11 ✓
- Tracker (daily state machine driver) → Task 10 ✓
- Country bboxes (FR/ES/IT/GR) in firm_request + country tagging → Task 1 ✓
- Seed pilot fires + runner + report scripts → Tasks 12-14 ✓
- `reports/` gitignored → Task 2 ✓
- Tests throughout, mocks for external calls → every task ✓
- Error handling: phase failures recorded, retried next tick → tracker `_dispatch` try/except, `failures` field ✓

**Placeholder scan:** No TBD/TODO. Pilot event coordinates are documented as replaceable placeholders (this is intentional — real FIRMS events must come from the live API).

**Type consistency:** `FireEvent.from_dict`/`to_dict`, `EventStore.save_event`/`update_event`/`load_event`/`list_events`, `analyze_prefire`/`fetch_daily_observations`/`analyze_postfire`/`analyze_recovery`/`process_event` signatures are consistent across the tasks they're defined and consumed in. `recovery_due` and `RECOVERY_OFFSETS_MONTHS` match between Task 9 and Task 10.
