# Fire Event Analytics — Before / During / After

## Overview

Extend the existing burn-scar pipeline into a per-fire-event analytics system that
produces a near-real-time report covering the full fire lifecycle for fires in
France, Spain, Italy, and Greece.

For each fire event the system tracks and analyzes:

- **Before:** pre-fire vegetation state (NDVI/NDWI dryness) plus a weather-based
  fire-danger proxy (Open-Meteo, no API key required).
- **During:** FIRMS thermal progression — daily FRP, detection counts, and bbox
  growth over time.
- **After:** post-fire burn severity (dNBR classification) and burned-area estimate.
- **Recovery:** monthly NDVI sampling for up to ~12 months post-fire, compared
  against the pre-fire baseline as a regrowth ratio.

Output per event is both structured data (JSON/CSV) and a self-contained HTML
report. Start with a seeded pilot set of known fires per country, then expand to
automatic tracking via the existing FIRMS clustering pipeline.

## Architecture

**Approach A — event-centric pipeline with a lightweight JSON store.** Each fire
is a first-class event that transitions through lifecycle states. A daily
scheduler runs whatever work the current state requires; a report builder renders
accumulated analytics.

```
seed_pilot_events ──▶ event store (JSON per event)
                             ▲
        ┌────────────────────┴────────────────────┐
        │          tracker (daily tick)           │
        ▼              ▼              ▼           ▼
  detected        active         ended       recovering
  ├─ prefire:     ├─ during:     ├─ postfire: ├─ recovery:
  │  NDVI/NDWI    │  FIRMS FRP   │  dNBR      │  monthly NDVI
  │  + weather    │  time series │  severity   │  regrowth ratio
  │  index        │  growth rate │  + area     │  (~12 months)
  └───────────────┴──────────────┴────────────┴──────────────
                    │                        │
                    ▼                        ▼
         event store updated          status=complete
                    │
                    ▼
     report.py ──▶ HTML report + JSON/CSV per event
```

### Lifecycle states

| State | Condition to enter | Work run on each tick |
|-------|-------------------|----------------------|
| `detected` | seeded or FIRMS-clustered | pre-fire vegetation + weather snapshot → move to `active` |
| `active` | pre-fire done | poll FIRMS for event bbox, append daily FRP/area observations |
| `ended` | FIRMS quiet for N consecutive days (configurable, default 3) | post-fire dNBR severity + burned area (+ Prithvi mask if `--use-model`) |
| `recovering` | post-fire done | monthly NDVI sampling vs pre-fire baseline |
| `complete` | 12 months after end date | finalize |

### Migration seam

The event store is a narrow interface (`load_event`, `save_event`, `list_events`,
`update_event`) backed by JSON files. Swapping to SQLite/Postgres for the future
API touches only `store.py`. Event data is plain dict/JSON to keep the swap
trivial.

## Modules

New top-level `analytics/` package, sitting beside `data_pipeline/` and reusing it.

| Module | Responsibility | Key outputs |
|--------|---------------|-------------|
| `event.py` | `FireEvent` model + lifecycle transitions | `status`, phase timestamps |
| `store.py` | JSON event store, thin interface (migration seam) | `load/save/list/update_event` |
| `prefire.py` | NDVI/NDWI dryness + weather index for the event bbox/window | `prefire_metrics` dict |
| `during.py` | FIRMS FRP progression + growth rate | `during_observations` list (daily rows) |
| `postfire.py` | dNBR severity + burned-area estimate | `postfire_assessment` dict |
| `recovery.py` | monthly NDVI vs pre-fire baseline | `recovery_samples` list |
| `fire_weather.py` | Open-Meteo fire-danger proxy fetcher | per-day weather index rows |
| `report.py` | HTML report + JSON/CSV exporters | `reports/{id}.html`, `.json`, `.csv` |
| `tracker.py` | daily state machine driver | updates event store |

### Scripts

- `scripts/seed_pilot_events.py` — hardcodes ~5-8 known recent fires across FR/ES/IT/GR into the store
- `scripts/run_event_tracker.py` — runs one daily tick over all tracked events
- `scripts/build_reports.py` — regenerates reports from the store

### Data shapes

```
FireEvent {
  event_id: uuid4, country, cluster_id,
  bbox: [min_lon, min_lat, max_lon, max_lat], centroid,
  start_date, end_date (FIRMS first/last detection),
  status: detected|active|ended|recovering|complete,
  prefire_metrics: { ndvi, ndwi, weather_index, fetched_on },
  during_observations: [ { date, frp_mw, detection_count, bbox_growth_deg } ],
  postfire_assessment: { dnbr, severity_classes, burned_area_ha },
  recovery_samples: [ { offset_months, ndvi, regrowth_ratio, fetched_on } ]
}
```

### Reuse over rebuild

- `prefire.py` and `postfire.py` call existing `sentinel_request.fetch_bbox` / `process_fire_event`
- `during.py` calls existing `firm_request.fetch_fire_events` filtered to the event bbox
- Burn-severity classification reuses the dNBR already computed by `compute_nbr`

### Weather source

Open-Meteo Archive API for the pilot — free, no API key, provides temperature,
wind, precipitation, and soil moisture for any lat/lon, from which a simple
fire-danger proxy is derived. EFFIS/ERA5 FWI can slot into `fire_weather.py`
behind the same interface later.

### Country scoping

Add country-level bboxes (FR/ES/IT/GR) to `firm_request.py` so events get tagged
with a `country` and FIRMS polling can be filtered per country.

## Error handling

- Each phase runs independently — a failure in one (e.g. cloudy post-fire image)
  marks that attempt as `failed` with a reason and does not block other phases or
  crash the tick.
- Failed phases are retried on the next daily tick (idempotent, safe to re-run).
- API/network errors in FIRMS/Sentinel/Open-Meteo are caught, logged, and retried.

### Idempotency

Every observation is keyed by `(event_id, phase, date)` — re-running a tick never
duplicates rows.

## Testing

pytest, matching existing style in `tests/test_data_pipeline.py`:

- State machine: transition logic, `N-days-quiet` rule
- `prefire.py`: NDVI/NDWI math with synthetic bands
- `postfire.py`: dNBR severity thresholds + burned-area pixel→hectare calc
- `report.py`: golden JSON export, HTML generated with expected sections
- `during.py`: FIRMS accumulation + dedup on re-run
- `store.py`: round-trip save/load, pure dict/JSON shape (verifies future SQLite swap)

Mocks for all external calls.

## Repo layout

```
analytics/
  __init__.py
  event.py  store.py  prefire.py  during.py
  postfire.py  recovery.py  fire_weather.py
  report.py  tracker.py
scripts/
  seed_pilot_events.py
  run_event_tracker.py
  build_reports.py
reports/            # gitignored
  events/{event_id}.json
  reports/{event_id}.html / .json / .csv
data_pipeline/      # firm_request.py gains country bboxes
tests/
  test_event_lifecycle.py
  test_prefire.py
  test_postfire.py
  test_during.py
  test_recovery.py
  test_report.py
  test_store.py
```

## Build order

1. event model + store
2. prefire
3. during
4. postfire
5. recovery
6. tracker
7. report builder
8. seed script
9. tests throughout

## Future API

The JSON store interface and structured outputs are designed so a FastAPI backend
(`GET /events`, `GET /events/{id}/report`) can be layered on top later with only
`store.py` and `report.py` touched.
