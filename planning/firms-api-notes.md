# FIRMS API Integration Notes

## API Summary

Fetches near-real-time fire detections from NASA FIRMS to trigger Sentinel-2 imagery acquisition.

## Endpoint

```
GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{source}/{region}/{days_back}
GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{source}/{region}/{days_back}/{date}
```

The optional trailing `{date}` (YYYY-MM-DD) **anchors the window**: it returns
detections for `[date, date + days_back)`. Without it, you get the most recent
`days_back` days. This is how the during-phase pages back through arbitrary
history for month-long fires — in 5-day chunks.

| Param | Values | Notes |
|-------|--------|-------|
| `API_KEY` | string | From `firm_map_key` env var |
| `source` | `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`, `MODIS_NRT` (+ `*_SP` archives) | See sources below |
| `region` | `world`, named region, or `west,south,east,north` bbox | `france`, `spain`, `italy`, `greece` supported |
| `days_back` | 1–5 | Per-request max is **5 days** (not 10); page with the `date` anchor to go further back |

## Data Sources and Archives

Each NRT source covers only the most recent ~3 months. The paired Standard
Processing (`_SP`) archive reaches back years, so the full history is seamless:

| Source | Archive coverage |
|--------|------------------|
| `MODIS_SP` | 2000-11-01 → present (~3 mo latency) |
| `VIIRS_SNPP_SP` | 2012-01-20 → present (~3 mo latency) |
| `VIIRS_NOAA20_SP` | 2018-04-01 → present |
| `VIIRS_SNPP_NRT` (etc.) | ~last 3 months |

`get_data_availability(source)` returns `(min_date, max_date)` per source and is
cached; `pick_source(window_start, window_end, source)` prefers the NRT source
when it covers the window and falls back to the `_SP` archive otherwise.

## Returned Fields (CSV)

| Field | Description |
|-------|-------------|
| `latitude`, `longitude` | Detection coordinates |
| `acq_date`, `acq_time` | Acquisition date/time |
| `bright_ti4`, `bright_ti5` | Brightness temp (K) — fire proxy |
| `frp` | Fire Radiative Power (MW) — fire intensity |
| `confidence` | `l`=low, `n`=nominal, `h`=high |
| `scan`, `track` | Pixel resolution |
| `satellite` | N = NOAA |
| `instrument` | VIIRS |
| `daynight` | D=day, N=night |

## Post-Processing Pipeline

1. **Filter by confidence** — `l`, `n`, or `h` (mapped to low/nominal/high)
2. **Convert to GeoDataFrame** — points with EPSG:4326 CRS
3. **Cluster with DBSCAN** — groups nearby detections into fire events
   - `eps=10km` (converted to radians for haversine)
   - `min_samples=5` (filters noise)
4. **Build event dict** per cluster:
   - `cluster_id`, `detection_count`, `total_frp_mw`
   - `start_date`, `end_date` (date range of detections)
   - `bbox` with 0.1° buffer (~10km)
   - `centroid_lat`, `centroid_lon`
5. **Filter by quality** — min FRP, min detections, max bbox size

## Current Status

- Logic is **exploratory** — in `notebooks/firm.ipynb`
- `data_pipeline/firm_request.py` is an **empty stub**
- `run_sentinel_demo.py` has a **hardcoded event** instead of using FIRMS data

## Expected Output Format

Fire events dict format (matches `process_fire_event` in `sentinel_request.py`):

```python
{
    "cluster_id": int,
    "detection_count": int,
    "total_frp_mw": float,
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "bbox": [min_lon, min_lat, max_lon, max_lat],
    "centroid_lat": float,
    "centroid_lon": float,
}
```

## References

- FIRMS API: https://firms.modaps.eosdis.nasa.gov/api
- Documentation: https://firms.modaps.eosdis.nasa.gov/userguidetechnotes.pdf