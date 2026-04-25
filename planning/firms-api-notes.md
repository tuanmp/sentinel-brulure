# FIRMS API Integration Notes

## API Summary

Fetches near-real-time fire detections from NASA FIRMS to trigger Sentinel-2 imagery acquisition.

## Endpoint

```
GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{source}/{region}/{days_back}
```

| Param | Values | Notes |
|-------|--------|-------|
| `API_KEY` | string | From `firm_map_key` env var |
| `source` | `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT` | VIIRS near-real-time satellites |
| `region` | `world` or named region | Used `world` for global |
| `days_back` | 1–10 | API max is 10 days |

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