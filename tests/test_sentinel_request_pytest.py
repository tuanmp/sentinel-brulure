import numpy as np
from sentinelhub import CRS, BBox

from data_pipeline import sentinel_request as sr


def _make_event(bbox):
    return {
        "cluster_id": 1,
        "start_date": "2026-04-11",
        "end_date": "2026-04-11",
        "bbox": bbox,
    }


def _fake_bands(height=4, width=5, nir=0.8, swir2=0.2):
    bands = np.zeros((7, height, width), dtype=np.float32)
    bands[3] = nir
    bands[5] = swir2
    bands[6] = 1
    return bands


def test_fetch_bbox_small_area_single_request_path(monkeypatch):
    call_counter = {"count": 0}

    def fake_compute_split_bboxes(_bbox, resolution=10):
        return [[BBox([0.0, 0.0, 0.1, 0.1], crs=CRS.WGS84)]], 1, 1

    def fake_fetch_bands(time_interval, bbox, height, width, evalscript=sr.evalscript):
        call_counter["count"] += 1
        assert (height, width) == (4, 5)
        return _fake_bands(height=height, width=width, nir=0.7, swir2=0.2)

    def fake_fetch_large_bbox(_bbox, _time_interval, resolution=10, split_result=None):
        raise AssertionError("Large bbox path should not be used for 1x1 split")

    monkeypatch.setattr(sr, "compute_split_bboxes", fake_compute_split_bboxes)
    monkeypatch.setattr(sr, "bbox_to_dimensions", lambda _bbox, resolution=10: (5, 4))
    monkeypatch.setattr(sr, "fetch_bands", fake_fetch_bands)
    monkeypatch.setattr(sr, "fetch_large_bbox", fake_fetch_large_bbox)

    bbox = BBox([103.0, 19.0, 103.01, 19.01], crs=CRS.WGS84)
    bands = sr.fetch_bbox(("2026-04-10", "2026-04-11"), bbox, resolution=10)

    assert call_counter["count"] == 1
    assert bands.shape == (7, 4, 5)


def test_fetch_bbox_large_area_partitioned_into_subtiles(monkeypatch):
    large_counter = {"count": 0}

    subtile_grid = [
        [
            BBox([0.0, 0.0, 0.1, 0.1], crs=CRS.WGS84),
            BBox([0.0, 0.1, 0.1, 0.2], crs=CRS.WGS84),
        ],
        [
            BBox([0.1, 0.0, 0.2, 0.1], crs=CRS.WGS84),
            BBox([0.1, 0.1, 0.2, 0.2], crs=CRS.WGS84),
        ],
    ]

    def fake_compute_split_bboxes(_bbox, resolution=10):
        return subtile_grid, 2, 2

    def fake_fetch_large_bbox(_bbox, _time_interval, resolution=10, split_result=None):
        large_counter["count"] += 1
        assert split_result is not None
        _subtiles, rows, cols = split_result
        assert (rows, cols) == (2, 2)
        return _fake_bands(height=8, width=10, nir=0.9, swir2=0.1)

    def fake_fetch_bands(time_interval, bbox, height, width, evalscript=sr.evalscript):
        raise AssertionError("Single tile path should not be used for multi-split bbox")

    monkeypatch.setattr(sr, "compute_split_bboxes", fake_compute_split_bboxes)
    monkeypatch.setattr(sr, "fetch_large_bbox", fake_fetch_large_bbox)
    monkeypatch.setattr(sr, "fetch_bands", fake_fetch_bands)

    bbox = BBox([100.9, 17.9, 102.7, 19.4], crs=CRS.WGS84)
    bands = sr.fetch_bbox(("2026-04-10", "2026-04-11"), bbox, resolution=10)

    assert large_counter["count"] == 1
    assert bands.shape == (7, 8, 10)
