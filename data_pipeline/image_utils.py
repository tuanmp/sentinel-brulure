import io
import logging
import math
import os

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from sentinelhub import BBox, bbox_to_dimensions

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
MAX_DIM = 2500
RESOLUTION = 10
OVERLAP = 0.1


def extract_bands_from_response(response: requests.Response) -> np.ndarray:
    if response.status_code != 200:
        return None
    with rasterio.open(io.BytesIO(response.content)) as src:
        data = src.read()
    return data


def to_rgb(bands: np.ndarray):
    rgb = bands[[2, 1, 0]]
    rgb = np.clip(rgb * 3.5, 0, 1)
    return rgb


def compute_split_bboxes(bbox: BBox, resolution=RESOLUTION, max_pixels=MAX_DIM):
    full_w, full_h = bbox_to_dimensions(bbox, resolution=resolution)

    n_cols = math.ceil(full_w / max_pixels)
    n_rows = math.ceil(full_h / max_pixels)

    if n_cols == 1 and n_rows == 1:
        return [[bbox]], n_rows, n_cols

    partitioned_boxes = bbox.get_partition(num_x=n_cols, num_y=n_rows)

    for i, col in enumerate(partitioned_boxes):
        for j, sub_bbox in enumerate(col):
            logging.info(f"Subtile ({i}, {j}): {sub_bbox}")

    flat = [sub for col in partitioned_boxes for sub in col]
    return flat, n_rows, n_cols


def stitch_tiles(sub_images, n_rows=None, n_cols=None):
    if n_rows is not None and n_cols is not None:
        rows = []
        for r in range(n_rows):
            row_tiles = sub_images[r * n_cols : (r + 1) * n_cols]
            rows.append(np.concatenate(list(row_tiles), axis=1))
        return np.concatenate(rows, axis=0)

    cols = []
    for col in sub_images:
        cols.append(np.concatenate(list(reversed(col)), axis=1))
    return np.concatenate(cols, axis=2)


def save_geotiff(data, filename, bbox, crs="EPSG:4326"):
    if data.ndim == 2:
        data_to_write = data[np.newaxis, :, :]
    elif data.ndim == 3:
        data_to_write = data
    else:
        raise ValueError("Expected data with shape (C, H, W) or (H, W)")

    n_channels, height, width = data_to_write.shape

    transform = from_bounds(
        bbox.min_x, bbox.min_y,
        bbox.max_x, bbox.max_y,
        width, height,
    )
    with rasterio.open(
        filename, "w",
        driver="GTiff",
        height=height, width=width,
        count=n_channels,
        dtype=data_to_write.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(n_channels):
            dst.write(data_to_write[i], i + 1)
