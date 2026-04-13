import io
import logging
import math
import os

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from sentinelhub import CRS, BBox, bbox_to_dimensions

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
MAX_DIM = 2500
RESOLUTION = 10    # metres — never change this
OVERLAP    = 0.1   # 10% overlap between sub-tiles

def extract_bands_from_response(response: requests.Response) -> np.ndarray:
    if response.status_code != 200:
        return None
    with rasterio.open(io.BytesIO(response.content)) as src:
        data = src.read()
    return data

def to_rgb(bands: np.ndarray):
    """Quick true-color preview from B04, B03, B02"""
    rgb = bands[[2, 1, 0]]  # R, G, B
    rgb = np.clip(rgb * 3.5, 0, 1)
    return rgb

def compute_split_bboxes(bbox: BBox, resolution=RESOLUTION, max_pixels=MAX_DIM):
    """
    Split a large BBox into a grid of non-overlapping sub-requests.
    Overlap is NOT needed here — we're just fetching pixels, not running a model.
    """
    min_lon, min_lat, max_lon, max_lat = (
        bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    )

    full_w, full_h = bbox_to_dimensions(bbox, resolution=resolution)

    print(f"Full bbox pixel dimensions at {resolution}m resolution: {full_w}×{full_h}")

    n_cols = math.ceil(full_w / max_pixels)
    n_rows = math.ceil(full_h / max_pixels)

    if n_cols == 1 and n_rows == 1:
        return [[bbox]], n_rows, n_cols

    tile_lon = (max_lon - min_lon) / n_cols
    tile_lat = (max_lat - min_lat) / n_rows

    partitioned_boxes = bbox.get_partition(num_x=n_cols, num_y=n_rows)

    print(partitioned_boxes)

    for i, col in enumerate(partitioned_boxes):
        for j, sub_bbox in enumerate(col):
            logging.info(f"Subtile ({i}, {j}): {sub_bbox}, pixel size: {bbox_to_dimensions(sub_bbox, resolution=resolution)}")

    # logging.info(f"Subtile pixel sizes: {[(bbox_to_dimensions(sub_bbox, resolution=resolution)) for sub_bbox in partitioned_boxes]}")

    return bbox.get_partition(num_x=n_cols, num_y=n_rows), n_rows, n_cols

    # sub_bboxes = []
    # for row in range(n_rows):
    #     for col in range(n_cols):
    #         left   = min_lon + col * tile_lon
    #         right  = min_lon + (col + 1) * tile_lon
    #         bottom = min_lat + row * tile_lat
    #         top    = min_lat + (row + 1) * tile_lat

    #         sub_bboxes.append(BBox([left, bottom, right, top], crs=CRS.WGS84))

    # logging.info(f"BBox split into {n_rows}×{n_cols} = {len(sub_bboxes)} sub-tiles")
    # logging.info(f"Subtile pixel sizes: {[(bbox_to_dimensions(sub_bbox, resolution=resolution)) for sub_bbox in sub_bboxes]}")

    # return sub_bboxes, n_rows, n_cols


def stitch_tiles(sub_images: list[list[np.ndarray]]):
    """
    Reassemble a grid of (H, W, C) arrays into one image.
    For overlapping tiles, averages the overlapping region.
    """
    # Stack into grid shape
    cols = []
    # for row in range(sub_images):
    for col in sub_images:
        # row_tiles = sub_images[row * n_cols : (row + 1) * n_cols]
        # print([tile.shape for tile in row_tiles])

        # Concatenate along width axis
        cols.append(np.concatenate(list(reversed(col)), axis=1))  # (C, H, W_total)

    # Concatenate along height axis
    stitched = np.concatenate(cols, axis=2)  # (C, H_total, W_total)
    return stitched

def save_geotiff(data, filename, bbox, crs="EPSG:4326"):
    """
    Save a (C, H, W) numpy array as a georeferenced GeoTIFF
    """
    if data.ndim == 2:
        data_to_write = data[np.newaxis, :, :]
    elif data.ndim == 3:
        data_to_write = data
    else:
        raise ValueError("Expected data with shape (C, H, W) or (H, W)")

    C, H, W = data_to_write.shape

    transform = from_bounds(
        bbox.min_x, bbox.min_y,
        bbox.max_x, bbox.max_y,
        W, H
    )
    with rasterio.open(
        filename, "w",
        driver="GTiff",
        height=H, width=W,
        count=C,
        dtype=data_to_write.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(C):
            dst.write(data_to_write[i], i + 1)