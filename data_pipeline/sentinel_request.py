import os

from dotenv import load_dotenv

load_dotenv("../.env")

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import requests

try:
    from .image_utils import (
        compute_split_bboxes,
        extract_bands_from_response,
        stitch_tiles,
        to_rgb,
    )
except ImportError:
    from image_utils import (
        compute_split_bboxes,
        extract_bands_from_response,
        stitch_tiles,
        to_rgb,
    )
try: 
    from .sentinel_utils import compute_nbr
except ImportError:
    from sentinel_utils import compute_nbr

from matplotlib import pyplot as plt
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from sentinelhub import CRS, BBox, bbox_to_dimensions

# sys.path.append(".")


MAX_RETRIES = 3
MAX_DIM = 2500
RESOLUTION = 10    # metres — never change this
OVERLAP    = 0.1   # 10% overlap between sub-tiles
SAFE_MARGIN = 60  # seconds before token expiry to consider it "expired"
ENV_ACCESS_TOKEN = "sentinel_access_token"
ENV_TOKEN_EXPIRY = "sentinel_token_expiry"

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04","B08","B11","B12","dataMask"],
    output: { bands: 7, sampleType: "FLOAT32" }
  }
}

function evaluatePixel(sample) {
        return [
            sample.B02,   // Blue
            sample.B03,   // Green
            sample.B04,   // Red
            sample.B08,   // NIR
            sample.B11,   // SWIR-1
            sample.B12,   // SWIR-2
            sample.dataMask  // 0 = no data, 1 = valid
        ];
    }
"""

def fetch_token():
    assert os.getenv("sentinel_client_id") is not None, "Missing Sentinel API client ID in environment variables"
    assert os.getenv("sentinel_client_secret") is not None, "Missing Sentinel API client secret in environment variables"
    client = BackendApplicationClient(client_id=os.getenv("sentinel_client_id"))
    oauth = OAuth2Session(client=client)

    num_retries = 0
    while num_retries < MAX_RETRIES:
        try:
            token = oauth.fetch_token(
                token_url=os.getenv("sentinel_token_url"),
                client_secret=os.getenv("sentinel_client_secret"),
                include_client_id=True,
            )
            os.environ[ENV_ACCESS_TOKEN] = token["access_token"]
            os.environ[ENV_TOKEN_EXPIRY] = str(token.get("expires_at", 0))
            return token
        except Exception as e:
            logging.warning(f"Error fetching token: {e}")
            num_retries += 1
            logging.warning(f"Retrying... ({num_retries}/{MAX_RETRIES})")

    return None

def fetch_large_bbox(
    bbox,
    time_interval,
    resolution=RESOLUTION,
    split_result=None,
):
    """
    Fetch a bbox of any size at fixed resolution by splitting
    into sub-tiles, fetching each, then stitching.
    """
    if split_result is None:
        sub_bboxes, n_rows, n_cols = compute_split_bboxes(bbox, resolution)
    else:
        sub_bboxes, n_rows, n_cols = split_result

    sub_images = []
    # for i, sub_bbox in enumerate(sub_bboxes):
    #     logging.info(f"  Fetching sub-tile {i+1}/{len(sub_bboxes)}...")
    #     bands = fetch_bands(time_interval, bbox=sub_bbox, resolution=resolution)
    #     sub_images.append(bands)  # each: (C, H_i, W_i)
    size=None
    for i, col in enumerate(sub_bboxes):
        this_col = []
        for j, sub_bbox in enumerate(col):
            if size is None: 
                w, h = bbox_to_dimensions(sub_bbox, resolution=resolution)
                size = (h, w)
                logging.info(f"Sub-tile pixel size: {size}")
            logging.info(f"  Fetching sub-tile ({i+1}, {j+1})/{(n_rows, n_cols)}...")
            bands = fetch_bands(time_interval, bbox=sub_bbox, height=h, width=w)
            this_col.append(bands)  # each: (C, H_i, W_i)
        sub_images.append(this_col)

    # Stitch sub-tiles back into one image
    stitched = stitch_tiles(sub_images)
    return stitched


def fetch_bbox(time_interval, bbox, resolution=RESOLUTION):
    """Fetch a bbox by selecting single-tile or multi-subtile strategy."""
    split_result = compute_split_bboxes(bbox, resolution)
    sub_bboxes, n_rows, n_cols = split_result

    if n_rows == 1 and n_cols == 1:
        width, height = bbox_to_dimensions(bbox, resolution=resolution)
        return fetch_bands(time_interval, bbox=bbox, height=height, width=width)

    return fetch_large_bbox(
        bbox,
        time_interval,
        resolution=resolution,
        split_result=split_result,
    )

def fetch_bands(time_interval, bbox, height, width, evalscript=evalscript):
    """Assume that the bbox is always small enough to fetch

    Args:
        time_interval (_type_): _description_
        bbox (_type_): _description_
        resolution (int, optional): _description_. Defaults to RESOLUTION.
    """
    data = make_json(bbox, time_interval[0], time_interval[1], height, width, evalscript)
    response = make_request(data)
    bands = extract_bands_from_response(response)
    if bands is None:
        raise RuntimeError("Failed to fetch bands from Sentinel response")
    logging.info("Fetched bands with shape: %s", bands.shape)
    return bands


def _token_from_env():
    access_token = os.getenv(ENV_ACCESS_TOKEN)
    expires_at_str = os.getenv(ENV_TOKEN_EXPIRY, "0")
    if not access_token:
        return None

    try:
        expires_at = float(expires_at_str)
    except (TypeError, ValueError):
        expires_at = 0

    return {
        "access_token": access_token,
        "expires_at": expires_at,
    }

def make_json(bbox, start, end, height, width, evalscript=evalscript):

    # size = bbox_to_dimensions(bbox, resolution=resolution)
    input_field =dict(
        bounds=dict(
            bbox=[bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y],
            properties=dict(
                crs='http://www.opengis.net/def/crs/OGC/1.3/CRS84'
            )
        ),
        data=[
            {
                "dataFilter": {
                "timeRange": {
                    "from": f"{start}T00:00:00Z",
                    "to": f"{end}T23:59:59Z"
                    },
                    "mosaickingOrder": "leastRecent"
                },
                "type": "sentinel-2-l2a"
            }
        ]
    )

    outputField = dict(
        height=height,
        width=width,
        responses=[
            dict(
                identifier="default",
                format=dict(
                    type="image/tiff"
                )
            )
        ]
    )

    return {
        "input": input_field,
        "output": outputField,
        "evalscript": evalscript
    }

def make_request(json_body: dict):
    token = _token_from_env()

    if token is None or time.time() > token.get("expires_at", 0) - SAFE_MARGIN:
        logging.warning("Token is expired or about to expire, fetching a new one...")
        token = fetch_token()

    if token is None:
        raise Exception("Failed to fetch access token for Sentinel API")
    
    # print(token["access_token"][:100])

    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Content-Type": "application/json"
    }

    # print(json_body)
    # print(headers)
    
    num_retries = 0
    while num_retries < MAX_RETRIES:
        try:
            response = requests.post(
                url=os.getenv("sentinel_request_url"),
                headers=headers,
                json=json_body,
            )
            response.raise_for_status()  # Raise an error for bad status codes
            return response
        except Exception as e:
            logging.warning(f"Error making request: {e}")
            num_retries += 1
            logging.warning(f"Retrying... ({num_retries}/{MAX_RETRIES})")
    return response

def process_fire_event(event, resolution=RESOLUTION):
    """
    Given a fire event dict from FIRMS, fetch pre and post
    Sentinel-2 tiles and compute dNBR.
    """
    bbox = BBox(event["bbox"], crs=CRS.WGS84)

    # Date windows: 30 days before fire start, fire end → +15 days after
    fire_start = datetime.strptime(event["start_date"], "%Y-%m-%d")
    fire_end   = datetime.strptime(event["end_date"],   "%Y-%m-%d")

    pre_window  = (
        (fire_start - timedelta(days=15)).strftime("%Y-%m-%d"),
        (fire_start - timedelta(days=1)).strftime("%Y-%m-%d")
    )
    post_window = (
        fire_end.strftime("%Y-%m-%d"),
        (fire_end + timedelta(days=15)).strftime("%Y-%m-%d")
    )

    logging.info(f"\nProcessing event #{event['cluster_id']}")
    logging.info(f"  Pre-fire window:  {pre_window}")
    logging.info(f"  Post-fire window: {post_window}")
    logging.info(f"  BBox: {event['bbox']}")


    # json_request = make_json(bbox, pre_window[0], pre_window[1], evalscript)

    # Reuse fetch_bands() from Step 2
    # pre_bands  = extract_bands(
    #     requests.post(
    #         os.getenv("sentinel-request-url"),
    #         headers={"Authorization": f"Bearer {token['access_token']}", "Content-Type": "application/json"},
    #         json=make_json(bbox, pre_window[0], pre_window[1], evalscript, 60)
    #     )
    # )

    pre_bands = fetch_bbox(pre_window, bbox, resolution=resolution)
    # post_bands = extract_bands(
    #     requests.post(
    #         os.getenv("sentinel-request-url"),
    #         headers={"Authorization": f"Bearer {token['access_token']}", "Content-Type": "application/json"},
    #         json=make_json(bbox, post_window[0], post_window[1], evalscript, 60)
    #     )
    # )
    post_bands = fetch_bbox(post_window, bbox, resolution=resolution)

    nbr_pre  = compute_nbr(pre_bands)
    nbr_post = compute_nbr(post_bands)
    dnbr     = nbr_pre - nbr_post

    # Stack into 8-channel tensor ready for the model
    post_8ch = np.concatenate(
        [post_bands, dnbr[np.newaxis, :, : ].astype(np.float32)],
        axis=0
    )

    return {
        "event":      event,
        "pre_bands":  pre_bands,
        "post_bands": post_bands,
        "dnbr":       dnbr,
        "tensor":     post_8ch,   # shape (8, H, W) — model-ready
    }

def plot_event_result(result, output_path="rhodes_dnbr_lowres.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(to_rgb(result["pre_bands"]).transpose(1, 2, 0))
    axes[0].set_title("Pre-fire (RGB)")
    axes[0].axis("off")

    axes[1].imshow(to_rgb(result["post_bands"]).transpose(1, 2, 0))
    axes[1].set_title("Post-fire (RGB)")
    axes[1].axis("off")

    dnbr_plot = axes[2].imshow(result["dnbr"], cmap="RdYlGn_r", vmin=-0.2, vmax=0.8)
    axes[2].set_title("dNBR (burn severity)")
    axes[2].axis("off")
    plt.colorbar(dnbr_plot, ax=axes[2], fraction=0.046, pad=0.04, label="dNBR value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
