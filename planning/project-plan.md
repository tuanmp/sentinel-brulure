# Wildfire Burn Scar Detector — Project Plan

## Overview

A personal project using publicly available satellite imagery and a fine-tuned geospatial AI model to detect and visualize wildfire burn scars in real-time via a web interface.

-----

## Architecture Overview

```
Sentinel-2 imagery (Copernicus API)
        ↓
  Preprocessing pipeline
        ↓
  Prithvi fine-tuned model
        ↓
  Segmentation mask (burn scar)
        ↓
  FastAPI backend → React frontend (live map)
```

-----

## 1. Data Pipeline

### Sources

- **Sentinel-2 L2A** — surface reflectance tiles via [Copernicus Data Space](https://dataspace.copernicus.eu/) (free, API available)
- **NASA FIRMS** — active fire alerts as a trigger to know where/when to pull new imagery
- **HLS Burn Scar Dataset** — fine-tuning labels ([HuggingFace](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars))

### Key Bands

|Band         |Description|Role             |
|-------------|-----------|-----------------|
|B02, B03, B04|RGB        |Visual context   |
|B08          |NIR        |Vegetation health|
|B11, B12     |SWIR       |Burn sensitivity |

### Pipeline Steps

1. Poll NASA FIRMS daily for new fire events
1. Query Copernicus API for Sentinel-2 tiles covering those coordinates
1. Filter for cloud cover < 20%
1. Compute NBR index: `(NIR - SWIR) / (NIR + SWIR)` — strong burn indicator
1. Stack bands + NBR into a 7-channel tensor
1. Tile into 224×224 patches (Prithvi’s expected input size)

### Tools

- `sentinelhub-py` or `pystac-client` — imagery retrieval
- `rasterio` + `numpy` — preprocessing

-----

## 2. Model: Fine-tuning Prithvi

### Why Prithvi

- Pretrained by IBM + NASA on Sentinel-2 & Landsat imagery
- Open on HuggingFace: `ibm-nasa-geospatial/Prithvi-100M`
- Has an existing burn scar segmentation config to build directly on

### Fine-tuning Setup

```python
from transformers import AutoModel

backbone = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")

# Attach a segmentation head (e.g. UperNet or simple Conv decoder)
model = SegmentationModel(backbone, num_classes=2)  # burn / no-burn

# Fine-tune on HLS Burn Scar dataset
# ~800 labeled scenes, split 80/10/10
# Loss: Dice + BCE (handles class imbalance well)
# Metric: IoU on burn class
```

### Training Details

- **Hardware:** Single A100 — Google Colab Pro+ or Lambda/RunPod (~$10–20 total)
- **Duration:** ~20–30 epochs, a few hours of GPU time
- **Target metric:** IoU > 0.75 on burn class

-----

## 3. Backend

### Stack

Python + FastAPI

### Endpoints

|Method|Route     |Description                                                     |
|------|----------|----------------------------------------------------------------|
|`GET` |`/events` |Returns list of recent fire events (from FIRMS cache)           |
|`POST`|`/predict`|Accepts `{lat, lon, date}`, returns GeoTIFF mask + overlay tiles|

### Prediction Flow

1. Receive coordinates + date
1. Fetch & preprocess Sentinel-2 tile (or serve from cache)
1. Run model inference → binary segmentation mask
1. Convert mask to GeoJSON or PNG overlay
1. Return to frontend

### Caching

Store preprocessed tiles + predictions in S3 or Cloudflare R2 (free tier) to avoid re-fetching the same tile twice.

### Deployment

Docker container on Railway, Render, or a small AWS EC2 instance.

-----

## 4. Frontend

### Stack

React + MapLibre GL (open-source Mapbox alternative)

### Key UI Features

- Base map with recent FIRMS fire event markers
- Click an event → triggers prediction → burn scar overlay appears on the map
- Before/after toggle slider (pre-fire vs post-fire Sentinel-2 true color)
- Confidence heatmap mode (show model probability, not just binary mask)

### Libraries

|Library                                  |Purpose                           |
|-----------------------------------------|----------------------------------|
|`maplibre-gl`                            |Map rendering                     |
|`react-map-gl`                           |React wrapper                     |
|`georaster-layer-for-leaflet` / `deck.gl`|Render GeoTIFF overlays in browser|

-----

## 5. Build Timeline

|Week|Milestone                                                                         |
|----|----------------------------------------------------------------------------------|
|1   |Data pipeline working — can pull & preprocess Sentinel-2 tiles for a given lat/lon|
|2   |Fine-tuning complete — model hits >0.75 IoU on validation set                     |
|3   |FastAPI backend serving predictions end-to-end                                    |
|4   |React frontend with map + overlay rendering                                       |
|5   |Polish, caching, and deployment                                                   |


> **Note:** The trickiest part in practice is the data pipeline — specifically handling cloud cover and tile alignment. Get this solid before touching the model.

-----

## 6. Key Resources

|Resource                        |URL                                                                                                                             |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
|Prithvi model + burn scar config|[huggingface.co/ibm-nasa-geospatial](https://huggingface.co/ibm-nasa-geospatial)                                                |
|HLS Burn Scar dataset           |[huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)|
|Sentinel-2 access               |[dataspace.copernicus.eu](https://dataspace.copernicus.eu)                                                                      |
|FIRMS fire data API             |[firms.modaps.eosdis.nasa.gov/api](https://firms.modaps.eosdis.nasa.gov/api)                                                    |
