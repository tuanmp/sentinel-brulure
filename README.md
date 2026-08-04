# Wildfire Burn Scar Detector

Fetch Sentinel-2 satellite imagery, compute NBR/dNBR burn indices, and prepare ML-ready tensors for burn scar segmentation.

## Status

Data pipeline is working. Model fine-tuning, backend, and frontend are planned.

## Setup

```bash
cp .env_template .env   # add your Copernicus API credentials
uv sync --group dev     # install dependencies + ruff
```

## Development

```bash
make lint    # ruff check
make format  # ruff format
make test    # pytest tests/
make sync    # uv sync
```

## Run Demo

```bash
PYTHONPATH=. uv run python scripts/run_sentinel_demo.py
```

## What It Does

1. **Fetch Sentinel-2 L2A tiles** via Copernicus Data Space API (OAuth2)
2. **Cloud filtering** via server-side evalscript
3. **Auto-split large areas** into 2500×2500px sub-tiles, stitch on return
4. **Compute NBR**: `(NIR − SWIR2) / (NIR + SWIR2)`
5. **Compute dNBR** between pre- and post-fire windows
6. **Return 8-channel tensor** ready for model inference

## Project Structure

```
data_pipeline/
  sentinel_request.py   # API, OAuth, bbox fetching
  sentinel_utils.py     # NBR computation
  image_utils.py        # bbox split, stitch, GeoTIFF
  firm_request.py      # TODO — FIRMS integration
scripts/
  run_sentinel_demo.py  # demo entry point
```

## Required Environment Variables

| Variable | Description |
|----------|-------------|
| `sentinel_client_id` | Copernicus Data Space OAuth client ID |
| `sentinel_client_secret` | OAuth client secret |
| `sentinel_token_url` | Token endpoint URL |
| `sentinel_request_url` | Process API endpoint |

See `.env_template` for the full list.

## Roadmap

- [x] Data pipeline (Sentinel-2 fetching, NBR/dNBR)
- [x] FIRMS integration (fire event polling)
- [x] Prithvi-EO-2.0-300M burn scar inference + evaluation harness
- [ ] Prithvi fine-tuning on Sentinel-2 data (designed, not run)
- [ ] FastAPI backend
- [ ] React frontend with live map

## Deployment

- **Local:** `docker compose up --build` then open http://localhost:8501
- **Live:** deploy the repo to Streamlit Community Cloud
  (main file `dashboard/app.py`). Add your API keys under
  Settings → Secrets (keys must match `.env_template`).

## Burn Scar Model

- Inference: `data_pipeline/model_inference_v2.py` (Prithvi-EO-2.0-300M)
- Validate vs official: `PYTHONPATH=. uv run python scripts/validate_prithvi_v2.py`
- Benchmark: `PYTHONPATH=. uv run python scripts/run_evaluation.py`
  (HLS test split: V2-300M IoU/Dice, plus dNBR cross-check)
- MLflow: `PYTHONPATH=. uv run python scripts/track_eval_mlflow.py [limit]`