# Agent Instructions — sentinel-sat

## Setup

```bash
uv sync
uv sync --group dev    # includes ruff for linting
cp .env_template .env  # then add your API credentials
```

## Dev Commands

```bash
make lint    # ruff check
make format  # ruff format
make test    # pytest tests/
make sync    # uv sync
make dev     # uv sync --group dev
```

## Run Demo

```bash
uv run python scripts/run_sentinel_demo.py
```

## Architecture

```
data_pipeline/
  sentinel_request.py  # Sentinel API, OAuth token mgmt, bbox fetching
  sentinel_utils.py   # NBR computation: (NIR - SWIR2) / (NIR + SWIR2)
  image_utils.py      # bbox splitting, tiling, stitch, GeoTIFF export
  firm_request.py     # STUB — FIRMS integration not implemented yet
  __init__.py
scripts/
  run_sentinel_demo.py  # Demo entry point
```

## Required Env Vars

```
sentinel_client_id      # Copernicus Data Space OAuth client ID
sentinel_client_secret  # OAuth client secret
sentinel_token_url      # Token endpoint URL
sentinel_request_url    # Process API endpoint
```

## Key Design Patterns

- **Token caching:** Access token stored in `os.environ`, refreshed 60s before expiry
- **Large bbox handling:** Areas > 2500px auto-split into sub-tiles, stitched before return
- **Cloud filtering:** Done server-side via embedded JavaScript evalscript (not post-fetch)
- **Dual import style:** `try: from .module import X; except ImportError: from module import X` — allows package and standalone execution

## Gotchas

- `firm_request.py` is an empty stub — FIRMS polling not yet implemented
- `make lint` and `make format` exclude notebooks (they have lint issues that are ignored)
- Tests in `tests/` use both pytest (explicit) and unittest (implicit via inheritance) styles
- `.env` is gitignored — never commit API credentials