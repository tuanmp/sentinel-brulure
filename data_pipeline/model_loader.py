"""
Download and cache Prithvi model files from HuggingFace.
"""
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar"
CHECKPOINT_NAME = "Prithvi_EO_1.0_100M_BurnScars.pt"
CONFIG_URL = (
    "https://raw.githubusercontent.com/NASA-IMPACT/hls-foundation-os/"
    "main/configs/burn_scars.py"
)

def get_model_dir():
    return Path.home() / ".cache" / "sentinel-sat" / "models"


def download_model_files():
    """Download checkpoint and config, return paths."""
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoint from HuggingFace
    ckpt_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_NAME,
        cache_dir=str(model_dir),
    )

    # Download config from GitHub (not in HF repo)
    config_path = model_dir / "burn_scars.py"
    if not config_path.exists():
        resp = requests.get(CONFIG_URL)
        resp.raise_for_status()
        config_path.write_text(resp.text)

    return ckpt_path, str(config_path)
