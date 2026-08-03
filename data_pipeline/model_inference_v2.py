"""Prithvi-EO-2.0-300M burn scar inference (TerraTorch-backed).

Drop-in for the legacy 100M module. Loads the fine-tuned V2-300M checkpoint
with LightningInferenceModel and runs sliding-window 512x512 inference,
returning softmax probabilities and an argmax class mask.
"""

from pathlib import Path

import torch

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
CHECKPOINT_NAME = "Prithvi_EO_V2_300M_BurnScars.pt"
PATCH_SIZE = 512
NUM_BANDS = 6

CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "prithvi_v2_burn_scars_inference.yaml"
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_dir() -> Path:
    return Path.home() / ".cache" / "sentinel-sat" / "models"


def download_checkpoint(cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CHECKPOINT_NAME,
            cache_dir=str(cache_dir or get_model_dir()),
        )
    )


def load_model(checkpoint_path: Path | None = None, device: str | None = None):
    """Load the fine-tuned V2-300M model via LightningInferenceModel.

    Returns the LightningInferenceModel wrapper exposing `.model`
    (the SemanticSegmentationTask) and `.datamodule` (for transforms).
    """
    from terratorch.cli_tools import LightningInferenceModel

    if checkpoint_path is None:
        checkpoint_path = download_checkpoint()
    device = device or pick_device()
    lightning_model = LightningInferenceModel.from_config(CONFIG_PATH, checkpoint_path)
    lightning_model.model.eval()
    lightning_model.model.to(device)
    return lightning_model
