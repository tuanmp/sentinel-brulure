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
