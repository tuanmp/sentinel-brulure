"""Prithvi-EO-2.0-300M burn scar inference (TerraTorch-backed).

Drop-in for the legacy 100M module. Loads the fine-tuned V2-300M checkpoint
with LightningInferenceModel and runs sliding-window 512x512 inference,
returning softmax probabilities and an argmax class mask.
"""

from pathlib import Path

import numpy as np
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


def scale_bands(bands: np.ndarray) -> np.ndarray:
    """Convert reflectance to 0-1 if needed. Accepts (C,H,W)."""
    bands = bands.astype(np.float32)
    if bands.shape[0] > NUM_BANDS:
        bands = bands[:NUM_BANDS]
    if bands.max() > 1:
        bands = bands / 10000.0
    return bands


def pad_to_multiple(img: np.ndarray, size: int) -> np.ndarray:
    """Reflect-pad the last two dims up to a multiple of *size*."""
    h, w = img.shape[-2:]
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * (img.ndim - 2) + [(0, pad_h), (0, pad_w)]
        img = np.pad(img, pad_width, mode="reflect")
    return img


def split_windows(img: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    """Split a (C,H,W) image into non-overlapping (N,C,size,size) windows.

    Returns (windows, n_rows, n_cols) where rows/cols are the padded grid dims.
    """
    img = pad_to_multiple(img, size)
    h1 = img.shape[-2] // size
    w1 = img.shape[-1] // size
    windows = np.stack(
        [
            img[:, y : y + size, x : x + size]
            for y in range(0, img.shape[-2], size)
            for x in range(0, img.shape[-1], size)
        ]
    )
    return windows, h1, w1


def merge_windows(
    windows: np.ndarray,
    n_rows: int,
    n_cols: int,
    out_h: int,
    out_w: int,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Inverse of split_windows for a (N,C,size,size) class/prob tensor.

    *windows* row-major order (rows of the padded grid), shape (N, C, size, size).
    Returns (C, out_h, out_w) cropped back to the original size.
    """
    n, c = windows.shape[:2]
    assert n == n_rows * n_cols
    grid = windows.reshape(n_rows, n_cols, c, size, size)
    grid = grid.transpose(2, 0, 3, 1, 4)  # (C, n_rows*size, n_cols*size)
    grid = grid.reshape(c, n_rows * size, n_cols * size)
    return grid[:, :out_h, :out_w]
