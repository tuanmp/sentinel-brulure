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


def predict(
    bands: np.ndarray,
    model=None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run V2-300M inference on a (6,H,W) reflectance array.

    Returns (probs, class_mask):
      probs      — softmax probability of the burn class, (H,W) float32 in [0,1]
      class_mask — argmax class, (H,W) uint8 (0 = not burned, 1 = burn scar)
    """
    if model is None:
        model = load_model(device=device)
    device = device or pick_device()
    model.model.to(device)

    datamodule = model.datamodule
    img = scale_bands(bands)
    windows, n_rows, n_cols = split_windows(img, PATCH_SIZE)

    logits = []
    with torch.no_grad():
        for window in windows:
            example = datamodule.test_transform(image=window.transpose(1, 2, 0))
            example["image"] = example["image"].unsqueeze(0).to(device)
            normalized = datamodule.aug(example)["image"].to(device)
            out = model.model(normalized).output  # (1, num_classes, 512, 512)
            logits.append(out.cpu().numpy())

    all_logits = np.concatenate(logits, axis=0)  # (N, num_classes, 512, 512)
    e = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)  # (N, num_classes, 512, 512)

    merged = merge_windows(probs_all, n_rows, n_cols, img.shape[1], img.shape[2])
    probs = merged[1].astype(np.float32)
    class_mask = np.argmax(merged, axis=0).astype(np.uint8)
    return probs, class_mask


def prepare_input(geotiff_path: str) -> np.ndarray:
    """Read a 6-band GeoTIFF into a (6,H,W) float32 reflectance array."""
    import rasterio

    with rasterio.open(geotiff_path) as src:
        data = src.read().astype(np.float32)
    if data.shape[0] > NUM_BANDS:
        data = data[:NUM_BANDS]
    return data


def run_inference(
    geotiff_path: str,
    model=None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: read a 6-band GeoTIFF and predict. Returns (probs, class_mask)."""
    return predict(prepare_input(geotiff_path), model=model, device=device)


def save_mask(mask: np.ndarray, output_path: str, reference_geotiff: str) -> None:
    """Write a single-band mask GeoTIFF sharing the reference georeferencing."""
    import rasterio

    with rasterio.open(reference_geotiff) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype=rasterio.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(rasterio.float32), 1)
