import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds

from data_pipeline import model_inference_v2 as v2


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=True)
def test_pick_device_prefers_cuda(mock_cuda, mock_mps):
    assert v2.pick_device() == "cuda"


@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.cuda.is_available", return_value=False)
def test_pick_device_falls_back_to_mps(mock_cuda, mock_mps):
    assert v2.pick_device() == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
def test_pick_device_falls_back_to_cpu(mock_cuda, mock_mps):
    assert v2.pick_device() == "cpu"


def test_constants_match_model_repo():
    assert v2.MODEL_REPO == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
    assert v2.CHECKPOINT_NAME == "Prithvi_EO_V2_300M_BurnScars.pt"
    assert v2.PATCH_SIZE == 512
    assert v2.NUM_BANDS == 6


@patch("huggingface_hub.hf_hub_download", return_value="/tmp/fake/checkpoint.pt")
def test_download_checkpoint_uses_hub(mock_download):
    path = v2.download_checkpoint()
    assert str(path) == "/tmp/fake/checkpoint.pt"
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["repo_id"] == v2.MODEL_REPO
    assert mock_download.call_args.kwargs["filename"] == v2.CHECKPOINT_NAME


def test_scale_bands_divides_large_values():
    bands = np.full((6, 64, 64), 5000.0, dtype=np.float32)
    scaled = v2.scale_bands(bands)
    assert scaled.max() <= 1.0
    assert np.allclose(scaled, 0.5)


def test_scale_bands_leaves_reflectance_untouched():
    bands = np.full((6, 64, 64), 0.4, dtype=np.float32)
    assert np.allclose(v2.scale_bands(bands), 0.4)


def test_pad_to_multiple_reflect_pads():
    img = np.ones((6, 500, 700), dtype=np.float32)
    padded = v2.pad_to_multiple(img, v2.PATCH_SIZE)
    assert padded.shape == (6, 512, 1024)
    assert np.allclose(padded[:, :500, :700], 1.0)


def test_split_windows_returns_grid_dims():
    img = np.zeros((6, 1024, 1024), dtype=np.float32)
    img[:, :512, :512] = 1.0
    windows, h1, w1 = v2.split_windows(img, v2.PATCH_SIZE)
    assert (h1, w1) == (2, 2)
    assert windows.shape == (4, 6, 512, 512)
    assert np.allclose(windows[0], 1.0)
    assert np.allclose(windows[3], 0.0)


def test_merge_windows_round_trips_grid():
    img = np.random.default_rng(0).random((1, 700, 900), dtype=np.float32)
    windows, h1, w1 = v2.split_windows(img, v2.PATCH_SIZE)
    class_tensor = np.stack([np.zeros_like(windows), windows], axis=1)  # (N,2,H,W)
    merged = v2.merge_windows(class_tensor, h1, w1, img.shape[1], img.shape[2])
    assert merged.shape == (2, 700, 900)
    assert np.allclose(merged[1], img, atol=1e-6)


def test_save_mask_writes_georeferenced_geotiff():
    bands = np.zeros((6, 32, 32), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "ref.tif")
        with rasterio.open(
            ref,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=6,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 32, 32),
        ) as dst:
            dst.write(bands)
        out = os.path.join(tmp, "mask.tif")
        mask = np.full((32, 32), 0.7, dtype=np.float32)
        v2.save_mask(mask, out, ref)
        with rasterio.open(out) as src:
            assert src.count == 1
            assert (src.height, src.width) == (32, 32)
            assert src.crs.to_string() == "EPSG:4326"
            assert np.allclose(src.read(1), 0.7)


def test_predict_applies_datamodule_transforms_and_returns_probs():
    logits = torch.zeros(1, 2, 512, 512, dtype=torch.float32)
    logits[:, 1] = 10.0  # burn class wins everywhere
    fake_logits = SimpleNamespace(output=logits)
    fake_model = Mock()
    fake_model.model.return_value = fake_logits
    fake_datamodule = Mock()
    fake_model.datamodule = fake_datamodule
    fake_datamodule.test_transform.return_value = {"image": torch.zeros(6, 512, 512)}
    fake_datamodule.aug.return_value = {"image": torch.zeros(1, 6, 512, 512)}

    bands = np.zeros((6, 64, 64), dtype=np.float32)
    probs, mask = v2.predict(bands, model=fake_model, device="cpu")

    assert probs.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert probs.max() <= 1.0
    assert probs.min() > 0.99  # entire map is the burn class
    assert mask.max() == 1  # burn class predicted where logits positive
