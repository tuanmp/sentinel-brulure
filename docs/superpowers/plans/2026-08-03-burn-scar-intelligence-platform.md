# Burn Scar Intelligence Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the project to the Prithvi-EO-2.0-300M burn-scar model, prove it beats baselines with an evaluation harness, surface results in the Streamlit dashboard, and ship it with Docker + CI + a live deploy.

**Architecture:** A standalone TerraTorch-backed inference module replaces the model work, an evaluation harness scores it against HLS-labeled data (V2-300M vs old 100M) plus a dNBR cross-check on live events, the existing Streamlit dashboard gains a burn-scar overlay and a "Model" tab reading eval artifacts, and Docker/CI/Streamlit Cloud make it shareable. The old 100M module is left untouched.

**Tech Stack:** Python 3.12, PyTorch, TerraTorch 1.2.6, Lightning, HuggingFace hub, rasterio, numpy, matplotlib, Streamlit, Plotly, pandas, pytest, ruff, uv, Docker, GitHub Actions, MLflow.

---

## Reference Facts (verify nothing against memory; use these)

- **Model repo:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`
- **Checkpoint file:** `Prithvi_EO_V2_300M_BurnScars.pt` (~1.3 GB)
- **Example chip (validation):** `examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif`
- **Official inference script:** `inference.py` (raw URL below)
- **Config in model repo:** `burn_scars_config.yaml` — has `backbone_pretrained: true`; we bundle a copy with that single change to `false`.
- **Model bands (order):** BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2 — matches our `evalscript` B02,B03,B04,B08,B11,B12 order.
- **Means/std** for normalization live in the config's `data.init_args` (per-band, HLS statistics).
- **Dataset repo:** `ibm-nasa-geospatial/hls_burn_scars`; tarball `hls_burn_scars.tar.gz` (2.65 GB) extracts to `training/` and `validation/` dirs. Images `*_merged.tif`, masks `*.mask.tif` (same basename, `_merged.tif` → `.mask.tif`).
- **Eval splits:** model repo `splits/test.txt` etc. — one tile prefix per line (e.g. `T10SEH.2018190.v1`). A test image is `<prefix>_merged.tif`.
- **100M model:** `data_pipeline/model_inference.py`, repo `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar`, checkpoint `burn_scars_Prithvi_100M.pth`. Same HLS band order.
- **Our band nuance:** the pipeline fetches B08, but both models were trained on HLS NIR_B8A. The HLS eval is apples-to-apples (both read HLS chips directly); the live-event cross-check inherits this known mismatch. Document it, don't fix it.
- **Env:** `make lint`, `make test` (`PYTHONPATH=. uv run pytest tests/ -v`). Tests live in `tests/`. Repo uses both unittest-style and pytest-style tests. `python-dotenv` loaded in `sentinel_request.py`.
- **Dashboard:** entry `dashboard/app.py` (Streamlit). Cached imagery under `reports/imagery/<event_id>/<phase>_<date>_r<res>.npz` (7 bands: 6 spectral + dataMask at index 6). `dashboard/imagery.py` has `load_cached_phase(event, phase, resolution)`.
- **Hardware:** MPS available (`torch.backends.mps.is_available() == True`), no CUDA. Current `model_inference.py` only checks cuda.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `data_pipeline/configs/prithvi_v2_burn_scars_inference.yaml` (create) | Bundled model config, `backbone_pretrained: false` |
| `data_pipeline/model_inference_v2.py` (create) | V2-300M load/inference/geotiff helpers |
| `tests/test_model_inference_v2.py` (create) | Offline unit tests + gated integration test |
| `scripts/validate_prithvi_v2.py` (create) | Cross-check vs official `inference.py` |
| `analytics/evaluation.py` (create) | Metric functions + aggregation |
| `tests/test_evaluation.py` (create) | Metric unit tests |
| `scripts/run_evaluation.py` (create) | HLS + dNBR eval, writes artifact |
| `dashboard/model_view.py` (create) | Model-tab renderers + overlay figure |
| `tests/test_model_view.py` (create) | Model-view unit tests |
| `dashboard/app.py` (modify) | Add overlay + Model tab + secrets shim |
| `requirements.txt` (create) | Runtime deps for Streamlit Cloud |
| `Dockerfile` (create) | Containerized dashboard |
| `docker-compose.yml` (create) | Local container orchestration |
| `scripts/track_eval_mlflow.py` (create) | MLflow-tracked eval run |
| `.env_template` (modify) | Document `RUN_MODEL_TESTS` |
| `pyproject.toml` (modify) | Add `mlflow` to dev group |
| `docs/superpowers/specs/2026-08-03-burn-scar-intelligence-platform-design.md` | Spec (exists) |

---

## Phase 1 — V2-300M Standalone Inference

### Task 1: Bundled config

**Files:**
- Create: `data_pipeline/configs/prithvi_v2_burn_scars_inference.yaml`

- [ ] **Step 1: Fetch the official config**

Fetch `https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/burn_scars_config.yaml` (use the webfetch tool / curl) and save it as the task file.

- [ ] **Step 2: Change exactly one field**

Edit `backbone_pretrained: true` → `backbone_pretrained: false`. Leave everything else byte-for-byte identical (backbone, UNetDecoder, means/stds, `ToTensorV2` test_transform, num_classes, class_names).

- [ ] **Step 3: Commit**

```bash
git add data_pipeline/configs/prithvi_v2_burn_scars_inference.yaml
git commit -m "feat: bundle Prithvi V2-300M burn scars inference config"
```

### Task 2: Module skeleton + device selection

**Files:**
- Create: `data_pipeline/model_inference_v2.py`
- Test: `tests/test_model_inference_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from data_pipeline import model_inference_v2 as v2


def test_pick_device_prefers_cuda(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert v2.pick_device() == "cuda"


def test_pick_device_falls_back_to_mps(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert v2.pick_device() == "mps"


def test_pick_device_falls_back_to_cpu(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert v2.pick_device() == "cpu"


def test_constants_match_model_repo():
    assert v2.MODEL_REPO == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
    assert v2.CHECKPOINT_NAME == "Prithvi_EO_V2_300M_BurnScars.pt"
    assert v2.PATCH_SIZE == 512
    assert v2.NUM_BANDS == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -v`
Expected: FAIL — `ModuleNotFoundError: data_pipeline.model_inference_v2`

- [ ] **Step 3: Write the module skeleton**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/model_inference_v2.py tests/test_model_inference_v2.py
git commit -m "feat: add Prithvi V2-300M inference module skeleton with device selection"
```

### Task 3: Download + load model

**Files:**
- Modify: `data_pipeline/model_inference_v2.py`
- Test: `tests/test_model_inference_v2.py`

- [ ] **Step 1: Write the failing test**

```python
def test_download_checkpoint_uses_hub(mocker):
    fake_path = "/tmp/fake/checkpoint.pt"
    mock_download = mocker.patch(
        "huggingface_hub.hf_hub_download", return_value=fake_path
    )
    path = v2.download_checkpoint()
    assert str(path) == fake_path
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["repo_id"] == v2.MODEL_REPO
    assert mock_download.call_args.kwargs["filename"] == v2.CHECKPOINT_NAME
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py::test_download_checkpoint_uses_hub -v`
Expected: FAIL — `AttributeError: module 'data_pipeline.model_inference_v2' has no attribute 'download_checkpoint'`

- [ ] **Step 3: Implement**

Append to `model_inference_v2.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py::test_download_checkpoint_uses_hub -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/model_inference_v2.py tests/test_model_inference_v2.py
git commit -m "feat: add V2 checkpoint download and LightningInferenceModel loader"
```

### Task 4: Input scaling + sliding window helpers

**Files:**
- Modify: `data_pipeline/model_inference_v2.py`
- Test: `tests/test_model_inference_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np


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
    img = np.random.default_rng(0).random((2, 700, 900), dtype=np.float32)
    windows, h1, w1 = v2.split_windows(img, v2.PATCH_SIZE)
    class_tensor = np.stack([np.zeros_like(windows), windows], axis=1)  # (N,2,H,W)
    merged = v2.merge_windows(class_tensor, h1, w1, img.shape[1], img.shape[2])
    assert merged.shape == (2, 700, 900)
    assert np.allclose(merged[1], img, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -k "scale_bands or pad_to_multiple or split_windows or merge_windows" -v`
Expected: FAIL — `AttributeError` for the new functions

- [ ] **Step 3: Implement**

Append to `model_inference_v2.py`:

```python
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
        img = np.pad(img, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -k "scale_bands or pad_to_multiple or split_windows or merge_windows" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/model_inference_v2.py tests/test_model_inference_v2.py
git commit -m "feat: add V2 input scaling and sliding-window helpers"
```

### Task 5: predict + run_inference + save_mask

**Files:**
- Modify: `data_pipeline/model_inference_v2.py`
- Test: `tests/test_model_inference_v2.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_inference_v2.py` (it already has `import pytest`,
`import numpy as np`, and `from data_pipeline import model_inference_v2 as v2`
from earlier tasks). Add these imports at the top of the file:

```python
import os
import tempfile

import rasterio
import torch
from rasterio.transform import from_bounds
```

Then append the tests:

```python
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


def test_predict_applies_datamodule_transforms_and_returns_probs(mocker):
    fake_model = mocker.Mock()
    fake_datamodule = mocker.Mock()
    fake_logits = mocker.Mock()
    fake_logits.output = torch.tensor(
        [[[[0.0], [10.0]]]], dtype=torch.float32  # (1, 2, 1, 1): burn wins
    )
    fake_model.model = fake_logits
    fake_model.datamodule = fake_datamodule
    fake_datamodule.test_transform.return_value = {"image": torch.zeros(6, 512, 512)}
    fake_datamodule.aug.return_value = {"image": torch.zeros(1, 6, 512, 512)}

    bands = np.zeros((6, 64, 64), dtype=np.float32)
    probs, mask = v2.predict(bands, model=fake_model, device="cpu")

    assert probs.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert probs.max() <= 1.0
    assert mask.max() == 1  # burn class predicted where logits positive
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -k "save_mask or predict" -v`
Expected: FAIL — `AttributeError` for `save_mask` / `predict`

- [ ] **Step 3: Implement**

Append to `model_inference_v2.py`:

```python
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
```

Note: `torch` is imported at the top of the test file by the step above; the module itself already imports `torch` at the top.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -k "save_mask or predict" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/model_inference_v2.py tests/test_model_inference_v2.py
git commit -m "feat: add V2 predict, run_inference, and save_mask"
```

### Task 6: Gated integration test

**Files:**
- Modify: `tests/test_model_inference_v2.py`

- [ ] **Step 1: Write the gated test**

```python
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to download the ~1.3GB checkpoint",
)


def test_run_inference_end_to_end(tmp_path):
    bands = np.random.default_rng(1).random((6, 64, 64), dtype=np.float32)
    tif = tmp_path / "input.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=6,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 64, 64),
    ) as dst:
        dst.write(bands)

    model = v2.load_model()
    probs, mask = v2.run_inference(str(tif), model=model)

    assert probs.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert mask.dtype == np.uint8
```

- [ ] **Step 2: Run the offline suite to confirm it's skipped**

Run: `PYTHONPATH=. uv run pytest tests/test_model_inference_v2.py -v`
Expected: integration test shows `SKIPPED` (reason: RUN_MODEL_TESTS unset)

- [ ] **Step 3: Run once with the env var to verify the real path (needs ~1.3GB download + a few minutes)**

Run: `PYTHONPATH=. RUN_MODEL_TESTS=1 uv run pytest tests/test_model_inference_v2.py::test_run_inference_end_to_end -v`
Expected: PASS (downloads checkpoint on first run). If the model throws a shape error here, fix the batch/transform shape handling before continuing.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_inference_v2.py
git commit -m "test: add gated V2-300M end-to-end integration test"
```

### Task 7: Validation script vs official inference

**Files:**
- Create: `scripts/validate_prithvi_v2.py`

- [ ] **Step 1: Write the script**

```python
"""Cross-check our V2-300M inference against the official inference.py.

Downloads the official example chip, runs both paths, and reports pixel-wise
agreement between the class masks.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from data_pipeline.model_inference_v2 import (
    download_checkpoint,
    load_model,
    run_inference,
    save_mask,
)

EXAMPLE_FILE = "examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
OFFICIAL_INFERENCE = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/"
    "resolve/main/inference.py"
)


def download_example(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
            filename=EXAMPLE_FILE,
            cache_dir=str(cache_dir),
        )
    )


def download_official_script(dest_dir: Path) -> Path:
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "inference.py"
    if not dest.exists():
        resp = requests.get(OFFICIAL_INFERENCE, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text)
    return dest


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero((a == 1) & (b == 1))
    union = np.count_nonzero((a == 1) | (b == 1))
    return float(inter / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/prithvi_v2_val"))
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    print("Downloading checkpoint (may take a while)...")
    ckpt = download_checkpoint()
    print(f"Checkpoint: {ckpt}")

    example = download_example(work)
    official = download_official_script(work)
    print(f"Example: {example}")

    config = Path(__file__).resolve().parent.parent / "data_pipeline" / "configs" \
        / "prithvi_v2_burn_scars_inference.yaml"

    print("Running our inference...")
    model = load_model(checkpoint_path=ckpt)
    probs, mask = run_inference(str(example), model=model)
    print(f"  probs range [{probs.min():.4f}, {probs.max():.4f}], "
          f"burn fraction {np.mean(mask == 1):.3%}")

    print("Running official inference.py...")
    out_dir = work / "official_out"
    cmd = [
        sys.executable, str(official),
        "--data_file", str(example),
        "--config", str(config),
        "--checkpoint", str(ckpt),
        "--output_dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(work))

    pred_file = out_dir / f"pred_{Path(example).stem}.tiff"
    with rasterio.open(pred_file) as src:
        official_mask = src.read(1)
    official_mask = (official_mask == 1).astype(np.uint8)

    accuracy = float(np.mean(mask == official_mask))
    print(f"  pixel accuracy: {accuracy:.4f}")
    print(f"  burn-class IoU:  {iou(mask, official_mask):.4f}")
    print(f"  burn fraction official: {np.mean(official_mask == 1):.3%}")

    save_path = work / "our_mask.tif"
    save_mask(mask.astype(np.float32), str(save_path), str(example))
    print(f"Our mask saved to {save_path}")

    if accuracy < 0.99:
        print("WARNING: masks disagree with the official script; investigate.")
        sys.exit(1)
    print("OK: our inference matches the official script.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=. uv run python scripts/validate_prithvi_v2.py`
Expected: prints accuracy and IoU; exits 0 only if accuracy ≥ 0.99. First run downloads the 1.3GB checkpoint + example chip.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_prithvi_v2.py
git commit -m "feat: add V2-300M validation script vs official inference"
```

Phase 1 gate: `validate_prithvi_v2.py` matches official output on the example chip.

---

## Phase 2 — Evaluation Harness

### Task 8: Metric functions

**Files:**
- Create: `analytics/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from analytics.evaluation import aggregate_metrics, binary_metrics, dnbr_mask


def test_binary_metrics_ignores_nodata_and_scores_known_case():
    pred = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    true = np.array([[1, 1, -1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8)
    m = binary_metrics(pred, true)
    # valid pixels: (0,0)=TP, (0,1)=TP, (1,1)=TP, (1,2)=TN, (2,1)=TN, (2,2)=FN
    assert m["tp"] == 3 and m["fp"] == 0 and m["fn"] == 1 and m["tn"] == 2
    assert m["iou"] == 3 / 4
    assert m["dice"] == 2 * 3 / (2 * 3 + 1)
    assert m["precision"] == 1.0
    assert m["recall"] == 3 / 4


def test_binary_metrics_no_positive_pixels():
    pred = np.zeros((2, 2), dtype=np.uint8)
    true = np.zeros((2, 2), dtype=np.int8)
    m = binary_metrics(pred, true)
    assert m["iou"] == 0.0 and m["dice"] == 0.0
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_dnbr_mask_thresholds_and_masks_nan():
    dnbr = np.array([[0.1, 0.3], [0.44, np.nan]], dtype=np.float32)
    m = dnbr_mask(dnbr, threshold=0.27)
    assert m.tolist() == [[False, True], [True, False]]


def test_aggregate_metrics_averages():
    results = [{"iou": 0.5, "dice": 0.6}, {"iou": 0.7, "dice": 0.8}]
    agg = aggregate_metrics(results)
    assert np.isclose(agg["iou"], 0.6)
    assert np.isclose(agg["dice"], 0.7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: analytics.evaluation`

- [ ] **Step 3: Implement**

```python
"""Binary segmentation metric helpers for burn scar evaluation."""
import numpy as np


def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Burn-class metrics with -1 (no data) pixels excluded.

    pred: (H,W) integer array, 1 = burn predicted.
    true: (H,W) integer array, 1 = burn, 0 = not burned, -1 = ignore.
    """
    valid = true != -1
    p = pred[valid] == 1
    t = true[valid] == 1
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def dnbr_mask(dnbr: np.ndarray, threshold: float = 0.27) -> np.ndarray:
    """Binary burned mask from dNBR (>= threshold), NaN pixels excluded."""
    return np.isfinite(dnbr) & (dnbr >= threshold)


def aggregate_metrics(results: list[dict]) -> dict[str, float]:
    """Mean of each float metric across results."""
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_evaluation.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add analytics/evaluation.py tests/test_evaluation.py
git commit -m "feat: add binary segmentation metric helpers"
```

### Task 9: HLS evaluation runner

**Files:**
- Create: `scripts/run_evaluation.py`

- [ ] **Step 1: Write the script**

```python
"""Evaluation harness for burn scar models on the HLS Burn Scars dataset.

Scores V2-300M and the legacy 100M model against the test split, plus a dNBR
cross-check on live events from the event store. Writes a benchmark artifact
to reports/evaluation/<timestamp>/.
"""
import argparse
import json
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from analytics.evaluation import (
    aggregate_metrics,
    binary_metrics,
    dnbr_mask,
)
from data_pipeline.model_inference import run_inference as run_100m
from data_pipeline.model_inference_v2 import load_model, run_inference as run_v2

DATASET_REPO = "ibm-nasa-geospatial/hls_burn_scars"
MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
CACHE_ROOT = Path.home() / ".cache" / "sentinel-sat"
EVAL_ROOT = Path("reports") / "evaluation"


def hls_data_root(cache_dir: Path) -> Path:
    """Download + extract the HLS tarball once; return the extraction dir."""
    from huggingface_hub import hf_hub_download

    data_dir = cache_dir / "hls_burn_scars_data"
    if (data_dir / "training").exists() and (data_dir / "validation").exists():
        return data_dir
    tarball = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename="hls_burn_scars.tar.gz",
            cache_dir=str(cache_dir),
        )
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(data_dir)
    return data_dir


def test_split_prefixes(cache_dir: Path) -> list[str]:
    from huggingface_hub import hf_hub_download

    split_file = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="splits/test.txt",
            cache_dir=str(cache_dir),
        )
    )
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def find_chip(data_root: Path, prefix: str) -> tuple[Path, Path]:
    image = next((data_root / d).glob(f"{prefix}_merged.tif") for d in ("training", "validation"))
    mask = Path(str(image).replace("_merged.tif", ".mask.tif"))
    if not mask.exists():
        raise FileNotFoundError(f"mask for {prefix} not found: {mask}")
    return image, mask


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def mask_from_probs(probs: np.ndarray) -> np.ndarray:
    return (probs >= 0.5).astype(np.uint8)


def evaluate_split(data_root: Path, prefixes: list[str], limit: int, device: str) -> dict:
    v2_model = load_model(device=device)
    rows_v2, rows_100m = [], []
    for prefix in prefixes[:limit]:
        image, mask_path = find_chip(data_root, prefix)
        true = read_mask(mask_path)
        probs_v2, mask_v2 = run_v2(str(image), model=v2_model, device=device)
        rows_v2.append(binary_metrics(mask_v2, true))

        probs_100m = run_100m(str(image), device=device)
        rows_100m.append(binary_metrics(mask_from_probs(probs_100m), true))

    return {
        "n": len(rows_v2),
        "v2_300m": aggregate_metrics(rows_v2),
        "v1_100m": aggregate_metrics(rows_100m),
    }


def dnbr_crosscheck(event_root: Path, limit: int, resolution: int, device: str) -> dict:
    from analytics.store import EventStore
    from dashboard.imagery import load_cached_phase
    from data_pipeline.sentinel_utils import compute_nbr

    store = EventStore(event_root)
    events = store.list_events()
    model = load_model(device=device)
    rows = []
    for event in events[:limit]:
        pre = load_cached_phase(event, "before", resolution)
        post = load_cached_phase(event, "after", resolution)
        if pre is None or post is None:
            continue
        dnbr = compute_nbr(pre) - compute_nbr(post)
        _, mask_v2 = run_v2(post[:6], model=model, device=device)
        baseline = dnbr_mask(dnbr).astype(np.uint8)
        rows.append(binary_metrics(mask_v2, baseline))
    return {"n": len(rows), "dNBR_crosscheck": aggregate_metrics(rows)}


def write_artifact(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))

    labels = ["V2-300M", "V1-100M"]
    ious = [result["v2_300m"]["iou"], result["v1_100m"]["iou"]]
    dices = [result["v2_300m"]["dice"], result["v1_100m"]["dice"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, ious, 0.4, label="IoU")
    ax.bar(x + 0.2, dices, 0.4, label="Dice")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Burn scar benchmark on HLS test split (n={result['n']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap chips evaluated (for smoke runs)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--events-root", type=Path, default=Path("reports/events"))
    parser.add_argument("--resolution", type=int, default=60)
    args = parser.parse_args()

    device = args.device or "cpu"
    data_root = hls_data_root(CACHE_ROOT)
    prefixes = test_split_prefixes(CACHE_ROOT)
    print(f"Evaluating {len(prefixes[:args.limit])} HLS test chips with V2-300M and V1-100M...")
    result = evaluate_split(data_root, prefixes, args.limit or len(prefixes), device)

    print("dNBR cross-check on live events...")
    result["events"] = dnbr_crosscheck(args.events_root, args.limit or 10, args.resolution, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_ROOT / timestamp
    write_artifact(out_dir, result)
    print(f"Artifact written to {out_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (limit 1) to validate the code path**

Run: `PYTHONPATH=. uv run python scripts/run_evaluation.py --limit 1`
Expected: downloads dataset + checkpoints (large), evaluates 1 chip, writes `reports/evaluation/<ts>/summary.json` and `benchmark.png`.

- [ ] **Step 3: Full run**

Run: `PYTHONPATH=. uv run python scripts/run_evaluation.py`
Expected: evaluates the full test split; artifact reproducible.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat: add HLS + dNBR evaluation harness"
```

Phase 2 gate: `run_evaluation.py` produces a reproducible benchmark artifact.

---

## Phase 3 — Dashboard Integration

### Task 10: Model-view module (overlay figure + model card + eval loader)

**Files:**
- Create: `dashboard/model_view.py`
- Test: `tests/test_model_view.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import numpy as np
import pytest

from dashboard.model_view import (
    load_latest_eval,
    model_card_markdown,
    render_overlay,
)


def _write_eval(root: Path, ts: str, iou: float) -> Path:
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"v2_300m": {"iou": iou}, "n": 1}), encoding="utf-8"
    )
    return out


def test_load_latest_eval_picks_newest(tmp_path):
    _write_eval(tmp_path, "20260101_000000", 0.5)
    _write_eval(tmp_path, "20260102_000000", 0.8)
    result = load_latest_eval(tmp_path)
    assert result is not None
    assert result["v2_300m"]["iou"] == 0.8


def test_load_latest_eval_none_when_empty(tmp_path):
    assert load_latest_eval(tmp_path) is None


def test_model_card_markdown_has_key_sections():
    md = model_card_markdown()
    assert "Prithvi-EO-2.0-300M" in md
    assert "Apache-2.0" in md
    assert "UNetDecoder" in md
    assert "B02" in md


def test_render_overlay_returns_figure():
    bands = np.zeros((6, 32, 32), dtype=np.float32)
    probs = np.zeros((32, 32), dtype=np.float32)
    probs[10:20, 10:20] = 0.8
    fig = render_overlay(bands, probs)
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_model_view.py -v`
Expected: FAIL — `ModuleNotFoundError: dashboard.model_view`

- [ ] **Step 3: Implement**

```python
"""Model tab renderers and burn-scar overlay for the dashboard."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.image_utils import to_rgb

EVAL_ROOT = Path("reports") / "evaluation"


def load_latest_eval(root: Path = EVAL_ROOT) -> dict | None:
    """Load the most recent benchmark artifact, or None."""
    if not root.exists():
        return None
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dirs:
        return None
    latest = dirs[-1]
    summary = latest / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8"))


def model_card_markdown() -> str:
    return """### Model card — Prithvi-EO-2.0-300M-BurnScars

| Property | Value |
|---|---|
| Backbone | prithvi_eo_v2_300 (embed_dim 1024, depth 24) |
| Decoder | UNetDecoder (512, 256, 128, 64) |
| Input bands | B02, B03, B04, B08, B11, B12 |
| Patch size | 512×512 |
| Classes | 0 = not burned, 1 = burn scar |
| Normalization | per-band mean/std (HLS statistics) |
| License | Apache-2.0 |
| Reference | arXiv 2412.02732 |
| Benchmark | 87.5 burned-class IoU on HLS test split |

Source: [ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars)
"""


def render_overlay(bands: np.ndarray, probs: np.ndarray):
    """Post-fire RGB + burn probability + class mask, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb = np.clip(to_rgb(bands).transpose(1, 2, 0), 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title("Post-fire RGB")
    im1 = axes[1].imshow(probs, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Burn probability (V2-300M)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb)
    axes[2].imshow(
        np.ma.masked_where(probs < 0.5, probs),
        cmap="Reds",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Burn overlay (prob ≥ 0.5)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_model_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/model_view.py tests/test_model_view.py
git commit -m "feat: add dashboard model view (overlay, model card, eval loader)"
```

### Task 11: Wire overlay + Model tab into the dashboard

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Add imports**

At the top of `dashboard/app.py`, after the existing `from dashboard.linked import build_linked_figure` line, add:

```python
from dashboard.model_view import (
    load_latest_eval,
    model_card_markdown,
    render_overlay,
)
```

- [ ] **Step 2: Add a burn-scar overlay in the satellite-progression tab**

After the linked figure block in `tab_progress` (after `_badge_if_low_valid(post_bands, "Post-fire scene")` and before the `if ndvi_mode == "delta":` caption block), add:

```python
            with st.expander("Burn scar overlay (Prithvi V2-300M)"):
                run_model = st.button("Run burn-scar model", key="run_model_btn")
                st.caption(
                    "Loads the ~1.3GB checkpoint on first use (cached in "
                    "session). Needs the six post-fire bands."
                )
                if run_model or st.session_state.get("v2_probs") is not None:
                    if st.session_state.get("v2_probs") is None:
                        try:
                            from data_pipeline.model_inference_v2 import (
                                load_model,
                                predict,
                            )

                            model = st.session_state.get("v2_model")
                            if model is None:
                                model = load_model()
                                st.session_state["v2_model"] = model
                            probs, mask = predict(post_bands[:6], model=model)
                            st.session_state["v2_probs"] = probs
                            st.session_state["v2_mask"] = mask
                        except Exception as exc:
                            st.error(f"Model inference failed: {exc}")
                            st.session_state["v2_probs"] = None
                    probs = st.session_state.get("v2_probs")
                    if probs is not None:
                        overlay_fig = render_overlay(post_bands[:6], probs)
                        st.pyplot(overlay_fig)
                        mask = st.session_state["v2_mask"]
                        st.metric(
                            "Burned fraction (model)",
                            f"{np.mean(mask == 1):.1%}",
                        )
```

Note: `np` is already imported in `app.py`.

- [ ] **Step 3: Add the Model tab**

Change the `st.tabs([...])` call to include a new tab. Replace:

```python
    tab_overview, tab_progress, tab_during, tab_post, tab_recovery, tab_failures = (
        st.tabs(
            [
                "Events",
                "Satellite progression",
                "During (FRP)",
                "Post-fire severity",
                "Recovery",
                "Failures",
            ]
        )
    )
```

with:

```python
    tab_overview, tab_progress, tab_during, tab_post, tab_recovery, tab_failures, tab_model = (
        st.tabs(
            [
                "Events",
                "Satellite progression",
                "During (FRP)",
                "Post-fire severity",
                "Recovery",
                "Failures",
                "Model",
            ]
        )
    )
```

Then, after the `with tab_failures:` block (still inside `main()`), add:

```python
    with tab_model:
        st.subheader("Burn scar model")
        st.markdown(model_card_markdown())
        eval_data = load_latest_eval()
        if eval_data is None:
            st.info(
                "No evaluation artifact found under reports/evaluation/. "
                "Run `PYTHONPATH=. uv run python scripts/run_evaluation.py`."
            )
        else:
            st.markdown(f"**Benchmark on HLS test split (n={eval_data.get('n')})**")
            v2 = eval_data.get("v2_300m", {})
            v1 = eval_data.get("v1_100m", {})
            rows = [
                {"model": "V2-300M", **{k: v2.get(k) for k in ("iou", "dice", "precision", "recall")}},
                {"model": "V1-100M", **{k: v1.get(k) for k in ("iou", "dice", "precision", "recall")}},
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            chart = Path("reports/evaluation")
            chart_dir = sorted(d for d in chart.iterdir() if d.is_dir())[-1] if chart.exists() and list(chart.iterdir()) else None
            if chart_dir and (chart_dir / "benchmark.png").exists():
                st.image(str(chart_dir / "benchmark.png"))
            events_eval = eval_data.get("events")
            if events_eval:
                st.markdown("**dNBR cross-check on live events**")
                st.write(events_eval)
```

Note: `Path` is imported in `app.py` (line 7); `pd` is imported (line 11).

- [ ] **Step 4: Sanity check the dashboard imports/runs**

Run: `PYTHONPATH=. uv run python -c "import dashboard.app"`
Expected: imports cleanly (no Streamlit execution — `main()` is only called under `__main__`).

- [ ] **Step 5: Run the existing tests to confirm no regressions**

Run: `make test`
Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add burn-scar overlay and Model tab to dashboard"
```

Phase 3 gate: dashboard Model tab + overlay works locally (`streamlit run dashboard/app.py`).

---

## Phase 4 — Deployment + Light MLOps

### Task 12: requirements.txt for Streamlit Cloud

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write the file**

```txt
numpy>=2.4.4
pandas>=3.0.2
rasterio>=1.5.0
matplotlib>=3.10.8
plotly>=6.7.0
streamlit>=1.60.0
sentinelhub>=3.11.5
pystac-client>=0.9.0
python-dotenv>=1.2.2
geopandas>=1.1.3
shapely>=2.1.2
scikit-learn>=1.8.0
seaborn>=0.13.2
requests
requests-oauthlib
oauthlib
torch>=2.4.0
torchvision
einops
timm>=1.0.26
lightning>=2.6.1
terratorch>=1.2.6
mmsegmentation
huggingface-hub
pyyaml
albumentations
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add runtime requirements for Streamlit Cloud"
```

### Task 13: Dockerfile + docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
```

- [ ] **Step 3: Write .dockerignore**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
notebooks/
reports/imagery/
reports/events/
*.png
*.tif
```

- [ ] **Step 4: Build smoke test (optional, requires Docker)**

Run: `docker compose build`
Expected: image builds. (If Docker is unavailable, skip and rely on CI.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "chore: add Docker packaging for the dashboard"
```

### Task 14: MLflow-tracked eval + pyproject dev dep + env template

**Files:**
- Create: `scripts/track_eval_mlflow.py`
- Modify: `pyproject.toml`
- Modify: `.env_template`

- [ ] **Step 1: Write the MLflow wrapper script**

```python
"""Run the HLS evaluation under MLflow tracking and log metrics."""
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    mlflow.set_experiment("burn-scar-benchmark")
    with mlflow.start_run(run_name=f"hls-eval-{limit or 'full'}") as run:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_evaluation.py")]
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        artifact_dir = max(
            (ROOT / "reports" / "evaluation").glob("*/"),
            key=lambda p: p.stat().st_mtime,
        )
        import json

        summary = json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        )
        for group, metrics in summary.items():
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{group}.{name}", value)
        mlflow.log_artifacts(str(artifact_dir))
        print(f"Run logged: {run.info.run_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add mlflow to dev dependencies**

In `pyproject.toml`, change:

```toml
[dependency-groups]
dev = ["ruff>=0.9.0"]
```

to:

```toml
[dependency-groups]
dev = ["ruff>=0.9.0", "mlflow>=2.16.0"]
```

- [ ] **Step 3: Update .env_template**

Append:

```
# Set to 1 to run the gated V2-300M integration test (downloads ~1.3GB checkpoint)
RUN_MODEL_TESTS=
```

- [ ] **Step 4: Sync and verify import**

Run: `uv sync --group dev && PYTHONPATH=. uv run python -c "import mlflow"`
Expected: no error.

- [ ] **Step 5: Smoke the tracking script (optional, small run)**

Run: `PYTHONPATH=. uv run python scripts/track_eval_mlflow.py 1`
Expected: runs eval with limit 1 and logs metrics to the local MLflow store.

- [ ] **Step 6: Commit**

```bash
git add scripts/track_eval_mlflow.py pyproject.toml .env_template uv.lock
git commit -m "feat: add MLflow-tracked eval run and dev dependency"
```

### Task 15: Secrets shim for Streamlit Cloud

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Add the secrets shim**

At the very top of `dashboard/app.py`, before the `sys.path.insert` line, add:

```python
# Streamlit Cloud serves secrets via st.secrets; expose them as env vars so the
# data pipeline (which reads os.environ) works unchanged.
try:
    import os
    import streamlit as st

    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass
```

Note: the existing file already imports `os`? No — verify. `app.py` imports `io, json, sys` and `from pathlib import Path`. `os` may not be imported. Keep the local `import os` inside the try block (shadowing is fine) OR add `import os` to the top imports. Prefer adding `import os` to the existing import block and use it inside the shim.

- [ ] **Step 2: Verify import**

Run: `PYTHONPATH=. uv run python -c "import dashboard.app"`
Expected: imports cleanly.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: expose Streamlit Cloud secrets as env vars for the pipeline"
```

### Task 16: Final verification + deploy notes

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full verification**

Run: `make lint && make test`
Expected: lint clean (no new E/F/I/W errors), all tests PASS.

- [ ] **Step 2: Update README**

Add a "Deployment" section and update the roadmap. Replace the `- [ ] Prithvi model fine-tuning (HLS Burn Scar dataset)` roadmap line with checked model items, and append:

```markdown
## Deployment

- **Local:** `docker compose up --build` then open http://localhost:8501
- **Live:** deploy the repo to Streamlit Community Cloud
  (main file `dashboard/app.py`). Add your API keys under
  Settings → Secrets (keys must match `.env_template`).

## Burn Scar Model

- Inference: `data_pipeline/model_inference_v2.py` (Prithvi-EO-2.0-300M)
- Validate vs official: `PYTHONPATH=. uv run python scripts/validate_prithvi_v2.py`
- Benchmark: `PYTHONPATH=. uv run python scripts/run_evaluation.py`
  (HLS test split: V2-300M vs V1-100M, plus dNBR cross-check)
- MLflow: `PYTHONPATH=. uv run python scripts/track_eval_mlflow.py [limit]`
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add deployment and model docs to README"
```

Phase 4 gate: live URL deployed (manual step in Streamlit Cloud) + CI green.

---

## Post-Plan Verification Checklist

- `make lint` clean
- `make test` green (integration tests skipped without `RUN_MODEL_TESTS=1`)
- `scripts/validate_prithvi_v2.py` accuracy ≥ 0.99 vs official
- `reports/evaluation/<ts>/summary.json` exists with v2_300m and v1_100m rows
- Dashboard Model tab renders benchmark + model card locally
- Docker image builds (if Docker available)
- Push to GitHub → CI lint+test green → deploy dashboard to Streamlit Cloud
