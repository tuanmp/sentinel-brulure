# Burn Scar Intelligence Platform — Model Evaluation + Live Demo

## Overview

Turn the existing burn-scar pipeline into a portfolio-ready product whose
centerpiece is a defensible ML evaluation story backed by a live demo.

The project already has: Sentinel-2 + FIRMS data pipeline, NBR/dNBR computation,
a hand-rolled Prithvi-EO-1.0-100M inference module, per-event lifecycle analytics
(before/during/after/recovery), auto-generated HTML reports, a Streamlit
dashboard with map + linked plots, and a test suite.

This build-out does four things:

1. **Upgrade the model** — integrate the newer, stronger
   `Prithvi-EO-2.0-300M-BurnScars` model (87.5 burned-class IoU) as a standalone
   TerraTorch-backed module, validated against the official inference script.
2. **Prove it works** — an evaluation harness scoring the V2-300M mask against
   the HLS Burn Scar dataset (headline numbers) plus a dNBR cross-check on live
   events, benchmarked against the dNBR baseline and the old 100M model.
3. **Show it live** — burn-scar overlay + before/after + confidence heatmap in the
   existing Streamlit dashboard, with a "Model" tab surfacing the benchmark.
4. **Ship it properly** — Docker, CI (lint + test), and deployment to Streamlit
   Community Cloud.

Fine-tuning is explicitly deferred (Phase 5, design-only).

## Phase 1 — V2-300M Standalone Inference

### Model facts

| Property | Value |
|----------|-------|
| Repo | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` |
| Checkpoint | `Prithvi_EO_V2_300M_BurnScars.pt` (~1.3 GB) |
| Backbone | `prithvi_eo_v2_300` (embed_dim 1024, depth 24) |
| Decoder | UNetDecoder, channels `[512, 256, 128, 64]` |
| Input bands | BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2 (6 bands) |
| Patch size | 512×512 |
| Classes | 0 = not burned, 1 = burn scar (ignore_index = -1) |
| License | Apache-2.0 |
| Reference | arXiv 2412.02732 |

Our existing `evalscript` (sentinel_request.py) already returns exactly these six
bands in this order (B02, B03, B04, B08, B11, B12), so no fetch-side changes.

### New module — `data_pipeline/model_inference_v2.py`

Mirrors the old `model_inference.py` API but is TerraTorch-backed. The old module
and `model_loader.py` are left untouched.

- `MODEL_REPO`, `CHECKPOINT_NAME` as above.
- `get_model_dir()` — reuse `~/.cache/sentinel-sat/models` (shared with 100M).
- `download_checkpoint()` — `hf_hub_download`, cached.
- `load_model(checkpoint_path, device)` → wraps `LightningInferenceModel`
  (`from_config` with the bundled config); exposes `.model` and `.datamodule`.
- `prepare_input(geotiff_path)` → `(1, 6, 1, H, W)` float32 tensor,
  `/10000` scale when values exceed 1 (matches official inference.py).
- `run_inference(geotiff_path, model, device)` → `(probs, class_mask)`:
  - `probs`: softmax of burn-class logits, `(H, W)` float32 in [0, 1]
  - `class_mask`: argmax, `(H, W)` uint8
  - Sliding-window 512×512, reflect-pad to a 512 multiple, non-overlapping
    windows via `unfold`/`rearrange`, per-window `datamodule.test_transform` +
    `datamodule.aug` (applies per-band mean/std), reassemble, crop back.
- `save_mask(mask, output_path, reference_geotiff)` — GeoTIFF write reusing the
  old module's pattern (count=1, float32 for probs).
- Device selection: `cuda > mps > cpu` (repo currently cuda-only).

### Bundled config — `data_pipeline/configs/prithvi_v2_burn_scars_inference.yaml`

Copy of the official `burn_scars_config.yaml` with exactly one change:
`backbone_pretrained: false`. This skips re-downloading the ~2 GB base
`Prithvi-EO-2.0-300M` model — the fine-tuned checkpoint is fully loaded over the
built architecture. Keeps backbone config, UNetDecoder, means/stds, and the
single `ToTensorV2` test_transform.

### Validation — `scripts/validate_prithvi_v2.py`

- Downloads the official example chip from the HF repo
  (`examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif`).
- Runs our `run_inference` → probs + class mask.
- Runs the official `inference.py` as a subprocess on the same chip →
  reference class mask.
- Reports pixel-wise accuracy / IoU vs the reference and burn-area fraction;
  saves a side-by-side PNG.

### Tests — `tests/test_model_inference_v2.py`

Offline, fast, no model download:

- `prepare_input` scaling (values > 1 → /10000; small values untouched).
- Pad / unfold / reassemble round-trip preserves spatial layout.
- `save_mask` writes a georeferenced single-band GeoTIFF.
- Device-selection helper prefers cuda > mps > cpu.

One integration test gated by `RUN_MODEL_TESTS=1` (skipped otherwise) that
downloads the checkpoint and asserts `run_inference` output shape/range on a tiny
synthetic 6-band GeoTIFF.

## Phase 2 — Evaluation Harness

### New module — `analytics/evaluation.py` + `scripts/run_evaluation.py`

Metrics on the burn class (class index 1): IoU, Dice, precision, recall. Reported
per-event and aggregated.

**Ground truth / baselines:**

- **Headline (labeled):** HLS Burn Scar dataset
  (`ibm-nasa-geospatial/hls_burn_scars`) — native domain, ~800 labeled 512×512
  chips with train/val/test splits in the model repo. Score V2-300M against the
  test split. This gives the defensible IoU/Dice numbers.
- **Cross-check (unlabeled live events):** score V2-300M vs a dNBR-threshold mask
  (dNBR > 0.27 ~ moderate+) on real events pulled through the existing pipeline.
  Frame as a detection-sanity check, not ground truth.

**Benchmarks (headline eval):** V2-300M on the HLS test split (labeled IoU/Dice
plus the dNBR cross-check on live events).

> **Decision:** the old 100M model is NOT a benchmark baseline. Its hand-rolled
> decoder never loads real weights (the checkpoint stores a UPerNet-style
> `neck.*`/`decode_head.*` decoder, but `model_inference.py` only loads
> `backbone.*` with `strict=False`), so its output is random noise on every
> fresh `load_model()`. The V2-300M model is scored against HLS labels plus the
> dNBR cross-check only.

Output artifact to `reports/evaluation/<run-timestamp>/`: metrics table (JSON/CSV)
and an IoU/Dice bar chart (matplotlib), consumed by the dashboard's Model tab.

### Scope note

The HLS eval runs in a standalone script/notebook harness, not inside the
Streamlit app (dataset download is heavy). The dashboard reads the produced
artifact.

## Phase 3 — Dashboard Integration

Extend the existing Streamlit app (`dashboard/app.py` and friends):

- Burn-scar overlay on the event map (V2-300M probs as a colormap layer /
  class mask as a binary overlay).
- Before/after slider (pre-fire vs post-fire true color, already partially
  present) plus a confidence heatmap toggle.
- Inference cached per tile (module-level LRU / simple dict) so the demo stays
  responsive.
- New "Model" tab:
  - Benchmark table (V2-300M vs 100M vs dNBR: IoU, Dice, P/R)
  - Model card: architecture, 6 bands, normalization, license, citation
  - Live metrics read from the latest `reports/evaluation/` artifact.

The V2 module is wired into the dashboard surface only. `process_fire_event` /
`--use-model` stays on the 100M path for now (no regressions to the CLI demo).

## Phase 4 — Deployment + Light MLOps

- `Dockerfile` (Python 3.12, dashboard + model deps) + `docker-compose.yml`.
- GitHub Actions workflow: `make lint` + `make test` on push/PR.
- Deploy the Streamlit dashboard to **Streamlit Community Cloud** (free tier,
  secrets from `.env`/`.env_template`, one-command GitHub deploy).
- One tracked MLflow run around the headline eval to demonstrate the MLOps
  signal (run name, metrics logged, artifact referenced). Adds `mlflow` to
  `pyproject.toml` dev dependencies.

### Env vars

No new required secrets. `.env_template` gains optional `RUN_MODEL_TESTS`
documented for the gated integration test. Sentinel/FIRMS vars unchanged.

## Phase 5 — Fine-tuning (deferred, design only)

Design a Colab notebook + dataset builder that seeds a small Sentinel-2 training
set from the pipeline's highest-confidence predictions and HLS labels, then
fine-tunes V2-300M and re-runs the eval to show a metric delta. **Not executed
in this build-out.**

## Out of Scope

- Executing Phase 5 fine-tuning.
- Deleting/refactoring the old 100M module or `model_loader.py`.
- FIRMS pipeline changes.
- FastAPI/React rewrite — Streamlit remains the product surface.

## Sequence & Gates

| Phase | Exit gate |
|-------|-----------|
| 1 | `validate_prithvi_v2.py` matches official inference output on the example chip |
| 2 | `run_evaluation.py` produces a reproducible benchmark artifact |
| 3 | Dashboard Model tab + overlay demo works locally |
| 4 | Live URL deployed; CI green |

Each phase is independently shippable; work is sequential but each gate is a
reviewable checkpoint.
