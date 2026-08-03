"""Evaluation harness for burn scar models on the HLS Burn Scars dataset.

Scores V2-300M and the legacy 100M model against the test split, plus a dNBR
cross-check on live events from the event store. Writes a benchmark artifact
to reports/evaluation/<timestamp>/.
"""
import argparse
import json
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
from data_pipeline.model_inference_v2 import load_model
from data_pipeline.model_inference_v2 import predict as predict_v2
from data_pipeline.model_inference_v2 import run_inference as run_v2

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
            repo_type="dataset",
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
    image = None
    for sub in ("training", "validation"):
        matches = list((data_root / sub).glob(f"*{prefix}*_merged.tif"))
        if matches:
            image = matches[0]
            break
    if image is None:
        raise FileNotFoundError(f"image for {prefix} not found under {data_root}")
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
        _, mask_v2 = predict_v2(post[:6], model=model, device=device)
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
