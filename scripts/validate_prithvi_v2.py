"""Cross-check our V2-300M inference against the official inference.py.

Downloads the official example chip, runs both paths, and reports pixel-wise
agreement between the class masks.
"""

import argparse
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

    config = (
        Path(__file__).resolve().parent.parent
        / "data_pipeline"
        / "configs"
        / "prithvi_v2_burn_scars_inference.yaml"
    )

    print("Running our inference...")
    model = load_model(checkpoint_path=ckpt)
    probs, mask = run_inference(str(example), model=model)
    print(
        f"  probs range [{probs.min():.4f}, {probs.max():.4f}], "
        f"burn fraction {np.mean(mask == 1):.3%}"
    )

    print("Running official inference.py...")
    out_dir = work / "official_out"
    cmd = [
        sys.executable,
        str(official),
        "--data_file",
        str(example),
        "--config",
        str(config),
        "--checkpoint",
        str(ckpt),
        "--output_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(work))

    pred_file = out_dir / f"pred_{Path(example).stem}.tiff"
    with rasterio.open(pred_file) as src:
        official_mask = src.read(1)
    official_mask = (official_mask > 0).astype(np.uint8)

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
