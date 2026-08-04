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
