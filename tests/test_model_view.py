import json
from pathlib import Path

import numpy as np

from dashboard.model_view import (
    latest_eval_dir,
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


def test_latest_eval_dir_picks_newest(tmp_path):
    _write_eval(tmp_path, "20260101_000000", 0.5)
    _write_eval(tmp_path, "20260102_000000", 0.8)
    assert latest_eval_dir(tmp_path) == tmp_path / "20260102_000000"


def test_latest_eval_dir_none_when_empty(tmp_path):
    assert latest_eval_dir(tmp_path) is None


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
