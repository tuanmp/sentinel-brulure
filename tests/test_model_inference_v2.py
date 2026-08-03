from unittest.mock import patch

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
