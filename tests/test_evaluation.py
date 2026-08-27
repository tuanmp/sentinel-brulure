import numpy as np

from analytics.evaluation import aggregate_metrics, binary_metrics, dnbr_mask


def test_binary_metrics_ignores_nodata_and_scores_known_case():
    pred = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    true = np.array([[1, 1, -1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8)
    m = binary_metrics(pred, true)
    # valid pixels: (0,0)=TP, (0,1)=TP, (1,0)=TN, (1,1)=TP, (1,2)=TN, (2,1)=TN, (2,2)=FP
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 0 and m["tn"] == 3
    assert m["iou"] == 3 / 4
    assert m["dice"] == 2 * 3 / (2 * 3 + 1)
    assert m["precision"] == 3 / 4
    assert m["recall"] == 1.0


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
