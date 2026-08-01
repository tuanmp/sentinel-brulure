from unittest.mock import Mock, patch

from data_pipeline import firm_request as fr


def _csv_text():
    return (
        "latitude,longitude,acq_date,acq_time,confidence,frp,daynight\n"
        "44.5,4.5,2026-07-12,1500,n,350.0,D\n"
        "44.6,4.6,2026-07-12,1600,h,500.0,D\n"
        "48.0,2.0,2026-07-12,1500,l,50.0,D\n"
    )


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_fetch_fire_events_supports_country_region(mock_getenv):
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = _csv_text()
        df = fr.fetch_fire_events(region="france", days_back=3, min_confidence="low")

    url = mock_get.call_args.args[0]
    assert "france" not in url
    assert "-5.5,41.0,9.5,51.5" in url
    assert len(df) == 3
    assert "confidence_label" in df.columns


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_fetch_fire_events_unknown_region_raises(mock_getenv):
    try:
        fr.fetch_fire_events(region="atlantis", days_back=3)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Unknown region" in str(exc)


def test_fetch_and_process_tags_country_on_events():
    with patch("data_pipeline.firm_request.fetch_fire_events") as mock_fetch, \
         patch("data_pipeline.firm_request.cluster_detections") as mock_cluster, \
         patch("data_pipeline.firm_request.filter_events") as mock_filter:
        mock_fetch.return_value = Mock()
        mock_cluster.return_value = [{"cluster_id": 1}]
        mock_filter.return_value = [{"cluster_id": 1}]
        events = fr.fetch_and_process(region="france", country="france")
    assert events[0]["country"] == "france"
