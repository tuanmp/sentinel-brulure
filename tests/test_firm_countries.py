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


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_fetch_fire_events_date_anchor_builds_paged_url(mock_getenv):
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = _csv_text()
        df = fr.fetch_fire_events(
            region="france", days_back=5, date="2026-07-01", min_confidence="low"
        )

    url = mock_get.call_args.args[0]
    assert url.endswith("/5/2026-07-01")
    assert len(df) == 3


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_fetch_fire_events_rejects_days_back_out_of_range(mock_getenv):
    for bad in (0, 6, 10):
        try:
            fr.fetch_fire_events(region="france", days_back=bad)
            raise AssertionError(f"expected ValueError for days_back={bad}")
        except ValueError as exc:
            assert "days_back" in str(exc)


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_get_data_availability_parses_min_max(mock_getenv):
    avail_text = "data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-04-27\n"
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = avail_text
        fr.get_data_availability.cache_clear()
        min_date, max_date = fr.get_data_availability("VIIRS_SNPP_SP")

    assert str(min_date) == "2012-01-20"
    assert str(max_date) == "2026-04-27"


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_pick_source_uses_nrt_when_covered(mock_getenv):
    avail_text = "data_id,min_date,max_date\nVIIRS_SNPP_NRT,2026-04-28,2026-08-01\n"
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = avail_text
        fr.get_data_availability.cache_clear()
        from datetime import date

        source = fr.pick_source(date(2026, 7, 1), date(2026, 7, 5), "VIIRS_SNPP_NRT")

    assert source == "VIIRS_SNPP_NRT"


@patch("data_pipeline.firm_request.os.getenv", return_value="test-api-key")
def test_pick_source_falls_back_to_archive(mock_getenv):
    avail_text = "data_id,min_date,max_date\nVIIRS_SNPP_NRT,2026-04-28,2026-08-01\n"
    with patch("data_pipeline.firm_request.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.text = avail_text
        fr.get_data_availability.cache_clear()
        from datetime import date

        source = fr.pick_source(date(2019, 9, 1), date(2019, 9, 5), "VIIRS_SNPP_NRT")

    assert source == "VIIRS_SNPP_SP"


def test_fetch_and_process_tags_country_on_events():
    with (
        patch("data_pipeline.firm_request.fetch_fire_events") as mock_fetch,
        patch("data_pipeline.firm_request.cluster_detections") as mock_cluster,
        patch("data_pipeline.firm_request.filter_events") as mock_filter,
    ):
        mock_fetch.return_value = Mock()
        mock_cluster.return_value = [{"cluster_id": 1}]
        mock_filter.return_value = [{"cluster_id": 1}]
        events = fr.fetch_and_process(region="france", country="france")
    assert events[0]["country"] == "france"
