from unittest.mock import Mock, patch

import pytest

from analytics import fire_weather as fw


def _daily_payload(**daily_overrides):
    daily = {
        "time": ["2026-07-01", "2026-07-02"],
        "temperature_2m_max": [30.0, 31.0],
        "wind_speed_10m_max": [20.0, 15.0],
        "precipitation_sum": [0.0, 1.0],
        "soil_moisture_0_to_10cm_mean": [0.12, 0.15],
    }
    daily.update(daily_overrides)
    return {"daily": daily}


def _mock_response(payload):
    mock_resp = Mock()
    mock_resp.json.return_value = payload
    return mock_resp


def test_fire_danger_low_for_cool_wet_conditions():
    score = fw.compute_fire_danger(
        temp_c=15, wind_kmh=5, precip_mm=20, soil_moisture=0.4
    )
    assert score == 0.0


def test_fire_danger_high_for_hot_dry_windy():
    score = fw.compute_fire_danger(
        temp_c=38, wind_kmh=45, precip_mm=0, soil_moisture=0.0
    )
    assert score == 100.0


def test_fire_danger_three_factor_saturated_without_soil():
    score = fw.compute_fire_danger(temp_c=38, wind_kmh=45, precip_mm=0)
    assert score == 100.0


def test_fire_danger_three_factor_cool_wet_without_soil():
    score = fw.compute_fire_danger(temp_c=15, wind_kmh=5, precip_mm=20)
    assert score == 0.0


def test_fetch_fire_weather_prefers_forecast_with_soil():
    payload = _daily_payload()
    mock_resp = _mock_response(payload)
    with patch(
        "analytics.fire_weather.requests.get", return_value=mock_resp
    ) as mock_get:
        rows = fw.fetch_fire_weather(44.5, 4.5, "2026-07-01", "2026-07-02")

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-01"
    assert rows[0]["soil_moisture"] == 0.12
    assert rows[0]["fire_danger_index"] > rows[1]["fire_danger_index"]
    assert mock_get.call_count == 1
    forecast_url = mock_get.call_args.args[0]
    assert forecast_url == fw.FORECAST_URL
    params = mock_get.call_args.kwargs["params"]
    assert params["latitude"] == 44.5
    assert "soil_moisture_0_to_10cm_mean" in params["daily"]


def test_fetch_falls_back_to_archive_when_forecast_out_of_range():
    forecast_payload = {
        "error": True,
        "reason": "Parameter 'start_date' is out of allowed range",
    }
    archive_payload = _daily_payload(soil_moisture_0_to_10cm_mean=[None, None])
    mock_forecast = _mock_response(forecast_payload)
    mock_archive = _mock_response(archive_payload)
    with patch(
        "analytics.fire_weather.requests.get",
        side_effect=[mock_forecast, mock_archive],
    ) as mock_get:
        rows = fw.fetch_fire_weather(44.5, 4.5, "2019-09-01", "2019-09-02")

    assert len(rows) == 2
    assert rows[0]["soil_moisture"] is None
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[0].args[0] == fw.FORECAST_URL
    assert mock_get.call_args_list[1].args[0] == fw.ARCHIVE_URL
    archive_params = mock_get.call_args_list[1].kwargs["params"]
    assert "soil_moisture_0_to_10cm_mean" not in archive_params["daily"]


def test_fetch_falls_back_when_forecast_lacks_soil():
    forecast_payload = _daily_payload(soil_moisture_0_to_10cm_mean=[None, None])
    archive_payload = _daily_payload(soil_moisture_0_to_10cm_mean=[None, None])
    mock_forecast = _mock_response(forecast_payload)
    mock_archive = _mock_response(archive_payload)
    with patch(
        "analytics.fire_weather.requests.get",
        side_effect=[mock_forecast, mock_archive],
    ) as mock_get:
        rows = fw.fetch_fire_weather(44.5, 4.5, "2026-07-01", "2026-07-02")

    assert len(rows) == 2
    assert rows[0]["soil_moisture"] is None
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[1].args[0] == fw.ARCHIVE_URL


def test_fetch_skips_days_with_missing_values():
    payload = _daily_payload(
        temperature_2m_max=[30.0, None],
    )
    mock_resp = _mock_response(payload)
    with patch("analytics.fire_weather.requests.get", return_value=mock_resp):
        rows = fw.fetch_fire_weather(44.5, 4.5, "2026-07-01", "2026-07-02")

    assert len(rows) == 1
    assert rows[0]["date"] == "2026-07-01"


def test_fetch_raises_on_archive_error_body():
    forecast_payload = {
        "error": True,
        "reason": "Parameter 'start_date' is out of allowed range",
    }
    archive_payload = {"error": True, "reason": "Invalid date range"}
    mock_forecast = _mock_response(forecast_payload)
    mock_archive = _mock_response(archive_payload)
    with patch(
        "analytics.fire_weather.requests.get",
        side_effect=[mock_forecast, mock_archive],
    ):
        with pytest.raises(ValueError, match="Invalid date range"):
            fw.fetch_fire_weather(44.5, 4.5, "2019-09-01", "2019-09-02")


def test_fire_danger_at_clip_bounds():
    score = fw.compute_fire_danger(
        temp_c=20, wind_kmh=5, precip_mm=10, soil_moisture=0.4
    )
    assert score == 0.0


def test_fire_danger_at_midpoint_scores():
    score = fw.compute_fire_danger(
        temp_c=27.5, wind_kmh=20, precip_mm=5, soil_moisture=0.2
    )
    assert score == 50.0
