from unittest.mock import Mock, patch

from analytics import fire_weather as fw


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


def test_fetch_fire_weather_parses_daily_rows():
    payload = {
        "daily": {
            "time": ["2026-07-01", "2026-07-02"],
            "temperature_2m_max": [30.0, 31.0],
            "wind_speed_10m_max": [20.0, 15.0],
            "precipitation_sum": [0.0, 1.0],
            "soil_moisture_0_to_10cm_mean": [0.12, 0.15],
        }
    }
    mock_resp = Mock()
    mock_resp.json.return_value = payload
    with patch(
        "analytics.fire_weather.requests.get", return_value=mock_resp
    ) as mock_get:
        rows = fw.fetch_fire_weather(44.5, 4.5, "2026-07-01", "2026-07-02")

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-01"
    assert rows[0]["fire_danger_index"] > rows[1]["fire_danger_index"]
    params = mock_get.call_args.kwargs["params"]
    assert params["latitude"] == 44.5
    assert "temperature_2m_max" in params["daily"]
