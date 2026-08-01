import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_fire_danger(
    temp_c: float, wind_kmh: float, precip_mm: float, soil_moisture: float
) -> float:
    """Simple 0-100 fire-danger proxy from daily weather fields."""
    temp_score = _clip((temp_c - 20.0) / 15.0, 0.0, 1.0)
    wind_score = _clip((wind_kmh - 5.0) / 30.0, 0.0, 1.0)
    precip_score = 1.0 - _clip(precip_mm / 10.0, 0.0, 1.0)
    moisture_score = 1.0 - _clip(soil_moisture / 0.4, 0.0, 1.0)
    return 100.0 * (temp_score + wind_score + precip_score + moisture_score) / 4.0


def fetch_fire_weather(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> list[dict]:
    """Fetch per-day weather rows for a lat/lon window from the Open-Meteo archive."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_max,wind_speed_10m_max,"
            "precipitation_sum,soil_moisture_0_to_10cm_mean"
        ),
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "Open-Meteo API error"))
    daily = data["daily"]

    rows = []
    for i, date in enumerate(daily["time"]):
        values = [
            daily["temperature_2m_max"][i],
            daily["wind_speed_10m_max"][i],
            daily["precipitation_sum"][i],
            daily["soil_moisture_0_to_10cm_mean"][i],
        ]
        if any(v is None for v in values):
            continue
        temp, wind, precip, moisture = values
        rows.append(
            {
                "date": date,
                "temperature_max_c": temp,
                "wind_max_kmh": wind,
                "precipitation_mm": precip,
                "soil_moisture": moisture,
                "fire_danger_index": compute_fire_danger(temp, wind, precip, moisture),
            }
        )
    return rows
