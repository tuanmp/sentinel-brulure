import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables shared by both endpoints; soil moisture is only offered by
# the forecast endpoint.
BASE_DAILY = "temperature_2m_max,wind_speed_10m_max,precipitation_sum"
SOIL_DAILY = "soil_moisture_0_to_10cm_mean"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_fire_danger(
    temp_c: float,
    wind_kmh: float,
    precip_mm: float,
    soil_moisture: float | None = None,
) -> float:
    """Simple 0-100 fire-danger proxy from daily weather fields.

    Uses a 4-factor score when soil moisture is available, otherwise a 3-factor
    score from temperature, wind, and precipitation.
    """
    temp_score = _clip((temp_c - 20.0) / 15.0, 0.0, 1.0)
    wind_score = _clip((wind_kmh - 5.0) / 30.0, 0.0, 1.0)
    precip_score = 1.0 - _clip(precip_mm / 10.0, 0.0, 1.0)
    if soil_moisture is None:
        return 100.0 * (temp_score + wind_score + precip_score) / 3.0
    moisture_score = 1.0 - _clip(soil_moisture / 0.4, 0.0, 1.0)
    return 100.0 * (temp_score + wind_score + precip_score + moisture_score) / 4.0


def _rows_from_daily(daily: dict) -> list[dict]:
    """Build per-day rows from an Open-Meteo daily payload, skipping None days."""
    rows = []
    for i, day in enumerate(daily["time"]):
        temp = daily["temperature_2m_max"][i]
        wind = daily["wind_speed_10m_max"][i]
        precip = daily["precipitation_sum"][i]
        if any(v is None for v in (temp, wind, precip)):
            continue
        soil = daily.get("soil_moisture_0_to_10cm_mean", [None] * len(daily["time"]))[i]
        rows.append(
            {
                "date": day,
                "temperature_max_c": temp,
                "wind_max_kmh": wind,
                "precipitation_mm": precip,
                "soil_moisture": soil,
                "fire_danger_index": compute_fire_danger(
                    temp, wind, precip, soil_moisture=soil
                ),
            }
        )
    return rows


def _has_soil(daily: dict) -> bool:
    soil = daily.get("soil_moisture_0_to_10cm_mean") or []
    return len(soil) > 0 and any(v is not None for v in soil)


def _get_daily(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "Open-Meteo API error"))
    return data


def fetch_fire_weather(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> list[dict]:
    """Fetch per-day weather rows for a lat/lon window.

    Prefers the Open-Meteo forecast endpoint, which provides soil moisture for
    roughly the last ~3 months. When the window predates forecast coverage (or
    the forecast lacks soil moisture), falls back to the archive endpoint with
    a 3-factor fire-danger index.
    """
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": f"{BASE_DAILY},{SOIL_DAILY}",
        "timezone": "UTC",
    }

    try:
        forecast = _get_daily(FORECAST_URL, common)
        if _has_soil(forecast.get("daily", {})):
            return _rows_from_daily(forecast["daily"])
    except ValueError:
        pass  # out of range or error body -> try archive

    archive = _get_daily(
        ARCHIVE_URL,
        {**common, "daily": BASE_DAILY},
    )
    return _rows_from_daily(archive["daily"])
