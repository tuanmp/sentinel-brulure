from analytics import report
from analytics.event import FireEvent


def _event():
    return FireEvent(
        event_id="e1",
        country="france",
        bbox=[4.2, 44.3, 4.8, 44.8],
        centroid_lat=44.55,
        centroid_lon=4.5,
        start_date="2026-07-12",
        end_date="2026-07-18",
        status="active",
        prefire_metrics={"ndvi": 0.5, "ndwi": -0.1, "weather_index": 55.0},
        during_observations=[
            {
                "date": "2026-07-18",
                "frp_mw": 100.0,
                "detection_count": 5,
                "bbox_growth_deg": 0.1,
            },
        ],
    )


def test_export_json_round_trips():
    event = _event()
    restored = FireEvent.from_dict(report.export_json(event))
    assert restored.country == "france"
    assert restored.status == "active"


def test_export_csv():
    csv_text = report.export_csv(_event().during_observations)
    assert "date,frp_mw,detection_count,bbox_growth_deg" in csv_text
    assert "2026-07-18,100.0,5,0.1" in csv_text


def test_export_csv_empty():
    assert report.export_csv([]) == ""


def test_frp_chart_returns_base64_png():
    import base64

    chart = report._frp_chart(_event())
    decoded = base64.b64decode(chart)
    assert decoded.startswith(b"\x89PNG")


def test_frp_chart_empty_observations_returns_png():
    import base64

    event = _event()
    event.during_observations = []
    chart = report._frp_chart(event)
    decoded = base64.b64decode(chart)
    assert decoded.startswith(b"\x89PNG")


def test_build_html_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_frp_chart", lambda event: "CHART_DATA")
    out = tmp_path / "e1.html"
    report.build_html_report(_event(), out)
    html = out.read_text()
    assert "france" in html
    assert "CHART_DATA" in html
    assert "Fire Radiative Power" in html
    assert "Recovery" in html


def test_render_html_returns_matching_string(monkeypatch):
    monkeypatch.setattr(report, "_frp_chart", lambda event: "CHART_DATA")
    html = report.render_html(_event())
    assert html.startswith("<!DOCTYPE html>")
    assert "france" in html
    assert "CHART_DATA" in html
