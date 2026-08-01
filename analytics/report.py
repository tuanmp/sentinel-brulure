import base64
import csv
import io
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .event import FireEvent


def export_json(event: FireEvent) -> dict:
    return event.to_dict()


def _esc(value) -> str:
    return escape(str(value)) if value is not None else "None"


def export_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\n")


def _frp_chart(event: FireEvent) -> str:
    obs = event.during_observations
    fig, ax = plt.subplots(figsize=(6, 3))
    dates = [o["date"] for o in obs]
    frp = [o["frp_mw"] for o in obs]
    if dates:
        ax.plot(dates, frp, marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("FRP (MW)")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Fire Radiative Power over time")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _metrics_table(event: FireEvent) -> str:
    pre = event.prefire_metrics or {}
    rows = [
        ("NDVI", pre.get("ndvi")),
        ("NDWI", pre.get("ndwi")),
        ("Weather index (0-100)", pre.get("weather_index")),
        ("Burned area (ha)", (event.postfire_assessment or {}).get("burned_area_ha")),
    ]
    cells = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(value)}</td></tr>" for name, value in rows
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{cells}</table>"


def _severity_table(event: FireEvent) -> str:
    classes = (event.postfire_assessment or {}).get("severity_classes", {})
    cells = "".join(
        f"<tr><td>{_esc(name)}</td><td>{fraction:.1%}</td></tr>"
        for name, fraction in classes.items()
    )
    return f"<table><tr><th>Class</th><th>Share</th></tr>{cells}</table>"


def _recovery_table(event: FireEvent) -> str:
    samples = event.recovery_samples
    if not samples:
        return "<p>No recovery samples yet.</p>"
    rows = "".join(
        f"<tr><td>{_esc(s['offset_months'])}</td><td>{_esc(s.get('ndvi'))}</td>"
        f"<td>{_esc(s.get('regrowth_ratio'))}</td></tr>"
        for s in samples
    )
    return (
        "<table><tr><th>Months</th><th>NDVI</th><th>Regrowth ratio</th></tr>"
        f"{rows}</table>"
    )


def build_html_report(event: FireEvent, output_path) -> None:
    pre = event.prefire_metrics or {}
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Fire {_esc(event.event_id)}</title></head>
<body>
<h1>Fire Report: {_esc(event.event_id)}</h1>
<p><strong>Country:</strong> {_esc(event.country)} &nbsp;
<strong>Status:</strong> {_esc(event.status)} &nbsp;
<strong>Window:</strong> {_esc(event.start_date)} → {_esc(event.end_date)}</p>
<h2>Before: Fuel State</h2>
<p>NDVI {_esc(pre.get("ndvi"))}, NDWI {_esc(pre.get("ndwi"))}, weather index {_esc(pre.get("weather_index"))}</p>
<h2>During: Fire Radiative Power</h2>
<img src="data:image/png;base64,{_frp_chart(event)}" alt="FRP over time">
<h2>After: Burn Severity</h2>
{_metrics_table(event)}
{_severity_table(event)}
<h2>Recovery</h2>
{_recovery_table(event)}
</body>
</html>
"""
    Path(output_path).write_text(html)
