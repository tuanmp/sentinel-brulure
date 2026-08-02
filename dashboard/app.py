import sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.store import EventStore
from dashboard.imagery import (
    fetch_phase_bands,
    peak_frp_date,
    render_after,
    render_before,
    render_during,
)

st.set_page_config(page_title="Fire Analytics Dashboard", layout="wide")

STATUS_ORDER = ["detected", "active", "ended", "recovering", "complete"]
STATUS_COLORS = {
    "detected": "#1f77b4",
    "active": "#ff7f0e",
    "ended": "#d62728",
    "recovering": "#2ca02c",
    "complete": "#7f7f7f",
}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _event_row(event) -> dict:
    pre = event.prefire_metrics or {}
    post = event.postfire_assessment or {}
    peak = 0.0
    if event.during_observations:
        peak = max(o["frp_mw"] for o in event.during_observations)
    return {
        "id": event.event_id[:8],
        "country": event.country,
        "status": event.status,
        "cluster_id": event.cluster_id,
        "start_date": event.start_date,
        "end_date": event.end_date,
        "quiet_days": event.quiet_days,
        "burned_ha": post.get("burned_area_ha"),
        "peak_frp_mw": peak,
        "ndvi": pre.get("ndvi"),
        "weather_index": pre.get("weather_index"),
        "obs": len(event.during_observations),
        "failures": len(event.failures),
    }


def _events_frame(events) -> pd.DataFrame:
    rows = [_event_row(e) for e in events]
    return pd.DataFrame(rows)


def _status_color(status: str) -> str:
    return STATUS_COLORS.get(status, "#999999")


def _render_phase_cached(event, phase, date, resolution):
    """Fetch bands with an inline spinner; return figure or None."""
    try:
        bands = fetch_phase_bands(event, phase, date=date, resolution=resolution)
    except Exception as exc:  # API/token failure -> surface, don't crash
        st.error(f"Imagery fetch failed for {phase}: {exc}")
        return None
    if bands is None:
        return None
    if phase == "before":
        return render_before(event, bands)
    if phase == "during":
        return render_during(event, bands, date)
    pre = fetch_phase_bands(event, "before", date=None, resolution=resolution)
    if pre is None:
        return None
    return render_after(event, pre, bands, resolution)


def _frp_figure(event):
    obs = event.during_observations
    dates = [o["date"] for o in obs]
    frp = [o["frp_mw"] for o in obs]
    dets = [o["detection_count"] for o in obs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=frp, mode="lines+markers", name="FRP (MW)"))
    fig.add_trace(go.Bar(x=dates, y=dets, name="Detections", yaxis="y2", opacity=0.4))
    fig.update_layout(
        title="Fire Radiative Power over time",
        yaxis_title="FRP (MW)",
        yaxis2=dict(title="Detections", overlaying="y", side="right"),
        hovermode="x unified",
        height=400,
    )
    return fig


def main():
    with st.sidebar:
        st.title("Fire Analytics")
        root = st.text_input("Event store root", value="reports/events")
        resolution = st.selectbox("Imagery resolution (m)", [60, 120, 240], index=0)
        status_filter = st.multiselect(
            "Status filter", STATUS_ORDER, default=STATUS_ORDER
        )
        if st.button("Refresh store"):
            st.cache_data.clear()
            st.rerun()

    store = EventStore(root)
    events = [e for e in store.list_events() if e.status in status_filter]
    events.sort(key=lambda e: e.start_date, reverse=True)

    if not events:
        st.info(f"No events match the current filters in {root}.")
        return

    df = _events_frame(events)

    tab_overview, tab_progress, tab_during, tab_post, tab_recovery, tab_failures = (
        st.tabs(
            [
                "Events",
                "Satellite progression",
                "During (FRP)",
                "Post-fire severity",
                "Recovery",
                "Failures",
            ]
        )
    )

    with tab_overview:
        col_map, col_table = st.columns([1, 2])
        with col_map:
            map_df = df.copy()
            map_df["color"] = map_df["status"].map(_status_color)
            map_df["lat"] = [e.centroid_lat for e in events]
            map_df["lon"] = [e.centroid_lon for e in events]
            st.subheader("Event map (color = status)")
            st.map(map_df, latitude="lat", longitude="lon", color="color")
            st.caption(
                " • ".join(
                    f"{s} = {_status_color(s)}"
                    for s in STATUS_ORDER
                    if s in status_filter
                )
            )
        with col_table:
            st.subheader("Events")
            sel = st.dataframe(
                df,
                selection_mode="single-row",
                on_select="rerun",
                key="events_table",
                hide_index=True,
                width="stretch",
            )

    selected_rows = []
    try:
        selected_rows = list(sel.selection.rows)
    except Exception:
        pass
    if not selected_rows:
        with tab_overview:
            st.info("Select a row in the events table to inspect its details.")
        return

    event = events[selected_rows[0]]
    pre = event.prefire_metrics or {}
    post = event.postfire_assessment or {}

    with tab_overview:
        st.subheader(f"Event {event.event_id[:8]} — {event.country}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Status", event.status)
        c2.metric("Burned area (ha)", _fmt(post.get("burned_area_ha")))
        c3.metric("Peak FRP (MW)", _fmt(_event_row(event)["peak_frp_mw"]))
        c4.metric("Pre NDVI", _fmt(pre.get("ndvi")))
        c5.metric("Weather index", _fmt(pre.get("weather_index")))
        c6.metric("Failures", len(event.failures))
        st.write(
            {
                "cluster_id": event.cluster_id,
                "start_date": event.start_date,
                "end_date": event.end_date,
                "quiet_days": event.quiet_days,
                "bbox": event.bbox,
                "centroid": [event.centroid_lat, event.centroid_lon],
            }
        )

    with tab_progress:
        st.subheader("Before / During / After — Sentinel-2")
        st.caption(
            f"Imagery fetched on demand ({resolution} m) and cached under "
            "`reports/imagery/`. LEAST-CC mosaicking picks the clearest scene."
        )
        st.markdown("#### Before (pre-fire)")
        with st.spinner("Fetching pre-fire imagery..."):
            fig = _render_phase_cached(event, "before", None, resolution)
        if fig is None:
            st.warning("No Sentinel-2 imagery available for the pre-fire window.")
        else:
            st.pyplot(fig)

        st.markdown("#### During (mid-fire)")
        obs_dates = sorted({o["date"] for o in event.during_observations})
        peak = peak_frp_date(event)
        if obs_dates:
            min_d, max_d = (
                _date.fromisoformat(obs_dates[0]),
                _date.fromisoformat(obs_dates[-1]),
            )
        else:
            min_d, max_d = (
                _date.fromisoformat(event.start_date),
                _date.fromisoformat(event.end_date),
            )
        default_d = _date.fromisoformat(peak)
        if not (min_d <= default_d <= max_d):
            default_d = min_d
        selected = st.slider(
            "During date",
            min_value=min_d,
            max_value=max_d,
            value=default_d,
            format="YYYY-MM-DD",
            key="during_slider",
        )
        with st.spinner("Fetching during imagery..."):
            fig = _render_phase_cached(
                event, "during", selected.isoformat(), resolution
            )
        if fig is None:
            st.warning("No Sentinel-2 imagery available for the selected during date.")
        else:
            st.pyplot(fig)

        st.markdown("#### After (post-fire)")
        with st.spinner("Fetching post-fire imagery..."):
            fig = _render_phase_cached(event, "after", None, resolution)
        if fig is None:
            st.warning("No Sentinel-2 imagery available for the post-fire window.")
        else:
            st.pyplot(fig)

    with tab_during:
        st.plotly_chart(_frp_figure(event), width="stretch")
        st.dataframe(
            pd.DataFrame(event.during_observations),
            hide_index=True,
            width="stretch",
        )

    with tab_post:
        if not post:
            st.info("Post-fire assessment not computed yet.")
        else:
            severity = post.get("severity_classes", {})
            if severity:
                sv = pd.DataFrame(
                    {
                        "class": list(severity.keys()),
                        "share": list(severity.values()),
                    }
                )
                fig = go.Figure(
                    go.Bar(x=sv["class"], y=sv["share"], marker_color="#e31a1c")
                )
                fig.update_layout(
                    title="Burn severity class share", yaxis_title="Fraction"
                )
                st.plotly_chart(fig, width="stretch")
            st.write({k: v for k, v in post.items() if k != "severity_classes"})

    with tab_recovery:
        samples = event.recovery_samples
        if not samples:
            st.info("No recovery samples yet.")
        else:
            st.dataframe(pd.DataFrame(samples), hide_index=True, width="stretch")
            fig = go.Figure(
                go.Scatter(
                    x=[s["offset_months"] for s in samples],
                    y=[s.get("ndvi") for s in samples],
                    mode="lines+markers",
                    name="NDVI",
                )
            )
            fig.update_layout(
                title="Recovery NDVI by month offset", xaxis_title="Months after fire"
            )
            st.plotly_chart(fig, width="stretch")

    with tab_failures:
        if not event.failures:
            st.info("No recorded failures.")
        else:
            st.dataframe(pd.DataFrame(event.failures), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
