import io
import json
import os
import sys
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta
from pathlib import Path

# Streamlit Cloud serves secrets via st.secrets; expose them as env vars so
# the data pipeline (which reads os.environ) works unchanged.
try:
    import streamlit as st

    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.report import export_csv, export_json, render_html
from analytics.store import EventStore
from dashboard import glossary
from dashboard.imagery import (
    cached_severity_areas,
    fetch_phase_bands,
    peak_frp_date,
    render_after,
    render_before,
    render_during,
    valid_fraction,
)
from dashboard.linked import build_linked_figure
from dashboard.model_view import (
    latest_eval_dir,
    load_latest_eval,
    model_card_markdown,
    render_overlay,
)
from data_pipeline.sentinel_request import compute_post_window

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
    return pd.DataFrame([_event_row(e) for e in events])


def _status_color(status: str) -> str:
    return STATUS_COLORS.get(status, "#999999")


def _events_column_config():
    g = glossary.GLOSSARY
    return {
        "status": st.column_config.TextColumn(
            "Status", help="Lifecycle stage — see the glossary in the sidebar."
        ),
        "cluster_id": st.column_config.NumberColumn("Cluster", help=g["cluster_id"]),
        "start_date": st.column_config.TextColumn(
            "Start", help="First day of fire detections."
        ),
        "end_date": st.column_config.TextColumn(
            "End", help="Last day of fire detections."
        ),
        "quiet_days": st.column_config.NumberColumn("Quiet days", help=g["quiet_days"]),
        "burned_ha": st.column_config.NumberColumn(
            "Burned (ha)", help=g["burned_area_ha"]
        ),
        "peak_frp_mw": st.column_config.NumberColumn("Peak FRP (MW)", help=g["FRP"]),
        "ndvi": st.column_config.NumberColumn("Pre NDVI", help=g["NDVI"]),
        "weather_index": st.column_config.NumberColumn(
            "Weather", help=g["weather_index"]
        ),
        "obs": st.column_config.NumberColumn(
            "Obs", help="Number of daily during-phase observations."
        ),
        "failures": st.column_config.NumberColumn(
            "Failures", help="Transient tracking failures recorded for this event."
        ),
    }


def _obs_column_config():
    g = glossary.GLOSSARY
    return {
        "date": st.column_config.TextColumn("Date", help="Observation day."),
        "frp_mw": st.column_config.NumberColumn("FRP (MW)", help=g["FRP"]),
        "detection_count": st.column_config.NumberColumn(
            "Detections", help=g["detection_count"]
        ),
        "bbox_growth_deg": st.column_config.NumberColumn(
            "BBox growth (deg)", help=g["bbox_growth_deg"]
        ),
    }


def _kpi_row(events) -> None:
    counts = {s: sum(1 for e in events if e.status == s) for s in STATUS_ORDER}
    total_ha = sum(
        (e.postfire_assessment or {}).get("burned_area_ha") or 0.0 for e in events
    )
    peaks = [
        max((o["frp_mw"] for o in e.during_observations), default=0.0) for e in events
    ]
    max_frp = max(peaks) if peaks else 0.0
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Events", len(events))
    c2.metric("Active", counts["active"])
    c3.metric("Ended", counts["ended"])
    c4.metric("Recovering", counts["recovering"])
    c5.metric("Complete", counts["complete"])
    c6.metric("Total burned (ha)", f"{total_ha:,.0f}")
    c7.metric("Max peak FRP (MW)", f"{max_frp:,.0f}")


def _load_bands(event, phase, date, resolution):
    try:
        return fetch_phase_bands(event, phase, date=date, resolution=resolution)
    except Exception as exc:  # API/token failure -> surface, don't crash
        st.error(f"Imagery fetch failed for {phase}: {exc}")
        return None


def _badge_if_low_valid(bands, label) -> None:
    frac = valid_fraction(bands)
    if frac < 0.30:
        st.warning(
            f"{label}: only {frac:.0%} of pixels have valid data — the scene may "
            "be cloudy or no-data."
        )


def _fig_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def _frp_figure(event):
    obs = event.during_observations
    dates = [o["date"] for o in obs]
    frp = [o["frp_mw"] for o in obs]
    dets = [o["detection_count"] for o in obs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=frp, mode="lines+markers", name="FRP (MW)"))
    fig.add_trace(go.Bar(x=dates, y=dets, name="Detections", yaxis="y2", opacity=0.4))
    if frp:
        peak_i = max(range(len(frp)), key=lambda i: frp[i])
        fig.add_trace(
            go.Scatter(
                x=[dates[peak_i]],
                y=[frp[peak_i]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star"),
                name="peak FRP",
            )
        )
    fig.update_layout(
        title="Fire Radiative Power over time",
        yaxis_title="FRP (MW)",
        yaxis2=dict(title="Detections", overlaying="y", side="right"),
        hovermode="x unified",
        height=400,
    )
    return fig


def _lifecycle_figure(event):
    s = _datetime.fromisoformat(event.start_date)
    e = _datetime.fromisoformat(event.end_date)
    post_s, post_e = compute_post_window(event.start_date, event.end_date)
    post_s, post_e = _datetime.fromisoformat(post_s), _datetime.fromisoformat(post_e)
    fig = go.Figure()
    rows = [
        ("pre-fire", s - timedelta(days=15), s - timedelta(days=1), "#1f77b4"),
        ("active (burning)", s, e, "#ff7f0e"),
        ("post-fire assessment", post_s, post_e, "#d62728"),
    ]
    for name, a, b, color in rows:
        fig.add_trace(
            go.Bar(
                y=[name],
                base=[a],
                x=[(b - a).days],
                orientation="h",
                marker_color=color,
                name=name,
                hovertemplate=f"{name}<br>{a.date()} → {b.date()}<extra></extra>",
            )
        )
    if event.recovery_samples:
        xs = [
            e + timedelta(days=30 * smp["offset_months"])
            for smp in event.recovery_samples
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=["recovery"] * len(xs),
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#2ca02c"),
                name="recovery sample",
            )
        )
    fig.update_layout(
        title="Lifecycle timeline",
        xaxis_title="Date",
        height=230,
        showlegend=False,
        barmode="overlay",
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
        with st.expander("Glossary / How to read this"):
            st.markdown(glossary.README_INTRO)
            for group, entries in glossary.TERM_GROUPS.items():
                st.markdown(f"**{group}**")
                for term, definition in entries:
                    st.markdown(f"- **{term}**: {definition}")

    store = EventStore(root)
    events = [e for e in store.list_events() if e.status in status_filter]
    events.sort(key=lambda e: e.start_date, reverse=True)

    if not events:
        st.info(f"No events match the current filters in {root}.")
        return

    df = _events_frame(events)

    tab_overview, tab_progress, tab_during, tab_post, tab_recovery, tab_failures, tab_model = (
        st.tabs(
            [
                "Events",
                "Satellite progression",
                "During (FRP)",
                "Post-fire severity",
                "Recovery",
                "Failures",
                "Model",
            ]
        )
    )

    with tab_overview:
        _kpi_row(events)
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
                column_config=_events_column_config(),
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
        g = glossary.GLOSSARY
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Status", event.status, help=g.get(event.status, "Lifecycle stage."))
        c2.metric(
            "Burned area (ha)",
            _fmt(post.get("burned_area_ha")),
            help=g["burned_area_ha"],
        )
        c3.metric(
            "Peak FRP (MW)", _fmt(_event_row(event)["peak_frp_mw"]), help=g["FRP"]
        )
        c4.metric("Pre NDVI", _fmt(pre.get("ndvi")), help=g["NDVI"])
        c5.metric(
            "Weather index", _fmt(pre.get("weather_index")), help=g["weather_index"]
        )
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
        st.plotly_chart(_lifecycle_figure(event), width="stretch")
        dc1, dc2, dc3 = st.columns(3)
        dc1.download_button(
            "Download JSON",
            json.dumps(export_json(event), indent=2),
            file_name=f"{event.event_id}.json",
            mime="application/json",
        )
        dc2.download_button(
            "Download observations CSV",
            export_csv(event.during_observations),
            file_name=f"{event.event_id}_observations.csv",
            mime="text/csv",
        )
        dc3.download_button(
            "Download HTML report",
            render_html(event),
            file_name=f"{event.event_id}.html",
            mime="text/html",
        )

    with tab_progress:
        st.subheader("Before / During / After — Sentinel-2")
        st.caption(
            f"Imagery fetched on demand ({resolution} m) and cached under "
            "`reports/imagery/`. LEAST-CC mosaicking picks the clearest scene."
        )
        st.caption(
            "All panels are linked: zoom or pan on any image and the rest follow "
            "to the same geographic segment."
        )
        ndvi_mode = (
            "delta"
            if st.radio(
                "NDVI view",
                ["Actual NDVI", "ΔNDVI vs pre-fire"],
                horizontal=True,
                key="ndvi_mode",
            )
            == "ΔNDVI vs pre-fire"
            else "actual"
        )

        pre_bands = _load_bands(event, "before", None, resolution)

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
        dur_bands = _load_bands(event, "during", selected.isoformat(), resolution)
        post_bands = _load_bands(event, "after", None, resolution)

        if pre_bands is None or dur_bands is None or post_bands is None:
            missing = [
                label
                for label, bands in [
                    ("pre-fire", pre_bands),
                    ("during", dur_bands),
                    ("post-fire", post_bands),
                ]
                if bands is None
            ]
            st.warning(f"No Sentinel-2 imagery available for: {', '.join(missing)}.")
        else:
            fig = build_linked_figure(
                event,
                pre_bands,
                dur_bands,
                post_bands,
                selected.isoformat(),
                ndvi_mode=ndvi_mode,
                resolution=resolution,
            )
            st.plotly_chart(fig, width="stretch")
            _badge_if_low_valid(pre_bands, "Pre-fire scene")
            _badge_if_low_valid(dur_bands, "During scene")
            _badge_if_low_valid(post_bands, "Post-fire scene")
            with st.expander("Burn scar overlay (Prithvi V2-300M)"):
                run_model = st.button("Run burn-scar model", key="run_model_btn")
                st.caption(
                    "Loads the ~1.3GB checkpoint on first use (cached in "
                    "session). Needs the six post-fire bands."
                )
                probs_key = f"v2_probs_{event.event_id}_{resolution}"
                mask_key = f"v2_mask_{event.event_id}_{resolution}"
                if run_model or st.session_state.get(probs_key) is not None:
                    if st.session_state.get(probs_key) is None:
                        try:
                            from data_pipeline.model_inference_v2 import (
                                load_model,
                                predict,
                            )

                            model = st.session_state.get("v2_model")
                            if model is None:
                                model = load_model()
                                st.session_state["v2_model"] = model
                            probs, mask = predict(post_bands[:6], model=model)
                            st.session_state[probs_key] = probs
                            st.session_state[mask_key] = mask
                        except Exception as exc:
                            st.error(f"Model inference failed: {exc}")
                            st.session_state[probs_key] = None
                    probs = st.session_state.get(probs_key)
                    if probs is not None:
                        overlay_fig = render_overlay(post_bands[:6], probs)
                        st.pyplot(overlay_fig)
                        mask = st.session_state.get(mask_key)
                        st.metric(
                            "Burned fraction (model)",
                            f"{np.mean(mask == 1):.1%}",
                        )
            if ndvi_mode == "delta":
                st.caption(
                    "ΔNDVI is measured relative to the pre-fire reference image."
                )

            st.markdown("**Download static images**")
            c1, c2, c3 = st.columns(3)
            pre_fig = render_before(event, pre_bands)
            c1.download_button(
                "Pre-fire PNG",
                _fig_png_bytes(pre_fig),
                file_name=f"{event.event_id[:8]}_before.png",
                mime="image/png",
            )
            dur_fig = render_during(
                event,
                dur_bands,
                selected.isoformat(),
                ndvi_mode=ndvi_mode,
                pre_bands=pre_bands,
            )
            c2.download_button(
                "During PNG",
                _fig_png_bytes(dur_fig),
                file_name=f"{event.event_id[:8]}_during_{selected.isoformat()}.png",
                mime="image/png",
            )
            post_fig = render_after(
                event, pre_bands, post_bands, resolution, ndvi_mode=ndvi_mode
            )
            c3.download_button(
                "Post-fire PNG",
                _fig_png_bytes(post_fig),
                file_name=f"{event.event_id[:8]}_after.png",
                mime="image/png",
            )

    with tab_during:
        sel_frp = st.plotly_chart(
            _frp_figure(event),
            on_select="rerun",
            selection_mode="points",
            key="frp_chart",
            width="stretch",
        )
        try:
            points = sel_frp.selection.points
            if points:
                clicked = points[0].get("x")
                if clicked and st.session_state.get("_last_frp_click") != clicked:
                    st.session_state["_last_frp_click"] = clicked
                    try:
                        st.session_state["during_slider"] = _date.fromisoformat(clicked)
                        st.rerun()
                    except ValueError:
                        pass
        except Exception:
            pass
        st.caption(
            "Click a point on the chart to jump the during-date slider in the "
            "Satellite progression tab."
        )
        st.dataframe(
            pd.DataFrame(event.during_observations),
            column_config=_obs_column_config(),
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
                    {"class": list(severity.keys()), "share": list(severity.values())}
                )
                fig = go.Figure(
                    go.Bar(x=sv["class"], y=sv["share"], marker_color="#e31a1c")
                )
                fig.update_layout(
                    title="Burn severity class share", yaxis_title="Fraction"
                )
                st.plotly_chart(fig, width="stretch")
            areas = cached_severity_areas(event, resolution=resolution)
            if areas is not None:
                area_df = pd.DataFrame(
                    {"class": list(areas.keys()), "area_ha": list(areas.values())}
                )
                st.dataframe(
                    area_df,
                    column_config={
                        "area_ha": st.column_config.NumberColumn(
                            "Area (ha)", help=g["burned_area_ha"]
                        )
                    },
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Per-class burned area computed from cached satellite imagery."
                )
            else:
                st.caption(
                    "View the Satellite progression tab to compute per-class burned "
                    "area (hectares)."
                )
            st.write({k: v for k, v in post.items() if k != "severity_classes"})

    with tab_recovery:
        samples = event.recovery_samples
        if not samples:
            st.info("No recovery samples yet.")
        else:
            st.dataframe(pd.DataFrame(samples), hide_index=True, width="stretch")
            offsets = [s["offset_months"] for s in samples]
            ndvis = [s.get("ndvi") for s in samples]
            regrows = [s.get("regrowth_ratio") for s in samples]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=offsets, y=ndvis, mode="lines+markers", name="NDVI")
            )
            baseline = pre.get("ndvi")
            if baseline is not None:
                fig.add_hline(
                    y=baseline,
                    line_dash="dash",
                    line_color="#888",
                    annotation_text="pre-fire NDVI baseline",
                    annotation_position="top right",
                )
            fig.add_trace(
                go.Scatter(
                    x=offsets,
                    y=regrows,
                    mode="lines+markers",
                    name="regrowth ratio",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title="Recovery by month offset",
                xaxis_title="Months after fire",
                yaxis_title="NDVI",
                yaxis2=dict(overlaying="y", side="right", title="regrowth ratio"),
            )
            st.plotly_chart(fig, width="stretch")

    with tab_failures:
        if not event.failures:
            st.info("No recorded failures.")
        else:
            st.dataframe(pd.DataFrame(event.failures), hide_index=True, width="stretch")

    with tab_model:
        st.subheader("Burn scar model")
        st.markdown(model_card_markdown())
        eval_data = load_latest_eval()
        if eval_data is None:
            st.info(
                "No evaluation artifact found under reports/evaluation/. "
                "Run `PYTHONPATH=. uv run python scripts/run_evaluation.py`."
            )
        else:
            st.markdown(f"**Benchmark on HLS test split (n={eval_data.get('n')})**")
            v2 = eval_data.get("v2_300m", {})
            rows = [
                {"model": "V2-300M", **{k: v2.get(k) for k in ("iou", "dice", "precision", "recall")}},
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            chart_dir = latest_eval_dir()
            if chart_dir and (chart_dir / "benchmark.png").exists():
                st.image(str(chart_dir / "benchmark.png"))
            events_eval = eval_data.get("events")
            if events_eval:
                st.markdown("**dNBR cross-check on live events**")
                st.write(events_eval)


if __name__ == "__main__":
    main()
