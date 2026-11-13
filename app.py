import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sources import dashboard_data as dd
from sources.config import DB_PATH, FIGURES_DIR

st.set_page_config(page_title="ISS Real-Time Analytics", page_icon="🛰️", layout="wide")

st.sidebar.title("🛰️ ISS Analytics")
auto = st.sidebar.toggle("Auto-refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh every (s)", 1, 30, 2)
n_recent = st.sidebar.slider("Recent events shown", 60, 3000, 720, step=60,
                             help="720 events = 2 hours of telemetry at 10-s cadence")
st.sidebar.caption(f"Database: `{Path(DB_PATH).relative_to(ROOT)}`")
st.sidebar.markdown("**Pipeline:** `python -m sources.stream_processor --speedup 20`")

COLORS = {"normal": "#3366aa", "anomaly": "#d62728", "forecast": "#ff7f0e", "trend": "#2ca02c"}


def fmt(v, nd=2, unit=""):
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:,.{nd}f}{unit}"


def live_view():
    conn = dd.connect()
    if conn is None:
        st.warning("No database yet. Run `python -m sources.stream_processor` (or `python run_all.py`) first.")
        return
    k = dd.kpis(conn)
    if not k["events"]:
        st.info("Database is empty - start the pipeline.")
        return
    latest = k["latest"]

    st.title("ISS Real-Time Data Analytics Platform")
    st.caption(f"Last event time: **{k['last_ts']}** · refreshed {time.strftime('%H:%M:%S')}")
    c = st.columns(7)
    c[0].metric("Altitude", fmt(latest.get("altitude_km"), 2, " km"),
                fmt(latest.get("alt_trend_km_per_min"), 3, " km/min"))
    c[1].metric("Speed", fmt(latest.get("velocity_kmh"), 1, " km/h"))
    c[2].metric("Position", f"{latest.get('latitude', 0):.2f}°, {latest.get('longitude', 0):.2f}°")
    c[3].metric("Region", str(latest.get("region", "–")))
    c[4].metric("Events processed", f"{k['events']:,}", f"{k['raw']:,} received")
    c[5].metric("Anomalies", f"{int(k['anomalies']):,}", f"{100 * k['anomaly_rate']:.1f} % of events", delta_color="off")
    c[6].metric("Forecast error (60 s)", fmt(k["forecast_mae_km"], 3, " km"), f"{k['forecast_n']:,} scored", delta_color="off")

    tel = dd.recent_telemetry(conn, n_recent)
    tabs = st.tabs(["🌍 Live track", "⏱️ Windows", "🚨 Anomalies", "🔮 Forecasts", "🧹 Data quality"])

    with tabs[0]:
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            norm, anom = tel[tel.is_anomaly == 0], tel[tel.is_anomaly == 1]
            fig.add_trace(go.Scattergeo(lat=norm.latitude, lon=norm.longitude, mode="markers",
                                        marker=dict(size=3, color=norm.altitude_km, colorscale="Viridis",
                                                    colorbar=dict(title="alt km")), name="ground track"))
            fig.add_trace(go.Scattergeo(lat=anom.latitude, lon=anom.longitude, mode="markers",
                                        marker=dict(size=8, color=COLORS["anomaly"], symbol="x"), name="anomaly"))
            fig.add_trace(go.Scattergeo(lat=[latest["latitude"]], lon=[latest["longitude"]], mode="markers",
                                        marker=dict(size=16, color="black", symbol="star"), name="ISS now"))
            fc = dd.latest_forecast(conn)
            if not fc.empty:
                fig.add_trace(go.Scattergeo(lat=fc.pred_latitude, lon=fc.pred_longitude, mode="markers",
                                            marker=dict(size=12, color=COLORS["forecast"], symbol="diamond"),
                                            name="forecast +60 s"))
            fig.update_geos(projection_type="natural earth", showcountries=True, showland=True, landcolor="#eeeeee")
            fig.update_layout(height=460, margin=dict(l=0, r=0, t=30, b=0), title="Ground track (recent events)",
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tel.timestamp, y=tel.altitude_km, mode="lines", name="altitude",
                                     line=dict(color=COLORS["normal"], width=1)))
            fig.add_trace(go.Scatter(x=tel.timestamp, y=tel.roll_alt_mean_km, mode="lines", name="5-min rolling mean",
                                     line=dict(color=COLORS["trend"], width=2)))
            a = tel[tel.is_anomaly == 1]
            fig.add_trace(go.Scatter(x=a.timestamp, y=a.altitude_km, mode="markers", name="anomaly",
                                     marker=dict(color=COLORS["anomaly"], size=7, symbol="x")))
            fig.update_layout(height=225, margin=dict(l=0, r=0, t=30, b=0), title="Altitude (km)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tel.timestamp, y=tel.velocity_kmh, mode="lines",
                                     line=dict(color="#9467bd", width=1)))
            fig.add_trace(go.Scatter(x=a.timestamp, y=a.velocity_kmh, mode="markers",
                                     marker=dict(color=COLORS["anomaly"], size=7, symbol="x")))
            fig.update_layout(height=225, margin=dict(l=0, r=0, t=30, b=0), title="Speed (km/h)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        wt = dd.windows(conn, "tumbling_60s", 360)
        ws = dd.windows(conn, "sliding_5m", 360)
        if wt.empty:
            st.info("No closed windows yet.")
        else:
            c1, c2 = st.columns(2)
            fig = px.line(wt, x="window_start", y=["avg_altitude_km", "min_altitude_km", "max_altitude_km"],
                          title="60-s tumbling windows: altitude (avg / min / max)")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c1.plotly_chart(fig, use_container_width=True)
            fig = px.line(ws, x="window_start", y="altitude_change_km_per_min",
                          title="5-min sliding window: altitude change rate (km/min)")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c2.plotly_chart(fig, use_container_width=True)
            c3, c4 = st.columns(2)
            fig = px.bar(wt, x="window_start", y="anomaly_count", title="Anomalies per 60-s window")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            c3.plotly_chart(fig, use_container_width=True)
            fig = px.line(wt, x="window_start", y="avg_ground_speed_kmh", title="Ground speed per window (km/h)")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            c4.plotly_chart(fig, use_container_width=True)
            st.dataframe(wt.sort_values("window_start", ascending=False).head(30), use_container_width=True, hide_index=True)

    with tabs[2]:
        an = dd.anomalies(conn, 1000)
        if an.empty:
            st.success("No anomalies flagged.")
        else:
            c1, c2 = st.columns([1, 2])
            by = an.groupby("reason").size().reset_index(name="count")
            fig = px.pie(by, names="reason", values="count", title="Alerts by dominant signal", hole=0.4)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c1.plotly_chart(fig, use_container_width=True)
            fig = px.scatter(tel, x="timestamp", y="rule_score", color=tel.is_anomaly.map({0: "normal", 1: "anomaly"}),
                             log_y=True, title="Anomaly score (robust z, log scale)",
                             color_discrete_map={"normal": COLORS["normal"], "anomaly": COLORS["anomaly"]})
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), legend_title="")
            c2.plotly_chart(fig, use_container_width=True)
            if tel["truth_anomaly"].notna().any():
                t = tel.dropna(subset=["truth_anomaly"])
                tp = int(((t.is_anomaly == 1) & (t.truth_anomaly == 1)).sum())
                fp = int(((t.is_anomaly == 1) & (t.truth_anomaly == 0)).sum())
                fn = int(((t.is_anomaly == 0) & (t.truth_anomaly == 1)).sum())
                p = tp / (tp + fp) if tp + fp else 0
                r = tp / (tp + fn) if tp + fn else 0
                st.caption(f"Live evaluation on the events shown (simulation ground truth): precision {p:.3f} · "
                           f"recall {r:.3f} · TP {tp} · FP {fp} · FN {fn}")
            st.subheader("Alert log")
            st.dataframe(an[["timestamp", "event_id", "reason", "rule_score", "iforest_score", "latitude", "longitude",
                             "altitude_km", "velocity_kmh", "pos_dev_km", "alt_dev_km", "vel_dev_kmh", "truth_type"]],
                         use_container_width=True, hide_index=True)

    with tabs[3]:
        pr = dd.predictions(conn, 2000)
        if pr.empty:
            st.info("No scored forecasts yet (needs 6 consecutive accepted readings + 60 s).")
        else:
            c1, c2 = st.columns(2)
            fig = px.line(pr, x="target_time", y="pos_error_km", title="60-s-ahead position error (km), prequential")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c1.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(pr, x="pos_error_km", nbins=60, title="Error distribution")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c2.plotly_chart(fig, use_container_width=True)
            st.caption(f"MAE {pr.pos_error_km.mean():.3f} km · P95 {pr.pos_error_km.quantile(.95):.3f} km · "
                       f"altitude MAE {pr.alt_error_km.mean():.4f} km over {len(pr):,} forecasts")

    with tabs[4]:
        rj = dd.rejected_summary(conn)
        c1, c2 = st.columns(2)
        if rj.empty:
            c1.success("No rejected events.")
        else:
            fig = px.bar(rj, x="reason", y="events", title="Rejected events by reason")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            c1.plotly_chart(fig, use_container_width=True)
        runs = dd.pipeline_runs(conn)
        c2.subheader("Pipeline runs")
        c2.dataframe(runs, use_container_width=True, hide_index=True)
    conn.close()


if auto and hasattr(st, "fragment"):
    st.fragment(run_every=refresh_sec)(live_view)()
else:
    live_view()

with st.expander("📊 Offline evaluation & performance results", expanded=False):
    res = dd.offline_results()
    if not res:
        st.info("Run `python run_all.py` to produce the evaluation results.")
    for key, title in (("anomaly_methods", "Anomaly detection (test split)"),
                       ("anomaly_noise_sensitivity", "Noise sensitivity"),
                       ("forecast_evaluation", "Forecasting (test split)"),
                       ("benchmark_batch_sweep", "Throughput / latency vs micro-batch size"),
                       ("benchmark_rate_controlled", "Rate-controlled load"),
                       ("distributed_scaling", "Distributed scaling")):
        if key in res:
            st.markdown(f"**{title}**")
            st.dataframe(res[key], use_container_width=True, hide_index=True)
    figs = sorted(FIGURES_DIR.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem, use_container_width=True)

if auto and not hasattr(st, "fragment"):
    time.sleep(refresh_sec)
    st.rerun()