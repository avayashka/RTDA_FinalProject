import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from sources.config import DB_PATH, RESULTS_DIR


def connect(db_path=DB_PATH) -> Optional[sqlite3.Connection]:
    p = Path(db_path)
    if not p.exists():
        return None
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=10)


def _q(conn, sql, params=()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def kpis(conn) -> Dict:
    k = _q(conn, "SELECT COUNT(*) AS events, COALESCE(SUM(is_anomaly),0) AS anomalies, MAX(timestamp) AS last_ts "
                 "FROM telemetry").iloc[0].to_dict()
    k["rejected"] = int(_q(conn, "SELECT COUNT(*) AS n FROM rejected_events").iloc[0]["n"])
    k["raw"] = int(_q(conn, "SELECT COUNT(*) AS n FROM raw_events").iloc[0]["n"])
    latest = _q(conn, "SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT 1")
    k["latest"] = latest.iloc[0].to_dict() if len(latest) else {}
    f = _q(conn, "SELECT AVG(pos_error_km) AS mae, COUNT(*) AS n FROM predictions WHERE status='scored'")
    k["forecast_mae_km"] = None if f.empty or pd.isna(f.iloc[0]["mae"]) else float(f.iloc[0]["mae"])
    k["forecast_n"] = 0 if f.empty else int(f.iloc[0]["n"])
    k["anomaly_rate"] = k["anomalies"] / k["events"] if k["events"] else 0.0
    return k


def recent_telemetry(conn, n=720) -> pd.DataFrame:
    df = _q(conn, "SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT ?", (int(n),))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
    return df


def windows(conn, window_type="tumbling_60s", n=240) -> pd.DataFrame:
    df = _q(conn, "SELECT * FROM window_metrics WHERE window_type=? ORDER BY window_start DESC LIMIT ?",
            (window_type, int(n)))
    if not df.empty:
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
        df = df.sort_values("window_start")
    return df


def anomalies(conn, n=500) -> pd.DataFrame:
    return _q(conn, "SELECT * FROM anomalies ORDER BY timestamp DESC LIMIT ?", (int(n),))


def predictions(conn, n=1000) -> pd.DataFrame:
    df = _q(conn, "SELECT * FROM predictions WHERE status='scored' ORDER BY target_time DESC LIMIT ?", (int(n),))
    if not df.empty:
        df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
        df = df.sort_values("target_time")
    return df


def latest_forecast(conn) -> pd.DataFrame:
    return _q(conn, "SELECT * FROM predictions ORDER BY issued_at DESC LIMIT 1")


def rejected_summary(conn) -> pd.DataFrame:
    return _q(conn, "SELECT reason, COUNT(*) AS events FROM rejected_events GROUP BY reason ORDER BY events DESC")


def pipeline_runs(conn) -> pd.DataFrame:
    return _q(conn, "SELECT run_id, started_at, finished_at, events_in, events_out, throughput_eps, config "
                    "FROM pipeline_runs ORDER BY run_id DESC LIMIT 10")


def offline_results() -> Dict:
    out = {}
    for name in ("anomaly_methods.csv", "forecast_evaluation.csv", "benchmark_batch_sweep.csv",
                 "benchmark_rate_controlled.csv", "distributed_scaling.csv", "anomaly_noise_sensitivity.csv"):
        p = RESULTS_DIR / name
        if p.exists():
            out[name.replace(".csv", "")] = pd.read_csv(p)
    for name in ("time_series_summary.json", "stream_run_summary.json", "anomaly_evaluation.json"):
        p = RESULTS_DIR / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text())
    return out
