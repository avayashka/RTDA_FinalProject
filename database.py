import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import DB_PATH

SCHEMA = {
    "raw_events": """
        CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT, received_at REAL, payload TEXT)""",
    "telemetry": """
        CREATE TABLE IF NOT EXISTS telemetry (
            event_id TEXT PRIMARY KEY, seq INTEGER, satellite_id TEXT, timestamp TEXT,
            latitude REAL, longitude REAL, altitude_km REAL, velocity_kmh REAL,
            hemisphere TEXT, region TEXT, segment_id INTEGER,
            pos_dev_km REAL, alt_dev_km REAL, vel_dev_kmh REAL, ground_speed_kmh REAL,
            roll_alt_mean_km REAL, roll_vel_mean_kmh REAL, alt_trend_km_per_min REAL, ewma_alt_km REAL,
            rule_score REAL, iforest_score REAL, is_anomaly INTEGER, anomaly_reason TEXT,
            truth_anomaly INTEGER, truth_type TEXT, processed_at REAL, latency_ms REAL)""",
    "window_metrics": """
        CREATE TABLE IF NOT EXISTS window_metrics (
            window_type TEXT, window_start TEXT, window_end TEXT, event_count INTEGER,
            completeness_pct REAL, avg_altitude_km REAL, min_altitude_km REAL, max_altitude_km REAL,
            std_altitude_km REAL, avg_velocity_kmh REAL, min_velocity_kmh REAL, max_velocity_kmh REAL,
            altitude_change_km_per_min REAL, distance_km REAL, avg_ground_speed_kmh REAL,
            anomaly_count INTEGER, dominant_hemisphere TEXT, dominant_region TEXT,
            PRIMARY KEY (window_type, window_start))""",
    "anomalies": """
        CREATE TABLE IF NOT EXISTS anomalies (
            event_id TEXT PRIMARY KEY, timestamp TEXT, reason TEXT, rule_score REAL, iforest_score REAL,
            latitude REAL, longitude REAL, altitude_km REAL, velocity_kmh REAL,
            pos_dev_km REAL, alt_dev_km REAL, vel_dev_kmh REAL, truth_type TEXT)""",
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            event_id TEXT, issued_at TEXT, target_time TEXT, horizon_sec INTEGER,
            pred_latitude REAL, pred_longitude REAL, pred_altitude_km REAL, pred_velocity_kmh REAL,
            actual_latitude REAL, actual_longitude REAL, pos_error_km REAL, alt_error_km REAL,
            vel_error_kmh REAL, status TEXT)""",
    "rejected_events": """
        CREATE TABLE IF NOT EXISTS rejected_events (
            event_id TEXT, reason TEXT, detail TEXT, payload TEXT, rejected_at REAL)""",
    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, finished_at TEXT,
            events_in INTEGER, events_out INTEGER, throughput_eps REAL, config TEXT, summary TEXT)""",
}
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tel_ts ON telemetry(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_tel_anom ON telemetry(is_anomaly)",
    "CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status)",
]


def _columns(table: str) -> List[str]:
    body = SCHEMA[table].split("(", 1)[1]
    cols = []
    for part in body.replace("\n", " ").split(","):
        tok = part.strip().split()
        if tok and tok[0].isidentifier() and tok[0].upper() not in ("PRIMARY", "INTEGER", "TEXT", "REAL"):
            cols.append(tok[0])
    return [c for c in cols if c not in ("id", "run_id")]


COLUMNS = {t: _columns(t) for t in SCHEMA}


class ISSDatabaseSink:
    def __init__(self, db_path=DB_PATH, reset: bool = False):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        if reset:
            self.reset()
        self._init_db()

    def _init_db(self):
        for ddl in SCHEMA.values():
            self.conn.execute(ddl)
        for ddl in INDEXES:
            self.conn.execute(ddl)
        self.conn.commit()

    def reset(self):
        for t in SCHEMA:
            self.conn.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
        self._init_db()

    def insert_rows(self, table: str, rows: Iterable[Dict[str, Any]], replace: bool = True, commit: bool = True):
        rows = list(rows)
        if not rows:
            return 0
        cols = COLUMNS[table]
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        self.conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
        if commit:
            self.conn.commit()
        return len(rows)

    def write_batch(self, raw=(), telemetry=(), windows=(), anomalies=(), predictions=(), rejected=()):
        with self.conn:
            self.insert_rows("raw_events", raw, replace=False, commit=False)
            self.insert_rows("telemetry", telemetry, commit=False)
            self.insert_rows("window_metrics", windows, commit=False)
            self.insert_rows("anomalies", anomalies, commit=False)
            self.insert_rows("predictions", predictions, replace=False, commit=False)
            self.insert_rows("rejected_events", rejected, replace=False, commit=False)

    def log_run(self, started_at: str, events_in: int, events_out: int, throughput: float,
                config: Dict[str, Any], summary: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                "INSERT INTO pipeline_runs (started_at, finished_at, events_in, events_out, throughput_eps, config, summary)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (started_at, time.strftime("%Y-%m-%dT%H:%M:%S"), events_in, events_out, throughput,
                 json.dumps(config, default=str), json.dumps(summary, default=str)))

    def save_events(self, df_events: pd.DataFrame):
        self.insert_rows("telemetry", df_events.to_dict(orient="records"))

    def query(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def fetch_recent_records(self, limit: int = 100) -> pd.DataFrame:
        return self.query("SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT ?", (int(limit),))

    def table_counts(self) -> Dict[str, int]:
        return {t: int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in SCHEMA}

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = ISSDatabaseSink()
    print(f"Database: {db.db_path}")
    print(json.dumps(db.table_counts(), indent=2))
    if db.table_counts()["telemetry"]:
        print(db.fetch_recent_records(5)[["event_id", "timestamp", "altitude_km", "velocity_kmh", "is_anomaly"]])