import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from sources.config import EARTH_RADIUS_KM


def parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wrap_lon(delta_deg):
    return (np.asarray(delta_deg) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def latlon_to_unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def unit_to_latlon(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


def angular_distance_km(u, v):
    dot = np.clip(np.sum(np.asarray(u) * np.asarray(v), axis=-1), -1.0, 1.0)
    return EARTH_RADIUS_KM * np.arccos(dot)


class StageTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0
            self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self, n_events: int) -> dict:
        total = sum(self.totals.values()) or 1e-12
        return {
            name: {
                "total_sec": round(t, 4),
                "share_pct": round(100 * t / total, 1),
                "us_per_event": round(1e6 * t / max(n_events, 1), 2),
            }
            for name, t in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }


def percentile_summary(values_ms) -> dict:
    arr = np.asarray(values_ms, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
