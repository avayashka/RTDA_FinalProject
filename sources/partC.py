import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import SAMPLING_INTERVAL_SEC, TUMBLING_WINDOW_SEC
from sources.utils import haversine_km, parse_ts


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_metrics(events: List[Dict[str, Any]], start: float, end: float, kind: str) -> Dict[str, Any]:
    alts = np.array([e["altitude_km"] for e in events])
    vels = np.array([e["velocity_kmh"] for e in events])
    lats = np.array([e["latitude"] for e in events])
    lons = np.array([e["longitude"] for e in events])
    steps = haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:]) if len(events) > 1 else np.array([0.0])
    flagged = [int(e.get("is_anomaly", 0)) for e in events]
    ok = np.array([not (a or b) for a, b in zip(flagged[:-1], flagged[1:])]) if len(events) > 1 else np.array([True])
    t = np.array([e["_event_time"] for e in events])
    dts = np.diff(t) if len(events) > 1 else np.array([SAMPLING_INTERVAL_SEC])
    good_steps, good_dts = steps[ok], dts[ok] if len(events) > 1 else dts
    expected = (end - start) / SAMPLING_INTERVAL_SEC
    return {
        "window_type": kind,
        "window_start": _iso(start),
        "window_end": _iso(end),
        "event_count": len(events),
        "completeness_pct": round(100 * len(events) / expected, 1),
        "avg_altitude_km": round(float(alts.mean()), 3),
        "min_altitude_km": round(float(alts.min()), 3),
        "max_altitude_km": round(float(alts.max()), 3),
        "std_altitude_km": round(float(alts.std()), 4),
        "avg_velocity_kmh": round(float(vels.mean()), 3),
        "min_velocity_kmh": round(float(vels.min()), 3),
        "max_velocity_kmh": round(float(vels.max()), 3),
        "altitude_change_km_per_min": round(float((alts[-1] - alts[0]) / max(t[-1] - t[0], 1) * 60), 4),
        "distance_km": round(float(good_steps.sum()), 2),
        "avg_ground_speed_kmh": round(float(good_steps.sum() / good_dts.sum() * 3600), 1) if good_dts.sum() > 0 else None,
        "anomaly_count": int(sum(flagged)),
        "dominant_hemisphere": Counter(e.get("hemisphere") for e in events).most_common(1)[0][0],
        "dominant_region": Counter(e.get("region") for e in events).most_common(1)[0][0],
    }


class TumblingWindowAggregator:
    def __init__(self, window_sec: int = TUMBLING_WINDOW_SEC):
        self.window_sec = window_sec
        self._current_start: Optional[float] = None
        self._events: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        t = event.get("_event_time") or parse_ts(event["timestamp"]).timestamp()
        event["_event_time"] = t
        start = (t // self.window_sec) * self.window_sec
        closed = []
        if self._current_start is not None and start != self._current_start:
            closed.append(self._close())
        if self._current_start != start:
            self._current_start, self._events = start, []
        self._events.append(event)
        return closed

    def _close(self) -> Dict[str, Any]:
        return _window_metrics(self._events, self._current_start, self._current_start + self.window_sec, "tumbling_60s")

    def flush(self) -> List[Dict[str, Any]]:
        if not self._events:
            return []
        out = [self._close()]
        self._events, self._current_start = [], None
        return out


class SlidingWindowAggregator:
    def __init__(self, size_sec: int = 300, slide_sec: int = 60):
        self.size_sec, self.slide_sec = size_sec, slide_sec
        self._buf: deque = deque()
        self._next_emit: Optional[float] = None

    def add(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        t = event["_event_time"]
        if self._next_emit is None:
            self._next_emit = (t // self.slide_sec + 1) * self.slide_sec + (self.size_sec - self.slide_sec)
        out = []
        while t >= self._next_emit:
            end = self._next_emit
            in_win = [e for e in self._buf if end - self.size_sec <= e["_event_time"] < end]
            if in_win:
                out.append(_window_metrics(in_win, end - self.size_sec, end, "sliding_5m"))
            self._next_emit += self.slide_sec
        self._buf.append(event)
        while self._buf and self._buf[0]["_event_time"] < t - self.size_sec:
            self._buf.popleft()
        return out


class ISSTumblingWindowProcessor:
    def __init__(self, window_size_sec: int = TUMBLING_WINDOW_SEC):
        self._agg = TumblingWindowAggregator(window_size_sec)
        self._results: List[Dict[str, Any]] = []

    def add_event(self, event: Dict[str, Any]):
        e = dict(event)
        e.setdefault("velocity_kmh", e.get("speed_kmph"))
        e.setdefault("hemisphere", e.get("Hemisphere"))
        self._results.extend(self._agg.add(e))

    def get_window_metrics(self) -> List[Dict[str, Any]]:
        return self._results + self._agg.flush()


if __name__ == "__main__":
    import pandas as pd
    from sources.partB import FeatureBuilder, StreamCleaner
    from sources.stream_generator import ISSStreamGenerator

    cleaner, fb = StreamCleaner(), FeatureBuilder()
    tumbling, sliding = TumblingWindowAggregator(), SlidingWindowAggregator()
    t_out, s_out = [], []
    for raw in ISSStreamGenerator(max_events=720).generate_events():
        for ev in cleaner.process(raw):
            ev = fb.transform(ev)
            ev["is_anomaly"] = ev["state_gate_flag"]
            t_out += tumbling.add(ev)
            s_out += sliding.add(ev)
    t_out += tumbling.flush()
    pd.set_option("display.width", 200)
    cols = ["window_start", "event_count", "avg_altitude_km", "avg_velocity_kmh",
            "altitude_change_km_per_min", "distance_km", "avg_ground_speed_kmh", "anomaly_count", "dominant_region"]
    print(f"{len(t_out)} tumbling windows, {len(s_out)} sliding windows")
    print(pd.DataFrame(t_out)[cols].head(8).to_string(index=False))
    print(pd.DataFrame(s_out)[cols].head(5).to_string(index=False))
