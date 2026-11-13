import heapq
import json
import math
import sys
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import (DEDUP_CACHE_SIZE, SLIDING_WINDOW_EVENTS, STATE_GATE,
                            WATERMARK_SEC)
from sources.partA import ISSTelemetryEvent, SchemaValidationError
from sources.utils import EARTH_RADIUS_KM, haversine_km, latlon_to_unit, parse_ts

FEATURE_COLUMNS = ["pos_dev_km", "alt_dev_km", "vel_dev_kmh", "ground_speed_dev_kmh",
                   "alt_roll_z", "vel_roll_z"]


class StreamCleaner:
    def __init__(self, watermark_sec: int = WATERMARK_SEC, dedup_cache_size: int = DEDUP_CACHE_SIZE):
        self.watermark_sec = watermark_sec
        self._seen = OrderedDict()
        self._dedup_cache_size = dedup_cache_size
        self._buffer: List[Tuple[float, int, Dict[str, Any]]] = []
        self._max_event_time: Optional[float] = None
        self._last_released_time: float = -math.inf
        self.stats = {"received": 0, "valid": 0, "released": 0, "duplicates": 0,
                      "late_dropped": 0, "reordered": 0, "rejected": {}}
        self.rejected: List[Dict[str, Any]] = []

    def _reject(self, raw: Dict[str, Any], reason: str, detail: str):
        self.stats["rejected"][reason] = self.stats["rejected"].get(reason, 0) + 1
        payload = {k: v for k, v in raw.items() if not k.startswith("_")}
        self.rejected.append({"event_id": str(raw.get("event_id", "")), "reason": reason, "detail": detail,
                              "payload": json.dumps(payload, default=str)})

    def process(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.stats["received"] += 1
        try:
            event = ISSTelemetryEvent.validate_and_parse(raw).to_dict()
        except SchemaValidationError as exc:
            self._reject(raw, exc.reason, str(exc))
            return []

        if event["event_id"] in self._seen:
            self.stats["duplicates"] += 1
            self._reject(raw, "duplicate", "event_id already processed")
            return []
        self._seen[event["event_id"]] = None
        if len(self._seen) > self._dedup_cache_size:
            self._seen.popitem(last=False)

        t = parse_ts(event["timestamp"]).timestamp()
        if self._max_event_time is not None and t < self._max_event_time:
            self.stats["reordered"] += 1
        watermark = (self._max_event_time or t) - self.watermark_sec
        if t < watermark or t <= self._last_released_time:
            self.stats["late_dropped"] += 1
            self._reject(raw, "late_event", f"event time behind watermark by {watermark - t:.0f}s")
            return []

        self.stats["valid"] += 1
        event["_event_time"] = t
        if "_truth" in raw:
            event["_truth"] = raw["_truth"]
        if "_ingest_perf" in raw:
            event["_ingest_perf"] = raw["_ingest_perf"]
        heapq.heappush(self._buffer, (t, event["seq"], event))
        self._max_event_time = t if self._max_event_time is None else max(self._max_event_time, t)
        return self._release(self._max_event_time - self.watermark_sec)

    def _release(self, up_to: float) -> List[Dict[str, Any]]:
        out = []
        while self._buffer and self._buffer[0][0] <= up_to:
            t, _, ev = heapq.heappop(self._buffer)
            self._last_released_time = t
            out.append(ev)
        self.stats["released"] += len(out)
        return out

    def flush(self) -> List[Dict[str, Any]]:
        return self._release(math.inf)

    def drain_rejected(self) -> List[Dict[str, Any]]:
        out, self.rejected = self.rejected, []
        return out


def _rotate_predict(a2: np.ndarray, a1: np.ndarray, dt12: float, dt: float) -> np.ndarray:
    axis = np.cross(a2, a1)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or dt12 <= 0:
        return a1
    axis /= norm
    angle = math.atan2(norm, float(np.dot(a2, a1))) / dt12 * dt
    c, s = math.cos(angle), math.sin(angle)
    return a1 * c + np.cross(axis, a1) * s + axis * np.dot(axis, a1) * (1 - c)


def _arc_km(u: np.ndarray, v: np.ndarray) -> float:
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, float(np.dot(u, v)))))


class FeatureBuilder:
    def __init__(self, window_events: int = SLIDING_WINDOW_EVENTS, gate: Dict[str, float] = STATE_GATE):
        self.window_events = window_events
        self.gate = gate
        self._accepted: deque = deque(maxlen=2)
        self._raw_prev: Optional[Dict[str, Any]] = None
        self._raw_prev_accepted = True
        self._roll: deque = deque(maxlen=window_events)
        self._ewma_alt: Optional[float] = None
        self.segment_id = 0

    @staticmethod
    def _state(ev):
        return {"t": ev["_event_time"], "u": latlon_to_unit(ev["latitude"], ev["longitude"]),
                "lat": ev["latitude"], "lon": ev["longitude"],
                "alt": ev["altitude_km"], "vel": ev["velocity_kmh"]}

    def _deviation_from_accepted(self, cur):
        a1 = self._accepted[-1]
        dt = cur["t"] - a1["t"]
        if len(self._accepted) == 2:
            a2 = self._accepted[0]
            dt12 = a1["t"] - a2["t"]
            pred = _rotate_predict(a2["u"], a1["u"], dt12, dt)
            pos = _arc_km(pred, cur["u"])
            alt_rate = (a1["alt"] - a2["alt"]) / dt12
            vel_rate = (a1["vel"] - a2["vel"]) / dt12
        else:
            pos = abs(_arc_km(a1["u"], cur["u"]) - 6.97 * dt)
            alt_rate = vel_rate = 0.0
        alt = abs(cur["alt"] - (a1["alt"] + alt_rate * dt))
        vel = abs(cur["vel"] - (a1["vel"] + vel_rate * dt))
        return pos, alt, vel

    def _deviation_from_raw_prev(self, cur):
        p = self._raw_prev
        dt = cur["t"] - p["t"]
        pos = abs(_arc_km(p["u"], cur["u"]) - 6.97 * dt)
        return pos, abs(cur["alt"] - p["alt"]), abs(cur["vel"] - p["vel"])

    def _gated(self, pos, alt, vel) -> bool:
        return pos > self.gate["pos_dev_km"] or alt > self.gate["alt_dev_km"] or vel > self.gate["vel_dev_kmh"]

    def transform(self, event: Dict[str, Any]) -> Dict[str, Any]:
        cur = self._state(event)
        out = dict(event)
        reanchor = False
        if not self._accepted:
            pos = alt = vel = 0.0
        else:
            pos, alt, vel = self._deviation_from_accepted(cur)
            if not self._raw_prev_accepted and self._raw_prev is not None:
                rp = self._deviation_from_raw_prev(cur)
                if not self._gated(*rp) and self._gated(pos, alt, vel):
                    pos, alt, vel = rp
                    reanchor = True
        gated = self._gated(pos, alt, vel)

        if self._raw_prev is not None and cur["t"] > self._raw_prev["t"]:
            step = float(haversine_km(self._raw_prev["lat"], self._raw_prev["lon"], cur["lat"], cur["lon"]))
            ground_speed = step / (cur["t"] - self._raw_prev["t"]) * 3600
        else:
            ground_speed = 6.97 * 3600
        roll_alts = [r["alt"] for r in self._roll]
        roll_vels = [r["vel"] for r in self._roll]
        if len(roll_alts) >= 5:
            am, asd = float(np.mean(roll_alts)), float(np.std(roll_alts)) + 0.05
            vm, vsd = float(np.mean(roll_vels)), float(np.std(roll_vels)) + 0.1
            span = self._roll[-1]["t"] - self._roll[0]["t"]
            trend = (self._roll[-1]["alt"] - self._roll[0]["alt"]) / span * 60 if span > 0 else 0.0
        else:
            am, asd, vm, vsd, trend = cur["alt"], 0.05, cur["vel"], 0.1, 0.0

        out.update({
            "pos_dev_km": round(pos, 4),
            "alt_dev_km": round(alt, 4),
            "vel_dev_kmh": round(vel, 4),
            "ground_speed_kmh": round(ground_speed, 2),
            "ground_speed_dev_kmh": round(abs(ground_speed - 6.97 * 3600), 2),
            "roll_alt_mean_km": round(am, 3),
            "roll_alt_std_km": round(asd, 4),
            "roll_vel_mean_kmh": round(vm, 3),
            "alt_roll_z": round((cur["alt"] - am) / asd, 3),
            "vel_roll_z": round((cur["vel"] - vm) / vsd, 3),
            "alt_trend_km_per_min": round(trend, 4),
            "ewma_alt_km": round(cur["alt"] if self._ewma_alt is None else 0.2 * cur["alt"] + 0.8 * self._ewma_alt, 3),
            "state_gate_flag": int(gated),
            "segment_id": self.segment_id + int(reanchor),
        })

        if not gated:
            if reanchor:
                self.segment_id += 1
                self._accepted.clear()
                self._accepted.append(self._raw_prev)
                self._roll.clear()
            self._accepted.append(cur)
            self._roll.append(cur)
            self._ewma_alt = out["ewma_alt_km"]
        self._raw_prev = cur
        self._raw_prev_accepted = not gated
        return out


def process_iss_stream_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaner = StreamCleaner()
    clean = []
    for ev in events:
        clean.extend(cleaner.process(ev))
    clean.extend(cleaner.flush())
    if not clean:
        return {"total_processed_events": 0, "status": "No valid events found", "cleaning": cleaner.stats}
    alts = np.array([e["altitude_km"] for e in clean])
    vels = np.array([e["velocity_kmh"] for e in clean])
    return {
        "total_processed_events": len(clean),
        "cleaning": cleaner.stats,
        "altitude_metrics": {"min_km": round(alts.min(), 2), "max_km": round(alts.max(), 2),
                             "avg_km": round(alts.mean(), 2), "std_km": round(alts.std(), 2)},
        "velocity_metrics": {"min_kmh": round(vels.min(), 2), "max_kmh": round(vels.max(), 2),
                             "avg_kmh": round(vels.mean(), 2), "std_kmh": round(vels.std(), 2)},
    }


if __name__ == "__main__":
    from sources.stream_generator import ISSStreamGenerator
    gen = ISSStreamGenerator(inject_faults=True)
    print(json.dumps(process_iss_stream_metrics(list(gen.generate_events())), indent=2))
