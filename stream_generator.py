import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import FAULT_RATES, RANDOM_SEED, RAW_DATA_PATH, SATELLITE_ID
from sources.utils import haversine_km

ANOMALY_TYPES = ("altitude_spike", "velocity_spike", "position_glitch", "stuck_sensor")
DQ_TYPES = ("duplicate", "missing_field", "invalid_value", "out_of_order", "late_event")
NOISE_SIGMA = {"pos_km": 0.1, "alt_km": 0.02, "vel_kmh": 0.05}
LATE_EVENT_DELAY = 12


def label_natural_discontinuities(df: pd.DataFrame) -> np.ndarray:
    step = haversine_km(df["latitude"].shift(), df["longitude"].shift(), df["latitude"], df["longitude"])
    d_alt = df["altitude_km"].diff().abs()
    d_vel = df["speed_kmph"].diff().abs()
    lab = (step < 65) | (step > 75) | (d_alt > 0.5) | (d_vel > 1.0)
    lab.iloc[0] = False
    return lab.to_numpy()


class ISSStreamGenerator:
    def __init__(self, csv_filepath=RAW_DATA_PATH, stream_delay_sec: float = 0.0,
                 speedup: Optional[float] = None, inject_faults: bool = False,
                 seed: int = RANDOM_SEED, satellite_id: str = SATELLITE_ID,
                 max_events: Optional[int] = None, noise_level: float = 0.0):
        self.csv_filepath = Path(csv_filepath)
        self.stream_delay_sec = stream_delay_sec
        self.speedup = speedup
        self.inject_faults = inject_faults
        self.seed = seed
        self.satellite_id = satellite_id
        self.noise_level = noise_level
        self.df = pd.read_csv(self.csv_filepath)
        if max_events:
            self.df = self.df.head(max_events).reset_index(drop=True)
        self.natural = label_natural_discontinuities(self.df)
        self._events = self._build_events()

    def _base_events(self) -> List[Dict[str, Any]]:
        events = []
        for i, row in enumerate(self.df.itertuples(index=False)):
            seq = i + 1
            events.append({
                "event_id": f"{self.satellite_id}-{seq:06d}",
                "seq": seq,
                "satellite_id": self.satellite_id,
                "timestamp": row.timestamp,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "altitude_km": float(row.altitude_km),
                "speed_kmph": float(row.speed_kmph),
                "Hemisphere": row.Hemisphere,
                "region": row.region,
                "_truth": {
                    "anomaly": int(self.natural[i]),
                    "type": "trajectory_discontinuity" if self.natural[i] else "normal",
                    "dq": None,
                },
            })
        if self.noise_level > 0:
            rng = np.random.default_rng(self.seed + 1)
            k = self.noise_level
            for e in events:
                e["latitude"] = float(np.clip(e["latitude"] + rng.normal(0, NOISE_SIGMA["pos_km"] * k / 111.0), -90, 90))
                coslat = max(np.cos(np.radians(e["latitude"])), 0.3)
                e["longitude"] = float((e["longitude"] + rng.normal(0, NOISE_SIGMA["pos_km"] * k / (111.0 * coslat)) + 180) % 360 - 180)
                e["altitude_km"] = float(e["altitude_km"] + rng.normal(0, NOISE_SIGMA["alt_km"] * k))
                e["speed_kmph"] = float(e["speed_kmph"] + rng.normal(0, NOISE_SIGMA["vel_kmh"] * k))
        return events

    def _pick(self, rng, n_total, rate, blocked: np.ndarray, gap=8, tail=10) -> List[int]:
        k = int(round(rate * n_total))
        picked = []
        candidates = rng.permutation(np.arange(3, n_total - tail))
        for idx in candidates:
            if len(picked) >= k:
                break
            lo, hi = max(0, idx - gap), min(n_total, idx + gap + 1)
            if blocked[lo:hi].any():
                continue
            picked.append(int(idx))
            blocked[lo:hi] = True
        return sorted(picked)

    def _build_events(self) -> List[Dict[str, Any]]:
        events = self._base_events()
        if not self.inject_faults:
            return events

        rng = np.random.default_rng(self.seed)
        n = len(events)
        blocked = np.zeros(n, dtype=bool)
        for i in np.where(self.natural)[0]:
            blocked[max(0, i - 3): i + 4] = True

        for i in self._pick(rng, n, FAULT_RATES["altitude_spike"], blocked):
            e = events[i]
            mag = rng.uniform(1.0, 20.0)
            e["altitude_km"] = round(e["altitude_km"] + rng.choice([-1, 1]) * mag, 2)
            e["_truth"].update(anomaly=1, type="altitude_spike", magnitude=round(float(mag), 2))
        for i in self._pick(rng, n, FAULT_RATES["velocity_spike"], blocked):
            e = events[i]
            mag = rng.uniform(5.0, 300.0)
            e["speed_kmph"] = round(e["speed_kmph"] + rng.choice([-1, 1]) * mag, 2)
            e["_truth"].update(anomaly=1, type="velocity_spike", magnitude=round(float(mag), 2))
        for i in self._pick(rng, n, FAULT_RATES["position_glitch"], blocked):
            e = events[i]
            e["latitude"] = float(np.clip(e["latitude"] + rng.choice([-1, 1]) * rng.uniform(1.0, 8.0), -89.9, 89.9))
            e["longitude"] = float((e["longitude"] + rng.choice([-1, 1]) * rng.uniform(1.0, 8.0) + 180) % 360 - 180)
            e["_truth"].update(anomaly=1, type="position_glitch")
        for i in self._pick(rng, n, FAULT_RATES["stuck_sensor"], blocked, gap=14):
            length = int(rng.integers(3, 7))
            src = events[i]
            for k in range(1, length + 1):
                e = events[i + k]
                for f in ("latitude", "longitude", "altitude_km", "speed_kmph"):
                    e[f] = src[f]
                e["_truth"].update(anomaly=1, type="stuck_sensor")

        dq_blocked = np.zeros(n, dtype=bool)
        missing_idx = self._pick(rng, n, FAULT_RATES["missing_field"], dq_blocked, gap=2)
        invalid_idx = self._pick(rng, n, FAULT_RATES["invalid_value"], dq_blocked, gap=2)
        dup_idx = set(self._pick(rng, n, FAULT_RATES["duplicate"], dq_blocked, gap=2))
        ooo_idx = set(self._pick(rng, n, FAULT_RATES["out_of_order"], dq_blocked, gap=3))
        late_idx = set(self._pick(rng, n, FAULT_RATES["late_event"], dq_blocked, gap=LATE_EVENT_DELAY + 2,
                                  tail=LATE_EVENT_DELAY + 2))
        for i in missing_idx:
            events[i].pop(rng.choice(["speed_kmph", "altitude_km"]))
            events[i]["_truth"]["dq"] = "missing_field"
        for i in invalid_idx:
            field, value = [("latitude", 999.0), ("altitude_km", -5.0), ("speed_kmph", float("nan"))][int(rng.integers(0, 3))]
            events[i][field] = value
            events[i]["_truth"]["dq"] = "invalid_value"

        order = list(range(n))
        for i in sorted(ooo_idx):
            pos = order.index(i)
            if pos + 1 < len(order):
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
                events[i]["_truth"]["dq"] = "out_of_order"
        for i in sorted(late_idx):
            pos = order.index(i)
            order.pop(pos)
            order.insert(min(pos + LATE_EVENT_DELAY, len(order)), i)
            events[i]["_truth"]["dq"] = "late_event"

        emitted = []
        for i in order:
            emitted.append(events[i])
            if i in dup_idx:
                dup = json.loads(json.dumps(events[i]))
                dup["_truth"] = {**dup["_truth"], "dq": "duplicate"}
                emitted.append(dup)
        return emitted

    def __len__(self):
        return len(self._events)

    def generate_events(self) -> Iterator[Dict[str, Any]]:
        delay = self.stream_delay_sec
        if self.speedup:
            delay = 10.0 / self.speedup
        for event in self._events:
            yield dict(event)
            if delay > 0:
                time.sleep(delay)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self._events:
            r = {k: v for k, v in e.items() if k != "_truth"}
            r["truth_anomaly"] = e["_truth"]["anomaly"]
            r["truth_type"] = e["_truth"]["type"]
            r["truth_dq"] = e["_truth"]["dq"]
            r["truth_magnitude"] = e["_truth"].get("magnitude")
            rows.append(r)
        return pd.DataFrame(rows)

    def fault_summary(self) -> Dict[str, int]:
        df = self.to_dataframe()
        out = df.loc[df.truth_anomaly == 1, "truth_type"].value_counts().to_dict()
        out.update(df["truth_dq"].dropna().value_counts().to_dict())
        out["emitted_events"] = len(df)
        return out


class LiveISSSource:
    URL = "https://api.wheretheiss.at/v1/satellites/25544"

    def __init__(self, poll_interval_sec: float = 10.0, max_events: Optional[int] = None):
        self.poll_interval_sec = poll_interval_sec
        self.max_events = max_events

    def generate_events(self) -> Iterator[Dict[str, Any]]:
        seq = 0
        while self.max_events is None or seq < self.max_events:
            with urllib.request.urlopen(self.URL, timeout=10) as resp:
                d = json.loads(resp.read())
            seq += 1
            ts = pd.Timestamp(d["timestamp"], unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            yield {
                "event_id": f"{SATELLITE_ID}-LIVE-{seq:06d}", "seq": seq, "satellite_id": SATELLITE_ID,
                "timestamp": ts, "latitude": d["latitude"], "longitude": d["longitude"],
                "altitude_km": d["altitude"], "speed_kmph": d["velocity"],
                "Hemisphere": "Northern" if d["latitude"] >= 0 else "Southern", "region": "Unknown",
                "_truth": {"anomaly": 0, "type": "live", "dq": None},
            }
            time.sleep(self.poll_interval_sec)


if __name__ == "__main__":
    clean = ISSStreamGenerator()
    print(f"Clean replay: {len(clean)} events, natural discontinuities: {int(clean.natural.sum())}")
    gen = ISSStreamGenerator(inject_faults=True)
    print("Fault-injected stream summary:", json.dumps(gen.fault_summary(), indent=2))
    for i, ev in enumerate(gen.generate_events()):
        if i < 2:
            print(json.dumps(ev))
        else:
            break