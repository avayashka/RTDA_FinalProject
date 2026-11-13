import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import (ISS_TELEMETRY_TOPIC, KAFKA_NUM_PARTITIONS, SATELLITE_ID,
                            VALID_RANGES)
from sources.utils import parse_ts, to_iso

RAW_FIELD_ALIASES = {
    "speed_kmph": "velocity_kmh",
    "velocity": "velocity_kmh",
    "altitude": "altitude_km",
    "Hemisphere": "hemisphere",
    "lat": "latitude",
    "lon": "longitude",
}

REQUIRED_FIELDS = ("event_id", "timestamp", "latitude", "longitude", "altitude_km", "velocity_kmh")

DATA_SOURCE_DESCRIPTION = {
    "name": "ISS orbital telemetry (24 h, 10-second cadence)",
    "file": "data/raw_iss_telemetry.csv",
    "records": 8640,
    "period": "2025-06-07T00:00:00Z .. 2025-06-07T23:59:50Z",
    "columns": {
        "timestamp": "ISO-8601 UTC event time",
        "latitude": "geodetic latitude of the sub-satellite point, degrees [-90, 90]",
        "longitude": "longitude of the sub-satellite point, degrees [-180, 180]",
        "altitude_km": "altitude above the reference ellipsoid, km",
        "speed_kmph": "orbital speed, km/h (renamed to velocity_kmh)",
        "Hemisphere": "Northern / Southern (renamed to hemisphere)",
        "region": "ocean or country below the ISS",
    },
    "live_alternative": "https://api.wheretheiss.at/v1/satellites/25544 "
                        "(same fields; see stream_generator.LiveISSSource)",
}


class SchemaValidationError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def normalize_field_names(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in payload.items():
        out[RAW_FIELD_ALIASES.get(key, key)] = value
    return out


@dataclass
class ISSTelemetryEvent:
    event_id: str
    seq: int
    satellite_id: str
    timestamp: str
    latitude: float
    longitude: float
    altitude_km: float
    velocity_kmh: float
    hemisphere: str
    region: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def validate_and_parse(cls, payload: Dict[str, Any]) -> "ISSTelemetryEvent":
        p = normalize_field_names(payload)
        missing = [f for f in REQUIRED_FIELDS if p.get(f) is None or p.get(f) == ""]
        if missing:
            raise SchemaValidationError("missing_field", f"missing {missing}")

        values = {}
        for field in ("latitude", "longitude", "altitude_km", "velocity_kmh"):
            try:
                v = float(p[field])
            except (TypeError, ValueError):
                raise SchemaValidationError("type_error", f"{field}={p[field]!r} is not numeric")
            if not math.isfinite(v):
                raise SchemaValidationError("null_value", f"{field} is NaN/inf")
            lo, hi = VALID_RANGES[field]
            if not lo <= v <= hi:
                raise SchemaValidationError("out_of_range", f"{field}={v} outside [{lo}, {hi}]")
            values[field] = v

        try:
            ts = to_iso(parse_ts(p["timestamp"]))
        except (TypeError, ValueError):
            raise SchemaValidationError("bad_timestamp", f"timestamp={p['timestamp']!r}")

        lat = values["latitude"]
        hemisphere = str(p.get("hemisphere") or ("Northern" if lat >= 0 else "Southern"))
        return cls(
            event_id=str(p["event_id"]),
            seq=int(p.get("seq", -1)),
            satellite_id=str(p.get("satellite_id", SATELLITE_ID)),
            timestamp=ts,
            latitude=lat,
            longitude=values["longitude"],
            altitude_km=values["altitude_km"],
            velocity_kmh=values["velocity_kmh"],
            hemisphere=hemisphere,
            region=str(p.get("region") or "Unknown"),
        )


ISSTelemetrySchema = ISSTelemetryEvent

KAFKA_CLUSTER_TOPOLOGY = {
    "topic_name": ISS_TELEMETRY_TOPIC,
    "num_partitions": KAFKA_NUM_PARTITIONS,
    "replication_factor": 1,
    "partition_key": "satellite_id (preserves per-satellite event order)",
    "consumer_groups": {
        "telemetry_processor": "cleaning, windowed analytics, anomaly detection, ML inference",
        "storage_sink": "persists raw + processed events into SQLite",
        "dashboard": "reads the processed tables for visualisation",
    },
    "delivery": "at-least-once (duplicates are removed by the cleaner's event_id cache)",
}

ARCHITECTURE_LAYERS = [
    ("1. Data source", "CSV replay / live API, fault injection", "stream_generator.py"),
    ("2. Ingestion", "producer -> partitioned topic -> consumer groups", "kafka_producer.py, kafka_consumer.py"),
    ("3. Cleaning & transformation", "schema validation, dedup, watermark re-ordering, features", "partB.py"),
    ("4. Real-time analytics", "tumbling + sliding windows, indicators", "partC.py"),
    ("5. Time-series & anomaly detection", "orbital-period analysis, robust-z + Isolation Forest", "time_series_analysis.py, anomaly_detector.py"),
    ("6. ML inference", "orbit forecaster, prequential evaluation", "ml_forecaster.py"),
    ("7. Distributed processing", "multiprocess partitioned workers, Spark windows", "distributed_pipeline.py, partD.py"),
    ("8. Storage", "SQLite (raw, processed, windows, alerts, predictions)", "database.py"),
    ("9. Visualisation", "Streamlit dashboard, auto refresh", "dashboard/app.py"),
]


if __name__ == "__main__":
    sample = {
        "event_id": "ISS-000001", "seq": 1, "timestamp": "2025-06-07T00:00:00Z",
        "latitude": -10.459, "longitude": 133.051, "altitude_km": 368.18,
        "speed_kmph": 27687.89, "Hemisphere": "Southern", "region": "Pacific Ocean",
    }
    print("Validated event:", ISSTelemetryEvent.validate_and_parse(sample).to_json())
    for bad in ({**sample, "latitude": 999}, {k: v for k, v in sample.items() if k != "speed_kmph"}):
        try:
            ISSTelemetryEvent.validate_and_parse(bad)
        except SchemaValidationError as exc:
            print("Rejected  ->", exc.reason, "|", exc)
    print("\nKafka topology:", json.dumps(KAFKA_CLUSTER_TOPOLOGY, indent=2))
    print("\nArchitecture layers:")
    for layer, role, files in ARCHITECTURE_LAYERS:
        print(f"  {layer:<38} {role:<60} [{files}]")