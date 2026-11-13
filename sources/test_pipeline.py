import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.kafka_consumer import ISSKafkaConsumer
from sources.kafka_producer import InMemoryBroker, ISSKafkaProducer
from sources.ml_forecaster import OrbitForecaster, build_supervised, forecast_metrics
from sources.partA import ISSTelemetryEvent, SchemaValidationError
from sources.partB import FeatureBuilder, StreamCleaner
from sources.partC import TumblingWindowAggregator
from sources.stream_generator import ISSStreamGenerator
from sources.utils import haversine_km, latlon_to_unit, unit_to_latlon

GOOD = {"event_id": "E1", "seq": 1, "timestamp": "2025-06-07T00:00:00Z", "latitude": 10.0, "longitude": 20.0,
        "altitude_km": 370.0, "speed_kmph": 27650.0, "Hemisphere": "Northern", "region": "X"}


def _ev(i, **kw):
    e = dict(GOOD, event_id=f"E{i}", seq=i, timestamp=f"2025-06-07T00:{i // 6:02d}:{(i % 6) * 10:02d}Z")
    e.update(kw)
    return e


def test_schema_maps_source_columns():
    ev = ISSTelemetryEvent.validate_and_parse(GOOD)
    assert ev.velocity_kmh == 27650.0 and ev.hemisphere == "Northern"


@pytest.mark.parametrize("bad,reason", [
    ({"latitude": 999}, "out_of_range"), ({"altitude_km": float("nan")}, "null_value"),
    ({"timestamp": "yesterday"}, "bad_timestamp"), ({"speed_kmph": None}, "missing_field"),
    ({"latitude": "abc"}, "type_error")])
def test_schema_rejects(bad, reason):
    with pytest.raises(SchemaValidationError) as exc:
        ISSTelemetryEvent.validate_and_parse({**GOOD, **bad})
    assert exc.value.reason == reason


def test_cleaner_dedup_reorder_and_late():
    c = StreamCleaner(watermark_sec=20)
    out = []
    for i in [0, 1, 3, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 1]:
        out += c.process(_ev(i))
    out += c.flush()
    seqs = [e["seq"] for e in out]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)) == 12
    assert c.stats["duplicates"] == 2 and c.stats["reordered"] >= 1


def test_unit_vector_roundtrip_and_wrap():
    lat, lon = unit_to_latlon(latlon_to_unit(np.array([45.0, -10.0]), np.array([179.9, -179.9])))
    assert np.allclose(lat, [45, -10]) and np.allclose(lon, [179.9, -179.9])
    assert haversine_km(0, 179.9, 0, -179.9) < 25


def test_feature_builder_flags_spike_but_not_following_event():
    gen = ISSStreamGenerator(max_events=60)
    c, fb = StreamCleaner(), FeatureBuilder()
    rows = []
    for i, raw in enumerate(gen.generate_events()):
        if i == 30:
            raw["altitude_km"] += 8.0
        rows += [fb.transform(e) for e in c.process(raw)]
    rows += [fb.transform(e) for e in c.flush()]
    flags = {r["seq"]: r["state_gate_flag"] for r in rows}
    assert flags[31] == 1
    assert flags[32] == 0


def test_natural_discontinuities_labelled():
    gen = ISSStreamGenerator()
    assert 300 < gen.natural.sum() < 360


def test_tumbling_window_counts():
    agg = TumblingWindowAggregator(60)
    out = []
    for i in range(18):
        e = ISSTelemetryEvent.validate_and_parse(_ev(i)).to_dict()
        out += agg.add(e)
    out += agg.flush()
    assert [w["event_count"] for w in out] == [6, 6, 6]


def test_broker_partitions_and_consumer_groups():
    broker = InMemoryBroker()
    prod = ISSKafkaProducer(backend="memory", broker=broker)
    for i in range(30):
        prod.publish_event({**_ev(i), "satellite_id": f"SAT-{i % 3}"})
    a = ISSKafkaConsumer("group_a", backend="memory", broker=broker)
    b = ISSKafkaConsumer("group_b", backend="memory", broker=broker)
    got_a = a.poll(100); a.commit()
    assert len(got_a) == 30 and len(b.poll(100)) == 30
    assert a.poll(100) == []


def test_ridge_beats_physics_baseline():
    from sources.train_models import build_feature_frame, chronological_split
    df = build_feature_frame(inject_faults=False, max_events=4000)
    train, _, test = chronological_split(df)
    Xtr, Ytr, _ = build_supervised(train, 6)
    Xte, Yte, _ = build_supervised(test, 6)
    ridge = forecast_metrics(OrbitForecaster(6, model="ridge").fit(Xtr, Ytr), Xte, Yte)
    cv = forecast_metrics(OrbitForecaster(6, model="constant_velocity").fit(Xtr, Ytr), Xte, Yte)
    assert ridge["pos_mae_km"] < cv["pos_mae_km"] < 5
