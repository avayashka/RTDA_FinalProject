import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.anomaly_detector import ISSAnomalyDetector, evaluate_flags
from sources.config import (ANOMALY_MODEL_PATH, DB_PATH, FORECAST_MODEL_PATH,
                            RAW_DATA_PATH, RESULTS_DIR, TRAIN_FRACTION, VAL_FRACTION)
from sources.database import ISSDatabaseSink
from sources.ml_forecaster import OnlineForecaster, OrbitForecaster
from sources.partB import FEATURE_COLUMNS, FeatureBuilder, StreamCleaner
from sources.partC import SlidingWindowAggregator, TumblingWindowAggregator
from sources.utils import StageTimer, percentile_summary, to_iso


def load_or_train_models():
    if not ANOMALY_MODEL_PATH.exists() or not FORECAST_MODEL_PATH.exists():
        print("[pipeline] models not found -> running offline training (sources/train_models.py) first ...")
        from sources.train_models import run_anomaly_experiment, run_forecast_experiment
        run_anomaly_experiment(verbose=False)
        run_forecast_experiment(verbose=False)
    return ISSAnomalyDetector.load(), OrbitForecaster.load()


class StreamPipelineProcessor:
    def __init__(self, db_path=DB_PATH, reset_db: bool = False, use_sink: bool = True,
                 store_raw: bool = True, models=None):
        self.cleaner = StreamCleaner()
        self.features = FeatureBuilder()
        detector, forecaster = models or load_or_train_models()
        self.detector: ISSAnomalyDetector = detector
        self.forecaster = OnlineForecaster(forecaster)
        self.tumbling = TumblingWindowAggregator()
        self.sliding = SlidingWindowAggregator()
        self.db = ISSDatabaseSink(db_path, reset=reset_db) if use_sink else None
        self.store_raw = store_raw
        self.timer = StageTimer()
        self.latencies_ms: List[float] = []
        self.events_in = 0
        self.events_out = 0
        self._eval_rows: List[tuple] = []
        self._pred_rows: List[dict] = []
        self._t_start = time.perf_counter()
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def process_batch(self, raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = time.perf_counter()
        raw_rows = []
        for r in raw_events:
            r.setdefault("_ingest_perf", now)
            if self.store_raw:
                raw_rows.append({"event_id": str(r.get("event_id")), "received_at": time.time(),
                                 "payload": json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, default=str)})
        self.events_in += len(raw_events)

        with self.timer.stage("1_clean_validate"):
            clean = []
            for r in raw_events:
                clean.extend(self.cleaner.process(r))
        return self._process_clean(clean, raw_rows)

    def process_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        out = self.process_batch([raw_event])
        return out[-1] if out else None

    def _process_clean(self, clean: List[Dict[str, Any]], raw_rows: List[dict]) -> List[Dict[str, Any]]:
        with self.timer.stage("2_features"):
            enriched = [self.features.transform(ev) for ev in clean]

        with self.timer.stage("3_anomaly_detection"):
            if enriched:
                feat = pd.DataFrame([{c: e[c] for c in FEATURE_COLUMNS} for e in enriched])
                scores = self.detector.score(feat, include_baseline=False)
                flags = self.detector.predict_all(feat, scores, include_baseline=False)
                reasons = self.detector._reasons(feat)
                primary = self.detector.primary_method
                for i, e in enumerate(enriched):
                    e["rule_score"] = round(float(scores["rule"][i]), 3)
                    e["iforest_score"] = round(float(scores["iforest"][i]), 4) if "iforest" in scores else None
                    e["is_anomaly"] = int(flags[primary][i])
                    e["anomaly_reason"] = str(reasons[i]) if e["is_anomaly"] else ""

        with self.timer.stage("4_ml_forecast"):
            for e in enriched:
                self.forecaster.update(e)
            preds = self.forecaster.drain_completed()
            for p in preds:
                p["target_time"] = to_iso(pd.Timestamp(p["target_time"], unit="s", tz="UTC"))
            self._pred_rows.extend(p for p in preds if p["status"] == "scored")

        with self.timer.stage("5_windows"):
            windows = []
            for e in enriched:
                windows += self.tumbling.add(e)
                windows += self.sliding.add(e)

        tel_rows, alert_rows = [], []
        t_proc = time.perf_counter()
        for e in enriched:
            truth = e.get("_truth") or {}
            e["truth_anomaly"] = truth.get("anomaly")
            e["truth_type"] = truth.get("type")
            e["processed_at"] = time.time()
            e["latency_ms"] = round((t_proc - e.get("_ingest_perf", t_proc)) * 1000, 3)
            tel_rows.append(e)
            if e["is_anomaly"]:
                alert_rows.append({**e, "reason": e["anomaly_reason"]})
            if truth:
                self._eval_rows.append((e["seq"], int(truth.get("anomaly", 0)), e["is_anomaly"], e["rule_score"]))

        with self.timer.stage("6_storage_sink"):
            if self.db is not None:
                self.db.write_batch(raw=raw_rows, telemetry=tel_rows, windows=windows, anomalies=alert_rows,
                                    predictions=preds, rejected=[{**r, "rejected_at": time.time()}
                                                                 for r in self.cleaner.drain_rejected()])
            else:
                self.cleaner.drain_rejected()

        t_done = time.perf_counter()
        self.latencies_ms.extend((t_done - e.get("_ingest_perf", t_done)) * 1000 for e in enriched)
        self.events_out += len(enriched)
        return enriched

    def finish(self) -> Dict[str, Any]:
        with self.timer.stage("1_clean_validate"):
            rest = self.cleaner.flush()
        self._process_clean(rest, [])
        windows = self.tumbling.flush()
        if self.db is not None and windows:
            self.db.write_batch(windows=windows)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self._t_start
        s = {
            "events_in": self.events_in,
            "events_out": self.events_out,
            "wall_time_sec": round(elapsed, 3),
            "throughput_eps": round(self.events_out / elapsed, 1) if elapsed > 0 else None,
            "latency_ms": {k: round(v, 3) for k, v in percentile_summary(self.latencies_ms).items()},
            "stage_breakdown": self.timer.summary(self.events_out),
            "cleaning": self.cleaner.stats,
            "segments_detected": self.features.segment_id,
            "anomaly_method": self.detector.primary_method,
            "anomalies_flagged": int(sum(r[2] for r in self._eval_rows)) if self._eval_rows else None,
        }
        if self._eval_rows:
            ev = np.array(self._eval_rows)
            max_seq = ev[:, 0].max()
            test = ev[:, 0] > max_seq * (TRAIN_FRACTION + VAL_FRACTION)
            s["online_detection_all"] = evaluate_flags(ev[:, 1], ev[:, 2], ev[:, 3])
            s["online_detection_test_period"] = evaluate_flags(ev[test, 1], ev[test, 2], ev[test, 3])
        if self._pred_rows:
            pe = np.array([p["pos_error_km"] for p in self._pred_rows])
            ae = np.array([p["alt_error_km"] for p in self._pred_rows])
            s["online_forecast"] = {"horizon_sec": self._pred_rows[0]["horizon_sec"], "scored": len(pe),
                                    "pos_mae_km": round(float(pe.mean()), 4),
                                    "pos_p95_km": round(float(np.percentile(pe, 95)), 4),
                                    "alt_mae_km": round(float(ae.mean()), 4)}
        if self.db is not None:
            s["db_table_counts"] = self.db.table_counts()
        return s


def _batched_source(events, batch_size: int, max_wait_sec: float = 1.0):
    batch, t_first = [], None
    for ev in events:
        ev["_ingest_perf"] = time.perf_counter()
        if not batch:
            t_first = time.perf_counter()
        batch.append(ev)
        if len(batch) >= batch_size or time.perf_counter() - t_first >= max_wait_sec:
            yield batch
            batch = []
    if batch:
        yield batch


def run_pipeline(source: str = "csv", backend: str = "memory", batch_size: int = 50, speedup: Optional[float] = None,
                 max_events: Optional[int] = None, inject_faults: bool = True, reset_db: bool = True,
                 db_path=DB_PATH, verbose: bool = True) -> Dict[str, Any]:
    from sources.stream_generator import ISSStreamGenerator, LiveISSSource

    processor = StreamPipelineProcessor(db_path=db_path, reset_db=reset_db)
    config = {"source": source, "backend": backend, "batch_size": batch_size, "speedup": speedup,
              "max_events": max_events, "inject_faults": inject_faults}
    if source == "live":
        gen = LiveISSSource(max_events=max_events)
    else:
        gen = ISSStreamGenerator(RAW_DATA_PATH, speedup=speedup, inject_faults=inject_faults, max_events=max_events)

    n = 0
    if source in ("csv", "live"):
        for batch in _batched_source(gen.generate_events(), batch_size):
            processor.process_batch(batch)
            n += len(batch)
            if verbose and n % 1000 < len(batch):
                print(f"  processed {n} events ...", flush=True)
    elif source == "broker":
        from sources.kafka_consumer import ISSKafkaConsumer
        from sources.kafka_producer import InMemoryBroker, ISSKafkaProducer
        broker = InMemoryBroker()
        producer = ISSKafkaProducer(backend=backend, broker=broker)
        consumer = ISSKafkaConsumer("telemetry_processor", backend=producer.backend, broker=broker)
        config["backend"] = producer.backend
        done = threading.Event()

        def produce():
            for ev in gen.generate_events():
                ev["_ingest_perf"] = time.perf_counter()
                producer.publish_event(ev)
            producer.flush()
            done.set()

        threading.Thread(target=produce, daemon=True).start()
        while True:
            batch = consumer.poll(max_records=batch_size, timeout_ms=1000)
            if batch:
                processor.process_batch(batch)
                consumer.commit()
                n += len(batch)
                if verbose and n % 1000 < len(batch):
                    print(f"  consumed {n} events (lag {sum(consumer.lag().values())}) ...", flush=True)
            elif done.is_set():
                if not consumer.poll(max_records=1):
                    break
            else:
                time.sleep(0.01)
        config["partition_sizes"] = producer.stats["per_partition"]
    else:
        raise ValueError(source)

    summary = processor.finish()
    summary["config"] = config
    if processor.db is not None:
        processor.db.log_run(processor._started_at, summary["events_in"], summary["events_out"],
                             summary["throughput_eps"] or 0, config,
                             {k: v for k, v in summary.items() if k != "db_table_counts"})
    with open(RESULTS_DIR / "stream_run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if verbose:
        print(json.dumps({k: summary[k] for k in ("events_in", "events_out", "throughput_eps", "latency_ms",
                                                  "cleaning", "online_detection_test_period", "online_forecast",
                                                  "db_table_counts") if k in summary}, indent=2, default=str))
        print("Stage breakdown:", json.dumps(summary["stage_breakdown"], indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ISS real-time analytics pipeline")
    ap.add_argument("--source", default="csv", choices=["csv", "broker", "live"])
    ap.add_argument("--backend", default="auto", choices=["auto", "memory", "kafka"],
                    help="broker backend when --source broker")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--speedup", type=float, default=None,
                    help="replay speed vs. real time (e.g. 20 -> one 10-s event every 0.5 s). Default: max speed")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--no-faults", action="store_true", help="replay the clean file without injected faults")
    ap.add_argument("--append", action="store_true", help="keep existing database contents")
    a = ap.parse_args()
    run_pipeline(a.source, a.backend, a.batch_size, a.speedup, a.max_events, not a.no_faults, not a.append)