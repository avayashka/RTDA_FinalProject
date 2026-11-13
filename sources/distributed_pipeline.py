import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import DATA_DIR, FIGURES_DIR, RANDOM_SEED, RESULTS_DIR

CHUNK = 200
DIST_DB = DATA_DIR / "iss_distributed_storage.db"


def partition_of(satellite_id: str, n_partitions: int) -> int:
    return int(satellite_id.split("-")[-1]) % n_partitions


def _worker(worker_id: int, in_q, out_q):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sources.anomaly_detector import ISSAnomalyDetector
    from sources.ml_forecaster import OrbitForecaster
    from sources.stream_processor import StreamPipelineProcessor
    models = (ISSAnomalyDetector.load(), OrbitForecaster.load())
    processors = {}
    n = 0
    t_busy = 0.0
    while True:
        msg = in_q.get()
        if msg is None:
            break
        t0 = time.perf_counter()
        sat = msg[0]["satellite_id"]
        proc = processors.get(sat)
        if proc is None:
            proc = processors[sat] = StreamPipelineProcessor(use_sink=False, store_raw=False, models=models)
        rows = proc.process_batch(msg)
        n += len(msg)
        t_busy += time.perf_counter() - t0
        out_q.put([_compact(r) for r in rows])
    for proc in processors.values():
        rows = proc._process_clean(proc.cleaner.flush(), [])
        out_q.put([_compact(r) for r in rows])
    out_q.put(("DONE", worker_id, n, round(t_busy, 3), os.getpid()))


def _compact(r):
    return {"event_id": r["event_id"], "satellite_id": r["satellite_id"], "timestamp": r["timestamp"],
            "latitude": r["latitude"], "longitude": r["longitude"], "altitude_km": r["altitude_km"],
            "velocity_kmh": r["velocity_kmh"], "rule_score": r["rule_score"], "is_anomaly": r["is_anomaly"],
            "truth_anomaly": (r.get("_truth") or {}).get("anomaly")}


def _sink(out_q, n_workers: int, db_path: str, report_q):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sources.database import ISSDatabaseSink
    db = ISSDatabaseSink(db_path, reset=True)
    done, workers, buf, total = 0, [], [], 0
    while done < n_workers:
        msg = out_q.get()
        if isinstance(msg, tuple) and msg[0] == "DONE":
            done += 1
            workers.append({"worker": msg[1], "events": msg[2], "busy_sec": msg[3], "pid": msg[4]})
            continue
        buf.extend(msg)
        if len(buf) >= 2000:
            db.insert_rows("telemetry", buf)
            total += len(buf)
            buf = []
    db.insert_rows("telemetry", buf)
    total += len(buf)
    per_sat = db.query("SELECT satellite_id, COUNT(*) AS events, SUM(is_anomaly) AS anomalies, "
                       "SUM(truth_anomaly) AS true_anomalies, SUM(is_anomaly * truth_anomaly) AS tp "
                       "FROM telemetry GROUP BY satellite_id ORDER BY satellite_id")
    db.close()
    report_q.put({"rows_written": total, "workers": workers, "per_satellite": per_sat.to_dict(orient="records")})


def build_constellation(n_satellites: int, events_per_sat: int):
    from sources.stream_generator import ISSStreamGenerator
    streams = []
    for k in range(n_satellites):
        gen = ISSStreamGenerator(inject_faults=True, seed=RANDOM_SEED + k, satellite_id=f"SAT-{k}",
                                 max_events=events_per_sat)
        streams.append(list(gen.generate_events()))
    out = []
    for i in range(max(len(s) for s in streams)):
        for s in streams:
            if i < len(s):
                out.append(s[i])
    return out


def run_distributed(n_workers: int, events: list) -> dict:
    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue(maxsize=64) for _ in range(n_workers)]
    out_q, report_q = ctx.Queue(maxsize=256), ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(i, in_qs[i], out_q)) for i in range(n_workers)]
    sink = ctx.Process(target=_sink, args=(out_q, n_workers, str(DIST_DB), report_q))
    for p in workers + [sink]:
        p.start()
    time.sleep(0.5)

    t0 = time.perf_counter()
    pending = {}
    for ev in events:
        key = (partition_of(ev["satellite_id"], n_workers), ev["satellite_id"])
        chunk = pending.setdefault(key, [])
        chunk.append(ev)
        if len(chunk) >= CHUNK:
            in_qs[key[0]].put(chunk)
            pending[key] = []
    for (p, _), chunk in pending.items():
        if chunk:
            in_qs[p].put(chunk)
    for q in in_qs:
        q.put(None)
    report = report_q.get()
    wall = time.perf_counter() - t0
    for p in workers + [sink]:
        p.join()
    report.update(n_workers=n_workers, wall_sec=round(wall, 3), events=len(events),
                  throughput_eps=round(len(events) / wall, 1))
    return report


def scaling_experiment(n_satellites=4, worker_counts=(1, 2, 4), events_per_sat=8640, verbose=True):
    events = build_constellation(n_satellites, events_per_sat)
    rows, reports, reference = [], {}, None
    for w in worker_counts:
        rep = run_distributed(w, events)
        reports[w] = rep
        per_sat = pd.DataFrame(rep["per_satellite"])
        if reference is None:
            reference = per_sat
        identical = per_sat.equals(reference)
        rows.append({"workers": w, "wall_sec": rep["wall_sec"], "throughput_eps": rep["throughput_eps"],
                     "rows_written": rep["rows_written"], "results_identical_to_1_worker": identical,
                     "worker_events": [x["events"] for x in sorted(rep["workers"], key=lambda x: x["worker"])]})
        if verbose:
            print(f"workers={w}: {rep['wall_sec']:.2f}s  {rep['throughput_eps']:.0f} events/s  identical={identical}")
    df = pd.DataFrame(rows)
    base = df.loc[df.workers == worker_counts[0], "wall_sec"].iloc[0]
    df["speedup"] = (base / df["wall_sec"]).round(2)
    df["efficiency"] = (df["speedup"] / (df["workers"] / worker_counts[0])).round(2)
    df["cpu_cores_available"] = os.cpu_count()
    df.to_csv(RESULTS_DIR / "distributed_scaling.csv", index=False)
    with open(RESULTS_DIR / "distributed_scaling.json", "w") as fh:
        json.dump({"satellites": n_satellites, "events_per_satellite": events_per_sat, "cpu_cores": os.cpu_count(),
                   "results": df.to_dict(orient="records"),
                   "per_satellite_quality": reports[worker_counts[0]]["per_satellite"]}, fh, indent=2, default=str)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(df.workers.astype(str), df.throughput_eps, color="#3366aa")
    ax[0].set_xlabel("worker processes"); ax[0].set_ylabel("events / s"); ax[0].set_title("Throughput")
    ax[1].plot(df.workers, df.speedup, "o-", label="measured")
    ax[1].plot(df.workers, df.workers / worker_counts[0], "--", color="grey", label="ideal (linear)")
    ax[1].axvline(os.cpu_count(), color="#cc3333", ls=":", label=f"CPU cores = {os.cpu_count()}")
    ax[1].set_xlabel("worker processes"); ax[1].set_ylabel("speed-up"); ax[1].legend(); ax[1].set_title("Speed-up")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "distributed_scaling.png", dpi=130); plt.close(fig)
    if verbose:
        print(df.drop(columns=["worker_events"]).to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--satellites", type=int, default=4)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--events", type=int, default=8640, help="events per satellite")
    a = ap.parse_args()
    print(f"CPU cores: {os.cpu_count()}; constellation of {a.satellites} satellites x {a.events} events")
    scaling_experiment(a.satellites, tuple(a.workers), a.events)
