import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.anomaly_detector import ISSAnomalyDetector
from sources.config import FIGURES_DIR, RESULTS_DIR
from sources.ml_forecaster import OrbitForecaster
from sources.stream_generator import ISSStreamGenerator
from sources.stream_processor import StreamPipelineProcessor, load_or_train_models
from sources.utils import percentile_summary

BENCH_DB = RESULTS_DIR / "benchmark_tmp.db"
PROC = psutil.Process(os.getpid())


def _events(n):
    return list(ISSStreamGenerator(inject_faults=True).generate_events())[:n]


def _fresh_processor(models):
    return StreamPipelineProcessor(db_path=BENCH_DB, reset_db=True, models=models)


def batch_size_sweep(models, n_events=3000, sizes=(1, 10, 50, 200, 500)):
    rows, stages = [], {}
    for bs in sizes:
        events = _events(n_events)
        proc = _fresh_processor(models)
        cpu0, t0 = PROC.cpu_times(), time.perf_counter()
        for i in range(0, len(events), bs):
            batch = events[i:i + bs]
            stamp = time.perf_counter()
            for e in batch:
                e["_ingest_perf"] = stamp
            proc.process_batch(batch)
        proc.finish()
        wall = time.perf_counter() - t0
        cpu1 = PROC.cpu_times()
        lat = percentile_summary(proc.latencies_ms)
        br = proc.timer.summary(proc.events_out)
        stages[bs] = {k: v["us_per_event"] for k, v in br.items()}
        rows.append({"batch_size": bs, "events": proc.events_out, "wall_sec": round(wall, 3),
                     "throughput_eps": round(proc.events_out / wall, 1),
                     "latency_p50_ms": round(lat["p50"], 3), "latency_p95_ms": round(lat["p95"], 3),
                     "latency_p99_ms": round(lat["p99"], 3),
                     "cpu_pct_of_one_core": round(100 * ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / wall, 1),
                     "rss_mb": round(PROC.memory_info().rss / 2 ** 20, 1),
                     "bottleneck_stage": next(iter(br))})
        proc.db.close()
        print(f"  batch={bs:>4}: {rows[-1]['throughput_eps']:>8.0f} ev/s  p95={rows[-1]['latency_p95_ms']:.2f} ms  "
              f"bottleneck={rows[-1]['bottleneck_stage']}")
    return pd.DataFrame(rows), pd.DataFrame(stages).T.rename_axis("batch_size")


def rate_controlled(models, rates=(100, 500, 1000, 2000, 4000, 8000), max_batch=100, max_wait=0.05,
                    seconds=3.0, max_events=6000):
    rows = []
    for rate in rates:
        n = int(min(max_events, rate * seconds))
        events = _events(n)
        proc = _fresh_processor(models)
        t0 = time.perf_counter()
        sched = t0 + np.arange(n) / rate
        i, pending = 0, []
        while i < n or pending:
            now = time.perf_counter()
            while i < n and sched[i] <= now:
                events[i]["_ingest_perf"] = sched[i]
                pending.append(events[i])
                i += 1
            if pending and (len(pending) >= max_batch or now - pending[0]["_ingest_perf"] >= max_wait or i >= n):
                proc.process_batch(pending)
                pending = []
                continue
            nxt = min(sched[i] if i < n else now, pending[0]["_ingest_perf"] + max_wait if pending else np.inf)
            if nxt > now:
                time.sleep(min(nxt - now, 0.005))
        proc.finish()
        wall = time.perf_counter() - t0
        lat = percentile_summary(proc.latencies_ms)
        rows.append({"target_rate_eps": rate, "events": proc.events_out, "achieved_eps": round(proc.events_out / wall, 1),
                     "latency_p50_ms": round(lat["p50"], 2), "latency_p95_ms": round(lat["p95"], 2),
                     "latency_p99_ms": round(lat["p99"], 2), "saturated": proc.events_out / wall < 0.95 * rate})
        proc.db.close()
        print(f"  rate={rate:>5}: achieved {rows[-1]['achieved_eps']:>7.0f} ev/s  p50={lat['p50']:.1f} ms  p99={lat['p99']:.1f} ms")
    return pd.DataFrame(rows)


def storage_pattern_comparison(n=1000):
    path = RESULTS_DIR / "benchmark_sink.db"
    rows = [(f"E{i}", "2025-06-07T00:00:00Z", 1.0, 2.0, 370.0, 27600.0, "Northern", 0) for i in range(n)]
    ddl = ("CREATE TABLE t (event_id TEXT PRIMARY KEY, timestamp TEXT, latitude REAL, longitude REAL, "
           "altitude_km REAL, velocity_kmh REAL, hemisphere TEXT, is_anomaly INTEGER)")
    sql = "INSERT OR REPLACE INTO t VALUES (?,?,?,?,?,?,?,?)"
    res = {}
    for name in ("per_event_connect_commit", "single_connection_batched"):
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
        c = sqlite3.connect(path)
        c.execute(ddl)
        c.commit()
        c.close()
        t0 = time.perf_counter()
        if name == "per_event_connect_commit":
            for r in rows:
                c = sqlite3.connect(path)
                c.execute(sql, r)
                c.commit()
                c.close()
        else:
            c = sqlite3.connect(path)
            c.execute("PRAGMA journal_mode=WAL")
            for i in range(0, n, 50):
                c.executemany(sql, rows[i:i + 50])
                c.commit()
            c.close()
        dt = time.perf_counter() - t0
        res[name] = {"events": n, "sec": round(dt, 4), "us_per_event": round(1e6 * dt / n, 1)}
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    res["speedup"] = round(res["per_event_connect_commit"]["sec"] / res["single_connection_batched"]["sec"], 1)
    return res


def run_all(n_events=3000):
    models = load_or_train_models()
    print("1) micro-batch sweep")
    sweep, stages = batch_size_sweep(models, n_events)
    print("2) rate-controlled load")
    rates = rate_controlled(models)
    print("3) storage pattern")
    storage = storage_pattern_comparison()
    print("  ", storage)

    sweep.to_csv(RESULTS_DIR / "benchmark_batch_sweep.csv", index=False)
    stages.to_csv(RESULTS_DIR / "benchmark_stage_us_per_event.csv")
    rates.to_csv(RESULTS_DIR / "benchmark_rate_controlled.csv", index=False)
    summary = {"machine": {"cpu_cores": os.cpu_count(), "python": sys.version.split()[0]},
               "batch_sweep": sweep.to_dict(orient="records"),
               "stage_us_per_event": stages.round(2).to_dict(orient="index"),
               "rate_controlled": rates.to_dict(orient="records"), "storage_pattern": storage}
    with open(RESULTS_DIR / "benchmark_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    BENCH_DB.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(BENCH_DB) + suffix).unlink(missing_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sweep.batch_size, sweep.throughput_eps, "o-", color="#3366aa")
    ax[0].set_xscale("log"); ax[0].set_xlabel("micro-batch size"); ax[0].set_ylabel("events / s")
    ax0b = ax[0].twinx()
    ax0b.plot(sweep.batch_size, sweep.latency_p95_ms, "s--", color="#cc3333")
    ax0b.set_ylabel("P95 latency (ms)", color="#cc3333"); ax[0].set_title("Throughput vs latency trade-off")
    stages.plot.bar(stacked=True, ax=ax[1], colormap="tab10")
    ax[1].set_ylabel("microseconds per event"); ax[1].set_title("Per-stage cost by batch size")
    ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=0)
    ax[2].plot(rates.target_rate_eps, rates.achieved_eps, "o-", label="achieved")
    ax[2].plot(rates.target_rate_eps, rates.target_rate_eps, "--", color="grey", label="target")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("input rate (events/s)")
    ax[2].set_ylabel("events / s"); ax[2].legend(loc="upper left")
    ax2b = ax[2].twinx()
    ax2b.plot(rates.target_rate_eps, rates.latency_p99_ms, "s--", color="#cc3333")
    ax2b.set_yscale("log"); ax2b.set_ylabel("P99 latency (ms)", color="#cc3333"); ax[2].set_title("Rate-controlled load")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "benchmark.png", dpi=130); plt.close(fig)
    print("\nBatch sweep:\n", sweep.to_string(index=False))
    print("\nRate-controlled:\n", rates.to_string(index=False))
    return summary


def run_performance_benchmark(rates=(10, 100, 500)):
    return batch_size_sweep(load_or_train_models(), n_events=max(rates), sizes=(1, 50))[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=3000)
    run_all(ap.parse_args().events)