# ISS Real-Time Data Analytics Platform

**Final Project – Real-Time Data Analysis**
Alua Muratova (BDA-2401) · Ayazhan Sailauova (BDA-2402)

An end-to-end streaming analytics platform for International Space Station telemetry:
continuous event ingestion → validation and cleaning → windowed real-time analytics →
time-series analysis → anomaly detection → ML forecasting → distributed processing →
SQLite storage → live Streamlit dashboard.


## Key results

| Component | Result |
|---|---|
| Cleaning | 86 of 86 injected bad messages rejected, 8,597 valid events kept in order |
| Anomaly detection (test split) | precision / recall / F1 = 1.00 / 1.00 / 1.00 (baseline F1 = 0.17) |
| Forecast 60 s ahead | position error 0.10 km (physics baseline 1.14 km) |
| Orbital period from data | 91.4 min (latitude phase), 91.9 min (vis-viva) |
| Throughput | ~2,000 events/s per process, saturates near 3,000 events/s |
| Distributed (2 workers) | 1.37× speed-up, identical results |

## Project structure

```
├── data/raw_iss_telemetry.csv     # 24 h of ISS telemetry, 10-s cadence (8,640 events)
├── sources/
│   ├── config.py                  # paths, window sizes, thresholds, splits
│   ├── utils.py                   # geodesy helpers, timers, percentiles
│   ├── partA.py                   # event schema, validation, Kafka topology
│   ├── stream_generator.py        # replay stream, labelled fault injection, live API
│   ├── partB.py                   # cleaning (validation, dedup, watermark) + features
│   ├── partC.py                   # 60-s tumbling and 5-min sliding windows
│   ├── time_series_analysis.py    # orbital period, ACF, window justification
│   ├── anomaly_detector.py        # robust-z rule, Isolation Forest, baseline
│   ├── ml_forecaster.py           # orbit forecaster + online (prequential) scoring
│   ├── train_models.py            # train / validation / test experiments
│   ├── kafka_producer.py          # producer + in-memory partitioned broker
│   ├── kafka_consumer.py          # consumer groups, manual offset commits
│   ├── distributed_pipeline.py    # multi-process workers partitioned by satellite
│   ├── partD.py                   # Apache Spark batch + Structured Streaming
│   ├── stream_processor.py        # the integrated end-to-end pipeline
│   ├── database.py                # SQLite storage (7 tables)
│   ├── benchmark_test.py          # throughput, latency, CPU, bottlenecks
│   └── dashboard_data.py          # queries for the dashboard
├── dashboard/app.py               # Streamlit dashboard (auto-refresh)
├── tests/test_pipeline.py         # 13 unit and integration tests
├── docs/architecture.png          # architecture diagram
├── results/                       # metrics (CSV/JSON) and figures
├── FinalProject.ipynb             # walkthrough of every component
├── run_all.py                     # reproduces all results
├── docker-compose.yml             # optional Kafka broker
└── requirements.txt
```

## Installation

Python 3.10–3.12 is required. Java 11 or 17 is needed only for the optional Spark step.

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run all commands from the project root.

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m pytest -q` | runs the 13 tests |
| 2 | `python run_all.py` | runs every experiment and writes `results/summary.md` (~3 min) |
| 3 | `python -m sources.time_series_analysis` | time-series analysis and figures |
| 4 | `python -m sources.train_models` | trains and evaluates the anomaly detector and forecaster |
| 5 | `python -m sources.stream_processor` | full streaming pipeline into SQLite |
| 6 | `python -m sources.stream_processor --source broker` | same pipeline via producer → topic → consumer |
| 7 | `python -m sources.distributed_pipeline` | scaling test with 1 / 2 / 4 worker processes |
| 8 | `python -m sources.benchmark_test` | throughput, latency and CPU benchmark |
| 9 | `python -m sources.partD --mode batch` | Spark window aggregation (optional) |

### Live dashboard

Run each command in its own terminal:

```bash
python -m sources.stream_processor --speedup 20 --batch-size 1
streamlit run dashboard/app.py          # opens http://localhost:8501
```

### Real Kafka (optional)

```bash
docker compose up -d
python -m sources.stream_processor --source broker --backend kafka
docker compose down
```

## Reproducibility

- All random choices use seed 42.
- The splits are chronological: 50 % train, 20 % validation, 30 % test.
- `python run_all.py` regenerates all models, metrics, figures and databases.
