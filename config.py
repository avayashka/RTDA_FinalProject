import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_PATH = Path(os.getenv("ISS_RAW_DATA_PATH", DATA_DIR / "raw_iss_telemetry.csv"))
DB_PATH = Path(os.getenv("ISS_DB_PATH", DATA_DIR / "iss_streaming_storage.db"))
SPARK_STREAM_INPUT_DIR = DATA_DIR / "spark_stream_input"

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"
FORECAST_MODEL_PATH = MODELS_DIR / "orbit_forecaster.joblib"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ISS_TELEMETRY_TOPIC = "iss.telemetry-events"
KAFKA_NUM_PARTITIONS = 3
SATELLITE_ID = "ISS-25544"

SAMPLING_INTERVAL_SEC = 10
EARTH_RADIUS_KM = 6371.0
ISS_ORBITAL_PERIOD_MIN = 91.4

VALID_RANGES = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "altitude_km": (300.0, 500.0),
    "velocity_kmh": (26000.0, 29000.0),
}

MAX_PLAUSIBLE_STEP_KM = 220.0

TUMBLING_WINDOW_SEC = int(os.getenv("ISS_TUMBLING_WINDOW_SEC", 60))
SLIDING_WINDOW_EVENTS = int(os.getenv("ISS_SLIDING_WINDOW_EVENTS", 30))
WATERMARK_SEC = int(os.getenv("ISS_WATERMARK_SEC", 20))
DEDUP_CACHE_SIZE = 10_000

STATE_GATE = {"pos_dev_km": 5.0, "alt_dev_km": 0.6, "vel_dev_kmh": 2.0}

RANDOM_SEED = 42
TRAIN_FRACTION = 0.50
VAL_FRACTION = 0.20
FORECAST_HORIZONS = (1, 6, 30)
STREAM_FORECAST_HORIZON = 6
FORECAST_LAGS = 6

FAULT_RATES = {
    "altitude_spike": 0.006,
    "velocity_spike": 0.006,
    "position_glitch": 0.004,
    "stuck_sensor": 0.002,
    "duplicate": 0.005,
    "missing_field": 0.002,
    "invalid_value": 0.002,
    "out_of_order": 0.004,
    "late_event": 0.001,
}