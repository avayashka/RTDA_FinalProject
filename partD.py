import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import (EARTH_RADIUS_KM, MAX_PLAUSIBLE_STEP_KM, RAW_DATA_PATH,
                            RESULTS_DIR, SPARK_STREAM_INPUT_DIR, VALID_RANGES)

TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"


def create_iss_spark_session(app_name="ISS_Realtime_Stream_Processing"):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app_name)
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def source_schema():
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType
    return StructType([
        StructField("timestamp", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("altitude_km", DoubleType(), True),
        StructField("speed_kmph", DoubleType(), True),
        StructField("Hemisphere", StringType(), True),
        StructField("region", StringType(), True),
    ])


def _normalise(df):
    from pyspark.sql import functions as F
    df = (df.withColumn("ts", F.to_timestamp("timestamp", TS_FORMAT))
            .withColumnRenamed("speed_kmph", "velocity_kmh")
            .withColumnRenamed("Hemisphere", "hemisphere")
            .withColumn("satellite_id", F.lit("ISS-25544")))
    cond = F.col("ts").isNotNull()
    for c, (lo, hi) in VALID_RANGES.items():
        cond = cond & F.col(c).between(lo, hi)
    return df.filter(cond)


def run_spark_batch_window_aggregation(csv_path=RAW_DATA_PATH, spark=None):
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    spark = spark or create_iss_spark_session()
    df = _normalise(spark.read.option("header", "true").schema(source_schema()).csv(str(csv_path)))

    w = Window.partitionBy("satellite_id").orderBy("ts")
    lat1, lon1 = F.radians(F.lag("latitude").over(w)), F.radians(F.lag("longitude").over(w))
    lat2, lon2 = F.radians("latitude"), F.radians("longitude")
    a = F.sin((lat2 - lat1) / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin((lon2 - lon1) / 2) ** 2
    df = (df.withColumn("step_km", 2 * EARTH_RADIUS_KM * F.asin(F.sqrt(a)))
            .withColumn("d_alt_km", F.col("altitude_km") - F.lag("altitude_km").over(w))
            .withColumn("discontinuity", ((F.col("step_km") > MAX_PLAUSIBLE_STEP_KM) |
                                          (F.col("step_km") < 65) | (F.abs("d_alt_km") > 0.5)).cast("int")))
    df = df.fillna({"discontinuity": 0})
    df.cache()

    aggs = [F.count("*").alias("event_count"),
            F.round(F.avg("altitude_km"), 3).alias("avg_altitude_km"),
            F.round(F.stddev("altitude_km"), 4).alias("std_altitude_km"),
            F.round(F.avg("velocity_kmh"), 3).alias("avg_velocity_kmh"),
            F.round(F.max("velocity_kmh") - F.min("velocity_kmh"), 3).alias("velocity_range_kmh"),
            F.sum("discontinuity").alias("discontinuities")]
    metric_cols = ["event_count", "avg_altitude_km", "std_altitude_km", "avg_velocity_kmh",
                   "velocity_range_kmh", "discontinuities"]
    tumbling = (df.groupBy(F.window("ts", "60 seconds"), "hemisphere").agg(*aggs)
                  .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
                          "hemisphere", *metric_cols)
                  .orderBy("window_start", "hemisphere"))
    sliding = (df.groupBy(F.window("ts", "5 minutes", "1 minute")).agg(*aggs)
                 .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
                         *metric_cols)
                 .orderBy("window_start"))
    stats = df.agg(F.count("*").alias("valid_events"), F.sum("discontinuity").alias("discontinuities"),
                   F.expr("percentile_approx(step_km, 0.5)").alias("median_step_km")).first().asDict()
    return tumbling, sliding, stats


def prepare_stream_input(csv_path=RAW_DATA_PATH, out_dir=SPARK_STREAM_INPUT_DIR, rows_per_file=360):
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    df = pd.read_csv(csv_path)
    for i in range(0, len(df), rows_per_file):
        df.iloc[i:i + rows_per_file].to_csv(out_dir / f"part_{i // rows_per_file:03d}.csv", index=False)
    return len(list(out_dir.glob("*.csv")))


def run_spark_structured_streaming(spark=None, timeout_sec=300):
    from pyspark.sql import functions as F
    spark = spark or create_iss_spark_session()
    n_files = prepare_stream_input()
    stream = (spark.readStream.option("header", "true").option("maxFilesPerTrigger", 1)
              .schema(source_schema()).csv(str(SPARK_STREAM_INPUT_DIR)))
    windows = (_normalise(stream)
               .withWatermark("ts", "20 seconds")
               .groupBy(F.window("ts", "60 seconds"))
               .agg(F.count("*").alias("event_count"),
                    F.round(F.avg("altitude_km"), 3).alias("avg_altitude_km"),
                    F.round(F.avg("velocity_kmh"), 3).alias("avg_velocity_kmh")))
    checkpoint = RESULTS_DIR / "spark_checkpoint"
    shutil.rmtree(checkpoint, ignore_errors=True)
    query = (windows.writeStream.outputMode("append").format("memory").queryName("iss_windows")
             .option("checkpointLocation", str(checkpoint)).trigger(processingTime="1 second").start())
    t0 = time.time()
    query.processAllAvailable()
    progress = [p for p in query.recentProgress if p.get("numInputRows", 0) > 0]
    query.stop()
    result = spark.sql("SELECT window.start AS window_start, window.end AS window_end, event_count, "
                       "avg_altitude_km, avg_velocity_kmh FROM iss_windows ORDER BY window_start").toPandas()
    rates = [p.get("processedRowsPerSecond", 0) for p in progress]
    return result, {"files": n_files, "micro_batches": len(progress), "wall_sec": round(time.time() - t0, 2),
                    "mean_processed_rows_per_sec": round(sum(rates) / max(len(rates), 1), 1)}


def pandas_reference_windows(csv_path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    g = df.groupby(df["ts"].dt.floor("60s"))
    return pd.DataFrame({"event_count": g.size(), "avg_altitude_km": g["altitude_km"].mean().round(3),
                         "avg_velocity_kmh": g["speed_kmph"].mean().round(3)}).reset_index(names="window_start")


def verify_against_pandas(spark_windows: pd.DataFrame) -> dict:
    ref = pandas_reference_windows()
    s = spark_windows.copy()
    s["window_start"] = pd.to_datetime(s["window_start"], utc=True)
    m = ref.merge(s, on="window_start", suffixes=("_pandas", "_spark"))
    return {"windows_pandas": len(ref), "windows_spark": len(s), "matched": len(m),
            "max_abs_diff_altitude": float((m.avg_altitude_km_pandas - m.avg_altitude_km_spark).abs().max()),
            "count_mismatches": int((m.event_count_pandas != m.event_count_spark).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["batch", "stream"], default="batch")
    args = ap.parse_args()
    try:
        import pyspark
    except ImportError:
        sys.exit("PySpark is not installed: pip install pyspark (and Java 11/17). See README.")
    spark = create_iss_spark_session()
    if args.mode == "batch":
        t0 = time.time()
        tumbling, sliding, stats = run_spark_batch_window_aggregation(spark=spark)
        tp, sp = tumbling.toPandas(), sliding.toPandas()
        print(f"Spark batch job finished in {time.time() - t0:.1f}s. Feature stats: {stats}")
        tumbling.show(8, truncate=False)
        sliding.show(5, truncate=False)
        tp.to_csv(RESULTS_DIR / "spark_tumbling_windows.csv", index=False)
        sp.to_csv(RESULTS_DIR / "spark_sliding_windows.csv", index=False)
        tp["alt_sum"] = tp["avg_altitude_km"] * tp["event_count"]
        overall = tp.groupby("window_start", as_index=False)[["event_count", "alt_sum"]].sum()
        overall["avg_altitude_km"] = (overall["alt_sum"] / overall["event_count"]).round(3)
        print("Verification vs pandas:", verify_against_pandas(overall))
    else:
        result, info = run_spark_structured_streaming(spark)
        print("Structured Streaming:", info)
        print(result.head(10).to_string(index=False))
        result.to_csv(RESULTS_DIR / "spark_streaming_windows.csv", index=False)
        print("Verification vs pandas:", verify_against_pandas(result))
    spark.stop()