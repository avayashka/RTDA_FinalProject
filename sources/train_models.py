import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.anomaly_detector import ISSAnomalyDetector, evaluate_flags
from sources.config import (FIGURES_DIR, FORECAST_HORIZONS, RESULTS_DIR,
                            STREAM_FORECAST_HORIZON, TRAIN_FRACTION, VAL_FRACTION)
from sources.ml_forecaster import OrbitForecaster, build_supervised, forecast_metrics
from sources.partB import FeatureBuilder, StreamCleaner
from sources.stream_generator import ISSStreamGenerator

METHODS = ["raw_iforest_baseline", "rule", "iforest", "hybrid"]


def build_feature_frame(inject_faults: bool = True, max_events=None, noise_level: float = 0.0) -> pd.DataFrame:
    gen = ISSStreamGenerator(inject_faults=inject_faults, max_events=max_events, noise_level=noise_level)
    cleaner, fb = StreamCleaner(), FeatureBuilder()
    rows = []
    for raw in gen.generate_events():
        rows.extend(fb.transform(ev) for ev in cleaner.process(raw))
    rows.extend(fb.transform(ev) for ev in cleaner.flush())
    df = pd.DataFrame(rows)
    truth = df.pop("_truth")
    df["truth_anomaly"] = truth.map(lambda t: t["anomaly"]).astype(int)
    df["truth_type"] = truth.map(lambda t: t["type"])
    df["truth_magnitude"] = truth.map(lambda t: t.get("magnitude"))
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    n = len(df)
    a, b = int(n * TRAIN_FRACTION), int(n * (TRAIN_FRACTION + VAL_FRACTION))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def run_anomaly_experiment(verbose=True) -> dict:
    df = build_feature_frame(inject_faults=True)
    train, val, test = chronological_split(df)
    det = ISSAnomalyDetector().fit(train)
    tuning = det.tune_thresholds(val, val["truth_anomaly"].to_numpy())
    det.save()

    y = test["truth_anomaly"].to_numpy()
    hours = (test["_event_time"].iloc[-1] - test["_event_time"].iloc[0]) / 3600
    flags = det.predict_all(test)
    scores = det.score(test)
    scores["hybrid"] = np.maximum(scores["rule"] / det.thresholds["rule"], scores["iforest"] / det.thresholds["iforest"])
    rows = []
    for m in METHODS:
        r = evaluate_flags(y, flags[m], scores[m], hours)
        r["method"] = m
        rows.append(r)
    table = pd.DataFrame(rows).set_index("method")

    primary = det.primary_method
    test = test.assign(pred=flags[primary])
    per_type = (test[test.truth_anomaly == 1].groupby("truth_type")["pred"]
                .agg(["count", "sum"]).rename(columns={"sum": "detected"}))
    per_type["recall"] = (per_type["detected"] / per_type["count"]).round(4)
    fp_by_type = int(((test.pred == 1) & (test.truth_anomaly == 0)).sum())

    mag = {}
    for typ, bins in (("altitude_spike", [1, 2, 5, 10, 20]), ("velocity_spike", [5, 15, 50, 150, 300])):
        sub = df[df.truth_type == typ].copy()
        sub["pred"] = det.predict_all(sub)[primary]
        sub["bucket"] = pd.cut(sub["truth_magnitude"], bins).astype(str)
        mag[typ] = sub.groupby("bucket")["pred"].agg(["count", "mean"]).round(3).to_dict(orient="index")

    result = {
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_anomaly_rate": round(float(y.mean()), 4),
        "test_hours": round(hours, 2),
        "thresholds": det.thresholds, "tuning": tuning, "primary_method": primary,
        "test_metrics": table.to_dict(orient="index"),
        "recall_by_type_test": per_type.to_dict(orient="index"),
        "false_positives_test": fp_by_type,
        "recall_by_magnitude_all_splits": mag,
        "scales": det.scales,
    }
    table.to_csv(RESULTS_DIR / "anomaly_methods.csv")
    with open(RESULTS_DIR / "anomaly_evaluation.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    _anomaly_figures(test, y, scores, flags, det, table)
    if verbose:
        print("Anomaly detection - test split")
        print(table[["precision", "recall", "f1", "roc_auc", "pr_auc", "false_alarms_per_hour"]].to_string())
        print("\nRecall by anomaly type (primary method =", primary + "):")
        print(per_type.to_string())
    return result


def _anomaly_figures(test, y, scores, flags, det, table):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, c in zip(METHODS, ["#999999", "#3366aa", "#dd8833", "#339966"]):
        p, r, _ = precision_recall_curve(y, scores[m])
        ax[0].plot(r, p, color=c, label=f"{m} (AP={table.loc[m, 'pr_auc']:.3f})")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("Precision-recall (test split)")
    ax[0].legend(fontsize=8)
    t = pd.to_datetime(test["timestamp"])
    ax[1].plot(t, test["altitude_km"], lw=0.6, color="#3366aa")
    hit = flags[det.primary_method] == 1
    ax[1].scatter(t[hit & (y == 1)], test["altitude_km"][hit & (y == 1)], s=10, c="#339966", label="true positive")
    ax[1].scatter(t[hit & (y == 0)], test["altitude_km"][hit & (y == 0)], s=18, c="#cc3333", marker="x", label="false positive")
    ax[1].scatter(t[~hit & (y == 1)], test["altitude_km"][~hit & (y == 1)], s=18, facecolors="none", edgecolors="k", label="missed")
    ax[1].set_ylabel("altitude (km)"); ax[1].set_title(f"Detections on test split ({det.primary_method})")
    ax[1].legend(fontsize=8); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "anomaly_evaluation.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    table[["precision", "recall", "f1"]].plot.bar(ax=ax, color=["#3366aa", "#dd8833", "#339966"])
    ax.set_ylim(0, 1.05); ax.set_title("Anomaly detection methods (test split)"); ax.set_xlabel("")
    plt.xticks(rotation=15); fig.tight_layout(); fig.savefig(FIGURES_DIR / "anomaly_methods.png", dpi=130); plt.close(fig)


def run_noise_sensitivity(levels=(0, 1, 3, 10), verbose=True) -> pd.DataFrame:
    rows = []
    for k in levels:
        df = build_feature_frame(inject_faults=True, noise_level=k)
        train, val, test = chronological_split(df)
        det = ISSAnomalyDetector().fit(train)
        det.tune_thresholds(val, val["truth_anomaly"].to_numpy())
        y = test["truth_anomaly"].to_numpy()
        flags, scores = det.predict_all(test), det.score(test)
        subtle = (test["truth_type"] == "altitude_spike") & (test["truth_magnitude"] < 5)
        for m in METHODS:
            r = evaluate_flags(y, flags[m], scores.get(m))
            rows.append({"noise_level": k, "method": m, "precision": r["precision"], "recall": r["recall"],
                         "f1": r["f1"], "pr_auc": r.get("pr_auc"),
                         "gate_false_flags": int(((test.state_gate_flag == 1) & (y == 0)).sum()),
                         "subtle_alt_recall": round(float(flags[m][subtle.to_numpy()].mean()), 3) if subtle.any() else None})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "anomaly_noise_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, c in zip(METHODS, ["#999999", "#3366aa", "#dd8833", "#339966"]):
        sub = table[table.method == m]
        ax.plot(sub.noise_level, sub.f1, "o-", color=c, label=m)
    ax.set_xscale("symlog"); ax.set_xticks(list(levels)); ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("sensor noise level (x base sigma)"); ax.set_ylabel("F1 (test)"); ax.set_ylim(0, 1.05)
    ax.set_title("Robustness of anomaly detection to sensor noise"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "anomaly_noise_sensitivity.png", dpi=130); plt.close(fig)
    if verbose:
        print("\nNoise sensitivity (test F1):")
        print(table.pivot(index="noise_level", columns="method", values="f1").to_string())
    return table


def run_forecast_experiment(verbose=True) -> dict:
    df = build_feature_frame(inject_faults=False)
    train, val, test = chronological_split(df)
    rows = []
    chosen = None
    for H in FORECAST_HORIZONS:
        Xtr, Ytr, _ = build_supervised(train, H)
        Xva, Yva, _ = build_supervised(val, H)
        Xte, Yte, _ = build_supervised(test, H)
        Xtv, Ytv = np.vstack([Xtr, Xva]), np.vstack([Ytr, Yva])
        val_scores = {}
        for name in ("persistence", "constant_velocity", "ridge", "gradient_boosting"):
            val_scores[name] = forecast_metrics(OrbitForecaster(H, model=name).fit(Xtr, Ytr), Xva, Yva)["pos_mae_km"]
            model = OrbitForecaster(H, model=name).fit(Xtv, Ytv)
            m = forecast_metrics(model, Xte, Yte)
            m.update(horizon_steps=H, horizon_sec=10 * H, model=name, n_train=len(Xtv),
                     val_pos_mae_km=val_scores[name])
            rows.append(m)
        if H == STREAM_FORECAST_HORIZON:
            ml = {k: v for k, v in val_scores.items() if k in ("ridge", "gradient_boosting")}
            chosen = min(ml, key=ml.get)
            OrbitForecaster(H, model=chosen).fit(Xtv, Ytv).save()
    table = pd.DataFrame(rows)[["horizon_sec", "model", "n", "n_train", "val_pos_mae_km", "pos_mae_km",
                                "pos_rmse_km", "pos_p95_km", "alt_mae_km", "vel_mae_kmh"]]
    table.to_csv(RESULTS_DIR / "forecast_evaluation.csv", index=False)
    with open(RESULTS_DIR / "forecast_evaluation.json", "w") as fh:
        json.dump({"stream_model": chosen, "stream_horizon_sec": 10 * STREAM_FORECAST_HORIZON,
                   "results": rows}, fh, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv = table.pivot(index="horizon_sec", columns="model", values="pos_mae_km")
    piv.plot.bar(ax=ax[0], logy=True)
    ax[0].set_ylabel("position MAE (km, log)"); ax[0].set_title("Position forecast error by horizon (test)")
    piv2 = table.pivot(index="horizon_sec", columns="model", values="alt_mae_km")
    piv2.plot.bar(ax=ax[1], logy=True)
    ax[1].set_ylabel("altitude MAE (km, log)"); ax[1].set_title("Altitude forecast error by horizon (test)")
    for a in ax:
        a.set_xlabel("horizon (s)"); a.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "forecast_evaluation.png", dpi=130); plt.close(fig)
    if verbose:
        print("\nForecasting - test split (models refit on train+validation)")
        print(table.to_string(index=False))
        print(f"\nStreaming model (H={10 * STREAM_FORECAST_HORIZON}s): {chosen}")
    return {"table": table, "stream_model": chosen}


if __name__ == "__main__":
    run_anomaly_experiment()
    run_noise_sensitivity()
    run_forecast_experiment()
