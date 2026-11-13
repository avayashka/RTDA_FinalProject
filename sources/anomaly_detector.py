import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import ANOMALY_MODEL_PATH, RANDOM_SEED
from sources.partB import FEATURE_COLUMNS

RULE_FEATURES = ["pos_dev_km", "alt_dev_km", "vel_dev_kmh"]
MAD_FLOOR = {"pos_dev_km": 0.05, "alt_dev_km": 0.01, "vel_dev_kmh": 0.01}
RAW_FEATURES = ["latitude", "longitude", "altitude_km", "velocity_kmh"]


def _iforest_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


class ISSAnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.contamination = contamination
        self.iforest = IsolationForest(n_estimators=200, random_state=random_state, n_jobs=1)
        self.raw_iforest = IsolationForest(n_estimators=200, contamination=contamination,
                                           random_state=random_state, n_jobs=1)
        self.medians: Dict[str, float] = {}
        self.scales: Dict[str, float] = {}
        self.thresholds = {"rule": 6.0, "iforest": 0.0}
        self.primary_method = "hybrid"
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ISSAnomalyDetector":
        for f in RULE_FEATURES:
            x = train_df[f].to_numpy(dtype=float)
            med = float(np.median(x))
            self.medians[f] = med
            self.scales[f] = max(1.4826 * float(np.median(np.abs(x - med))), MAD_FLOOR[f])
        self.iforest.fit(_iforest_matrix(train_df))
        self.raw_iforest.fit(train_df[RAW_FEATURES].to_numpy(dtype=float))
        self.thresholds["iforest"] = float(np.percentile(self._iforest_score(train_df), 99))
        self.is_fitted = True
        return self

    def tune_thresholds(self, val_df: pd.DataFrame, y_val: np.ndarray) -> Dict[str, dict]:
        report = {}
        scores = self.score(val_df)
        for method, cand in (("rule", np.r_[np.arange(3, 20, 0.5), np.arange(20, 200, 5)]),
                             ("iforest", np.unique(np.percentile(scores["iforest"], np.arange(80, 99.9, 0.1))))):
            best = max(cand, key=lambda th: f1_score(y_val, scores[method] > th, zero_division=0))
            self.thresholds[method] = float(best)
            report[method] = {"threshold": float(best),
                              "val_f1": float(f1_score(y_val, scores[method] > best, zero_division=0))}
        self.primary_method = "hybrid"
        flags = self.predict_all(val_df)
        f1s = {m: f1_score(y_val, flags[m], zero_division=0) for m in ("rule", "iforest", "hybrid")}
        self.primary_method = max(f1s, key=f1s.get)
        report["hybrid"] = {"val_f1": float(f1s["hybrid"])}
        report["primary_method"] = self.primary_method
        return report

    def _rule_score(self, df: pd.DataFrame) -> np.ndarray:
        z = [(df[f].to_numpy(dtype=float) - self.medians[f]) / self.scales[f] for f in RULE_FEATURES]
        return np.max(np.vstack(z), axis=0)

    def _iforest_score(self, df: pd.DataFrame) -> np.ndarray:
        return -self.iforest.decision_function(_iforest_matrix(df))

    def score(self, df: pd.DataFrame, include_baseline: bool = True) -> Dict[str, np.ndarray]:
        out = {"rule": self._rule_score(df), "iforest": self._iforest_score(df)}
        if include_baseline:
            out["raw_iforest_baseline"] = -self.raw_iforest.decision_function(df[RAW_FEATURES].to_numpy(dtype=float))
        return out

    def predict_all(self, df: pd.DataFrame, scores: Optional[Dict[str, np.ndarray]] = None,
                    include_baseline: bool = True) -> Dict[str, np.ndarray]:
        s = scores if scores is not None else self.score(df, include_baseline)
        rule = s["rule"] > self.thresholds["rule"]
        out = {"rule": rule.astype(int)}
        if "iforest" in s:
            iso = s["iforest"] > self.thresholds["iforest"]
            out["iforest"] = iso.astype(int)
            out["hybrid"] = (rule | iso).astype(int)
        if include_baseline:
            out["raw_iforest_baseline"] = (self.raw_iforest.predict(df[RAW_FEATURES].to_numpy(dtype=float)) == -1).astype(int)
        return out

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        flags = self.predict_all(out)
        scores = self.score(out)
        out["rule_score"] = np.round(scores["rule"], 3)
        out["iforest_score"] = np.round(scores["iforest"], 4)
        out["rule_flag"] = flags["rule"]
        out["iforest_flag"] = flags["iforest"]
        out["is_anomaly"] = flags[self.primary_method]
        out["anomaly_score"] = out["rule_score"]
        out["anomaly_reason"] = np.where(out["is_anomaly"] == 1, self._reasons(out), "")
        return out

    def _reasons(self, df: pd.DataFrame) -> np.ndarray:
        z = np.vstack([(df[f].to_numpy(dtype=float) - self.medians[f]) / self.scales[f] for f in RULE_FEATURES]).T
        names = np.array(["position", "altitude", "speed"])
        return names[z.argmax(axis=1)]

    def save(self, path=ANOMALY_MODEL_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=ANOMALY_MODEL_PATH) -> "ISSAnomalyDetector":
        return joblib.load(path)


def evaluate_flags(y_true: np.ndarray, flags: np.ndarray, scores: Optional[np.ndarray] = None,
                   hours: Optional[float] = None) -> Dict[str, float]:
    y_true, flags = np.asarray(y_true), np.asarray(flags)
    out = {
        "precision": round(float(precision_score(y_true, flags, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, flags, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, flags, zero_division=0)), 4),
        "tp": int(((flags == 1) & (y_true == 1)).sum()),
        "fp": int(((flags == 1) & (y_true == 0)).sum()),
        "fn": int(((flags == 0) & (y_true == 1)).sum()),
        "tn": int(((flags == 0) & (y_true == 0)).sum()),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    if hours:
        out["false_alarms_per_hour"] = round(out["fp"] / hours, 3)
    return out


if __name__ == "__main__":
    from sources.train_models import build_feature_frame, chronological_split
    df = build_feature_frame(inject_faults=True)
    train, val, test = chronological_split(df)
    det = ISSAnomalyDetector().fit(train)
    print("Tuning:", det.tune_thresholds(val, val["truth_anomaly"].to_numpy()))
    res = det.detect_anomalies(test)
    print("Test (primary method):", evaluate_flags(test["truth_anomaly"], res["is_anomaly"]))
