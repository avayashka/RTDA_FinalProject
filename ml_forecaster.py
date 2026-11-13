import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import (FORECAST_LAGS, FORECAST_MODEL_PATH, RANDOM_SEED,
                            STREAM_FORECAST_HORIZON)
from sources.partB import _rotate_predict
from sources.utils import angular_distance_km, latlon_to_unit, unit_to_latlon

TARGETS = ["dx", "dy", "dz", "d_alt", "d_vel"]


def _lag_features(U: np.ndarray, alt: np.ndarray, vel: np.ndarray) -> np.ndarray:
    dU = np.diff(U, axis=0)
    return np.concatenate([U[-1], dU.ravel() * 100, [alt[-1]], np.diff(alt), [vel[-1] / 1000], np.diff(vel)])


def build_supervised(df: pd.DataFrame, horizon: int, lags: int = FORECAST_LAGS):
    U = latlon_to_unit(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    alt = df["altitude_km"].to_numpy(dtype=float)
    vel = df["velocity_kmh"].to_numpy(dtype=float)
    seg = df["segment_id"].to_numpy()
    bad = df["state_gate_flag"].to_numpy().astype(bool)
    t = df["_event_time"].to_numpy(dtype=float)
    X, Y, idx = [], [], []
    bad_cum = np.r_[0, np.cumsum(bad)]
    for i in range(lags - 1, len(df) - horizon):
        lo, hi = i - lags + 1, i + horizon
        if seg[lo] != seg[hi] or bad_cum[hi + 1] - bad_cum[lo] > 0:
            continue
        if abs((t[hi] - t[i]) - 10 * horizon) > 1 or abs((t[i] - t[lo]) - 10 * (lags - 1)) > 1:
            continue
        X.append(_lag_features(U[lo:i + 1], alt[lo:i + 1], vel[lo:i + 1]))
        Y.append(np.r_[U[hi] - U[i], alt[hi] - alt[i], vel[hi] - vel[i]])
        idx.append(i)
    meta = pd.DataFrame({"row": idx})
    return np.array(X), np.array(Y), meta


class OrbitForecaster:
    def __init__(self, horizon: int = STREAM_FORECAST_HORIZON, lags: int = FORECAST_LAGS,
                 model: str = "ridge", random_state: int = RANDOM_SEED):
        self.horizon, self.lags, self.model_name = horizon, lags, model
        if model == "ridge":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
        elif model == "gradient_boosting":
            self.model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, random_state=random_state))
        elif model in ("persistence", "constant_velocity"):
            self.model = None
        else:
            raise ValueError(model)
        self.y_scale = np.array([100, 100, 100, 1, 1], dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OrbitForecaster":
        if self.model is not None:
            self.model.fit(X, Y * self.y_scale)
        self._fast = None
        if self.model_name == "ridge":
            sc, rg = self.model.named_steps["standardscaler"], self.model.named_steps["ridge"]
            W = rg.coef_ / sc.scale_
            b = rg.intercept_ - W @ sc.mean_
            self._fast = (W.T / self.y_scale, b / self.y_scale)
        return self

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        L = self.lags
        if self.model_name == "persistence":
            return np.zeros((len(X), 5))
        if self.model_name == "constant_velocity":
            out = []
            for x in X:
                u1 = x[:3]
                du_last = x[3 + 3 * (L - 2): 3 + 3 * (L - 1)] / 100
                u0 = u1 - du_last
                pred = _rotate_predict(u0 / np.linalg.norm(u0), u1, 1.0, float(self.horizon))
                d_alt_last = x[3 + 3 * (L - 1) + 1 + (L - 2)]
                d_vel_last = x[-1]
                out.append(np.r_[pred - u1, d_alt_last * self.horizon, d_vel_last * self.horizon])
            return np.array(out)
        if getattr(self, "_fast", None) is not None:
            W, b = self._fast
            return X @ W + b
        return self.model.predict(X) / self.y_scale

    def predict_absolute(self, X: np.ndarray):
        d = self.predict_delta(X)
        L = self.lags
        u = X[:, :3] + d[:, :3]
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        alt = X[:, 3 + 3 * (L - 1)] + d[:, 3]
        vel = X[:, 3 + 3 * (L - 1) + L] * 1000 + d[:, 4]
        return u, alt, vel

    def save(self, path=FORECAST_MODEL_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=FORECAST_MODEL_PATH) -> "OrbitForecaster":
        return joblib.load(path)


def forecast_metrics(model: OrbitForecaster, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    u_pred, alt_pred, vel_pred = model.predict_absolute(X)
    L = model.lags
    u_true = X[:, :3] + Y[:, :3]
    u_true /= np.linalg.norm(u_true, axis=1, keepdims=True)
    pos_err = angular_distance_km(u_pred, u_true)
    alt_true = X[:, 3 + 3 * (L - 1)] + Y[:, 3]
    vel_true = X[:, 3 + 3 * (L - 1) + L] * 1000 + Y[:, 4]
    return {
        "n": int(len(X)),
        "pos_mae_km": round(float(pos_err.mean()), 4),
        "pos_rmse_km": round(float(np.sqrt((pos_err ** 2).mean())), 4),
        "pos_p95_km": round(float(np.percentile(pos_err, 95)), 4),
        "alt_mae_km": round(float(np.abs(alt_pred - alt_true).mean()), 4),
        "vel_mae_kmh": round(float(np.abs(vel_pred - vel_true).mean()), 4),
    }


class OnlineForecaster:
    def __init__(self, model: OrbitForecaster):
        self.model = model
        self.L, self.H = model.lags, model.horizon
        self._hist: deque = deque(maxlen=self.L)
        self._segment = None
        self._pending: Dict[float, dict] = {}
        self.completed: List[dict] = []

    def update(self, ev: Dict) -> Optional[dict]:
        t = ev["_event_time"]
        self._score_pending(ev)
        if ev.get("is_anomaly") or ev.get("state_gate_flag"):
            return None
        if ev["segment_id"] != self._segment:
            self._hist.clear()
            self._segment = ev["segment_id"]
            for p in self._pending.values():
                p["status"] = "invalidated_by_discontinuity"
                self.completed.append(p)
            self._pending.clear()
        if self._hist and abs(t - self._hist[-1]["t"] - 10) > 1:
            self._hist.clear()
        self._hist.append({"t": t, "lat": ev["latitude"], "lon": ev["longitude"],
                           "alt": ev["altitude_km"], "vel": ev["velocity_kmh"]})
        if len(self._hist) < self.L:
            return None
        h = list(self._hist)
        U = latlon_to_unit(np.array([r["lat"] for r in h]), np.array([r["lon"] for r in h]))
        x = _lag_features(U, np.array([r["alt"] for r in h]), np.array([r["vel"] for r in h]))[None, :]
        u, alt, vel = self.model.predict_absolute(x)
        lat, lon = unit_to_latlon(u[0])
        target_t = t + 10 * self.H
        pred = {"event_id": ev["event_id"], "issued_at": ev["timestamp"], "target_time": target_t,
                "horizon_sec": 10 * self.H, "pred_latitude": float(lat), "pred_longitude": float(lon),
                "pred_altitude_km": float(alt[0]), "pred_velocity_kmh": float(vel[0]), "status": "pending"}
        self._pending[target_t] = pred
        return pred

    def _score_pending(self, ev) -> Optional[dict]:
        now = ev["_event_time"]
        for target in [k for k in self._pending if k < now]:
            expired = self._pending.pop(target)
            expired["status"] = "target_missing"
            self.completed.append(expired)
        p = self._pending.pop(now, None)
        if p is None:
            return None
        if ev.get("is_anomaly") or ev.get("state_gate_flag") or ev["segment_id"] != self._segment:
            p["status"] = "target_not_comparable"
        else:
            u_p = latlon_to_unit(p["pred_latitude"], p["pred_longitude"])
            u_t = latlon_to_unit(ev["latitude"], ev["longitude"])
            p.update(status="scored", actual_latitude=ev["latitude"], actual_longitude=ev["longitude"],
                     pos_error_km=float(angular_distance_km(u_p, u_t)),
                     alt_error_km=float(abs(p["pred_altitude_km"] - ev["altitude_km"])),
                     vel_error_kmh=float(abs(p["pred_velocity_kmh"] - ev["velocity_kmh"])))
        self.completed.append(p)
        return p

    def drain_completed(self) -> List[dict]:
        out, self.completed = self.completed, []
        return out


ISSOrbitForecaster = OrbitForecaster


if __name__ == "__main__":
    from sources.train_models import build_feature_frame, chronological_split
    df = build_feature_frame(inject_faults=False)
    train, _, test = chronological_split(df)
    for H in (1, 6, 30):
        Xtr, Ytr, _ = build_supervised(train, H)
        Xte, Yte, _ = build_supervised(test, H)
        for name in ("persistence", "constant_velocity", "ridge"):
            m = OrbitForecaster(horizon=H, model=name).fit(Xtr, Ytr)
            print(f"H={H:>2} {name:<18}", forecast_metrics(m, Xte, Yte))