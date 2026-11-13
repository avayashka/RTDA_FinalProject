import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import EARTH_RADIUS_KM, FIGURES_DIR, RAW_DATA_PATH, RESULTS_DIR
from sources.stream_generator import label_natural_discontinuities

MU_EARTH = 398600.4418
ISS_INCLINATION_DEG = 51.64


def load_clean_series(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"speed_kmph": "velocity_kmh", "Hemisphere": "hemisphere"})
    df["discontinuity"] = label_natural_discontinuities(df.rename(columns={"velocity_kmh": "speed_kmph"}))
    df["segment"] = np.cumsum(df["discontinuity"])
    return df


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    d = (x * x).sum()
    return np.array([1.0] + [(x[:-k] * x[k:]).sum() / d for k in range(1, max_lag + 1)])


def within_segment_acf(x: np.ndarray, seg: np.ndarray, lags) -> dict:
    out = {}
    for k in lags:
        m = seg[k:] == seg[:-k]
        out[int(k)] = {"acf": round(float(np.corrcoef(x[:-k][m], x[k:][m])[0, 1]), 4), "pairs": int(m.sum())}
    return out


def orbital_period_from_latitude(df: pd.DataFrame) -> dict:
    lat = np.radians(df["latitude"].to_numpy())
    s = np.clip(np.sin(lat) / np.sin(np.radians(ISS_INCLINATION_DEG)), -1, 1)
    ascending = np.r_[np.nan, np.diff(lat)] > 0
    u = np.where(ascending, np.arcsin(s), np.pi - np.arcsin(s))
    du = (np.r_[np.nan, np.diff(u)] + np.pi) % (2 * np.pi) - np.pi
    seg = df["segment"].to_numpy()
    same = np.r_[False, seg[1:] == seg[:-1]] & np.r_[False, ascending[1:] == ascending[:-1]]
    ok = same & (np.abs(s) < 0.9)
    periods = 2 * np.pi / np.abs(du[ok] / 10.0) / 60.0
    return {"median_min": round(float(np.median(periods)), 2),
            "p05_min": round(float(np.percentile(periods, 5)), 2),
            "p95_min": round(float(np.percentile(periods, 95)), 2),
            "n_samples": int(ok.sum()), "u": u}


def orbital_period_from_visviva(df: pd.DataFrame) -> dict:
    r = EARTH_RADIUS_KM + df["altitude_km"].to_numpy()
    v = df["velocity_kmh"].to_numpy() / 3600.0
    a = 1.0 / np.median(2.0 / r - v ** 2 / MU_EARTH)
    v_circ = np.sqrt(MU_EARTH / r) * 3600
    return {"semi_major_axis_km": round(float(a), 1),
            "period_min": round(float(2 * np.pi * np.sqrt(a ** 3 / MU_EARTH) / 60), 2),
            "speed_minus_circular_speed_kmh": {"mean": round(float((df["velocity_kmh"] - v_circ).mean()), 3),
                                               "std": round(float((df["velocity_kmh"] - v_circ).std()), 3)},
            "v_circ": v_circ}


def run_analysis(save_figures: bool = True) -> dict:
    df = load_clean_series()
    seg = df["segment"].to_numpy()
    seg_len = df.groupby("segment").size()
    lat_p = orbital_period_from_latitude(df)
    vv = orbital_period_from_visviva(df)
    df["u_deg"] = np.degrees(lat_p.pop("u"))
    v_circ = vv.pop("v_circ")

    period_samples = int(round(lat_p["median_min"] * 6))
    df["alt_orbit_mean"] = df["altitude_km"].rolling(period_samples, center=True, min_periods=period_samples // 2).mean()
    hours = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600
    slope_km_per_day = float(np.polyfit(hours, df["altitude_km"], 1)[0] * 24)

    lags = [1, 3, 6, 12, 30, 60, 90]
    summary = {
        "n_events": int(len(df)),
        "time_span": [df["timestamp"].iloc[0].isoformat(), df["timestamp"].iloc[-1].isoformat()],
        "sampling_interval_sec": df["timestamp"].diff().dt.total_seconds().value_counts().to_dict(),
        "natural_discontinuities": int(df["discontinuity"].sum()),
        "segments": {"count": int(seg_len.size), "median_len": float(seg_len.median()),
                     "p75_len": float(seg_len.quantile(0.75)), "max_len": int(seg_len.max())},
        "descriptive": df[["altitude_km", "velocity_kmh", "latitude"]].describe().round(3).to_dict(),
        "corr_altitude_speed": round(float(df["altitude_km"].corr(df["velocity_kmh"])), 4),
        "orbital_period_latitude_phase": lat_p,
        "orbital_period_vis_viva": vv,
        "naive_fft_dominant_period_min": None,
        "acf_whole_series": {},
        "acf_within_segment": {},
        "altitude_trend_km_per_day": round(slope_km_per_day, 3),
        "window_justification": {},
    }
    for col in ("altitude_km", "velocity_kmh", "latitude"):
        x = df[col].to_numpy()
        a = acf(x, 90)
        summary["acf_whole_series"][col] = {int(k): round(float(a[k]), 4) for k in lags}
        summary["acf_within_segment"][col] = within_segment_acf(x, seg, lags)
    f = np.abs(np.fft.rfft(df["altitude_km"] - df["altitude_km"].mean()))
    freqs = np.fft.rfftfreq(len(df), 10)
    summary["naive_fft_dominant_period_min"] = round(float(1 / freqs[f[1:].argmax() + 1] / 60), 1)
    ws = summary["acf_within_segment"]["altitude_km"]
    summary["window_justification"] = {
        "tumbling_60s": f"6 samples; altitude ACF(lag 6) = {ws[6]['acf']} -> samples are near-redundant, mean is stable; "
                        f"60 s = {60 / (lat_p['median_min'] * 60) * 100:.1f}% of an orbit so orbital shape is preserved",
        "sliding_5min": f"30 samples; ACF(lag 30) = {ws[30]['acf']} (speed {summary['acf_within_segment']['velocity_kmh'][30]['acf']}) "
                        f"-> still locally predictable; also ~ p75 segment length ({seg_len.quantile(0.75):.0f} samples)",
        "beyond_10min": f"ACF(lag 60) = {ws[60]['acf']} -> history older than 10 min carries little information",
        "trend_window": f"1 orbit = {period_samples} samples, removes the orbital oscillation for trend estimation",
    }

    if save_figures:
        _figures(df, v_circ, summary, seg_len)
    with open(RESULTS_DIR / "time_series_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def _figures(df, v_circ, summary, seg_len):
    t = df["timestamp"]
    disc = df["discontinuity"].to_numpy()

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for a, col, lab in zip(ax, ["altitude_km", "velocity_kmh", "latitude"], ["Altitude (km)", "Speed (km/h)", "Latitude (deg)"]):
        a.plot(t, df[col], lw=0.6, color="#3366aa")
        a.scatter(t[disc], df[col][disc], s=6, color="#cc3333", label="trajectory discontinuity", zorder=3)
        a.set_ylabel(lab)
    ax[0].plot(t, df["alt_orbit_mean"], color="#222222", lw=1.5, label="1-orbit rolling mean (trend)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("ISS telemetry over 24 h (10-s cadence)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "ts_overview.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    lags = list(summary["acf_within_segment"]["altitude_km"].keys())
    for col, c in zip(["altitude_km", "velocity_kmh", "latitude"], ["#3366aa", "#dd8833", "#339966"]):
        ax[0].plot(lags, [summary["acf_whole_series"][col][k] for k in lags], "o--", color=c, label=f"{col} (whole series)")
        ax[0].plot(lags, [summary["acf_within_segment"][col][k]["acf"] for k in lags], "o-", color=c, label=f"{col} (within segment)")
    ax[0].axvline(6, color="grey", ls=":"); ax[0].axvline(30, color="grey", ls=":")
    ax[0].set_xlabel("lag (samples of 10 s)"); ax[0].set_ylabel("autocorrelation"); ax[0].legend(fontsize=7)
    ax[0].set_title("ACF: whole series vs within continuous segments")
    ax[1].hist(seg_len, bins=40, color="#3366aa")
    ax[1].set_xlabel("segment length (samples)"); ax[1].set_ylabel("count"); ax[1].set_title("Continuous segment lengths")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "acf_segments.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    order = np.argsort(df["altitude_km"].to_numpy())
    ax[0].scatter(df["altitude_km"], df["velocity_kmh"], s=3, alpha=0.4, label="observed")
    ax[0].plot(df["altitude_km"].to_numpy()[order], v_circ[order], color="#cc3333", label="circular orbit v = sqrt(mu/r)")
    ax[0].set_xlabel("altitude (km)"); ax[0].set_ylabel("speed (km/h)"); ax[0].legend()
    ax[0].set_title(f"Speed is set by altitude (corr = {summary['corr_altitude_speed']})")
    ax[1].scatter(df["u_deg"] % 360, df["altitude_km"], s=3, alpha=0.4)
    ax[1].set_xlabel("argument of latitude u (deg)"); ax[1].set_ylabel("altitude (km)")
    ax[1].set_title("Altitude vs orbital phase")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "orbit_physics.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["altitude_km"], s=2, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="altitude (km)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Ground track (24 h)")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "ground_track.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    s = run_analysis()
    print(json.dumps({k: s[k] for k in ("natural_discontinuities", "segments", "corr_altitude_speed",
                                        "orbital_period_latitude_phase", "orbital_period_vis_viva",
                                        "naive_fft_dominant_period_min", "altitude_trend_km_per_day",
                                        "window_justification")}, indent=2))
