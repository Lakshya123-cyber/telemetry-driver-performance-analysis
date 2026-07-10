"""
Telemetry-Based Driver Performance Analysis in Racing Simulation Using Assetto Corsa

Pipeline:
- Robust preprocessing (clipping, smoothing, lap-progress normalization)
- Reliable lap detection with boundary-lap trimming
- Exact sector time interpolation at sector boundaries
- Rich per-lap and per-sector feature extraction (shared helper, no duplication)
- Racing line deviation vs. best lap
- Strong/weak sector labeling
- Driver feedback generation (table-driven rules)
- Reproducible CSV outputs and plots
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================


@dataclass
class Config:
    csv_file: str = "ks_monza66_ks_ferrari_f2004_telemetria_20260705_123135.csv"
    output_dir: str = "output"

    num_sectors: int = 3
    min_samples_per_lap: int = 50

    # smoothing / cleaning
    smoothing_window: int = 5
    brake_threshold: float = 0.05
    throttle_threshold: float = 0.05

    # outlier clipping quantiles
    low_quantile: float = 0.005
    high_quantile: float = 0.995

    # racing line comparison
    interp_points_per_lap: int = 400

    # feedback thresholds
    time_loss_threshold_sec: float = 0.08
    avg_speed_diff_threshold_kmh: float = 2.0
    line_deviation_threshold_m: float = 2.5
    brake_ratio_diff_threshold: float = 0.05
    throttle_ratio_diff_threshold: float = 0.05
    corner_speed_diff_threshold_kmh: float = 3.0
    steering_smoothness_ratio_threshold: float = 1.15

    # plotting
    show_plots: bool = True


CFG = Config()


REQUIRED_COLUMNS = [
    "TrackName", "CarName", "X", "Y", "Z", "SpeedKMH", "RPM", "Throttle", "Brake", "Gear",
    "G_Lat", "G_Long", "LapProgress", "LapTime_ms", "CGHeight", "Steer",
]

RENAME_MAP = {
    "TrackName": "track_name", "CarName":"car_name", "X": "x", "Y": "y", "Z": "z", "SpeedKMH": "speed", "RPM": "rpm",
    "Throttle": "throttle", "Brake": "brake", "Gear": "gear",
    "G_Lat": "lat_g", "G_Long": "long_g", "LapProgress": "lap_progress",
    "LapTime_ms": "lap_time_ms", "CGHeight": "cg_height", "Steer": "steer",
}


# =========================================================
# UTILITIES
# =========================================================


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def robust_clip_series(s: pd.Series, low_q: float, high_q: float) -> pd.Series:
    return s.clip(lower=s.quantile(low_q), upper=s.quantile(high_q))


def rolling_median(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, center=True, min_periods=1).median()


def safe_std(s: pd.Series) -> float:
    return float(np.std(s, ddof=0)) if len(s) > 1 else 0.0


def usage_ratio(mask: pd.Series) -> float:
    """Fraction of samples where a condition is True."""
    return float(mask.mean()) if len(mask) else 0.0


def mean_abs_diff(s: pd.Series) -> float:
    return float(np.abs(np.diff(s.to_numpy())).mean()) if len(s) > 1 else 0.0


def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y)


def dedupe_sort_by_progress(progress: np.ndarray, *value_arrays: np.ndarray):
    """
    Sort all arrays by `progress`, drop duplicate progress values (keeping the
    last occurrence), and return progress plus each value array aligned to it.
    Sorting is done once and reused across all value arrays.
    """
    order = np.argsort(progress)
    progress_sorted = progress[order]
    sorted_values = [v[order] for v in value_arrays]

    unique_progress, unique_idx = np.unique(progress_sorted, return_index=True)
    unique_values = [v[unique_idx] for v in sorted_values]
    return (unique_progress, *unique_values)


# =========================================================
# LOAD + CLEAN
# =========================================================


def load_and_prepare_csv(csv_file: str) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found:\n{csv_file}")

    df = pd.read_csv(csv_file)
    validate_columns(df, REQUIRED_COLUMNS)
    df = df.rename(columns=RENAME_MAP).copy()

    print("CSV loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def normalize_lap_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize LapProgress to a 0-1 scale before any filtering."""
    df = df.copy()
    progress = df["lap_progress"].dropna()
    if progress.empty:
        raise ValueError("LapProgress column is empty after loading.")

    if progress.max() > 1.5:
        print("LapProgress appears to be 0-100. Converting to 0-1 scale.")
        df["lap_progress"] = df["lap_progress"] / 100.0

    return df


def clean_and_smooth(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=["x", "z", "speed", "lap_progress", "lap_time_ms"]).copy()
    df = df[(df["lap_progress"] >= -0.02) & (df["lap_progress"] <= 1.05)].copy()

    clip_cols = ["speed", "rpm", "throttle", "brake", "lat_g", "long_g", "steer"]
    for col in clip_cols:
        if col in df.columns:
            df[col] = robust_clip_series(df[col], cfg.low_quantile, cfg.high_quantile)

    smooth_cols = ["speed", "throttle", "brake", "lat_g", "long_g", "steer"]
    for col in smooth_cols:
        df[col] = rolling_median(df[col], cfg.smoothing_window)

    df["throttle"] = df["throttle"].clip(0, 1)
    df["brake"] = df["brake"].clip(0, 1)

    return df.reset_index(drop=True)


# =========================================================
# LAP DETECTION
# =========================================================


def detect_laps(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    lap_reset = df["lap_progress"].diff() < -0.5
    df["lap_number"] = lap_reset.cumsum() + 1

    lap_sizes = df.groupby("lap_number").size()
    valid_laps = lap_sizes[lap_sizes > cfg.min_samples_per_lap].index
    df = df[df["lap_number"].isin(valid_laps)].copy()

    if df.empty:
        raise ValueError("No laps left after initial lap-size filtering.")

    # Drop first/last laps: they are likely mid-lap starts/ends.
    for edge, label in [(df["lap_number"].min, "first"), (df["lap_number"].max, "last")]:
        if df["lap_number"].nunique() > 1:
            edge_lap = edge()
            print(f"Excluding Lap {edge_lap} because it likely {label}s mid-lap.")
            df = df[df["lap_number"] != edge_lap].copy()

    if df.empty:
        raise ValueError("No valid laps left after removing incomplete boundary laps.")

    lap_map = {old: new for new, old in enumerate(sorted(df["lap_number"].unique()), start=1)}
    df["lap_number"] = df["lap_number"].map(lap_map)

    return df.reset_index(drop=True)


# =========================================================
# GEOMETRY / DISTANCE / TIME
# =========================================================


def add_derived_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    df["x_plot"] = df["x"]
    df["y_plot"] = df["z"]  # AC track plane is X-Z, not X-Y

    df["lap_time_sec"] = df["lap_time_ms"] / 1000.0

    df["sector"] = (np.floor(df["lap_progress"] * cfg.num_sectors) + 1).astype(int)
    df["sector"] = df["sector"].clip(lower=1, upper=cfg.num_sectors)

    df["dx"] = df["x_plot"].diff().fillna(0)
    df["dy"] = df["y_plot"].diff().fillna(0)
    df["distance_step"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    lap_change = df["lap_number"].diff().fillna(0) != 0
    df.loc[lap_change, "distance_step"] = 0.0
    df["lap_distance"] = df.groupby("lap_number")["distance_step"].cumsum()

    df["combined_g"] = np.sqrt(df["lat_g"] ** 2 + df["long_g"] ** 2)
    df["abs_steer"] = df["steer"].abs()
    df["steer_delta_abs"] = df.groupby("lap_number")["steer"].diff().abs().fillna(0)

    return df


# =========================================================
# SHARED TELEMETRY SUMMARY (used by both lap and sector stats)
# =========================================================


def summarize_telemetry(segment: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """
    Compute the shared set of driving-behavior metrics for any slice of
    telemetry (a full lap or a single sector). Keeping this in one place
    avoids maintaining two near-identical dict-building blocks.
    """
    return {
        "avg_speed_kmh": float(segment["speed"].mean()),
        "max_speed_kmh": float(segment["speed"].max()),
        "speed_std_kmh": safe_std(segment["speed"]),
        "avg_throttle": float(segment["throttle"].mean()),
        "throttle_std": safe_std(segment["throttle"]),
        "throttle_usage_ratio": usage_ratio(segment["throttle"] > cfg.throttle_threshold),
        "avg_brake": float(segment["brake"].mean()),
        "brake_std": safe_std(segment["brake"]),
        "brake_usage_ratio": usage_ratio(segment["brake"] > cfg.brake_threshold),
        "avg_abs_steer": float(segment["abs_steer"].mean()),
        "steering_smoothness": mean_abs_diff(segment["steer"]),
        "max_lat_g": float(segment["lat_g"].max()),
        "min_lat_g": float(segment["lat_g"].min()),
        "max_long_g": float(segment["long_g"].max()),
        "min_long_g": float(segment["long_g"].min()),
        "avg_combined_g": float(segment["combined_g"].mean()),
        "max_combined_g": float(segment["combined_g"].max()),
        "samples": int(len(segment)),
    }


# =========================================================
# LAP-LEVEL STATS
# =========================================================


def compute_lap_stats(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    lap_rows = []

    for lap, lap_df in df.groupby("lap_number"):
        row = {
            "lap_number": lap,
            "lap_time_sec": float(lap_df["lap_time_sec"].max()),
            **summarize_telemetry(lap_df, cfg),
            "lap_distance": float(lap_df["lap_distance"].max()),
        }
        lap_rows.append(row)

    lap_stats = pd.DataFrame(lap_rows).sort_values("lap_number").reset_index(drop=True)
    if lap_stats.empty:
        raise ValueError("No valid laps available after filtering.")

    return lap_stats


# =========================================================
# EXACT SECTOR TIMING WITH INTERPOLATION
# =========================================================


def compute_sector_boundaries(num_sectors: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num_sectors + 1)


def _corner_phase_speeds(seg_sorted: pd.DataFrame) -> Tuple[float, float, float]:
    """Entry/apex/exit speed using the first/last 15% of samples as proxies."""
    n = len(seg_sorted)
    edge_n = max(1, int(n * 0.15))
    entry = float(seg_sorted["speed"].iloc[:edge_n].mean())
    apex = float(seg_sorted["speed"].min())
    exit_ = float(seg_sorted["speed"].iloc[-edge_n:].mean())
    return entry, apex, exit_


def sector_time_features_for_lap(lap_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    lap_df = lap_df.sort_values("lap_progress").copy()

    progress = lap_df["lap_progress"].to_numpy()
    elapsed = lap_df["lap_time_sec"].to_numpy()
    p_unique, t_unique = dedupe_sort_by_progress(progress, elapsed)

    if len(p_unique) < 2:
        return pd.DataFrame()

    boundaries = compute_sector_boundaries(cfg.num_sectors)
    boundary_times = interpolate_1d(p_unique, t_unique, boundaries)

    sector_rows = []
    for s in range(1, cfg.num_sectors + 1):
        start_p, end_p = boundaries[s - 1], boundaries[s]
        seg = lap_df[(lap_df["lap_progress"] >= start_p) & (lap_df["lap_progress"] <= end_p)]

        if seg.empty:
            continue

        seg_sorted = seg.sort_values("lap_progress")
        entry, apex, exit_ = _corner_phase_speeds(seg_sorted)

        row = {
            "lap_number": int(lap_df["lap_number"].iloc[0]),
            "sector": s,
            "sector_progress_start": start_p,
            "sector_progress_end": end_p,
            "sector_time_sec": float(boundary_times[s] - boundary_times[s - 1]),
            **summarize_telemetry(seg, cfg),
            "sector_distance": float(seg["distance_step"].sum()),
            "entry_speed_kmh": entry,
            "apex_speed_kmh": apex,
            "exit_speed_kmh": exit_,
        }
        sector_rows.append(row)

    return pd.DataFrame(sector_rows)


def compute_sector_stats(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sector_parts = [
        sector_time_features_for_lap(lap_df, cfg)
        for _, lap_df in df.groupby("lap_number")
    ]
    sector_parts = [s for s in sector_parts if not s.empty]

    if not sector_parts:
        raise ValueError("Unable to compute sector statistics.")

    sector_stats = pd.concat(sector_parts, ignore_index=True)
    return sector_stats.sort_values(["lap_number", "sector"]).reset_index(drop=True)


# =========================================================
# RACING LINE DEVIATION
# =========================================================


def resample_lap_xy(lap_df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    lap_df = lap_df.sort_values("lap_progress")

    progress = lap_df["lap_progress"].to_numpy()
    x_vals = lap_df["x_plot"].to_numpy()
    y_vals = lap_df["y_plot"].to_numpy()

    p_unique, x_unique, y_unique = dedupe_sort_by_progress(progress, x_vals, y_vals)
    if len(p_unique) < 2:
        return pd.DataFrame()

    grid = np.linspace(0.0, 1.0, n_points)
    return pd.DataFrame({
        "lap_progress": grid,
        "x_plot": interpolate_1d(p_unique, x_unique, grid),
        "y_plot": interpolate_1d(p_unique, y_unique, grid),
    })


def _deviation_from_best(lap_df: pd.DataFrame, best_resampled: pd.DataFrame, cfg: Config):
    """Returns (resampled_lap, deviation_array) or (None, None) if unavailable."""
    lap_resampled = resample_lap_xy(lap_df, cfg.interp_points_per_lap)
    if lap_resampled.empty:
        return None, None

    dx = lap_resampled["x_plot"].to_numpy() - best_resampled["x_plot"].to_numpy()
    dy = lap_resampled["y_plot"].to_numpy() - best_resampled["y_plot"].to_numpy()
    return lap_resampled, np.sqrt(dx**2 + dy**2)


def compute_racing_line_deviation(df: pd.DataFrame, best_lap_number: int, cfg: Config) -> pd.DataFrame:
    best_df = df[df["lap_number"] == best_lap_number]
    best_resampled = resample_lap_xy(best_df, cfg.interp_points_per_lap)
    if best_resampled.empty:
        raise ValueError("Best lap could not be resampled for racing line comparison.")

    rows = []
    for lap, lap_df in df.groupby("lap_number"):
        _, deviation = _deviation_from_best(lap_df, best_resampled, cfg)
        if deviation is None:
            continue
        rows.append({
            "lap_number": lap,
            "mean_line_deviation_m": float(deviation.mean()),
            "max_line_deviation_m": float(deviation.max()),
        })

    return pd.DataFrame(rows).sort_values("lap_number").reset_index(drop=True)


def compute_sector_line_deviation(df: pd.DataFrame, best_lap_number: int, cfg: Config) -> pd.DataFrame:
    best_df = df[df["lap_number"] == best_lap_number]
    best_resampled = resample_lap_xy(best_df, cfg.interp_points_per_lap)
    boundaries = compute_sector_boundaries(cfg.num_sectors)

    rows = []
    for lap, lap_df in df.groupby("lap_number"):
        lap_resampled, deviation = _deviation_from_best(lap_df, best_resampled, cfg)
        if deviation is None:
            continue

        progress = lap_resampled["lap_progress"].to_numpy()
        for s in range(1, cfg.num_sectors + 1):
            mask = (progress >= boundaries[s - 1]) & (progress <= boundaries[s])
            if not np.any(mask):
                continue
            rows.append({
                "lap_number": lap,
                "sector": s,
                "mean_line_deviation_m": float(deviation[mask].mean()),
                "max_line_deviation_m": float(deviation[mask].max()),
            })

    return pd.DataFrame(rows).sort_values(["lap_number", "sector"]).reset_index(drop=True)


# =========================================================
# COMPARISON TO BEST LAP
# =========================================================


def find_best_lap(lap_stats: pd.DataFrame) -> Tuple[int, float]:
    best_row = lap_stats.loc[lap_stats["lap_time_sec"].idxmin()]
    return int(best_row["lap_number"]), float(best_row["lap_time_sec"])


def add_best_lap_comparisons(
    lap_stats: pd.DataFrame,
    sector_stats: pd.DataFrame,
    line_dev_lap: pd.DataFrame,
    line_dev_sector: pd.DataFrame,
    best_lap_number: int,
    best_lap_time: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lap_stats = lap_stats.merge(line_dev_lap, on="lap_number", how="left")
    lap_stats["time_lost_vs_best_lap_sec"] = lap_stats["lap_time_sec"] - best_lap_time

    best_cols = [
        "sector", "sector_time_sec", "avg_speed_kmh", "avg_brake", "avg_throttle",
        "brake_usage_ratio", "throttle_usage_ratio",
        "entry_speed_kmh", "apex_speed_kmh", "exit_speed_kmh",
    ]
    rename_as_best = {
        "sector_time_sec": "best_sector_time_sec",
        "avg_speed_kmh": "best_sector_avg_speed_kmh",
        "avg_brake": "best_sector_avg_brake",
        "avg_throttle": "best_sector_avg_throttle",
        "brake_usage_ratio": "best_sector_brake_usage_ratio",
        "throttle_usage_ratio": "best_sector_throttle_usage_ratio",
        "entry_speed_kmh": "best_entry_speed_kmh",
        "apex_speed_kmh": "best_apex_speed_kmh",
        "exit_speed_kmh": "best_exit_speed_kmh",
    }
    best_sector = (
        sector_stats.loc[sector_stats["lap_number"] == best_lap_number, best_cols]
        .rename(columns=rename_as_best)
    )

    sector_stats = sector_stats.merge(best_sector, on="sector", how="left")
    sector_stats = sector_stats.merge(line_dev_sector, on=["lap_number", "sector"], how="left")

    sector_stats["time_lost_vs_best_sector_sec"] = (
        sector_stats["sector_time_sec"] - sector_stats["best_sector_time_sec"]
    )
    # e.g. avg_speed_kmh -> avg_speed_diff_vs_best, entry_speed_kmh -> entry_speed_diff_vs_best
    for phase, best_col in [
        ("avg_speed_kmh", "best_sector_avg_speed_kmh"),
        ("entry_speed_kmh", "best_entry_speed_kmh"),
        ("apex_speed_kmh", "best_apex_speed_kmh"),
        ("exit_speed_kmh", "best_exit_speed_kmh"),
    ]:
        diff_col = phase.replace("_kmh", "") + "_diff_vs_best"
        sector_stats[diff_col] = sector_stats[phase] - sector_stats[best_col]

    sector_stats["sector_quality"] = np.select(
        [
            sector_stats["time_lost_vs_best_sector_sec"] <= 0.03,
            sector_stats["time_lost_vs_best_sector_sec"] <= 0.10,
        ],
        ["strong", "average"],
        default="weak",
    )

    return lap_stats, sector_stats


# =========================================================
# FEEDBACK ENGINE (causal diagnosis, ranked by time impact)
# =========================================================
#
# Design goals, based on feedback that the flat symptom-list version was
# noisy and repetitive:
#   1. Group correlated symptoms into one root-cause diagnosis instead of
#      listing "lower entry speed" / "average speed lower" / "lost Xs" as
#      three unrelated bullets when they're really one braking issue.
#   2. Rank sectors within a lap by time lost, worst first, so the biggest
#      opportunity is always read first.
#   3. Don't bury a [strong] label under a pile of minor nitpicks — only
#      flag things on a strong sector if they're independently significant.
#   4. Add a one-line lap summary (total time lost + biggest opportunity)
#      so the driver isn't left to add up sector losses by hand.


def _diagnose_sector(row: pd.Series, cfg: Config) -> Tuple[List[str], float]:
    """
    Turn a sector-stat row into a short list of driver-facing diagnosis
    strings, plus the sector's time loss (for ranking). Correlated speed
    symptoms are collapsed into a single root-cause note rather than
    reported as separate, seemingly unrelated bullets.
    """
    time_lost = row["time_lost_vs_best_sector_sec"]
    has_time_loss = pd.notna(time_lost) and time_lost > cfg.time_loss_threshold_sec

    entry_down = pd.notna(row["entry_speed_diff_vs_best"]) and \
        row["entry_speed_diff_vs_best"] < -cfg.corner_speed_diff_threshold_kmh
    apex_down = pd.notna(row["apex_speed_diff_vs_best"]) and \
        row["apex_speed_diff_vs_best"] < -cfg.corner_speed_diff_threshold_kmh
    exit_down = pd.notna(row["exit_speed_diff_vs_best"]) and \
        row["exit_speed_diff_vs_best"] < -cfg.corner_speed_diff_threshold_kmh
    brakes_longer = (
        pd.notna(row["brake_usage_ratio"]) and pd.notna(row["best_sector_brake_usage_ratio"])
        and row["brake_usage_ratio"] > row["best_sector_brake_usage_ratio"] + cfg.brake_ratio_diff_threshold
    )
    throttle_less = (
        pd.notna(row["throttle_usage_ratio"]) and pd.notna(row["best_sector_throttle_usage_ratio"])
        and row["throttle_usage_ratio"] < row["best_sector_throttle_usage_ratio"] - cfg.throttle_ratio_diff_threshold
    )
    line_off = pd.notna(row["mean_line_deviation_m"]) and \
        row["mean_line_deviation_m"] > cfg.line_deviation_threshold_m

    notes: List[str] = []
    time_str = f" (-{time_lost:.2f}s)" if has_time_loss else ""

    # --- Root-cause grouping for the three speed-phase symptoms ---
    if entry_down and exit_down:
        # Slow in and slow out almost always means braking too early/hard
        # for this corner rather than three independent problems.
        cause = "braking too early or too hard into this sector" if brakes_longer else \
            "carrying too little speed into and out of this sector"
        notes.append(f"{cause}{time_str}")
    elif entry_down and not exit_down:
        notes.append(f"braking eats into entry speed, but you recover it by exit{time_str}")
    elif apex_down and not entry_down and not exit_down:
        notes.append(f"losing speed mid-corner (apex) despite a clean entry and exit{time_str}")
    elif exit_down and not entry_down:
        notes.append(f"good entry but throttle pickup on exit is late{time_str}")
    elif has_time_loss:
        # Time was lost but not clearly explained by entry/apex/exit — keep
        # it generic rather than inventing a cause.
        notes.append(f"time lost in this sector without a clear speed-phase pattern{time_str}")

    if throttle_less and not (entry_down or exit_down):
        notes.append("throttle applied for less of the sector than on the best lap")

    if line_off:
        notes.append(f"racing line drifts {row['mean_line_deviation_m']:.1f} m from the best lap on average")

    return notes, (time_lost if pd.notna(time_lost) else 0.0)


def _steering_note(row: pd.Series, best_smoothness: pd.Series, cfg: Config) -> str | None:
    ref = best_smoothness.get(int(row["sector"]))
    if (
        pd.notna(row["steering_smoothness"]) and row["steering_smoothness"] > 0
        and ref is not None
        and row["steering_smoothness"] > cfg.steering_smoothness_ratio_threshold * ref
    ):
        return "steering inputs are less smooth than on the best lap"
    return None


def generate_driver_feedback(
    sector_stats: pd.DataFrame, best_lap_number: int, cfg: Config, output_dir: Path
) -> None:
    best_smoothness = (
        sector_stats.loc[sector_stats["lap_number"] == best_lap_number]
        .set_index("sector")["steering_smoothness"]
    )

    lines = ["DRIVER FEEDBACK REPORT", "=" * 60, f"Reference lap: Lap {best_lap_number}", ""]

    for lap in sorted(sector_stats["lap_number"].unique()):
        if lap == best_lap_number:
            continue

        lap_df = sector_stats[sector_stats["lap_number"] == lap]
        total_lost = lap_df["time_lost_vs_best_sector_sec"].sum()

        lines.append(f"Lap {lap}  (total time lost vs best: {total_lost:.2f}s)")
        lines.append("-" * 60)

        # Diagnose every sector first, then print worst-time-loss first so
        # the biggest opportunity is always at the top of the lap block.
        diagnosed = []
        for _, row in lap_df.iterrows():
            notes, time_lost = _diagnose_sector(row, cfg)
            steer_note = _steering_note(row, best_smoothness, cfg)

            # On a sector already labeled "strong", don't pile on unless
            # there's something independently worth flagging (steering).
            if row["sector_quality"] == "strong" and not notes:
                if steer_note:
                    notes = [steer_note]
                else:
                    diagnosed.append((int(row["sector"]), row["sector_quality"], time_lost, []))
                    continue
            elif steer_note:
                notes.append(steer_note)

            diagnosed.append((int(row["sector"]), row["sector_quality"], time_lost, notes))

        diagnosed.sort(key=lambda d: d[2], reverse=True)

        any_feedback = False
        for sector, quality, time_lost, notes in diagnosed:
            if not notes:
                continue
            any_feedback = True
            lines.append(f"Sector {sector} [{quality}]: " + "; ".join(notes))

        if not any_feedback:
            lines.append("No major differences from best lap detected.")
        lines.append("")

    feedback_path = output_dir / "driver_feedback.txt"
    feedback_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {feedback_path}")


# =========================================================
# SAVE OUTPUTS
# =========================================================


def save_outputs(df: pd.DataFrame, lap_stats: pd.DataFrame, sector_stats: pd.DataFrame, output_dir: Path) -> None:
    outputs = {
        "processed_telemetry.csv": df,
        "lap_stats.csv": lap_stats,
        "sector_stats.csv": sector_stats,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")


# =========================================================
# PLOTS
# =========================================================


def plot_trajectory_by_lap(df: pd.DataFrame,
    track_name: str,
    car_name: str
    ) -> None:
    plt.figure(figsize=(10, 6))
    for lap in sorted(df["lap_number"].unique()):
        lap_df = df[df["lap_number"] == lap]
        plt.plot(lap_df["x_plot"], lap_df["y_plot"], linewidth=1, label=f"Lap {lap}")
        plt.scatter(lap_df["x_plot"].iloc[0], lap_df["y_plot"].iloc[0], s=20)
    plt.title(f"Track Trajectory by Lap\n{track_name} | {car_name}")
    plt.xlabel("Track X")
    plt.ylabel("Track Z")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_track_colored_by_speed(df: pd.DataFrame,
    track_name: str,
    car_name: str
    ) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["x_plot"], df["y_plot"], c=df["speed"], s=4)
    plt.title(f"Track Colored by Speed\n{track_name} | {car_name}")
    plt.xlabel("Track X")
    plt.ylabel("Track Z")
    plt.colorbar(scatter, label="Speed (km/h)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_speed_vs_progress(df: pd.DataFrame,
    track_name: str,
    car_name: str
    ) -> None:
    plt.figure(figsize=(10, 6))
    for lap in sorted(df["lap_number"].unique()):
        lap_df = df[df["lap_number"] == lap]
        plt.plot(lap_df["lap_progress"] * 100, lap_df["speed"], linewidth=1, label=f"Lap {lap}")
    plt.title(f"Speed vs Lap Progress\n{track_name} | {car_name}")
    plt.xlabel("Lap Progress (%)")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sector_time_loss(sector_stats: pd.DataFrame, best_lap_number: int, track_name: str, car_name: str) -> None:
    if sector_stats.empty:
        return

    plt.figure(figsize=(10, 6))
    for lap in sorted(sector_stats["lap_number"].unique()):
        lap_df = sector_stats[sector_stats["lap_number"] == lap]
        is_best = lap == best_lap_number
        plt.plot(
            lap_df["sector"],
            lap_df["time_lost_vs_best_sector_sec"],
            marker="o",
            label=f"Lap {lap} (best)" if is_best else f"Lap {lap}",
            linewidth=2.5 if is_best else 1,
            linestyle="--" if is_best else "-",
            color="black" if is_best else None,
            zorder=10 if is_best else 1,
        )
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.title(f"Sector Time Loss vs Best Lap\n{track_name} | {car_name}")
    plt.xlabel("Sector")
    plt.ylabel("Time Loss (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN PIPELINE
# =========================================================


def main(cfg: Config) -> None:
    output_dir = ensure_output_dir(cfg.output_dir)

    df = load_and_prepare_csv(cfg.csv_file)
    track_name = str(df["track_name"].iloc[0]).replace("ks_", "")
    car_name = str(df["car_name"].iloc[0]).replace("ks_", "")

    print(f"Track : {track_name}")
    print(f"Car   : {car_name}")

    df = normalize_lap_progress(df)
    df = clean_and_smooth(df, cfg)
    df = detect_laps(df, cfg)
    df = add_derived_columns(df, cfg)

    print("x range:", df["x_plot"].min(), "to", df["x_plot"].max())
    print("y range:", df["y_plot"].min(), "to", df["y_plot"].max())

    lap_stats = compute_lap_stats(df, cfg)
    sector_stats = compute_sector_stats(df, cfg)

    best_lap_number, best_lap_time = find_best_lap(lap_stats)

    line_dev_lap = compute_racing_line_deviation(df, best_lap_number, cfg)
    line_dev_sector = compute_sector_line_deviation(df, best_lap_number, cfg)

    lap_stats, sector_stats = add_best_lap_comparisons(
        lap_stats, sector_stats, line_dev_lap, line_dev_sector, best_lap_number, best_lap_time
    )

    print("\nLap stats:")
    print(lap_stats)
    print(f"\nBest lap: Lap {best_lap_number} ({best_lap_time:.3f} s)")

    print("\nSector stats:")
    print(sector_stats.head(20))

    generate_driver_feedback(sector_stats, best_lap_number, cfg, output_dir)
    save_outputs(df, lap_stats, sector_stats, output_dir)

    if cfg.show_plots:
        plot_trajectory_by_lap(df, track_name, car_name)
        plot_track_colored_by_speed(df, track_name, car_name)
        plot_speed_vs_progress(df, track_name, car_name)
        plot_sector_time_loss(sector_stats, best_lap_number, track_name, car_name)


if __name__ == "__main__":
    main(CFG)