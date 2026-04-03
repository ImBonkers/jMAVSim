"""Flight data analysis and statistics computation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FlightStats:
    """Summary statistics for a single flight CSV."""
    duration_s: float

    # Position hold quality (NED)
    pos_drift_mean: float  # mean 2D distance from hover center
    pos_drift_max: float
    pos_drift_std: float
    alt_error_mean: float
    alt_error_max: float

    # Attitude
    roll_mean: float
    roll_std: float
    roll_max_abs: float
    pitch_mean: float
    pitch_std: float
    pitch_max_abs: float

    # Velocity
    ground_speed_mean: float
    ground_speed_max: float
    vertical_speed_mean: float
    vertical_speed_max_abs: float

    # Control tracking error (setpoint - actual)
    roll_err_rms: float
    pitch_err_rms: float
    yaw_err_rms: float

    # EKF health
    ekf_vel_ratio_mean: float
    ekf_vel_ratio_max: float
    ekf_pos_horiz_ratio_mean: float
    ekf_pos_horiz_accuracy_mean: float
    ekf_pos_vert_accuracy_mean: float

    # Vibration
    vib_x_mean: float
    vib_y_mean: float
    vib_z_mean: float
    vib_magnitude_max: float
    clipping_total: int

    # NPU
    npu_ms_mean: float
    npu_fps_mean: float

    # Comms
    comm_drop_rate_max: float
    comm_errors_max: int


def compute_flight_stats(df: pd.DataFrame) -> FlightStats:
    """Compute summary statistics from a flight DataFrame."""
    dur = (df["elapsed_ms"].iloc[-1] - df["elapsed_ms"].iloc[0]) / 1000.0

    # Position drift from center of mass during flight
    armed_mask = df["armed"] == True  # noqa: E712
    if armed_mask.any():
        adf = df[armed_mask]
    else:
        adf = df

    drift_2d = np.sqrt(adf["x"] ** 2 + adf["y"] ** 2) if len(adf) > 0 else pd.Series([0.0])
    alt = adf["altitude"] if "altitude" in adf.columns else -adf["z"]

    def _safe(series, func, default=0.0):
        s = series.dropna()
        return func(s) if len(s) > 0 else default

    def _rms(series):
        s = series.dropna()
        return float(np.sqrt((s ** 2).mean())) if len(s) > 0 else 0.0

    vib_mag = np.sqrt(
        df.get("vibration_x", pd.Series([0.0])) ** 2 +
        df.get("vibration_y", pd.Series([0.0])) ** 2 +
        df.get("vibration_z", pd.Series([0.0])) ** 2
    )

    clip_cols = ["clipping_0", "clipping_1", "clipping_2"]
    clip_total = sum(int(_safe(df[c], lambda s: s.max(), 0)) for c in clip_cols if c in df.columns)

    return FlightStats(
        duration_s=dur,
        pos_drift_mean=_safe(drift_2d, lambda s: s.mean()),
        pos_drift_max=_safe(drift_2d, lambda s: s.max()),
        pos_drift_std=_safe(drift_2d, lambda s: s.std()),
        alt_error_mean=_safe(alt, lambda s: s.std()),
        alt_error_max=_safe(alt, lambda s: (s - s.mean()).abs().max()),

        roll_mean=_safe(adf["roll_deg"], lambda s: s.mean()),
        roll_std=_safe(adf["roll_deg"], lambda s: s.std()),
        roll_max_abs=_safe(adf["roll_deg"], lambda s: s.abs().max()),
        pitch_mean=_safe(adf["pitch_deg"], lambda s: s.mean()),
        pitch_std=_safe(adf["pitch_deg"], lambda s: s.std()),
        pitch_max_abs=_safe(adf["pitch_deg"], lambda s: s.abs().max()),

        ground_speed_mean=_safe(adf["ground_speed"], lambda s: s.mean()),
        ground_speed_max=_safe(adf["ground_speed"], lambda s: s.max()),
        vertical_speed_mean=_safe(adf["vertical_speed"], lambda s: s.mean()),
        vertical_speed_max_abs=_safe(adf["vertical_speed"], lambda s: s.abs().max()),

        roll_err_rms=_rms(df.get("roll_err_deg", pd.Series(dtype=float))),
        pitch_err_rms=_rms(df.get("pitch_err_deg", pd.Series(dtype=float))),
        yaw_err_rms=_rms(df.get("yaw_err_deg", pd.Series(dtype=float))),

        ekf_vel_ratio_mean=_safe(df.get("ekf_vel_ratio", pd.Series(dtype=float)), lambda s: s.mean()),
        ekf_vel_ratio_max=_safe(df.get("ekf_vel_ratio", pd.Series(dtype=float)), lambda s: s.max()),
        ekf_pos_horiz_ratio_mean=_safe(df.get("ekf_pos_horiz_ratio", pd.Series(dtype=float)), lambda s: s.mean()),
        ekf_pos_horiz_accuracy_mean=_safe(df.get("ekf_pos_horiz_accuracy", pd.Series(dtype=float)), lambda s: s.mean()),
        ekf_pos_vert_accuracy_mean=_safe(df.get("ekf_pos_vert_accuracy", pd.Series(dtype=float)), lambda s: s.mean()),

        vib_x_mean=_safe(df.get("vibration_x", pd.Series(dtype=float)), lambda s: s.mean()),
        vib_y_mean=_safe(df.get("vibration_y", pd.Series(dtype=float)), lambda s: s.mean()),
        vib_z_mean=_safe(df.get("vibration_z", pd.Series(dtype=float)), lambda s: s.mean()),
        vib_magnitude_max=_safe(vib_mag, lambda s: s.max()),
        clipping_total=clip_total,

        npu_ms_mean=_safe(df.get("npu_ms", pd.Series(dtype=float)), lambda s: s.mean()),
        npu_fps_mean=_safe(df.get("npu_fps", pd.Series(dtype=float)), lambda s: s.mean()),

        comm_drop_rate_max=_safe(df.get("comm_drop_rate", pd.Series(dtype=float)), lambda s: s.max()),
        comm_errors_max=int(_safe(df.get("comm_errors", pd.Series(dtype=float)), lambda s: s.max())),
    )


def compute_hover_quality(df: pd.DataFrame, hover_step_indices: Optional[list[int]] = None) -> dict:
    """Compute hover-specific quality metrics during hover steps."""
    if hover_step_indices:
        mask = df["step_index"].isin(hover_step_indices)
        hdf = df[mask]
    else:
        # Try to find hover steps by type
        if "step_type" in df.columns:
            hdf = df[df["step_type"] == "hover"]
        else:
            hdf = df

    if len(hdf) == 0:
        return {"hover_data": False}

    cx, cy = hdf["x"].mean(), hdf["y"].mean()
    drift = np.sqrt((hdf["x"] - cx) ** 2 + (hdf["y"] - cy) ** 2)
    alt_target = -hdf["z"].median()

    return {
        "hover_data": True,
        "center_x": float(cx),
        "center_y": float(cy),
        "drift_mean_m": float(drift.mean()),
        "drift_max_m": float(drift.max()),
        "drift_95pct_m": float(drift.quantile(0.95)),
        "alt_target_m": float(alt_target),
        "alt_std_m": float(hdf["altitude"].std()) if "altitude" in hdf.columns else float(hdf["z"].std()),
        "roll_std_deg": float(hdf["roll_deg"].std()),
        "pitch_std_deg": float(hdf["pitch_deg"].std()),
        "duration_s": float((hdf["elapsed_ms"].iloc[-1] - hdf["elapsed_ms"].iloc[0]) / 1000.0),
    }


def compare_runs(stats_list: list[tuple[str, FlightStats]]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple (label, FlightStats) pairs."""
    rows = []
    for label, stats in stats_list:
        row = {"label": label}
        for field_name in stats.__dataclass_fields__:
            row[field_name] = getattr(stats, field_name)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")
