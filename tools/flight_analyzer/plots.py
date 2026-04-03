"""Plotting functions for jMAVSim flight data visualization."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


STEP_COLORS = {
    "reboot": "#999999",
    "arm": "#2ecc71",
    "takeoff": "#3498db",
    "hover": "#9b59b6",
    "goto": "#e67e22",
    "orbit": "#e74c3c",
    "land": "#1abc9c",
    "disarm": "#95a5a6",
    "setWind": "#f39c12",
    "setNpuLoad": "#8e44ad",
    "wait": "#bdc3c7",
}


def _add_step_shading(ax, boundaries: list[dict], ymin=None, ymax=None):
    """Add colored vertical spans for each scenario step."""
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1]["elapsed_s"] if i + 1 < len(boundaries) else None
        if end is None:
            continue
        color = STEP_COLORS.get(b["step_type"], "#dddddd")
        ax.axvspan(b["elapsed_s"], end, alpha=0.08, color=color, zorder=0)
        mid = (b["elapsed_s"] + end) / 2
        if ymax is not None:
            ax.text(mid, ymax * 0.98, b["step_type"], fontsize=6, ha="center",
                    va="top", color=color, alpha=0.7, rotation=90)


def plot_flight_overview(df: pd.DataFrame, boundaries: list[dict],
                         title: str = "Flight Overview",
                         output: Optional[Path] = None, show: bool = False):
    """Generate a multi-panel flight overview plot.

    Panels: Position XY, Altitude, Attitude, Ground Speed,
            EKF ratios, Vibration, Control Error, NPU.
    """
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(8, 2, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.98, bottom=0.03)
    t = df["elapsed_s"]

    # --- 1. XY trajectory (top-left, square) ---
    ax_xy = fig.add_subplot(gs[0, 0])
    sc = ax_xy.scatter(df["x"], df["y"], c=t, cmap="viridis", s=1, alpha=0.7)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectory")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.plot(df["x"].iloc[0], df["y"].iloc[0], "go", ms=8, label="Start")
    ax_xy.plot(df["x"].iloc[-1], df["y"].iloc[-1], "rs", ms=8, label="End")
    ax_xy.legend(fontsize=7)
    plt.colorbar(sc, ax=ax_xy, label="Time (s)", shrink=0.8)

    # --- 2. 3D trajectory (top-right) ---
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.plot(df["x"], df["y"], df["altitude"] if "altitude" in df.columns else -df["z"],
               linewidth=0.5, alpha=0.8)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Alt")
    ax_3d.set_title("3D Flight Path")

    # --- Time series panels ---
    panel_configs = [
        (1, "Altitude & Vertical Speed", [
            ("altitude", "Alt (m)", "tab:blue"),
            ("vertical_speed", "Vz (m/s)", "tab:orange"),
        ]),
        (2, "Attitude", [
            ("roll_deg", "Roll (deg)", "tab:red"),
            ("pitch_deg", "Pitch (deg)", "tab:green"),
            ("yaw_deg", "Yaw (deg)", "tab:blue"),
        ]),
        (3, "Ground Speed & Velocity", [
            ("ground_speed", "GndSpd (m/s)", "tab:blue"),
            ("vx", "Vx", "tab:red"),
            ("vy", "Vy", "tab:green"),
        ]),
        (4, "EKF Ratios", [
            ("ekf_vel_ratio", "Vel", "tab:blue"),
            ("ekf_pos_horiz_ratio", "PosH", "tab:orange"),
            ("ekf_pos_vert_ratio", "PosV", "tab:green"),
            ("ekf_mag_ratio", "Mag", "tab:red"),
        ]),
        (5, "Vibration", [
            ("vibration_x", "Vib X", "tab:red"),
            ("vibration_y", "Vib Y", "tab:green"),
            ("vibration_z", "Vib Z", "tab:blue"),
        ]),
        (6, "Control Tracking Error", [
            ("roll_err_deg", "Roll err (deg)", "tab:red"),
            ("pitch_err_deg", "Pitch err (deg)", "tab:green"),
            ("yaw_err_deg", "Yaw err (deg)", "tab:blue"),
        ]),
        (7, "NPU & Comms", [
            ("npu_ms", "NPU latency (ms)", "tab:purple"),
            ("npu_fps", "NPU FPS", "tab:orange"),
            ("comm_drop_rate", "CommDrop (%)", "tab:red"),
        ]),
    ]

    for row, panel_title, traces in panel_configs:
        ax = fig.add_subplot(gs[row, :])
        has_data = False
        for col, label, color in traces:
            if col in df.columns and df[col].notna().any():
                ax.plot(t, df[col], label=label, color=color, linewidth=0.5, alpha=0.85)
                has_data = True
        if has_data:
            _add_step_shading(ax, boundaries)
            ax.legend(fontsize=7, ncol=len(traces), loc="upper right")
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_comparison(data: list[tuple[str, pd.DataFrame]], columns: list[str],
                    title: str = "Run Comparison",
                    output: Optional[Path] = None, show: bool = False):
    """Overlay time-series from multiple flights for comparison."""
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    cmap = plt.cm.tab10
    for col_idx, col in enumerate(columns):
        ax = axes[col_idx]
        for i, (label, df) in enumerate(data):
            if col in df.columns:
                ax.plot(df["elapsed_s"], df[col], label=label,
                        color=cmap(i % 10), linewidth=0.6, alpha=0.8)
        ax.set_ylabel(col)
        ax.legend(fontsize=7, ncol=min(len(data), 5))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_stats_comparison(stats_df: pd.DataFrame, metrics: Optional[list[str]] = None,
                          title: str = "Statistics Comparison",
                          output: Optional[Path] = None, show: bool = False):
    """Bar chart comparing statistics across runs."""
    if metrics is None:
        metrics = ["pos_drift_mean", "pos_drift_max", "roll_std", "pitch_std",
                    "ekf_vel_ratio_mean", "vib_magnitude_max", "npu_ms_mean"]
    metrics = [m for m in metrics if m in stats_df.columns]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = stats_df[metric]
        bars = ax.bar(range(len(vals)), vals, color=plt.cm.tab10(i % 10), alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(stats_df.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return fig
