"""Generate thesis-quality figures for PCA and equivalence analysis.

Figures:
1. Box plots: 5 key metrics across 5 duty levels (representative scenarios)
2. TOST equivalence forest plots (5 panels by metric)
3. Spearman rho heatmap (21 scenarios x 5 metrics)
4. PCA scatter plots (within-STM32N6 + cross-platform)
5. NPU inference latency distribution across duty levels
6. CPU utilization vs duty level
7. Cross-platform box plots on key metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STATS_CSV = Path(__file__).resolve().parent.parent.parent / "logs" / "stats" / "all_flight_stats.csv"
INCLUDED_RUNS = [
    "logs/npu_v2_20260418_131423",   # NPU 0%
    "logs/npu_v2_20260418_185511",   # NPU 25%
    "logs/npu_v2_20260419_003555",   # NPU 50%
    "logs/npu_v2_20260419_061806",   # NPU 75%
    "logs/npu_v2_20260419_115740",   # NPU 100%
    "logs/pixhawk6c_20260420_014914",
    "logs/npu_v2_20260420_160601",   # S27
]
EXCLUDED_RUNS = []
OUTLIER_FILTER = lambda df: df

KEY_METRICS = [
    ("ekf_pos_vert_accuracy_mean", "EKF Vert. Accuracy", "m", 2.0),
    ("vertical_speed_mean", "Vert. Speed Mean", "m/s", 0.2),
    ("pitch_mean", "Pitch Mean", "deg", 2.0),
    ("roll_mean", "Roll Mean", "deg", 2.0),
    ("roll_err_rms", "Roll Error RMS", "deg", 2.0),
    ("pitch_max_abs", "Pitch Max Abs", "deg", 2.0),
    ("pos_drift_max", "Pos. Drift Max", "m", 3.0),
    ("pos_drift_std", "Pos. Drift Std", "m", 0.5),
    ("roll_std", "Roll Std", "deg", 2.0),
    ("vertical_speed_max_abs", "Vert. Speed Max", "m/s", 0.5),
    ("roll_max_abs", "Roll Max Abs", "deg", 2.0),
    ("ekf_vel_ratio_mean", "EKF Vel. Ratio", "", 0.5),
    ("ground_speed_mean", "Ground Speed Mean", "m/s", 1.0),
    ("xte_rms", "Cross-Track Error", "m", 0.5),
    ("pix_traj_rmse", "Pix. Traj. RMSE", "m", 1.0),
    ("vib_z_mean", "Vibration Z Mean", "m/s²", 0.1),
    ("vib_magnitude_max", "Vibration Max", "m/s²", 0.5),
]

DUTY_LEVELS = [0, 25, 50, 75, 100]
DUTY_COLORS = {0: "#2c3e50", 25: "#2980b9", 50: "#27ae60", 75: "#f39c12", 100: "#e74c3c"}

REPRESENTATIVE_SCENARIOS = [
    "S01_hover_calm",
    "S06_waypoint_calm",
    "S13_fast_altitude_changes",
    "S19_low_battery_failsafe",
]

SCENARIO_SHORT = {
    "S01_hover_calm": "S01 Hover",
    "S02_hover_moderate_wind": "S02 Mod. Wind",
    "S03_hover_strong_wind": "S03 Strong Wind",
    "S04_hover_gust": "S04 Gust",
    "S05_long_hover": "S05 Long Hover",
    "S06_waypoint_calm": "S06 Waypoint",
    "S07_waypoint_crosswind": "S07 WP Crosswind",
    "S08_complex_mission": "S08 Complex",
    "S09_takeoff_land_single": "S09 Takeoff/Land",
    "S11_rapid_attitude_changes": "S11 Rapid Att.",
    "S12_tight_orbit": "S12 Tight Orbit",
    "S13_fast_altitude_changes": "S13 Fast Alt.",
    "S14_hover_degraded_gps": "S14 Degraded GPS",
    "S15_mission_gps_dropout": "S15 GPS Dropout",
    "S18_rtl_during_mission": "S18 RTL",
    "S19_low_battery_failsafe": "S19 Low Battery",
    "S20_sensor_failure": "S20 Sensor Fail",
    "S21_mavlink_telemetry_integrity": "S21 MAVLink",
    "S23_hitl_loop_validation": "S23 HITL Loop",
    "S25_cross_platform_parity": "S25 Cross-Plat.",
    "S26_wind_gps_degradation": "S26 Wind+GPS",
}


def fig1_boxplots_by_duty(df: pd.DataFrame, out_dir: Path):
    """Box plots of 5 key metrics across duty levels for representative scenarios."""
    for scenario in REPRESENTATIVE_SCENARIOS:
        sdf = df[df["scenario"] == scenario]
        if len(sdf) == 0:
            continue

        n_metrics = len(KEY_METRICS)
        ncols = 5
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
        axes = axes.flatten()
        sc_short = SCENARIO_SHORT.get(scenario, scenario)

        for ax, (metric, label, unit, _) in zip(axes, KEY_METRICS):
            data = [sdf[sdf["npu_load"] == d][metric].dropna().values for d in DUTY_LEVELS]
            bp = ax.boxplot(data, tick_labels=[f"{d}%" for d in DUTY_LEVELS],
                           patch_artist=True, widths=0.6, medianprops=dict(color="black", linewidth=1.5))
            for patch, duty in zip(bp["boxes"], DUTY_LEVELS):
                patch.set_facecolor(DUTY_COLORS[duty])
                patch.set_alpha(0.7)
            ax.set_xlabel("NPU Duty")
            ax.set_ylabel(f"{unit}")
            ax.set_title(f"{label}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=0.3)

        for ax in axes[len(KEY_METRICS):]:
            ax.set_visible(False)
        fig.suptitle(f"{sc_short} — Key Metrics by NPU Duty Level", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = out_dir / f"fig1_boxplots_{scenario}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname.name}")


def fig2_tost_forest(tost_csv: Path, out_dir: Path,
                     title: str = "TOST Equivalence: NPU 0% vs 100%",
                     fname: str = "fig2_tost_forest.png"):
    """TOST equivalence forest plots — one panel per metric."""
    tost = pd.read_csv(tost_csv)
    if len(tost) == 0:
        print("  [skip] No TOST results")
        return

    metrics_in_data = list(tost["metric"].unique())
    n_metrics = len(metrics_in_data)
    ncols = min(n_metrics, 3)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Build metric info from data + KEY_METRICS lookup
    metric_info = {m: (label, unit, margin) for m, label, unit, margin in KEY_METRICS}
    metrics_ordered = [m for m in metrics_in_data if m in metric_info]
    # Also handle metrics not in KEY_METRICS (cross-platform)
    try:
        from flight_analyzer.equivalence_tests import CROSS_PLATFORM_METRICS
    except ImportError:
        from equivalence_tests import CROSS_PLATFORM_METRICS
    for m, unit, margin in CROSS_PLATFORM_METRICS:
        if m not in metric_info:
            label = m.replace("_", " ").title()
            metric_info[m] = (label, unit, margin)
    metrics_ordered = [m for m in metrics_in_data if m in metric_info]

    for ax, metric in zip(axes, metrics_ordered):
        label, unit, margin = metric_info[metric]
        mdf = tost[tost["metric"] == metric].copy()
        if len(mdf) == 0:
            continue
        # Override margin from data if available
        if "margin" in mdf.columns:
            margin = mdf["margin"].iloc[0]
        mdf = mdf.sort_values("scenario", ascending=False)

        scenarios = mdf["scenario"].values
        diffs = mdf["mean_diff"].values
        p_bh = mdf["p_tost_bh"].values
        n_a = mdf["n_a"].values
        n_b = mdf["n_b"].values

        # Approximate CI: mean_diff +/- t * SE
        ci_half = []
        for i, row in mdf.iterrows():
            se = np.sqrt(row["mean_diff"] ** 2 / max(row["n_a"] + row["n_b"] - 2, 1)) if row["mean_diff"] != 0 else margin * 0.1
            # Use actual data to compute SE
            ci_half.append(margin * 0.3)  # placeholder width for visual

        y = np.arange(len(scenarios))

        # Equivalence region
        ax.axvspan(-margin, margin, alpha=0.15, color="green", label=f"Equiv. margin (±{margin})")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(-margin, color="green", linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(margin, color="green", linewidth=1, linestyle="--", alpha=0.5)

        for i in range(len(scenarios)):
            color = "#2ecc71" if p_bh[i] < 0.05 else "#e74c3c"
            marker = "o" if p_bh[i] < 0.05 else "s"
            ax.plot(diffs[i], y[i], marker=marker, color=color, markersize=5, zorder=5)

        sc_labels = [SCENARIO_SHORT.get(s, s) for s in scenarios]
        ax.set_yticks(y)
        ax.set_yticklabels(sc_labels, fontsize=7)
        ax.set_xlabel(f"Mean Difference ({unit})")
        ax.set_title(f"{label}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="Equivalent (p<0.05)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=8, label="Inconclusive"),
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.15, label="Equivalence margin"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def fig3_spearman_heatmap(sp_csv: Path, out_dir: Path):
    """Spearman rho heatmap — scenarios x metrics."""
    sp = pd.read_csv(sp_csv)
    if len(sp) == 0:
        print("  [skip] No Spearman results")
        return

    pivot = sp.pivot(index="scenario", columns="metric", values="rho")
    # Order columns by KEY_METRICS order
    col_order = [m for m, _, _, _ in KEY_METRICS if m in pivot.columns]
    pivot = pivot[col_order]
    # Sort scenarios
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)

    metric_labels = {m: label for m, label, _, _ in KEY_METRICS}
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([metric_labels.get(m, m) for m in col_order], fontsize=7, rotation=45, ha="right")

    sc_labels = [SCENARIO_SHORT.get(s, s) for s in pivot.index]
    ax.set_yticks(range(len(sc_labels)))
    ax.set_yticklabels(sc_labels, fontsize=7)

    # Annotate
    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    ax.set_title("Spearman Rank Correlation: NPU Duty vs Flight Metrics", fontsize=11)
    fig.tight_layout()
    fname = out_dir / "fig3_spearman_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig4_pca_scatters(df: pd.DataFrame, out_dir: Path):
    """PCA scatter plots — within-STM32N6 (representative) + cross-platform."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "duration_s", "origin_dist_mean", "pos_drift_max", "pos_drift_std",
        "alt_error_mean", "alt_error_max",
        "roll_mean", "roll_std", "roll_max_abs",
        "pitch_mean", "pitch_std", "pitch_max_abs",
        "ground_speed_mean", "ground_speed_max",
        "vertical_speed_mean", "vertical_speed_max_abs",
        "roll_err_rms", "pitch_err_rms", "yaw_err_rms",
        "ekf_vel_ratio_mean", "ekf_vel_ratio_max",
        "ekf_pos_horiz_ratio_mean", "ekf_pos_horiz_accuracy_mean", "ekf_pos_vert_accuracy_mean",
        "vib_x_mean", "vib_y_mean", "vib_z_mean", "vib_magnitude_max",
        "clipping_total", "comm_drop_rate_max", "comm_errors_max",
    ]

    def run_pca_scatter(sdf, group_col, group_labels, colors, title, fname):
        features = [c for c in feature_cols if c in sdf.columns]
        X = sdf[features].copy()
        X = X.loc[:, X.std() > 1e-10]
        X = X.fillna(X.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for grp in sorted(sdf[group_col].unique()):
            mask = sdf[group_col].values == grp
            label = group_labels.get(grp, str(grp))
            ax.scatter(scores[mask, 0], scores[mask, 1],
                      c=colors.get(grp, "#999"), label=label,
                      alpha=0.6, s=30, edgecolors="none")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, markerscale=1.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname.name}")

    npu_v2 = df[df["board"] == "npu_v2"]
    npu_labels = {d: f"NPU {d}%" for d in DUTY_LEVELS}

    # Within-STM32N6 for representative scenarios
    for scenario in ["S01_hover_calm", "S06_waypoint_calm", "S19_low_battery_failsafe"]:
        sdf = npu_v2[npu_v2["scenario"] == scenario].reset_index(drop=True)
        sc_short = SCENARIO_SHORT.get(scenario, scenario)
        run_pca_scatter(sdf, "npu_load", npu_labels, DUTY_COLORS,
                       f"PCA — {sc_short} (npu_v2 only)",
                       out_dir / f"fig4_pca_within_{scenario}.png")

    # Cross-platform: both boards at NPU 0%
    baseline = df[df["npu_load"] == 0].reset_index(drop=True)
    board_labels = {"pixhawk6c": "Pixhawk 6C", "npu_v2": "STM32N6"}
    board_colors = {"pixhawk6c": "#e74c3c", "npu_v2": "#3498db"}
    run_pca_scatter(baseline, "board", board_labels, board_colors,
                   "PCA — Cross-Platform (NPU 0%, all scenarios)",
                   out_dir / "fig4_pca_cross_platform.png")


def fig5_npu_latency(df: pd.DataFrame, out_dir: Path):
    """NPU inference latency distribution across duty levels."""
    npu = df[(df["board"] == "npu_v2") & (df["npu_load"] > 0) & (df["npu_ms_mean"] > 0)]
    # Exclude the known outlier
    npu = npu[npu["npu_ms_mean"] < 50]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [npu[npu["npu_load"] == d]["npu_ms_mean"].dropna().values for d in DUTY_LEVELS[1:]]
    parts = ax.violinplot(data, positions=range(len(DUTY_LEVELS[1:])),
                          showmeans=True, showmedians=True, showextrema=False)

    for i, (pc, duty) in enumerate(zip(parts["bodies"], DUTY_LEVELS[1:])):
        pc.set_facecolor(DUTY_COLORS[duty])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    ax.set_xticks(range(len(DUTY_LEVELS[1:])))
    ax.set_xticklabels([f"{d}%" for d in DUTY_LEVELS[1:]])
    ax.set_xlabel("NPU Duty Level")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("NPU Inference Latency by Duty Level")
    ax.grid(axis="y", alpha=0.3)

    # Annotate means
    for i, duty in enumerate(DUTY_LEVELS[1:]):
        vals = npu[npu["npu_load"] == duty]["npu_ms_mean"]
        ax.text(i, vals.mean() + 0.1, f"{vals.mean():.2f} ms", ha="center", fontsize=8, color="black")

    fig.tight_layout()
    fname = out_dir / "fig5_npu_latency.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig6_cpu_utilization(out_dir: Path):
    """CPU utilization vs duty level — derived from per-flight cpu_load column."""
    import glob

    root = Path(__file__).resolve().parent.parent.parent
    runs = {
        0:   root / "logs/npu_v2_20260418_131423",
        25:  root / "logs/npu_v2_20260418_185511",
        50:  root / "logs/npu_v2_20260419_003555",
        75:  root / "logs/npu_v2_20260419_061806",
        100: root / "logs/npu_v2_20260419_115740",
    }

    duties, means, stds = [], [], []
    all_flight_means = {}  # duty -> list of per-flight means
    for duty in sorted(runs):
        csvs = sorted(glob.glob(str(runs[duty] / "**" / "*.csv"), recursive=True))
        flight_means = []
        for c in csvs:
            try:
                df = pd.read_csv(c, usecols=["cpu_load"])
                vals = df["cpu_load"].dropna()
                if len(vals) > 0:
                    flight_means.append(vals.mean())
            except Exception:
                continue
        if flight_means:
            arr = np.array(flight_means)
            duties.append(duty)
            means.append(arr.mean())
            stds.append(arr.std())
            all_flight_means[duty] = arr

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(duties, means, yerr=stds, fmt="o-", color="#2c3e50",
                capsize=5, markersize=8, linewidth=2, label="Mean ± SD (n=201 flights each)")

    # Overlay individual flight means as jittered dots
    for duty in duties:
        if duty in all_flight_means:
            jitter = np.random.default_rng(42).normal(0, 1.2, len(all_flight_means[duty]))
            ax.scatter(duty + jitter, all_flight_means[duty], alpha=0.08, s=6,
                       color="#2c3e50", edgecolors="none")

    # Linear fit
    z = np.polyfit(duties, means, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(-5, 105, 100)
    ax.plot(x_fit, p(x_fit), "--", color="#e74c3c", alpha=0.7,
            label=f"Linear fit (slope={z[0]:.1f}‰ per 1% duty)")

    ax.set_xlabel("NPU Duty Level (%)")
    ax.set_ylabel("PX4 Reported CPU Load (permille)")
    ax.set_title("PX4 Reported CPU Load vs NPU Duty Level")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 1050)

    # Add right y-axis in percent
    ax2 = ax.twinx()
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("CPU Load (%)")

    fig.tight_layout()
    fname = out_dir / "fig6_cpu_utilization.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig7_cross_platform_boxplots(df: pd.DataFrame, out_dir: Path):
    """Cross-platform box plots comparing Pixhawk 6C vs STM32N6 at NPU 0%."""
    baseline = df[df["npu_load"] == 0]

    # PCA-identified cross-platform discriminators
    compare_metrics = [
        ("roll_mean", "Roll Mean", "deg"),
        ("roll_max_abs", "Roll Max Abs", "deg"),
        ("ground_speed_max", "Ground Speed Max", "m/s"),
        ("ekf_vel_ratio_max", "EKF Vel. Ratio Max", ""),
        ("vib_z_mean", "Vibration Z Mean", "m/s²"),
    ]

    n_metrics = len(compare_metrics)
    ncols = min(n_metrics, 3)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()
    board_colors = {"pixhawk6c": "#e74c3c", "npu_v2": "#3498db"}

    for ax, (metric, label, unit) in zip(axes, compare_metrics):
        data_pix = baseline[baseline["board"] == "pixhawk6c"][metric].dropna().values
        data_n6 = baseline[baseline["board"] == "npu_v2"][metric].dropna().values

        bp = ax.boxplot([data_pix, data_n6], tick_labels=["Pixhawk 6C", "STM32N6"],
                       patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor(board_colors["pixhawk6c"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(board_colors["npu_v2"])
        bp["boxes"][1].set_alpha(0.7)

        ylabel = f"{label} ({unit})" if unit else label
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians
        for i, (vals, board) in enumerate([(data_pix, "Pix"), (data_n6, "N6")]):
            med = np.median(vals)
            ax.text(i + 1, med, f" {med:.3f}", va="bottom", fontsize=7, color="black")

    for ax in axes[len(compare_metrics):]:
        ax.set_visible(False)
    fig.suptitle("Cross-Platform Comparison: Pixhawk 6C vs STM32N6 (NPU 0%)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "fig7_cross_platform_boxplots.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def all_scenario_spearman_scatters(df: pd.DataFrame, sp_csv: Path, out_dir: Path):
    """Per-scenario Spearman scatter plots: metric vs duty with rho annotation."""
    sc_dir = out_dir / "per_scenario" / "spearman"
    sc_dir.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(sp_csv) if sp_csv.exists() else pd.DataFrame()

    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        if len(sdf) < 10:
            continue

        n_metrics = len(KEY_METRICS)
        ncols = 5
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
        axes = axes.flatten()
        sc_short = SCENARIO_SHORT.get(scenario, scenario)

        for ax, (metric, label, unit, _) in zip(axes, KEY_METRICS):
            vals = sdf[[metric, "npu_load"]].dropna()
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            for duty in DUTY_LEVELS:
                dv = vals[vals["npu_load"] == duty][metric].values
                jitter = np.random.default_rng(42).normal(0, 1.5, len(dv))
                ax.scatter(duty + jitter, dv, alpha=0.5, s=15,
                           color=DUTY_COLORS[duty], edgecolors="none")

            # Annotate Spearman rho/p from CSV
            if len(sp) > 0:
                row = sp[(sp["scenario"] == scenario) & (sp["metric"] == metric)]
                if len(row) == 1:
                    rho = row.iloc[0]["rho"]
                    p_bh = row.iloc[0]["p_spearman_bh"]
                    sig = "*" if p_bh < 0.05 else ""
                    color = "#e74c3c" if p_bh < 0.05 else "#7f8c8d"
                    ax.text(0.97, 0.97, f"rho={rho:+.2f}{sig}\np_BH={p_bh:.3f}",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7,
                            color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            ax.set_xlabel("NPU Duty (%)")
            ax.set_ylabel(unit)
            ax.set_title(label, fontsize=9)
            ax.set_xticks(DUTY_LEVELS)
            ax.grid(axis="y", alpha=0.3)

        for ax in axes[n_metrics:]:
            ax.set_visible(False)
        fig.suptitle(f"{sc_short} — Spearman: Metric vs NPU Duty", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = sc_dir / f"spearman_{scenario}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {len(df['scenario'].unique())} plots saved to per_scenario/spearman/")


def all_scenario_boxplots(df: pd.DataFrame, out_dir: Path):
    """Box plots for every scenario, saved to a subdirectory."""
    sc_dir = out_dir / "per_scenario" / "boxplots"
    sc_dir.mkdir(parents=True, exist_ok=True)
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        if len(sdf) < 10:
            continue

        n_metrics = len(KEY_METRICS)
        ncols = 5
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
        axes = axes.flatten()
        sc_short = SCENARIO_SHORT.get(scenario, scenario)

        for ax, (metric, label, unit, _) in zip(axes, KEY_METRICS):
            data = [sdf[sdf["npu_load"] == d][metric].dropna().values for d in DUTY_LEVELS]
            bp = ax.boxplot(data, tick_labels=[f"{d}%" for d in DUTY_LEVELS],
                           patch_artist=True, widths=0.6, medianprops=dict(color="black", linewidth=1.5))
            for patch, duty in zip(bp["boxes"], DUTY_LEVELS):
                patch.set_facecolor(DUTY_COLORS[duty])
                patch.set_alpha(0.7)
            ax.set_xlabel("NPU Duty")
            ax.set_ylabel(f"{unit}")
            ax.set_title(f"{label}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=0.3)

        for ax in axes[len(KEY_METRICS):]:
            ax.set_visible(False)
        fig.suptitle(f"{sc_short} — Key Metrics by NPU Duty Level", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(sc_dir / f"boxplot_{scenario}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  {len(list(sc_dir.glob('*.png')))} plots saved to {sc_dir.relative_to(out_dir)}/")


PCA_FLAGGED = {
    "S01_hover_calm": "roll_mean",
    "S02_hover_moderate_wind": "ekf_pos_vert_accuracy_mean",
    "S03_hover_strong_wind": "roll_err_rms",
    "S04_hover_gust": "ekf_pos_vert_accuracy_mean",
    "S07_waypoint_crosswind": "ekf_pos_vert_accuracy_mean",
    "S08_complex_mission": "pitch_max_abs",
    "S09_takeoff_land_single": "pos_drift_max",
    "S11_rapid_attitude_changes": "pos_drift_std",
    "S12_tight_orbit": "vertical_speed_mean",
    "S13_fast_altitude_changes": "roll_std",
    "S14_hover_degraded_gps": "origin_dist_mean",
    "S15_mission_gps_dropout": "vertical_speed_max_abs",
    "S18_rtl_during_mission": "roll_max_abs",
    "S19_low_battery_failsafe": "pitch_mean",
    "S20_sensor_failure": "vertical_speed_mean",
    "S23_hitl_loop_validation": "pitch_mean",
    "S25_cross_platform_parity": "ekf_vel_ratio_mean",
    "S26_wind_gps_degradation": "ground_speed_mean",
}


def all_scenario_tost_forests(tost_csv: Path, out_dir: Path):
    """Per-scenario TOST forest plots, saved to a subdirectory."""
    import pandas as pd
    tost = pd.read_csv(tost_csv)
    if len(tost) == 0:
        print("  [skip] No TOST results")
        return

    sc_dir = out_dir / "per_scenario" / "tost"
    sc_dir.mkdir(parents=True, exist_ok=True)

    metric_info = {m: (label, unit, margin) for m, label, unit, margin in KEY_METRICS}
    try:
        try:
            from flight_analyzer.equivalence_tests import CROSS_PLATFORM_METRICS
        except ImportError:
            from equivalence_tests import CROSS_PLATFORM_METRICS
        for m, unit, margin in CROSS_PLATFORM_METRICS:
            if m not in metric_info:
                metric_info[m] = (m.replace("_", " ").title(), unit, margin)
    except ImportError:
        pass

    for scenario in sorted(tost["scenario"].unique()):
        sdf = tost[tost["scenario"] == scenario]
        metrics = [m for m in sdf["metric"].unique() if m in metric_info]
        if not metrics:
            continue

        # Put PCA-flagged metric first
        flagged = PCA_FLAGGED.get(scenario)
        if flagged and flagged in metrics:
            metrics.remove(flagged)
            metrics.insert(0, flagged)

        fig, ax = plt.subplots(figsize=(8, max(3, len(metrics) * 0.6)))
        sc_short = SCENARIO_SHORT.get(scenario, scenario)
        y = np.arange(len(metrics))

        for i, metric in enumerate(metrics):
            row = sdf[sdf["metric"] == metric].iloc[0]
            label, unit, default_margin = metric_info[metric]
            margin = row["margin"] if "margin" in row.index else default_margin
            diff = row["mean_diff"]
            p = row["p_tost_bh"]
            is_flagged = metric == flagged

            if p < 0.05:
                color = "#1a7a2e" if is_flagged else "#2ecc71"
            else:
                color = "#c0392b" if is_flagged else "#e74c3c"
            marker = "D" if is_flagged else ("o" if p < 0.05 else "s")
            size = 10 if is_flagged else 6

            if i == 0:
                ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

            ax.plot(diff, y[i], marker=marker, color=color, markersize=size, zorder=5)
            bar_alpha = 0.4 if is_flagged else 0.2
            ax.plot([-margin, margin], [y[i], y[i]], color="green", alpha=bar_alpha,
                    linewidth=10 if is_flagged else 8, zorder=1)

        ylabels = []
        for m in metrics:
            label_text = metric_info[m][0]
            if m == flagged:
                label_text = f"* {label_text} (PCA)"
            ylabels.append(label_text)

        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8)
        for tick, m in zip(ax.get_yticklabels(), metrics):
            if m == flagged:
                tick.set_fontweight("bold")
        ax.set_xlabel("Mean Difference")
        ax.set_title(f"TOST — {sc_short}", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#1a7a2e", markersize=9,
                   label="PCA-flagged (equiv.)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7,
                   label="Other (equiv.)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=7,
                   label="Inconclusive"),
            plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.3, label="Margin"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

        fig.tight_layout()
        fig.savefig(sc_dir / f"tost_{scenario}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {len(list(sc_dir.glob('*.png')))} plots saved to {sc_dir.relative_to(out_dir)}/")


def fig9_pca_focused_jt(jt_csv: Path, out_dir: Path):
    """PCA-focused Jonckheere-Terpstra forest plot — one test per scenario."""
    from matplotlib.lines import Line2D

    jt = pd.read_csv(jt_csv)
    if len(jt) == 0:
        print("  [skip] No PCA-focused JT results")
        return

    jt = jt.sort_values("scenario", ascending=False)
    scenarios = jt["scenario"].values
    z_vals = jt["jt_z"].values
    p_bh = jt["p_jt_bh"].values
    metrics = jt["metric"].values

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(scenarios))

    # Significance threshold lines
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvspan(-1.96, 1.96, alpha=0.06, color="green", label="No trend (|z| < 1.96)")

    for i in range(len(scenarios)):
        sig = p_bh[i] < 0.05
        color = "#e74c3c" if sig else "#2ecc71"
        marker = "D" if sig else "o"
        size = 9 if sig else 6
        ax.plot(z_vals[i], y[i], marker=marker, color=color, markersize=size,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    sc_labels = []
    for s, m in zip(scenarios, metrics):
        short = SCENARIO_SHORT.get(s, s)
        m_label = m.replace("_", " ")
        sc_labels.append(f"{short}  [{m_label}]")

    ax.set_yticks(y)
    ax.set_yticklabels(sc_labels, fontsize=7.5)
    for tick, p in zip(ax.get_yticklabels(), p_bh):
        if p < 0.05:
            tick.set_fontweight("bold")

    ax.set_xlabel("JT z-statistic", fontsize=10)
    ax.set_title("Jonckheere-Terpstra Trend Test — PCA-Flagged Metric per Scenario", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e74c3c", markersize=9,
               markeredgecolor="white", label=f"Monotonic trend (p_BH < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7,
               markeredgecolor="white", label="No trend"),
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.06, label="|z| < 1.96 region"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    fname = out_dir / "fig9_pca_focused_jt.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig10_pca_focused_spearman(sp_csv: Path, out_dir: Path):
    """PCA-focused Spearman rho lollipop plot — one test per scenario."""
    from matplotlib.lines import Line2D

    sp = pd.read_csv(sp_csv)
    if len(sp) == 0:
        print("  [skip] No PCA-focused Spearman results")
        return

    sp = sp.sort_values("scenario", ascending=False)
    scenarios = sp["scenario"].values
    rho_vals = sp["rho"].values
    p_bh = sp["p_spearman_bh"].values
    metrics = sp["metric"].values

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(scenarios))

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    for i in range(len(scenarios)):
        sig = p_bh[i] < 0.05
        color = "#e74c3c" if sig else "#2ecc71"
        # Lollipop: stem from 0 to rho
        ax.plot([0, rho_vals[i]], [y[i], y[i]], color=color, linewidth=2, alpha=0.7)
        marker = "D" if sig else "o"
        size = 9 if sig else 6
        ax.plot(rho_vals[i], y[i], marker=marker, color=color, markersize=size,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    sc_labels = []
    for s, m in zip(scenarios, metrics):
        short = SCENARIO_SHORT.get(s, s)
        m_label = m.replace("_", " ")
        sc_labels.append(f"{short}  [{m_label}]")

    ax.set_yticks(y)
    ax.set_yticklabels(sc_labels, fontsize=7.5)
    for tick, p in zip(ax.get_yticklabels(), p_bh):
        if p < 0.05:
            tick.set_fontweight("bold")

    ax.set_xlabel("Spearman ρ (duty level vs metric)", fontsize=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("Spearman Rank Correlation — PCA-Flagged Metric per Scenario", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e74c3c", markersize=9,
               markeredgecolor="white", label=f"Significant (p_BH < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7,
               markeredgecolor="white", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    fname = out_dir / "fig10_pca_focused_spearman.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


CROSS_PLATFORM_METRICS = [
    ("roll_mean", "Roll Mean", "deg", 2.0),
    ("roll_max_abs", "Roll Max Abs", "deg", 2.0),
    ("ground_speed_max", "Ground Speed Max", "m/s", 1.0),
    ("ekf_vel_ratio_max", "EKF Vel. Ratio Max", "", 0.5),
    ("vib_z_mean", "Vibration Z Mean", "m/s²", 1.0),
]

BOARD_COLORS = {"pixhawk6c": "#e74c3c", "npu_v2": "#3498db"}


def all_scenario_cross_platform_boxplots(df: pd.DataFrame, out_dir: Path):
    """Per-scenario box plots comparing Pixhawk 6C vs STM32N6 at NPU 0%."""
    baseline = df[df["npu_load"] == 0]
    sc_dir = out_dir / "per_scenario" / "cross_platform_boxplots"
    sc_dir.mkdir(parents=True, exist_ok=True)

    try:
        from flight_analyzer.equivalence_tests import PCA_FLAGGED_BOARDS
    except ImportError:
        from equivalence_tests import PCA_FLAGGED_BOARDS

    for scenario in sorted(baseline["scenario"].unique()):
        sdf = baseline[baseline["scenario"] == scenario]
        n_pix = (sdf["board"] == "pixhawk6c").sum()
        n_n6 = (sdf["board"] == "npu_v2").sum()
        if n_pix < 2 or n_n6 < 2:
            continue

        n_metrics = len(CROSS_PLATFORM_METRICS)
        ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes = axes.flatten()
        sc_short = SCENARIO_SHORT.get(scenario, scenario)
        flagged = PCA_FLAGGED_BOARDS.get(scenario)

        for ax, (metric, label, unit, _margin) in zip(axes, CROSS_PLATFORM_METRICS):
            data_pix = sdf[sdf["board"] == "pixhawk6c"][metric].dropna().values
            data_n6 = sdf[sdf["board"] == "npu_v2"][metric].dropna().values

            bp = ax.boxplot([data_pix, data_n6], tick_labels=["Pixhawk 6C", "STM32N6"],
                           patch_artist=True, widths=0.5,
                           medianprops=dict(color="black", linewidth=1.5))
            bp["boxes"][0].set_facecolor(BOARD_COLORS["pixhawk6c"])
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor(BOARD_COLORS["npu_v2"])
            bp["boxes"][1].set_alpha(0.7)

            ylabel = f"{unit}" if unit else ""
            ax.set_ylabel(ylabel, fontsize=9)
            title_text = f"* {label} (PCA)" if metric == flagged else label
            ax.set_title(title_text, fontsize=9,
                        fontweight="bold" if metric == flagged else "normal")
            ax.grid(axis="y", alpha=0.3)

            # Annotate medians
            for i, vals in enumerate([data_pix, data_n6]):
                if len(vals) > 0:
                    med = np.median(vals)
                    ax.text(i + 1, med, f" {med:.3f}", va="bottom", fontsize=7)

        for ax in axes[n_metrics:]:
            ax.set_visible(False)
        fig.suptitle(f"{sc_short} — Cross-Platform (NPU 0%)\n"
                     f"Pixhawk 6C (n={n_pix}) vs STM32N6 (n={n_n6})",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(sc_dir / f"xp_boxplot_{scenario}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {len(list(sc_dir.glob('*.png')))} plots saved to {sc_dir.relative_to(out_dir)}/")


def all_scenario_cross_platform_tost_forests(tost_csv: Path, out_dir: Path):
    """Per-scenario TOST forest plots for cross-platform comparison."""
    tost = pd.read_csv(tost_csv)
    if len(tost) == 0:
        print("  [skip] No cross-platform TOST results")
        return

    sc_dir = out_dir / "per_scenario" / "cross_platform_tost"
    sc_dir.mkdir(parents=True, exist_ok=True)

    try:
        from flight_analyzer.equivalence_tests import PCA_FLAGGED_BOARDS
    except ImportError:
        from equivalence_tests import PCA_FLAGGED_BOARDS
    metric_info = {m: (label, unit, margin) for m, label, unit, margin in CROSS_PLATFORM_METRICS}

    for scenario in sorted(tost["scenario"].unique()):
        sdf = tost[tost["scenario"] == scenario]
        metrics = [m for m in sdf["metric"].unique() if m in metric_info]
        if not metrics:
            continue

        flagged = PCA_FLAGGED_BOARDS.get(scenario)
        if flagged and flagged in metrics:
            metrics.remove(flagged)
            metrics.insert(0, flagged)

        fig, ax = plt.subplots(figsize=(8, max(3, len(metrics) * 0.6)))
        sc_short = SCENARIO_SHORT.get(scenario, scenario)
        y = np.arange(len(metrics))

        for i, metric in enumerate(metrics):
            row = sdf[sdf["metric"] == metric].iloc[0]
            label, unit, default_margin = metric_info[metric]
            margin = row["margin"] if "margin" in row.index else default_margin
            diff = row["mean_diff"]
            p = row["p_tost_bh"]
            is_flagged = metric == flagged

            if p < 0.05:
                color = "#1a7a2e" if is_flagged else "#2ecc71"
            else:
                color = "#c0392b" if is_flagged else "#e74c3c"
            marker = "D" if is_flagged else ("o" if p < 0.05 else "s")
            size = 10 if is_flagged else 6

            if i == 0:
                ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

            ax.plot(diff, y[i], marker=marker, color=color, markersize=size, zorder=5)
            bar_alpha = 0.4 if is_flagged else 0.2
            ax.plot([-margin, margin], [y[i], y[i]], color="green", alpha=bar_alpha,
                    linewidth=10 if is_flagged else 8, zorder=1)

        ylabels = []
        for m in metrics:
            label_text = metric_info[m][0]
            if m == flagged:
                label_text = f"* {label_text} (PCA)"
            ylabels.append(label_text)

        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8)
        for tick, m in zip(ax.get_yticklabels(), metrics):
            if m == flagged:
                tick.set_fontweight("bold")
        ax.set_xlabel("Mean Difference (Pixhawk 6C → STM32N6)")
        ax.set_title(f"Cross-Platform TOST — {sc_short}", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#1a7a2e", markersize=9,
                   label="PCA-flagged (equiv.)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b", markersize=9,
                   label="PCA-flagged (not equiv.)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7,
                   label="Other (equiv.)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=7,
                   label="Not equivalent"),
            plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.3, label="Margin"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

        fig.tight_layout()
        fig.savefig(sc_dir / f"xp_tost_{scenario}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  {len(list(sc_dir.glob('*.png')))} plots saved to {sc_dir.relative_to(out_dir)}/")


def fig13_cross_platform_flight_quality_tost(tost_csv: Path, out_dir: Path):
    """TOST forest for all 14 flight-quality metrics between boards."""
    tost = pd.read_csv(tost_csv)
    if len(tost) == 0:
        print("  [skip] No flight-quality cross-platform TOST results")
        return

    metric_info = {m: (label, unit, margin) for m, label, unit, margin in KEY_METRICS}
    metrics_in_data = [m for m in tost["metric"].unique() if m in metric_info]

    n_metrics = len(metrics_in_data)
    ncols = min(n_metrics, 3)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, metric in zip(axes, metrics_in_data):
        label, unit, margin = metric_info[metric]
        mdf = tost[tost["metric"] == metric].copy()
        if len(mdf) == 0:
            continue
        if "margin" in mdf.columns:
            margin = mdf["margin"].iloc[0]
        mdf = mdf.sort_values("scenario", ascending=False)

        scenarios = mdf["scenario"].values
        diffs = mdf["mean_diff"].values
        p_bh = mdf["p_tost_bh"].values
        y = np.arange(len(scenarios))

        ax.axvspan(-margin, margin, alpha=0.15, color="green", label=f"±{margin}")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(-margin, color="green", linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(margin, color="green", linewidth=1, linestyle="--", alpha=0.5)

        for i in range(len(scenarios)):
            color = "#2ecc71" if p_bh[i] < 0.05 else "#e74c3c"
            marker = "o" if p_bh[i] < 0.05 else "s"
            ax.plot(diffs[i], y[i], marker=marker, color=color, markersize=5, zorder=5)

        sc_labels = [SCENARIO_SHORT.get(s, s) for s in scenarios]
        ax.set_yticks(y)
        ax.set_yticklabels(sc_labels, fontsize=7)
        ax.set_xlabel(f"Mean Diff ({unit})" if unit else "Mean Diff")
        n_eq = (p_bh < 0.05).sum()
        ax.set_title(f"{label} ({n_eq}/{len(p_bh)} equiv.)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8,
               label="Equivalent (p<0.05)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=8,
               label="Not equivalent"),
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.15, label="Equivalence margin"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Cross-Platform TOST: Pixhawk 6C vs STM32N6 (NPU 0%)\nFlight-Quality Metrics",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fname = out_dir / "fig13_cross_platform_flight_quality_tost.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig14_cross_platform_equivalence_heatmap(tost_csv: Path, mw_csv: Path, out_dir: Path):
    """Heatmap: rows = 14 metrics, columns = 20 scenarios, cells = TOST equiv / MW diff."""
    tost = pd.read_csv(tost_csv)
    mw = pd.read_csv(mw_csv)
    if len(tost) == 0:
        return

    metric_info = {m: label for m, label, _, _ in KEY_METRICS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, df_in, pcol, title, cmap_colors in [
        (ax1, tost, "p_tost_bh", "TOST Equivalence (green=equivalent)", ["#e74c3c", "#2ecc71"]),
        (ax2, mw, "p_mw_bh", "Mann-Whitney U (red=different)", ["#2ecc71", "#e74c3c"]),
    ]:
        # Build binary matrix: significant or not
        metric_order = [m for m, _, _, _ in KEY_METRICS if m in df_in["metric"].unique()]
        scenario_order = sorted(df_in["scenario"].unique())

        matrix = np.full((len(metric_order), len(scenario_order)), np.nan)
        for i, metric in enumerate(metric_order):
            for j, scenario in enumerate(scenario_order):
                row = df_in[(df_in["metric"] == metric) & (df_in["scenario"] == scenario)]
                if len(row) > 0:
                    matrix[i, j] = 1.0 if row.iloc[0][pcol] < 0.05 else 0.0

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_colors)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

        ax.set_yticks(range(len(metric_order)))
        ax.set_yticklabels([metric_info.get(m, m) for m in metric_order], fontsize=7)
        sc_labels = [SCENARIO_SHORT.get(s, s) for s in scenario_order]
        ax.set_xticks(range(len(scenario_order)))
        ax.set_xticklabels(sc_labels, fontsize=6, rotation=60, ha="right")
        ax.set_title(title, fontsize=10)

        # Annotate with counts
        for i in range(len(metric_order)):
            n_sig = int(np.nansum(matrix[i, :]))
            n_total = int(np.sum(~np.isnan(matrix[i, :])))
            ax.text(len(scenario_order), i, f" {n_sig}/{n_total}",
                    va="center", fontsize=7, color="black")

    fig.suptitle("Cross-Platform: Pixhawk 6C vs STM32N6 — Flight-Quality Metrics",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fname = out_dir / "fig14_cross_platform_equivalence_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig11_pca_focused_cross_platform_tost(tost_csv: Path, out_dir: Path):
    """PCA-focused cross-platform TOST forest — one PCA-flagged metric per scenario."""
    from matplotlib.lines import Line2D

    tost = pd.read_csv(tost_csv)
    if len(tost) == 0:
        print("  [skip] No PCA-focused cross-platform TOST results")
        return

    tost = tost.sort_values("scenario", ascending=False)
    scenarios = tost["scenario"].values
    diffs = tost["mean_diff"].values
    margins = tost["margin"].values
    p_bh = tost["p_tost_bh"].values
    metrics = tost["metric"].values

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(scenarios))

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    for i in range(len(scenarios)):
        margin = margins[i]
        equiv = p_bh[i] < 0.05
        color = "#2ecc71" if equiv else "#e74c3c"
        marker = "o" if equiv else "s"

        # Margin bar
        ax.plot([-margin, margin], [y[i], y[i]], color="green", alpha=0.2,
                linewidth=8, zorder=1)
        # Point
        ax.plot(diffs[i], y[i], marker=marker, color=color, markersize=7,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    sc_labels = []
    for s, m in zip(scenarios, metrics):
        short = SCENARIO_SHORT.get(s, s)
        m_label = m.replace("_", " ")
        sc_labels.append(f"{short}  [{m_label}]")

    ax.set_yticks(y)
    ax.set_yticklabels(sc_labels, fontsize=7.5)
    ax.set_xlabel("Mean Difference (Pixhawk 6C → STM32N6)")
    ax.set_title("Cross-Platform TOST — PCA-Flagged Metric per Scenario\n(Pixhawk 6C vs STM32N6 at NPU 0%)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8,
               label="Equivalent (p < 0.05)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=8,
               label="Not equivalent"),
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.2, label="Equivalence margin"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    fname = out_dir / "fig11_pca_focused_cross_platform_tost.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig12_discriminator_comparison(out_dir: Path):
    """Compare PCA-flagged discriminators: NPU duty vs Board per scenario.

    Shows which parameter axis each factor (NPU load, board) acts on,
    highlighting orthogonality.
    """
    try:
        from flight_analyzer.equivalence_tests import PCA_FLAGGED, PCA_FLAGGED_BOARDS
    except ImportError:
        from equivalence_tests import PCA_FLAGGED, PCA_FLAGGED_BOARDS

    scenarios = sorted(set(PCA_FLAGGED.keys()) & set(PCA_FLAGGED_BOARDS.keys()))

    # Collect unique metrics across both
    all_metrics = sorted(set(
        list(PCA_FLAGGED.values()) + [PCA_FLAGGED_BOARDS[s] for s in scenarios]
    ))
    metric_to_idx = {m: i for i, m in enumerate(all_metrics)}

    fig, ax = plt.subplots(figsize=(10, 7))

    y_npu = []
    y_board = []
    y_pos = np.arange(len(scenarios))

    for i, scenario in enumerate(scenarios):
        npu_m = PCA_FLAGGED[scenario]
        board_m = PCA_FLAGGED_BOARDS[scenario]
        npu_x = metric_to_idx[npu_m]
        board_x = metric_to_idx[board_m]

        same = npu_m == board_m

        ax.plot(npu_x, i, marker="^", color="#3498db", markersize=9 if not same else 12,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(board_x, i, marker="v", color="#e74c3c", markersize=9 if not same else 12,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)

        if not same:
            # Draw connecting line to show separation
            ax.plot([npu_x, board_x], [i, i], color="#95a5a6", linewidth=1,
                    linestyle="--", alpha=0.5, zorder=1)
        else:
            # Highlight overlap
            ax.plot(npu_x, i, marker="o", color="#f39c12", markersize=16, zorder=3,
                    alpha=0.3, markeredgecolor="none")

    sc_labels = [SCENARIO_SHORT.get(s, s) for s in scenarios]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sc_labels, fontsize=8)

    metric_labels = [m.replace("_", "\n") for m in all_metrics]
    ax.set_xticks(range(len(all_metrics)))
    ax.set_xticklabels(metric_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("PCA-Flagged Metric", fontsize=10)
    ax.set_title("PCA Discriminator Comparison: NPU Duty vs Cross-Platform\n"
                 "(which parameter axis each factor acts on)", fontsize=11)
    ax.grid(True, alpha=0.15)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#3498db", markersize=10,
               label="NPU duty discriminator"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c", markersize=10,
               label="Board discriminator"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12", markersize=12,
               alpha=0.4, label="Overlap (same metric)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.tight_layout()
    fname = out_dir / "fig12_discriminator_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig15_s27_validation(out_dir: Path):
    """S27 randomised validation: segment metrics by duty level (no grouping)."""
    seg_csv = Path(__file__).resolve().parent.parent.parent / "logs" / "plots" / "pca" / "s27_randomized" / "s27_segments.csv"
    if not seg_csv.exists():
        print("  [skip] S27 segments CSV not found")
        return

    df = pd.read_csv(seg_csv)
    metrics = [
        ("roll_mean", "Roll Mean (deg)"),
        ("roll_std", "Roll Std (deg)"),
        ("vertical_speed_mean", "Vert. Speed Mean (m/s)"),
        ("ekf_pos_vert_accuracy_mean", "EKF Vert. Accuracy (m)"),
        ("ground_speed_mean", "Ground Speed (m/s)"),
        ("vib_z_mean", "Vibration Z (m/s²)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        for duty in DUTY_LEVELS:
            sub = df[df["duty"] == duty]
            vals = sub[metric].dropna()
            jitter = np.random.default_rng(42).normal(0, 0.15, len(vals))
            x = duty + jitter
            ax.scatter(x, vals, alpha=0.35, s=12, color=DUTY_COLORS[duty],
                       label=f"{duty}%" if duty == 0 else None, edgecolors="none")
        # Add KW p-value
        groups = [df[df["duty"] == d][metric].dropna().values for d in DUTY_LEVELS]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 3:
            H, p = stats.kruskal(*groups)
            ax.text(0.97, 0.97, f"KW p={p:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        ax.set_xlabel("NPU Duty (%)")
        ax.set_ylabel(label)
        ax.set_xticks(DUTY_LEVELS)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("S27 Randomised Validation — Segment Metrics by Duty Level\n"
                 "(300 segments, 10 flights, fully randomised ordering)",
                 fontsize=11)
    fig.tight_layout()
    fname = out_dir / "fig15_s27_validation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig16_sesoi_sensitivity(out_dir: Path):
    """Bar chart showing equivalence rates at 0.5x, 1x, 2x margins."""
    multipliers = ["0.5×", "1.0×", "2.0×"]
    equiv = [335, 339, 340]
    total = 340
    pcts = [e / total * 100 for e in equiv]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(multipliers, pcts, color=["#e74c3c", "#27ae60", "#2980b9"],
                  width=0.5, edgecolor="black", linewidth=0.5)
    for bar, e, pct in zip(bars, equiv, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{e}/{total}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(96, 101)
    ax.set_ylabel("Equivalence Rate (%)")
    ax.set_xlabel("SESOI Margin Multiplier")
    ax.set_title("TOST Equivalence at Varied SESOI Margins\n(17 metrics × 20 scenarios)")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "fig16_sesoi_sensitivity.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig17_experimental_design(out_dir: Path):
    """Experimental design overview diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 22)
    ax.set_axis_off()

    scenarios = [
        "S01 Hover Calm", "S02 Mod. Wind", "S03 Strong Wind", "S04 Gust",
        "S06 Waypoint", "S07 WP Crosswind", "S08 Complex", "S09 Takeoff/Land",
        "S11 Rapid Att.", "S12 Tight Orbit", "S13 Fast Alt.",
        "S14 Degraded GPS", "S15 GPS Dropout",
        "S18 RTL", "S19 Low Battery", "S20 Sensor Fail",
        "S21 MAVLink", "S23 HITL Loop", "S25 Cross-Plat.", "S26 Wind+GPS",
    ]
    duties = ["0%", "25%", "50%", "75%", "100%"]
    boards = ["STM32N6\n(npu_v2)", "Pixhawk 6C"]

    # Column headers
    ax.text(0, 21.5, "Scenario", fontsize=9, fontweight="bold", va="center")
    for i, d in enumerate(duties):
        ax.text(1.5 + i * 0.8, 21.5, d, fontsize=8, ha="center", va="center",
                fontweight="bold", color=list(DUTY_COLORS.values())[i])
    ax.text(5.8, 21.5, "Pix 6C", fontsize=8, ha="center", va="center",
            fontweight="bold", color="#7f8c8d")

    # Grid
    for row, sc in enumerate(scenarios):
        y = 20.5 - row
        ax.text(0, y, sc, fontsize=7, va="center")
        for i in range(5):
            ax.add_patch(plt.Rectangle((1.2 + i * 0.8, y - 0.3), 0.6, 0.6,
                                        facecolor=list(DUTY_COLORS.values())[i],
                                        alpha=0.6, edgecolor="black", linewidth=0.3))
            ax.text(1.5 + i * 0.8, y, "10", fontsize=6, ha="center", va="center", color="white", fontweight="bold")
        # Pixhawk column
        ax.add_patch(plt.Rectangle((5.5, y - 0.3), 0.6, 0.6,
                                    facecolor="#bdc3c7", alpha=0.8, edgecolor="black", linewidth=0.3))
        ax.text(5.8, y, "10", fontsize=6, ha="center", va="center", fontweight="bold")

    # S05 row (excluded)
    y = 20.5 - len(scenarios)
    ax.text(0, y, "S05 Long Hover", fontsize=7, va="center", color="#95a5a6")
    for i in range(5):
        ax.add_patch(plt.Rectangle((1.2 + i * 0.8, y - 0.3), 0.6, 0.6,
                                    facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=0.3))
        ax.text(1.5 + i * 0.8, y, "1", fontsize=6, ha="center", va="center", color="#95a5a6")
    ax.add_patch(plt.Rectangle((5.5, y - 0.3), 0.6, 0.6,
                                facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=0.3))
    ax.text(5.8, y, "1", fontsize=6, ha="center", va="center", color="#95a5a6")
    ax.text(6.3, y, "(excluded)", fontsize=6, va="center", color="#95a5a6")

    # Totals
    y_tot = y - 1.2
    ax.text(0, y_tot, "Column total:", fontsize=8, fontweight="bold", va="center")
    for i in range(5):
        ax.text(1.5 + i * 0.8, y_tot, "201", fontsize=8, ha="center", va="center", fontweight="bold")
    ax.text(5.8, y_tot, "201", fontsize=8, ha="center", va="center", fontweight="bold")

    # Title and annotations
    ax.set_title("Experimental Design: 1,206 Flights\n"
                 "20 scenarios × 5 duty levels × 10 reps (STM32N6) + 20 scenarios × 10 reps (Pixhawk 6C)",
                 fontsize=11, pad=10)
    ax.text(3, y_tot - 1, "S27 randomised validation (10 flights, 300 segments) run separately",
            fontsize=8, ha="center", va="center", style="italic", color="#7f8c8d")

    fig.tight_layout()
    fname = out_dir / "fig17_experimental_design.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname.name}")


def fig18_allpairs_tost(df: pd.DataFrame, out_dir: Path):
    """All-pairs TOST equivalence heatmap — 5x5 duty level matrix."""
    from itertools import combinations
    from statsmodels.stats.weightstats import ttost_ind

    npu = df[(df["board"] == "npu_v2") & (df["scenario"] != "S05_long_hover")]
    duties = [0, 25, 50, 75, 100]
    pairs = list(combinations(duties, 2))
    scenarios = sorted(npu["scenario"].unique())
    metric_list = [(m, margin) for m, _, _, margin in KEY_METRICS]

    # Compute equivalence rate per pair
    pair_rates = {}
    for d_a, d_b in pairs:
        eq, total = 0, 0
        for scenario in scenarios:
            sdf = npu[npu["scenario"] == scenario]
            for metric, margin in metric_list:
                a = sdf[sdf["npu_load"] == d_a][metric].dropna().values
                b = sdf[sdf["npu_load"] == d_b][metric].dropna().values
                if len(a) < 2 or len(b) < 2:
                    continue
                p, _, _ = ttost_ind(a, b, -margin, margin)
                total += 1
                if p < 0.05:
                    eq += 1
        pair_rates[(d_a, d_b)] = (eq, total)
        pair_rates[(d_b, d_a)] = (eq, total)

    # Build matrix
    n = len(duties)
    matrix = np.full((n, n), np.nan)
    annot = [['' for _ in range(n)] for _ in range(n)]
    for i, d_a in enumerate(duties):
        for j, d_b in enumerate(duties):
            if i == j:
                matrix[i, j] = 100.0
                annot[i][j] = '—'
            elif (d_a, d_b) in pair_rates:
                eq, total = pair_rates[(d_a, d_b)]
                pct = eq / total * 100 if total > 0 else 0
                matrix[i, j] = pct
                annot[i][j] = f'{eq}/{total}\n({pct:.1f}%)'

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=98, vmax=100.1, aspect='equal')

    # Labels
    labels = [f'{d}%' for d in duties]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel('NPU Duty Level')
    ax.set_ylabel('NPU Duty Level')

    # Annotations
    for i in range(n):
        for j in range(n):
            ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=8,
                    color='black' if matrix[i, j] > 99.5 else 'black')

    ax.set_title('All-Pairs TOST Equivalence\n(17 metrics × 20 scenarios per pair)')
    fig.colorbar(im, ax=ax, label='Equivalence Rate (%)', shrink=0.8)
    fig.tight_layout()
    fname = out_dir / 'fig18_allpairs_tost.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {fname.name}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--csv", default=str(STATS_CSV))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if INCLUDED_RUNS:
        df = df[df["run_dir"].isin(INCLUDED_RUNS)]
    elif EXCLUDED_RUNS:
        df = df[~df["run_dir"].isin(EXCLUDED_RUNS)]
    df = OUTLIER_FILTER(df)
    out_dir = Path(args.output_dir) if args.output_dir else STATS_CSV.parent.parent / "plots" / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating thesis figures to {out_dir}/")
    print(f"Loaded {len(df)} flights\n")

    # Paths to pre-computed test results
    pca_dir = STATS_CSV.parent.parent / "plots" / "pca"
    tost_csv = pca_dir / "tost_results.csv"
    sp_csv = pca_dir / "spearman_results.csv"

    npu_v2 = df[df["board"] == "npu_v2"]

    print("Figure 1: Box plots by duty level (representative scenarios)")
    fig1_boxplots_by_duty(npu_v2, out_dir)

    print("\nFigure 2: TOST equivalence forest plots")
    if tost_csv.exists():
        fig2_tost_forest(tost_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py first")

    print("\nFigure 3: Spearman rho heatmap")
    if sp_csv.exists():
        fig3_spearman_heatmap(sp_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py first")

    print("\nFigure 4: PCA scatter plots")
    fig4_pca_scatters(df, out_dir)

    print("\nFigure 5: NPU inference latency")
    fig5_npu_latency(df, out_dir)

    print("\nFigure 6: CPU utilization vs duty level")
    fig6_cpu_utilization(out_dir)

    print("\nFigure 7: Cross-platform box plots")
    fig7_cross_platform_boxplots(df, out_dir)

    print("\nFigure 8: Cross-platform TOST forest plot")
    xp_tost_csv = pca_dir / "cross_platform_tost.csv"
    if xp_tost_csv.exists():
        fig2_tost_forest(xp_tost_csv, out_dir,
                         title="TOST Equivalence: Pixhawk 6C vs STM32N6 (NPU 0%)",
                         fname="fig8_cross_platform_tost_forest.png")
    else:
        print("  [skip] Run equivalence_tests.py --compare-boards first")

    print("\nFigure 9: PCA-focused JT trend plot")
    pca_jt_csv = pca_dir / "pca_focused_jt.csv"
    if pca_jt_csv.exists():
        fig9_pca_focused_jt(pca_jt_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --pca-focused first")

    print("\nFigure 10: PCA-focused Spearman correlation plot")
    pca_sp_csv = pca_dir / "pca_focused_spearman.csv"
    if pca_sp_csv.exists():
        fig10_pca_focused_spearman(pca_sp_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --pca-focused first")

    print("\nFigure 13: Cross-platform flight-quality TOST forest")
    fq_tost_csv = pca_dir / "cross_platform_flight_quality_tost.csv"
    if fq_tost_csv.exists():
        fig13_cross_platform_flight_quality_tost(fq_tost_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --compare-boards first")

    print("\nFigure 14: Cross-platform equivalence heatmap")
    fq_mw_csv = pca_dir / "cross_platform_flight_quality_mw.csv"
    if fq_tost_csv.exists() and fq_mw_csv.exists():
        fig14_cross_platform_equivalence_heatmap(fq_tost_csv, fq_mw_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --compare-boards first")

    print("\nFigure 11: PCA-focused cross-platform TOST")
    pca_xp_tost_csv = pca_dir / "pca_focused_cross_platform_tost.csv"
    if pca_xp_tost_csv.exists():
        fig11_pca_focused_cross_platform_tost(pca_xp_tost_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --pca-focused-boards first")

    print("\nFigure 12: Discriminator comparison (NPU duty vs Board)")
    fig12_discriminator_comparison(out_dir)

    print("\nFigure 18: All-pairs TOST equivalence")
    fig18_allpairs_tost(df, out_dir)

    print("\nFigure 15: S27 randomised validation")
    fig15_s27_validation(out_dir)

    print("\nFigure 16: SESOI margin sensitivity")
    fig16_sesoi_sensitivity(out_dir)

    print("\nFigure 17: Experimental design overview")
    fig17_experimental_design(out_dir)

    print("\nPer-scenario cross-platform box plots")
    all_scenario_cross_platform_boxplots(df, out_dir)

    print("\nPer-scenario cross-platform TOST forests")
    xp_tost_csv = pca_dir / "cross_platform_tost.csv"
    if xp_tost_csv.exists():
        all_scenario_cross_platform_tost_forests(xp_tost_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py --compare-boards first")

    print("\nPer-scenario box plots (all scenarios)")
    all_scenario_boxplots(npu_v2, out_dir)

    print("\nPer-scenario TOST forests (all scenarios)")
    if tost_csv.exists():
        all_scenario_tost_forests(tost_csv, out_dir)
    else:
        print("  [skip] Run equivalence_tests.py first")

    print("\nPer-scenario Spearman scatter plots")
    sp_csv = pca_dir / "spearman_results.csv"
    all_scenario_spearman_scatters(npu_v2, sp_csv, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
