"""Per-segment analysis for S27 randomized duty cycle flights.

Segments each flight CSV into 30 duty-level windows, discards settling
time, computes per-segment metrics, and runs statistical tests to assess
whether duty level or temporal order affects flight dynamics.

Run standalone:
    PYTHONPATH=tools tools/flight_analyzer/.venv/bin/python -m flight_analyzer.s27_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from flight_analyzer.analysis import FlightStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S27_RUN_DIR = Path("logs/npu_v2_20260420_160601")
SCENARIO_DIR = Path("scenarios")
SETTLE_SECONDS = 10  # discard first 10s of each segment
DUTY_LEVELS = [0, 25, 50, 75, 100]

# Metrics to compute per segment (subset of FlightStats)
SEGMENT_METRICS = [
    "roll_mean", "roll_std", "roll_max_abs",
    "pitch_mean", "pitch_std",
    "origin_dist_mean", "pos_drift_max", "pos_drift_std",
    "vertical_speed_mean", "vertical_speed_max_abs",
    "ground_speed_mean",
    "ekf_vel_ratio_mean", "ekf_pos_vert_accuracy_mean",
    "vib_z_mean",
]


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


def _compute_segment_metrics(df: pd.DataFrame) -> dict:
    """Compute metrics from a segment DataFrame (already filtered to
    measurement window only)."""
    def _safe(series, func, default=0.0):
        s = series.dropna()
        return float(func(s)) if len(s) > 0 else default

    # Hold quality: distance from target position (or mean if no target given)
    # target_pos is set externally via _compute_segment_metrics_with_target
    cx, cy = df["x"].mean(), df["y"].mean()
    hold_drift = np.sqrt((df["x"] - cx) ** 2 + (df["y"] - cy) ** 2)

    return {
        "roll_mean": _safe(df["roll_deg"], np.mean),
        "roll_std": _safe(df["roll_deg"], np.std),
        "roll_max_abs": _safe(df["roll_deg"], lambda s: np.max(np.abs(s))),
        "pitch_mean": _safe(df["pitch_deg"], np.mean),
        "pitch_std": _safe(df["pitch_deg"], np.std),
        "hold_rmse_m": _safe(hold_drift, lambda s: float(np.sqrt(np.mean(s**2)))),
        "hold_drift_max": _safe(hold_drift, np.max),
        "hold_drift_std": _safe(hold_drift, np.std),
        "vertical_speed_mean": _safe(df["vertical_speed"], np.mean),
        "vertical_speed_max_abs": _safe(df["vertical_speed"],
                                        lambda s: np.max(np.abs(s))),
        "ground_speed_mean": _safe(df["ground_speed"], np.mean),
        "ekf_vel_ratio_mean": _safe(df.get("ekf_vel_ratio",
                                            pd.Series(dtype=float)), np.mean),
        "ekf_pos_vert_accuracy_mean": _safe(
            df.get("ekf_pos_vert_accuracy", pd.Series(dtype=float)), np.mean),
        "vib_z_mean": _safe(df.get("vibration_z", pd.Series(dtype=float)),
                            np.mean),
        "n_samples": len(df),
    }


def extract_segments(csv_path: Path, scenario_path: Path,
                     settle_s: float = SETTLE_SECONDS) -> pd.DataFrame:
    """Extract per-segment metrics from a single S27 flight CSV.

    Returns a DataFrame with one row per segment, columns:
        flight, segment_index, duty, task, time_index, + all metrics
    """
    # Load scenario JSON for segment metadata
    with open(scenario_path) as f:
        scenario = json.load(f)
    seg_meta = scenario["segments"]

    # Load flight CSV
    df = pd.read_csv(csv_path)

    # Identify segment boundaries from step structure.
    # Pattern: setNpuLoad -> hover (for hover task)
    #          setNpuLoad -> goto -> hover (for goto task)
    # We measure during the LAST hover step of each segment.
    step_groups = df.groupby("step_index").agg(
        step_type=("step_type", "first"),
        t_start=("elapsed_ms", "min"),
        t_end=("elapsed_ms", "max"),
        n=("elapsed_ms", "count"),
    ).reset_index()

    # Find all setNpuLoad steps (these mark segment boundaries)
    npu_steps = step_groups[step_groups["step_type"] == "setNpuLoad"]
    npu_indices = npu_steps["step_index"].values

    # Get scenario steps for target extraction
    scenario_steps = scenario["steps"]

    rows = []
    for seg_i, npu_step_idx in enumerate(npu_indices):
        if seg_i >= len(seg_meta):
            break
        meta = seg_meta[seg_i]

        if meta["task"] == "goto":
            # Transit step is npu_step_idx + 1 (goto), hold step is npu_step_idx + 2 (hover)
            transit_step_idx = npu_step_idx + 1
            measure_step_idx = npu_step_idx + 2
        else:
            # Measurement is the hover right after setNpuLoad: step_index = npu_step_idx + 1
            transit_step_idx = None
            measure_step_idx = npu_step_idx + 1

        # Get the hold measurement data (after settling)
        seg_df = df[df["step_index"] == measure_step_idx].copy()
        if len(seg_df) == 0:
            continue

        # Discard settling time
        t0 = seg_df["elapsed_ms"].iloc[0]
        settle_cutoff = t0 + settle_s * 1000
        measure_df = seg_df[seg_df["elapsed_ms"] >= settle_cutoff]
        if len(measure_df) < 100:  # need at least 100 samples (~0.4s)
            continue

        metrics = _compute_segment_metrics(measure_df)

        # For goto segments: compute XTE during transit and hold RMSE from target
        if meta["task"] == "goto" and transit_step_idx is not None:
            # Get target from the goto step definition
            goto_step_def = scenario_steps[int(transit_step_idx)]
            if goto_step_def.get("type") == "goto":
                tx, ty, tz = goto_step_def["x"], goto_step_def["y"], goto_step_def["z"]

                # Hold RMSE from target (during settled hover at waypoint)
                hold_dist = np.sqrt(
                    (measure_df["x"].values - tx)**2 +
                    (measure_df["y"].values - ty)**2 +
                    (measure_df["z"].values - tz)**2
                )
                metrics["hold_rmse_m"] = float(np.sqrt(np.mean(hold_dist**2)))

                # XTE during transit phase
                transit_df = df[df["step_index"] == transit_step_idx]
                if len(transit_df) > 5:
                    # Previous position = position at start of transit
                    ax, ay = transit_df["x"].iloc[0], transit_df["y"].iloc[0]
                    bx, by = tx, ty
                    # Cross-track error: perpendicular distance to line a->b
                    ab = np.array([bx - ax, by - ay])
                    ab_len = np.linalg.norm(ab)
                    if ab_len > 0.1:
                        ab_unit = ab / ab_len
                        xte_vals = []
                        for _, row_data in transit_df.iterrows():
                            ap = np.array([row_data["x"] - ax, row_data["y"] - ay])
                            proj = np.dot(ap, ab_unit)
                            perp = ap - proj * ab_unit
                            xte_vals.append(np.linalg.norm(perp))
                        metrics["xte_rms"] = float(np.sqrt(np.mean(np.array(xte_vals)**2)))
                    else:
                        metrics["xte_rms"] = 0.0
                else:
                    metrics["xte_rms"] = np.nan
            else:
                metrics["xte_rms"] = np.nan
        else:
            metrics["xte_rms"] = np.nan

        metrics.update({
            "segment_index": seg_i,
            "duty": meta["duty"],
            "task": meta["task"],
            "time_index": seg_i,  # temporal order
            "duration_s": (measure_df["elapsed_ms"].iloc[-1] -
                           measure_df["elapsed_ms"].iloc[0]) / 1000.0,
        })
        rows.append(metrics)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load all S27 flights
# ---------------------------------------------------------------------------


def load_s27_segments(run_dir: Path = S27_RUN_DIR,
                      scenario_dir: Path = SCENARIO_DIR) -> pd.DataFrame:
    """Load and segment all 6 S27 flights. Returns combined DataFrame."""
    summary_path = run_dir / "test_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    all_segments = []
    for run in summary["runs"]:
        scenario_name = run["scenario"]
        flight_num = int(scenario_name.split("_f")[-1])
        csv_path = None
        for lf in run["logFiles"]:
            if lf.endswith(".csv"):
                csv_path = Path(lf)
                if not csv_path.exists():
                    csv_path = run_dir.parent / lf
                break
        if csv_path is None or not csv_path.exists():
            print(f"  Warning: no CSV for {scenario_name}")
            continue

        scenario_path = scenario_dir / f"{scenario_name}.json"
        if not scenario_path.exists():
            print(f"  Warning: no scenario JSON for {scenario_name}")
            continue

        seg_df = extract_segments(csv_path, scenario_path)
        seg_df["flight"] = flight_num
        all_segments.append(seg_df)
        print(f"  Flight {flight_num}: {len(seg_df)} segments extracted")

    combined = pd.concat(all_segments, ignore_index=True)
    print(f"\nTotal: {len(combined)} segments across {len(all_segments)} flights")
    return combined


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def test_temporal_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between segment time_index and each metric.

    If no temporal drift, correlations should be non-significant.
    """
    results = []
    for metric in SEGMENT_METRICS:
        if metric not in df.columns:
            continue
        vals = df[metric].dropna()
        times = df.loc[vals.index, "time_index"]
        if len(vals) < 10:
            continue
        rho, p = stats.spearmanr(times, vals)
        results.append({
            "metric": metric,
            "rho": round(rho, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "n": len(vals),
        })
    return pd.DataFrame(results)


def test_duty_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test across duty levels for each metric.

    If no duty effect, tests should be non-significant.
    """
    results = []
    for metric in SEGMENT_METRICS:
        if metric not in df.columns:
            continue
        groups = []
        for duty in DUTY_LEVELS:
            vals = df[df["duty"] == duty][metric].dropna().values
            if len(vals) >= 3:
                groups.append(vals)
        if len(groups) < 3:
            continue
        H, p = stats.kruskal(*groups)

        # Epsilon-squared effect size
        k = len(groups)
        N = sum(len(g) for g in groups)
        eps_sq = max(0.0, (H - k + 1) / (N - k)) if (N - k) > 0 else 0.0

        results.append({
            "metric": metric,
            "kruskal_H": round(H, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "epsilon_sq": round(eps_sq, 6),
            "n_groups": len(groups),
            "n_total": N,
        })
    return pd.DataFrame(results)


def test_duty_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between duty level and each metric.

    Same test as the main analysis broad Spearman — correlates duty against
    metric value across all individual segments.
    """
    results = []
    for metric in SEGMENT_METRICS:
        if metric not in df.columns:
            continue
        vals = df[metric].dropna()
        duties = df.loc[vals.index, "duty"]
        if len(vals) < 10:
            continue
        rho, p = stats.spearmanr(duties, vals)
        results.append({
            "metric": metric,
            "rho": round(rho, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "n": len(vals),
        })
    return pd.DataFrame(results)


def test_duty_tost(df: pd.DataFrame) -> pd.DataFrame:
    """TOST equivalence between NPU 0% and 100% segments.

    Uses the same SESOI margins as the main analysis.
    """
    from statsmodels.stats.weightstats import ttost_ind

    MARGINS = {
        "roll_mean": 2.0, "roll_std": 2.0, "roll_max_abs": 2.0,
        "pitch_mean": 2.0, "pitch_std": 2.0,
        "vertical_speed_mean": 0.2, "vertical_speed_max_abs": 0.5,
        "ground_speed_mean": 1.0,
        "ekf_vel_ratio_mean": 0.5, "ekf_pos_vert_accuracy_mean": 2.0,
        "vib_z_mean": 0.1,
    }

    results = []
    for metric in SEGMENT_METRICS:
        if metric not in df.columns or metric not in MARGINS:
            continue
        margin = MARGINS[metric]
        a = df[df["duty"] == 0][metric].dropna().values
        b = df[df["duty"] == 100][metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            p_tost, _, _ = ttost_ind(a, b, -margin, margin)
        except Exception:
            p_tost = 1.0

        mean_diff = b.mean() - a.mean()
        results.append({
            "metric": metric,
            "n_0pct": len(a),
            "n_100pct": len(b),
            "mean_diff": round(mean_diff, 6),
            "margin": margin,
            "p_tost": round(p_tost, 6),
            "equivalent": p_tost < 0.05,
        })
    return pd.DataFrame(results)


def test_task_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney between hover and goto segments for each metric."""
    results = []
    for metric in SEGMENT_METRICS:
        if metric not in df.columns:
            continue
        hover = df[df["task"] == "hover"][metric].dropna().values
        goto = df[df["task"] == "goto"][metric].dropna().values
        if len(hover) < 3 or len(goto) < 3:
            continue
        U, p = stats.mannwhitneyu(hover, goto, alternative="two-sided")
        results.append({
            "metric": metric,
            "U": U,
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "n_hover": len(hover),
            "n_goto": len(goto),
            "hover_median": round(np.median(hover), 6),
            "goto_median": round(np.median(goto), 6),
        })
    return pd.DataFrame(results)


def compare_with_monotonic(df_s27: pd.DataFrame,
                           stats_csv: Path) -> pd.DataFrame:
    """Compare S27 randomized hover segments with existing S01 monotonic data.

    Tests whether the duty-level distributions differ between the two designs.
    """
    if not stats_csv.exists():
        print(f"  Warning: {stats_csv} not found, skipping comparison")
        return pd.DataFrame()

    s01 = pd.read_csv(stats_csv)
    s01 = s01[(s01["scenario"] == "S01_hover_calm") & (s01["board"] == "npu_v2")]
    s01 = s01[~s01["run_dir"].isin(["logs/npu_v2_20260331_175500"])]

    # S27 hover-only segments
    s27_hover = df_s27[df_s27["task"] == "hover"]

    results = []
    for metric in ["roll_mean", "roll_std", "origin_dist_mean",
                    "vertical_speed_mean", "roll_max_abs"]:
        if metric not in s27_hover.columns or metric not in s01.columns:
            continue
        for duty in DUTY_LEVELS:
            s27_vals = s27_hover[s27_hover["duty"] == duty][metric].dropna().values
            s01_vals = s01[s01["npu_load"] == duty][metric].dropna().values
            if len(s27_vals) < 3 or len(s01_vals) < 3:
                continue
            U, p = stats.mannwhitneyu(s27_vals, s01_vals, alternative="two-sided")
            results.append({
                "metric": metric,
                "duty": duty,
                "n_s27": len(s27_vals),
                "n_s01": len(s01_vals),
                "s27_median": round(np.median(s27_vals), 6),
                "s01_median": round(np.median(s01_vals), 6),
                "U": U,
                "p_value": round(p, 6),
                "significant": p < 0.05,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame, temporal: pd.DataFrame,
                  duty: pd.DataFrame, duty_spearman: pd.DataFrame,
                  duty_tost: pd.DataFrame, task: pd.DataFrame,
                  comparison: pd.DataFrame):
    """Print human-readable summary."""
    print("\n" + "=" * 72)
    print("  S27 RANDOMIZED DUTY CYCLE ANALYSIS")
    print("=" * 72)

    # Data overview
    print(f"\n  Segments: {len(df)} total "
          f"({(df['task']=='hover').sum()} hover, "
          f"{(df['task']=='goto').sum()} goto)")
    print(f"  Flights: {df['flight'].nunique()}")
    for duty_val in DUTY_LEVELS:
        n = (df["duty"] == duty_val).sum()
        print(f"    {duty_val:>3}%: {n} segments")

    # Temporal drift
    print(f"\n  1. TEMPORAL DRIFT (Spearman: time_index vs metric)")
    print(f"     {'Metric':35s} {'rho':>8s} {'p':>10s} {'Sig':>5s}")
    print(f"     {'-'*35} {'-'*8} {'-'*10} {'-'*5}")
    n_sig = 0
    for _, row in temporal.iterrows():
        sig = "*" if row["significant"] else ""
        if row["significant"]:
            n_sig += 1
        print(f"     {row['metric']:35s} {row['rho']:>+8.4f} "
              f"{row['p_value']:>10.4f} {sig:>5s}")
    print(f"     {n_sig}/{len(temporal)} significant — "
          f"{'no temporal drift detected' if n_sig == 0 else 'temporal drift detected!'}")

    # Duty effect — KW
    print(f"\n  2. DUTY EFFECT (Kruskal-Wallis across 5 duty levels)")
    print(f"     {'Metric':35s} {'H':>8s} {'p':>10s} {'eps_sq':>8s} {'Sig':>5s}")
    print(f"     {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*5}")
    n_sig = 0
    for _, row in duty.iterrows():
        sig = "*" if row["significant"] else ""
        if row["significant"]:
            n_sig += 1
        print(f"     {row['metric']:35s} {row['kruskal_H']:>8.3f} "
              f"{row['p_value']:>10.4f} {row['epsilon_sq']:>8.5f} {sig:>5s}")
    print(f"     {n_sig}/{len(duty)} significant — "
          f"{'no duty effect detected' if n_sig == 0 else 'duty effect detected!'}")

    # Duty effect — Spearman
    print(f"\n  3. DUTY-METRIC CORRELATION (Spearman: duty vs metric)")
    print(f"     {'Metric':35s} {'rho':>8s} {'p':>10s} {'Sig':>5s}")
    print(f"     {'-'*35} {'-'*8} {'-'*10} {'-'*5}")
    n_sig = 0
    for _, row in duty_spearman.iterrows():
        sig = "*" if row["significant"] else ""
        if row["significant"]:
            n_sig += 1
        print(f"     {row['metric']:35s} {row['rho']:>+8.4f} "
              f"{row['p_value']:>10.4f} {sig:>5s}")
    print(f"     {n_sig}/{len(duty_spearman)} significant — "
          f"{'no duty correlation' if n_sig == 0 else 'duty correlation detected!'}")

    # Duty effect — TOST
    print(f"\n  4. TOST EQUIVALENCE (0% vs 100% segments)")
    print(f"     {'Metric':35s} {'Diff':>10s} {'Margin':>8s} {'p':>10s} {'Equiv':>6s}")
    print(f"     {'-'*35} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")
    n_eq = 0
    for _, row in duty_tost.iterrows():
        eq = "YES" if row["equivalent"] else "NO"
        if row["equivalent"]:
            n_eq += 1
        print(f"     {row['metric']:35s} {row['mean_diff']:>+10.5f} "
              f"{row['margin']:>8.2f} {row['p_tost']:>10.4f} {eq:>6s}")
    print(f"     {n_eq}/{len(duty_tost)} equivalent")

    # Task effect
    print(f"\n  5. TASK EFFECT (Mann-Whitney: hover vs goto)")
    print(f"     {'Metric':35s} {'p':>10s} {'hover med':>12s} {'goto med':>12s} {'Sig':>5s}")
    print(f"     {'-'*35} {'-'*10} {'-'*12} {'-'*12} {'-'*5}")
    n_sig = 0
    for _, row in task.iterrows():
        sig = "*" if row["significant"] else ""
        if row["significant"]:
            n_sig += 1
        print(f"     {row['metric']:35s} {row['p_value']:>10.4f} "
              f"{row['hover_median']:>12.4f} {row['goto_median']:>12.4f} {sig:>5s}")
    print(f"     {n_sig}/{len(task)} significant")

    # Comparison with S01 monotonic
    if len(comparison) > 0:
        print(f"\n  6. COMPARISON: S27 randomized hover vs S01 monotonic")
        print(f"     {'Metric':25s} {'Duty':>5s} {'S27 med':>10s} {'S01 med':>10s} "
              f"{'p':>10s} {'Sig':>5s}")
        print(f"     {'-'*25} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
        n_sig = 0
        for _, row in comparison.iterrows():
            sig = "*" if row["significant"] else ""
            if row["significant"]:
                n_sig += 1
            print(f"     {row['metric']:25s} {row['duty']:>4}% "
                  f"{row['s27_median']:>10.4f} {row['s01_median']:>10.4f} "
                  f"{row['p_value']:>10.4f} {sig:>5s}")
        print(f"     {n_sig}/{len(comparison)} significant — "
              f"{'designs produce equivalent results' if n_sig == 0 else 'some differences between designs'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="S27 randomized duty cycle segment analysis"
    )
    parser.add_argument(
        "--run-dir", default=str(S27_RUN_DIR),
        help=f"S27 test run directory (default: {S27_RUN_DIR})",
    )
    parser.add_argument(
        "--scenario-dir", default=str(SCENARIO_DIR),
        help="Scenario JSON directory (default: scenarios/)",
    )
    parser.add_argument(
        "--stats-csv", default="logs/stats/all_flight_stats.csv",
        help="Path to all_flight_stats.csv for S01 comparison",
    )
    parser.add_argument(
        "--output-dir", default="logs/plots/pca/s27_randomized/",
        help="Output directory",
    )
    parser.add_argument(
        "--settle", type=float, default=SETTLE_SECONDS,
        help=f"Settling time to discard per segment (default: {SETTLE_SECONDS}s)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scenario_dir = Path(args.scenario_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading S27 segments...")
    df = load_s27_segments(run_dir, scenario_dir)
    if len(df) == 0:
        print("No segments extracted.")
        sys.exit(1)

    # Save raw segments
    df.to_csv(output_dir / "s27_segments.csv", index=False)
    print(f"Saved: {output_dir / 's27_segments.csv'}")

    # Run tests
    print("\nRunning statistical tests...")
    temporal = test_temporal_drift(df)
    duty = test_duty_effect(df)
    duty_spearman = test_duty_spearman(df)
    duty_tost = test_duty_tost(df)
    task = test_task_effect(df)
    comparison = compare_with_monotonic(df, Path(args.stats_csv))

    # Save results
    temporal.to_csv(output_dir / "temporal_drift.csv", index=False)
    duty.to_csv(output_dir / "duty_effect.csv", index=False)
    duty_spearman.to_csv(output_dir / "duty_spearman.csv", index=False)
    duty_tost.to_csv(output_dir / "duty_tost.csv", index=False)
    task.to_csv(output_dir / "task_effect.csv", index=False)
    if len(comparison) > 0:
        comparison.to_csv(output_dir / "s27_vs_s01_comparison.csv", index=False)

    # Print summary
    print_summary(df, temporal, duty, duty_spearman, duty_tost, task, comparison)

    print(f"\n  All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
