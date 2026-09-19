#!/usr/bin/env python3
"""Statistical comparison of STM32N6 (NPU v2) vs Pixhawk 6C flight test data.

Loads all test runs from logs/, computes per-flight summary statistics using
the existing FlightStats infrastructure, then applies non-parametric
statistical tests to compare platforms and quantify NPU load impact.

Usage:
    python -m tools.flight_analyzer.statistical_analysis [--logs-dir logs/] [--out-dir logs/stats/]

Or directly:
    cd jMAVSim
    .venv/bin/python tools/flight_analyzer/statistical_analysis.py
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Reuse existing loader and analysis code
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.flight_analyzer.loader import (
    TestRun, RunEntry, discover_test_runs, load_flight_csv,
)
from tools.flight_analyzer.analysis import (
    FlightStats, compute_flight_stats, compute_hover_quality,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS_POSITION = [
    "origin_dist_mean", "pos_drift_max", "pos_drift_std",
    "alt_error_mean", "alt_error_max",
]
METRICS_ATTITUDE = [
    "roll_std", "roll_max_abs", "pitch_std", "pitch_max_abs",
]
METRICS_TRACKING = [
    "roll_err_rms", "pitch_err_rms", "yaw_err_rms",
]
METRICS_EKF = [
    "ekf_vel_ratio_mean", "ekf_vel_ratio_max",
    "ekf_pos_horiz_ratio_mean",
    "ekf_pos_horiz_accuracy_mean", "ekf_pos_vert_accuracy_mean",
]
METRICS_VIBRATION = [
    "vib_x_mean", "vib_y_mean", "vib_z_mean", "vib_magnitude_max",
]
METRICS_COMMS = [
    "comm_drop_rate_max",
]
METRICS_NPU = [
    "npu_ms_mean", "npu_fps_mean",
]

# Human-readable labels with units for plot axes
METRIC_LABELS = {
    "origin_dist_mean": "Position Drift Mean (m)",
    "pos_drift_max": "Position Drift Max (m)",
    "pos_drift_std": "Position Drift Std (m)",
    "alt_error_mean": "Altitude Error Mean (m)",
    "alt_error_max": "Altitude Error Max (m)",
    "roll_mean": "Roll Mean (deg)",
    "roll_std": "Roll Std (deg)",
    "roll_max_abs": "Roll Max Abs (deg)",
    "pitch_mean": "Pitch Mean (deg)",
    "pitch_std": "Pitch Std (deg)",
    "pitch_max_abs": "Pitch Max Abs (deg)",
    "ground_speed_mean": "Ground Speed Mean (m/s)",
    "ground_speed_max": "Ground Speed Max (m/s)",
    "vertical_speed_mean": "Vertical Speed Mean (m/s)",
    "vertical_speed_max_abs": "Vertical Speed Max Abs (m/s)",
    "roll_err_rms": "Roll Error RMS (deg)",
    "pitch_err_rms": "Pitch Error RMS (deg)",
    "yaw_err_rms": "Yaw Error RMS (deg)",
    "ekf_vel_ratio_mean": "EKF Velocity Ratio Mean",
    "ekf_vel_ratio_max": "EKF Velocity Ratio Max",
    "ekf_pos_horiz_ratio_mean": "EKF Horiz. Position Ratio Mean",
    "ekf_pos_horiz_accuracy_mean": "EKF Horiz. Position Accuracy (m)",
    "ekf_pos_vert_accuracy_mean": "EKF Vert. Position Accuracy (m)",
    "vib_x_mean": "Vibration X Mean (m/s\u00b2)",
    "vib_y_mean": "Vibration Y Mean (m/s\u00b2)",
    "vib_z_mean": "Vibration Z Mean (m/s\u00b2)",
    "vib_magnitude_max": "Vibration Magnitude Max (m/s\u00b2)",
    "clipping_total": "Clipping Events (count)",
    "npu_ms_mean": "NPU Inference Latency (ms)",
    "npu_fps_mean": "NPU Throughput (FPS)",
    "comm_drop_rate_max": "Comm Drop Rate Max (%)",
    "comm_errors_max": "Comm Errors Max (count)",
}
# All metrics suitable for cross-platform comparison (excludes NPU-only)
METRICS_CORE = (
    METRICS_POSITION + METRICS_ATTITUDE + METRICS_TRACKING +
    METRICS_EKF + METRICS_VIBRATION + METRICS_COMMS
)
ALL_METRICS = METRICS_CORE + METRICS_NPU

# Key metrics for summary tables — the most interesting ones
KEY_METRICS = [
    "origin_dist_mean", "pos_drift_max", "alt_error_mean",
    "roll_err_rms", "pitch_err_rms",
    "ekf_vel_ratio_mean", "ekf_pos_horiz_accuracy_mean",
    "vib_magnitude_max", "comm_drop_rate_max",
    "npu_ms_mean", "npu_fps_mean",
]

# Scenarios that are most useful for platform comparison
PARITY_SCENARIOS = [
    "S01_hover_calm", "S02_hover_moderate_wind", "S03_hover_strong_wind",
    "S04_hover_gust", "S06_waypoint_calm", "S07_waypoint_crosswind",
    "S08_complex_mission", "S09_takeoff_land_single",
    "S11_rapid_attitude_changes", "S12_tight_orbit",
    "S13_fast_altitude_changes", "S25_cross_platform_parity",
]

NPU_LOADS = [0, 25, 50, 75, 100]

# Significance level
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_stats(logs_dir: Path, downsample: int = 4) -> pd.DataFrame:
    """Load every test run, compute FlightStats per flight, return as DataFrame.

    Each row is one flight (one CSV). Columns include board, npu_load,
    scenario, repetition, pass/fail, and all FlightStats fields.
    """
    test_runs = discover_test_runs(logs_dir)
    print(f"Discovered {len(test_runs)} test runs")

    rows = []
    for tr in test_runs:
        for run in tr.runs:
            if run.csv_path is None or not Path(run.csv_path).exists():
                continue
            try:
                df = load_flight_csv(run.csv_path, downsample=downsample)
                fs = compute_flight_stats(df)
            except Exception as e:
                print(f"  Warning: {run.csv_path}: {e}")
                continue

            row = {
                "board": tr.board,
                "npu_load": tr.npu_load if tr.npu_load is not None else 0,
                "scenario": run.scenario,
                "repetition": run.repetition,
                "passed": run.passed,
                "run_timestamp": tr.timestamp,
                "run_dir": str(tr.path),
            }
            for f in fields(FlightStats):
                row[f.name] = getattr(fs, f.name)
            rows.append(row)

    result = pd.DataFrame(rows)
    print(f"Loaded {len(result)} flight records "
          f"({result['board'].nunique()} boards, "
          f"{result['scenario'].nunique()} scenarios)")
    return result


def tag_npu_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column indicating whether NPU telemetry is valid."""
    df = df.copy()
    df["has_npu_telemetry"] = df["npu_ms_mean"] > 0
    # For 0% NPU load, telemetry is expected to be zero — mark as valid
    df.loc[df["npu_load"] == 0, "has_npu_telemetry"] = True
    return df


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_summary(df: pd.DataFrame, metrics: list[str],
                        groupby: list[str]) -> pd.DataFrame:
    """Compute descriptive stats for each metric grouped by given columns."""
    agg_funcs = {
        m: ["count", "mean", "std", "median",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            "min", "max"]
        for m in metrics if m in df.columns
    }
    summary = df.groupby(groupby).agg(agg_funcs)
    # Flatten multi-level columns
    summary.columns = [f"{m}_{stat}" if stat != "<lambda_0>" and stat != "<lambda_1>"
                       else f"{m}_{'q25' if '<lambda_0>' in stat else 'q75'}"
                       for m, stat in summary.columns]
    # Fix lambda names
    new_cols = []
    lambda_counter = {}
    for m, stat in [(c.rsplit("_", 1)[0], c.rsplit("_", 1)[1])
                    for c in summary.columns]:
        if "lambda" in stat:
            key = m
            lambda_counter[key] = lambda_counter.get(key, 0) + 1
            stat = "q25" if lambda_counter[key] % 2 == 1 else "q75"
        new_cols.append(f"{m}_{stat}")
    summary.columns = new_cols
    return summary


def coefficient_of_variation(df: pd.DataFrame, metrics: list[str],
                             groupby: list[str]) -> pd.DataFrame:
    """Compute CV (std/mean * 100) for each metric per group."""
    rows = []
    for name, grp in df.groupby(groupby):
        row = dict(zip(groupby, name)) if isinstance(name, tuple) else {groupby[0]: name}
        row["n"] = len(grp)
        for m in metrics:
            if m in grp.columns:
                mean = grp[m].mean()
                std = grp[m].std()
                row[f"{m}_cv"] = (std / mean * 100) if mean != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mann_whitney_test(group_a: pd.Series, group_b: pd.Series,
                      label_a: str = "A", label_b: str = "B") -> dict:
    """Mann-Whitney U test with rank-biserial effect size."""
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"test": "Mann-Whitney U", "skipped": True,
                "reason": f"n_a={len(a)}, n_b={len(b)}"}

    u_stat, p_value = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
    n1, n2 = len(a), len(b)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "label_a": label_a, "n_a": n1, "median_a": float(a.median()),
        "label_b": label_b, "n_b": n2, "median_b": float(b.median()),
        "U": float(u_stat), "p": float(p_value),
        "rank_biserial_r": float(r_rb),
        "significant": p_value < ALPHA,
    }


def kruskal_wallis_test(groups: dict[str, pd.Series]) -> dict:
    """Kruskal-Wallis H test across k groups."""
    clean = {k: v.dropna() for k, v in groups.items() if len(v.dropna()) >= 2}
    if len(clean) < 2:
        return {"test": "Kruskal-Wallis", "skipped": True,
                "reason": f"only {len(clean)} groups with n>=2"}

    arrays = list(clean.values())
    h_stat, p_value = sp_stats.kruskal(*arrays)
    # Effect size: eta-squared = (H - k + 1) / (N - k)
    k = len(arrays)
    N = sum(len(a) for a in arrays)
    eta_sq = (h_stat - k + 1) / (N - k) if N > k else np.nan

    return {
        "test": "Kruskal-Wallis",
        "groups": {k: {"n": len(v), "median": float(v.median())}
                   for k, v in clean.items()},
        "H": float(h_stat), "p": float(p_value),
        "eta_squared": float(eta_sq),
        "significant": p_value < ALPHA,
    }


def dunns_posthoc(groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Dunn's post-hoc test with Bonferroni correction.

    Implemented manually since scikit-posthocs may not be installed.
    Uses the normal approximation approach.
    """
    clean = {k: v.dropna().values for k, v in groups.items()
             if len(v.dropna()) >= 2}
    keys = sorted(clean.keys())
    if len(keys) < 2:
        return pd.DataFrame()

    # Pool all values and rank them
    all_vals = np.concatenate([clean[k] for k in keys])
    N = len(all_vals)
    ranks = sp_stats.rankdata(all_vals)

    # Assign ranks back to groups
    group_ranks = {}
    idx = 0
    for k in keys:
        n = len(clean[k])
        group_ranks[k] = ranks[idx:idx + n]
        idx += n

    mean_ranks = {k: np.mean(r) for k, r in group_ranks.items()}

    # Compute tied-rank correction
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_correction = 1 - np.sum(tie_counts**3 - tie_counts) / (N**3 - N)

    results = []
    pairs = [(keys[i], keys[j]) for i in range(len(keys))
             for j in range(i + 1, len(keys))]
    n_comparisons = len(pairs)

    for k1, k2 in pairs:
        n1, n2 = len(clean[k1]), len(clean[k2])
        diff = mean_ranks[k1] - mean_ranks[k2]
        se = np.sqrt(tie_correction * (N * (N + 1) / 12) * (1/n1 + 1/n2))
        z = diff / se if se > 0 else 0
        p = 2 * sp_stats.norm.sf(abs(z))
        p_adj = min(p * n_comparisons, 1.0)  # Bonferroni
        results.append({
            "group_a": k1, "group_b": k2,
            "mean_rank_a": mean_ranks[k1], "mean_rank_b": mean_ranks[k2],
            "z": z, "p_raw": p, "p_adj_bonferroni": p_adj,
            "significant": p_adj < ALPHA,
        })

    return pd.DataFrame(results)


def jonckheere_terpstra_test(groups: dict[str, pd.Series]) -> dict:
    """Jonckheere-Terpstra test for monotonic ordered trend.

    Tests H1: values tend to increase with group order.
    Groups dict keys should be orderable (e.g. "0", "25", "50", ...).
    """
    ordered_keys = sorted(groups.keys(), key=lambda x: int(x))
    clean = {k: groups[k].dropna().values for k in ordered_keys
             if len(groups[k].dropna()) >= 1}
    keys = list(clean.keys())
    if len(keys) < 2:
        return {"test": "Jonckheere-Terpstra", "skipped": True}

    # Compute J statistic: count of concordant pairs across groups
    J = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for xi in clean[keys[i]]:
                for xj in clean[keys[j]]:
                    if xj > xi:
                        J += 1
                    elif xj == xi:
                        J += 0.5

    # Expected value and variance under null
    ns = [len(clean[k]) for k in keys]
    N = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4

    # Variance (assuming no ties for simplicity)
    num = (2 * N**3 + 3 * N**2 - sum(n**2 * (2*n + 3) for n in ns))
    Var_J = num / 72

    z = (J - E_J) / np.sqrt(Var_J) if Var_J > 0 else 0
    # One-sided p-value (increasing trend)
    p_increasing = sp_stats.norm.sf(z)
    # Two-sided
    p_two = 2 * sp_stats.norm.sf(abs(z))

    return {
        "test": "Jonckheere-Terpstra",
        "J": float(J), "E_J": float(E_J), "Var_J": float(Var_J),
        "z": float(z),
        "p_increasing": float(p_increasing),
        "p_two_sided": float(p_two),
        "trend": "increasing" if z > 0 else "decreasing",
        "significant_increasing": p_increasing < ALPHA,
        "groups": {k: len(clean[k]) for k in keys},
    }


def spearman_correlation(values: pd.Series, loads: pd.Series) -> dict:
    """Spearman rank correlation between a metric and NPU load."""
    mask = values.notna() & loads.notna()
    v, l = values[mask], loads[mask]
    if len(v) < 3:
        return {"test": "Spearman", "skipped": True}

    rho, p = sp_stats.spearmanr(l, v)
    return {
        "test": "Spearman",
        "rho": float(rho), "p": float(p),
        "n": len(v),
        "significant": p < ALPHA,
    }


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                  ci: float = 0.95, statistic=np.mean) -> tuple[float, float]:
    """BCa bootstrap confidence interval."""
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(42)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def pass_rate_fisher(group_a: pd.Series, group_b: pd.Series,
                     label_a: str = "A", label_b: str = "B") -> dict:
    """Fisher's exact test on pass/fail counts."""
    a_pass = group_a.sum()
    a_fail = len(group_a) - a_pass
    b_pass = group_b.sum()
    b_fail = len(group_b) - b_pass

    table = [[int(a_pass), int(a_fail)], [int(b_pass), int(b_fail)]]
    _, p = sp_stats.fisher_exact(table)

    return {
        "test": "Fisher exact",
        label_a: {"pass": int(a_pass), "fail": int(a_fail),
                  "rate": float(a_pass / len(group_a)) if len(group_a) > 0 else 0},
        label_b: {"pass": int(b_pass), "fail": int(b_fail),
                  "rate": float(b_pass / len(group_b)) if len(group_b) > 0 else 0},
        "p": float(p),
        "significant": p < ALPHA,
    }


# ---------------------------------------------------------------------------
# High-level analysis functions
# ---------------------------------------------------------------------------

def analyze_platform_parity(df: pd.DataFrame,
                            scenarios: Optional[list[str]] = None
                            ) -> dict:
    """Compare Pixhawk vs STM32N6 @ 0% NPU across all shared scenarios."""
    pix = df[(df["board"] == "pixhawk6c")]
    npu0 = df[(df["board"] == "npu_v2") & (df["npu_load"] == 0)]

    if scenarios is None:
        shared = sorted(set(pix["scenario"]) & set(npu0["scenario"]))
    else:
        shared = [s for s in scenarios
                  if s in pix["scenario"].values and s in npu0["scenario"].values]

    results = {}
    for scenario in shared:
        p_data = pix[pix["scenario"] == scenario]
        n_data = npu0[npu0["scenario"] == scenario]

        scenario_results = {}
        for metric in METRICS_CORE:
            if metric not in df.columns:
                continue
            mw = mann_whitney_test(
                p_data[metric], n_data[metric],
                label_a="pixhawk6c", label_b="npu_v2_0pct"
            )
            scenario_results[metric] = mw

        # Pass rate comparison
        if len(p_data) > 0 and len(n_data) > 0:
            scenario_results["pass_rate"] = pass_rate_fisher(
                p_data["passed"], n_data["passed"],
                label_a="pixhawk6c", label_b="npu_v2_0pct"
            )

        results[scenario] = scenario_results

    return results


def analyze_npu_load_effect(df: pd.DataFrame,
                            scenarios: Optional[list[str]] = None
                            ) -> dict:
    """Kruskal-Wallis + Jonckheere-Terpstra across NPU load levels."""
    npu = df[df["board"] == "npu_v2"]

    if scenarios is None:
        scenarios = sorted(npu["scenario"].unique())

    results = {}
    for scenario in scenarios:
        sdf = npu[npu["scenario"] == scenario]
        if len(sdf) == 0:
            continue

        scenario_results = {}
        for metric in ALL_METRICS:
            if metric not in sdf.columns:
                continue

            groups = {str(load): sdf[sdf["npu_load"] == load][metric]
                      for load in NPU_LOADS
                      if load in sdf["npu_load"].values}

            if len(groups) < 2:
                continue

            kw = kruskal_wallis_test(groups)
            jt = jonckheere_terpstra_test(groups)
            sp = spearman_correlation(sdf[metric], sdf["npu_load"])

            metric_result = {
                "kruskal_wallis": kw,
                "jonckheere_terpstra": jt,
                "spearman": sp,
            }

            # Post-hoc if Kruskal-Wallis is significant
            if not kw.get("skipped") and kw.get("significant"):
                metric_result["dunns_posthoc"] = dunns_posthoc(groups).to_dict("records")

            # Bootstrap CIs per group
            boot = {}
            for load_key, series in groups.items():
                vals = series.dropna().values
                if len(vals) >= 2:
                    ci_lo, ci_hi = bootstrap_ci(vals)
                    boot[load_key] = {
                        "mean": float(np.mean(vals)),
                        "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
                        "n": len(vals),
                    }
            metric_result["bootstrap_ci"] = boot

            scenario_results[metric] = metric_result

        results[scenario] = scenario_results

    return results


def analyze_pooled_npu_effect(df: pd.DataFrame) -> dict:
    """Pool all scenarios and test NPU load effect on each metric."""
    npu = df[df["board"] == "npu_v2"]
    results = {}

    for metric in ALL_METRICS:
        if metric not in npu.columns:
            continue

        groups = {str(load): npu[npu["npu_load"] == load][metric]
                  for load in NPU_LOADS
                  if load in npu["npu_load"].values}
        if len(groups) < 2:
            continue

        results[metric] = {
            "kruskal_wallis": kruskal_wallis_test(groups),
            "jonckheere_terpstra": jonckheere_terpstra_test(groups),
            "spearman": spearman_correlation(npu[metric], npu["npu_load"]),
        }

    return results


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def make_parity_summary_table(parity: dict, metrics: list[str]) -> pd.DataFrame:
    """Create a readable summary of Mann-Whitney results."""
    rows = []
    for scenario, metric_results in sorted(parity.items()):
        for metric in metrics:
            if metric not in metric_results:
                continue
            r = metric_results[metric]
            if r.get("skipped"):
                continue
            rows.append({
                "scenario": scenario,
                "metric": metric,
                "pixhawk_median": r.get("median_a"),
                "npu_v2_0pct_median": r.get("median_b"),
                "U": r.get("U"),
                "p": r.get("p"),
                "r_rb": r.get("rank_biserial_r"),
                "significant": r.get("significant"),
            })
    return pd.DataFrame(rows)


def make_npu_effect_summary_table(npu_effect: dict,
                                  metrics: list[str]) -> pd.DataFrame:
    """Summarize Kruskal-Wallis + Jonckheere-Terpstra results."""
    rows = []
    for scenario, metric_results in sorted(npu_effect.items()):
        for metric in metrics:
            if metric not in metric_results:
                continue
            r = metric_results[metric]
            kw = r.get("kruskal_wallis", {})
            jt = r.get("jonckheere_terpstra", {})
            sp = r.get("spearman", {})

            if kw.get("skipped") or jt.get("skipped"):
                continue

            rows.append({
                "scenario": scenario,
                "metric": metric,
                "KW_H": kw.get("H"),
                "KW_p": kw.get("p"),
                "KW_eta2": kw.get("eta_squared"),
                "KW_sig": kw.get("significant"),
                "JT_z": jt.get("z"),
                "JT_p_incr": jt.get("p_increasing"),
                "JT_trend": jt.get("trend"),
                "JT_sig": jt.get("significant_increasing"),
                "Spearman_rho": sp.get("rho"),
                "Spearman_p": sp.get("p"),
            })
    return pd.DataFrame(rows)


def make_pooled_summary_table(pooled: dict) -> pd.DataFrame:
    """One-row-per-metric summary of pooled NPU load effect."""
    rows = []
    for metric, r in pooled.items():
        kw = r.get("kruskal_wallis", {})
        jt = r.get("jonckheere_terpstra", {})
        sp = r.get("spearman", {})
        if kw.get("skipped"):
            continue

        medians = {}
        if "groups" in kw:
            for k, info in kw["groups"].items():
                medians[f"median_{k}pct"] = info["median"]

        rows.append({
            "metric": metric,
            **medians,
            "KW_H": kw.get("H"),
            "KW_p": kw.get("p"),
            "KW_eta2": kw.get("eta_squared"),
            "KW_sig": kw.get("significant"),
            "JT_z": jt.get("z"),
            "JT_p_incr": jt.get("p_increasing"),
            "JT_trend": jt.get("trend"),
            "Spearman_rho": sp.get("rho"),
            "Spearman_p": sp.get("p"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_by_npu_load(df: pd.DataFrame, metric: str,
                            scenario: Optional[str] = None,
                            out_dir: Optional[Path] = None):
    """Box + strip plot of a metric across NPU load levels, with Pixhawk ref."""
    is_npu_metric = metric in METRICS_NPU
    npu = df[df["board"] == "npu_v2"]
    # For NPU-specific metrics, only use runs with valid NPU telemetry
    if is_npu_metric:
        npu = npu[npu["has_npu_telemetry"] & (npu["npu_load"] > 0)]
    pix = df[df["board"] == "pixhawk6c"]
    if scenario:
        npu = npu[npu["scenario"] == scenario]
        pix = pix[pix["scenario"] == scenario]
    if metric not in npu.columns or npu[metric].isna().all():
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    title_scenario = scenario or "All scenarios (pooled)"

    # NPU box plot
    plot_data = []
    for load in NPU_LOADS:
        vals = npu[npu["npu_load"] == load][metric].dropna()
        if len(vals) > 0:
            for v in vals:
                plot_data.append({"NPU Load (%)": load, metric: v})
    if not plot_data:
        plt.close(fig)
        return

    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(data=plot_df, x="NPU Load (%)", y=metric, ax=ax,
                color="lightblue", fliersize=3, width=0.5)
    sns.stripplot(data=plot_df, x="NPU Load (%)", y=metric, ax=ax,
                  color="navy", alpha=0.5, size=4, jitter=True)

    # Pixhawk reference line (skip for NPU-only metrics)
    pix_vals = pix[metric].dropna()
    if len(pix_vals) > 0 and not is_npu_metric:
        pix_median = pix_vals.median()
        ax.axhline(pix_median, color="red", linestyle="--", linewidth=1.5,
                   label=f"Pixhawk median ({pix_median:.4f})")
        ax.axhspan(pix_vals.quantile(0.25), pix_vals.quantile(0.75),
                   color="red", alpha=0.08, label="Pixhawk IQR")
        ax.legend(fontsize=8)

    ylabel = METRIC_LABELS.get(metric, metric)
    ax.set_title(f"{ylabel}\n{title_scenario}", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")

    if out_dir:
        fname = f"npu_load_{metric}{'_' + scenario if scenario else '_pooled'}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_parity_heatmap(parity_table: pd.DataFrame,
                        out_dir: Optional[Path] = None):
    """Heatmap of Mann-Whitney p-values for platform parity."""
    if parity_table.empty:
        return

    pivot = parity_table.pivot_table(index="scenario", columns="metric",
                                     values="p", aggfunc="first")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.2),
                                     max(6, len(pivot) * 0.5)))
    # Log-transform p-values for better color scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_p = -np.log10(pivot.clip(lower=1e-10))

    sns.heatmap(log_p, ax=ax, cmap="RdYlGn_r", annot=pivot.round(3),
                fmt="", linewidths=0.5, vmin=0, vmax=3,
                cbar_kws={"label": "-log10(p)"})
    ax.set_title("Platform Parity: Mann-Whitney p-values\n"
                 "(Pixhawk 6C vs STM32N6 @ 0% NPU)\n"
                 "Green = similar, Red = significantly different", fontsize=11)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if out_dir:
        fig.savefig(out_dir / "parity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_npu_effect_heatmap(effect_table: pd.DataFrame,
                            stat_col: str = "Spearman_rho",
                            out_dir: Optional[Path] = None):
    """Heatmap of Spearman rho (NPU load correlation) per scenario/metric."""
    if effect_table.empty:
        return

    pivot = effect_table.pivot_table(index="scenario", columns="metric",
                                     values=stat_col, aggfunc="first")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.2),
                                     max(6, len(pivot) * 0.5)))

    vmin, vmax = (-1, 1) if "rho" in stat_col else (None, None)
    cmap = "RdBu_r" if "rho" in stat_col else "YlOrRd"

    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, center=0, vmin=vmin, vmax=vmax,
                cbar_kws={"label": stat_col})
    ax.set_title(f"NPU Load Effect: {stat_col} per Scenario\n"
                 "Red = increases with NPU load, Blue = decreases", fontsize=11)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if out_dir:
        fname = f"npu_effect_heatmap_{stat_col}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_descriptive_boxplots(df: pd.DataFrame, metrics: list[str],
                              out_dir: Optional[Path] = None):
    """Per-scenario boxplots: Pixhawk vs NPU at each load level."""
    scenarios = sorted(df["scenario"].unique())

    for metric in metrics:
        if metric not in df.columns or df[metric].isna().all():
            continue

        is_npu_metric = metric in METRICS_NPU
        ylabel = METRIC_LABELS.get(metric, metric)

        for scenario in scenarios:
            sdf = df[df["scenario"] == scenario]

            plot_rows = []
            # Pixhawk — skip for NPU-only metrics
            if not is_npu_metric:
                pix = sdf[sdf["board"] == "pixhawk6c"][metric].dropna()
                for v in pix:
                    plot_rows.append({"group": "Pixhawk", "value": v})
            # NPU per load
            for load in NPU_LOADS:
                mask = (sdf["board"] == "npu_v2") & (sdf["npu_load"] == load)
                if is_npu_metric:
                    mask = mask & (sdf["has_npu_telemetry"]) & (sdf["npu_load"] > 0)
                vals = sdf[mask][metric].dropna()
                for v in vals:
                    plot_rows.append({"group": f"NPU {load}%", "value": v})

            if not plot_rows:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df = pd.DataFrame(plot_rows)
            order = ["Pixhawk"] + [f"NPU {l}%" for l in NPU_LOADS]
            order = [o for o in order if o in plot_df["group"].values]

            palette = {"Pixhawk": "#e74c3c"}
            palette.update({f"NPU {l}%": plt.cm.Blues(0.3 + 0.14 * i)
                            for i, l in enumerate(NPU_LOADS)})

            sns.boxplot(data=plot_df, x="group", y="value", ax=ax,
                        order=order, palette=palette, fliersize=3, width=0.6)
            sns.stripplot(data=plot_df, x="group", y="value", ax=ax,
                          order=order, color="black", alpha=0.3, size=3,
                          jitter=True)

            ax.set_title(f"{ylabel} — {scenario}", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis="y")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            if out_dir:
                scenario_dir = out_dir / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(scenario_dir / f"boxplot_{metric}.png", dpi=150,
                            bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of jMAVSim HITL test logs")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"),
                        help="Root logs directory")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/stats"),
                        help="Output directory for results and plots")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Downsample CSV by this factor (default: 4)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading flight data")
    print("=" * 70)
    all_stats = load_all_stats(args.logs_dir, downsample=args.downsample)
    all_stats = tag_npu_telemetry(all_stats)

    # Save raw stats table
    all_stats.to_csv(out_dir / "all_flight_stats.csv", index=False)
    print(f"Saved all_flight_stats.csv ({len(all_stats)} rows)")

    # Print data inventory
    print("\nData inventory:")
    inv = all_stats.groupby(["board", "npu_load"]).agg(
        flights=("scenario", "count"),
        scenarios=("scenario", "nunique"),
        pass_rate=("passed", "mean"),
    )
    print(inv.to_string())
    print()

    # ------------------------------------------------------------------
    # 2. Descriptive statistics
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: Descriptive statistics")
    print("=" * 70)

    desc_platform = descriptive_summary(
        all_stats, KEY_METRICS, ["board", "npu_load"])
    desc_platform.to_csv(out_dir / "descriptive_by_platform_npu.csv")
    print("Saved descriptive_by_platform_npu.csv")

    desc_scenario = descriptive_summary(
        all_stats, KEY_METRICS, ["board", "npu_load", "scenario"])
    desc_scenario.to_csv(out_dir / "descriptive_by_scenario.csv")
    print("Saved descriptive_by_scenario.csv")

    cv = coefficient_of_variation(
        all_stats, METRICS_CORE, ["board", "npu_load"])
    cv.to_csv(out_dir / "coefficient_of_variation.csv", index=False)
    print("Saved coefficient_of_variation.csv")
    print("\nCV summary (repeatability):")
    print(cv.to_string(index=False, max_cols=10))
    print()

    # ------------------------------------------------------------------
    # 3. Platform parity analysis (Pixhawk vs NPU @ 0%)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 3: Platform parity — Pixhawk 6C vs STM32N6 @ 0% NPU")
    print("=" * 70)

    parity = analyze_platform_parity(all_stats)
    parity_table = make_parity_summary_table(parity, METRICS_CORE)

    if not parity_table.empty:
        parity_table.to_csv(out_dir / "parity_mann_whitney.csv", index=False)
        print(f"Saved parity_mann_whitney.csv ({len(parity_table)} tests)")

        sig = parity_table[parity_table["significant"]]
        print(f"\nSignificant differences (p < {ALPHA}): "
              f"{len(sig)}/{len(parity_table)}")
        if len(sig) > 0:
            print(sig[["scenario", "metric", "pixhawk_median",
                        "npu_v2_0pct_median", "p", "r_rb"]
                       ].to_string(index=False))
        else:
            print("  No significant differences — platforms are equivalent "
                  "at 0% NPU load!")
    print()

    # ------------------------------------------------------------------
    # 4. NPU load effect (per scenario)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 4: NPU load effect — Kruskal-Wallis + Jonckheere-Terpstra")
    print("=" * 70)

    npu_effect = analyze_npu_load_effect(all_stats)
    npu_effect_table = make_npu_effect_summary_table(npu_effect, ALL_METRICS)

    if not npu_effect_table.empty:
        npu_effect_table.to_csv(out_dir / "npu_effect_per_scenario.csv",
                                index=False)
        print(f"Saved npu_effect_per_scenario.csv ({len(npu_effect_table)} tests)")

        sig_kw = npu_effect_table[npu_effect_table["KW_sig"] == True]
        sig_jt = npu_effect_table[npu_effect_table["JT_sig"] == True]
        print(f"\nKruskal-Wallis significant: {len(sig_kw)}/{len(npu_effect_table)}")
        print(f"Jonckheere-Terpstra significant (increasing): "
              f"{len(sig_jt)}/{len(npu_effect_table)}")

        if len(sig_jt) > 0:
            print("\nMetrics with significant monotonic increase with NPU load:")
            cols = ["scenario", "metric", "JT_z", "JT_p_incr",
                    "Spearman_rho", "KW_eta2"]
            cols = [c for c in cols if c in sig_jt.columns]
            print(sig_jt[cols].to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 5. Pooled NPU effect (all scenarios combined)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 5: Pooled NPU load effect (all scenarios combined)")
    print("=" * 70)

    pooled = analyze_pooled_npu_effect(all_stats)
    pooled_table = make_pooled_summary_table(pooled)

    if not pooled_table.empty:
        pooled_table.to_csv(out_dir / "npu_effect_pooled.csv", index=False)
        print("Saved npu_effect_pooled.csv")
        print("\nPooled NPU load effect:")
        display_cols = ["metric", "KW_H", "KW_p", "KW_sig",
                        "JT_z", "JT_p_incr", "JT_trend",
                        "Spearman_rho", "Spearman_p"]
        display_cols = [c for c in display_cols if c in pooled_table.columns]
        print(pooled_table[display_cols].to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 6. Save full results as JSON
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 6: Saving full results")
    print("=" * 70)

    full_results = {
        "platform_parity": parity,
        "npu_effect_per_scenario": _sanitize_for_json(npu_effect),
        "npu_effect_pooled": _sanitize_for_json(pooled),
    }
    with open(out_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print("Saved full_results.json")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    if not args.skip_plots:
        print("\n" + "=" * 70)
        print("STEP 7: Generating plots")
        print("=" * 70)

        # Parity heatmap
        plot_parity_heatmap(parity_table, out_dir=plots_dir)
        print("  parity_heatmap.png")

        # NPU effect heatmaps
        plot_npu_effect_heatmap(npu_effect_table, "Spearman_rho",
                                out_dir=plots_dir)
        print("  npu_effect_heatmap_Spearman_rho.png")

        plot_npu_effect_heatmap(npu_effect_table, "KW_eta2",
                                out_dir=plots_dir)
        print("  npu_effect_heatmap_KW_eta2.png")

        # Per-scenario boxplots for key metrics
        print("  Generating per-scenario boxplots...")
        plot_descriptive_boxplots(all_stats, KEY_METRICS, out_dir=plots_dir)
        n_scenarios = all_stats["scenario"].nunique()
        print(f"  {n_scenarios} scenarios x {len(KEY_METRICS)} metrics "
              f"= boxplots in plots/<scenario>/")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"  CSV tables:  {len(list(out_dir.glob('*.csv')))} files")
    print(f"  JSON:        {len(list(out_dir.glob('*.json')))} files")
    if not args.skip_plots:
        n_plots = len(list(plots_dir.rglob('*.png')))
        print(f"  Plots:       {n_plots} files")


def _sanitize_for_json(obj):
    """Convert numpy types and other non-serializable objects for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
