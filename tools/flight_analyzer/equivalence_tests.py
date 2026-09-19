"""Equivalence and trend testing for NPU duty cycle impact.

Tests:
1. TOST (Two One-Sided Tests) for equivalence within operational margins
2. Jonckheere-Terpstra trend test for monotonic degradation with duty level
3. Spearman rank correlation between duty level and each metric

All tests run per-scenario on operationally relevant metrics.
BH FDR correction applied across all tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ttost_ind

STATS_CSV = Path(__file__).resolve().parent.parent.parent / "logs" / "stats" / "all_flight_stats.csv"
INCLUDED_RUNS = [
    # N6 n=10 full regression (new NPU module, aligned EKF params)
    "logs/npu_v2_20260418_131423",   # NPU 0%
    "logs/npu_v2_20260418_185511",   # NPU 25%
    "logs/npu_v2_20260419_003555",   # NPU 50%
    "logs/npu_v2_20260419_061806",   # NPU 75%
    "logs/npu_v2_20260419_115740",   # NPU 100%
    # Pixhawk n=10 baseline
    "logs/pixhawk6c_20260420_014914",
    # S27 randomised validation (n=10)
    "logs/npu_v2_20260420_160601",
]
EXCLUDED_RUNS = []  # Using INCLUDED_RUNS allowlist instead
OUTLIER_FILTER = lambda df: df  # No outlier exclusions for fresh dataset

KEY_METRICS = [
    # PCA-identified top discriminators (non-artifact, balanced data)
    # Margins are SESOI per PX4 documented thresholds (see PCA_REPORT.md Section 4.3)
    ("ekf_pos_vert_accuracy_mean", "m", 2.0),       # MAVLink ESTIMATOR_STATUS 1-STD accuracy
    ("vertical_speed_mean", "m/s", 0.2),             # PX4 EKF2_REQ_VDRIFT = 0.2 m/s (GPS vertical drift gate)
    ("pitch_mean", "deg", 2.0),                      # PX4 EKF2 preflight gate
    ("roll_mean", "deg", 2.0),                       # Same as pitch_mean
    ("roll_err_rms", "deg", 2.0),                    # PX4 attitude setpoint error operational band
    ("pitch_max_abs", "deg", 2.0),                   # PX4 EKF2 preflight gate (0.25 rad ≈ 14.3°), conservative
    ("pos_drift_max", "m", 3.0),                     # PX4 EKF2_REQ_EPH = 3.0 m (GPS horizontal accuracy gate)
    ("pos_drift_std", "m", 0.5),                     # EKF2_GPS_P_NOISE = 0.5 m; 25% of MPC_XY_ERR_MAX (2.0 m)
    ("roll_std", "deg", 2.0),                        # Same as attitude error band
    ("vertical_speed_max_abs", "m/s", 0.5),          # 2× baseline SD; conservative bound above EKF2_GPS_V_NOISE
    ("roll_max_abs", "deg", 2.0),                    # PX4 EKF2 preflight gate, conservative
    ("ekf_vel_ratio_mean", "", 0.5),                 # PX4 COM_ARM_EKF_VEL = 0.5 (innovation ratio arming gate)
    ("ground_speed_mean", "m/s", 1.0),               # ~8% of PX4 MPC_XY_VEL_MAX (12 m/s)
    ("xte_rms", "m", 0.5),                           # PX4 EKF2_GPS_P_NOISE = 0.5 m; cross-track error
    ("pix_traj_rmse", "m", 1.0),                     # 2× within-group Pixhawk consistency (~0.5 m)
    ("vib_z_mean", "m/s²", 0.1),                     # HITL vibration artifact — included for completeness
    ("vib_magnitude_max", "m/s²", 0.5),              # HITL vibration artifact — included for completeness
]

# PCA-identified cross-platform discriminators (non-artifact + artifact for completeness)
CROSS_PLATFORM_METRICS = [
    ("roll_mean", "deg", 2.0),           # PX4 EKF2 preflight gate
    ("roll_max_abs", "deg", 2.0),        # PX4 EKF2 preflight gate, conservative
    ("ground_speed_max", "m/s", 1.0),    # ~8% of MPC_XY_VEL_MAX
    ("ekf_vel_ratio_max", "", 0.5),      # MAVLink ESTIMATOR_STATUS normal band
    ("vib_z_mean", "m/s²", 1.0),        # HITL noise floor (document: ±1 m/s² for HITL)
]

DUTY_LEVELS = [0, 25, 50, 75, 100]

# Per-scenario PCA-flagged top discriminator (from per-scenario PCA on balanced data)
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
    "S14_hover_degraded_gps": "ekf_pos_vert_accuracy_mean",
    "S15_mission_gps_dropout": "vertical_speed_max_abs",
    "S18_rtl_during_mission": "roll_max_abs",
    "S19_low_battery_failsafe": "pitch_mean",
    "S20_sensor_failure": "vertical_speed_mean",
    "S23_hitl_loop_validation": "pitch_mean",
    "S25_cross_platform_parity": "ekf_vel_ratio_mean",
    "S26_wind_gps_degradation": "ground_speed_mean",
}


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_pvals[i] * n / (i + 1)
        )
    return np.clip(adjusted, 0, 1)


def tost_per_scenario(df: pd.DataFrame, metric: str, margin: float,
                      npu_a: int = 0, npu_b: int = 100):
    """Run TOST for equivalence between two NPU levels within each scenario."""
    results = []
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        a = sdf[sdf["npu_load"] == npu_a][metric].dropna().values
        b = sdf[sdf["npu_load"] == npu_b][metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        # TOST: test if difference is within [-margin, +margin]
        p_value, (t1, p1, df1), (t2, p2, df2) = ttost_ind(a, b, -margin, margin)
        mean_diff = b.mean() - a.mean()
        n1, n2 = len(a), len(b)
        sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
        cohens_d = mean_diff / sp if sp > 0 else 0.0
        # 90% CI on mean difference (dual of two one-sided tests at alpha=0.05)
        se = sp * np.sqrt(1 / n1 + 1 / n2)
        dof = n1 + n2 - 2
        t_crit = stats.t.ppf(0.95, dof)
        ci_90_lo = mean_diff - t_crit * se
        ci_90_hi = mean_diff + t_crit * se
        results.append({
            "scenario": scenario,
            "metric": metric,
            "n_a": n1,
            "n_b": n2,
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "margin": margin,
            "ci_90_lo": ci_90_lo,
            "ci_90_hi": ci_90_hi,
            "p_tost": p_value,
        })
    return results


def jonckheere_terpstra(groups: list[np.ndarray]) -> tuple[float, float]:
    """Jonckheere-Terpstra test for ordered alternatives.

    Tests H1: group_1 <= group_2 <= ... <= group_k (monotonic increase).
    Returns (J statistic, two-sided p-value).
    """
    k = len(groups)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            # Mann-Whitney U between groups[i] and groups[j]
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        J += 1
                    elif xj == xi:
                        J += 0.5

    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N = sum(ns)
    E_J = (N * N - sum(n * n for n in ns)) / 4

    # Variance formula
    term1 = N * N * (2 * N + 3)
    term2 = sum(n * n * (2 * n + 3) for n in ns)
    var_J = (term1 - term2) / 72

    if var_J <= 0:
        return J, 1.0

    z = (J - E_J) / np.sqrt(var_J)
    p = 2 * stats.norm.sf(abs(z))  # two-sided
    return z, p


def _bootstrap_jt_ci(groups: list[np.ndarray], n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for JT z-statistic."""
    rng = np.random.default_rng(seed)
    z_boot = []
    for _ in range(n_boot):
        resampled = [rng.choice(g, size=len(g), replace=True) for g in groups]
        z, _ = jonckheere_terpstra(resampled)
        z_boot.append(z)
    z_boot = np.array(z_boot)
    lo = np.percentile(z_boot, 100 * alpha / 2)
    hi = np.percentile(z_boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def jt_per_scenario(df: pd.DataFrame, metric: str, bootstrap_ci: bool = False):
    """Run Jonckheere-Terpstra per scenario across duty levels."""
    results = []
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        groups = []
        for duty in DUTY_LEVELS:
            vals = sdf[sdf["npu_load"] == duty][metric].dropna().values
            if len(vals) < 2:
                groups = []
                break
            groups.append(vals)
        if len(groups) < 3:
            continue
        z, p = jonckheere_terpstra(groups)
        N = sum(len(g) for g in groups)
        jt_r = z / np.sqrt(N) if N > 0 else 0.0
        row = {
            "scenario": scenario,
            "metric": metric,
            "jt_z": z,
            "jt_r": jt_r,
            "p_jt": p,
        }
        if bootstrap_ci:
            ci_lo, ci_hi = _bootstrap_jt_ci(groups)
            row["jt_z_ci_lo"] = ci_lo
            row["jt_z_ci_hi"] = ci_hi
        results.append(row)
    return results


def _bootstrap_spearman_ci(duties: np.ndarray, vals: np.ndarray,
                            n_boot: int = 2000, alpha: float = 0.05,
                            seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    rho_boot = []
    n = len(duties)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(duties[idx], vals[idx])
        if not np.isnan(r):
            rho_boot.append(r)
    if len(rho_boot) < 10:
        return np.nan, np.nan
    rho_boot = np.array(rho_boot)
    lo = np.percentile(rho_boot, 100 * alpha / 2)
    hi = np.percentile(rho_boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def spearman_per_scenario(df: pd.DataFrame, metric: str, bootstrap_ci: bool = False):
    """Spearman rank correlation between duty level and metric per scenario."""
    results = []
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        vals = sdf[metric].dropna()
        duties = sdf.loc[vals.index, "npu_load"]
        if len(vals) < 5:
            continue
        rho, p = stats.spearmanr(duties, vals)
        row = {
            "scenario": scenario,
            "metric": metric,
            "rho": rho,
            "p_spearman": p,
            "n": len(vals),
        }
        if bootstrap_ci:
            ci_lo, ci_hi = _bootstrap_spearman_ci(duties.values, vals.values)
            row["rho_ci_lo"] = ci_lo
            row["rho_ci_hi"] = ci_hi
        results.append(row)
    return results


def mann_whitney_per_scenario(df: pd.DataFrame, metric: str, group_col: str,
                              val_a, val_b):
    """Mann-Whitney U test between two groups per scenario."""
    results = []
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        a = sdf[sdf[group_col] == val_a][metric].dropna().values
        b = sdf[sdf[group_col] == val_b][metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial r as effect size
        n1, n2 = len(a), len(b)
        r = 1 - (2 * u) / (n1 * n2)
        cliffs_delta = (2 * u) / (n1 * n2) - 1
        results.append({
            "scenario": scenario,
            "metric": metric,
            "n_a": n1,
            "n_b": n2,
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "median_a": np.median(a),
            "median_b": np.median(b),
            "U": u,
            "r_effect": r,
            "cliffs_delta": cliffs_delta,
            "p_mw": p,
        })
    return results


# Per-scenario PCA-flagged top discriminator for cross-platform (pixhawk6c vs npu_v2 at NPU 0%)
PCA_FLAGGED_BOARDS = {
    "S01_hover_calm": "roll_mean",
    "S02_hover_moderate_wind": "roll_max_abs",
    "S03_hover_strong_wind": "vib_z_mean",
    "S04_hover_gust": "ground_speed_max",
    "S06_waypoint_calm": "vib_z_mean",
    "S07_waypoint_crosswind": "vib_z_mean",
    "S08_complex_mission": "vib_z_mean",
    "S09_takeoff_land_single": "roll_mean",
    "S11_rapid_attitude_changes": "vib_z_mean",
    "S12_tight_orbit": "vib_z_mean",
    "S13_fast_altitude_changes": "roll_mean",
    "S14_hover_degraded_gps": "vib_z_mean",
    "S15_mission_gps_dropout": "vib_z_mean",
    "S18_rtl_during_mission": "roll_mean",
    "S19_low_battery_failsafe": "roll_mean",
    "S20_sensor_failure": "roll_mean",
    "S21_mavlink_telemetry_integrity": "ekf_vel_ratio_max",
    "S23_hitl_loop_validation": "roll_mean",
    "S25_cross_platform_parity": "vib_z_mean",
    "S26_wind_gps_degradation": "roll_mean",
}

# Margins for cross-platform PCA-flagged metrics
BOARD_METRIC_MARGINS = {
    "roll_mean": 2.0,
    "roll_max_abs": 2.0,
    "vib_z_mean": 1.0,
    "ground_speed_max": 1.0,
    "ekf_vel_ratio_max": 0.5,
}


def run_pca_focused_tests(df: pd.DataFrame, out_dir: Path):
    """Run JT and Spearman on only the PCA-flagged metric per scenario.

    This is a targeted test: PCA identified the single most discriminating
    parameter per scenario, and we test whether even that parameter shows
    a monotonic dose-response trend across NPU duty levels.
    """
    # Build margin lookup from KEY_METRICS
    margin_lookup = {m: margin for m, _, margin in KEY_METRICS}

    all_jt = []
    all_sp = []
    all_tost = []

    print(f"\n{'='*72}")
    print(f"  PCA-FOCUSED TREND TESTS (one metric per scenario)")
    print(f"{'='*72}")
    print(f"  Each scenario tested on its PCA-identified top discriminator only.\n")

    for scenario in sorted(PCA_FLAGGED.keys()):
        metric = PCA_FLAGGED[scenario]
        sdf = df[df["scenario"] == scenario]
        if len(sdf) < 10:
            continue

        # JT (with bootstrap CIs for PCA-focused)
        jt_results = jt_per_scenario(sdf, metric, bootstrap_ci=True)
        for r in jt_results:
            r["pca_flagged"] = True
        all_jt.extend(jt_results)

        # Spearman (with bootstrap CIs for PCA-focused)
        sp_results = spearman_per_scenario(sdf, metric, bootstrap_ci=True)
        for r in sp_results:
            r["pca_flagged"] = True
        all_sp.extend(sp_results)

        # TOST (0% vs 100%) for completeness
        margin = margin_lookup.get(metric, 0.5)
        tost_results = tost_per_scenario(sdf, metric, margin, 0, 100)
        for r in tost_results:
            r["pca_flagged"] = True
        all_tost.extend(tost_results)

    jt_df = pd.DataFrame(all_jt)
    sp_df = pd.DataFrame(all_sp)
    tost_df = pd.DataFrame(all_tost)

    if len(jt_df) > 0:
        jt_df["p_jt_bh"] = benjamini_hochberg(jt_df["p_jt"].values)
    if len(sp_df) > 0:
        sp_df["p_spearman_bh"] = benjamini_hochberg(sp_df["p_spearman"].values)
    if len(tost_df) > 0:
        tost_df["p_tost_bh"] = benjamini_hochberg(tost_df["p_tost"].values)

    # Print JT
    print(f"\n  JONCKHEERE-TERPSTRA (PCA-flagged metric only)")
    has_jt_ci = "jt_z_ci_lo" in jt_df.columns if len(jt_df) > 0 else False
    if has_jt_ci:
        print(f"  {'Scenario':35s} {'Metric':30s} {'z':>8s} {'95% CI':>18s} {'p(BH)':>10s} {'Result':>10s}")
        print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*18} {'-'*10} {'-'*10}")
    else:
        print(f"  {'Scenario':35s} {'Metric':30s} {'z':>8s} {'p(BH)':>10s} {'Result':>10s}")
        print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in jt_df.iterrows():
        sig = "TREND" if row["p_jt_bh"] < 0.05 else "none"
        if has_jt_ci:
            ci_str = f"[{row['jt_z_ci_lo']:+.2f}, {row['jt_z_ci_hi']:+.2f}]"
            print(f"  {row['scenario']:35s} {row['metric']:30s} {row['jt_z']:>+8.3f} "
                  f"{ci_str:>18s} {row['p_jt_bh']:>10.4f} {sig:>10s}")
        else:
            print(f"  {row['scenario']:35s} {row['metric']:30s} {row['jt_z']:>+8.3f} "
                  f"{row['p_jt_bh']:>10.4f} {sig:>10s}")
    n_trend = (jt_df["p_jt_bh"] < 0.05).sum() if len(jt_df) > 0 else 0
    print(f"\n  JT Summary: {n_trend}/{len(jt_df)} monotonic trends (p_BH < 0.05)")

    # Print Spearman
    print(f"\n  SPEARMAN (PCA-flagged metric only)")
    has_sp_ci = "rho_ci_lo" in sp_df.columns if len(sp_df) > 0 else False
    if has_sp_ci:
        print(f"  {'Scenario':35s} {'Metric':30s} {'rho':>8s} {'95% CI':>18s} {'p(BH)':>10s} {'n':>5s} {'Result':>10s}")
        print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*18} {'-'*10} {'-'*5} {'-'*10}")
    else:
        print(f"  {'Scenario':35s} {'Metric':30s} {'rho':>8s} {'p(BH)':>10s} {'n':>5s} {'Result':>10s}")
        print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*5} {'-'*10}")
    for _, row in sp_df.iterrows():
        sig = "CORR" if row["p_spearman_bh"] < 0.05 else "none"
        if has_sp_ci:
            ci_str = f"[{row['rho_ci_lo']:+.2f}, {row['rho_ci_hi']:+.2f}]"
            print(f"  {row['scenario']:35s} {row['metric']:30s} {row['rho']:>+8.3f} "
                  f"{ci_str:>18s} {row['p_spearman_bh']:>10.4f} {row['n']:>5.0f} {sig:>10s}")
        else:
            print(f"  {row['scenario']:35s} {row['metric']:30s} {row['rho']:>+8.3f} "
                  f"{row['p_spearman_bh']:>10.4f} {row['n']:>5.0f} {sig:>10s}")
    n_corr = (sp_df["p_spearman_bh"] < 0.05).sum() if len(sp_df) > 0 else 0
    print(f"\n  Spearman Summary: {n_corr}/{len(sp_df)} significant correlations (p_BH < 0.05)")

    # Print TOST
    print(f"\n  TOST (PCA-flagged metric only, 0% vs 100%)")
    print(f"  {'Scenario':35s} {'Metric':30s} {'Margin':>8s} {'Diff':>10s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in tost_df.iterrows():
        sig = "EQUIV" if row["p_tost_bh"] < 0.05 else "INCONC"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['margin']:>8.3f} "
              f"{row['mean_diff']:>+10.4f} {row['p_tost_bh']:>10.4f} {sig:>10s}")
    n_equiv = (tost_df["p_tost_bh"] < 0.05).sum() if len(tost_df) > 0 else 0
    print(f"\n  TOST Summary: {n_equiv}/{len(tost_df)} equivalent (p_BH < 0.05)")

    # Overall
    print(f"\n{'='*72}")
    print(f"  PCA-FOCUSED SUMMARY (most sensitive metric per scenario)")
    print(f"{'='*72}")
    print(f"  TOST equivalence:       {n_equiv}/{len(tost_df)} confirmed within margins")
    print(f"  Jonckheere-Terpstra:    {n_trend}/{len(jt_df)} monotonic trends detected")
    print(f"  Spearman correlation:   {n_corr}/{len(sp_df)} significant correlations")
    total = len(tost_df) + len(jt_df) + len(sp_df)
    print(f"  BH FDR correction applied within each test family ({total} total tests)")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    jt_df.to_csv(out_dir / "pca_focused_jt.csv", index=False)
    sp_df.to_csv(out_dir / "pca_focused_spearman.csv", index=False)
    tost_df.to_csv(out_dir / "pca_focused_tost.csv", index=False)
    print(f"\n  Results saved to {out_dir}/")


def run_pca_focused_cross_platform(df: pd.DataFrame, out_dir: Path):
    """Run TOST and Mann-Whitney on only the PCA-flagged metric per scenario for board comparison."""
    baseline = df[df["npu_load"] == 0]
    board_a, board_b = "pixhawk6c", "npu_v2"

    print(f"\n{'='*72}")
    print(f"  PCA-FOCUSED CROSS-PLATFORM TESTS (one metric per scenario)")
    print(f"  {board_a} vs {board_b} at NPU 0%")
    print(f"{'='*72}")
    print(f"  Each scenario tested on its PCA-identified top board discriminator.\n")

    all_tost = []
    all_mw = []

    for scenario in sorted(PCA_FLAGGED_BOARDS.keys()):
        metric = PCA_FLAGGED_BOARDS[scenario]
        margin = BOARD_METRIC_MARGINS.get(metric, 0.5)
        sdf = baseline[baseline["scenario"] == scenario]
        if len(sdf) < 6:
            continue

        # TOST: remap board to npu_load so we can reuse tost_per_scenario
        sdf_mapped = sdf.assign(npu_load=sdf["board"].map({board_a: 0, board_b: 100}))
        tost_results = tost_per_scenario(sdf_mapped, metric, margin, npu_a=0, npu_b=100)
        for r in tost_results:
            r["group_a"] = board_a
            r["group_b"] = board_b
            r["pca_flagged"] = True
        all_tost.extend(tost_results)

        # Mann-Whitney
        mw_results = mann_whitney_per_scenario(sdf, metric, "board", board_a, board_b)
        for r in mw_results:
            r["pca_flagged"] = True
        all_mw.extend(mw_results)

    tost_df = pd.DataFrame(all_tost)
    mw_df = pd.DataFrame(all_mw)

    if len(tost_df) > 0:
        tost_df["p_tost_bh"] = benjamini_hochberg(tost_df["p_tost"].values)
    if len(mw_df) > 0:
        mw_df["p_mw_bh"] = benjamini_hochberg(mw_df["p_mw"].values)

    # Print TOST
    print(f"\n  TOST EQUIVALENCE (PCA-flagged metric only)")
    print(f"  {'Scenario':35s} {'Metric':25s} {'Margin':>8s} {'Diff':>10s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in tost_df.iterrows():
        sig = "EQUIV" if row["p_tost_bh"] < 0.05 else "DIFFER"
        print(f"  {row['scenario']:35s} {row['metric']:25s} {row['margin']:>8.3f} "
              f"{row['mean_diff']:>+10.4f} {row['p_tost_bh']:>10.4f} {sig:>10s}")
    n_equiv = (tost_df["p_tost_bh"] < 0.05).sum() if len(tost_df) > 0 else 0
    print(f"\n  TOST Summary: {n_equiv}/{len(tost_df)} equivalent (p_BH < 0.05)")

    # Print Mann-Whitney
    print(f"\n  MANN-WHITNEY U (PCA-flagged metric only)")
    print(f"  {'Scenario':35s} {'Metric':25s} {'r':>8s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*25} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in mw_df.iterrows():
        sig = "DIFF" if row["p_mw_bh"] < 0.05 else "same"
        print(f"  {row['scenario']:35s} {row['metric']:25s} {row['r_effect']:>+8.3f} "
              f"{row['p_mw_bh']:>10.4f} {sig:>10s}")
    n_diff = (mw_df["p_mw_bh"] < 0.05).sum() if len(mw_df) > 0 else 0
    print(f"\n  MW Summary: {n_diff}/{len(mw_df)} significantly different (p_BH < 0.05)")

    # Overlap analysis with NPU PCA-flagged
    print(f"\n  PCA DISCRIMINATOR COMPARISON (NPU duty vs Board)")
    print(f"  {'Scenario':35s} {'NPU Duty':25s} {'Board':25s} {'Same?':>6s}")
    print(f"  {'-'*35} {'-'*25} {'-'*25} {'-'*6}")
    overlap_count = 0
    for scenario in sorted(set(PCA_FLAGGED.keys()) & set(PCA_FLAGGED_BOARDS.keys())):
        npu_m = PCA_FLAGGED[scenario]
        board_m = PCA_FLAGGED_BOARDS[scenario]
        same = "YES" if npu_m == board_m else ""
        if npu_m == board_m:
            overlap_count += 1
        print(f"  {scenario:35s} {npu_m:25s} {board_m:25s} {same:>6s}")
    n_shared = len(set(PCA_FLAGGED.keys()) & set(PCA_FLAGGED_BOARDS.keys()))
    print(f"\n  Overlap: {overlap_count}/{n_shared} scenarios share the same PCA-flagged metric")

    # Summary
    print(f"\n{'='*72}")
    print(f"  PCA-FOCUSED CROSS-PLATFORM SUMMARY")
    print(f"{'='*72}")
    print(f"  TOST equivalence:  {n_equiv}/{len(tost_df)} within margins")
    print(f"  Mann-Whitney U:    {n_diff}/{len(mw_df)} significantly different")
    print(f"  Discriminator overlap with NPU duty: {overlap_count}/{n_shared}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    tost_df.to_csv(out_dir / "pca_focused_cross_platform_tost.csv", index=False)
    mw_df.to_csv(out_dir / "pca_focused_cross_platform_mw.csv", index=False)
    print(f"\n  Results saved to {out_dir}/")


def _run_board_tests(baseline: pd.DataFrame, metrics_list, board_a: str, board_b: str,
                     label: str):
    """Run TOST + Mann-Whitney between boards for a given metric set. Returns (tost_df, mw_df)."""
    all_tost = []
    all_mw = []

    for metric, unit, margin in metrics_list:
        tost_results = tost_per_scenario(
            baseline.assign(npu_load=baseline["board"].map({board_a: 0, board_b: 100})),
            metric, margin, npu_a=0, npu_b=100,
        )
        for r in tost_results:
            r["group_a"] = board_a
            r["group_b"] = board_b

        mw_results = mann_whitney_per_scenario(baseline, metric, "board", board_a, board_b)
        all_tost.extend(tost_results)
        all_mw.extend(mw_results)

    tost_df = pd.DataFrame(all_tost)
    mw_df = pd.DataFrame(all_mw)

    if len(tost_df) > 0:
        tost_df["p_tost_bh"] = benjamini_hochberg(tost_df["p_tost"].values)
    if len(mw_df) > 0:
        mw_df["p_mw_bh"] = benjamini_hochberg(mw_df["p_mw"].values)

    return tost_df, mw_df


def _print_board_results(tost_df, mw_df, board_a, board_b, label):
    """Print TOST and MW results for a board comparison."""
    # Print TOST
    print(f"\n{'='*72}")
    print(f"  TOST EQUIVALENCE ({label}): {board_a} vs {board_b}")
    print(f"  (significant = boards are EQUIVALENT within margin)")
    print(f"{'='*72}")
    print(f"  {'Scenario':35s} {'Metric':30s} {'Margin':>8s} {'Diff':>10s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in tost_df.iterrows():
        sig = "EQUIV" if row["p_tost_bh"] < 0.05 else "DIFFER"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['margin']:>8.3f} "
              f"{row['mean_diff']:>+10.4f} {row['p_tost_bh']:>10.4f} {sig:>10s}")

    n_equiv = (tost_df["p_tost_bh"] < 0.05).sum() if len(tost_df) > 0 else 0
    n_total = len(tost_df)
    print(f"\n  Summary: {n_equiv}/{n_total} equivalent, {n_total - n_equiv}/{n_total} not equivalent")

    # Print Mann-Whitney
    print(f"\n{'='*72}")
    print(f"  MANN-WHITNEY U ({label}): {board_a} vs {board_b}")
    print(f"  (significant = boards are DIFFERENT)")
    print(f"{'='*72}")
    print(f"  {'Scenario':35s} {'Metric':30s} {'r':>8s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in mw_df.iterrows():
        sig = "DIFF" if row["p_mw_bh"] < 0.05 else "same"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['r_effect']:>+8.3f} "
              f"{row['p_mw_bh']:>10.4f} {sig:>10s}")

    n_diff = (mw_df["p_mw_bh"] < 0.05).sum() if len(mw_df) > 0 else 0
    n_total_mw = len(mw_df)
    print(f"\n  Summary: {n_diff}/{n_total_mw} significantly different")

    return n_equiv, n_total, n_diff, n_total_mw


def run_cross_platform(df: pd.DataFrame, out_dir: Path):
    """Run TOST and Mann-Whitney between boards at NPU 0%.

    Two metric sets:
    1. Board PCA discriminators (5 metrics) — what PCA says separates the boards
    2. Flight-quality metrics (14 metrics, same as NPU duty) — are the boards
       equivalent on the parameters that matter for flight performance?
    """
    baseline = df[df["npu_load"] == 0]
    board_a, board_b = "pixhawk6c", "npu_v2"

    print(f"\nCross-platform comparison: {board_a} vs {board_b} at NPU 0%")
    print(f"  {board_a}: {(baseline['board'] == board_a).sum()} flights")
    print(f"  {board_b}: {(baseline['board'] == board_b).sum()} flights")

    # --- Set 1: Board PCA discriminators (5 metrics) ---
    tost_pca, mw_pca = _run_board_tests(baseline, CROSS_PLATFORM_METRICS, board_a, board_b,
                                          "board PCA discriminators")
    s1 = _print_board_results(tost_pca, mw_pca, board_a, board_b, "board PCA discriminators")

    # --- Set 2: Flight-quality metrics (18 metrics, same KEY_METRICS as NPU duty) ---
    tost_fq, mw_fq = _run_board_tests(baseline, KEY_METRICS, board_a, board_b,
                                        "flight-quality metrics")
    s2 = _print_board_results(tost_fq, mw_fq, board_a, board_b, "flight-quality metrics")

    # Overall summary
    print(f"\n{'='*72}")
    print(f"  CROSS-PLATFORM SUMMARY")
    print(f"{'='*72}")
    print(f"  Board PCA discriminators (5 metrics × 20 scenarios):")
    print(f"    TOST equivalence:  {s1[0]}/{s1[1]} within margins")
    print(f"    Mann-Whitney U:    {s1[2]}/{s1[3]} significantly different")
    print(f"  Flight-quality metrics (18 metrics × 20 scenarios):")
    print(f"    TOST equivalence:  {s2[0]}/{s2[1]} within margins")
    print(f"    Mann-Whitney U:    {s2[2]}/{s2[3]} significantly different")
    total = s1[1] + s1[3] + s2[1] + s2[3]
    print(f"  BH FDR correction applied within each metric set")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    tost_pca.to_csv(out_dir / "cross_platform_tost.csv", index=False)
    mw_pca.to_csv(out_dir / "cross_platform_mann_whitney.csv", index=False)
    tost_fq.to_csv(out_dir / "cross_platform_flight_quality_tost.csv", index=False)
    mw_fq.to_csv(out_dir / "cross_platform_flight_quality_mw.csv", index=False)
    print(f"\n  Results saved to {out_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Equivalence and trend tests")
    parser.add_argument("--csv", default=str(STATS_CSV))
    parser.add_argument("--board", default="npu_v2")
    parser.add_argument("--compare-boards", action="store_true",
                        help="Run cross-platform tests (Pixhawk 6C vs STM32N6 at NPU 0%%)")
    parser.add_argument("--pca-focused", action="store_true",
                        help="Run JT/Spearman/TOST on only the PCA-flagged metric per scenario")
    parser.add_argument("--pca-focused-boards", action="store_true",
                        help="Run TOST/MW on PCA-flagged board discriminator per scenario")
    parser.add_argument("--npu-a", type=int, default=0)
    parser.add_argument("--npu-b", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output directory for CSV results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if INCLUDED_RUNS:
        df = df[df["run_dir"].isin(INCLUDED_RUNS)]
    elif EXCLUDED_RUNS:
        df = df[~df["run_dir"].isin(EXCLUDED_RUNS)]
    df = OUTLIER_FILTER(df)
    out_dir = Path(args.output) if args.output else STATS_CSV.parent.parent / "plots" / "pca"

    if args.compare_boards:
        run_cross_platform(df, out_dir)
        print("\nDone.")
        return

    if args.pca_focused:
        df_npu = df[df["board"] == "npu_v2"] if "board" in df.columns else df
        print(f"Loaded {len(df_npu)} npu_v2 flights, {df_npu['scenario'].nunique()} scenarios")
        run_pca_focused_tests(df_npu, out_dir)
        print("\nDone.")
        return

    if args.pca_focused_boards:
        run_pca_focused_cross_platform(df, out_dir)
        print("\nDone.")
        return

    if args.board:
        df = df[df["board"] == args.board]
    print(f"Loaded {len(df)} flights, {df['scenario'].nunique()} scenarios")

    all_tost = []
    all_jt = []
    all_spearman = []

    for metric, unit, margin in KEY_METRICS:
        print(f"\n{'='*72}")
        print(f"  {metric} (margin: +/-{margin} {unit})")
        print(f"{'='*72}")

        tost_results = tost_per_scenario(df, metric, margin, args.npu_a, args.npu_b)
        jt_results = jt_per_scenario(df, metric)
        sp_results = spearman_per_scenario(df, metric)

        all_tost.extend(tost_results)
        all_jt.extend(jt_results)
        all_spearman.extend(sp_results)

    # Apply BH correction across all tests
    tost_df = pd.DataFrame(all_tost)
    jt_df = pd.DataFrame(all_jt)
    sp_df = pd.DataFrame(all_spearman)

    if len(tost_df) > 0:
        tost_df["p_tost_bh"] = benjamini_hochberg(tost_df["p_tost"].values)
    if len(jt_df) > 0:
        jt_df["p_jt_bh"] = benjamini_hochberg(jt_df["p_jt"].values)
    if len(sp_df) > 0:
        sp_df["p_spearman_bh"] = benjamini_hochberg(sp_df["p_spearman"].values)

    # Print TOST results
    print(f"\n{'='*72}")
    print(f"  TOST EQUIVALENCE: NPU {args.npu_a}% vs {args.npu_b}%")
    print(f"  (significant = difference is within margin, i.e., EQUIVALENT)")
    print(f"{'='*72}")
    print(f"  {'Scenario':35s} {'Metric':30s} {'Margin':>8s} {'Diff':>10s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in tost_df.iterrows():
        sig = "EQUIV" if row["p_tost_bh"] < 0.05 else "INCONC"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['margin']:>8.3f} "
              f"{row['mean_diff']:>+10.4f} {row['p_tost_bh']:>10.4f} {sig:>10s}")

    n_equiv = (tost_df["p_tost_bh"] < 0.05).sum() if len(tost_df) > 0 else 0
    n_total = len(tost_df)
    print(f"\n  Summary: {n_equiv}/{n_total} tests show equivalence (p_BH < 0.05)")

    # Print JT results
    print(f"\n{'='*72}")
    print(f"  JONCKHEERE-TERPSTRA TREND TEST")
    print(f"  (significant = monotonic trend with duty level)")
    print(f"{'='*72}")
    print(f"  {'Scenario':35s} {'Metric':30s} {'z':>8s} {'p(BH)':>10s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in jt_df.iterrows():
        sig = "TREND" if row["p_jt_bh"] < 0.05 else "none"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['jt_z']:>+8.3f} "
              f"{row['p_jt_bh']:>10.4f} {sig:>10s}")

    n_trend = (jt_df["p_jt_bh"] < 0.05).sum() if len(jt_df) > 0 else 0
    n_total_jt = len(jt_df)
    print(f"\n  Summary: {n_trend}/{n_total_jt} tests show monotonic trend (p_BH < 0.05)")

    # Print Spearman results
    print(f"\n{'='*72}")
    print(f"  SPEARMAN RANK CORRELATION (duty level vs metric)")
    print(f"{'='*72}")
    print(f"  {'Scenario':35s} {'Metric':30s} {'rho':>8s} {'p(BH)':>10s} {'n':>5s} {'Result':>10s}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*10} {'-'*5} {'-'*10}")
    for _, row in sp_df.iterrows():
        sig = "CORR" if row["p_spearman_bh"] < 0.05 else "none"
        print(f"  {row['scenario']:35s} {row['metric']:30s} {row['rho']:>+8.3f} "
              f"{row['p_spearman_bh']:>10.4f} {row['n']:>5.0f} {sig:>10s}")

    n_corr = (sp_df["p_spearman_bh"] < 0.05).sum() if len(sp_df) > 0 else 0
    n_total_sp = len(sp_df)
    print(f"\n  Summary: {n_corr}/{n_total_sp} tests show significant correlation (p_BH < 0.05)")

    # Overall summary
    print(f"\n{'='*72}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*72}")
    print(f"  TOST equivalence:       {n_equiv}/{n_total} confirmed within margins")
    print(f"  Jonckheere-Terpstra:    {n_trend}/{n_total_jt} monotonic trends detected")
    print(f"  Spearman correlation:   {n_corr}/{n_total_sp} significant correlations")
    print(f"  BH FDR correction applied across all {n_total + n_total_jt + n_total_sp} tests")

    # Save CSVs
    out_dir = Path(args.output) if args.output else STATS_CSV.parent.parent / "plots" / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    tost_df.to_csv(out_dir / "tost_results.csv", index=False)
    jt_df.to_csv(out_dir / "jt_results.csv", index=False)
    sp_df.to_csv(out_dir / "spearman_results.csv", index=False)
    print(f"\n  Results saved to {out_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
