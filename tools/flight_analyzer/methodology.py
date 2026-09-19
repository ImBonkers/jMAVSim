"""Methodological validation analyses for flight stats.

Implements split-half PCA validation, TOST sensitivity analysis,
post-hoc power analysis, Shapiro-Wilk normality tests, and outlier
sensitivity analysis.

Run standalone:
    PYTHONPATH=tools tools/flight_analyzer/.venv/bin/python -m flight_analyzer.methodology
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import ttost_ind

from flight_analyzer.equivalence_tests import (
    jonckheere_terpstra, PCA_FLAGGED, KEY_METRICS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "duration_s", "origin_dist_mean", "pos_drift_max", "pos_drift_std",
    "alt_error_mean", "alt_error_max",
    "roll_mean", "roll_std", "roll_max_abs",
    "pitch_mean", "pitch_std", "pitch_max_abs",
    "ground_speed_mean", "ground_speed_max",
    "vertical_speed_mean", "vertical_speed_max_abs",
    "roll_err_rms", "pitch_err_rms", "yaw_err_rms",
    "ekf_vel_ratio_mean", "ekf_vel_ratio_max",
    "ekf_pos_horiz_ratio_mean", "ekf_pos_horiz_accuracy_mean",
    "ekf_pos_vert_accuracy_mean",
    "vib_x_mean", "vib_y_mean", "vib_z_mean", "vib_magnitude_max",
    "clipping_total",
    "comm_drop_rate_max", "comm_errors_max",
]

# KEY_METRICS imported from equivalence_tests (17 metrics, canonical source)

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
OUTLIER_SCENARIO = None
OUTLIER_RUN = None
OUTLIER_REP = None

DUTY_LEVELS = [0, 25, 50, 75, 100]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(csv_path: Path, exclude_outlier: bool = True) -> pd.DataFrame:
    """Load CSV, apply exclusions, filter to npu_v2 board."""
    df = pd.read_csv(csv_path)
    if INCLUDED_RUNS:
        df = df[df["run_dir"].isin(INCLUDED_RUNS)]
    elif EXCLUDED_RUNS:
        df = df[~df["run_dir"].isin(EXCLUDED_RUNS)]
    if exclude_outlier and OUTLIER_SCENARIO:
        df = df[~(
            (df["scenario"] == OUTLIER_SCENARIO)
            & (df["run_dir"] == OUTLIER_RUN)
            & (df["repetition"] == OUTLIER_REP)
        )]
    df = df[df["board"] == "npu_v2"]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two arrays (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                 / (na + nb - 2))
    if sp < 1e-12:
        return 0.0
    return (b.mean() - a.mean()) / sp


def _top_discriminator_pca(sub_df: pd.DataFrame) -> str | None:
    """Run PCA on sub_df and return the feature with highest |Cohen's d|
    between NPU 0% and 100% in scaled space."""
    features = [c for c in FEATURE_COLS if c in sub_df.columns]
    X = sub_df[features].copy()
    X = X.loc[:, X.std() > 1e-10]
    used = list(X.columns)
    if len(used) < 2:
        return None
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=used, index=sub_df.index)

    mask_0 = sub_df["npu_load"].values == 0
    mask_100 = sub_df["npu_load"].values == 100
    if mask_0.sum() == 0 or mask_100.sum() == 0:
        return None

    effects = {}
    for feat in used:
        effects[feat] = abs(_cohens_d(
            X_scaled.loc[mask_0, feat].values,
            X_scaled.loc[mask_100, feat].values,
        ))
    return max(effects, key=effects.get)


def _bh_correction(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    # Enforce monotonicity (step-up)
    adjusted = np.minimum(adjusted, 1.0)
    # Walk from largest rank down
    adj_sorted_idx = np.argsort(ranked)[::-1]
    running_min = 1.0
    result = adjusted.copy()
    for idx in adj_sorted_idx:
        running_min = min(running_min, adjusted[idx])
        result[idx] = running_min
    return result


# ---------------------------------------------------------------------------
# 1. Split-half PCA validation
# ---------------------------------------------------------------------------


def split_half_pca(df: pd.DataFrame, output_dir: Path, n_iter: int = 100,
                   seed: int = 42) -> pd.DataFrame:
    """Split-half PCA concordance analysis."""
    print("\n" + "=" * 72)
    print("  1. SPLIT-HALF PCA VALIDATION")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    scenarios = sorted(df["scenario"].unique())
    results = []

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario].reset_index(drop=True)
        counts_per_duty = {}
        for duty in DUTY_LEVELS:
            counts_per_duty[duty] = (sdf["npu_load"] == duty).sum()

        # Need at least 6 per group to split 3+3
        if counts_per_duty.get(0, 0) < 6 or counts_per_duty.get(100, 0) < 6:
            print(f"  {scenario}: skipped (insufficient data for split-half)")
            continue

        top_metrics_half1 = []
        top_metrics_half2 = []

        for _ in range(n_iter):
            # For each duty level, randomly split rows into two halves
            half1_idx = []
            half2_idx = []
            for duty in DUTY_LEVELS:
                duty_idx = sdf.index[sdf["npu_load"] == duty].tolist()
                rng.shuffle(duty_idx)
                mid = len(duty_idx) // 2
                half1_idx.extend(duty_idx[:mid])
                half2_idx.extend(duty_idx[mid:])

            h1 = sdf.loc[half1_idx].reset_index(drop=True)
            h2 = sdf.loc[half2_idx].reset_index(drop=True)

            t1 = _top_discriminator_pca(h1)
            t2 = _top_discriminator_pca(h2)

            if t1 is not None:
                top_metrics_half1.append(t1)
            if t2 is not None:
                top_metrics_half2.append(t2)

        # Concordance: how often do the two halves agree?
        n_valid = min(len(top_metrics_half1), len(top_metrics_half2))
        if n_valid == 0:
            continue
        concordant = sum(
            1 for a, b in zip(top_metrics_half1, top_metrics_half2) if a == b
        )
        concordance_rate = concordant / n_valid

        # Most common metrics across all iterations (both halves combined)
        from collections import Counter
        combined = top_metrics_half1 + top_metrics_half2
        counts = Counter(combined)
        most_common = counts.most_common(2)
        mc1 = most_common[0][0] if len(most_common) > 0 else ""
        mc2 = most_common[1][0] if len(most_common) > 1 else ""

        results.append({
            "scenario": scenario,
            "concordance_rate": round(concordance_rate, 3),
            "most_common_metric": mc1,
            "second_most_common": mc2,
        })
        print(f"  {scenario}: concordance={concordance_rate:.1%}  "
              f"top={mc1}, 2nd={mc2}")

    result_df = pd.DataFrame(results)
    out_path = output_dir / "split_half_concordance.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    if len(result_df) > 0:
        mean_conc = result_df["concordance_rate"].mean()
        print(f"  Mean concordance across scenarios: {mean_conc:.1%}")

    return result_df


# ---------------------------------------------------------------------------
# 2. TOST sensitivity analysis
# ---------------------------------------------------------------------------


def _tost_sensitivity_core(df: pd.DataFrame, group_col: str, group_a, group_b,
                           output_dir: Path, label: str, fname: str) -> pd.DataFrame:
    """TOST at 0.5x, 1x, 2x margins between two groups."""
    multipliers = [0.5, 1.0, 2.0]
    scenarios = sorted(df["scenario"].unique())
    rows = []

    for mult in multipliers:
        pvals = []
        meta = []

        for scenario in scenarios:
            sdf = df[df["scenario"] == scenario]
            ga = sdf[sdf[group_col] == group_a]
            gb = sdf[sdf[group_col] == group_b]
            if len(ga) < 2 or len(gb) < 2:
                continue

            for metric, unit, base_margin in KEY_METRICS:
                if metric not in sdf.columns:
                    continue
                margin = base_margin * mult
                a = ga[metric].dropna().values
                b = gb[metric].dropna().values
                if len(a) < 2 or len(b) < 2:
                    continue
                try:
                    p_tost, _, _ = ttost_ind(a, b, -margin, margin)
                except Exception:
                    p_tost = 1.0
                pvals.append(p_tost)
                meta.append({
                    "multiplier": mult,
                    "scenario": scenario,
                    "metric": metric,
                    "margin": margin,
                    "p_tost": p_tost,
                })

        # BH correction within this multiplier level
        pvals_arr = np.array(pvals)
        adj = _bh_correction(pvals_arr)
        for i, m in enumerate(meta):
            m["p_adj"] = adj[i]
            m["equivalent"] = adj[i] < 0.05
        rows.extend(meta)

    result_df = pd.DataFrame(rows)
    out_path = output_dir / fname
    result_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Summary
    for mult in multipliers:
        sub = result_df[result_df["multiplier"] == mult]
        n_eq = sub["equivalent"].sum()
        n_total = len(sub)
        print(f"  {mult:.1f}x margin: {n_eq}/{n_total} equivalent "
              f"({n_eq/n_total:.1%})" if n_total > 0 else f"  {mult:.1f}x margin: no tests")

    return result_df


def tost_sensitivity(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """TOST margin sensitivity: NPU 0% vs 100%."""
    print("\n" + "=" * 72)
    print("  2a. TOST SENSITIVITY — NPU 0% vs 100%")
    print("=" * 72)
    return _tost_sensitivity_core(df, "npu_load", 0, 100, output_dir,
                                  "NPU 0% vs 100%", "tost_sensitivity.csv")


def tost_sensitivity_cross_platform(df_all: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """TOST margin sensitivity: Pixhawk 6C vs STM32N6 at NPU 0%."""
    print("\n" + "=" * 72)
    print("  2b. TOST SENSITIVITY — Cross-platform (Pixhawk vs N6 @ 0%)")
    print("=" * 72)
    baseline = df_all[df_all["npu_load"] == 0].copy()
    return _tost_sensitivity_core(baseline, "board", "pixhawk6c", "npu_v2", output_dir,
                                  "Pixhawk vs N6", "tost_sensitivity_cross_platform.csv")


# ---------------------------------------------------------------------------
# 3. Post-hoc TOST power analysis
# ---------------------------------------------------------------------------


def _tost_power_core(df: pd.DataFrame, group_col: str, group_a, group_b,
                     output_dir: Path, label: str, fname: str) -> pd.DataFrame:
    """Post-hoc power analysis for TOST equivalence tests between two groups."""
    scenarios = sorted(df["scenario"].unique())
    rows = []

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario]
        ga = sdf[sdf[group_col] == group_a]
        gb = sdf[sdf[group_col] == group_b]
        if len(ga) < 2 or len(gb) < 2:
            continue

        for metric, unit, margin in KEY_METRICS:
            if metric not in sdf.columns:
                continue
            a = ga[metric].dropna().values
            b = gb[metric].dropna().values
            na, nb = len(a), len(b)
            if na < 2 or nb < 2:
                continue

            # Pooled SD
            sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2
                          + (nb - 1) * b.std(ddof=1) ** 2)
                         / (na + nb - 2))
            se = sp * np.sqrt(1.0 / na + 1.0 / nb)
            df_val = na + nb - 2

            if se < 1e-15:
                power_val = 1.0
            else:
                t_crit = t_dist.ppf(1 - 0.05, df_val)  # one-sided alpha
                ncp = margin / se
                # Power of TOST when true difference = 0
                power_val = float(t_dist.sf(t_crit - ncp, df_val))

            rows.append({
                "scenario": scenario,
                "metric": metric,
                "margin": margin,
                "n1": na,
                "n2": nb,
                "pooled_sd": round(sp, 6),
                "se": round(se, 6),
                "ncp": round(margin / se, 3) if se > 1e-15 else np.inf,
                "power": round(power_val, 4),
            })

    result_df = pd.DataFrame(rows)
    out_path = output_dir / fname
    result_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Per-metric summary
    if len(result_df) > 0:
        print(f"\n  {'Metric':40s}  {'Median':>8s}  {'Min':>8s}  {'<0.8':>5s}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*5}")
        for metric, _, _ in KEY_METRICS:
            sub = result_df[result_df["metric"] == metric]
            if len(sub) == 0:
                continue
            med = sub["power"].median()
            mn = sub["power"].min()
            underpowered = (sub["power"] < 0.8).sum()
            print(f"  {metric:40s}  {med:8.3f}  {mn:8.3f}  {underpowered:5d}")

        total_under = (result_df["power"] < 0.8).sum()
        total = len(result_df)
        print(f"\n  Overall: {total_under}/{total} underpowered tests "
              f"({total_under/total:.1%})")

    return result_df


def tost_power(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Post-hoc power: NPU 0% vs 100%."""
    print("\n" + "=" * 72)
    print("  3a. POST-HOC TOST POWER — NPU 0% vs 100%")
    print("=" * 72)
    return _tost_power_core(df, "npu_load", 0, 100, output_dir,
                            "NPU 0% vs 100%", "tost_power.csv")


def tost_power_cross_platform(df_all: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Post-hoc power: Pixhawk 6C vs STM32N6 at NPU 0%."""
    print("\n" + "=" * 72)
    print("  3b. POST-HOC TOST POWER — Cross-platform (Pixhawk vs N6 @ 0%)")
    print("=" * 72)
    baseline = df_all[df_all["npu_load"] == 0].copy()
    return _tost_power_core(baseline, "board", "pixhawk6c", "npu_v2", output_dir,
                            "Pixhawk vs N6", "tost_power_cross_platform.csv")


# ---------------------------------------------------------------------------
# 4. Shapiro-Wilk normality tests
# ---------------------------------------------------------------------------


def shapiro_wilk(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Shapiro-Wilk normality test for each scenario x metric x duty cell."""
    print("\n" + "=" * 72)
    print("  4. SHAPIRO-WILK NORMALITY TESTS")
    print("=" * 72)

    scenarios = sorted(df["scenario"].unique())
    rows = []

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario]
        for metric, unit, margin in KEY_METRICS:
            if metric not in sdf.columns:
                continue
            for duty in DUTY_LEVELS:
                cell = sdf[sdf["npu_load"] == duty][metric].dropna().values
                if len(cell) < 3:
                    continue
                try:
                    stat, pval = shapiro(cell)
                except Exception:
                    stat, pval = np.nan, np.nan
                rows.append({
                    "scenario": scenario,
                    "metric": metric,
                    "duty": duty,
                    "n": len(cell),
                    "shapiro_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
                    "p_value": round(pval, 6) if not np.isnan(pval) else np.nan,
                    "reject_normality": pval < 0.05 if not np.isnan(pval) else False,
                })

    result_df = pd.DataFrame(rows)
    out_path = output_dir / "shapiro_wilk.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    n_reject = result_df["reject_normality"].sum()
    n_total = len(result_df)
    pct = n_reject / n_total * 100 if n_total > 0 else 0
    print(f"\n  {n_reject}/{n_total} cells reject normality ({pct:.1f}%)")

    # Per-metric summary
    if len(result_df) > 0:
        print(f"\n  {'Metric':40s}  {'Reject':>7s}  {'Total':>6s}  {'%':>7s}")
        print(f"  {'-'*40}  {'-'*7}  {'-'*6}  {'-'*7}")
        for metric, _, _ in KEY_METRICS:
            sub = result_df[result_df["metric"] == metric]
            if len(sub) == 0:
                continue
            rej = sub["reject_normality"].sum()
            tot = len(sub)
            print(f"  {metric:40s}  {rej:7d}  {tot:6d}  {rej/tot*100:6.1f}%")

    return result_df


# ---------------------------------------------------------------------------
# 5. Outlier sensitivity
# ---------------------------------------------------------------------------


def outlier_sensitivity(csv_path: Path, output_dir: Path) -> pd.DataFrame:
    """Compare TOST results with and without the S18 outlier."""
    print("\n" + "=" * 72)
    print("  5. OUTLIER SENSITIVITY ANALYSIS")
    print("=" * 72)

    df_with = load_data(csv_path, exclude_outlier=False)
    df_without = load_data(csv_path, exclude_outlier=True)

    def _run_tost(dframe: pd.DataFrame) -> dict[tuple[str, str], bool]:
        """Run TOST for all scenario x metric combos, return equivalence dict."""
        scenarios = sorted(dframe["scenario"].unique())
        pvals = []
        keys = []
        for scenario in scenarios:
            sdf = dframe[dframe["scenario"] == scenario]
            g0 = sdf[sdf["npu_load"] == 0]
            g100 = sdf[sdf["npu_load"] == 100]
            if len(g0) < 2 or len(g100) < 2:
                continue
            for metric, unit, margin in KEY_METRICS:
                if metric not in sdf.columns:
                    continue
                a = g0[metric].dropna().values
                b = g100[metric].dropna().values
                if len(a) < 2 or len(b) < 2:
                    continue
                try:
                    p_tost, _, _ = ttost_ind(a, b, -margin, margin)
                except Exception:
                    p_tost = 1.0
                pvals.append(p_tost)
                keys.append((scenario, metric))

        adj = _bh_correction(np.array(pvals))
        return {k: adj[i] < 0.05 for i, k in enumerate(keys)}

    equiv_with = _run_tost(df_with)
    equiv_without = _run_tost(df_without)

    # Build comparison
    all_keys = sorted(set(equiv_with.keys()) | set(equiv_without.keys()))
    rows = []
    changed = []
    for key in all_keys:
        ew = equiv_with.get(key, None)
        ewo = equiv_without.get(key, None)
        rows.append({
            "scenario": key[0],
            "metric": key[1],
            "equiv_with_outlier": ew,
            "equiv_without_outlier": ewo,
            "changed": ew != ewo,
        })
        if ew != ewo:
            changed.append(key)

    result_df = pd.DataFrame(rows)
    out_path = output_dir / "outlier_sensitivity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    n_with = sum(1 for v in equiv_with.values() if v)
    n_without = sum(1 for v in equiv_without.values() if v)
    print(f"\n  With outlier:    {n_with}/{len(equiv_with)} equivalent")
    print(f"  Without outlier: {n_without}/{len(equiv_without)} equivalent")
    print(f"  Changed results: {len(changed)}")
    for sc, met in changed:
        w = equiv_with.get((sc, met))
        wo = equiv_without.get((sc, met))
        label_w = "equiv" if w else "non-equiv"
        label_wo = "equiv" if wo else "non-equiv"
        print(f"    {sc} / {met}: {label_w} -> {label_wo}")

    return result_df


# ---------------------------------------------------------------------------
# 6. Permutation test for duty-ordering confound
# ---------------------------------------------------------------------------


def _identify_sessions(sdf: pd.DataFrame) -> list[list[str]]:
    """Identify day-sessions: groups of 5 consecutive runs (one per duty level).

    Each session is a list of 5 run_dirs ordered by timestamp (which matches
    the duty ordering 0→25→50→75→100).
    """
    # Get unique runs sorted by name (which encodes timestamp)
    runs = sorted(sdf["run_dir"].unique())
    duty_per_run = sdf.groupby("run_dir")["npu_load"].first().to_dict()

    sessions = []
    session = []
    for run in runs:
        session.append(run)
        if len(session) == 5:
            sessions.append(session)
            session = []
    if session:
        sessions.append(session)  # partial session
    return sessions


def permutation_ordering_test(df: pd.DataFrame, output_dir: Path,
                               n_perm: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Permutation test: shuffle duty labels within each day-session.

    For each scenario × PCA-flagged metric, computes the observed JT z-statistic,
    then generates a null distribution by permuting the duty-level assignment
    within each day-session (preserving temporal structure but breaking
    the duty-time aliasing). Reports a two-sided permutation p-value.
    """
    print("\n" + "=" * 72)
    print("  6. PERMUTATION TEST FOR DUTY-ORDERING CONFOUND")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    results = []

    for scenario in sorted(PCA_FLAGGED.keys()):
        metric = PCA_FLAGGED[scenario]
        sdf = df[df["scenario"] == scenario].copy()
        if len(sdf) < 10:
            continue

        # Observed JT
        groups_obs = []
        for duty in DUTY_LEVELS:
            vals = sdf[sdf["npu_load"] == duty][metric].dropna().values
            if len(vals) < 2:
                break
            groups_obs.append(vals)
        if len(groups_obs) < 5:
            continue

        z_obs, _ = jonckheere_terpstra(groups_obs)

        # Identify sessions for this scenario
        sessions = _identify_sessions(sdf)

        # Permutation null distribution
        z_null = []
        for _ in range(n_perm):
            # Shuffle duty labels within each session
            sdf_perm = sdf.copy()
            for session_runs in sessions:
                mask = sdf_perm["run_dir"].isin(session_runs)
                session_rows = sdf_perm.loc[mask]
                # Get the duty levels present in this session
                duties_in_session = session_rows.groupby("run_dir")["npu_load"].first()
                # Shuffle the mapping: which run gets which duty
                shuffled_duties = duties_in_session.values.copy()
                rng.shuffle(shuffled_duties)
                run_to_new_duty = dict(zip(duties_in_session.index, shuffled_duties))
                # Apply
                for run_dir, new_duty in run_to_new_duty.items():
                    sdf_perm.loc[sdf_perm["run_dir"] == run_dir, "npu_load"] = new_duty

            # Recompute JT on permuted data
            groups_perm = []
            for duty in DUTY_LEVELS:
                vals = sdf_perm[sdf_perm["npu_load"] == duty][metric].dropna().values
                if len(vals) < 2:
                    break
                groups_perm.append(vals)
            if len(groups_perm) < 5:
                continue

            z_perm, _ = jonckheere_terpstra(groups_perm)
            z_null.append(z_perm)

        z_null = np.array(z_null)
        # Two-sided permutation p-value
        p_perm = np.mean(np.abs(z_null) >= np.abs(z_obs))

        results.append({
            "scenario": scenario,
            "metric": metric,
            "jt_z_observed": round(z_obs, 4),
            "z_null_mean": round(z_null.mean(), 4),
            "z_null_std": round(z_null.std(), 4),
            "z_null_2.5pct": round(np.percentile(z_null, 2.5), 4),
            "z_null_97.5pct": round(np.percentile(z_null, 97.5), 4),
            "p_permutation": round(p_perm, 4),
            "significant": p_perm < 0.05,
        })

        sig_str = "SIG" if p_perm < 0.05 else "ns"
        print(f"  {scenario:35s} {metric:30s} z={z_obs:+.3f}  "
              f"null=[{np.percentile(z_null, 2.5):+.2f}, {np.percentile(z_null, 97.5):+.2f}]  "
              f"p={p_perm:.4f} {sig_str}")

    result_df = pd.DataFrame(results)
    out_path = output_dir / "permutation_ordering_test.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    if len(result_df) > 0:
        n_sig = result_df["significant"].sum()
        n_total = len(result_df)
        print(f"\n  {n_sig}/{n_total} scenarios significant (p < 0.05)")
        print(f"  If duty-ordering confound were driving the JT trends,")
        print(f"  permutation p-values should be large (trend disappears")
        print(f"  when duty-time aliasing is broken).")

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Methodological validation analyses for flight stats"
    )
    parser.add_argument(
        "--csv",
        default="logs/stats/all_flight_stats.csv",
        help="Path to all_flight_stats.csv (default: logs/stats/all_flight_stats.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/plots/pca/methodology/",
        help="Output directory (default: logs/plots/pca/methodology/)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path, exclude_outlier=True)
    print(f"Loaded {len(df)} flights (npu_v2, outlier excluded), "
          f"{df['scenario'].nunique()} scenarios")

    # Load all data (both boards) for cross-platform analyses
    df_all = pd.read_csv(csv_path)
    if INCLUDED_RUNS:
        df_all = df_all[df_all["run_dir"].isin(INCLUDED_RUNS)]
    print(f"Loaded {len(df_all)} flights (all boards) for cross-platform analyses")

    # 1. Split-half PCA validation
    split_half_pca(df, output_dir)

    # 2. TOST sensitivity analysis
    tost_sensitivity(df, output_dir)
    tost_sensitivity_cross_platform(df_all, output_dir)

    # 3. Post-hoc TOST power analysis
    tost_power(df, output_dir)
    tost_power_cross_platform(df_all, output_dir)

    # 4. Shapiro-Wilk normality tests
    shapiro_wilk(df, output_dir)

    # 5. Outlier sensitivity (needs raw CSV path to load both with/without)
    outlier_sensitivity(csv_path, output_dir)

    # 6. Permutation test for duty-ordering confound
    permutation_ordering_test(df, output_dir)

    print("\n" + "=" * 72)
    print("  All methodology analyses complete.")
    print(f"  Results saved to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
