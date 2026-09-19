"""Per-step flight analysis — extracts timing, tracking error, and hold
quality for each scenario step across reps, duty levels, and boards.

Metrics per step:
  - duration_s: time spent in step
  - For goto/takeoff steps:
    - transit_s: time in TRANSIT phase (before first entering tolerance)
    - settle_s: time in SETTLING phase
    - overshoot_m: max distance from target during transit
    - hold_rmse_m: position RMSE during settled portion
    - arrival_speed: speed when first entering tolerance
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INCLUDED_RUNS = [
    "logs/npu_v2_20260418_131423",   # NPU 0%
    "logs/npu_v2_20260418_185511",   # NPU 25%
    "logs/npu_v2_20260419_003555",   # NPU 50%
    "logs/npu_v2_20260419_061806",   # NPU 75%
    "logs/npu_v2_20260419_115740",   # NPU 100%
    "logs/pixhawk6c_20260420_014914",
]

# Step targets from scenario JSONs (loaded dynamically)
SCENARIO_DIR = Path("scenarios")


def load_scenario_steps(scenario_name: str) -> list[dict]:
    """Load step definitions from scenario JSON."""
    path = SCENARIO_DIR / f"{scenario_name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("steps", [])


def get_step_target(step_def: dict) -> tuple[float, float, float] | None:
    """Extract target position from a goto/takeoff step definition."""
    step_type = step_def.get("type", "")
    if step_type == "goto":
        return (step_def["x"], step_def["y"], step_def["z"])
    elif step_type == "takeoff":
        alt = step_def.get("altitude", 10.0)
        return (0.0, 0.0, -alt)  # NED: takeoff at origin, altitude is negative Z
    return None


def analyse_flight(csv_path: str, steps: list[dict]) -> list[dict]:
    """Extract per-step metrics from a single flight CSV."""
    df = pd.read_csv(csv_path)
    results = []

    for si in sorted(df["step_index"].unique()):
        si_int = int(si)
        if si_int >= len(steps):
            continue

        step_def = steps[si_int]
        step_type = step_def.get("type", "unknown")
        sub = df[df["step_index"] == si].copy()

        if len(sub) < 2:
            continue

        t0 = sub["elapsed_ms"].iloc[0]
        duration_s = (sub["elapsed_ms"].iloc[-1] - t0) / 1000.0

        row = {
            "step_index": si_int,
            "step_type": step_type,
            "duration_s": duration_s,
        }

        target = get_step_target(step_def)
        if target is not None:
            tx, ty, tz = target
            dist = np.sqrt((sub["x"] - tx)**2 + (sub["y"] - ty)**2 + (sub["z"] - tz)**2)
            tolerance = step_def.get("tolerance", 0.5)

            # Find transit/settle boundary: first time distance < tolerance
            within = dist.values < tolerance
            if within.any():
                first_within_idx = np.argmax(within)
                transit_samples = sub.iloc[:first_within_idx]
                settle_samples = sub.iloc[first_within_idx:]

                transit_s = (transit_samples["elapsed_ms"].iloc[-1] - t0) / 1000.0 if len(transit_samples) > 1 else 0.0
                settle_s = duration_s - transit_s

                row["transit_s"] = transit_s
                row["settle_s"] = settle_s
                row["overshoot_m"] = float(dist.iloc[:first_within_idx].max()) if first_within_idx > 0 else 0.0
                row["hold_rmse_m"] = float(np.sqrt((dist.iloc[first_within_idx:]**2).mean()))

                # Arrival speed
                if first_within_idx > 0 and first_within_idx < len(sub):
                    row["arrival_speed"] = float(sub["ground_speed"].iloc[first_within_idx])
                else:
                    row["arrival_speed"] = 0.0
            else:
                # Never reached target
                row["transit_s"] = duration_s
                row["settle_s"] = 0.0
                row["overshoot_m"] = float(dist.max())
                row["hold_rmse_m"] = np.nan
                row["arrival_speed"] = np.nan

            row["mean_dist_m"] = float(dist.mean())
            row["final_dist_m"] = float(dist.iloc[-1])

        results.append(row)

    return results


def find_csvs(run_dir: str, scenario: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "*", scenario, "*.csv")))


def main():
    parser = argparse.ArgumentParser(description="Per-step flight analysis")
    parser.add_argument("--output", type=Path, default=Path("logs/plots/pca/step_analysis"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Identify runs
    n6_runs = {
        0: INCLUDED_RUNS[0], 25: INCLUDED_RUNS[1], 50: INCLUDED_RUNS[2],
        75: INCLUDED_RUNS[3], 100: INCLUDED_RUNS[4],
    }
    pix_run = INCLUDED_RUNS[5]

    # Discover scenarios
    board_dirs = [d for d in glob.glob(os.path.join(INCLUDED_RUNS[0], "*")) if os.path.isdir(d)]
    scenarios = sorted([os.path.basename(d) for d in glob.glob(os.path.join(board_dirs[0], "*")) if os.path.isdir(d)])

    print(f"Scenarios: {len(scenarios)}")
    all_rows = []

    for scenario in scenarios:
        steps = load_scenario_steps(scenario)
        if not steps:
            print(f"  {scenario}: no scenario JSON found, skipping")
            continue

        # N6 runs
        for duty, run_dir in n6_runs.items():
            for csv_path in find_csvs(run_dir, scenario):
                rep = csv_path.split("rep")[1][:2]
                flight_steps = analyse_flight(csv_path, steps)
                for row in flight_steps:
                    row["scenario"] = scenario
                    row["board"] = "npu_v2"
                    row["npu_load"] = duty
                    row["repetition"] = int(rep)
                    all_rows.append(row)

        # Pixhawk
        for csv_path in find_csvs(pix_run, scenario):
            rep = csv_path.split("rep")[1][:2]
            flight_steps = analyse_flight(csv_path, steps)
            for row in flight_steps:
                row["scenario"] = scenario
                row["board"] = "pixhawk6c"
                row["npu_load"] = 0
                row["repetition"] = int(rep)
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output / "step_metrics.csv", index=False)
    print(f"\nTotal step records: {len(df)}")

    # Filter to goto/takeoff steps only
    flight_steps = df[df["step_type"].isin(["goto", "takeoff"])].copy()

    # --- Within-platform: duty effect on per-step metrics ---
    print("\n" + "=" * 80)
    print("  NPU DUTY EFFECT ON PER-STEP METRICS (N6 only, Kruskal-Wallis across 5 duty levels)")
    print("=" * 80)

    step_metrics = ["duration_s", "transit_s", "hold_rmse_m", "overshoot_m", "arrival_speed"]
    n6_steps = flight_steps[flight_steps["board"] == "npu_v2"]

    kw_results = []
    for scenario in scenarios:
        for si in sorted(n6_steps[n6_steps["scenario"] == scenario]["step_index"].unique()):
            for metric in step_metrics:
                sdf = n6_steps[(n6_steps["scenario"] == scenario) & (n6_steps["step_index"] == si)]
                groups = [g[metric].dropna().values for _, g in sdf.groupby("npu_load")]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) >= 2:
                    H, p = stats.kruskal(*groups)
                    kw_results.append({
                        "scenario": scenario, "step_index": si,
                        "step_type": sdf["step_type"].iloc[0],
                        "metric": metric, "H": H, "p": p,
                    })

    kw_df = pd.DataFrame(kw_results)
    sig = kw_df[kw_df["p"] < 0.05]
    print(f"\n  Total tests: {len(kw_df)}")
    print(f"  Significant (p < 0.05): {len(sig)}")
    if len(sig) > 0:
        print(f"\n  Significant results:")
        for _, r in sig.iterrows():
            print(f"    {r['scenario']:35s} step {r['step_index']:2.0f} ({r['step_type']:8s}) "
                  f"{r['metric']:15s} H={r['H']:.2f} p={r['p']:.4f}")

    # --- Cross-platform: N6 0% vs Pixhawk per-step ---
    print("\n" + "=" * 80)
    print("  CROSS-PLATFORM PER-STEP COMPARISON (N6 0% vs Pixhawk, Mann-Whitney)")
    print("=" * 80)

    n6_0 = flight_steps[(flight_steps["board"] == "npu_v2") & (flight_steps["npu_load"] == 0)]
    pix = flight_steps[flight_steps["board"] == "pixhawk6c"]

    mw_results = []
    for scenario in scenarios:
        for si in sorted(n6_0[n6_0["scenario"] == scenario]["step_index"].unique()):
            for metric in step_metrics:
                a = n6_0[(n6_0["scenario"] == scenario) & (n6_0["step_index"] == si)][metric].dropna()
                b = pix[(pix["scenario"] == scenario) & (pix["step_index"] == si)][metric].dropna()
                if len(a) >= 2 and len(b) >= 2:
                    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    mw_results.append({
                        "scenario": scenario, "step_index": si,
                        "step_type": a.index[0] if False else flight_steps.loc[a.index[0], "step_type"],
                        "metric": metric,
                        "n6_median": a.median(), "pix_median": b.median(),
                        "diff": a.median() - b.median(),
                        "U": U, "p": p,
                    })

    mw_df = pd.DataFrame(mw_results)
    sig_xp = mw_df[mw_df["p"] < 0.05]
    print(f"\n  Total tests: {len(mw_df)}")
    print(f"  Significant (p < 0.05): {len(sig_xp)}")
    if len(sig_xp) > 0:
        print(f"\n  Significant results (top 20 by |diff|):")
        sig_xp_sorted = sig_xp.reindex(sig_xp["diff"].abs().sort_values(ascending=False).index)
        for _, r in sig_xp_sorted.head(20).iterrows():
            print(f"    {r['scenario']:35s} step {r['step_index']:2.0f} "
                  f"{r['metric']:15s} N6={r['n6_median']:8.3f} Pix={r['pix_median']:8.3f} "
                  f"diff={r['diff']:+8.3f} p={r['p']:.4f}")

    # Save
    kw_df.to_csv(args.output / "kw_duty_effect.csv", index=False)
    mw_df.to_csv(args.output / "mw_cross_platform.csv", index=False)

    # Summary table: per-scenario, goto steps only, show hold_rmse
    print("\n" + "=" * 80)
    print("  HOLD RMSE AT WAYPOINTS (metres, median across reps)")
    print("=" * 80)
    goto_steps = flight_steps[flight_steps["step_type"] == "goto"]
    print(f"\n  {'Scenario':35s} {'Step':>5s} {'0%':>7s} {'25%':>7s} {'50%':>7s} {'75%':>7s} {'100%':>7s} {'Pix':>7s}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for scenario in scenarios:
        sc_data = goto_steps[goto_steps["scenario"] == scenario]
        for si in sorted(sc_data["step_index"].unique()):
            row = f"  {scenario:35s} {si:5.0f}"
            for duty in [0, 25, 50, 75, 100]:
                v = sc_data[(sc_data["step_index"] == si) & (sc_data["board"] == "npu_v2") &
                            (sc_data["npu_load"] == duty)]["hold_rmse_m"]
                row += f" {v.median():7.3f}" if len(v) > 0 else f" {'—':>7s}"
            v = sc_data[(sc_data["step_index"] == si) & (sc_data["board"] == "pixhawk6c")]["hold_rmse_m"]
            row += f" {v.median():7.3f}" if len(v) > 0 else f" {'—':>7s}"
            print(row)

    print(f"\n  Results saved to {args.output}/")


if __name__ == "__main__":
    main()
