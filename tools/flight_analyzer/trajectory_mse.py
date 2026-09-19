"""Trajectory MSE analysis — compares flight paths across reps and boards.

Two metrics:
1. Within-group consistency: per-duty-level trajectory variance (how repeatable?)
2. Cross-platform MSE: N6 mean trajectory vs Pixhawk mean trajectory
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

INCLUDED_RUNS = [
    "logs/npu_v2_20260418_131423",   # NPU 0%
    "logs/npu_v2_20260418_185511",   # NPU 25%
    "logs/npu_v2_20260419_003555",   # NPU 50%
    "logs/npu_v2_20260419_061806",   # NPU 75%
    "logs/npu_v2_20260419_115740",   # NPU 100%
    "logs/pixhawk6c_20260420_014914",
]

# Steps before flight (reboot, setNpu, wait, arm) — skip these
FLIGHT_START_STEP = 4  # takeoff


def load_trajectory(csv_path: str, resample_hz: int = 50) -> pd.DataFrame | None:
    """Load a flight CSV and return resampled x,y,z trajectory from takeoff onwards.

    Returns DataFrame with columns: t, x, y, z (t in seconds from takeoff start).
    """
    df = pd.read_csv(csv_path)

    # Filter to flight steps (takeoff onwards, exclude disarm)
    flight = df[df["step_index"] >= FLIGHT_START_STEP].copy()
    if len(flight) < 10:
        return None

    # Remove the very last step (disarm, usually 1 row)
    max_step = flight["step_index"].max()
    flight = flight[flight["step_index"] < max_step]
    if len(flight) < 10:
        return None

    # Time from takeoff start in seconds
    t0 = flight["elapsed_ms"].iloc[0]
    flight["t"] = (flight["elapsed_ms"] - t0) / 1000.0

    # Resample to common grid
    t_max = flight["t"].iloc[-1]
    dt = 1.0 / resample_hz
    t_grid = np.arange(0, t_max, dt)

    if len(t_grid) < 5:
        return None

    resampled = pd.DataFrame({"t": t_grid})
    for col in ["x", "y", "z"]:
        f = interpolate.interp1d(flight["t"].values, flight[col].values,
                                 kind="linear", fill_value="extrapolate")
        resampled[col] = f(t_grid)

    return resampled


def align_trajectories(trajectories: list[pd.DataFrame]) -> list[np.ndarray]:
    """Truncate all trajectories to the shortest common length.

    Returns list of (N, 3) arrays where N is the common length.
    """
    min_len = min(len(t) for t in trajectories)
    return [t[["x", "y", "z"]].values[:min_len] for t in trajectories]


def within_group_consistency(trajectories: list[np.ndarray]) -> dict:
    """Compute per-timestep position variance across trajectories.

    Returns dict with mean/max/std of per-timestep 3D position variance.
    """
    if len(trajectories) < 2:
        return {"mean_var": np.nan, "max_var": np.nan, "n_traj": len(trajectories)}

    stacked = np.stack(trajectories)  # (n_reps, n_time, 3)
    # Per-timestep mean position
    mean_traj = stacked.mean(axis=0)  # (n_time, 3)
    # Per-timestep squared distance from mean
    sq_dist = np.sum((stacked - mean_traj[np.newaxis]) ** 2, axis=2)  # (n_reps, n_time)
    # Mean squared distance per timestep (= position variance)
    mse_per_t = sq_dist.mean(axis=0)  # (n_time,)

    return {
        "mean_mse": float(mse_per_t.mean()),
        "max_mse": float(mse_per_t.max()),
        "rmse": float(np.sqrt(mse_per_t.mean())),
        "n_traj": len(trajectories),
        "n_samples": len(mse_per_t),
    }


def cross_group_mse(traj_a: list[np.ndarray], traj_b: list[np.ndarray]) -> dict:
    """Compute MSE between mean trajectories of two groups."""
    if not traj_a or not traj_b:
        return {"mse": np.nan, "rmse": np.nan}

    # Align to common length across both groups
    min_len = min(
        min(len(t) for t in traj_a),
        min(len(t) for t in traj_b),
    )
    a_stack = np.stack([t[:min_len] for t in traj_a])
    b_stack = np.stack([t[:min_len] for t in traj_b])

    mean_a = a_stack.mean(axis=0)  # (n_time, 3)
    mean_b = b_stack.mean(axis=0)  # (n_time, 3)

    sq_dist = np.sum((mean_a - mean_b) ** 2, axis=1)  # (n_time,)

    return {
        "mse": float(sq_dist.mean()),
        "rmse": float(np.sqrt(sq_dist.mean())),
        "max_dist": float(np.sqrt(sq_dist.max())),
        "n_samples": int(min_len),
    }


def find_csvs(run_dir: str, scenario: str) -> list[str]:
    """Find all CSV files for a scenario in a run directory."""
    pattern = os.path.join(run_dir, "*", scenario, "*.csv")
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Trajectory MSE analysis")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path, default=Path("logs/plots/pca/trajectory"))
    parser.add_argument("--resample-hz", type=int, default=50)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Discover scenarios from the first N6 run
    first_run = INCLUDED_RUNS[0]
    board_dirs = [d for d in glob.glob(os.path.join(first_run, "*")) if os.path.isdir(d)]
    if not board_dirs:
        print(f"No board directories in {first_run}")
        return
    scenario_dirs = sorted(glob.glob(os.path.join(board_dirs[0], "*")))
    scenarios = [os.path.basename(d) for d in scenario_dirs if os.path.isdir(d)]

    # Identify N6 and Pixhawk runs
    n6_runs = {
        0: INCLUDED_RUNS[0],
        25: INCLUDED_RUNS[1],
        50: INCLUDED_RUNS[2],
        75: INCLUDED_RUNS[3],
        100: INCLUDED_RUNS[4],
    }
    pix_run = INCLUDED_RUNS[5]

    print(f"Scenarios: {len(scenarios)}")
    print(f"Resample: {args.resample_hz} Hz")
    print()

    all_within = []
    all_cross = []

    for scenario in scenarios:
        # Load all trajectories per duty level
        duty_trajs = {}
        for duty, run_dir in n6_runs.items():
            csvs = find_csvs(run_dir, scenario)
            trajs = []
            for c in csvs:
                t = load_trajectory(c, args.resample_hz)
                if t is not None:
                    trajs.append(t)
            if trajs:
                aligned = align_trajectories(trajs)
                duty_trajs[duty] = aligned

        # Pixhawk trajectories
        pix_csvs = find_csvs(pix_run, scenario)
        pix_trajs = []
        for c in pix_csvs:
            t = load_trajectory(c, args.resample_hz)
            if t is not None:
                pix_trajs.append(t)

        # Within-group consistency per duty level
        for duty, trajs in duty_trajs.items():
            wg = within_group_consistency(trajs)
            wg["scenario"] = scenario
            wg["duty"] = duty
            wg["board"] = "npu_v2"
            all_within.append(wg)

        if pix_trajs:
            pix_aligned = align_trajectories(pix_trajs)
            wg = within_group_consistency(pix_aligned)
            wg["scenario"] = scenario
            wg["duty"] = 0
            wg["board"] = "pixhawk6c"
            all_within.append(wg)

        # Cross-platform MSE (N6 @ 0% vs Pixhawk)
        if 0 in duty_trajs and pix_trajs:
            pix_aligned = align_trajectories(pix_trajs)
            xp = cross_group_mse(duty_trajs[0], pix_aligned)
            xp["scenario"] = scenario
            xp["comparison"] = "n6_0pct_vs_pixhawk"
            all_cross.append(xp)

        # Cross-duty MSE (N6 @ 0% vs N6 @ 100%)
        if 0 in duty_trajs and 100 in duty_trajs:
            cd = cross_group_mse(duty_trajs[0], duty_trajs[100])
            cd["scenario"] = scenario
            cd["comparison"] = "n6_0pct_vs_n6_100pct"
            all_cross.append(cd)

    # Save and print results
    within_df = pd.DataFrame(all_within)
    cross_df = pd.DataFrame(all_cross)

    within_df.to_csv(args.output / "within_group_consistency.csv", index=False)
    cross_df.to_csv(args.output / "cross_group_mse.csv", index=False)

    print("=" * 72)
    print("  WITHIN-GROUP TRAJECTORY CONSISTENCY (RMSE in metres)")
    print("=" * 72)
    print(f"  {'Scenario':35s} {'0%':>6s} {'25%':>6s} {'50%':>6s} {'75%':>6s} {'100%':>6s} {'Pix':>6s}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for scenario in scenarios:
        row = f"  {scenario:35s}"
        for duty in [0, 25, 50, 75, 100]:
            sub = within_df[(within_df["scenario"] == scenario) &
                            (within_df["duty"] == duty) &
                            (within_df["board"] == "npu_v2")]
            if len(sub):
                row += f" {sub.iloc[0]['rmse']:6.3f}"
            else:
                row += f" {'—':>6s}"
        sub = within_df[(within_df["scenario"] == scenario) &
                        (within_df["board"] == "pixhawk6c")]
        if len(sub):
            row += f" {sub.iloc[0]['rmse']:6.3f}"
        else:
            row += f" {'—':>6s}"
        print(row)

    print()
    print("=" * 72)
    print("  CROSS-GROUP TRAJECTORY MSE")
    print("=" * 72)
    print(f"  {'Scenario':35s} {'N6 0% vs 100%':>15s} {'N6 vs Pixhawk':>15s}")
    print(f"  {'':35s} {'RMSE (m)':>15s} {'RMSE (m)':>15s}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    for scenario in scenarios:
        row = f"  {scenario:35s}"
        for comp in ["n6_0pct_vs_n6_100pct", "n6_0pct_vs_pixhawk"]:
            sub = cross_df[(cross_df["scenario"] == scenario) &
                           (cross_df["comparison"] == comp)]
            if len(sub):
                row += f" {sub.iloc[0]['rmse']:15.3f}"
            else:
                row += f" {'—':>15s}"
        print(row)

    print()
    print(f"  Results saved to {args.output}/")


if __name__ == "__main__":
    main()
