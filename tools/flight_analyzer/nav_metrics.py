"""Navigation metrics: cross-track error and Pixhawk trajectory deviation.

Adds two columns to all_flight_stats.csv:
  - xte_rms: RMS cross-track error (perpendicular distance from ideal straight-line
    path between consecutive waypoints). Measures path-following quality during transit.
  - pix_traj_rmse: RMSE of 3D position relative to the mean Pixhawk trajectory
    (time-aligned per step). Measures how much each N6 flight deviates from the
    Pixhawk baseline path.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

SCENARIO_DIR = Path("scenarios")
FLIGHT_START_STEP = 4  # takeoff step index

INCLUDED_RUNS = [
    "logs/npu_v2_20260418_131423",   # NPU 0%
    "logs/npu_v2_20260418_185511",   # NPU 25%
    "logs/npu_v2_20260419_003555",   # NPU 50%
    "logs/npu_v2_20260419_061806",   # NPU 75%
    "logs/npu_v2_20260419_115740",   # NPU 100%
    "logs/pixhawk6c_20260420_014914",
]


def load_scenario_waypoints(scenario_name: str) -> list[tuple[float, float, float]]:
    """Extract ordered waypoint positions from scenario JSON (takeoff + gotos)."""
    path = SCENARIO_DIR / f"{scenario_name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    waypoints = [(0.0, 0.0, 0.0)]  # origin (takeoff position)
    for step in data.get("steps", []):
        if step.get("type") == "takeoff":
            alt = step.get("altitude", 10.0)
            waypoints.append((0.0, 0.0, -alt))
        elif step.get("type") == "goto":
            waypoints.append((step["x"], step["y"], step["z"]))
    return waypoints


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to line segment a-b."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def compute_xte_rms(csv_path: str, scenario_name: str) -> float:
    """Compute RMS cross-track error for a flight.

    For each sample during goto steps, computes perpendicular distance to the
    straight-line segment between the previous and current waypoint.
    """
    waypoints = load_scenario_waypoints(scenario_name)
    if len(waypoints) < 2:
        return np.nan

    df = pd.read_csv(csv_path)
    # Only consider flight steps (takeoff onwards, exclude land/disarm)
    flight = df[df["step_index"] >= FLIGHT_START_STEP].copy()
    if len(flight) < 10:
        return np.nan

    # Load scenario steps to map step_index to waypoint pairs
    path = SCENARIO_DIR / f"{scenario_name}.json"
    with open(path) as f:
        steps = json.load(f)["steps"]

    xte_values = []
    wp_idx = 0  # index into waypoints list (0 = origin)

    for si in sorted(flight["step_index"].unique()):
        si_int = int(si)
        if si_int >= len(steps):
            continue
        step_def = steps[si_int]
        step_type = step_def.get("type", "")

        if step_type == "takeoff":
            wp_idx += 1
            # During takeoff, the "segment" is vertical from ground to takeoff alt
            sub = flight[flight["step_index"] == si]
            a = np.array([0.0, 0.0, 0.0])
            b = np.array([0.0, 0.0, -step_def.get("altitude", 10.0)])
            for _, row in sub.iterrows():
                p = np.array([row["x"], row["y"], row["z"]])
                xte_values.append(point_to_segment_distance(p, a, b))

        elif step_type == "goto":
            wp_idx += 1
            sub = flight[flight["step_index"] == si]
            if wp_idx < 2 or wp_idx > len(waypoints):
                continue
            a = np.array(waypoints[wp_idx - 1])
            b = np.array(waypoints[wp_idx])
            for _, row in sub.iterrows():
                p = np.array([row["x"], row["y"], row["z"]])
                xte_values.append(point_to_segment_distance(p, a, b))

    if not xte_values:
        return np.nan
    return float(np.sqrt(np.mean(np.array(xte_values) ** 2)))


def build_pixhawk_mean_trajectories(stats_csv: Path, resample_hz: int = 50) -> dict:
    """Build per-scenario mean Pixhawk trajectory (time-aligned per step).

    Returns dict: scenario -> {step_index -> (t_grid, mean_x, mean_y, mean_z)}
    """
    df = pd.read_csv(stats_csv)
    pix_run = [r for r in INCLUDED_RUNS if "pixhawk" in r][0]

    # Find all Pixhawk CSVs per scenario
    board_dirs = [d for d in glob.glob(os.path.join(pix_run, "*")) if os.path.isdir(d)]
    if not board_dirs:
        return {}

    result = {}
    scenario_dirs = sorted(glob.glob(os.path.join(board_dirs[0], "*")))

    for sc_dir in scenario_dirs:
        if not os.path.isdir(sc_dir):
            continue
        scenario = os.path.basename(sc_dir)
        csvs = sorted(glob.glob(os.path.join(sc_dir, "*.csv")))
        if not csvs:
            continue

        # Per-step trajectories across all reps
        step_trajs = {}  # step_index -> list of (t, x, y, z) arrays

        for csv_path in csvs:
            flight_df = pd.read_csv(csv_path)
            for si in sorted(flight_df["step_index"].unique()):
                si_int = int(si)
                if si_int < FLIGHT_START_STEP:
                    continue
                sub = flight_df[flight_df["step_index"] == si]
                if len(sub) < 5:
                    continue
                t = (sub["elapsed_ms"].values - sub["elapsed_ms"].values[0]) / 1000.0
                if t[-1] < 0.1:
                    continue
                if si_int not in step_trajs:
                    step_trajs[si_int] = []
                step_trajs[si_int].append((t, sub["x"].values, sub["y"].values, sub["z"].values))

        # Compute mean trajectory per step
        scenario_means = {}
        dt = 1.0 / resample_hz
        for si_int, trajs in step_trajs.items():
            # Common time grid (shortest duration)
            min_dur = min(t[-1] for t, _, _, _ in trajs)
            t_grid = np.arange(0, min_dur, dt)
            if len(t_grid) < 3:
                continue

            xs, ys, zs = [], [], []
            for t, x, y, z in trajs:
                fx = interpolate.interp1d(t, x, fill_value="extrapolate")
                fy = interpolate.interp1d(t, y, fill_value="extrapolate")
                fz = interpolate.interp1d(t, z, fill_value="extrapolate")
                xs.append(fx(t_grid))
                ys.append(fy(t_grid))
                zs.append(fz(t_grid))

            scenario_means[si_int] = (
                t_grid,
                np.mean(xs, axis=0),
                np.mean(ys, axis=0),
                np.mean(zs, axis=0),
            )

        result[scenario] = scenario_means

    return result


def compute_pix_traj_rmse(csv_path: str, scenario_name: str,
                          pix_means: dict, resample_hz: int = 50) -> float:
    """Compute RMSE of a flight's position relative to the mean Pixhawk trajectory.

    Alignment is per-step: each step is independently time-aligned to avoid
    cumulative timing offsets.
    """
    if scenario_name not in pix_means:
        return np.nan

    scenario_means = pix_means[scenario_name]
    df = pd.read_csv(csv_path)
    dt = 1.0 / resample_hz

    sq_errors = []

    for si_int, (t_grid, mean_x, mean_y, mean_z) in scenario_means.items():
        sub = df[df["step_index"] == si_int]
        if len(sub) < 5:
            continue

        t = (sub["elapsed_ms"].values - sub["elapsed_ms"].values[0]) / 1000.0
        if t[-1] < 0.1:
            continue

        # Resample this flight's step to the same grid (truncate to common length)
        n = min(len(t_grid), int(t[-1] / dt))
        if n < 3:
            continue
        t_common = t_grid[:n]

        fx = interpolate.interp1d(t, sub["x"].values, fill_value="extrapolate")
        fy = interpolate.interp1d(t, sub["y"].values, fill_value="extrapolate")
        fz = interpolate.interp1d(t, sub["z"].values, fill_value="extrapolate")

        dx = fx(t_common) - mean_x[:n]
        dy = fy(t_common) - mean_y[:n]
        dz = fz(t_common) - mean_z[:n]

        sq_errors.extend((dx**2 + dy**2 + dz**2).tolist())

    if not sq_errors:
        return np.nan
    return float(np.sqrt(np.mean(sq_errors)))


def compute_nav_metrics(stats_csv: Path, logs_dir: Path) -> pd.DataFrame:
    """Compute xte_rms and pix_traj_rmse for all flights and merge with stats."""
    df = pd.read_csv(stats_csv)

    if INCLUDED_RUNS:
        df = df[df["run_dir"].isin(INCLUDED_RUNS)].copy()

    print(f"Computing navigation metrics for {len(df)} flights...")

    # Build Pixhawk mean trajectories (once)
    print("  Building Pixhawk mean trajectories...")
    pix_means = build_pixhawk_mean_trajectories(stats_csv)
    print(f"  Done: {len(pix_means)} scenarios")

    # Find CSV paths from the test run structure
    # We need to reconstruct CSV paths from run_dir + scenario + rep
    xte_values = []
    pix_rmse_values = []

    for idx, row in df.iterrows():
        run_dir = row["run_dir"]
        scenario = row["scenario"]
        rep = row["repetition"]

        # Find the CSV
        board_dirs = [d for d in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(d)]
        csv_path = None
        for bd in board_dirs:
            pattern = os.path.join(bd, scenario, f"*rep{rep:02d}*.csv")
            matches = glob.glob(pattern)
            if matches:
                csv_path = matches[0]
                break

        if csv_path is None:
            xte_values.append(np.nan)
            pix_rmse_values.append(np.nan)
            continue

        xte = compute_xte_rms(csv_path, scenario)
        pix_rmse = compute_pix_traj_rmse(csv_path, scenario, pix_means)
        xte_values.append(xte)
        pix_rmse_values.append(pix_rmse)

    df["xte_rms"] = xte_values
    df["pix_traj_rmse"] = pix_rmse_values

    print(f"  xte_rms: {sum(1 for v in xte_values if not np.isnan(v))}/{len(xte_values)} computed")
    print(f"  pix_traj_rmse: {sum(1 for v in pix_rmse_values if not np.isnan(v))}/{len(pix_rmse_values)} computed")

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute navigation metrics")
    parser.add_argument("--stats-csv", type=Path,
                        default=Path("logs/stats/all_flight_stats.csv"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path,
                        default=Path("logs/stats/all_flight_stats.csv"),
                        help="Output CSV (overwrites stats CSV with added columns)")
    args = parser.parse_args()

    result = compute_nav_metrics(args.stats_csv, args.logs_dir)
    result.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output} ({len(result)} rows, {len(result.columns)} columns)")

    # Quick summary
    inc = result[result["run_dir"].isin(INCLUDED_RUNS)]
    n6 = inc[inc["board"] == "npu_v2"]
    print(f"\n=== XTE RMS Summary (N6, per duty level) ===")
    for duty in [0, 25, 50, 75, 100]:
        v = n6[n6["npu_load"] == duty]["xte_rms"].dropna()
        if len(v) > 0:
            print(f"  {duty:3d}%: median={v.median():.3f} m  mean={v.mean():.3f} m  n={len(v)}")

    print(f"\n=== Pixhawk Trajectory RMSE Summary (N6, per duty level) ===")
    for duty in [0, 25, 50, 75, 100]:
        v = n6[n6["npu_load"] == duty]["pix_traj_rmse"].dropna()
        if len(v) > 0:
            print(f"  {duty:3d}%: median={v.median():.3f} m  mean={v.mean():.3f} m  n={len(v)}")


if __name__ == "__main__":
    main()
