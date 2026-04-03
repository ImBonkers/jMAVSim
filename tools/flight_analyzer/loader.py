"""Data loading for jMAVSim flight logs.

Discovers test runs, parses test_summary.json / per-scenario JSON,
and loads 250 Hz CSV flight telemetry into pandas DataFrames.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing import Optional

import pandas as pd


@dataclass
class StepResult:
    index: int
    type: str
    name: str
    passed: bool
    elapsed_seconds: float
    details: Optional[str] = None


@dataclass
class ScenarioResult:
    scenario: str
    description: str
    board: str
    timestamp: str
    total_time_seconds: float
    passed: int
    failed: int
    total: int
    success: bool
    steps: list[StepResult] = field(default_factory=list)


@dataclass
class RunEntry:
    scenario: str
    repetition: int
    result: str
    passed: bool
    duration: float
    start_time: str
    end_time: str
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    log_path: Optional[Path] = None


@dataclass
class TestRun:
    """A complete test run (one execution of a routine)."""
    path: Path
    board: str
    description: str
    routine: str
    variables: dict
    timestamp: str
    total_runs: int
    passed: int
    failed: int
    all_passed: bool
    duration_seconds: float
    runs: list[RunEntry] = field(default_factory=list)

    @property
    def npu_load(self) -> Optional[int]:
        return self.variables.get("NPU_LOAD")

    @property
    def scenarios(self) -> list[str]:
        return sorted(set(r.scenario for r in self.runs))


def discover_test_runs(logs_dir: Path) -> list[TestRun]:
    """Find all test runs under a logs directory."""
    logs_dir = Path(logs_dir)
    runs = []
    for entry in sorted(logs_dir.iterdir()):
        summary_path = entry / "test_summary.json"
        if entry.is_dir() and summary_path.exists():
            try:
                runs.append(_load_test_run(entry, summary_path))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping {entry.name}: {e}")
    return runs


def _load_test_run(run_dir: Path, summary_path: Path) -> TestRun:
    with open(summary_path) as f:
        data = json.load(f)

    run_entries = []
    for r in data.get("runs", []):
        csv_path = json_path = log_path = None
        for lf in r.get("logFiles", []):
            p = run_dir.parent / lf if not os.path.isabs(lf) else Path(lf)
            # Also try relative to the run directory's parent (logs/)
            if not p.exists():
                p = run_dir.parent.parent / lf
            if not p.exists():
                p = Path(lf)
            if p.suffix == ".csv":
                csv_path = p
            elif p.suffix == ".json":
                json_path = p
            elif p.suffix == ".log":
                log_path = p

        run_entries.append(RunEntry(
            scenario=r["scenario"],
            repetition=r["repetition"],
            result=r["result"],
            passed=r.get("passed", r["result"] == "PASS"),
            duration=r["duration"],
            start_time=r["startTime"],
            end_time=r["endTime"],
            csv_path=csv_path,
            json_path=json_path,
            log_path=log_path,
        ))

    return TestRun(
        path=run_dir,
        board=data["board"],
        description=data.get("description", ""),
        routine=data.get("routine", ""),
        variables=data.get("variables", {}),
        timestamp=data["timestamp"],
        total_runs=data["totalRuns"],
        passed=data["passed"],
        failed=data["failed"],
        all_passed=data["allPassed"],
        duration_seconds=data.get("durationSeconds", 0),
        runs=run_entries,
    )


def load_scenario_result(json_path: Path) -> ScenarioResult:
    """Load a per-scenario JSON result file."""
    with open(json_path) as f:
        data = json.load(f)

    steps = []
    for s in data.get("steps", []):
        steps.append(StepResult(
            index=s["index"],
            type=s["type"],
            name=s["name"],
            passed=s["passed"],
            elapsed_seconds=s["elapsedSeconds"],
            details=s.get("details"),
        ))

    return ScenarioResult(
        scenario=data["scenario"],
        description=data.get("description", ""),
        board=data.get("board", "Unknown"),
        timestamp=data.get("timestamp", ""),
        total_time_seconds=data.get("totalTimeSeconds", 0),
        passed=data.get("passed", 0),
        failed=data.get("failed", 0),
        total=data.get("total", 0),
        success=data.get("success", False),
        steps=steps,
    )


def load_flight_csv(csv_path: Path, downsample: Optional[int] = None,
                    align: str = "liftoff") -> pd.DataFrame:
    """Load a flight CSV into a DataFrame.

    Args:
        csv_path: Path to the CSV file.
        downsample: If set, keep every Nth row (useful for large files).
        align: Time alignment mode:
            "liftoff" — t=0 when altitude first exceeds 0.5m (best for cross-run comparison)
            "arm"     — t=0 at the arm step
            "none"    — t=0 at recording start (raw)

    Returns:
        DataFrame with elapsed_s column added, EKF transients removed.
    """
    df = pd.read_csv(csv_path)
    df["elapsed_s"] = df["elapsed_ms"] / 1000.0

    # Remove EKF initialization transients: detect position jumps > 2m
    # between consecutive samples and remove the outlier segments.
    # During reboot, LOCAL_POSITION_NED can report wild values before
    # the EKF converges, then snap back to (0,0,0).
    if "x" in df.columns and len(df) > 1:
        dx = df["x"].diff()
        dy = df["y"].diff()
        dz = df["z"].diff()
        jump_dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Find jumps > 2m in a single 4ms sample
        jump_mask = jump_dist > 2.0
        if jump_mask.any():
            jump_indices = df.index[jump_mask].tolist()
            # Typically comes in pairs: jump out, then jump back.
            # Mark everything between paired jumps for removal.
            remove_mask = pd.Series(False, index=df.index)
            i = 0
            while i < len(jump_indices) - 1:
                start = jump_indices[i]
                end = jump_indices[i + 1]
                # Only pair jumps that are reasonably close (< 30s apart)
                dt = df.loc[end, "elapsed_ms"] - df.loc[start, "elapsed_ms"]
                if dt < 30_000:
                    remove_mask.iloc[start:end] = True
                    i += 2
                else:
                    # Unpaired — remove a small window around the jump
                    remove_mask.iloc[max(0, start - 1):min(len(df), start + 10)] = True
                    i += 1
            # Handle trailing unpaired jump
            if i < len(jump_indices):
                idx = jump_indices[i]
                remove_mask.iloc[max(0, idx - 1):min(len(df), idx + 10)] = True

            df = df[~remove_mask].reset_index(drop=True)

    # Time alignment: shift t=0 to a meaningful flight event and drop pre-event data.
    if align != "none":
        t0 = None
        if align == "liftoff" and "altitude" in df.columns:
            liftoff = df[df["altitude"] > 0.5]
            if len(liftoff) > 0:
                t0 = liftoff.iloc[0]["elapsed_ms"]
        if (t0 is None or align == "arm") and "step_type" in df.columns:
            # Fallback to arm if no liftoff found, or if arm was requested
            arm_rows = df[df["step_type"] == "arm"]
            if len(arm_rows) > 0:
                t0 = arm_rows.iloc[0]["elapsed_ms"]
        if t0 is not None:
            df["elapsed_ms_raw"] = df["elapsed_ms"]
            df["elapsed_ms"] = df["elapsed_ms"] - t0
            df["elapsed_s"] = df["elapsed_ms"] / 1000.0
            df = df[df["elapsed_ms"] >= 0].reset_index(drop=True)

    if downsample and downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    return df


def get_step_boundaries(df: pd.DataFrame) -> list[dict]:
    """Extract step transition points from a flight CSV."""
    boundaries = []
    if "step_index" not in df.columns or "step_type" not in df.columns:
        return boundaries

    prev_idx = None
    for _, row in df.iterrows():
        si = int(row["step_index"]) if pd.notna(row["step_index"]) else None
        if si is not None and si != prev_idx:
            boundaries.append({
                "step_index": si,
                "step_type": row["step_type"],
                "elapsed_s": row["elapsed_s"],
                "elapsed_ms": row["elapsed_ms"],
            })
            prev_idx = si
    return boundaries
