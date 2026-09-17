#!/usr/bin/env python3
"""
Automated test runner for jMAVSim HIL scenarios.

Reads one or more test routine files (JSON) and runs each scenario via 'make test-batch'.
Uses tqdm for progress display with live jMAVSim step status.

Usage:
    python3 tools/test_runner/run_tests.py routines/quick_smoke.json
    python3 tools/test_runner/run_tests.py routines/npu_v2_50pct.json routines/npu_v2_75pct.json
    python3 tools/test_runner/run_tests.py routines/npu_v2_*.json -v
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm


class C:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def format_duration(secs):
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
    elif secs >= 60:
        return f"{secs // 60}m{secs % 60:02d}s"
    else:
        return f"{secs}s"


def term_width():
    return shutil.get_terminal_size((100, 24)).columns


def find_project_root():
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "build.xml").exists():
            return d
        d = d.parent
    return Path.cwd()


def load_routine(path):
    with open(path) as f:
        routine = json.load(f)
    for key in ("board", "scenarios"):
        if key not in routine:
            sys.exit(f"ERROR: Routine file missing required key: '{key}'")
    return routine


def get_scenario_info(scenario_file):
    try:
        with open(scenario_file) as f:
            data = json.load(f)
        return {
            "timeout": data.get("globalTimeoutSeconds", 300),
            "steps": len(data.get("steps", [])),
            "name": data.get("name", Path(scenario_file).stem),
        }
    except Exception:
        return {"timeout": 300, "steps": 0, "name": Path(scenario_file).stem}


def apply_variables(scenario_file, variables, tmp_path):
    if not variables:
        return scenario_file
    with open(scenario_file) as f:
        content = f.read()
    for key, value in variables.items():
        content = content.replace(f'"${key}"', str(value))
    with open(tmp_path, "w") as f:
        f.write(content)
    return str(tmp_path)


class StatusVar:
    """Thread-safe shared status string. Writer: LogWatcher. Reader: main thread."""

    def __init__(self):
        self._status = ""
        self._steps_done = 0
        self._lock = threading.Lock()

    def set(self, status, steps_done=None):
        with self._lock:
            self._status = status
            if steps_done is not None:
                self._steps_done = steps_done

    def get(self):
        with self._lock:
            return self._status, self._steps_done


class LogWatcher:
    """Watches a log file and writes parsed status to a StatusVar."""

    STEP_PASS = re.compile(r'^\[(.+?)\]\s+PASS')
    STEP_FAIL = re.compile(r'^\[(.+?)\]\s+FAIL')
    PROGRESS = re.compile(r'^\s*\[(\w+)\]\s+(.+)')
    CONNECTED = re.compile(r'Vehicle connected')
    WAITING = re.compile(r'Waiting for vehicle connection')
    STEP_NAME = re.compile(r'^\[([^\]]+)\]')

    def __init__(self, log_path, status_var):
        self.log_path = log_path
        self.sv = status_var
        self.steps_done = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _watch(self):
        for _ in range(100):
            if os.path.exists(self.log_path) or self._stop.is_set():
                break
            time.sleep(0.1)
        try:
            with open(self.log_path, "r") as f:
                while not self._stop.is_set():
                    line = f.readline()
                    if line:
                        self._parse(line.rstrip())
                    else:
                        time.sleep(0.05)
        except Exception:
            pass

    def _short_name(self, raw):
        m = self.STEP_NAME.match(raw)
        if m:
            return m.group(1).strip()
        return raw.strip()

    def _parse(self, line):
        if not line:
            return

        if self.WAITING.search(line):
            self.sv.set("connecting", self.steps_done)
            return
        if self.CONNECTED.search(line):
            self.sv.set("connected", self.steps_done)
            return

        m = self.STEP_PASS.match(line)
        if m:
            self.steps_done += 1
            name = self._short_name(m.group(0))
            self.sv.set(name, self.steps_done)
            return

        m = self.STEP_FAIL.match(line)
        if m:
            self.steps_done += 1
            name = self._short_name(m.group(0))
            self.sv.set(name, self.steps_done)
            return

        m = self.PROGRESS.match(line)
        if m:
            step_type = m.group(1)
            self.sv.set(step_type, self.steps_done)


GUI_MODE = False


def run_with_status(cmd, log_path, status_var, total_steps, step_bar, overall_bar):
    """Run cmd via subprocess, tail log, and update tqdm bars from main thread."""
    watcher = LogWatcher(log_path, status_var)
    watcher.start()

    result_holder = [None]

    def _run():
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT,
            )
            proc.wait()
            result_holder[0] = proc.returncode

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()

    last_status = ""
    last_steps = 0
    while runner.is_alive():
        status, steps = status_var.get()
        changed = (status != last_status) or (steps != last_steps)
        if changed:
            if steps > last_steps:
                step_bar.update(steps - last_steps)
            if status != last_status:
                # Truncate long status to keep bar clean
                short = status[:30] if len(status) > 30 else status
                step_bar.set_postfix_str(short)
            last_status = status
            last_steps = steps
        time.sleep(0.15)

    runner.join()
    watcher.stop()

    # Final update — clamp to total
    status, steps = status_var.get()
    remaining = total_steps - step_bar.n
    if remaining > 0:
        step_bar.update(remaining)

    return result_holder[0] if result_holder[0] is not None else 1


def run_routine(routine_path, output_base, verbose, bar_file, bar_cols):
    """Run a single routine. Returns (results list, board name, duration)."""
    routine = load_routine(routine_path)
    board = routine["board"]
    description = routine.get("description", "")
    conn = routine.get("connection", {})
    serial_port = conn.get("port", "/dev/ttyACM0")
    baud = conn.get("baudRate", 921600)
    pause = routine.get("pauseBetweenRunsSeconds", 5)
    variables = routine.get("variables", {})

    if output_base is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", f"{board}_{ts}")
    else:
        out_dir = output_base

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(routine_path, os.path.join(out_dir, "routine.json"))

    # Build scenario groups
    scenario_groups = []
    total_runs = 0
    total_estimated_secs = 0
    for sc in routine["scenarios"]:
        info = get_scenario_info(sc["file"])
        reps = sc.get("repetitions", 1)
        scenario_groups.append({"file": sc["file"], "reps": reps, "info": info})
        total_runs += reps
        total_estimated_secs += info["timeout"] * reps + pause * reps

    eta = datetime.now() + timedelta(seconds=total_estimated_secs)
    cols = term_width()
    sep = "─" * min(cols - 4, 60)

    # Header
    print()
    print(f"  {C.BOLD}Routine: {Path(routine_path).stem}{C.NC}")
    print(f"  {C.DIM}{sep}{C.NC}")
    print(f"  Board:       {C.BOLD}{board}{C.NC}")
    if description:
        print(f"  Description: {description}")
    print(f"  Serial:      {serial_port} @ {baud}")
    print(f"  Output:      {out_dir}")
    print(f"  Runs:        {total_runs}  ({len(scenario_groups)} scenarios)")
    print(f"  Max time:    {format_duration(total_estimated_secs)}  (ETA {eta.strftime('%H:%M')})")
    if variables:
        var_str = ", ".join(f"{k}={v}" for k, v in variables.items())
        print(f"  Variables:   {var_str}")
    print(f"  {C.DIM}{sep}{C.NC}")
    print()

    # Overall progress bar
    overall = tqdm(
        total=total_runs,
        desc="Overall",
        unit="run",
        bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=bar_cols,
        position=1,
        file=bar_file,
        leave=True,
        colour="cyan",
        disable=not bar_file.isatty(),
    )

    routine_start = time.time()
    results = []
    run_num = 0

    for sg in scenario_groups:
        scenario_file = sg["file"]
        reps = sg["reps"]
        info = sg["info"]
        scenario_name = Path(scenario_file).stem
        num_steps = info["steps"]

        for rep in range(1, reps + 1):
            run_num += 1
            start_time = datetime.now()
            start_str = start_time.strftime("%Y%m%d_%H%M%S")
            rep_fmt = f"{rep:02d}"

            # Step-level bar
            label = f"{scenario_name} r{rep}"
            step_bar = tqdm(
                total=num_steps,
                desc=label,
                unit="step",
                bar_format="{desc}: {bar} {n_fmt}/{total_fmt} {postfix}",
                ncols=bar_cols,
                position=0,
                file=bar_file,
                leave=False,
                colour="green",
                disable=not bar_file.isatty(),
            )
            step_bar.set_postfix_str("starting")

            # Prepare temp dirs
            run_dir = os.path.join(out_dir, ".tmp_run")
            os.makedirs(run_dir, exist_ok=True)
            tmp_scenario = os.path.join(out_dir, ".tmp_scenario.json")
            actual_scenario = apply_variables(scenario_file, variables, tmp_scenario)

            target = "test-batch-gui" if GUI_MODE else "test-batch"
            cmd = (f'make {target} SERIAL="{serial_port}" BAUD="{baud}" '
                   f'SCENARIO="{actual_scenario}" OUTPUT="{run_dir}"')
            log_path = os.path.join(out_dir, ".tmp_stdout.log")

            if verbose:
                proc = subprocess.run(cmd, shell=True)
                exit_code = proc.returncode
            else:
                sv = StatusVar()
                exit_code = run_with_status(
                    cmd, log_path, sv, num_steps, step_bar, overall,
                )

            end_time = datetime.now()
            end_str = end_time.strftime("%Y%m%d_%H%M%S")
            duration = (end_time - start_time).total_seconds()

            if exit_code == 69:
                result_str, color = "ABORTED", C.YELLOW
            elif exit_code != 0:
                result_str, color = "FAIL", C.RED
            else:
                result_str, color = "PASS", C.GREEN

            step_bar.close()

            tqdm.write(
                f"  {label}: {color}{result_str}{C.NC}  {format_duration(duration)}",
                file=bar_file,
            )

            overall.update(1)

            # Move log files
            final_dir = os.path.join(out_dir, board, scenario_name)
            os.makedirs(final_dir, exist_ok=True)
            prefix = f"{board}_{scenario_name}_rep{rep_fmt}_{start_str}_{end_str}_{result_str}"

            log_files = []
            for f in Path(run_dir).iterdir():
                if f.is_file():
                    dest = os.path.join(final_dir, f"{prefix}{f.suffix}")
                    shutil.move(str(f), dest)
                    log_files.append(dest)
            if not verbose and os.path.exists(log_path):
                dest = os.path.join(final_dir, f"{prefix}.log")
                shutil.move(log_path, dest)
                log_files.append(dest)
            shutil.rmtree(run_dir, ignore_errors=True)

            results.append({
                "scenario": scenario_name,
                "repetition": rep,
                "result": result_str,
                "exitCode": exit_code,
                "passed": exit_code == 0,
                "duration": round(duration, 1),
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
                "logFiles": log_files,
            })

            # Pause between runs
            is_last = (sg == scenario_groups[-1] and rep == reps)
            if not is_last and pause > 0:
                step_bar = tqdm(
                    total=pause,
                    desc="pause",
                    unit="s",
                    bar_format="{desc}: {bar} {n:.0f}/{total:.0f}s",
                    ncols=bar_cols,
                    position=0,
                    file=bar_file,
                    leave=False,
                    colour="yellow",
                )
                for _ in range(pause):
                    time.sleep(1)
                    step_bar.update(1)
                step_bar.close()

    overall.close()
    total_duration = time.time() - routine_start

    print(file=bar_file)

    # Routine summary
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print()
    print(f"  {C.DIM}{sep}{C.NC}")
    print(f"  {C.BOLD}COMPLETE{C.NC} — {board} in {format_duration(total_duration)}")
    print(f"  Passed: {C.GREEN}{passed}{C.NC}    Failed: {C.RED if failed else C.DIM}{failed}{C.NC}")

    if failed > 0:
        print()
        for r in results:
            if not r["passed"]:
                c = C.YELLOW if r["result"] == "ABORTED" else C.RED
                print(f"    {c}x{C.NC} {r['scenario']} rep {r['repetition']}")
    print()

    # JSON summary
    summary = {
        "board": board, "description": description,
        "serial": serial_port, "baudRate": baud,
        "routine": os.path.basename(routine_path),
        "variables": variables,
        "timestamp": datetime.now().isoformat(),
        "totalRuns": len(results), "passed": passed, "failed": failed,
        "allPassed": failed == 0,
        "durationSeconds": round(total_duration, 1),
        "outputDir": out_dir, "runs": results,
    }
    with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results, board, total_duration, out_dir


def main():
    parser = argparse.ArgumentParser(description="jMAVSim automated test runner")
    parser.add_argument("routines", nargs="+", help="Path(s) to test routine JSON file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full simulator output")
    parser.add_argument("--gui", action="store_true",
                        help="Show the 3D window for each run (demo/recording)")
    args = parser.parse_args()

    global GUI_MODE
    GUI_MODE = args.gui

    project_root = find_project_root()
    os.chdir(project_root)

    cols = term_width()
    sep = "─" * min(cols - 4, 60)
    bar_file = sys.stderr
    bar_cols = min(cols, 120)

    if len(args.routines) > 1:
        print()
        print(f"  {C.BOLD}jMAVSim Test Runner{C.NC} — {len(args.routines)} routines queued")
        print(f"  {C.DIM}{sep}{C.NC}")
        for i, rp in enumerate(args.routines, 1):
            print(f"    {i}. {Path(rp).stem}")
        print(f"  {C.DIM}{sep}{C.NC}")

    grand_start = time.time()
    all_results = []
    routine_summaries = []

    for idx, routine_path in enumerate(args.routines, 1):
        if len(args.routines) > 1:
            print()
            print(f"  {C.BOLD}[{idx}/{len(args.routines)}]{C.NC} {Path(routine_path).stem}")

        results, board, duration, out_dir = run_routine(
            routine_path, args.output, args.verbose, bar_file, bar_cols,
        )
        passed = sum(1 for r in results if r["passed"])
        failed = len(results) - passed
        all_results.extend(results)
        routine_summaries.append({
            "routine": Path(routine_path).stem,
            "board": board,
            "passed": passed,
            "failed": failed,
            "duration": duration,
            "output": out_dir,
        })

    # Grand summary (only if multiple routines)
    if len(args.routines) > 1:
        grand_duration = time.time() - grand_start
        grand_passed = sum(1 for r in all_results if r["passed"])
        grand_failed = len(all_results) - grand_passed

        print()
        print(f"  {C.DIM}{sep}{C.NC}")
        print(f"  {C.BOLD}ALL ROUTINES COMPLETE{C.NC} — {format_duration(grand_duration)}")
        print(f"  {C.DIM}{sep}{C.NC}")
        for rs in routine_summaries:
            status = f"{C.GREEN}PASS{C.NC}" if rs["failed"] == 0 else f"{C.RED}FAIL{C.NC}"
            print(f"    {status}  {rs['routine']:<30s}  "
                  f"{rs['passed']}/{rs['passed']+rs['failed']}  "
                  f"{format_duration(rs['duration'])}")
        print(f"  {C.DIM}{sep}{C.NC}")
        print(f"  Total: {C.GREEN}{grand_passed} passed{C.NC}    "
              f"{C.RED if grand_failed else C.DIM}{grand_failed} failed{C.NC}")
        print()
        if grand_failed == 0:
            print(f"  {C.GREEN}{C.BOLD}ALL TESTS PASSED{C.NC}")
        else:
            print(f"  {C.RED}{C.BOLD}SOME TESTS FAILED{C.NC}")
        print()
    else:
        # Single routine — just print pass/fail
        passed = sum(1 for r in all_results if r["passed"])
        failed = len(all_results) - passed
        if failed == 0:
            print(f"  {C.GREEN}{C.BOLD}ALL TESTS PASSED{C.NC}")
        else:
            print(f"  {C.RED}{C.BOLD}SOME TESTS FAILED{C.NC}")
        print(f"  {C.DIM}Output: {routine_summaries[0]['output']}{C.NC}")
        print()

    any_failed = any(not r["passed"] for r in all_results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
