#!/usr/bin/env python3
"""CLI for jMAVSim flight data analysis.

Usage:
    flight-analyzer list [--logs-dir DIR]
    flight-analyzer summary RUN [--scenario S]
    flight-analyzer plot RUN SCENARIO [--rep N] [--output DIR]
    flight-analyzer analyze RUN SCENARIO [--rep N]
    flight-analyzer compare RUN1 RUN2 --scenario S [--columns C1,C2]
    flight-analyzer report RUN [--output DIR]
"""

import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .loader import (
    discover_test_runs, load_flight_csv, load_scenario_result,
    get_step_boundaries, TestRun, RunEntry,
)
from .analysis import compute_flight_stats, compute_hover_quality, compare_runs
from .plots import plot_flight_overview, plot_comparison, plot_stats_comparison

console = Console()

DEFAULT_LOGS = Path(__file__).resolve().parent.parent.parent / "logs"


def _resolve_logs_dir(logs_dir: str | None) -> Path:
    if logs_dir:
        return Path(logs_dir)
    return DEFAULT_LOGS


def _find_run(runs: list[TestRun], run_id: str) -> TestRun | None:
    """Find a run by directory name prefix or index."""
    # Try exact match
    for r in runs:
        if r.path.name == run_id:
            return r
    # Try prefix
    for r in runs:
        if r.path.name.startswith(run_id):
            return r
    # Try numeric index
    try:
        idx = int(run_id)
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass
    return None


def _get_run_entries(run: TestRun, scenario: str | None, rep: int | None) -> list[RunEntry]:
    entries = run.runs
    if scenario:
        entries = [e for e in entries if e.scenario == scenario or
                   e.scenario.lower().startswith(scenario.lower())]
    if rep is not None:
        entries = [e for e in entries if e.repetition == rep]
    return entries


@click.group()
@click.option("--logs-dir", "-d", default=None, help="Path to logs directory")
@click.pass_context
def cli(ctx, logs_dir):
    """jMAVSim Flight Data Analyzer"""
    ctx.ensure_object(dict)
    ctx.obj["logs_dir"] = _resolve_logs_dir(logs_dir)


@cli.command("list")
@click.pass_context
def list_runs(ctx):
    """List all test runs."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    if not runs:
        console.print("[yellow]No test runs found.[/yellow]")
        return

    table = Table(title="Test Runs", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Run ID", style="cyan")
    table.add_column("Board", style="green")
    table.add_column("NPU %", justify="right")
    table.add_column("Routine")
    table.add_column("Total", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Duration", justify="right")
    table.add_column("Timestamp")

    for i, run in enumerate(runs):
        npu = str(run.npu_load) + "%" if run.npu_load is not None else "-"
        dur = f"{run.duration_seconds / 60:.0f}m"
        fail_style = "red bold" if run.failed > 0 else ""
        table.add_row(
            str(i), run.path.name, run.board, npu, run.routine,
            str(run.total_runs), str(run.passed),
            Text(str(run.failed), style=fail_style),
            dur, run.timestamp[:16],
        )

    console.print(table)


@cli.command()
@click.argument("run_id")
@click.option("--scenario", "-s", default=None, help="Filter by scenario name")
@click.pass_context
def summary(ctx, run_id, scenario):
    """Show detailed summary for a test run."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    run = _find_run(runs, run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    # Header
    console.print(Panel(
        f"[bold]{run.board}[/bold] — {run.description}\n"
        f"Routine: {run.routine}  |  NPU: {run.npu_load}%  |  "
        f"Duration: {run.duration_seconds / 60:.1f} min\n"
        f"Total: {run.total_runs}  |  "
        f"[green]Pass: {run.passed}[/green]  |  "
        f"[red]Fail: {run.failed}[/red]",
        title=run.path.name,
    ))

    # Group by scenario
    by_scenario = defaultdict(list)
    for entry in run.runs:
        by_scenario[entry.scenario].append(entry)

    table = Table(show_lines=True)
    table.add_column("Scenario", style="cyan")
    table.add_column("Reps", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Avg Duration", justify="right")

    for sc_name in sorted(by_scenario.keys()):
        if scenario and not sc_name.lower().startswith(scenario.lower()):
            continue
        entries = by_scenario[sc_name]
        n_pass = sum(1 for e in entries if e.passed)
        n_fail = sum(1 for e in entries if not e.passed)
        avg_dur = sum(e.duration for e in entries) / len(entries)
        table.add_row(
            sc_name, str(len(entries)), str(n_pass), str(n_fail),
            f"{avg_dur:.1f}s",
        )

    console.print(table)

    # Show step details for failed runs
    failed = [e for e in run.runs if not e.passed]
    if scenario:
        failed = [e for e in failed if e.scenario.lower().startswith(scenario.lower())]
    if failed:
        console.print(f"\n[red bold]Failed runs ({len(failed)}):[/red bold]")
        for entry in failed[:10]:
            console.print(f"  {entry.scenario} rep{entry.repetition}: {entry.result}")
            if entry.json_path and entry.json_path.exists():
                sr = load_scenario_result(entry.json_path)
                for step in sr.steps:
                    if not step.passed:
                        console.print(f"    [red]FAIL[/red] step {step.index} "
                                      f"({step.type}): {step.details or ''}")


@cli.command()
@click.argument("run_id")
@click.argument("scenario")
@click.option("--rep", "-r", type=int, default=None, help="Repetition number (default: first)")
@click.option("--output", "-o", default=None, help="Output directory for plots")
@click.option("--downsample", type=int, default=10, help="Keep every Nth sample (default: 10)")
@click.pass_context
def plot(ctx, run_id, scenario, rep, output, downsample):
    """Generate flight overview plots."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    run = _find_run(runs, run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    entries = _get_run_entries(run, scenario, rep)
    if not entries:
        console.print(f"[red]No matching entries for scenario '{scenario}'.[/red]")
        return

    out_dir = Path(output) if output else ctx.obj["logs_dir"] / "plots"

    for entry in entries:
        if not entry.csv_path or not entry.csv_path.exists():
            console.print(f"[yellow]No CSV for {entry.scenario} rep{entry.repetition}[/yellow]")
            continue

        console.print(f"Plotting {entry.scenario} rep{entry.repetition}...")
        df = load_flight_csv(entry.csv_path, downsample=downsample)
        boundaries = get_step_boundaries(df)

        title = (f"{run.board} | {entry.scenario} rep{entry.repetition} | "
                 f"NPU={run.npu_load}% | {entry.result}")
        out_file = out_dir / f"{run.board}_{entry.scenario}_rep{entry.repetition:02d}.png"
        plot_flight_overview(df, boundaries, title=title, output=out_file)


@cli.command()
@click.argument("run_id")
@click.argument("scenario")
@click.option("--rep", "-r", type=int, default=None)
@click.pass_context
def analyze(ctx, run_id, scenario, rep):
    """Compute and display flight statistics."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    run = _find_run(runs, run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    entries = _get_run_entries(run, scenario, rep)
    if not entries:
        console.print(f"[red]No matching entries.[/red]")
        return

    all_stats = []
    for entry in entries:
        if not entry.csv_path or not entry.csv_path.exists():
            continue
        df = load_flight_csv(entry.csv_path)
        stats = compute_flight_stats(df)
        all_stats.append((f"rep{entry.repetition}", stats))

        # Also show hover quality if applicable
        hover_q = compute_hover_quality(df)
        if hover_q.get("hover_data"):
            entry._hover_quality = hover_q

    if not all_stats:
        console.print("[yellow]No CSV data found.[/yellow]")
        return

    # Display stats table
    table = Table(title=f"Flight Statistics: {scenario}", show_lines=True)
    table.add_column("Metric", style="cyan")
    for label, _ in all_stats:
        table.add_column(label, justify="right")

    # Pick key metrics to display
    key_metrics = [
        ("duration_s", "Duration (s)", ".1f"),
        ("pos_drift_mean", "Pos drift mean (m)", ".3f"),
        ("pos_drift_max", "Pos drift max (m)", ".3f"),
        ("alt_error_mean", "Alt error std (m)", ".3f"),
        ("roll_std", "Roll std (deg)", ".2f"),
        ("pitch_std", "Pitch std (deg)", ".2f"),
        ("roll_max_abs", "Roll max (deg)", ".1f"),
        ("pitch_max_abs", "Pitch max (deg)", ".1f"),
        ("ground_speed_max", "Max ground speed (m/s)", ".2f"),
        ("roll_err_rms", "Roll error RMS (deg)", ".2f"),
        ("pitch_err_rms", "Pitch error RMS (deg)", ".2f"),
        ("ekf_vel_ratio_mean", "EKF vel ratio mean", ".3f"),
        ("ekf_vel_ratio_max", "EKF vel ratio max", ".3f"),
        ("ekf_pos_horiz_accuracy_mean", "EKF pos H acc (m)", ".3f"),
        ("vib_magnitude_max", "Vib magnitude max", ".4f"),
        ("clipping_total", "Clipping events", "d"),
        ("npu_ms_mean", "NPU latency mean (ms)", ".2f"),
        ("npu_fps_mean", "NPU FPS mean", ".1f"),
        ("comm_drop_rate_max", "Comm drop rate max (%)", ".1f"),
        ("comm_errors_max", "Comm errors max", "d"),
    ]

    for attr, label, fmt in key_metrics:
        vals = []
        for _, stats in all_stats:
            v = getattr(stats, attr, None)
            if v is not None:
                vals.append(f"{v:{fmt}}")
            else:
                vals.append("-")
        table.add_row(label, *vals)

    console.print(table)

    # Show averages if multiple reps
    if len(all_stats) > 1:
        comp_df = compare_runs(all_stats)
        console.print("\n[bold]Averages across repetitions:[/bold]")
        means = comp_df.mean(numeric_only=True)
        for attr, label, fmt in key_metrics:
            if attr in means.index:
                console.print(f"  {label}: {means[attr]:{fmt}}")


@cli.command()
@click.argument("run_ids", nargs=-1, required=True)
@click.option("--scenario", "-s", required=True, help="Scenario to compare")
@click.option("--rep", "-r", type=int, default=1, help="Repetition number")
@click.option("--columns", "-c", default="altitude,roll_deg,pitch_deg,ground_speed",
              help="Comma-separated columns to compare")
@click.option("--output", "-o", default=None, help="Output file path")
@click.pass_context
def compare(ctx, run_ids, scenario, rep, columns, output):
    """Compare flights across multiple runs."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    cols = [c.strip() for c in columns.split(",")]

    flight_data = []
    stat_pairs = []

    for run_id in run_ids:
        run = _find_run(runs, run_id)
        if not run:
            console.print(f"[yellow]Run '{run_id}' not found, skipping.[/yellow]")
            continue

        entries = _get_run_entries(run, scenario, rep)
        if not entries:
            console.print(f"[yellow]No match for {scenario} rep{rep} in {run_id}.[/yellow]")
            continue

        entry = entries[0]
        if not entry.csv_path or not entry.csv_path.exists():
            continue

        df = load_flight_csv(entry.csv_path, downsample=10)
        label = f"{run.board} NPU={run.npu_load}%"
        flight_data.append((label, df))
        stat_pairs.append((label, compute_flight_stats(df)))

    if len(flight_data) < 2:
        console.print("[red]Need at least 2 runs to compare.[/red]")
        return

    # Print comparison table
    comp_df = compare_runs(stat_pairs)
    table = Table(title=f"Comparison: {scenario} rep{rep}", show_lines=True)
    table.add_column("Metric", style="cyan")
    for label in comp_df.index:
        table.add_column(str(label), justify="right")

    for attr in ["duration_s", "pos_drift_mean", "pos_drift_max",
                  "roll_std", "pitch_std", "roll_err_rms", "pitch_err_rms",
                  "ekf_vel_ratio_mean", "vib_magnitude_max", "npu_ms_mean"]:
        if attr in comp_df.columns:
            vals = [f"{comp_df.loc[l, attr]:.3f}" for l in comp_df.index]
            table.add_row(attr.replace("_", " "), *vals)

    console.print(table)

    # Generate overlay plot
    out_path = Path(output) if output else ctx.obj["logs_dir"] / "plots" / f"compare_{scenario}_rep{rep:02d}.png"
    title = f"Comparison: {scenario} rep{rep}"
    plot_comparison(flight_data, cols, title=title, output=out_path)


@cli.command()
@click.argument("run_id")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--downsample", type=int, default=10)
@click.pass_context
def report(ctx, run_id, output, downsample):
    """Generate a full report with plots for all scenarios in a run."""
    runs = discover_test_runs(ctx.obj["logs_dir"])
    run = _find_run(runs, run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        return

    out_dir = Path(output) if output else ctx.obj["logs_dir"] / "plots" / run.path.name

    console.print(Panel(
        f"Generating report for [bold]{run.board}[/bold]\n"
        f"{run.description}\n"
        f"Scenarios: {len(run.scenarios)}  |  Total runs: {run.total_runs}",
        title="Report Generation",
    ))

    by_scenario = defaultdict(list)
    for entry in run.runs:
        by_scenario[entry.scenario].append(entry)

    all_stats = []

    for sc_name in sorted(by_scenario.keys()):
        entries = by_scenario[sc_name]
        console.print(f"  Processing {sc_name} ({len(entries)} reps)...")

        # Plot first rep
        entry = entries[0]
        if entry.csv_path and entry.csv_path.exists():
            df = load_flight_csv(entry.csv_path, downsample=downsample)
            boundaries = get_step_boundaries(df)
            title = (f"{run.board} | {sc_name} rep1 | NPU={run.npu_load}% | {entry.result}")
            out_file = out_dir / f"{sc_name}_rep01.png"
            plot_flight_overview(df, boundaries, title=title, output=out_file)

            stats = compute_flight_stats(df)
            all_stats.append((sc_name, stats))

    # Summary stats comparison
    if all_stats:
        comp_df = compare_runs(all_stats)
        out_file = out_dir / "stats_comparison.png"
        plot_stats_comparison(comp_df, title=f"{run.board} NPU={run.npu_load}% — Stats Overview",
                              output=out_file)

    console.print(f"\n[green]Report saved to {out_dir}[/green]")


def main():
    cli()


if __name__ == "__main__":
    main()
