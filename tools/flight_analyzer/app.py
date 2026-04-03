#!/usr/bin/env python3
"""jMAVSim Flight Analyzer — Interactive Dashboard.

Launch:  python -m tools.flight_analyzer.app [--logs-dir DIR] [--port 8050]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from .loader import (
    discover_test_runs, load_flight_csv, load_scenario_result,
    get_step_boundaries, TestRun, RunEntry,
)
from .analysis import compute_flight_stats, compute_hover_quality, compare_runs

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
LOGS_DIR: Path = Path()
ALL_RUNS: list[TestRun] = []

STEP_COLORS = {
    "reboot": "rgba(153,153,153,0.15)", "arm": "rgba(46,204,113,0.15)",
    "takeoff": "rgba(52,152,219,0.15)", "hover": "rgba(155,89,182,0.15)",
    "goto": "rgba(230,126,34,0.15)",    "orbit": "rgba(231,76,60,0.15)",
    "land": "rgba(26,188,156,0.15)",    "disarm": "rgba(149,165,166,0.15)",
    "setWind": "rgba(243,156,18,0.15)", "setNpuLoad": "rgba(142,68,173,0.15)",
    "wait": "rgba(189,195,199,0.10)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_label(r: TestRun) -> str:
    npu = f" NPU={r.npu_load}%" if r.npu_load is not None else ""
    return f"{r.board}{npu} ({r.path.name[:20]})"


def _run_options():
    return [{"label": _run_label(r), "value": i} for i, r in enumerate(ALL_RUNS)]


def _scenario_options(run_indices: list[int]) -> list[dict]:
    scenarios = set()
    for idx in run_indices:
        if 0 <= idx < len(ALL_RUNS):
            scenarios.update(ALL_RUNS[idx].scenarios)
    return [{"label": s, "value": s} for s in sorted(scenarios)]


def _rep_options(run_idx: int, scenario: str) -> list[dict]:
    if run_idx is None or not scenario:
        return []
    run = ALL_RUNS[run_idx]
    reps = [e.repetition for e in run.runs if e.scenario == scenario]
    return [{"label": f"Rep {r}", "value": r} for r in sorted(set(reps))]


def _get_entry(run_idx: int, scenario: str, rep: int) -> RunEntry | None:
    run = ALL_RUNS[run_idx]
    for e in run.runs:
        if e.scenario == scenario and e.repetition == rep:
            return e
    return None


def _add_step_shapes(fig, boundaries, row_count):
    """Add step shading as background shapes across all subplot rows."""
    for i, b in enumerate(boundaries):
        t0 = b["elapsed_s"]
        t1 = boundaries[i + 1]["elapsed_s"] if i + 1 < len(boundaries) else None
        if t1 is None:
            continue
        color = STEP_COLORS.get(b["step_type"], "rgba(200,200,200,0.08)")
        for row in range(1, row_count + 1):
            fig.add_vrect(x0=t0, x1=t1, fillcolor=color, line_width=0,
                          row=row, col=1)
        # Label on top row
        fig.add_annotation(
            x=(t0 + t1) / 2, y=1.0, yref="paper",
            text=b["step_type"], showarrow=False,
            font=dict(size=9, color="gray"), yanchor="bottom",
        )


TRACE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "jMAVSim Flight Analyzer"

def build_layout():
    return html.Div([
        # Header
        html.Div([
            html.H1("jMAVSim Flight Analyzer", style={"margin": "0", "fontSize": "24px"}),
            html.Span(f"{len(ALL_RUNS)} test runs loaded", style={"color": "#888", "marginLeft": "16px"}),
        ], style={"display": "flex", "alignItems": "center", "padding": "12px 20px",
                  "borderBottom": "2px solid #ddd", "backgroundColor": "#fafafa"}),

        # Tabs
        dcc.Tabs(id="tabs", value="tab-overview", children=[
            dcc.Tab(label="Run Overview", value="tab-overview"),
            dcc.Tab(label="Flight Viewer", value="tab-flight"),
            dcc.Tab(label="Compare Runs", value="tab-compare"),
            dcc.Tab(label="Statistics", value="tab-stats"),
        ], style={"margin": "0 20px"}),

        html.Div(id="tab-content", style={"padding": "20px"}),
    ], style={"fontFamily": "system-ui, -apple-system, sans-serif"})


# ---- Tab: Overview --------------------------------------------------------

def render_overview():
    # Build summary table
    rows = []
    for i, run in enumerate(ALL_RUNS):
        npu = f"{run.npu_load}%" if run.npu_load is not None else "-"
        rows.append({
            "#": i,
            "Run": run.path.name,
            "Board": run.board,
            "NPU Load": npu,
            "Routine": run.routine,
            "Total": run.total_runs,
            "Passed": run.passed,
            "Failed": run.failed,
            "Duration": f"{run.duration_seconds / 60:.0f} min",
            "Timestamp": run.timestamp[:16],
        })
    df = pd.DataFrame(rows)

    # Pass/fail bar chart
    fig = go.Figure()
    boards_npus = [_run_label(r) for r in ALL_RUNS]
    fig.add_trace(go.Bar(name="Passed", x=boards_npus, y=[r.passed for r in ALL_RUNS],
                         marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Failed", x=boards_npus, y=[r.failed for r in ALL_RUNS],
                         marker_color="#e74c3c"))
    fig.update_layout(barmode="stack", title="Pass / Fail by Run", height=300,
                      margin=dict(t=40, b=80))

    # Scenario breakdown heatmap
    all_scenarios = sorted(set(s for r in ALL_RUNS for s in r.scenarios))
    pass_matrix = []
    for run in ALL_RUNS:
        row = []
        by_sc = defaultdict(list)
        for e in run.runs:
            by_sc[e.scenario].append(e.passed)
        for sc in all_scenarios:
            entries = by_sc.get(sc, [])
            if entries:
                row.append(sum(entries) / len(entries) * 100)
            else:
                row.append(np.nan)
        pass_matrix.append(row)

    heatmap = go.Figure(data=go.Heatmap(
        z=pass_matrix, x=all_scenarios, y=boards_npus,
        colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1.0, "#2ecc71"]],
        zmin=0, zmax=100, text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in pass_matrix],
        texttemplate="%{text}", hovertemplate="Run: %{y}<br>Scenario: %{x}<br>Pass rate: %{z:.0f}%<extra></extra>",
    ))
    heatmap.update_layout(title="Pass Rate by Scenario & Run", height=50 + 40 * len(ALL_RUNS),
                          margin=dict(t=40, b=100), xaxis_tickangle=-45)

    return html.Div([
        dcc.Graph(figure=fig),
        dcc.Graph(figure=heatmap),
        html.H3("Run Details"),
        html.Table(
            [html.Tr([html.Th(col, style={"padding": "6px 12px", "borderBottom": "2px solid #ddd",
                                           "textAlign": "left"}) for col in df.columns])] +
            [html.Tr([
                html.Td(row[col], style={
                    "padding": "4px 12px", "borderBottom": "1px solid #eee",
                    "color": "#e74c3c" if col == "Failed" and row[col] > 0 else "inherit",
                    "fontWeight": "bold" if col == "Failed" and row[col] > 0 else "normal",
                }) for col in df.columns
            ]) for _, row in df.iterrows()],
            style={"borderCollapse": "collapse", "width": "100%"},
        ),
    ])


# ---- Tab: Flight Viewer ---------------------------------------------------

def render_flight_viewer():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Run:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="fv-run", options=_run_options(), value=0,
                             clearable=False, style={"width": "300px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Scenario:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="fv-scenario", style={"width": "260px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Rep:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="fv-rep", style={"width": "100px"}),
            ]),
            html.Div([
                html.Label("Align t=0:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="fv-align", options=[
                    {"label": "Liftoff (alt > 0.5m)", "value": "liftoff"},
                    {"label": "Arm command", "value": "arm"},
                    {"label": "Raw (recording start)", "value": "none"},
                ], value="liftoff", clearable=False, style={"width": "200px"}),
            ], style={"marginLeft": "20px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "16px"}),

        html.Div(id="fv-info", style={"marginBottom": "12px", "color": "#555"}),
        dcc.Loading(dcc.Graph(id="fv-trajectory", style={"height": "400px"})),
        dcc.Loading(dcc.Graph(id="fv-timeseries", style={"height": "900px"})),
    ])


# ---- Tab: Compare Runs ----------------------------------------------------

def render_compare():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Runs to Compare:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="cmp-runs", options=_run_options(), multi=True,
                             value=[0], style={"width": "600px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Scenario:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="cmp-scenario", style={"width": "260px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Rep:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="cmp-rep", value=1, style={"width": "100px"}),
            ]),
            html.Div([
                html.Label("Align t=0:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="cmp-align", options=[
                    {"label": "Liftoff (alt > 0.5m)", "value": "liftoff"},
                    {"label": "Arm command", "value": "arm"},
                    {"label": "Raw (recording start)", "value": "none"},
                ], value="liftoff", clearable=False, style={"width": "200px"}),
            ], style={"marginLeft": "20px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "16px"}),

        html.Div([
            html.Label("Metrics:", style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.Dropdown(id="cmp-metrics", multi=True, value=[
                "altitude", "roll_deg", "pitch_deg", "ground_speed", "ekf_vel_ratio", "thrust_sp",
            ], options=[{"label": c, "value": c} for c in [
                "altitude", "x", "y", "z", "vx", "vy", "vz",
                "roll_deg", "pitch_deg", "yaw_deg",
                "ground_speed", "vertical_speed",
                "roll_err_deg", "pitch_err_deg", "yaw_err_deg",
                "thrust_sp",
                "ekf_vel_ratio", "ekf_pos_horiz_ratio", "ekf_pos_vert_ratio",
                "ekf_pos_horiz_accuracy", "ekf_pos_vert_accuracy",
                "vibration_x", "vibration_y", "vibration_z",
                "npu_ms", "npu_fps",
                "cpu_load", "comm_drop_rate",
                "wind_x", "wind_y", "wind_z",
            ]], style={"width": "800px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

        dcc.Loading(dcc.Graph(id="cmp-plot")),
        dcc.Loading(html.Div(id="cmp-stats-table", style={"marginTop": "20px"})),
    ])


# ---- Tab: Statistics -------------------------------------------------------

def render_stats():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Runs:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="stats-runs", options=_run_options(), multi=True,
                             value=list(range(len(ALL_RUNS))), style={"width": "600px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Scenario:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="stats-scenario", style={"width": "260px"}),
            ]),
        ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "16px"}),

        dcc.Loading(dcc.Graph(id="stats-bars", style={"height": "600px"})),
        dcc.Loading(html.Div(id="stats-detail")),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-overview":
        return render_overview()
    elif tab == "tab-flight":
        return render_flight_viewer()
    elif tab == "tab-compare":
        return render_compare()
    elif tab == "tab-stats":
        return render_stats()
    return html.Div("Select a tab")


# ---- Flight Viewer callbacks -----------------------------------------------

@app.callback(Output("fv-scenario", "options"), Output("fv-scenario", "value"),
              Input("fv-run", "value"))
def fv_update_scenarios(run_idx):
    if run_idx is None:
        return [], None
    opts = _scenario_options([run_idx])
    val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(Output("fv-rep", "options"), Output("fv-rep", "value"),
              Input("fv-run", "value"), Input("fv-scenario", "value"))
def fv_update_reps(run_idx, scenario):
    if run_idx is None or not scenario:
        return [], None
    opts = _rep_options(run_idx, scenario)
    val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(
    Output("fv-info", "children"),
    Output("fv-trajectory", "figure"),
    Output("fv-timeseries", "figure"),
    Input("fv-rep", "value"),
    Input("fv-align", "value"),
    State("fv-run", "value"), State("fv-scenario", "value"),
)
def fv_update_plots(rep, align_mode, run_idx, scenario):
    empty_fig = go.Figure()
    if run_idx is None or not scenario or rep is None:
        return "", empty_fig, empty_fig

    entry = _get_entry(run_idx, scenario, rep)
    if not entry or not entry.csv_path or not entry.csv_path.exists():
        return "No CSV data available.", empty_fig, empty_fig

    run = ALL_RUNS[run_idx]
    df = load_flight_csv(entry.csv_path, downsample=5, align=align_mode or "liftoff")
    boundaries = get_step_boundaries(df)
    t = df["elapsed_s"]

    # Info line
    info = f"{run.board} | {scenario} rep{rep} | NPU={run.npu_load}% | {entry.result} | {len(df)} samples | t=0 at {align_mode}"

    # Step result details
    if entry.json_path and entry.json_path.exists():
        sr = load_scenario_result(entry.json_path)
        step_strs = []
        for s in sr.steps:
            icon = "+" if s.passed else "X"
            step_strs.append(f"[{icon}] {s.name} ({s.elapsed_seconds:.1f}s)")
        info += " | Steps: " + " -> ".join(step_strs)

    # --- Trajectory plot (XY + 3D side by side) ---
    traj_fig = make_subplots(rows=1, cols=2,
                              specs=[[{"type": "xy"}, {"type": "scene"}]],
                              subplot_titles=["XY Trajectory", "3D Flight Path"])

    traj_fig.add_trace(go.Scattergl(
        x=df["x"], y=df["y"], mode="markers", marker=dict(size=2, color=t, colorscale="Viridis",
        colorbar=dict(title="Time(s)", x=0.45, len=0.8)),
        hovertemplate="X: %{x:.3f}m<br>Y: %{y:.3f}m<extra></extra>",
    ), row=1, col=1)
    traj_fig.add_trace(go.Scatter(x=[df["x"].iloc[0]], y=[df["y"].iloc[0]], mode="markers",
                                   marker=dict(size=10, color="green", symbol="circle"),
                                   name="Start", showlegend=True), row=1, col=1)
    traj_fig.add_trace(go.Scatter(x=[df["x"].iloc[-1]], y=[df["y"].iloc[-1]], mode="markers",
                                   marker=dict(size=10, color="red", symbol="square"),
                                   name="End", showlegend=True), row=1, col=1)
    traj_fig.update_xaxes(title_text="X (m)", scaleanchor="y", row=1, col=1)
    traj_fig.update_yaxes(title_text="Y (m)", row=1, col=1)

    alt = df["altitude"] if "altitude" in df.columns else -df["z"]
    traj_fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=alt, mode="lines",
        line=dict(width=3, color=t, colorscale="Viridis"),
        hovertemplate="X: %{x:.3f}m<br>Y: %{y:.3f}m<br>Alt: %{z:.3f}m<extra></extra>",
    ), row=1, col=2)
    traj_fig.update_layout(height=380, margin=dict(t=30, b=10, l=10, r=10), showlegend=True,
                           legend=dict(x=0, y=1))
    # Increase axis tick precision for small values
    traj_fig.update_scenes(
        xaxis=dict(tickformat=".2f", title="X (m)"),
        yaxis=dict(tickformat=".2f", title="Y (m)"),
        zaxis=dict(tickformat=".2f", title="Alt (m)"),
    )

    # --- Time series panels ---
    panels = [
        ("Altitude & Vertical Speed", [("altitude", "Alt (m)"), ("vertical_speed", "Vz (m/s)")]),
        ("Attitude", [("roll_deg", "Roll"), ("pitch_deg", "Pitch"), ("yaw_deg", "Yaw")]),
        ("Ground Speed", [("ground_speed", "GndSpd"), ("vx", "Vx"), ("vy", "Vy")]),
        ("Control Tracking Error", [("roll_err_deg", "Roll err"), ("pitch_err_deg", "Pitch err"),
                                     ("yaw_err_deg", "Yaw err")]),
        ("EKF Ratios", [("ekf_vel_ratio", "Vel"), ("ekf_pos_horiz_ratio", "PosH"),
                         ("ekf_pos_vert_ratio", "PosV"), ("ekf_mag_ratio", "Mag")]),
        ("Vibration", [("vibration_x", "X"), ("vibration_y", "Y"), ("vibration_z", "Z")]),
        ("NPU & Comms", [("npu_ms", "NPU ms"), ("npu_fps", "FPS"), ("comm_drop_rate", "CommDrop")]),
    ]

    ts_fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                            subplot_titles=[p[0] for p in panels], vertical_spacing=0.035)

    for row_i, (panel_name, traces) in enumerate(panels, 1):
        for trace_name, label in traces:
            if trace_name in df.columns and df[trace_name].notna().any():
                ts_fig.add_trace(go.Scattergl(
                    x=t, y=df[trace_name], name=label, mode="lines",
                    line=dict(width=1),
                    hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
                    legendgroup=panel_name, legendgrouptitle_text=panel_name,
                ), row=row_i, col=1)

    _add_step_shapes(ts_fig, boundaries, len(panels))
    ts_fig.update_layout(height=140 * len(panels), margin=dict(t=20, b=30, l=50, r=20))
    ts_fig.update_xaxes(title_text="Time (s)", row=len(panels), col=1)

    return info, traj_fig, ts_fig


# ---- Compare callbacks -----------------------------------------------------

@app.callback(Output("cmp-scenario", "options"), Output("cmp-scenario", "value"),
              Input("cmp-runs", "value"))
def cmp_update_scenarios(run_indices):
    if not run_indices:
        return [], None
    opts = _scenario_options(run_indices)
    val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(Output("cmp-rep", "options"), Output("cmp-rep", "value"),
              Input("cmp-runs", "value"), Input("cmp-scenario", "value"))
def cmp_update_reps(run_indices, scenario):
    if not run_indices or not scenario:
        return [], None
    # Intersection of available reps
    all_reps = None
    for idx in run_indices:
        reps = set(e.repetition for e in ALL_RUNS[idx].runs if e.scenario == scenario)
        all_reps = reps if all_reps is None else all_reps & reps
    if not all_reps:
        return [], None
    opts = [{"label": f"Rep {r}", "value": r} for r in sorted(all_reps)]
    return opts, opts[0]["value"]

@app.callback(
    Output("cmp-plot", "figure"), Output("cmp-stats-table", "children"),
    Input("cmp-rep", "value"), Input("cmp-metrics", "value"),
    Input("cmp-align", "value"),
    State("cmp-runs", "value"), State("cmp-scenario", "value"),
)
def cmp_update(rep, metrics, align_mode, run_indices, scenario):
    empty = go.Figure()
    if not run_indices or not scenario or rep is None or not metrics:
        return empty, ""

    # Load data for each run
    flight_data = []  # (label, df)
    stat_pairs = []   # (label, FlightStats)
    for idx in run_indices:
        run = ALL_RUNS[idx]
        entry = _get_entry(idx, scenario, rep)
        if not entry or not entry.csv_path or not entry.csv_path.exists():
            continue
        df = load_flight_csv(entry.csv_path, downsample=5, align=align_mode or "liftoff")
        label = _run_label(run)
        flight_data.append((label, df))
        stat_pairs.append((label, compute_flight_stats(df)))

    if not flight_data:
        return empty, "No data available for selected runs."

    # Overlay plot
    n = len(metrics)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=metrics, vertical_spacing=0.03)

    for col_i, col in enumerate(metrics):
        for run_i, (label, df) in enumerate(flight_data):
            if col in df.columns and df[col].notna().any():
                fig.add_trace(go.Scattergl(
                    x=df["elapsed_s"], y=df[col], name=label, mode="lines",
                    line=dict(width=1.2, color=TRACE_COLORS[run_i % len(TRACE_COLORS)]),
                    legendgroup=label,
                    showlegend=(col_i == 0),
                    hovertemplate=f"{label}<br>{col}: %{{y:.3f}}<extra></extra>",
                ), row=col_i + 1, col=1)

    fig.update_layout(height=max(400, 160 * n), margin=dict(t=30, b=30, l=50, r=20))
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)

    # Stats comparison table
    if stat_pairs:
        comp_df = compare_runs(stat_pairs)
        key_metrics = [
            "duration_s", "pos_drift_mean", "pos_drift_max", "alt_error_mean",
            "roll_std", "pitch_std", "roll_err_rms", "pitch_err_rms",
            "ground_speed_max", "ekf_vel_ratio_mean", "ekf_vel_ratio_max",
            "ekf_pos_horiz_accuracy_mean", "vib_magnitude_max", "clipping_total",
            "npu_ms_mean", "npu_fps_mean", "comm_drop_rate_max",
        ]
        key_metrics = [m for m in key_metrics if m in comp_df.columns]

        table_header = [html.Tr(
            [html.Th("Metric", style={"padding": "6px 10px", "borderBottom": "2px solid #ddd", "textAlign": "left"})] +
            [html.Th(label, style={"padding": "6px 10px", "borderBottom": "2px solid #ddd", "textAlign": "right"})
             for label in comp_df.index]
        )]
        table_rows = []
        for m in key_metrics:
            vals = comp_df[m]
            # Highlight best/worst
            cells = [html.Td(m.replace("_", " "), style={"padding": "4px 10px", "borderBottom": "1px solid #eee",
                                                           "fontWeight": "bold"})]
            for v in vals:
                cells.append(html.Td(f"{v:.4f}" if isinstance(v, float) else str(v),
                                     style={"padding": "4px 10px", "borderBottom": "1px solid #eee",
                                            "textAlign": "right"}))
            table_rows.append(html.Tr(cells))

        stats_table = html.Div([
            html.H3("Statistics Comparison", style={"marginTop": "20px"}),
            html.Table(table_header + table_rows,
                       style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px"}),
        ])
    else:
        stats_table = ""

    return fig, stats_table


# ---- Statistics callbacks --------------------------------------------------

@app.callback(Output("stats-scenario", "options"), Output("stats-scenario", "value"),
              Input("stats-runs", "value"))
def stats_update_scenarios(run_indices):
    if not run_indices:
        return [], None
    opts = _scenario_options(run_indices)
    val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(Output("stats-bars", "figure"), Output("stats-detail", "children"),
              Input("stats-scenario", "value"), State("stats-runs", "value"))
def stats_update(scenario, run_indices):
    empty = go.Figure()
    if not run_indices or not scenario:
        return empty, ""

    stat_pairs = []
    for idx in run_indices:
        run = ALL_RUNS[idx]
        entry = _get_entry(idx, scenario, 1)
        if not entry or not entry.csv_path or not entry.csv_path.exists():
            continue
        df = load_flight_csv(entry.csv_path, downsample=5)
        stat_pairs.append((_run_label(run), compute_flight_stats(df)))

    if not stat_pairs:
        return empty, "No data for this scenario across selected runs."

    comp_df = compare_runs(stat_pairs)
    bar_metrics = ["pos_drift_mean", "pos_drift_max", "roll_std", "pitch_std",
                   "roll_err_rms", "pitch_err_rms", "ekf_vel_ratio_mean",
                   "vib_magnitude_max", "npu_ms_mean"]
    bar_metrics = [m for m in bar_metrics if m in comp_df.columns]

    fig = make_subplots(rows=1, cols=len(bar_metrics),
                        subplot_titles=[m.replace("_", " ") for m in bar_metrics])

    for i, m in enumerate(bar_metrics):
        for j, label in enumerate(comp_df.index):
            fig.add_trace(go.Bar(
                x=[label], y=[comp_df.loc[label, m]], name=label,
                marker_color=TRACE_COLORS[j % len(TRACE_COLORS)],
                showlegend=(i == 0),
                hovertemplate=f"{label}<br>{m}: %{{y:.4f}}<extra></extra>",
            ), row=1, col=i + 1)

    fig.update_layout(height=500, margin=dict(t=40, b=120), barmode="group", showlegend=True)

    # Hover quality for hover scenarios
    hover_info = ""
    if "hover" in scenario.lower():
        hover_rows = []
        for idx in run_indices:
            run = ALL_RUNS[idx]
            entry = _get_entry(idx, scenario, 1)
            if not entry or not entry.csv_path or not entry.csv_path.exists():
                continue
            df = load_flight_csv(entry.csv_path)
            hq = compute_hover_quality(df)
            if hq.get("hover_data"):
                hover_rows.append({
                    "Run": _run_label(run),
                    "Drift mean (m)": f"{hq['drift_mean_m']:.4f}",
                    "Drift max (m)": f"{hq['drift_max_m']:.4f}",
                    "Drift 95% (m)": f"{hq['drift_95pct_m']:.4f}",
                    "Alt std (m)": f"{hq['alt_std_m']:.4f}",
                    "Roll std (deg)": f"{hq['roll_std_deg']:.3f}",
                    "Pitch std (deg)": f"{hq['pitch_std_deg']:.3f}",
                })
        if hover_rows:
            hdf = pd.DataFrame(hover_rows)
            hover_info = html.Div([
                html.H3("Hover Quality Analysis"),
                html.Table(
                    [html.Tr([html.Th(c, style={"padding": "6px 10px", "borderBottom": "2px solid #ddd"})
                              for c in hdf.columns])] +
                    [html.Tr([html.Td(row[c], style={"padding": "4px 10px", "borderBottom": "1px solid #eee",
                                                      "textAlign": "right" if i > 0 else "left"})
                              for i, c in enumerate(hdf.columns)]) for _, row in hdf.iterrows()],
                    style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px", "marginTop": "12px"},
                ),
            ])

    return fig, hover_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LOGS_DIR, ALL_RUNS

    parser = argparse.ArgumentParser(description="jMAVSim Flight Analyzer Dashboard")
    parser.add_argument("--logs-dir", "-d",
                        default=str(Path(__file__).resolve().parent.parent.parent / "logs"),
                        help="Path to logs directory")
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to serve on")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    LOGS_DIR = Path(args.logs_dir)
    print(f"Loading test runs from {LOGS_DIR}...")
    ALL_RUNS = discover_test_runs(LOGS_DIR)
    print(f"Loaded {len(ALL_RUNS)} test runs ({sum(r.total_runs for r in ALL_RUNS)} total scenario runs)")

    app.layout = build_layout()
    print(f"\nDashboard: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
