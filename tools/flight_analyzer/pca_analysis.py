"""PCA analysis of flight stats across NPU configurations.

Loads all_flight_stats.csv, runs PCA on the numeric flight parameters,
and identifies which parameters most differentiate configurations
(e.g. NPU 100% vs NPU 0%).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STATS_CSV = Path(__file__).resolve().parent.parent.parent / "logs" / "stats" / "all_flight_stats.csv"

# Features to include in PCA (exclude identifiers, metadata, and direct NPU measures)
FEATURE_COLS = [
    "duration_s", "origin_dist_mean", "pos_drift_max", "pos_drift_std",
    "alt_error_mean", "alt_error_max",
    "roll_mean", "roll_std", "roll_max_abs",
    "pitch_mean", "pitch_std", "pitch_max_abs",
    "ground_speed_mean", "ground_speed_max",
    "vertical_speed_mean", "vertical_speed_max_abs",
    "roll_err_rms", "pitch_err_rms", "yaw_err_rms",
    "ekf_vel_ratio_mean", "ekf_vel_ratio_max",
    "ekf_pos_horiz_ratio_mean", "ekf_pos_horiz_accuracy_mean", "ekf_pos_vert_accuracy_mean",
    "vib_x_mean", "vib_y_mean", "vib_z_mean", "vib_magnitude_max",
    "clipping_total",
    "comm_drop_rate_max", "comm_errors_max",
]


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
OUTLIER_FILTER = lambda df: df

def load_and_prepare(csv_path: Path, board: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if INCLUDED_RUNS:
        df = df[df["run_dir"].isin(INCLUDED_RUNS)]
    elif EXCLUDED_RUNS:
        df = df[~df["run_dir"].isin(EXCLUDED_RUNS)]
    df = OUTLIER_FILTER(df)
    if board:
        df = df[df["board"] == board]
    return df


def run_pca(df: pd.DataFrame, n_components: int = 5):
    """Run PCA on the feature columns, return results dict."""
    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].copy()

    # Drop columns that are constant (zero variance)
    X = X.loc[:, X.std() > 1e-10]
    used_features = list(X.columns)

    # Handle NaN by filling with column median
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, len(used_features), len(X))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    return {
        "pca": pca,
        "scaler": scaler,
        "scores": scores,
        "features": used_features,
        "n_components": n_components,
        "df": df,
    }


def print_explained_variance(result: dict):
    pca = result["pca"]
    print("\n=== Explained Variance ===")
    cumulative = 0.0
    for i, (var, cum) in enumerate(
        zip(pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_))
    ):
        cumulative = cum
        print(f"  PC{i+1}: {var:6.1%}  (cumulative: {cum:6.1%})")
    print(f"\n  Total with {result['n_components']} components: {cumulative:.1%}")


def print_top_loadings(result: dict, n_top: int = 10):
    pca = result["pca"]
    features = result["features"]

    print("\n=== Top Feature Loadings per Principal Component ===")
    for i in range(min(3, result["n_components"])):
        loadings = pca.components_[i]
        order = np.argsort(np.abs(loadings))[::-1]
        print(f"\n  PC{i+1} ({pca.explained_variance_ratio_[i]:.1%} variance):")
        for j in order[:n_top]:
            sign = "+" if loadings[j] > 0 else "-"
            print(f"    {sign} {features[j]:40s}  {loadings[j]:+.3f}")


def print_group_separation(result: dict, group_col: str, group_labels: dict | None = None):
    """Show how well PCA separates groups (NPU loads or boards)."""
    df = result["df"]
    scores = result["scores"]

    if group_col not in df.columns:
        return

    label_fn = (lambda v: group_labels.get(v, str(v))) if group_labels else str
    title = "Board" if group_col == "board" else "NPU Configuration"

    print(f"\n=== {title} Separation (PC1-PC2 centroids) ===")
    for val in sorted(df[group_col].unique()):
        mask = df[group_col] == val
        centroid = scores[mask.values, :2].mean(axis=0)
        print(f"  {label_fn(val):>15s}:  PC1={centroid[0]:+.3f}  PC2={centroid[1]:+.3f}  (n={mask.sum()})")


def print_discriminating_features(result: dict, group_col: str, val_a, val_b, n_top: int = 15):
    """Find features that most differ between two groups."""
    df = result["df"]
    features = result["features"]
    scaler = result["scaler"]

    mask_a = df[group_col] == val_a
    mask_b = df[group_col] == val_b

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        print(f"\n  [skip] {val_a} or {val_b} has no data")
        return

    X = df[features].fillna(df[features].median())
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features, index=df.index)

    mean_a = X_scaled[mask_a].mean()
    mean_b = X_scaled[mask_b].mean()
    std_pooled = X_scaled.std()
    std_pooled = std_pooled.replace(0, 1)  # avoid division by zero

    # Cohen's d effect size
    effect = (mean_b - mean_a) / std_pooled
    effect_sorted = effect.abs().sort_values(ascending=False)

    # Cliff's delta for top features (bounded [-1, +1])
    raw_vals_a = X[mask_a]
    raw_vals_b = X[mask_b]
    cliffs = {}
    for feat in effect_sorted.head(n_top).index:
        va = raw_vals_a[feat].values
        vb = raw_vals_b[feat].values
        n_a, n_b = len(va), len(vb)
        more = sum(1 for a in va for b in vb if b > a)
        less = sum(1 for a in va for b in vb if b < a)
        cliffs[feat] = (more - less) / (n_a * n_b)

    label_a, label_b = str(val_a), str(val_b)
    print(f"\n=== Most Discriminating Features: {label_a} vs {label_b} ===")
    print(f"  (Cohen's d and Cliff's delta effect sizes)")
    print(f"  {'Feature':40s}  {'d':>8s}  {'Cliff':>8s}  {'Mean '+label_a:>14s}  {'Mean '+label_b:>14s}")
    print(f"  {'':40s}  {'':>8s}  {'delta':>8s}  {'(raw)':>14s}  {'(raw)':>14s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*14}")

    raw_a = X[mask_a].mean()
    raw_b = X[mask_b].mean()

    for feat in effect_sorted.head(n_top).index:
        d = effect[feat]
        cd = cliffs.get(feat, float('nan'))
        print(f"  {feat:40s}  {d:+8.3f}  {cd:+8.3f}  {raw_a[feat]:14.4f}  {raw_b[feat]:14.4f}")

    return effect, cliffs


def plot_pca_scatter(result: dict, output: Path, scenario: str = "",
                     group_col: str = "npu_load"):
    """2D scatter of PC1 vs PC2, colored by group."""
    df = result["df"]
    scores = result["scores"]
    pca = result["pca"]

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = sorted(df[group_col].unique())

    if group_col == "board":
        colors = {"pixhawk6c": "#e74c3c", "npu_v2": "#3498db"}
        for grp in groups:
            mask = df[group_col].values == grp
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       c=colors.get(grp, "#999999"), label=grp,
                       alpha=0.6, s=30, edgecolors="none")
    else:
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(groups), vmax=max(groups))
        for grp in groups:
            mask = df[group_col].values == grp
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       c=[cmap(norm(grp))], label=f"NPU {grp}%",
                       alpha=0.6, s=30, edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    title = scenario.replace("_", " ") if scenario else "All Scenarios"
    ax.set_title(f"PCA — {title}")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loading_heatmap(result: dict, output: Path, n_features: int = 20):
    """Heatmap of top feature loadings on first few PCs."""
    pca = result["pca"]
    features = result["features"]

    output.parent.mkdir(parents=True, exist_ok=True)

    n_pcs = min(5, result["n_components"])
    loadings = pd.DataFrame(
        pca.components_[:n_pcs].T,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_pcs)],
    )

    # Select features with highest absolute loading on any PC
    max_abs = loadings.abs().max(axis=1)
    top_features = max_abs.sort_values(ascending=False).head(n_features).index
    loadings_top = loadings.loc[top_features]

    fig, ax = plt.subplots(figsize=(8, max(6, n_features * 0.35)))
    im = ax.imshow(loadings_top.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace("_", " ") for f in top_features], fontsize=8)
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels(loadings_top.columns)

    # Annotate cells
    for i in range(len(top_features)):
        for j in range(n_pcs):
            val = loadings_top.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
    ax.set_title("PCA Feature Loadings (top features by absolute magnitude)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Loading heatmap saved to {output}")
    plt.close(fig)


def run_per_scenario(df: pd.DataFrame, group_col: str, val_a, val_b,
                     n_components: int, n_top: int, out_dir: Path):
    """Run PCA independently for each scenario and print results."""
    scenarios = sorted(df["scenario"].unique())
    summary_rows = []

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario].reset_index(drop=True)
        n_flights = len(sdf)
        group_counts = sdf[group_col].value_counts().to_dict()

        print(f"\n{'='*72}")
        print(f"  {scenario}  (n={n_flights}, {group_col}: {group_counts})")
        print(f"{'='*72}")

        if n_flights < 6:
            print("  [skip] Too few samples for PCA")
            continue

        result = run_pca(sdf, n_components=n_components)
        print_explained_variance(result)
        print_group_separation(result, group_col)
        effect = print_discriminating_features(result, group_col, val_a, val_b, n_top=n_top)

        # Collect top discriminating feature for cross-scenario summary
        if effect is not None:
            cohens_d, cliffs_delta = effect
            top_feat = cohens_d.abs().idxmax()
            summary_rows.append({
                "scenario": scenario,
                "n": n_flights,
                "pc1_var": result["pca"].explained_variance_ratio_[0],
                "top_feature": top_feat,
                "top_effect_d": cohens_d[top_feat],
                "top_cliffs_delta": cliffs_delta.get(top_feat, float('nan')),
            })

        # Per-scenario plots
        sc_dir = out_dir / scenario
        plot_pca_scatter(result, sc_dir / "pca_scatter.png", scenario=scenario,
                         group_col=group_col)
        plot_loading_heatmap(result, sc_dir / "pca_loadings.png", n_features=n_top)

    # Cross-scenario summary
    if summary_rows:
        print(f"\n{'='*72}")
        print(f"  CROSS-SCENARIO SUMMARY: {val_a} vs {val_b}")
        print(f"{'='*72}")
        print(f"  {'Scenario':40s}  {'PC1 var':>8s}  {'Top feature':>30s}  {'d':>8s}  {'Cliff δ':>8s}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*30}  {'-'*8}  {'-'*8}")
        for row in summary_rows:
            print(f"  {row['scenario']:40s}  {row['pc1_var']:7.1%}  "
                  f"{row['top_feature']:>30s}  {row['top_effect_d']:+7.2f}  "
                  f"{row['top_cliffs_delta']:+7.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PCA analysis of flight configurations")
    parser.add_argument("--csv", default=str(STATS_CSV), help="Path to all_flight_stats.csv")
    parser.add_argument("--board", default=None, help="Filter to a specific board")
    parser.add_argument("--scenario", default=None, help="Run for a single scenario (e.g. S04_hover_gust)")
    parser.add_argument("--compare-boards", action="store_true",
                        help="Compare boards (pixhawk6c vs npu_v2) at NPU 0%% instead of NPU levels")
    parser.add_argument("--npu-a", type=int, default=0, help="First NPU config to compare (default: 0)")
    parser.add_argument("--npu-b", type=int, default=100, help="Second NPU config to compare (default: 100)")
    parser.add_argument("--components", type=int, default=5, help="Number of PCA components")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--top", type=int, default=10, help="Number of top features to show")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = load_and_prepare(csv_path, board=args.board)

    if args.compare_boards:
        # Board comparison mode: filter to NPU 0% only, compare boards
        df = df[df["npu_load"] == 0]
        group_col = "board"
        val_a, val_b = "pixhawk6c", "npu_v2"
        out_subdir = "pca_boards"
        print(f"Board comparison mode: NPU 0% only")
        print(f"  pixhawk6c: {(df['board'] == 'pixhawk6c').sum()} flights")
        print(f"  npu_v2:    {(df['board'] == 'npu_v2').sum()} flights")
    else:
        group_col = "npu_load"
        val_a, val_b = args.npu_a, args.npu_b
        out_subdir = "pca"

    print(f"Loaded {len(df)} flights, {df['scenario'].nunique()} scenarios")

    if args.scenario:
        matches = [s for s in df["scenario"].unique()
                   if s.lower().startswith(args.scenario.lower())]
        if not matches:
            print(f"Error: no scenario matching '{args.scenario}'")
            sys.exit(1)
        df = df[df["scenario"].isin(matches)]
        print(f"Filtered to: {matches}")

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent.parent / "plots" / out_subdir
    run_per_scenario(df, group_col=group_col, val_a=val_a, val_b=val_b,
                     n_components=args.components, n_top=args.top, out_dir=out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
