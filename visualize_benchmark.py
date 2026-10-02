"""
visualize_benchmark.py

Generates publication-quality figures from tests/mock_results.xlsx:
1. Violin plots comparing metric distributions across alignment pipelines.
2. Heatmap displaying normalized and raw mean performance metrics.
3. Scatter plots comparing Wall-Clock Execution Time (s) vs Peak Memory Usage (MB),
   colored by key performance metrics (Structure Conservation Index SCI, TCS, Covariation).

Saves generated figures to tests/figures/ directory.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_visualizations(excel_path: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"

    df = pd.read_excel(excel_path, sheet_name="Alignment Metrics")

    # ---------------------------------------------------------
    # 1. Violin Plots: Key Performance Metric Distributions
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ("structure_conservation_index_sci", "Structure Conservation Index (SCI)", axes[0, 0], "Blues"),
        ("transitive_consistency_score_tcs", "Transitive Consistency Score (TCS %)", axes[0, 1], "Greens"),
        ("mean_mi_apc_covariation", "Mean MI-APC Covariation", axes[1, 0], "Purples"),
        ("normalized_shannon_entropy_hn", "Normalized Shannon Entropy (H_N)", axes[1, 1], "Oranges_r"),
    ]

    for col, title, ax, palette in metrics_to_plot:
        sns.violinplot(
            data=df,
            x="pipeline",
            y=col,
            hue="pipeline",
            ax=ax,
            palette=palette,
            inner="quartile",
            density_norm="width",
            legend=False
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    violin_path = output_dir / "metric_distributions_violin.png"
    plt.savefig(violin_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Heatmap: Mean Performance Across Pipelines
    # ---------------------------------------------------------
    mean_df = df.groupby("pipeline").mean(numeric_only=True)

    heat_cols = [
        "structure_conservation_index_sci",
        "transitive_consistency_score_tcs",
        "mean_mi_apc_covariation",
        "consensus_bp_covariation_score",
        "compensatory_mutation_count",
        "normalized_shannon_entropy_hn",
        "mean_overlap_score_mos",
        "mean_sequence_similarity",
        "execution_time_seconds",
        "memory_peak_mb"
    ]

    sub_mean = mean_df[heat_cols].copy()

    # Min-max normalization for heatmap color intensity
    norm_mean = (sub_mean - sub_mean.min()) / (sub_mean.max() - sub_mean.min() + 1e-9)
    # Invert entropy so higher is better
    norm_mean["normalized_shannon_entropy_hn"] = 1.0 - norm_mean["normalized_shannon_entropy_hn"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        norm_mean.T,
        annot=sub_mean.T.round(2),
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized Relative Score (0-1)"},
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("Mean Alignment Pipeline Performance Matrix (Annotated Raw Values)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25)
    plt.tight_layout()
    heatmap_path = output_dir / "pipeline_performance_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Scatter Plot: Wall-Clock Time vs Memory Usage (Colored by SCI)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        df["execution_time_seconds"],
        df["memory_peak_mb"],
        c=df["structure_conservation_index_sci"],
        cmap="viridis",
        alpha=0.8,
        s=60,
        edgecolor="k",
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Structure Conservation Index (SCI)", fontsize=12)

    # Highlight mean points per pipeline
    for pipe, group in df.groupby("pipeline"):
        mean_time = group["execution_time_seconds"].mean()
        mean_mem = group["memory_peak_mb"].mean()
        ax.plot(mean_time, mean_mem, marker="X", markersize=14, color="red", markeredgecolor="black")
        ax.annotate(
            pipe,
            (mean_time, mean_mem),
            xytext=(10, 10),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
        )

    ax.set_title("Runtime vs Peak Memory Footprint (Colored by SCI Score)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Wall-Clock Execution Time (Seconds)", fontsize=12)
    ax.set_ylabel("Peak Memory Usage (MB)", fontsize=12)

    plt.tight_layout()
    scatter_path = output_dir / "runtime_memory_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    create_visualizations(
        excel_path=Path("tests/mock_results.xlsx"),
        output_dir=Path("tests/figures")
    )
