"""
plot_benchmark.py

Generates comparative bar charts for pipeline performance metrics
(MMseqs2_simweighted vs RiboSeek_simweighted vs RiboSeek_None)
across datasets, arranged by mean Adjusted Rand Index (ARI) in descending order.
Includes visualizations for glued references, split references, and pair accuracy metrics
(accurate / faulty / missing pairs) at COV 50%, 75%, and 90% thresholds.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16
})


PIPELINE_RENAME_MAP = {
    "MMseqs2_simweighted": "MMseqs2 (simweighted)",
    "RiboSeek_simweighted": "RiboSeek (simweighted)",
    "RiboSeek_None": "RiboSeek (None)"
}


def load_and_prepare_data(excel_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads benchmark dataset sheets from Excel file, calculates dataset order
    by mean ARI descending, and returns the merged DataFrame and dataset order.

    :param excel_path: Path to DPC_benchmark.xlsx
    :return: Tuple of (merged DataFrame, list of dataset names sorted by mean ARI desc)
    """
    xl = pd.ExcelFile(excel_path)
    df_list = []

    for sheet_name in xl.sheet_names:
        df_sheet = xl.parse(sheet_name)
        df_sheet["pipeline"] = PIPELINE_RENAME_MAP.get(sheet_name, sheet_name)
        df_list.append(df_sheet)

    merged_df = pd.concat(df_list, ignore_index=True)

    # Calculate mean ARI per dataset across all pipelines
    mean_ari_per_dataset = (
        merged_df.groupby("dataset")["adjusted_rand_score"]
        .mean()
        .sort_values(ascending=False)
    )

    dataset_order = mean_ari_per_dataset.index.tolist()

    # Convert dataset column to Categorical with explicit order
    merged_df["dataset"] = pd.Categorical(
        merged_df["dataset"], categories=dataset_order, ordered=True
    )
    merged_df = merged_df.sort_values("dataset")

    return merged_df, dataset_order


def create_comparative_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    annotate_values: bool = True
) -> Path:
    """
    Creates a grouped comparative bar chart for a specified metric across datasets
    arranged by mean ARI descending.

    :param df: Prepared DataFrame
    :param metric_col: Name of the metric column to plot
    :param title: Figure title
    :param ylabel: Y-axis label
    :param output_path: Destination PNG file path
    :param annotate_values: Whether to add numerical labels above bars
    :return: Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    pipelines = df["pipeline"].unique().tolist()
    datasets = df["dataset"].cat.categories.tolist()

    x = np.arange(len(datasets))
    n_pipelines = len(pipelines)
    total_width = 0.8
    bar_width = total_width / n_pipelines

    # Colors for pipelines
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for idx, pipeline in enumerate(pipelines):
        pipe_df = df[df["pipeline"] == pipeline].set_index("dataset")
        values = [pipe_df.loc[d, metric_col] if d in pipe_df.index else 0 for d in datasets]

        offset = x - (total_width / 2) + (idx + 0.5) * bar_width
        bars = ax.bar(
            offset,
            values,
            width=bar_width,
            label=pipeline,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.5
        )

        if annotate_values:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    if isinstance(height, float) and abs(height) < 10:
                        text = f"{height:.2f}"
                    else:
                        text = f"{int(round(height)):,}" if abs(height) >= 1000 else f"{int(round(height))}"

                    ax.annotate(
                        text,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90
                    )

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel("Dataset (Arranged by Mean ARI Descending)", labelpad=10, fontweight="bold")
    ax.set_ylabel(ylabel, labelpad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(title="Pipeline", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust y limits to make room for annotations if enabled
    y_max = df[metric_col].max()
    if not np.isnan(y_max):
        ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def create_pairs_breakdown_figure(
    df: pd.DataFrame,
    dataset_order: List[str],
    cov_percentage: str,
    pair_type: str,  # 'ref_pairs_in_pred' or 'pred_pairs_in_ref'
    output_path: Path
) -> Path:
    """
    Creates a 1x3 panel figure comparing Accurate (Right), Faulty (Wrong), and Missing pairs
    at a specific COV coverage threshold (e.g. '50%', '75%', '90%').

    :param df: Prepared DataFrame
    :param dataset_order: Datasets ordered by mean ARI desc
    :param cov_percentage: Coverage threshold string e.g. '50%'
    :param pair_type: 'ref_pairs_in_pred' or 'pred_pairs_in_ref'
    :param output_path: Destination PNG path
    :return: Saved file path
    """
    if pair_type == "ref_pairs_in_pred":
        prefix = "ref_pairs_in_pred_clusters"
        type_title = "Reference Pairs in Predicted Clusters"
    else:
        prefix = "pred_pairs_in_ref_clusters"
        type_title = "Predicted Pairs in Reference Clusters"

    metric_cols = [
        (f"accurate_{prefix}_COV_{cov_percentage}", "Accurate (Right) Pairs", "#2ca02c"),
        (f"faulty_{prefix}_COV_{cov_percentage}", "Faulty (Wrong) Pairs", "#d62728"),
        (f"missing_{prefix}_COV_{cov_percentage}", "Missing Pairs", "#ff7f0e")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    pipelines = df["pipeline"].unique().tolist()
    x = np.arange(len(dataset_order))
    n_pipelines = len(pipelines)
    total_width = 0.8
    bar_width = total_width / n_pipelines
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for ax_idx, (col_name, status_title, status_color) in enumerate(metric_cols):
        ax = axes[ax_idx]
        for idx, pipeline in enumerate(pipelines):
            pipe_df = df[df["pipeline"] == pipeline].set_index("dataset")
            values = [pipe_df.loc[d, col_name] if (d in pipe_df.index and col_name in pipe_df.columns) else 0 for d in dataset_order]
            offset = x - (total_width / 2) + (idx + 0.5) * bar_width

            ax.bar(
                offset,
                values,
                width=bar_width,
                label=pipeline,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.5
            )

        ax.set_title(f"{status_title} ({cov_percentage} Coverage)", fontweight="bold", fontsize=12)
        ax.set_ylabel("Pair Count", fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if ax_idx == 0:
            ax.legend(title="Pipeline", fontsize=9, title_fontsize=10)

    fig.suptitle(
        f"{type_title} Comparison at COV {cov_percentage} (Datasets Ordered by Mean ARI Descending)",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def create_summary_grid_figure(
    df: pd.DataFrame,
    dataset_order: List[str],
    output_path: Path
) -> Path:
    """
    Creates a 2x2 panel summary figure showing ARI, NMI, Glued References, and Split References.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    metrics = [
        ("adjusted_rand_score", "Adjusted Rand Index (ARI)", "ARI Score"),
        ("normalized_mutual_info_score", "Normalized Mutual Information (NMI)", "NMI Score"),
        ("n_glued_refs", "Glued Reference Clusters Count", "Number of Glued Refs"),
        ("n_split_refs", "Split Reference Clusters Count", "Number of Split Refs")
    ]

    pipelines = df["pipeline"].unique().tolist()
    x = np.arange(len(dataset_order))
    n_pipelines = len(pipelines)
    total_width = 0.8
    bar_width = total_width / n_pipelines
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for ax_idx, (metric_col, title, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for idx, pipeline in enumerate(pipelines):
            pipe_df = df[df["pipeline"] == pipeline].set_index("dataset")
            values = [pipe_df.loc[d, metric_col] if d in pipe_df.index else 0 for d in dataset_order]
            offset = x - (total_width / 2) + (idx + 0.5) * bar_width

            ax.bar(
                offset,
                values,
                width=bar_width,
                label=pipeline,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.5
            )

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if ax_idx == 0:
            ax.legend(title="Pipeline", fontsize=9, title_fontsize=10)

    fig.suptitle(
        "Performance Comparison Across Pipelines (Datasets Ordered by Mean ARI Descending)",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def generate_all_plots(excel_path: Path, output_dir: Path) -> List[Path]:
    """
    Loads data and generates all visual comparative bar chart figures.

    :param excel_path: Path to input benchmark Excel file
    :param output_dir: Directory where figures will be saved
    :return: List of generated figure file paths
    """
    df, dataset_order = load_and_prepare_data(excel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # 1. Comparative bar chart for ARI
    p_ari = create_comparative_bar_chart(
        df=df,
        metric_col="adjusted_rand_score",
        title="Adjusted Rand Index (ARI) Performance per Pipeline and Dataset",
        ylabel="Adjusted Rand Index (ARI)",
        output_path=output_dir / "ari_comparison.png"
    )
    generated_files.append(p_ari)

    # 2. Comparative bar chart for NMI
    p_nmi = create_comparative_bar_chart(
        df=df,
        metric_col="normalized_mutual_info_score",
        title="Normalized Mutual Information (NMI) Performance per Pipeline and Dataset",
        ylabel="Normalized Mutual Information (NMI)",
        output_path=output_dir / "nmi_comparison.png"
    )
    generated_files.append(p_nmi)

    # 3. Comparative bar chart for Glued References
    p_glued = create_comparative_bar_chart(
        df=df,
        metric_col="n_glued_refs",
        title="Glued Reference Clusters Count per Pipeline and Dataset",
        ylabel="Number of Glued References",
        output_path=output_dir / "glued_refs_comparison.png"
    )
    generated_files.append(p_glued)

    # 4. Comparative bar chart for Split References
    p_split = create_comparative_bar_chart(
        df=df,
        metric_col="n_split_refs",
        title="Split Reference Clusters Count per Pipeline and Dataset",
        ylabel="Number of Split References",
        output_path=output_dir / "split_refs_comparison.png"
    )
    generated_files.append(p_split)

    # 5. Comparative bar chart for Execution Time
    p_time = create_comparative_bar_chart(
        df=df,
        metric_col="full_time",
        title="Total Execution Time (Seconds) per Pipeline and Dataset",
        ylabel="Full Execution Time (s)",
        output_path=output_dir / "execution_time_comparison.png"
    )
    generated_files.append(p_time)

    # 6. Summary grid figure
    p_summary = create_summary_grid_figure(
        df=df,
        dataset_order=dataset_order,
        output_path=output_dir / "summary_metrics_comparison.png"
    )
    generated_files.append(p_summary)

    # 7-12. Pair accuracy breakdown figures at COV 50%, 75%, and 90%
    cov_thresholds = ["50%", "75%", "90%"]
    pair_types = [
        ("ref_pairs_in_pred", "ref_pairs"),
        ("pred_pairs_in_ref", "pred_pairs")
    ]

    for pair_type, label in pair_types:
        for cov in cov_thresholds:
            clean_cov = cov.replace("%", "")
            fig_path = create_pairs_breakdown_figure(
                df=df,
                dataset_order=dataset_order,
                cov_percentage=cov,
                pair_type=pair_type,
                output_path=output_dir / f"{label}_accuracy_cov_{clean_cov}.png"
            )
            generated_files.append(fig_path)

            # Standalone charts for accurate (right), faulty (wrong), missing pairs
            prefix = "ref_pairs_in_pred_clusters" if pair_type == "ref_pairs_in_pred" else "pred_pairs_in_ref_clusters"
            p_acc = create_comparative_bar_chart(
                df=df,
                metric_col=f"accurate_{prefix}_COV_{cov}",
                title=f"Accurate Pairs ({label.replace('_', ' ').title()}) at COV {cov}",
                ylabel="Accurate Pairs Count",
                output_path=output_dir / f"{label}_accurate_cov_{clean_cov}.png"
            )
            generated_files.append(p_acc)

            p_fault = create_comparative_bar_chart(
                df=df,
                metric_col=f"faulty_{prefix}_COV_{cov}",
                title=f"Faulty (Wrong) Pairs ({label.replace('_', ' ').title()}) at COV {cov}",
                ylabel="Faulty Pairs Count",
                output_path=output_dir / f"{label}_faulty_cov_{clean_cov}.png"
            )
            generated_files.append(p_fault)

            p_miss = create_comparative_bar_chart(
                df=df,
                metric_col=f"missing_{prefix}_COV_{cov}",
                title=f"Missing Pairs ({label.replace('_', ' ').title()}) at COV {cov}",
                ylabel="Missing Pairs Count",
                output_path=output_dir / f"{label}_missing_cov_{clean_cov}.png"
            )
            generated_files.append(p_miss)

    return generated_files


if __name__ == "__main__":
    excel_file = Path("DPC_benchmark.xlsx")
    plots_dir = Path("plots")
    if excel_file.exists():
        files = generate_all_plots(excel_file, plots_dir)
        print(f"Successfully generated {len(files)} comparative visualization plots in '{plots_dir}'.")
