"""
test_visualizations.py

Unit tests for plot_benchmark.py module.
"""

from pathlib import Path
import pandas as pd
import pytest

from plot_benchmark import load_and_prepare_data, generate_all_plots, create_comparative_bar_chart, create_pairs_breakdown_figure


@pytest.fixture
def mock_excel_file(tmp_path: Path) -> Path:
    excel_path = tmp_path / "mock_benchmark.xlsx"

    # Create 3 mock sheets with 3 datasets each
    datasets = ["dataset_A", "dataset_B", "dataset_C"]

    def build_mock_dict(ari_vals):
        d = {
            "dataset": datasets,
            "adjusted_rand_score": ari_vals,
            "normalized_mutual_info_score": [0.3, 0.85, 0.55],
            "n_glued_refs": [10, 2, 5],
            "n_split_refs": [12, 1, 6],
            "full_time": [100.0, 200.0, 150.0]
        }
        for cov in ["50%", "75%", "90%"]:
            for prefix in ["ref_pairs_in_pred_clusters", "pred_pairs_in_ref_clusters"]:
                d[f"accurate_{prefix}_COV_{cov}"] = [100, 200, 150]
                d[f"faulty_{prefix}_COV_{cov}"] = [10, 20, 15]
                d[f"missing_{prefix}_COV_{cov}"] = [5, 2, 8]
        return d

    data1 = build_mock_dict([0.2, 0.8, 0.5])
    data2 = build_mock_dict([0.3, 0.9, 0.6])
    data3 = build_mock_dict([0.4, 0.7, 0.7])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(data1).to_excel(writer, sheet_name="MMseqs2_simweighted", index=False)
        pd.DataFrame(data2).to_excel(writer, sheet_name="RiboSeek_simweighted", index=False)
        pd.DataFrame(data3).to_excel(writer, sheet_name="RiboSeek_None", index=False)

    return excel_path


def test_load_and_prepare_data(mock_excel_file: Path):
    df, dataset_order = load_and_prepare_data(mock_excel_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9  # 3 datasets * 3 sheets
    assert "pipeline" in df.columns

    # Order should be ['dataset_B', 'dataset_C', 'dataset_A']
    assert dataset_order == ["dataset_B", "dataset_C", "dataset_A"]


def test_generate_all_plots(mock_excel_file: Path, tmp_path: Path):
    out_dir = tmp_path / "plots"
    files = generate_all_plots(mock_excel_file, out_dir)

    # 6 base charts + (2 pair types * 3 covs * 4 files per cov) = 6 + 24 = 30 files
    assert len(files) == 30
    for f in files:
        assert f.exists()
        assert f.stat().st_size > 0


def test_create_pairs_breakdown_figure(mock_excel_file: Path, tmp_path: Path):
    df, dataset_order = load_and_prepare_data(mock_excel_file)
    out_path = tmp_path / "pairs_breakdown.png"

    fig_path = create_pairs_breakdown_figure(
        df=df,
        dataset_order=dataset_order,
        cov_percentage="50%",
        pair_type="ref_pairs_in_pred",
        output_path=out_path
    )

    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
