"""
test_visualizations.py

Unit tests for plot_benchmark.py module.
"""

from pathlib import Path
import pandas as pd
import pytest

from plot_benchmark import load_and_prepare_data, generate_all_plots, create_comparative_bar_chart


@pytest.fixture
def mock_excel_file(tmp_path: Path) -> Path:
    excel_path = tmp_path / "mock_benchmark.xlsx"

    # Create 3 mock sheets with 3 datasets each
    datasets = ["dataset_A", "dataset_B", "dataset_C"]

    # Sheet 1
    data1 = {
        "dataset": datasets,
        "adjusted_rand_score": [0.2, 0.8, 0.5],
        "normalized_mutual_info_score": [0.3, 0.85, 0.55],
        "n_glued_refs": [10, 2, 5],
        "n_split_refs": [12, 1, 6],
        "full_time": [100.0, 200.0, 150.0]
    }

    # Sheet 2
    data2 = {
        "dataset": datasets,
        "adjusted_rand_score": [0.3, 0.9, 0.6],
        "normalized_mutual_info_score": [0.35, 0.92, 0.65],
        "n_glued_refs": [8, 1, 4],
        "n_split_refs": [10, 0, 5],
        "full_time": [110.0, 210.0, 160.0]
    }

    # Sheet 3
    data3 = {
        "dataset": datasets,
        "adjusted_rand_score": [0.4, 0.7, 0.7],
        "normalized_mutual_info_score": [0.45, 0.75, 0.75],
        "n_glued_refs": [7, 3, 3],
        "n_split_refs": [9, 2, 4],
        "full_time": [105.0, 205.0, 155.0]
    }

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

    # Check dataset ordering by mean ARI desc
    # Dataset mean ARIs:
    # dataset_A: (0.2 + 0.3 + 0.4) / 3 = 0.3
    # dataset_B: (0.8 + 0.9 + 0.7) / 3 = 0.8
    # dataset_C: (0.5 + 0.6 + 0.7) / 3 = 0.6
    # Order should be ['dataset_B', 'dataset_C', 'dataset_A']
    assert dataset_order == ["dataset_B", "dataset_C", "dataset_A"]


def test_generate_all_plots(mock_excel_file: Path, tmp_path: Path):
    out_dir = tmp_path / "plots"
    files = generate_all_plots(mock_excel_file, out_dir)

    assert len(files) == 6
    for f in files:
        assert f.exists()
        assert f.stat().st_size > 0
