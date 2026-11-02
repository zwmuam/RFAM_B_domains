import pytest
import numpy as np
from clustering import dpc
from domain_finder import filter_spurious_domains

def test_dpc_empty_peak_classification():
    # If all points have density below min_dens, return empty clusters and all points as noise
    # 2 isolated points with 0 similarity
    rows = np.array([], dtype=np.int64)
    cols = np.array([], dtype=np.int64)
    sims = np.array([], dtype=np.float64)
    peak_to_cluster, n_noise = dpc(
        rows=rows,
        cols=cols,
        sims=sims,
        n=2,
        min_sim=0.5,
        min_dens=10.0, # impossibly high density threshold
        filter_transparency=1.0,
        halo_method=None,
        backend="numpy"
    )
    # When rows is empty and n=2, dpc returns singletons if rows.size == 0
    # Let's test non-empty rows where max density is below min_dens:
    rows = np.array([0], dtype=np.int64)
    cols = np.array([1], dtype=np.int64)
    sims = np.array([0.1], dtype=np.float64)
    peak_to_cluster, n_noise = dpc(
        rows=rows,
        cols=cols,
        sims=sims,
        n=2,
        min_sim=0.5,
        min_dens=5.0, # min_dens 5.0, but densities are only 1.1
        filter_transparency=1.0,
        halo_method=None,
        backend="numpy"
    )
    assert len(peak_to_cluster) == 0
    assert n_noise == 2

def test_spurious_filtering_biological_priority():
    # Candidate 0: compact motif (len 50, high support N=40)
    # Candidate 1: long match (len 160, low support N=5)
    # Candidate 1 completely overlaps Candidate 0
    cluster_map = np.array([0, 1], dtype=np.int32)
    suspect_rows = np.array([0], dtype=np.int32)
    suspect_cols = np.array([1], dtype=np.int32)
    suspect_inters = np.array([50], dtype=np.int64)
    suspect_lens = np.array([50, 160], dtype=np.int64)

    # Priority ranks: candidate 0 has rank 0 (best), candidate 1 has rank 1 (worst)
    domain_ranks = np.array([0, 1], dtype=np.int32)

    # Filtering with reassign_targets=True
    filtered_map = filter_spurious_domains(
        cluster_map=cluster_map.copy(),
        suspect_rows=suspect_rows,
        suspect_cols=suspect_cols,
        suspect_intersections=suspect_inters,
        suspect_lengths=suspect_lens,
        min_containment=0.5,
        reassign_targets=True,
        domain_ranks=domain_ranks
    )
    # Candidate 1 should be pruned and reassigned to Candidate 0
    assert filtered_map[0] == 0
    assert filtered_map[1] == 0
