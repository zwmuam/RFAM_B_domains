import gc
import json
import random
import time
from pathlib import Path
import psutil
import numpy as np

from domain_finder import find_domains


def generate_simulated_tsv(tsv_path: Path, fasta_path: Path, n_hits: int, n_queries: int = 500, n_targets: int = 2000):
    """
    Generate a deterministic simulated MMseqs2 BLAST-TAB TSV file with `n_hits` total alignment records,
    grouped contiguously by query sequence ID.
    """
    random.seed(42)
    np.random.seed(42)

    queries = [f"seq_{i}" for i in range(1, n_queries + 1)]
    targets = [f"seq_{j}" for j in range(1, n_targets + 1)]

    # Generate FASTA
    with fasta_path.open("w") as f:
        for q in queries:
            f.write(f">{q}\n" + "ACGT" * 250 + "\n")
        for t in targets:
            if t not in queries:
                f.write(f">{t}\n" + "ACGT" * 250 + "\n")

    hits_per_query = n_hits // n_queries
    extra_hits = n_hits % n_queries

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w") as f:
        for i, q in enumerate(queries):
            num_h = hits_per_query + (1 if i < extra_hits else 0)
            for _ in range(num_h):
                t = random.choice(targets)
                if t == q:
                    continue
                q_start = random.randint(1, 800)
                q_end = q_start + random.randint(30, 150)

                # Strand orientation: 80% forward, 20% reverse
                if random.random() < 0.8:
                    t_start = random.randint(1, 800)
                    t_end = t_start + random.randint(30, 150)
                else:
                    t_start_raw = random.randint(1, 800)
                    t_end_raw = t_start_raw + random.randint(30, 150)
                    t_start, t_end = t_end_raw, t_start_raw  # Inverted target coordinates

                pident = 95.0
                length = abs(q_end - q_start) + 1
                line = f"{t}\t{q}\t{pident}\t{length}\t0\t0\t{t_start}\t{t_end}\t{q_start}\t{q_end}\t1e-10\t200.0\n"
                f.write(line)


def run_baseline_benchmark(dataset_name: str, n_hits: int):
    bench_dir = Path(f"/tmp/benchmark_tmp/{dataset_name}")
    bench_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = bench_dir / "simulated.tsv"
    fasta_path = bench_dir / "sequences.fna"
    out_gff = bench_dir / "baseline.gff"

    if not tsv_path.exists():
        print(f"Generating simulated dataset {dataset_name} ({n_hits} hits)...")
        generate_simulated_tsv(tsv_path, fasta_path, n_hits=n_hits)

    attributes = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_path),
        "min_dom_sim": 0.35,
        "dom_halo": 0.5,
        "min_dom_dens": 1.0,
        "min_fam_sim": 0.25,
        "fam_halo": 0.5,
        "min_fam_dens": 1.0,
        "halo_method": "knn",
        "reassign_targets": True,
        "min_containment": 0.6,
    }

    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()
    find_domains(
        fasta_path=fasta_path,
        attributes=attributes,
        output_gff=out_gff,
        tmp_dir=bench_dir / "tmp",
        result_dir=bench_dir / "results"
    )
    elapsed = time.perf_counter() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_mem = max(mem_after, mem_before)

    # Read output GFF features count
    with out_gff.open("r") as f:
        gff_lines = [line for line in f if not line.startswith("#") and line.strip()]

    result_data = {
        "dataset": dataset_name,
        "n_hits": n_hits,
        "runtime_sec": round(elapsed, 3),
        "peak_mem_mb": round(peak_mem, 2),
        "gff_feature_count": len(gff_lines),
    }

    print(f"Baseline {dataset_name} result: {result_data}")
    with (bench_dir / "baseline_metrics.json").open("w") as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    run_baseline_benchmark("simulated_25k", 25000)
    run_baseline_benchmark("simulated_250k", 250000)
