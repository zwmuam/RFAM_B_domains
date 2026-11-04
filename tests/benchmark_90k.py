# fmt: off
"""
Benchmark script comparing pre-revision baseline vs revised pipeline on a simulated dataset
of 90,000 MMseqs2 BLAST-TAB hits (query_fasta = target_fasta) featuring a mixture of long (500-2000 bp)
and short (100-500 bp) RNAs across variable cluster sizes (singletons to 250 members).
"""

import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil

# Add repo to sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

BASELINE_DIR = Path("/tmp/baseline_code").resolve()


def generate_simulated_dataset(output_dir: Path,
                               target_hits: int = 90_000,
                               seed: int = 42) -> Tuple[Path, Path, int]:
    """
    Generate single-fasta all-against-all dataset with mixture of long and short RNAs
    belonging to variable-sized clusters.
    """
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "simulated_rna_90k.fna"
    tsv_path = output_dir / "simulated_rna_90k.tsv"

    # Define cluster distribution:
    # mixture of singletons, small (2-10), medium (20-50), large (100-250) clusters
    cluster_sizes = (
        [1] * 50 +                  # 50 singletons
        [random.randint(2, 5) for _ in range(40)] +     # small clusters
        [random.randint(8, 20) for _ in range(25)] +    # medium clusters
        [random.randint(30, 80) for _ in range(15)] +   # larger clusters
        [random.randint(100, 250) for _ in range(5)]    # large clusters up to 250
    )

    sequences: List[Dict] = []
    seq_counter = 0

    for cluster_id, size in enumerate(cluster_sizes):
        is_long = random.random() < 0.5
        base_len = random.randint(500, 2000) if is_long else random.randint(100, 500)
        dom_start = int(base_len * random.uniform(0.1, 0.3))
        dom_len = max(30, int(base_len * random.uniform(0.4, 0.7)))
        dom_end = min(base_len, dom_start + dom_len)

        for _ in range(size):
            seq_counter += 1
            s_id = f"rna_{seq_counter:05d}"
            s_len = max(dom_end + 10, int(base_len * random.uniform(0.9, 1.1)))
            sequences.append({
                "id": s_id,
                "cluster_id": cluster_id,
                "length": s_len,
                "dom_start": dom_start,
                "dom_end": dom_end
            })

    bases = ["A", "C", "G", "U"]
    with fasta_path.open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(f">{seq['id']}\n")
            seq_str = "".join(random.choices(bases, k=seq["length"]))
            f.write(seq_str + "\n")

    print(f"Generated FASTA with {len(sequences)} sequences across {len(cluster_sizes)} clusters.")

    seq_dict = {s["id"]: s for s in sequences}
    seq_ids = [s["id"] for s in sequences]
    n_seqs = len(sequences)

    clusters: Dict[int, List[str]] = {}
    for s in sequences:
        clusters.setdefault(s["cluster_id"], []).append(s["id"])

    total_written_hits = 0
    with tsv_path.open("w", encoding="utf-8") as tsv:
        for q_seq in sequences:
            q_id = q_seq["id"]
            q_c_id = q_seq["cluster_id"]
            cluster_members = clusters[q_c_id]

            potential_targets = [t for t in cluster_members if t != q_id]
            hits_for_q = []
            if potential_targets:
                k = min(len(potential_targets), random.randint(10, 50))
                for t_id in random.sample(potential_targets, k=k):
                    t_seq = seq_dict[t_id]
                    q_s = max(1, q_seq["dom_start"] + random.randint(-5, 5))
                    q_e = min(q_seq["length"], q_seq["dom_end"] + random.randint(-5, 5))
                    t_s = max(1, t_seq["dom_start"] + random.randint(-5, 5))
                    t_e = min(t_seq["length"], t_seq["dom_end"] + random.randint(-5, 5))
                    if random.random() < 0.2:
                        t_s, t_e = t_e, t_s
                    pident = round(random.uniform(75.0, 99.0), 1)
                    aln_len = abs(q_e - q_s) + 1
                    hits_for_q.append((q_id, t_id, pident, aln_len, 0, 0, q_s, q_e, t_s, t_e, 1e-15, 300.0))

            if len(hits_for_q) < 40 and total_written_hits < target_hits:
                needed = random.randint(15, 45)
                noise_targets = [t for t in random.sample(seq_ids, k=min(needed * 2, n_seqs)) if t != q_id and t not in cluster_members][:needed]
                for t_id in noise_targets:
                    t_seq = seq_dict[t_id]
                    q_s = random.randint(1, max(1, q_seq["length"] - 50))
                    q_e = q_s + random.randint(30, min(120, q_seq["length"] - q_s))
                    t_s = random.randint(1, max(1, t_seq["length"] - 50))
                    t_e = t_s + random.randint(30, min(120, t_seq["length"] - t_s))
                    if random.random() < 0.5:
                        t_s, t_e = t_e, t_s
                    pident = round(random.uniform(60.0, 85.0), 1)
                    aln_len = abs(q_e - q_s) + 1
                    hits_for_q.append((q_id, t_id, pident, aln_len, 0, 0, q_s, q_e, t_s, t_e, 1e-5, 120.0))

            for hit in hits_for_q:
                tsv.write(f"{hit[0]}\t{hit[1]}\t{hit[2]}\t{hit[3]}\t{hit[4]}\t{hit[5]}\t{hit[6]}\t{hit[7]}\t{hit[8]}\t{hit[9]}\t{hit[10]}\t{hit[11]}\n")
                total_written_hits += 1

    if total_written_hits < target_hits:
        print(f"First pass generated {total_written_hits} hits. Expanding to {target_hits}...")
        query_hits: Dict[str, List[str]] = {}
        with tsv_path.open() as f:
            for line in f:
                q = line.split("\t", 1)[0]
                query_hits.setdefault(q, []).append(line)

        q_keys = list(query_hits.keys())
        idx = 0
        while total_written_hits < target_hits:
            q_id = q_keys[idx % len(q_keys)]
            idx += 1
            q_seq = seq_dict[q_id]
            t_id = random.choice(seq_ids)
            if t_id == q_id:
                continue
            t_seq = seq_dict[t_id]
            q_s = random.randint(1, max(1, q_seq["length"] - 50))
            q_e = q_s + random.randint(30, min(120, q_seq["length"] - q_s))
            t_s = random.randint(1, max(1, t_seq["length"] - 50))
            t_e = t_s + random.randint(30, min(120, t_seq["length"] - t_s))
            pident = 80.0
            aln_len = abs(q_e - q_s) + 1
            line = f"{q_id}\t{t_id}\t{pident}\t{aln_len}\t0\t0\t{q_s}\t{q_e}\t{t_s}\t{t_e}\t1e-6\t150.0\n"
            query_hits[q_id].append(line)
            total_written_hits += 1

        with tsv_path.open("w", encoding="utf-8") as f:
            for q_id in sequences:
                if q_id["id"] in query_hits:
                    for line in query_hits[q_id["id"]]:
                        f.write(line)

    print(f"Successfully generated dataset with {total_written_hits} BLAST-TAB hits at {tsv_path}.")
    return fasta_path, tsv_path, total_written_hits


def measure_pipeline(version: str,
                     code_dir: Path,
                     fasta_path: Path,
                     tsv_path: Path,
                     output_gff: Path,
                     cpu: int) -> Dict:
    """
    Execute pipeline in isolated subprocess and track peak memory, elapsed time, and GFF output.
    """
    runner_script = output_gff.parent / f"run_{version}_cpu{cpu}.py"
    metrics_path = output_gff.parent / f"metrics_{version}_cpu{cpu}.json"

    runner_script.write_text(f"""
import sys
import os
import time
import json
import psutil
from pathlib import Path

sys.path.insert(0, "{str(code_dir)}")
import domain_finder

if __name__ == '__main__':
    proc = psutil.Process()
    mem_start_mb = proc.memory_info().rss / (1024 * 1024)
    peak_rss_mb = mem_start_mb

    import threading
    stop_monitor = False

    def monitor():
        global peak_rss_mb
        while not stop_monitor:
            try:
                mem = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                m_mb = mem / (1024 * 1024)
                if m_mb > peak_rss_mb:
                    peak_rss_mb = m_mb
            except Exception:
                pass
            time.sleep(0.02)

    t_thread = threading.Thread(target=monitor, daemon=True)
    t_thread.start()

    t0 = time.perf_counter()
    attrs = {{
        "aligner": "mmseqs2",
        "tsv": "{str(tsv_path)}",
        "min_dom_sim": 0.35,
        "dom_halo": 0.5,
        "min_dom_dens": 1.0,
        "min_fam_sim": 0.25,
        "fam_halo": 0.5,
        "min_fam_dens": 1.0,
        "halo_method": "similarity_weighted",
        "point_specific": False,
        "reassign_targets": True,
        "min_containment": 0.6,
    }}

    domain_finder.find_domains(
        fasta_path=Path("{str(fasta_path)}"),
        attributes=attrs,
        output_gff=Path("{str(output_gff)}"),
        cpu={cpu}
    )
    elapsed = time.perf_counter() - t0
    stop_monitor = True
    t_thread.join(timeout=1.0)

    with open("{str(output_gff)}") as gff:
        lines = [line.strip() for line in gff if not line.startswith("#") and line.strip()]

    features_count = len(lines)
    res = {{
        "version": "{version}",
        "cpu": {cpu},
        "time_sec": round(elapsed, 3),
        "peak_memory_mb": round(peak_rss_mb, 2),
        "memory_growth_mb": round(max(0.0, peak_rss_mb - mem_start_mb), 2),
        "gff_features": features_count,
    }}
    with open("{str(metrics_path)}", "w") as f:
        json.dump(res, f)
""")

    print(f"\n[Benchmarking {version.upper()} with {cpu} CPU cores]...")
    subprocess.run([sys.executable, str(runner_script)], check=True)
    with metrics_path.open() as f:
        metrics = json.load(f)
    print(f"Metrics: {metrics}")
    return metrics


def run_benchmark():
    work_dir = Path("/tmp/benchmark_90k").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = work_dir / "simulated_rna_90k.fna"
    tsv_path = work_dir / "simulated_rna_90k.tsv"
    if not (fasta_path.exists() and tsv_path.exists()):
        fasta_path, tsv_path, n_hits = generate_simulated_dataset(work_dir, target_hits=90_000, seed=42)

    max_cores = psutil.cpu_count(logical=True) or 4
    core_configurations = [1, 2, max_cores]

    results = []

    for c in core_configurations:
        out_gff_baseline = work_dir / f"baseline_cpu{c}.gff"
        m_base = measure_pipeline("pre_revision", BASELINE_DIR, fasta_path, tsv_path, out_gff_baseline, cpu=c)
        results.append(m_base)

        out_gff_revised = work_dir / f"revised_cpu{c}.gff"
        m_rev = measure_pipeline("revised", REPO_DIR, fasta_path, tsv_path, out_gff_revised, cpu=c)
        results.append(m_rev)

        # Check result equivalence
        base_lines = sorted([l for l in out_gff_baseline.read_text().splitlines() if not l.startswith("#") and l.strip()])
        rev_lines = sorted([l for l in out_gff_revised.read_text().splitlines() if not l.startswith("#") and l.strip()])
        print(f"Comparison (CPU={c}): Baseline features={len(base_lines)}, Revised features={len(rev_lines)}")
        assert len(base_lines) == len(rev_lines), f"Feature count mismatch on CPU {c}!"
        assert base_lines == rev_lines, f"Feature content mismatch on CPU {c}!"

    summary_path = work_dir / "benchmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=======================================================")
    print("BENCHMARK SUMMARY (90,000 MMseqs2 HITS)")
    print("=======================================================")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_benchmark()
# fmt: on
