# fmt: off
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import psutil

# Paths
OLD_DIR = Path("/tmp/file_attachments/old_version").resolve()
NEW_DIR = Path("/app").resolve()


def generate_dataset(output_dir: Path,
                     prefix: str,
                     n_queries: int,
                     n_targets: int,
                     hits_per_query_range: tuple,
                     seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{prefix}.fna"
    tsv_path = output_dir / f"{prefix}.tsv"

    queries = [f"seq_{i}" for i in range(1, n_queries + 1)]
    targets = [f"seq_{j}" for j in range(1, n_targets + 1)]

    with fasta_path.open("w") as f:
        for q in queries:
            f.write(f">{q}\n" + "ACGT" * 250 + "\n")
        for t in targets:
            if t not in queries:
                f.write(f">{t}\n" + "ACGT" * 250 + "\n")

    total_hits = 0
    with tsv_path.open("w") as f:
        for q in queries:
            n_hits = random.randint(*hits_per_query_range)
            for _ in range(n_hits):
                t = random.choice(targets)
                if t == q:
                    continue
                q_start = random.randint(1, 800)
                q_end = q_start + random.randint(30, 150)
                if random.random() < 0.7:
                    t_start = random.randint(1, 800)
                    t_end = t_start + random.randint(30, 150)
                else:
                    t_start_raw = random.randint(1, 800)
                    t_end_raw = t_start_raw + random.randint(30, 150)
                    t_start, t_end = t_end_raw, t_start_raw

                pident = 95.0
                length = abs(q_end - q_start) + 1
                # Write standard BLAST-TAB (q in col 0, t in col 1, q_start in 6, q_end in 7, t_start in 8, t_end in 9)
                line = f"{q}\t{t}\t{pident}\t{length}\t0\t0\t{q_start}\t{q_end}\t{t_start}\t{t_end}\t1e-10\t200.0\n"
                f.write(line)
                total_hits += 1

    return fasta_path, tsv_path, total_hits


def run_single_comparison(name: str, n_q: int, n_t: int, hit_range: tuple, base_dir: Path):
    print(f"\n=======================================================")
    print(f"BENCHMARKING DATASET: {name}")
    print(f"=======================================================")
    ds_dir = base_dir / name
    fasta_path, tsv_path, total_hits = generate_dataset(ds_dir, name, n_q, n_t, hit_range)
    print(f"Dataset generated: {total_hits} hits across {n_q} queries and {n_t} targets.")

    # Create old format TSV where query is in fields[1], cols 8, 9 as required by old parser
    old_tsv_path = ds_dir / f"{name}_old.tsv"
    with tsv_path.open() as in_f, old_tsv_path.open("w") as out_f:
        for line in in_f:
            fields = line.rstrip("\n").split("\t")
            # old parser expects:
            # fields[1] = query, fields[8], fields[9] = query_start, query_end
            # fields[0] = target, fields[6], fields[7] = target_start, target_end
            # In our standard BLAST line:
            # fields[0] = q, fields[1] = t
            # fields[6] = q_s, fields[7] = q_e
            # fields[8] = t_s, fields[9] = t_e
            # So swapping fields[0] <-> fields[1] and fields[6:8] <-> fields[8:10] provides exact mapping:
            old_fields = list(fields)
            old_fields[0] = fields[1]
            old_fields[1] = fields[0]
            old_fields[6] = fields[8]
            old_fields[7] = fields[9]
            old_fields[8] = fields[6]
            old_fields[9] = fields[7]
            out_f.write("\t".join(old_fields) + "\n")

    common_params = {
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

    # 1. Run Old Pipeline in subprocess
    import subprocess
    worker_script = ds_dir / "run_old_worker.py"
    worker_script.write_text(f"""
import sys
import time
import json
import psutil
from pathlib import Path

sys.path.insert(0, "{str(OLD_DIR)}")
import mmseqs_reexaminer
import aligners

proc = psutil.Process()
mem_start = proc.memory_info().rss / (1024 * 1024)
t0 = time.perf_counter()

tsv_path = Path("{str(old_tsv_path)}")
old_search = aligners.parse_mmseqs_tsv(tsv_path)
old_finder = mmseqs_reexaminer.DomainFinder(search=old_search)

old_finder.find_domain_instances(min_dom_sim={common_params['min_dom_sim']},
                                halo_stringency={common_params['dom_halo']},
                                min_dom_dens={common_params['min_dom_dens']},
                                halo_method='{common_params['halo_method']}')
n_doms_raw = old_finder.n_domains

discarded, kept = old_finder.filter_spurious_domains(
    min_containment={common_params['min_containment']},
    reassign_targets={common_params['reassign_targets']}
)

old_finder.merge_targets()
old_finder.calculate_target_overlaps()
n_intersections = len(old_finder.target_intersections)

old_families = list(old_finder.families(min_fam_sim={common_params['min_fam_sim']},
                                        halo_stringency={common_params['fam_halo']},
                                        min_fam_dens={common_params['min_fam_dens']},
                                        halo_method='{common_params['halo_method']}'))
old_gff_features = sum(len(f[1]) for f in old_families)

elapsed = time.perf_counter() - t0
mem_peak = (proc.memory_info().rss / (1024 * 1024)) - mem_start

res = {{
    "old_domains_raw": n_doms_raw,
    "old_domains_kept": kept,
    "old_domains_discarded": discarded,
    "old_intersections": n_intersections,
    "old_families": len(old_families),
    "old_gff_features": old_gff_features,
    "old_time_sec": round(elapsed, 3),
    "old_mem_mb": round(max(mem_peak, 0.0), 2)
}}
with open("{str(ds_dir / 'old_metrics.json')}", "w") as f:
    json.dump(res, f)
""")

    print("\n[Running Old Pipeline via worker]...")
    subprocess.run([sys.executable, str(worker_script)], check=True)
    with (ds_dir / "old_metrics.json").open() as f:
        old_res = json.load(f)

    # 2. Run New Pipeline
    print("\n[Running New Pipeline]...")
    sys.path.insert(0, str(NEW_DIR))
    import domain_finder as df_new
    gc.collect()
    proc = psutil.Process()
    mem_start_new = proc.memory_info().rss / (1024 * 1024)
    t0_new = time.perf_counter()

    new_out_gff = ds_dir / "new_pred.gff"
    attrs = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_path),
        **common_params
    }

    df_new.find_domains(fasta_path=fasta_path,
                        attributes=attrs,
                        output_gff=new_out_gff,
                        cpu=4)

    elapsed_new = time.perf_counter() - t0_new
    mem_peak_new = (proc.memory_info().rss / (1024 * 1024)) - mem_start_new

    with new_out_gff.open() as f:
        new_lines = [line for line in f if not line.startswith("#") and line.strip()]
    new_gff_features = len(new_lines)

    rec = {
        "dataset": name,
        "hits": total_hits,
        "queries": n_q,
        "targets": n_t,
        **old_res,
        "new_gff_features": new_gff_features,
        "new_time_sec": round(elapsed_new, 3),
        "new_mem_mb": round(max(mem_peak_new, 0.0), 2),
        "speedup": round(old_res["old_time_sec"] / max(elapsed_new, 0.001), 2),
    }
    print(f"\nCompleted {name}: {rec}")
    return rec


def main():
    benchmarks = [
        ("small_dense", 50, 100, (100, 200)),
        ("medium_sparse", 500, 2000, (5, 15)),
        ("large_dense", 200, 500, (150, 250)),
    ]
    base_dir = Path("/tmp/empirical_benchmark")
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name, n_q, n_t, hit_range in benchmarks:
        r = run_single_comparison(name, n_q, n_t, hit_range, base_dir)
        all_results.append(r)

    with (base_dir / "all_comparison_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=======================================================")
    print("ALL EMPIRICAL BENCHMARKS COMPLETED")
    print(json.dumps(all_results, indent=2))
    print("=======================================================")


if __name__ == "__main__":
    main()
# fmt: on
