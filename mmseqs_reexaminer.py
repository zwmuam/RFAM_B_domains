#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jakub Barylski"
__maintainer__ = "Jakub Barylski"
__license__ = "GNU GENERAL PUBLIC LICENSE"
__email__ = "jakub.barylski@gmail.com"

from collections import defaultdict
from itertools import count
from pathlib import Path
from time import time
from typing import Dict, List, Generator, Tuple

import numpy as np
import pandas as pd
import namer
from scipy.sparse import csgraph, coo_matrix
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from shutil import rmtree
from tqdm import tqdm

from aligners import parse_mmseqs_tsv, Span
from clustering import (compare_sorted_spans,
                        jaccard_index,
                        max_containment,
                        dpc)
from evaluation import evaluate
from tweaks import default_threads, logger, timeout_signal

# kill after 30 minutes (1800 seconds) to prevent hanging on large datasets
@timeout_signal(seconds=1800)
def domain_families(tsv: Path,
                    min_dom_sim: float,
                    dom_halo: float,
                    min_dom_dens: float,
                    min_fam_sim: float,
                    fam_halo: float,
                    min_fam_dens: float,
                    halo_method: str,
                    reassign_targets: bool,
                    min_containment: float) -> Dict[str, float]:
    """
    Runs mmseqs, build and partition alignment network and extract and reorient sequences
    :param tsv: path to the MMseqs2 or RiboSeek output TSV file containing the alignment results
    :param min_dom_sim: minimum domain similarity for clustering
    :param dom_halo: stringency of the halo filtering during last DPC step (higher values are more stringent)
    :param min_dom_dens: minimum domain density for clustering
    :param min_fam_sim: minimum family similarity for clustering
    :param fam_halo: halo family similarity for clustering
    :param min_fam_dens: minimum family density for clustering
    :param halo_method: halo filtering algorithm: 'traditional' or 'knn'
    :param reassign_targets: minimum coverage ratio for filtering spurious domains
    :param min_containment: minimum containment for filtering spurious domains
    :return: adjusted Rand score of the predicted domain families against the reference GFF
    """
    start_time = time()
    search = parse_mmseqs_tsv(tsv)
    run_name = namer.generate(category='microbiology', separator='_')
    report_subdir = TMP_DIR.joinpath(f'{run_name}')
    report_subdir.mkdir(parents=True, exist_ok=True)

    report_gff = report_subdir.joinpath('report.gff')

    domains = DomainFinder(search=search)
    domains.find_domain_instances(min_dom_sim=min_dom_sim,
                                  halo_stringency=dom_halo,
                                  min_dom_dens=min_dom_dens,
                                  halo_method=halo_method)
    ds_discarded, ds_kept = domains.filter_spurious_domains(min_containment=min_containment,
                                                            reassign_targets=reassign_targets)
    metric_dict = {'n_domains_discarded': ds_discarded,
                   'n_domains_kept': ds_kept,
                   'run_name': run_name}

    domains.merge_targets()
    domains.calculate_target_overlaps()

    with report_gff.open('w') as gff_writer:
        for fam_id, fam_domains in domains.families(min_fam_sim=min_fam_sim,
                                                    halo_stringency=fam_halo,
                                                    min_fam_dens=min_fam_dens,
                                                    halo_method=halo_method):
            for domain in fam_domains:
                gff_writer.write(domain.to_gff(cluster_id=fam_id) + '\n')

    metric_dict.update(evaluate(fasta_path=FASTA_PATH,
                                ground_truth_gff=REF_GFF,
                                pred_gff=report_gff,
                                annotation_tsv=annotation_tsv_path,
                                report_dir=report_subdir))

    metric_dict.update({'n_halo_alns_ignored_during_domain_finding': domains.n_halo_alns,
                        'n_halo_domains_ignored_during_family_finding': domains.n_halo_domains,
                        'elapsed_time': time() - start_time})

    return metric_dict


class DomainInstance(Span):
    instance_counter = count()

    __slots__ = ('seq_id', 'start', 'end', 'length', 'number', 'strand', 'targets')

    def __init__(self,
                 seq_id: str,
                 start: int,
                 summary_strand: int,
                 end: int,
                 targets: List[Span]):
        super().__init__(seq_id=seq_id,
                         start=start,
                         end=end,
                         strand=(summary_strand > 0) - (summary_strand < 0),
                         partners=targets)

    def to_gff(self, cluster_id) -> str:
        strand_string = '+' if self.strand > 0 else '-' if self.strand < 0 else '.'
        return f'{self.seq_id}\tmmseqs2\tDomain\t{self.start}\t{self.end}\t.\t{strand_string}\t.\tID={cluster_id}'

    def total_target_length(self) -> int:
        return sum(target.length for target in self.partners)

    def coverage(self) -> float:
        """
        Return number of joined targets as a proxy of the alignment coverage of the domain instance.
        :return: coverage ratio
        """
        return len(self.partners)


class DomainFinder:
    """
    Selects representative conserved domain instances from the alignments on all
    Each instance corresponds to distinct peak of the alignment density on a given query seq
    A sequence may contain multiple distinct instances, that will be classified to
    respective domain families in the next step of the pipeline.
    Hence, "domain instances" are equivalent of "primary clusters" in DPCfam (Russo et al. 2022)
    """

    def __init__(self,
                 search: dict[str, List[Span]]) -> None:
        """

        :param search: dictionary of query sequences and their corresponding list of alignment spans
        """
        self.sorted_queries = search
        self.domains_per_seq = {}
        self.n_domains = 0
        self.domain_index = {}
        self.domain_total_lengths = {}
        self.targets_per_seq = {}
        self.target_intersections = defaultdict(lambda: 0)
        self.n_halo_alns = 0
        self.n_halo_domains = 0

    def find_domain_instances(self,
                              min_dom_sim: float,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: str):
        """
        Find domain instances (conserved regions that align to multiple targets) by
        analysing the alignment coverage of each query sequence and identifying peaks in the alignment density.
        This is analogous to the primary clustering step in DPCfam (Russo et al. 2022).
        """
        with tqdm(total=len(self.sorted_queries), desc="Finding domain instances") as pbar:
            for sorted_query_spans in self.sorted_queries.values():
                n_spans = len(sorted_query_spans)
                if n_spans == 0:
                    raise ValueError('Impossible')  # TODO REMOVE
                elif n_spans == 1:
                    peak_span, = sorted_query_spans
                    self.create_new_domain(peak_span=peak_span,
                                           targets=peak_span.partners)
                else:
                    rows, cols, intersections = compare_sorted_spans(sorted_query_spans)
                    if rows.size == 0:
                        # fully disjoined nodes
                        peak_to_cluster = {p: np.array([p]) for p in np.arange(n_spans)}
                    else:
                        span_lengths = np.array([s.length for s in sorted_query_spans])
                        weights = np.array([s.n_partners() for s in sorted_query_spans])

                        similarities = jaccard_index(intersections=intersections,
                                                     row_lengths=span_lengths[rows],
                                                     column_lengths=span_lengths[cols])

                        peak_to_cluster, n_noise = dpc(rows=rows,
                                                       cols=cols,
                                                       sims=similarities,
                                                       n=n_spans,
                                                       min_sim=min_dom_sim,
                                                       weights=weights,
                                                       backend='numpy',
                                                       filter_transparency=halo_stringency,
                                                       halo_method=halo_method,
                                                       min_dens=min_dom_dens)
                        self.n_halo_alns += n_noise

                    for peak_id, member_ids in peak_to_cluster.items():
                        peak_span = sorted_query_spans[peak_id]
                        member_targets = []
                        for m_id in member_ids:
                            member_targets.extend(sorted_query_spans[int(m_id)].partners)
                        self.create_new_domain(peak_span=peak_span,
                                               targets=member_targets)

                pbar.update(1)
        logger.info(f'Found {self.n_domains} domain instances across {len(self.sorted_queries)} query sequences')

    def create_new_domain(self,
                          peak_span: Span,
                          targets: List[Span]):

        """
        Create a new domain instance based on the peak span and
        bind it to the corresponding targets.
        Overlapping target spans are merged to prevent faulty overlap calculations.
        :param peak_span: query span that is the density peak used to define the domain instance
        :param targets: targets of all query alignments assigned to the peak cluster
        """

        raw_targets = defaultdict(list)
        strands = []
        for target in targets:
            raw_targets[target.seq_id].append(target)
            strands.append(target.strand)
        # add the peak span to the list of targets to ensure it is included in the overlap calculations
        supplemented_targets = targets + [peak_span]
        new_domain = DomainInstance(seq_id=peak_span.seq_id,
                                    start=peak_span.start,
                                    end=peak_span.end,
                                    summary_strand=sum(strands),
                                    targets=supplemented_targets)
        if new_domain.seq_id not in self.domains_per_seq:
            self.domains_per_seq[new_domain.seq_id] = []
        self.domains_per_seq[new_domain.seq_id].append(new_domain)
        self.n_domains += 1

    def filter_spurious_domains(self,
                                min_containment: float,
                                reassign_targets: bool):
        """
        Filter out spurious domains that formed by random merging of nearby domains

        :param min_containment: minimal Jaccard containment to consider a domain embedded in another domain
        :param reassign_targets:
        :return:
        """
        dc = count()
        n_discarded = 0
        n_kept = 0
        for seq_id, domain_list in self.domains_per_seq.items():
            sorted_domains = list(sorted(domain_list, key=lambda d: d.start))
            domain_lengths = np.array([s.length for s in sorted_domains])
            rows, columns, intersections = compare_sorted_spans(sorted_domains)
            if rows.size == 0:
                connected_mask = np.array([], dtype=bool)
            else:
                containment = max_containment(intersections=intersections,
                                              row_lengths=domain_lengths[rows],
                                              column_lengths=domain_lengths[columns])
                connected_mask = containment >= min_containment
            if not any(connected_mask):
                kept_domains = sorted_domains
                n_kept += len(kept_domains)
            else:
                sources, targets = rows[connected_mask], columns[connected_mask]
                remaining_edges = [{s, t} for s, t in zip(sources, targets)]
                indices_to_remove = set()
                while remaining_edges:
                    connected_dom_indices = set()
                    for edge in remaining_edges:
                        connected_dom_indices.update(edge)
                    connected_domains = {i: sorted_domains[i] for i in connected_dom_indices}
                    # domains are sorted by alignment coverage (number of targets) to remove the least covered first
                    # ties are resolved by the length (longer alignments may be formed by merging of nearby domains)
                    scoring_dict = {i: (d.coverage(), -d.length) for i, d in connected_domains.items()}
                    # remove the node with the lowest score (least covered) and all its edges
                    weak_i = min(connected_domains, key=lambda i: scoring_dict[i])
                    indices_to_remove.add(weak_i)
                    if reassign_targets:
                        removed_domain = sorted_domains[weak_i]
                        removed_edges = [e for e in remaining_edges if weak_i in e]
                        # find nodes (domains) connected to the removed node (based on the edges)
                        connected_to_removed = set()
                        for edge in removed_edges:
                            connected_to_removed.update(edge)
                        connected_to_removed.discard(weak_i)
                        # resolve ties by length (longer alignments may be formed by merging of nearby domains)
                        heir_domain = connected_domains[max(connected_to_removed, key=lambda i: scoring_dict[i])]
                        # reassign targets of the removed domain to the strongest connected node
                        heir_domain.partners.extend(removed_domain.partners)
                    remaining_edges = [e for e in remaining_edges if weak_i not in e]
                # remove the weakest domains from the list
                kept_domains = [d for i, d in enumerate(sorted_domains) if i not in indices_to_remove]
                n_discarded += len(indices_to_remove)
                n_kept += len(kept_domains)

            self.domain_index.update({next(dc): d for d in kept_domains})

        return n_discarded, n_kept

    def merge_targets(self):
        """
        Merge overlapping targets to prevent faulty overlap calculations and impossible Jaccard indices.
        This fuction is also the only place where the domain pointers are set and domain_total_lengths are calculated,
        so it has to be called before calculating the target overlaps.
        """
        # overlapping target have to be merged to prevent faulty overlap calculations and impossible Jaccard indices
        logger.info(f'Merging overlapping targets for {len(self.domain_index)} domains')
        n_merged = 0
        n_distinct = 0
        n_singletons = 0
        with tqdm(total=len(self.domain_index), desc="Merging overlapping targets") as pbar:
            for d_id, domain in self.domain_index.items():
                self.domain_total_lengths[d_id] = 0
                domain_targets_by_seq = defaultdict(list)
                for target in domain.partners:
                    domain_targets_by_seq[target.seq_id].append(target)
                for t_seq_id, targets in domain_targets_by_seq.items():
                    if t_seq_id not in self.targets_per_seq:
                        self.targets_per_seq[t_seq_id] = []
                    n_targets = len(targets)
                    if n_targets > 1:
                        sorted_group = sorted(targets, key=lambda s: s.start)
                        rows, cols, intersections = compare_sorted_spans(sorted_group)
                        similarity_matrix = coo_matrix((intersections, (rows, cols)),
                                                       shape=(n_targets, n_targets)).tocsr()
                        components, labels = csgraph.connected_components(similarity_matrix, directed=False,
                                                                          return_labels=True)
                        for component_id in range(components):
                            component_index = {i: s for i, s in enumerate(sorted_group) if labels[i] == component_id}
                            if len(component_index) == 1:
                                distinct_target, = component_index.values()
                                distinct_target.domain = d_id
                                self.targets_per_seq[t_seq_id].append(distinct_target)
                                self.domain_total_lengths[d_id] += distinct_target.length
                                n_distinct += 1
                            else:
                                merged_start = min([s.start for s in component_index.values()])
                                merged_end = max([s.end for s in component_index.values()])
                                # strand becomes irrelevant after merging, so we set it to 0 (unknown)
                                merged_target = Span(seq_id=t_seq_id,
                                                     start=merged_start,
                                                     end=merged_end,
                                                     strand=0)
                                merged_target.domain = d_id
                                self.targets_per_seq[t_seq_id].append(merged_target)
                                self.domain_total_lengths[d_id] += merged_target.length
                                n_merged += len(component_index) - 1
                    else:
                        singleton_target, = targets
                        singleton_target.domain = d_id
                        self.targets_per_seq[t_seq_id].append(singleton_target)
                        self.domain_total_lengths[d_id] += singleton_target.length
                        n_singletons += 1
                pbar.update(1)

        logger.info(f'Merged: {n_merged}; Distinct: {n_distinct}; Singletons: {n_singletons}')

    def calculate_target_overlaps(self):
        """
        Calculate overlaps between targets and update the global sparse similarity matrix.
        :return: None
        """
        with tqdm(total=len(self.targets_per_seq), desc="Calculating target overlaps") as pbar:
            for seq_id, targets in self.targets_per_seq.items():
                n_targets = len(targets)
                if n_targets > 1:
                    local_rows, local_cols, intersections = compare_sorted_spans(targets)
                    if local_rows.size > 0:
                        domain_ids = np.array([t.domain for t in targets])
                        global_rows = domain_ids[local_rows]
                        global_cols = domain_ids[local_cols]
                        for r, c, intersect in zip(global_rows, global_cols, intersections):
                            canonical_key = tuple(sorted([r, c]))
                            assert r != c, f'Unexpected self-overlap for domain {r} in sequence {seq_id}'
                            self.target_intersections[canonical_key] += intersect
                pbar.update(1)

    def families(self,
                 min_fam_sim: float,
                 min_fam_dens: float,
                 halo_stringency: float,
                 halo_method: str) -> Generator[Tuple[str, List[DomainInstance]], None, None]:
        """
        Get connected components assuming that unconnected edges
        simply do not exist in the internal "self" dictionary (was never set).
        Only existing edges are considered, so the dict keys are the only edges in the graph.
        Then generate jaccard matrices for each connected component and yield exported as numpy arrays
        """

        # filter the noise (using Jaccard DISTANCE instead of similarity so we keep those with distance <= threshold)
        if len(self.target_intersections) == 0:
            logger.warning(f'No edges in the overlap graph. This may indicate error in input data or noise threshold.')
        else:
            n_domains = len(self.domain_index)
            n_intersections = len(self.target_intersections)

            logger.info(f'Calculating Jaccard similarities for {n_domains} domains')
            rows = np.zeros(n_intersections, dtype=np.int32)
            cols = np.zeros(n_intersections, dtype=np.int32)
            row_domain_lengths = np.zeros(n_intersections, dtype=np.int32)
            col_domain_lengths = np.zeros(n_intersections, dtype=np.int32)
            intersections = np.zeros(n_intersections, dtype=np.int32)
            for i, ((row, col), intersection) in enumerate(self.target_intersections.items()):
                rows[i] = row
                cols[i] = col
                intersections[i] = intersection
                row_domain_lengths[i] = self.domain_total_lengths[row]
                col_domain_lengths[i] = self.domain_total_lengths[col]
                if row_domain_lengths[i] <= 0 or col_domain_lengths[i] <= 0:
                    raise ValueError(f'Unexpected non-positive domain length for domains {row} and {col}. '
                                     f'Row length: {row_domain_lengths[i]}, Col length: {col_domain_lengths[i]}')
            similarities = jaccard_index(intersections=intersections,
                                         row_lengths=row_domain_lengths,
                                         column_lengths=col_domain_lengths)
            logger.info(f'Clustering {n_domains} domains into families using DPC with min_fam_sim={min_fam_sim}, '
                        f'halo_stringency={halo_stringency}, min_fam_dens={min_fam_dens}, halo_method={halo_method}')
            peak_to_cluster, n_halo = dpc(rows=rows,
                                          cols=cols,
                                          sims=similarities,
                                          n=n_domains,
                                          min_sim=min_fam_sim,
                                          backend='scipy',
                                          filter_transparency=halo_stringency,
                                          halo_method=halo_method,
                                          min_dens=min_fam_dens)
            self.n_halo_domains = n_halo

            n_clusters = len(peak_to_cluster)

            with tqdm(total=n_clusters, desc="Evaluating clusters") as pbar:
                for peak_id, member_ids in peak_to_cluster.items():
                    # sort the domains back to original sequences
                    cluster_domains_per_seq = defaultdict(list)
                    for d_id in member_ids:
                        d = self.domain_index[int(d_id)]
                        cluster_domains_per_seq[d.seq_id].append(d)

                    component_domains = []
                    # merge any overlapping same-cluster domains (alternate versions of the same domain) to clean up the output
                    for seq_id, domains in cluster_domains_per_seq.items():
                        sorted_domains = sorted(domains, key=lambda d: d.start)
                        rows, cols, intersections = compare_sorted_spans(sorted_domains)
                        if rows.size == 0:
                            component_domains.extend(sorted_domains)
                        else:
                            similarity_matrix = coo_matrix((intersections, (rows, cols)),
                                                           shape=(len(sorted_domains), len(sorted_domains))).tocsr()
                            components, labels = csgraph.connected_components(similarity_matrix, directed=False,
                                                                              return_labels=True)
                            for component_id in range(components):
                                component_index = {i: s for i, s in enumerate(sorted_domains) if
                                                   labels[i] == component_id}
                                if len(component_index) == 1:
                                    distinct_domain, = component_index.values()
                                    component_domains.append(distinct_domain)
                                else:
                                    merged_start = min([s.start for s in component_index.values()])
                                    merged_end = max([s.end for s in component_index.values()])
                                    # strand becomes irrelevant after merging, so we set it to 0 (unknown)
                                    merged_domain = DomainInstance(seq_id=seq_id,
                                                                   start=merged_start,
                                                                   end=merged_end,
                                                                   summary_strand=0,
                                                                   targets=[self.domain_index[int(i)] for i in
                                                                            component_index])
                                    component_domains.append(merged_domain)
                    pbar.update(1)
                    yield f'{peak_id}', component_domains


if __name__ == '__main__':
    FASTA_PATH = Path('/home/michalstanoch/data/mmseq2net2/mutated_datasets/AsB_chimeras.spacer_1_mut/sequences.fna')
    REF_GFF = Path('/home/michalstanoch/data/mmseq2net2/mutated_datasets/AsB_chimeras.spacer_1_mut/annotations.gff')
    SEARCH_TSV_DIR = Path('/home/michalstanoch/data/mmseq2net2/reevaluation')
    TMP_DIR = Path('/home/michalstanoch/data/mmseq2net2/reevaluation_run_data')

    SEARCH_RESULT_DIR = TMP_DIR.joinpath('results_tsvs')
    SEARCH_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMISER_RECORDS = []
    OPTIMISER_TABLE = TMP_DIR.joinpath('optimiser_table.xlsx')
    TIMEOUT = 900  # seconds
    main_log_file = TMP_DIR.joinpath('network_builderH_opt.log')
    annotation_tsv_path = Path('original_datasets/rfam_sample_joined.tsv')
    logger.add(main_log_file, level="INFO")

    rnaseek_space = [Categorical([f for f in SEARCH_TSV_DIR.iterdir() if f.suffix == '.tsv'], name='tsv'),
                     Integer(5, 15, name='min_dom_sim'),
                     Integer(1, 20, name='dom_halo'),
                     Integer(0, 3, name='min_dom_dens'),
                     Integer(1, 10, name='min_fam_sim'),
                     Integer(1, 20, name='fam_halo'),
                     Integer(0, 5, name='min_fam_dens'),
                     Categorical([None, 'traditional', 'similarity_weighted', 'knn', 'ndcg'], name='halo_method'),
                     Categorical([0, 1], name='reassign_targets'),
                     Integer(10, 20, name='min_containment')]


    def divide_by(value, divisor):
        return value / divisor


    transformed_parsms = {'min_dom_sim': (divide_by, 20),
                          'dom_halo': (divide_by, 20),
                          'min_fam_sim': (divide_by, 20),
                          'fam_halo': (divide_by, 20),
                          'min_containment': (divide_by, 20)}


    @use_named_args(rnaseek_space)
    def objective(**params):
        """
        Objective function for Bayesian hyperparameter optimisation.
        :param params: entire parameter space to be optimised, including MMseqs and DPC parameters
        :return:
        """

        if not hasattr(objective, "iteration"):
            objective.iteration = 0

        transformed_params = {}
        for param_name, param_value in params.items():
            if param_name in transformed_parsms:
                transform_func, divisor = transformed_parsms[param_name]
                transformed_params[param_name] = transform_func(param_value, divisor)
            else:
                transformed_params[param_name] = param_value

        report_dict = {**transformed_params}
        report_dict['iteration'] = objective.iteration

        try:
            metric_dict = domain_families(**transformed_params)
            evaluation_score = metric_dict['adjusted_rand_score']
            logger.info(f"\n{report_dict}")
            # gather all params and metrics into a dictionary for logging
            report_dict.update(metric_dict)

        except Exception as e:
            logger.error(f"Error occurred while evaluating parameters: {e}")
            report_dict.update({'exception': str(e)})
            evaluation_score = 0
            report_dict.update({'adjusted_rand_score': evaluation_score})

        OPTIMISER_RECORDS.append(report_dict)

        with pd.ExcelWriter(OPTIMISER_TABLE, engine='openpyxl', mode='w') as writer:
            # sort the records in descending order of adjusted Rand score before saving
            pd.DataFrame.from_records(OPTIMISER_RECORDS).sort_values(by='adjusted_rand_score',
                                                                     ascending=False).to_excel(writer, index=False)
            logger.info(f"Saved optimisation records to {OPTIMISER_TABLE}")

        objective.iteration += 1
        return -evaluation_score  # score is minimized (treated as penalty)


    # optimise for maximum adjusted Rand score and save best parameters
    res = gp_minimize(objective,
                      rnaseek_space,
                      n_initial_points=150,
                      n_calls=200)
