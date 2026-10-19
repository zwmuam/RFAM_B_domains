from collections import defaultdict, Counter
from itertools import count, combinations
from math import comb as n_comb
from pathlib import Path
from typing import Tuple, Dict, Collection

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from annotations import NucleotideSeq
from tweaks import logger

u_prefix = 'UNL_'  # prefix for generated IDs of unlabeled regions (e.g., UNL_1, UNL_2, etc.)


def read_fasta(fasta_path: Path) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Dict]]:
    """
    Read fasta file to create the dict of zero-initialized numpy arrays
    :param fasta_path: path to fasta file to read
    :return: 1. dict mapping sequence id to offset of the sequence in the concatenated sequence array
    """
    seq_offsets = {}

    offset_start, offset_end, n = 0, 0, 0
    for n, seq in enumerate(NucleotideSeq.read_fasta(fasta_path), 1):
        offset_start, offset_end = offset_start, offset_start + seq.length
        seq_offsets[seq.seq_id] = (offset_start, offset_end)
        offset_start = offset_end

    general_stats = {'input': {'total_seqs': n,
                               'total_nt': offset_end}}

    return seq_offsets, general_stats


def read_gff(gff_path: Path,
             seq_offsets: Dict[str, Tuple[int, int]],
             label_type: str,
             annotation_tsv: Path = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Read single gff file to generate label dictionary.
    :param gff_path: path to gff file to read
    :param seq_offsets: dict mapping sequence id to offset of the sequence in the concatenated sequence array
    :param label_type: string identifier of the label type (e.g. 'ref'
    :param annotation_tsv: path to csv file mapping reference IDs to human-readable names (optional, only for reference gff)
    :return: dataframe gathering all domain instances (one per row) with columns:
             1. 'label' - string identifier of the domain type (e.g. RFAM ID)
             2. 'seq_id' - string identifier of the sequence (e.g. accession)
             3. 'start' - 0-based index of the domain 5' terminus in sequence
             4. 'end' - 0-based index of the domain 3' terminus in sequence (start < end)
             5. 'numeric_id' - 1-based unique integer ID assigned to the label (0 is for the unlabeled)
             6. 'length' - length of the domain in nucleotide positions
    """
    label_positions = defaultdict(list)
    domain_records = []
    label_type_stats = {'domains': defaultdict(lambda: 0),
                        'nucleotides': defaultdict(lambda: 0),
                        'seqs': defaultdict(set),
                        'fully_labeled_seqs': 0,
                        'partially_labeled_seqs': 0,
                        'fully_UNLABELED_seqs': 0,
                        'UNLABELED_regions': 0,
                        'UNLABELED_nucleotides': 0}
    with gff_path.open() as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # skip comments and empty lines
            seq_id, _, feature, start, end, _, _, _, attributes = line.strip().split('\t')
            seq_offset = seq_offsets[seq_id]  # get sequence offset for the current seq_id
            if feature in ('source', 'Domain'):
                start, end = int(start) - 1, int(end)  # adjust to 0-based array indexing
                label = attributes.split('ID=')[1].split(';')[0]  # extract label from attributes
                label_positions[seq_id].append((label, start, end))
                domain_records.append({'label': label,
                                       'seq_id': seq_id,
                                       'start': start,
                                       'o_start': seq_offset[0] + start,
                                       'o_end': seq_offset[0] + end,
                                       'end': end})
                label_type_stats['domains'][label] += 1
                label_type_stats['nucleotides'][label] += end - start
                label_type_stats['seqs'][label].add(seq_id)
    domain_tab = pd.DataFrame(domain_records)
    assert domain_records, line
    domain_tab['length'] = domain_tab['end'] - domain_tab['start']  # coordinates are already 0-based

    label_type_stats['seqs'] = {l: len(seqs) for l, seqs in
                                label_type_stats['seqs'].items()}  # convert sets of seq_ids to counts

    if annotation_tsv is not None:
        # assign human-readable descriptions to reference labels
        descriptions = pd.read_csv(annotation_tsv, sep='\t', usecols=['acc', 'desc'])
        description_dict = {acc: desc for acc, desc in descriptions.itertuples(index=False, name=None)}
        domain_tab['desc'] = domain_tab['label'].map(description_dict)

    domain_tab['label_type'] = label_type

    return domain_tab, label_type_stats


def read_inputs(inputs_fasta: Path,
                ground_truth_gff: Path,
                pred_gff: Path,
                annotation_tsv: Path,
                cutoffs: Tuple[float] = (0.5, 0.75, 0.9)) -> Tuple[Dict[str, Dict], pd.DataFrame, np.ndarray, list]:
    """
    Read input files: fasta and gff annotations
    :param inputs_fasta: path to fasta file with sequence dataset
    :param ground_truth_gff: path to gff file with reference annotations (e.g., RFAM id of each domain)
    :param pred_gff: path to gff file with cluster numbers as labels
    :param annotation_tsv: path to csv file mapping reference IDs to human-readable names
    :param cutoffs: overlap cutoffs for labeing predicted and reference domains as 'match'
           (used to calculate domain pairings)
    :return: 1.
    """
    label_numbers = count(1)

    def new_unlabeled_id():
        return f'{u_prefix}{next(label_numbers)}'

    # create 0-filled sequence arrays to fill them with numeric labels later
    seq_offsets, general_stats = read_fasta(inputs_fasta)
    general_stats['input']['pairing_cutoffs'] = cutoffs
    domains = []
    for label_type, gff, annot in (('ref', ground_truth_gff, annotation_tsv), ('pred', pred_gff, None)):
        domain_tab, label_type_stats = read_gff(gff_path=gff,
                                                seq_offsets=seq_offsets,
                                                label_type=label_type,
                                                annotation_tsv=annot)
        domains.append(domain_tab)
        general_stats[label_type] = label_type_stats

    # concatenate reference and predicted domain dataframes into a single dict for easier access
    domains = pd.concat(domains, ignore_index=True)

    # create structured array to store labels for each nucleotide position in the concatenated sequence array
    label_matrix_dtype = [('seq_id', 'U25'), ('ref', 'U25'), ('pred', 'U25')]
    label_matrix = np.full((general_stats['input']['total_nt']), fill_value=np.nan, dtype=label_matrix_dtype)

    # track overlaps for warning report
    overlap_records = []

    coords = domains[['seq_id', 'label_type', 'label', 'o_start', 'o_end', 'length']]

    # iterate over seqeunces
    with tqdm(total=coords.shape[0], desc='filling label matrix') as pbar:
        for seq_id, label_type, label, o_start, o_end, _ in coords.itertuples(index=False):
            potential_overlap = label_matrix[label_type][o_start:o_end]
            if np.all(potential_overlap == 'nan'):
                overlap_records.append({'seq_id': seq_id,
                                        'label_type': label_type,
                                        'label': label,
                                        'structure': set(potential_overlap)})
            label_matrix[label_type][o_start:o_end] = label
            pbar.update()

    cutoff_array = np.empty((len(coords), len(cutoffs)), dtype='U25')
    with tqdm(total=coords.shape[0], desc='looking for paired labels') as pbar:
        for master_idx, seq_id, label_type, label, o_start, o_end, length in coords.itertuples():
            other_type = {'ref': 'pred', 'pred': 'ref'}[label_type]
            other_labels = label_matrix[other_type][o_start:o_end]
            most_common_partner, mc_count = Counter(other_labels).most_common(1)[0]
            mc_frequency = mc_count / length
            cutoff_array[master_idx] = [most_common_partner if mc_frequency >= c else np.nan for c in cutoffs]
            pbar.update()
    domains[[f'{c}' for c in cutoffs]] = cutoff_array

    with tqdm(total=len(seq_offsets), desc='filling unlabeled regions') as pbar:
        for seq_id, (offset_start, offset_end) in seq_offsets.items():
            label_matrix['seq_id'][offset_start:offset_end] = seq_id
            unlabeled_masks = {'ref': label_matrix['ref'][offset_start:offset_end] == 'nan',
                               'pred': label_matrix['pred'][offset_start:offset_end] == 'nan'}
            for label_type, unlabeled_mask in unlabeled_masks.items():
                if np.any(unlabeled_mask):
                    if np.all(unlabeled_mask):
                        general_stats[label_type]['fully_UNLABELED_seqs'] += 1
                    else:
                        general_stats[label_type]['partially_labeled_seqs'] += 1
                    unlabeled_positions = np.where(unlabeled_mask)[0]
                    unlabeled_regions = np.split(unlabeled_positions,
                                                 np.where(np.diff(unlabeled_positions) != 1)[0] + 1)
                    for u_positions in unlabeled_regions:
                        new_uid = new_unlabeled_id()
                        o_positions = u_positions + offset_start
                        assert np.all(label_matrix[label_type][o_positions] == 'nan')  # TODO remove after debugging
                        label_matrix[label_type][o_positions] = new_uid
                        general_stats[label_type]['UNLABELED_regions'] += 1
                        general_stats[label_type]['UNLABELED_nucleotides'] += len(o_positions)
                else:
                    general_stats[label_type]['fully_labeled_seqs'] += 1
            pbar.update()

    return general_stats, domains, label_matrix, overlap_records


def vformat(d: dict):
    """
    Format dictionary values into a string that can be fitted into Excel table
    :param d: dictionary to format
    :return: values separated by '; '
    """
    return '; '.join([f'{e}' for e in d.values()])


def cformat(c: Collection):
    """
    Format collection values into a string that can be fitted into Excel table
    :param c: collection to format
    :return: values separated by '; '
    """
    return '; '.join([f'{e}' for e in c])


def vsum(d: dict):
    """
    Sum values of a dictionary
    :param d: a dict with numeric values
    :return: sum of values
    """
    return sum(d.values())


def evaluate(fasta_path: Path,
             ground_truth_gff: Path,
             pred_gff: Path,
             annotation_tsv: Path,
             report_dir: Path):
    """
    Wrapper for evaluate_gffs_legacy to match new interface
    :param fasta_path: path to fasta file with sequence dataset
    :param ground_truth_gff: path to gff file with reference annotations (e.g., RFAM id of each domain)
    :param pred_gff: path to gff file with cluster numbers as labels
    :param annotation_tsv: path to csv file mapping reference IDs to human-readable names
    :param report_dir: path to directory where the evaluation report will be saved
    :return:
    """
    report_dir.mkdir(exist_ok=True, parents=True)

    # read all inputs and prepare data structures for evaluation
    general_stats, domains, label_matrix, overlap_records = read_inputs(inputs_fasta=fasta_path,
                                                                        ground_truth_gff=ground_truth_gff,
                                                                        pred_gff=pred_gff,
                                                                        annotation_tsv=annotation_tsv)

    report_path = report_dir.joinpath(f'domains.xlsx')
    logger.info(f'writing domain table to: {report_path}')
    domains.to_excel(report_path, index=False)

    # generate statistics on domain pairings between reference and predicted labels at different overlap cutoffs
    pair_report = {}
    for label_type, type_group in domains.groupby('label_type'):
        other_type = {'ref': 'pred', 'pred': 'ref'}[label_type]
        for cutoff in general_stats['input']['pairing_cutoffs']:
            good_pairs = 0  # number of pairs between the same 'other' label
            bad_pairs = 0  # number of pairs between different 'other' labels
            all_pairs = 0  # total number of pairs of domains in the analysed clusters ('ref' or 'pred')
            missing_pairs = 0  # pairs lost compared to the "ideal" (starting from domain sith no significant overlap)
            for label, group in type_group.groupby('label'):
                # extract values of "other" (ref for pred / pred for ref) paired with specific label
                paired_domains = group[group[f'{cutoff}'] != 'nan'][f'{cutoff}'].values
                other_count = Counter(paired_domains)  # only overlapping domains considered
                # generate all combinations of different "other" labels
                other_combinations = combinations(list(other_count.keys()), 2)

                # number of incorrect pairs is equal to the sum of product of each label count and the count
                bad_pairs_for_label = sum([other_count[t1] * other_count[t2] for t1, t2 in other_combinations])
                bad_pairs += bad_pairs_for_label

                good_pairs_for_label = sum([n_comb(n, 2) for n in other_count.values() if n > 1])
                good_pairs += good_pairs_for_label

                # all possible pairs of domains in a cluster include unmapped domains
                all_possible_pairs_in_label = n_comb(group.shape[0], 2)
                all_pairs += all_possible_pairs_in_label
                missing_pairs_in_label = all_possible_pairs_in_label - (good_pairs_for_label + bad_pairs_for_label)
                missing_pairs += missing_pairs_in_label

            good_percent = good_pairs / all_pairs * 100
            bad_percent = bad_pairs / all_pairs * 100
            ugly_percent = missing_pairs / all_pairs * 100
            pair_report[f'accurate_{other_type}_pairs_in_{label_type}_clusters_COV_{cutoff:.0%}'] = good_percent
            pair_report[f'faulty_{other_type}_pairs_in_{label_type}_clusters_COV_{cutoff:.0%}'] = bad_percent
            pair_report[f'missing_{other_type}_pairs_in_{label_type}_clusters_COV_{cutoff:.0%}'] = ugly_percent

    # concatenate reference and predicted labels into two long 1D arrays for ARS computation
    logger.info('calculating Adjusted Rand Score and Normalized Mutual Information Score')
    encoded_ref, encoded_pred = np.unique(np.array([label_matrix['ref'], label_matrix['pred']]), return_inverse=True)[1]
    ars = adjusted_rand_score(labels_true=encoded_ref, labels_pred=encoded_pred)
    nmi = normalized_mutual_info_score(labels_true=encoded_ref, labels_pred=encoded_pred)

    # get counts for all reference-prediction combinations
    logger.info('collecting reference-prediction pairs for error analysis')
    pair_nt_counter = Counter([r_p_pair for r_p_pair in zip(label_matrix['ref'], label_matrix['pred'])])
    pair_seq_unique = np.unique(label_matrix)
    pair_seq_counter = Counter([(r, p) for s, r, p in pair_seq_unique])

    # map each reference to all predicted clusters it was distributed into
    ref_to_pred = defaultdict(set)
    for r, p in pair_seq_counter:
        # pairs of labeled domains (unlabeled regions ignored)
        if not r.startswith(u_prefix) and not p.startswith(u_prefix):
            ref_to_pred[r].add(p)

    # map each predicted cluster to all references it contains
    pred_to_ref = defaultdict(set)
    for r, p in pair_seq_counter:
        # pairs of labeled domains (unlabeled regions ignored)
        if not r.startswith(u_prefix) and not p.startswith(u_prefix):
            pred_to_ref[p].add(r)

    logger.info('All label pairs collected')

    # retrieve unique 'label:description' mappings from domain dataframe
    ref_descriptions = \
    domains[domains['label_type'] == 'ref'].drop_duplicates(subset=['label'])[['label', 'desc']].set_index('label')[
        'desc'].to_dict()

    error_net = nx.Graph()

    # analyze glued references
    glued_ref_records = []
    with tqdm(total=len(pred_to_ref), desc='analyzing glued references') as progress_bar:
        for p, refs in pred_to_ref.items():
            if len(refs) > 1:
                r_descriptions = {r: ref_descriptions.get(r, f'*{r}*') for r in refs}
                nts = {r: pair_nt_counter[(r, p)] for r in refs}
                seqs = {r: pair_seq_counter[(r, p)] for r in refs}
                [error_net.add_edge(p, r, nt=nts[r], seqs=seqs[r]) for r in refs]
                glued_ref_records.append({'pred_id': p,
                                          'ref_ids': cformat(refs),
                                          'ref_descriptions': vformat(r_descriptions),
                                          'nt_counts_breakdown': vformat(nts),
                                          'nt_sum': vsum(nts),
                                          'seq_counts_breakdown': vformat(seqs),
                                          'seq_sum': vsum(seqs)})
            progress_bar.update()

    # analyze split references
    split_ref_records = []
    with tqdm(total=len(ref_to_pred), desc='analyzing split references') as progress_bar:
        for r, preds in ref_to_pred.items():
            r_desc = ref_descriptions.get(r, f'*{r}*')
            nts = {p: pair_nt_counter[(r, p)] for p in preds}
            seqs = {p: pair_seq_counter[(r, p)] for p in preds}
            [error_net.add_edge(r, p, nt=nts[p], seqs=seqs[p]) for p in preds]
            split_ref_records.append({'ref_id': r,
                                      'ref_description': r_desc,
                                      'pred_ids': cformat(preds),
                                      'nt_counts_breakdown': vformat(nts),
                                      'nt_sum': vsum(nts),
                                      'seq_counts_breakdown': vformat(seqs),
                                      'seq_sum': vsum(seqs)})
            progress_bar.update()

    excel_report_path = report_dir.joinpath('evaluation_report.xlsx')

    summary_metrics = {'total_seqs': general_stats['input']['total_seqs'],
                       'total_nt': general_stats['input']['total_nt'],
                       'adjusted_rand_score': ars,
                       'normalized_mutual_info_score': nmi,
                       'n_glued_refs': len(glued_ref_records),
                       'n_split_refs': len(split_ref_records)}
    summary_metrics.update(pair_report)

    r_vs_p = {}
    for label_type in ('ref', 'pred'):
        per_label_domains = np.array([n for n in general_stats[label_type]['domains'].values()])
        per_label_seqs = np.array([n for n in general_stats[label_type]['seqs'].values()])
        per_label_nucleotides = np.array([n for n in general_stats[label_type]['nucleotides'].values()])
        r_vs_p[label_type] = {'cluster_number': per_label_domains.shape[0],
                              'max_cluster_domains': per_label_domains.max(),
                              'cluster_domains_Q1': np.percentile(per_label_domains, 25),
                              'cluster_domains_median': np.percentile(per_label_domains, 50),
                              'cluster_domains_mean': per_label_domains.mean(),
                              'cluster_domains_Q3': np.percentile(per_label_domains, 75),
                              'max_cluster_seqs': per_label_seqs.max(),
                              'cluster_seqs_Q1': np.percentile(per_label_seqs, 25),
                              'cluster_seqs_median': np.percentile(per_label_seqs, 50),
                              'cluster_seqs_mean': per_label_seqs.mean(),
                              'cluster_seqs_Q3': np.percentile(per_label_seqs, 75),
                              'max_cluster_nucleotides': per_label_nucleotides.max(),
                              'cluster_nucleotides_Q1': np.percentile(per_label_nucleotides, 25),
                              'cluster_nucleotides_median': np.percentile(per_label_nucleotides, 50),
                              'cluster_nucleotides_mean': per_label_nucleotides.mean(),
                              'cluster_nucleotides_Q3': np.percentile(per_label_nucleotides, 75),
                              'fully_labeled_seqs': general_stats[label_type]['fully_labeled_seqs'],
                              'partially_labeled_seqs': general_stats[label_type]['partially_labeled_seqs'],
                              'fully_UNLABELED_seqs': general_stats[label_type]['fully_UNLABELED_seqs'],
                              'UNLABELED_regions': general_stats[label_type]['UNLABELED_regions'],
                              'UNLABELED_nucleotides': general_stats[label_type]['UNLABELED_nucleotides']}
        summary_metrics[f'{label_type}_cluster_number'] = per_label_domains.shape[0]
        summary_metrics[f'{label_type}_max_cluster_domains'] = per_label_domains.max()
        summary_metrics[f'{label_type}_cluster_domains_Q1'] = np.percentile(per_label_domains, 25)
        summary_metrics[f'{label_type}_cluster_domains_median'] = np.percentile(per_label_domains, 50)
        summary_metrics[f'{label_type}_cluster_domains_mean'] = per_label_domains.mean()
        summary_metrics[f'{label_type}_cluster_domains_Q3'] = np.percentile(per_label_domains, 75)
        summary_metrics[f'{label_type}_max_cluster_seqs'] = per_label_seqs.max()
        summary_metrics[f'{label_type}_cluster_seqs_Q1'] = np.percentile(per_label_seqs, 25)
        summary_metrics[f'{label_type}_cluster_seqs_median'] = np.percentile(per_label_seqs, 50)
        summary_metrics[f'{label_type}_cluster_seqs_mean'] = per_label_seqs.mean()
        summary_metrics[f'{label_type}_cluster_seqs_Q3'] = np.percentile(per_label_seqs, 75)
        summary_metrics[f'{label_type}_max_cluster_nucleotides'] = per_label_nucleotides.max()
        summary_metrics[f'{label_type}_cluster_nucleotides_Q1'] = np.percentile(per_label_nucleotides, 25)
        summary_metrics[f'{label_type}_cluster_nucleotides_median'] = np.percentile(per_label_nucleotides, 50)
        summary_metrics[f'{label_type}_cluster_nucleotides_mean'] = per_label_nucleotides.mean()
        summary_metrics[f'{label_type}_cluster_nucleotides_Q3'] = np.percentile(per_label_nucleotides, 75)
        summary_metrics[f'{label_type}_fully_labeled_seqs'] = general_stats[label_type]['fully_labeled_seqs']
        summary_metrics[f'{label_type}_partially_labeled_seqs'] = general_stats[label_type]['partially_labeled_seqs']
        summary_metrics[f'{label_type}_fully_UNLABELED_seqs'] = general_stats[label_type]['fully_UNLABELED_seqs']
        summary_metrics[f'{label_type}_UNLABELED_regions'] = general_stats[label_type]['UNLABELED_regions']
        summary_metrics[f'{label_type}_UNLABELED_nucleotides'] = general_stats[label_type]['UNLABELED_nucleotides']

    with pd.ExcelWriter(excel_report_path) as writer:
        pd.DataFrame([summary_metrics]).T.to_excel(writer, sheet_name='genral_metrics', index=True)
        pd.DataFrame(r_vs_p).to_excel(writer, sheet_name='reference_vs_predicted_clusters', index=True)
        pd.DataFrame(glued_ref_records).to_excel(writer, sheet_name='glued_refs', index=False)
        pd.DataFrame(split_ref_records).to_excel(writer, sheet_name='split_refs', index=False)

        nx.set_node_attributes(error_net, ref_descriptions, name='description')
        for label_type in ('ref', 'pred'):
            # add total nucleotide, domain, and sequence counts as node attributes
            nx.set_node_attributes(error_net, general_stats[label_type]['domains'], name='domain_count')
            nx.set_node_attributes(error_net, general_stats[label_type]['seqs'], name='seq_count')
            nx.set_node_attributes(error_net, general_stats[label_type]['nucleotides'], name='nt_count')

        error_network_path = report_dir.joinpath('error_network.gml')
        nx.write_gml(error_net, error_network_path)

    return summary_metrics
