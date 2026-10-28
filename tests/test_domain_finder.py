# fmt: off
from pathlib import Path
import pytest
from aligners import Span
from domain_finder import (DomainFinder, DomainInstance, _normalize_attributes,
                           find_domains)


def test_normalize_attributes():
    raw_attrs = {
        "--aligner": "riboseek",
        "gap-open": 11.0,
        "-min-seq-id": 0.5,
        "min.dom.sim": 0.25,
    }
    normalized = _normalize_attributes(raw_attrs)
    assert normalized["aligner"] == "riboseek"
    assert normalized["gap_open"] == 11.0
    assert normalized["min_seq_id"] == 0.5
    assert normalized["min_dom_sim"] == 0.25


def test_domain_instance():
    target1 = Span(seq_id="seqB", start=10, end=50, strand=1)
    target2 = Span(seq_id="seqC", start=20, end=70, strand=-1)
    domain = DomainInstance(seq_id="seqA",
                            start=5,
                            summary_strand=1,
                            end=100,
                            targets=[target1, target2],
                            source="riboseek")

    assert domain.seq_id == "seqA"
    assert domain.start == 5
    assert domain.end == 100
    assert domain.length == 96
    assert domain.strand == 1
    assert domain.coverage() == 2
    assert domain.total_target_length() == (41 + 51)
    gff_line = domain.to_gff(cluster_id="0")
    assert "seqA\triboseek\tDomain\t5\t100\t.\t+\t.\tID=0" in gff_line


def test_domain_finder_workflow(tmp_path):
    # Setup mock search hits
    span1 = Span(seq_id="seq1", start=10, end=100, strand=1)
    target1 = Span(seq_id="seq2", start=15, end=95, strand=1)
    span1.partners = [target1]

    span2 = Span(seq_id="seq1", start=12, end=98, strand=1)
    target2 = Span(seq_id="seq3", start=20, end=90, strand=1)
    span2.partners = [target2]

    search_hits = {"seq1": [span1, span2]}

    finder = DomainFinder(search=search_hits, source="mmseqs2")
    finder.find_domain_instances(min_dom_sim=0.1,
                                 halo_stringency=0.5,
                                 min_dom_dens=0.0,
                                 halo_method="knn")
    assert finder.n_domains > 0

    discarded, kept = finder.filter_spurious_domains(min_containment=0.5,
                                                      reassign_targets=True)
    assert kept > 0

    finder.merge_targets()
    finder.calculate_target_overlaps()

    families = list(finder.families(min_fam_sim=0.1,
                                    min_fam_dens=0.0,
                                    halo_stringency=0.5,
                                    halo_method="knn"))
    assert len(families) > 0


def test_find_domains_with_tsv(tmp_path):
    tsv_content = (
        "seq1\tseq2\t100.0\t100\t0\t0\t10\t100\t10\t100\t1e-10\t200\n"
        "seq1\tseq3\t100.0\t100\t0\t0\t20\t90\t12\t98\t1e-10\t200\n"
    )
    tsv_file = tmp_path / "mock_search.tsv"
    tsv_file.write_text(tsv_content)

    fasta_file = tmp_path / "sequences.fna"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n>seq3\nACGT\n")

    output_gff = tmp_path / "output.gff"

    attributes = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_file),
        "min_dom_sim": 0.1,
        "dom_halo": 0.5,
        "min_dom_dens": 0.0,
        "min_fam_sim": 0.1,
        "fam_halo": 0.5,
        "min_fam_dens": 0.0,
        "halo_method": "knn",
        "reassign_targets": True,
        "min_containment": 0.5,
    }

    find_domains(fasta_path=fasta_file,
                 attributes=attributes,
                 output_gff=output_gff,
                 tmp_dir=tmp_path / "tmp",
                 result_dir=tmp_path / "results_tsvs")

    assert output_gff.exists()
    assert output_gff.stat().st_size > 0
    content = output_gff.read_text()
    assert "seq1\tmmseqs2\tDomain\t" in content


def test_parse_mmseqs_tsv_stream_reverse_strand(tmp_path):
    from aligners import SequenceRegistry, parse_mmseqs_tsv_stream

    tsv_content = (
        "qseq1\ttseq1\t0.864\t103\t14\t0\t103\t1\t5\t107\t3.45e-27\t123\n"
    )
    tsv_file = tmp_path / "rev_strand.tsv"
    tsv_file.write_text(tsv_content)

    registry = SequenceRegistry()
    blocks = list(parse_mmseqs_tsv_stream(tsv_file, registry=registry))
    assert len(blocks) == 1
    q_id, spans = blocks[0]
    assert registry.get_str(q_id) == "qseq1"
    assert len(spans) == 1
    span = spans[0]
    assert span.start == 1
    assert span.end == 103
    assert span.length == 103
    assert len(span.partners) == 1
    partner = span.partners[0]
    assert registry.get_str(partner.seq_id) == "tseq1"
    assert partner.start == 5
    assert partner.end == 107
    assert partner.strand == -1


def test_non_contiguous_tsv_raises_error(tmp_path):
    tsv_content = (
        "seq1\ttargetA\t100.0\t100\t0\t0\t10\t100\t10\t100\t1e-10\t200\n"
        "seq2\ttargetA\t100.0\t100\t0\t0\t10\t100\t10\t100\t1e-10\t200\n"
        "seq1\ttargetB\t100.0\t100\t0\t0\t10\t100\t10\t100\t1e-10\t200\n"
    )
    tsv_file = tmp_path / "non_contiguous.tsv"
    tsv_file.write_text(tsv_content)

    fasta_file = tmp_path / "sequences.fna"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n")

    output_gff = tmp_path / "output.gff"

    attributes = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_file),
    }

    with pytest.raises(ValueError) as exc_info:
        find_domains(fasta_path=fasta_file,
                     attributes=attributes,
                     output_gff=output_gff,
                     tmp_dir=tmp_path / "tmp",
                     result_dir=tmp_path / "results_tsvs")

    assert "🛑" in str(exc_info.value)
    assert "not grouped contiguously" in str(exc_info.value)


def test_inverted_target_coordinates(tmp_path):
    tsv_content = (
        "seq1\tseq2\t100.0\t91\t0\t0\t10\t100\t100\t10\t1e-10\t200\n"
    )
    tsv_file = tmp_path / "inverted_target.tsv"
    tsv_file.write_text(tsv_content)

    fasta_file = tmp_path / "sequences.fna"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n")

    output_gff = tmp_path / "output.gff"

    attributes = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_file),
        "min_dom_sim": 0.1,
        "dom_halo": 0.5,
        "min_dom_dens": 0.0,
        "min_fam_sim": 0.1,
        "fam_halo": 0.5,
        "min_fam_dens": 0.0,
        "halo_method": "knn",
        "reassign_targets": True,
        "min_containment": 0.5,
    }

    find_domains(fasta_path=fasta_file,
                 attributes=attributes,
                 output_gff=output_gff,
                 tmp_dir=tmp_path / "tmp",
                 result_dir=tmp_path / "results_tsvs")

    assert output_gff.exists()
    assert output_gff.stat().st_size > 0


def test_find_domains_with_sample_subset_tsv(tmp_path):
    sample_tsv = Path(__file__).parent / "fixtures" / "sample_subset.tsv"
    assert sample_tsv.exists()

    fasta_file = tmp_path / "sequences.fna"
    fasta_file.write_text(">URS00036AA4AB\nACGT\n>URS00019BAC15\nACGT\n")

    output_gff = tmp_path / "sample_output.gff"

    attributes = {
        "aligner": "mmseqs2",
        "tsv": str(sample_tsv),
        "min_dom_sim": 0.1,
        "dom_halo": 0.5,
        "min_dom_dens": 0.0,
        "min_fam_sim": 0.1,
        "fam_halo": 0.5,
        "min_fam_dens": 0.0,
        "halo_method": "knn",
        "reassign_targets": True,
        "min_containment": 0.5,
    }

    find_domains(fasta_path=fasta_file,
                 attributes=attributes,
                 output_gff=output_gff,
                 tmp_dir=tmp_path / "tmp",
                 result_dir=tmp_path / "results_tsvs")

    assert output_gff.exists()
    assert output_gff.stat().st_size > 0
# fmt: on
