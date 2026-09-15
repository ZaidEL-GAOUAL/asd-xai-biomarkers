"""Compare reference gene coverage without labels, expression rows or models.

Uses the prepared Entrez dictionary and training CSV header only. The original
Hallmark and broader Reactome collections remain separate. No pathways or genes
are filtered from the study based on these descriptive counts.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
PROJECT = STUDY
HALLMARK = PROJECT / "data/reference/hallmark/msigdb_2026.1.Hs/source_manifest.json"
REACTOME = STUDY / "data/reference/reactome/msigdb_2026.1.Hs/source_manifest.json"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def gene_id(value):
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Not a canonical positive Entrez ID: {value!r}")
    return value


def identifier_guard(input_dir):
    """Block diagnosis metadata, test files, and writes to prepared inputs."""
    root = Path(input_dir).resolve()
    opened = set()

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root):
            flags = args[2] or 0
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing or path.parent != root or path.name not in {
                "gene_dictionary.csv", "X_train_log2.csv.gz"
            }:
                raise PermissionError(f"Outside identifier-only scope: {path.name}")
            opened.add(path.name)

    sys.addaudithook(guard)
    return opened


def read_project_genes(input_dir):
    with (input_dir / "gene_dictionary.csv").open(newline="", encoding="utf-8") as handle:
        genes = [gene_id(row["gene_id"]) for row in csv.DictReader(handle)]
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("Empty or duplicate gene dictionary identifiers")
    with gzip.open(input_dir / "X_train_log2.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))  # No expression data rows are parsed.
    if header[0] != "sample_id":
        raise ValueError("Unexpected training matrix index")
    columns = [gene_id(value) for value in header[1:]]
    if len(columns) != len(set(columns)) or set(columns) != set(genes):
        raise ValueError("Training feature header differs from gene dictionary")
    return set(genes), hashlib.sha256("\t".join(columns).encode()).hexdigest()


def parse_gmt(text, prefix):
    pathways = {}
    duplicates = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3 or not fields[0].startswith(prefix):
            raise ValueError(f"Invalid gene-set row {number}")
        name, source, *members = fields
        if name in pathways:
            raise ValueError(f"Duplicate gene-set name: {name}")
        unique = {gene_id(value) for value in members}
        duplicates += len(members) - len(unique)
        pathways[name] = {"genes": unique, "source_url": source}
    if not pathways:
        raise ValueError("Empty gene-set collection")
    return pathways, duplicates


def read_reference(manifest_path, prefix):
    manifest = json.loads(manifest_path.read_text())
    filename = manifest["file"]
    if Path(filename).name != filename or manifest["identifier_type"] != "NCBI Entrez Gene ID":
        raise ValueError("Invalid reference path or identifier namespace")
    path = manifest_path.parent / filename
    if sha256(path) != manifest["sha256"] or path.stat().st_size != manifest["bytes"]:
        raise ValueError("Reference bytes differ from the frozen manifest")
    pathways, duplicates = parse_gmt(path.read_text(), prefix)
    if len(pathways) != manifest["expected_sets"]:
        raise ValueError("Reference gene-set count differs from its manifest")
    return pathways, duplicates, path


def coverage(pathways, project_genes, duplicates=0):
    if not pathways or not project_genes:
        raise ValueError("Coverage requires nonempty reference and project genes")
    membership = Counter()
    signatures = defaultdict(list)
    details = []
    reference_genes = set()
    for name, row in sorted(pathways.items()):
        genes = row["genes"]
        present = genes & project_genes
        reference_genes.update(genes)
        membership.update(present)
        if present:
            signatures[tuple(sorted(present, key=int))].append(name)
        details.append({
            "gene_set": name, "source_url": row["source_url"],
            "reference_gene_count": len(genes), "matched_gene_count": len(present),
            "reference_members_observed_fraction": len(present) / len(genes),
            "matched_gene_ids": sorted(present, key=int),
            "unmeasured_reference_gene_ids": sorted(genes - project_genes, key=int),
        })
    covered = set(membership)
    sizes = [row["matched_gene_count"] for row in details]
    identical = [names for names in signatures.values() if len(names) > 1]
    summary = {
        "gene_set_count": len(pathways), "project_gene_count": len(project_genes),
        "reference_unique_gene_count": len(reference_genes),
        "covered_project_gene_count": len(covered),
        "project_gene_coverage_fraction": len(covered) / len(project_genes),
        "uncovered_project_gene_count": len(project_genes - covered),
        "unmeasured_reference_gene_count": len(reference_genes - project_genes),
        "matched_genes_per_set_min": min(sizes),
        "matched_genes_per_set_median": statistics.median(sizes),
        "matched_genes_per_set_max": max(sizes),
        "sets_with_zero_matches": [row["gene_set"] for row in details if row["matched_gene_count"] == 0],
        "sets_with_one_to_four_matches": sum(1 <= size < 5 for size in sizes),
        "covered_genes_in_multiple_sets": sum(count > 1 for count in membership.values()),
        "largest_memberships_for_one_gene": max(membership.values(), default=0),
        "identical_nonempty_matched_set_groups": identical,
        "duplicate_within_set_memberships_removed": duplicates,
    }
    return summary, details, covered


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(input_dir=PROJECT / "phase2_outputs", output=STUDY / "results/pathway_coverage"):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Results already exist: {output}")
    opened = identifier_guard(input_dir)
    genes, header_hash = read_project_genes(input_dir)
    if len(genes) != 17263:
        raise ValueError("The active study must contain 17,263 prepared gene IDs")
    files = [input_dir / "gene_dictionary.csv", HALLMARK, REACTOME, Path(__file__).resolve()]
    summaries, details, covered = {}, {}, {}
    for label, manifest, prefix in (("Hallmark", HALLMARK, "HALLMARK_"), ("Reactome", REACTOME, "REACTOME_")):
        pathways, duplicates, path = read_reference(manifest, prefix)
        files.append(path)
        summaries[label], details[label], covered[label] = coverage(pathways, genes, duplicates)
    hashes = {str(path): sha256(path) for path in files}
    comparison = {
        "genes_in_both": len(covered["Hallmark"] & covered["Reactome"]),
        "reactome_only_gene_count": len(covered["Reactome"] - covered["Hallmark"]),
        "hallmark_only_gene_count": len(covered["Hallmark"] - covered["Reactome"]),
        "union_covered_gene_count": len(covered["Hallmark"] | covered["Reactome"]),
        "union_gene_coverage_fraction": len(covered["Hallmark"] | covered["Reactome"]) / len(genes),
        "union_is_descriptive_not_selected_analysis": True,
    }
    protocol = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "identifier-only comparison of preselected Reactome and existing Hallmark",
        "python_version": sys.version, "source_and_dictionary_sha256": hashes,
        "training_header_sha256": header_hash,
        "input_files_opened": sorted(opened), "expression_rows_parsed": False,
        "diagnosis_labels_read": False, "test_files_opened": False,
        "models_fitted": False, "explanations_recalculated": False,
        "mapping": "Exact Entrez ID intersection; no symbol or obsolete-ID rescue; absent members not filled with zeros",
        "pathway_filters_applied": None,
        "small_set_counts": "Descriptive only; fewer than five matches is not a medical cutoff",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "summary.json", {"collections": summaries, "comparison": comparison})
    write_json(output / "pathway_membership.json", details)
    write_json(output / "uncovered_gene_ids.json", {
        label: sorted(genes - present, key=int) for label, present in covered.items()
    })
    lines = ["# Pathway coverage", "", "Identifier check only. No new model or XAI results.", "",
             "| Collection | Gene sets | Our genes covered | Coverage of our 17,263 genes |",
             "| --- | ---: | ---: | ---: |"]
    for label, summary in summaries.items():
        lines.append(f"| {label} | {summary['gene_set_count']:,} | {summary['covered_project_gene_count']:,} | {summary['project_gene_coverage_fraction']:.1%} |")
    r = summaries["Reactome"]
    lines.extend(["", "## Decision", "",
        "Reactome is selected as the broader reference for the next explanation comparison. "
        "Hallmark remains the original comparator. The classifier, gene inputs, saved SHAP values "
        "and earlier results are unchanged. This is not an accuracy or explanation-quality improvement.", "",
        f"Reactome covers {comparison['reactome_only_gene_count']:,} genes absent from Hallmark, "
        f"but misses {comparison['hallmark_only_gene_count']:,} genes covered by Hallmark. "
        "It is broader, not a strict superset. The collections have not been merged.", "",
        "## Mapping checks", "",
        f"- Reactome sets have {r['matched_genes_per_set_min']} to {r['matched_genes_per_set_max']:,} "
        f"measured genes (median {r['matched_genes_per_set_median']:g}).",
        f"- {len(r['sets_with_zero_matches'])} sets have no matching genes; "
        f"{r['sets_with_one_to_four_matches']} have only one to four. No size cutoff was applied.",
        f"- {r['covered_genes_in_multiple_sets']:,} covered genes occur in multiple Reactome sets. "
        "Shared or nested pathways are not independent biological evidence.",
        f"- {len(r['identical_nonempty_matched_set_groups'])} groups of pathway labels become "
        "identical after matching to our measured genes. Their names are retained in summary.json.", "",
        "Coverage means membership by gene ID, not expression strength, ASD relevance or biological activation. "
        "Missing reference genes are absent measurements, not zero expression. Old array identifiers and "
        "current reference identifiers may differ; no identifier rescue was attempted.", "",
        "## Reproduction and sources", "",
        "Run `python scripts/pipeline/06_gene_set_coverage.py --output results/pathway_coverage_recheck` "
        "from the repository root. Existing output directories are never overwritten.", "",
        "The audit reads the gene dictionary and parses only the training matrix header. "
        "It does not read diagnosis labels, parse expression rows, access test files or fit models.", "",
        "See the [Reactome reference record](../../data/reference/reactome/README.md) and "
        "[source ledger](../../data/reference/REFERENCES.md). Full memberships and unmapped IDs are retained in JSON.", "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines))
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Reference or dictionary changed during audit")
    protocol.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_and_dictionary_unchanged=True)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"collections": summaries, "comparison": comparison}, indent=2))
    return summaries, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "phase2_outputs")
    parser.add_argument("--output", type=Path, default=STUDY / "results/pathway_coverage")
    args = parser.parse_args()
    run(args.input_dir, args.output)
