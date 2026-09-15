"""Compare fixed gene-SHAP summaries across versioned biological collections.

No prepared expression/label files are opened. No model fitting or new gene
explanation calculation occurs. Existing source/results are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from .common import sha256, write_json
from .explanation import (allocation_matrix, aggregate_attributions, exploratory_percentile,
                          fold_top_sets, permute_membership, stability_rows, top_indices)
from .pathways import gene_id, read_reference

STUDY = Path(__file__).resolve().parents[2]
PROJECT = STUDY
SOURCE = STUDY / "results/gene_pathway_explanations"
EXTRA = STUDY / "data/reference/collections/msigdb_2026.1.Hs"
COLLECTIONS = {
    "hallmark": ("Hallmark", PROJECT / "data/reference/hallmark/msigdb_2026.1.Hs/source_manifest.json", "HALLMARK_"),
    "reactome": ("Reactome", STUDY / "data/reference/reactome/msigdb_2026.1.Hs/source_manifest.json", "REACTOME_"),
    "canonical": ("Canonical pathways", EXTRA / "canonical/source_manifest.json", ""),
    "go_bp": ("GO Biological Process", EXTRA / "go_bp/source_manifest.json", "GOBP_"),
}
SIZE_RULES = {"primary": (1, None), "size_15_500": (15, 500)}


def source_guard():
    """Refuse prepared-data access and changes to previously completed analyses."""
    prepared = (PROJECT / "phase2_outputs").resolve()
    protected = [SOURCE.resolve(), (STUDY / "results/model_comparison").resolve()]
    opened = set()

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(prepared):
            raise PermissionError("This comparison cannot access prepared expression or labels")
        if any(path.is_relative_to(root) for root in protected):
            flags = args[2] or 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError("Completed source results are read-only")
            if path.suffix == ".csv":
                raise PermissionError("Use saved attribution arrays and JSON, not record/label tables")
            opened.add(str(path.relative_to(PROJECT)))
    sys.addaudithook(guard)
    return opened


def load_saved_folds(source):
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol.get("status") != "complete" or protocol.get("test_policy") != "never read held-out test files":
        raise ValueError("A completed training-only gene explanation run is required")
    folds, seen, universe = [], set(), None
    for index in range(5):
        with np.load(source / "attributions" / f"fold_{index}.npz", allow_pickle=False) as archive:
            genes = archive["gene_ids"].astype(str).tolist()
            ids = archive["sample_ids"].astype(str).tolist()
            values = archive["gene_shap_log_odds"].copy()
            base = archive["base_log_odds"].copy()
            decision = archive["model_decision_log_odds"].copy()
            original_groups = archive["pathway_allocations_log_odds"].copy()
            original_residual = archive["unassigned_log_odds"].copy()
        if len(set(genes)) != len(genes) or any(gene_id(g) != g for g in genes):
            raise ValueError("Invalid gene universe")
        if universe is None:
            universe = genes
        if genes != universe or values.shape != (len(ids), len(genes)):
            raise ValueError("Fold gene order or attribution dimensions differ")
        if (base.shape != (len(ids),) or decision.shape != (len(ids),)
                or original_residual.shape != (len(ids),)
                or original_groups.shape != (len(ids), 50)):
            raise ValueError("Saved reconstruction arrays have unexpected dimensions")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate outer-validation record")
        seen.update(ids)
        if not all(np.isfinite(array).all() for array in (values, base, decision, original_groups, original_residual)):
            raise ValueError("Non-finite saved attributions")
        np.testing.assert_allclose(base + values.sum(axis=1), decision, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(base + original_groups.sum(axis=1) + original_residual,
                                   decision, atol=1e-8, rtol=1e-8)
        folds.append({"ids": ids, "values": values, "base": base, "decision": decision,
                      "original_groups": original_groups, "original_residual": original_residual})
    if len(seen) != 261 or len(universe) != 17263:
        raise ValueError("The active study requires 261 explained records and 17,263 genes")
    return universe, folds


def membership_matrix(pathways, genes, minimum=1, maximum=None):
    """Filter using measured gene counts, preserving reference order and labels."""
    if (not genes or len(set(genes)) != len(genes)
            or minimum < 1 or (maximum is not None and maximum < minimum)):
        raise ValueError("Gene universe must be unique and size bounds must be positive and ordered")
    lookup = {gene: index for index, gene in enumerate(genes)}
    names, rows, columns, details = [], [], [], []
    for name, entry in pathways.items():
        present = sorted(entry["genes"] & lookup.keys(), key=int)
        count = len(present)
        included = count >= minimum and (maximum is None or count <= maximum)
        details.append({"group": name, "source_url": entry["source_url"],
                        "reference_gene_count": len(entry["genes"]), "measured_gene_count": count,
                        "included": included, "measured_gene_ids": present})
        if included:
            column = len(names)
            names.append(name)
            rows.extend(lookup[gene] for gene in present)
            columns.extend([column] * count)
    if not names:
        raise ValueError("No gene groups survive the size rule")
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(genes), len(names)))
    return names, matrix, details


def observed_metrics(sets):
    stability = stability_rows(sets)
    return {
        "mean_covered_gene_overlap": float(np.mean([entry["covered_gene_overlap"] for entry in sets])),
        "mean_all_gene_overlap": float(np.mean([entry["all_gene_overlap"] for entry in sets])),
        "mean_pathway_jaccard": float(np.mean([entry["jaccard"] for entry in stability if entry["level"] == "pathways"])),
    }, stability


def random_reference(fold_values, membership, draws=100, seed=20260908, progress=None):
    """Same control and RNG ordering as the original Hallmark experiment."""
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        randomized = permute_membership(membership, rng)
        sets = [fold_top_sets(values, randomized) for values in fold_values]
        observed, _ = observed_metrics(sets)
        rows.append({"draw": draw, "seed": seed, **observed})
        if progress and (draw + 1) % 10 == 0:
            print(f"{progress}: {draw + 1}/{draws} random groups", flush=True)
    return rows


def compare_to_random(observed, controls):
    result = {}
    for metric, value in observed.items():
        random = np.asarray([row[metric] for row in controls])
        result[metric] = {"observed": value, "random_mean": float(random.mean()),
                          "random_q025": float(np.quantile(random, .025)),
                          "random_q975": float(np.quantile(random, .975)),
                          "midrank_percentile": exploratory_percentile(value, random)}
    return result


def group_rankings(values, membership, names, genes, group_values):
    weights = allocation_matrix(membership).tocsc()
    gene_importance = np.abs(values).mean(axis=0)
    unsigned = np.asarray(np.abs(values) @ weights)
    importance = np.abs(group_values).mean(axis=0)
    records = []
    for rank, column in enumerate(top_indices(importance, len(names)), 1):
        start, end = weights.indptr[column:column + 2]
        indices = weights.indices[start:end]
        mass = gene_importance[indices] * weights.data[start:end]
        order = top_indices(mass, min(5, len(mass)))
        total = float(mass.sum())
        ratio = np.divide(np.abs(group_values[:, column]), unsigned[:, column],
                          out=np.ones(len(values)), where=unsigned[:, column] > 0)
        records.append({"rank": rank, "group": names[column], "measured_gene_count": len(indices),
                        "mean_absolute_signed_allocation_log_odds": float(importance[column]),
                        "mean_signed_allocation_log_odds": float(group_values[:, column].mean()),
                        "mean_absolute_gene_mass_log_odds": total,
                        "mean_cancellation_fraction": float(np.mean(1 - ratio)),
                        "top_driver_gene_id": genes[indices[order[0]]],
                        "top_driver_mass_share": float(mass[order[0]] / total) if total else None,
                        "top5_driver_mass_share": float(mass[order].sum() / total) if total else None})
    return records


def summarize_variant(key, label, rule, pathways, genes, folds, output, draws, seed):
    names, membership, details = membership_matrix(pathways, genes, *SIZE_RULES[rule])
    values = np.concatenate([fold["values"] for fold in folds])
    group_values, residual, _ = aggregate_attributions(values, membership)
    base = np.concatenate([fold["base"] for fold in folds])
    decisions = np.concatenate([fold["decision"] for fold in folds])
    np.testing.assert_allclose(base + group_values.sum(axis=1) + residual, decisions, atol=1e-8, rtol=1e-8)
    sets = [fold_top_sets(fold["values"], membership) for fold in folds]
    observed, stability = observed_metrics(sets)
    controls = random_reference([fold["values"] for fold in folds], membership, draws, seed, f"{label}/{rule}")
    covered = np.asarray(membership.sum(axis=1)).ravel() > 0
    sizes = np.asarray(membership.sum(axis=0)).ravel().astype(int)
    degree = np.asarray(membership.sum(axis=1)).ravel().astype(int)
    duplicate_groups = defaultdict(list)
    for detail in details:
        if detail["included"]:
            duplicate_groups[tuple(detail["measured_gene_ids"])].append(detail["group"])
    duplicates = [names for names in duplicate_groups.values() if len(names) > 1]
    rankings = group_rankings(values, membership, names, genes, group_values)
    summary = {
        "collection": key, "label": label, "size_rule": rule,
        "source_group_count": len(pathways), "included_group_count": len(names),
        "zero_match_groups": sum(row["measured_gene_count"] == 0 for row in details),
        "excluded_by_size_count": sum(not row["included"] and row["measured_gene_count"] > 0 for row in details),
        "covered_gene_count": int(covered.sum()), "covered_gene_fraction": float(covered.mean()),
        "covered_absolute_gene_attribution_fraction": float(np.abs(values[:, covered]).sum() / np.abs(values).sum()),
        "groups_with_one_to_four_genes": int(np.sum((sizes >= 1) & (sizes <= 4))),
        "groups_above_500_genes": int(np.sum(sizes > 500)),
        "group_size_min": int(sizes.min()), "group_size_median": float(np.median(sizes)),
        "group_size_max": int(sizes.max()), "genes_in_multiple_groups": int(np.sum(degree > 1)),
        "max_gene_membership_count": int(degree.max()), "identical_mapped_group_labels": duplicates,
        "observed": observed, "random_control_comparison": compare_to_random(observed, controls),
        "max_logit_reconstruction_error": float(np.max(np.abs(base + group_values.sum(axis=1) + residual - decisions))),
        "top_groups": rankings[:10], "records_explained": len(values),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_json(output / "rankings.json", rankings)
    write_json(output / "membership.json", details)
    write_json(output / "random_controls.json", controls)
    write_json(output / "fold_stability.json", stability)
    write_json(output / "fold_top_lists.json", [
        {"fold": i, "top20_gene_ids": [genes[g] for g in sorted(entry["genes_all"])],
         "top20_covered_gene_ids": [genes[g] for g in sorted(entry["genes_covered"])],
         "top5_groups": [names[g] for g in sorted(entry["pathways"])],
         "covered_gene_overlap": entry["covered_gene_overlap"], "all_gene_overlap": entry["all_gene_overlap"]}
        for i, entry in enumerate(sets)
    ])
    np.savez_compressed(output / "allocations.npz", sample_ids=np.asarray([record for fold in folds for record in fold["ids"]]),
                        fold=np.concatenate([np.full(len(fold["ids"]), i) for i, fold in enumerate(folds)]),
                        group_names=np.asarray(names), group_allocations_log_odds=group_values,
                        unassigned_log_odds=residual, base_log_odds=base, decision_log_odds=decisions,
                        fold_mean_absolute_allocations=np.stack([
                            np.abs(aggregate_attributions(fold["values"], membership)[0]).mean(axis=0) for fold in folds]))
    if key == "hallmark" and rule == "primary":
        np.testing.assert_allclose(group_values, np.concatenate([fold["original_groups"] for fold in folds]), atol=1e-10, rtol=1e-10)
        original = json.loads((SOURCE / "summary.json").read_text())
        for metric in observed:
            np.testing.assert_allclose(observed[metric], original["observed"][metric], atol=1e-12, rtol=0)
            for statistic in summary["random_control_comparison"][metric]:
                np.testing.assert_allclose(summary["random_control_comparison"][metric][statistic],
                                           original["random_control_comparison"][metric][statistic], atol=1e-10, rtol=0)
    return summary, names


def make_figures(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [row for row in results if row["size_rule"] == "primary"]
    labels = [row["label"] for row in primary]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(primary))
    ax.bar(x - .18, [100 * r["covered_gene_fraction"] for r in primary], .36, label="Genes covered")
    ax.bar(x + .18, [100 * r["covered_absolute_gene_attribution_fraction"] for r in primary], .36,
           label="Absolute gene SHAP mass covered")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel="Percent", title="Reference coverage of the unchanged gene model")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, rule in zip(axes, SIZE_RULES):
        rows = [row for row in results if row["size_rule"] == rule]
        for y, row in enumerate(rows):
            stat = row["random_control_comparison"]["mean_pathway_jaccard"]
            ax.plot([stat["random_q025"], stat["random_q975"]], [y, y], color="#b3bbc4", linewidth=7)
            ax.scatter(stat["random_mean"], y, c="#526d82", marker="o", label="Random mean" if y == 0 else None)
            ax.scatter(stat["observed"], y, c="#ae542d", marker="D", label="Biological groups" if y == 0 else None, zorder=3)
        ax.set(yticks=range(len(rows)), yticklabels=[row["label"] for row in rows], xlim=(0, 1),
               xlabel="Top-five agreement across folds (Jaccard)",
               title="All nonempty groups" if rule == "primary" else "15–500 measured genes per group")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Each collection compared with its own random groups\nGrey bars: 2.5th–97.5th percentiles, not confidence intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "group_stability.png", dpi=160)
    plt.close(fig)


def write_report(output, results):
    lines = ["# Biological gene-group comparison", "", "Completed using the same saved gene SHAP values for 261 development records. "
             "No models were fitted, no new gene SHAP values were calculated, and no test data were accessed.", "",
             "## Coverage", "", "| Collection | Groups | Genes covered | Gene coverage | Absolute gene SHAP mass covered |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in results:
        if row["size_rule"] == "primary":
            lines.append(f"| {row['label']} | {row['included_group_count']:,} | {row['covered_gene_count']:,} | "
                         f"{row['covered_gene_fraction']:.1%} | {row['covered_absolute_gene_attribution_fraction']:.1%} |")
    lines.extend(["", "These are annotation/attribution coverage percentages, not accuracy or biological validation. "
                  "GO Biological Process is a functional ontology collection, not exclusively reaction pathways. "
                  "Canonical pathways contain Reactome; these collections are not independent replications.", ""])
    for rule in SIZE_RULES:
        lines.extend(["## " + ("Primary comparison: all nonempty groups" if rule == "primary" else "Predefined size sensitivity: 15–500 measured genes"), "",
                      "| Collection | Groups | Real top-five agreement | Random mean | Descriptive percentile | Covered top-gene overlap: real / random |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in results:
            if row["size_rule"] != rule:
                continue
            a = row["random_control_comparison"]["mean_pathway_jaccard"]
            b = row["random_control_comparison"]["mean_covered_gene_overlap"]
            lines.append(f"| {row['label']} | {row['included_group_count']:,} | {a['observed']:.3f} | {a['random_mean']:.3f} | "
                         f"{a['midrank_percentile']:g} | {b['observed']:.1%} / {b['random_mean']:.1%} |")
        lines.append("")
    lines.extend(["## Interpretation boundaries", "",
        "Compare each real grouping with its own matched random reference. Do not rank collections by raw "
        "Jaccard alone: they differ in group count, size, overlap and gene coverage. The covered top-20 gene "
        "list also differs between collections. The 100 relabellings yield descriptive percentiles, not "
        "formal p-values or multiple-testing-adjusted findings. A missing advantage is not proof of equivalence.", "",
        "The size range is a sensitivity rule borrowed from familiar GSEA practice, not a medical or XAI "
        "validity threshold. Both variants are retained. Shared and identical labels can inflate apparent "
        "repeatability; random controls preserve those structural features. They do not preserve expression "
        "correlation or each gene's original number of memberships.", "",
        "Groups are signed sums of allocated gene SHAP values. Each gene's contribution is shared across "
        "its containing groups, so changing the collection changes the allocation denominator. Opposing "
        "contributions can cancel; driver concentration and cancellation are saved in rankings.json. "
        "These are not independently computed pathway Shapley values or proof of biological activation.", "",
        "The same five fixed models and same records are used across collections. Across folds, however, "
        "models have different selected settings and validation records. Thus stability includes model "
        "refitting/selection and sample variation. Pooled magnitudes mix model log-odds scales and do not "
        "define a validated biomarker panel. Earlier adaptive exploration and blood confounding remain.", "",
        "## Files and reproduction", "",
        "Each collection/rule folder contains summary.json, rankings.json, membership.json, "
        "fold_top_lists.json, fold_stability.json, random_controls.json and allocations.npz. "
        "Identical Hallmark primary/sensitivity memberships reuse the same results, explicitly recorded.", "",
        "Run `python scripts/pipeline/07_compare_explanations.py --output results/group_comparison_recheck` "
        "from the repository root. Existing outputs are not overwritten. Source code, plan and input "
        "checksums are preserved. Review canonical-reference redistribution terms before releasing memberships.", "",
        "See [the comparison plan](../../src/asd_blood/comparison_plan.md), [references](../../data/reference/REFERENCES.md), "
        "and [additional reference provenance](../../data/reference/collections/README.md).", "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines))


def run(output=STUDY / "results/group_comparison", draws=100, seed=20260908):
    if draws != 100 or seed != 20260908:
        raise ValueError("Use the fixed recorded 100-draw, seed-20260908 protocol")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    opened = source_guard()
    files = [SOURCE / "protocol.json", SOURCE / "summary.json", STUDY / "src/asd_blood/comparison_plan.md",
             *[SOURCE / "attributions" / f"fold_{i}.npz" for i in range(5)],
             Path(__file__), Path(__file__).with_name("explanation.py"), Path(__file__).with_name("common.py"),
             Path(__file__).with_name("pathways.py")]
    for _, manifest, _ in COLLECTIONS.values():
        files.append(manifest)
        files.append(manifest.parent / json.loads(manifest.read_text())["file"])
    hashes = {str(path.resolve()): sha256(path) for path in files}
    original = json.loads((SOURCE / "protocol.json").read_text())
    if sha256(Path(__file__).with_name("explanation.py")) != original["source_code_sha256"]:
        raise ValueError("Original explanation implementation changed")
    for name, expected in original["model_sha256"].items():
        path = STUDY / "results/model_comparison/outer_models" / name
        if sha256(path) != expected:
            raise ValueError("An original explained model changed")
        hashes[str(path.resolve())] = expected
    genes, folds = load_saved_folds(SOURCE)
    references = {}
    for key, (label, manifest, prefix) in COLLECTIONS.items():
        pathways, _, _ = read_reference(manifest, prefix)
        references[key] = pathways
        covered = set().union(*(entry["genes"] for entry in pathways.values())) & set(genes)
        print(f"Coverage: {label}: {len(covered):,}/{len(genes):,} genes ({len(covered)/len(genes):.1%})", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": hashes, "collections": list(COLLECTIONS), "size_rules": SIZE_RULES,
                "gene_top_k": 20, "group_top_k": 5, "random_draws": draws, "random_seed": seed,
                "prepared_data_access": False, "models_fitted": False, "gene_shap_recomputed": False,
                "test_evaluated": False, "python_version": sys.version, "numpy_version": np.__version__,
                "source_population": "five saved outer-validation SHAP arrays, 261 development records"}
    write_json(output / "protocol.json", protocol)
    snapshots = output / "source"
    snapshots.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("explanation.py"), Path(__file__).with_name("common.py"),
                 Path(__file__).with_name("pathways.py"), STUDY / "src/asd_blood/comparison_plan.md"):
        shutil.copyfile(path, snapshots / path.name)
    results = []
    with threadpool_limits(limits=1):
        for key, (label, _, _) in COLLECTIONS.items():
            primary, primary_names = summarize_variant(key, label, "primary", references[key], genes, folds,
                                                        output / key / "primary", draws, seed)
            results.append(primary)
            sensitivity_names, _, _ = membership_matrix(references[key], genes, *SIZE_RULES["size_15_500"])
            if sensitivity_names == primary_names:
                shutil.copytree(output / key / "primary", output / key / "size_15_500")
                sensitivity = {**primary, "size_rule": "size_15_500", "identical_to_primary": True}
                write_json(output / key / "size_15_500/summary.json", sensitivity)
            else:
                sensitivity, _ = summarize_variant(key, label, "size_15_500", references[key], genes, folds,
                                                    output / key / "size_15_500", draws, seed)
            results.append(sensitivity)
            write_json(output / "comparison.json", results)
            print(f"Completed {label}", flush=True)
    make_figures(output, results)
    write_report(output, results)
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("A source file changed during the comparison")
    protocol.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_files_opened=sorted(opened), input_hashes_unchanged=True,
                    original_hallmark_reproduced=True)
    write_json(output / "protocol.json", protocol)
    print(f"Completed collection comparison: {output}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "results/group_comparison")
    args = parser.parse_args()
    run(args.output)
