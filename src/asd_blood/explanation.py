"""Training-fold SHAP explanations and descriptive Hallmark aggregation.

Pathway values are allocated sums of gene SHAP values, not independently
estimated group Shapley values. Positive values increase the model's ASD log
odds relative to its outer-training reference population; they do not establish
pathway activation or disease causation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .common import load_training, sha256, training_data_guard, write_json


def read_pathways(gmt: Path, gene_ids: list[str]):
    """Return binary membership for all pathways, including unmapped genes."""
    lookup = {str(gene): i for i, gene in enumerate(gene_ids)}
    names, rows, columns, source_sizes = [], [], [], []
    with gmt.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or not fields[0]:
                raise ValueError(f"Invalid GMT record at line {line_number}")
            name, genes = fields[0], set(fields[2:])
            if name in names:
                raise ValueError(f"Duplicate pathway: {name}")
            column = len(names)
            names.append(name)
            source_sizes.append(len(genes))
            for gene in genes:
                if gene in lookup:
                    rows.append(lookup[gene])
                    columns.append(column)
    membership = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(gene_ids), len(names)), dtype=np.float64,
    )
    if not names or np.any(np.asarray(membership.sum(axis=0)).ravel() == 0):
        raise ValueError("GMT must contain nonempty mapped pathways")
    return names, membership, np.asarray(source_sizes)


def allocation_matrix(membership):
    """Split a gene equally across its memberships; uncovered genes stay zero."""
    degree = np.asarray(membership.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sparse.diags(inverse) @ membership


def aggregate_attributions(gene_values, membership):
    """Preserve total attribution with a separate unassigned-gene residual."""
    weights = allocation_matrix(membership)
    pathway_values = np.asarray(gene_values @ weights)
    covered = np.asarray(membership.sum(axis=1)).ravel() > 0
    residual = gene_values[:, ~covered].sum(axis=1)
    np.testing.assert_allclose(
        pathway_values.sum(axis=1) + residual,
        gene_values.sum(axis=1), atol=1e-9, rtol=1e-9,
    )
    return pathway_values, residual, weights


def top_indices(values, count):
    """Stable tie-breaking follows the frozen feature/pathway order."""
    values = np.asarray(values)
    return np.argsort(-values, kind="stable")[:min(count, len(values))]


def jaccard(first, second):
    first, second = set(first), set(second)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def fold_top_sets(gene_values, membership, gene_k=20, pathway_k=5):
    covered = np.flatnonzero(np.asarray(membership.sum(axis=1)).ravel() > 0)
    gene_importance = np.mean(np.abs(gene_values), axis=0)
    pathways, _, _ = aggregate_attributions(gene_values, membership)
    top_all = top_indices(gene_importance, gene_k)
    top_covered = covered[top_indices(gene_importance[covered], gene_k)]
    top_pathways = top_indices(np.mean(np.abs(pathways), axis=0), pathway_k)
    contained = set(np.flatnonzero(
        np.asarray(membership[:, top_pathways].sum(axis=1)).ravel() > 0
    ))
    return {
        "genes_all": set(top_all),
        "genes_covered": set(top_covered),
        "pathways": set(top_pathways),
        "covered_gene_overlap": len(set(top_covered) & contained) / len(top_covered),
        "all_gene_overlap": len(set(top_all) & contained) / len(top_all),
    }


def stability_rows(top_sets):
    rows = []
    for first, second in combinations(range(len(top_sets)), 2):
        for level in ("genes_all", "genes_covered", "pathways"):
            rows.append({
                "fold_a": first, "fold_b": second, "level": level,
                "jaccard": jaccard(top_sets[first][level], top_sets[second][level]),
            })
    return rows


def permute_membership(membership, rng):
    """One global relabelling preserves pathway sizes and pairwise overlaps.

    Only covered genes are shuffled. Expression correlation and the association
    between a particular gene and its membership degree are not preserved.
    """
    covered = np.flatnonzero(np.asarray(membership.sum(axis=1)).ravel() > 0)
    order = np.arange(membership.shape[0])
    order[covered] = rng.permutation(covered)
    return membership[order].tocsr()


def random_controls(fold_values, membership, draws=100, seed=20260908):
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        # The same permutation is shared across every fold in this draw.
        random_membership = permute_membership(membership, rng)
        sets = [fold_top_sets(values, random_membership) for values in fold_values]
        pairwise = stability_rows(sets)
        rows.append({
            "draw": draw, "seed": seed,
            "mean_covered_gene_overlap": float(np.mean([
                row["covered_gene_overlap"] for row in sets
            ])),
            "mean_all_gene_overlap": float(np.mean([
                row["all_gene_overlap"] for row in sets
            ])),
            "mean_pathway_jaccard": float(np.mean([
                row["jaccard"] for row in pairwise if row["level"] == "pathways"
            ])),
        })
    return pd.DataFrame(rows)


def exploratory_percentile(observed, random_values):
    """Midrank percentile; not a clinical validation or a formal p-value."""
    random_values = np.asarray(random_values)
    return float(100 * (
        np.mean(random_values < observed) + 0.5 * np.mean(random_values == observed)
    ))


def annotate_sfari(genes, reference):
    """Exact-symbol context only; SFARI scores are not blood-biomarker labels."""
    required = {"gene_symbol", "score", "syndromic"}
    if not required <= set(reference.columns):
        raise ValueError("SFARI reference requires gene_symbol, score and syndromic")
    if reference.gene_symbol.isna().any() or reference.gene_symbol.duplicated().any():
        raise ValueError("SFARI symbols must be nonempty and unique for an exact join")
    selected = reference.loc[:, ["gene_symbol", "score", "syndromic"]].copy()
    selected = selected.rename(columns={"score": "sfari_score", "syndromic": "sfari_syndromic"})
    selected["sfari_exact_symbol_match"] = True
    result = genes.merge(selected, how="left", on="gene_symbol", validate="many_to_one", sort=False)
    result["sfari_exact_symbol_match"] = result.sfari_exact_symbol_match.eq(True)
    result["sfari_score_1_or_2"] = pd.to_numeric(result.sfari_score, errors="coerce").isin([1, 2])
    return result


def attribution_rankings(values, membership, gene_ids, symbols, pathway_names):
    pathways, residual, weights = aggregate_attributions(values, membership)
    gene_importance = np.mean(np.abs(values), axis=0)
    genes = pd.DataFrame({
        "gene_id": gene_ids, "gene_symbol": symbols,
        "mean_absolute_shap_log_odds": gene_importance,
        "mean_signed_shap_log_odds": values.mean(axis=0),
        "hallmark_memberships": np.asarray(membership.sum(axis=1)).ravel().astype(int),
    }).sort_values("mean_absolute_shap_log_odds", ascending=False, kind="stable")
    genes.insert(0, "rank", np.arange(1, len(genes) + 1))
    unsigned_allocations = np.asarray(np.abs(values) @ weights)
    pathway_rows = []
    for index, name in enumerate(pathway_names):
        column = weights[:, index].tocoo()
        driver_scores = gene_importance[column.row] * column.data
        driver_order = top_indices(driver_scores, min(5, len(driver_scores)))
        total = float(driver_scores.sum())
        driver_index = int(column.row[driver_order[0]])
        denominator = unsigned_allocations[:, index]
        cancellation = np.divide(
            np.abs(pathways[:, index]), denominator,
            out=np.ones_like(denominator), where=denominator > 0,
        )
        pathway_rows.append({
            "pathway": name, "mapped_gene_count": len(column.row),
            "mean_absolute_signed_allocation_log_odds": float(np.abs(pathways[:, index]).mean()),
            "mean_signed_allocation_log_odds": float(pathways[:, index].mean()),
            "mean_sum_absolute_gene_allocation_log_odds": total,
            "mean_within_sample_cancellation_fraction": float(np.mean(1 - cancellation)),
            "top_driver_gene_id": gene_ids[driver_index],
            "top_driver_symbol": symbols[driver_index],
            "top_driver_absolute_share": float(driver_scores[driver_order[0]] / total) if total else 0.0,
            "top5_drivers_absolute_share": float(driver_scores[driver_order].sum() / total) if total else 0.0,
        })
    pathway_frame = pd.DataFrame(pathway_rows).sort_values(
        "mean_absolute_signed_allocation_log_odds", ascending=False, kind="stable"
    )
    pathway_frame.insert(0, "rank", np.arange(1, len(pathway_frame) + 1))
    return genes, pathway_frame, pathways, residual


def sample_driver_rows(values, membership, gene_ids, symbols, pathway_names, ids, fold):
    pathways, _, weights = aggregate_attributions(values, membership)
    rows = []
    for sample, sample_id in enumerate(ids):
        for index in top_indices(np.abs(pathways[sample]), 5):
            column = weights[:, index].tocoo()
            contributions = values[sample, column.row] * column.data
            order = top_indices(np.abs(contributions), 3)
            denominator = float(np.abs(contributions).sum())
            rows.append({
                "fold": fold, "sample_id": sample_id, "pathway": pathway_names[index],
                "signed_allocation_log_odds": float(pathways[sample, index]),
                "sum_absolute_gene_allocation_log_odds": denominator,
                "top_driver_gene_id": gene_ids[column.row[order[0]]],
                "top_driver_symbol": symbols[column.row[order[0]]],
                "top_driver_signed_allocation_log_odds": float(contributions[order[0]]),
                "top_driver_absolute_share": float(abs(contributions[order[0]]) / denominator) if denominator else 0.0,
                "top3_driver_gene_ids": "|".join(gene_ids[column.row[i]] for i in order),
                "top3_driver_symbols": "|".join(symbols[column.row[i]] for i in order),
            })
    return rows


def verify_linear_shap(explainer, model, background, validation):
    explanation = explainer(validation)
    values = np.asarray(explanation.values, dtype=np.float64)
    base = np.asarray(explanation.base_values, dtype=np.float64)
    if values.shape != validation.shape:
        raise ValueError("Expected binary logistic-regression SHAP matrix")
    expected_values = model.coef_[0] * (validation - background.mean(axis=0))
    np.testing.assert_allclose(values, expected_values, atol=1e-8, rtol=1e-8)
    decision = np.asarray(model.decision_function(validation))
    np.testing.assert_allclose(base + values.sum(axis=1), decision, atol=1e-8, rtol=1e-8)
    return values, base, decision


def make_figures(output, genes, pathways, controls, observed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    for frame, label_column, value_column, title, file_name in (
        (genes.head(20), "gene_symbol", "mean_absolute_shap_log_odds",
         "Gene contributions: pooled outer-validation explanations", "gene_importance.png"),
        (pathways.head(10), "pathway", "mean_absolute_signed_allocation_log_odds",
         "Hallmark summaries: allocated gene SHAP contributions", "pathway_importance.png"),
    ):
        shown = frame.iloc[::-1]
        labels = shown[label_column].astype(str).str.replace("HALLMARK_", "", regex=False)
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(labels, shown[value_column], color="#356a85")
        axis.set_xlabel("Mean absolute contribution (model log odds)")
        axis.set_title(title, fontsize=11)
        figure.tight_layout()
        figure.savefig(figure_dir / file_name, dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(
        axes,
        ("mean_covered_gene_overlap", "mean_pathway_jaccard"),
        ("Top-gene overlap with top pathways", "Top-pathway cross-fold agreement"),
    ):
        axis.hist(controls[metric], bins=12, color="#bfd2dc", edgecolor="white")
        axis.axvline(observed[metric], color="#aa4827", label="Real Hallmark groups")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Overlap fraction" if "overlap" in metric else "Mean Jaccard")
        axis.set_ylabel("Random gene relabellings")
        axis.legend(fontsize=8)
    figure.suptitle("Exploratory controls: pathway size and overlap structure preserved", fontsize=11)
    figure.tight_layout()
    figure.savefig(figure_dir / "random_group_controls.png", dpi=160)
    plt.close(figure)


def write_report(output, summary, genes, pathways):
    observed = summary["observed"]
    stability = summary["mean_pairwise_topk_jaccard"]
    null = summary["random_control_comparison"]
    top_names = ", ".join(pathways.head(5)["pathway"].tolist())
    if isinstance(summary["sfari_comparison"], dict):
        sfari = summary["sfari_comparison"]
        sfari_text = (
            f"Exact-symbol SFARI annotation was run using the frozen reference. "
            f"Among the pooled top 20 genes, {sfari['top20_exact_matches']} matched "
            f"a SFARI entry and {sfari['top20_score_1_or_2']} had score 1 or 2. "
            f"The corresponding full-universe counts were {sfari['universe_exact_matches']} "
            f"and {sfari['universe_score_1_or_2']}, out of {summary['gene_count']:,} genes. "
            "These are descriptive counts, not an enrichment test. Missing scores "
            "remain missing; syndromic status is separate. Unmatched symbols may "
            "include aliases and are not evidence of no ASD relationship. SFARI "
            "genetic-risk evidence is not a list of validated blood-expression "
            "biomarkers, and a match does not validate a model or establish causation."
        )
    else:
        sfari_text = (
            "SFARI comparison was not run: no frozen reference was supplied. "
            "A pathway-membership check is not a substitute for a separate "
            "genetic-risk reference, and SFARI risk evidence is not a list of "
            "validated blood biomarkers."
        )
    text = f"""# Gene and pathway explanations

## Analysis population and estimator

This exploratory analysis explains {summary['sample_count']} existing training
records using five logistic-regression models. Each record was explained only
by its outer-fold model, which did not use that record for fitting or selecting
hyperparameters. No held-out test files were read or reevaluated. The folds share
training records and are not independent replications. This internal exercise
does not establish transportability to a new cohort or clinical usefulness.

SHAP LinearExplainer used the complete corresponding outer-training reference
population with an Independent masker. Values explain model log odds, not changes
in ASD risk caused by genes. This assumes an interventional feature treatment;
correlated gene expression can complicate the interpretation. The exact linear
formula and reconstruction of each model decision were verified numerically.

## Main observations

- Gene universe: {summary['gene_count']:,}; Hallmark-covered genes:
  {summary['covered_gene_count']:,} ({summary['covered_gene_fraction']:.1%}).
- The {summary['pathway_count']} Hallmark groups cover only part of the model.
  Unassigned genes account for {summary['unassigned_absolute_gene_share']:.1%}
  of the total absolute gene-attribution mass in these explanations.
- Mean pairwise top-20 gene Jaccard: {stability['genes_all']:.3f}; restricting
  gene selection to covered genes: {stability['genes_covered']:.3f}.
- Mean pairwise top-5 pathway Jaccard: {stability['pathways']:.3f}.
  These raw gene/pathway values are not directly comparable because the
  universes and top-list lengths differ.
- Mean proportion of top-20 covered genes appearing in the top-five pathways:
  {observed['mean_covered_gene_overlap']:.1%}. Across the unrestricted top-20
  genes the corresponding proportion is {observed['mean_all_gene_overlap']:.1%}.
- Real pathway agreement is at the {null['mean_pathway_jaccard']['midrank_percentile']:.1f}th
  exploratory percentile of {summary['random_draws']} random relabellings;
  covered-gene overlap is at the
  {null['mean_covered_gene_overlap']['midrank_percentile']:.1f}th percentile.

The leading pooled pathway labels are: {top_names}.
Their ranking is descriptive and does not prove that these pathways are
activated, causal, specific to ASD, or relevant to brain tissue.

## What was compared

Each gene's signed contribution is divided equally between the Hallmark groups
containing it. Summing all groups and the unassigned-gene residual reconstructs
the full gene attribution. These are post-hoc allocated summaries, **not
independent pathway Shapley values**. Agreement between the two levels is partly
built into this construction and is not independent biological validation.

Pathway rankings use the mean absolute value of the *signed sum*. Separate
columns report unsigned contribution mass, within-sample cancellation and the
largest gene's share. This distinguishes aligned model contributions from
opposing model contributions and summaries dominated by a single gene; it is
not evidence of biological coordination. No threshold
defines a medically acceptable overlap, cancellation or driver share.

The random controls globally permute covered-gene labels in the membership
matrix, using the same permutation across all five folds. They preserve every
pathway's size, pairwise intersection sizes and the distribution of membership
degrees. They do not preserve expression correlation within groups or each
particular gene's membership degree. Percentiles are descriptive exploratory
comparisons under that particular null, not clinical validation or formal
multiple-testing-adjusted significance results. No patient data were generated.

## Scope and limitations

This is the primary LR explanation analysis. Random Forest and KNN remain
supporting classifier comparisons; their explanations have not been computed
and cannot be inferred from LR. Model accuracy is unchanged by this analysis.

Blood-cell composition, cohort effects and correlated predictors can influence
these associations. Pooled fold rankings mix explanations from five fitted
models with different training-reference values. The original training data
have also been used in earlier exploratory research; this is a prospectively
recorded analysis within this run, not a preregistered independent study.

{sfari_text}

## Files

- `protocol.json`: analysis settings, input/model checksums, reference and scope.
- `summary.json`: observations, descriptive control percentiles and audit.
- `gene_rankings.csv`, `pathway_rankings.csv`: pooled outer-validation rankings.
- `fold_gene_rankings.csv`, `fold_pathway_rankings.csv`: fold-specific rankings.
- `gene_pathway_overlap.csv`: foldwise top-list overlap and gene membership.
- `fold_stability.csv`: pairwise top-list Jaccard values.
- `random_controls.csv`: every seeded random-control result.
- `sample_pathway_drivers.csv`: drivers of each sample's five strongest summaries.
- `validation_predictions.csv`: outer-validation predictions and reconstructed logits.
- `attributions/fold_*.npz`: full-precision sample-level arrays and feature IDs.
- `figures/`: static descriptive figures.

No new methodological or ASD-specific novelty claim follows from these results
alone. References and how they were used are recorded in the parent research
documentation.
"""
    (output / "RESULTS.md").write_text(text)


def run_analysis(input_dir, models_dir, gmt, output, draws=100, seed=20260908, sfari_csv=None):
    """Run a frozen, training-only analysis and never overwrite an existing run."""
    import shap

    input_dir, models_dir, gmt, output = map(Path, (input_dir, models_dir, gmt, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}")
    if draws < 100:
        raise ValueError("At least 100 random-control draws are required")
    model_files = [models_dir / "outer_models" / f"fold_{i}_logistic_regression.joblib" for i in range(5)]
    for path in [gmt, *model_files]:
        if not path.is_file():
            raise FileNotFoundError(path)
    gmt_manifest_path = gmt.with_name("source_manifest.json")
    gmt_manifest = json.loads(gmt_manifest_path.read_text())
    if gmt.name != gmt_manifest["file"] or sha256(gmt) != gmt_manifest["sha256"]:
        raise ValueError("GMT differs from its verified reference manifest")
    if sfari_csv is not None:
        sfari_csv = Path(sfari_csv)
        if not sfari_csv.is_file():
            raise FileNotFoundError(sfari_csv)
        sfari_manifest_path = sfari_csv.with_name("source_manifest.json")
        sfari_manifest = json.loads(sfari_manifest_path.read_text())
        if sfari_csv.name != sfari_manifest["clean_file"] or sha256(sfari_csv) != sfari_manifest["clean_sha256"]:
            raise ValueError("SFARI CSV differs from its verified projected-reference manifest")
    opened = training_data_guard(input_dir)
    X, y, metadata = load_training(input_dir)
    dictionary = pd.read_csv(input_dir / "gene_dictionary.csv", dtype={"gene_id": str})
    symbol_lookup = dictionary.set_index("gene_id")["gene_symbol"].fillna("").to_dict()
    gene_ids = [str(column) for column in X.columns if str(column) in symbol_lookup]
    if not gene_ids:
        raise ValueError("No input expression features match the gene dictionary")
    symbols = [symbol_lookup[gene] or gene for gene in gene_ids]
    gene_positions = {gene: index for index, gene in enumerate(gene_ids)}
    names, membership, source_sizes = read_pathways(gmt, gene_ids)
    if len(names) != gmt_manifest["expected_sets"] or len(names) != 50:
        raise ValueError("The primary protocol requires the verified 50 Hallmark gene sets")
    covered = np.asarray(membership.sum(axis=1)).ravel() > 0
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "analysis": "outer-validation logistic-regression SHAP with allocated Hallmark summaries",
        "gene_top_k": 20, "pathway_top_k": 5,
        "pathway_ranking": "mean absolute signed allocated contribution",
        "shap_version": shap.__version__, "explainer": "LinearExplainer",
        "masker": "Independent, all corresponding outer-training standardized rows",
        "output_scale": "log odds of positive class 1 (ASD)",
        "aggregation": "split each gene equally among memberships; retain unassigned residual",
        "random_draws": draws, "random_seed": seed,
        "randomization": "covered-gene label permutation, paired across folds",
        "draw_interpretation": "exploratory midrank percentiles, not formal p-values",
        "sfari": ({
            "file_name": sfari_csv.name, "sha256": sha256(sfari_csv),
            "reference_release": sfari_manifest["release_label"],
            "manifest_sha256": sha256(sfari_manifest_path),
            "method": "exact gene-symbol annotation; descriptive all-entry and score-1-or-2 counts",
            "role": "external genetic-risk context, not feature selection or blood-biomarker validation",
        } if sfari_csv else "not run; no frozen verified input supplied"),
        "test_policy": "never read held-out test files",
        "already_explored_training_data": True,
        "input_sha256": {name: sha256(input_dir / name) for name in (
            "X_train_log2.csv.gz", "y_train.csv", "metadata_train.csv", "gene_dictionary.csv"
        )},
        "model_sha256": {path.name: sha256(path) for path in model_files},
        "gmt_name": gmt.name, "gmt_sha256": sha256(gmt),
        "gmt_manifest_sha256": sha256(gmt_manifest_path),
        "source_code_sha256": sha256(Path(__file__)),
        "feature_order": "gene-expression input order; ties stable in this order",
        "pathway_order": "GMT file order; ties stable in this order",
    }
    write_json(output / "protocol.json", protocol)
    source_dir = output / "source"
    source_dir.mkdir()
    for source_path in (Path(__file__), Path(__file__).with_name("common.py")):
        shutil.copy2(source_path, source_dir / source_path.name)
    attribution_dir = output / "attributions"
    attribution_dir.mkdir()
    pd.DataFrame({
        "pathway": names, "source_gene_count": source_sizes,
        "mapped_gene_count": np.asarray(membership.sum(axis=0)).ravel().astype(int),
    }).to_csv(output / "pathway_coverage.csv", index=False)
    all_values, all_genes, all_pathways = [], [], []
    all_drivers, predictions, overlap_rows, sets = [], [], [], []
    seen_ids = set()
    audit = []
    for fold, path in enumerate(model_files):
        bundle = joblib.load(path)
        model, preprocessor = bundle["model"], bundle["preprocessor"]
        if list(model.classes_) != [0, 1] or np.asarray(model.coef_).shape[0] != 1:
            raise ValueError("Expected binary LR with class 1 denoting ASD")
        train_ids = [str(value) for value in bundle["train_ids"]]
        validation_ids = [str(value) for value in bundle["validation_ids"]]
        if set(train_ids) & set(validation_ids) or seen_ids & set(validation_ids):
            raise ValueError("Overlapping training/validation or repeated validation records")
        if set(train_ids) | set(validation_ids) != set(X.index.astype(str)):
            raise ValueError("Outer fold must partition the current training records")
        expected_ids = set(metadata.index[metadata.training_cv_fold.eq(fold)].astype(str))
        if set(validation_ids) != expected_ids:
            raise ValueError("Model validation records differ from the preserved outer fold")
        if set(metadata.loc[train_ids, "relatedness_group"]) & set(metadata.loc[validation_ids, "relatedness_group"]):
            raise ValueError("Relatedness groups overlap between model training and validation")
        seen_ids.update(validation_ids)
        background_frame = preprocessor.transform(X.loc[train_ids])
        validation_frame = preprocessor.transform(X.loc[validation_ids])
        feature_names = [str(name) for name in bundle["feature_names"]]
        if list(background_frame.columns.astype(str)) != feature_names:
            raise ValueError("Bundle feature order differs from preprocessing output")
        if list(validation_frame.columns.astype(str)) != feature_names:
            raise ValueError("Validation feature order differs from preprocessing output")
        if not set(feature_names) <= set(gene_ids):
            raise ValueError("Model contains features outside the recorded gene universe")
        background = background_frame.to_numpy(dtype=np.float64)
        validation = validation_frame.to_numpy(dtype=np.float64)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(model, masker)
        values, base, decision = verify_linear_shap(explainer, model, background, validation)
        full_values = np.zeros((len(validation_ids), len(gene_ids)), dtype=np.float64)
        positions = [gene_positions[gene] for gene in feature_names]
        full_values[:, positions] = values
        genes, pathways, pathway_values, residual = attribution_rankings(
            full_values, membership, gene_ids, symbols, names
        )
        np.testing.assert_allclose(
            base + pathway_values.sum(axis=1) + residual, decision,
            atol=1e-8, rtol=1e-8,
        )
        np.savez_compressed(
            attribution_dir / f"fold_{fold}.npz",
            sample_ids=np.asarray(validation_ids), gene_ids=np.asarray(gene_ids),
            gene_shap_log_odds=full_values, model_gene_ids=np.asarray(feature_names),
            standardized_model_features=validation, base_log_odds=base,
            pathway_names=np.asarray(names), pathway_allocations_log_odds=pathway_values,
            unassigned_log_odds=residual, model_decision_log_odds=decision,
            training_background_mean=background.mean(axis=0),
        )
        genes.insert(0, "fold", fold)
        pathways.insert(0, "fold", fold)
        all_genes.append(genes)
        all_pathways.append(pathways)
        all_values.append(full_values)
        all_drivers.extend(sample_driver_rows(
            full_values, membership, gene_ids, symbols, names, validation_ids, fold
        ))
        current_sets = fold_top_sets(full_values, membership)
        sets.append(current_sets)
        for index in sorted(current_sets["genes_all"] | current_sets["genes_covered"]):
            containing = [names[p] for p in sorted(current_sets["pathways"]) if membership[index, p] != 0]
            overlap_rows.append({
                "fold": fold, "gene_id": gene_ids[index], "gene_symbol": symbols[index],
                "in_top20_all_genes": index in current_sets["genes_all"],
                "in_top20_covered_genes": index in current_sets["genes_covered"],
                "hallmark_covered": bool(covered[index]),
                "in_top5_pathway": bool(containing), "top_pathway_memberships": "|".join(containing),
            })
        predicted = model.predict(validation)
        probability = model.predict_proba(validation)[:, 1]
        for position, sample_id in enumerate(validation_ids):
            truth, prediction = int(y.loc[sample_id]), int(predicted[position])
            outcome = {(1, 1): "true_positive", (1, 0): "false_negative", (0, 1): "false_positive", (0, 0): "true_negative"}[truth, prediction]
            predictions.append({
                "fold": fold, "sample_id": sample_id, "label": truth,
                "prediction": prediction, "probability_asd": float(probability[position]),
                "decision_log_odds": float(decision[position]), "outcome": outcome,
            })
        audit.append({
            "fold": fold, "train_count": len(train_ids), "validation_count": len(validation_ids),
            "model_gene_count": len(feature_names), "background_count": len(background),
            "max_formula_error": float(np.max(np.abs(values - model.coef_[0] * (validation - background.mean(axis=0))))),
            "max_reconstruction_error": float(np.max(np.abs(base + values.sum(axis=1) - decision))),
            "params": bundle["params"],
        })
        print(f"Explained outer fold {fold + 1}/5 ({len(validation_ids)} validation records)", flush=True)
    if seen_ids != set(X.index.astype(str)):
        raise ValueError("Outer validation records do not cover training data exactly once")
    combined_values = np.concatenate(all_values)
    pooled_genes, pooled_pathways, _, _ = attribution_rankings(
        combined_values, membership, gene_ids, symbols, names
    )
    sfari_summary = "not_run_no_frozen_reference"
    if sfari_csv is not None:
        sfari_reference = pd.read_csv(sfari_csv, dtype={"gene_symbol": str})
        annotated = annotate_sfari(pooled_genes, sfari_reference)
        annotated.to_csv(output / "gene_sfari_annotation.csv", index=False)
        fold_annotated = annotate_sfari(pd.concat(all_genes, ignore_index=True), sfari_reference)
        fold_annotated.to_csv(output / "fold_gene_sfari_annotation.csv", index=False)
        sfari_summary = {
            "method": "exact_symbol_descriptive_only",
            "universe_exact_matches": int(annotated.sfari_exact_symbol_match.sum()),
            "universe_score_1_or_2": int(annotated.sfari_score_1_or_2.sum()),
            "top20_exact_matches": int(annotated.head(20).sfari_exact_symbol_match.sum()),
            "top20_score_1_or_2": int(annotated.head(20).sfari_score_1_or_2.sum()),
        }
    pooled_genes.to_csv(output / "gene_rankings.csv", index=False)
    pooled_pathways.to_csv(output / "pathway_rankings.csv", index=False)
    pd.concat(all_genes, ignore_index=True).to_csv(output / "fold_gene_rankings.csv", index=False)
    pd.concat(all_pathways, ignore_index=True).to_csv(output / "fold_pathway_rankings.csv", index=False)
    pd.DataFrame(all_drivers).to_csv(output / "sample_pathway_drivers.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output / "gene_pathway_overlap.csv", index=False)
    stability = pd.DataFrame(stability_rows(sets))
    stability.to_csv(output / "fold_stability.csv", index=False)
    controls = random_controls(all_values, membership, draws, seed)
    controls.to_csv(output / "random_controls.csv", index=False)
    mean_stability = stability.groupby("level")["jaccard"].mean().to_dict()
    observed = {
        "mean_covered_gene_overlap": float(np.mean([row["covered_gene_overlap"] for row in sets])),
        "mean_all_gene_overlap": float(np.mean([row["all_gene_overlap"] for row in sets])),
        "mean_pathway_jaccard": float(mean_stability["pathways"]),
    }
    null_comparison = {
        metric: {
            "observed": value, "random_mean": float(controls[metric].mean()),
            "random_q025": float(controls[metric].quantile(.025)),
            "random_q975": float(controls[metric].quantile(.975)),
            "midrank_percentile": exploratory_percentile(value, controls[metric]),
        } for metric, value in observed.items()
    }
    total_absolute = np.abs(combined_values).sum()
    summary = {
        "sample_count": len(y), "gene_count": len(gene_ids), "pathway_count": len(names),
        "covered_gene_count": int(covered.sum()), "covered_gene_fraction": float(covered.mean()),
        "unassigned_absolute_gene_share": float(np.abs(combined_values[:, ~covered]).sum() / total_absolute),
        "random_draws": draws, "observed": observed,
        "mean_pairwise_topk_jaccard": mean_stability,
        "random_control_comparison": null_comparison,
        "fold_audit": audit, "input_files_opened": sorted(str(p) for p in opened),
        "test_evaluated": False, "sfari_comparison": sfari_summary,
    }
    write_json(output / "summary.json", summary)
    write_report(output, summary, pooled_genes, pooled_pathways)
    make_figures(output, pooled_genes, pooled_pathways, controls, observed)
    for name, expected in protocol["input_sha256"].items():
        if sha256(input_dir / name) != expected:
            raise ValueError(f"Training input changed during analysis: {name}")
    for path in model_files:
        if sha256(path) != protocol["model_sha256"][path.name]:
            raise ValueError(f"Model changed during analysis: {path.name}")
    if sha256(gmt) != protocol["gmt_sha256"] or sha256(gmt_manifest_path) != protocol["gmt_manifest_sha256"]:
        raise ValueError("Hallmark reference changed during analysis")
    if sfari_csv and (sha256(sfari_csv) != protocol["sfari"]["sha256"] or sha256(sfari_manifest_path) != protocol["sfari"]["manifest_sha256"]):
        raise ValueError("SFARI reference changed during analysis")
    if sha256(Path(__file__)) != protocol["source_code_sha256"]:
        raise ValueError("Explanation source changed during analysis")
    protocol["status"] = "complete"
    protocol["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["integrity_checks_passed"] = True
    write_json(output / "protocol.json", protocol)
    print(f"Completed training-only explanation analysis: {output}", flush=True)
    return summary


def main():
    research = Path(__file__).resolve().parents[2]
    fresh = research
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=fresh / "phase2_outputs")
    parser.add_argument("--models-dir", type=Path, default=research / "results" / "model_comparison")
    parser.add_argument("--gmt", type=Path, default=fresh / "data" / "reference" / "hallmark" / "msigdb_2026.1.Hs" / "h.all.v2026.1.Hs.entrez.gmt")
    parser.add_argument("--output", type=Path, default=research / "results" / "gene_pathway_explanations")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--sfari-csv", type=Path, help="Frozen gene_symbol,score,syndromic CSV; optional descriptive annotation")
    args = parser.parse_args()
    run_analysis(args.input_dir, args.models_dir, args.gmt, args.output, args.draws, args.seed, args.sfari_csv)


if __name__ == "__main__":
    main()
