"""Gene matching, train-only transformations, and preparation checks. No classifier."""
from __future__ import annotations
import gzip
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2_outputs"
REPORT = ROOT / "reports/preparation"
GNUSE_THRESHOLD = 1.25
GNUSE_SOURCE = "https://pubmed.ncbi.nlm.nih.gov/21548974/"


def gnuse_exclusion_mask(platforms, scores):
    """Published GPL570 rule; no diagnoses, model scores, or subjective plot decisions."""
    supported = platforms.eq("GPL570")
    if not platforms.isin(["GPL570", "GPL6244"]).all():
        raise ValueError("Unexpected platform: define an appropriate QC policy first")
    if not np.isfinite(scores.loc[supported]).all() or (scores.loc[supported] <= 0).any():
        raise ValueError("Missing or invalid GNUSE for GPL570: do not silently mark as passed")
    return supported & scores.gt(GNUSE_THRESHOLD)


def apply_quality_filter(intake_manifest):
    """Apply the authorized QC rule after per-array fRMA, before any fitted scaling.

    Phase 1 remains the 331-record acquisition log. Phase 2 exports the final
    327-record manifest. Preserve the original holdout and retained CV fold IDs.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if not intake_manifest.index.is_unique:
        raise ValueError("Intake sample IDs must be unique")
    quality_rows = []
    for gsm, row in intake_manifest.iterrows():
        q = pd.read_csv(ROOT / f"data/whole_blood/frma/{row.platform}/{gsm}_qc.csv")
        if len(q) != 1 or q.sample_id.iloc[0] != gsm or q.platform.iloc[0] != row.platform:
            raise ValueError(f"QC/sample mismatch: {gsm}")
        quality_rows.append(q)
    quality = pd.concat(quality_rows).set_index("sample_id").loc[intake_manifest.index]
    decisions = intake_manifest.copy()
    decisions["gnuse_median"] = quality.gnuse_median
    rejected = gnuse_exclusion_mask(decisions.platform, decisions.gnuse_median)
    decisions["qc_status"] = np.where(decisions.platform.eq("GPL6244"),
                                      "GNUSE_not_available_for_core_output", "passes_GNUSE_rule")
    decisions.loc[rejected, "qc_status"] = "excluded_GNUSE_above_1.25"
    decisions["included_after_qc"] = ~rejected
    decisions["qc_exclusion_reason"] = ""
    decisions.loc[rejected, "qc_exclusion_reason"] = (
        "GPL570 median GNUSE > 1.25; published technical-quality exclusion criterion")
    excluded = decisions.loc[rejected].copy()
    excluded["included"] = False
    excluded["exclusion_reason"] = excluded.qc_exclusion_reason
    retained = decisions.loc[~rejected].copy()
    assert len(intake_manifest) == 331 and len(retained) == 327 and len(excluded) == 4
    assert retained.target.value_counts().to_dict() == {1: 204, 0: 123}
    original_test = intake_manifest.index[intake_manifest.split.eq("test")]
    assert retained.index[retained.split.eq("test")].equals(original_test), "Do not reshuffle the holdout"
    assert retained[["split", "training_cv_fold"]].equals(
        intake_manifest.loc[retained.index, ["split", "training_cv_fold"]])
    decisions.to_csv(OUT / "qc_decisions_all_intake_samples.csv")
    excluded.to_csv(OUT / "qc_excluded_samples.csv")
    retained.to_csv(OUT / "final_sample_manifest.csv")
    previous_exclusions = pd.read_csv(ROOT / "phase1_outputs/excluded_samples.csv", index_col="sample_id")
    all_excluded = pd.concat([previous_exclusions, excluded])
    assert all_excluded.index.is_unique and len(all_excluded) == 14
    all_excluded.to_csv(OUT / "all_excluded_samples.csv")
    retained.groupby(["cohort", "label"]).size().unstack(fill_value=0).to_csv(OUT / "cohort_counts.csv")
    retained.groupby(["split", "cohort", "label"]).size().unstack(fill_value=0).to_csv(OUT / "split_counts.csv")
    excluded[["cohort", "label", "split", "gnuse_median", "qc_exclusion_reason"]].to_csv(
        REPORT / "quality_exclusions.csv")
    policy = {
        "adopted_date": "2026-09-06", "platform": "GPL570", "metric": "median GNUSE",
        "operator": ">", "threshold": GNUSE_THRESHOLD, "source": GNUSE_SOURCE,
        "decision_basis": "User approved published exclusion criterion after viewing QC flags, before classifier training",
        "not_preregistered_before_data_inspection": True,
        "diagnosis_and_model_performance_not_used_by_rule": True,
        "GPL6244": "GNUSE unavailable at core level; not assessed by this rule, not certified as passed",
        "raw_files_preserved": True, "test_membership_unchanged": True,
        "retained_training_cv_folds_unchanged": True,
        "retained_records": len(retained), "excluded_records": len(excluded),
        "excluded_sample_ids": excluded.index.tolist(),
        "train_samples": int(retained.split.eq("train").sum()),
        "test_samples": int(retained.split.eq("test").sum()),
        "actual_test_fraction": float(retained.split.eq("test").mean()),
        "preprocessing": "Refit cohort means and scaler on retained training records only"
    }
    (REPORT / "quality_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return retained, decisions


def platform_mapping(platform):
    """Keep probes with exactly one Entrez ID and one nonempty gene symbol."""
    with gzip.open(ROOT / f"data/whole_blood/source/{platform}.gz", "rt") as f:
        for line in f:
            if line.startswith("!platform_table_begin"): break
        else: raise ValueError("Missing platform table")
        annotation = pd.read_csv(f, sep="\t", comment="!", dtype=str, keep_default_na=False)
    rows = []
    for _, row in annotation.iterrows():
        ids = set(t.strip() for t in row["Gene ID"].split("///") if t.strip())
        symbols = set(t.strip() for t in row["Gene symbol"].split("///") if t.strip())
        valid_id = len(ids) == 1 and bool(re.fullmatch(r"[1-9]\d*", next(iter(ids), "")))
        reason = ""
        if row.ID.startswith("AFFX"): reason = "control probe"
        elif not valid_id: reason = "missing or ambiguous Entrez ID"
        elif len(symbols) != 1: reason = "missing or ambiguous symbol"
        rows.append({"probe_id": row.ID, "gene_id": next(iter(ids)) if valid_id else "",
                     "gene_symbol": next(iter(symbols)) if len(symbols) == 1 else "",
                     "gene_title": row["Gene title"], "included_mapping": not bool(reason),
                     "mapping_exclusion_reason": reason})
    audit = pd.DataFrame(rows)
    assert audit.probe_id.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / f"{platform}_mapping_audit.csv.gz", index=False)
    return audit.loc[audit.included_mapping].set_index("probe_id"), audit


def merge_genes(manifest):
    """Median of unambiguous probes per gene, then intersection across platforms."""
    matrices, mapping_counts, annotations, qc = [], [], [], []
    gene_sets = {}
    for platform, members in manifest.groupby("platform", sort=True):
        mapping, audit = platform_mapping(platform)
        sample_series = []
        expected_probes = None
        for gsm in members.index:
            path = ROOT / f"data/whole_blood/frma/{platform}/{gsm}.tsv.gz"
            probes = pd.read_csv(path, sep="\t", dtype={"probe_id": str}).set_index("probe_id")
            assert probes.index.is_unique and np.isfinite(probes.log2_expression).all()
            if expected_probes is None: expected_probes = probes.index
            assert probes.index.equals(expected_probes), f"Probe mismatch for {gsm}"
            matched = probes.join(mapping[["gene_id"]], how="inner")
            genes = matched.groupby("gene_id").log2_expression.median().rename(gsm)
            sample_series.append(genes)
            qc.append(pd.read_csv(ROOT / f"data/whole_blood/frma/{platform}/{gsm}_qc.csv"))
        matrix = pd.concat(sample_series, axis=1).T
        gene_sets[platform] = set(matrix.columns)
        matrices.append(matrix)
        used = mapping.loc[mapping.index.intersection(expected_probes)].reset_index()
        used["platform"] = platform
        annotations.append(used)
        mapping_counts.append({"platform": platform, "annotation_rows": len(audit),
                               "frma_features": len(expected_probes),
                               "unambiguous_measured_probes": len(used),
                               "unique_mapped_genes": matrix.shape[1]})
    common = sorted(set.intersection(*gene_sets.values()), key=int)
    assert len(common) > 1000, "Too few genes: stop and investigate mapping"
    merged = pd.concat([m.loc[:, common] for m in matrices]).loc[manifest.index]
    merged.index.name = "sample_id"
    merged.columns.name = "gene_id"
    assert merged.index.is_unique and merged.columns.is_unique
    assert np.isfinite(merged.to_numpy()).all()
    pd.DataFrame(mapping_counts).to_csv(OUT / "mapping_counts.csv", index=False)
    annotated = pd.concat(annotations, ignore_index=True)
    annotated.loc[annotated.gene_id.isin(common)].to_csv(OUT / "retained_probe_gene_mapping.csv.gz", index=False)
    info = annotated.loc[annotated.gene_id.isin(common)].groupby("gene_id").agg(
        gene_symbol=("gene_symbol", lambda x: " | ".join(sorted(set(x)))),
        gene_title=("gene_title", lambda x: " | ".join(sorted(set(x)))),
        platforms=("platform", lambda x: " | ".join(sorted(set(x)))))
    info = info.reindex(common)
    info["symbol_disagreement"] = info.gene_symbol.str.contains(" | ", regex=False)
    info.to_csv(OUT / "gene_dictionary.csv")
    pd.concat(qc).set_index("sample_id").loc[manifest.index].to_csv(OUT / "sample_quality.csv")
    merged.to_csv(OUT / "all_samples_common_genes_log2.csv.gz", float_format="%.8g")
    return merged, info


class WholeBloodPreprocessor(TransformerMixin, BaseEstimator):
    """Simple known-cohort mean adjustment followed by global standard scaling.

    Input: gene columns plus a string 'cohort' column. Targets are never consulted.
    Fit only on a training partition. Refit inside EACH future CV training fold.
    This adjusts mean offsets, not every platform effect or biological confounder.
    It cannot transform an unseen cohort without a different validation protocol.
    """
    def __init__(self, center_cohorts=True):
        self.center_cohorts = center_cohorts

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or "cohort" not in X:
            raise ValueError("Provide a DataFrame with gene columns and 'cohort'.")
        self.input_genes_ = X.columns.drop("cohort").to_numpy()
        values = X.loc[:, self.input_genes_].astype(float)
        if not np.isfinite(values.to_numpy()).all(): raise ValueError("Non-finite values")
        self.genes_ = values.columns[values.var(ddof=0) > 0].to_numpy()
        values = values.loc[:, self.genes_]
        self.global_mean_ = values.mean()
        self.cohort_means_ = values.groupby(X.cohort).mean()
        centered = self._center(values, X.cohort)
        # After centering a gene could become constant; StandardScaler handles that.
        self.scaler_ = StandardScaler().fit(centered)
        self.n_features_in_ = X.shape[1]
        return self

    def _center(self, values, cohorts):
        unknown = set(cohorts) - set(self.cohort_means_.index)
        if unknown: raise ValueError(f"Unseen cohort(s): {sorted(unknown)}")
        if not self.center_cohorts: return values.copy()
        offsets = self.cohort_means_.loc[cohorts].to_numpy() - self.global_mean_.to_numpy()
        return values - offsets

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        if list(X.columns.drop("cohort")) != list(self.input_genes_):
            raise ValueError("Gene columns/order differ from training")
        values = X.loc[:, self.genes_].astype(float)
        if not np.isfinite(values.to_numpy()).all(): raise ValueError("Non-finite values")
        centered = self._center(values, X.cohort)
        return pd.DataFrame(self.scaler_.transform(centered), index=X.index, columns=self.genes_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "genes_")
        return self.genes_.copy()


def split_and_prepare(merged, manifest):
    train_ids = manifest.index[manifest.split == "train"]
    test_ids = manifest.index[manifest.split == "test"]
    train, test = merged.loc[train_ids], merged.loc[test_ids]
    train_input = train.assign(cohort=manifest.loc[train_ids, "cohort"])
    test_input = test.assign(cohort=manifest.loc[test_ids, "cohort"])
    transformer = WholeBloodPreprocessor()
    train_prepared = transformer.fit_transform(train_input)
    test_prepared = transformer.transform(test_input)
    for split, raw, prepared in [("train", train, train_prepared), ("test", test, test_prepared)]:
        raw.to_csv(OUT / f"X_{split}_log2.csv.gz", float_format="%.8g")
        prepared.to_csv(OUT / f"X_{split}_prepared.csv.gz", float_format="%.8g")
        manifest.loc[raw.index, ["target"]].to_csv(OUT / f"y_{split}.csv")
        manifest.loc[raw.index].to_csv(OUT / f"metadata_{split}.csv")
    joblib.dump(transformer, OUT / "training_preprocessor.joblib")
    transformer.cohort_means_.to_csv(OUT / "training_cohort_gene_means.csv.gz")
    pd.DataFrame({"gene_id": transformer.genes_, "global_training_mean": transformer.global_mean_,
                  "scaler_mean": transformer.scaler_.mean_, "scaler_scale": transformer.scaler_.scale_}).to_csv(
                      OUT / "training_scaling_parameters.csv.gz", index=False)
    return train, test, train_prepared, test_prepared, transformer


def training_diagnostics(train, prepared, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True, exist_ok=True)
    meta = manifest.loc[train.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    variance = {}
    for ax, name, values in [(axes[0], "Before cohort-mean adjustment", StandardScaler().fit_transform(train)),
                             (axes[1], "After cohort-mean adjustment", prepared.to_numpy())]:
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        coords = pca.fit_transform(values)
        variance[name] = pca.explained_variance_ratio_.tolist()
        for cohort, color in zip(sorted(meta.cohort.unique()), ["#257491", "#b05b28", "#73589b"]):
            mask = (meta.cohort == cohort).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=19, alpha=.7, label=cohort, color=color)
        ax.set(title=name, xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",
               ylabel=f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle("Training samples only — PCA is a diagnostic, not a model input", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORT / "training_pca.png", dpi=160)
    plt.close(fig)
    qc = pd.read_csv(OUT / "sample_quality.csv", index_col="sample_id").join(manifest[["cohort", "split"]])
    qc["gnuse_exclusion_rule_triggered"] = gnuse_exclusion_mask(qc.platform, qc.gnuse_median)
    qc["gnuse_assessed"] = qc.platform.eq("GPL570")
    assert not qc.gnuse_exclusion_rule_triggered.any(), "Run the quality filter before diagnostics"
    qc.to_csv(OUT / "sample_quality.csv")
    return variance, qc


def validate_outputs(train, test, train_prepared, test_prepared, transformer, manifest):
    checks = {}
    checks["327_records_after_QC"] = len(train) + len(test) == 327
    checks["261_train_66_test"] = len(train) == 261 and len(test) == 66
    decisions = pd.read_csv(OUT / "qc_decisions_all_intake_samples.csv", index_col="sample_id")
    expected_rejections = gnuse_exclusion_mask(decisions.platform, decisions.gnuse_median)
    checks["published_QC_rule_applied_exactly"] = (
        set(manifest.index) == set(decisions.index[~expected_rejections]) and expected_rejections.sum() == 4)
    intake = pd.read_csv(ROOT / "phase1_outputs/split_manifest.csv", index_col="sample_id")
    checks["original_holdout_membership_preserved"] = test.index.equals(intake.index[intake.split.eq("test")])
    checks["retained_cv_folds_preserved"] = manifest.training_cv_fold.equals(intake.loc[manifest.index, "training_cv_fold"])
    checks["preprocessor_refitted_on_retained_train_only"] = (
        np.allclose(transformer.global_mean_, train.loc[:, transformer.genes_].mean(), atol=1e-12)
        and np.allclose(transformer.cohort_means_, train.loc[:, transformer.genes_].groupby(
            manifest.loc[train.index, "cohort"]).mean(), atol=1e-12))
    checks["no_shared_sample_ids"] = not bool(set(train.index) & set(test.index))
    checks["no_shared_known_or_possible_groups"] = not bool(
        set(manifest.loc[train.index, "relatedness_group"]) & set(manifest.loc[test.index, "relatedness_group"]))
    checks["grouped_training_cv"] = manifest.loc[train.index].groupby("relatedness_group").training_cv_fold.nunique().max() == 1
    checks["same_gene_order"] = train.columns.equals(test.columns) and train_prepared.columns.equals(test_prepared.columns)
    checks["all_values_finite"] = all(np.isfinite(x.to_numpy()).all() for x in [train, test, train_prepared, test_prepared])
    checks["no_exact_duplicate_gene_vectors"] = not pd.concat([train, test]).duplicated().any()
    checks["train_scaled_mean_zero"] = np.allclose(train_prepared.mean(), 0, atol=1e-10)
    checks["train_cohort_means_zero"] = np.allclose(train_prepared.groupby(manifest.loc[train.index, "cohort"]).mean(), 0, atol=1e-10)
    checks["mr_dd_and_repeated_title_excluded"] = "GSM650700" not in manifest.index and not manifest.original_diagnosis.str.contains("mental retardation|developmental delay", case=False).any()
    for split, ids in [("train", train.index), ("test", test.index)]:
        checks[f"{split}_both_labels_each_cohort"] = manifest.loc[ids].groupby("cohort").target.nunique().eq(2).all()
    # Transforming a modified test set must not alter fitted means or training output.
    original_means = transformer.cohort_means_.copy(deep=True)
    changed_test = (test + 100).assign(cohort=manifest.loc[test.index, "cohort"])
    transformer.transform(changed_test)
    train_again = transformer.transform(train.assign(cohort=manifest.loc[train.index, "cohort"]))
    checks["test_transform_does_not_change_fit"] = transformer.cohort_means_.equals(original_means) and np.array_equal(train_again, train_prepared)
    try:
        transformer.transform(test.assign(cohort="unseen_cohort"))
        checks["unseen_cohort_rejected"] = False
    except ValueError:
        checks["unseen_cohort_rejected"] = True
    checks = {k: bool(v) for k, v in checks.items()}
    (REPORT / "validation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    assert all(checks.values()), checks
    return checks


def summarize_preparation(merged, manifest, train, test, prepared_train, pca, qc, checks):
    summary = {"samples": len(merged), "ASD": int(manifest.target.sum()),
               "Control": int((manifest.target == 0).sum()), "unique_title_stem_groups": manifest.relatedness_group.nunique(),
               "common_genes": merged.shape[1], "prepared_genes": prepared_train.shape[1],
               "train_samples": len(train), "test_samples": len(test),
               "intake_records_before_QC": 331,
               "quality_excluded_records": len(pd.read_csv(OUT / "qc_excluded_samples.csv")),
               "retained_GPL570_GNUSE_passes": int(qc.gnuse_assessed.sum()),
               "retained_GPL6244_GNUSE_not_assessed": int((~qc.gnuse_assessed).sum()),
               "gnuse_missing": int(qc.gnuse_median.isna().sum()),
               "training_pca_explained_variance": pca,
               "checks_passed": sum(checks.values()), "no_classifiers_trained": True,
               "not_external_validation": True, "no_synthetic_samples": True}
    (REPORT / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def run_preparation():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    intake_manifest = pd.read_csv(ROOT / "phase1_outputs/split_manifest.csv", index_col="sample_id")
    manifest, decisions = apply_quality_filter(intake_manifest)
    merged, genes = merge_genes(manifest)
    train, test, prepared_train, prepared_test, transformer = split_and_prepare(merged, manifest)
    pca, qc = training_diagnostics(train, prepared_train, manifest)
    checks = validate_outputs(train, test, prepared_train, prepared_test, transformer, manifest)
    return summarize_preparation(merged, manifest, train, test, prepared_train, pca, qc, checks)


if __name__ == "__main__":
    run_preparation()
