"""Verify saved training-only model results without fitting models or reading test data."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from asd_blood.common import DEFAULT_INPUT, STUDY, TRAINING_FILES, sha256, training_data_guard

MEASURES = ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")
FAMILIES = ("logistic_regression", "random_forest", "knn")
TOLERANCE = 1e-11


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE), message)


def read_json(path):
    return json.loads(path.read_text())


def independent_metrics(truth, prediction, probability):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {"n": len(truth), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp + tn) / len(truth),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "roc_auc": roc_auc_score(truth, probability)}


def compare_metrics(actual, expected, context):
    for name, value in actual.items():
        close(value, expected[name], f"{context}: {name}")


def validate_preprocessor(bundle, train, cohort):
    """Reconstruct fitted sufficient statistics directly; never call fit()."""
    prep = bundle["preprocessor"]
    require(list(prep.input_genes_) == list(train.columns), "Preprocessor input order")
    retained = train.columns[train.var(ddof=0) > 0].tolist()
    require(list(prep.genes_) == retained == bundle["feature_names"], "Variance filter or feature order")
    raw = train.loc[:, retained]
    means, grand = raw.groupby(cohort).mean(), raw.mean()
    close(prep.global_mean_.to_numpy(), grand.to_numpy(), "Training-only global mean")
    require(prep.cohort_means_.index.equals(means.index), "Cohort names")
    close(prep.cohort_means_.to_numpy(), means.to_numpy(), "Training-only cohort means")
    adjusted = raw.to_numpy() - means.loc[cohort].to_numpy() + grand.to_numpy()
    close(prep.scaler_.mean_, adjusted.mean(axis=0), "Training-only scaler mean")
    close(prep.scaler_.var_, adjusted.var(axis=0), "Training-only scaler variance")
    close(prep.scaler_.n_samples_seen_, len(train), "Preprocessor fitted sample count")
    require(prep.center_cohorts is True, "Unexpected centering setting")
    require(bundle["model"].n_features_in_ == len(retained), "Model feature count")
    require(list(bundle["model"].classes_) == [0, 1], "Positive-class probability index")


def bootstrap_intervals(predictions, repetitions=1000):
    rng = np.random.default_rng(42)
    names = predictions.relatedness_group.unique()
    groups = {name: np.flatnonzero(predictions.relatedness_group.to_numpy() == name) for name in names}
    calculated = []
    for _ in range(repetitions):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = predictions.iloc[np.concatenate([groups[name] for name in selected])]
        if sample.true_label.nunique() == 2:
            calculated.append(independent_metrics(sample.true_label, sample.predicted_label, sample.probability_asd))
    return pd.DataFrame(calculated)


def verify(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    opened = training_data_guard(input_dir)
    protocol = read_json(output_dir / "protocol.json")
    require(protocol["status"] == "complete", "Training protocol is not complete")
    require(protocol["candidate_count"] == 54 and protocol["candidate_fits"] == 1350,
            "Unexpected fixed search budget")
    require(protocol["outer_and_final_refits"] == 18, "Unexpected refit count")
    require((protocol["outer_folds"], protocol["inner_folds"], protocol["full_training_search_folds"]) == (5, 4, 5),
            "Unexpected fold design")
    require(not any(protocol[name] for name in ("smote", "pca", "synthetic_samples")), "Unexpected analysis branch")
    require(set(protocol["accessed_training_files"]) == set(TRAINING_FILES), "Training access record")
    input_hashes = {name: sha256(input_dir / name) for name in TRAINING_FILES}
    require(input_hashes == protocol["input_sha256"], "Prepared training input hashes differ")
    files = sorted(path for path in output_dir.rglob("*") if path.is_file()
                   and path.name not in {"verification.json", "RESULTS.md"})
    preserved_hashes = {str(path.relative_to(output_dir)): sha256(path) for path in files}
    source_checks = {}
    for filename in ("common.py", "training.py"):
        snapshot = sha256(output_dir / "source" / filename)
        current = sha256(STUDY / "src" / "asd_blood" / filename)
        require(snapshot == current, f"Source differs from executed snapshot: {filename}")
        source_checks[filename] = snapshot
    versions = {name: importlib.metadata.version(name) for name in protocol["environment"]}
    require(versions == protocol["environment"], "Reconstruction software differs from run environment")

    expression = pd.read_csv(input_dir / "X_train_log2.csv.gz", index_col="sample_id")
    labels = pd.read_csv(input_dir / "y_train.csv", index_col="sample_id").target
    metadata = pd.read_csv(input_dir / "metadata_train.csv", index_col="sample_id")
    require(expression.shape == (261, 17263), "Training matrix shape")
    require(expression.index.equals(labels.index) and expression.index.equals(metadata.index), "Training alignment")
    require(metadata.split.eq("train").all() and labels.equals(metadata.target), "Training partition or targets")
    require(labels.value_counts().to_dict() == {1: 163, 0: 98}, "Class counts")
    require(expression.index.is_unique and expression.columns.is_unique, "Duplicate training identifiers")
    require(np.isfinite(expression.to_numpy()).all(), "Nonfinite training expression")
    fold_ids = metadata.training_cv_fold
    require(set(fold_ids) == set(range(5)), "Outer fold identifiers")
    require(metadata.groupby("relatedness_group").training_cv_fold.nunique().max() == 1, "Relatedness crosses folds")
    saved_folds = pd.read_csv(output_dir / "training_folds.csv", index_col="sample_id")
    pd.testing.assert_frame_equal(saved_folds, metadata[saved_folds.columns], check_dtype=False)
    oof = pd.read_csv(output_dir / "out_of_fold_predictions.csv")
    require(len(oof) == 783 and set(oof.model) == set(FAMILIES), "OOF record count")
    require(not oof.duplicated(["model", "sample_id"]).any(), "Duplicate OOF predictions")
    require(np.isfinite(oof.probability_asd).all() and oof.probability_asd.between(0, 1).all(), "Invalid probability")
    saved_fold_metrics = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    require(len(saved_fold_metrics) == 15, "Fold metric row count")
    summaries = {entry["model"]: entry for entry in read_json(output_dir / "summary.json")}
    choices = read_json(output_dir / "selected_parameters.json")
    history = read_json(output_dir / "selection_history.json")
    require(len(history) == 18, "Selection history count")
    histories = {(entry["stage"], entry["model"]): entry for entry in history}
    require(len(histories) == 18, "Duplicate selection histories")
    grids = {name: list(ParameterGrid(protocol["grids"][name])) for name in FAMILIES}
    require({name: len(grid) for name, grid in grids.items()} ==
            {"logistic_regression": 12, "random_forest": 18, "knn": 24}, "Candidate grids")
    expected_bundles, fit_rows, candidate_rows, roundoff_ties = {}, 0, 0, []
    for stage in [f"outer_{fold}" for fold in range(5)] + ["full_training"]:
        folds = 5 if stage == "full_training" else 4
        validation_folds = list(range(5)) if stage == "full_training" else [
            fold for fold in range(5) if fold != int(stage.rsplit("_", 1)[1])]
        for family, grid in grids.items():
            rows = pd.read_csv(output_dir / "searches" / f"{stage}_{family}_fold_scores.csv")
            candidates = pd.read_csv(output_dir / "searches" / f"{stage}_{family}_candidates.csv")
            require(len(rows) == len(grid) * folds and len(candidates) == len(grid), "Search row counts")
            require(set(rows.model) == {family} and set(candidates.model) == {family}, "Search family labels")
            require(not rows.duplicated(["candidate", "fold"]).any(), "Repeated search fit row")
            require(set(rows.candidate) == set(range(len(grid))) and set(rows.fold) == set(range(folds)), "Search IDs")
            require(set(candidates.candidate) == set(range(len(grid))), "Candidate summary IDs")
            for inner_index, original_fold in enumerate(validation_folds):
                recorded = rows[rows.fold.eq(inner_index)]
                truth = labels[fold_ids.eq(original_fold)]
                require(recorded.n.eq(len(truth)).all(), "Search validation record counts")
                require((recorded.tn + recorded.fp).eq(int(truth.eq(0).sum())).all(), "Search control counts")
                require((recorded.tp + recorded.fn).eq(int(truth.eq(1).sum())).all(), "Search ASD counts")
                require(recorded.roc_auc.between(0, 1).all(), "Invalid search ROC AUC")
                close(recorded.accuracy, (recorded.tp + recorded.tn) / recorded.n, "Search accuracy/count mismatch")
                close(recorded.sensitivity, recorded.tp / (recorded.tp + recorded.fn), "Search sensitivity/count mismatch")
                close(recorded.specificity, recorded.tn / (recorded.tn + recorded.fp), "Search specificity/count mismatch")
                close(recorded.balanced_accuracy, (recorded.sensitivity + recorded.specificity) / 2,
                      "Search balanced accuracy/count mismatch")
            for index, params in enumerate(grid):
                subset = rows[rows.candidate.eq(index)]
                require(len(subset) == folds and all(json.loads(value) == params for value in subset.parameters), "Grid parameter mismatch")
                summary = candidates[candidates.candidate.eq(index)].iloc[0]
                require(json.loads(summary.parameters) == params, "Summary parameters differ")
                for measure in MEASURES:
                    close(subset[measure].mean(), summary[measure], f"{stage}/{family}/{index}/{measure}")
            require(candidates.selected.sum() == 1 and bool(candidates.iloc[0].selected), "Selected candidate flag/order")
            ranked = candidates.sort_values(["balanced_accuracy", "roc_auc", "candidate"], ascending=[False, False, True])
            if int(ranked.iloc[0].candidate) != int(candidates.iloc[0].candidate):
                close(ranked.iloc[0].balanced_accuracy, candidates.iloc[0].balanced_accuracy, "Selection BA not maximal")
                close(ranked.iloc[0].roc_auc, candidates.iloc[0].roc_auc, "Selection AUC not maximal among ties")
                roundoff_ties.append({"stage": stage, "model": family})
            selected = json.loads(candidates.iloc[0].parameters)
            expected_bundles[(stage, family)] = selected
            require(histories[(stage, family)]["parameters"] == selected, "History selected parameters")
            close(histories[(stage, family)]["selection_cv_balanced_accuracy"], candidates.iloc[0].balanced_accuracy, "History BA")
            close(histories[(stage, family)]["selection_cv_accuracy"], candidates.iloc[0].accuracy, "History accuracy")
            if stage == "full_training":
                require(choices[family] == selected, "Final selected parameter JSON")
            fit_rows += len(rows)
            candidate_rows += len(candidates)
    require(fit_rows == 1350 and candidate_rows == 324, "Total search ledger size")

    checks, calculated_folds, maximum_probability_error = [], [], 0.0
    with threadpool_limits(limits=1):
        for fold in range(5):
            train_ids = expression.index[fold_ids.ne(fold)]
            validation_ids = expression.index[fold_ids.eq(fold)]
            train_meta, validation_meta = metadata.loc[train_ids], metadata.loc[validation_ids]
            require(not set(train_meta.relatedness_group) & set(validation_meta.relatedness_group), "Outer relatedness leakage")
            for inner in sorted(train_meta.training_cv_fold.unique()):
                left = train_meta[train_meta.training_cv_fold.ne(inner)]
                right = train_meta[train_meta.training_cv_fold.eq(inner)]
                require(not set(left.relatedness_group) & set(right.relatedness_group), "Inner relatedness leakage")
                for part in (left, right):
                    require(set(part.cohort) == set(metadata.cohort) and part.groupby("cohort").target.nunique().min() == 2,
                            "Inner cohort/class coverage")
            for family in FAMILIES:
                bundle = joblib.load(output_dir / "outer_models" / f"fold_{fold}_{family}.joblib")
                require(bundle["train_ids"] == list(train_ids) and bundle["validation_ids"] == list(validation_ids), "Bundle record membership")
                require(bundle["params"] == expected_bundles[(f"outer_{fold}", family)], "Outer selected parameters")
                expected_params = {**protocol["fixed_parameters"][family], **bundle["params"]}
                require(bundle["model"].get_params() == expected_params, "Full estimator parameter record")
                validate_preprocessor(bundle, expression.loc[train_ids], metadata.loc[train_ids, "cohort"])
                source = expression.loc[validation_ids].assign(cohort=metadata.loc[validation_ids, "cohort"])
                transformed = bundle["preprocessor"].transform(source).to_numpy()
                prediction = bundle["model"].predict(transformed)
                probability = bundle["model"].predict_proba(transformed)[:, 1]
                recorded = oof[oof.model.eq(family) & oof.fold.eq(fold)]
                require(recorded.sample_id.tolist() == list(validation_ids), "OOF validation order")
                require(np.array_equal(recorded.true_label, labels.loc[validation_ids]), "OOF ground truth")
                require(np.array_equal(recorded.relatedness_group, metadata.loc[validation_ids, "relatedness_group"]), "OOF relatedness labels")
                require(np.array_equal(recorded.cohort, metadata.loc[validation_ids, "cohort"]), "OOF cohort labels")
                require(np.array_equal(prediction, recorded.predicted_label), "Saved prediction mismatch")
                close(probability, recorded.probability_asd, "Saved probability mismatch")
                maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(probability - recorded.probability_asd))))
                computed = independent_metrics(labels.loc[validation_ids], prediction, probability)
                metric_row = saved_fold_metrics[saved_fold_metrics.model.eq(family) & saved_fold_metrics.fold.eq(fold)]
                require(len(metric_row) == 1, "Missing/repeated fold metric")
                compare_metrics(computed, metric_row.iloc[0], f"{family}/fold{fold}")
                calculated_folds.append({"model": family, "fold": fold, **computed})
                checks.append({"model": family, "fold": fold, "validation_records": len(validation_ids), "prediction_match": True,
                               "training_only_preprocessor_statistics": True, "group_separation": True})
            print(f"Verified outer fold {fold + 1}/5", flush=True)
        for family in FAMILIES:
            bundle = joblib.load(output_dir / "models" / f"{family}.joblib")
            require(bundle["train_ids"] == list(expression.index) and bundle["validation_ids"] == [], "Final bundle IDs")
            require(bundle["params"] == choices[family], "Final bundle selected parameters")
            require(bundle["model"].get_params() == {**protocol["fixed_parameters"][family], **choices[family]}, "Final fixed parameters")
            validate_preprocessor(bundle, expression, metadata.cohort)

    frame, pooled = pd.DataFrame(calculated_folds), {}
    model_comparison = pd.read_csv(output_dir / "model_comparison.csv").set_index("model")
    cohort_metrics = pd.read_csv(output_dir / "cohort_metrics.csv")
    for family in FAMILIES:
        subset = oof[oof.model.eq(family)]
        require(set(subset.sample_id) == set(expression.index), "Incomplete OOF coverage")
        calculated = independent_metrics(subset.true_label, subset.predicted_label, subset.probability_asd)
        pooled[family] = calculated
        compare_metrics(calculated, summaries[family]["pooled_oof"], f"{family}/pooled")
        for measure in MEASURES:
            values = frame[frame.model.eq(family)][measure]
            for statistic, value in (("mean", values.mean()), ("sd", values.std(ddof=0))):
                close(value, summaries[family][f"{statistic}_{measure}"], f"{family}/{statistic}/{measure}")
                close(value, model_comparison.loc[family, f"{statistic}_{measure}"], "Comparison CSV/JSON mismatch")
        intervals = bootstrap_intervals(subset)
        for measure, bounds in summaries[family]["descriptive_95pct_group_bootstrap"].items():
            close(intervals[measure].quantile(.025), bounds["low"], "Bootstrap lower bound")
            close(intervals[measure].quantile(.975), bounds["high"], "Bootstrap upper bound")
        for cohort, records in subset.groupby("cohort"):
            expected = cohort_metrics[cohort_metrics.model.eq(family) & cohort_metrics.cohort.eq(cohort)]
            require(len(expected) == 1, "Cohort metric rows")
            compare_metrics(independent_metrics(records.true_label, records.predicted_label, records.probability_asd),
                            expected.iloc[0], f"{family}/{cohort}")
    require(preserved_hashes == {str(path.relative_to(output_dir)): sha256(path) for path in files}, "Run artefact bytes changed during verification")
    require(input_hashes == {name: sha256(input_dir / name) for name in TRAINING_FILES}, "Input bytes changed during verification")
    report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only reconstruction from approved training inputs; no fitting or test-data access",
              "outer_models_reconstructed": 15, "final_model_bundles_checked": 3,
              "predictions_reconstructed": 783, "unique_training_records": 261,
              "unique_candidate_settings": 54, "candidate_summary_rows_checked": candidate_rows,
              "candidate_fold_rows_checked": fit_rows, "bootstrap_repetitions_per_model": 1000,
              "maximum_probability_absolute_error": maximum_probability_error,
              "absolute_and_relative_tolerance": TOLERANCE, "csv_roundoff_selection_ties": roundoff_ties,
              "accessed_training_files": sorted(opened), "input_sha256": input_hashes,
              "source_snapshot_sha256": source_checks, "verified_output_sha256": preserved_hashes,
              "verification_script_sha256": sha256(Path(__file__)), "environment": versions,
              "fold_checks": checks, "pooled_oof_recomputed": pooled,
              "limitations": ["Checks stored inner-search ledgers without refitting all 1350 candidate models.",
                              "Does not establish clinical validity, causal interpretation or independent validation.",
                              "Group-bootstrap intervals are conditional on saved predictions, not retraining uncertainty."]}
    target = output_dir / "verification.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "predictions_reconstructed": 783,
                      "maximum_probability_absolute_error": maximum_probability_error}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=STUDY / "results/model_comparison")
    args = parser.parse_args()
    verify(args.input_dir, args.output_dir)
