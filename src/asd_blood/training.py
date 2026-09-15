"""Bounded LR, Random Forest, and KNN tuning within preserved training folds."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .common import (DEFAULT_INPUT, METRICS, STUDY, TRAINING_FILES, WholeBloodPreprocessor,
                     load_training, metrics, sha256, training_data_guard, validate_folds, write_json)

SEED = 42
GRIDS = {
    "logistic_regression": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                            "class_weight": [None, "balanced"]},
    "random_forest": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5],
                      "class_weight": [None, "balanced"]},
    "knn": {"n_neighbors": [3, 5, 7, 11, 15, 21], "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]},
}


def make_model(name, params):
    if name == "logistic_regression":
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=1,
                                      random_state=SEED, **params)
    if name == "knn":
        return KNeighborsClassifier(algorithm="brute", n_jobs=1, **params)
    raise ValueError(name)


def prepared_splits(X, y, meta, n_folds):
    prepared = []
    for tr, va in validate_folds(meta, expected_folds=n_folds):
        prepare = WholeBloodPreprocessor().fit(X.iloc[tr])
        prepared.append((prepare.transform(X.iloc[tr]).to_numpy(), y.iloc[tr].to_numpy(),
                         prepare.transform(X.iloc[va]).to_numpy(), y.iloc[va].to_numpy()))
    return prepared


def evaluate_candidate(name, params, index, pairs):
    rows = []
    for fold, (Xtr, ytr, Xva, yva) in enumerate(pairs):
        model = make_model(name, params).fit(Xtr, ytr)
        probability = model.predict_proba(Xva)[:, 1]
        rows.append({"model": name, "candidate": index, "fold": fold,
                     "parameters": json.dumps(params, sort_keys=True),
                     **metrics(yva, model.predict(Xva), probability)})
    return rows


def select_candidate(rows):
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "candidate", "parameters"], sort=False)[list(METRICS)].mean().reset_index()
    summary = summary.sort_values(["balanced_accuracy", "roc_auc", "candidate"],
                                  ascending=[False, False, True])
    selected = summary.iloc[0]
    summary["selected"] = summary.candidate.eq(selected.candidate)
    return json.loads(selected.parameters), summary


def search(pairs, out, stage, jobs):
    choices, totals = {}, []
    for name, grid in GRIDS.items():
        candidates = list(ParameterGrid(grid))
        groups = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(evaluate_candidate)(name, params, i, pairs) for i, params in enumerate(candidates))
        rows = [r for group in groups for r in group]
        params, summary = select_candidate(rows)
        pd.DataFrame(rows).to_csv(out / "searches" / f"{stage}_{name}_fold_scores.csv", index=False)
        summary.to_csv(out / "searches" / f"{stage}_{name}_candidates.csv", index=False)
        choices[name] = params
        totals.append({"stage": stage, "model": name, "parameters": params,
                       "selection_cv_balanced_accuracy": float(summary.iloc[0].balanced_accuracy),
                       "selection_cv_accuracy": float(summary.iloc[0].accuracy)})
        print(f"{stage}: {name}, best search balanced accuracy={summary.iloc[0].balanced_accuracy:.3f}", flush=True)
    return choices, totals


def fit_bundle(X, y, params, name, train_ids, validation_ids):
    prepare = WholeBloodPreprocessor().fit(X)
    Z = prepare.transform(X)
    model = make_model(name, params).fit(Z.to_numpy(), y.to_numpy())
    return {"preprocessor": prepare, "model": model, "feature_names": Z.columns.tolist(),
            "train_ids": list(train_ids), "validation_ids": list(validation_ids), "params": params}


def group_bootstrap(oof, repetitions=1000):
    """Descriptive intervals for fixed OOF predictions, not full retraining uncertainty."""
    rng = np.random.default_rng(SEED)
    groups = oof.relatedness_group.unique()
    indices = {g: np.flatnonzero(oof.relatedness_group.to_numpy() == g) for g in groups}
    values = []
    for _ in range(repetitions):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = oof.iloc[np.concatenate([indices[g] for g in selected])]
        if sample.true_label.nunique() == 2:
            values.append(metrics(sample.true_label, sample.predicted_label, sample.probability_asd))
    frame = pd.DataFrame(values)
    return {key: {"low": float(frame[key].quantile(.025)), "high": float(frame[key].quantile(.975))}
            for key in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity")}


def render_results(out, summary, choices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for ax, row in zip(axes, summary):
        m = row["pooled_oof"]
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=163)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "ASD"],
               yticklabels=["Control", "ASD"], xlabel="Predicted", ylabel="Observed",
               title=row["model"].replace("_", " ").title())
    fig.suptitle("Training-only nested out-of-fold predictions")
    fig.savefig(out / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    table = "\n".join(
        f"| {r['model']} | {r['mean_accuracy']:.1%} | {r['mean_balanced_accuracy']:.1%} | "
        f"{r['mean_roc_auc']:.3f} | {r['mean_sensitivity']:.1%} | {r['mean_specificity']:.1%} |"
        for r in summary)
    parameter_lines = "\n".join(f"- {name}: `{json.dumps(params, sort_keys=True)}`" for name, params in choices.items())
    best = max(summary, key=lambda r: r["mean_balanced_accuracy"])
    text = f"""# Classifier comparison

## Evaluation

These are exploratory nested cross-validation results on 261 training assay records.
The 66-record historical test partition was not accessed. No independent final
test estimate is produced by this experiment. The three original classifiers,
17,263 gene inputs, retained quality exclusions, and group-separated folds are preserved.

| Classifier | Accuracy | Balanced accuracy | ROC AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Values are means over five outer folds. Each outer model was tuned using the
remaining four folds. The highest mean balanced accuracy in this comparison is
{best['mean_balanced_accuracy']:.1%} ({best['model']}); this is not evidence of a
statistically significant difference or of clinical usefulness.

## Parameters for the full-training models

{parameter_lines}

Full-training parameters were selected with five-fold CV. Their search scores
are selection scores, not substitutes for the nested estimates above. The grid
is finite; no claim of globally optimal hyperparameters is made. L2 logistic
regression, 300-tree Random Forest, and brute-force KNN were used. Class weighting
changes training emphasis without generating samples. PCA, SMOTE, supervised
gene selection, and decision-threshold tuning were not used.

## Interpretation and limitations

ASD is the positive class. Sensitivity measures the fraction of ASD records
identified; specificity measures the fraction of controls correctly classified.
Balanced accuracy averages the two. Confusion matrices use predictions from
models that did not train or tune on those outer-validation records.

The nested procedure separates this grid's tuning from its assessment, but does
not undo earlier adaptive exploration of the same data. The relatedness groups
are conservative metadata groupings, not verified unrelated-patient counts.
All cohorts are represented in training and validation: this is not external
cohort validation. Mean adjustment does not eliminate all biological confounding.

`summary.json` includes relatedness-group bootstrap intervals for the fixed OOF
predictions. These describe sample variability conditional on the saved fits;
they do not include all uncertainty from model selection or retraining. Fold SDs
are descriptive because training sets overlap.

The primary explanation analysis remains logistic regression, selected before
this grid for the XAI study. RF and KNN provide supporting classifier comparisons.
All candidates and selected parameters are retained in `searches/`.
"""
    (out / "RESULTS.md").write_text(text)


def run_training(input_dir=DEFAULT_INPUT, output=STUDY / "results/model_comparison", jobs=2):
    input_dir, out = Path(input_dir).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Results already exist: {out}")
    if out.is_relative_to(input_dir):
        raise ValueError("Output must not overwrite prepared inputs")
    opened = training_data_guard(input_dir)
    hashes = {name: sha256(input_dir / name) for name in TRAINING_FILES}
    X, y, meta = load_training(input_dir)
    out.mkdir(parents=True)
    for name in ("searches", "outer_models", "models", "source"):
        (out / name).mkdir()
    for source in (Path(__file__), Path(__file__).with_name("common.py")):
        shutil.copy2(source, out / "source" / source.name)
    count = sum(len(ParameterGrid(grid)) for grid in GRIDS.values())
    protocol = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "training_records": len(X), "gene_features": X.shape[1] - 1, "seed": SEED,
                "input_sha256": hashes, "grids": GRIDS, "candidate_count": count,
                "candidate_fits": count * 25, "outer_and_final_refits": 18,
                "outer_folds": 5, "inner_folds": 4, "full_training_search_folds": 5,
                "selection": "mean inner balanced accuracy, then ROC AUC, then ParameterGrid order",
                "preprocessing": "cohort offsets and StandardScaler fitted separately in each training fold",
                "decision_rule": "classifier.predict, no threshold optimization",
                "fixed_parameters": {n: make_model(n, {}).get_params() for n in GRIDS},
                "smote": False, "pca": False, "synthetic_samples": False,
                "primary_xai_model": "logistic_regression",
                "test_policy": "No access; historical test already examined in prior research",
                "interpretation": "Exploratory after prior adaptive analysis, not a new independent validation",
                "environment": {p: importlib.metadata.version(p) for p in
                                ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "shap")}}
    write_json(out / "protocol.json", protocol)
    meta[["cohort", "target", "relatedness_group", "training_cv_fold"]].to_csv(out / "training_folds.csv")
    started = time.perf_counter()
    fold_rows, predictions, selection_rows = [], [], []
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(validate_folds(meta, 5)):
            Xtr, ytr, mtr = X.iloc[tr], y.iloc[tr], meta.iloc[tr]
            choices, selected = search(prepared_splits(Xtr, ytr, mtr, 4), out, f"outer_{fold}", jobs)
            selection_rows.extend(selected)
            for name, params in choices.items():
                bundle = fit_bundle(Xtr, ytr, params, name, Xtr.index, X.iloc[va].index)
                joblib.dump(bundle, out / "outer_models" / f"fold_{fold}_{name}.joblib", compress=3)
                Zva = bundle["preprocessor"].transform(X.iloc[va]).to_numpy()
                probability = bundle["model"].predict_proba(Zva)[:, 1]
                predicted = bundle["model"].predict(Zva)
                fold_rows.append({"model": name, "fold": fold, **metrics(y.iloc[va], predicted, probability)})
                for i, sample_id in enumerate(X.iloc[va].index):
                    predictions.append({"sample_id": sample_id, "model": name, "fold": fold,
                                        "relatedness_group": meta.loc[sample_id, "relatedness_group"],
                                        "cohort": meta.loc[sample_id, "cohort"],
                                        "true_label": int(y.loc[sample_id]), "predicted_label": int(predicted[i]),
                                        "probability_asd": float(probability[i])})
            print(f"Completed outer fold {fold + 1}/5", flush=True)
        choices, selected = search(prepared_splits(X, y, meta, 5), out, "full_training", jobs)
        selection_rows.extend(selected)
        for name, params in choices.items():
            joblib.dump(fit_bundle(X, y, params, name, X.index, []), out / "models" / f"{name}.joblib", compress=3)
    frame, oof = pd.DataFrame(fold_rows), pd.DataFrame(predictions)
    frame.to_csv(out / "outer_fold_metrics.csv", index=False)
    oof.to_csv(out / "out_of_fold_predictions.csv", index=False)
    summary = []
    for name in GRIDS:
        f, p = frame[frame.model.eq(name)], oof[oof.model.eq(name)]
        if len(f) != 5 or len(p) != 261 or p.sample_id.nunique() != 261:
            raise ValueError("Incomplete outer-fold results")
        row = {"model": name, **{f"mean_{key}": float(f[key].mean()) for key in METRICS},
               **{f"sd_{key}": float(f[key].std(ddof=0)) for key in METRICS},
               "pooled_oof": metrics(p.true_label, p.predicted_label, p.probability_asd),
               "descriptive_95pct_group_bootstrap": group_bootstrap(p)}
        summary.append(row)
    cohort_rows = [{"model": name, "cohort": cohort, **metrics(p.true_label, p.predicted_label, p.probability_asd)}
                   for (name, cohort), p in oof.groupby(["model", "cohort"])]
    pd.DataFrame(cohort_rows).to_csv(out / "cohort_metrics.csv", index=False)
    write_json(out / "summary.json", summary)
    write_json(out / "selected_parameters.json", choices)
    write_json(out / "selection_history.json", selection_rows)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in summary]).to_csv(
        out / "model_comparison.csv", index=False)
    if hashes != {name: sha256(input_dir / name) for name in TRAINING_FILES}:
        raise ValueError("Prepared inputs changed during analysis")
    render_results(out, summary, choices)
    protocol.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    accessed_training_files=sorted(opened), completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "protocol.json", protocol)
    print(json.dumps({r["model"]: r["mean_balanced_accuracy"] for r in summary}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=STUDY / "results/model_comparison")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run_training(args.input_dir, args.output, args.jobs)
