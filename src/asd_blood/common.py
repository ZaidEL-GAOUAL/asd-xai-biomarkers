"""Data validation, fold-fitted preprocessing, and evaluation utilities."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

STUDY = Path(__file__).resolve().parents[2]
PROJECT = STUDY
DEFAULT_INPUT = PROJECT / "phase2_outputs"
TRAINING_FILES = ("X_train_log2.csv.gz", "y_train.csv", "metadata_train.csv", "gene_dictionary.csv")
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def training_data_guard(input_dir):
    """Refuse test-partition access and input writes in this analysis process."""
    input_dir = Path(input_dir).resolve()
    opened = set()
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(input_dir):
            flags = args[2] or 0
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if path.parent != input_dir or path.name not in TRAINING_FILES or writing:
                raise PermissionError(f"Not an approved training input: {path}")
            opened.add(path.name)
    sys.addaudithook(guard)
    return opened


def load_training(input_dir=DEFAULT_INPUT):
    input_dir = Path(input_dir)
    genes = pd.read_csv(input_dir / TRAINING_FILES[0], index_col="sample_id")
    y = pd.read_csv(input_dir / TRAINING_FILES[1], index_col="sample_id").target
    meta = pd.read_csv(input_dir / TRAINING_FILES[2], index_col="sample_id")
    if not genes.index.is_unique or not genes.columns.is_unique:
        raise ValueError("Duplicate sample or feature identifiers")
    if not genes.index.equals(y.index) or not genes.index.equals(meta.index):
        raise ValueError("Expression, metadata, and label order differ")
    if not meta.split.eq("train").all() or not y.equals(meta.target):
        raise ValueError("Partition or label mismatch")
    if not np.isfinite(genes.to_numpy()).all() or set(y.unique()) != {0, 1}:
        raise ValueError("Invalid expression or labels")
    if (len(genes), genes.shape[1]) != (261, 17263) or y.value_counts().to_dict() != {1: 163, 0: 98}:
        raise ValueError("Prepared training dataset differs from the study protocol")
    validate_folds(meta, expected_folds=5)
    return genes.assign(cohort=meta.cohort), y, meta


def validate_folds(meta, expected_folds):
    folds = meta.training_cv_fold
    if folds.isna().any() or folds.nunique() != expected_folds:
        raise ValueError("Incorrect number of preserved folds")
    if meta.groupby("relatedness_group").training_cv_fold.nunique().max() != 1:
        raise ValueError("A relatedness group crosses folds")
    splits = list(PredefinedSplit(folds.astype(int).to_numpy()).split())
    for tr, va in splits:
        left, right = meta.iloc[tr], meta.iloc[va]
        if set(left.relatedness_group) & set(right.relatedness_group):
            raise ValueError("Relatedness leakage")
        for part in (left, right):
            if set(part.cohort) != set(meta.cohort) or part.groupby("cohort").target.nunique().min() != 2:
                raise ValueError("A cohort/class is absent from a fold")
    return splits


class WholeBloodPreprocessor(TransformerMixin, BaseEstimator):
    """Known-cohort mean adjustment and standard scaling fitted on training only.

    Equivalent to the preserved preparation implementation. This removes cohort
    mean offsets, not all technical or biological confounding. Unseen cohorts are
    unsupported. Cohort identifiers are not classifier features.
    """
    def __init__(self, center_cohorts=True):
        self.center_cohorts = center_cohorts

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or "cohort" not in X:
            raise ValueError("Expected gene columns and cohort metadata")
        self.input_genes_ = X.columns.drop("cohort").to_numpy()
        values = X.loc[:, self.input_genes_].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Non-finite expression")
        self.genes_ = values.columns[values.var(ddof=0) > 0].to_numpy()
        values = values.loc[:, self.genes_]
        self.global_mean_ = values.mean()
        self.cohort_means_ = values.groupby(X.cohort).mean()
        self.scaler_ = StandardScaler().fit(self._center(values, X.cohort))
        self.n_features_in_ = X.shape[1]
        return self

    def _center(self, values, cohorts):
        unknown = set(cohorts) - set(self.cohort_means_.index)
        if unknown:
            raise ValueError(f"Unseen cohorts: {sorted(unknown)}")
        if not self.center_cohorts:
            return values.copy()
        offsets = self.cohort_means_.loc[cohorts].to_numpy() - self.global_mean_.to_numpy()
        return values - offsets

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        if list(X.columns.drop("cohort")) != list(self.input_genes_):
            raise ValueError("Gene columns or ordering differ from training")
        values = X.loc[:, self.genes_].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Non-finite expression")
        centered = self._center(values, X.cohort)
        return pd.DataFrame(self.scaler_.transform(centered), index=X.index, columns=self.genes_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "genes_")
        return self.genes_.copy()


def metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "roc_auc": roc_auc_score(y, probability),
            "sensitivity": recall_score(y, prediction, pos_label=1),
            "specificity": recall_score(y, prediction, pos_label=0),
            "precision": precision_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
