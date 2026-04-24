"""Inside-fold residualization for sex + age covariates.

Follows the same API contract as sklearn transformers: fit on training data
then transform train and test. Coefficients are learned from training samples
only, then applied to both train and test residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """Regress out covariates per gene inside a CV fold.

    For each gene g, fit g_i = beta0 + beta1 * sex_i + beta2 * age_i on training
    samples, then return residuals (g_i - predicted g_i) for both train and test.

    Parameters
    ----------
    covariates_df : pd.DataFrame indexed by sample id, with columns for the
        covariates to regress out. Must cover all samples referenced in fit/transform.
    """

    def __init__(self, covariates_df: pd.DataFrame):
        self.covariates_df = covariates_df

    def _design_matrix(self, sample_ids):
        C = self.covariates_df.loc[sample_ids].copy()
        # One-hot encode any non-numeric columns, drop first to avoid collinearity
        C = pd.get_dummies(C, drop_first=True).astype(float)
        # Add intercept
        C.insert(0, "_intercept", 1.0)
        return C.values  # (n_samples, k_covariates + 1)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit per-gene OLS coefficients on the training X."""
        D = self._design_matrix(X.index)
        # Solve (D^T D) beta = D^T X for all genes in one lstsq
        # Using lstsq for stability with near-collinear covariates
        self.beta_, *_ = np.linalg.lstsq(D, X.values, rcond=None)
        # beta_ shape: (k_covariates+1, n_genes)
        self.feature_names_in_ = X.columns.values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        D = self._design_matrix(X.index)
        predicted = D @ self.beta_                 # (n_samples, n_genes)
        residuals = X.values - predicted
        return pd.DataFrame(residuals, index=X.index, columns=X.columns)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
