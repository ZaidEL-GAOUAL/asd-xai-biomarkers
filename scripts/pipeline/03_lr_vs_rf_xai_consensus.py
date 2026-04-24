"""LR vs RF XAI consensus on ComBat-corrected data with covariate residualization inside CV.

Deliverables:
  - per-method, per-model gene rankings
  - cross-model consensus table (LR top-K ∩ RF top-K)
  - figure
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
import shap

from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residualize import CovariateResidualizer
import GEOparse

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RANDOM_STATE = 42
TOP_K = 20

# ─────────────────────── Load ComBat-corrected matrix + covariates ───────────────────────
print("Loading ComBat-corrected matrix...")
combat_df = pd.read_parquet(OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet")
meta_cols = ["diagnosis", "platform_id"]
X_all = combat_df.drop(columns=meta_cols)
meta = combat_df[meta_cols].copy()
print(f"X shape: {X_all.shape}")

# Build covariate frame (sex + age) from GEO
print("Fetching covariates...")
gse = GEOparse.get_GEO("GSE18123", destdir=str(Path(__file__).resolve().parent.parent.parent / "data" / "raw"), silent=True)
sex, age = {}, {}
for n, gsm in gse.gsms.items():
    for item in gsm.metadata.get("characteristics_ch1", []):
        if ":" not in item: continue
        k, v = item.split(":", 1)
        k = k.strip().lower(); v = v.strip()
        if k == "gender": sex[n] = v
        elif k == "age at blood drawing (months)":
            try: age[n] = float(v)
            except: pass
cov = pd.DataFrame({"sex": pd.Series(sex), "age_months": pd.Series(age)})
cov = cov.loc[X_all.index]
print(f"Covariates: {cov.shape}, complete rows: {cov.notna().all(axis=1).sum()}")

y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)

# ─────────────────────── Signature stability on ComBat-corrected data ───────────────────────
print("\n=== Stability analysis on ComBat data (binary target, k=100) ===")
cv_stab = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
selector = SelectKBest(score_func=f_classif, k=100)
sel_counts = Counter()
total = 0
for train_idx, _ in cv_stab.split(X_all, y_bin):
    X_train_df = X_all.iloc[train_idx]
    y_train_s = y_bin.iloc[train_idx]

    # Residualize inside fold: fit on training indices only (full cov frame OK since fit uses X.index)
    res = CovariateResidualizer(cov)
    X_train_res = res.fit_transform(X_train_df)

    selector.fit(X_train_res, y_train_s)
    sel_counts.update(X_all.columns[selector.get_support()].tolist())
    total += 1

stability = pd.DataFrame([
    {"gene": g, "count": c, "freq": c/total} for g, c in sel_counts.items()
]).sort_values("count", ascending=False).reset_index(drop=True)
stability.to_csv(OUT_DIR / "tables" / "signature_stability_combat.csv", index=False)
print(f"Genes stable in 100% of splits: {(stability['freq']==1.0).sum()}")
print(f"Genes stable in >=80%: {(stability['freq']>=0.8).sum()}")
print(f"Top 15:")
print(stability.head(15).to_string(index=False))

# Take top 100
top100 = stability.head(100)["gene"].tolist()

# ─────────────────────── Shared CV-fold XAI helper ───────────────────────
def fit_transform_fold(X_train_df, X_test_df, cov_frame):
    """Residualize (fit on train), standardize (fit on train), apply to train/test.

    Full covariate frame is passed; fit only uses training sample indices via X_train_df.index.
    """
    res = CovariateResidualizer(cov_frame)  # full covariate frame
    X_train_res = res.fit_transform(X_train_df)  # beta learned from train only
    X_test_res = res.transform(X_test_df)         # apply beta to test

    scaler = StandardScaler().fit(X_train_res.values)
    X_train_sc = pd.DataFrame(scaler.transform(X_train_res.values),
                              index=X_train_res.index, columns=X_train_res.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test_res.values),
                             index=X_test_res.index, columns=X_test_res.columns)
    return X_train_sc, X_test_sc

def lr_xai_methods(model, X_test, y_test, feat_names):
    """Return a dict {method: score_array_of_len_n_features} for LR."""
    scores = {}
    # PFI on held-out test fold
    pfi = permutation_importance(model, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="balanced_accuracy",
                                  n_jobs=-1)
    scores["PFI"] = pfi.importances_mean
    # SHAP (linear explainer)
    explainer = shap.LinearExplainer(model, X_test)
    sv = explainer.shap_values(X_test)
    if sv.ndim == 3:
        sv_abs = np.abs(sv).mean(axis=(0,2))
    else:
        sv_abs = np.abs(sv).mean(axis=0)
    scores["SHAP"] = sv_abs
    # Model coefficients
    coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    scores["Coef"] = coef
    return scores

def rf_xai_methods(model, X_test, y_test, feat_names):
    """Return a dict {method: score_array_of_len_n_features} for RF."""
    scores = {}
    # PFI
    pfi = permutation_importance(model, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="balanced_accuracy",
                                  n_jobs=-1)
    scores["PFI"] = pfi.importances_mean
    # TreeSHAP
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv, list):
        sv_abs = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
    elif sv.ndim == 3:
        sv_abs = np.abs(sv).mean(axis=(0,2))
    else:
        sv_abs = np.abs(sv).mean(axis=0)
    scores["SHAP"] = sv_abs
    # RF feature importance
    scores["Impurity"] = model.feature_importances_
    return scores

# ─────────────────────── K-fold XAI aggregation ───────────────────────
def aggregate_kfold_xai(X_genes, y, cov_frame, model_factory, xai_fn, n_splits=5):
    """Run model_factory across stratified folds, collect XAI rankings on each held-out fold, aggregate."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    # Running sums of per-method scores across folds (arithmetic mean of absolute importances)
    agg = {}  # method -> np.array of length n_features
    n_features = X_genes.shape[1]
    fold_test_accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_genes, y), start=1):
        X_tr_df = X_genes.iloc[tr_idx]
        X_te_df = X_genes.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        X_tr_sc, X_te_sc = fit_transform_fold(X_tr_df, X_te_df, cov_frame)

        # SMOTE on training only (binary + few samples: only if minority < majority)
        if len(np.unique(y_tr)) > 1:
            min_count = min(Counter(y_tr).values())
            if min_count >= 6:  # default k_neighbors=5
                sm = SMOTE(random_state=RANDOM_STATE)
                X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
            else:
                X_tr_res, y_tr_res = X_tr_sc, y_tr
        else:
            X_tr_res, y_tr_res = X_tr_sc, y_tr

        model = model_factory()
        model.fit(X_tr_res, y_tr_res)
        fold_test_accs.append(balanced_accuracy_score(y_te, model.predict(X_te_sc)))

        fold_scores = xai_fn(model, X_te_sc, y_te, list(X_genes.columns))
        for meth, arr in fold_scores.items():
            if meth not in agg:
                agg[meth] = np.zeros(n_features)
            agg[meth] += arr / n_splits

    return agg, fold_test_accs

X_top = X_all[top100]
y = y_bin

# ── LR model family ──
print("\n=== LR pipeline (residualize + scale + SMOTE + LR) ===")
lr_factory = lambda: LogisticRegression(solver="saga", max_iter=8000,
                                         random_state=RANDOM_STATE, class_weight="balanced")
lr_scores, lr_accs = aggregate_kfold_xai(X_top, y, cov, lr_factory, lr_xai_methods)
print(f"LR fold balanced_accuracy: {np.mean(lr_accs):.3f} +/- {np.std(lr_accs):.3f}")

# ── RF model family ──
print("\n=== RF pipeline (residualize + scale + SMOTE + RF) ===")
rf_factory = lambda: RandomForestClassifier(n_estimators=500, max_depth=None,
                                              random_state=RANDOM_STATE,
                                              class_weight="balanced", n_jobs=-1)
rf_scores, rf_accs = aggregate_kfold_xai(X_top, y, cov, rf_factory, rf_xai_methods)
print(f"RF fold balanced_accuracy: {np.mean(rf_accs):.3f} +/- {np.std(rf_accs):.3f}")

# ─────────────────────── Consensus table ───────────────────────
def topk_set(score_arr, feat_names, k=TOP_K):
    idx = np.argsort(-score_arr)[:k]
    return set(np.asarray(feat_names)[idx])

feat_names = top100
lr_tops = {m: topk_set(s, feat_names) for m, s in lr_scores.items()}
rf_tops = {m: topk_set(s, feat_names) for m, s in rf_scores.items()}

# Cross-model consensus:
# a) Gene in top-K of at least one LR method AND at least one RF method
lr_union = set().union(*lr_tops.values())
rf_union = set().union(*rf_tops.values())
cross_union = lr_union & rf_union

# b) Gene in top-K of all LR methods AND top-K of all RF methods (strict)
lr_inter = set.intersection(*lr_tops.values())
rf_inter = set.intersection(*rf_tops.values())
cross_strict = lr_inter & rf_inter

# Consensus score per gene: count (out of 6 methods) how many include the gene in top-K
all_methods = {**{f"LR_{k}": v for k, v in lr_tops.items()}, **{f"RF_{k}": v for k, v in rf_tops.items()}}
gene_counts = Counter()
for top in all_methods.values():
    gene_counts.update(top)

# Build table
union_genes = sorted(set().union(*all_methods.values()))
rows = []
for g in union_genes:
    row = {"gene": g}
    row["total_methods_in_top20_6"] = gene_counts[g]
    row["LR_methods_in_top20_3"] = sum(1 for s in lr_tops.values() if g in s)
    row["RF_methods_in_top20_3"] = sum(1 for s in rf_tops.values() if g in s)
    for m, s in lr_tops.items():
        row[f"LR_{m}"] = int(g in s)
    for m, s in rf_tops.items():
        row[f"RF_{m}"] = int(g in s)
    rows.append(row)
consensus_df = pd.DataFrame(rows).sort_values(
    ["total_methods_in_top20_6", "LR_methods_in_top20_3", "RF_methods_in_top20_3", "gene"],
    ascending=[False, False, False, True]
).reset_index(drop=True)
consensus_df.to_csv(OUT_DIR / "tables" / "lr_vs_rf_consensus.csv", index=False)

print(f"\nGenes in top-20 of ALL 3 LR methods AND ALL 3 RF methods: {len(cross_strict)}")
print(f"  -> {sorted(cross_strict)}")
print(f"\nGenes in top-20 of AT LEAST ONE LR method AND AT LEAST ONE RF method: {len(cross_union)}")
print(f"\nFull consensus table (top 30):")
print(consensus_df.head(30).to_string(index=False))

# ─────────────────────── Figure: LR vs RF consensus ───────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
top_genes = consensus_df.head(20)
genes_rev = top_genes["gene"].tolist()[::-1]
lr_vals = top_genes["LR_methods_in_top20_3"].values[::-1]
rf_vals = top_genes["RF_methods_in_top20_3"].values[::-1]
y_pos = np.arange(len(genes_rev))
ax.barh(y_pos - 0.2, lr_vals, height=0.4, color="#2b7bba", label="LR (PFI + SHAP + Coef)")
ax.barh(y_pos + 0.2, rf_vals, height=0.4, color="#d95f02", label="RF (PFI + TreeSHAP + Impurity)")
ax.set_yticks(y_pos)
ax.set_yticklabels(genes_rev, fontsize=9)
ax.set_xlabel("Number of XAI methods (out of 3 per model) placing gene in top-20", fontsize=10)
ax.set_xticks([0, 1, 2, 3])
ax.axvline(3, linestyle="--", alpha=0.25, color="black")
ax.legend(loc="lower right", frameon=True)
ax.set_title("LR vs RF XAI consensus on ComBat-corrected, covariate-residualized data",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "lr_vs_rf_consensus.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: {fig_path}")
