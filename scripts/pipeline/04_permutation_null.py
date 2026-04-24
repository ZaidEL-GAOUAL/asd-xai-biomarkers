"""Permutation null test on ComBat + residualized pipeline.

Part 1 (Stability null)
    500 permutations. For each, shuffle y_bin and run RepeatedStratifiedKFold(5,5)
    with SelectKBest(k=100) in each fit (residualization applied inside fold).
    Records per-permutation: max selection count, 95th percentile of counts, and the
    total null selection count per gene across all permutations.

Part 2 (Consensus null)
    50 permutations of the cross-model XAI. For each, shuffle y_bin, restrict to
    the observed top-100 stability genes, run 5-fold CV with LR (PFI/SHAP/Coef)
    and RF (PFI/TreeSHAP/Impurity), count genes in top-20 of all 6 methods.
    Records null distribution of strict-consensus size plus 5/6 and 4/6 sizes.

All residualization, scaling, selection, SMOTE, model fitting is inside-fold.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residualize import CovariateResidualizer
import GEOparse

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RANDOM_STATE = 42
N_PERM_STAB = 500
N_PERM_XAI = 50
K_FEATURES = 100
TOP_K_XAI = 20

import warnings
warnings.filterwarnings("ignore")

print("Loading ComBat-corrected matrix...")
combat_df = pd.read_parquet(OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet")
X_all = combat_df.drop(columns=["diagnosis", "platform_id"])
meta = combat_df[["diagnosis", "platform_id"]].copy()

gse = GEOparse.get_GEO("GSE18123", destdir=str(Path(__file__).resolve().parent.parent.parent / "data" / "raw"), silent=True)
sex, age = {}, {}
for n, gsm in gse.gsms.items():
    for item in gsm.metadata.get("characteristics_ch1", []):
        if ":" not in item: continue
        k, v = item.split(":", 1); k = k.strip().lower(); v = v.strip()
        if k == "gender": sex[n] = v
        elif k == "age at blood drawing (months)":
            try: age[n] = float(v)
            except: pass
cov = pd.DataFrame({"sex": pd.Series(sex), "age_months": pd.Series(age)}).loc[X_all.index]

y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)

# ─────────────────── PART 1: Stability null ───────────────────
print(f"\n================ PART 1: Stability null ({N_PERM_STAB} perms) ================")

def one_stability_run(X, y, n_splits=5, n_repeats=5, k=K_FEATURES, seed=RANDOM_STATE):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    counts = Counter()
    for tr, _ in cv.split(X, y):
        X_tr = X.iloc[tr]; y_tr = y.iloc[tr]
        res = CovariateResidualizer(cov)
        X_tr_r = res.fit_transform(X_tr)
        sel = SelectKBest(f_classif, k=k).fit(X_tr_r.values, y_tr)
        counts.update(X.columns[sel.get_support()].tolist())
    return counts

print("Observed stability run...")
t0 = time.time()
obs_counts = one_stability_run(X_all, y_bin)
obs_arr = np.array([obs_counts.get(g, 0) for g in X_all.columns])
print(f"  observed max count: {obs_arr.max()} / 25")
print(f"  100%-stable genes: {(obs_arr == 25).sum()}")
print(f"  >=80%-stable genes: {(obs_arr >= 20).sum()}")
print(f"  elapsed: {time.time()-t0:.1f}s")

print(f"Running {N_PERM_STAB} permutations...")
rng = np.random.default_rng(RANDOM_STATE)
null_max = np.zeros(N_PERM_STAB, dtype=int)
null_p95 = np.zeros(N_PERM_STAB, dtype=float)
gene_null_counts = np.zeros(len(X_all.columns), dtype=np.int64)
t0 = time.time()
for p in range(N_PERM_STAB):
    y_perm = pd.Series(rng.permutation(y_bin.values), index=y_bin.index)
    c = one_stability_run(X_all, y_perm, seed=RANDOM_STATE + p + 1)
    arr = np.array([c.get(g, 0) for g in X_all.columns])
    null_max[p] = arr.max()
    null_p95[p] = np.percentile(arr, 95)
    gene_null_counts += arr
    if (p+1) % 25 == 0:
        print(f"  [{p+1}/{N_PERM_STAB}] elapsed={time.time()-t0:.0f}s, null_max_median={np.median(null_max[:p+1]):.1f}")

print(f"\n--- Stability null summary ---")
print(f"Null max selection count: median={np.median(null_max):.1f}, 95th pct={np.percentile(null_max, 95):.1f}, max={null_max.max()}")
print(f"Observed max: {obs_arr.max()}")

candidates = ["CES1", "EGR1", "SPG20", "TIGD7",
              "LOC728554", "MALAT1", "UBXN2A", "ZNF117"]
print(f"\nCandidate gene selection counts and permutation p-values:")
for g in candidates:
    if g in X_all.columns:
        obs_c = obs_counts.get(g, 0)
        pval = (null_max >= obs_c).mean()
        print(f"  {g:12s}  observed={obs_c}/25   p_max={pval:.4f}")

pd.DataFrame({
    "perm": np.arange(N_PERM_STAB),
    "null_max_sel_count": null_max,
    "null_p95_sel_count": null_p95,
}).to_csv(OUT_DIR / "tables" / "permutation_null_stability.csv", index=False)

gene_null_df = pd.DataFrame({
    "gene": X_all.columns,
    "observed_count": obs_arr,
    "null_total_count": gene_null_counts,
    "null_mean_count_per_perm": gene_null_counts / N_PERM_STAB,
}).sort_values("observed_count", ascending=False)
gene_null_df.to_csv(OUT_DIR / "tables" / "permutation_null_per_gene.csv", index=False)

# ─────────────────── PART 2: Consensus null ───────────────────
print(f"\n================ PART 2: Consensus null ({N_PERM_XAI} perms) ================")
obs_stab = pd.DataFrame({"gene": X_all.columns, "count": obs_arr})
obs_stab = obs_stab.sort_values("count", ascending=False)
top100_obs = obs_stab.head(100)["gene"].tolist()
X_top = X_all[top100_obs]

def fit_transform_fold(X_train_df, X_test_df):
    res = CovariateResidualizer(cov)
    X_tr_r = res.fit_transform(X_train_df)
    X_te_r = res.transform(X_test_df)
    scl = StandardScaler().fit(X_tr_r.values)
    X_tr_sc = pd.DataFrame(scl.transform(X_tr_r.values), index=X_tr_r.index, columns=X_tr_r.columns)
    X_te_sc = pd.DataFrame(scl.transform(X_te_r.values), index=X_te_r.index, columns=X_te_r.columns)
    return X_tr_sc, X_te_sc

def aggregate_xai_methods(X_features, y, n_splits=5, rf_trees=200):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    methods = ["LR_PFI", "LR_SHAP", "LR_Coef", "RF_PFI", "RF_SHAP", "RF_Impurity"]
    agg = {m: np.zeros(X_features.shape[1]) for m in methods}
    for tr, te in skf.split(X_features, y):
        X_tr_df = X_features.iloc[tr]; X_te_df = X_features.iloc[te]
        y_tr = y.iloc[tr]; y_te = y.iloc[te]
        X_tr, X_te = fit_transform_fold(X_tr_df, X_te_df)
        if min(Counter(y_tr).values()) >= 6:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)
        else:
            X_tr_r, y_tr_r = X_tr, y_tr

        lr = LogisticRegression(solver="saga", max_iter=8000, random_state=RANDOM_STATE,
                                 class_weight="balanced").fit(X_tr_r, y_tr_r)
        pfi_lr = permutation_importance(lr, X_te, y_te, n_repeats=10,
                                         random_state=RANDOM_STATE, scoring="balanced_accuracy",
                                         n_jobs=-1).importances_mean
        expl_lr = shap.LinearExplainer(lr, X_te)
        sv_lr = expl_lr.shap_values(X_te)
        sv_lr_abs = np.abs(sv_lr).mean(axis=(0,2)) if sv_lr.ndim == 3 else np.abs(sv_lr).mean(axis=0)
        coef_lr = np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_).ravel()

        rf = RandomForestClassifier(n_estimators=rf_trees, random_state=RANDOM_STATE,
                                      class_weight="balanced", n_jobs=-1).fit(X_tr_r, y_tr_r)
        pfi_rf = permutation_importance(rf, X_te, y_te, n_repeats=10,
                                         random_state=RANDOM_STATE, scoring="balanced_accuracy",
                                         n_jobs=-1).importances_mean
        expl_rf = shap.TreeExplainer(rf)
        sv_rf = expl_rf.shap_values(X_te, check_additivity=False)
        if isinstance(sv_rf, list):
            sv_rf_abs = np.mean([np.abs(s) for s in sv_rf], axis=0).mean(axis=0)
        elif sv_rf.ndim == 3:
            sv_rf_abs = np.abs(sv_rf).mean(axis=(0,2))
        else:
            sv_rf_abs = np.abs(sv_rf).mean(axis=0)
        imp_rf = rf.feature_importances_

        agg["LR_PFI"] += pfi_lr / n_splits
        agg["LR_SHAP"] += sv_lr_abs / n_splits
        agg["LR_Coef"] += coef_lr / n_splits
        agg["RF_PFI"] += pfi_rf / n_splits
        agg["RF_SHAP"] += sv_rf_abs / n_splits
        agg["RF_Impurity"] += imp_rf / n_splits
    return agg

def consensus_counts(agg, feat_names, top_k=TOP_K_XAI):
    top_sets = [set(np.asarray(feat_names)[np.argsort(-arr)[:top_k]]) for arr in agg.values()]
    c = Counter()
    for s in top_sets:
        c.update(s)
    return {
        "6of6": sum(1 for g, v in c.items() if v == 6),
        "5of6": sum(1 for g, v in c.items() if v >= 5),
        "4of6": sum(1 for g, v in c.items() if v >= 4),
    }

print("Observed cross-model XAI on real labels...")
t0 = time.time()
obs_agg = aggregate_xai_methods(X_top, y_bin, rf_trees=500)
obs_cc = consensus_counts(obs_agg, top100_obs)
print(f"  observed: {obs_cc}  (elapsed: {time.time()-t0:.0f}s)")

print(f"\nRunning {N_PERM_XAI} permutations of cross-model XAI...")
rng2 = np.random.default_rng(RANDOM_STATE + 10000)
null_6of6 = np.zeros(N_PERM_XAI, dtype=int)
null_5of6 = np.zeros(N_PERM_XAI, dtype=int)
null_4of6 = np.zeros(N_PERM_XAI, dtype=int)
t0 = time.time()
for p in range(N_PERM_XAI):
    y_perm = pd.Series(rng2.permutation(y_bin.values), index=y_bin.index)
    agg = aggregate_xai_methods(X_top, y_perm, rf_trees=200)
    cc = consensus_counts(agg, top100_obs)
    null_6of6[p] = cc["6of6"]; null_5of6[p] = cc["5of6"]; null_4of6[p] = cc["4of6"]
    if (p+1) % 5 == 0:
        elapsed = time.time()-t0
        print(f"  [{p+1}/{N_PERM_XAI}]  6/6 so far: median={np.median(null_6of6[:p+1]):.1f}, "
              f"95th={np.percentile(null_6of6[:p+1], 95):.1f}  "
              f"elapsed={elapsed:.0f}s  eta={elapsed/(p+1)*(N_PERM_XAI-p-1):.0f}s")

print(f"\n--- Consensus null summary ---")
for label, null_arr, obs_val in [
    ("6/6 strict", null_6of6, obs_cc["6of6"]),
    ("5/6",        null_5of6, obs_cc["5of6"]),
    ("4/6",        null_4of6, obs_cc["4of6"]),
]:
    p95 = np.percentile(null_arr, 95)
    pval = (null_arr >= obs_val).mean()
    print(f"  {label:12s}  null median={np.median(null_arr):.1f}, 95th={p95:.1f}, max={null_arr.max()}   "
          f"observed={obs_val}   p={pval:.4f}")

pd.DataFrame({
    "perm": np.arange(N_PERM_XAI),
    "null_6of6": null_6of6,
    "null_5of6": null_5of6,
    "null_4of6": null_4of6,
}).to_csv(OUT_DIR / "tables" / "permutation_null_consensus.csv", index=False)

# Figures
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
bins = max(5, int(null_max.max()) - int(null_max.min()) + 1)
ax.hist(null_max, bins=bins, color="#1f77b4", alpha=0.7, edgecolor="white")
ax.axvline(obs_arr.max(), color="#d62728", linestyle="--", linewidth=2,
           label=f"Observed max = {obs_arr.max()}")
ax.axvline(np.percentile(null_max, 95), color="#2ca02c", linestyle=":", linewidth=2,
           label=f"Null 95th = {np.percentile(null_max, 95):.1f}")
ax.set_xlabel("Max selection count across 25 CV fits")
ax.set_ylabel("Permutations")
ax.set_title(f"Stability null (n={N_PERM_STAB})")
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
bins = max(5, null_6of6.max() - null_6of6.min() + 1)
ax.hist(null_6of6, bins=bins, color="#ff7f0e", alpha=0.7, edgecolor="white")
ax.axvline(obs_cc["6of6"], color="#d62728", linestyle="--", linewidth=2,
           label=f"Observed 6/6 = {obs_cc['6of6']}")
ax.axvline(np.percentile(null_6of6, 95), color="#2ca02c", linestyle=":", linewidth=2,
           label=f"Null 95th = {np.percentile(null_6of6, 95):.1f}")
ax.set_xlabel("Genes in top-20 of all 6 XAI methods")
ax.set_ylabel("Permutations")
ax.set_title(f"Cross-model consensus null (n={N_PERM_XAI})")
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.suptitle("Permutation null tests (ComBat + residualized pipeline)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "permutation_null.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure: {fig_path}")

print("\nDone.")
