"""Multi-seed x multi-n_estimators robustness check for the consensus biomarker set.

For each combination of (seed in 10 values) x (n_estimators in {200, 500, 1000}):
  1. Recompute stability top-100 using the seed to drive RepeatedStratifiedKFold.
  2. Run 5-fold cross-model XAI (LR 3 methods + RF 3 methods at n_estimators).
  3. Record 6/6 and 5/6 consensus gene sets plus their sizes.

Outputs per-gene inclusion frequency, consensus-size distributions, figures.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import time, warnings
warnings.filterwarnings("ignore")
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
SEEDS = list(range(42, 52))                 # 10 seeds
TREES = [200, 500, 1000]                    # 3 settings
K_FEATURES = 100
TOP_K = 20
LIME_SAMPLES = 0                             # not used here

# Load
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

def stability_top100(seed):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    counts = Counter()
    for tr, _ in cv.split(X_all, y_bin):
        X_tr = X_all.iloc[tr]; y_tr = y_bin.iloc[tr]
        res = CovariateResidualizer(cov)
        X_tr_r = res.fit_transform(X_tr)
        sel = SelectKBest(f_classif, k=K_FEATURES).fit(X_tr_r.values, y_tr)
        counts.update(X_all.columns[sel.get_support()].tolist())
    df = pd.DataFrame([{"gene": g, "count": c} for g, c in counts.items()])
    df = df.sort_values(["count", "gene"], ascending=[False, True]).reset_index(drop=True)
    return df.head(100)["gene"].tolist()

def fit_transform_fold(X_tr, X_te):
    res = CovariateResidualizer(cov)
    X_tr_r = res.fit_transform(X_tr); X_te_r = res.transform(X_te)
    scl = StandardScaler().fit(X_tr_r.values)
    X_tr_sc = pd.DataFrame(scl.transform(X_tr_r.values), index=X_tr_r.index, columns=X_tr_r.columns)
    X_te_sc = pd.DataFrame(scl.transform(X_te_r.values), index=X_te_r.index, columns=X_te_r.columns)
    return X_tr_sc, X_te_sc

def run_xai(top100, seed, n_trees):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_top = X_all[top100]
    methods = ["LR_PFI", "LR_SHAP", "LR_Coef", "RF_PFI", "RF_SHAP", "RF_Impurity"]
    agg = {m: np.zeros(len(top100)) for m in methods}
    for tr, te in skf.split(X_top, y_bin):
        X_tr_df = X_top.iloc[tr]; X_te_df = X_top.iloc[te]
        y_tr = y_bin.iloc[tr]; y_te = y_bin.iloc[te]
        X_tr, X_te = fit_transform_fold(X_tr_df, X_te_df)
        if min(Counter(y_tr).values()) >= 6:
            sm = SMOTE(random_state=seed)
            X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)
        else:
            X_tr_r, y_tr_r = X_tr, y_tr
        lr = LogisticRegression(solver="saga", max_iter=8000, random_state=seed,
                                 class_weight="balanced").fit(X_tr_r, y_tr_r)
        agg["LR_PFI"] += permutation_importance(lr, X_te, y_te, n_repeats=10,
            random_state=seed, scoring="balanced_accuracy", n_jobs=-1).importances_mean / 5
        sv_lr = shap.LinearExplainer(lr, X_te).shap_values(X_te)
        agg["LR_SHAP"] += (np.abs(sv_lr).mean(axis=(0,2)) if sv_lr.ndim == 3 else np.abs(sv_lr).mean(axis=0)) / 5
        agg["LR_Coef"] += (np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_).ravel()) / 5

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed,
                                      class_weight="balanced", n_jobs=-1).fit(X_tr_r, y_tr_r)
        agg["RF_PFI"] += permutation_importance(rf, X_te, y_te, n_repeats=10,
            random_state=seed, scoring="balanced_accuracy", n_jobs=-1).importances_mean / 5
        sv_rf = shap.TreeExplainer(rf).shap_values(X_te, check_additivity=False)
        if isinstance(sv_rf, list):
            sv_rf_abs = np.mean([np.abs(s) for s in sv_rf], axis=0).mean(axis=0)
        elif sv_rf.ndim == 3:
            sv_rf_abs = np.abs(sv_rf).mean(axis=(0,2))
        else:
            sv_rf_abs = np.abs(sv_rf).mean(axis=0)
        agg["RF_SHAP"] += sv_rf_abs / 5
        agg["RF_Impurity"] += rf.feature_importances_ / 5

    top_sets = {m: set(np.asarray(top100)[np.argsort(-arr)[:TOP_K]]) for m, arr in agg.items()}
    return top_sets

# Run all
records = []            # rows: {seed, trees, 6of6_size, 5of6_size, 4of6_size, genes_6of6, genes_5of6}
gene_in_5of6 = Counter()
gene_in_6of6 = Counter()

t_start = time.time()
for seed in SEEDS:
    top100 = stability_top100(seed)
    for ntrees in TREES:
        t0 = time.time()
        top_sets = run_xai(top100, seed, ntrees)
        method_counts = Counter()
        for s in top_sets.values():
            method_counts.update(s)
        genes_6 = [g for g, c in method_counts.items() if c == 6]
        genes_5 = [g for g, c in method_counts.items() if c >= 5]
        genes_4 = [g for g, c in method_counts.items() if c >= 4]
        gene_in_5of6.update(genes_5)
        gene_in_6of6.update(genes_6)
        records.append({
            "seed": seed, "trees": ntrees,
            "n_6of6": len(genes_6), "n_5of6": len(genes_5), "n_4of6": len(genes_4),
            "genes_6of6": "|".join(sorted(genes_6)),
            "genes_5of6": "|".join(sorted(genes_5)),
        })
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        print(f"seed={seed}, trees={ntrees}:  6/6={len(genes_6)}  5/6={len(genes_5)}  "
              f"(fold elapsed={elapsed:.0f}s, total={total_elapsed:.0f}s)")

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "tables" / "multiseed_consensus.csv", index=False)
print(f"\nSaved per-run records: {OUT_DIR}/tables/multiseed_consensus.csv")

# Per-gene inclusion frequency
n_runs = len(records)
per_gene = pd.DataFrame({
    "gene": list(set(list(gene_in_5of6.keys()) + list(gene_in_6of6.keys()))),
})
per_gene["freq_5of6"] = per_gene["gene"].map(lambda g: gene_in_5of6.get(g, 0) / n_runs)
per_gene["freq_6of6"] = per_gene["gene"].map(lambda g: gene_in_6of6.get(g, 0) / n_runs)
per_gene = per_gene.sort_values("freq_5of6", ascending=False).reset_index(drop=True)
per_gene.to_csv(OUT_DIR / "tables" / "multiseed_gene_frequency.csv", index=False)
print(f"Saved per-gene frequencies: {OUT_DIR}/tables/multiseed_gene_frequency.csv")

# Summary stats
print("\n--- Consensus size distribution across 30 runs ---")
for ntrees in TREES:
    sub = df[df['trees'] == ntrees]
    print(f"  trees={ntrees}:")
    print(f"    6/6: median={sub['n_6of6'].median():.1f}, IQR=[{sub['n_6of6'].quantile(0.25):.1f}, {sub['n_6of6'].quantile(0.75):.1f}], min={sub['n_6of6'].min()}, max={sub['n_6of6'].max()}")
    print(f"    5/6: median={sub['n_5of6'].median():.1f}, IQR=[{sub['n_5of6'].quantile(0.25):.1f}, {sub['n_5of6'].quantile(0.75):.1f}], min={sub['n_5of6'].min()}, max={sub['n_5of6'].max()}")

# Top recurring 5/6 genes
print("\n--- Top 20 genes by 5/6 inclusion frequency across 30 runs ---")
print(per_gene.head(20).to_string(index=False))

# Figures
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title in [
    (axes[0], "n_6of6", "Strict 6/6 consensus size"),
    (axes[1], "n_5of6", "Majority 5/6 consensus size"),
]:
    data = [df[df['trees'] == t][col].values for t in TREES]
    parts = ax.boxplot(data, labels=[str(t) for t in TREES], patch_artist=True)
    for p, c in zip(parts['boxes'], ["#a6cee3", "#1f78b4", "#b2df8a"]):
        p.set_facecolor(c); p.set_alpha(0.8)
    ax.set_xlabel("RF n_estimators")
    ax.set_ylabel("Number of genes")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle(f"Consensus robustness across {len(SEEDS)} seeds × {len(TREES)} RF tree counts",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "multiseed_consensus.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: {fig_path}")

# Per-gene inclusion bar
top_genes = per_gene.head(20)
fig, ax = plt.subplots(figsize=(8, 10))
y_pos = np.arange(len(top_genes))[::-1]
ax.barh(y_pos, top_genes["freq_5of6"].values, color="#1f77b4", alpha=0.85, label="5/6")
ax.barh(y_pos, top_genes["freq_6of6"].values, color="#d62728", alpha=0.85, label="6/6")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_genes["gene"].values, fontsize=9)
ax.set_xlabel(f"Fraction of {n_runs} runs including gene")
ax.axvline(0.8, linestyle=":", alpha=0.5, color="black", label="80% threshold")
ax.set_title(f"Per-gene inclusion frequency across {n_runs} runs (10 seeds × 3 tree counts)")
ax.legend(loc="lower right", frameon=True)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "multiseed_gene_frequency.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure: {fig_path}")

print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
