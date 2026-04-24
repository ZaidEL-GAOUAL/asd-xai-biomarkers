"""No-residualization multi-seed control on GSE18123 (Adjustment 2).

Third config-invariance check, layered on top of:
  - Discovery: sex + age residualization (script 06)
  - Control 1: age-only residualization (script 08)
  - Control 2: no residualization at all (this script)

Reports 5/6 and 6/6 inclusion frequency for the 4 primary genes + UBXN2A + the
partial set, across 30 runs (10 seeds x 3 tree counts).

Decision rule: any gene dropping below 80% 5/6 under no-residualization gets
demoted from "config-invariant" to "config-sensitive."
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import os

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

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
SEEDS = list(range(42, 52))
TREES = [200, 500, 1000]
K_FEATURES = 100
TOP_K = 20

CANDIDATES = ["SPG20", "TIGD7", "CES1", "UBXN2A", "EGR1",
              "MALAT1", "PNOC", "LOC728554", "DDX3X"]

# Load ComBat-corrected matrix
combat_df = pd.read_parquet(OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet")
X_all = combat_df.drop(columns=["diagnosis","platform_id"])
meta = combat_df[["diagnosis","platform_id"]].copy()
y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)
print(f"X={X_all.shape}, y balance={dict(Counter(y_bin))}")
print("Config: NO RESIDUALIZATION (third control)")

def stab_top100(seed):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    counts = Counter()
    for tr, _ in cv.split(X_all, y_bin):
        X_tr = X_all.iloc[tr]; y_tr = y_bin.iloc[tr]
        sel = SelectKBest(f_classif, k=K_FEATURES).fit(X_tr.values, y_tr)
        counts.update(X_all.columns[sel.get_support()].tolist())
    df = pd.DataFrame([{"gene":g,"count":c} for g,c in counts.items()])
    return df.sort_values(["count","gene"],ascending=[False,True]).head(K_FEATURES)["gene"].tolist()

def run_xai(top100, seed, n_trees):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_top = X_all[top100]
    methods = ["LR_PFI","LR_SHAP","LR_Coef","RF_PFI","RF_SHAP","RF_Impurity"]
    agg = {m: np.zeros(len(top100)) for m in methods}
    for tr, te in skf.split(X_top, y_bin):
        X_tr_df = X_top.iloc[tr]; X_te_df = X_top.iloc[te]
        y_tr = y_bin.iloc[tr]; y_te = y_bin.iloc[te]
        scl = StandardScaler().fit(X_tr_df.values)
        X_tr = pd.DataFrame(scl.transform(X_tr_df.values), index=X_tr_df.index, columns=X_tr_df.columns)
        X_te = pd.DataFrame(scl.transform(X_te_df.values), index=X_te_df.index, columns=X_te_df.columns)
        if min(Counter(y_tr).values()) >= 6:
            sm_ = SMOTE(random_state=seed)
            X_tr_f, y_tr_f = sm_.fit_resample(X_tr, y_tr)
        else:
            X_tr_f, y_tr_f = X_tr, y_tr
        lr = LogisticRegression(solver="saga", max_iter=8000, random_state=seed,
                                 class_weight="balanced").fit(X_tr_f, y_tr_f)
        agg["LR_PFI"] += permutation_importance(lr, X_te, y_te, n_repeats=10,
            random_state=seed, scoring="balanced_accuracy", n_jobs=-1).importances_mean / 5
        sv = shap.LinearExplainer(lr, X_te).shap_values(X_te)
        agg["LR_SHAP"] += (np.abs(sv).mean(axis=(0,2)) if sv.ndim==3 else np.abs(sv).mean(axis=0)) / 5
        agg["LR_Coef"] += (np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim>1 else np.abs(lr.coef_).ravel()) / 5
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed,
                                      class_weight="balanced", n_jobs=-1).fit(X_tr_f, y_tr_f)
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

records = []
gene5 = Counter(); gene6 = Counter()
stab_hits = Counter()
t0 = time.time()
for seed in SEEDS:
    top100 = stab_top100(seed)
    for g in CANDIDATES:
        if g in top100: stab_hits[g] += 1
    for ntrees in TREES:
        t1 = time.time()
        top_sets = run_xai(top100, seed, ntrees)
        mc = Counter()
        for s in top_sets.values():
            mc.update(s)
        g6 = [g for g,c in mc.items() if c==6]
        g5 = [g for g,c in mc.items() if c>=5]
        gene5.update(g5); gene6.update(g6)
        records.append({"seed":seed,"trees":ntrees,"n_6of6":len(g6),"n_5of6":len(g5),
                         "genes_6of6":"|".join(sorted(g6)),"genes_5of6":"|".join(sorted(g5))})
        print(f"  seed={seed}, trees={ntrees}: 6/6={len(g6)}, 5/6={len(g5)}  "
              f"({time.time()-t1:.0f}s, total={time.time()-t0:.0f}s)")

df_r = pd.DataFrame(records)
df_r.to_csv(OUT_DIR / "tables" / "multiseed_no_resid.csv", index=False)
n_runs = len(records)

# Summary
disc = pd.read_csv(OUT_DIR / "tables" / "multiseed_gene_frequency.csv").set_index("gene")
age_only_df = pd.read_csv(OUT_DIR / "tables" / "multiseed_age_only_candidate_frequencies.csv").set_index("gene")

print("\n" + "="*80)
print("THREE-WAY CONFIG COMPARISON for key genes")
print("="*80)
print(f"{'gene':12s}  {'disc (sex+age)':>15s}  {'age-only':>10s}  {'no-resid':>10s}  verdict")
for g in CANDIDATES:
    disc_f = float(disc.loc[g, "freq_5of6"]) if g in disc.index else 0.0
    age_f  = float(age_only_df.loc[g, "5of6_freq"]) if g in age_only_df.index else 0.0
    nor_f  = gene5.get(g, 0) / n_runs
    # config-invariance: all three >= 0.80
    invariant = min(disc_f, age_f, nor_f) >= 0.80
    verdict = "CONFIG-INVARIANT" if invariant else ("demoted" if any(x >= 0.80 for x in (disc_f, age_f, nor_f)) else "weak")
    print(f"{g:12s}  {disc_f:>14.2f}   {age_f:>9.2f}   {nor_f:>9.2f}   {verdict}")

# Save no-resid gene frequencies
per_gene = pd.DataFrame({"gene": sorted(set(gene5.keys()) | set(gene6.keys()))})
per_gene["freq_5of6"] = per_gene["gene"].map(lambda g: gene5.get(g, 0)/n_runs)
per_gene["freq_6of6"] = per_gene["gene"].map(lambda g: gene6.get(g, 0)/n_runs)
per_gene = per_gene.sort_values("freq_5of6", ascending=False).reset_index(drop=True)
per_gene.to_csv(OUT_DIR / "tables" / "multiseed_no_resid_gene_frequency.csv", index=False)

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
