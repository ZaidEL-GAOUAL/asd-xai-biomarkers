"""Residualization-config control: multi-seed on GSE18123 with AGE-ONLY residualization.

Mirrors 06_multiseed_robustness.py exactly, but replaces the sex+age covariate frame
with age-only. This isolates whether the Tier-1/2 gene frequencies observed in the
discovery pipeline (sex+age config) survive the residualization configuration used
in the GSE25507 replication (age-only, forced by GSE25507 lacking sex metadata).

Per-gene reporting focuses on the genes carried forward to the GSE25507 check:
  Tier-1/2: SPG20, TIGD7, CES1, UBXN2A, EGR1
  Partial:  MALAT1, PNOC, LOC728554
  Reference: DDX3X
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
SEEDS = list(range(42, 52))
TREES = [200, 500, 1000]
K_FEATURES = 100
TOP_K = 20

CANDIDATES = ["SPG20", "TIGD7", "CES1", "UBXN2A", "EGR1",    # Tier-1/2
              "MALAT1", "PNOC", "LOC728554",                     # Partial
              "DDX3X"]                                            # Reference

# ─────────────────────── Load matrix + covariates ───────────────────────
print("Loading ComBat-corrected matrix...")
combat_df = pd.read_parquet(OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet")
X_all = combat_df.drop(columns=["diagnosis", "platform_id"])
meta = combat_df[["diagnosis", "platform_id"]].copy()

gse = GEOparse.get_GEO("GSE18123",
                        destdir=str(_P(__file__).resolve().parent.parent.parent / "data" / "raw"),
                        silent=True)
age = {}
for n, gsm in gse.gsms.items():
    for item in gsm.metadata.get("characteristics_ch1", []):
        if ":" not in item: continue
        k, v = item.split(":", 1); k = k.strip().lower(); v = v.strip()
        if k == "age at blood drawing (months)":
            try: age[n] = float(v)
            except: pass

# AGE-ONLY covariate frame (the discovery ran with sex+age; this is the control)
cov = pd.DataFrame({"age_months": pd.Series(age)}).loc[X_all.index]
print(f"Covariate frame: {cov.shape}, columns={list(cov.columns)} (age-only, NO sex)")

y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)

# ─────────────────────── Pipeline helpers (mirror script 06) ───────────────────────
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
    return df.head(K_FEATURES)["gene"].tolist()

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

# ─────────────────────── Run ───────────────────────
records = []
gene_in_5of6 = Counter()
gene_in_6of6 = Counter()
candidate_stab_hits = Counter()  # times each candidate in stability top-100

t_start = time.time()
for seed in SEEDS:
    top100 = stability_top100(seed)
    for g in CANDIDATES:
        if g in top100:
            candidate_stab_hits[g] += 1
    for ntrees in TREES:
        t0 = time.time()
        top_sets = run_xai(top100, seed, ntrees)
        method_counts = Counter()
        for s in top_sets.values():
            method_counts.update(s)
        genes_6 = [g for g, c in method_counts.items() if c == 6]
        genes_5 = [g for g, c in method_counts.items() if c >= 5]
        gene_in_5of6.update(genes_5)
        gene_in_6of6.update(genes_6)
        records.append({
            "seed": seed, "trees": ntrees,
            "n_6of6": len(genes_6), "n_5of6": len(genes_5),
            "genes_6of6": "|".join(sorted(genes_6)),
            "genes_5of6": "|".join(sorted(genes_5)),
        })
        elapsed = time.time() - t0; total = time.time() - t_start
        print(f"seed={seed}, trees={ntrees}:  6/6={len(genes_6)}  5/6={len(genes_5)}  "
              f"(elapsed={elapsed:.0f}s, total={total:.0f}s)")

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "tables" / "multiseed_consensus_age_only.csv", index=False)

n_runs = len(records)

# Per-candidate summary
summary_rows = []
for g in CANDIDATES:
    summary_rows.append({
        "gene": g,
        "stability_top100_hits": candidate_stab_hits.get(g, 0),
        "stability_top100_frac": candidate_stab_hits.get(g, 0) / len(SEEDS),
        "5of6_hits": gene_in_5of6.get(g, 0),
        "5of6_freq": gene_in_5of6.get(g, 0) / n_runs,
        "6of6_hits": gene_in_6of6.get(g, 0),
        "6of6_freq": gene_in_6of6.get(g, 0) / n_runs,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "tables" / "multiseed_age_only_candidate_frequencies.csv", index=False)

# Full per-gene frequency table (for completeness)
all_genes = set(gene_in_5of6.keys()) | set(gene_in_6of6.keys())
per_gene = pd.DataFrame({"gene": sorted(all_genes)})
per_gene["freq_5of6"] = per_gene["gene"].map(lambda g: gene_in_5of6.get(g, 0) / n_runs)
per_gene["freq_6of6"] = per_gene["gene"].map(lambda g: gene_in_6of6.get(g, 0) / n_runs)
per_gene = per_gene.sort_values("freq_5of6", ascending=False).reset_index(drop=True)
per_gene.to_csv(OUT_DIR / "tables" / "multiseed_age_only_all_genes.csv", index=False)

# ─────────────────────── Report ───────────────────────
print("\n" + "="*72)
print("AGE-ONLY GSE18123 CONTROL — Candidate gene frequencies across 30 runs")
print("="*72)
print(f"{'gene':12s}  {'stab_top100':>12s}  {'5/6':>8s}  {'6/6':>8s}")
for _, r in summary.iterrows():
    print(f"{r['gene']:12s}  {int(r['stability_top100_hits']):>6d}/10       "
          f"{int(r['5of6_hits']):>3d}/30    {int(r['6of6_hits']):>3d}/30")

print("\n--- Interpretation threshold ---")
print("Tier-1/2 genes at >=80% 5/6 in DISCOVERY (sex+age): SPG20=100%, TIGD7=100%, CES1=100%, UBXN2A=93%, EGR1=80%")
print("Under AGE-ONLY, whether these same 5 genes remain at >=80% is the key question.")

# Compare discovery (sex+age) vs control (age-only)
disc = pd.read_csv(OUT_DIR / "tables" / "multiseed_gene_frequency.csv").set_index("gene")
print(f"\n{'gene':12s}  {'disc 5/6':>10s}  {'ctrl 5/6':>10s}  {'Δ':>6s}")
for g in CANDIDATES:
    disc_freq = disc.loc[g, "freq_5of6"] if g in disc.index else 0.0
    ctrl_freq = gene_in_5of6.get(g, 0) / n_runs
    delta = ctrl_freq - disc_freq
    tag = "  [Tier-1/2]" if g in {"SPG20","TIGD7","CES1","UBXN2A","EGR1"} else ""
    print(f"{g:12s}  {disc_freq:>9.2f}   {ctrl_freq:>9.2f}   {delta:+6.2f}{tag}")

# Figure: side-by-side discovery vs control
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(CANDIDATES))
disc_vals = [disc.loc[g, "freq_5of6"] if g in disc.index else 0.0 for g in CANDIDATES]
ctrl_vals = [gene_in_5of6.get(g, 0) / n_runs for g in CANDIDATES]
ax.barh(y_pos - 0.2, disc_vals, height=0.4, color="#1f77b4", label="Discovery (sex+age)")
ax.barh(y_pos + 0.2, ctrl_vals, height=0.4, color="#ff7f0e", label="Control (age-only)")
ax.set_yticks(y_pos); ax.set_yticklabels(CANDIDATES); ax.invert_yaxis()
ax.set_xlabel("5/6 consensus inclusion frequency (30 runs each)")
ax.set_title("Residualization-config control on GSE18123\nsex+age vs age-only")
ax.axvline(0.8, linestyle=":", alpha=0.5, color="black", label="80% threshold")
ax.legend(loc="lower right", frameon=True)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "residualization_config_control.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved:")
print(f"  {OUT_DIR}/tables/multiseed_consensus_age_only.csv")
print(f"  {OUT_DIR}/tables/multiseed_age_only_candidate_frequencies.csv")
print(f"  {OUT_DIR}/tables/multiseed_age_only_all_genes.csv")
print(f"  {OUT_DIR}/figures/residualization_config_control.png")
print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
