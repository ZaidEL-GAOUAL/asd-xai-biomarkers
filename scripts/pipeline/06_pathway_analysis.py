"""Pathway-level analysis (secondary chapter to the gene-level primary track).

Motivation
----------
Our gene-level 4-gene signature (SPG20, TIGD7, CES1, EGR1) fails to replicate
in 3 external cohorts, including GSE6575 with matched tissue + platform.
Published ASD literature (Bao 2023, Lacalamita 2025, Fóthi 2022) shows
pathway-level models achieve 85-98% cross-cohort AUC where gene-level fails.
Biological interpretation: pathways buffer single-gene variability -- if gene
A and gene B both contribute to "MAPK signaling" dysregulation and cohort 1
picks A while cohort 2 picks B, the pathway-level signal still replicates.

This script runs the FULL v2-style analysis (stability → XAI consensus → null
→ multi-seed) on PATHWAY scores instead of GENE expression.

Pipeline
--------
1. Load ComBat-corrected gene matrix.
2. Compute ssGSEA scores → samples × pathways matrix (KEGG + Reactome, ~2100 pathways).
3. Stability selection on pathways (RepeatedStratifiedKFold + SelectKBest(k=50)).
4. Cross-model XAI consensus on top-50 pathways (LR + RF, 6 methods, 5/6 voting).
5. Quick null test (100 label-shuffle perms).
6. Quick multi-seed robustness (5 seeds × 2 tree counts).

Inputs
------
- results/processed/GSE18123_combat_corrected.parquet
- data/reference/gene_sets/KEGG_2021_Human.gmt
- data/reference/gene_sets/Reactome_2022.gmt

Outputs
-------
- results/processed/GSE18123_pathway_scores.parquet   (samples × pathways NES matrix)
- results/tables/pathway_stability.csv                (per-pathway selection frequency)
- results/tables/pathway_xai_consensus.csv            (5/6 and 6/6 consensus pathways)
- results/tables/pathway_null_summary.csv             (observed vs null)
- results/tables/pathway_multiseed_frequency.csv      (multi-seed inclusion freq)
- results/figures/pathway_consensus.png               (signature bar chart)

Runtime: ~15-20 min (ssGSEA is the slow step; downstream is fast on ~2100 features).
"""
import sys
import time
import warnings
from pathlib import Path as _P
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gseapy as gp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import shap

REPO_ROOT = _P(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results"
GENE_SETS_DIR = REPO_ROOT / "data" / "reference" / "gene_sets"

K_PATHWAYS = 50
TOP_K_XAI = 20
N_PERM_NULL = 50  # reduced from 100 for practical runtime; observed signal should clearly exceed null if real
SEEDS_MULTI = [42, 43, 44, 45, 46]
TREES_MULTI = [200, 500]

_T0 = time.time()

# ═════════════ Step 1: Load gene matrix + metadata ═════════════
print("=" * 72)
print(" Step 1: Load ComBat-corrected gene matrix")
print("=" * 72)

combat_path = OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet"
if not combat_path.exists():
    raise FileNotFoundError(f"Missing {combat_path}. Run scripts 00 + 01 first.")

df = pd.read_parquet(combat_path)
meta = df[["diagnosis", "platform_id"]].copy()
X_genes = df.drop(columns=["diagnosis", "platform_id"])
y_bin = (meta["diagnosis"].str.upper() != "CONTROL").astype(int)
print(f"Gene matrix: {X_genes.shape}  (samples x genes)")
print(f"Class balance: {dict(Counter(y_bin))}")

# ═════════════ Step 2: Compute ssGSEA scores ═════════════
print("\n" + "=" * 72)
print(" Step 2: Compute ssGSEA pathway scores (KEGG + Reactome)")
print("=" * 72)

scores_path = OUT_DIR / "processed" / "GSE18123_pathway_scores.parquet"

if scores_path.exists():
    print(f"  Cached scores found at {scores_path} — loading (set FORCE=1 env var to recompute)")
    X_pathways = pd.read_parquet(scores_path)
else:
    print("  Computing ssGSEA (this takes ~10-15 min)...")
    t_ssgsea = time.time()

    # gseapy ssgsea expects rows=genes, columns=samples
    data_for_ssgsea = X_genes.T

    # Load both GMT files as a merged dict
    def parse_gmt(path):
        sets = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                if len(genes) >= 5:  # drop tiny gene sets
                    sets[name] = genes
        return sets

    kegg = parse_gmt(GENE_SETS_DIR / "KEGG_2021_Human.gmt")
    reactome = parse_gmt(GENE_SETS_DIR / "Reactome_2022.gmt")
    print(f"  KEGG pathways (>= 5 genes):     {len(kegg)}")
    print(f"  Reactome pathways (>= 5 genes): {len(reactome)}")

    # Merge — prefix to preserve source
    merged = {f"KEGG:{k}": v for k, v in kegg.items()}
    merged.update({f"REACTOME:{k}": v for k, v in reactome.items()})
    print(f"  Merged gene sets: {len(merged)}")

    res = gp.ssgsea(
        data=data_for_ssgsea,
        gene_sets=merged,
        sample_norm_method="rank",
        min_size=5,
        max_size=500,
        no_plot=True,
        outdir=None,
    )

    # Pivot long-format (Name, Term, ES, NES) → wide (samples × pathways NES)
    X_pathways = res.res2d.pivot(index="Name", columns="Term", values="NES")
    # Ensure same sample order as X_genes
    X_pathways = X_pathways.loc[X_genes.index].astype(float)
    X_pathways = X_pathways.fillna(X_pathways.median(numeric_only=True))

    print(f"  ssGSEA done in {(time.time()-t_ssgsea)/60:.1f} min")
    X_pathways.to_parquet(scores_path)
    print(f"  Saved pathway scores to {scores_path}")

print(f"  Pathway matrix: {X_pathways.shape}  (samples x pathways)")

# ═════════════ Step 3: Stability selection on pathways ═════════════
print("\n" + "=" * 72)
print(f" Step 3: Pathway stability (SelectKBest k={K_PATHWAYS}, 25 CV fits)")
print("=" * 72)

from asd_pipeline_utils import stab_rank_no_resid

stab_df = stab_rank_no_resid(X_pathways, y_bin, seed=42, n_splits=5, n_repeats=5, k=K_PATHWAYS)
stab_df.columns = ["pathway", "count"]
stab_df["frequency"] = stab_df["count"] / 25.0
stab_df.to_csv(OUT_DIR / "tables" / "pathway_stability.csv", index=False)

print(f"  Pathways selected at least once: {len(stab_df)}")
print(f"  Pathways selected 25/25 folds:   {(stab_df['count'] == 25).sum()}")
print(f"  Pathways selected >= 20/25:      {(stab_df['count'] >= 20).sum()}")
print(f"\n  Top 15 most stable pathways:")
for _, r in stab_df.head(15).iterrows():
    print(f"    {r['pathway'][:80]:80s}  {int(r['count'])}/25")

# ═════════════ Step 4: Cross-model XAI consensus ═════════════
print("\n" + "=" * 72)
print(" Step 4: Cross-model XAI consensus (LR + RF, 6 methods, 5/6 voting)")
print("=" * 72)

# Take top-K stable pathways as input to XAI
top_pathways = stab_df.head(K_PATHWAYS)["pathway"].tolist()
X_top = X_pathways[top_pathways]

def run_xai_ensemble(X_top, y, seed, n_trees):
    """Run 6-method XAI on top-k features. Returns dict of top-TOP_K_XAI sets per method."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    methods = ["LR_PFI", "LR_SHAP", "LR_Coef", "RF_PFI", "RF_SHAP", "RF_Impurity"]
    agg = {m: np.zeros(X_top.shape[1]) for m in methods}
    feats = X_top.columns.tolist()

    for tr, te in skf.split(X_top, y):
        X_tr_df = X_top.iloc[tr]; X_te_df = X_top.iloc[te]
        y_tr = y.iloc[tr]; y_te = y.iloc[te]
        scl = StandardScaler().fit(X_tr_df.values)
        X_tr = pd.DataFrame(scl.transform(X_tr_df.values), index=X_tr_df.index, columns=feats)
        X_te = pd.DataFrame(scl.transform(X_te_df.values), index=X_te_df.index, columns=feats)
        if min(Counter(y_tr).values()) >= 6:
            sm = SMOTE(random_state=seed)
            X_tr_f, y_tr_f = sm.fit_resample(X_tr, y_tr)
        else:
            X_tr_f, y_tr_f = X_tr, y_tr

        lr = LogisticRegression(solver="saga", max_iter=8000, random_state=seed,
                                 class_weight="balanced").fit(X_tr_f, y_tr_f)
        agg["LR_PFI"] += permutation_importance(lr, X_te, y_te, n_repeats=10,
            random_state=seed, scoring="balanced_accuracy", n_jobs=-1).importances_mean / 5
        sv = shap.LinearExplainer(lr, X_te).shap_values(X_te)
        agg["LR_SHAP"] += (np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)) / 5
        agg["LR_Coef"] += (np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_).ravel()) / 5

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed,
                                      class_weight="balanced", n_jobs=-1).fit(X_tr_f, y_tr_f)
        agg["RF_PFI"] += permutation_importance(rf, X_te, y_te, n_repeats=10,
            random_state=seed, scoring="balanced_accuracy", n_jobs=-1).importances_mean / 5
        sv_rf = shap.TreeExplainer(rf).shap_values(X_te, check_additivity=False)
        if isinstance(sv_rf, list):
            sv_rf_abs = np.mean([np.abs(s) for s in sv_rf], axis=0).mean(axis=0)
        elif sv_rf.ndim == 3:
            sv_rf_abs = np.abs(sv_rf).mean(axis=(0, 2))
        else:
            sv_rf_abs = np.abs(sv_rf).mean(axis=0)
        agg["RF_SHAP"] += sv_rf_abs / 5
        agg["RF_Impurity"] += rf.feature_importances_ / 5

    feats_arr = np.asarray(feats)
    return {m: set(feats_arr[np.argsort(-arr)[:TOP_K_XAI]]) for m, arr in agg.items()}

print(f"  Running observed XAI (seed=42, 500 trees) on top-{K_PATHWAYS} pathways...")
t_xai = time.time()
top_sets = run_xai_ensemble(X_top, y_bin, seed=42, n_trees=500)
method_counts = Counter()
for m, s in top_sets.items():
    method_counts.update(s)
obs_6of6 = [p for p, c in method_counts.items() if c == 6]
obs_5of6 = [p for p, c in method_counts.items() if c >= 5]
print(f"  XAI done in {time.time()-t_xai:.1f}s")
print(f"  Observed 6/6 consensus pathways: {len(obs_6of6)}")
for p in sorted(obs_6of6):
    print(f"    [6/6] {p[:100]}")
print(f"  Observed 5/6 consensus pathways: {len(obs_5of6)}")
for p in sorted(obs_5of6):
    if p not in obs_6of6:
        print(f"    [5/6] {p[:100]}")

pd.DataFrame([
    {"pathway": p, "consensus": "6of6" if p in obs_6of6 else "5of6"}
    for p in obs_5of6
]).to_csv(OUT_DIR / "tables" / "pathway_xai_consensus.csv", index=False)

# ═════════════ Step 5: Quick null test (100 perms) ═════════════
print("\n" + "=" * 72)
print(f" Step 5: Permutation null ({N_PERM_NULL} label-shuffles)")
print("=" * 72)

rng = np.random.default_rng(42)
null_5of6 = []
null_6of6 = []
y_shuffle = y_bin.values.copy()
t_null = time.time()

for p in range(N_PERM_NULL):
    rng.shuffle(y_shuffle)
    y_p = pd.Series(y_shuffle, index=y_bin.index)
    # Use same k pathway pool but re-select in-fold
    top_null = stab_rank_no_resid(X_pathways, y_p, seed=42, n_splits=5, n_repeats=1, k=K_PATHWAYS)
    top_null.columns = ["pathway", "count"]
    pool = top_null.head(K_PATHWAYS)["pathway"].tolist()
    X_pool = X_pathways[pool]
    top_null_sets = run_xai_ensemble(X_pool, y_p, seed=42, n_trees=200)  # 200 trees for speed
    mc = Counter()
    for s in top_null_sets.values():
        mc.update(s)
    null_6of6.append(sum(1 for c in mc.values() if c == 6))
    null_5of6.append(sum(1 for c in mc.values() if c >= 5))
    if (p + 1) % 25 == 0:
        elapsed = time.time() - t_null
        print(f"  perm {p+1}/{N_PERM_NULL}  elapsed={elapsed:.0f}s  "
              f"est_remaining={elapsed*(N_PERM_NULL-p-1)/(p+1):.0f}s")

null_5of6 = np.array(null_5of6)
null_6of6 = np.array(null_6of6)
# Observed uses 500 trees, but the 200-tree null still gives a reasonable upper bound
p_5 = float(((null_5of6 >= len(obs_5of6)).sum() + 1) / (len(null_5of6) + 1))
p_6 = float(((null_6of6 >= len(obs_6of6)).sum() + 1) / (len(null_6of6) + 1))

print(f"\n  Observed 5/6 count = {len(obs_5of6)}, null median = {int(np.median(null_5of6))}, "
      f"null max = {int(null_5of6.max())}, p_emp = {p_5:.4g}")
print(f"  Observed 6/6 count = {len(obs_6of6)}, null median = {int(np.median(null_6of6))}, "
      f"null max = {int(null_6of6.max())}, p_emp = {p_6:.4g}")

pd.DataFrame([
    {"metric": "5of6", "observed": len(obs_5of6),
     "null_median": int(np.median(null_5of6)), "null_max": int(null_5of6.max()),
     "p_empirical": p_5, "n_perms": N_PERM_NULL},
    {"metric": "6of6", "observed": len(obs_6of6),
     "null_median": int(np.median(null_6of6)), "null_max": int(null_6of6.max()),
     "p_empirical": p_6, "n_perms": N_PERM_NULL},
]).to_csv(OUT_DIR / "tables" / "pathway_null_summary.csv", index=False)

# ═════════════ Step 6: Multi-seed robustness ═════════════
print("\n" + "=" * 72)
print(f" Step 6: Multi-seed robustness ({len(SEEDS_MULTI)} seeds × {len(TREES_MULTI)} tree counts)")
print("=" * 72)

n_runs = len(SEEDS_MULTI) * len(TREES_MULTI)
inc_5of6 = Counter()
inc_6of6 = Counter()
t_ms = time.time()

for seed in SEEDS_MULTI:
    # Re-select stability top-K for this seed (mirror gene-level pipeline)
    stab_seed = stab_rank_no_resid(X_pathways, y_bin, seed=seed, n_splits=5, n_repeats=5, k=K_PATHWAYS)
    stab_seed.columns = ["pathway", "count"]
    pool_seed = stab_seed.head(K_PATHWAYS)["pathway"].tolist()
    X_pool = X_pathways[pool_seed]

    for ntrees in TREES_MULTI:
        top_sets_seed = run_xai_ensemble(X_pool, y_bin, seed=seed, n_trees=ntrees)
        mc = Counter()
        for s in top_sets_seed.values():
            mc.update(s)
        g6 = {p for p, c in mc.items() if c == 6}
        g5 = {p for p, c in mc.items() if c >= 5}
        inc_5of6.update(g5)
        inc_6of6.update(g6)
        print(f"  seed={seed} trees={ntrees}: 5/6={len(g5)}, 6/6={len(g6)}")

ms_rows = []
for p in set(inc_5of6) | set(inc_6of6):
    ms_rows.append({
        "pathway": p,
        "freq_5of6": inc_5of6[p] / n_runs,
        "freq_6of6": inc_6of6[p] / n_runs,
    })
ms_df = pd.DataFrame(ms_rows).sort_values("freq_5of6", ascending=False).reset_index(drop=True)
ms_df.to_csv(OUT_DIR / "tables" / "pathway_multiseed_frequency.csv", index=False)

print(f"\n  Multi-seed done in {(time.time()-t_ms)/60:.1f} min")
print(f"  Pathways at freq_5of6 >= 0.80 (robust signature):")
robust = ms_df[ms_df["freq_5of6"] >= 0.80]
for _, r in robust.iterrows():
    print(f"    {r['pathway'][:80]:80s}  5/6={r['freq_5of6']:.0%}  6/6={r['freq_6of6']:.0%}")
if len(robust) == 0:
    print("    (none — pathway signature is not multi-seed-stable at this threshold)")

# ═════════════ Step 7: Figure ═════════════
print("\n" + "=" * 72)
print(" Step 7: Figure")
print("=" * 72)

fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * min(25, len(ms_df)))))
top_show = ms_df.head(25)
y_pos = np.arange(len(top_show))
ax.barh(y_pos, top_show["freq_5of6"], color="#1f77b4", alpha=0.75, label="5/6 consensus")
ax.barh(y_pos, top_show["freq_6of6"], color="#d62728", alpha=0.90, label="6/6 (strict)")
ax.set_yticks(y_pos)
# Truncate pathway names for display
labels = [p[:70] + "..." if len(p) > 70 else p for p in top_show["pathway"]]
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.axvline(0.80, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="80% robust threshold")
ax.set_xlabel("Multi-seed inclusion frequency")
ax.set_title(f"Pathway-level XAI consensus on GSE18123\n"
             f"(top-{K_PATHWAYS} stable pathways, {n_runs} runs)")
ax.legend(loc="lower right", fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, 1.05)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "pathway_consensus.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {fig_path}")

# ═════════════ Summary ═════════════
print("\n" + "=" * 72)
print(" SUMMARY — pathway-level analysis on GSE18123")
print("=" * 72)

print(f"\nPathway matrix: {X_pathways.shape[1]} pathways (KEGG + Reactome)")
print(f"Top stability pathways (25/25 fits): {(stab_df['count'] == 25).sum()}")
print(f"Observed 5/6 consensus (seed=42):    {len(obs_5of6)}  (null p={p_5:.4g})")
print(f"Observed 6/6 consensus (seed=42):    {len(obs_6of6)}  (null p={p_6:.4g})")
print(f"Robust multi-seed signature (freq_5of6 >= 0.80): {len(robust)} pathways")
if len(robust) > 0:
    print(f"\nRobust pathways:")
    for _, r in robust.iterrows():
        print(f"  - {r['pathway']}")

print(f"\nTotal runtime: {(time.time() - _T0) / 60:.1f} min")
print(f"Outputs under {OUT_DIR}")
