"""Unified external validation: gene + pathway signatures on 3 external cohorts.

This is THE test of the thesis: does either signature (gene-level or pathway-level)
produced by GSE18123 transfer to independent cohorts?

Replaces current scripts 07, 09, 10, 14 (all the per-cohort external validation
scripts were doing the same work separately). One script, one answer, consistent
methodology across cohorts.

Cohorts
-------
- GSE18123 (n=285, whole blood, GPL570+GPL6244)           [discovery]
- GSE25507 (n=146, lymphocytes, GPL570)                   [external]
- GSE42133 (n=147, leukocytes, GPL10558 Illumina)         [external]
- GSE6575  (n=47 ASD+TD, whole blood, GPL570)             [external, tissue-match]

Phases
------
A. Load all 4 cohorts + extract diagnosis labels
B. Joint 4-cohort ComBat (mod=diagnosis preserved; 4 batch levels)
C. Compute ssGSEA pathway scores on joint matrix
D. GENE-LEVEL transfer: train on GSE18123 [SPG20, TIGD7, CES1, EGR1], test on each external cohort
E. PATHWAY-LEVEL transfer: train on GSE18123 [6 pathways from script 06], test on each external cohort
F. Cohort-specific pathway consensus: does each external cohort's top-N pathway overlap with discovery?

Outputs
-------
- results/processed/four_cohort_combat.parquet           joint 4-cohort ComBat gene matrix
- results/processed/four_cohort_pathway_scores.parquet   joint ssGSEA pathway matrix
- results/tables/gene_external_validation.csv           gene-level transfer results
- results/tables/pathway_external_validation.csv        pathway-level transfer results
- results/tables/pathway_signature_overlap.csv          which of 6 pathways appear in each cohort
- results/figures/external_validation_comparison.png    gene vs pathway transfer bars

Runtime: ~30 min (dominated by ssGSEA on 4 cohorts).
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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from imblearn.over_sampling import SMOTE
import GEOparse

from asd_pipeline_utils import load_analysis_frame, split_features_and_metadata, clean_symbol, stab_rank_no_resid
from inmoose.pycombat import pycombat_norm

REPO_ROOT = _P(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results"
DATA_RAW = REPO_ROOT / "data" / "raw"
GENE_SETS_DIR = REPO_ROOT / "data" / "reference" / "gene_sets"

GENE_SIGNATURE = ["SPG20", "TIGD7", "CES1", "LOC728554"]
PATHWAY_SIGNATURE_SOURCE = OUT_DIR / "tables" / "pathway_multiseed_frequency.csv"

_T0 = time.time()

# ═════════════ Phase A: Load all 4 cohorts ═════════════
print("=" * 72)
print(" A. Load all 4 cohorts")
print("=" * 72)

# GSE18123 from cached processed parquet
raw18 = load_analysis_frame()
X18_full, meta18 = split_features_and_metadata(raw18)
X18_full = X18_full.fillna(X18_full.median(numeric_only=True))
y18 = (meta18["diagnosis"].str.upper() != "CONTROL").astype(int)
print(f"GSE18123: {X18_full.shape}, labels={dict(Counter(y18))}")

def load_geo_cohort(gse_id, gpl_list, diag_keys, diag_map):
    """Load a cohort from cached GEO SOFT, return X (samples x genes) and y.

    Args:
      gse_id: e.g. 'GSE25507'
      gpl_list: list of (gpl_id, gene_col) tuples to try for annotation
      diag_keys: list of lowercased keys to search in characteristics_ch1
      diag_map: function that takes characteristic value and returns 0/1/None
    """
    g = GEOparse.get_GEO(gse_id, destdir=str(DATA_RAW), silent=True)
    expr = g.pivot_samples("VALUE").apply(pd.to_numeric, errors="coerce")

    diag = {}
    for n, gsm in g.gsms.items():
        # Try source_name_ch1 first, then characteristics_ch1
        for src in [gsm.metadata.get("source_name_ch1", [""])[0]] + gsm.metadata.get("characteristics_ch1", []):
            src_l = src.lower()
            for dk in diag_keys:
                if dk in src_l:
                    label = diag_map(src_l)
                    if label is not None:
                        diag[n] = label
                        break
            if n in diag:
                break

    # Find first working GPL annotation
    p2g = {}
    for gpl_id, gene_col in gpl_list:
        if gpl_id not in g.gpls:
            continue
        gpl_table = g.gpls[gpl_id].table
        if gene_col not in gpl_table.columns:
            continue
        ann = gpl_table[["ID", gene_col]].dropna()
        ann[gene_col] = ann[gene_col].map(clean_symbol)
        ann = ann.dropna()
        p2g = dict(zip(ann["ID"].astype(str), ann[gene_col].astype(str)))
        print(f"  {gse_id}: using {gpl_id}/{gene_col} annotation ({len(p2g)} probes mapped)")
        break
    if not p2g:
        raise RuntimeError(f"No working GPL annotation for {gse_id}")

    samples = [s for s in expr.columns if s in diag]
    sub = expr[samples].copy()
    sub.index = sub.index.map(lambda pid: p2g.get(str(pid), np.nan))
    sub = sub[sub.index.notna()]
    sub = sub.groupby(sub.index).mean()
    X = sub.T

    mx = np.nanmax(X.to_numpy())
    if mx > 50:
        X = np.log2(X + 1)
    X = X.fillna(X.median(numeric_only=True))
    y = pd.Series({s: diag[s] for s in samples}).loc[X.index]
    return X, y

# GSE25507
print("\nLoading GSE25507...")
X25_full, y25 = load_geo_cohort(
    "GSE25507",
    gpl_list=[("GPL570", "Gene Symbol")],
    diag_keys=["group", "diagnosis"],
    diag_map=lambda v: 0 if "control" in v else (1 if "autism" in v or "autis" in v else None),
)
print(f"GSE25507: {X25_full.shape}, labels={dict(Counter(y25))}")

# GSE42133
print("\nLoading GSE42133...")
X42_full, y42 = load_geo_cohort(
    "GSE42133",
    gpl_list=[("GPL10558", "ILMN_Gene"), ("GPL10558", "Symbol")],
    diag_keys=["dx", "diagnosis"],
    diag_map=lambda v: 0 if "control" in v else (1 if "asd" in v or "autis" in v else None),
)
print(f"GSE42133: {X42_full.shape}, labels={dict(Counter(y42))}")

# GSE6575 (exclude MRDD from primary binary comparison)
print("\nLoading GSE6575...")
X6575_full, y6575 = load_geo_cohort(
    "GSE6575",
    gpl_list=[("GPL570", "Gene Symbol")],
    diag_keys=["autism", "general population", "mental retardation"],
    diag_map=lambda v: (0 if "general population" in v
                         else (1 if "autism" in v
                               else None)),  # explicitly drop MRDD samples
)
print(f"GSE6575:  {X6575_full.shape}, labels={dict(Counter(y6575))}")

# Common genes across all 4 cohorts
common = (X18_full.columns
          .intersection(X25_full.columns)
          .intersection(X42_full.columns)
          .intersection(X6575_full.columns))
print(f"\nCommon genes across all 4 cohorts: {len(common)}")
for g in GENE_SIGNATURE + ["DDX3X", "UBXN2A", "MALAT1"]:
    print(f"  {g}: {g in common}")

X18 = X18_full[common]
X25 = X25_full[common]
X42 = X42_full[common]
X6575 = X6575_full[common]

# ═════════════ Phase B: Joint 4-cohort ComBat ═════════════
print("\n" + "=" * 72)
print(" B. Joint 4-cohort ComBat (mod=diagnosis preserved)")
print("=" * 72)

X_joint = pd.concat([X18, X25, X42, X6575], axis=0)
batch = []
for s in X_joint.index:
    if s in X18.index:
        plat = meta18.loc[s, "platform_id"]
        batch.append(f"GSE18123_{plat}")
    elif s in X25.index:
        batch.append("GSE25507_GPL570")
    elif s in X42.index:
        batch.append("GSE42133_GPL10558")
    else:
        batch.append("GSE6575_GPL570")
batch = np.array(batch)
print(f"Batches: {dict(Counter(batch))}")

y_joint = pd.concat([y18, y25, y42, y6575])
mod = pd.get_dummies(y_joint, drop_first=True).astype(float).values
print("Running pycombat_norm on joint 4-cohort matrix...")
X_joint_T = pycombat_norm(X_joint.T, batch=batch, mod=mod)
X_joint_c = pd.DataFrame(X_joint_T.T.values, index=X_joint.index, columns=X_joint.columns)

X18_c = X_joint_c.loc[X18.index]
X25_c = X_joint_c.loc[X25.index]
X42_c = X_joint_c.loc[X42.index]
X6575_c = X_joint_c.loc[X6575.index]

OUT_PARQUET = OUT_DIR / "processed" / "four_cohort_combat.parquet"
X_joint_c.to_parquet(OUT_PARQUET)
print(f"Saved: {OUT_PARQUET}")

# Quick PCA diagnostic (cohort silhouette)
Xsc_after = StandardScaler().fit_transform(X_joint_c.values)
p2 = PCA(n_components=2, random_state=42).fit(Xsc_after)
pc_after = p2.transform(Xsc_after)
cohort_labels = np.array(["GSE18123" if s in X18.index else
                            ("GSE25507" if s in X25.index else
                             ("GSE42133" if s in X42.index else "GSE6575"))
                           for s in X_joint.index])
sil_cohort = silhouette_score(pc_after, cohort_labels)
sil_batch = silhouette_score(pc_after, batch)
print(f"Silhouette after ComBat — cohort: {sil_cohort:.3f}, batch: {sil_batch:.3f}")

# ═════════════ Phase C: ssGSEA pathway scores on joint matrix ═════════════
print("\n" + "=" * 72)
print(" C. ssGSEA pathway scores on joint ComBat matrix")
print("=" * 72)

path_parquet = OUT_DIR / "processed" / "four_cohort_pathway_scores.parquet"

if path_parquet.exists():
    print(f"Cached: {path_parquet}. Loading.")
    X_paths = pd.read_parquet(path_parquet)
else:
    def parse_gmt(path):
        sets = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                if len(genes) >= 5:
                    sets[name] = genes
        return sets

    kegg = parse_gmt(GENE_SETS_DIR / "KEGG_2021_Human.gmt")
    reactome = parse_gmt(GENE_SETS_DIR / "Reactome_2022.gmt")
    merged = {f"KEGG:{k}": v for k, v in kegg.items()}
    merged.update({f"REACTOME:{k}": v for k, v in reactome.items()})
    print(f"Running ssGSEA on joint {X_joint_c.shape[0]}-sample matrix across {len(merged)} gene sets...")
    t_s = time.time()
    res = gp.ssgsea(data=X_joint_c.T, gene_sets=merged, sample_norm_method="rank",
                      min_size=5, max_size=500, no_plot=True, outdir=None)
    X_paths = res.res2d.pivot(index="Name", columns="Term", values="NES")
    X_paths = X_paths.loc[X_joint_c.index].astype(float)
    X_paths = X_paths.fillna(X_paths.median(numeric_only=True))
    X_paths.to_parquet(path_parquet)
    print(f"ssGSEA done in {(time.time()-t_s)/60:.1f} min, saved: {path_parquet}")

print(f"Pathway matrix: {X_paths.shape}")

# Slice pathway scores per cohort
P18 = X_paths.loc[X18.index]
P25 = X_paths.loc[X25.index]
P42 = X_paths.loc[X42.index]
P6575 = X_paths.loc[X6575.index]

# ═════════════ Phase D: GENE-LEVEL transfer ═════════════
print("\n" + "=" * 72)
print(" D. GENE-LEVEL transfer: GSE18123 4-gene signature → each external cohort")
print("=" * 72)

def train_transfer(X_train, y_train, X_test, y_test, test_name, feat_name):
    """Train LR + RF on X_train[features], test on X_test[features]."""
    rows = []
    scl = StandardScaler().fit(X_train.values)
    X_tr = pd.DataFrame(scl.transform(X_train.values), index=X_train.index, columns=X_train.columns)
    X_te = pd.DataFrame(scl.transform(X_test.values), index=X_test.index, columns=X_test.columns)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr, y_train)
    for name, clf in [
        ("LR", LogisticRegression(solver="saga", max_iter=8000, random_state=42, class_weight="balanced").fit(X_res, y_res)),
        ("RF", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1).fit(X_res, y_res)),
    ]:
        yp = clf.predict(X_te)
        yo = clf.predict_proba(X_te)[:, 1]
        bal = balanced_accuracy_score(y_test, yp)
        auc = roc_auc_score(y_test, yo)
        rows.append({"features": feat_name, "train": "GSE18123", "test": test_name,
                      "model": name, "bal_acc": float(bal), "roc_auc": float(auc),
                      "n_features": X_train.shape[1], "n_test": len(y_test)})
        print(f"  {feat_name:>8s}  {test_name:>8s}  {name}  bal_acc={bal:.3f}  ROC-AUC={auc:.3f}")
    return rows

gene_rows = []
gene_feats = [g for g in GENE_SIGNATURE if g in X18_c.columns]
print(f"Gene features used ({len(gene_feats)} of {len(GENE_SIGNATURE)}): {gene_feats}")
X18_gene = X18_c[gene_feats]

for test_name, X_test_c, y_test in [("GSE25507", X25_c, y25), ("GSE42133", X42_c, y42), ("GSE6575", X6575_c, y6575)]:
    X_test = X_test_c[gene_feats]
    gene_rows.extend(train_transfer(X18_gene, y18, X_test, y_test, test_name, "gene"))

# Baseline: internal CV on GSE18123
from sklearn.model_selection import cross_val_score
for name, clf in [("LR", LogisticRegression(solver="saga", max_iter=8000, random_state=42, class_weight="balanced")),
                    ("RF", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1))]:
    scl = StandardScaler().fit(X18_gene.values)
    X18_s = pd.DataFrame(scl.transform(X18_gene.values), index=X18_gene.index, columns=gene_feats)
    scores = cross_val_score(clf, X18_s, y18, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               scoring="balanced_accuracy", n_jobs=-1)
    gene_rows.append({"features": "gene", "train": "GSE18123", "test": "GSE18123 (internal CV)",
                       "model": name, "bal_acc": float(scores.mean()), "roc_auc": np.nan,
                       "n_features": len(gene_feats), "n_test": len(y18)})
    print(f"  {'gene':>8s}  {'GSE18123 CV':>8s}  {name}  bal_acc={scores.mean():.3f} ± {scores.std():.3f}")

pd.DataFrame(gene_rows).to_csv(OUT_DIR / "tables" / "gene_external_validation.csv", index=False)

# ═════════════ Phase E: PATHWAY-LEVEL transfer ═════════════
print("\n" + "=" * 72)
print(" E. PATHWAY-LEVEL transfer: GSE18123 pathway signature → each external cohort")
print("=" * 72)

# Load the pathway signature from script 06's multi-seed output
ms_df = pd.read_csv(PATHWAY_SIGNATURE_SOURCE)
path_signature = ms_df[ms_df["freq_5of6"] >= 0.80]["pathway"].tolist()
print(f"Pathway signature ({len(path_signature)} pathways at freq_5of6 >= 80%):")
for p in path_signature:
    print(f"  - {p}")

path_feats = [p for p in path_signature if p in P18.columns]
print(f"Pathway features available: {len(path_feats)} of {len(path_signature)}")
P18_sig = P18[path_feats]

path_rows = []
for test_name, P_test, y_test in [("GSE25507", P25, y25), ("GSE42133", P42, y42), ("GSE6575", P6575, y6575)]:
    P_test_sig = P_test[path_feats]
    path_rows.extend(train_transfer(P18_sig, y18, P_test_sig, y_test, test_name, "pathway"))

# Baseline: internal CV on GSE18123 pathways
for name, clf in [("LR", LogisticRegression(solver="saga", max_iter=8000, random_state=42, class_weight="balanced")),
                    ("RF", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1))]:
    scl = StandardScaler().fit(P18_sig.values)
    P18_s = pd.DataFrame(scl.transform(P18_sig.values), index=P18_sig.index, columns=path_feats)
    scores = cross_val_score(clf, P18_s, y18, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               scoring="balanced_accuracy", n_jobs=-1)
    path_rows.append({"features": "pathway", "train": "GSE18123", "test": "GSE18123 (internal CV)",
                       "model": name, "bal_acc": float(scores.mean()), "roc_auc": np.nan,
                       "n_features": len(path_feats), "n_test": len(y18)})
    print(f"  {'pathway':>8s}  {'GSE18123 CV':>8s}  {name}  bal_acc={scores.mean():.3f} ± {scores.std():.3f}")

pd.DataFrame(path_rows).to_csv(OUT_DIR / "tables" / "pathway_external_validation.csv", index=False)

# ═════════════ Phase F: Pathway signature overlap per cohort ═════════════
print("\n" + "=" * 72)
print(" F. Cohort-specific pathway rankings — does each cohort independently find these 6 pathways?")
print("=" * 72)

overlap_rows = []
for cohort_name, P_cohort, y_cohort in [("GSE25507", P25, y25), ("GSE42133", P42, y42), ("GSE6575", P6575, y6575)]:
    print(f"\n  Computing stability ranking on {cohort_name} pathways...")
    rank_df = stab_rank_no_resid(P_cohort, y_cohort, seed=42, n_splits=5, n_repeats=5, k=50)
    rank_df.columns = ["pathway", "count"]
    for N in [50, 100, 200]:
        top_N = set(rank_df.head(N)["pathway"])
        hits = [p for p in path_feats if p in top_N]
        overlap_rows.append({"cohort": cohort_name, "N": N,
                              "n_signature_found": len(hits),
                              "n_signature_total": len(path_feats),
                              "pct_recovered": 100.0 * len(hits) / len(path_feats) if path_feats else 0,
                              "pathways_found": "|".join(hits)})
        print(f"    top-{N:>3d}: {len(hits)}/{len(path_feats)} signature pathways found")
        if hits:
            for p in hits:
                r = rank_df.index[rank_df["pathway"] == p][0] + 1
                print(f"      [#{r}] {p[:80]}")

pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "tables" / "pathway_signature_overlap.csv", index=False)

# ═════════════ Figure: gene vs pathway transfer comparison ═════════════
print("\n" + "=" * 72)
print(" Figure: gene vs pathway transfer performance")
print("=" * 72)

gdf = pd.DataFrame(gene_rows)
pdf = pd.DataFrame(path_rows)
# Drop internal CV rows for the external-transfer panel
gdf_ext = gdf[~gdf["test"].str.contains("internal CV")].copy()
pdf_ext = pdf[~pdf["test"].str.contains("internal CV")].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, metric, ylim in zip(axes, ["bal_acc", "roc_auc"], [(0.3, 0.85), (0.3, 0.85)]):
    width = 0.35
    cohorts = ["GSE25507", "GSE42133", "GSE6575"]
    x = np.arange(len(cohorts))
    for i, model in enumerate(["LR", "RF"]):
        offset = (i - 0.5) * width * 0.5
        gv = [gdf_ext[(gdf_ext["test"] == c) & (gdf_ext["model"] == model)][metric].iloc[0] for c in cohorts]
        pv = [pdf_ext[(pdf_ext["test"] == c) & (pdf_ext["model"] == model)][metric].iloc[0] for c in cohorts]
        ax.bar(x - width/2 + offset, gv, width * 0.45, label=f"gene-4   ({model})", alpha=0.8,
                color="#1f77b4" if model == "LR" else "#aec7e8")
        ax.bar(x + width/2 + offset, pv, width * 0.45, label=f"pathway-{len(path_feats)} ({model})", alpha=0.85,
                color="#d62728" if model == "LR" else "#ff9896")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel(metric)
    ax.set_title(f"Transfer {metric} (GSE18123 → external)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "figures" / "external_validation_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {fig_path}")

# ═════════════ Summary ═════════════
print("\n" + "=" * 72)
print(" SUMMARY — external validation at gene + pathway level")
print("=" * 72)

print(f"\n{'features':>10s}  {'train':>10s}  {'test':>22s}  {'model':>5s}  {'bal_acc':>8s}  {'ROC_AUC':>8s}")
for r in gene_rows + path_rows:
    bal_s = f"{r['bal_acc']:.3f}"
    auc_s = f"{r['roc_auc']:.3f}" if not pd.isna(r['roc_auc']) else "---"
    print(f"{r['features']:>10s}  {r['train']:>10s}  {r['test']:>22s}  {r['model']:>5s}  {bal_s:>8s}  {auc_s:>8s}")

# Compute win counts
wins_gene = sum(1 for r in gene_rows if "internal" not in r["test"] and r["bal_acc"] > 0.60)
wins_path = sum(1 for r in path_rows if "internal" not in r["test"] and r["bal_acc"] > 0.60)
print(f"\nTransfers achieving bal_acc > 0.60 (of 6 = 3 cohorts × 2 models):")
print(f"  Gene-level signature:    {wins_gene}/6")
print(f"  Pathway-level signature: {wins_path}/6")

if wins_path > wins_gene:
    print("  → Pathways transfer better than genes. Supports the thesis pivot.")
elif wins_path == wins_gene == 0:
    print("  → Both signatures fail external replication. Honest negative at both levels.")
elif wins_path == wins_gene:
    print("  → Both signatures comparable in transfer performance.")
else:
    print("  → Genes transfer better than pathways (unexpected).")

print(f"\nTotal runtime: {(time.time() - _T0) / 60:.1f} min")
print(f"Outputs under {OUT_DIR}")
