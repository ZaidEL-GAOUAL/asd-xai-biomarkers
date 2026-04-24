"""ComBat with mod=diagnosis preserved, plus PCA before/after."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import os

import numpy as np
import pandas as pd
from pathlib import Path
from asd_pipeline_utils import load_analysis_frame, split_features_and_metadata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inmoose.pycombat import pycombat_norm

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ── Load ──
final_df = load_analysis_frame()
X, meta = split_features_and_metadata(final_df)
X_imp = X.fillna(X.median(numeric_only=True))
print(f"X shape: {X_imp.shape}")

# Build diagnosis design matrix (drop first level to avoid collinearity with intercept)
diag_dummies = pd.get_dummies(meta["diagnosis"].astype(str), drop_first=True)
diag_dummies.index = X_imp.index
print(f"Diagnosis dummies: {list(diag_dummies.columns)}  shape={diag_dummies.shape}")

batch = meta["platform_id"].astype(str).values

# ── ComBat with mod preserved ──
print("\nRunning ComBat with mod=diagnosis preserved...")
X_combat_T = pycombat_norm(X_imp.T, batch=batch, mod=diag_dummies.values.astype(float))
X_combat = pd.DataFrame(X_combat_T.T.values, index=X_imp.index, columns=X_imp.columns)
print(f"ComBat output: {X_combat.shape}")

# ── Save ──
X_combat.join(meta).to_parquet(OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet", index=True)
print("Saved corrected matrix.")

# ── PCA before / after ──
def fit_pca(X_df):
    X_sc = StandardScaler().fit_transform(X_df.values)
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(X_sc), p.explained_variance_ratio_ * 100

pc_before, var_before = fit_pca(X_imp)
pc_after, var_after = fit_pca(X_combat)

print(f"\nBefore: PC1={var_before[0]:.1f}%, PC2={var_before[1]:.1f}%")
print(f"After:  PC1={var_after[0]:.1f}%, PC2={var_after[1]:.1f}%")

sil_before = silhouette_score(pc_before, meta["platform_id"].values)
sil_after  = silhouette_score(pc_after,  meta["platform_id"].values)
print(f"\nPlatform silhouette: before={sil_before:.3f}, after={sil_after:.3f}")

# Centroid distance
centroid_before = {p: pc_before[meta["platform_id"].values == p].mean(axis=0) for p in ["GPL570", "GPL6244"]}
centroid_after  = {p: pc_after[meta["platform_id"].values == p].mean(axis=0)  for p in ["GPL570", "GPL6244"]}
d_before = np.linalg.norm(centroid_before["GPL570"] - centroid_before["GPL6244"])
d_after  = np.linalg.norm(centroid_after["GPL570"]  - centroid_after["GPL6244"])
print(f"Centroid distance: before={d_before:.2f}, after={d_after:.2f}")

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"GPL570": "#d62728", "GPL6244": "#1f77b4"}
labels = {"GPL570": "GPL570 (n=99)", "GPL6244": "GPL6244 (n=186)"}

for ax, scores, var, title in [
    (axes[0], pc_before, var_before, "Before ComBat"),
    (axes[1], pc_after, var_after, "After ComBat (diagnosis preserved)"),
]:
    for plat in ["GPL570", "GPL6244"]:
        mask = meta["platform_id"].values == plat
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=colors[plat], label=labels[plat],
                   alpha=0.7, s=32, edgecolors="white", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Platform effect before and after ComBat (mod=diagnosis)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "combat_pca_before_after.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: {fig_path}")

# ── Secondary: PCA colored by diagnosis to see if diagnosis signal is preserved ──
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
diag_colors = {"CONTROL": "#2ca02c", "AUTISM": "#d62728",
               "PDD-NOS": "#ff7f0e", "ASPERGER'S DISORDER": "#9467bd"}
for ax, scores, var, title in [
    (axes2[0], pc_before, var_before, "Before ComBat (by diagnosis)"),
    (axes2[1], pc_after, var_after, "After ComBat (by diagnosis)"),
]:
    for diag, col in diag_colors.items():
        mask = meta["diagnosis"].values == diag
        ax.scatter(scores[mask, 0], scores[mask, 1], c=col, label=f"{diag} (n={mask.sum()})",
                   alpha=0.7, s=28, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.suptitle("PCA colored by diagnosis — before and after ComBat",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig2_path = OUT_DIR / "figures" / "combat_pca_diagnosis.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Diagnosis PCA: {fig2_path}")
