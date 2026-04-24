"""Biological annotation of gene + pathway signatures.

Replaces old script 13 (SFARI enrichment on top-100 genes). The new analysis is
more meaningful: for each pathway in our discovery signature, list the constituent
genes and check which are known ASD genes according to SFARI.

This answers: "when our pipeline flags MAPK1/ERK2 Activation as predictive of ASD,
which specific genes INSIDE that pathway are SFARI-curated ASD genes?"

Inputs
------
- data/reference/sfari_genes.csv                          SFARI Q4 2025 snapshot
- data/reference/gene_sets/KEGG_2021_Human.gmt            KEGG pathway members
- data/reference/gene_sets/Reactome_2022.gmt              Reactome pathway members
- results/tables/pathway_multiseed_frequency.csv          discovery pathway signature
- results/tables/multiseed_gene_frequency.csv             discovery gene signature
- results/processed/GSE18123_combat_corrected.parquet     to filter pathway members
                                                          to genes actually expressed

Outputs
-------
- results/tables/gene_signature_sfari.csv                 per-gene SFARI annotation
- results/tables/pathway_sfari_members.csv                per-(pathway, gene) SFARI annotation
- results/tables/pathway_sfari_summary.csv                per-pathway SFARI enrichment summary

Runtime: <30 s.
"""
import sys
import time
import warnings
from pathlib import Path as _P

warnings.filterwarnings("ignore")
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = _P(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results"
GENE_SETS_DIR = REPO_ROOT / "data" / "reference" / "gene_sets"
SFARI_PATH = REPO_ROOT / "data" / "reference" / "sfari_genes.csv"

_T0 = time.time()

# ═════════════ Load SFARI reference ═════════════
print("=" * 72)
print(" Load SFARI reference and define strict subset")
print("=" * 72)

sfari = pd.read_csv(SFARI_PATH)
sfari["gene-symbol"] = sfari["gene-symbol"].astype(str).str.strip().str.upper()
score1 = set(sfari.loc[sfari["gene-score"] == 1, "gene-symbol"])
score2 = set(sfari.loc[sfari["gene-score"] == 2, "gene-symbol"])
syndromic = set(sfari.loc[sfari["syndromic"] == 1, "gene-symbol"])
sfari_strict = score1 | syndromic
sfari_broad = score1 | score2 | syndromic
print(f"  Score 1 only: {len(score1)}")
print(f"  Score 2 only: {len(score2)}")
print(f"  Syndromic:    {len(syndromic)}")
print(f"  strict = Score 1 ∪ Syndromic:           {len(sfari_strict)}")
print(f"  broad  = Score 1 ∪ Score 2 ∪ Syndromic: {len(sfari_broad)}")

sfari_info = {r["gene-symbol"]: {"score": r["gene-score"], "syndromic": r["syndromic"],
                                    "eagle": r.get("eagle")} for _, r in sfari.iterrows()}

def annotate_gene(symbol):
    """Return dict with SFARI metadata for a gene, or None-filled dict if absent."""
    info = sfari_info.get(str(symbol).upper())
    if info is None:
        return {"in_sfari_strict": False, "in_sfari_broad": False,
                 "sfari_score": None, "sfari_syndromic": None}
    score = info.get("score")
    syn = info.get("syndromic") == 1
    return {"in_sfari_strict": (score == 1) or syn,
             "in_sfari_broad": (score in (1, 2)) or syn,
             "sfari_score": int(score) if pd.notna(score) else None,
             "sfari_syndromic": bool(syn)}

# ═════════════ Gene signature annotation ═════════════
print("\n" + "=" * 72)
print(" Gene signature: SFARI annotation of individual genes")
print("=" * 72)

gene_ms = pd.read_csv(OUT_DIR / "tables" / "multiseed_gene_frequency.csv")
gene_signature = gene_ms[gene_ms["freq_5of6"] >= 0.80]["gene"].tolist()
print(f"Gene signature (freq_5of6 >= 80%): {len(gene_signature)} genes")

gene_rows = []
for g in gene_signature:
    info = annotate_gene(g)
    freq = float(gene_ms[gene_ms["gene"] == g]["freq_5of6"].iloc[0])
    gene_rows.append({"gene": g, "freq_5of6": freq, **info})

# Also annotate the top-15 by frequency even if <80%
for g in gene_ms.head(15)["gene"].tolist():
    if g not in gene_signature:
        info = annotate_gene(g)
        freq = float(gene_ms[gene_ms["gene"] == g]["freq_5of6"].iloc[0])
        gene_rows.append({"gene": g, "freq_5of6": freq, **info})

gene_df = pd.DataFrame(gene_rows)
gene_df.to_csv(OUT_DIR / "tables" / "gene_signature_sfari.csv", index=False)

n_in_strict = gene_df["in_sfari_strict"].sum()
n_in_broad = gene_df["in_sfari_broad"].sum()
print(f"\nGene-level SFARI hits (of {len(gene_df)} annotated genes):")
print(f"  strict (Score 1 ∪ Syndromic): {n_in_strict}")
print(f"  broad:  {n_in_broad}")
print(f"\n  {'gene':10s}  {'freq_5of6':>9s}  {'SFARI score':>11s}  {'Syndromic':>9s}")
for _, r in gene_df.iterrows():
    score = f"{r['sfari_score']}" if r["sfari_score"] else "—"
    syn = "yes" if r["sfari_syndromic"] else "no" if r["sfari_syndromic"] is False else "—"
    print(f"  {r['gene']:10s}  {r['freq_5of6']:>8.1%}   {score:>11s}  {syn:>9s}")

# ═════════════ Pathway signature — member gene analysis ═════════════
print("\n" + "=" * 72)
print(" Pathway signature: SFARI-annotated member genes per pathway")
print("=" * 72)

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

path_ms = pd.read_csv(OUT_DIR / "tables" / "pathway_multiseed_frequency.csv")
path_signature = path_ms[path_ms["freq_5of6"] >= 0.80]["pathway"].tolist()
print(f"Pathway signature: {len(path_signature)} pathways")
for p in path_signature:
    print(f"  - {p}")

# Load expressed gene background (intersection of all expressed genes in the discovery cohort)
X_path = OUT_DIR / "processed" / "GSE18123_combat_corrected.parquet"
X = pd.read_parquet(X_path)
expressed = set(c.upper() for c in X.columns if c not in ("diagnosis", "platform_id"))
print(f"\nGSE18123 expressed gene universe: {len(expressed)}")
print(f"  SFARI strict genes present in GSE18123 universe: {len(sfari_strict & expressed)}")
print(f"  SFARI broad genes present in GSE18123 universe:  {len(sfari_broad & expressed)}")

member_rows = []
pathway_summary_rows = []

for path in path_signature:
    if path not in merged:
        print(f"WARN: pathway {path} not in GMT library, skipping")
        continue
    members_all = [g.upper() for g in merged[path]]
    # Filter to genes actually measured in GSE18123 (the biologically relevant subset)
    members_expressed = [g for g in members_all if g in expressed]

    for g in members_expressed:
        info = annotate_gene(g)
        member_rows.append({"pathway": path, "gene": g, **info})

    # Pathway-level SFARI enrichment: Fisher exact of (SFARI ∩ members) vs background
    M = len(members_expressed)
    S_strict_in_bg = len(sfari_strict & expressed)
    S_broad_in_bg = len(sfari_broad & expressed)
    G_bg = len(expressed)

    k_strict = len([g for g in members_expressed if g in sfari_strict])
    k_broad = len([g for g in members_expressed if g in sfari_broad])

    # Fisher's exact 2x2: (in_members, not_in_members) x (in_sfari, not_in_sfari)
    def fisher(k, S, M, G):
        if k < 0 or M < 0 or G - M < 0 or S - k < 0:
            return None, None
        table = [[k, S - k], [M - k, (G - M) - (S - k)]]
        if any(cell < 0 for row in table for cell in row):
            return None, None
        or_, p = fisher_exact(table, alternative="greater")
        return (float(or_) if np.isfinite(or_) else None), float(p)

    or_s, p_s = fisher(k_strict, S_strict_in_bg, M, G_bg)
    or_b, p_b = fisher(k_broad, S_broad_in_bg, M, G_bg)

    pathway_summary_rows.append({
        "pathway": path,
        "n_members_total": len(members_all),
        "n_members_in_gse18123": M,
        "strict_hits": k_strict,
        "strict_expected": round(M * S_strict_in_bg / G_bg, 2) if G_bg else 0,
        "strict_OR": round(or_s, 3) if or_s else None,
        "strict_p": round(p_s, 4) if p_s else None,
        "broad_hits": k_broad,
        "broad_expected": round(M * S_broad_in_bg / G_bg, 2) if G_bg else 0,
        "broad_OR": round(or_b, 3) if or_b else None,
        "broad_p": round(p_b, 4) if p_b else None,
    })

    # Pretty print
    hits_strict_list = [g for g in members_expressed if g in sfari_strict]
    print(f"\n  {path}")
    print(f"    members expressed in GSE18123: {M} (of {len(members_all)})")
    print(f"    SFARI strict hits: {k_strict}/{M}  ({k_strict/M:.1%})  expected~{round(M*S_strict_in_bg/G_bg,2)}"
          + (f"  OR={or_s:.2f}, p={p_s:.4f}" if or_s else ""))
    if hits_strict_list:
        print(f"    strict hit genes: {hits_strict_list[:10]}" + ("..." if len(hits_strict_list) > 10 else ""))

members_df = pd.DataFrame(member_rows)
members_df.to_csv(OUT_DIR / "tables" / "pathway_sfari_members.csv", index=False)

summary_df = pd.DataFrame(pathway_summary_rows)
# BH correction across strict + broad pathway tests
from statsmodels.stats.multitest import multipletests
all_ps = np.concatenate([summary_df["strict_p"].dropna().values, summary_df["broad_p"].dropna().values])
if len(all_ps):
    bh_q = multipletests(all_ps, method="fdr_bh")[1]
    # map back
    n_strict = summary_df["strict_p"].notna().sum()
    summary_df["strict_q_BH"] = np.nan
    summary_df["broad_q_BH"] = np.nan
    idx_strict = summary_df.index[summary_df["strict_p"].notna()]
    idx_broad = summary_df.index[summary_df["broad_p"].notna()]
    summary_df.loc[idx_strict, "strict_q_BH"] = bh_q[:n_strict].round(4)
    summary_df.loc[idx_broad, "broad_q_BH"] = bh_q[n_strict:].round(4)
summary_df.to_csv(OUT_DIR / "tables" / "pathway_sfari_summary.csv", index=False)

# ═════════════ Figure: pathway-level SFARI enrichment ═════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_df = summary_df.copy()
fig_df["neg_log10_q_strict"] = -np.log10(fig_df["strict_q_BH"].clip(lower=1e-6))
fig_df = fig_df.sort_values("neg_log10_q_strict", ascending=True)
fig_df["pathway_short"] = fig_df["pathway"].str.replace(" R-HSA-.*", "", regex=True).str.replace("REACTOME:", "Reactome: ").str.replace("KEGG:", "KEGG: ")

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#2C7FB8" if q < 0.05 else "#B0B0B0" for q in fig_df["strict_q_BH"].fillna(1).values]
ax.barh(fig_df["pathway_short"], fig_df["neg_log10_q_strict"], color=colors, edgecolor="black", linewidth=0.6)
ax.axvline(-np.log10(0.05), color="firebrick", linestyle="--", linewidth=1.2, label="q_BH = 0.05")
ax.set_xlabel("$-\\log_{10}$(q$_{BH}$) — SFARI-strict enrichment", fontsize=10)
ax.set_title("Pathway-level SFARI-strict enrichment (Fisher's exact, BH-corrected)", fontsize=11)
for i, (q, hits) in enumerate(zip(fig_df["strict_q_BH"].values, fig_df["strict_hits"].values)):
    q_label = f"q = {q:.3f}" if pd.notna(q) else "q = —"
    ax.text(0.05, i, f"{int(hits)} SFARI hit(s)  {q_label}", va="center", fontsize=8, color="black")
ax.legend(loc="lower right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig_path = OUT_DIR / "figures" / "pathway_sfari_enrichment.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure: {fig_path}")

# ═════════════ Summary ═════════════
print("\n" + "=" * 72)
print(" SUMMARY — biological annotation")
print("=" * 72)

print(f"\nGene signature SFARI annotation:")
print(f"  {len(gene_df)} genes annotated")
print(f"  SFARI strict hits: {int(n_in_strict)}  ({100*n_in_strict/max(1,len(gene_df)):.1f}%)")

print(f"\nPathway signature SFARI-member analysis:")
print(f"  {len(path_signature)} pathways tested")
print(f"{'pathway':<65s} {'members':>8s} {'strict':>8s} {'OR':>6s} {'q_BH':>8s}")
for _, r in summary_df.iterrows():
    p_short = r["pathway"][:60] + ".." if len(r["pathway"]) > 62 else r["pathway"]
    or_s = f"{r['strict_OR']:.2f}" if pd.notna(r["strict_OR"]) else "—"
    q_s = f"{r['strict_q_BH']:.3f}" if pd.notna(r.get("strict_q_BH")) else "—"
    marker = " *" if pd.notna(r.get("strict_q_BH")) and r["strict_q_BH"] < 0.05 else "  "
    print(f"  {p_short:<63s} {r['n_members_in_gse18123']:>5d}   {r['strict_hits']:>5d}   {or_s:>5s}  {q_s:>6s}{marker}")

n_sig_path = (summary_df.get("strict_q_BH") < 0.05).sum() if "strict_q_BH" in summary_df else 0
print(f"\n{n_sig_path} pathway(s) significantly enriched for SFARI strict genes (q_BH < 0.05)")

print(f"\nRuntime: {time.time() - _T0:.1f}s")
print(f"Outputs under {OUT_DIR}")
