# Multi-XAI Consensus for ASD Biomarker Discovery

Explainable AI (XAI) comparison pipeline for identifying robust gene expression biomarkers in Autism Spectrum Disorder from blood transcriptomic data, and testing their cross-cohort reproducibility at both the gene level and the pathway level.

## Key findings

This thesis produces two parallel analysis tracks (gene-level and pathway-level) with rigorous internal validity and cross-cohort external validation on 3 independent datasets.

### Gene-level track (primary): cohort-specific, fails external replication

- **GSE18123 (n=285, whole blood)**: robust 4-gene signature — **SPG20, TIGD7, CES1, EGR1** (5/6 multi-XAI consensus, null-resistant, multi-seed stable, config-invariant).
- Cross-cohort transfer (GSE18123 → GSE25507 / GSE42133 / GSE6575) is at chance (bal_acc 0.44 – 0.54 in all 6 LR + RF transfers). 
- 0/4 primary genes appear in the multi-seed consensus of any external cohort — including GSE6575 which matches tissue and platform exactly.
- Other cohorts produce their own disjoint signatures (GSE25507: RAB8A/LIX1/HIST1H2BG; GSE42133: MRRF/AK3/TFAP2A/TSHZ2) that likewise do not replicate elsewhere. The gene-level story is a clean, rigorous **negative external-validation finding**.

### Pathway-level track (secondary): partial cross-cohort replication + real biology

The same pipeline run on ssGSEA pathway activation scores (KEGG + Reactome, ~2000 pathways) produces a 6-pathway multi-seed signature:

- REACTOME: **MAPK1 (ERK2) Activation** — canonical RASopathy pathway, long-established ASD link
- REACTOME: **GPI-anchored Protein Synthesis** — PIGA/PIGN/PIGT mutations cause ID + autism
- REACTOME: Uptake and Function of Diphtheria Toxin
- REACTOME: **Nuclear Events (Kinase and Transcription Factor Activation)**
- KEGG: Systemic Lupus Erythematosus (immune gene module)
- REACTOME: **Serine Biosynthesis** — D-serine is an NMDA receptor co-agonist

**Three findings at the pathway level that the gene-level track does not deliver:**

1. **Transfer above chance on one external cohort.** On GSE42133, the 6-pathway signature achieves bal_acc 0.608 (LR) / 0.617 (RF), where the 4-gene signature gives 0.457 / 0.486.
2. **Cross-cohort pathway stability.** REACTOME: Serine Biosynthesis appears in the top-200 of GSE42133 AND the top-19 of GSE6575 stability rankings — the only pathway with cohort-independent selection signal.
3. **Biological grounding via SFARI enrichment.** The **Nuclear Events (Kinase/TF Activation) pathway is statistically enriched for SFARI Score 1 + Syndromic ASD genes**: 6 of 61 expressed members are SFARI-curated (OR = 4.74, Fisher p = 0.0026, **q_BH = 0.016** after BH correction across all pathway tests). The 6 SFARI hits — **CHD4, EP300, MEF2C, PPP2CA, PPP2R5D, RPS6KA3** — are canonical neurodevelopmental genes (MEF2C haploinsufficiency syndrome, Coffin-Lowry, Jordan's syndrome, Rubinstein-Taybi). KEGG: Systemic Lupus Erythematosus additionally contains GRIN2A + GRIN2B (NMDA receptor subunits; major ASD genes), and MAPK1 contains PTPN11 (Noonan syndrome).

The XAI pipeline, blind to SFARI during training, independently identifies a pathway significantly enriched for canonical SFARI ASD genes.

### Methodological conclusion

Rigorous internal validity is necessary but not sufficient for cross-cohort **gene-level** biomarker generalization in blood ASD transcriptomics — the 4-gene discovery signature is cohort-specific across 4 independent cohorts. The same pipeline at the **pathway level** (a) identifies published ASD-relevant pathways, (b) is statistically significantly enriched for canonical SFARI ASD genes in one pathway (q_BH = 0.016), (c) transfers to one of three external cohorts above chance, and (d) has one pathway (Serine Biosynthesis) that shows cross-cohort stability. Pathway-level abstraction is the more defensible unit of analysis for ASD blood transcriptomics at current sample sizes.

## Pipeline

```
scripts/pipeline/
├── 00_base_preprocessing.py                    # GEO download + probe-to-gene mapping for GSE18123
├── 01_combat_with_mod_diagnosis.py             # ComBat batch correction on GSE18123
├── 02_stability_selection.py                   # Gene-level stability + k-sensitivity sweep (v2 Step 9) + leakage-safety demo
├── 03_lr_vs_rf_xai_consensus.py                # Gene-level LR + RF multi-XAI consensus
├── 04_permutation_null.py                      # Null tests for stability and consensus
├── 05_multiseed_robustness.py                  # 30-run multi-seed robustness (10 seeds × 3 tree counts)
├── 06_pathway_analysis.py                      # Pathway-level track: ssGSEA + stability + XAI + null + multi-seed
├── 07_external_validation.py                   # Unified: gene + pathway signatures tested on 3 external cohorts
├── 08_biological_annotation.py                 # SFARI annotation of gene + pathway-member signatures
└── README.md                                   # Execution order, dependencies, runtimes

scripts/appendix/
├── age_only_control.py                         # Residualization-config sensitivity (age only)
└── no_resid_control.py                         # Residualization-config sensitivity (no residualization)
```

See `scripts/pipeline/README.md` for per-script input/output/runtime details.

## Setup

```bash
pip install -r requirements.txt
```

Then run the pipeline in numerical order:

```bash
python scripts/pipeline/00_base_preprocessing.py
python scripts/pipeline/01_combat_with_mod_diagnosis.py
# Scripts 02-05 are the gene-level track (independent after 01 completes).
# Script 06 is the pathway-level track (depends on 01).
# Script 07 unifies external validation for both tracks (depends on 03, 05, 06).
# Script 08 is the biological annotation layer (depends on 05, 06).
```

All scripts resolve paths relative to the repo root; run them from any working directory.

## Data

| Dataset | Samples | Tissue | Platform | Role |
|---------|---------|--------|----------|------|
| GSE18123 | 285 (115 ctrl, 72 aut, 24 asp, 74 PDD-NOS) | Whole blood | GPL570 + GPL6244 | Discovery |
| GSE25507 | 146 (64 ctrl, 82 aut) | Lymphocytes | GPL570 | External validation |
| GSE42133 | 147 (56 ctrl, 91 ASD) | Leukocytes | GPL10558 (Illumina) | External validation |
| GSE6575 | 47 ASD+ctrl (35 ASD, 12 typical; 9 MR/DD excluded) | Whole blood | GPL570 | Like-for-like replication (tissue + platform match to GSE18123) |

GEO SOFT files are cached in `data/raw/` (gitignored for size). Processed matrices in `data/processed/` (GSE18123 only) and `results/processed/` (joint ComBat parquets + ssGSEA pathway scores).

SFARI Gene reference (Q4 2025 snapshot) bundled at `data/reference/sfari_genes.csv` with attribution. KEGG_2021_Human and Reactome_2022 gene sets bundled at `data/reference/gene_sets/` (downloaded from Enrichr).

## Results

Main outputs in `results/`:

- `results/figures/` — PCA plots, consensus figures, robustness figures, external validation comparison, pathway consensus bar chart
- `results/tables/` — per-gene, per-run, per-cohort numerical tables, pathway signature + SFARI annotation, external validation transfer summaries
- `results/processed/` — ComBat-corrected gene matrices (single, 3-cohort joint, 4-cohort joint) + ssGSEA pathway score matrices

## Project status

Code pipeline is complete. Thesis writeup is pending. Biological interpretation chapter: gene-level (negative/cohort-specific) + pathway-level (positive in one cohort + SFARI-enriched Nuclear Events pathway).
