# Pipeline scripts

Numbered execution order, runtimes, and I/O for the full pipeline (v3 structure: gene-level track + pathway-level track + unified external validation).

## Dependency graph

```
00 (base preprocessing)
 └── 01 (ComBat on GSE18123)
      ╔═══════════ GENE-LEVEL TRACK (primary) ═══════════╗
      ├── 02 (gene-level stability + leakage-safe performance)
      ├── 03 (gene LR+RF multi-XAI consensus)
      ├── 04 (gene permutation null tests)
      ├── 05 (gene multi-seed robustness, 30 runs)
      ╚══════════════════════════════════════════════════╝
      ╔═══════════ PATHWAY-LEVEL TRACK (secondary) ═══════╗
      ├── 06 (ssGSEA + stability + XAI + null + multi-seed on pathways)
      ╚══════════════════════════════════════════════════╝
      ╔═══════════ VALIDATION + INTERPRETATION ═══════════╗
      ├── 07 (unified: gene + pathway signatures tested on GSE25507, GSE42133, GSE6575)
      └── 08 (SFARI annotation of gene + pathway-member signatures)
      ╚══════════════════════════════════════════════════╝
```

Scripts 02-05 (gene track) and 06 (pathway track) depend only on the ComBat parquet produced by 01; they can be run in parallel after 01 completes. Script 07 depends on 03, 05, 06 outputs (uses their signatures). Script 08 depends on 05 and 06 outputs.

Sensitivity-check scripts at `scripts/appendix/` (age-only residualization control, no-residualization control) are not part of the main pipeline; they can be run optionally to validate robustness of the discovery gene signature to residualization-config choices.

## I/O and runtime

| # | Script | Inputs | Outputs | Runtime |
|---|--------|--------|---------|---------|
| 00 | `00_base_preprocessing.py` | GEO (GSE18123) | `data/processed/GSE18123_commonGenes_2platforms.parquet` | 2 min (idempotent; `FORCE=1` re-downloads) |
| 01 | `01_combat_with_mod_diagnosis.py` | base parquet | `results/processed/GSE18123_combat_corrected.parquet`, PCA figures | <1 min |
| 02 | `02_stability_selection.py` | ComBat parquet | `results/tables/signature_stability_combat.csv`, `gene_stability_k{50,80,100,150}.csv`, `gene_stability_k_sensitivity_summary.csv`, `staged_performance_*_infold.csv`, figure | <1 min |
| 03 | `03_lr_vs_rf_xai_consensus.py` | ComBat parquet | `results/tables/lr_vs_rf_consensus.csv`, `signature_stability_combat.csv`, figure | ~3 min |
| 04 | `04_permutation_null.py` | ComBat parquet | `results/tables/permutation_null_*.csv`, figure | ~25 min |
| 05 | `05_multiseed_robustness.py` | ComBat parquet | `results/tables/multiseed_*.csv`, figures | ~18 min |
| 06 | `06_pathway_analysis.py` | ComBat parquet, bundled GMTs | `results/processed/GSE18123_pathway_scores.parquet`, `results/tables/pathway_stability.csv`, `pathway_multiseed_frequency.csv`, `pathway_null_summary.csv`, figure | ~10 min |
| 07 | `07_external_validation.py` | ComBat parquet, pathway signature, GSE25507/42133/6575 GEO, bundled GMTs | `results/processed/four_cohort_combat.parquet`, `four_cohort_pathway_scores.parquet`, `results/tables/gene_external_validation.csv`, `pathway_external_validation.csv`, `pathway_signature_overlap.csv`, figure | ~3 min (cached ssGSEA) / ~30 min (fresh) |
| 08 | `08_biological_annotation.py` | SFARI CSV, bundled GMTs, gene + pathway multi-seed CSVs | `results/tables/gene_signature_sfari.csv`, `pathway_sfari_members.csv`, `pathway_sfari_summary.csv` | <1 s |

## Running a single script

Scripts resolve all paths relative to the repo root, so they can be run from any working directory:

```bash
python scripts/pipeline/01_combat_with_mod_diagnosis.py
# or
cd /tmp && python /abs/path/to/scripts/pipeline/01_combat_with_mod_diagnosis.py
```

Script 00 is idempotent: it skips regeneration if `data/processed/GSE18123_commonGenes_2platforms.parquet` already exists. Set `FORCE=1` to re-download and reprocess from raw GEO data.

Script 06 caches its ssGSEA output at `results/processed/GSE18123_pathway_scores.parquet`. Script 07 caches its 4-cohort ssGSEA output at `results/processed/four_cohort_pathway_scores.parquet`. Delete those files to force recomputation.

## Required packages

- `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `imbalanced-learn`, `matplotlib`
- `inmoose` — `pycombat_norm` (ComBat batch correction)
- `shap` — `LinearExplainer` (LR) and `TreeExplainer` (RF)
- `GEOparse` — fetching GSE18123, GSE25507, GSE42133, GSE6575
- `gseapy` — `ssgsea` (pathway activation scoring) with local KEGG + Reactome GMT files
- `lime` — legacy (only used by appendix scripts; main pipeline dropped LIME in favor of 6-method consensus)
- `CovariateResidualizer` class in `../../src/residualize.py`

## Script-by-script summary

1. **00** Load GSE18123 from GEO, map probes to gene symbols for both platforms (GPL570 direct, GPL6244 via `gene_assignment` parsing), intersect to common genes, log2-transform, save parquet.
2. **01** ComBat batch correction with `mod=diagnosis` preserving biological variance. Before/after PCA by platform and by diagnosis.
3. **02** Gene-level stability selection + k-sensitivity sweep (v2 Step 5 + v2 Step 9) + leakage-safety demo. Section A runs SelectKBest stability ranking at k=100 (primary, feeds downstream scripts via `signature_stability_combat.csv`). Section B sweeps k ∈ {50, 80, 100, 150} and computes pairwise + all-way top-25 overlap. Section C reproduces the in-fold-vs-outside-fold leakage-safety comparison from old script 02.
4. **03** Gene-level cross-model XAI consensus: LR with (PFI, SHAP, Coef) and RF with (PFI, TreeSHAP, Impurity). 5/6 and 6/6 consensus voting. Produces the GSE18123 signature `signature_stability_combat.csv` + 4-gene consensus.
5. **04** 500 permutations of the stability pipeline (reveals that stability-count alone is not null-supported on p-greater-than-n data), then 50 permutations of the cross-model XAI consensus (shows 5/6 majority consensus is null-resistant; 6/6 strict is more borderline).
6. **05** Gene-level 30-run robustness (10 seeds × {200, 500, 1000} trees) on GSE18123. Produces per-gene inclusion frequency table.
7. **06** Pathway-level track: ssGSEA converts 285 × 17,707 gene matrix to 285 × ~2000 pathway matrix (KEGG_2021_Human + Reactome_2022 bundled in `data/reference/gene_sets/`). Then applies the gene-level v2-style pipeline to pathways: stability ranking, cross-model XAI consensus (6 methods), quick null test (50 perms), multi-seed robustness (5 seeds × 2 tree counts). Produces 6-pathway multi-seed signature at freq_5of6 ≥ 80%.
8. **07** Unified external validation: joint 4-cohort ComBat (GSE18123 + GSE25507 + GSE42133 + GSE6575), joint ssGSEA on the full 625-sample matrix, model transfer of both the 4-gene gene signature and the 6-pathway signature from GSE18123 to each external cohort via SMOTE + LR + RF, cohort-specific pathway ranking overlap with the discovery signature. Produces side-by-side bar chart comparing gene-level vs pathway-level transfer performance.
9. **08** Biological annotation: SFARI Q4 2025 snapshot cross-referenced against (a) each gene in the gene signature and (b) each member gene of each pathway in the pathway signature. Fisher's exact test of SFARI enrichment per pathway, with BH correction. Identifies which pathways contain canonical ASD genes.

## Known methodological choices

- SHAP background set is `X_test` for all XAI runs (standard practice uses a training sample; ranking is not materially affected).
- `pd.get_dummies(drop_first=True)` in `CovariateResidualizer._design_matrix` could fail with extremely unbalanced folds; on GSE18123 class balance this does not occur in practice.
- `build_platform_matrix` in `asd_pipeline_utils.py` handles GPL570 and GPL6244. GPL10558 (Illumina) is handled inline in script 07 since it is only used once.
- All RF calls are seeded via the loop variable, so changing `SEEDS` in scripts 05, 06, 07 produces a different but reproducible robustness picture.
- Gene sets (KEGG_2021_Human, Reactome_2022) are bundled as local GMT files in `data/reference/gene_sets/` to avoid runtime dependency on Enrichr's download API (gseapy 1.1.13 has a download bug on Python 3.13).
- v2 Step 9 (k-sensitivity sweep at k ∈ {50, 80, 100, 150}) **restored** in script 02. Measured on GSE18123: 18 of top-25 genes (72%) are shared across all 4 k values — stability ranking is k-robust. Pairwise top-25 overlaps range from 72% (k=50 vs k=150) to 96% (k=80 vs k=100).
