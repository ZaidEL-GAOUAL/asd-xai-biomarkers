# Multi-XAI Consensus for ASD Biomarker Discovery

Explainable AI (XAI) comparison pipeline for identifying robust gene expression biomarkers in Autism Spectrum Disorder from blood microarray data.

## Key Findings

- **PRPF38B** validated as XAI consensus gene across 3 independent datasets (GSE18123, GSE25507, GSE42133)
- **PNN** validated in 2/3 datasets
- 8 genes reproducible across >= 2 datasets: ATP6AP2, C2orf57, DEK, IDI1, KIF5B, PNN, PRPF38B, S100A6

## Pipeline

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_loading_and_metadata` | Load GSE18123 from GEO |
| 2 | `02_probe_mapping_and_gene_matrix` | Map probes to genes, merge GPL570 + GPL6244 |
| 3 | `03_qc_and_preprocessing` | Quality control |
| 4 | `04_baseline_models` | Binary + multiclass CV baselines |
| 5 | `05_signature_extraction` | SelectKBest stability analysis |
| 6 | `06_smote_pfi` | SMOTE + Permutation Feature Importance |
| 7 | `07_synthetic_generation` | SMOTE augmentation + comparison |
| 8 | `08_xai_comparison` | 5 XAI methods: PFI, SHAP, LIME, Coefficients, Ablation |
| 9 | `09_robustness_analysis` | Sensitivity to feature selection threshold k |
| 10 | `10_external_validation` | Validation on GSE25507 + GSE42133 |

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks sequentially from `notebooks/` directory.

## Data

- **GSE18123**: 285 blood samples, 2 platforms (GPL570 + GPL6244), 4 diagnosis groups
- **GSE25507**: 146 blood samples, GPL570, binary (autism/control)
- **GSE42133**: 147 blood samples, GPL10558 (Illumina), binary (ASD/control)
