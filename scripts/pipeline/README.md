# Running the pipeline

## Review without downloading data

The four notebooks under `scripts/notebooks/` display included aggregate results.
They do not access source expression, diagnosis tables, models or the historical
test partition. The stored figures and tables are sufficient to follow the findings.

For a rerun, use Python 3.13 with the recorded package versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Software test suites and administrative release tools are not bundled in this
pipeline-only copy. They remain in the complete working package.
The completed study's numerical-verification reports remain historical evidence
under `results/published/`.

The single root `requirements.txt` specifies the recorded analysis environment.

## Full rerun from public sources

Run the following stages in order from the repository root unless stated otherwise.
This is a deliberate computational rerun, not required to open the notebooks.
It was not repeated during release preparation. Historical results can be compared
with a rerun, but compressed-file timestamps and run metadata can change byte hashes.

### 1 Source metadata and raw arrays

Obtain the three series-matrix files for GSE18123/GPL570, GSE18123/GPL6244 and
GSE6575/GPL570, plus the **GEO annotation** files for GPL570 and GPL6244. Do not
substitute the much larger platform-family SOFT downloads: these are different files.
Expected names, source links and decompressed SHA256 values are recorded in
`data/source_manifest.json`. Place them in `data/whole_blood/source/` under those names.

```bash
python scripts/pipeline/00_download_data.py --download
```

The preflight stops on a missing or changed metadata source. The program then
constructs the acquisition manifest and grouped partition and retrieves its
referenced CEL arrays. Source diagnoses and eligibility rules are defined in
the code. Do not replace the recorded metadata with an unverified newer snapshot.

### 2 Normalization and preparation

Use an isolated R 4.5 / Bioconductor 3.22 environment. Review the installer before
running it; package installation and raw-array processing can take substantial time.

```bash
Rscript src/install_r_preparation.R
Rscript scripts/pipeline/01_normalize_arrays.R GPL570
Rscript scripts/pipeline/01_normalize_arrays.R GPL6244
python scripts/pipeline/02_prepare_data.py
```

The scripts preserve the grouped partition, perform per-array frozen RMA,
match Entrez genes and apply the recorded QC policy. Preparation necessarily
creates training and test matrices; the later active analyses read development
inputs only. The globally prepared/scaled matrix is not used as a substitute for
the fold-fitted transformations in model training.

### 3 Biological references

```bash
python scripts/pipeline/03_download_references.py
```

This retrieves the four versioned MSigDB GMT files only and verifies their exact
hashes. Reference licenses remain those of the providers. It sends no project data.
For the optional SFARI context, obtain the frozen 2026 Q2 human-gene download and
follow `data/reference/sfari/source_manifest.json`: project its recorded
columns to `human_genes.csv`, preserving order and missing values. The source
manifest contains raw and projected hashes. A current, different SFARI release
is not an exact reproduction and must be documented separately.

### 4 Classification and gene explanations

```bash
python scripts/pipeline/04_train_models.py --jobs 2
python scripts/tools/verify_training.py
python scripts/pipeline/05_gene_explanations.py --sfari-csv data/reference/sfari/human_genes.csv
```

If the frozen SFARI file is unavailable, omit `--sfari-csv`; this omits the
descriptive SFARI annotation, not the gene SHAP calculation. Record that deviation
when comparing outputs. Model search uses 54 settings with fold-fitted preprocessing.
No SMOTE, new feature search, threshold optimization or test reevaluation is included.

### 5 Biological-group comparison

From the repository root:

```bash
python scripts/pipeline/06_gene_set_coverage.py
python scripts/pipeline/07_compare_explanations.py
python scripts/tools/verify_collections.py
python scripts/tools/plot_collections.py
```

These commands use the newly generated development inputs/explanations, not the
published aggregate summaries. Existing analysis destinations are never overwritten
by the model/XAI commands. Use a new working copy for a full rerun rather than mixing
new outputs with partial old results. The original protocol expects the original
gene universe and record counts and deliberately fails if they differ.

Fresh outputs are written to `results/model_comparison/`,
`results/gene_pathway_explanations/`, `results/pathway_coverage/` and
`results/group_comparison/`. The distributed aggregate results remain separate
under `results/published/`. Their historical provenance names the paths and
source hashes used before the folder reorganization. A complete fresh run
generates its own provenance; the verification tools target fresh outputs.

To refresh only the four review notebooks from the unchanged saved aggregates:

```bash
python scripts/tools/build_review_notebooks.py --execute
```

This command does not download data or train models.

## Reporting boundaries

The shared cohorts come from two GEO series, not three independent studies.
The 66-record test partition has already been examined historically. Nested
development evaluation does not erase earlier exploration. Reproduction confirms
implementation consistency, not clinical validity or biological causation.
