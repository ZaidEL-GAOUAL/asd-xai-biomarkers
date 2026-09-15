# ASD blood expression and explainable classification

Pipeline for preparing whole-blood expression data, comparing logistic regression,
Random Forest and KNN, and examining gene SHAP explanations through biological
groups. Grouping changes the explanation summary, not the model's predictions.

## Pipeline

Run the numbered steps in order. Setup and required input files are described in
[scripts/pipeline/README.md](scripts/pipeline/README.md).

```text
scripts/pipeline/
  00_download_data.py          verify metadata and download raw arrays
  01_normalize_arrays.R       normalize each array with frozen RMA
  02_prepare_data.py          match genes, apply QC and prepare the split
  03_download_references.py   obtain the frozen biological collections
  04_train_models.py          compare logistic regression, RF and KNN
  05_gene_explanations.py     calculate gene SHAP and Hallmark summaries
  06_gene_set_coverage.py     check reference coverage
  07_compare_explanations.py  compare biological-group explanations

src/                         shared implementation
data/                        source configurations and reference records
results/published/           saved aggregate results and analysis plots
scripts/notebooks/           four notebooks for reading those results
scripts/tools/               result checks and notebook/plot generation
```

## Setup

Use Python 3.13 and the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Normalization additionally requires R 4.5 and Bioconductor 3.22. Follow the
[execution guide](scripts/pipeline/README.md) before downloading or processing data.
The numbered Python scripts locate the repository and run its shared code.

## Read existing results

Open the notebooks in numerical order under `scripts/notebooks/`. They display
saved aggregate results only; executing them does not retrain the models.
New analysis runs write under `results/`, separately from `results/published/`.

The recorded study uses 327 assay records from GSE18123 and GSE6575, with
261 development records, 66 historical internal test records and 17,263 shared
genes. These are not a verified count of unrelated people. Logistic regression
had 75.1% mean outer-fold accuracy and 74.8% balanced accuracy in the saved run.
Biological grouping did not establish consistently better explanations.

The test partition was examined historically. This is exploratory research,
not a clinically validated diagnostic tool. The model search and full data
processing were not repeated when simplifying the folder layout. Saved JSON
provenance retains the paths and hashes of the original run; a fresh run records
the current paths and code hashes. Do not mix old model files with a new run.

## Sources and scope

The six source manifests are required input configurations for locating and
checking the original data and biological-reference versions. Raw patient-level
files, fitted models and restricted gene-set downloads are not included.
See [sources and method references](data/reference/REFERENCES.md) and
[reference terms](data/reference/collections/README.md).

This folder contains no progress report, literature-review document or Draw.io
diagram. Only actual analysis plots accompany the saved results. No open-source
license is granted by this copy; the author must choose a code license before
offering licensed reuse. Reference data retain their providers' terms.

## AI assistance

AI tools assisted with explanations of concepts, literature identification,
programming and debugging, execution of computational analyses, and documentation.
The author is responsible for checking the sources, methods and interpretations
and for meeting institutional disclosure requirements before submission.
