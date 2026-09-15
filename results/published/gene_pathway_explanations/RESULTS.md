# Gene and pathway explanations

## Analysis population and estimator

This exploratory analysis explains 261 existing training
records using five logistic-regression models. Each record was explained only
by its outer-fold model, which did not use that record for fitting or selecting
hyperparameters. No held-out test files were read or reevaluated. The folds share
training records and are not independent replications. This internal exercise
does not establish transportability to a new cohort or clinical usefulness.

SHAP LinearExplainer used the complete corresponding outer-training reference
population with an Independent masker. Values explain model log odds, not changes
in ASD risk caused by genes. This assumes an interventional feature treatment;
correlated gene expression can complicate the interpretation. The exact linear
formula and reconstruction of each model decision were verified numerically.

## Main observations

- Gene universe: 17,263; Hallmark-covered genes:
  4,211 (24.4%).
- The 50 Hallmark groups cover only part of the model.
  Unassigned genes account for 76.5%
  of the total absolute gene-attribution mass in these explanations.
- Mean pairwise top-20 gene Jaccard: 0.065; restricting
  gene selection to covered genes: 0.076.
- Mean pairwise top-5 pathway Jaccard: 0.335.
  These raw gene/pathway values are not directly comparable because the
  universes and top-list lengths differ.
- Mean proportion of top-20 covered genes appearing in the top-five pathways:
  24.0%. Across the unrestricted top-20
  genes the corresponding proportion is 2.0%.
- Real pathway agreement is at the 59.5th
  exploratory percentile of 100 random relabellings;
  covered-gene overlap is at the
  32.0th percentile.

The leading pooled pathway labels are: HALLMARK_HEME_METABOLISM, HALLMARK_INTERFERON_GAMMA_RESPONSE, HALLMARK_ALLOGRAFT_REJECTION, HALLMARK_G2M_CHECKPOINT, HALLMARK_IL2_STAT5_SIGNALING.
Their ranking is descriptive and does not prove that these pathways are
activated, causal, specific to ASD, or relevant to brain tissue.

## What was compared

Each gene's signed contribution is divided equally between the Hallmark groups
containing it. Summing all groups and the unassigned-gene residual reconstructs
the full gene attribution. These are post-hoc allocated summaries, **not
independent pathway Shapley values**. Agreement between the two levels is partly
built into this construction and is not independent biological validation.

Pathway rankings use the mean absolute value of the *signed sum*. Separate
columns report unsigned contribution mass, within-sample cancellation and the
largest gene's share. This distinguishes aligned model contributions from
opposing model contributions and summaries dominated by a single gene; it is
not evidence of biological coordination. No threshold
defines a medically acceptable overlap, cancellation or driver share.

The random controls globally permute covered-gene labels in the membership
matrix, using the same permutation across all five folds. They preserve every
pathway's size, pairwise intersection sizes and the distribution of membership
degrees. They do not preserve expression correlation within groups or each
particular gene's membership degree. Percentiles are descriptive exploratory
comparisons under that particular null, not clinical validation or formal
multiple-testing-adjusted significance results. No patient data were generated.

## Scope and limitations

This is the primary LR explanation analysis. Random Forest and KNN remain
supporting classifier comparisons; their explanations have not been computed
and cannot be inferred from LR. Model accuracy is unchanged by this analysis.

Blood-cell composition, cohort effects and correlated predictors can influence
these associations. Pooled fold rankings mix explanations from five fitted
models with different training-reference values. The original training data
have also been used in earlier exploratory research; this is a prospectively
recorded analysis within this run, not a preregistered independent study.

Exact-symbol SFARI annotation was run using the frozen reference. Among the pooled top 20 genes, 1 matched a SFARI entry and 1 had score 1 or 2. The corresponding full-universe counts were 1188 and 874, out of 17,263 genes. These are descriptive counts, not an enrichment test. Missing scores remain missing; syndromic status is separate. Unmatched symbols may include aliases and are not evidence of no ASD relationship. SFARI genetic-risk evidence is not a list of validated blood-expression biomarkers, and a match does not validate a model or establish causation.

## Original run files

The inventory below describes the full recorded experiment. This public package
includes aggregate JSON summaries and figures only; CSV tables and sample-level
arrays are not bundled. The root reproduction instructions regenerate them.

- `protocol.json`: analysis settings, input/model checksums, reference and scope.
- `summary.json`: observations, descriptive control percentiles and audit.
- `gene_rankings.csv`, `pathway_rankings.csv`: pooled outer-validation rankings.
- `fold_gene_rankings.csv`, `fold_pathway_rankings.csv`: fold-specific rankings.
- `gene_pathway_overlap.csv`: foldwise top-list overlap and gene membership.
- `fold_stability.csv`: pairwise top-list Jaccard values.
- `random_controls.csv`: every seeded random-control result.
- `sample_pathway_drivers.csv`: drivers of each sample's five strongest summaries.
- `validation_predictions.csv`: outer-validation predictions and reconstructed logits.
- `attributions/fold_*.npz`: full-precision sample-level arrays and feature IDs.
- `figures/`: static descriptive figures.

No new methodological or ASD-specific novelty claim follows from these results
alone. References and how they were used are recorded in the parent research
documentation.
