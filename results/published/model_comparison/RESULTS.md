# Classifier comparison

## Evaluation

These are exploratory nested cross-validation results on 261 training assay records.
The 66-record historical test partition was not accessed. No independent final
test estimate is produced by this experiment. The three original classifiers,
17,263 gene inputs, retained quality exclusions, and group-separated folds are preserved.

| Classifier | Accuracy | Balanced accuracy | ROC AUC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 75.1% | 74.8% | 0.823 | 76.1% | 73.4% |
| random_forest | 67.1% | 59.3% | 0.725 | 90.8% | 27.7% |
| knn | 63.6% | 57.4% | 0.644 | 82.8% | 31.9% |

Values are means over five outer folds. Each outer model was tuned using the
remaining four folds. The highest mean balanced accuracy in this comparison is
74.8% (logistic_regression); this is not evidence of a
statistically significant difference or of clinical usefulness.

## Parameters for the full-training models

- logistic_regression: `{"C": 0.001, "class_weight": "balanced"}`
- random_forest: `{"class_weight": "balanced", "max_depth": 5, "min_samples_leaf": 5}`
- knn: `{"metric": "manhattan", "n_neighbors": 5, "weights": "distance"}`

Full-training parameters were selected with five-fold CV. Their search scores
are selection scores, not substitutes for the nested estimates above. The grid
is finite; no claim of globally optimal hyperparameters is made. L2 logistic
regression, 300-tree Random Forest, and brute-force KNN were used. Class weighting
changes training emphasis without generating samples. PCA, SMOTE, supervised
gene selection, and decision-threshold tuning were not used.

## Interpretation and limitations

ASD is the positive class. Sensitivity measures the fraction of ASD records
identified; specificity measures the fraction of controls correctly classified.
Balanced accuracy averages the two. Confusion matrices use predictions from
models that did not train or tune on those outer-validation records.

The nested procedure separates this grid's tuning from its assessment, but does
not undo earlier adaptive exploration of the same data. The relatedness groups
are conservative metadata groupings, not verified unrelated-patient counts.
All cohorts are represented in training and validation: this is not external
cohort validation. Mean adjustment does not eliminate all biological confounding.

`summary.json` includes relatedness-group bootstrap intervals for the fixed OOF
predictions. These describe sample variability conditional on the saved fits;
they do not include all uncertainty from model selection or retraining. Fold SDs
are descriptive because training sets overlap.

The primary explanation analysis remains logistic regression, selected before
this grid for the XAI study. RF and KNN provide supporting classifier comparisons.
All candidates and selected parameters were retained in the original run's
`searches/` directory. This public package distributes aggregate summaries and
figures, not the search tables or fitted models. Follow the root reproduction
instructions to regenerate the complete outputs.
