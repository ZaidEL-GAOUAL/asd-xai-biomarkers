# Biological gene-group comparison

Completed using the same saved gene SHAP values for 261 development records. No models were fitted, no new gene SHAP values were calculated, and no test data were accessed.

## Coverage

| Collection | Groups | Genes covered | Gene coverage | Absolute gene SHAP mass covered |
| --- | ---: | ---: | ---: | ---: |
| Hallmark | 50 | 4,211 | 24.4% | 23.5% |
| Reactome | 1,839 | 10,025 | 58.1% | 56.8% |
| Canonical pathways | 4,114 | 11,961 | 69.3% | 68.2% |
| GO Biological Process | 7,535 | 14,677 | 85.0% | 84.2% |

These are annotation/attribution coverage percentages, not accuracy or biological validation. GO Biological Process is a functional ontology collection, not exclusively reaction pathways. Canonical pathways contain Reactome; these collections are not independent replications.

## Primary comparison: all nonempty groups

| Collection | Groups | Real top-five agreement | Random mean | Descriptive percentile | Covered top-gene overlap: real / random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hallmark | 50 | 0.335 | 0.325 | 59.5 | 24.0% / 26.5% |
| Reactome | 1,839 | 0.800 | 0.620 | 95 | 39.0% / 40.1% |
| Canonical pathways | 4,114 | 0.595 | 0.535 | 73.5 | 30.0% / 29.1% |
| GO Biological Process | 7,535 | 0.464 | 0.548 | 26.5 | 26.0% / 36.1% |

## Predefined size sensitivity: 15–500 measured genes

| Collection | Groups | Real top-five agreement | Random mean | Descriptive percentile | Covered top-gene overlap: real / random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hallmark | 50 | 0.335 | 0.325 | 59.5 | 24.0% / 26.5% |
| Reactome | 1,153 | 0.500 | 0.505 | 54 | 17.0% / 13.6% |
| Canonical pathways | 2,531 | 0.533 | 0.422 | 86.5 | 12.0% / 12.9% |
| GO Biological Process | 3,993 | 0.349 | 0.398 | 35 | 10.0% / 13.8% |

## Interpretation boundaries

Compare each real grouping with its own matched random reference. Do not rank collections by raw Jaccard alone: they differ in group count, size, overlap and gene coverage. The covered top-20 gene list also differs between collections. The 100 relabellings yield descriptive percentiles, not formal p-values or multiple-testing-adjusted findings. A missing advantage is not proof of equivalence.

The size range is a sensitivity rule borrowed from familiar GSEA practice, not a medical or XAI validity threshold. Both variants are retained. Shared and identical labels can inflate apparent repeatability; random controls preserve those structural features. They do not preserve expression correlation or each gene's original number of memberships.

Groups are signed sums of allocated gene SHAP values. Each gene's contribution is shared across its containing groups, so changing the collection changes the allocation denominator. Opposing contributions can cancel; driver concentration and cancellation are saved in rankings.json. These are not independently computed pathway Shapley values or proof of biological activation.

The same five fixed models and same records are used across collections. Across folds, however, models have different selected settings and validation records. Thus stability includes model refitting/selection and sample variation. Pooled magnitudes mix model log-odds scales and do not define a validated biomarker panel. Earlier adaptive exploration and blood confounding remain.

## Files and reproduction

Each original collection/rule folder contained summary.json, rankings.json,
membership.json, fold_top_lists.json, fold_stability.json, random_controls.json
and allocations.npz. This public package omits membership.json and the sample-level
allocations.npz arrays. Identical Hallmark primary/sensitivity memberships reuse
the same results, explicitly recorded.

After regenerating the inputs described in the root reproduction instructions,
run `python scripts/pipeline/07_compare_explanations.py --output results/group_comparison_recheck`
from the repository root. Existing outputs are not overwritten. Source code,
plan and input checksums are preserved. Review canonical-reference redistribution
terms before releasing memberships.

See [the comparison plan](../../../src/asd_blood/comparison_plan.md), [references](../../../data/reference/REFERENCES.md), and [additional reference provenance](../../../data/reference/collections/README.md).
