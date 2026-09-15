# Gene-set explanation comparison

Specified on 8 September 2026 before additional collection coverage and XAI
comparison results. This is an exploratory extension of previously examined
data and Hallmark results, not a preregistered independent study.

## Fixed scope

Compare human MSigDB 2026.1.Hs Hallmark, Reactome, combined canonical pathways
(C2:CP), and GO Biological Process (C5:GO:BP). The latter is a biological-process
ontology collection, not exclusively pathways. Canonical pathways contain
Reactome and are not independent evidence. Do not select a collection by its
explanation scores or search additional collections until a desired result occurs.

Reuse the five existing outer-validation gene SHAP arrays for all 261 development
records. All four collections receive identical gene contributions, record IDs,
models and backgrounds within a fold. Do not fit classifiers, tune settings,
calculate new gene SHAP values, or open any prepared training/test expression or
label files. Validate the saved additive reconstruction and freeze input hashes.

## Mapping and summaries

Match exact Entrez IDs to the frozen gene universe. Report total coverage and
the share of absolute gene attribution belonging to covered genes. Missing
members are absent measurements, not zero expression. Exclude only zero-match
groups from the primary scoring. Report small groups, repeated memberships and
groups that become identical after mapping; retain nonempty duplicate labels
in the primary analysis to preserve the published reference definitions.

Apply the existing signed allocation rule: divide each gene's SHAP contribution
equally among the included groups containing it. Keep uncovered genes in a
separate residual. Rank groups by mean absolute signed contribution. This is
an allocated gene-SHAP summary, not an independent group Shapley estimator.
Changing the collection also changes each gene's allocation denominator.

Primary: all nonempty mapped groups. One predefined sensitivity check includes
groups with **15 to 500 measured genes**, inclusive. This follows the familiar
GSEA size-range convention as a pragmatic sensitivity check, not a validated
XAI or medical threshold. Count size using genes actually available here; it
is not asserted to reproduce GSEA. Recompute memberships/allocations after
filtering. Report both analyses, without selecting a preferred outcome.

## Comparisons

Keep top 20 genes and top 5 groups. Report all-gene overlap and covered-gene
overlap separately, fold-specific lists, and mean pairwise top-list Jaccard.
The covered-gene universe differs across collections. Raw gene/group Jaccards
and raw group Jaccards across different collection sizes are not directly
comparable evidence of improved explanations.

Use **100 random global covered-gene relabellings**, seed 20260908, for each
collection and size rule. One relabelling is shared across all five folds per
draw. It preserves the group sizes, group intersections and the distribution of
gene membership counts, but not within-group expression correlation or each
gene's original membership degree. No patient records are created. Report
observed values, random means/ranges and descriptive midrank percentiles, not
formal p-values, FDR, equivalence claims or clinical pass/fail decisions.

Save pooled and fold-specific group ranks, concentration of contributions in
the largest gene and five largest genes, within-sample cancellation, matched
gene IDs, allocation arrays and numerical checks. Inspect whether broad or
small groups dominate rankings. No new biological enrichment or SFARI-based
selection is performed. The original descriptive SFARI annotation is unchanged.

## Interpretation

Ask whether biological grouping shows agreement/stability beyond this specified
random-group reference, not whether grouping increases prediction accuracy.
Pooled magnitudes combine fitted models with different settings and scales;
fold stability also combines different evaluation records. Blood-cell mixture,
technical cohorts and other covariates remain possible explanations. A stable
label is not proof of reliable medical explanation, causation or novelty.

The original Hallmark primary results must be numerically reproduced before
interpreting new collections. Preserve previous outputs and all new outcomes.
