# What we learned

**A broader reference solves much of the coverage problem, but does not by itself
give us demonstrably better explanations.** These results concern summaries of
the same logistic-regression gene SHAP values, not new classifiers.

## 1. More genes now have a biological group

| Reference | Our genes covered | Coverage |
| --- | ---: | ---: |
| Hallmark | 4,211 | 24.4% |
| Reactome | 10,025 | 58.1% |
| Combined canonical pathways | 11,961 | 69.3% |
| GO Biological Process | 14,677 | 85.0% |

GO also covers 84.2% of the total absolute gene contribution, compared with
Hallmark's 23.5%. The additional coverage therefore includes substantial model
contributions. It is not just a larger list of genes with negligible importance.
GO describes biological functions; it is not a complete reaction-pathway map.
The remaining 2,586 genes are still present in the gene explanations.

## 2. A repeated group name is not automatically a stronger explanation

Reactome initially has high agreement between its top-five lists: **0.800**,
versus **0.620** on average for randomized groups. This is near the upper end of
the random results (95th descriptive percentile), but not a 95% confidence claim
or a multiple-testing-adjusted finding.

Its five highest-ranked pooled groups are broad, containing **668–1,353 measured
genes** each. In the predefined comparison allowing 15–500 measured genes per
group, agreement becomes **0.500 versus 0.505** for randomized groups. The apparent
advantage is therefore sensitive to the group-size rule. This check changes both
the available groups and contribution-sharing weights; it does not isolate size
as the sole cause.

GO covers the most genes but does not show higher agreement than its randomized
reference. Canonical pathways show modestly higher agreement than their random
mean, within the random reference range in both checks. None establishes a
robust advantage across the reported measures and size rules. This does not prove
biological groups and random groups are equivalent.

## 3. Do the important genes belong to the important groups?

Among each collection's top 20 covered genes, the average fractions belonging to
its top five groups are **24% Hallmark, 39% Reactome, 30% canonical and 26% GO**.
Their randomized references give about **26%, 40%, 29% and 36%**, respectively.
The membership check does not consistently favor the biological groups.

Every gene in a covered top-20 list has an annotation somewhere in that collection.
The 26% for GO means that only 5.2 of those 20 genes, on average, belong to the
five leading groups. It does not mean that 74% are missing from GO.

This is not evidence that these genes or processes are medically irrelevant.
The calculation describes model contributions; overlapping groups, opposing
contributions, correlated inputs and varying fitted models affect the rankings.
Missing annotation is a separate issue for the unrestricted top-gene list, not
an explanation for absence from the top five in a covered-only comparison.
The calculation does not test pathway activation or establish ASD causation.

## Practical conclusion

All four comparisons are preserved as the research record. The revised report
focuses on GO and documents the completed GO-only rerun, which reproduced these
results. Narrowing the discussion is not a new accuracy or explanation gain.
The useful observation is **coverage and apparent
explanation stability depend on the reference and grouping rules**. We have not
established a new ASD biomarker or a superior explanation method. Human usefulness
and clinical validity were not evaluated, and novelty remains unconfirmed.

The previously reported LR cross-validation accuracy remains **75.1%**: grouping
the explanations cannot change predictions. This extension reused explanations
for the same **261 development records**, with no model refitting and no access
to the **66 historical test records**.

See [complete numerical results](RESULTS.md),
[the plan fixed before these comparisons](../../../src/asd_blood/comparison_plan.md),
and [sources and their use](../../../data/reference/REFERENCES.md).
