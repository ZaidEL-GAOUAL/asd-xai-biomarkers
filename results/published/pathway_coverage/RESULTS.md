# Pathway coverage

Identifier check only. No new model or XAI results.

| Collection | Gene sets | Our genes covered | Coverage of our 17,263 genes |
| --- | ---: | ---: | ---: |
| Hallmark | 50 | 4,211 | 24.4% |
| Reactome | 1,839 | 10,025 | 58.1% |

## Decision

Reactome is selected as the broader reference for the next explanation comparison. Hallmark remains the original comparator. The classifier, gene inputs, saved SHAP values and earlier results are unchanged. This is not an accuracy or explanation-quality improvement.

Reactome covers 6,501 genes absent from Hallmark, but misses 687 genes covered by Hallmark. It is broader, not a strict superset. The collections have not been merged.

## Mapping checks

- Reactome sets have 1 to 1,353 measured genes (median 21).
- 0 sets have no matching genes; 26 have only one to four. No size cutoff was applied.
- 9,613 covered genes occur in multiple Reactome sets. Shared or nested pathways are not independent biological evidence.
- 1 groups of pathway labels become identical after matching to our measured genes. Their names are retained in summary.json.

Coverage means membership by gene ID, not expression strength, ASD relevance or biological activation. Missing reference genes are absent measurements, not zero expression. Old array identifiers and current reference identifiers may differ; no identifier rescue was attempted.

## Reproduction and sources

After regenerating the inputs described in the root reproduction instructions,
run `python scripts/pipeline/06_gene_set_coverage.py --output results/pathway_coverage_recheck`
from the repository root. Existing output directories are never overwritten.

The audit reads the gene dictionary and parses only the training matrix header. It does not read diagnosis labels, parse expression rows, access test files or fit models.

See the [Reactome reference record](../../../data/reference/reactome/README.md) and [source ledger](../../../data/reference/REFERENCES.md).
Full memberships and unmapped IDs were retained in the original run's JSON files.
Only aggregate summaries are distributed here; reference memberships are regenerated
from the recorded sources.
