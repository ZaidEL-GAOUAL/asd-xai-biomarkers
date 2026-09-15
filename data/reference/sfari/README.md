# SFARI Gene reference

Frozen public download from the [Human Gene module](https://gene.sfari.org/database/human-gene/),
release **2026 Q2**, updated **12 July 2026**, retrieved **8 September 2026**.
The source page's download link was followed directly; no account or patient
records were involved.

This public copy includes the source manifest only. The two CSV files described
below belong to the original local study and must be obtained/recreated separately
as explained in the root reproduction instructions.

- `human_genes_download.csv`: unchanged downloaded bytes, 1,289 gene rows.
- `human_genes.csv`: four-column projection for annotation, with original row
  order and values preserved.
- `source_manifest.json`: source links, SHA-256 hashes, release information,
  transformations, validation and attribution.

## Columns

| Column | Meaning |
| --- | --- |
| `gene_id` | Ensembl human gene identifier. **Not an Entrez identifier.** Five rows lack this value. |
| `gene_symbol` | Official-download gene symbol; unique in this snapshot. |
| `score` | SFARI evidence category: 1 = High Confidence, 2 = Strong Candidate, 3 = Suggestive Evidence. Empty remains missing, not zero. |
| `syndromic` | Separate original indicator: 1 denotes syndromic designation. Do not discard it when interpreting a missing numeric score. |

The numeric categories contain 245, 712 and 238 genes, respectively; 94 genes
have no numeric category. Evidence categories are not probabilities or expression
effect sizes. The interpretation follows the [official scoring guide](https://gene.sfari.org/about-gene-scoring/).

For the project's Entrez-based expression features, use its existing gene
dictionary to obtain symbols, then perform an explicit exact-symbol match. Do not
join the Ensembl `gene_id` column directly to Entrez IDs. Retain unmatched and
ambiguous project mappings as such. No alias rescue or patient-data analysis was
performed when preparing this reference.

Use this reference only after model fitting for descriptive biological context.
An unmatched gene is not a negative ASD gene. An overlap is not proof that its
blood expression causes ASD or is a validated diagnostic biomarker.

## Attribution and use

SFARI Gene, Simons Foundation Autism Research Initiative; curated by MindSpec.
Human Gene module, release 2026 Q2, accessed 8 September 2026.

The [site terms](https://gene.sfari.org/terms-and-privacy/) encourage publication
use with citation of the underlying primary data, SFARI Gene and relevant
database publications. This snapshot does not apply a new open-data licence or
imply SFARI endorsement. Consult the underlying gene evidence before making
gene-specific biological claims.
