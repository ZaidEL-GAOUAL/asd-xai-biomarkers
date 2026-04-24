# Reference data

Third-party reference data bundled for reproducibility.

## sfari_genes.csv

Source: SFARI Gene database, downloaded from
https://gene.sfari.org/wp-content/themes/sfari-gene/utilities/download-csv.php?api-endpoint=genes

Date downloaded: 2026-04-23 (Q4 2025 release, last updated 2026-01-14 per the
SFARI Gene website at time of retrieval).

Total rows: 1,267 genes.

Columns:
- `status` — curation status code.
- `gene-symbol` — HGNC gene symbol.
- `gene-name` — full gene name.
- `ensembl-id` — Ensembl ID.
- `chromosome` — chromosome.
- `genetic-category` — one or more of: Rare Single Gene Mutation, Syndromic,
  Functional, Genetic Association.
- `gene-score` — SFARI confidence score: 1 (High Confidence, 244 genes),
  2 (Strong Candidate, 708 genes), 3 (Suggestive Evidence, 221 genes), blank (94 genes).
- `syndromic` — 1 if flagged as Syndromic autism gene (312 genes), else 0.
- `eagle` — EAGLE score.
- `number-of-reports` — count of supporting reports.

This snapshot is used as-is by `scripts/pipeline/13_sfari_enrichment.py` for
gene-set enrichment tests. Two filters are applied inside the script:

- **Strict** = Score 1 ∪ Syndromic (435 unique genes)
- **Broad** = Score 1 ∪ Score 2 ∪ Syndromic (1,091 unique genes)

To refresh the snapshot:

```bash
curl -sL "https://gene.sfari.org/wp-content/themes/sfari-gene/utilities/download-csv.php?api-endpoint=genes" \
  -o data/reference/sfari_genes.csv
```

## License / attribution

SFARI Gene is a database maintained by the Simons Foundation. The data is made
available for non-commercial research use. Cite:
- Abrahams BS, Arking DE, Campbell DB, et al. "SFARI Gene 2.0: a
  community-driven knowledgebase for the autism spectrum disorders (ASDs)."
  *Molecular Autism* 4:36 (2013).
- Banerjee-Basu S, Packer A. "SFARI Gene: an evolving database for the autism
  research community." *Dis Model Mech* 3:133-135 (2010).
