# Additional gene-set references

Selected before explanation results on 8 September 2026:

- **Canonical pathways (MSigDB C2:CP):** a combined curated pathway collection,
  including Reactome and other sources. It is not independent of Reactome.
- **GO Biological Process (MSigDB C5:GO:BP):** groups defined by annotated
  biological processes. These are functional gene groups, not exclusively
  reaction pathways. Broad parent terms and narrower child terms overlap.

Both use the same **MSigDB 2026.1.Hs human Entrez ID release** as the existing
Hallmark and Reactome references. Source bytes, hashes and release information
are recorded in separate manifests. This public copy contains those provenance
records, not the GMT files or derived memberships. Source files are retrieved
separately using the root reproduction instructions. No new patient data are downloaded.

The current question concerns biological process/pathway summaries. Therefore
we do not search every MSigDB collection for maximal coverage: chromosome bands,
regulatory targets and disease-phenotype annotations answer different questions.
All four selected collections are compared, irrespective of their results.

## Sources and use

- [MSigDB collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp): collection definitions and published counts.
- [MSigDB 2026.1.Hs release notes](https://docs.gsea-msigdb.org/MSigDB/Release_Notes/MSigDB_2026.1.Hs/): source versions and upstream processing.
- [Official download index](https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/): immutable-version GMT filenames.
- [GO citation policy](https://geneontology.org/docs/go-citation-policy/): attribution and reference release reporting.
- [MSigDB terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp): reuse conditions, including source-specific exceptions.

## Redistribution

MSigDB generally uses CC BY 4.0, but the combined canonical collection contains
source-specific exceptions, including KEGG MEDICUS, other KEGG and BioCarta
sets. Do not assign a blanket license to the combined GMT or publish it without
reviewing those terms. It is retained for local research; reproduction can use
the official download. GO and MSigDB attribution must accompany the GO subset.

The source files and downstream memberships remain unchanged or explicitly
derived by matching gene IDs. Neither reference establishes ASD causation,
pathway activation, clinical validity, or a complete map of human biology.
