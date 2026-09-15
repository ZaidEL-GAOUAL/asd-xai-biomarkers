# Reactome reference

Selected on 8 September 2026 as the broader pathway collection for the next
explanation comparison. Hallmark remains the original reference comparison.
Reactome was selected for its curated biological pathways and broader scope,
before calculating its coverage or any explanation scores. It is not a complete
map of human biology, an ASD-specific reference or an additional patient cohort.

## Frozen source

Use the human Reactome subset distributed in **MSigDB 2026.1.Hs**, in Entrez Gene
IDs, matching the release and identifier system used for the Hallmark analysis.
The published MSigDB collection contains **1,839 gene sets**. This is a processed
MSigDB subset based on Reactome **release 95**, not the entire current Reactome
database. MSigDB already applies some redundancy filtering using pathway overlap
and hierarchy. The [source manifest](msigdb_2026.1.Hs/source_manifest.json) records
the unchanged GMT's URL and checksum.

- [MSigDB human collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
- [Release notes](https://docs.gsea-msigdb.org/MSigDB/Release_Notes/MSigDB_2026.1.Hs/)
- [Official download index](https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/)
- [Reactome Knowledgebase 2026](https://doi.org/10.1093/nar/gkaf1223)

## Attribution

This MSigDB distribution identifies its gene-set contents as CC BY 4.0.
Attribution: Broad Institute, Massachusetts Institute of Technology, and Regents
of the University of California; original pathway knowledge from Reactome.
The original study retained the GMT without changes; this public copy distributes
its provenance only. Retrieve the source using the root reproduction instructions.
Any locally matched subsets are
derived identifier intersections, not revisions to Reactome biology.

- [MSigDB license notice on a Reactome gene set](https://www.gsea-msigdb.org/gsea/msigdb/cards/REACTOME_S_PHASE)
- [MSigDB terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Scope

The completed [coverage audit](../../../results/published/pathway_coverage/RESULTS.md) matches
**10,025 of 17,263 genes (58.1%)**, compared with Hallmark's 4,211 (24.4%). Reactome
leaves 7,238 project genes outside the collection. This establishes broader
identifier coverage only. The subsequent
[four-collection XAI comparison](../../../results/published/group_comparison/RESULTS.md) is now
complete, reusing the same gene SHAP values with no model refitting or test access.

Preparation checks gene-ID availability only. It does not filter patient records,
change expression measurements, choose classifier settings, recalculate SHAP or
access test records. Overlapping and nested pathways need to remain visible in
interpretation; more group labels do not imply more independent evidence.
