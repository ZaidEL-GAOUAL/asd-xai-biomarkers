# Pipeline references

These sources document the inputs and methods used by the pipeline. The thesis
literature review and article-by-article comparisons are kept separately.

## Data and preparation

- [GEO GSE18123](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18123)
  and [GEO GSE6575](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6575):
  array data and sample metadata. Required snapshots and hashes are in
  [the data source configuration](../source_manifest.json).
- McCall, Bolstad and Irizarry (2010),
  [Frozen robust multiarray analysis](https://doi.org/10.1093/biostatistics/kxp059),
  and McCall, Jaffee and Irizarry (2012),
  [fRMA ST](https://doi.org/10.1093/bioinformatics/bts588): per-array normalization.
- McCall et al. (2011),
  [Assessing Affymetrix GeneChip microarray quality](https://doi.org/10.1186/1471-2105-12-137):
  the GNUSE quality criterion used for the GPL570 arrays.

## Evaluation and gene explanations

- Cawley and Talbot (2010),
  [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/beta/papers/v11/cawley10a.html):
  separating parameter selection from performance assessment.
- Lundberg and Lee (2017),
  [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html),
  and [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html):
  gene-level explanations of fitted logistic-regression models.

## Biological grouping

- [MSigDB human collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
  and [the 2026.1.Hs release](https://docs.gsea-msigdb.org/MSigDB/Release_Notes/MSigDB_2026.1.Hs/):
  frozen Hallmark, Reactome, canonical and GO Biological Process gene sets.
  [Reference records](collections/README.md) describe
  provenance and redistribution limits.
- Liberzon et al. (2015),
  [The Molecular Signatures Database hallmark gene set collection](https://doi.org/10.1016/j.cels.2015.12.004):
  the initial compact biological collection.
- [The Reactome Knowledgebase 2026](https://doi.org/10.1093/nar/gkaf1223)
  and [The Gene Ontology knowledgebase in 2026](https://doi.org/10.1093/nar/gkaf1292):
  upstream biological-reference context. The pipeline uses their frozen MSigDB subsets.
- Segura-Lepe, Keun and Ebbels (2019),
  [Predictive modelling using pathway scores](https://doi.org/10.1186/s12859-019-3163-0):
  motivation for comparing biological summaries with randomized groups. The
  pipeline adapts that idea to explanations, not prediction by pathway scores.
- [GSEA User Guide](https://docs.gsea-msigdb.org/GSEA/GSEA_User_Guide/):
  the 15 to 500 measured-gene convention adapted as a sensitivity check, not a
  validated explanation-quality threshold.
- [SFARI Gene scoring](https://gene.sfari.org/about-gene-scoring/): optional
  descriptive ASD-gene annotation using the frozen version documented in
  [the SFARI reference record](sfari/README.md).

The [comparison plan](../../src/asd_blood/comparison_plan.md) defines the implemented
allocation and randomization rules. Group summaries are allocated gene SHAP,
not independently estimated pathway Shapley values or measures of activation.
