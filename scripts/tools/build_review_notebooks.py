"""Build portable notebooks from the distributed aggregate results only."""
import argparse
from pathlib import Path
import sys
from textwrap import dedent

import nbformat as nbf

STUDY = Path(__file__).resolve().parents[2]


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


SETUP = '''
from pathlib import Path
import json
from IPython.display import Markdown, Image, display

current = Path.cwd().resolve()
STUDY = next(p for p in (current, *current.parents)
             if (p / "results/published").is_dir())
RESULTS = STUDY / "results/published"

def read_json(relative):
    path = (RESULTS / relative).resolve()
    if not path.is_relative_to(RESULTS.resolve()) or path.suffix != ".json":
        raise ValueError("Only published JSON summaries may be read")
    return json.loads(path.read_text())

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    display(Markdown("\\n".join(lines)))

print("Review mode: included aggregate results only; no training or participant-table access.")
'''


def definitions():
    return {
        "01_data_review.ipynb": [
            md("""# 01 Data review

            This study uses whole-blood expression from GSE18123 on GPL570 and
            GPL6244, and GSE6575 on GPL570. These are three platform cohorts from
            two studies, not three independent studies. The counts below describe
            assay records, not a verified number of unrelated people.

            See the [pipeline overview](../../README.md) and
            [data and method references](../../data/reference/REFERENCES.md).
            """), code(SETUP),
            code('''
            s = read_json("preparation/summary.json")
            table(["Measure", "Count"], [["Retained records", s["samples"]],
                ["ASD", s["ASD"]], ["Control", s["Control"]],
                ["Development records", s["train_samples"]], ["Historical test records", s["test_samples"]],
                ["Shared genes", s["common_genes"]], ["QC exclusions", s["quality_excluded_records"]]])
            assert s["train_samples"] + s["test_samples"] == s["samples"]
            '''),
            md("""## Preparation and evaluation

            Raw arrays were normalized individually with frozen RMA. Unambiguous
            probes were mapped to Entrez gene IDs, multiple probes were combined
            by their median, and only genes shared across platforms were retained.
            Four GPL570 training controls exceeded the recorded GNUSE quality rule.
            GNUSE was unavailable for GPL6244 core output; unavailable is not a pass.

            The 261 development records form five grouped folds. Each fold was
            predicted using a model fitted on the other folds; the same record was
            not both fitted and validated in that outer step. Tuning occurred
            inside the training portion. This procedure was already completed.

            The separate 66-record internal test set was examined in earlier work.
            It was not reopened for the active model and explanation comparisons.
            Unknown relatedness and blood/cohort confounding remain limitations.

            [Reproduction steps](../pipeline/README.md) and
            [preparation summary](../../results/published/preparation/summary.json).
            """),
        ],
        "02_model_comparison.ipynb": [
            md("""# 02 Model comparison

            Compare logistic regression, Random Forest and KNN after a fixed
            54-setting search. The reported means are across five outer validation
            folds. Balanced accuracy gives ASD and control recognition equal weight.
            No synthetic sampling, PCA input or threshold optimization was used.
            """), code(SETUP),
            code('''
            scores = read_json("model_comparison/summary.json")
            table(["Model", "Accuracy", "Balanced accuracy", "ROC AUC", "Sensitivity", "Specificity"], [
                [s["model"].replace("_", " "), f'{s["mean_accuracy"]:.1%}',
                 f'{s["mean_balanced_accuracy"]:.1%}', f'{s["mean_roc_auc"]:.3f}',
                 f'{s["mean_sensitivity"]:.1%}', f'{s["mean_specificity"]:.1%}'] for s in scores])
            display(Image(filename=str(RESULTS / "model_comparison/confusion_matrices.png")))
            '''),
            md("""## Interpretation

            Logistic regression remains the strongest of these benchmarks. Its
            result is essentially unchanged from the previous estimate, not a
            newly improved diagnostic score. Across its 261 validation predictions,
            124 ASD records and 72 controls were correctly classified; 39 ASD
            records were missed and 26 controls were labeled ASD.

            False negatives are missed ASD records; false positives are controls
            labeled ASD. Both are errors. Their clinical tradeoff cannot be settled
            by this dataset or accuracy alone. This classifier is not a diagnostic test.

            The mean fold AUC and the pooled out-of-fold AUC are different summaries;
            predictions from separately fitted models need not share calibration.
            [Full model report](../../results/published/model_comparison/RESULTS.md).

            The active pipeline retains the original model without oversampling.
            Temporary experiments are kept separately and are not required steps.
            """),
        ],
        "03_gene_pathway_explanations.ipynb": [
            md("""# 03 Gene and pathway explanations

            Standard Linear SHAP explains the logistic-regression models only.
            It describes which gene inputs push a prediction toward ASD or control,
            relative to the corresponding model's training background. Contributions
            are on a log-odds scale, not a measure of pathway activation.
            """), code(SETUP),
            code('''
            s = read_json("gene_pathway_explanations/summary.json")
            table(["Measure", "Value"], [["Explained development records", s["sample_count"]],
                ["Gene inputs", s["gene_count"]], ["Hallmark groups", s["pathway_count"]],
                ["Gene coverage", f'{s["covered_gene_fraction"]:.1%}'],
                ["Absolute gene attribution outside Hallmark", f'{s["unassigned_absolute_gene_share"]:.1%}']])
            display(Image(filename=str(RESULTS / "gene_pathway_explanations/figures/gene_importance.png")))
            display(Image(filename=str(RESULTS / "gene_pathway_explanations/figures/pathway_importance.png")))
            '''),
            md("""## What grouping does

            Signed gene contributions are added within biological groups. A gene
            occurring in several groups shares its contribution equally among
            them; unassigned contributions remain separate. Opposite signs can
            cancel. This is allocated gene SHAP, not a separately calculated
            pathway Shapley estimator. Predictions do not change.

            Hallmark covers a minority of our genes. Its top groups recur across
            folds, but no clear advantage over matched randomized groups was
            established. Randomization rearranges annotation memberships, not
            patient data. One pooled top-20 gene matches the frozen SFARI reference;
            genetic-risk annotation is not confirmation of a blood biomarker.

            [Explanation report](../../results/published/gene_pathway_explanations/RESULTS.md).
            The next notebook compares broader references using the same gene values.
            """),
        ],
        "04_collection_comparison.ipynb": [
            md("""# 04 Biological collection comparison

            Compare Hallmark, Reactome, combined canonical pathways and GO Biological
            Process using identical saved gene contributions within each fold.
            GO groups functions, not exclusively reaction pathways. Canonical
            pathways include Reactome and do not provide independent replication.

            This records the completed four-collection comparison. GO results
            can be read directly in the saved comparison below. See the
            [comparison results](../../results/published/group_comparison/RESULTS.md).
            """), code(SETUP),
            code('''
            comparison = read_json("group_comparison/comparison.json")
            primary = [r for r in comparison if r["size_rule"] == "primary"]
            table(["Collection", "Genes covered", "Gene coverage", "Attribution coverage"], [
                [r["label"], r["covered_gene_count"], f'{r["covered_gene_fraction"]:.1%}',
                 f'{r["covered_absolute_gene_attribution_fraction"]:.1%}'] for r in primary])
            display(Image(filename=str(RESULTS / "group_comparison/coverage.png")))
            '''),
            md("""## Consistency against randomized groups

            Top-five agreement measures how much the group lists share across
            folds: 0 means no shared names; 1 means identical lists. There is no
            medical pass mark. Each collection is compared with 100 random
            covered-gene relabellings preserving group sizes and intersections.

            We also specified a 15–500 measured-gene size restriction before this
            comparison. It is a sensitivity rule, not a validated XAI threshold.
            Both analyses are retained; Hallmark's identical memberships reuse results.
            """),
            code('''
            table(["Collection", "Size rule", "Real agreement", "Random mean"], [
                [r["label"], r["size_rule"], f'{r["observed"]["mean_pathway_jaccard"]:.3f}',
                 f'{r["random_control_comparison"]["mean_pathway_jaccard"]["random_mean"]:.3f}']
                for r in comparison])
            display(Image(filename=str(RESULTS / "group_comparison/group_stability.png")))
            '''),
            md("""## Conclusion

            GO increases gene coverage to 85.0% but does not show higher agreement
            than its randomized reference. Reactome initially looks more consistent,
            but its advantage disappears under the predefined size rule. That rule
            changes both group availability and contribution-sharing weights, so
            it does not isolate size as the sole cause.

            The result is not proof that pathways are medically irrelevant or
            equivalent to random groups. It shows that broader coverage does not
            establish better explanations in this experiment. Different validation
            records, fitted settings, correlated genes and blood confounding remain.

            The 26.0% covered-gene membership means an average of 5.2 of the top
            20 GO-covered genes occurred in the five leading groups. All 20 genes
            have an annotation somewhere in GO; this is not 26.0% annotation coverage.

            [Plain-language interpretation](../../results/published/group_comparison/INTERPRETATION.md),
            [full results](../../results/published/group_comparison/RESULTS.md),
            [comparison plan](../../src/asd_blood/comparison_plan.md) and [sources](../../data/reference/REFERENCES.md).
            """),
        ],
    }


def build(execute=False):
    folder = STUDY / "scripts/notebooks"
    folder.mkdir(exist_ok=True)
    for filename, cells in definitions().items():
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}})
        if execute:
            from nbclient import NotebookClient
            client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                    resources={"metadata": {"path": str(STUDY)}})
            client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
            client.execute()
        for cell in notebook.cells:
            cell.metadata = {}
        nbf.validate(notebook)
        nbf.write(notebook, folder / filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    build(parser.parse_args().execute)
