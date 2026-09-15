"""Independently reconstruct saved group allocations and comparison summaries."""
from itertools import combinations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from scipy import sparse

from asd_blood.common import sha256, write_json
from asd_blood.compare_explanations import COLLECTIONS, SOURCE, STUDY, source_guard


def metrics(values, member, fold_labels):
    degree = np.asarray(member.sum(axis=1)).ravel()
    weights = sparse.diags(1 / np.maximum(degree, 1)) @ member
    allocated = np.asarray(values @ weights)
    tops, overlap, all_overlap = [], [], []
    covered = np.flatnonzero(degree)
    for f in range(5):
        rows = fold_labels == f
        importance = np.abs(values[rows]).mean(axis=0)
        genes = covered[np.argsort(-importance[covered], kind="stable")[:20]]
        all_genes = np.argsort(-importance, kind="stable")[:20]
        groups = np.argsort(-np.abs(allocated[rows]).mean(axis=0), kind="stable")[:5]
        contained = set(np.flatnonzero(np.asarray(member[:, groups].sum(axis=1)).ravel()))
        tops.append(set(groups))
        overlap.append(len(set(genes) & contained) / len(genes))
        all_overlap.append(len(set(all_genes) & contained) / len(all_genes))
    agreement = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    return {"mean_pathway_jaccard": float(np.mean(agreement)),
            "mean_covered_gene_overlap": float(np.mean(overlap)),
            "mean_all_gene_overlap": float(np.mean(all_overlap))}


def run():
    source_guard()
    output = STUDY / "results/group_comparison"
    target = output / "verification.json"
    if target.exists():
        raise FileExistsError(target)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["status"] == "complete"
    for path, expected in protocol["input_sha256"].items():
        assert sha256(path) == expected, path
    values, ids, base, decisions, fold_labels = [], [], [], [], []
    for i in range(5):
        with np.load(SOURCE / "attributions" / f"fold_{i}.npz", allow_pickle=False) as a:
            if i == 0:
                genes = a["gene_ids"].tolist()
            assert genes == a["gene_ids"].tolist()
            values.append(a["gene_shap_log_odds"])
            ids.extend(a["sample_ids"].tolist())
            base.extend(a["base_log_odds"])
            decisions.extend(a["model_decision_log_odds"])
            fold_labels.extend([i] * len(a["sample_ids"]))
    values, base, decisions = np.concatenate(values), np.array(base), np.array(decisions)
    fold_labels = np.array(fold_labels)
    assert len(ids) == len(set(ids)) == 261
    lookup = {g: i for i, g in enumerate(genes)}
    checked, max_error = [], 0.
    for summary in json.loads((output / "comparison.json").read_text()):
        key, rule = summary["collection"], summary["size_rule"]
        directory = output / key / rule
        manifest = COLLECTIONS[key][1]
        gmt = manifest.parent / json.loads(manifest.read_text())["file"]
        names, memberships = [], []
        for line in gmt.read_text().splitlines():
            name, url, *reference_ids = line.split("\t")
            present = sorted({lookup[g] for g in reference_ids if g in lookup})
            if present and (rule == "primary" or 15 <= len(present) <= 500):
                names.append(name)
                memberships.append(present)
        row = [g for group in memberships for g in group]
        col = [c for c, group in enumerate(memberships) for _ in group]
        member = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(genes), len(names)))
        degree = np.bincount(row, minlength=len(genes))
        # Compute each group independently from its gene list and degrees.
        reconstructed = np.column_stack([(values[:, group] / degree[group]).sum(axis=1)
                                         for group in memberships])
        residual = values[:, degree == 0].sum(axis=1)
        with np.load(directory / "allocations.npz", allow_pickle=False) as a:
            assert a["sample_ids"].tolist() == ids
            assert a["group_names"].tolist() == names
            np.testing.assert_array_equal(a["fold"], fold_labels)
            np.testing.assert_array_equal(a["base_log_odds"], base)
            np.testing.assert_array_equal(a["decision_log_odds"], decisions)
            np.testing.assert_allclose(a["group_allocations_log_odds"], reconstructed, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(a["unassigned_log_odds"], residual, atol=1e-10)
            for f in range(5):
                np.testing.assert_allclose(a["fold_mean_absolute_allocations"][f],
                                          np.abs(reconstructed[fold_labels == f]).mean(axis=0), atol=1e-10)
        error = float(np.max(np.abs(base + reconstructed.sum(axis=1) + residual - decisions)))
        max_error = max(max_error, error)
        assert error < 1e-8
        assert summary["covered_gene_count"] == int(np.count_nonzero(degree))
        np.testing.assert_allclose(summary["covered_absolute_gene_attribution_fraction"],
                                   np.abs(values[:, degree > 0]).sum() / np.abs(values).sum())
        observed = metrics(values, member, fold_labels)
        for metric, value in observed.items():
            np.testing.assert_allclose(summary["observed"][metric], value, atol=1e-12)
        controls = json.loads((directory / "random_controls.json").read_text())
        assert len(controls) == 100
        rng = np.random.default_rng(20260908)
        covered = np.flatnonzero(degree)
        for draw in range(100):
            order = np.arange(len(genes))
            order[covered] = rng.permutation(covered)
            if draw in (0, 49, 99):
                randomized = member[order]
                np.testing.assert_array_equal(member.sum(axis=0), randomized.sum(axis=0))
                expected = metrics(values, randomized, fold_labels)
                for metric, value in expected.items():
                    np.testing.assert_allclose(controls[draw][metric], value, atol=1e-12)
        for metric, value in observed.items():
            random = np.array([r[metric] for r in controls])
            stats = summary["random_control_comparison"][metric]
            for stat, expected in (("observed", value), ("random_mean", random.mean()),
                                   ("random_q025", np.quantile(random, .025)),
                                   ("random_q975", np.quantile(random, .975)),
                                   ("midrank_percentile", 100 * ((random < value).mean() + .5 * (random == value).mean()))):
                np.testing.assert_allclose(stats[stat], expected, atol=1e-12)
        checked.append({"collection": key, "rule": rule, "groups": len(names),
                        "reconstruction_error": error, "random_draws_independently_recomputed": [0, 49, 99]})
    result = {"status": "passed", "records": 261, "genes": len(genes), "variants": checked,
              "max_reconstruction_error": max_error, "input_hashes_unchanged": True,
              "models_refitted": False, "prepared_data_access": False, "test_data_access": False,
              "verifier_sha256": sha256(Path(__file__))}
    write_json(target, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
