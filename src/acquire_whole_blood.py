"""Reproducible, resumable intake of the three peripheral-whole-blood cohorts.

No modelling and no use of the deposited (incompatibly processed) expression values.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/whole_blood/source"
OUT = ROOT / "phase1_outputs"
COHORTS = {
    "GSE18123_GPL570": "GSE18123-GPL570_series_matrix.txt.gz",
    "GSE18123_GPL6244": "GSE18123-GPL6244_series_matrix.txt.gz",
    "GSE6575_GPL570": "GSE6575_series_matrix.txt.gz",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path):
    metadata = defaultdict(list)
    with gzip.open(path, "rt", newline="") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                cells = next(csv.reader([line], delimiter="\t"))
                metadata[cells[0]].append(cells[1:])
    n = len(metadata["!Sample_geo_accession"][0])
    assert all(len(row) == n for rows in metadata.values() for row in rows)
    return metadata


def build_manifest():
    SOURCE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / "data/source_manifest.json").read_text())["files"]
    expected = {entry["file"]: entry for entry in expected}
    provenance = []
    for filename in list(COHORTS.values()) + ["GPL570.gz", "GPL6244.gz"]:
        dst = SOURCE / filename
        if not dst.exists():
            raise FileNotFoundError(f"Missing source {filename}. See data/source_manifest.json and scripts/pipeline/README.md.")
        with gzip.open(dst, "rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected[filename]["decompressed_sha256"]:
            raise ValueError(f"Source metadata differs from the recorded study: {filename}")
        accession = filename.split("_")[0].split("-")[0].split(".")[0]
        provenance.append({"file": filename, "sha256": sha256(dst),
                           "bytes": dst.stat().st_size,
                           "source": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"})
    pd.DataFrame(provenance).to_csv(OUT / "source_checksums.csv", index=False)
    records = []
    for cohort, filename in COHORTS.items():
        study, platform = cohort.split("_")
        meta = read_header(SOURCE / filename)
        for i, gsm in enumerate(meta["!Sample_geo_accession"][0]):
            title = meta["!Sample_title"][0][i]
            source = meta["!Sample_source_name_ch1"][0][i]
            chars = [row[i] for row in meta["!Sample_characteristics_ch1"]]
            keyed = {v.split(":", 1)[0].strip().lower(): v.split(":", 1)[1].strip()
                     for v in chars if ":" in v}
            diagnosis = keyed.get("diagnosis", chars[0])
            if study == "GSE18123":
                label = {"CONTROL": "Control", "AUTISM": "ASD", "PDD-NOS": "ASD",
                         "ASPERGER'S DISORDER": "ASD"}.get(diagnosis)
            else:
                label = "ASD" if diagnosis.lower().startswith("autism ") else (
                    "Control" if diagnosis.lower().startswith("general population") else None)
            if label is None and not (study == "GSE6575" and
                    re.search("mental retardation|developmental delay", diagnosis, re.I)):
                raise ValueError(f"Unrecognized diagnosis for {gsm}: {diagnosis}")
            sex_match = re.search(r"\b(female|male)\b", " | ".join(chars), re.I)
            reason = "" if label else "MR/DD excluded: neither ASD nor typical control"
            if gsm == "GSM650700":
                reason = "Repeated GSE18123 sample title AC02-1189-01; retain GPL570 GSM650579"
            # A common stem may indicate relatives or repeat visits, NOT a proven identity.
            stem = re.sub(r"-(?:P\d|A\d|\d{2})$", "", title) if study == "GSE18123" else title
            urls = [row[i].replace("ftp://", "https://")
                    for row in meta["!Sample_supplementary_file"] if ".cel" in row[i].lower()]
            assert len(urls) == 1, (gsm, urls)
            assert meta["!Sample_platform_id"][0][i] == platform
            records.append({"sample_id": gsm, "sample_title": title, "study": study,
                            "platform": platform, "cohort": cohort,
                            "blood_material": "peripheral whole blood", "source_name": source,
                            "original_diagnosis": diagnosis, "label": label or "Excluded",
                            "target": int(label == "ASD") if label else np.nan,
                            "sex": sex_match.group(1).title() if sex_match else "Unknown",
                            "age_months": keyed.get("age at blood drawing (months)", ""),
                            "relatedness_group": f"{study}:{stem}",
                            "group_evidence": "conservative title-stem heuristic; not verified family IDs",
                            "included": not bool(reason), "exclusion_reason": reason,
                            "cel_url": urls[0], "characteristics_raw": " | ".join(chars)})
    full = pd.DataFrame(records)
    full["age_months"] = pd.to_numeric(full.age_months, errors="coerce")
    assert full.sample_id.is_unique
    full.to_csv(OUT / "all_samples.csv", index=False)
    full.loc[~full.included].to_csv(OUT / "excluded_samples.csv", index=False)
    manifest = full.loc[full.included].copy().reset_index(drop=True)
    manifest["target"] = manifest.target.astype(int)
    assert len(manifest) == 331
    assert manifest.label.value_counts().to_dict() == {"ASD": 204, "Control": 127}
    shared = full.duplicated("relatedness_group", keep=False)
    full.loc[shared].sort_values(["relatedness_group", "sample_id"]).to_csv(
        OUT / "possible_related_samples_review.csv", index=False)
    manifest.to_csv(OUT / "included_samples.csv", index=False)
    manifest.groupby(["cohort", "label"]).size().unstack(fill_value=0).to_csv(OUT / "cohort_counts.csv")
    return manifest


def assign_split(manifest):
    """Choose among five candidate folds using counts only, never expression or scores."""
    manifest = manifest.copy()
    strata = manifest.cohort + ":" + manifest.label
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    total = strata.value_counts()
    candidates = []
    for fold, (tr, te) in enumerate(splitter.split(manifest, strata, manifest.relatedness_group)):
        fractions = strata.iloc[te].value_counts().reindex(total.index, fill_value=0) / total
        score = float(((fractions - .2) ** 2).sum() + (len(te) / len(manifest) - .2) ** 2)
        candidates.append((score, fold, tr, te))
    _, chosen, tr, te = min(candidates, key=lambda x: (x[0], x[1]))
    manifest["split"] = "train"
    manifest.loc[te, "split"] = "test"
    manifest["training_cv_fold"] = -1
    train = manifest.loc[tr]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    for fold, (_, val) in enumerate(cv.split(train, strata.loc[tr], train.relatedness_group)):
        manifest.loc[train.index[val], "training_cv_fold"] = fold
    assert not set(manifest.loc[tr, "relatedness_group"]) & set(manifest.loc[te, "relatedness_group"])
    assert manifest.groupby("relatedness_group").split.nunique().max() == 1
    assert manifest.loc[tr].groupby("relatedness_group").training_cv_fold.nunique().max() == 1
    manifest.to_csv(OUT / "split_manifest.csv", index=False)
    manifest.groupby(["split", "cohort", "label"]).size().unstack(fill_value=0).to_csv(
        OUT / "split_counts.csv")
    (OUT / "split_policy.json").write_text(json.dumps({
        "seed": 42, "training_cv_seed": 43, "chosen_candidate_fold": chosen,
        "desired_test_fraction": .2, "actual_test_fraction": len(te) / len(manifest),
        "strata": "cohort + label", "group": "relatedness_group",
        "selection": "minimize squared deviation from 20% per stratum plus overall, among 5 folds",
        "no_expression_or_model_performance_used": True,
        "not_external_validation": True,
        "caveat": "Title-stem grouping is conservative; unknown relationships may remain."
    }, indent=2) + "\n")
    return manifest


def download_one(row):
    dest = ROOT / "data/whole_blood/cel" / row.platform / f"{row.sample_id}.CEL.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            if not dest.exists():
                request = urllib.request.Request(row.cel_url, headers={"User-Agent": "ASD-student-research/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, dest.with_suffix(".part").open("wb") as f:
                    shutil.copyfileobj(response, f)
                # Validate gzip before replacing the final path.
                with gzip.open(dest.with_suffix(".part"), "rb") as f:
                    for _ in iter(lambda: f.read(1024 * 1024), b""): pass
                dest.with_suffix(".part").replace(dest)
            return {"sample_id": row.sample_id, "url": row.cel_url,
                    "path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size,
                    "sha256": sha256(dest)}
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def download_cels(manifest, workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, row) for row in manifest.itertuples()]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(manifest):
                print(f"Verified {len(results)}/{len(manifest)} CEL files", flush=True)
    pd.DataFrame(results).sort_values("sample_id").to_csv(OUT / "cel_checksums.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    m = assign_split(build_manifest())
    print(m.groupby(["split", "cohort", "label"]).size().to_string(), flush=True)
    if args.download: download_cels(m)
