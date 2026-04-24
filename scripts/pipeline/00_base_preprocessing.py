"""Base preprocessing for GSE18123: GEO download + probe-to-gene mapping.

Produces data/processed/GSE18123_commonGenes_2platforms.parquet, which is the
starting point for script 01_combat_with_mod_diagnosis.py.

This script replaces the original notebooks 01 (data loading) and 02 (probe
mapping + gene matrix). It is idempotent — if the parquet already exists, the
script skips recomputation unless FORCE=1 is set in the environment.

Steps:
  1. Load GSE18123 from GEO (cached to data/raw/)
  2. Extract diagnosis + platform metadata from each sample
  3. Build per-platform probe-to-gene matrices (GPL570, GPL6244)
  4. Intersect to common genes, apply log2(x+1) if needed
  5. Save to parquet + gzipped CSV in data/processed/
"""
import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))

os.chdir(str(_P(__file__).resolve().parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from asd_pipeline_utils import (
    load_geo_series, extract_sample_metadata, build_common_gene_dataset,
    save_analysis_frame, save_metadata, nan_report,
    PARQUET_PATH, DESTDIR,
)

REPO_ROOT = _P(__file__).resolve().parent.parent.parent

# Force-regenerate flag
force = os.environ.get("FORCE", "0") == "1"

# Paths (repo-relative)
proc_parquet = REPO_ROOT / "data" / "processed" / "GSE18123_commonGenes_2platforms.parquet"
proc_csv = REPO_ROOT / "data" / "processed" / "GSE18123_commonGenes_2platforms.csv.gz"
proc_meta = REPO_ROOT / "data" / "processed" / "GSE18123_sample_metadata.csv"

if proc_parquet.exists() and not force:
    print(f"Parquet already exists at {proc_parquet}")
    print("Set FORCE=1 to regenerate from raw GEO data.")
    sys.exit(0)

# Ensure raw data directory exists for the GEOparse cache
raw_dir = REPO_ROOT / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

print("Step 1: Load GSE18123 from GEO...")
gse, expr_probes = load_geo_series(gse_id="GSE18123", destdir=str(raw_dir) + "/")
print(f"  expression matrix (probes x samples): {expr_probes.shape}")

print("\nStep 2: Extract diagnosis + platform metadata...")
expr_probes, diagnosis_df, platform_s = extract_sample_metadata(gse, expr_probes)
print(f"  samples with valid diagnosis: {len(diagnosis_df)}")
print(f"  platforms: {dict(platform_s.value_counts())}")
print(f"  diagnosis: {dict(diagnosis_df['diagnosis'].value_counts())}")

proc_meta.parent.mkdir(parents=True, exist_ok=True)
save_metadata(diagnosis_df, platform_s, path=proc_meta)
print(f"  saved metadata: {proc_meta}")

print("\nStep 3-4: Build per-platform gene matrices, intersect common genes, log2-transform...")
X, meta, final_df = build_common_gene_dataset(
    gse, expr_probes, diagnosis_df, platform_s, apply_log2=True,
)
print(f"\nFinal matrix (samples x genes): {X.shape}")

nan_report(X, title="GENES ONLY")

print("\nStep 5: Save to parquet + CSV...")
# Direct save to our chosen locations (not the utility's defaults, which point to notebooks/)
final_df.to_parquet(proc_parquet, index=True)
final_df.to_csv(proc_csv, compression="gzip", index=True)
print(f"  parquet: {proc_parquet}")
print(f"  csv.gz:  {proc_csv}")
print("\nDone. Ready for 01_combat_with_mod_diagnosis.py.")
