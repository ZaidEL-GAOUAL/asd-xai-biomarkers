"""Download versioned MSigDB references and validate bytes before retaining them.

No patient data are read or submitted. Reference data are not relicensed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = [ROOT / "data/reference/hallmark/msigdb_2026.1.Hs/source_manifest.json",
             ROOT / "data/reference/reactome/msigdb_2026.1.Hs/source_manifest.json",
             *sorted((ROOT / "data/reference/collections").rglob("source_manifest.json"))]


def run():
    for path in MANIFESTS:
        manifest = json.loads(path.read_text())
        name = manifest["file"]
        if Path(name).name != name:
            raise ValueError("Reference name must be a filename")
        target = path.parent / name
        if target.exists():
            with target.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != manifest["sha256"]:
                raise ValueError(f"Existing reference differs from manifest: {name}")
            print(f"Verified existing {name}")
            continue
        url = manifest["source_url"]
        if not url.startswith("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"):
            raise ValueError("Expected official versioned MSigDB download")
        request = urllib.request.Request(url, headers={"User-Agent": "ASD-Blood-XAI-reference-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(manifest["bytes"] + 1)
        if len(data) != manifest["bytes"] or hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ValueError(f"Downloaded reference changed: {name}")
        with target.open("xb") as stream:
            stream.write(data)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
