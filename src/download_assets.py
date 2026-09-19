"""Download versioned release assets, verify SHA-256, then extract them."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes destination: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive must not contain symlinks")
            if target.exists() and not item.is_dir():
                raise FileExistsError(f"Refusing to overwrite {target}; use a clean output directory")
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["ppi", "pri", "benchmarks"])
    parser.add_argument("--archive", type=Path, help="Verify/extract an already downloaded ZIP")
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    spec = json.loads((ROOT / "config/assets.json").read_text())[args.asset]
    with tempfile.TemporaryDirectory(prefix="colbert_ppi_asset_") as temp:
        archive = args.archive or Path(temp) / spec["filename"]
        try:
            if not args.archive:
                print(f"Downloading {spec['url']}", flush=True)
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "ColBERT-PPI/0.2.0"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if sha256(archive) != spec["sha256"]:
                raise ValueError("SHA-256 mismatch; obtain the exact version listed in config/assets.json")
            extract(archive, args.destination)
        except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
            parser.exit(1, f"Asset setup failed: {error}\nSee README.md for manual download.\n")
    print(f"Verified and extracted {spec['filename']} to {args.destination}")


if __name__ == "__main__":
    main()
