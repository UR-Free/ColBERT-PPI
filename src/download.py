"""Download versioned release assets, verify SHA-256, then extract them."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile



def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination, strip_prefix=None):
    with zipfile.ZipFile(archive) as bundle:
        targets = []
        for item in bundle.infolist():
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if strip_prefix:
                if not relative.parts or relative.parts[0] != strip_prefix:
                    raise ValueError(f"Expected {strip_prefix}/ prefix: {item.filename}")
                relative = PurePosixPath(*relative.parts[1:])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes destination: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive must not contain symlinks")
            if target.exists() and not item.is_dir():
                raise FileExistsError(f"Refusing to overwrite {target}; use a clean output directory")
            targets.append((item, target))
        for item, target in targets:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def fetch(spec, archive):
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "ColBERT-PPI/0.3.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    except urllib.error.HTTPError as error:
        gh = shutil.which("gh")
        if error.code not in (403, 404) or gh is None:
            raise
        prefix = "https://github.com/UR-Free/ColBERT-PPI/releases/download/"
        if not spec["url"].startswith(prefix):
            raise
        tag = spec["url"][len(prefix):].split("/")[0]
        result = subprocess.run([gh, "release", "download", tag, "--repo", "UR-Free/ColBERT-PPI",
                                 "--pattern", spec["filename"], "--dir", str(archive.parent), "--clobber"])
        if result.returncode:
            raise RuntimeError("Download failed; authenticate with gh auth login or supply --archive")


def run(args):
    manifest = args.root / "data/weights/manifest.json"
    spec = json.loads(manifest.read_text())[args.asset]
    with tempfile.TemporaryDirectory(prefix="colbert_ppi_asset_") as temp:
        archive = args.archive or Path(temp) / spec["filename"]
        if not args.archive:
            print(f"Downloading {spec['url']}", flush=True)
            fetch(spec, archive)
        if sha256(archive) != spec["sha256"]:
            raise ValueError("SHA-256 mismatch; obtain the exact version in data/weights/manifest.json")
        destination = args.destination
        if args.asset == "benchmarks":
            destination = destination / "data/benchmarks"
        extract(archive, destination, strip_prefix="data" if args.asset == "benchmarks" else None)
    print(f"Verified and extracted {spec['filename']} to {destination}")


if __name__ == "__main__":
    from colbert_ppi.options import run_script

    run_script("download", run)
