"""Prepare PPI inference tokens from a CSV of single-chain PDB structure pairs."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def structure_tokens(path, foldseek, tokenizer, max_residues=1024):
    if not path.is_file():
        raise ValueError(f"Structure does not exist: {path}")
    with tempfile.TemporaryDirectory(prefix="colbert_ppi_") as temp:
        output = Path(temp) / "descriptor.tsv"
        subprocess.run(
            [foldseek, "structureto3didescriptor", "-v", "0", "--threads", "1",
             "--chain-name-mode", "1", str(path), str(output)],
            check=True, capture_output=True, text=True, timeout=300,
        )
        rows = [line.split("\t") for line in output.read_text().splitlines() if line]
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError(f"Expected exactly one protein chain in {path}; split chains first")
    aa, di = rows[0][1].upper(), rows[0][2].lower()
    if not aa or len(aa) != len(di):
        raise ValueError(f"Invalid amino-acid/3Di alignment in {path}")
    if len(aa) > max_residues:
        raise ValueError(f"{path}: {len(aa)} residues exceed {max_residues}; crop explicitly")
    sequence = "".join(a + d for a, d in zip(aa, di))
    ids = tokenizer(sequence, add_special_tokens=True)["input_ids"]
    if len(ids) != len(aa) + 2 or tokenizer.unk_token_id in ids:
        raise ValueError(f"SaProt tokenization failed for {path}; check backbone vocabulary")
    return ids


def prepare(manifest, output, foldseek, tokenizer):
    records, cache, seen = [], {}, set()
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"pair_id", "structure_a", "structure_b"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires pair_id,structure_a,structure_b columns")
        for row in reader:
            pair_id = row["pair_id"].strip()
            if not pair_id or pair_id in seen:
                raise ValueError("pair_id must be nonempty and unique")
            seen.add(pair_id)
            record = {"pair_id": pair_id}
            for column, side in [("structure_a", "left"), ("structure_b", "right")]:
                path = (manifest.parent / row[column]).resolve()
                if path not in cache:
                    cache[path] = structure_tokens(path, foldseek, tokenizer)
                record[side + "_tokens"] = cache[path]
            records.append(record)
    if not records:
        raise ValueError("No pairs in manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    return len(records)


def run(args):
    binary = shutil.which(args.foldseek_bin)
    if not binary:
        raise ValueError("Foldseek not found; supply --foldseek-bin /path/to/foldseek")
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained(args.saprot_dir)
    try:
        count = prepare(args.manifest, args.output, binary, tokenizer)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Input preparation failed: {error}") from error
    print(f"Prepared {count} PPI pairs in {args.output}")


if __name__ == "__main__":
    from colbert_ppi.options import run_script

    run_script("prepare", run)
