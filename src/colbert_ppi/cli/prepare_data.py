"""Prepare contact-supervised ColBERT-PPI inputs from paired monomer PDB files.

Each manifest row must point to two single-chain PDB files extracted from the
same complex coordinate frame. Foldseek supplies 3Di tokens, while C-beta
coordinates from the PDB files define positive and negative residue contacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
MODIFIED_AA3_TO_1 = {
    "ASH": "D",
    "CME": "C",
    "CSO": "C",
    "FME": "M",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HYP": "P",
    "MSE": "M",
    "PCA": "E",
    "PTR": "Y",
    "SEC": "C",
    "SEP": "S",
    "TPO": "T",
}
RESIDUE_MAP = {**AA3_TO_1, **MODIFIED_AA3_TO_1}
REQUIRED_COLUMNS = {"protein_a", "protein_b", "structure_a", "structure_b"}


@dataclass(frozen=True)
class PairRecord:
    protein_a: str
    protein_b: str
    structure_a: Path
    structure_b: Path

    @property
    def pair_id(self) -> str:
        return f"{self.protein_a}:{self.protein_b}"


@dataclass(frozen=True)
class ProteinRecord:
    label: str
    structure: Path
    sequence: str
    three_di: str
    coordinates: np.ndarray
    chain_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with protein_a, protein_b, structure_a and structure_b columns",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Output prefix, such as train or val")
    parser.add_argument("--saprot-dir", type=Path, required=True)
    parser.add_argument(
        "--foldseek-bin",
        default="foldseek",
        help="Foldseek executable name or path",
    )
    parser.add_argument("--positive-threshold", type=float, default=8.0)
    parser.add_argument("--negative-threshold", type=float, default=12.0)
    parser.add_argument(
        "--max-negatives-per-pair",
        type=int,
        default=2048,
        help="Maximum stored non-contact pairs; 0 keeps all",
    )
    parser.add_argument("--max-residues", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise ValueError("Protein labels must not be empty")
    if ":" in label:
        raise ValueError(f"Protein label must not contain ':': {label!r}")
    return label


def resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    return path


def load_manifest(path: Path) -> tuple[list[PairRecord], dict[str, Path]]:
    path = path.resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        pairs: list[PairRecord] = []
        structures: dict[str, Path] = {}
        seen_pairs: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            protein_a = normalize_label(row["protein_a"])
            protein_b = normalize_label(row["protein_b"])
            structure_a = resolve_manifest_path(row["structure_a"], path.parent)
            structure_b = resolve_manifest_path(row["structure_b"], path.parent)
            record = PairRecord(protein_a, protein_b, structure_a, structure_b)
            if record.pair_id in seen_pairs:
                raise ValueError(f"Duplicate pair at manifest row {row_number}: {record.pair_id}")
            seen_pairs.add(record.pair_id)
            for label, structure in (
                (protein_a, structure_a),
                (protein_b, structure_b),
            ):
                existing = structures.setdefault(label, structure)
                if existing != structure:
                    raise ValueError(
                        f"Protein {label!r} maps to multiple structures: "
                        f"{existing} and {structure}"
                    )
            pairs.append(record)

    if not pairs:
        raise ValueError(f"Manifest contains no pairs: {path}")
    return pairs, structures


def parse_single_chain_coordinates(path: Path) -> tuple[str, np.ndarray, str]:
    chains: OrderedDict[str, OrderedDict[tuple[str, str], dict]] = OrderedDict()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in {"CA", "CB"}:
                continue
            if line[16].strip() not in {"", "A", "1"}:
                continue
            try:
                coordinate = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            chain_id = line[21].strip() or "_"
            residue_key = (line[22:26].strip(), line[26].strip())
            residue = chains.setdefault(chain_id, OrderedDict()).setdefault(
                residue_key,
                {"resname": line[17:20].strip().upper(), "atoms": {}},
            )
            residue["atoms"].setdefault(atom, coordinate)

    parsed: list[tuple[str, str, np.ndarray]] = []
    for chain_id, residues in chains.items():
        sequence: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for residue in residues.values():
            amino_acid = RESIDUE_MAP.get(str(residue["resname"]))
            coordinate = residue["atoms"].get("CB") or residue["atoms"].get("CA")
            if amino_acid is None or coordinate is None:
                continue
            sequence.append(amino_acid)
            coordinates.append(coordinate)
        if sequence:
            parsed.append(
                (
                    chain_id,
                    "".join(sequence),
                    np.asarray(coordinates, dtype=np.float32),
                )
            )

    if len(parsed) != 1:
        chain_ids = [chain_id for chain_id, _, _ in parsed]
        raise ValueError(
            f"{path} must contain exactly one recognized protein chain; found {chain_ids}"
        )
    chain_id, sequence, coordinates = parsed[0]
    return sequence, coordinates, chain_id


def resolve_executable(value: str) -> str:
    explicit = Path(value).expanduser()
    if explicit.is_file():
        return str(explicit.resolve())
    discovered = shutil.which(value)
    if discovered:
        return discovered
    raise FileNotFoundError(f"Foldseek executable not found: {value}")


def extract_three_di(pdb_path: Path, foldseek_bin: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="colbert_ppi_foldseek_") as tmpdir:
        output_path = Path(tmpdir) / "descriptor.tsv"
        command = [
            foldseek_bin,
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(pdb_path),
            str(output_path),
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Foldseek failed for {pdb_path}: {process.stderr.strip()[:500]}"
            )
        rows: list[tuple[str, str]] = []
        if output_path.exists():
            with output_path.open() as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) >= 3:
                        rows.append((fields[1].upper(), fields[2].lower()))
        if len(rows) != 1:
            raise ValueError(
                f"Foldseek must return exactly one chain for {pdb_path}; found {len(rows)}"
            )
        sequence, three_di = rows[0]
        if len(sequence) != len(three_di):
            raise ValueError(f"Foldseek AA/3Di length mismatch for {pdb_path}")
        return sequence, three_di


def contact_indices(
    coordinates_a: np.ndarray,
    coordinates_b: np.ndarray,
    *,
    positive_threshold: float,
    negative_threshold: float,
    max_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    distance_squared = (
        (coordinates_a[:, None, :] - coordinates_b[None, :, :]) ** 2
    ).sum(axis=-1)
    positives = np.argwhere(distance_squared < positive_threshold**2).astype(np.int32)
    negatives = np.argwhere(distance_squared > negative_threshold**2).astype(np.int32)
    if max_negatives > 0 and negatives.shape[0] > max_negatives:
        selected = rng.choice(negatives.shape[0], size=max_negatives, replace=False)
        negatives = negatives[np.sort(selected)]
    return positives, negatives


def prepare_proteins(
    structures: dict[str, Path],
    *,
    foldseek_bin: str,
    max_residues: int,
) -> dict[str, ProteinRecord]:
    proteins: dict[str, ProteinRecord] = {}
    for label, path in sorted(structures.items()):
        pdb_sequence, coordinates, chain_id = parse_single_chain_coordinates(path)
        foldseek_sequence, three_di = extract_three_di(path, foldseek_bin)
        if pdb_sequence != foldseek_sequence:
            raise ValueError(
                f"PDB/Foldseek sequence mismatch for {label!r}: "
                f"{len(pdb_sequence)} versus {len(foldseek_sequence)} residues"
            )
        length = min(len(pdb_sequence), max_residues)
        if length < 1:
            raise ValueError(f"Protein {label!r} contains no usable residues")
        proteins[label] = ProteinRecord(
            label=label,
            structure=path,
            sequence=pdb_sequence[:length],
            three_di=three_di[:length],
            coordinates=coordinates[:length],
            chain_id=chain_id,
        )
    return proteins


def write_outputs(
    pairs: list[PairRecord],
    proteins: dict[str, ProteinRecord],
    args: argparse.Namespace,
) -> dict[str, object]:
    from transformers import EsmTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = EsmTokenizer.from_pretrained(str(args.saprot_dir))
    tokenized: dict[str, dict[str, torch.Tensor]] = {}
    labels: list[str] = []
    offsets = np.zeros(len(proteins) + 1, dtype=np.int64)
    coordinate_blocks: list[np.ndarray] = []

    for index, (label, protein) in enumerate(sorted(proteins.items())):
        combined = "".join(
            amino_acid + three_di
            for amino_acid, three_di in zip(protein.sequence, protein.three_di)
        )
        encoded = tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            max_length=len(protein.sequence) + 2,
        )
        tokenized[label] = {
            "input_ids": encoded["input_ids"].cpu().long(),
            "attention_mask": encoded["attention_mask"].cpu().bool(),
        }
        labels.append(label)
        coordinate_blocks.append(protein.coordinates)
        offsets[index + 1] = offsets[index] + protein.coordinates.shape[0]

    contacts: dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)
    for pair in pairs:
        protein_a = proteins[pair.protein_a]
        protein_b = proteins[pair.protein_b]
        positives, negatives = contact_indices(
            protein_a.coordinates,
            protein_b.coordinates,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            max_negatives=args.max_negatives_per_pair,
            rng=rng,
        )
        if positives.shape[0] < 1 or negatives.shape[0] < 1:
            raise ValueError(
                f"{pair.pair_id} requires at least one contact and one non-contact; "
                f"found {len(positives)} and {len(negatives)}"
            )
        contacts[pair.pair_id] = {
            "shape": (len(protein_a.sequence), len(protein_b.sequence)),
            "pos_i": torch.from_numpy(positives[:, 0]),
            "pos_j": torch.from_numpy(positives[:, 1]),
            "neg_i": torch.from_numpy(negatives[:, 0]),
            "neg_j": torch.from_numpy(negatives[:, 1]),
        }

    prefix = args.prefix
    pairs_path = args.output_dir / f"{prefix}_pairs.csv"
    inputs_path = args.output_dir / f"{prefix}_saprot_inputs.pt"
    coordinates_path = args.output_dir / f"{prefix}_cb.npz"
    contacts_path = args.output_dir / f"{prefix}_contacts.pt"
    summary_path = args.output_dir / f"{prefix}_summary.json"

    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PAIRID", "HASH", "CLUSTER", "LEN"])
        for pair in pairs:
            length_a = len(proteins[pair.protein_a].sequence)
            length_b = len(proteins[pair.protein_b].sequence)
            writer.writerow([pair.pair_id, "", "", f"{length_a}:{length_b}"])

    torch.save(tokenized, inputs_path)
    np.savez_compressed(
        coordinates_path,
        labels=np.asarray(labels, dtype=np.str_),
        domain_ids=np.asarray(labels, dtype=np.str_),
        offsets=offsets,
        coords=np.concatenate(coordinate_blocks, axis=0).astype(np.float32),
    )
    torch.save({"contacts": contacts, "format": "colbert_ppi_explicit_v1"}, contacts_path)

    summary: dict[str, object] = {
        "pairs": len(pairs),
        "proteins": len(proteins),
        "positive_threshold_angstrom": args.positive_threshold,
        "negative_threshold_angstrom": args.negative_threshold,
        "max_negatives_per_pair": args.max_negatives_per_pair,
        "files": {
            "pairs_csv": str(pairs_path),
            "saprot_inputs": str(inputs_path),
            "cb_coordinates": str(coordinates_path),
            "contacts": str(contacts_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.positive_threshold <= 0:
        raise ValueError("--positive-threshold must be positive")
    if args.negative_threshold <= args.positive_threshold:
        raise ValueError("--negative-threshold must exceed --positive-threshold")
    if args.max_residues < 1:
        raise ValueError("--max-residues must be positive")
    if args.max_negatives_per_pair < 0:
        raise ValueError("--max-negatives-per-pair must be non-negative")
    if not args.saprot_dir.is_dir():
        raise FileNotFoundError(f"SaProt directory not found: {args.saprot_dir}")

    pairs, structures = load_manifest(args.manifest)
    foldseek_bin = resolve_executable(args.foldseek_bin)
    proteins = prepare_proteins(
        structures,
        foldseek_bin=foldseek_bin,
        max_residues=args.max_residues,
    )
    summary = write_outputs(pairs, proteins, args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
