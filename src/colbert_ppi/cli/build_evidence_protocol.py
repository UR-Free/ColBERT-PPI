#!/usr/bin/env python3
"""Build auditable three-state PPI labels for PINDER retrieval evaluation.

The PINDER validation/test tables provide experimentally resolved positive
complexes, but the off-diagonal cells in their Cartesian retrieval matrices
are not experimentally confirmed non-interactions.  This builder therefore
separates four concepts that the legacy diagonal/zero matrix conflated:

* structural positives: PINDER complexes, expanded across duplicate UniProt
  records within the same split;
* verified negative evidence: exact UniProt pairs in Negatome 2.0 manual
  stringent, unless any positive/association evidence conflicts;
* unlabelled candidates: biologically eligible same-species pairs with no
  positive or negative evidence;
* censored/ineligible cells: STRING associations, train/validation edge
  overlap, cross-species pairs, and cells with missing taxonomy.

STRING is deliberately used only as a censoring source.  Its association score
does not establish direct physical binding and must not promote a pair to a
physical-interaction positive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path.cwd().resolve()
ACCESSION_RE = re.compile(r"_([A-Za-z0-9]+(?:-\d+)?)-[RL](?:\.pdb)?$")
PROTOCOL_VERSION = "pinder_ppi_evidence_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Use a project-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def canonical_accession(value: str) -> str:
    return value.strip().split("-", 1)[0]


def normalize_label(value: str) -> str:
    value = value.strip().lstrip(">")
    return value[:-4] if value.endswith(".pdb") else value


def accession_from_label(value: str) -> str:
    match = ACCESSION_RE.search(value.strip())
    if match is None:
        raise ValueError(f"Cannot parse UniProt accession from {value!r}")
    return canonical_accession(match.group(1))


def unordered_edge(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((canonical_accession(a), canonical_accession(b))))


def species_group(organism: str) -> str:
    """Collapse strain/isolate annotations while retaining the species name."""
    organism = " ".join(str(organism).split())
    if not organism:
        return ""
    # UniProt organism strings generally append strain/isolate detail in
    # parentheses.  The pre-parenthesis name is a stable species-level key and
    # avoids treating E. coli K-12 and E. coli O157:H7 as different species.
    return organism.split(" (", 1)[0].strip().casefold()


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    receptor_label: str
    ligand_label: str
    receptor_accession: str
    ligand_accession: str


def read_pairs(path: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PAIRID" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a PAIRID column")
        for row in reader:
            pair_id = row["PAIRID"].strip()
            receptor, ligand = pair_id.split(":", 1)
            records.append(
                PairRecord(
                    pair_id=pair_id,
                    receptor_label=normalize_label(receptor),
                    ligand_label=normalize_label(ligand),
                    receptor_accession=accession_from_label(receptor),
                    ligand_accession=accession_from_label(ligand),
                )
            )
    if not records:
        raise ValueError(f"No pairs found in {path}")
    return records


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(handle, dialect=dialect)
        rows = {}
        for row in reader:
            accession = canonical_accession(row.get("uniprot", ""))
            if accession:
                rows[accession] = {
                    "taxon_id": row.get("taxon_id", "").strip(),
                    "organism": row.get("organism", "").strip(),
                    "species_group": species_group(row.get("organism", "")),
                }
    return rows


def read_string_associations(path: Path, minimum_score: float) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    edges: set[tuple[str, str]] = set()
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        for row in csv.DictReader(handle, dialect=dialect):
            score = float(row.get("score", 0) or 0)
            if score >= minimum_score:
                edges.add(unordered_edge(row["uniprot_a"], row["uniprot_b"]))
    return edges


def read_negatome(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    evidence: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            edge = unordered_edge(fields[0], fields[1])
            evidence.setdefault(edge, []).append(
                {
                    "source": "Negatome2_manual_stringent",
                    "pmid": fields[2].strip() if len(fields) > 2 else "",
                    "method": fields[3].strip() if len(fields) > 3 else "",
                    "line": str(line_number),
                }
            )
    return evidence


def edge_set(records: Iterable[PairRecord]) -> set[tuple[str, str]]:
    return {
        unordered_edge(record.receptor_accession, record.ligand_accession)
        for record in records
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_split(
    *,
    name: str,
    source_path: Path,
    records: list[PairRecord],
    metadata: dict[str, dict[str, str]],
    train_edges: set[tuple[str, str]],
    selection_edges: set[tuple[str, str]],
    string_edges: set[tuple[str, str]],
    negatome: dict[tuple[str, str], list[dict[str, str]]],
    output_dir: Path,
) -> dict[str, object]:
    n = len(records)
    receptor_accessions = np.asarray([r.receptor_accession for r in records])
    ligand_accessions = np.asarray([r.ligand_accession for r in records])
    receptor_species = np.asarray([
        metadata.get(accession, {}).get("species_group", "")
        for accession in receptor_accessions
    ])
    ligand_species = np.asarray([
        metadata.get(accession, {}).get("species_group", "")
        for accession in ligand_accessions
    ])

    split_edges = edge_set(records)
    positive = np.zeros((n, n), dtype=bool)
    negative = np.zeros((n, n), dtype=bool)
    candidate = np.zeros((n, n), dtype=bool)
    observed_label = np.zeros((n, n), dtype=bool)
    association_censor = np.zeros((n, n), dtype=bool)
    train_overlap = np.zeros((n, n), dtype=bool)
    selection_overlap = np.zeros((n, n), dtype=bool)
    same_species = np.zeros((n, n), dtype=bool)
    missing_taxonomy = np.zeros((n, n), dtype=bool)
    negative_conflict = np.zeros((n, n), dtype=bool)
    ledger: list[dict[str, object]] = []

    for i, receptor in enumerate(records):
        rec_species = receptor_species[i]
        for j, ligand in enumerate(records):
            lig_species = ligand_species[j]
            edge = unordered_edge(receptor.receptor_accession, ligand.ligand_accession)
            is_positive = edge in split_edges
            has_taxonomy = bool(rec_species and lig_species)
            is_same_species = has_taxonomy and rec_species == lig_species
            is_train_overlap = edge in train_edges
            is_selection_overlap = edge in selection_edges
            has_association = edge in string_edges and not is_positive
            has_negative = edge in negatome

            same_species[i, j] = is_same_species
            missing_taxonomy[i, j] = not has_taxonomy
            train_overlap[i, j] = is_train_overlap
            selection_overlap[i, j] = is_selection_overlap
            association_censor[i, j] = has_association

            eligible = (
                is_same_species
                and not is_train_overlap
                and not is_selection_overlap
            )
            candidate[i, j] = eligible
            positive[i, j] = eligible and is_positive

            conflict = has_negative and (
                is_positive or has_association or is_train_overlap or is_selection_overlap
            )
            negative_conflict[i, j] = conflict
            negative[i, j] = eligible and has_negative and not conflict

            # Observed-label AP is a positive-vs-unlabelled sensitivity, not a
            # verified binary metric.  Known STRING associations are censored
            # so they cannot be counted as nominal zeroes.
            observed_label[i, j] = eligible and not has_association

            statuses: list[str] = []
            if is_positive:
                statuses.append("pinder_structural_positive")
            if has_negative:
                statuses.append("negatome_negative_evidence")
            if has_association:
                statuses.append("string_association_censor")
            if is_train_overlap:
                statuses.append("train_edge_overlap")
            if is_selection_overlap:
                statuses.append("validation_edge_overlap")
            if not is_same_species:
                statuses.append("cross_species_or_missing_taxonomy")
            if conflict:
                statuses.append("evidence_conflict")
            if positive[i, j]:
                label_state = "structural_positive"
            elif negative[i, j]:
                label_state = "verified_negative"
            elif candidate[i, j]:
                label_state = "unlabelled_candidate"
                statuses.append("unlabelled_candidate")
            else:
                label_state = "ineligible"
            ledger.append(
                {
                    "receptor_row": i,
                    "ligand_col": j,
                    "receptor_id": receptor.receptor_label,
                    "ligand_id": ligand.ligand_label,
                    "receptor_uniprot": receptor.receptor_accession,
                    "ligand_uniprot": ligand.ligand_accession,
                    "receptor_species": rec_species,
                    "ligand_species": lig_species,
                    "label_state": label_state,
                    "statuses": "|".join(statuses),
                    "primary_positive": int(positive[i, j]),
                    "verified_negative": int(negative[i, j]),
                    "retrieval_candidate": int(candidate[i, j]),
                    "observed_label_evaluable": int(observed_label[i, j]),
                    "same_organism": int(same_species[i, j]),
                    "train_overlap": int(train_overlap[i, j]),
                    "selection_overlap": int(selection_overlap[i, j]),
                    "string_association_censor": int(association_censor[i, j]),
                    "evidence_conflict": int(negative_conflict[i, j]),
                    "negatome_evidence": json.dumps(
                        negatome.get(edge, []), sort_keys=True
                    ),
                }
            )

    if np.any(positive & negative):
        raise RuntimeError(f"{name}: positive and negative masks overlap")
    if np.any(positive & ~candidate) or np.any(negative & ~candidate):
        raise RuntimeError(f"{name}: judged labels must be retrieval candidates")
    if np.any(positive & ~observed_label):
        raise RuntimeError(f"{name}: every primary positive must be observed-label evaluable")

    unknown = candidate & ~positive & ~negative
    judged = positive | negative
    output_npz = output_dir / f"{name}.evidence_labels.npz"
    np.savez_compressed(
        output_npz,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        receptor_labels=np.asarray([r.receptor_label for r in records]),
        ligand_labels=np.asarray([r.ligand_label for r in records]),
        receptor_accessions=receptor_accessions,
        ligand_accessions=ligand_accessions,
        positive_mask=positive,
        verified_negative_mask=negative,
        unknown_mask=unknown,
        judged_mask=judged,
        candidate_mask=candidate,
        observed_label_mask=observed_label,
        association_censor_mask=association_censor,
        train_overlap_mask=train_overlap,
        selection_overlap_mask=selection_overlap,
        same_species_mask=same_species,
        missing_taxonomy_mask=missing_taxonomy,
        negative_conflict_mask=negative_conflict,
    )
    ledger_path = output_dir / f"{name}.evidence_ledger.csv"
    write_csv(
        ledger_path,
        ledger,
        [
            "receptor_row", "ligand_col", "receptor_id", "ligand_id",
            "receptor_uniprot", "ligand_uniprot", "receptor_species",
            "ligand_species", "label_state", "statuses", "primary_positive",
            "verified_negative", "retrieval_candidate",
            "observed_label_evaluable", "same_organism", "train_overlap",
            "selection_overlap", "string_association_censor",
            "evidence_conflict", "negatome_evidence",
        ],
    )

    diagonal_positive = np.diag(positive)
    diagonal_missing_taxonomy = np.diag(missing_taxonomy)
    diagonal_same_species = np.diag(same_species)
    diagonal_train_overlap = np.diag(train_overlap)
    diagonal_selection_overlap = np.diag(selection_overlap)
    diagonal_exclusion_combinations: Counter[str] = Counter()
    for index in np.flatnonzero(~diagonal_positive):
        reasons: list[str] = []
        if diagonal_missing_taxonomy[index]:
            reasons.append("missing_taxonomy")
        elif not diagonal_same_species[index]:
            reasons.append("cross_species")
        if diagonal_train_overlap[index]:
            reasons.append("train_overlap")
        if diagonal_selection_overlap[index]:
            reasons.append("validation_overlap")
        diagonal_exclusion_combinations["+".join(reasons) or "other"] += 1
    candidate_per_row = candidate.sum(axis=1)
    candidate_per_col = candidate.sum(axis=0)
    summary: dict[str, object] = {
        "split": name,
        "source_csv": display_path(source_path),
        "source_sha256": sha256(source_path),
        "records": n,
        "matrix_cells": int(n * n),
        "unique_receptor_accessions": int(len(set(receptor_accessions.tolist()))),
        "unique_ligand_accessions": int(len(set(ligand_accessions.tolist()))),
        "split_structural_edges": int(len(split_edges)),
        "primary_positive_cells": int(positive.sum()),
        "primary_diagonal_positives": int(diagonal_positive.sum()),
        "excluded_diagonal_positives": int((~diagonal_positive).sum()),
        "diagonal_exclusion_combinations": dict(
            sorted(diagonal_exclusion_combinations.items())
        ),
        "verified_negative_cells": int(negative.sum()),
        "unknown_candidate_cells": int(unknown.sum()),
        "retrieval_candidate_cells": int(candidate.sum()),
        "observed_label_evaluable_cells": int(observed_label.sum()),
        "string_association_censored_cells": int(association_censor.sum()),
        "train_overlap_cells": int(train_overlap.sum()),
        "selection_overlap_cells": int(selection_overlap.sum()),
        "cross_species_cells": int((~same_species & ~missing_taxonomy).sum()),
        "missing_taxonomy_cells": int(missing_taxonomy.sum()),
        "negative_conflict_cells": int(negative_conflict.sum()),
        "positive_rows_with_at_least_2_candidates": int(
            np.sum(positive.any(axis=1) & (candidate_per_row >= 2))
        ),
        "positive_cols_with_at_least_2_candidates": int(
            np.sum(positive.any(axis=0) & (candidate_per_col >= 2))
        ),
        "npz": display_path(output_npz),
        "npz_sha256": sha256(output_npz),
        "evidence_ledger": display_path(ledger_path),
        "evidence_ledger_sha256": sha256(ledger_path),
        "evidence_ledger_rows": len(ledger),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default="data/processed_pinder_afdb_esmatlas/pinder_train_afdb.csv",
    )
    parser.add_argument(
        "--val-csv",
        default="data/processed_pinder_afdb_esmatlas/pinder_val_hetero_afdb.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed_pinder_afdb_esmatlas/pinder_test_hetero_afdb.csv",
    )
    parser.add_argument(
        "--uniprot-metadata",
        default="results/ppi_database_calibration/uniprot_metadata.tsv",
    )
    parser.add_argument(
        "--string-edges",
        default="results/ppi_database_calibration/string_edges.tsv",
    )
    parser.add_argument(
        "--negatome",
        default="data/external_binary_ppi/upstream/negatome2_manual_stringent_20191101.txt",
    )
    parser.add_argument("--string-minimum-score", type=float, default=0.7)
    parser.add_argument("--output-dir", default="results/ppi_label_protocol_v2")
    args = parser.parse_args()

    paths = {
        "train": ROOT / args.train_csv,
        "validation": ROOT / args.val_csv,
        "test": ROOT / args.test_csv,
        "uniprot_metadata": ROOT / args.uniprot_metadata,
        "string_edges": ROOT / args.string_edges,
        "negatome": ROOT / args.negatome,
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} input: {path}")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_pairs(paths["train"])
    val_records = read_pairs(paths["validation"])
    test_records = read_pairs(paths["test"])
    metadata = read_metadata(paths["uniprot_metadata"])
    string_edges = read_string_associations(paths["string_edges"], args.string_minimum_score)
    negatome = read_negatome(paths["negatome"])
    train_known = edge_set(train_records)
    val_known = edge_set(val_records)

    val_summary = build_split(
        name="pinder_val_hetero_afdb",
        source_path=paths["validation"],
        records=val_records,
        metadata=metadata,
        train_edges=train_known,
        selection_edges=set(),
        string_edges=string_edges,
        negatome=negatome,
        output_dir=output_dir,
    )
    test_summary = build_split(
        name="pinder_test_hetero_afdb",
        source_path=paths["test"],
        records=test_records,
        metadata=metadata,
        train_edges=train_known,
        # Exact validation edges are unavailable for an untouched final test.
        selection_edges=val_known,
        string_edges=string_edges,
        negatome=negatome,
        output_dir=output_dir,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Three-state PPI retrieval labels: structural positive, verified "
            "negative evidence, or unlabelled; no absence-as-negative assumption."
        ),
        "rules": {
            "organism_group": (
                "UniProt organism names are normalized by removing parenthetical "
                "strain/isolate annotations and applying case folding; missing or "
                "different groups are excluded from primary within-organism retrieval."
            ),
            "positive": (
                "PINDER experimental-complex edge, expanded across duplicate "
                "canonical UniProt records within the split; primary analysis "
                "requires same-species eligibility and no train/selection overlap."
            ),
            "negative": (
                "Negatome 2.0 manual-stringent exact canonical UniProt pair only; "
                "conflicts with any positive, STRING association, train edge, or "
                "validation edge are censored rather than forced negative."
            ),
            "unknown": (
                "All other biologically eligible same-species candidate pairs."
            ),
            "string": (
                "High-confidence STRING association is censoring evidence only, "
                "not proof of direct physical interaction."
            ),
            "cross_species": (
                "Excluded from the primary within-species retrieval task; known "
                "cross-species PINDER complexes are reported separately."
            ),
            "test_firewall": (
                "Exact train and validation edges are excluded from the final "
                "test candidate universe."
            ),
        },
        "input_files": {
            key: {"path": display_path(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "implementation": {
            "builder": display_path(Path(__file__)),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
        "string_minimum_score": args.string_minimum_score,
        "metadata_accessions": len(metadata),
        "string_association_edges": len(string_edges),
        "negatome_edges": len(negatome),
        "splits": [val_summary, test_summary],
        "interpretation": {
            "primary_rank_metrics": (
                "MRR and Hit/Recall@K of known structural partners among eligible "
                "same-species candidates; unlabelled candidates remain competitors."
            ),
            "observed_label_auprc": (
                "Positive-versus-unlabelled sensitivity after censoring known "
                "associations; it is not a true-negative binary AUPRC."
            ),
            "strict_binary_metrics": (
                "Computed only over structural positives plus verified Negatome "
                "negative evidence and reported as unavailable when either class "
                "has insufficient coverage."
            ),
        },
    }
    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
