#!/usr/bin/env python3
"""Build auditable PPI labels for PINDER retrieval evaluation.

The PINDER validation/test tables provide experimentally resolved positive
complexes, but the off-diagonal cells in their Cartesian retrieval matrices
are not experimentally confirmed non-interactions.  This builder therefore
separates four concepts that the legacy diagonal/zero matrix conflated:

* structural positives: PINDER complexes, expanded across duplicate UniProt
  records within the same split;
* operational negatives: eligible same-species pairs absent from the PINDER
  positive set and the prespecified STRING association set;
* verified negative evidence: the subset of operational negatives supported
  by an exact UniProt pair in Negatome 2.0 manual-stringent;
* unlabelled candidates: eligible pairs carrying STRING association evidence
  but no PINDER structural-positive label;
* censored/ineligible cells: STRING associations, train/validation edge
  overlap, cross-species pairs, and cells with missing taxonomy.

This follows the common benchmark convention, also used by RF2-PPI, of using
database-absence pairs as negative controls. Such pairs are explicitly named
"operational negatives": STRING absence is not proof that two proteins cannot
interact. STRING associations are censoring evidence and never promote a pair
to a direct physical-interaction positive.
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
PROTOCOL_VERSION = "pinder_ppi_evidence_v3_1_string_any_coverage"


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


def read_string_mapping(path: Path) -> set[str]:
    """Return canonical UniProt accessions with a successful STRING mapping."""
    mapped: set[str] = set()
    with path.open(newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = "\t" if header.count("\t") > header.count(",") else ","
        for row in csv.DictReader(handle, delimiter=delimiter):
            accession = canonical_accession(row.get("uniprot", ""))
            string_id = row.get("string_id", "").strip()
            if accession and string_id:
                mapped.add(accession)
    return mapped


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
    string_mapped_accessions: set[str],
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
    operational_negative = np.zeros((n, n), dtype=bool)
    verified_negative = np.zeros((n, n), dtype=bool)
    candidate = np.zeros((n, n), dtype=bool)
    observed_label = np.zeros((n, n), dtype=bool)
    association_censor = np.zeros((n, n), dtype=bool)
    string_coverage = np.zeros((n, n), dtype=bool)
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
            has_string_coverage = (
                receptor.receptor_accession in string_mapped_accessions
                and ligand.ligand_accession in string_mapped_accessions
            )
            has_negative = edge in negatome

            same_species[i, j] = is_same_species
            missing_taxonomy[i, j] = not has_taxonomy
            train_overlap[i, j] = is_train_overlap
            selection_overlap[i, j] = is_selection_overlap
            association_censor[i, j] = has_association
            string_coverage[i, j] = has_string_coverage

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
            operational_negative[i, j] = (
                eligible
                and has_string_coverage
                and not is_positive
                and not has_association
            )
            verified_negative[i, j] = (
                operational_negative[i, j] and has_negative and not conflict
            )
            observed_label[i, j] = positive[i, j] or operational_negative[i, j]

            statuses: list[str] = []
            if is_positive:
                statuses.append("pinder_structural_positive")
            if has_negative:
                statuses.append("negatome_negative_evidence")
            if has_association:
                statuses.append("string_association_censor")
            if not has_string_coverage:
                statuses.append("string_mapping_incomplete")
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
            elif verified_negative[i, j]:
                label_state = "operational_negative_verified_by_negatome"
            elif operational_negative[i, j]:
                label_state = "operational_negative"
            elif candidate[i, j]:
                label_state = "unjudged_candidate"
                statuses.append("unjudged_candidate")
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
                    "operational_negative": int(operational_negative[i, j]),
                    "verified_negative": int(verified_negative[i, j]),
                    "retrieval_candidate": int(candidate[i, j]),
                    "observed_label_evaluable": int(observed_label[i, j]),
                    "same_organism": int(same_species[i, j]),
                    "train_overlap": int(train_overlap[i, j]),
                    "selection_overlap": int(selection_overlap[i, j]),
                    "string_association_censor": int(association_censor[i, j]),
                    "string_mapping_coverage": int(string_coverage[i, j]),
                    "evidence_conflict": int(negative_conflict[i, j]),
                    "negatome_evidence": json.dumps(
                        negatome.get(edge, []), sort_keys=True
                    ),
                }
            )

    if np.any(positive & operational_negative):
        raise RuntimeError(f"{name}: positive and operational-negative masks overlap")
    if np.any(verified_negative & ~operational_negative):
        raise RuntimeError(f"{name}: verified negatives must be operational negatives")
    if np.any(positive & ~candidate) or np.any(operational_negative & ~candidate):
        raise RuntimeError(f"{name}: judged labels must be retrieval candidates")
    if np.any(positive & ~observed_label):
        raise RuntimeError(f"{name}: every primary positive must be observed-label evaluable")

    unknown = candidate & ~positive & ~operational_negative
    judged = positive | operational_negative
    strict_judged = positive | verified_negative
    output_npz = output_dir / f"{name}.evidence_labels.npz"
    np.savez_compressed(
        output_npz,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        receptor_labels=np.asarray([r.receptor_label for r in records]),
        ligand_labels=np.asarray([r.ligand_label for r in records]),
        receptor_accessions=receptor_accessions,
        ligand_accessions=ligand_accessions,
        positive_mask=positive,
        operational_negative_mask=operational_negative,
        verified_negative_mask=verified_negative,
        unknown_mask=unknown,
        judged_mask=judged,
        strict_judged_mask=strict_judged,
        candidate_mask=candidate,
        observed_label_mask=observed_label,
        association_censor_mask=association_censor,
        string_coverage_mask=string_coverage,
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
            "operational_negative", "verified_negative", "retrieval_candidate",
            "observed_label_evaluable", "same_organism", "train_overlap",
            "selection_overlap", "string_association_censor",
            "string_mapping_coverage", "evidence_conflict", "negatome_evidence",
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
        "operational_negative_cells": int(operational_negative.sum()),
        "verified_negative_cells": int(verified_negative.sum()),
        "unknown_candidate_cells": int(unknown.sum()),
        "retrieval_candidate_cells": int(candidate.sum()),
        "observed_label_evaluable_cells": int(observed_label.sum()),
        "operational_positive_prevalence": (
            float(positive.sum() / observed_label.sum())
            if observed_label.any() else None
        ),
        "string_association_censored_cells": int(association_censor.sum()),
        "string_covered_candidate_cells": int((candidate & string_coverage).sum()),
        "string_uncovered_candidate_cells": int((candidate & ~string_coverage).sum()),
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
        default="results/ppi_database_calibration_string_any_v3/uniprot_metadata.tsv",
    )
    parser.add_argument(
        "--string-edges",
        default="results/ppi_database_calibration_string_any_v3/string_edges.tsv",
    )
    parser.add_argument(
        "--string-mapping",
        default=(
            "results/ppi_database_calibration_string_any_v3/"
            "string_mapped_accessions.tsv"
        ),
    )
    parser.add_argument(
        "--string-provenance",
        default=(
            "results/ppi_database_calibration_string_any_v3/"
            "source_provenance.json"
        ),
    )
    parser.add_argument(
        "--negatome",
        default="data/external_binary_ppi/upstream/negatome2_manual_stringent_20191101.txt",
    )
    parser.add_argument("--string-minimum-score", type=float, default=0.0)
    parser.add_argument("--output-dir", default="results/ppi_label_protocol_v3_1")
    args = parser.parse_args()

    paths = {
        "train": ROOT / args.train_csv,
        "validation": ROOT / args.val_csv,
        "test": ROOT / args.test_csv,
        "uniprot_metadata": ROOT / args.uniprot_metadata,
        "string_edges": ROOT / args.string_edges,
        "string_mapping": ROOT / args.string_mapping,
        "string_provenance": ROOT / args.string_provenance,
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
    string_mapped_accessions = read_string_mapping(paths["string_mapping"])
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
        string_mapped_accessions=string_mapped_accessions,
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
        string_mapped_accessions=string_mapped_accessions,
        negatome=negatome,
        output_dir=output_dir,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "PPI retrieval labels with structural positives, database-absence "
            "operational negatives, association-censored candidates, and an "
            "independent Negatome evidence tier."
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
            "operational_negative": (
                "Eligible same-species pair absent from the split's PINDER "
                "structural-positive edges and from the full STRING API network "
                "query (required_score=0; current API floor 0.15). Both accessions "
                "must have successful STRING mappings. This is an assumed "
                "benchmark negative, not proof of non-interaction."
            ),
            "verified_negative_evidence": (
                "Negatome 2.0 manual-stringent exact canonical UniProt pair; it "
                "is retained only when it is also an operational negative and "
                "has no structural, STRING, train, or validation conflict."
            ),
            "unknown": (
                "Eligible non-positive pairs with any returned STRING association "
                "or incomplete STRING mapping coverage; retained as ranking "
                "competitors but excluded from the operational binary endpoint."
            ),
            "string": (
                "Any association returned by the frozen STRING required_score=0 "
                "query is censoring evidence only, not proof of direct physical "
                "interaction."
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
        "string_mapped_accessions": len(string_mapped_accessions),
        "negatome_edges": len(negatome),
        "splits": [val_summary, test_summary],
        "interpretation": {
            "primary_rank_metrics": (
                "MRR and Hit/Recall@K of known structural partners among eligible "
                "same-species candidates; unlabelled candidates remain competitors."
            ),
            "operational_binary_metrics": (
                "AUPRC and AUROC over structural positives versus prespecified "
                "database-absence operational negatives. The negative-control "
                "construction and resulting prevalence must accompany the metric."
            ),
            "strict_binary_metrics": (
                "Computed only over structural positives plus verified Negatome "
                "negative evidence as a sensitivity analysis; reported as "
                "unavailable when either class has insufficient coverage."
            ),
        },
    }
    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
