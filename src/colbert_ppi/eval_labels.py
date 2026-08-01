"""Utilities for calibrated PPI retrieval labels."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch


_PINDER_UNIPROT_RE = re.compile(r"_([A-Za-z0-9]+(?:-\d+)?)-[RL]$")


@dataclass(frozen=True)
class RetrievalLabelProtocol:
    """Three-state, evidence-aware labels for a retrieval matrix.

    ``candidate_mask`` defines the biologically eligible retrieval universe.
    ``positive_mask`` and ``verified_negative_mask`` are disjoint judged
    subsets.  The remaining candidate cells are explicitly unlabelled.
    ``observed_label_mask`` is the prespecified positive-vs-unlabelled
    sensitivity set after censoring known associations; it must not be
    described as a matrix of confirmed negatives.
    """

    positive_mask: torch.Tensor
    verified_negative_mask: torch.Tensor
    candidate_mask: torch.Tensor
    observed_label_mask: torch.Tensor
    unknown_mask: torch.Tensor
    source: str
    protocol_version: str

    def validate(self) -> None:
        masks = {
            "positive_mask": self.positive_mask,
            "verified_negative_mask": self.verified_negative_mask,
            "candidate_mask": self.candidate_mask,
            "observed_label_mask": self.observed_label_mask,
            "unknown_mask": self.unknown_mask,
        }
        shapes = {tuple(mask.shape) for mask in masks.values()}
        if len(shapes) != 1:
            raise ValueError(f"Protocol masks have inconsistent shapes: {shapes}")
        shape = next(iter(shapes))
        if len(shape) != 2:
            raise ValueError(f"Protocol masks must be matrices, got {shape}")
        if (self.positive_mask & self.verified_negative_mask).any():
            raise ValueError("Positive and verified-negative masks overlap")
        judged = self.positive_mask | self.verified_negative_mask
        if (judged & ~self.candidate_mask).any():
            raise ValueError("Judged labels must be retrieval candidates")
        if (self.positive_mask & ~self.observed_label_mask).any():
            raise ValueError("Every primary positive must be observed-label evaluable")
        if (self.observed_label_mask & ~self.candidate_mask).any():
            raise ValueError("Observed-label cells must be retrieval candidates")
        expected_unknown = (
            self.candidate_mask
            & ~self.positive_mask
            & ~self.verified_negative_mask
        )
        if not torch.equal(self.unknown_mask, expected_unknown):
            raise ValueError("unknown_mask is inconsistent with candidate/judged masks")


def normalize_pair_label(label: str) -> str:
    label = label.strip()
    if label.startswith(">"):
        label = label[1:]
    if label.endswith(".pdb"):
        label = label[:-4]
    return label


def extract_uniprot_accession(label: str) -> str:
    """Return the canonical UniProt accession encoded in a PINDER label.

    Labels that do not encode an accession deliberately remain distinct.  This
    avoids accidentally collapsing unrelated records on a weak name heuristic.
    """
    normalized = normalize_pair_label(label)
    match = _PINDER_UNIPROT_RE.search(normalized)
    if match is None:
        return f"RECORD::{normalized}"
    return match.group(1).split("-", 1)[0]


def collapse_retrieval_by_uniprot(
    scores: np.ndarray,
    positive_mask: np.ndarray | torch.Tensor | None,
    samples: Sequence[Tuple[str, str]],
    *,
    reduction: str = "max",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Collapse a record-level retrieval matrix to unique UniProt entities.

    Score cells mapping to the same UniProt pair are reduced with either the
    maximum or arithmetic mean. Positive labels are always collapsed with OR,
    so duplicate records cannot inflate the number of positive edges.
    """
    score_array = np.asarray(scores)
    n = len(samples)
    if score_array.shape != (n, n):
        raise ValueError(
            f"score shape {score_array.shape} does not match {n} dataset samples"
        )
    if reduction not in {"max", "mean"}:
        raise ValueError(f"Unsupported collapse reduction: {reduction}")

    if positive_mask is None:
        positive_array = np.eye(n, dtype=bool)
    elif torch.is_tensor(positive_mask):
        positive_array = positive_mask.detach().cpu().numpy().astype(bool, copy=False)
    else:
        positive_array = np.asarray(positive_mask, dtype=bool)
    if positive_array.shape != (n, n):
        raise ValueError(
            f"positive mask shape {positive_array.shape} does not match {(n, n)}"
        )

    receptor_ids = [extract_uniprot_accession(left) for left, _ in samples]
    ligand_ids = [extract_uniprot_accession(right) for _, right in samples]

    def _ordered_groups(labels: Sequence[str]) -> tuple[list[str], list[list[int]]]:
        ordered: list[str] = []
        groups_by_label: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            if label not in groups_by_label:
                ordered.append(label)
                groups_by_label[label] = []
            groups_by_label[label].append(index)
        return ordered, [groups_by_label[label] for label in ordered]

    unique_receptors, receptor_groups = _ordered_groups(receptor_ids)
    unique_ligands, ligand_groups = _ordered_groups(ligand_ids)
    collapsed_scores = np.empty(
        (len(receptor_groups), len(ligand_groups)), dtype=score_array.dtype
    )
    collapsed_positive = np.zeros(collapsed_scores.shape, dtype=bool)

    for row, row_indices in enumerate(receptor_groups):
        for col, col_indices in enumerate(ligand_groups):
            cell_scores = score_array[np.ix_(row_indices, col_indices)]
            if reduction == "max":
                collapsed_scores[row, col] = np.max(cell_scores)
            else:
                collapsed_scores[row, col] = np.mean(cell_scores)
            collapsed_positive[row, col] = bool(
                positive_array[np.ix_(row_indices, col_indices)].any()
            )

    metadata: dict[str, object] = {
        "record_rows": n,
        "record_cols": n,
        "unique_receptors": len(unique_receptors),
        "unique_ligands": len(unique_ligands),
        "record_positive_pairs": int(positive_array.sum()),
        "unique_positive_edges": int(collapsed_positive.sum()),
        "receptor_uniprots": unique_receptors,
        "ligand_uniprots": unique_ligands,
        "reduction": reduction,
    }
    return collapsed_scores, collapsed_positive, metadata


def collapse_retrieval_protocol_by_uniprot(
    scores: np.ndarray,
    protocol: RetrievalLabelProtocol,
    samples: Sequence[Tuple[str, str]],
    *,
    reduction: str = "max",
) -> tuple[np.ndarray, RetrievalLabelProtocol, dict[str, object]]:
    """Collapse scores and every evidence-aware mask to UniProt entities."""
    score_array = np.asarray(scores)
    n = len(samples)
    if score_array.shape != (n, n):
        raise ValueError(
            f"score shape {score_array.shape} does not match {n} dataset samples"
        )
    protocol.validate()
    if tuple(protocol.positive_mask.shape) != (n, n):
        raise ValueError("Protocol shape does not match samples")
    if reduction not in {"max", "mean"}:
        raise ValueError(f"Unsupported collapse reduction: {reduction}")

    receptor_ids = [extract_uniprot_accession(left) for left, _ in samples]
    ligand_ids = [extract_uniprot_accession(right) for _, right in samples]

    def ordered_groups(labels: Sequence[str]) -> tuple[list[str], list[list[int]]]:
        ordered: list[str] = []
        mapping: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            if label not in mapping:
                ordered.append(label)
                mapping[label] = []
            mapping[label].append(index)
        return ordered, [mapping[label] for label in ordered]

    unique_receptors, receptor_groups = ordered_groups(receptor_ids)
    unique_ligands, ligand_groups = ordered_groups(ligand_ids)
    shape = (len(receptor_groups), len(ligand_groups))
    collapsed_scores = np.empty(shape, dtype=score_array.dtype)
    mask_names = (
        "positive_mask",
        "verified_negative_mask",
        "candidate_mask",
        "observed_label_mask",
    )
    source_masks = {
        name: getattr(protocol, name).detach().cpu().numpy().astype(bool, copy=False)
        for name in mask_names
    }
    collapsed_masks = {name: np.zeros(shape, dtype=bool) for name in mask_names}

    for row, row_indices in enumerate(receptor_groups):
        for col, col_indices in enumerate(ligand_groups):
            block = score_array[np.ix_(row_indices, col_indices)]
            collapsed_scores[row, col] = (
                np.max(block) if reduction == "max" else np.mean(block)
            )
            for name in mask_names:
                collapsed_masks[name][row, col] = bool(
                    source_masks[name][np.ix_(row_indices, col_indices)].any()
                )

    # Evidence is keyed at the UniProt-pair level.  Still resolve any
    # unexpected conflict conservatively so it cannot survive entity collapse.
    conflict = (
        collapsed_masks["positive_mask"]
        & collapsed_masks["verified_negative_mask"]
    )
    collapsed_masks["verified_negative_mask"][conflict] = False
    unknown = (
        collapsed_masks["candidate_mask"]
        & ~collapsed_masks["positive_mask"]
        & ~collapsed_masks["verified_negative_mask"]
    )
    collapsed_protocol = RetrievalLabelProtocol(
        positive_mask=torch.from_numpy(collapsed_masks["positive_mask"]),
        verified_negative_mask=torch.from_numpy(
            collapsed_masks["verified_negative_mask"]
        ),
        candidate_mask=torch.from_numpy(collapsed_masks["candidate_mask"]),
        observed_label_mask=torch.from_numpy(
            collapsed_masks["observed_label_mask"]
        ),
        unknown_mask=torch.from_numpy(unknown),
        source=protocol.source,
        protocol_version=protocol.protocol_version,
    )
    collapsed_protocol.validate()
    metadata: dict[str, object] = {
        "record_rows": n,
        "record_cols": n,
        "unique_receptors": len(unique_receptors),
        "unique_ligands": len(unique_ligands),
        "record_positive_pairs": int(protocol.positive_mask.sum().item()),
        "unique_positive_edges": int(collapsed_protocol.positive_mask.sum().item()),
        "unique_verified_negative_edges": int(
            collapsed_protocol.verified_negative_mask.sum().item()
        ),
        "unique_candidate_pairs": int(collapsed_protocol.candidate_mask.sum().item()),
        "receptor_uniprots": unique_receptors,
        "ligand_uniprots": unique_ligands,
        "reduction": reduction,
        "evidence_conflicts_removed": int(conflict.sum()),
    }
    return collapsed_scores, collapsed_protocol, metadata


def load_positive_mask(
    csv_path: str | Path,
    samples: Sequence[Tuple[str, str]],
) -> torch.Tensor:
    """Load a calibrated receptor-by-ligand positive mask.

    Parameters
    ----------
    csv_path
        CSV produced by `calibrate_pinder_with_ppi_databases.py`, with
        `receptor_id` and `ligand_id` columns.
    samples
        Dataset sample order, as `(receptor_label, ligand_label)` pairs.
    """
    path = Path(csv_path)
    n = len(samples)
    receptor_to_rows: dict[str, list[int]] = {}
    ligand_to_cols: dict[str, list[int]] = {}
    for idx, (receptor, ligand) in enumerate(samples):
        receptor_to_rows.setdefault(normalize_pair_label(receptor), []).append(idx)
        ligand_to_cols.setdefault(normalize_pair_label(ligand), []).append(idx)

    mask = torch.zeros((n, n), dtype=torch.bool)
    missing = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            receptor = normalize_pair_label(row.get("receptor_id", ""))
            ligand = normalize_pair_label(row.get("ligand_id", ""))
            rows = receptor_to_rows.get(receptor, [])
            cols = ligand_to_cols.get(ligand, [])
            if not rows or not cols:
                missing += 1
                continue
            for i in rows:
                for j in cols:
                    mask[i, j] = True
    if missing:
        raise ValueError(f"{missing} calibrated pairs in {path} did not match dataset labels")
    return mask


def load_retrieval_label_protocol(
    npz_path: str | Path,
    samples: Sequence[Tuple[str, str]],
) -> RetrievalLabelProtocol:
    """Load and order-check an evidence-aware PPI label protocol.

    The NPZ stores the exact receptor/ligand record order used by the builder.
    Evaluation aborts on any mismatch rather than silently applying labels to
    a filtered or reordered dataset.
    """
    path = Path(npz_path)
    payload = np.load(path, allow_pickle=False)
    required = {
        "protocol_version",
        "receptor_labels",
        "ligand_labels",
        "positive_mask",
        "verified_negative_mask",
        "unknown_mask",
        "candidate_mask",
        "observed_label_mask",
    }
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"{path} is missing protocol arrays: {sorted(missing)}")

    expected_receptors = [normalize_pair_label(left) for left, _ in samples]
    expected_ligands = [normalize_pair_label(right) for _, right in samples]
    stored_receptors = [normalize_pair_label(str(x)) for x in payload["receptor_labels"]]
    stored_ligands = [normalize_pair_label(str(x)) for x in payload["ligand_labels"]]
    if stored_receptors != expected_receptors or stored_ligands != expected_ligands:
        mismatch = next(
            (
                index
                for index, values in enumerate(
                    zip(
                        stored_receptors,
                        stored_ligands,
                        expected_receptors,
                        expected_ligands,
                    )
                )
                if values[0] != values[2] or values[1] != values[3]
            ),
            None,
        )
        raise ValueError(
            f"Protocol/sample order mismatch for {path}; first mismatch index={mismatch}"
        )

    def as_bool(name: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(payload[name], dtype=bool).copy())

    protocol = RetrievalLabelProtocol(
        positive_mask=as_bool("positive_mask"),
        verified_negative_mask=as_bool("verified_negative_mask"),
        candidate_mask=as_bool("candidate_mask"),
        observed_label_mask=as_bool("observed_label_mask"),
        unknown_mask=as_bool("unknown_mask"),
        source=str(path),
        protocol_version=str(np.asarray(payload["protocol_version"]).item()),
    )
    expected_shape = (len(samples), len(samples))
    if tuple(protocol.positive_mask.shape) != expected_shape:
        raise ValueError(
            f"Protocol shape {tuple(protocol.positive_mask.shape)} does not match "
            f"dataset matrix {expected_shape}"
        )
    protocol.validate()
    return protocol
