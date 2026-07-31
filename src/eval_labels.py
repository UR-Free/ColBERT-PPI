"""Utilities for calibrated PPI retrieval labels."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch


_PINDER_UNIPROT_RE = re.compile(r"_([A-Za-z0-9]+(?:-\d+)?)-[RL]$")


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
