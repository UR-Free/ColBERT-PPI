#!/usr/bin/env python3
"""Shared T07 scoring primitives for frozen validation and one-time test.

The score is fixed to mutual-top-1/top-10 in each explicitly evaluated role
assignment, followed by an arithmetic orientation mean.  Exact prepared-input
sequence groups are then Max-collapsed.  This module has no CLI and never
chooses a split; callers must provide the dataset and label matrix explicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_K = 1
CANONICAL_N = 10
SELECTION_METRIC = "val_t02_collapsed_orientation_mean_auprc"
# The frozen T02 grid was evaluated jointly up to k=10 and N=100.  Even though
# T07 consumes only k=1/N=10, retaining these maxima reproduces T02's top-k tie
# handling and floating-point cumulative-sum order exactly.
T02_GRID_MAX_K = 10
T02_GRID_MAX_N = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad3(items: list[torch.Tensor], max_len: int) -> torch.Tensor:
    padded: list[torch.Tensor] = []
    for tensor in items:
        if tensor.size(1) < max_len:
            tensor = torch.cat(
                [
                    tensor,
                    tensor.new_zeros(
                        tensor.size(0), max_len - tensor.size(1), tensor.size(2)
                    ),
                ],
                dim=1,
            )
        padded.append(tensor)
    return torch.cat(padded, dim=0)


def pad2(items: list[torch.Tensor], max_len: int) -> torch.Tensor:
    padded: list[torch.Tensor] = []
    for tensor in items:
        if tensor.size(1) < max_len:
            tensor = torch.cat(
                [tensor, tensor.new_zeros(tensor.size(0), max_len - tensor.size(1))],
                dim=1,
            )
        padded.append(tensor)
    return torch.cat(padded, dim=0).bool()


@torch.inference_mode()
def encode_both_roles(
    model: torch.nn.Module,
    loader: DataLoader,
    train_args: dict,
    device: torch.device,
) -> dict[str, object]:
    """Encode genuine query/partner roles; never infer reverse by transpose."""
    collections: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("r_q", "l_p", "r_p", "l_q", "mr", "ml")
    }
    receptor_keys: list[tuple[int, ...]] = []
    ligand_keys: list[tuple[int, ...]] = []
    for batch in loader:
        raw_mr_cpu = batch["mask1"].bool()
        raw_ml_cpu = batch["mask2"].bool()
        for index in range(raw_mr_cpu.size(0)):
            receptor_keys.append(
                tuple(batch["input_ids1"][index][raw_mr_cpu[index]].tolist())
            )
            ligand_keys.append(
                tuple(batch["input_ids2"][index][raw_ml_cpu[index]].tolist())
            )
        raw_mr = raw_mr_cpu.to(device)
        raw_ml = raw_ml_cpu.to(device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and bool(train_args.get("bf16", True)),
        ):
            outputs = model(
                input_ids1=batch["input_ids1"].to(device),
                input_ids2=batch["input_ids2"].to(device),
                mask1=raw_mr,
                mask2=raw_ml,
                return_opposite=True,
            )
        if len(outputs) != 9:
            raise RuntimeError("Model did not return four role-specific embeddings")
        r_q, l_p, _, _, _, r_p, l_q, _, _ = outputs
        for key, tensor in (
            ("r_q", r_q),
            ("l_p", l_p),
            ("r_p", r_p),
            ("l_q", l_q),
        ):
            collections[key].append(tensor.detach())
        collections["mr"].append(raw_mr[:, 1:-1].detach())
        collections["ml"].append(raw_ml[:, 1:-1].detach())

    max_r = max(tensor.size(1) for tensor in collections["r_q"])
    max_l = max(tensor.size(1) for tensor in collections["l_p"])
    return {
        "r_q": pad3(collections["r_q"], max_r),
        "l_p": pad3(collections["l_p"], max_l),
        "r_p": pad3(collections["r_p"], max_r),
        "l_q": pad3(collections["l_q"], max_l),
        "mr": pad2(collections["mr"], max_r),
        "ml": pad2(collections["ml"], max_l),
        "receptor_keys": receptor_keys,
        "ligand_keys": ligand_keys,
    }


def stable_assignment(keys: Iterable[tuple[int, ...]]) -> tuple[np.ndarray, int]:
    lookup: dict[tuple[int, ...], int] = {}
    assignment: list[int] = []
    for key in keys:
        if key not in lookup:
            lookup[key] = len(lookup)
        assignment.append(lookup[key])
    return np.asarray(assignment, dtype=np.int64), len(lookup)


def max_collapse(
    matrix: np.ndarray,
    row_assignment: np.ndarray,
    col_assignment: np.ndarray,
    shape: tuple[int, int],
    *,
    logical: bool = False,
) -> np.ndarray:
    row_grid = np.broadcast_to(row_assignment[:, None], matrix.shape)
    col_grid = np.broadcast_to(col_assignment[None, :], matrix.shape)
    if logical:
        output = np.zeros(shape, dtype=bool)
        np.logical_or.at(output, (row_grid, col_grid), matrix.astype(bool))
    else:
        output = np.full(shape, -np.inf, dtype=np.float64)
        np.maximum.at(output, (row_grid, col_grid), matrix.astype(np.float64))
        if not np.isfinite(output).all():
            raise RuntimeError("Prepared-sequence Max-collapse left an empty block")
    return output


def retrieval_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Micro AUPRC primary; query AP/MRR/Hit@1 are secondary diagnostics."""
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise RuntimeError("Invalid score or label matrix")
    query_ap: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_at_1: list[float] = []
    for directed_scores, directed_labels in ((scores, labels), (scores.T, labels.T)):
        for index in range(directed_scores.shape[0]):
            truth = directed_labels[index].astype(bool)
            if not truth.any():
                continue
            values = directed_scores[index]
            query_ap.append(
                float(average_precision_score(truth.astype(np.int8), values))
            )
            order = np.argsort(-values, kind="stable")
            first = int(np.flatnonzero(truth[order])[0]) + 1
            reciprocal_ranks.append(1.0 / first)
            hit_at_1.append(float(first == 1))
    return {
        "auprc": float(average_precision_score(labels.ravel(), scores.ravel())),
        "macro_query_ap": float(np.mean(query_ap)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit_at_1": float(np.mean(hit_at_1)),
    }


def mutual_rank_matrix_t02(logits: torch.Tensor) -> torch.Tensor:
    """Exact rank-mask construction used by the frozen T02 grid search."""
    rows, cols = logits.shape
    row_take = min(T02_GRID_MAX_K, cols)
    col_take = min(T02_GRID_MAX_K, rows)
    sentinel = T02_GRID_MAX_K + 1
    row_rank = torch.full(
        (rows, cols), sentinel, dtype=torch.int16, device=logits.device
    )
    row_indices = logits.topk(
        row_take, dim=1, largest=True, sorted=True
    ).indices
    row_values = torch.arange(
        1, row_take + 1, device=logits.device, dtype=torch.int16
    )
    row_rank.scatter_(1, row_indices, row_values.unsqueeze(0).expand(rows, -1))
    col_rank = torch.full_like(row_rank, sentinel)
    col_indices = logits.topk(
        col_take, dim=0, largest=True, sorted=True
    ).indices
    col_values = torch.arange(
        1, col_take + 1, device=logits.device, dtype=torch.int16
    )
    col_rank.scatter_(0, col_indices, col_values.unsqueeze(1).expand(-1, cols))
    return torch.maximum(row_rank, col_rank)


@torch.inference_mode()
def t02_score_matrix(
    query: torch.Tensor,
    partner: torch.Tensor,
    query_mask: torch.Tensor,
    partner_mask: torch.Tensor,
    inv_temperature: torch.Tensor,
    *,
    query_chunk: int = 1,
) -> torch.Tensor:
    """Bitwise-equivalent k=1/N=10 branch of T02 ``score_grid``."""
    n_items = query.size(0)
    scores = torch.zeros(
        (n_items, n_items), dtype=torch.float32, device=query.device
    )
    partner_float = partner.float()
    inv_temp = inv_temperature.to(device=query.device, dtype=torch.float32)
    for start in range(0, n_items, query_chunk):
        end = min(start + query_chunk, n_items)
        residue_scores = torch.einsum(
            "cld,bmd->cblm", query[start:end].float(), partner_float
        ) * inv_temp
        residue_scores.clamp_(min=-1e2, max=1e2)
        for local_index, query_index in enumerate(range(start, end)):
            valid_query = query_mask[query_index]
            for partner_index in range(n_items):
                logits = residue_scores[local_index, partner_index][valid_query][
                    :, partner_mask[partner_index]
                ]
                ranks = mutual_rank_matrix_t02(logits)
                kept = logits[ranks <= CANONICAL_K]
                take = min(T02_GRID_MAX_N, kept.numel())
                values = kept.topk(take, largest=True, sorted=True).values
                cumulative = values.cumsum(0)
                used = min(CANONICAL_N, kept.numel())
                scores[query_index, partner_index] = cumulative[used - 1]
        del residue_scores
    return scores


@torch.inference_mode()
def canonical_score_dataset(
    model: torch.nn.Module,
    dataset,
    train_args: dict,
    device: torch.device,
    labels: np.ndarray,
    *,
    batch_size: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return record/collapsed matrices plus the complete T02 metric payload."""
    labels = np.asarray(labels, dtype=bool)
    if labels.shape != (len(dataset), len(dataset)) or not labels.any():
        raise RuntimeError(
            f"Label matrix {labels.shape} is incompatible with {len(dataset)} records"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=device.type == "cuda",
    )
    embeddings = encode_both_roles(model, loader, train_args, device)
    inv_temperature = model.get_inv_temperature()
    forward = t02_score_matrix(
        embeddings["r_q"],
        embeddings["l_p"],
        embeddings["mr"],
        embeddings["ml"],
        inv_temperature,
        query_chunk=1,
    )
    reverse_l_by_r = t02_score_matrix(
        embeddings["l_q"],
        embeddings["r_p"],
        embeddings["ml"],
        embeddings["mr"],
        inv_temperature,
        query_chunk=1,
    )
    matrices = {
        "forward": forward.float().cpu().numpy().astype(np.float32),
        "reverse": reverse_l_by_r.T.float().cpu().numpy().astype(np.float32),
    }
    matrices["orientation_mean"] = (
        (matrices["forward"].astype(np.float64) + matrices["reverse"].astype(np.float64))
        / 2.0
    ).astype(np.float32)

    row_assignment, n_rows = stable_assignment(embeddings["receptor_keys"])
    col_assignment, n_cols = stable_assignment(embeddings["ligand_keys"])
    shape = (n_rows, n_cols)
    collapsed_labels = max_collapse(
        labels, row_assignment, col_assignment, shape, logical=True
    )
    collapsed = {
        orientation: max_collapse(matrix, row_assignment, col_assignment, shape)
        for orientation, matrix in matrices.items()
    }
    metrics = {
        scope: {
            orientation: retrieval_metrics(matrix, metric_labels)
            for orientation, matrix in scope_matrices.items()
        }
        for scope, scope_matrices, metric_labels in (
            ("record", matrices, labels),
            ("collapsed", collapsed, collapsed_labels),
        )
    }
    arrays = {
        "labels": labels.astype(np.int8),
        "collapsed_labels": collapsed_labels.astype(np.int8),
        "receptor_sequence_assignment": row_assignment,
        "ligand_sequence_assignment": col_assignment,
        **{f"record_{key}": value for key, value in matrices.items()},
        **{f"collapsed_{key}": value.astype(np.float32) for key, value in collapsed.items()},
    }
    payload: dict[str, object] = {
        "scoring": {
            "aggregation": "mutual row/column top-k followed by top-N sum",
            "k": CANONICAL_K,
            "N": CANONICAL_N,
            "pair_score": "mean(A-as-query:B-as-partner, B-as-query:A-as-partner)",
            "role_swap": "model return_opposite embeddings; no transpose shortcut",
            "collapse": "exact prepared-input sequence Max-collapse after orientation mean",
        },
        "counts": {
            "record_rows": int(labels.shape[0]),
            "record_cols": int(labels.shape[1]),
            "record_positives": int(labels.sum()),
            "collapsed_rows": int(n_rows),
            "collapsed_cols": int(n_cols),
            "collapsed_positives": int(collapsed_labels.sum()),
        },
        "metrics": metrics,
    }
    return arrays, payload
