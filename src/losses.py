"""Late-interaction scoring and retrieval metrics for ColBERT-PPI."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score


# ===========================================================================
# Mutual top-k PPI scoring (evaluation only)
# ===========================================================================

def topk_both_values(scores: torch.Tensor, k: int = 10) -> torch.Tensor:
    """返回同时位于每行和每列 top-k 的元素值，结果为 1D tensor。

    Parameters
    ----------
    scores : (L1, L2) float tensor
        Residue-level similarity matrix.
    k : int
        Number of top elements per row / column.

    Returns
    -------
    1D tensor of elements that are in both row top-k and column top-k.
    """
    if scores.dim() != 2:
        raise ValueError("scores must be 2D")
    n_rows, n_cols = scores.shape
    kr = min(max(0, k), n_cols)
    kc = min(max(0, k), n_rows)

    if kr > 0:
        _, row_idx = torch.topk(scores, k=kr, dim=1, largest=True, sorted=False)
        row_mask = torch.zeros_like(scores, dtype=torch.bool)
        row_mask.scatter_(1, row_idx, True)
    else:
        row_mask = torch.zeros_like(scores, dtype=torch.bool)

    if kc > 0:
        _, col_idx = torch.topk(scores.t(), k=kc, dim=1, largest=True, sorted=False)
        col_mask_t = torch.zeros((n_cols, n_rows), dtype=torch.bool, device=scores.device)
        col_mask_t.scatter_(1, col_idx.to(scores.device), True)
        col_mask = col_mask_t.t()
    else:
        col_mask = torch.zeros_like(scores, dtype=torch.bool)

    keep_mask = row_mask & col_mask
    return scores[keep_mask].view(-1)


def mutual_topk_ppi_scores(
    h1: torch.Tensor,
    h2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    inv_temperature: torch.Tensor,
    *,
    k: int = 10,
    top_n: int = 20,
    query_chunk: int = 8,
    attn1: Optional[torch.Tensor] = None,
    attn2: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute B×B PPI scores via mutual top-k residue filtering + top-n sum.

    For each protein pair (i, j):
      1. Compute residue-level logits: h1[i] @ h2[j].T × inv_temperature
      2. (optional) Multiply by attn1[i, :] × attn2[j, :] if provided
      3. Keep only mutual top-k (row & column top-k simultaneously)
      4. Sum top-n values as the protein-pair PPI score

    Parameters
    ----------
    h1, h2 : (B, L1, D), (B, L2, D)
        L2-normalised residue embeddings.
    mask1, mask2 : (B, L1), (B, L2) bool
        Valid residue masks (True = valid).
    inv_temperature : scalar tensor
        Learnable temperature for score scaling.
    k : int
        Mutual top-k filtering parameter.
    top_n : int
        Number of top values to sum for the protein-pair score.
    query_chunk : int
        Number of query proteins to process at once (memory bound).
    attn1, attn2 : (B, L1), (B, L2) or None
        Optional per-residue attention weights in [0, 1].
        When provided, residue logits are scaled by attn1[i,a] * attn2[j,b].
    reduction : {"sum", "mean", "max", "sqrt"}
        How to combine the retained top residue-pair logits. The historical
        default is ``sum``. ``mean``/``sqrt`` reduce peptide-length bias for
        receptor-wise peptide screening tasks.
    """
    B = h1.size(0)
    device = h1.device
    scores = torch.zeros(B, B, device=device, dtype=torch.float32)
    inv_temp = inv_temperature.to(device=device, dtype=torch.float32)
    h2_f = h2.float()
    use_attn = attn1 is not None and attn2 is not None

    for i_start in range(0, B, query_chunk):
        i_end = min(i_start + query_chunk, B)
        h1_chunk = h1[i_start:i_end].float()
        m1_chunk = mask1[i_start:i_end]
        C = i_end - i_start

        # Batch residue logits: (C, B, L1_i, L2_j)
        res_scores = torch.einsum("cld,bmd->cblm", h1_chunk, h2_f) * inv_temp
        res_scores = torch.clamp(res_scores, min=-1e2, max=1e2)

        for c in range(C):
            i = i_start + c
            m1_valid = m1_chunk[c].to(device=device, dtype=torch.bool)
            for j in range(B):
                m2_valid = mask2[j].to(device=device, dtype=torch.bool)
                if not m1_valid.any() or not m2_valid.any():
                    continue
                # Boolean indexing is required here. Padding masks are usually
                # contiguous, but pocket/interface scoring masks are sparse and
                # non-contiguous; slicing by mask.sum() would incorrectly score
                # the N-terminal prefix instead of the requested residues.
                logits_ij = res_scores[c, j][m1_valid][:, m2_valid]
                # Attention weighting: scale by attn1[i, a] * attn2[j, b]
                if use_attn:
                    a1 = attn1[i].to(device=device, dtype=torch.float32)[m1_valid]
                    a2 = attn2[j].to(device=device, dtype=torch.float32)[m2_valid]
                    logits_ij = logits_ij * a1.unsqueeze(1)**2 * a2.unsqueeze(0)**2
                filtered = topk_both_values(logits_ij, k=k)
                if filtered.numel() > 0:
                    n = min(top_n, filtered.numel())
                    values = filtered.topk(n).values
                    if reduction == "sum":
                        scores[i, j] = values.sum()
                    elif reduction == "mean":
                        scores[i, j] = values.mean()
                    elif reduction == "max":
                        scores[i, j] = values.max()
                    elif reduction == "sqrt":
                        scores[i, j] = values.sum() / (float(n) ** 0.5)
                    else:
                        raise ValueError(f"Unknown mutual_topk reduction: {reduction}")

    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def mutual_topk_ppi_scores_hardgate(
    h1: torch.Tensor,
    h2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    inv_temperature: torch.Tensor,
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    *,
    k: int = 10,
    top_n: int = 20,
    query_chunk: int = 8,
    keep_ratio: float = 0.5,
) -> torch.Tensor:
    """Hard-gate version: use attention to filter residues before similarity.

    For each protein:
      1. Keep only top ``keep_ratio`` (default 50%) residues by attention weight.
         Non-selected residues' embeddings are set to zero.
      2. Compute residue logits: (gated_h1[i]) @ (gated_h2[j]).T × inv_temp
      3. Mutual top-k + top-n sum as usual.

    This preserves embedding dynamic range while using attention purely for
    residue selection — not for scaling similarity scores.

    Parameters
    ----------
    h1, h2 : (B, L1, D), (B, L2, D)  L2-normalised residue embeddings.
    mask1, mask2 : (B, L1), (B, L2) bool
    inv_temperature : scalar tensor
    attn1, attn2 : (B, L1), (B, L2)  Per-residue attention in [0, 1].
    k, top_n, query_chunk : same as mutual_topk_ppi_scores.
    keep_ratio : float  Fraction of residues to keep per protein.
    """
    B = h1.size(0)
    device = h1.device
    scores = torch.zeros(B, B, device=device, dtype=torch.float32)
    inv_temp = inv_temperature.to(device=device, dtype=torch.float32)

    # --- Build hard-gated embeddings ---
    def _gate(h, mask, attn):
        """Zero out embeddings of residues with attention below median."""
        gated = h.float().clone()
        for i in range(h.size(0)):
            Li = int(mask[i].sum().item())
            if Li < 2:
                continue
            a = attn[i, :Li]
            k_gate = max(1, int(Li * keep_ratio))
            _, top_idx = torch.topk(a, k=k_gate, largest=True)
            keep = torch.zeros(Li, dtype=torch.bool, device=device)
            keep[top_idx] = True
            # Zero out non-selected residues
            gated[i, :Li] = gated[i, :Li] * keep.unsqueeze(-1).float()
        return gated

    h1_gated = _gate(h1, mask1, attn1)
    h2_gated = _gate(h2, mask2, attn2)

    # --- Normal mutual top-k + top-n sum on gated embeddings ---
    h2_f = h2_gated.float()

    for i_start in range(0, B, query_chunk):
        i_end = min(i_start + query_chunk, B)
        h1_chunk = h1_gated[i_start:i_end].float()
        m1_chunk = mask1[i_start:i_end]
        C = i_end - i_start

        res_scores = torch.einsum("cld,bmd->cblm", h1_chunk, h2_f) * inv_temp
        res_scores = torch.clamp(res_scores, min=-1e2, max=1e2)

        for c in range(C):
            i = i_start + c
            Li = int(m1_chunk[c].sum().item())
            for j in range(B):
                Lj = int(mask2[j].sum().item())
                if Li == 0 or Lj == 0:
                    continue
                logits_ij = res_scores[c, j, :Li, :Lj]
                filtered = topk_both_values(logits_ij, k=k)
                if filtered.numel() > 0:
                    n = min(top_n, filtered.numel())
                    scores[i, j] = filtered.topk(n).values.sum()

    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


# ===========================================================================
# Retrieval metrics (evaluation only)
# ===========================================================================

def batch_acc(ppi_scores: torch.Tensor) -> float:
    """Diagonal accuracy for a square PPI score matrix (symmetric)."""
    B = ppi_scores.shape[0]
    if B < 2:
        return 0.0
    labels = torch.arange(B, device=ppi_scores.device)
    acc_ab = (ppi_scores.argmax(dim=1) == labels).float().mean()
    acc_ba = (ppi_scores.T.argmax(dim=1) == labels).float().mean()
    return float((0.5 * (acc_ab + acc_ba)).item())


def topk_accuracy(ppi_scores: torch.Tensor, k: int = 5) -> float:
    """Top-k retrieval accuracy (symmetric)."""
    B = ppi_scores.shape[0]
    if B < 2:
        return 0.0
    k = min(k, B)
    labels = torch.arange(B, device=ppi_scores.device)
    hit_ab = (
        (ppi_scores.topk(k=k, dim=1).indices == labels.unsqueeze(1))
        .any(dim=1).float().mean()
    )
    hit_ba = (
        (ppi_scores.topk(k=k, dim=0).indices == labels.unsqueeze(0))
        .any(dim=1).float().mean()
    )
    return float((0.5 * (hit_ab + hit_ba)).item())


def hit_at_k_batch(
    ppi_scores: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[int, float]:
    """Hit@k for multiple k values (symmetric)."""
    B = ppi_scores.shape[0]
    if B < 2:
        return {k: 0.0 for k in ks}
    labels = torch.arange(B, device=ppi_scores.device)
    max_k = min(max(ks), B)
    topk_ab = ppi_scores.topk(k=max_k, dim=1).indices
    topk_ba = ppi_scores.topk(k=max_k, dim=0).indices
    return {
        k: float(
            (
                0.5 * (
                    (topk_ab[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean()
                    + (topk_ba[:k, :] == labels.unsqueeze(0)).any(dim=1).float().mean()
                )
            ).item()
        )
        for k in ks if k <= B
    }


def compute_auprc(ppi_scores: np.ndarray, positive_mask: np.ndarray | None = None) -> float:
    """Compute AUPRC from a score matrix and optional positive-pair mask."""
    N = ppi_scores.shape[0]
    if N < 2:
        return 0.0
    if not np.isfinite(ppi_scores).all():
        return 0.0
    if positive_mask is None:
        y_true = np.eye(N, dtype=np.int64).flatten()
    else:
        y_true = positive_mask.astype(np.int64).flatten()
        if y_true.sum() == 0:
            return 0.0
    y_score = ppi_scores.flatten()
    return float(average_precision_score(y_true, y_score))


def compute_retrieval_metrics(
    ppi_scores: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute comprehensive retrieval metrics from a square PPI score matrix.

    Parameters
    ----------
    ppi_scores : (N, N) float
        Square matrix of protein-level PPI scores.  Diagonal = positive pairs.
    positive_mask : (N, N) bool, optional
        Positive receptor-by-ligand labels. If provided, metrics use all
        positives in the mask instead of assuming only the diagonal is positive.

    Returns
    -------
    metrics : dict
        acc, top100, top200, top300, mrr, auprc
    """
    B = ppi_scores.shape[0]
    if B < 2:
        return {"acc": 0.0, "top100": 0.0, "top200": 0.0, "top300": 0.0,
                "mrr": 0.0, "auprc": 0.0}

    if positive_mask is None:
        # Preserve the historical symmetric diagonal-only metric.
        scores = 0.5 * (ppi_scores + ppi_scores.T)
    else:
        scores = ppi_scores
        positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
        if positive_mask.shape != scores.shape:
            raise ValueError(
                f"positive_mask shape {tuple(positive_mask.shape)} does not match "
                f"score shape {tuple(scores.shape)}"
            )

    # Guard against NaN/Inf in scores (model collapse / numerical instability)
    if not torch.isfinite(scores).all():
        return {"acc": 0.0, "top100": 0.0, "top200": 0.0, "top300": 0.0,
                "mrr": 0.0, "auprc": 0.0}

    if positive_mask is None:
        metrics = {"acc": batch_acc(scores)}

        # Top-k accuracy (equivalent to hit@k in this symmetric implementation)
        for k in (100, 200, 300):
            metrics[f"top{k}"] = topk_accuracy(scores, k=k)

        # MRR (Mean Reciprocal Rank)
        labels = torch.arange(B, device=scores.device)
        ranks_ab = (scores.argsort(dim=1, descending=True) == labels.unsqueeze(1)).float()
        ranks_ba = (scores.argsort(dim=0, descending=True) == labels.unsqueeze(0)).float()
        mrr_ab = (1.0 / (ranks_ab.argmax(dim=1).float() + 1)).mean()
        mrr_ba = (1.0 / (ranks_ba.argmax(dim=1).float() + 1)).mean()
        metrics["mrr"] = float((0.5 * (mrr_ab + mrr_ba)).item())

        # AUPRC
        metrics["auprc"] = compute_auprc(scores.detach().cpu().numpy())
        return metrics

    def _hit_at_k(score_mat: torch.Tensor, pos_mask: torch.Tensor, k: int, dim: int) -> torch.Tensor:
        k = min(k, score_mat.size(dim))
        if dim == 1:
            valid = pos_mask.any(dim=1)
            if not valid.any():
                return score_mat.new_tensor(0.0)
            topk = score_mat.topk(k=k, dim=1).indices
            hits = pos_mask.gather(1, topk).any(dim=1)
            return hits[valid].float().mean()
        valid = pos_mask.any(dim=0)
        if not valid.any():
            return score_mat.new_tensor(0.0)
        topk = score_mat.topk(k=k, dim=0).indices
        hits = pos_mask.gather(0, topk).any(dim=0)
        return hits[valid].float().mean()

    def _mrr(score_mat: torch.Tensor, pos_mask: torch.Tensor, dim: int) -> torch.Tensor:
        if dim == 1:
            valid = pos_mask.any(dim=1)
            if not valid.any():
                return score_mat.new_tensor(0.0)
            order = score_mat.argsort(dim=1, descending=True)
            ranked_pos = pos_mask.gather(1, order)
            first = ranked_pos.float().argmax(dim=1).float() + 1.0
            return (1.0 / first[valid]).mean()
        valid = pos_mask.any(dim=0)
        if not valid.any():
            return score_mat.new_tensor(0.0)
        order = score_mat.argsort(dim=0, descending=True)
        ranked_pos = pos_mask.gather(0, order)
        first = ranked_pos.float().argmax(dim=0).float() + 1.0
        return (1.0 / first[valid]).mean()

    acc_ab = _hit_at_k(scores, positive_mask, k=1, dim=1)
    acc_ba = _hit_at_k(scores, positive_mask, k=1, dim=0)
    metrics = {"acc": float((0.5 * (acc_ab + acc_ba)).item())}

    for k in (100, 200, 300):
        hit_ab = _hit_at_k(scores, positive_mask, k=k, dim=1)
        hit_ba = _hit_at_k(scores, positive_mask, k=k, dim=0)
        metrics[f"top{k}"] = float((0.5 * (hit_ab + hit_ba)).item())

    mrr_ab = _mrr(scores, positive_mask, dim=1)
    mrr_ba = _mrr(scores, positive_mask, dim=0)
    metrics["mrr"] = float((0.5 * (mrr_ab + mrr_ba)).item())
    metrics["auprc"] = compute_auprc(
        scores.detach().cpu().numpy(),
        positive_mask.detach().cpu().numpy(),
    )

    return metrics
