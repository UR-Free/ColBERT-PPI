"""
Training and evaluation loops for v8 SaProt ColBERT PPI model.

The only training objective is residue-contact CLIP / InfoNCE.

Pipeline:
  1. SaProt residues → independent query/candidate MLP projections
  2. Sample positive (CB < 8Å) and negative (CB > 12Å) residue pairs from contact matrix
  3. Build cross-chain similarity matrix with learnable temperature
  4. Bidirectional InfoNCE loss (v1-style clip_loss)

Mutual top-k PPI scoring is computed for evaluation only (no PPI loss):
  bidirectional residue logits → mutual top-10 filter → top-20 sum.

Key functions:
  - _sample_contact_pairs: extract pos/neg residue embeddings from contact matrix
  - _residue_clip_loss:   v1-style bidirectional InfoNCE on residue similarity matrix
  - compute_batch_losses:  per-batch forward + contact-only loss
  - run_train_epoch:       single training epoch
  - run_eval_epoch:        validation epoch with retrieval metrics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .retrieval import (
    mutual_topk_ppi_scores,
    compute_retrieval_metrics,
)

logger = logging.getLogger(__name__)

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _safe_binary_metrics(labels: list[np.ndarray], scores: list[np.ndarray]) -> tuple[float, float]:
    if not labels or not scores:
        return float("nan"), float("nan")
    y_true = np.concatenate(labels).astype(np.int64)
    y_score = np.concatenate(scores).astype(np.float32)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.size == 0:
        return float("nan"), float("nan")
    if y_true.sum() == 0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_true, y_score))
    if np.unique(y_true).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
    return auprc, auroc


# ===========================================================================
# Residue pair sampling from contact matrix
# ===========================================================================

def _sample_contact_pairs(
    h1: torch.Tensor,
    h2: torch.Tensor,
    contact_matrix: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    pos_per_sample: int = 5,
    neg_per_sample: int = 5,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample positive and negative residue pairs from a contact matrix.

    Positive pairs:  contact_matrix ==  1  (CB < 8 Å)
    Negative pairs:  contact_matrix == -1  (CB > 12 Å)

    Gathers the corresponding residue embeddings from the encoded
    representations h1, h2. Also returns positive residue indices for
    residue-level evaluation.

    Parameters
    ----------
    h1, h2 : (B, L1, D), (B, L2, D)
        L2-normalised residue embeddings from the encoder.
    contact_matrix : (B, L1, L2) int
        1 = contact, -1 = non-contact, 0 = ignored.
    contact_mask : (B, L1, L2) bool
        Valid residue pair mask.
    pos_per_sample, neg_per_sample : int
        Max number of positive / negative pairs to sample per complex.
    Returns
    -------
    pos_h1, pos_h2 : (N_pos, D)   paired positive residue embeddings
    neg_h1, neg_h2 : (N_neg, D)   paired negative residue embeddings
    N_pos : int                   number of positive pairs
    pos_batch_ids : (N_pos,)      batch index for each positive pair
    pos_i, pos_j : (N_pos,)       residue indices for each positive pair
    """
    B = h1.size(0)
    pos_i_parts: list[torch.Tensor] = []
    pos_j_parts: list[torch.Tensor] = []
    pos_batch_parts: list[torch.Tensor] = []
    neg_i_parts: list[torch.Tensor] = []
    neg_j_parts: list[torch.Tensor] = []
    neg_batch_parts: list[torch.Tensor] = []

    for b in range(B):
        valid = contact_mask[b] & (contact_matrix[b] != 0)
        if not valid.any():
            continue

        for sign, (i_buf, j_buf, bid_buf), limit in [
            (1,  (pos_i_parts, pos_j_parts, pos_batch_parts), pos_per_sample),
            (-1, (neg_i_parts, neg_j_parts, neg_batch_parts), neg_per_sample),
        ]:
            idx = torch.nonzero(
                valid & (contact_matrix[b] == sign), as_tuple=False
            )
            if idx.size(0) > limit:
                idx = idx[
                    torch.randperm(idx.size(0), device=device)[:limit]
                ]
            if idx.size(0) > 0:
                i_buf.append(idx[:, 0])
                j_buf.append(idx[:, 1])
                bid_buf.append(
                    torch.full((idx.size(0),), b, device=device, dtype=torch.long)
                )

    # --- Gather positive embeddings ---
    if pos_i_parts:
        pos_batch = torch.cat(pos_batch_parts, dim=0)
        pos_i = torch.cat(pos_i_parts, dim=0)
        pos_j = torch.cat(pos_j_parts, dim=0)
        pos_h1 = h1[pos_batch, pos_i]
        pos_h2 = h2[pos_batch, pos_j]
        pos_batch_ids = pos_batch
    else:
        pos_h1 = h1.new_zeros(0, h1.size(-1))
        pos_h2 = h1.new_zeros(0, h1.size(-1))
        pos_batch_ids = h1.new_zeros(0, dtype=torch.long)
        pos_i = h1.new_zeros(0, dtype=torch.long)
        pos_j = h1.new_zeros(0, dtype=torch.long)
    N_pos = pos_h1.size(0)

    # --- Gather negative embeddings ---
    if neg_i_parts:
        neg_batch = torch.cat(neg_batch_parts, dim=0)
        neg_i = torch.cat(neg_i_parts, dim=0)
        neg_j = torch.cat(neg_j_parts, dim=0)
        neg_h1 = h1[neg_batch, neg_i]
        neg_h2 = h2[neg_batch, neg_j]
    else:
        neg_h1 = h1.new_zeros(0, h1.size(-1))
        neg_h2 = h1.new_zeros(0, h1.size(-1))
    return pos_h1, pos_h2, neg_h1, neg_h2, N_pos, pos_batch_ids, pos_i, pos_j


# ===========================================================================
# Residue-level CLIP / InfoNCE loss  (v1-style)
# ===========================================================================

def _residue_clip_loss(
    logits: torch.Tensor,
    num_pos: int,
) -> torch.Tensor:
    """Bidirectional InfoNCE / CLIP loss on a residue similarity matrix.

    Assumes the first num_pos rows AND cols correspond to positive pairs
    arranged on the diagonal.  Logits are expected to already include
    temperature scaling.

    Loss = 0.5 * (CE(rows[:num_pos]) + CE(cols[:num_pos]))

    This mirrors v1's clip_loss but operates on a single pre-scaled matrix.

    Parameters
    ----------
    logits : (N_total, N_total)
        Cross-chain residue similarity matrix, scaled by inv_temperature.
    num_pos : int
        Number of positive pairs (diagonal block size).

    Returns
    -------
    Scalar loss tensor.
    """
    labels = torch.arange(num_pos, device=logits.device)

    # Row direction: each positive receptor residue → correct ligand residue
    loss_r = F.cross_entropy(logits[:num_pos], labels)

    # Column direction: each positive ligand residue → correct receptor residue
    loss_l = F.cross_entropy(logits.T[:num_pos], labels)

    return 0.5 * (loss_r + loss_l)


# ===========================================================================
# Per-batch loss computation
# ===========================================================================


def compute_batch_losses(
    model: torch.nn.Module,
    batch: dict,
    *,
    pos_per_sample: int = 5,
    neg_per_sample: int = 5,
    device: torch.device,
    bf16: bool = False,
) -> dict[str, torch.Tensor]:
    """Forward pass plus the sole training objective: residue-contact InfoNCE.

    Supports two batch formats:
      - Pre-extracted: batch has "repr1", "repr2"
      - LoRA / tokenized: batch has "input_ids1", "input_ids2"
    """
    use_lora = "input_ids1" in batch

    if use_lora:
        input_ids1 = batch["input_ids1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
    else:
        repr1 = batch["repr1"].to(device)
        repr2 = batch["repr2"].to(device)

    mask1 = batch["mask1"].to(device)
    mask2 = batch["mask2"].to(device)
    if "contact_matrix" not in batch or "contact_mask" not in batch:
        raise ValueError(
            "Contact-only training requires contact_matrix and contact_mask in every batch"
        )
    contact_matrix = batch["contact_matrix"].to(device)
    contact_mask = batch["contact_mask"].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        if use_lora:
            h1, h2, inv_temp = model(
                input_ids1=input_ids1, input_ids2=input_ids2,
                mask1=mask1, mask2=mask2,
            )
        else:
            h1, h2, inv_temp = model(
                repr1=repr1, repr2=repr2,
                mask1=mask1, mask2=mask2,
            )
        contact_loss = h1.new_tensor(0.0)
        pos_h1, pos_h2, neg_h1, neg_h2, num_pos, _, _, _ = _sample_contact_pairs(
                h1, h2, contact_matrix, contact_mask,
                pos_per_sample=pos_per_sample,
                neg_per_sample=neg_per_sample,
                device=device,
            )
        if num_pos > 0:
            all_h1 = torch.cat([pos_h1, neg_h1], dim=0)
            all_h2 = torch.cat([pos_h2, neg_h2], dim=0)
            logits = torch.matmul(all_h1, all_h2.T)
            logits = torch.clamp(logits, min=-1e2, max=1e2)
            logits = logits * inv_temp.to(dtype=logits.dtype)
            contact_loss = _residue_clip_loss(logits, num_pos)

    return {
        "loss": contact_loss,
        "contact_loss": contact_loss.detach(),
    }


# ===========================================================================
# Training epoch
# ===========================================================================

def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    *,
    pos_per_sample: int = 5,
    neg_per_sample: int = 5,
    device: torch.device,
    bf16: bool = False,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    epoch: int = 0,
    show_progress: bool = True,
) -> dict[str, float | None]:
    """Run one training epoch.

    Returns dict of average losses over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_contact_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False,
                disable=not show_progress, dynamic_ncols=True, ascii=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    for batch_idx, batch in enumerate(pbar):
        losses = compute_batch_losses(
            model, batch,
            pos_per_sample=pos_per_sample,
            neg_per_sample=neg_per_sample,
            device=device, bf16=bf16,
        )
        loss = losses["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_contact_loss += losses["contact_loss"].item()
        n_batches += 1

        if batch_idx % log_interval == 0:
            inv_temp_val = _unwrap_model(model).get_inv_temperature().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "contact": f"{losses['contact_loss'].item():.4f}",
                "1/tau": f"{inv_temp_val:.2f}",
            })

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_contact_loss": total_contact_loss / max(n_batches, 1),
    }


# ===========================================================================
# Evaluation epoch
# ===========================================================================

@torch.no_grad()
def run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    pos_per_sample: int = 5,
    neg_per_sample: int = 5,
    device: torch.device,
    bf16: bool = False,
    epoch: int = 0,
    stage: str = "val",
    show_progress: bool = True,
    positive_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    observed_label_mask: torch.Tensor | None = None,
    verified_negative_mask: torch.Tensor | None = None,
    score_mask1_by_label: dict[str, torch.Tensor] | None = None,
    score_output: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Run one evaluation epoch.

    Computes:
      - Contact contrastive loss (as validation loss)
      - Evidence-aware PPI retrieval metrics (MRR and Hit@K as primary;
        positive-vs-unlabelled AUPRC as secondary) on validation pairs.

    Returns dict of all metrics.
    """
    model.eval()
    total_loss = 0.0
    total_contact_loss = 0.0
    n_batches = 0
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1

    # Collect all encoded proteins for retrieval evaluation (single-GPU only)
    # In DDP mode, skip collection to avoid wasting memory.
    all_h1 = [] if not use_ddp else None
    all_h2 = [] if not use_ddp else None
    all_mask1 = [] if not use_ddp else None
    all_mask2 = [] if not use_ddp else None
    all_score_mask1 = [] if not use_ddp else None
    contact_pair_labels: list[np.ndarray] = []
    contact_pair_scores: list[np.ndarray] = []
    contact_res_labels: list[np.ndarray] = []
    contact_res_scores: list[np.ndarray] = []
    sample_offset = 0
    positive_sample_mask = None
    if positive_mask is not None and not use_ddp:
        if positive_mask.dim() == 2:
            positive_sample_mask = positive_mask.diag().bool()
        elif positive_mask.dim() == 1:
            positive_sample_mask = positive_mask.bool()

    pbar = tqdm(
        loader,
        desc=f"Eval {stage.upper()} E{epoch:03d}",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True, ascii=False,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )
    for batch_idx, batch in enumerate(pbar):
        losses = compute_batch_losses(
            model, batch,
            pos_per_sample=pos_per_sample,
            neg_per_sample=neg_per_sample,
            device=device, bf16=bf16,
        )

        total_loss += losses["loss"].item()
        total_contact_loss += losses["contact_loss"].item()
        n_batches += 1

        # Collect all proteins for retrieval evaluation (single-GPU only)
        if not use_ddp:
            use_lora = "input_ids1" in batch
            mask1_raw = batch["mask1"].to(device)
            mask2_raw = batch["mask2"].to(device)
            batch_size = int(mask1_raw.size(0))
            if positive_sample_mask is not None:
                batch_positive = positive_sample_mask[
                    sample_offset:sample_offset + batch_size
                ].to(device=device)
                if batch_positive.numel() != batch_size:
                    batch_positive = torch.ones(
                        batch_size, dtype=torch.bool, device=device
                    )
            else:
                batch_positive = torch.ones(
                    batch_size, dtype=torch.bool, device=device
                )
            sample_offset += batch_size

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                if use_lora:
                    input_ids1 = batch["input_ids1"].to(device)
                    input_ids2 = batch["input_ids2"].to(device)
                    h1_eval, h2_eval, _ = model(
                        input_ids1=input_ids1, input_ids2=input_ids2,
                        mask1=mask1_raw, mask2=mask2_raw,
                    )
                    # Strip <cls>/<eos> from masks so they align with residue embeddings
                    mask1_eval = mask1_raw[:, 1:-1]
                    mask2_eval = mask2_raw[:, 1:-1]
                else:
                    repr1 = batch["repr1"].to(device)
                    repr2 = batch["repr2"].to(device)
                    h1_eval, h2_eval, _ = model(
                        repr1=repr1, repr2=repr2,
                        mask1=mask1_raw, mask2=mask2_raw,
                    )
                    mask1_eval = mask1_raw
                    mask2_eval = mask2_raw
            all_h1.append(h1_eval)
            all_h2.append(h2_eval)
            all_mask1.append(mask1_eval)
            all_mask2.append(mask2_eval)
            if score_mask1_by_label is not None:
                score_masks = []
                for label, mask in zip(batch["label_a"], mask1_eval):
                    score_mask = score_mask1_by_label.get(label)
                    if score_mask is None:
                        score_mask = torch.ones(int(mask.sum().item()), dtype=torch.bool)
                    score_mask = score_mask.to(device=device, dtype=torch.bool)
                    if score_mask.numel() < mask.numel():
                        pad = mask.numel() - score_mask.numel()
                        score_mask = torch.cat([score_mask, score_mask.new_zeros(pad)], dim=0)
                    elif score_mask.numel() > mask.numel():
                        score_mask = score_mask[: mask.numel()]
                    score_masks.append(score_mask & mask)
                all_score_mask1.append(torch.stack(score_masks, dim=0))
            contact_matrix = batch.get("contact_matrix")
            contact_mask = batch.get("contact_mask")
            if (
                contact_matrix is not None
                and contact_mask is not None
                and batch_positive.any()
            ):
                h1_contact = h1_eval[batch_positive]
                h2_contact = h2_eval[batch_positive]
                mask1_contact = mask1_eval[batch_positive]
                mask2_contact = mask2_eval[batch_positive]
                contact_matrix = contact_matrix.to(device)
                contact_mask = contact_mask.to(device)
                contact_matrix = contact_matrix[batch_positive]
                contact_mask = contact_mask[batch_positive]
                inv_temp_eval = _unwrap_model(model).get_inv_temperature().to(
                    device=device, dtype=torch.float32
                )
                pair_scores = torch.einsum(
                    "bld,bmd->blm", h1_contact.float(), h2_contact.float()
                ) * inv_temp_eval
                pair_scores = torch.clamp(pair_scores, min=-1e2, max=1e2)
                valid_pairs = contact_mask & (contact_matrix != 0)
                if valid_pairs.any():
                    contact_pair_labels.append(
                        (contact_matrix[valid_pairs] == 1).detach().cpu().numpy()
                    )
                    contact_pair_scores.append(
                        pair_scores[valid_pairs].detach().cpu().numpy()
                    )

                neg_inf = torch.finfo(pair_scores.dtype).min
                masked_scores = pair_scores.masked_fill(~contact_mask, neg_inf)
                res1_valid = mask1_contact & contact_mask.any(dim=2)
                res2_valid = mask2_contact & contact_mask.any(dim=1)
                res1_true = ((contact_matrix == 1) & contact_mask).any(dim=2)
                res2_true = ((contact_matrix == 1) & contact_mask).any(dim=1)
                res1_score = masked_scores.max(dim=2).values
                res2_score = masked_scores.max(dim=1).values

                if res1_valid.any():
                    contact_res_labels.append(res1_true[res1_valid].detach().cpu().numpy())
                    contact_res_scores.append(res1_score[res1_valid].detach().cpu().numpy())
                if res2_valid.any():
                    contact_res_labels.append(res2_true[res2_valid].detach().cpu().numpy())
                    contact_res_scores.append(res2_score[res2_valid].detach().cpu().numpy())

    if use_ddp:
        agg = torch.tensor(
            [total_loss, total_contact_loss, float(n_batches)],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, total_contact_loss, n_batches = agg.tolist()

    metrics = {
        f"{stage}_loss": total_loss / max(n_batches, 1.0),
        f"{stage}_contact_loss": total_contact_loss / max(n_batches, 1.0),
    }
    pair_auprc, pair_auroc = _safe_binary_metrics(contact_pair_labels, contact_pair_scores)
    res_auprc, res_auroc = _safe_binary_metrics(contact_res_labels, contact_res_scores)
    metrics[f"{stage}_contact_res_pair_auprc"] = pair_auprc
    metrics[f"{stage}_contact_res_pair_auroc"] = pair_auroc
    metrics[f"{stage}_contact_res_auprc"] = res_auprc
    metrics[f"{stage}_contact_res_auroc"] = res_auroc

    # --- ColBERT PPI retrieval evaluation ---
    # NOTE: In DDP mode we skip the expensive all_gather + cross-GPU retrieval
    # to avoid NCCL timeouts (all_gather_object is pickle-based and the
    # mutual_topk_ppi_scores triple loop is too slow for live validation).
    # Full PPI retrieval metrics are computed at test-time after DDP teardown.
    if use_ddp:
        return metrics

    # --- Single-process retrieval eval ---
    if all_h1:
        # Pad all collected tensors to consistent seq_len before cat.
        max_l1_local = max(h.size(1) for h in all_h1)
        max_l2_local = max(h.size(1) for h in all_h2)

        def _pad_local_3d(items, max_len):
            out = []
            for t in items:
                pad = max_len - t.size(1)
                if pad > 0:
                    t = torch.cat([t, t.new_zeros(t.size(0), pad, t.size(2))], dim=1)
                out.append(t)
            return out

        def _pad_local_2d(items, max_len):
            out = []
            for t in items:
                pad = max_len - t.size(1)
                if pad > 0:
                    t = torch.cat([t, t.new_zeros(t.size(0), pad)], dim=1)
                out.append(t)
            return out

        h1_cat = torch.cat(_pad_local_3d(all_h1, max_l1_local), dim=0)
        h2_cat = torch.cat(_pad_local_3d(all_h2, max_l2_local), dim=0)
        m1_cat = torch.cat(_pad_local_2d(all_mask1, max_l1_local), dim=0)
        m2_cat = torch.cat(_pad_local_2d(all_mask2, max_l2_local), dim=0)
        if all_score_mask1:
            score_m1_cat = torch.cat(_pad_local_2d(all_score_mask1, max_l1_local), dim=0)
            score_m1_cat = score_m1_cat & m1_cat
        else:
            score_m1_cat = m1_cat
        if h1_cat.size(0) >= 2:
            base_model = _unwrap_model(model)
            inv_temp = base_model.get_inv_temperature()

            ppi_scores = mutual_topk_ppi_scores(
                h1_cat, h2_cat, score_m1_cat, m2_cat, inv_temp,
            )
            if score_output is not None:
                score_output["scores"] = ppi_scores.detach().cpu().numpy()
            label_mask = (
                positive_mask.to(device=ppi_scores.device, dtype=torch.bool)
                if positive_mask is not None else None
            )
            ret_metrics = compute_retrieval_metrics(
                ppi_scores,
                positive_mask=label_mask,
                candidate_mask=(
                    candidate_mask.to(device=ppi_scores.device, dtype=torch.bool)
                    if candidate_mask is not None else None
                ),
                observed_label_mask=(
                    observed_label_mask.to(device=ppi_scores.device, dtype=torch.bool)
                    if observed_label_mask is not None else None
                ),
                verified_negative_mask=(
                    verified_negative_mask.to(device=ppi_scores.device, dtype=torch.bool)
                    if verified_negative_mask is not None else None
                ),
            )
            for k, v in ret_metrics.items():
                metrics[f"{stage}_{k}"] = v

            if label_mask is not None:
                metrics[f"{stage}_positive_pairs"] = float(label_mask.sum().item())
            if all_score_mask1:
                metrics[f"{stage}_score_mask1_mean_residues"] = float(
                    score_m1_cat.sum(dim=1).float().mean().item()
                )

    return metrics
