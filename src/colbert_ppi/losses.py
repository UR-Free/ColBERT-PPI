"""Symmetric sampled residue-contact InfoNCE."""

import torch
from torch.nn import functional as F


def sampled_contact_losses(
    p: torch.Tensor,
    n: torch.Tensor,
    contact: torch.Tensor,
    contact_mask: torch.Tensor,
    p_attention: torch.Tensor | None = None,
    n_attention: torch.Tensor | None = None,
    inv_temperature: torch.Tensor | None = None,
    max_positive: int = 5,
    max_negative: int = 5,
) -> dict[str, torch.Tensor]:
    """Contact-only InfoNCE objective adapted to protein–nucleic pairs."""
    pos_left, pos_right, neg_left, neg_right = [], [], [], []
    pos_pa, pos_na, neg_pa, neg_na = [], [], [], []
    for b in range(p.size(0)):
        for sign, limit, left, right, pa_parts, na_parts in (
            (1, max_positive, pos_left, pos_right, pos_pa, pos_na),
            (-1, max_negative, neg_left, neg_right, neg_pa, neg_na),
        ):
            pairs = torch.nonzero(contact_mask[b] & contact[b].eq(sign), as_tuple=False)
            if pairs.size(0) > limit:
                pairs = pairs[torch.randperm(pairs.size(0), device=p.device)[:limit]]
            if pairs.numel():
                left.append(p[b, pairs[:, 0]])
                right.append(n[b, pairs[:, 1]])
                if p_attention is not None and n_attention is not None:
                    pa_parts.append(p_attention[b, pairs[:, 0]])
                    na_parts.append(n_attention[b, pairs[:, 1]])
    if not pos_left:
        zero = p.new_tensor(0.0)
        return {"contact_loss": zero}
    positive_left, positive_right = torch.cat(pos_left), torch.cat(pos_right)
    all_left = (
        torch.cat((positive_left, torch.cat(neg_left)), dim=0)
        if neg_left
        else positive_left
    )
    all_right = (
        torch.cat((positive_right, torch.cat(neg_right)), dim=0)
        if neg_right
        else positive_right
    )
    logits = (all_left @ all_right.T).clamp(min=-1e2, max=1e2)
    if pos_pa:
        positive_pa, positive_na = torch.cat(pos_pa), torch.cat(pos_na)
        negative_pa = torch.cat(neg_pa) if neg_pa else p.new_zeros(0)
        negative_na = torch.cat(neg_na) if neg_na else n.new_zeros(0)
        all_pa = torch.cat((positive_pa, negative_pa), dim=0) if neg_pa else positive_pa
        all_na = torch.cat((positive_na, negative_na), dim=0) if neg_na else positive_na
        logits = logits * all_pa.unsqueeze(1) * all_na.unsqueeze(0)
    else:
        positive_pa = positive_na = negative_pa = negative_na = None
    if inv_temperature is not None:
        logits = logits * inv_temperature.to(dtype=logits.dtype)
    n_positive = positive_left.size(0)
    labels = torch.arange(n_positive, device=p.device)
    contact_loss = 0.5 * (
        F.cross_entropy(logits[:n_positive], labels)
        + F.cross_entropy(logits.T[:n_positive], labels)
    )

    return {"contact_loss": contact_loss}
