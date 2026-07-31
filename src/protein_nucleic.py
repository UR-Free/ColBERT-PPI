"""Nucleic-acid / SaProt protein late-interaction components.

This module deliberately keeps the two molecular modalities separate until the
late-interaction stage.  Protein inputs are SaProt AA+3Di tokens; nucleic-acid
inputs are base-level RiNALMo, OmniBioTE, or structure-aware ERNIE-RNA tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.model import FlashAttentionEncoder


@dataclass
class PNABatch:
    protein_ids: torch.Tensor
    protein_mask: torch.Tensor
    nucleic_ids: torch.Tensor
    nucleic_mask: torch.Tensor
    contact: torch.Tensor
    contact_mask: torch.Tensor
    pair_ids: list[str]
    nucleic_structure: torch.Tensor
    nucleic_relations: torch.Tensor


class ProteinNucleicDataset(torch.utils.data.Dataset):
    """Dataset backed by a torch-saved list of prepared PNA pair records.

    Each record stores token tensors with special tokens included and sparse
    ``positive``/``negative`` base-residue index pairs in residue coordinates.
    Keeping labels sparse avoids serialising full L_protein x L_nucleic maps.
    """

    def __init__(self, path: str, max_protein_len: int | None = 256,
                 max_nucleic_len: int | None = 512,
                 random_interface_nucleic_crop: bool = False):
        self.records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(self.records, list) or not self.records:
            raise ValueError(f"Prepared PNA data is empty or invalid: {path}")
        self.max_protein_len = max_protein_len
        self.max_nucleic_len = max_nucleic_len
        self.random_interface_nucleic_crop = bool(random_interface_nucleic_crop)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _as_long(value) -> torch.Tensor:
        return value.long() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        p = self._as_long(row["protein_input_ids"]).flatten()
        n = self._as_long(row["nucleic_input_ids"]).flatten()
        # Special tokens are present at both termini.  Crop in residue/base space
        # while retaining EOS, so frozen backbones see their expected layout.
        if self.max_protein_len is not None and p.numel() > self.max_protein_len + 2:
            p = torch.cat((p[: self.max_protein_len + 1], p[-1:]))
        nucleic_start = 0
        if self.max_nucleic_len is not None and n.numel() > self.max_nucleic_len + 2:
            full_length = n.numel() - 2
            max_start = full_length - self.max_nucleic_len
            if self.random_interface_nucleic_crop:
                positives = self._as_long(row["positive"]).reshape(-1, 2)
                protein_limit = max(p.numel() - 2, 0)
                anchors = positives[positives[:, 0] < protein_limit, 1]
                if anchors.numel():
                    anchor = int(anchors[torch.randint(anchors.numel(), (1,))].item())
                    lower = max(0, anchor - self.max_nucleic_len + 1)
                    upper = min(anchor, max_start)
                    nucleic_start = int(torch.randint(lower, upper + 1, (1,)).item())
                else:
                    nucleic_start = int(torch.randint(max_start + 1, (1,)).item())
            n = torch.cat((
                n[:1],
                n[1 + nucleic_start: 1 + nucleic_start + self.max_nucleic_len],
                n[-1:],
            ))
        lp, ln = max(p.numel() - 2, 0), max(n.numel() - 2, 0)

        def filter_pairs(value):
            pairs = self._as_long(value).reshape(-1, 2)
            pairs = pairs.clone()
            pairs[:, 1] -= nucleic_start
            keep = (
                (pairs[:, 0] >= 0) & (pairs[:, 0] < lp)
                & (pairs[:, 1] >= 0) & (pairs[:, 1] < ln)
            )
            return pairs[keep]

        raw_edges = row.get("rna_structure_edges", [])
        edges = torch.as_tensor(raw_edges, dtype=torch.float32)
        if edges.numel():
            if edges.ndim != 2 or edges.size(1) < 3:
                raise ValueError(
                    f"rna_structure_edges must be [N,>=3], got {tuple(edges.shape)} "
                    f"for {row.get('pair_id', index)}"
                )
        else:
            edges = edges.reshape(0, 3)
        if edges.numel():
            edges = edges.clone()
            edges[:, 0] -= nucleic_start
            edges[:, 1] -= nucleic_start
        return {
            "protein_input_ids": p,
            "nucleic_input_ids": n,
            "positive": filter_pairs(row["positive"]),
            "negative": filter_pairs(row["negative"]),
            "pair_id": str(row.get("pair_id", index)),
            "nucleic_pad_id": int(row.get("nucleic_pad_id", 0)),
            "rna_structure_edges": edges,
            "rna_structure_directed": bool(row.get("rna_structure_directed", False)),
        }

    @staticmethod
    def collate_fn(rows: Sequence[dict]) -> PNABatch:
        protein_ids = pad_sequence([x["protein_input_ids"] for x in rows], batch_first=True, padding_value=1)
        pad_ids = {int(x["nucleic_pad_id"]) for x in rows}
        if len(pad_ids) != 1:
            raise ValueError(f"Mixed nucleic padding IDs in one batch: {sorted(pad_ids)}")
        nucleic_pad_id = pad_ids.pop()
        nucleic_ids = pad_sequence(
            [x["nucleic_input_ids"] for x in rows],
            batch_first=True,
            padding_value=nucleic_pad_id,
        )
        protein_mask = protein_ids.ne(1)
        nucleic_mask = nucleic_ids.ne(nucleic_pad_id)
        lp, ln = protein_ids.size(1) - 2, nucleic_ids.size(1) - 2
        contact = torch.zeros((len(rows), lp, ln), dtype=torch.int8)
        contact_mask = torch.zeros((len(rows), lp, ln), dtype=torch.bool)
        # ERNIE-RNA includes CLS/EOS in its 2D bias, so residue-space edge
        # indices are shifted by one. Other nucleic backbones receive zeros.
        nucleic_structure = torch.zeros(
            (len(rows), nucleic_ids.size(1), nucleic_ids.size(1)),
            dtype=torch.float32,
        )
        # Channels: relation type, orientation, direction, confidence. Codes
        # stay categorical until the trainable RNA relation adapter.
        nucleic_relations = torch.zeros(
            (len(rows), nucleic_ids.size(1), nucleic_ids.size(1), 4),
            dtype=torch.float32,
        )
        for b, row in enumerate(rows):
            for key, sign in (("positive", 1), ("negative", -1)):
                pairs = row[key]
                if pairs.numel():
                    contact[b, pairs[:, 0], pairs[:, 1]] = sign
                    contact_mask[b, pairs[:, 0], pairs[:, 1]] = True
            edges = row["rna_structure_edges"]
            if edges.numel():
                keep = (
                    (edges[:, 0] >= 0) & (edges[:, 0] < ln)
                    & (edges[:, 1] >= 0) & (edges[:, 1] < ln)
                )
                edges = edges[keep]
                if edges.numel():
                    left = edges[:, 0].long() + 1
                    right = edges[:, 1].long() + 1
                    value = edges[:, 2].float()
                    nucleic_structure[b, left, right] = value
                    if edges.size(1) >= 7:
                        nucleic_relations[b, left, right, 0] = edges[:, 3]
                        nucleic_relations[b, left, right, 1] = edges[:, 4]
                        nucleic_relations[b, left, right, 2] = edges[:, 5]
                        nucleic_relations[b, left, right, 3] = edges[:, 6]
                    if not row["rna_structure_directed"]:
                        nucleic_structure[b, right, left] = value
                        if edges.size(1) >= 7:
                            nucleic_relations[b, right, left] = nucleic_relations[b, left, right]
        return PNABatch(protein_ids, protein_mask, nucleic_ids, nucleic_mask,
                        contact, contact_mask, [x["pair_id"] for x in rows],
                        nucleic_structure, nucleic_relations)


class RNARelationAdapter(nn.Module):
    """Gated categorical message passing over RNA base-pair/stacking edges."""

    def __init__(self, hidden_size: int, rank: int = 32, dropout: float = 0.0,
                 gate_init: float = -6.9):
        super().__init__()
        self.rank = int(rank)
        self.dropout = float(dropout)
        # Codes 1..6 correspond to the explicit relation schema. Code 0 is no edge.
        self.relation_down = nn.ModuleList([
            nn.Linear(hidden_size, rank, bias=False) for _ in range(6)
        ])
        self.orientation_scale = nn.Embedding(5, 1, padding_idx=0)
        self.direction_scale = nn.Embedding(3, 1, padding_idx=0)
        nn.init.ones_(self.orientation_scale.weight)
        nn.init.ones_(self.direction_scale.weight)
        with torch.no_grad():
            self.orientation_scale.weight[0].zero_()
            self.direction_scale.weight[0].zero_()
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.up.weight)
        self.message_norm = nn.LayerNorm(rank)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden: torch.Tensor, relations: torch.Tensor | None) -> torch.Tensor:
        if relations is None or relations.numel() == 0:
            return hidden
        relation_type = relations[..., 0].long().clamp(0, 6)
        orientation = relations[..., 1].long().clamp(0, 4)
        direction = relations[..., 2].long().clamp(0, 2)
        confidence = relations[..., 3].to(hidden)
        confidence = confidence * self.orientation_scale(orientation).squeeze(-1).to(hidden)
        confidence = confidence * self.direction_scale(direction).squeeze(-1).to(hidden)
        if self.training and self.dropout > 0:
            confidence = F.dropout(confidence, p=self.dropout, training=True)
        message = hidden.new_zeros((*hidden.shape[:2], self.rank))
        degree = hidden.new_zeros(hidden.shape[:2])
        for code, projection in enumerate(self.relation_down, start=1):
            adjacency = confidence * relation_type.eq(code).to(confidence)
            message = message + torch.bmm(adjacency, projection(hidden))
            degree = degree + adjacency.abs().sum(dim=-1)
        message = message / degree.clamp_min(1.0).unsqueeze(-1)
        update = self.up(self.message_norm(message))
        return hidden + torch.sigmoid(self.gate_logit) * update


class ProteinNucleicColBERT(nn.Module):
    """SaProt–nucleic-language-model asymmetric ColBERT model.

    The language-model backbones can be either frozen or LoRA-adapted, followed
    by modality-specific contextual encoders.  The resulting residue/base vectors share one metric space; their
    dot-product matrix is used for both contact supervision and retrieval.
    """

    def __init__(self, saprot_dir: str, rinalmo_name: str,
                 hidden_dim: int = 512, num_heads: int = 16,
                 num_layers: int = 3, dropout: float = 0.1,
                 temperature: float = 0.07, tune_saprot: bool = False,
                 nucleic_model_type: str = "rinalmo",
                 omnibiote_checkpoint: str | None = None,
                 omnibiote_code_dir: str = "third_party/OmniBioTE/src",
                 ernie_rna_checkpoint: str | None = None,
                 ernie_rna_code_dir: str = "third_party/ERNIE-RNA",
                 rna_structure_weight: float = 2.0,
                 rna_relation_adapter_rank: int = 0,
                 rna_relation_dropout: float = 0.0,
                 rna_relation_gate_init: float = -6.9,
                 saprot_lora_rank: int = 0,
                 saprot_lora_alpha: int = 8,
                 saprot_lora_dropout: float = 0.1,
                 nucleic_lora_rank: int = 0,
                 nucleic_lora_alpha: int = 8,
                 nucleic_lora_dropout: float = 0.1):
        super().__init__()
        from transformers import EsmConfig, EsmForMaskedLM

        cfg = EsmConfig.from_pretrained(saprot_dir)
        self.saprot = EsmForMaskedLM(cfg)
        state = torch.load(f"{saprot_dir}/pytorch_model.bin", map_location="cpu", weights_only=False)
        self.saprot.load_state_dict({k: v for k, v in state.items() if not k.startswith("lm_head.")}, strict=False)
        self.saprot.lm_head = None
        self.nucleic_model_type = nucleic_model_type.lower()
        if self.nucleic_model_type == "rinalmo":
            from multimolecule import RiNALMoModel
            self.rinalmo = RiNALMoModel.from_pretrained(rinalmo_name)
            nucleic_hidden_size = int(self.rinalmo.config.hidden_size)
        elif self.nucleic_model_type == "omnibiote":
            if not omnibiote_checkpoint:
                raise ValueError("omnibiote_checkpoint is required for nucleic_model_type='omnibiote'")
            from src.omnibiote_adapter import OmniBioTEBackbone
            self.rinalmo = OmniBioTEBackbone(omnibiote_checkpoint, omnibiote_code_dir)
            nucleic_hidden_size = self.rinalmo.hidden_size
        elif self.nucleic_model_type == "ernie_rna":
            if not ernie_rna_checkpoint:
                raise ValueError("ernie_rna_checkpoint is required for nucleic_model_type='ernie_rna'")
            from src.ernie_rna_adapter import ERNIERNABackbone
            self.rinalmo = ERNIERNABackbone(
                ernie_rna_checkpoint,
                ernie_rna_code_dir,
                structure_weight=rna_structure_weight,
            )
            nucleic_hidden_size = self.rinalmo.hidden_size
        else:
            raise ValueError(f"Unknown nucleic_model_type: {nucleic_model_type!r}")
        self.tune_saprot = tune_saprot
        self.saprot_lora_enabled = saprot_lora_rank > 0
        self.nucleic_lora_enabled = nucleic_lora_rank > 0
        if tune_saprot and self.saprot_lora_enabled:
            raise ValueError("Use either full SaProt tuning or SaProt LoRA, not both.")
        if tune_saprot:
            for p in self.saprot.parameters():
                p.requires_grad = True
        else:
            for p in self.saprot.parameters():
                p.requires_grad = False
            if self.saprot_lora_enabled:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    r=saprot_lora_rank,
                    lora_alpha=saprot_lora_alpha,
                    lora_dropout=saprot_lora_dropout,
                    target_modules=["query", "key", "value", "dense"],
                    inference_mode=False,
                )
                self.saprot.esm = get_peft_model(self.saprot.esm, peft_config)
            else:
                self.saprot.eval()  # freeze dropout/batchnorm in backbone
        for p in self.rinalmo.parameters():
            p.requires_grad = False
        if self.nucleic_lora_enabled:
            from src.lora_adapters import inject_lora_linear_layers
            if self.nucleic_model_type == "omnibiote":
                target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.nucleic_model_type == "ernie_rna":
                # ERNIE-RNA's custom attention fast path accesses projection
                # weights directly, so only FFN LoRA is guaranteed active.
                target_modules = ["fc1", "fc2"]
            else:
                target_modules = None
            target = self.rinalmo.model if self.nucleic_model_type in {"omnibiote", "ernie_rna"} else self.rinalmo
            self.nucleic_lora_summary = inject_lora_linear_layers(
                target,
                rank=nucleic_lora_rank,
                alpha=nucleic_lora_alpha,
                dropout=nucleic_lora_dropout,
                target_modules=target_modules,
            )
            if hasattr(self.rinalmo, "set_force_eval"):
                self.rinalmo.set_force_eval(False)
        else:
            self.rinalmo.eval()  # freeze dropout/batchnorm in backbone

        self.rna_relation_adapter = (
            RNARelationAdapter(
                nucleic_hidden_size,
                rank=rna_relation_adapter_rank,
                dropout=rna_relation_dropout,
                gate_init=rna_relation_gate_init,
            )
            if rna_relation_adapter_rank > 0 else None
        )

        self.protein_encoder = FlashAttentionEncoder(cfg.hidden_size + 1, hidden_dim, num_heads, num_layers, dropout)
        self.nucleic_encoder = FlashAttentionEncoder(nucleic_hidden_size + 1, hidden_dim, num_heads, num_layers, dropout)
        self.protein_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))
        self.nucleic_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))
        self.log_inv_temperature = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def train(self, mode: bool = True):
        super().train(mode)
        if not (self.tune_saprot or self.saprot_lora_enabled):
            self.saprot.eval()
        if not self.nucleic_lora_enabled:
            self.rinalmo.eval()
        return self

    @property
    def inv_temperature(self) -> torch.Tensor:
        return self.log_inv_temperature.exp().clamp(0.05, 20.0)

    @staticmethod
    def _strip(output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output[:, 1:-1], mask[:, 1:-1]

    def forward(self, protein_ids: torch.Tensor, protein_mask: torch.Tensor,
                nucleic_ids: torch.Tensor, nucleic_mask: torch.Tensor,
                nucleic_structure: torch.Tensor | None = None,
                nucleic_relations: torch.Tensor | None = None):
        # Frozen backbones run under no_grad; LoRA/full-tuned backbones must keep
        # gradients through the frozen base path into trainable adapters.
        p_kwargs = {"input_ids": protein_ids, "attention_mask": protein_mask.float(), "return_dict": True}
        if self.tune_saprot or self.saprot_lora_enabled:
            p_out = self.saprot.esm(**p_kwargs).last_hidden_state
        else:
            with torch.no_grad():
                p_out = self.saprot.esm(**p_kwargs).last_hidden_state
        if self.nucleic_lora_enabled:
            if self.nucleic_model_type == "rinalmo":
                n_out = self.rinalmo(input_ids=nucleic_ids, attention_mask=nucleic_mask, return_dict=True).last_hidden_state
            elif self.nucleic_model_type == "omnibiote":
                n_out = self.rinalmo(nucleic_ids, nucleic_mask)
            else:
                n_out = self.rinalmo(nucleic_ids, nucleic_mask, nucleic_structure)
        else:
            with torch.no_grad():
                if self.nucleic_model_type == "rinalmo":
                    n_out = self.rinalmo(input_ids=nucleic_ids, attention_mask=nucleic_mask, return_dict=True).last_hidden_state
                elif self.nucleic_model_type == "omnibiote":
                    n_out = self.rinalmo(nucleic_ids, nucleic_mask)
                else:
                    n_out = self.rinalmo(nucleic_ids, nucleic_mask, nucleic_structure)
        if self.rna_relation_adapter is not None:
            n_out = self.rna_relation_adapter(n_out, nucleic_relations)
        p_out, p_mask = self._strip(p_out, protein_mask)
        n_out, n_mask = self._strip(n_out, nucleic_mask)
        p_role = torch.ones((*p_out.shape[:2], 1), device=p_out.device, dtype=p_out.dtype)
        n_role = torch.zeros((*n_out.shape[:2], 1), device=n_out.device, dtype=n_out.dtype)
        p = F.normalize(self.protein_encoder(torch.cat((p_out, p_role), -1), p_mask), p=2, dim=-1)
        n = F.normalize(self.nucleic_encoder(torch.cat((n_out, n_role), -1), n_mask), p=2, dim=-1)
        pa = torch.sigmoid(self.protein_attention(p).squeeze(-1)).masked_fill(~p_mask, 0)
        na = torch.sigmoid(self.nucleic_attention(n).squeeze(-1)).masked_fill(~n_mask, 0)
        return p, n, p_mask, n_mask, pa, na, self.inv_temperature


def sampled_contact_infonce(
    p: torch.Tensor,
    n: torch.Tensor,
    contact: torch.Tensor,
    contact_mask: torch.Tensor,
    p_attention: torch.Tensor | None = None,
    n_attention: torch.Tensor | None = None,
    inv_temperature: torch.Tensor | None = None,
    max_positive: int = 5,
    max_negative: int = 5,
) -> torch.Tensor:
    """Symmetric contact InfoNCE with explicit structural negatives."""
    return sampled_contact_losses(
        p, n, contact, contact_mask,
        p_attention=p_attention,
        n_attention=n_attention,
        inv_temperature=inv_temperature,
        max_positive=max_positive,
        max_negative=max_negative,
    )["contact_loss"]


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
    all_left = torch.cat((positive_left, torch.cat(neg_left)), dim=0) if neg_left else positive_left
    all_right = torch.cat((positive_right, torch.cat(neg_right)), dim=0) if neg_right else positive_right
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


def mutual_topk_score(
    p: torch.Tensor,
    n: torch.Tensor,
    inv_temperature: torch.Tensor | float | None = None,
    p_attention: torch.Tensor | None = None,
    n_attention: torch.Tensor | None = None,
    k: int = 2,
    top_n: int = 30,
) -> torch.Tensor:
    """Aggregate a protein--nucleic residue score matrix.

    The defaults are retained only for backwards compatibility with legacy
    protein--RNA runs.  Canonical evaluations must pass the independently
    validation-selected ``k`` and ``top_n`` values explicitly; they are not
    inherited from the PPI retrieval protocol.
    """
    score = p @ n.T
    if inv_temperature is not None:
        score = score * torch.as_tensor(inv_temperature, device=score.device, dtype=score.dtype)
    score = torch.clamp(score, min=-1e2, max=1e2)
    if p_attention is not None and n_attention is not None:
        score = score * p_attention.to(score).unsqueeze(1).square() * n_attention.to(score).unsqueeze(0).square()
    kr, kc = min(k, score.size(1)), min(k, score.size(0))
    rows = torch.zeros_like(score, dtype=torch.bool)
    cols = torch.zeros_like(score, dtype=torch.bool)
    rows.scatter_(1, score.topk(kr, dim=1).indices, True)
    cols.scatter_(0, score.topk(kc, dim=0).indices, True)
    values = score[rows & cols]
    return values.topk(min(top_n, values.numel())).values.sum() if values.numel() else score.new_tensor(0.0)
