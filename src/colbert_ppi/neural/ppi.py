"""SaProt with LoRA and a shared role-conditioned residue encoder."""

import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .ppi_context import FlashAttentionEncoder


class ProteinPairModel(nn.Module):
    def __init__(
        self, saprot_dir, hidden_dim=512, num_heads=16, num_layers=3, dropout=0.1
    ):
        super().__init__()
        from transformers import EsmConfig, EsmForMaskedLM
        from peft import LoraConfig, get_peft_model

        config = EsmConfig.from_pretrained(saprot_dir)
        self.saprot_backbone = EsmForMaskedLM(config)
        state = torch.load(
            Path(saprot_dir) / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        self.saprot_backbone.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("lm_head.")},
            strict=False,
        )
        self.saprot_backbone.lm_head = None
        for parameter in self.saprot_backbone.parameters():
            parameter.requires_grad = False
        self.saprot_backbone.esm = get_peft_model(
            self.saprot_backbone.esm,
            LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"],
            ),
        )
        self.encoder = FlashAttentionEncoder(
            config.hidden_size + 1, hidden_dim, num_heads, num_layers, dropout
        )
        self.residue_attn = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 4, 32), 1),
        )
        self.log_inv_temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, ids, mask, role):
        hidden = self.saprot_backbone.esm(
            input_ids=ids, attention_mask=mask.float(), return_dict=True
        ).last_hidden_state[:, 1:-1]
        positions = torch.arange(1, ids.shape[1] - 1, device=ids.device)[None, :]
        residues = mask[:, 1:-1] & (positions < mask.sum(1)[:, None] - 1)
        indicator = torch.full(
            (*hidden.shape[:2], 1),
            1.0 if role == 0 else 0.0,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        vectors = self.encoder(torch.cat([hidden, indicator], -1), residues)
        vectors = F.normalize(vectors.float(), dim=-1)
        attention = torch.sigmoid(self.residue_attn(vectors).squeeze(-1)).masked_fill(
            ~residues, 0
        )
        return vectors, residues, attention

    def forward(self, left_ids, left_mask, right_ids, right_mask):
        left, lmask, la = self.encode(left_ids, left_mask, 0)
        right, rmask, ra = self.encode(right_ids, right_mask, 1)
        temperature = self.log_inv_temperature.exp().clamp(0.05, 20)
        return left, right, lmask, rmask, la, ra, temperature
