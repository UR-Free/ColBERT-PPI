"""Contextual encoders for protein–RNA adaptation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout
        )
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask
    probabilities = torch.softmax(scores, dim=-1)
    if dropout > 0.0:
        probabilities = F.dropout(probabilities, p=dropout, training=True)
    return torch.matmul(probabilities, value)


class FlashAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        expanded = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = float(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)
        batch, length, width = normalized.shape
        query, key, value = self.qkv_proj(normalized).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        attention_mask = (
            padding_mask.unsqueeze(1).unsqueeze(2) if padding_mask is not None else None
        )
        attended = _scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            attention_mask=attention_mask,
            dropout=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        x = residual + self.out_proj(attended)
        return x + self.ffn(self.norm2(x))


class FlashAttentionEncoder(nn.Module):
    """Exact state-dict-compatible encoder used by the formal PPI source."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                FlashAttentionLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        residue_repr: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.dropout(self.in_proj(residue_repr))
        for layer in self.layers:
            hidden = layer(hidden, padding_mask=padding_mask)
        return self.final_norm(hidden)
