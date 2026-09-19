"""Role-conditioned PPI contextual encoder."""

from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

    # PyTorch < 2.0 fallback (manual attention).
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(~attn_mask, neg_inf)
        else:
            attn_scores = attn_scores + attn_mask

    attn_probs = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    return torch.matmul(attn_probs, v)


class FlashAttentionLayer(nn.Module):
    """Transformer layer using PyTorch scaled_dot_product_attention (FlashAttention).

    Uses pre-LayerNorm with GELU-activated FFN (mlp_ratio=4.0).
    Supports padding mask via boolean attn_mask in sdpa.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # QKV projection (combined for efficiency; FlashAttention expects (B, H, L, D))
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, L, D) float tensor
        padding_mask : (B, L) bool tensor, True = valid residue (NOT padding).
                       Converted to additive mask internally for sdpa.
        """
        residual = x
        x_norm = self.norm1(x)

        B, L, D = x_norm.shape

        # Combined QKV → split & reshape for multi-head
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, D)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, L, D_h)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Build attention mask: masked positions get -inf attention
        attn_mask = None
        if padding_mask is not None:
            # padding_mask: (B, L), True = valid.  Convert to (B, 1, 1, L) boolean.
            # sdpa expects True = attend (keep), False = mask out.
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        attn_out = _scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            training=self.training,
        )

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FlashAttentionEncoder(nn.Module):
    """Transformer encoder with FlashAttention backbone.

    Replaces the v7 DistanceAwareEncoder: projects input features to hidden_dim,
    passes through num_layers of FlashAttentionLayer, and outputs
    (normed_hidden, raw_hidden).
    """

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
                FlashAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        residue_repr: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        residue_repr : (B, L, input_dim) float tensor
        padding_mask : (B, L) bool tensor, True = valid, False = pad.

        Returns
        -------
        normed : (B, L, hidden_dim)  LayerNorm output
        """
        x = self.dropout(self.in_proj(residue_repr))
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        x = self.final_norm(x)
        return x
