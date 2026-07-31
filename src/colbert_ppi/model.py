"""
ColBERTPPIModel — SaProt backbone + FlashAttentionEncoder + ColBERT late interaction.

Architecture:
  1. SaProt (frozen, pre-extracted) → per-residue representations (1280-dim)
  2. Side indicator: receptor=1.0, ligand=0.0 appended to each residue
  3. Shared FlashAttentionEncoder with efficient self-attention
  4. L2-normalise → ColBERT late interaction: h1 @ h2^T
  5. Contact head: pairwise residue contact prediction (auxiliary)
  6. Residue attention MLP: learnable per-residue importance weights
  7. Learnable inverse temperature for scoring

The SaProt backbone is NOT included in this model — features are pre-extracted
and loaded from disk (see pre_extract.py).  This keeps training efficient.
"""
from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ===========================================================================
# FlashAttention Transformer Layer & Encoder
# ===========================================================================

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

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D_h)
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
        self.layers = nn.ModuleList([
            FlashAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
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


# ===========================================================================
# Per-residue MLP projector  (v1-style R_MLP / L_MLP)
# ===========================================================================

class ResidueMLP(nn.Module):
    """Per-residue MLP projection with L2-normalised output.

    Applies the same MLP independently to every residue in a batch:
        Linear(in_dim, hidden) → ReLU → Linear(hidden, out_dim) → L2-norm
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project per-residue features.

        Parameters
        ----------
        x : (B, L, in_dim)

        Returns
        -------
        (B, L, out_dim)  L2-normalised along last dim.
        """
        return F.normalize(self.net(x), p=2, dim=-1)



# ===========================================================================
# Main PPI model
# ===========================================================================

class ColBERTPPIModel(nn.Module):
    """SaProt-powered residue-level ColBERT model for PPI scoring.

    SaProt extracts sequence+structure (foldseek 3Di) residue representations.
    These are fed into a shared FlashAttentionEncoder with efficient
    self-attention, and ColBERT late interaction computes cross-protein
    residue scores.

    Training uses contact-map contrastive learning (InfoNCE on residue pairs).
    Mutual top-k PPI scoring is used for evaluation only (not in loss).

    Parameters
    ----------
    input_dim : int
        SaProt per-residue embedding dim (1280 for SaProt_650M).
    hidden_dim : int
        Hidden size of the FlashAttentionEncoder (= residue embedding dim).
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dropout : float
        Dropout probability.
    ppi_temperature : float
        Initial temperature for the learnable logit scale.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ppi_temperature: float = 0.07,
        untied_encoder: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.untied_encoder = bool(untied_encoder)

        # --- Shared FlashAttention encoder ---
        self.encoder = FlashAttentionEncoder(
            input_dim=input_dim + 1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        if self.untied_encoder:
            # Start the ablation from identical towers so that the only
            # architectural change is whether their parameters remain tied.
            self.partner_encoder = copy.deepcopy(self.encoder)

        # --- Learnable inverse temperature ---
        init_inv_temp = 1.0 / max(ppi_temperature, 1e-6)
        self.log_inv_temperature = nn.Parameter(
            torch.ones([]) * math.log(max(min(init_inv_temp, 20.0), 0.05))
        )
        self.min_log_inv_temp = math.log(0.05)
        self.max_log_inv_temp = math.log(20.0)

        # --- Residue attention MLP ---
        self.residue_attn = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 4, 32), 1),
        )

        self._init_weights()
        self._synchronize_untied_encoder_initialization()

    def _synchronize_untied_encoder_initialization(self) -> None:
        """Make both context towers identical at initialization."""
        if self.untied_encoder:
            self.partner_encoder.load_state_dict(self.encoder.state_dict())

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier-uniform for linear weights, zero for biases."""
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Side indicator
    # ------------------------------------------------------------------
    @staticmethod
    def _augment_side_indicator(
        emb: torch.Tensor, side_value: float
    ) -> torch.Tensor:
        """Append per-residue scalar: receptor=1.0, ligand=0.0."""
        side_feature = torch.full(
            (*emb.shape[:2], 1),
            side_value,
            device=emb.device,
            dtype=emb.dtype,
        )
        return torch.cat([emb, side_feature], dim=-1)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_protein(
        self,
        residue_repr: torch.Tensor,
        residue_mask: torch.Tensor,
        role_id: int = 0,
    ) -> torch.Tensor:
        """Encode one protein into L2-normalised residue embeddings.

        Parameters
        ----------
        residue_repr : (B, L, input_dim)
            SaProt per-residue features.
        residue_mask : (B, L) bool, optional
            True = valid residue, False = padding.
        role_id : int
            0 = receptor (side=1.0), 1 = ligand (side=0.0).

        Returns
        -------
        h : (B, L, hidden_dim)  L2-normalised residue embeddings.
        """
        side_value = 1.0 if role_id == 0 else 0.0
        residue_repr = self._augment_side_indicator(residue_repr, side_value)

        encoder = (
            self.partner_encoder
            if self.untied_encoder and role_id == 1
            else self.encoder
        )
        h = encoder(residue_repr, residue_mask)

        # L2-normalise
        h = F.normalize(h.float(), p=2, dim=-1).to(dtype=h.dtype)

        return h

    def _select_role_ids(self) -> tuple[int, int]:
        """Assign side-1/side-2 inputs to their context-encoder roles.

        Untied encoders are deliberately role-specific: input 1 always uses
        the receptor tower and input 2 always uses the ligand tower.  The
        historical random role swap is retained only for the shared-encoder
        model, where it does not exchange independent tower parameters.
        """
        if self.untied_encoder:
            return 0, 1

        swap = self.training and torch.rand(1).item() > 0.5
        return (1, 0) if swap else (0, 1)

    # ------------------------------------------------------------------
    # Residue attention
    # ------------------------------------------------------------------
    def compute_residue_attention(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-residue importance weights via sigmoid (independent per residue).

        Each residue gets a weight in [0, 1] independently — no across-residue
        normalisation.  This preserves gradient signal when weights are multiplied
        into training logits, unlike per-protein softmax.

        Parameters
        ----------
        h : (B, L, hidden_dim)  L2-normalised residue embeddings.
        mask : (B, L) bool      True = valid residue.

        Returns
        -------
        weights : (B, L)  Sigmoid weights in [0, 1] (zero for padding).
        """
        raw = self.residue_attn(h).squeeze(-1)  # (B, L)
        return torch.sigmoid(raw).masked_fill(~mask, 0.0)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def get_inv_temperature(self, *, detach: bool = True) -> torch.Tensor:
        """Clamped learnable inverse temperature for score scaling."""
        inv_temperature = torch.clamp(
            self.log_inv_temperature.exp(),
            min=math.exp(self.min_log_inv_temp),
            max=math.exp(self.max_log_inv_temp),
        )
        return inv_temperature.detach() if detach else inv_temperature

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        repr1: torch.Tensor,
        repr2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        return_opposite: bool = False,
    ):
        """Full forward pass for a batch of protein pairs.

        Returns
        -------
        Tuple of (h1, h2, inv_temperature, attn_w1, attn_w2)
            h1, h2: (B, L, D)  L2-normalised residue embeddings
            inv_temperature: scalar  clamped learnable temperature
            attn_w1, attn_w2: (B, L)  per-residue attention weights (sigmoid, [0,1])
        """
        role1, role2 = self._select_role_ids()

        h1 = self.encode_protein(repr1, mask1, role_id=role1)
        h2 = self.encode_protein(repr2, mask2, role_id=role2)

        attn_w1 = self.compute_residue_attention(h1, mask1)
        attn_w2 = self.compute_residue_attention(h2, mask2)

        if return_opposite:
            h1_opp = self.encode_protein(repr1, mask1, role_id=1 - role1)
            h2_opp = self.encode_protein(repr2, mask2, role_id=1 - role2)
            attn_w1_opp = self.compute_residue_attention(h1_opp, mask1)
            attn_w2_opp = self.compute_residue_attention(h2_opp, mask2)
            return (
                h1, h2, self.get_inv_temperature(), attn_w1, attn_w2,
                h1_opp, h2_opp, attn_w1_opp, attn_w2_opp,
            )

        return h1, h2, self.get_inv_temperature(), attn_w1, attn_w2


# ===========================================================================
# Model factory
# ===========================================================================

def create_model(model_type: str, **kwargs) -> nn.Module:
    """Build a PPI model by type string.

    Parameters
    ----------
    model_type : str
        ``"colbert"`` builds the pre-extracted-feature model;
        ``"colbert_lora"`` builds the live SaProt+LoRA model.
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    nn.Module
    """
    model_type = model_type.lower().strip()
    if model_type == "colbert":
        return ColBERTPPIModel(**kwargs)
    if model_type == "colbert_lora":
        return ColBERTPPIModelWithLoRA(**kwargs)
    raise ValueError(
        f"Unknown model_type '{model_type}'. Supported: 'colbert', 'colbert_lora'."
    )


# ===========================================================================
# ColBERT model with SaProt + LoRA backbone
# ===========================================================================

class ColBERTPPIModelWithLoRA(ColBERTPPIModel):
    """SaProt + LoRA → FlashAttention → ColBERT PPI model.

    Unlike ColBERTPPIModel (which consumes pre-extracted features), this model
    includes the SaProt backbone with LoRA adapters.  Tokenized foldseek
    sequences (input_ids, attention_mask) are passed directly to the model.

    Parameters
    ----------
    saprot_dir : str
        Path to the local SaProt HuggingFace checkpoint directory.
    lora_r : int
        LoRA rank (default 8).
    lora_alpha : int
        LoRA scaling factor (default 8).
    lora_dropout : float
        LoRA dropout probability (default 0.1).
    input_dim : int
        SaProt hidden dim (1280 for SaProt_650M_AF2).  Ignored – determined
        from the SaProt config.
    hidden_dim, num_heads, num_layers, dropout, ppi_temperature :
        Same as ColBERTPPIModel (FlashAttention encoder + scoring).
    """

    def __init__(
        self,
        saprot_dir: str,
        *,
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        input_dim: int = 1280,          # ignored — read from SaProt config
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ppi_temperature: float = 0.07,
        sequence_only: bool = False,
        untied_encoder: bool = False,
    ):
        # Determine input_dim from SaProt config
        from transformers import EsmConfig, EsmForMaskedLM
        import os as _os
        saprot_cfg = EsmConfig.from_pretrained(saprot_dir)
        _input_dim = saprot_cfg.hidden_size

        # Bypass ColBERTPPIModel.__init__ — call nn.Module.__init__ directly
        # and set up the shared components ourselves.
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.sequence_only = bool(sequence_only)
        self.untied_encoder = bool(untied_encoder)

        # Build a vocabulary-derived 3Di -> "#" token remapping rather than
        # relying on checkpoint-specific numeric token IDs.
        vocab_path = Path(saprot_dir) / "vocab.txt"
        vocab = vocab_path.read_text().splitlines()
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        sequence_only_ids = torch.arange(len(vocab), dtype=torch.long)
        for idx, token in enumerate(vocab):
            if len(token) == 2 and token[0].isalpha():
                sequence_token = f"{token[0]}#"
                if sequence_token not in token_to_id:
                    raise ValueError(
                        f"Missing sequence-only token {sequence_token!r} in {vocab_path}"
                    )
                sequence_only_ids[idx] = token_to_id[sequence_token]
        self.register_buffer(
            "sequence_only_token_map", sequence_only_ids, persistent=False
        )

        # --- SaProt backbone (frozen) ---
        # Load manually to bypass transformers' torch>=2.6 requirement for
        # .bin checkpoints (CVE-2025-32434).  The local SaProt checkpoint is
        # trusted and we discard the lm_head immediately.
        self.saprot_backbone = EsmForMaskedLM(saprot_cfg)
        ckpt_path = _os.path.join(saprot_dir, "pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Discard lm_head weights (unused — we drop the head below)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head.")}
        self.saprot_backbone.load_state_dict(state_dict, strict=False)
        self.saprot_backbone.lm_head = None
        # Freeze backbone before applying LoRA
        for param in self.saprot_backbone.esm.parameters():
            param.requires_grad = False

        # --- Apply LoRA via PEFT ---
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            inference_mode=False,
        )
        self.saprot_backbone.esm = get_peft_model(
            self.saprot_backbone.esm, peft_config
        )

        # --- FlashAttention encoder (shared, same as ColBERTPPIModel) ---
        self.encoder = FlashAttentionEncoder(
            input_dim=_input_dim + 1,  # +1 for side indicator
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        if self.untied_encoder:
            self.partner_encoder = copy.deepcopy(self.encoder)

        # --- Learnable inverse temperature ---
        init_inv_temp = 1.0 / max(ppi_temperature, 1e-6)
        self.log_inv_temperature = nn.Parameter(
            torch.ones([]) * math.log(max(min(init_inv_temp, 20.0), 0.05))
        )
        self.min_log_inv_temp = math.log(0.05)
        self.max_log_inv_temp = math.log(20.0)

        # --- Residue attention MLP ---
        self.residue_attn = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 4, 32), 1),
        )

        # Only init the new (non-SaProt) parts — PEFT already initialises
        # LoRA adapters correctly (A~kaiming, B~zeros).
        self._init_new_weights()
        self._synchronize_untied_encoder_initialization()

    def _init_new_weights(self):
        """Xavier-init only the encoder + residue_attn (not SaProt/LoRA)."""
        for name, param in self.named_parameters():
            if name.startswith("saprot_backbone"):
                continue  # SaProt frozen + LoRA already init'd by PEFT
            if "weight" in name and param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Forward pass (overrides ColBERTPPIModel)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        repr1: torch.Tensor = None,   # ignored — kept for API compat
        repr2: torch.Tensor = None,   # ignored
        return_opposite: bool = False,
    ):
        """Full forward pass with live SaProt + LoRA.

        Parameters
        ----------
        input_ids1, input_ids2 : (B, L) long
            Tokenized foldseek structure-aware sequences.
        mask1, mask2 : (B, L) bool
            True = valid token, False = padding.

        Returns
        -------
        Same as ColBERTPPIModel.forward.
        """
        if self.sequence_only:
            input_ids1 = self.sequence_only_token_map[input_ids1]
            input_ids2 = self.sequence_only_token_map[input_ids2]

        # --- 1. Run SaProt backbone (frozen + LoRA) ---
        # We run receptor and ligand separately through SaProt.
        # SaProt ESM uses attention_mask where 1 = attend, 0 = pad.
        attn_mask1 = mask1.float()
        attn_mask2 = mask2.float()

        out1 = self.saprot_backbone.esm(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            output_hidden_states=False,
            return_dict=True,
        )
        out2 = self.saprot_backbone.esm(
            input_ids=input_ids2,
            attention_mask=attn_mask2,
            output_hidden_states=False,
            return_dict=True,
        )
        # Take last hidden state, strip <cls> (idx 0) and <eos> (idx L-1)
        repr1_raw = out1.last_hidden_state[:, 1:-1, :]  # (B, L-2, D)
        repr2_raw = out2.last_hidden_state[:, 1:-1, :]

        # Adjust mask to match stripped sequence
        mask1_stripped = mask1[:, 1:-1]
        mask2_stripped = mask2[:, 1:-1]

        # --- 2. Side indicator + FlashAttention encoder ---
        role1, role2 = self._select_role_ids()

        h1 = self.encode_protein(repr1_raw, mask1_stripped, role_id=role1)
        h2 = self.encode_protein(repr2_raw, mask2_stripped, role_id=role2)

        # --- 3. Residue attention ---
        attn_w1 = self.compute_residue_attention(h1, mask1_stripped)
        attn_w2 = self.compute_residue_attention(h2, mask2_stripped)

        # --- 4. Temperature (zero-multiply for DDP static_graph) ---
        inv_temp = self.get_inv_temperature()
        inv_temp = inv_temp + 0.0 * self.log_inv_temperature

        if return_opposite:
            h1_opp = self.encode_protein(repr1_raw, mask1_stripped, role_id=1 - role1)
            h2_opp = self.encode_protein(repr2_raw, mask2_stripped, role_id=1 - role2)
            attn_w1_opp = self.compute_residue_attention(h1_opp, mask1_stripped)
            attn_w2_opp = self.compute_residue_attention(h2_opp, mask2_stripped)
            return (
                h1, h2, inv_temp, attn_w1, attn_w2,
                h1_opp, h2_opp, attn_w1_opp, attn_w2_opp,
            )

        return h1, h2, inv_temp, attn_w1, attn_w2
