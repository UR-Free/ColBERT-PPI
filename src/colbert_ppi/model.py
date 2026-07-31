"""SaProt residue representations followed by a direct per-residue MLP."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def component_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the complete trainable LoRA/MLP scoring state on CPU."""
    if hasattr(model, "module"):
        model = model.module
    prefixes = ("query_projector.", "candidate_projector.")
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if "lora_" in name or name.startswith(prefixes) or name == "log_inv_temperature":
            state[name] = value.detach().cpu().clone()
    if not any("lora_" in name for name in state):
        raise RuntimeError("No LoRA adapter parameters found while exporting components")
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in state):
            raise RuntimeError(f"No parameters found for {prefix.rstrip('.')}")
    return state


def load_component_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | dict,
) -> tuple[list[str], list[str]]:
    """Load an epoch LoRA/MLP component checkpoint into a constructed model."""
    if hasattr(model, "module"):
        model = model.module
    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    state = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected component keys: {unexpected}")
    return list(incompatible.missing_keys), unexpected

# ===========================================================================
# Direct per-residue MLP projector
# ===========================================================================

class ResidueMLP(nn.Module):
    """Per-residue MLP projection with L2-normalised output.

    Applies the same MLP independently to every residue in a batch:
        Linear(in_dim, hidden) → GELU → Dropout → Linear(hidden, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
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
        projected = self.net(x)
        return F.normalize(projected.float(), p=2, dim=-1).to(projected.dtype)



# ===========================================================================
# Main PPI model
# ===========================================================================

class ColBERTPPIModel(nn.Module):
    """SaProt-powered residue-level ColBERT model for PPI scoring.

    SaProt extracts sequence+structure (foldseek 3Di) residue representations.
    Independent query and candidate MLPs map each residue. No post-SaProt
    Transformer, residue weighting, or contextual encoder is present.

    Training uses contact-map contrastive learning (InfoNCE on residue pairs).
    Mutual top-k PPI scoring is used for evaluation only (not in loss).

    Parameters
    ----------
    input_dim : int
        SaProt per-residue embedding dim (1280 for SaProt_650M).
    hidden_dim : int
        Hidden and output size of the direct residue MLP.
    dropout : float
        Dropout probability.
    ppi_temperature : float
        Initial temperature for the learnable logit scale.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        ppi_temperature: float = 0.07,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_projector = ResidueMLP(
            in_dim=input_dim,
            hidden=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.candidate_projector = ResidueMLP(
            in_dim=input_dim,
            hidden=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # --- Learnable inverse temperature ---
        init_inv_temp = 1.0 / max(ppi_temperature, 1e-6)
        self.log_inv_temperature = nn.Parameter(
            torch.ones([]) * math.log(max(min(init_inv_temp, 20.0), 0.05))
        )
        self.min_log_inv_temp = math.log(0.05)
        self.max_log_inv_temp = math.log(20.0)

        self._init_weights()

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

    def encode_protein(
        self,
        residue_repr: torch.Tensor,
        residue_mask: torch.Tensor,
        *,
        tower: str,
    ) -> torch.Tensor:
        """Encode one protein into L2-normalised residue embeddings.

        Parameters
        ----------
        residue_repr : (B, L, input_dim)
            SaProt per-residue features.
        residue_mask : (B, L) bool, optional
            True = valid residue, False = padding.
        tower : {"query", "candidate"}
            Projection head to apply.

        Returns
        -------
        h : (B, L, hidden_dim)  L2-normalised residue embeddings.
        """
        if tower == "query":
            projector = self.query_projector
        elif tower == "candidate":
            projector = self.candidate_projector
        else:
            raise ValueError(f"Unknown projection tower: {tower!r}")
        h = projector(residue_repr)
        return h.masked_fill(~residue_mask.unsqueeze(-1), 0.0)

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
    ):
        """Full forward pass for a batch of protein pairs.

        Returns
        -------
        Tuple of (query_embeddings, candidate_embeddings, inv_temperature)
            embeddings: (B, L, D) L2-normalised residue embeddings
            inv_temperature: scalar  clamped learnable temperature
        """
        query_embeddings = self.encode_protein(
            repr1, mask1, tower="query"
        )
        candidate_embeddings = self.encode_protein(
            repr2, mask2, tower="candidate"
        )
        return query_embeddings, candidate_embeddings, self.get_inv_temperature()


# ===========================================================================
# Model factory
# ===========================================================================

def create_model(model_type: str, **kwargs) -> nn.Module:
    """Build a PPI model by type string.

    Parameters
    ----------
    model_type : str
        "colbert"     → ColBERTPPIModel (query/candidate residue MLPs)
        "colbert_lora"→ ColBERTPPIModelWithLoRA (SaProt+LoRA + two MLP heads)
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    nn.Module
    """
    model_type = model_type.lower().strip()
    if model_type == "colbert":
        return ColBERTPPIModel(**kwargs)
    elif model_type == "colbert_lora":
        return ColBERTPPIModelWithLoRA(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: 'colbert', 'colbert_lora'."
        )


# ===========================================================================
# ColBERT model with SaProt + LoRA backbone
# ===========================================================================

class ColBERTPPIModelWithLoRA(ColBERTPPIModel):
    """SaProt + LoRA followed by query and candidate per-residue MLPs.

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
    hidden_dim, dropout, ppi_temperature :
        Same as ColBERTPPIModel.
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
        dropout: float = 0.1,
        ppi_temperature: float = 0.07,
        sequence_only: bool = False,
        gradient_checkpointing: bool = False,
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
        self.gradient_checkpointing = bool(gradient_checkpointing)

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
        if self.gradient_checkpointing:
            self.saprot_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.saprot_backbone.enable_input_require_grads()

        # Two direct residue-wise mappings; no post-SaProt Transformer.
        self.query_projector = ResidueMLP(
            in_dim=_input_dim,
            hidden=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.candidate_projector = ResidueMLP(
            in_dim=_input_dim,
            hidden=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # --- Learnable inverse temperature ---
        init_inv_temp = 1.0 / max(ppi_temperature, 1e-6)
        self.log_inv_temperature = nn.Parameter(
            torch.ones([]) * math.log(max(min(init_inv_temp, 20.0), 0.05))
        )
        self.min_log_inv_temp = math.log(0.05)
        self.max_log_inv_temp = math.log(20.0)

        # Only init the new (non-SaProt) parts — PEFT already initialises
        # LoRA adapters correctly (A~kaiming, B~zeros).
        self._init_new_weights()

    def _init_new_weights(self):
        """Xavier-init only the MLP heads (not SaProt/LoRA)."""
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
        # Query and candidate proteins pass through the shared SaProt backbone.
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

        # --- 2. Independent query/candidate residue projections ---
        h1 = self.encode_protein(
            repr1_raw, mask1_stripped, tower="query"
        )
        h2 = self.encode_protein(
            repr2_raw, mask2_stripped, tower="candidate"
        )

        # --- 3. Temperature (zero-multiply keeps the parameter in the graph) ---
        inv_temp = self.get_inv_temperature()
        inv_temp = inv_temp + 0.0 * self.log_inv_temperature
        return h1, h2, inv_temp
