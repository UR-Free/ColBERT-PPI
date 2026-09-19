"""Protein and RNA encoders with residue contact supervision."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .pri_context import FlashAttentionEncoder


class RNARelationAdapter(nn.Module):
    """Gated categorical message passing over RNA base-pair/stacking edges."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 32,
        dropout: float = 0.0,
        gate_init: float = -6.9,
    ):
        super().__init__()
        self.rank = int(rank)
        self.dropout = float(dropout)
        # Codes 1..6 correspond to the explicit relation schema. Code 0 is no edge.
        self.relation_down = nn.ModuleList(
            [nn.Linear(hidden_size, rank, bias=False) for _ in range(6)]
        )
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

    def forward(
        self, hidden: torch.Tensor, relations: torch.Tensor | None
    ) -> torch.Tensor:
        if relations is None or relations.numel() == 0:
            return hidden
        relation_type = relations[..., 0].long().clamp(0, 6)
        orientation = relations[..., 1].long().clamp(0, 4)
        direction = relations[..., 2].long().clamp(0, 2)
        confidence = relations[..., 3].to(hidden)
        confidence = confidence * self.orientation_scale(orientation).squeeze(-1).to(
            hidden
        )
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

    def __init__(
        self,
        saprot_dir: str,
        rinalmo_name: str,
        hidden_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 0.07,
        tune_saprot: bool = False,
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
        nucleic_lora_dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import EsmConfig, EsmForMaskedLM

        cfg = EsmConfig.from_pretrained(saprot_dir)
        self.saprot = EsmForMaskedLM(cfg)
        state = torch.load(
            f"{saprot_dir}/pytorch_model.bin", map_location="cpu", weights_only=False
        )
        self.saprot.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("lm_head.")},
            strict=False,
        )
        self.saprot.lm_head = None
        self.nucleic_model_type = nucleic_model_type.lower()
        if self.nucleic_model_type == "rinalmo":
            from multimolecule import RiNALMoModel

            self.rinalmo = RiNALMoModel.from_pretrained(rinalmo_name)
            nucleic_hidden_size = int(self.rinalmo.config.hidden_size)
        elif self.nucleic_model_type == "omnibiote":
            if not omnibiote_checkpoint:
                raise ValueError(
                    "omnibiote_checkpoint is required for nucleic_model_type='omnibiote'"
                )
            from src.omnibiote_adapter import OmniBioTEBackbone

            self.rinalmo = OmniBioTEBackbone(omnibiote_checkpoint, omnibiote_code_dir)
            nucleic_hidden_size = self.rinalmo.hidden_size
        elif self.nucleic_model_type == "ernie_rna":
            if not ernie_rna_checkpoint:
                raise ValueError(
                    "ernie_rna_checkpoint is required for nucleic_model_type='ernie_rna'"
                )
            from .rna_backbone import ERNIERNABackbone

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
            from .lora import inject_lora_linear_layers

            if self.nucleic_model_type == "omnibiote":
                target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.nucleic_model_type == "ernie_rna":
                # ERNIE-RNA's custom attention fast path accesses projection
                # weights directly, so only FFN LoRA is guaranteed active.
                target_modules = ["fc1", "fc2"]
            else:
                target_modules = None
            target = (
                self.rinalmo.model
                if self.nucleic_model_type in {"omnibiote", "ernie_rna"}
                else self.rinalmo
            )
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
            if rna_relation_adapter_rank > 0
            else None
        )

        self.protein_encoder = FlashAttentionEncoder(
            cfg.hidden_size + 1, hidden_dim, num_heads, num_layers, dropout
        )
        self.nucleic_encoder = FlashAttentionEncoder(
            nucleic_hidden_size + 1, hidden_dim, num_heads, num_layers, dropout
        )
        self.protein_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.nucleic_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.log_inv_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature))
        )

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
    def _strip(
        output: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs are right-padded BOS/residue/EOS sequences. A batch-wide slice
        # removes EOS only for the longest row; exclude each row's own EOS
        # before contextual encoding and pooling while preserving tensor width.
        positions = torch.arange(1, mask.shape[1] - 1, device=mask.device)[None, :]
        residue_mask = mask[:, 1:-1].bool() & (
            positions < (mask.sum(dim=1) - 1)[:, None]
        )
        return output[:, 1:-1], residue_mask

    def forward(
        self,
        protein_ids: torch.Tensor,
        protein_mask: torch.Tensor,
        nucleic_ids: torch.Tensor,
        nucleic_mask: torch.Tensor,
        nucleic_structure: torch.Tensor | None = None,
        nucleic_relations: torch.Tensor | None = None,
    ):
        # Frozen backbones run under no_grad; LoRA/full-tuned backbones must keep
        # gradients through the frozen base path into trainable adapters.
        p_kwargs = {
            "input_ids": protein_ids,
            "attention_mask": protein_mask.float(),
            "return_dict": True,
        }
        if self.tune_saprot or self.saprot_lora_enabled:
            p_out = self.saprot.esm(**p_kwargs).last_hidden_state
        else:
            with torch.no_grad():
                p_out = self.saprot.esm(**p_kwargs).last_hidden_state
        if self.nucleic_lora_enabled:
            if self.nucleic_model_type == "rinalmo":
                n_out = self.rinalmo(
                    input_ids=nucleic_ids, attention_mask=nucleic_mask, return_dict=True
                ).last_hidden_state
            elif self.nucleic_model_type == "omnibiote":
                n_out = self.rinalmo(nucleic_ids, nucleic_mask)
            else:
                n_out = self.rinalmo(nucleic_ids, nucleic_mask, nucleic_structure)
        else:
            with torch.no_grad():
                if self.nucleic_model_type == "rinalmo":
                    n_out = self.rinalmo(
                        input_ids=nucleic_ids,
                        attention_mask=nucleic_mask,
                        return_dict=True,
                    ).last_hidden_state
                elif self.nucleic_model_type == "omnibiote":
                    n_out = self.rinalmo(nucleic_ids, nucleic_mask)
                else:
                    n_out = self.rinalmo(nucleic_ids, nucleic_mask, nucleic_structure)
        if self.rna_relation_adapter is not None:
            n_out = self.rna_relation_adapter(n_out, nucleic_relations)
        p_out, p_mask = self._strip(p_out, protein_mask)
        n_out, n_mask = self._strip(n_out, nucleic_mask)
        p_role = torch.ones(
            (*p_out.shape[:2], 1), device=p_out.device, dtype=p_out.dtype
        )
        n_role = torch.zeros(
            (*n_out.shape[:2], 1), device=n_out.device, dtype=n_out.dtype
        )
        p = F.normalize(
            self.protein_encoder(torch.cat((p_out, p_role), -1), p_mask), p=2, dim=-1
        )
        n = F.normalize(
            self.nucleic_encoder(torch.cat((n_out, n_role), -1), n_mask), p=2, dim=-1
        )
        pa = torch.sigmoid(self.protein_attention(p).squeeze(-1)).masked_fill(
            ~p_mask, 0
        )
        na = torch.sigmoid(self.nucleic_attention(n).squeeze(-1)).masked_fill(
            ~n_mask, 0
        )
        return p, n, p_mask, n_mask, pa, na, self.inv_temperature
