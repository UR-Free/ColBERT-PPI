"""OmniBioTE adapter and base-aligned DNA/RNA tokenisation.

OmniBioTE's published single-character vocabulary uses the same A/T/C/G/N
base tokens for DNA and RNA, while an explicit leading header token preserves
the molecular type.  RNA U is therefore encoded as T exactly as in the
official implementation; the ``<RNA>``/``<DNA>`` token keeps the distinction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


BASE_IDS = {"A": 4, "T": 5, "C": 6, "G": 7, "N": 8}
TYPE_IDS = {"dna": 9, "mrna": 10, "rna": 11, "rrna": 12, "trna": 13}
EOS_ID = 3


def encode_nucleic_sequence(sequence: str, nucleic_type: str) -> torch.Tensor:
    """Encode one DNA/RNA sequence into L+2 OmniBioTE char tokens."""
    kind = nucleic_type.lower().replace("-", "")
    if kind not in TYPE_IDS:
        raise ValueError(f"Unsupported nucleic type {nucleic_type!r}; expected DNA or RNA")
    sequence = sequence.upper().replace("U", "T")
    bases = [BASE_IDS.get(base, BASE_IDS["N"]) for base in sequence]
    return torch.tensor([TYPE_IDS[kind], *bases, EOS_ID], dtype=torch.long)


class OmniBioTEBackbone(nn.Module):
    """Load an official pickled OmniBioTE model as a token encoder."""

    def __init__(self, checkpoint: str, code_dir: str):
        super().__init__()
        code_path = str(Path(code_dir).resolve())
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Published checkpoints are OmniBioTA objects.  Accept the wrapper used
        # for binding-energy checkpoints as a defensive compatibility measure.
        if hasattr(model, "omnibiota"):
            model = model.omnibiota
        if not hasattr(model, "config") or not hasattr(model.config, "n_embd"):
            raise TypeError(f"Unsupported OmniBioTE checkpoint object: {type(model)!r}")
        self.model = model
        self.hidden_size = int(model.config.n_embd)
        self.context_length = int(model.config.context_length)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.force_eval = True
        self.model.eval()

    def set_force_eval(self, force_eval: bool) -> None:
        self.force_eval = bool(force_eval)
        if self.force_eval:
            self.eval()

    def train(self, mode: bool = True):
        # Frozen backbones must stay deterministic.  LoRA-enabled backbones set
        # force_eval=False so adapter dropout and transformer dropout can train.
        actual_mode = bool(mode) and not self.force_eval
        super().train(actual_mode)
        self.model.train(actual_mode)
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"OmniBioTE input length {input_ids.size(1)} exceeds context {self.context_length}"
            )
        # SDPA boolean masks use True for positions that are allowed to attend.
        mask = attention_mask.bool()[:, None, None, :]
        if self.force_eval:
            self.model.eval()
        return self.model(input_ids, attn_mask=mask, return_embeddings=True)
