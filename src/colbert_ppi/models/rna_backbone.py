"""ERNIE-RNA backbone adapter with an explicit RNA-structure input channel.

The upstream ERNIE-RNA checkpoint uses a sequence-derived canonical base-pair
matrix as a two-dimensional attention bias.  This adapter preserves that
pre-trained path and adds a sparse RNA-internal relation matrix before the
official nonlinear 2D projection.  The current preprocessing uses directed,
type-aware base-pair and stacking relations; no protein--RNA contact labels are
used to construct the structure matrix.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


def encode_ernie_rna_sequence(sequence: str) -> torch.Tensor:
    """Return official ERNIE-RNA character IDs with CLS/EOS tokens."""
    token_ids = {"G": 4, "A": 5, "U": 6, "T": 6, "C": 7}
    sequence = sequence.upper().replace("T", "U")
    return torch.tensor(
        [0, *(token_ids.get(base, 3) for base in sequence), 2],
        dtype=torch.long,
    )


def _vendor_utils(code_dir: str):
    """Load the upstream ``src`` namespace under a collision-free alias."""
    source_root = Path(code_dir).resolve() / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"ERNIE-RNA source directory not found: {source_root}")
    alias = "_colbert_ernie_rna_vendor"
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [str(source_root)]
        package.__package__ = alias
        sys.modules[alias] = package
    return importlib.import_module(f"{alias}.utils")


@contextmanager
def _trusted_checkpoint_loading():
    """Restore the pre-PyTorch-2.6 torch.load default for the trusted checkpoint."""
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


class ERNIERNABackbone(nn.Module):
    """Base-aligned ERNIE-RNA representations from sequence and RNA relations."""

    hidden_size = 768
    pad_token_id = 1

    def __init__(
        self,
        checkpoint: str,
        code_dir: str = "third_party/ERNIE-RNA",
        structure_weight: float = 2.0,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"ERNIE-RNA checkpoint not found: {checkpoint_path}"
            )
        vendor = _vendor_utils(code_dir)
        dictionary_dir = str((Path(code_dir).resolve() / "src" / "dict"))
        with _trusted_checkpoint_loading():
            pretrained = vendor.load_pretrained_ernierna(
                str(checkpoint_path), {"data": dictionary_dir}
            )
        # The official feature extractor passes this encoder to ErnieRNAOnestage.
        self.model = pretrained.encoder
        self.structure_weight = float(structure_weight)
        self._force_eval = True

        # Official creatmat weights: AU=2, GC=3, GU=0.8, in both directions.
        pair_prior = torch.zeros((30, 30), dtype=torch.float32)
        for left, right, value in (
            (5, 6, 2.0),
            (6, 5, 2.0),
            (4, 7, 3.0),
            (7, 4, 3.0),
            (4, 6, 0.8),
            (6, 4, 0.8),
        ):
            pair_prior[left, right] = value
        self.register_buffer("pair_prior", pair_prior, persistent=False)

    def set_force_eval(self, enabled: bool) -> None:
        self._force_eval = bool(enabled)
        if self._force_eval:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._force_eval:
            self.model.eval()
        return self

    def _two_dimensional_input(
        self,
        input_ids: torch.Tensor,
        structure: torch.Tensor | None,
    ) -> torch.Tensor:
        safe_ids = input_ids.clamp(min=0, max=self.pair_prior.size(0) - 1)
        prior = self.pair_prior[safe_ids.unsqueeze(2), safe_ids.unsqueeze(1)]
        if structure is not None:
            if structure.shape != prior.shape:
                raise ValueError(
                    f"RNA structure matrix shape {tuple(structure.shape)} does not match "
                    f"token-pair shape {tuple(prior.shape)}"
                )
            prior = prior + self.structure_weight * structure.to(prior)
        return prior.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask  # Padding is inferred by the official encoder from ID 1.
        two_d = self._two_dimensional_input(input_ids, structure)
        _, _, extra = self.model(
            input_ids,
            twod_tokens=two_d,
            is_twod=True,
            extra_only=True,
            masked_only=False,
        )
        # Official inner states are T x B x C; the final state is base aligned.
        return extra["inner_states"][-1].transpose(0, 1)
