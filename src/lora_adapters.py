"""Small dependency-light LoRA adapters for non-HuggingFace backbones."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap ``nn.Linear`` with a frozen base path plus trainable LoRA update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass
class LoRAInjectionSummary:
    layers: int
    trainable_parameters: int


def _matches(name: str, leaf: str, target_modules: Iterable[str] | None) -> bool:
    if target_modules is None:
        return True
    return any(target == leaf or name.endswith(target) or target in name for target in target_modules)


def inject_lora_linear_layers(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str] | None = None,
    prefix: str = "",
) -> LoRAInjectionSummary:
    """Recursively replace matching ``nn.Linear`` children with ``LoRALinear``."""
    layers = 0
    trainable = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and _matches(full_name, child_name, target_modules):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, child_name, wrapped)
            layers += 1
            trainable += sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        else:
            summary = inject_lora_linear_layers(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                prefix=full_name,
            )
            layers += summary.layers
            trainable += summary.trainable_parameters
    return LoRAInjectionSummary(layers=layers, trainable_parameters=trainable)
