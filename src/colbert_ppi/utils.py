"""
Utilities for ColBERT-PPI training: logging, seeding and checkpointing.
"""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    logger_obj: logging.Logger,
    output_dir: Path,
    *,
    enable: bool = True,
) -> Optional[Path]:
    """Configure file + console logging. Returns log file path."""
    if not enable:
        logger_obj.setLevel(logging.WARNING)
        return None
    if logger_obj.handlers:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"train_{timestamp}.log"

    logger_obj.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger_obj.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger_obj.addHandler(ch)

    return log_file


# ---------------------------------------------------------------------------
# Device / DDP
# ---------------------------------------------------------------------------

def parse_gpu_ids(gpu_str: str) -> list[int]:
    """Parse GPU ID string, e.g. '0' → [0], '0,1' → [0, 1]."""
    if not gpu_str or gpu_str.lower() == "none":
        return []
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    if is_dist_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    metrics: dict,
    path: Path,
    *,
    include_optimizer_state: bool = True,
    include_scaler_state: bool = True,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if include_optimizer_state and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if include_scaler_state and scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)
    parts = ["model"]
    if include_optimizer_state and optimizer is not None:
        parts.append("optimizer")
    if include_scaler_state and scaler is not None:
        parts.append("scaler")
    logger.info(f"Checkpoint saved: {path} ({'+'.join(parts)})")


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load training checkpoint. Returns checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from {path} (epoch {checkpoint.get('epoch', '?')})")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ---------------------------------------------------------------------------
# Gradient scaler
# ---------------------------------------------------------------------------

def create_grad_scaler(bf16: bool) -> Optional[torch.cuda.amp.GradScaler]:
    # bf16 autocast does not need loss scaling — GradScaler is a no-op.
    return None


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 5,
    steps_per_epoch: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing with linear warmup."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
