#!/usr/bin/env python3
"""
v8 SaProt ColBERT PPI training entry point.

Pipeline:
  1. Pre-extract SaProt features:  python pre_extract.py
  2. Train:                        python train.py

Training uses residue-contact InfoNCE exclusively. Evaluation uses
residue-level late interaction exclusively.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure v8/ is on sys.path so `from src.xxx` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.config import V8Config
from src.dataset import (
    EntityUniqueDistributedBatchSampler,
    MixedContactDataset,
    MixedStructureContactDataset,
    SaProtContactDataset,
    SaProtLoRAExplicitContactDataset,
    SaProtLoRAContactDataset,
    SaProtLoRAContactDatasetV3,
)
from src.eval_labels import load_positive_mask
from src.model import create_model
from src.trainer import (
    run_train_epoch,
    run_eval_epoch,
)
from src.utils import (
    count_parameters,
    create_grad_scaler,
    create_scheduler,
    load_checkpoint,
    save_checkpoint,
    parse_gpu_ids,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(cfg: Optional[V8Config] = None) -> argparse.Namespace:
    """Parse CLI args with defaults pulled from V8Config (single source of truth).

    Parameters
    ----------
    cfg : V8Config, optional
        If provided, used as the default base.  Otherwise V8Config() is used.

    Returns
    -------
    argparse.Namespace
    """
    if cfg is None:
        cfg = V8Config()

    parser = argparse.ArgumentParser(
        description="v8 SaProt ColBERT PPI Training — defaults from src/config.py"
    )

    # Optional JSON config file to override V8Config fields
    parser.add_argument("--config", type=str, default=None,
                        help="JSON file with V8Config overrides")

    # Data — paths default from V8Config; no longer required on CLI
    parser.add_argument("--data_mode", type=str,
                        default=getattr(cfg, "data_mode", "ddi"),
                        choices=["ddi", "pinder", "mix", "pinder_structure_mix"],
                        help=("Training data mode: ddi, pinder, mix (DDI+PINDER), "
                              "or pinder_structure_mix (PDB+AFDB)"))
    parser.add_argument("--mix_ratio", type=float,
                        default=getattr(cfg, "mix_ratio", 2.0),
                        help="DDI:PINDER ratio in mix mode (default: 2.0)")
    parser.add_argument("--pdb_afdb_mix_ratio", type=float,
                        default=getattr(cfg, "pdb_afdb_mix_ratio", 1.0),
                        help="PDB:AFDB sample ratio per structure-mix epoch")
    parser.add_argument("--entity_unique_batches", action="store_true",
                        default=getattr(cfg, "entity_unique_batches", False),
                        help="Forbid protein reuse across the full distributed batch")
    parser.add_argument("--train_csv", type=str, default=cfg.train_csv)
    parser.add_argument("--val_csv", type=str, default=cfg.val_csv)
    parser.add_argument("--test_csv", type=str, default=cfg.test_csv)
    parser.add_argument("--train_saprot", type=str, default=getattr(cfg, "train_saprot", ""))
    parser.add_argument("--val_saprot", type=str, default=getattr(cfg, "val_saprot", ""))
    parser.add_argument("--test_saprot", type=str, default=getattr(cfg, "test_saprot", ""))
    parser.add_argument("--train_cb", type=str, default=cfg.train_cb)
    parser.add_argument("--val_cb", type=str, default=cfg.val_cb)
    parser.add_argument("--test_cb", type=str, default=cfg.test_cb)
    parser.add_argument("--data_dir", type=str, default=cfg.data_dir)
    parser.add_argument("--processed_prefix", type=str, default=None)
    # PINDER train paths (used when data_mode == "pinder" or "mix")
    parser.add_argument("--pinder_train_csv", type=str,
                        default=getattr(cfg, "pinder_train_csv", ""))
    parser.add_argument("--pinder_train_cb", type=str,
                        default=getattr(cfg, "pinder_train_cb", ""))
    parser.add_argument("--pinder_train_saprot_inputs", type=str,
                        default=getattr(cfg, "pinder_train_saprot_inputs", ""))
    parser.add_argument("--pinder_train_saprot", type=str,
                        default=getattr(cfg, "pinder_train_saprot", ""))
    parser.add_argument("--pinder_train_contact_map", type=str,
                        default=getattr(cfg, "pinder_train_contact_map", ""),
                        help="Optional explicit sparse contact labels for PINDER training")
    parser.add_argument("--val_contact_map", type=str,
                        default=getattr(cfg, "val_contact_map", ""),
                        help="Optional explicit sparse contact labels for validation")
    parser.add_argument("--test_contact_map", type=str,
                        default=getattr(cfg, "test_contact_map", ""),
                        help="Optional explicit sparse contact labels for test")

    # Model
    parser.add_argument("--model_type", type=str, default=cfg.model_type,
                        choices=["mlp", "colbert"],
                        help="Model architecture: 'mlp' (default) or 'colbert'")
    parser.add_argument("--input_dim", type=int, default=cfg.saprot_input_dim)
    # MLP model
    parser.add_argument("--mlp_hidden", type=int, default=cfg.mlp_hidden)
    parser.add_argument("--mlp_output_dim", type=int, default=cfg.mlp_output_dim)
    # ColBERT model
    parser.add_argument("--hidden_dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--num_heads", type=int, default=cfg.num_heads)
    parser.add_argument("--num_layers", type=int, default=cfg.num_layers)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    # Shared
    parser.add_argument("--ppi_temperature", type=float, default=cfg.ppi_temperature)
    parser.add_argument(
        "--untied_encoder", action="store_true",
        default=getattr(cfg, "untied_encoder", False),
        help="Use independent receptor/ligand FlashAttention context encoders",
    )
    parser.add_argument(
        "--sequence_only", action="store_true",
        default=getattr(cfg, "sequence_only", False),
        help="Mask SaProt 3Di symbols by mapping every residue token to AA#",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup_epochs", type=int, default=cfg.warmup_epochs)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)

    # Loss
    parser.add_argument("--pos_per_sample", type=int, default=cfg.pos_per_sample)
    parser.add_argument("--neg_per_sample", type=int, default=cfg.neg_per_sample)

    # Eval
    parser.add_argument("--eval_freq", type=int, default=cfg.eval_freq)
    parser.add_argument(
        "--validation_during_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable validation/checkpoint selection during training. Disable only "
            "for fixed-update control trajectories whose final epoch is the "
            "pre-registered endpoint."
        ),
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save best_model.pt from the final training epoch without optimizer "
            "state. Intended for fixed-update control trajectories."
        ),
    )
    parser.add_argument(
        "--save_initial_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save the rank-0 model state immediately after DDP parameter "
            "synchronisation and before any optimiser update. This provides "
            "an auditable 0%%-PPI source checkpoint for data-scaling studies."
        ),
    )
    parser.add_argument("--sync_eval", action="store_true",
                        default=getattr(cfg, "sync_eval", False),
                        help="Run PINDER validation/test evaluation synchronously on the training GPU")
    parser.add_argument("--val_calibrated_pairs", type=str,
                        default=getattr(cfg, "val_calibrated_pairs", ""),
                        help="Optional calibrated positive-pair CSV for validation retrieval metrics")
    parser.add_argument("--test_calibrated_pairs", type=str,
                        default=getattr(cfg, "test_calibrated_pairs", ""),
                        help="Optional calibrated positive-pair CSV for test retrieval metrics")
    parser.add_argument("--eval_selection_metric", type=str,
                        default=getattr(cfg, "eval_selection_metric", "val_auprc"),
                        help="Validation metric used to select best_model.pt")
    parser.add_argument(
        "--eval_test_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=getattr(cfg, "eval_test_each_epoch", True),
        help=(
            "Legacy per-epoch test monitoring. Disable for validation-only checkpoint "
            "selection and run test once after training."
        ),
    )
    parser.add_argument(
        "--eval_worker_script",
        type=str,
        default="",
        help=(
            "Optional repository-relative asynchronous evaluator. Empty keeps "
            "scripts/eval_worker.py; confirmatory protocols may provide a frozen "
            "validation-only evaluator."
        ),
    )
    parser.add_argument(
        "--final_test_after_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the legacy in-process test evaluation after training. Disable when "
            "a protocol-specific one-time test evaluator is required."
        ),
    )
    parser.add_argument(
        "--eval_shutdown_timeout",
        type=int,
        default=120,
        help="Seconds to wait for the final asynchronous validation before termination.",
    )
    parser.add_argument("--val_score_mask1", type=str,
                        default=getattr(cfg, "val_score_mask1", ""),
                        help="Optional dict[label_a] bool mask restricting side-1 residues for PPI scoring")
    parser.add_argument("--test_score_mask1", type=str,
                        default=getattr(cfg, "test_score_mask1", ""),
                        help="Optional dict[label_a] bool mask restricting side-1 residues for PPI scoring")

    # Data loading
    parser.add_argument("--max_seq_len", type=int, default=cfg.max_seq_len,
                        help="Max seq len for eval (None = no truncation)")
    parser.add_argument("--train_max_seq_len", type=int, default=cfg.train_max_seq_len,
                        help="Max seq len for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--train_subset_ratio", type=float, default=cfg.train_subset_ratio,
                        help="Fraction of training set to use (1.0 = full)")

    # Output
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Initialize model weights from checkpoint but start a fresh fine-tuning run",
    )

    # GPU
    parser.add_argument("--gpu", type=str, default=cfg.gpu)
    parser.add_argument("--bf16", action="store_true", default=cfg.bf16,
                        help="Enable bf16 mixed-precision (autocast + no GradScaler)")
    parser.add_argument("--no_bf16", action="store_false", dest="bf16",
                        help="Disable mixed-precision (full FP32)")
    parser.add_argument("--eval_port", type=int, default=29599,
                        help="Socket port for train↔eval communication (default: 29599)")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=cfg.use_lora,
                        help="Enable LoRA fine-tuning of SaProt backbone")
    parser.add_argument("--lora_r", type=int, default=cfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=cfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=cfg.lora_dropout)
    parser.add_argument("--saprot_dir", type=str, default=cfg.saprot_dir,
                        help="Local SaProt HuggingFace checkpoint directory")
    parser.add_argument("--train_saprot_inputs", type=str,
                        default=getattr(cfg, "train_saprot_inputs", ""))
    parser.add_argument("--val_saprot_inputs", type=str,
                        default=getattr(cfg, "val_saprot_inputs", ""))
    parser.add_argument("--test_saprot_inputs", type=str,
                        default=getattr(cfg, "test_saprot_inputs", ""))
    # --- DDI v3 (per-pair, PAE-filtered) ---
    parser.add_argument("--ddi_v3", action="store_true", default=False,
                        help="Use DDI v3 format (per-pair coords, PAE-filtered contacts)")
    parser.add_argument("--ddi_v3_contact_pae", type=str, default="",
                        help="Path to ddi_v3_contact_pae_lt1.pt")

    # Misc
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--log_interval", type=int, default=cfg.log_interval)
    parser.add_argument("--compile", action="store_true",
                        default=getattr(cfg, "compile_model", False),
                        help="Enable torch.compile (disabled by default — CUDA graph hangs on A100)")

    args = parser.parse_args()

    # If --config JSON was passed, merge it on top (CLI args already won)
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r") as fh:
                json_overrides = json.load(fh)
            # Only set attributes that are still at their V8Config default
            cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
            for key, val in json_overrides.items():
                if key in cfg_dict and getattr(args, key) == cfg_dict[key]:
                    setattr(args, key, val)
                elif key not in cfg_dict and hasattr(args, key) and getattr(args, key) in ("", None):
                    setattr(args, key, val)

    return args


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def _broadcast_object(obj, src: int = 0):
    if dist.is_available() and dist.is_initialized():
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]
    return obj


def _init_distributed(local_rank: int, world_size: int, use_cuda: bool) -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", str(local_rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=3600),
        )


def _resolve_device(
    *,
    local_rank: int,
    world_size: int,
    gpu_ids: list[int],
) -> torch.device:
    if not torch.cuda.is_available() or not gpu_ids:
        return torch.device("cpu")

    if world_size > 1:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            device_index = local_rank
        elif gpu_ids:
            device_index = gpu_ids[local_rank]
        else:
            device_index = local_rank
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            device_index = 0
        elif gpu_ids:
            device_index = gpu_ids[0]
        else:
            device_index = 0

    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


# ---------------------------------------------------------------------------
# Async eval helpers (rank-0 only)
# ---------------------------------------------------------------------------

def _spawn_eval_worker(
    output_dir: Path,
    eval_gpu: int,
    python_bin: str = "python",
    eval_port: int = 29599,
    worker_script: str = "",
) -> subprocess.Popen:
    """Launch the eval worker subprocess on a dedicated GPU."""
    if worker_script:
        worker_path = Path(worker_script)
        if not worker_path.is_absolute():
            worker_path = Path(__file__).resolve().parent.parent / worker_path
        worker_path = worker_path.resolve()
    else:
        worker_path = Path(__file__).resolve().parent / "eval_worker.py"
    if not worker_path.is_file():
        raise FileNotFoundError(f"Evaluation worker not found: {worker_path}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    proc = subprocess.Popen(
        [python_bin, str(worker_path),
         "--output_dir", str(output_dir),
         "--eval_gpu", str(eval_gpu),
         "--eval_port", str(eval_port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # setsid — survives parent exit
    )
    logger.info(
        f"Spawned eval worker (PID={proc.pid}) on GPU {eval_gpu}: {worker_path}"
    )
    return proc


def _check_eval_results(output_dir: Path, best_val_auprc: float,
                        output_dir_path: Path,
                        selection_metric: str = "val_auprc") -> tuple[float, bool]:
    """Check for completed eval results and manage checkpoint files.

    For each eval_result_epoch{N}.json found:
      - If val_auprc improved: copy temp_ckpt_epoch{N}.pt → best_model.pt
      - Otherwise: delete temp_ckpt_epoch{N}.pt
      - Delete the result file.

    Returns (new_best_val_auprc, found_any).
    """
    def _result_epoch(path: Path) -> int:
        try:
            return int(path.stem.replace("eval_result_epoch", ""))
        except ValueError:
            return sys.maxsize

    # Numeric ordering preserves the documented earliest-epoch tie rule even
    # if more than one asynchronous result becomes visible at once (plain
    # lexical ordering would place epoch 10 before epoch 2).
    result_files = sorted(
        output_dir.glob("eval_result_epoch*.json"), key=_result_epoch
    )
    found_any = False
    for rf in result_files:
        found_any = True
        try:
            epoch = int(rf.stem.replace("eval_result_epoch", ""))
        except ValueError:
            continue

        try:
            with rf.open("r") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            val_auprc = metrics.get(selection_metric, 0.0)
            val_attn_auprc = metrics.get("val_attn_auprc", None)
        except Exception:
            val_auprc = 0.0
            val_attn_auprc = None

        temp_ckpt = output_dir / f"temp_ckpt_epoch{epoch}.pt"

        attn_str = f", attn_auprc={val_attn_auprc:.4f}" if val_attn_auprc is not None else ""
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_path = output_dir_path / "best_model.pt"
            if temp_ckpt.exists():
                try:
                    os.replace(temp_ckpt, best_path)
                    logger.info(
                        f"  >> New best {selection_metric}={val_auprc:.4f}{attn_str} "
                        f"(epoch {epoch}) — moved to best_model.pt"
                    )
                except OSError:
                    shutil.copy2(temp_ckpt, best_path)
                    logger.info(
                        f"  >> New best {selection_metric}={val_auprc:.4f}{attn_str} "
                        f"(epoch {epoch}) — copied to best_model.pt"
                    )
        else:
            logger.info(
                f"  Eval epoch {epoch}: {selection_metric}={val_auprc:.4f}{attn_str} "
                f"(no improvement over {best_val_auprc:.4f})"
            )

        # Clean up temp checkpoint (keep storage minimal)
        if temp_ckpt.exists():
            temp_ckpt.unlink()

        # Clean up result and signal
        rf.unlink()

    return best_val_auprc, found_any


def _send_eval_signal(epoch: int, eval_port: int = 29599) -> None:
    """Send eval signal for the given epoch via socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", eval_port))
        sock.sendall(json.dumps({"epoch": epoch, "action": "eval"}).encode("utf-8"))
        sock.close()
    except Exception as e:
        logger.warning(f"Failed to send eval signal for epoch {epoch}: {e}")


def _send_stop_signal(eval_port: int = 29599) -> None:
    """Send stop signal to the eval worker via socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", eval_port))
        sock.sendall(json.dumps({"action": "stop"}).encode("utf-8"))
        sock.close()
    except Exception as e:
        logger.warning(f"Failed to send stop signal: {e}")


def _save_temp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    train_metrics: dict,
    temp_ckpt_path: Path,
    eval_port: int = 29599,
) -> None:
    """Save temp checkpoint synchronously in the main thread.

    CRITICAL: state_dict is moved to CPU BEFORE torch.save, so the I/O path
    never touches CUDA.  This avoids the CUDA-context thread contention that
    caused multi-hour stalls with ThreadPoolExecutor.

    Typical save time: ~30-45 s for a 2.5 GB model — acceptable overhead
    relative to ~2 h per epoch.
    """
    tmp_path = temp_ckpt_path.with_suffix(temp_ckpt_path.suffix + ".tmp")
    save_start = time.perf_counter()

    try:
        # --- Move state_dict to CPU (fast: pure device→host copy) ---
        model_to_save = model.module if hasattr(model, "module") else model
        state_dict_cpu = model_to_save.state_dict()
        state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict_cpu.items()}

        # --- Build checkpoint (all CPU tensors) ---
        checkpoint = {
            "model_state_dict": state_dict_cpu,
            "epoch": epoch,
            "metrics": train_metrics,
        }

        # --- Save to disk (pure CPU I/O — no CUDA, no GIL contention issue) ---
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, temp_ckpt_path)

        save_elapsed = time.perf_counter() - save_start
        temp_size_mb = temp_ckpt_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Temp checkpoint saved for epoch {epoch} in {save_elapsed:.1f}s "
            f"({temp_size_mb:.1f} MB), signalling eval worker..."
        )

        # --- Signal eval worker ---
        _send_eval_signal(epoch, eval_port)

    except Exception:
        logger.exception(f"Checkpoint save failed for epoch {epoch}")
        if tmp_path.exists():
            tmp_path.unlink()


def _run_worker(
    local_rank: int,
    args: argparse.Namespace,
    gpu_ids: list[int],
    world_size: Optional[int] = None,
) -> None:
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    use_cuda = torch.cuda.is_available() and bool(gpu_ids)

    device = _resolve_device(local_rank=local_rank, world_size=world_size, gpu_ids=gpu_ids)

    if use_ddp:
        _init_distributed(local_rank, world_size, use_cuda)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    main_rank = rank == 0

    set_seed(args.seed + rank)

    start_time = datetime.now().astimezone()
    run_id = start_time.strftime("run_%Y%m%d_%H%M%S") if main_rank else ""
    run_id = _broadcast_object(run_id, src=0)
    # Resolve output_dir relative to v8 project root (not cwd)
    v8_root = Path(__file__).resolve().parent.parent  # v8/
    output_dir = (v8_root / args.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logger, output_dir, enable=main_rank)
    if not main_rank:
        logger.disabled = True

    if main_rank:
        logger.info("=" * 60)
        logger.info("v8 SaProt ColBERT PPI Training")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Args: {vars(args)}")

        with (output_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=2, default=str)

        # --- Spawn async eval worker on dedicated GPU ---
        # Resolve eval GPU: default to the next GPU after training GPUs,
        # or use EVAL_GPU env var if set.
        eval_port = getattr(args, "eval_port", 29599)
        eval_worker_proc = None
        if not args.validation_during_training:
            logger.info(
                "Validation during training disabled; the fixed final epoch will "
                "be saved only when --save_final_checkpoint is set"
            )
        elif args.sync_eval:
            if use_ddp:
                logger.error("sync_eval is only supported for single-GPU training")
                if dist.is_initialized():
                    dist.destroy_process_group()
                sys.exit(1)
            logger.info("Synchronous PINDER val/test evaluation enabled on the training GPU")
        else:
            eval_gpu_str = os.environ.get("EVAL_GPU", "")
            if eval_gpu_str:
                eval_gpu = int(eval_gpu_str)
            else:
                # Default: pick the GPU after the last training GPU
                # If training uses 0,1,2,3,4 → eval uses 5
                eval_gpu = max(gpu_ids) + 1

            # --- Validate: eval GPU MUST NOT overlap with training GPUs ---
            if eval_gpu in gpu_ids:
                logger.error(
                    f"GPU CONFLICT: eval GPU {eval_gpu} is also in training GPUs {gpu_ids}! "
                    f"Please set EVAL_GPU to a GPU outside the training set. "
                    f"Current training GPUs: {gpu_ids}"
                )
                if use_ddp and dist.is_initialized():
                    dist.destroy_process_group()
                sys.exit(1)

            eval_worker_proc = _spawn_eval_worker(
                output_dir, eval_gpu,
                python_bin=os.environ.get("PYTHON_BIN", sys.executable),
                eval_port=eval_port,
                worker_script=args.eval_worker_script,
            )
            logger.info(f"Async eval worker launched on GPU {eval_gpu}")

    # --- Datasets ---
    data_dir = v8_root / args.data_dir
    prefix = args.processed_prefix

    DatasetClass = SaProtLoRAContactDataset if args.use_lora else SaProtContactDataset

    if main_rank:
        logger.info(f"Loading training dataset (mode={args.data_mode})...")

    # --- Build training dataset(s) depending on data_mode ---
    mix_dataset = None  # only set in "mix" mode
    if args.data_mode == "pinder_structure_mix":
        if not args.use_lora:
            raise ValueError("PINDER PDB+AFDB structure mixing requires tokenized LoRA inputs")
        if not getattr(args, "pinder_train_contact_map", ""):
            raise ValueError("AFDB structure mixing requires projected explicit contact labels")

        pdb_dataset = SaProtLoRAContactDataset(
            csv_file=args.train_csv,
            saprot_inputs_pt=args.train_saprot_inputs,
            cb_npz=args.train_cb,
            processed_dir=data_dir,
            processed_prefix=prefix or Path(args.train_csv).stem,
            max_seq_len=args.train_max_seq_len,
            build_contacts=True,
        )
        afdb_dataset = SaProtLoRAExplicitContactDataset(
            csv_file=args.pinder_train_csv,
            saprot_inputs_pt=args.pinder_train_saprot_inputs,
            cb_npz=args.pinder_train_cb,
            contact_map_pt=args.pinder_train_contact_map,
            max_seq_len=args.train_max_seq_len,
            build_contacts=True,
        )
        train_dataset = MixedStructureContactDataset(pdb_dataset, afdb_dataset)
        base_train_dataset = train_dataset
        full_train_len = len(train_dataset)
        if main_rank:
            logger.info(
                f"PINDER structure mix: PDB={len(pdb_dataset)} + AFDB={len(afdb_dataset)} "
                f"(epoch ratio={args.pdb_afdb_mix_ratio}:1), "
                f"combined records={full_train_len}, feat_dim={train_dataset.feat_dim}"
            )

    elif args.data_mode == "mix":
        # ---- DDI sub-dataset ----
        use_ddi_v3 = getattr(args, "ddi_v3", False)
        if use_ddi_v3:
            # V3: per-pair storage, PAE-filtered contacts
            ddi_v3_pae = getattr(args, "ddi_v3_contact_pae", "") or str(
                Path(args.train_csv).parent / "ddi_v3_contact_pae_lt1.pt"
            )
            ddi_dataset = SaProtLoRAContactDatasetV3(
                pair_csv=args.train_csv,
                saprot_inputs_pt=args.train_saprot_inputs,
                cb_npz=args.train_cb,
                contact_pae_pt=ddi_v3_pae,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
        else:
            # V2: per-domain storage (backward-compatible)
            ddi_kwargs = dict(
                csv_file=args.train_csv,
                cb_npz=args.train_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.train_csv).stem,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
            if args.use_lora:
                ddi_kwargs["saprot_inputs_pt"] = args.train_saprot_inputs or (
                    args.train_saprot.replace("_saprot.pt", "_saprot_inputs.pt")
                    if args.train_saprot else ""
                )
            else:
                ddi_kwargs["saprot_pt"] = args.train_saprot
            ddi_dataset = DatasetClass(**ddi_kwargs)

        # ---- PINDER train sub-dataset ----
        if getattr(args, "pinder_train_contact_map", ""):
            if not args.use_lora:
                raise ValueError("Explicit contact maps currently require --use_lora tokenized inputs")
            pinder_dataset = SaProtLoRAExplicitContactDataset(
                csv_file=args.pinder_train_csv,
                saprot_inputs_pt=args.pinder_train_saprot_inputs,
                cb_npz=args.pinder_train_cb,
                contact_map_pt=args.pinder_train_contact_map,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
        else:
            pinder_kwargs = dict(
                csv_file=args.pinder_train_csv,
                cb_npz=args.pinder_train_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.pinder_train_csv).stem,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
            if args.use_lora:
                pinder_kwargs["saprot_inputs_pt"] = args.pinder_train_saprot_inputs
            else:
                pinder_kwargs["saprot_pt"] = args.pinder_train_saprot or args.pinder_train_saprot_inputs.replace(
                    "_saprot_inputs.pt", "_saprot.pt"
                )
            pinder_dataset = DatasetClass(**pinder_kwargs)

        mix_kwargs = dict(
            dataset_a=ddi_dataset,
            dataset_b=pinder_dataset,
            ratio=args.mix_ratio,
            shuffle_a=True,
            seed=args.seed,
        )
        mix_dataset = MixedContactDataset(**mix_kwargs)
        train_dataset = mix_dataset
        base_train_dataset = mix_dataset
        ddi_len_mix = len(ddi_dataset)
        pinder_len_mix = len(pinder_dataset)
        full_train_len = len(mix_dataset)
        if main_rank:
            logger.info(
                f"Mix mode: DDI={ddi_len_mix} + PINDER={pinder_len_mix} → "
                f"{full_train_len} pairs/epoch (ratio={args.mix_ratio}:1), "
                f"feat_dim={mix_dataset.feat_dim}"
            )

    elif args.data_mode == "pinder":
        # ---- PINDER-only training ----
        if getattr(args, "pinder_train_contact_map", ""):
            if not args.use_lora:
                raise ValueError("Explicit contact maps currently require --use_lora tokenized inputs")
            train_dataset = SaProtLoRAExplicitContactDataset(
                csv_file=args.pinder_train_csv,
                saprot_inputs_pt=args.pinder_train_saprot_inputs,
                cb_npz=args.pinder_train_cb,
                contact_map_pt=args.pinder_train_contact_map,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
        else:
            dataset_kwargs = dict(
                csv_file=args.pinder_train_csv,
                cb_npz=args.pinder_train_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.pinder_train_csv).stem,
                max_seq_len=args.train_max_seq_len,
                build_contacts=True,
            )
            if args.use_lora:
                dataset_kwargs["saprot_inputs_pt"] = args.pinder_train_saprot_inputs
            else:
                dataset_kwargs["saprot_pt"] = args.pinder_train_saprot or args.pinder_train_saprot_inputs.replace(
                    "_saprot_inputs.pt", "_saprot.pt"
                )
            train_dataset = DatasetClass(**dataset_kwargs)
        base_train_dataset = train_dataset
        full_train_len = len(base_train_dataset)
        if main_rank:
            logger.info(
                f"PINDER train: {full_train_len} pairs, "
                f"feat_dim={base_train_dataset.feat_dim}"
            )
    else:
        # ---- DDI-only training (v3 per-pair, PAE-filtered contacts) ----
        ddi_v3_pae = getattr(args, "ddi_v3_contact_pae", "") or str(
            Path(args.train_csv).parent / "ddi_v3_contact_pae_lt1.pt"
        )
        train_dataset = SaProtLoRAContactDatasetV3(
            pair_csv=args.train_csv,
            saprot_inputs_pt=args.train_saprot_inputs,
            cb_npz=args.train_cb,
            contact_pae_pt=ddi_v3_pae if Path(ddi_v3_pae).exists() else None,
            max_seq_len=args.train_max_seq_len,
            build_contacts=True,
        )
        base_train_dataset = train_dataset
        full_train_len = len(base_train_dataset)
        if main_rank:
            logger.info(
                f"DDI v3 train: {full_train_len} pairs, "
                f"feat_dim={base_train_dataset.feat_dim}"
            )

    # --- Subset (applies to any mode) ---
    subset_ratio = float(args.train_subset_ratio)
    if subset_ratio <= 0:
        raise ValueError("train_subset_ratio must be > 0")
    if subset_ratio < 1.0:
        subset_size = max(1, int(full_train_len * subset_ratio))
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        indices = torch.randperm(full_train_len, generator=generator)[:subset_size].tolist()
        train_dataset = Subset(base_train_dataset, indices)
        if main_rank:
            logger.info(
                f"Train subset: {subset_size}/{full_train_len} pairs "
                f"({subset_ratio:.4f})"
            )

    train_sampler = None
    if args.entity_unique_batches:
        if not hasattr(base_train_dataset, "entity_pairs"):
            raise ValueError(
                "entity_unique_batches currently requires pinder_structure_mix"
            )
        if isinstance(train_dataset, Subset):
            subset_indices = [int(index) for index in train_dataset.indices]
            entity_pairs = [base_train_dataset.entity_pairs[index] for index in subset_indices]
            source_ids = [base_train_dataset.source_ids[index] for index in subset_indices]
        else:
            entity_pairs = base_train_dataset.entity_pairs
            source_ids = base_train_dataset.source_ids
        train_sampler = EntityUniqueDistributedBatchSampler(
            entity_pairs,
            batch_size=args.batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
            drop_last=True,
            source_ids=source_ids,
            pdb_to_afdb_ratio=args.pdb_afdb_mix_ratio,
        )
        if main_rank:
            sampler_audit = train_sampler.audit()
            logger.info(f"Entity-unique global batch sampler audit: {sampler_audit}")
            if sampler_audit["entity_reuse_violations"]:
                raise RuntimeError(f"Entity-unique sampler audit failed: {sampler_audit}")
    elif use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            # Keep matched conditions on the same per-seed sample order while
            # allowing the three T07 seeds to represent distinct training
            # trajectories.  PyTorch's implicit default is otherwise seed 0
            # for every run, irrespective of args.seed.
            seed=args.seed,
        )

    if args.entity_unique_batches:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=base_train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=base_train_dataset.collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    val_loader = None
    val_positive_mask = None
    val_score_mask1 = None
    if args.sync_eval:
        if main_rank:
            logger.info("Loading synchronous validation dataset...")
        if getattr(args, "val_contact_map", ""):
            if not args.use_lora:
                raise ValueError("Explicit contact maps currently require --use_lora tokenized inputs")
            val_dataset = SaProtLoRAExplicitContactDataset(
                csv_file=args.val_csv,
                saprot_inputs_pt=args.val_saprot_inputs or (
                    args.val_saprot.replace("_saprot.pt", "_saprot_inputs.pt")
                    if args.val_saprot else ""
                ),
                cb_npz=args.val_cb,
                contact_map_pt=args.val_contact_map,
                max_seq_len=args.max_seq_len,
                build_contacts=True,
            )
        else:
            val_kwargs = dict(
                csv_file=args.val_csv,
                cb_npz=args.val_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.val_csv).stem,
                max_seq_len=args.max_seq_len,
                build_contacts=True,
            )
            if args.use_lora:
                val_kwargs["saprot_inputs_pt"] = args.val_saprot_inputs or (
                    args.val_saprot.replace("_saprot.pt", "_saprot_inputs.pt")
                    if args.val_saprot else ""
                )
            else:
                val_kwargs["saprot_pt"] = args.val_saprot
            val_dataset = DatasetClass(**val_kwargs)
        if getattr(args, "val_calibrated_pairs", ""):
            val_positive_mask = load_positive_mask(args.val_calibrated_pairs, val_dataset.samples)
        if getattr(args, "val_score_mask1", ""):
            val_score_mask1 = torch.load(args.val_score_mask1, map_location="cpu", weights_only=False)
            if main_rank:
                logger.info(f"Loaded validation side-1 scoring masks: {len(val_score_mask1)} labels")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        if main_rank:
            logger.info(f"Sync val: {len(val_dataset)} pairs, feat_dim={val_dataset.feat_dim}")

    # --- Model ---
    effective_model_type = "colbert_lora" if args.use_lora else args.model_type
    if main_rank:
        logger.info(f"Building model (type={effective_model_type})...")
    model_kwargs = dict(
        input_dim=args.input_dim,
        ppi_temperature=args.ppi_temperature,
    )
    if args.use_lora:
        model_kwargs.update(
            saprot_dir=args.saprot_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            sequence_only=args.sequence_only,
            untied_encoder=args.untied_encoder,
        )
    elif args.model_type == "mlp":
        model_kwargs.update(
            mlp_hidden=args.mlp_hidden,
            output_dim=args.mlp_output_dim,
        )
    else:  # colbert
        model_kwargs.update(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            untied_encoder=args.untied_encoder,
        )

    model = create_model(effective_model_type, **model_kwargs).to(device)

    if use_ddp:
        # LoRA/SaProt can leave small parameter paths unused on some PINDER
        # batches, so keep DDP in the dynamic graph mode.
        _needs_unused = True
        ddp_kwargs = {
            "find_unused_parameters": _needs_unused,
            "static_graph": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update({
                "device_ids": [device.index],
                "output_device": device.index,
            })
        model = DDP(model, **ddp_kwargs)

    if main_rank and args.save_initial_checkpoint:
        initial_path = output_dir / "initial_model.pt"
        temporary = initial_path.with_suffix(initial_path.suffix + ".tmp")
        model_to_save = model.module if hasattr(model, "module") else model
        initial_state = {
            key: value.detach().cpu()
            for key, value in model_to_save.state_dict().items()
        }
        torch.save(
            {
                "model_state_dict": initial_state,
                "epoch": 0,
                "metrics": {},
                "role": "pre-optimisation PPI-0% source state",
            },
            temporary,
        )
        os.replace(temporary, initial_path)
        logger.info("Saved pre-optimisation checkpoint: %s", initial_path)

    # --- torch.compile: JIT-optimise the model graph (~10-20% speedup on A100) ---
    # Applied after DDP so compile wraps the per-rank replicated model.
    # "reduce-overhead" uses CUDA graphs for minimised kernel launch overhead,
    # at the cost of slightly higher first-epoch memory.  Falls back gracefully
    # on PyTorch < 2.0 or unsupported ops.
    use_compile = getattr(args, "compile", False)
    if hasattr(torch, "compile") and use_compile:
        model = torch.compile(model, mode="reduce-overhead")
        if main_rank:
            logger.info("torch.compile enabled (mode=reduce-overhead)")
    elif use_compile and main_rank:
        logger.info("torch.compile requested but not available")
    else:
        if main_rank:
            logger.info("torch.compile disabled (default)")

    if main_rank:
        total, trainable = count_parameters(model)
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        steps_per_epoch=len(train_loader),
    )
    scaler = create_grad_scaler(args.bf16 and device.type == "cuda")

    # --- Resume ---
    start_epoch = 1
    best_val_auprc = 0.0
    if args.init_checkpoint:
        load_checkpoint(model, Path(args.init_checkpoint), device=device)
        if main_rank:
            logger.info(f"Initialized model weights from {args.init_checkpoint}")
    if args.resume:
        ckpt = load_checkpoint(model, Path(args.resume), optimizer, scaler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auprc = ckpt.get("metrics", {}).get("val_auprc", 0.0)
        if main_rank:
            logger.info(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    if main_rank:
        logger.info(f"Training for {args.epochs} epochs (starting at {start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Reset dataset shuffle buffer (fresh shuffle each epoch)
        if hasattr(train_dataset, "on_epoch_begin"):
            train_dataset.on_epoch_begin()

        # Read-only T07/T11 resource telemetry. Resetting PyTorch's peak-memory
        # counters does not touch tensors, allocator state, gradients, RNGs, or
        # optimiser state; it only changes the reported high-water marks.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_train_started = time.perf_counter()

        train_metrics = run_train_epoch(
            model, train_loader, optimizer, scaler,
            pos_per_sample=args.pos_per_sample,
            neg_per_sample=args.neg_per_sample,
            device=device, bf16=args.bf16,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            epoch=epoch,
            show_progress=main_rank,
        )
        scheduler.step()

        if device.type == "cuda":
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            visible_list = [value.strip() for value in visible_devices.split(",") if value.strip()]
            local_index = int(device.index or 0)
            physical_gpu = (
                visible_list[local_index]
                if local_index < len(visible_list)
                else str(local_index)
            )
            memory_record = {
                "epoch": epoch,
                "rank": rank,
                "local_rank": local_rank,
                "physical_gpu": physical_gpu,
                "device_name": torch.cuda.get_device_name(device),
                "train_elapsed_seconds": time.perf_counter() - epoch_train_started,
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "memory_allocated_end_bytes": int(torch.cuda.memory_allocated(device)),
                "memory_reserved_end_bytes": int(torch.cuda.memory_reserved(device)),
            }
            memory_path = output_dir / f"gpu_memory_rank{rank}.jsonl"
            with memory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(memory_record, sort_keys=True) + "\n")

        if main_rank:
            logger.info(
                f"Epoch {epoch:>3d}/{args.epochs:<3d} | "
                f"loss={train_metrics['train_loss']:>7.4f}  "
                f"contact={train_metrics['train_contact_loss']:>7.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # --- Eval: save temp checkpoint + signal eval worker (rank 0 only) ---
        # Checkpoint is saved synchronously in the main thread AFTER moving
        # state_dict to CPU.  This guarantees that torch.save never touches
        # CUDA, eliminating the thread-contention stall seen with the old
        # ThreadPoolExecutor approach.
        # The eval worker runs on a dedicated GPU in a separate process, so
        # evaluation is fully parallel with the NEXT training epoch.
        do_eval = args.validation_during_training and (
            epoch % args.eval_freq == 0 or epoch == args.epochs
        )
        if do_eval and main_rank:
            if args.sync_eval:
                if val_loader is None:
                    raise RuntimeError("sync_eval requested but validation loader was not built")
                val_metrics = run_eval_epoch(
                    model, val_loader,
                    pos_per_sample=args.pos_per_sample,
                    neg_per_sample=args.neg_per_sample,
                    device=device, bf16=args.bf16,
                    epoch=epoch, stage="val",
                    show_progress=True,
                    positive_mask=val_positive_mask,
                    score_mask1_by_label=val_score_mask1,
                )
                val_auprc = val_metrics.get("val_auprc", 0.0)
                val_attn_auprc = val_metrics.get("val_attn_auprc", None)
                attn_str = f", attn_auprc={val_attn_auprc:.4f}" if val_attn_auprc is not None else ""
                history_path = output_dir / "eval_history.jsonl"
                with history_path.open("a") as fh:
                    fh.write(json.dumps({"epoch": epoch, "metrics": val_metrics}, default=str) + "\n")
                best_path = output_dir / "best_model.pt"
                if val_auprc >= best_val_auprc or not best_path.exists():
                    best_val_auprc = val_auprc
                    save_checkpoint(
                        model, optimizer, scaler, epoch, val_metrics, best_path,
                        include_optimizer_state=False,
                        include_scaler_state=False,
                    )
                    logger.info(
                        f"  >> New best val_auprc={val_auprc:.4f}{attn_str} "
                        f"(epoch {epoch}) — saved to best_model.pt"
                    )
                else:
                    logger.info(
                        f"  Eval epoch {epoch}: val_auprc={val_auprc:.4f}{attn_str} "
                        f"(no improvement over {best_val_auprc:.4f})"
                    )
            else:
                # 1) Check for completed eval results from previous epochs
                best_val_auprc, _ = _check_eval_results(
                    output_dir, best_val_auprc, output_dir, args.eval_selection_metric
                )

                # 2) Save temp checkpoint for current epoch (synchronous, CPU-only I/O)
                temp_ckpt_path = output_dir / f"temp_ckpt_epoch{epoch}.pt"
                _save_temp_checkpoint(
                    model, optimizer, scaler, epoch,
                    train_metrics,
                    temp_ckpt_path,
                    eval_port=eval_port,
                )
                # Eval worker is signalled inside _save_temp_checkpoint after
                # the file is fully written and atomically renamed.

        # Do not add an epoch-level DDP barrier here.
        # Rank 0 may still be busy writing the checkpoint or signaling the eval worker,
        # and blocking all ranks on a long-lived collective can trip NCCL watchdogs.
        # Training correctness is preserved by per-step DDP synchronisation.

    # --- Final: stop eval worker & process remaining results ---
    if main_rank:
        # Check for any remaining eval results
        if not args.validation_during_training:
            logger.info("Validation worker was disabled for this fixed-update run.")
        elif args.sync_eval:
            logger.info("Synchronous eval used; no async eval worker to stop.")
        else:
            best_val_auprc, _ = _check_eval_results(
                output_dir, best_val_auprc, output_dir, args.eval_selection_metric
            )

            # Send stop signal to eval worker
            logger.info("Sending stop signal to eval worker...")
            _send_stop_signal(eval_port)

            # Wait for eval worker to finish (with timeout)
            try:
                eval_worker_proc.wait(timeout=args.eval_shutdown_timeout)
                logger.info("Eval worker exited cleanly.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Eval worker did not exit within %ss, terminating...",
                    args.eval_shutdown_timeout,
                )
                eval_worker_proc.kill()
                eval_worker_proc.wait()

            # Final check for results (eval worker may have finished last epoch)
            best_val_auprc, _ = _check_eval_results(
                output_dir, best_val_auprc, output_dir, args.eval_selection_metric
            )

        if args.save_final_checkpoint:
            final_path = output_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scaler, args.epochs, train_metrics, final_path,
                include_optimizer_state=False,
                include_scaler_state=False,
            )
            logger.info(
                "Saved fixed-update final checkpoint at epoch %d: %s",
                args.epochs, final_path,
            )

        elapsed = (datetime.now().astimezone() - start_time).total_seconds()
        logger.info(
            f"Training complete. Elapsed: {elapsed:.0f}s. "
            f"Best {args.eval_selection_metric}: {best_val_auprc:.4f}"
        )
        logger.info(f"Outputs: {output_dir}")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    # =====================================================================
    # Test evaluation — load best checkpoint and evaluate on test split
    #   MUST run after DDP teardown so run_eval_epoch uses the single‑GPU
    #   path (no all_gather), otherwise non‑rank‑0 ranks would hang.
    # =====================================================================
    test_ckpt_path = output_dir / "best_model.pt"
    if test_ckpt_path.exists() and main_rank and args.final_test_after_training:
        logger.info("=" * 60)
        logger.info("Test evaluation — loading best checkpoint")
        load_checkpoint(model, test_ckpt_path, device=device)

        logger.info("Loading test dataset...")
        if getattr(args, "test_contact_map", ""):
            if not args.use_lora:
                raise ValueError(
                    "Explicit contact maps currently require --use_lora tokenized inputs"
                )
            test_dataset = SaProtLoRAExplicitContactDataset(
                csv_file=args.test_csv,
                saprot_inputs_pt=args.test_saprot_inputs or (
                    args.test_saprot.replace("_saprot.pt", "_saprot_inputs.pt")
                    if args.test_saprot else ""
                ),
                cb_npz=args.test_cb,
                contact_map_pt=args.test_contact_map,
                max_seq_len=args.max_seq_len,
                build_contacts=True,
            )
        elif args.use_lora:
            test_dataset = SaProtLoRAContactDataset(
                csv_file=args.test_csv,
                saprot_inputs_pt=args.test_saprot_inputs or (
                    args.test_saprot.replace("_saprot.pt", "_saprot_inputs.pt") if args.test_saprot else ""
                ),
                cb_npz=args.test_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.test_csv).stem,
                max_seq_len=args.max_seq_len,
                build_contacts=True,
            )
        else:
            test_dataset = SaProtContactDataset(
                csv_file=args.test_csv,
                saprot_pt=args.test_saprot,
                cb_npz=args.test_cb,
                processed_dir=data_dir,
                processed_prefix=prefix or Path(args.test_csv).stem,
                max_seq_len=args.max_seq_len,
                build_contacts=True,
            )
        logger.info(f"Test: {len(test_dataset)} pairs, feat_dim={test_dataset.feat_dim}")
        test_positive_mask = None
        if getattr(args, "test_calibrated_pairs", ""):
            test_positive_mask = load_positive_mask(args.test_calibrated_pairs, test_dataset.samples)
            logger.info(
                f"Test calibrated positive mask: {int(test_positive_mask.sum().item())} positives "
                f"from {args.test_calibrated_pairs}"
            )
        test_score_mask1 = None
        if getattr(args, "test_score_mask1", ""):
            test_score_mask1 = torch.load(args.test_score_mask1, map_location="cpu", weights_only=False)
            logger.info(f"Loaded test side-1 scoring masks: {len(test_score_mask1)} labels")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
        )

        test_metrics = run_eval_epoch(
            model, test_loader,
            pos_per_sample=args.pos_per_sample,
            neg_per_sample=args.neg_per_sample,
            device=device, bf16=args.bf16,
            epoch=0, stage="test",
            show_progress=True,
            positive_mask=test_positive_mask,
            score_mask1_by_label=test_score_mask1,
        )

        # --- Log test metrics (same format as val) ---
        loss_str = f"loss={test_metrics['test_loss']:.4f}"
        ctc_str = f"ctc={test_metrics.get('test_contact_loss', 0):.4f}"
        ret_acc = f"acc={test_metrics.get('test_acc', 0):.4f}"
        ret_top = f"top100={test_metrics.get('test_top100', 0):.4f} top200={test_metrics.get('test_top200', 0):.4f} top300={test_metrics.get('test_top300', 0):.4f}"
        rank_str = f"mrr={test_metrics.get('test_mrr', 0):.4f} auprc={test_metrics.get('test_auprc', 0):.4f}"
        logger.info(
            f"  TEST | {loss_str}  {ctc_str}  |  "
            f"{ret_acc}  {ret_top}  |  "
            f"{rank_str}"
        )

        # --- Save test metrics to JSON ---
        test_json_path = output_dir / "test_metrics.json"
        with test_json_path.open("w") as fh:
            json.dump(test_metrics, fh, indent=2, default=str)
        logger.info(f"Test metrics saved to {test_json_path}")
        logger.info("=" * 60)
    elif main_rank and not args.final_test_after_training:
        logger.info(
            "Final in-process test evaluation disabled; checkpoint remains frozen for "
            "the protocol-specific one-time evaluator."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu)

    if _is_torchrun():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _run_worker(local_rank, args, gpu_ids)
        return

    if len(gpu_ids) > 1 and torch.cuda.is_available():
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        mp.spawn(
            _run_worker,
            nprocs=len(gpu_ids),
            args=(args, gpu_ids, len(gpu_ids)),
            join=True,
        )
        return

    _run_worker(0, args, gpu_ids, 1)


if __name__ == "__main__":
    main()
