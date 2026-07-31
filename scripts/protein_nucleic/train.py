#!/usr/bin/env python3
"""Train the SaProt–nucleic-LM late-interaction model on prepared PNA contacts."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.protein_nucleic import (  # noqa: E402
    ProteinNucleicColBERT,
    ProteinNucleicDataset,
    sampled_contact_losses,
)


@dataclass
class Config:
    train_data: str = "data/protein_nucleic/processed/train.pt"
    val_data: str = "data/protein_nucleic/processed/val.pt"
    saprot_dir: str = "SaProt/weights/PLMs/SaProt_650M_PDB"
    rinalmo_name: str = "data/protein_nucleic/models/rinalmo-giga"
    nucleic_model_type: str = "rinalmo"
    omnibiote_checkpoint: str | None = None
    omnibiote_code_dir: str = "third_party/OmniBioTE/src"
    ernie_rna_checkpoint: str | None = None
    ernie_rna_code_dir: str = "third_party/ERNIE-RNA"
    rna_structure_weight: float = 2.0
    rna_relation_adapter_rank: int = 0
    rna_relation_dropout: float = 0.0
    rna_relation_gate_init: float = -6.9
    init_checkpoint: str | None = None
    protein_init_checkpoint: str | None = None
    output_dir: str = "outputs/protein_nucleic"
    device: str = "cuda:0"
    epochs: int = 30
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    max_protein_len: int = 256
    max_nucleic_len: int = 512
    eval_max_protein_len: int | None = None
    eval_max_nucleic_len: int | None = None
    hidden_dim: int = 512
    num_heads: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    saprot_lora_rank: int = 0
    saprot_lora_alpha: int = 8
    saprot_lora_dropout: float = 0.1
    nucleic_lora_rank: int = 0
    nucleic_lora_alpha: int = 8
    nucleic_lora_dropout: float = 0.1
    balance_nucleic_types: bool = False
    length_stratified_sampling: bool = False
    rna_length_bin_edges: list[int] | None = None
    rna_length_target_fractions: list[float] | None = None
    partner_diversity_weighting: bool = False
    partner_diversity_exponent: float = 0.5
    entity_safe_batch_sampling: bool = False
    random_interface_nucleic_crop: bool = False
    seed: int = 42
    save_full_model: bool = False
    early_stop_patience: int = 0
    min_epochs: int = 0
    min_delta: float = 0.0
    scheduler_type: str = "constant"
    checkpoint_selection: str = "val_loss"
    save_epoch_checkpoints: bool = False
    smoke: bool = False


class WeightedDistributedSampler(Sampler[int]):
    """Deterministic weighted sampling with DDP rank sharding."""

    def __init__(self, weights: torch.Tensor, num_replicas: int, rank: int,
                 seed: int = 0, replacement: bool = True):
        self.weights = weights.double().cpu()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = replacement
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class EntitySafeDistributedBatchSampler(Sampler[list[int]]):
    """Weighted local batches without known off-diagonal positive entity edges."""

    def __init__(self, dataset: ProteinNucleicDataset, weights: torch.Tensor,
                 batch_size: int, num_replicas: int, rank: int, seed: int = 0):
        self.weights = weights.double().cpu().clamp_min(0)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = int(math.floor(
            math.ceil(len(dataset) / self.num_replicas) / self.batch_size
        ))
        self.protein_keys = [str(row.get("protein_sequence", "")) for row in dataset.records]
        self.rna_keys = [
            str(row.get("nucleic_sequence", row.get("rna_sequence", "")))
            for row in dataset.records
        ]
        self.known_edges = set(zip(self.protein_keys, self.rna_keys))

    def _safe(self, index: int, proteins: set[str], rnas: set[str]) -> bool:
        protein, rna = self.protein_keys[index], self.rna_keys[index]
        if protein in proteins or rna in rnas:
            return False
        if any((protein, other_rna) in self.known_edges for other_rna in rnas):
            return False
        if any((other_protein, rna) in self.known_edges for other_protein in proteins):
            return False
        return True

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1009 * self.epoch + 104729 * self.rank)
        for _ in range(self.num_batches):
            batch, proteins, rnas = [], set(), set()
            for _position in range(self.batch_size):
                chosen = None
                proposals = torch.multinomial(
                    self.weights, 256, replacement=True, generator=generator
                ).tolist()
                for index in proposals:
                    if self._safe(index, proteins, rnas):
                        chosen = index
                        break
                if chosen is None:
                    safe = [
                        index for index in range(len(self.weights))
                        if self._safe(index, proteins, rnas)
                    ]
                    if not safe:
                        raise RuntimeError(
                            f"Cannot construct entity-safe batch of size {self.batch_size}"
                        )
                    position = int(torch.multinomial(
                        self.weights[safe], 1, replacement=True, generator=generator
                    ).item())
                    chosen = safe[position]
                batch.append(chosen)
                proteins.add(self.protein_keys[chosen])
                rnas.add(self.rna_keys[chosen])
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def nucleic_type_weights(dataset: ProteinNucleicDataset) -> torch.Tensor:
    counts: dict[str, int] = {}
    labels = []
    for record in dataset.records:
        label = str(record.get("nucleic_type", "unknown")).lower()
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1
    return torch.tensor([1.0 / max(counts[label], 1) for label in labels], dtype=torch.double)


def rna_training_weights(dataset: ProteinNucleicDataset, cfg: Config) -> tuple[torch.Tensor, dict]:
    """Length-stratified weights with optional protein/RNA-cluster diversity."""
    edges = list(cfg.rna_length_bin_edges or [25, 50, 100])
    targets = list(cfg.rna_length_target_fractions or [0.25, 0.25, 0.25, 0.25])
    if len(targets) != len(edges) + 1 or any(value < 0 for value in targets):
        raise ValueError("rna_length_target_fractions must be non-negative and one longer than bin edges")
    target_sum = sum(targets)
    if target_sum <= 0:
        raise ValueError("rna_length_target_fractions must have positive sum")
    targets = [value / target_sum for value in targets]

    def bin_index(length: int) -> int:
        return sum(length > edge for edge in edges)

    bins = [bin_index(len(str(row.get("nucleic_sequence", row.get("rna_sequence", ""))))) for row in dataset.records]
    bin_counts = np.bincount(bins, minlength=len(targets))
    weights = np.asarray([targets[b] / max(bin_counts[b], 1) for b in bins], dtype=np.float64)

    diversity_summary = None
    if cfg.partner_diversity_weighting:
        protein_keys = [str(row.get("protein_cluster_id", row.get("protein_sequence", i))) for i, row in enumerate(dataset.records)]
        rna_keys = [str(row.get("rna_cluster_id", row.get("nucleic_sequence", i))) for i, row in enumerate(dataset.records)]
        protein_counts = {key: protein_keys.count(key) for key in set(protein_keys)}
        rna_counts = {key: rna_keys.count(key) for key in set(rna_keys)}
        exponent = float(cfg.partner_diversity_exponent) / 2.0
        diversity = np.asarray([
            (1.0 / max(protein_counts[p] * rna_counts[r], 1)) ** exponent
            for p, r in zip(protein_keys, rna_keys)
        ])
        weights *= diversity
        diversity_summary = {
            "protein_clusters": len(protein_counts),
            "rna_clusters": len(rna_counts),
            "exponent": cfg.partner_diversity_exponent,
        }
    weights /= weights.mean()
    effective = np.asarray([weights[np.asarray(bins) == b].sum() for b in range(len(targets))])
    effective /= effective.sum()
    return torch.as_tensor(weights, dtype=torch.double), {
        "bin_edges": edges,
        "observed_counts": bin_counts.tolist(),
        "target_fractions": targets,
        "effective_fractions_after_diversity": effective.tolist(),
        "diversity": diversity_summary,
    }


def read_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one train and validation batch, then exit.")
    ns = parser.parse_args()
    def load_config(path: Path, seen: set[Path]) -> dict:
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Cyclic base_config inheritance at {path}")
        seen.add(path)
        payload = json.loads(path.read_text())
        base_config = payload.pop("base_config", None)
        if base_config:
            base_path = Path(base_config)
            if not base_path.is_absolute():
                root_candidate = ROOT / base_path
                base_path = root_candidate if root_candidate.exists() else path.parent / base_path
            base_payload = load_config(base_path, seen)
            base_payload.update(payload)
            payload = base_payload
        seen.remove(path)
        return payload

    payload = load_config(ns.config, set())
    cfg = Config(**payload)
    if ns.device:
        cfg.device = ns.device
    if ns.epochs:
        cfg.epochs = ns.epochs
    cfg.smoke = ns.smoke
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch, device: torch.device):
    return [getattr(batch, name).to(device) for name in (
        "protein_ids", "protein_mask", "nucleic_ids", "nucleic_mask",
        "contact", "contact_mask", "nucleic_structure", "nucleic_relations"
    )]


def run_epoch(model, loader, optimizer, cfg: Config, train: bool) -> dict:
    model.train(train)
    total = contact_total = 0.0
    n_batches = 0
    accum = cfg.gradient_accumulation_steps if train else 1
    distributed = dist.is_available() and dist.is_initialized()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.device.startswith("cuda"))
    step = -1
    loader_steps = len(loader)
    final_window = loader_steps % accum or accum
    for step, batch in enumerate(loader):
        p_ids, p_mask, n_ids, n_mask, contact, contact_mask, n_structure, n_relations = move(
            batch, next(model.parameters()).device
        )
        # The final partial accumulation window must also run a synchronized
        # backward pass.  Stepping directly after a no_sync() backward causes
        # DDP ranks to diverge whenever len(loader) is not divisible by accum.
        final_step = step + 1 == loader_steps
        optimizer_step = (step + 1) % accum == 0 or final_step or cfg.smoke
        sync_ctx = (
            model.no_sync()
            if distributed and train and not optimizer_step
            else contextlib.nullcontext()
        )
        with sync_ctx, torch.set_grad_enabled(train), autocast:
            p, n, pm, nm, pa, na, inv_t = model(
                p_ids, p_mask, n_ids, n_mask, nucleic_structure=n_structure,
                nucleic_relations=n_relations,
            )
            losses = sampled_contact_losses(
                p, n, contact, contact_mask,
                p_attention=pa, n_attention=na, inv_temperature=inv_t,
            )
            c_loss = losses["contact_loss"]
            full_loss = c_loss
            in_final_partial_window = (
                final_window < accum and step >= loader_steps - final_window
            )
            loss = full_loss / (final_window if in_final_partial_window else accum)
        if train:
            loss.backward()
            if optimizer_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += float(full_loss.detach())
        contact_total += float(c_loss.detach())
        n_batches += 1
        if cfg.smoke:
            break
    stats = torch.tensor(
        [total, contact_total, n_batches],
        device=next(model.parameters()).device,
    )
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total, contact_total, n_batches = stats.tolist()
    return {
        "loss": total / max(n_batches, 1),
        "contact_loss": contact_total / max(n_batches, 1),
        "batches": int(n_batches),
    }


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {name: state[name].detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def trainable_parameter_sha256(model: torch.nn.Module) -> str:
    """Bitwise digest used to prove that all DDP ranks ended synchronized."""
    raw_model = model.module if isinstance(model, DDP) else model
    digest = hashlib.sha256()
    for name, parameter in sorted(raw_model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict,
    cfg: Config,
    *,
    include_optimizer: bool,
) -> dict:
    raw_model = model.module if isinstance(model, DDP) else model
    state = raw_model.state_dict() if cfg.save_full_model else trainable_state_dict(raw_model)
    payload = {
        "model": state,
        "state_kind": "full" if cfg.save_full_model else "trainable",
        "epoch": epoch,
        "val": val_metrics,
        "config": asdict(cfg),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json_write(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    scheduler_type = cfg.scheduler_type.lower()
    if scheduler_type == "constant":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs, 1),
            eta_min=0.0,
        )
    raise ValueError(f"Unsupported scheduler_type={cfg.scheduler_type!r}; use 'constant' or 'cosine'.")


def main() -> int:
    cfg = read_args()
    if cfg.checkpoint_selection not in {"val_loss", "external_retrieval_auprc"}:
        raise ValueError(
            "checkpoint_selection must be 'val_loss' or 'external_retrieval_auprc', "
            f"got {cfg.checkpoint_selection!r}"
        )
    set_seed(cfg.seed)
    distributed = "RANK" in os.environ
    rank, local_rank = 0, 0
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    out = Path(cfg.output_dir) / run_id
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if distributed:
        dist.barrier()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"train_rank{rank}.log")])
    log = logging.getLogger("pna-train")
    device = torch.device(f"cuda:{local_rank}" if distributed else (cfg.device if torch.cuda.is_available() else "cpu"))
    train_ds = ProteinNucleicDataset(
        cfg.train_data, cfg.max_protein_len, cfg.max_nucleic_len,
        random_interface_nucleic_crop=cfg.random_interface_nucleic_crop,
    )
    val_ds = ProteinNucleicDataset(cfg.val_data, cfg.max_protein_len, cfg.max_nucleic_len)
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=ProteinNucleicDataset.collate_fn,
                       pin_memory=device.type == "cuda")
    sampling_summary = None
    if cfg.length_stratified_sampling:
        weights, sampling_summary = rna_training_weights(train_ds, cfg)
        if cfg.balance_nucleic_types:
            weights *= nucleic_type_weights(train_ds)
    elif cfg.balance_nucleic_types:
        weights = nucleic_type_weights(train_ds)
    if cfg.length_stratified_sampling or cfg.balance_nucleic_types:
        train_sampler = (
            WeightedDistributedSampler(
                weights,
                num_replicas=dist.get_world_size(),
                rank=rank,
                seed=cfg.seed,
            )
            if distributed else WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    if cfg.entity_safe_batch_sampling:
        if not (cfg.length_stratified_sampling or cfg.balance_nucleic_types):
            weights = torch.ones(len(train_ds), dtype=torch.double)
        entity_batch_sampler = EntitySafeDistributedBatchSampler(
            train_ds, weights, cfg.batch_size,
            num_replicas=dist.get_world_size() if distributed else 1,
            rank=rank, seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=entity_batch_sampler,
            num_workers=cfg.num_workers,
            collate_fn=ProteinNucleicDataset.collate_fn,
            pin_memory=device.type == "cuda",
        )
        train_sampler = entity_batch_sampler
    else:
        # Keep every optimisation step at the configured local contrastive-pool
        # size (24 complexes -> up to 120 sampled contacts), including under DDP.
        train_loader = DataLoader(
            train_ds, shuffle=train_sampler is None, sampler=train_sampler,
            drop_last=True, **loader_args,
        )
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_args)
    model = ProteinNucleicColBERT(
        cfg.saprot_dir, cfg.rinalmo_name, cfg.hidden_dim,
        cfg.num_heads, cfg.num_layers, cfg.dropout,
        nucleic_model_type=cfg.nucleic_model_type,
        omnibiote_checkpoint=cfg.omnibiote_checkpoint,
        omnibiote_code_dir=cfg.omnibiote_code_dir,
        ernie_rna_checkpoint=cfg.ernie_rna_checkpoint,
        ernie_rna_code_dir=cfg.ernie_rna_code_dir,
        rna_structure_weight=cfg.rna_structure_weight,
        rna_relation_adapter_rank=cfg.rna_relation_adapter_rank,
        rna_relation_dropout=cfg.rna_relation_dropout,
        rna_relation_gate_init=cfg.rna_relation_gate_init,
        saprot_lora_rank=cfg.saprot_lora_rank,
        saprot_lora_alpha=cfg.saprot_lora_alpha,
        saprot_lora_dropout=cfg.saprot_lora_dropout,
        nucleic_lora_rank=cfg.nucleic_lora_rank,
        nucleic_lora_alpha=cfg.nucleic_lora_alpha,
        nucleic_lora_dropout=cfg.nucleic_lora_dropout,
    )
    if cfg.init_checkpoint:
        initial = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        state = initial.get("model", initial)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in init_checkpoint: {incompatible.unexpected_keys[:20]}"
            )
        loaded = len(set(state).intersection(model.state_dict()))
        if loaded == 0:
            raise RuntimeError(f"No compatible parameters found in {cfg.init_checkpoint}")
        if rank == 0:
            log.info(
                "warm-start checkpoint=%s source_epoch=%s loaded_keys=%d missing_keys=%d",
                cfg.init_checkpoint, initial.get("epoch"), loaded,
                len(incompatible.missing_keys),
            )
    if cfg.protein_init_checkpoint:
        protein_initial = torch.load(
            cfg.protein_init_checkpoint, map_location="cpu", weights_only=False
        )
        protein_state = protein_initial.get("model", protein_initial)
        allowed_prefixes = ("saprot.", "protein_encoder.", "protein_attention.")
        invalid = [key for key in protein_state if not key.startswith(allowed_prefixes)]
        if invalid:
            raise RuntimeError(
                f"Non-protein keys in protein_init_checkpoint: {invalid[:20]}"
            )
        incompatible = model.load_state_dict(protein_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected protein warm-start keys: "
                f"{incompatible.unexpected_keys[:20]}"
            )
        loaded = len(set(protein_state).intersection(model.state_dict()))
        if loaded != len(protein_state):
            raise RuntimeError(
                f"Only {loaded}/{len(protein_state)} protein warm-start keys loaded"
            )
        if rank == 0:
            log.info(
                "protein warm-start checkpoint=%s source_epoch=%s loaded_keys=%d",
                cfg.protein_init_checkpoint, protein_initial.get("source_epoch"), loaded,
            )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    if rank == 0:
        log.info("device=%s world_size=%d train=%d val=%d trainable=%d", device, dist.get_world_size() if distributed else 1, len(train_ds), len(val_ds),
                 sum(p.numel() for p in model.parameters() if p.requires_grad))
        if sampling_summary is not None:
            log.info("RNA training sampler=%s", json.dumps(sampling_summary, sort_keys=True))
            atomic_json_write(sampling_summary, out / "sampling_summary.json")
    best_val_loss = float("inf")
    bad_epochs = 0
    completed_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(model, train_loader, optimizer, cfg, True)
        val_metrics = run_epoch(model, val_loader, optimizer, cfg, False)
        if scheduler is not None:
            scheduler.step()
        summary = {"epoch": epoch, "lr": epoch_lr, "train": train_metrics, "val": val_metrics}
        if rank == 0:
            log.info(json.dumps(summary, sort_keys=True))
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
        val_loss = val_metrics.get("loss", float("inf"))
        improved = val_loss < best_val_loss - cfg.min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if rank == 0:
            if cfg.checkpoint_selection == "val_loss" and improved:
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, epoch, val_metrics, cfg,
                        include_optimizer=True,
                    ),
                    out / "best_model.pt",
                )
            if cfg.save_epoch_checkpoints or cfg.checkpoint_selection == "external_retrieval_auprc":
                temp_path = out / f"temp_ckpt_epoch{epoch}.pt"
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, epoch, val_metrics, cfg,
                        include_optimizer=False,
                    ),
                    temp_path,
                )
                log.info("saved epoch checkpoint=%s", temp_path)

        if distributed:
            dist.barrier()
        completed_epoch = epoch
        if (cfg.checkpoint_selection == "val_loss" and cfg.early_stop_patience > 0
                and epoch >= cfg.min_epochs and bad_epochs >= cfg.early_stop_patience):
            if rank == 0:
                log.info("early stopping epoch=%d best_val_loss=%.6f patience=%d",
                         epoch, best_val_loss, cfg.early_stop_patience)
            break
        if cfg.smoke:
            break
    parameter_hash = trainable_parameter_sha256(model)
    rank_hashes = [parameter_hash]
    if distributed:
        rank_hashes = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rank_hashes, parameter_hash)
        if len(set(rank_hashes)) != 1:
            raise RuntimeError(f"DDP trainable parameters diverged across ranks: {rank_hashes}")
    if rank == 0:
        atomic_json_write({
            "status": "complete",
            "completed_epoch": completed_epoch,
            "configured_epochs": cfg.epochs,
            "checkpoint_selection": cfg.checkpoint_selection,
            "best_val_loss": best_val_loss,
            "ddp_trainable_parameter_sha256": parameter_hash,
            "ddp_rank_hashes": rank_hashes,
            "ddp_parameters_bitwise_equal": len(set(rank_hashes)) == 1,
        }, out / "training_complete.json")
        log.info("completed output=%s best_val_loss=%.6f", out, best_val_loss)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
