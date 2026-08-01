#!/usr/bin/env python3
"""
Asynchronous evaluation worker — runs on a dedicated GPU (e.g. GPU 4)
in parallel with training on GPUs 0-3.

Listens for eval requests via Unix socket from train.py after each epoch,
loads the temp checkpoint, runs full PPI retrieval evaluation on the
validation set, and writes results back.

Communication protocol (socket-based, 127.0.0.1:29599):
  train → {"action": "eval", "epoch": N}   evaluate epoch N
  eval  → eval_result_epoch{N}.json        results written to disk
  train → {"action": "stop"}               eval worker exits

Results are still written to disk (output_dir) for train.py to discover.

Usage:
  python eval_worker.py --output_dir outputs/run_xxx --eval_gpu 4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from colbert_ppi.config import TrainingConfig
from colbert_ppi.dataset import (
    SaProtContactDataset,
    SaProtLoRAContactDataset,
    SaProtLoRAExplicitContactDataset,
)
from colbert_ppi.eval_labels import (
    RetrievalLabelProtocol,
    collapse_retrieval_by_uniprot,
    collapse_retrieval_protocol_by_uniprot,
    load_positive_mask,
    load_retrieval_label_protocol,
)
from colbert_ppi.retrieval import compute_retrieval_metrics
from colbert_ppi.model import create_model
from colbert_ppi.trainer import run_eval_epoch
from colbert_ppi.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

EVAL_CSV_COLUMNS = [
    "epoch",
    "val_auprc",
    "val_observed_label_auprc",
    "val_strict_binary_auprc",
    "val_strict_binary_auroc",
    "val_mrr",
    "test_auprc",
    "test_observed_label_auprc",
    "test_strict_binary_auprc",
    "test_strict_binary_auroc",
    "test_mrr",
    "val_uniprot_max_auprc",
    "val_uniprot_max_observed_label_auprc",
    "val_uniprot_max_mrr",
    "val_uniprot_max_hit_at_1",
    "val_uniprot_max_hit_at_5",
    "val_uniprot_max_hit_at_10",
    "val_uniprot_mean_auprc",
    "test_uniprot_max_auprc",
    "test_uniprot_max_observed_label_auprc",
    "test_uniprot_max_mrr",
    "test_uniprot_max_hit_at_1",
    "test_uniprot_max_hit_at_5",
    "test_uniprot_max_hit_at_10",
    "test_uniprot_mean_auprc",
    "val_uniprot_unique_positive_edges",
    "test_uniprot_unique_positive_edges",
    "val_contact_res_pair_auprc",
    "val_contact_res_pair_auroc",
    "val_contact_res_auprc",
    "val_contact_res_auroc",
    "test_contact_res_pair_auprc",
    "test_contact_res_pair_auroc",
    "test_contact_res_auprc",
    "test_contact_res_auroc",
]


def add_uniprot_collapse_metrics(
    metrics: dict,
    *,
    stage: str,
    score_output: dict,
    positive_mask: Optional[torch.Tensor],
    label_protocol: Optional[RetrievalLabelProtocol],
    samples,
) -> None:
    """Append UniProt Max/Mean-collapse metrics to an eval result dict."""
    scores = score_output.get("scores")
    if scores is None:
        return
    for reduction in ("max", "mean"):
        if label_protocol is not None:
            collapsed_scores, collapsed_protocol, metadata = (
                collapse_retrieval_protocol_by_uniprot(
                    scores, label_protocol, samples, reduction=reduction
                )
            )
            collapsed_metrics = compute_retrieval_metrics(
                torch.from_numpy(collapsed_scores).float(),
                positive_mask=collapsed_protocol.positive_mask,
                candidate_mask=collapsed_protocol.candidate_mask,
                observed_label_mask=collapsed_protocol.observed_label_mask,
                verified_negative_mask=collapsed_protocol.verified_negative_mask,
            )
        else:
            collapsed_scores, collapsed_positive, metadata = collapse_retrieval_by_uniprot(
                scores, positive_mask, samples, reduction=reduction
            )
            collapsed_metrics = compute_retrieval_metrics(
                torch.from_numpy(collapsed_scores).float(),
                positive_mask=torch.from_numpy(collapsed_positive),
            )
        prefix = f"{stage}_uniprot_{reduction}_"
        for key, value in collapsed_metrics.items():
            metrics[f"{prefix}{key}"] = value
        if reduction == "max":
            metrics[f"{stage}_uniprot_record_rows"] = int(metadata["record_rows"])
            metrics[f"{stage}_uniprot_record_cols"] = int(metadata["record_cols"])
            metrics[f"{stage}_uniprot_unique_receptors"] = int(metadata["unique_receptors"])
            metrics[f"{stage}_uniprot_unique_ligands"] = int(metadata["unique_ligands"])
            metrics[f"{stage}_uniprot_record_positive_pairs"] = int(
                metadata["record_positive_pairs"]
            )
            metrics[f"{stage}_uniprot_unique_positive_edges"] = int(
                metadata["unique_positive_edges"]
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_from_args(args: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the model from the serialised training args dict."""
    model_type = "colbert_lora" if args.get("use_lora", False) else args.get("model_type", "colbert")
    model_kwargs = dict(
        input_dim=args.get("input_dim", 1280),
        ppi_temperature=args.get("ppi_temperature", 0.07),
    )
    if model_type == "colbert_lora":
        model_kwargs.update(
            saprot_dir=args["saprot_dir"],
            lora_r=args.get("lora_r", 8),
            lora_alpha=args.get("lora_alpha", 8),
            lora_dropout=args.get("lora_dropout", 0.1),
            hidden_dim=args.get("hidden_dim", 256),
            dropout=args.get("dropout", 0.1),
            sequence_only=args.get("sequence_only", False),
            gradient_checkpointing=args.get("gradient_checkpointing", False),
        )
    else:
        model_kwargs.update(
            hidden_dim=args.get("hidden_dim", 256),
            dropout=args.get("dropout", 0.1),
        )
    model = create_model(model_type, **model_kwargs).to(device)
    return model


def load_val_dataset(args: dict, v8_root: Path):
    """Load the full (non-distributed) validation dataset."""
    data_dir = v8_root / args.get("data_dir", "data")
    DatasetClass = SaProtLoRAContactDataset if args.get("use_lora", False) else SaProtContactDataset

    if args.get("val_contact_map"):
        if not args.get("use_lora", False):
            raise ValueError("Explicit contact maps currently require tokenized LoRA inputs")
        return SaProtLoRAExplicitContactDataset(
            csv_file=args["val_csv"],
            saprot_inputs_pt=args.get("val_saprot_inputs", ""),
            cb_npz=args["val_cb"],
            contact_map_pt=args["val_contact_map"],
            max_seq_len=args.get("max_seq_len", None),
            build_contacts=True,
        )

    dataset_kwargs = dict(
        csv_file=args["val_csv"],
        cb_npz=args["val_cb"],
        processed_dir=data_dir,
        processed_prefix=Path(args["val_csv"]).stem,
        max_seq_len=args.get("max_seq_len", None),
        build_contacts=True,
    )
    if args.get("use_lora", False):
        dataset_kwargs["saprot_inputs_pt"] = args.get("val_saprot_inputs", "")
    else:
        dataset_kwargs["saprot_pt"] = args.get("val_saprot", "")
    dataset = DatasetClass(**dataset_kwargs)
    return dataset


def load_test_dataset(args: dict, v8_root: Path):
    """Load the full (non-distributed) test dataset."""
    data_dir = v8_root / args.get("data_dir", "data")
    DatasetClass = SaProtLoRAContactDataset if args.get("use_lora", False) else SaProtContactDataset

    if args.get("test_contact_map"):
        if not args.get("use_lora", False):
            raise ValueError("Explicit contact maps currently require tokenized LoRA inputs")
        return SaProtLoRAExplicitContactDataset(
            csv_file=args["test_csv"],
            saprot_inputs_pt=args.get("test_saprot_inputs", ""),
            cb_npz=args["test_cb"],
            contact_map_pt=args["test_contact_map"],
            max_seq_len=args.get("max_seq_len", None),
            build_contacts=True,
        )

    dataset_kwargs = dict(
        csv_file=args["test_csv"],
        cb_npz=args["test_cb"],
        processed_dir=data_dir,
        processed_prefix=Path(args["test_csv"]).stem,
        max_seq_len=args.get("max_seq_len", None),
        build_contacts=True,
    )
    if args.get("use_lora", False):
        dataset_kwargs["saprot_inputs_pt"] = args.get("test_saprot_inputs", "")
    else:
        dataset_kwargs["saprot_pt"] = args.get("test_saprot", "")
    dataset = DatasetClass(**dataset_kwargs)
    return dataset


def append_eval_metrics_csv(
    csv_path: Path,
    *,
    epoch: int,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    """Append one epoch of selected eval metrics to a persistent CSV table."""
    row = {"epoch": epoch}
    merged = {}
    merged.update(val_metrics)
    merged.update(test_metrics)
    for col in EVAL_CSV_COLUMNS:
        if col == "epoch":
            continue
        row[col] = merged.get(col, "")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVAL_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval_worker(output_dir: Path, device: torch.device, eval_port: int = 29599) -> None:
    """Main loop: listen on socket, evaluate, write results."""

    # --- Load training config ---
    args_path = output_dir / "args.json"
    if not args_path.exists():
        logger.error(f"args.json not found in {output_dir}")
        sys.exit(1)
    with args_path.open("r") as f:
        train_args = json.load(f)
    logger.info(f"Loaded training config from {args_path}")

    v8_root = Path(__file__).resolve().parent.parent

    # --- Build model ---
    logger.info("Building model for evaluation...")
    model = build_model_from_args(train_args, device)
    model.eval()

    # --- Load validation dataset ---
    logger.info("Loading validation dataset...")
    val_dataset = load_val_dataset(train_args, v8_root)
    logger.info(f"Val dataset: {len(val_dataset)} pairs")
    val_positive_mask = None
    val_label_protocol = None
    if train_args.get("val_label_protocol"):
        val_label_protocol = load_retrieval_label_protocol(
            train_args["val_label_protocol"], val_dataset.samples
        )
        val_positive_mask = val_label_protocol.positive_mask
        logger.info(
            "Val evidence protocol %s: positive=%d verified_negative=%d candidate=%d",
            val_label_protocol.protocol_version,
            int(val_label_protocol.positive_mask.sum()),
            int(val_label_protocol.verified_negative_mask.sum()),
            int(val_label_protocol.candidate_mask.sum()),
        )
    elif train_args.get("val_calibrated_pairs"):
        val_positive_mask = load_positive_mask(train_args["val_calibrated_pairs"], val_dataset.samples)
        logger.info(
            f"Val calibrated positive mask: {int(val_positive_mask.sum().item())} positives "
            f"from {train_args['val_calibrated_pairs']}"
        )
    val_score_mask1 = None
    if train_args.get("val_score_mask1"):
        val_score_mask1 = torch.load(train_args["val_score_mask1"], map_location="cpu", weights_only=False)
        logger.info(f"Val side-1 scoring masks: {len(val_score_mask1)} labels")

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_args.get("batch_size", 8),
        shuffle=False,
        num_workers=0,  # MUST be 0: CUDA already initialized, fork would deadlock
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    # A clean validation-only training run must not even load the test split.
    # The default remains True for backwards compatibility with historical runs;
    # new confirmatory/matched runs set eval_test_each_epoch=False explicitly.
    run_test_each_epoch = bool(train_args.get("eval_test_each_epoch", True))
    test_dataset = None
    test_loader = None
    test_positive_mask = None
    test_label_protocol = None
    test_score_mask1 = None
    if run_test_each_epoch:
        logger.info("Loading test dataset...")
        test_dataset = load_test_dataset(train_args, v8_root)
        logger.info(f"Test dataset: {len(test_dataset)} pairs")
        if train_args.get("test_label_protocol"):
            test_label_protocol = load_retrieval_label_protocol(
                train_args["test_label_protocol"], test_dataset.samples
            )
            test_positive_mask = test_label_protocol.positive_mask
            logger.info(
                "Test evidence protocol %s: positive=%d verified_negative=%d candidate=%d",
                test_label_protocol.protocol_version,
                int(test_label_protocol.positive_mask.sum()),
                int(test_label_protocol.verified_negative_mask.sum()),
                int(test_label_protocol.candidate_mask.sum()),
            )
        elif train_args.get("test_calibrated_pairs"):
            test_positive_mask = load_positive_mask(
                train_args["test_calibrated_pairs"], test_dataset.samples
            )
            logger.info(
                f"Test calibrated positive mask: {int(test_positive_mask.sum().item())} positives "
                f"from {train_args['test_calibrated_pairs']}"
            )
        if train_args.get("test_score_mask1"):
            test_score_mask1 = torch.load(
                train_args["test_score_mask1"], map_location="cpu", weights_only=False
            )
            logger.info(f"Test side-1 scoring masks: {len(test_score_mask1)} labels")
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_args.get("batch_size", 8),
            shuffle=False,
            num_workers=0,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=False,
        )
    else:
        logger.info(
            "Validation-only firewall active: the test dataset will not be loaded or "
            "evaluated during checkpoint selection."
        )

    processed_epochs: set[int] = set()
    best_val_auprc = 0.0
    eval_csv_path = output_dir / "eval_metrics.csv"

    # --- Create socket server ---
    HOST = "127.0.0.1"
    PORT = eval_port
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    logger.info(f"Eval worker ready, listening on {HOST}:{PORT} for signals...")

    while True:
        # --- Accept connection (blocking — no polling) ---
        try:
            conn, addr = server_sock.accept()
        except Exception:
            break

        epoch = None
        try:
            data = conn.recv(4096)
            conn.close()
            if not data:
                continue

            msg = json.loads(data.decode("utf-8"))
            action = msg.get("action", "")

            if action == "stop":
                logger.info("Received stop signal, exiting.")
                break

            if action != "eval":
                logger.warning(f"Unknown action '{action}', ignoring.")
                continue

            epoch = int(msg.get("epoch", -1))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid message received: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing socket message: {e}")
            continue

        if epoch is None or epoch < 0:
            continue

        # Skip if already processed (e.g. duplicate signal)
        if epoch in processed_epochs:
            continue

        logger.info(f"Processing eval request for epoch {epoch}...")

        # Skip if result already exists (e.g. worker restart)
        result_path = output_dir / f"eval_result_epoch{epoch}.json"
        if result_path.exists():
            logger.info(f"Result for epoch {epoch} already exists, skipping.")
            processed_epochs.add(epoch)
            continue

        # Load temp checkpoint
        ckpt_path = output_dir / f"temp_ckpt_epoch{epoch}.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint {ckpt_path} not found, skipping epoch {epoch}")
            processed_epochs.add(epoch)
            continue

        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            processed_epochs.add(epoch)
            continue

        # --- Run validation evaluation ---
        logger.info(f"Running val evaluation for epoch {epoch}...")
        eval_start = time.time()

        val_score_output: dict = {}
        val_metrics = run_eval_epoch(
            model, val_loader,
            pos_per_sample=train_args.get("pos_per_sample", 5),
            neg_per_sample=train_args.get("neg_per_sample", 5),
            device=device,
            bf16=train_args.get("bf16", train_args.get("fp16", False)),
            epoch=epoch,
            stage="val",
            show_progress=True,
            positive_mask=val_positive_mask,
            candidate_mask=(
                val_label_protocol.candidate_mask if val_label_protocol else None
            ),
            observed_label_mask=(
                val_label_protocol.observed_label_mask if val_label_protocol else None
            ),
            verified_negative_mask=(
                val_label_protocol.verified_negative_mask if val_label_protocol else None
            ),
            score_mask1_by_label=val_score_mask1,
            score_output=val_score_output,
        )
        add_uniprot_collapse_metrics(
            val_metrics,
            stage="val",
            score_output=val_score_output,
            positive_mask=val_positive_mask,
            label_protocol=val_label_protocol,
            samples=val_dataset.samples,
        )

        # --- Optional legacy test monitoring ---
        test_metrics: dict = {}
        if run_test_each_epoch:
            logger.info(f"Running test evaluation for epoch {epoch}...")
            test_score_output: dict = {}
            test_metrics = run_eval_epoch(
                model, test_loader,
                pos_per_sample=train_args.get("pos_per_sample", 5),
                neg_per_sample=train_args.get("neg_per_sample", 5),
                device=device,
                bf16=train_args.get("bf16", train_args.get("fp16", False)),
                epoch=epoch,
                stage="test",
                show_progress=True,
                positive_mask=test_positive_mask,
                candidate_mask=(
                    test_label_protocol.candidate_mask if test_label_protocol else None
                ),
                observed_label_mask=(
                    test_label_protocol.observed_label_mask if test_label_protocol else None
                ),
                verified_negative_mask=(
                    test_label_protocol.verified_negative_mask
                    if test_label_protocol else None
                ),
                score_mask1_by_label=test_score_mask1,
                score_output=test_score_output,
            )
            add_uniprot_collapse_metrics(
                test_metrics,
                stage="test",
                score_output=test_score_output,
                positive_mask=test_positive_mask,
                label_protocol=test_label_protocol,
                samples=test_dataset.samples,
            )

        eval_elapsed = time.time() - eval_start
        logger.info(f"Evaluation completed in {eval_elapsed:.1f}s")

        # --- Write result (val + test metrics for train.py handoff) ---
        result_path = output_dir / f"eval_result_epoch{epoch}.json"
        with result_path.open("w") as f:
            json.dump({
                "epoch": epoch,
                "metrics": val_metrics,          # val metrics (for train.py best-ckpt selection)
                "test_metrics": test_metrics,     # test metrics (for monitoring)
                "eval_time_s": eval_elapsed,
            }, f, indent=2, default=str)
        logger.info(f"Results written to {result_path}")

        append_eval_metrics_csv(
            eval_csv_path,
            epoch=epoch,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        logger.info(f"Selected metrics appended to {eval_csv_path}")

        # Track the preregistered validation metric for logging.
        selection_metric = train_args.get("eval_selection_metric", "val_auprc")
        selection_value = val_metrics.get(selection_metric)
        if not isinstance(selection_value, (int, float)):
            raise RuntimeError(
                f"Selection metric {selection_metric!r} is unavailable: "
                f"{selection_value!r}"
            )
        val_auprc = float(selection_value)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            message = (
                f"  >> New best {selection_metric}={val_auprc:.4f} "
                f"(val_auprc={val_auprc:.4f})"
            )
        else:
            message = (
                f"  Val {selection_metric}={val_auprc:.4f} "
                f"(best={best_val_auprc:.4f})"
            )
        if run_test_each_epoch:
            message += (
                f", test_auprc={test_metrics.get('test_auprc', 0.0):.4f}"
            )
        logger.info(message)

        # Mark as processed
        processed_epochs.add(epoch)

        # Clean up old epochs from processed set (keep last 5 for safety)
        if len(processed_epochs) > 10:
            processed_epochs = set(sorted(processed_epochs)[-5:])

    server_sock.close()
    logger.info("Eval worker exiting.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Async eval worker for ColBERT PPI")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Training run output directory")
    parser.add_argument("--eval_gpu", type=int, default=4,
                        help="GPU device id for evaluation (default: 4)")
    parser.add_argument("--eval_port", type=int, default=29599,
                        help="Socket port for train↔eval communication (default: 29599)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: output_dir {output_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # GPU is already isolated via CUDA_VISIBLE_DEVICES in the parent process.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    # Set up logging
    log_file = output_dir / "eval_worker.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    set_seed(42)
    logger.info(f"Eval worker starting — output_dir={output_dir}, device={device}")
    run_eval_worker(output_dir, device, eval_port=args.eval_port)


if __name__ == "__main__":
    main()
