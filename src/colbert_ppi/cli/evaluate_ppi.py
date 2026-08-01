#!/usr/bin/env python3
"""Evaluate one checkpoint with the frozen bidirectional PPI scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from colbert_ppi.cli.eval_worker import (
    build_model_from_args,
    load_test_dataset,
    load_val_dataset,
)
from colbert_ppi.eval_labels import (
    collapse_retrieval_protocol_by_uniprot,
    load_retrieval_label_protocol,
)
from colbert_ppi.retrieval import compute_retrieval_metrics
from colbert_ppi.scoring import canonical_score_dataset
from colbert_ppi.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Evidence-aware PPI label NPZ in the exact dataset order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Base directory used to resolve relative data paths (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    train_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(int(train_args.get("seed", 42)))

    model = build_model_from_args(train_args, device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = (
        load_val_dataset(train_args, args.project_root.resolve())
        if args.split == "val"
        else load_test_dataset(train_args, args.project_root.resolve())
    )
    protocol = load_retrieval_label_protocol(args.protocol, dataset.samples)
    labels = protocol.positive_mask.numpy()

    arrays, payload = canonical_score_dataset(
        model,
        dataset,
        train_args,
        device,
        labels,
        batch_size=args.batch_size,
    )
    payload["checkpoint"] = {
        "epoch": int(checkpoint.get("epoch", 0)),
        "split": args.split,
    }
    record_scores = arrays["record_orientation_mean"]
    record_metrics = compute_retrieval_metrics(
        torch.from_numpy(record_scores).float(),
        positive_mask=protocol.positive_mask,
        candidate_mask=protocol.candidate_mask,
        observed_label_mask=protocol.observed_label_mask,
        verified_negative_mask=protocol.verified_negative_mask,
    )
    entity_scores, entity_protocol, entity_metadata = (
        collapse_retrieval_protocol_by_uniprot(
            record_scores,
            protocol,
            dataset.samples,
            reduction="max",
        )
    )
    entity_metrics = compute_retrieval_metrics(
        torch.from_numpy(entity_scores).float(),
        positive_mask=entity_protocol.positive_mask,
        candidate_mask=entity_protocol.candidate_mask,
        observed_label_mask=entity_protocol.observed_label_mask,
        verified_negative_mask=entity_protocol.verified_negative_mask,
    )
    arrays["uniprot_max_orientation_mean"] = entity_scores.astype("float32")
    payload["label_protocol"] = {
        "version": protocol.protocol_version,
        "source": str(args.protocol),
        "semantics": (
            "Binary AUPRC uses prespecified database-absence operational "
            "negatives; Negatome-supported negatives are a stricter "
            "sensitivity tier, not a requirement for the primary metric."
        ),
    }
    payload["metrics"] = {
        "record": record_metrics,
        "uniprot_max": entity_metrics,
    }
    payload["uniprot_max_metadata"] = entity_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "scores.npz", **arrays)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
