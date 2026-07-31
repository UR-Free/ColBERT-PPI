#!/usr/bin/env python3
"""Evaluate one checkpoint with the frozen bidirectional PPI scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from colbert_ppi.cli.eval_worker import (
    build_model_from_args,
    load_test_dataset,
    load_val_dataset,
)
from colbert_ppi.scoring import canonical_score_dataset
from colbert_ppi.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-key", default="labels")
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
    label_archive = np.load(args.labels, allow_pickle=False)
    labels = np.asarray(label_archive[args.label_key], dtype=bool)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "scores.npz", **arrays)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
