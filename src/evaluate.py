"""Evaluate a fixed checkpoint on a specified labelled mini dataset."""

from pathlib import Path
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colbert_ppi.model_loading import build_model, restore_components
from colbert_ppi.inference import encode_pairs, retrieval_metrics
from colbert_ppi.training_data import read_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ppi", "pri"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--saprot-dir", required=True)
    parser.add_argument("--ernie-checkpoint")
    parser.add_argument("--ernie-code")
    parser.add_argument("--reference-bank")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(
        args.task, args.saprot_dir, args.ernie_checkpoint, args.ernie_code
    ).to(device)
    restore_components(model, args.checkpoint)
    rows = read_pairs(args.data)
    encoded = encode_pairs(model, rows, args.task, device)
    banks = dict(np.load(args.reference_bank)) if args.reference_bank else None
    if args.task == "ppi" and banks is None:
        raise ValueError(
            "Supply the training-only bank saved with the selected checkpoint"
        )
    metrics, scores, truth = retrieval_metrics(encoded, rows, args.task, banks)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", scores=scores, labels=truth)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
