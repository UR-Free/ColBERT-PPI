#!/usr/bin/env python3
"""Evaluate any PPI score matrix with the evidence-aware PINDER protocol.

This model-agnostic evaluator is the common endpoint for ColBERT-PPI,
FlashPPI, and other baselines.  Every method must supply scores in the same
record order; labels, candidate eligibility, censoring, and entity collapse
are then identical across methods.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


from colbert_ppi.eval_labels import (
    collapse_retrieval_protocol_by_uniprot,
    load_retrieval_label_protocol,
    normalize_pair_label,
)
from colbert_ppi.retrieval import compute_retrieval_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PAIRID" not in reader.fieldnames:
            raise ValueError(f"{path} must contain PAIRID")
        for row in reader:
            left, right = row["PAIRID"].split(":", 1)
            samples.append((normalize_pair_label(left), normalize_pair_label(right)))
    return samples


def read_scores(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    payload = np.load(path, allow_pickle=False)
    if key:
        if key not in payload.files:
            raise KeyError(f"{key!r} not found in {path}; keys={payload.files}")
        return np.asarray(payload[key], dtype=np.float32)
    if len(payload.files) != 1:
        raise ValueError(f"--score-key is required for multi-array NPZ: {payload.files}")
    return np.asarray(payload[payload.files[0]], dtype=np.float32)


def metrics(scores: np.ndarray, protocol) -> dict[str, float | None]:
    return compute_retrieval_metrics(
        torch.from_numpy(scores).float(),
        positive_mask=protocol.positive_mask,
        candidate_mask=protocol.candidate_mask,
        observed_label_mask=protocol.observed_label_mask,
        verified_negative_mask=protocol.verified_negative_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        required=True,
        help="Square record-level .npy or .npz score matrix",
    )
    parser.add_argument("--score-key", default="", help="Array key for an NPZ input")
    parser.add_argument("--pairs", required=True, help="PAIRID CSV in score-matrix order")
    parser.add_argument("--protocol", required=True, help="Evidence-label NPZ")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--entity-reduction", choices=("max", "mean"), default="max")
    args = parser.parse_args()

    score_path = Path(args.scores)
    pair_path = Path(args.pairs)
    protocol_path = Path(args.protocol)
    output_path = Path(args.output)
    samples = read_samples(pair_path)
    scores = read_scores(score_path, args.score_key)
    expected = (len(samples), len(samples))
    if scores.shape != expected:
        raise ValueError(f"Score shape {scores.shape} does not match pairs {expected}")
    if not np.isfinite(scores).all():
        raise ValueError("Score matrix contains NaN or infinity")

    protocol = load_retrieval_label_protocol(protocol_path, samples)
    record_metrics = metrics(scores, protocol)
    entity_scores, entity_protocol, entity_metadata = (
        collapse_retrieval_protocol_by_uniprot(
            scores,
            protocol,
            samples,
            reduction=args.entity_reduction,
        )
    )
    entity_metrics = metrics(entity_scores, entity_protocol)
    output = {
        "protocol_version": protocol.protocol_version,
        "semantics": {
            "primary": "entity MRR and bidirectional Hit@1/5/10/20",
            "observed_label_auprc": "positive-versus-unlabelled sensitivity",
            "strict_binary": "positives versus verified negative evidence only",
        },
        "inputs": {
            "scores": str(score_path),
            "scores_sha256": sha256(score_path),
            "score_key": args.score_key,
            "pairs": str(pair_path),
            "pairs_sha256": sha256(pair_path),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
        },
        "record_metrics": record_metrics,
        "entity_reduction": args.entity_reduction,
        "entity_metrics": entity_metrics,
        "entity_metadata": entity_metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
