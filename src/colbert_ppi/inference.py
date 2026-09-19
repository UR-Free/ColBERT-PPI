"""Encode prepared pairs and evaluate the small demonstration retrieval grid."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import collate
from .scoring import score_protein_pair, score_protein_rna


@torch.inference_mode()
def encode_pairs(model, records, task, device):
    model.eval()
    encoded = []
    for row in records:
        left, lm, right, rm, _, _ = collate([row], device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            if task == "ppi":
                parts = {}
                for side, ids, mask in [("left", left, lm), ("right", right, rm)]:
                    for role, name in [(0, "query"), (1, "candidate")]:
                        v, valid, _ = model.encode(ids, mask, role)
                        parts[side + "_" + name] = v[0, valid[0]].float().cpu().numpy()
            else:
                p, n, pm, nm, *_ = model(left, lm, right, rm)
                parts = {
                    "left": p[0, pm[0]].float().cpu().numpy(),
                    "right": n[0, nm[0]].float().cpu().numpy(),
                }
        encoded.append(parts)
    return encoded


def training_bank(encoded):
    """Use training pairs only; sample up to 16 positions per monomer and role."""
    banks = {}
    for role in ["query", "candidate"]:
        vectors = []
        for pair in encoded:
            for side in ["left", "right"]:
                v = pair[side + "_" + role]
                positions = np.linspace(0, len(v) - 1, min(16, len(v)), dtype=int)
                vectors.append(v[positions])
        banks[role] = np.concatenate(vectors)
    return banks


def retrieval_metrics(encoded, records, task, banks=None):
    scores = np.empty((len(records), len(records)))
    left_keys = [tuple(r["left_tokens"]) for r in records]
    right_keys = [tuple(r["right_tokens"]) for r in records]
    known = set(zip(left_keys, right_keys))
    truth = np.array([[(a, b) in known for b in right_keys] for a in left_keys])
    for i, a in enumerate(encoded):
        for j, b in enumerate(encoded):
            if task == "ppi":
                scores[i, j] = score_protein_pair(
                    a["left_query"],
                    a["left_candidate"],
                    b["right_query"],
                    b["right_candidate"],
                    banks["query"],
                    banks["candidate"],
                )["score"]
            else:
                scores[i, j] = score_protein_rna(a["left"], b["right"])
    return (
        {
            "auprc": float(average_precision_score(truth.ravel(), scores.ravel())),
            "auroc": float(roc_auc_score(truth.ravel(), scores.ravel())),
            "positive_pairs": int(truth.sum()),
            "negative_pairs": int((~truth).sum()),
        },
        scores,
        truth,
    )
