"""Load multi-vector model components and score tokenized PPI or PRI pairs."""

from pathlib import Path
import json
import numpy as np
import torch

from colbert_ppi.models.loading import build_model, restore_components
from colbert_ppi.inference import encode_pairs
from colbert_ppi.scoring import score_protein_pair, score_protein_rna
from colbert_ppi.data import read_inference_pairs


def run(args):
    try:
        records = read_inference_pairs(args.input)
        if not Path(args.checkpoint).is_file():
            raise ValueError("Checkpoint missing; download components as described in README.md")
        if args.task == "ppi" and (not args.reference_bank or not Path(args.reference_bank).is_file()):
            raise ValueError("PPI requires the reference bank distributed with the selected checkpoint")
        cfg = Path(args.checkpoint).with_name("config.json")
        if cfg.is_file():
            settings = json.loads(cfg.read_text())
            if settings.get("sequence_only") or settings.get("representation") == "single_vector":
                raise ValueError("Pair inference supports complete multi-vector models; use the matching multi-vector checkpoint")
            if settings.get("task", args.task) != args.task:
                raise ValueError("Checkpoint task does not match --task")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; check the NVIDIA driver, PyTorch CUDA build and selected GPU")
        banks = dict(np.load(args.reference_bank)) if args.reference_bank else None
        if args.task == "ppi" and not {"query", "candidate"}.issubset(banks):
            raise ValueError("Reference bank must contain query and candidate arrays")
    except (ValueError, OSError) as error:
        raise ValueError(str(error)) from error
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(
        args.task, args.saprot_dir, args.ernie_checkpoint, args.ernie_code
    ).to(device)
    restore_components(model, args.checkpoint)
    features = encode_pairs(model, records, args.task, device)
    results = []
    for row, x in zip(records, features):
        if args.task == "ppi":
            if banks is None:
                raise ValueError(
                    "PPI inference requires the bank saved with the selected checkpoint"
                )
            score = score_protein_pair(
                x["left_query"],
                x["left_candidate"],
                x["right_query"],
                x["right_candidate"],
                banks["query"],
                banks["candidate"],
            )["score"]
        else:
            score = score_protein_rna(x["left"], x["right"])
        results.append({"pair_id": row.get("pair_id", "input"), "score": score})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
