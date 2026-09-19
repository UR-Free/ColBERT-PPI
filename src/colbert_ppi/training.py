"""Train on contacts, select an epoch on validation, then evaluate the holdout."""

from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from .data import read_pairs, collate
from .losses import sampled_contact_losses
from .models.loading import (
    build_model,
    component_state,
    restore_components,
    initialise_protein_branch,
)
from .inference import encode_pairs, training_bank, retrieval_metrics


def train(args):
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30)
    dataset = Path(args.data_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training = read_pairs(dataset / "train.json")
    validation = read_pairs(dataset / "validation.json")
    model = build_model(
        args.task, args.saprot_dir, args.ernie_checkpoint, args.ernie_code
    ).to(device)
    transferred = (
        initialise_protein_branch(model, args.protein_init) if args.protein_init else 0
    )
    initial = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and "lora_B" in name
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-6,
    )
    history = []
    best = -float("inf")
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(training))
        losses = []
        for start_index in range(0, len(order), args.batch_size):
            rows = [
                training[i] for i in order[start_index : start_index + args.batch_size]
            ]
            left, lm, right, rm, contacts, observed = collate(rows, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                p, n, pm, nm, pa, na, scale = model(left, lm, right, rm)
                loss = sampled_contact_losses(
                    p,
                    n,
                    contacts,
                    observed,
                    p_attention=pa,
                    n_attention=na,
                    inv_temperature=scale,
                )["contact_loss"]
                if args.task == "ppi":
                    loss = loss + 0.01 * (
                        (pa * (1 - pa))[pm].mean() + (na * (1 - na))[nm].mean()
                    )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        banks = (
            training_bank(encode_pairs(model, training, args.task, device))
            if args.task == "ppi"
            else None
        )
        metrics, _, _ = retrieval_metrics(
            encode_pairs(model, validation, args.task, device),
            validation,
            args.task,
            banks,
        )
        entry = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_auprc": metrics["auprc"],
        }
        history.append(entry)
        print(entry, flush=True)
        if metrics["auprc"] > best:
            best = metrics["auprc"]
            torch.save(
                {"model": component_state(model), "epoch": epoch, "task": args.task},
                output / "best.pt",
            )
            if banks:
                np.savez_compressed(output / "reference_bank.npz", **banks)
    changed = any(
        not torch.equal(initial[name], p.detach().cpu())
        for name, p in model.named_parameters()
        if name in initial
    )
    if not changed:
        raise RuntimeError("Training did not update the SaProt LoRA parameters")
    selected = restore_components(model, output / "best.pt")
    # Test records are first read after selection and are never used to choose an epoch.
    test = read_pairs(dataset / "test.json")
    banks = dict(np.load(output / "reference_bank.npz")) if args.task == "ppi" else None
    encoded = encode_pairs(model, test, args.task, device)
    metrics, scores, truth = retrieval_metrics(encoded, test, args.task, banks)
    np.savez_compressed(output / "test_predictions.npz", scores=scores, labels=truth)
    report = {
        "task": args.task,
        "training_pairs": len(training),
        "validation_pairs": len(validation),
        "test_pairs": len(test),
        "epochs": args.epochs,
        "optimizer_steps": steps,
        "selected_epoch": selected["epoch"],
        "lora_updated": changed,
        "protein_transfer_tensors": transferred,
        "test": metrics,
        "elapsed_seconds": time.perf_counter() - start,
        "purpose": "Small-data workflow demonstration, not manuscript benchmark performance",
    }
    (output / "training_history.json").write_text(json.dumps(history, indent=2))
    (output / "run_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
