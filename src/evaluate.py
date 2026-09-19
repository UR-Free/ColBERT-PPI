"""Run fixed-weight retrieval on the supplied full benchmark inputs and masks."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from colbert_ppi.models.loading import build_model, restore_components
from colbert_ppi.scoring import reference_background, smooth_maxsim


def metrics(labels, scores):
    return dict(
        auprc=float(average_precision_score(labels, scores)),
        auroc=float(roc_auc_score(labels, scores)),
        positives=int(np.sum(labels)),
        pairs=len(labels),
    )


def pooled(vectors, attention):
    return F.normalize((vectors * attention[:, None]).sum(0), dim=0)


def collapse(scores, left, right):
    return np.array([[scores[np.ix_(a, b)].max() for b in right] for a in left])


def cohort_metrics(scores, labels):
    report = {}
    for key in labels.files:
        if key == "positive":
            name, positive, negative = "validation", labels[key], labels["negative"]
        elif key.endswith("_positive"):
            name = key[:-9]
            positive, negative = labels[key], labels[name + "_negative"]
        else:
            continue
        assert scores.shape == positive.shape and not (positive & negative).any()
        mask = positive | negative
        report[name] = metrics(positive[mask], scores[mask])
    return report


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.root
    data = root / "data/benchmarks" / args.task
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.scores:
        if args.task != "pri":
            raise ValueError(
                "--scores accepts a PRI exact-entity NPZ; PPI/Y2H use --checkpoint"
            )
        scores = np.load(args.scores)["scores"]
    else:
        if not args.checkpoint or not args.saprot_dir:
            raise ValueError("Supply --checkpoint and --saprot-dir")
        cfg_path = Path(args.checkpoint).with_name("config.json")
        config = json.loads(cfg_path.read_text())
        single = config.get("representation") == "single_vector"
        sequence = bool(config.get("sequence_only", False))
        device = torch.device(args.device)
        model = (
            build_model(
                "pri" if args.task == "pri" else "ppi",
                args.saprot_dir,
                args.ernie_checkpoint,
                args.ernie_code,
            )
            .to(device)
            .eval()
        )
        restore_components(model, args.checkpoint)
        scale = float(model.log_inv_temperature.exp().clamp(0.05, 20))
        mapping = None
        if sequence:
            from transformers import EsmTokenizer

            vocab = EsmTokenizer.from_pretrained(args.saprot_dir).get_vocab()
            mapping = torch.arange(max(vocab.values()) + 1, device=device)
            for token, index in vocab.items():
                if len(token) == 2 and token[0] + "#" in vocab:
                    mapping[index] = vocab[token[0] + "#"]
        inputs = torch.load(
            data / f"{args.split}_inputs.pt", map_location="cpu", weights_only=True
        )

        def tensor(ids):
            return torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        if args.task == "pri":
            cache = []
            for i, row in enumerate(inputs):
                a, b = tensor(row["protein_input_ids"]), tensor(
                    row["nucleic_input_ids"]
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    p, n, pm, nm, pa, na, _ = model(a, a.ne(1), b, b.ne(1))
                p, n = p[0, pm[0]].float(), n[0, nm[0]].float()
                if single:
                    p, n = pooled(p, pa[0, pm[0]].float()), pooled(
                        n, na[0, nm[0]].float()
                    )
                cache.append((p.cpu().numpy(), n.cpu().numpy()))
                if i % 50 == 0:
                    print(f"Encoded {i + 1}/{len(inputs)} records", flush=True)
            scores = np.empty((len(cache), len(cache)), dtype=np.float32)
            for i, (p, _) in enumerate(cache):
                for j, (_, n) in enumerate(cache):
                    scores[i, j] = (
                        np.clip(p @ n * scale, -100, 100)
                        if single
                        else smooth_maxsim(p @ n.T, 0.001)
                    )
            groups = json.loads(
                (
                    data
                    / (
                        "duplicate_groups.json"
                        if args.split == "test"
                        else "validation_groups.json"
                    )
                ).read_text()
            )
            scores = collapse(scores, groups["protein_groups"], groups["rna_groups"])
        else:
            banks = None
            if not single:
                bp = (
                    Path(args.reference_bank)
                    if args.reference_bank
                    else Path(args.checkpoint).with_name("reference_bank.npz")
                )
                banks = np.load(bp)
            cache = {}
            for i, (key, row) in enumerate(inputs.items()):
                ids = tensor(row["input_ids"])
                if mapping is not None:
                    ids = mapping[ids]
                entry = []
                for role in [0, 1]:
                    v, mask, attn = model.encode(ids, ids.ne(1), role)
                    v, attn = v[0, mask[0]], attn[0, mask[0]]
                    if single:
                        entry.append((pooled(v, attn).cpu().numpy(), None))
                    else:
                        a = v.cpu().numpy()
                        bg = reference_background(
                            a, banks["candidate" if role == 0 else "query"], 0.03
                        )
                        entry.append((a, bg))
                cache[key] = entry
                if i % 50 == 0:
                    print(f"Encoded {i + 1}/{len(inputs)} proteins", flush=True)

            def score(a, b):
                (aq, aqb), (ac, acb) = cache[a]
                (bq, bqb), (bc, bcb) = cache[b]
                if single:
                    return float(np.clip(0.5 * (aq @ bc + bq @ ac) * scale, -100, 100))
                ab = aq @ bc.T - 0.5 * (aqb[:, None] + bcb[None, :])
                ba = bq @ ac.T - 0.5 * (bqb[:, None] + acb[None, :])
                return 0.5 * (smooth_maxsim(ab, 0.03) + smooth_maxsim(ba, 0.03))

            if args.task == "y2h":
                if args.split != "test":
                    raise ValueError("Y2H is an independent test screen")
                pairs = pd.read_csv(data / "test_pairs.csv")
                pairs["prediction"] = [
                    score(a, b) for a, b in zip(pairs.protein_a, pairs.protein_b)
                ]
            else:
                order = pd.read_csv(data / f"{args.split}_input_order.csv")
                labels = np.load(data / f"{args.split}_labels.npz")
                aliases = dict(zip(order.left_id, order.left_input_key)) | dict(
                    zip(order.right_id, order.right_input_key)
                )
                rows = []
                for i, a in enumerate(labels["receptor_labels"]):
                    for j, b in enumerate(labels["ligand_labels"]):
                        if not (
                            labels["positive_mask"][i, j]
                            or labels["operational_negative_mask"][i, j]
                        ):
                            continue
                        u, v = sorted(
                            [
                                str(labels["receptor_accessions"][i]),
                                str(labels["ligand_accessions"][j]),
                            ]
                        )
                        rows.append(
                            (
                                u,
                                v,
                                int(labels["positive_mask"][i, j]),
                                score(aliases[a], aliases[b]),
                            )
                        )
                pairs = pd.DataFrame(
                    rows, columns=["protein_a", "protein_b", "label", "prediction"]
                )
                pairs = pairs.groupby(["protein_a", "protein_b"], as_index=False).agg(
                    label=("label", "max"), prediction=("prediction", "max")
                )
            pairs.to_csv(output / "pair_scores.csv", index=False)
            result = metrics(pairs.label.values, pairs.prediction.values)
            (output / "metrics.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return
    labels = np.load(
        data / ("labels.npz" if args.split == "test" else "validation_labels.npz")
    )
    result = cohort_metrics(scores, labels)
    np.savez_compressed(output / "scores.npz", scores=scores)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    from colbert_ppi.options import run_script

    run_script("evaluate", run)
