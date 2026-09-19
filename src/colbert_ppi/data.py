"""Read the small contact datasets and pad tokens without changing residue indices."""

import json
from pathlib import Path


def read_pairs(path):
    records = json.loads(Path(path).read_text())
    for row in records:
        for side in ["left", "right"]:
            ids = row[side + "_tokens"]
            if ids[0] != 0 or ids[-1] != 2:
                raise ValueError("Expected BOS and EOS tokens")
        for field in ["positive", "negative"]:
            for i, j in row[field]:
                if not (
                    0 <= i < len(row["left_tokens"]) - 2
                    and 0 <= j < len(row["right_tokens"]) - 2
                ):
                    raise ValueError("Contact index outside residue range")
    return records


def collate(records, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    left = pad_sequence(
        [torch.tensor(r["left_tokens"]) for r in records],
        batch_first=True,
        padding_value=1,
    ).to(device)
    right = pad_sequence(
        [torch.tensor(r["right_tokens"]) for r in records],
        batch_first=True,
        padding_value=1,
    ).to(device)
    labels = torch.zeros(
        (len(records), left.shape[1] - 2, right.shape[1] - 2),
        dtype=torch.int8,
        device=device,
    )
    for b, row in enumerate(records):
        for field, sign in [("positive", 1), ("negative", -1)]:
            pairs = torch.tensor(row.get(field, []), device=device)
            if pairs.numel() == 0:
                continue
            labels[b, pairs[:, 0], pairs[:, 1]] = sign
    return left, left.ne(1), right, right.ne(1), labels, labels.ne(0)


def read_inference_pairs(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Input must be a nonempty JSON list of pair records")
    seen = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be an object")
        pair_id = row.setdefault("pair_id", f"pair_{index + 1}")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError("pair_id must be a unique nonempty string")
        seen.add(pair_id)
        for side in ["left", "right"]:
            ids = row.get(side + "_tokens")
            if not isinstance(ids, list) or len(ids) < 3 or len(ids) > 1026:
                raise ValueError(f"{pair_id}: {side}_tokens must have 3–1026 tokens")
            if any(type(x) is not int or x < 0 for x in ids):
                raise ValueError(f"{pair_id}: token IDs must be nonnegative integers")
            if ids[0] != 0 or ids[-1] != 2 or any(x in (0, 1, 2) for x in ids[1:-1]):
                raise ValueError(f"{pair_id}: expected BOS=0, residues, EOS=2 without padding")
    return records
