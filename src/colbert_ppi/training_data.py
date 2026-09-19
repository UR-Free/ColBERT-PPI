"""Read the small contact datasets and pad tokens without changing residue indices."""

import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence


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
