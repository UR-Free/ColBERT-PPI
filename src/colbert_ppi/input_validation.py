"""Validate label-free token records before loading large neural models."""

import json
from pathlib import Path


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
