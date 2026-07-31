from __future__ import annotations

import csv

import numpy as np
import torch

from colbert_ppi.dataset import SaProtLoRAExplicitContactDataset


def test_window_records_load_exact_inputs_and_expose_parent_labels(tmp_path) -> None:
    record_a = "parent_A__window_000000_000003"
    record_b = "parent_B__window_000000_000003"
    record_id = f"{record_a}:{record_b}"
    input_a = "structure_A__window_000000_000003"
    input_b = "structure_B__window_000000_000003"

    pair_csv = tmp_path / "pairs.csv"
    with pair_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PAIRID"])
        writer.writerow([record_id])

    inputs = {
        key: {
            "input_ids": torch.tensor([[0, 4, 5, 6, 2]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        }
        for key in (input_a, input_b)
    }
    inputs_path = tmp_path / "inputs.pt"
    torch.save(inputs, inputs_path)

    coordinates_path = tmp_path / "coordinates.npz"
    np.savez_compressed(
        coordinates_path,
        labels=np.asarray([input_a, input_b]),
        offsets=np.asarray([0, 3, 6]),
        coords=np.zeros((6, 3), dtype=np.float32),
    )

    contacts_path = tmp_path / "contacts.pt"
    torch.save(
        {
            "format": "colbert_ppi_windowed_explicit_v1",
            "contacts": {
                record_id: {
                    "pos_i": torch.tensor([0]),
                    "pos_j": torch.tensor([1]),
                    "neg_i": torch.tensor([2]),
                    "neg_j": torch.tensor([2]),
                    "input_key1": input_a,
                    "input_key2": input_b,
                    "parent_label1": "parent_A",
                    "parent_label2": "parent_B",
                    "scoring_only": False,
                }
            },
        },
        contacts_path,
    )

    dataset = SaProtLoRAExplicitContactDataset(
        pair_csv,
        inputs_path,
        coordinates_path,
        contacts_path,
        max_seq_len=514,
        build_contacts=True,
    )

    assert dataset.samples == [("parent_A", "parent_B")]
    item = dataset[0]
    assert item["pair_id"] == record_id
    assert item["afdb_label_a"] == input_a
    assert item["afdb_label_b"] == input_b
    assert item["contact_matrix"].shape == (3, 3)
    assert item["contact_matrix"][0, 1].item() == 1
    assert item["contact_matrix"][2, 2].item() == -1

