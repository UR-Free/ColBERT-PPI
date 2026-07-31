from __future__ import annotations

import numpy as np

from colbert_ppi.cli.prepare_data import contact_indices, load_manifest


def test_manifest_paths_are_resolved_relative_to_manifest(tmp_path) -> None:
    structure_a = tmp_path / "a.pdb"
    structure_b = tmp_path / "b.pdb"
    structure_a.write_text("")
    structure_b.write_text("")
    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "protein_a,protein_b,structure_a,structure_b\n"
        "protein_a,protein_b,a.pdb,b.pdb\n"
    )

    pairs, structures = load_manifest(manifest)

    assert [pair.pair_id for pair in pairs] == ["protein_a:protein_b"]
    assert structures == {
        "protein_a": structure_a.resolve(),
        "protein_b": structure_b.resolve(),
    }


def test_contact_indices_apply_distance_thresholds_and_negative_cap() -> None:
    coordinates_a = np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    coordinates_b = np.asarray([[0.0, 0.0, 5.0], [60.0, 0.0, 0.0]])

    positives, negatives = contact_indices(
        coordinates_a,
        coordinates_b,
        positive_threshold=8.0,
        negative_threshold=12.0,
        max_negatives=2,
        rng=np.random.default_rng(42),
    )

    assert positives.tolist() == [[0, 0]]
    assert negatives.shape == (2, 2)
