import numpy as np
import pytest
from colbert_ppi.scoring import score_protein_pair, score_protein_rna, normalise_vectors


def test_pair_order_and_residue_order_do_not_change_retrieval():
    rng = np.random.default_rng(19)
    aq, ac = rng.normal(size=(2, 4, 8))
    bq, bc = rng.normal(size=(2, 7, 8))
    q, c = rng.normal(size=(2, 11, 8))
    result = score_protein_pair(aq, ac, bq, bc, q, c)
    swapped = score_protein_pair(bq, bc, aq, ac, q, c)
    permuted = score_protein_pair(aq[::-1], ac[::-1], bq, bc, q[::-1], c)
    assert result['score'] == pytest.approx(swapped['score'])
    assert result['score'] == pytest.approx(permuted['score'])
    np.testing.assert_allclose(result['raw_local_matrix'], swapped['raw_local_matrix'].T)


def test_calibration_does_not_change_localisation():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(4, 5, 8))
    banks = rng.normal(size=(2, 9, 8))
    calibrated = score_protein_pair(*vectors, *banks)
    raw = score_protein_pair(*vectors, alpha=0)
    np.testing.assert_array_equal(calibrated['raw_local_matrix'], raw['raw_local_matrix'])
    assert calibrated['score'] != pytest.approx(raw['score'])


def test_pri_mask_excludes_padding():
    p, r = np.eye(3), np.eye(3)[::-1]
    padded = np.vstack([p, np.zeros((2, 3))])
    assert score_protein_rna(p, r) == pytest.approx(
        score_protein_rna(padded, r, protein_mask=[1, 1, 1, 0, 0]))


@pytest.mark.parametrize('vectors', [np.zeros((1, 3)), np.empty((0, 3)), [[np.nan, 1]]])
def test_invalid_residue_vectors_fail(vectors):
    with pytest.raises(ValueError):
        normalise_vectors(vectors)


def test_calibrated_scoring_requires_reference_bank():
    with pytest.raises(ValueError, match='banks'):
        score_protein_pair(*[np.eye(3)] * 4)
