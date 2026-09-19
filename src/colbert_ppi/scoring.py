"""Portable NumPy implementation of the manuscript's frozen readouts."""

import numpy as np
from scipy.special import logsumexp


def normalise_vectors(x, mask=None):
    """Remove masked positions and L2-normalise each residue vector."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected [residues, features]")
    if mask is not None:
        x = x[np.asarray(mask, dtype=bool)]
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite vectors")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError("Zero vectors must be excluded by the residue mask")
    return x / n


def log_mean_exp(x, tau, axis):
    """Smooth maximum normalised by the number of matching positions."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * (logsumexp(x / tau, axis=axis) - np.log(x.shape[axis]))


def reference_background(x, opposite_bank, tau):
    """Estimate each residue's matching tendency against the opposite-role bank."""
    bank = normalise_vectors(opposite_bank)
    return np.concatenate(
        [
            log_mean_exp(part @ bank.T, tau, 1)
            for part in np.array_split(x, max(1, (len(x) + 127) // 128))
        ]
    )


def smooth_maxsim(matrix, tau):
    """Average row-wise and column-wise smooth matching evidence."""
    return float(
        0.5
        * (log_mean_exp(matrix, tau, 1).mean() + log_mean_exp(matrix, tau, 0).mean())
    )


def score_protein_pair(
    a_query,
    a_candidate,
    b_query,
    b_candidate,
    bank_query=None,
    bank_candidate=None,
    alpha=1.0,
    tau=0.03,
):
    """Score both explicit role assignments of a protein pair.

    The four inputs have shape [residues, features] and exclude special tokens
    and padding. Query/candidate outputs for the same protein must align.
    Banks have shape [reference residues, features] and are required when
    alpha is nonzero. tau is the log-mean-exp temperature.

    Returns a scalar retrieval score, a raw local cosine matrix, and the
    aligned calibrated matrix. Localisation uses ``raw_local_matrix``;
    protein retrieval pools each role assignment before averaging.
    """
    aq, ac, bq, bc = map(
        normalise_vectors, [a_query, a_candidate, b_query, b_candidate]
    )
    if len(aq) != len(ac) or len(bq) != len(bc):
        raise ValueError("Encoder role residue indices must align")
    ab = aq @ bc.T
    ba = bq @ ac.T
    raw = 0.5 * (ab + ba.T)
    if alpha:
        if bank_query is None or bank_candidate is None:
            raise ValueError("Calibration requires both training banks")
        ab = ab - alpha * 0.5 * (
            reference_background(aq, bank_candidate, tau)[:, None]
            + reference_background(bc, bank_query, tau)[None, :]
        )
        ba = ba - alpha * 0.5 * (
            reference_background(bq, bank_candidate, tau)[:, None]
            + reference_background(ac, bank_query, tau)[None, :]
        )
    # Aggregate each explicit role assignment first; averaging matrices before pooling is different.
    return {
        "score": 0.5 * (smooth_maxsim(ab, tau) + smooth_maxsim(ba, tau)),
        "raw_local_matrix": raw,
        "calibrated_matrix": 0.5 * (ab + ba.T),
    }


def score_protein_rna(protein, rna, tau=0.001, protein_mask=None, rna_mask=None):
    """Score a residue-by-nucleotide cosine matrix with bidirectional smooth MaxSim.

    Input shapes are [protein residues, features] and [RNA nucleotides,
    features]. Optional Boolean masks select valid positions before pooling.
    This PRI readout uses no reference correction (alpha = 0).
    """
    return smooth_maxsim(
        normalise_vectors(protein, protein_mask) @ normalise_vectors(rna, rna_mask).T,
        tau,
    )
