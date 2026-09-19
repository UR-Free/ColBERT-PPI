"""Run the frozen-vector example and check its expected output."""

from pathlib import Path
import json
import time
import numpy as np

from colbert_ppi.scoring import score_protein_pair, score_protein_rna



def run(args):
    root = args.root
    start = time.perf_counter()
    x = np.load(root / "data/examples/scoring/vectors.npz")
    expected = np.load(root / "data/examples/scoring/expected.npz")
    y = score_protein_pair(
        **{
            k: x[k]
            for k in [
                "a_query",
                "a_candidate",
                "b_query",
                "b_candidate",
                "bank_query",
                "bank_candidate",
            ]
        }
    )
    errs = {
        key: float(np.max(np.abs(y[key] - expected[ref])))
        for key, ref in [
            ("raw_local_matrix", "raw"),
            ("calibrated_matrix", "corrected"),
        ]
    }
    assert max(errs.values()) < 2e-6, errs
    # These are unit checks on the same vectors, not claims of PRI predictive accuracy.
    a = x["a_query"]
    b = x["b_candidate"]
    v = score_protein_rna(a, b)
    assert np.isclose(
        v,
        score_protein_rna(
            np.vstack([a, np.zeros_like(a[:3])]),
            b,
            protein_mask=np.arange(len(a) + 3) < len(a),
        ),
    )
    assert np.isclose(v, score_protein_rna(b, a))
    report = {
        "status": "PASS",
        "example": "real frozen PPI output vectors, first cohort complex",
        "protein_pair_score": y["score"],
        "max_absolute_matrix_errors": errs,
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "cached-vector readout only; no raw-sequence inference or retraining",
    }
    print(json.dumps(report, indent=2))
