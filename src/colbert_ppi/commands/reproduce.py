"""Recalculate the manuscript metrics from the supplied scores and labels."""

import json
from pathlib import Path
import time

from colbert_ppi.metrics import (
    check_retrieval,
    check_localisation,
    check_protein_rna,
)


def run(args):
    data = args.root / "data/benchmarks"
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    retrieval = check_retrieval(data / "source_data")
    localisation = check_localisation(data, args.output)
    run_count, transfer = check_protein_rna(data, args.output)

    report = {
        "status": "PASS",
        "fig2": retrieval,
        "localisation_means": localisation,
        "pri_runs_and_cohorts": run_count,
        "pri_clean": transfer,
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "recompute numerical evidence from frozen predictions; does not retrain models, rerun PLM encoders, or remeasure hardware timing",
    }
    (args.output / "reproduction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
