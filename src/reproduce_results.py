"""Recalculate the manuscript metrics from the supplied scores and labels."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colbert_ppi.evaluation import (
    check_retrieval,
    check_localisation,
    check_protein_rna,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/validation_reports",
        help="Directory for recalculated tables and the comparison report",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    retrieval = check_retrieval(root / "data/source_data")
    localisation = check_localisation(root, args.output)
    run_count, transfer = check_protein_rna(root, args.output)

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


if __name__ == "__main__":
    main()
