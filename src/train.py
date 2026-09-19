"""Train on the small real-data example and select a checkpoint using validation."""

from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colbert_ppi.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ppi", "pri"], required=True)
    parser.add_argument("--data-dir", default="data/examples/training")
    parser.add_argument("--saprot-dir", required=True)
    parser.add_argument("--ernie-checkpoint")
    parser.add_argument("--ernie-code")
    parser.add_argument(
        "--protein-init",
        help="Optional best.pt from the PPI training example, for PRI initialization",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.protein_init and args.task != "pri":
        parser.error("--protein-init applies to PRI only")
    train(args)


if __name__ == "__main__":
    main()
