"""Shared script arguments and task defaults from config/*.json."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path


def build_parser(command):
    parser = argparse.ArgumentParser()
    parser.set_defaults(command=command)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Repository directory")
    p = parser
    p.add_argument("--dry-run", action="store_true", help="Print resolved settings without running")
    if command in {"predict", "evaluate", "train"}:
        tasks = ["ppi", "pri", "y2h"] if command == "evaluate" else ["ppi", "pri"]
        p.add_argument("--task", choices=tasks, default="ppi")
        p.add_argument("--config", type=Path, help="Override config/ppi.json or config/pri.json")
        p.add_argument("--device", help="PyTorch device (default from config: cuda:0)")
        p.add_argument("--saprot-dir")
        p.add_argument("--ernie-code")
        p.add_argument("--ernie-checkpoint")
        p.add_argument("--output", type=Path)
    if command in {"predict", "evaluate"}:
        p.add_argument("--checkpoint")
        p.add_argument("--reference-bank")
    if command == "predict":
        p.add_argument("--input", help="Prepared pair JSON; defaults to the bundled task example")
    elif command == "evaluate":
        p.add_argument("--split", choices=["validation", "test"], default="test")
        p.add_argument("--scores", help="Optional frozen PRI score matrix, instead of neural encoding")
        p.add_argument("--threads", type=int, default=4)
    elif command == "train":
        p.add_argument("--data-dir", help="Directory containing train.json, validation.json and test.json")
        p.add_argument("--epochs", type=int)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--learning-rate", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--protein-init", help="PPI component checkpoint for PRI transfer; empty string disables transfer")
    elif command == "prepare":
        p.add_argument("--manifest", required=True, type=Path, help="CSV: pair_id,structure_a,structure_b")
        p.add_argument("--output", type=Path, default=Path("data/user/pairs.json"))
        p.add_argument("--saprot-dir")
        p.add_argument("--foldseek-bin", default="foldseek")
    elif command == "download":
        p.add_argument("asset", choices=["ppi", "pri", "benchmarks"])
        p.add_argument("--archive", type=Path, help="Use an already downloaded ZIP")
        p.add_argument("--destination", type=Path, help="Alternative repository directory for extraction")
    elif command in {"reproduce", "plot"}:
        p.add_argument("--output", type=Path)
        if command == "plot":
            p.add_argument("--official", action="store_true", help="Export the supplied SVG compositions")
    return parser

def resolve_args(args):
    """Explicit CLI arguments override the two readable task configurations."""
    args.root = args.root.expanduser().resolve()
    if args.command in {"predict", "evaluate", "train", "prepare"}:
        task = getattr(args, "task", "ppi")
        task_config = "pri" if task == "pri" else "ppi"
        config_path = getattr(args, "config", None) or Path("config") / f"{task_config}.json"
        config_path = args.root / config_path
        defaults = json.loads(config_path.read_text())
        if args.command == "train":
            defaults = {**defaults, **defaults.get("training", {})}
        for name, value in vars(args).copy().items():
            if value is None and name in defaults:
                setattr(args, name, defaults[name])
        if args.command != "prepare":
            args.config = config_path
            if args.output is None:
                base = Path("data/results") / task
                args.output = {
                    "predict": base / "predictions.json",
                    "evaluate": base / "evaluation" / getattr(args, "split", "test"),
                    "train": base / "training",
                }[args.command]
            args.output = Path(args.output)
            required = ["device", "saprot_dir"]
            if args.command == "predict":
                required += ["checkpoint", "input"]
            if args.command == "train":
                required += ["data_dir", "epochs", "batch_size", "learning_rate", "seed"]
                if task == "ppi" and args.protein_init:
                    raise ValueError("--protein-init applies to PRI training only")
            for name in required:
                if getattr(args, name, None) is None:
                    raise ValueError(f"Missing {name}; set it in {config_path} or on the command line")
    elif args.command == "download":
        args.destination = args.destination or args.root
    elif args.command in {"reproduce", "plot"}:
        args.output = args.output or Path("data/results") / args.command
    return args


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_script(command, run, argv=None):
    parser = build_parser(command)
    args = parser.parse_args(argv)
    try:
        args = resolve_args(args)
        if args.dry_run:
            print(json.dumps(vars(args), indent=2, default=str))
            return
        with working_directory(args.root):
            run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
