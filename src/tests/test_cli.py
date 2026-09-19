import json
from pathlib import Path

from colbert_ppi.cli import build_parser, resolve_args, main

ROOT = Path(__file__).resolve().parents[2]


def resolved(*argv):
    return resolve_args(build_parser().parse_args(['--root', str(ROOT), *argv]))


def test_task_defaults_and_explicit_overrides():
    args = resolved('predict')
    assert args.device == 'cuda:0'
    assert args.checkpoint.endswith('ppi_full/weights.pt')
    assert args.output == Path('data/results/ppi/predictions.json')
    args = resolved('predict', '--task', 'pri', '--device', 'cuda:2', '--input', 'own.json')
    assert args.device == 'cuda:2'
    assert args.input == 'own.json'
    assert args.checkpoint.endswith('pri_ppi_100_seed42/weights.pt')
    assert args.reference_bank is None


def test_transfer_can_be_disabled_without_losing_other_defaults():
    args = resolved('train', '--task', 'pri', '--protein-init', '', '--epochs', '3')
    assert args.protein_init == ''
    assert args.epochs == 3
    assert args.batch_size == 2
    assert args.data_dir == 'data/examples/pri'


def test_dry_run_from_outside_repository(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(['--root', str(ROOT), 'evaluate', '--dry-run'])
    settings = json.loads(capsys.readouterr().out)
    assert settings['root'] == str(ROOT)
    assert settings['device'] == 'cuda:0'
    assert Path.cwd() == tmp_path
