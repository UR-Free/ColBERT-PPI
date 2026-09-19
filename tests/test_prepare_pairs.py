import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prepare_pairs', Path(__file__).parents[1] / 'src/prepare_pairs.py')
prepare_pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_pairs)


def test_manifest_paths_are_relative_and_repeated_structures_are_cached(tmp_path, monkeypatch):
    calls = []
    def tokens(path, *args):
        calls.append(path)
        return [0, 5, 2]
    monkeypatch.setattr(prepare_pairs, 'structure_tokens', tokens)
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('pair_id,structure_a,structure_b\na,x.pdb,y.pdb\nb,y.pdb,x.pdb\n')
    output = tmp_path / 'output.json'
    assert prepare_pairs.prepare(manifest, output, 'foldseek', None) == 2
    assert calls == [tmp_path / 'x.pdb', tmp_path / 'y.pdb']
    assert json.loads(output.read_text())[1]['pair_id'] == 'b'


@pytest.mark.parametrize('descriptor', ['a\tAAAA\tpppp\nb\tAAAA\tpppp\n', 'a\tAA\tppp\n', 'a\t' + 'A'*1025 + '\t' + 'p'*1025 + '\n'])
def test_ambiguous_or_overlong_structures_fail(tmp_path, monkeypatch, descriptor):
    path = tmp_path / 'input.pdb'
    path.touch()
    def run(command, **kwargs):
        Path(command[-1]).write_text(descriptor)
    monkeypatch.setattr(prepare_pairs.subprocess, 'run', run)
    with pytest.raises(ValueError):
        prepare_pairs.structure_tokens(path, 'foldseek', None)
