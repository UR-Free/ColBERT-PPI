from pathlib import Path
import zipfile
import pytest

import download as assets


def test_rejects_path_escape_before_extracting_any_files(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('valid.txt', 'ok')
        bundle.writestr('../escape.txt', 'bad')
    destination = tmp_path / 'out'
    with pytest.raises(ValueError):
        assets.extract(archive, destination)
    assert not (destination / 'valid.txt').exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_verified_archive_layout_and_no_overwrite(tmp_path):
    archive = tmp_path / 'ok.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('data/example.txt', 'hello')
    destination = tmp_path / 'out'
    assets.extract(archive, destination)
    assert (destination / 'data/example.txt').read_text() == 'hello'
    with pytest.raises(FileExistsError):
        assets.extract(archive, destination)


def test_benchmark_prefix_is_removed(tmp_path):
    archive = tmp_path / 'benchmark.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('data/', '')
        bundle.writestr('data/ppi/labels.txt', 'labels')
    destination = tmp_path / 'data/benchmarks'
    assets.extract(archive, destination, strip_prefix='data')
    assert (destination / 'ppi/labels.txt').read_text() == 'labels'
    assert not (destination / 'data').exists()


def test_wrong_benchmark_prefix_is_rejected_before_writes(tmp_path):
    archive = tmp_path / 'bad-prefix.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('data/ppi/labels.txt', 'labels')
        bundle.writestr('unexpected/file', 'bad')
    destination = tmp_path / 'out'
    with pytest.raises(ValueError):
        assets.extract(archive, destination, strip_prefix='data')
    assert not destination.exists()
