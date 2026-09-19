import importlib.util
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('download_assets', Path(__file__).parents[1] / 'download_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


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
