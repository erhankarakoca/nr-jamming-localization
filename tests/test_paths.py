from pathlib import Path

from jamloc import paths


def test_defaults_are_inside_repo(monkeypatch):
    monkeypatch.delenv("JAMLOC_DATA_DIR", raising=False)
    monkeypatch.delenv("JAMLOC_RESULTS_DIR", raising=False)
    repo = Path(__file__).resolve().parents[1]
    assert paths.data_dir() == repo / "data"
    assert paths.results_dir() == repo / "results"


def test_env_vars_override_and_dirs_are_created(monkeypatch, tmp_path):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "drive" / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "drive" / "results"))
    assert paths.data_dir() == tmp_path / "drive" / "data"
    assert paths.results_dir() == tmp_path / "drive" / "results"
    assert paths.data_dir().is_dir()
    assert paths.results_dir().is_dir()
