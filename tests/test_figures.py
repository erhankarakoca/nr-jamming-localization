import json

import pytest
from matplotlib.figure import Figure

from jamloc.experiments import figures

MODEL_FREE = ["attack_patterns", "system_diagram", "residual_explained", "constellations"]


@pytest.mark.parametrize("name", MODEL_FREE)
def test_model_free_figures(name):
    fig, caption = figures.FIGURES[name]()
    assert isinstance(fig, Figure) and fig.axes
    assert caption.count(". ") >= 1 and len(caption) > 120  # at least two real sentences


def test_main_writes_files_and_captions(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    figures.main(["--only", "attack_patterns"])
    out = tmp_path / "figures"
    assert (out / "attack_patterns.pdf").is_file() and (out / "attack_patterns.png").is_file()
    captions = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    assert "attack_patterns" in captions


def _tiny_project(tmp_path, monkeypatch):
    from jamloc.dataset import generate
    from jamloc.model import VARIANTS
    from jamloc.train import train

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    generate("test", num_slots=60, batch_size=32)
    for variant in VARIANTS:
        train(variant, epochs=2, batch_size=32)


def test_data_figures_and_gallery(tmp_path, monkeypatch):
    _tiny_project(tmp_path, monkeypatch)
    for name in ("feature_distributions", "detector_outputs", "training_curves"):
        fig, caption = figures.FIGURES[name]()
        assert fig.axes and caption.count(". ") >= 1 and len(caption) > 120, name

    figures.main([])
    out = tmp_path / "results" / "figures"
    html = (out / "index.html").read_text(encoding="utf-8")
    captions = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    for name in figures.FIGURES:
        assert f'src="{name}.png"' in html, name
        assert name in captions
    assert "Not produced yet" not in html


def test_gallery_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    figures.main([])
    html = (tmp_path / "results" / "figures" / "index.html").read_text(encoding="utf-8")
    assert 'src="attack_patterns.png"' in html
    assert "Not produced yet" in html and "detector_outputs" in html
