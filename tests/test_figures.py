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
