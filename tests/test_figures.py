import json
import xml.etree.ElementTree as ET

import pytest
from matplotlib.figure import Figure

from jamloc.experiments import figures
from jamloc.paths import REPO_ROOT

MODEL_FREE = ["attack_patterns", "residual_explained", "constellations"]


def test_drawio_sources_are_xml():
    sources = sorted((REPO_ROOT / "diagrams").glob("*.drawio"))
    assert sources
    for source in sources:
        root = ET.parse(source).getroot()
        assert root.tag == "mxfile", source
        for cell in root.findall(".//mxCell[@vertex='1']"):
            assert cell.find("mxGeometry[@as='geometry']") is not None, cell.get("id")


def test_drawio_export(tmp_path, monkeypatch):
    from jamloc import diagrams

    if diagrams.find_drawio() is None:
        pytest.skip("draw.io desktop is not installed")

    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    diagrams.main()
    for source in (REPO_ROOT / "diagrams").glob("*.drawio"):
        stem = tmp_path / "figures" / source.stem
        assert stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
        assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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
    for name in (
        "feature_distributions",
        "detector_outputs",
        "training_curves",
        "constellations_with_map",
    ):
        fig, caption = figures.FIGURES[name]()
        assert fig.axes and caption.count(". ") >= 1 and len(caption) > 120, name

    figures.main([])
    out = tmp_path / "results" / "figures"
    html = (out / "index.html").read_text(encoding="utf-8")
    captions = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    import html as html_lib

    for name in figures.FIGURES:
        assert f'src="{name}.png"' in html, name
        assert html_lib.escape(captions[name]) in html, name  # every caption is on the page
        assert f"<b>{name}</b>: Not produced yet" not in html


def test_gallery_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    figures.main([])
    html = (tmp_path / "results" / "figures" / "index.html").read_text(encoding="utf-8")
    assert 'src="attack_patterns.png"' in html
    assert "Not produced yet" in html and "detector_outputs" in html


def test_power_is_compared_per_cell_and_evm_per_prb():
    import numpy as np

    from jamloc.features import FEATURE_NAMES
    from jamloc.jammer import ATTACKS

    names = list(FEATURE_NAMES)
    f = np.zeros((1, 6, 24, 14), dtype=np.float32)
    f[0, names.index("alloc_mask"), 0:6] = 1
    f[0, names.index("dmrs_mask"), 0:6, 2] = 1
    f[0, names.index("data_mask"), 0:6] = 1
    f[0, names.index("data_mask"), 0:6, 2] = 0
    labels = np.zeros((1, 24, 14), dtype=np.uint8)
    labels[0, 2:4, 2] = 1  # pilot jamming of PRBs 2-3
    f[0, names.index("power_db"), 2:4, 2] = 10.0
    d = {
        "features": f,
        "labels": labels,
        "attack": np.array([ATTACKS.index("pilot")]),
        "jsr_db": np.array([-3.0]),
    }

    clean, jammed = figures._feature_groups(d, "pilot", "power_db", "alloc_mask")
    assert jammed.tolist() == [10.0, 10.0] and clean.size == 6 * 14 - 2
    clean, jammed = figures._feature_groups(d, "pilot", "evm_db", "data_mask")
    assert jammed.size == 2 * 13  # data cells of the pilot-jammed PRBs


@pytest.mark.parametrize("channel", ["A", "C"])
def test_example_seed_has_no_fade_on_the_jammed_block(channel):
    seed = figures._example_seed(channel)
    assert figures._block_power_ok(figures._example_batch("none", 0.0, 20.0, channel, seed))


def test_gallery_only_keeps_existing_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    out = tmp_path / "figures"
    out.mkdir(parents=True)
    (out / "attack_patterns.png").write_bytes(b"not really a png")
    captions = {"attack_patterns": "Caption <with> special & characters."}
    (out / "captions.json").write_text(json.dumps(captions), encoding="utf-8")

    figures.main(["--gallery-only"])
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'src="attack_patterns.png"' in page
    assert "Caption &lt;with&gt; special &amp; characters." in page
    assert not (out / "system_diagram.png").exists()  # nothing was regenerated
    assert "python -m jamloc.diagrams" in page
    assert "python -m jamloc.experiments.ber_bler" in page  # missing result figure explained

    (out / "system_diagram.png").write_bytes(b"externally rendered preview")
    figures.main(["--gallery-only"])
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'src="system_diagram.png"' in page
    assert "second pass through the same slot" in page
