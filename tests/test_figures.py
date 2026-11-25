import csv
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from matplotlib.figure import Figure

from jamloc.experiments import figures
from jamloc.paths import REPO_ROOT

MODEL_FREE = ["attack_patterns", "residual_explained", "constellations", "shrinkage_example"]


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


def test_residual_panels_show_the_fit_and_the_pooled_residual():
    from jamloc.features import delay_subspace, dmrs_ls

    fig, _ = figures.residual_explained()
    fit, prbs = fig.axes
    seed = figures._example_seed("C")
    series = {line.get_label(): line for ax in (fit, prbs) for line in ax.lines}
    for attack in ("none", "pilot"):
        batch = figures._example_batch(attack, 0.0, 20.0, "C", seed)
        ls, sc, power = dmrs_ls(batch.y_alloc, batch.tx)
        basis = delay_subspace(sc, figures.CFG.scs, figures.CFG.tau_max).cpu().numpy()
        projection = basis @ basis.conj().T
        estimates = ls[0].cpu().numpy()
        residual = estimates - projection @ estimates
        energy = abs(residual) ** 2 / (batch.no.item() / power)
        kept = 1 - np.diag(projection).real
        pooled = energy.reshape(-1, 6).sum(1) / kept.reshape(-1, 6).sum(1)
        label = "Benign slot" if attack == "none" else "Jammed slot"
        np.testing.assert_allclose(series[label].get_ydata(), 10 * np.log10(pooled), atol=0.002)
        if attack == "pilot":
            # (a) shows the jammed slot: every pilot once, and the smooth curve is its projection
            shown = np.r_[series["Clean PRBs"].get_ydata(), series["Jammed PRBs"].get_ydata()]
            np.testing.assert_allclose(np.sort(shown), np.sort(abs(estimates)), rtol=1e-5)
            np.testing.assert_allclose(
                series["Subspace fit"].get_ydata(), abs(projection @ estimates), rtol=1e-5
            )
            jammed = series["Jammed PRBs"].get_xdata()
            assert len(jammed) == 6 * figures.JAM[1]
            assert jammed.min() > figures.JAM[0] - 0.5 and jammed.max() < sum(figures.JAM) - 0.5


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

    # A miniature interval export, with other seeds, attacks and JSRs to distinguish.
    out = tmp_path / "results" / "intervals"
    out.mkdir()
    with (out / "test_pd_ci.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("detector", "attack", "seed", "n_seeds", "jsr_db", "pd", "pd_lo", "pd_hi"))
        for i, detector in enumerate(
            ("fused", "no_residual", "power", "evm", "block_test", "evm_threshold", "residual_cfar")
        ):
            pd = (i + 1) / 10
            writer.writerow((detector, "pilot", "mean", 3, -15, pd, pd - 0.03, pd + 0.05))
            for attack, seed, db in (
                ("pilot", "0", -15),
                ("data", "mean", -18),
                ("pilot", "mean", -9),
            ):
                writer.writerow((detector, attack, seed, 3, db, 0.99, 0.98, 1.0))


def test_data_figures_and_gallery(tmp_path, monkeypatch):
    _tiny_project(tmp_path, monkeypatch)
    for name in (
        "system_views",
        "feature_distributions",
        "detector_outputs",
        "training_curves",
        "constellations_with_map",
    ):
        fig, caption = figures.FIGURES[name]()
        assert fig.axes and caption.count(". ") >= 1 and len(caption) > 120, name
        if name == "detector_outputs":
            truth, evm, unet, pd = fig.axes
            for ax in (truth, evm, unet):
                assert ax.images[0].get_array().shape == (24, 14)
            expected = np.arange(1, 8) / 10
            np.testing.assert_allclose(pd.lines[0].get_xdata(), expected)
            segments = pd.collections[0].get_segments()
            np.testing.assert_allclose(
                [s[:, 0] for s in segments], np.c_[expected - 0.03, expected + 0.05]
            )

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


def test_shrinkage_lies_between_the_cell_and_its_row_mean():
    from jamloc.experiments.figures import CFG, _estimates_without_map, shrinkage_example

    # Block data jammer: the rows it does not touch shrink to their (near zero) row mean
    est = _estimates_without_map(CFG, "data", 9.0, seed=7)
    cell, row, shrink = est["cell"], est["prb"], est["shrink"]
    ok = ~np.isnan(cell)
    lo, hi = np.minimum(cell, row), np.maximum(cell, row)
    assert np.all(shrink[ok] >= np.maximum(lo[ok], 0) - 1e-6)
    assert np.all(shrink[ok] <= np.maximum(hi[ok], 0) + 1e-6)
    clean = ~est["truth"].any(axis=1)[:, None] & ok  # data cells of rows without any jammed cell
    assert clean.any()
    assert np.allclose(shrink[clean], row[clean], atol=0.05)

    fig, caption = shrinkage_example()
    assert isinstance(fig, Figure) and len(fig.axes) >= 2 and "shrinkage" in caption
