import pytest

from jamloc import paper


def test_init_and_figure_sync(tmp_path, monkeypatch):
    results = tmp_path / "results"
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(results))
    dest = tmp_path / "overleaf"
    dest.mkdir()
    (dest / "main.tex").write_text("blank Overleaf template", encoding="utf-8")

    # every figure the draft includes, as an empty file in results/
    tex = (paper.REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    figures = paper.figure_paths(tex)
    assert figures and all(f.endswith(".pdf") for f in figures)
    for rel in figures[:-1]:  # leave the last one out
        (results / rel).parent.mkdir(parents=True, exist_ok=True)
        (results / rel).write_bytes(b"%PDF")

    with pytest.raises(SystemExit):  # a missing figure is an error
        paper.main(["--dest", str(dest), "--init"])
    written = (dest / "main.tex").read_text(encoding="utf-8")
    assert "\\graphicspath{{figures/}}" in written and "../results" not in written
    assert "\\bibliography{references}" in written
    assert (dest / "references.bib").is_file()
    for rel in figures[:-1]:
        assert (dest / "figures" / rel).is_file()

    # Once the clone holds the paper, --init refuses to overwrite it
    with pytest.raises(SystemExit, match="already holds the paper"):
        paper.init_sources(dest)


def test_figure_paths():
    tex = r"\includegraphics[width=\columnwidth]{a/b.pdf} \includegraphics{c/d} \includegraphics{a/b.pdf}"
    assert paper.figure_paths(tex) == ["a/b.pdf", "c/d.pdf"]


def test_sync_copies_the_figures_of_every_version(tmp_path, monkeypatch):
    results = tmp_path / "results"
    for rel in ("figures/a.pdf", "figures/b.pdf"):
        (results / rel).parent.mkdir(parents=True, exist_ok=True)
        (results / rel).write_bytes(b"%PDF-")
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(results))
    dest = tmp_path / "clone"
    dest.mkdir()
    (dest / "main.tex").write_text("\\includegraphics{figures/a.pdf}", encoding="utf-8")
    (dest / "conference.tex").write_text("\\includegraphics[width=1in]{figures/b}", encoding="utf-8")
    copied, missing = paper.sync_figures(dest)
    assert sorted(copied) == ["figures/a.pdf", "figures/b.pdf"] and not missing
    assert (dest / "figures" / "figures" / "b.pdf").is_file()
