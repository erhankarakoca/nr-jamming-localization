"""Assemble and check the static showcase."""

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urljoin, urlsplit

from jamloc.paths import REPO_ROOT, results_dir
from jamloc.site.results import write_results

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
_MODULE_REFERENCE = re.compile(
    r"^\s*(?:import|export)\s+(?:[^;\n]*?\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE
)


class _PageReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src", "poster", "data", "action") and value is not None:
                self.references.append((tag, key, value.strip()))


def _local_path(root: Path, base: str, reference: str) -> Path:
    # A dummy authority lets urljoin resolve relative references with normal URL rules.
    url = urlsplit(urljoin(f"https://build.invalid/{base}", reference))
    path = root / unquote(url.path).lstrip("/")
    if url.netloc != "build.invalid" or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{base}: reference leaves the build: {reference}")
    if path.is_dir():
        path /= "index.html"
    if not path.is_file():
        raise ValueError(f"{base}: missing local file {reference}")
    return path


def check_build(root: Path) -> None:
    """Check local page references, script origins, and files named by the case manifest."""
    for page in sorted(root.rglob("*.html")):
        name = page.relative_to(root).as_posix()
        parser = _PageReferences()
        parser.feed(page.read_text(encoding="utf-8"))
        for tag, key, value in parser.references:
            url = urlsplit(value)
            external = url.scheme or url.netloc or value.startswith("//")
            if tag == "script" and key == "src" and external:
                if value != PLOTLY_URL:
                    raise ValueError(
                        f"{name}: script source must be local or pinned Plotly: {value}"
                    )
                if name != "results.html":
                    raise ValueError(f"{name}: Plotly is only allowed on results.html")
            if tag == "script" and key == "src" and not url.path:
                raise ValueError(f"{name}: script source is missing a file path: {value!r}")
            if not external and url.path:
                _local_path(root, name, value)

    for module in sorted(root.rglob("*.js")):
        name = module.relative_to(root).as_posix()
        for reference in _MODULE_REFERENCE.findall(module.read_text(encoding="utf-8")):
            if not reference.startswith(("./", "../")):
                raise ValueError(f"{name}: expected a relative module import: {reference}")
            _local_path(root, name, reference)

    manifest_path = "data/manifest.json"
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(f"{manifest_path}: expected schema version 2")
    for entry in manifest["cases"]:
        reference = entry["file"]
        if urlsplit(reference).scheme or reference.startswith("/"):
            raise ValueError(f"{manifest_path}: expected a relative case file: {reference}")
        case = _local_path(root, manifest_path, reference)
        if not case.resolve().is_relative_to((root / "data").resolve()):
            raise ValueError(f"{manifest_path}: case must be inside data/: {reference}")

    results = json.loads((root / "data" / "results.json").read_text(encoding="utf-8"))
    for section in results.values():
        for download in section["downloads"]:
            _local_path(root, "results.html", download)


def build_site(*, results_from: Path | None = None) -> Path:
    """Check a staged build before replacing the last successful build."""
    results = results_dir()
    folder = results / "site"
    output = folder / "build"
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-", dir=folder) as temp:
        stage = Path(temp) / "build"
        shutil.copytree(REPO_ROOT / "site", stage)
        shutil.copytree(folder / "data", stage / "data")
        write_results(results if results_from is None else results_from, stage)
        (stage / ".nojekyll").touch()
        check_build(stage)

        previous = Path(temp) / "previous"
        if output.exists():
            output.rename(previous)
        try:
            stage.rename(output)
        except OSError:
            if previous.exists() and not output.exists():
                previous.rename(output)
            raise
    return output
