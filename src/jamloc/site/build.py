"""Assemble and check the static showcase."""

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urljoin, urlsplit

from jamloc.paths import REPO_ROOT, results_dir
from jamloc.site.cases import CASES
from jamloc.site.results import write_results

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
_MODULE_REFERENCE = re.compile(
    r"^\s*(?:import|export)\s+(?:[^;\n]*?\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE
)
_NUMBER = (int, float)
_NULLABLE_NUMBER = (*_NUMBER, type(None))


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


def _read_json(path: Path, name: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc


def _require_fields(obj, location, **fields):
    if not isinstance(obj, dict):
        raise ValueError(f"{location}: expected an object")  # noqa: TRY004 (invalid JSON data)
    for field, types in fields.items():
        if field not in obj:
            raise ValueError(f"{location}: missing field {field}")
        allowed = types if isinstance(types, tuple) else (types,)
        if type(obj[field]) not in allowed:
            expected = " or ".join(t.__name__ for t in allowed)
            raise ValueError(f"{location}: field {field} must be {expected}")


def _require_v2(document, location):
    _require_fields(document, location, schema_version=int)
    if document["schema_version"] != 2:
        raise ValueError(f"{location}: expected schema version 2")


def _require_shape(array, location, size, columns=None):
    if not isinstance(array, list) or len(array) != size:
        raise ValueError(f"{location}: expected an array of length {size}")
    if columns is not None and any(not isinstance(r, list) or len(r) != columns for r in array):
        raise ValueError(f"{location}: expected {columns} entries per row")


def _check_sweep(document, location, width):
    _require_fields(document, location, prb=dict, benign=dict, sweep=list)
    _require_fields(document["prb"], f"{location}.prb", threshold=list)
    _require_shape(document["prb"]["threshold"], f"{location}.prb.threshold", width)
    frames = document["sweep"]
    if not frames:
        raise ValueError(f"{location}.sweep: expected JSR frames")
    for index, frame in enumerate([document["benign"], *frames]):
        label = f"{location}.sweep[{index - 1}]" if index else f"{location}.benign"
        _require_fields(
            frame, label, jsr_db=_NUMBER if index else type(None), r_b=list, pilots=dict
        )
        _require_shape(frame["r_b"], f"{label}.r_b", width)
        pilots = frame["pilots"]
        _require_fields(pilots, f"{label}.pilots", subcarrier=list, residual_power=list)
        for field in ("subcarrier", "residual_power"):
            _require_shape(pilots[field], f"{label}.pilots.{field}", width * 6)
    if document["metadata"]["jsr_db"] not in [frame["jsr_db"] for frame in frames]:
        raise ValueError(f"{location}.sweep: missing the selection JSR frame")


def _check_case(document, location, name):
    _require_v2(document, location)
    _require_fields(
        document,
        location,
        metadata=dict,
        grids=dict,
        detectors=dict,
        receivers=dict,
        x_hat=dict,
        abs_llr=dict,
    )
    meta = document["metadata"]
    label = f"{location}.metadata"
    _require_fields(
        meta,
        label,
        name=str,
        title=str,
        channel=str,
        snr_db=_NUMBER,
        jsr_db=_NULLABLE_NUMBER,
        noise_estimate=str,
        seed=int,
        detector_seed=int,
        batch_size=int,
        alloc=list,
        config=dict,
        selection=dict,
        thresholds=dict,
        batch_crc=dict,
    )
    if meta["name"] != name:
        raise ValueError(f"{label}.name: expected manifest case name {name}")
    cfg = meta["config"]
    _require_fields(cfg, f"{label}.config", dmrs_symbol=int, num_prb=int, num_symbols=int)
    prbs, symbols = cfg["num_prb"], cfg["num_symbols"]
    if prbs < 1 or symbols < 2 or not 0 <= cfg["dmrs_symbol"] < symbols:
        raise ValueError(f"{label}.config: invalid grid dimensions or DMRS symbol")
    _require_shape(meta["alloc"], f"{label}.alloc", 2)
    start, width = meta["alloc"]
    if (
        any(type(x) is not int for x in (start, width))
        or start < 0
        or not 0 < width <= prbs - start
    ):
        raise ValueError(f"{label}.alloc: expected a PRB start and width within the grid")

    selection = meta["selection"]
    _require_fields(
        selection,
        f"{label}.selection",
        rule=str,
        slot=int,
        hit_rate=_NULLABLE_NUMBER,
        batch_median=_NULLABLE_NUMBER,
    )
    selection_fields = {"first_crc_flip": "flip_count", "first_true_soft_failure": "failure_count"}
    if field := selection_fields.get(selection["rule"]):
        _require_fields(selection, f"{label}.selection", **{field: int})
    _require_fields(meta["thresholds"], f"{label}.thresholds", fused=dict)
    _require_fields(
        meta["thresholds"]["fused"],
        f"{label}.thresholds.fused",
        split=str,
        training_data=str,
        false_alarm_rate=_NUMBER,
        value=_NUMBER,
    )
    if name == "pilot_whole_comb":
        _require_fields(meta, label, noise_variance=_NUMBER, true_noise_variance=_NUMBER)

    grids = document["grids"]
    grid_fields = ("allocation", "truth", "power_db", "residual_db", "evm_db")
    for field in grid_fields:
        _require_shape(grids.get(field), f"{location}.grids.{field}", prbs, symbols)
    maps = document["detectors"]
    definition = CASES.get(name)
    required_maps = definition.detectors if definition else ("fused",)
    _require_fields(maps, f"{location}.detectors", **dict.fromkeys(required_maps, dict))
    for detector, layer in maps.items():
        label = f"{location}.detectors.{detector}"
        _require_fields(layer, label, scores=list, decisions=list, unscored=list)
        for field in ("scores", "decisions", "unscored"):
            _require_shape(layer[field], f"{label}.{field}", prbs, symbols)

    receivers = document["receivers"]
    decode_maps = definition.decode_maps if definition else ("fused",)
    required_receivers = (
        "standard",
        "true_soft",
        "true_prb_soft",
        *(f"{m}_soft" for m in decode_maps),
    )
    _require_fields(receivers, f"{location}.receivers", **dict.fromkeys(required_receivers, dict))
    for rx, receiver in receivers.items():
        _require_fields(receiver, f"{location}.receivers.{rx}", crc=bool, x_hat=str, abs_llr=str)
        counts = meta["batch_crc"].get(rx)
        _require_fields(counts, f"{location}.metadata.batch_crc.{rx}", slots=int, failures=int)
        for field, columns in (("x_hat", 2), ("abs_llr", None)):
            key = receiver[field]
            _require_shape(
                document[field].get(key),
                f"{location}.{field}.{key}",
                width * 12 * (symbols - 1),
                columns,
            )

    if name == "jsr_sweep":
        _require_fields(meta, f"{location}.metadata", jam=list)
        _require_shape(meta["jam"], f"{location}.metadata.jam", 2)
        _check_sweep(document, location, width)


def _check_cases(root):
    manifest_path = "data/manifest.json"
    manifest = _read_json(root / manifest_path, manifest_path)
    _require_v2(manifest, manifest_path)
    _require_fields(manifest, manifest_path, cases=list)
    cases = {}
    for index, entry in enumerate(manifest["cases"]):
        _require_fields(entry, f"{manifest_path}: cases[{index}]", name=str, title=str, file=str)
        name = entry["name"]
        if name in cases:
            raise ValueError(f"{manifest_path}: duplicate case {name}")
        cases[name] = entry["file"]
    # The default export registry also supplies the manifest that drives Explore and the gallery.
    # fig3 is a standalone export that the pages do not load.
    missing = CASES.keys() - cases.keys() - {"fig3"}
    if missing:
        raise ValueError(f"{manifest_path}: missing required cases: {', '.join(sorted(missing))}")
    for name, reference in cases.items():
        if urlsplit(reference).scheme or reference.startswith("/"):
            raise ValueError(f"{manifest_path}: expected a relative case file: {reference}")
        case = _local_path(root, manifest_path, reference)
        if not case.resolve().is_relative_to((root / "data").resolve()):
            raise ValueError(f"{manifest_path}: case must be inside data/: {reference}")
        label = case.relative_to(root).as_posix()
        _check_case(_read_json(case, label), label, name)


def check_build(root: Path) -> None:
    """Check local references, script origins, and the pages' versioned case data."""
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

    _check_cases(root)

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
