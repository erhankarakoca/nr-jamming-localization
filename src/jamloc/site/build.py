"""Assemble and check the static showcase."""

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urljoin, urlsplit

from jamloc.paths import REPO_ROOT, results_dir
from jamloc.site.cases import AIDED_RECEIVER, CASES
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


def _require_numeric_shape(array, location, size, columns=None):
    _require_shape(array, location, size, columns)
    values = array if columns is None else (value for row in array for value in row)
    if any(type(value) not in _NULLABLE_NUMBER for value in values):
        raise ValueError(f"{location}: expected numbers or null")


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


def _check_channel(channel, location, width, dmrs_symbol):
    _require_fields(
        channel,
        location,
        dmrs_symbol=int,
        num_sc=int,
        pilot_sc=list,
        pilot_jammed=list,
        ls=list,
        fit=list,
        true=list,
        standard=list,
        map=list,
        residual_prb=list,
        threshold_prb=list,
    )
    num_sc, num_pilots = width * 12, width * 6
    if channel["dmrs_symbol"] != dmrs_symbol or channel["num_sc"] != num_sc:
        raise ValueError(f"{location}: DMRS symbol or subcarrier count differs from the grid")
    for field in ("pilot_sc", "pilot_jammed"):
        _require_shape(channel[field], f"{location}.{field}", num_pilots)
    if any(type(sc) is not int or not 0 <= sc < num_sc for sc in channel["pilot_sc"]):
        raise ValueError(f"{location}.pilot_sc: expected allocation-relative subcarriers")
    if any(type(jammed) is not bool for jammed in channel["pilot_jammed"]):
        raise ValueError(f"{location}.pilot_jammed: expected booleans")
    for field in ("ls", "fit"):
        _require_numeric_shape(channel[field], f"{location}.{field}", num_pilots, 2)
    for field in ("true", "standard", "map"):
        _require_numeric_shape(channel[field], f"{location}.{field}", num_sc, 2)
    for field in ("residual_prb", "threshold_prb"):
        _require_numeric_shape(channel[field], f"{location}.{field}", width)


def _check_levels(document, location, levels, check_frame):
    """The Explore slider: one entry per jammer power, the case's own power as a bare marker."""
    _require_fields(document, location, levels=list)
    entries = document["levels"]
    for index, entry in enumerate(entries):
        _require_fields(entry, f"{location}.levels[{index}]", jsr_db=_NULLABLE_NUMBER)
    powers = [entry["jsr_db"] for entry in entries]
    own = document["metadata"]["jsr_db"]
    if (levels is not None and powers != list(levels)) or own not in powers:
        raise ValueError(f"{location}.levels: expected the jammer powers {levels or powers}")
    for index, entry in enumerate(entries):
        label = f"{location}.levels[{index}]"
        if entry["jsr_db"] == own:
            if set(entry) != {"jsr_db"}:
                raise ValueError(f"{label}: the case's own power refers to the top-level fields")
            continue
        _require_fields(
            entry,
            label,
            grids=dict,
            detectors=dict,
            receivers=dict,
            x_hat=dict,
            abs_llr=dict,
            batch_crc=dict,
        )
        check_frame(entry, label, entry["batch_crc"], f"{label}.batch_crc")
        if entry["jsr_db"] is None and any(any(row) for row in entry["grids"]["truth"]):
            raise ValueError(f"{label}.grids.truth: the switched-off jammer hit no cell")


def _check_symbol_refs(document, location, receiver, label, num_points):
    _require_fields(receiver, label, crc=bool, x_hat=str, abs_llr=str)
    for field, columns in (("x_hat", 2), ("abs_llr", None)):
        key = receiver[field]
        _require_numeric_shape(
            document[field].get(key), f"{location}.{field}.{key}", num_points, columns
        )


def _check_passes(document, location, width, num_points, iterations):
    _require_fields(document, location, passes=list)
    passes = document["passes"]
    if not passes or (iterations and not 2 <= len(passes) <= iterations + 1):
        raise ValueError(f"{location}.passes: invalid number of decoding passes")
    for index, step in enumerate(passes):
        label = f"{location}.passes[{index}]"
        _require_fields(step, label, h=list)
        _require_numeric_shape(step["h"], f"{label}.h", width * 12, 2)
        _check_symbol_refs(document, location, step, label, num_points)


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
    selection_fields = {
        "first_crc_flip": "flip_count",
        "first_aided_crc_flip": "flip_count",
        "first_true_soft_failure": "failure_count",
    }
    if field := selection_fields.get(selection["rule"]):
        _require_fields(selection, f"{label}.selection", **{field: int})
    if selection["rule"] == "first_aided_crc_flip":
        _require_fields(selection, f"{label}.selection", seed=int, flip_frequency=_NUMBER)
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

    definition = CASES.get(name)
    required_maps = definition.detectors if definition else ("fused",)
    decode_maps = definition.decode_maps if definition else ("fused",)
    iterations = definition.iterations if definition else 0
    required_receivers = (
        "standard",
        "true_soft",
        "true_prb_soft",
        *(f"{m}_soft" for m in decode_maps),
        *((AIDED_RECEIVER,) if iterations else ()),
    )
    num_points = width * 12 * (symbols - 1 - cfg.get("dmrs_additional_position", 0))

    def check_frame(frame, location, batch_crc, crc_label):
        grids = frame["grids"]
        for field in ("allocation", "truth", "power_db", "residual_db", "evm_db"):
            _require_shape(grids.get(field), f"{location}.grids.{field}", prbs, symbols)
        maps = frame["detectors"]
        _require_fields(maps, f"{location}.detectors", **dict.fromkeys(required_maps, dict))
        for detector, layer in maps.items():
            label = f"{location}.detectors.{detector}"
            _require_fields(layer, label, scores=list, decisions=list, unscored=list)
            for field in ("scores", "decisions", "unscored"):
                _require_shape(layer[field], f"{label}.{field}", prbs, symbols)
        receivers = frame["receivers"]
        _require_fields(
            receivers, f"{location}.receivers", **dict.fromkeys(required_receivers, dict)
        )
        for rx, receiver in receivers.items():
            _check_symbol_refs(frame, location, receiver, f"{location}.receivers.{rx}", num_points)
            counts = batch_crc.get(rx)
            _require_fields(counts, f"{crc_label}.{rx}", slots=int, failures=int)

    check_frame(document, location, meta["batch_crc"], f"{location}.metadata.batch_crc")
    if "levels" in document or (definition and definition.levels):
        _check_levels(document, location, definition.levels if definition else None, check_frame)

    if "channel" in document or (definition and definition.include_channel):
        _require_fields(document, location, channel=dict)
        _check_channel(document["channel"], f"{location}.channel", width, cfg["dmrs_symbol"])
    if "passes" in document or iterations:
        _check_passes(document, location, width, num_points, iterations)

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
