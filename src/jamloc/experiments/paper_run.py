"""Every dataset, model and result the paper uses, in stages that can be resumed.

The plan of review v6 (``docs/plan-v6.md``) changes the datasets (a seventh feature, the
empty-comb power; grids of small batches; new splits) and so every number. This script rebuilds
them all in one place, so that no result is left from an older version of the data:

1. ``data``   - generate every split that a later stage reads (existing files are kept, so a
               stopped run can be resumed; delete ``data/`` for a clean rebuild);
2. ``train``  - every model: the four U-Net variants, the U-Net with the empty-comb input, the
               per-cell MLPs (64 and 252 hidden units), the U-Net trained with noise jitter, on
               mixed channels, on mixed attacks (with and without spoofing), on the per-slot
               noise estimate, and the narrower U-Nets of the size sweep (existing checkpoints
               are kept);
3. ``eval``   - detection on the test and scheduling-aware grids, the unseen conditions, the
               intervals and paired differences, calibration, the residual's theory, mixed
               training and attacks, spoofing, the neighbor-cell interferer;
4. ``margin`` - BLER of the unprotected link (with and without jamming, several SNRs) and the
               detection margin at each SNR;
5. ``figures``- the explanatory figures and the gallery page;
6. ``zip``    - ``results/paper_run.zip`` with everything a later stage or the paper reads.

Data generation and training are meant for a GPU (Colab). The recovery experiment is not part of
this script: it is CPU-bound (its LMMSE solves run on the CPU) and runs best as parallel processes
on a multi-core machine (see the README). Latency must be measured on the target CPU.

    python -m jamloc.experiments.paper_run --stages data train eval margin figures zip
"""

import argparse
import importlib
import time
import zipfile

from jamloc.dataset import MARGIN_SNRS_DB, NOISE_ERRORS_DB, SPLITS, generate, margin_split
from jamloc.dataset import noise_split as _noise_split
from jamloc.model import VARIANTS
from jamloc.paths import data_dir, results_dir
from jamloc.train import PIXEL_HIDDEN, PIXEL_HIDDEN_MATCHED, checkpoint_path, train

STAGES = ("data", "train", "eval", "margin", "figures", "zip")
# (variant, options of train / checkpoint_path) of every model besides the four nominal U-Nets
EXTRA_MODELS = (
    ("fused_comb", {}),
    ("fused", {"arch": "pixel", "hidden": PIXEL_HIDDEN}),
    ("fused", {"arch": "pixel", "hidden": PIXEL_HIDDEN_MATCHED}),
    ("fused", {"jitter": 1.0}),
    ("fused", {"data": "atk"}),
    ("fused", {"data": "atk2"}),
    ("fused", {"data": "nest"}),
    *((v, {"data": "mix"}) for v in VARIANTS),  # experiments/robust_training
)
SIZE_WIDTHS = (4, 6, 8)  # narrower U-Nets of the size sweep (12 is the nominal one)


def _elapsed(start: float) -> str:
    return f"{(time.perf_counter() - start) / 60:.1f} min"


def all_splits() -> list[str]:
    """Every split that a stage reads (``neighbor`` only when its jammer exists)."""
    splits = [s for s in SPLITS if not s.startswith("margin") and not s.startswith("noise_")]
    splits += [margin_split(snr) for snr in (15.0, *MARGIN_SNRS_DB)]
    splits += [_noise_split(e) for e in NOISE_ERRORS_DB]
    return splits


def stage_data(small: int | None):
    for split in all_splits():
        if (data_dir() / f"{split}.npz").is_file():
            print(f"[data] keeping {split}.npz", flush=True)
            continue
        generate(split, small, batch_size=128)


def stage_train(seeds, epochs: int, small: bool):
    bs = 32 if small else 128
    jobs = [(v, {}) for v in VARIANTS] + list(EXTRA_MODELS)
    jobs += [("fused", {"width": w}) for w in SIZE_WIDTHS]
    for variant, options in jobs:
        for seed in seeds:
            if checkpoint_path(variant, seed, **options).is_file():
                continue
            train(variant, epochs=epochs, batch_size=bs, seed=seed, **options)


def _run(module: str, argv: list[str]):
    """Run ``jamloc.<module>.main(argv)``, reporting instead of stopping on an error, so that one
    failing experiment does not cost the others."""
    start = time.perf_counter()
    try:
        importlib.import_module(f"jamloc.{module}").main(argv)
        print(f"[run] {module} {' '.join(argv)}: {_elapsed(start)}", flush=True)
    except Exception as err:  # noqa: BLE001 - keep going, the log shows the traceback
        import traceback

        traceback.print_exc()
        print(f"[run] FAILED {module} {' '.join(argv)}: {err!r}", flush=True)


def stage_eval(seeds, small: bool):
    s = [str(x) for x in seeds]
    _run("evaluate", ["--split", "test"])
    _run("evaluate", ["--split", "sched"])
    for lo, hi in ((5, 15), (15, 25)):
        _run("evaluate", ["--split", "test", "--snr-db", str(lo), str(hi)])
    _run("experiments.generalization", [])
    for split in ("test", "sched", "combjam", "hopjam", "pulsejam", "tonejam", "spoofjam"):
        _run("experiments.intervals", ["--split", split, *(["--boot", "50"] if small else [])])
    _run("experiments.calibration", ["--seeds", *s])
    _run("experiments.calibration", ["--seeds", *s, "--estimator"])
    _run("experiments.residual_theory", [])
    _run("experiments.multislot", [])
    _run("experiments.robust_training", ["--seeds", *s])
    _run("experiments.attack_training", ["--seeds", *s])
    _run("experiments.spoofing", ["--seeds", *s])
    _run("experiments.size_sweep", ["--seeds", *s])
    _run("experiments.neighbor", [])


def stage_margin(seeds, small: bool):
    s = [str(x) for x in seeds]
    quick = ["--batch-size", "8", "--max-mc-iter", "1"] if small else []
    for channel in ("A", "C"):
        _run("experiments.ber_bler", ["--channel", channel, "--benign", *quick])
        for snr in (15.0, *MARGIN_SNRS_DB):
            _run("experiments.ber_bler", ["--channel", channel, "--snr-db", f"{snr:g}", *quick])
    for snr in (15.0, *MARGIN_SNRS_DB):
        _run("experiments.detection_margin", ["--seeds", *s, "--snr-db", f"{snr:g}"])


def stage_zip():
    archive = results_dir() / "paper_run.zip"
    folders = (
        "eval",
        "models",
        "figures",
        "generalization",
        "intervals",
        "calibration",
        "theory",
        "robust_training",
        "attack_training",
        "spoofing",
        "size_sweep",
        "neighbor",
        "ber_bler",
        "margin",
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            for path in sorted((results_dir() / folder).glob("*")):
                if path.is_file():
                    z.write(path, f"{folder}/{path.name}")
    print(f"Wrote {archive}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--small", type=int, default=None, help="tiny splits (smoke test): slots")
    args = p.parse_args(argv)
    start = time.perf_counter()
    epochs = 1 if args.small else args.epochs
    if "data" in args.stages:
        stage_data(args.small)
        print(f"[data] done after {_elapsed(start)}", flush=True)
    if "train" in args.stages:
        stage_train(args.seeds, epochs, bool(args.small))
        print(f"[train] done after {_elapsed(start)}", flush=True)
    if "eval" in args.stages:
        stage_eval(args.seeds, bool(args.small))
        print(f"[eval] done after {_elapsed(start)}", flush=True)
    if "margin" in args.stages:
        stage_margin(args.seeds, bool(args.small))
        print(f"[margin] done after {_elapsed(start)}", flush=True)
    if "figures" in args.stages:
        _run("experiments.figures", [])
    if "zip" in args.stages:
        stage_zip()
    print(f"[paper_run] finished after {_elapsed(start)}", flush=True)


if __name__ == "__main__":
    main()
