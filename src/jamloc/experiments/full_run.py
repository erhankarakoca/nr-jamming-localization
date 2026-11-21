"""The whole detection experiment in one command, ending in one zip file to download.

Meant for a Colab GPU; runs the same way on a laptop CPU (slower training):

    python -m jamloc.experiments.full_run --seeds 0 1 2

1. generate the train / val / test / sched datasets,
2. train every U-Net variant once per seed (with per-epoch loss histories),
3. evaluate on the test split, the scheduling-aware split, and two SNR ranges of the test split,
   with bootstrap intervals for the test split,
4. run the map-aware recovery experiment for TDL-A and TDL-C (seed-0 detectors),
5. draw the explanatory figures and the gallery page,
6. pack ``results/{eval,models,recovery,figures}`` into ``results/full_run.zip``.

The BER/BLER sweep, the feature-map figure and the CPU latency benchmark are separate scripts
(see docs/reproducing.md); latency in particular should be measured on the target CPU, not on Colab.
"""

import argparse
import time
import zipfile

from jamloc import evaluate
from jamloc.dataset import CORE_SPLITS, generate
from jamloc.experiments import figures, intervals, recovery
from jamloc.model import VARIANTS
from jamloc.paths import data_dir, results_dir
from jamloc.train import train

SMALL_SPLITS = {"train": 64, "val": 32, "test": 60, "sched": 60}  # for a quick smoke test
SNR_RANGES = ((5, 15), (15, 25))


def _elapsed(start: float) -> str:
    return f"{(time.perf_counter() - start) / 60:.1f} min"


def main(argv=None):
    p = argparse.ArgumentParser(description="Datasets, training, evaluation, zip.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--reuse-data", action="store_true", help="keep existing dataset files")
    p.add_argument("--small", action="store_true", help="tiny datasets and 1 epoch (smoke test)")
    args = p.parse_args(argv)
    start = time.perf_counter()

    # 1. Datasets
    for split in CORE_SPLITS:
        if args.reuse_data and (data_dir() / f"{split}.npz").is_file():
            print(f"[data] keeping existing {split}.npz")
            continue
        generate(split, SMALL_SPLITS[split] if args.small else None, batch_size=128)
    print(f"[data] done after {_elapsed(start)}")

    # 2. Training
    epochs = 1 if args.small else args.epochs
    for variant in VARIANTS:
        for seed in args.seeds:
            train(variant, epochs=epochs, batch_size=32 if args.small else 128, seed=seed)
    print(f"[train] done after {_elapsed(start)}")

    # 3. Evaluation
    evaluate.main(["--split", "test"])
    evaluate.main(["--split", "sched"])
    # Bootstrap intervals of the test split (the detector-output figure plots them)
    intervals.main(["--split", "test", "--boot", "20" if args.small else "1000"])
    if not args.small:
        for lo, hi in SNR_RANGES:
            evaluate.main(["--split", "test", "--snr-db", str(lo), str(hi)])
    print(f"[eval] done after {_elapsed(start)}")

    # 4. What the map is good for: recovery with the detectors' maps
    small = ["--batch-size", "8", "--max-mc-iter", "1", "--jsr-db", "0"] if args.small else []
    for channel in ("A", "C"):
        recovery.main(["--channel", channel, "--seeds", str(args.seeds[0]), *small])
    print(f"[recovery] done after {_elapsed(start)}")

    # 5. Explanatory figures and the gallery page
    figures.main([])
    print(f"[figures] done after {_elapsed(start)}")

    # 6. One file to download
    archive = results_dir() / "full_run.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("eval", "models", "recovery", "figures"):
            for path in sorted((results_dir() / folder).glob("*")):
                z.write(path, f"{folder}/{path.name}")
    print(f"Wrote {archive} after {_elapsed(start)}")


if __name__ == "__main__":
    main()
