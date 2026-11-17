"""CPU latency per slot (batch of one): feature extraction plus the U-Net forward pass.

Feature extraction starts from what the receiver already has (received grid, equalized symbols
and their noise variances) and includes the power, DMRS residual and EVM maps and the masks. The
delay subspace is computed once per allocation size and cached, as a gNB would precompute it.
Timing does not depend on the trained weights, so untrained networks of the same size are used.
By default PyTorch runs on one thread: for a 24 x 14 input, multithreading only adds overhead
(the U-Net takes ~0.5 ms on one M1 Pro core and ~1.3 ms with 8 threads), and a gNB would give the
detector one core per slot.

    python -m jamloc.experiments.latency --reps 200
"""

import argparse
import csv
import time

import numpy as np
import torch
from sionna.phy import config

from jamloc.config import SimConfig
from jamloc.features import extract
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.model import VARIANTS, TinyUNet, select_inputs
from jamloc.paths import results_dir


def _median_ms(fn, reps: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def benchmark(reps: int, alloc_prb: int) -> list[dict]:
    """One row per detector variant (runs on whatever ``config.device`` is set to)."""
    cfg = SimConfig()
    rng = seed_everything(0)
    scenario = draw_scenario(rng, 1, cfg, "A", attacks=(1,), alloc=(0, alloc_prb))
    batch = JammedPUSCHLink(cfg)(scenario)

    rows = []
    with torch.inference_mode():
        feature_ms = _median_ms(lambda: extract(batch, cfg), reps)
        features = extract(batch, cfg)
        for variant, channels in VARIANTS.items():
            model = TinyUNet(len(channels)).eval()
            x = select_inputs(features, variant)
            model_ms = _median_ms(lambda m=model, x=x: m(x), reps)
            rows.append(
                {
                    "variant": variant,
                    "params": sum(q.numel() for q in model.parameters()),
                    "feature_ms": feature_ms,
                    "model_ms": model_ms,
                    "total_ms": feature_ms + model_ms,
                    "threads": torch.get_num_threads(),
                }
            )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--alloc-prb", type=int, default=24)
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = p.parse_args(argv)

    previous_device, previous_threads = config.device, torch.get_num_threads()
    config.device = "cpu"  # this benchmark is about CPU latency
    torch.set_num_threads(args.threads)
    try:
        rows = benchmark(args.reps, args.alloc_prb)
    finally:
        config.device = previous_device
        torch.set_num_threads(previous_threads)

    with open(results_dir() / "latency.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['variant']:12s} {r['params']:6d} params  features {r['feature_ms']:.3f} ms  "
            f"U-Net {r['model_ms']:.3f} ms  total {r['total_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
