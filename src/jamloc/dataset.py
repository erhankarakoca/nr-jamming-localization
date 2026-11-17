"""Labelled datasets of feature maps, generated from the link with fixed seeds.

Splits (batch ``i`` of a split uses seed ``base + i``, so the splits never share channel
realizations, allocations or jammer draws):

* ``train`` / ``val`` - random attack (incl. benign), JSR in [-15, 10] dB, SNR in [5, 25] dB,
  TDL-A and TDL-C alternating per batch, jammer anywhere on the carrier.
* ``test`` - grid over channel x attack x JSR (-15 ... 9 dB in 3 dB steps), plus benign slots.
* ``sched`` - same grid, but the jammer hits exactly the scheduled PRBs (never seen in training).

Each split is one ``.npz`` file in ``data_dir()``: ``features`` ``[N, 6, 24, 14]`` (stored as
float16), ``labels`` ``[N, 24, 14]`` and one value per slot for every key in ``META_KEYS``.

    python -m jamloc.dataset --split train
"""

import argparse
import math
import time

import numpy as np

from jamloc.config import SimConfig
from jamloc.features import extract
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.paths import data_dir

CHANNELS = ("A", "C")
TEST_JSR_DB = tuple(float(j) for j in range(-15, 10, 3))
META_KEYS = (
    "attack",
    "jsr_db",
    "snr_db",
    "channel",
    "alloc_start",
    "alloc_size",
    "jam_start",
    "jam_size",
    "scheduling_aware",
)

SPLITS = {
    "train": {"seed": 1_000_000, "num_slots": 40_000},
    "val": {"seed": 2_000_000, "num_slots": 5_000},
    "test": {"seed": 3_000_000, "per_point": 500, "benign_per_channel": 2_000, "sched": False},
    "sched": {"seed": 4_000_000, "per_point": 300, "benign_per_channel": 0, "sched": True},
}


def _random_jobs(num_slots: int, batch_size: int) -> list[dict]:
    jobs = []
    for i in range(math.ceil(num_slots / batch_size)):
        n = min(batch_size, num_slots - i * batch_size)
        jobs.append(
            {
                "n": n,
                "channel": CHANNELS[i % 2],
                "attacks": (0, 1, 2, 3),
                "jsr_db": (-15.0, 10.0),
                "scheduling_aware": False,
            }
        )
    return jobs


def _grid_jobs(spec: dict, num_slots, batch_size: int) -> list[dict]:
    per_point, benign = spec["per_point"], spec["benign_per_channel"]
    if num_slots is not None:  # smaller version of the same grid
        scale = num_slots / (len(CHANNELS) * 3 * len(TEST_JSR_DB) * per_point)
        per_point = max(1, int(per_point * scale))
        benign = max(1, round(benign * scale)) if benign else 0

    def batches(total, **kw):
        return [dict(n=min(batch_size, total - s), **kw) for s in range(0, total, batch_size)]

    jobs = []
    for channel in CHANNELS:
        for attack in (1, 2, 3):
            for jsr in TEST_JSR_DB:
                jobs += batches(
                    per_point,
                    channel=channel,
                    attacks=(attack,),
                    jsr_db=jsr,
                    scheduling_aware=spec["sched"],
                )
        jobs += batches(benign, channel=channel, attacks=(0,), jsr_db=0.0, scheduling_aware=False)
    return jobs


def generate(split: str, num_slots: int | None = None, batch_size: int = 64):
    """Simulate a split and save it to ``data_dir()/{split}.npz``. Returns the file path."""
    spec = SPLITS[split]
    if "per_point" in spec:
        jobs = _grid_jobs(spec, num_slots, batch_size)
    else:
        jobs = _random_jobs(num_slots or spec["num_slots"], batch_size)

    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    features, labels, meta = [], [], {k: [] for k in META_KEYS}
    start = time.perf_counter()
    for i, job in enumerate(jobs):
        rng = seed_everything(spec["seed"] + i)
        sc = draw_scenario(
            rng,
            job["n"],
            cfg,
            job["channel"],
            attacks=job["attacks"],
            jsr_db=job["jsr_db"],
            scheduling_aware=job["scheduling_aware"],
        )
        batch = link(sc)
        features.append(extract(batch, cfg).cpu().numpy().astype(np.float16))
        labels.append(batch.labels.cpu().numpy().astype(np.uint8))

        n = sc.batch_size
        meta["attack"].append(sc.attack.numpy())
        meta["jsr_db"].append(sc.jsr_db.numpy())
        meta["snr_db"].append(sc.snr_db.numpy())
        meta["channel"].append(np.full(n, CHANNELS.index(sc.channel)))
        meta["alloc_start"].append(np.full(n, sc.alloc_start))
        meta["alloc_size"].append(np.full(n, sc.alloc_size))
        meta["jam_start"].append(sc.jam_start.numpy())
        meta["jam_size"].append(sc.jam_size.numpy())
        meta["scheduling_aware"].append(np.full(n, job["scheduling_aware"]))

        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            done = sum(len(f) for f in features)
            rate = done / (time.perf_counter() - start)
            print(f"[{split}] {done} slots, {rate:.0f} slots/s", flush=True)

    path = data_dir() / f"{split}.npz"
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        **{k: np.concatenate(v) for k, v in meta.items()},
    )
    return path


def load(split: str) -> dict[str, np.ndarray]:
    """Load a split; features are returned as float32."""
    with np.load(data_dir() / f"{split}.npz") as f:
        d = {k: f[k] for k in f.files}
    d["features"] = d["features"].astype(np.float32)
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a dataset split.")
    p.add_argument("--split", choices=list(SPLITS), required=True)
    p.add_argument("--num-slots", type=int, default=None, help="smaller split (for testing)")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    print("Wrote", generate(args.split, args.num_slots, args.batch_size))


if __name__ == "__main__":
    main()
