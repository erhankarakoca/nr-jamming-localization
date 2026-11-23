"""Localization after decoding: a known-symbol residual on the data cells of decoded slots.

Once a transport block passes its CRC, the gNB knows every data symbol of the slot, so every data
resource element gives an LS value of the channel. With the channel constant over the slot, the
LS values of a cell minus their prediction from the other symbols of the same subcarriers have a
known law without jamming (Gamma(12, 1/12) for the cell's normalized energy with QPSK), whatever
the jammer's support. Outliers are peeled off first so that jammed cells do not leak into the
predictions (``MapAwareReceiver.decoded_cell_test``).

This script replays the slots of a test grid exactly (same seeds and jobs as ``dataset.generate``,
keeping the first ``--per-point`` batches of every grid point), runs the fused U-Net, decodes every
slot with the soft map-aware receiver and its data-aided passes, and compares maps cell by cell:

* ``unet``    - the U-Net map at its validation threshold (1% of clean validation cells)
* ``decoded`` - on the scheduled data cells of slots whose CRC passed, the leave-one-out test of
                the decoded symbols (``MapAwareReceiver.decoded_cell_test``) at its analytic
                ``--alpha`` threshold replaces the U-Net decision; every other cell keeps the
                U-Net decision
* ``union``   - the U-Net map plus the cells the known-symbol test flags
* ``genie``   - as ``decoded``, but with the transmitted symbols on every slot (what perfect
                decoding would allow), to separate the test from decoding failures

The counts (true and false positives, misses) per attack, JSR and map, over all cells and over the
scheduled cells, go to ``results/decoded_map/<split>_seed<s>.csv``.

    python -m jamloc.experiments.decoded_map --split test --seed 0 --per-point 4
    python -m jamloc.experiments.decoded_map --split hopjam --seed 0 --per-point 4
"""

import argparse
import csv
import dataclasses
import time
from collections import defaultdict

import numpy as np
import torch

from jamloc.config import SimConfig
from jamloc.dataset import GRID_BATCH_SIZE, SPLITS, _grid_jobs
from jamloc.experiments.recovery import detector_cells, val_threshold
from jamloc.jammer import ATTACKS as ATTACK_NAMES
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.paths import results_dir
from jamloc.recovery import MapAwareReceiver
from jamloc.train import load_model

MAPS = ("unet", "decoded", "union", "genie")


def _jobs(split: str, per_point: int):
    """(index, job) of the split's grid jobs, keeping the first ``per_point`` batches of every
    grid point. The index is the one ``dataset.generate`` seeds the batch with."""
    spec = SPLITS[split]
    jobs = _grid_jobs(spec, None, spec.get("batch_size", GRID_BATCH_SIZE))
    seen = defaultdict(int)
    for i, job in enumerate(jobs):
        key = (job["channel"], job["attacks"], job["jsr_db"])
        seen[key] += 1
        if seen[key] <= per_point:
            yield i, job


def run(split: str, seed: int, per_point: int, alpha: float, passes: int, refine: bool):
    spec = SPLITS[split]
    conditions = spec.get("mix", (spec.get("cfg", {}),))
    if len(conditions) > 1:
        raise ValueError("mixed-condition splits are not supported")
    cfg = dataclasses.replace(
        SimConfig(), **{k: v for k, v in conditions[0].items() if k[0] != "_"}
    )
    link = JammedPUSCHLink(cfg)
    model = load_model("fused", seed)
    thr = val_threshold("fused", model)
    receivers = {}
    counts = defaultdict(lambda: np.zeros(6, dtype=np.int64))  # tp fp fn, scheduled tp fp fn
    decoded = defaultdict(lambda: np.zeros(2, dtype=np.int64))  # slots decoded, slots
    start = time.perf_counter()
    done = 0
    for i, job in _jobs(split, per_point):
        rng = seed_everything(spec["seed"] + i)
        sc = draw_scenario(
            rng,
            job["n"],
            cfg,
            job["channel"],
            attacks=job["attacks"],
            jsr_db=job["jsr_db"],
            snr_db=job["snr_db"],
            scheduling_aware=job["scheduling_aware"],
            noise_error_db=job.get("noise_error_db", 0.0),
        )
        out = link(sc, decode=False)
        alloc = slice(sc.alloc_start, sc.alloc_start + sc.alloc_size)
        labels = out.labels.bool()
        unet = detector_cells(out, cfg, "fused", model, thr)
        rx = receivers.get(sc.alloc_size) or receivers.setdefault(
            sc.alloc_size, MapAwareReceiver(out.tx, cfg)
        )
        with torch.no_grad():
            res = rx(
                out.y_alloc,
                out.no,
                unet[:, alloc],
                mode="soft",
                iterations=passes,
                cellwise=refine,
                refine=refine,
            )
        crc = res["tb_crc"].reshape(-1).bool()
        test = rx.decoded_cell_test(out.y_alloc, out.no, rx.decoded_symbols(res["b_hat"]), alpha)[0]
        genie = rx.decoded_cell_test(out.y_alloc, out.no, rx.decoded_symbols(out.b), alpha)[0]
        data = torch.zeros_like(unet)
        data[:, alloc] = True
        data[:, :, list(cfg.dmrs_symbols)] = False  # scheduled data cells
        maps = {"unet": unet}
        for name, cells, ok in (("decoded", test, crc), ("genie", genie, torch.ones_like(crc))):
            carrier = unet.clone()
            region = data & ok[:, None, None]
            full = torch.zeros_like(unet)
            full[:, alloc] = cells
            carrier[region] = full[region]
            maps[name] = carrier
        union = unet.clone()
        full = torch.zeros_like(unet)
        full[:, alloc] = test & crc[:, None, None]
        maps["union"] = union | full

        sched = torch.zeros_like(unet)
        sched[:, alloc] = True
        attack = ATTACK_NAMES[int(sc.attack[0])]
        key = (job["channel"], attack, float(job["jsr_db"]) if attack != "none" else float("nan"))
        for name, m in maps.items():
            for j, region in enumerate((torch.ones_like(sched), sched)):
                tp = (m & labels & region).sum().item()
                fp = (m & ~labels & region).sum().item()
                fn = (~m & labels & region).sum().item()
                counts[(*key, name)][3 * j : 3 * j + 3] += [tp, fp, fn]
        decoded[key] += [int(crc.sum()), len(crc)]
        done += len(crc)
        if done % 512 < len(crc):
            print(
                f"[{split} seed {seed}] {done} slots, {time.perf_counter() - start:.0f} s",
                flush=True,
            )

    rows = []
    for (channel, attack, jsr, name), c in sorted(counts.items(), key=str):
        d = decoded[(channel, attack, jsr)]
        rows.append(
            {
                "split": split,
                "seed": seed,
                "channel": channel,
                "attack": attack,
                "jsr_db": jsr,
                "map": name,
                "tp": c[0],
                "fp": c[1],
                "fn": c[2],
                "sched_tp": c[3],
                "sched_fp": c[4],
                "sched_fn": c[5],
                "decoded": d[0],
                "slots": d[1],
            }
        )
    out_dir = results_dir() / "decoded_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_refine" if refine else ""
    path = out_dir / f"{split}_seed{seed}{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary(rows)
    print(path)
    return rows


def summary(rows):
    total = defaultdict(lambda: np.zeros(3))
    for r in rows:
        total[r["map"]] += [r["tp"], r["fp"], r["fn"]]
    for name in MAPS:
        tp, fp, fn = total[name]
        print(f"{name:8s} F1 {2 * tp / max(2 * tp + fp + fn, 1):.4f}  fp {int(fp)}  fn {int(fn)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-point", type=int, default=4, help="batches per grid point")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--refine", action="store_true", help="passes refine the map")
    args = p.parse_args(argv)
    run(args.split, args.seed, args.per_point, args.alpha, args.passes, args.refine)


if __name__ == "__main__":
    main()
