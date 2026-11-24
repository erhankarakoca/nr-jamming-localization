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

* ``unet``  - the U-Net map at its validation threshold (1% of clean validation cells)
* ``post``  - analytic tests at level ``--alpha``. The unscheduled cells are tested on their
              received power, whose law without jamming is exact, in every slot. The data and DMRS
              cells of slots whose CRC passed are tested on their slot residual
              (``MapAwareReceiver.decoded_cell_test``). The allocation of a slot that failed keeps
              the U-Net decisions.
* ``fused`` - as ``post``, but with the U-Net map as a prior. A cell the U-Net flags is tested at
              the looser level ``--alpha-prior``, any other cell at ``--alpha``.
* ``genie`` - as ``post``, with the transmitted symbols on every slot (what perfect decoding would
              allow), to separate the tests from decoding failures

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

MAPS = ("unet", "post", "fused", "genie")
PEAKS = False  # combine each cell's sum test with a test on its largest error (see _with_peak)


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


def _slot_pvalues(rx, out, x_data) -> torch.Tensor:
    """P-values ``[B, n, num_symbols]`` of the data and DMRS cells of the allocation under the
    Gamma(m, 1/m) law of their slot residual (1 where a cell has no known symbol)."""
    _, r, count = rx.decoded_cell_test(out.y_alloc, out.no, x_data)
    m = count.to(r.dtype)
    pv = torch.special.gammaincc(m.clamp_min(1), m * torch.nan_to_num(r, nan=0.0))
    if PEAKS:
        pv = _with_peak(pv, torch.nan_to_num(rx.last_peaks, nan=0.0), m)
    return torch.where(count > 0, pv, torch.ones_like(pv))


def _with_peak(p_sum, peak, m):
    """Combined p-value of the sum test and the test on the largest of m unit exponentials, each
    at half the level (Bonferroni), so that a jammer on one subcarrier of a cell is not diluted."""
    p_peak = 1 - (1 - torch.exp(-peak)) ** m.clamp_min(1)
    return torch.clamp(2 * torch.minimum(p_sum, p_peak), max=1.0)


def _power_pvalues(y, no) -> torch.Tensor:
    """P-values ``[B, B_c, num_symbols]`` of the received power of every carrier cell without the
    UE's signal: its mean energy over the 12 subcarriers divided by N0 is Gamma(12, 1/12)."""
    unit = (y.abs() ** 2).reshape(y.shape[0], y.shape[1], -1, 12) / no[:, None, None, None]
    pv = torch.special.gammaincc(torch.tensor(12.0), unit.sum(-1))
    if PEAKS:
        pv = _with_peak(pv, unit.amax(-1), torch.tensor(12.0))
    return pv.transpose(1, 2)


def run(
    split: str,
    seed: int,
    per_point: int,
    alpha: float,
    passes: int,
    refine: bool,
    alpha_prior: float = 0.1,
):
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
                cellwise="shrink" if refine else False,
                refine=refine,
            )
        crc = res["tb_crc"].reshape(-1).bool()
        sched = torch.zeros_like(unet)
        sched[:, alloc] = True
        power = _power_pvalues(out.y, out.no)
        maps = {"unet": unet}
        for name, bits, ok in (
            ("post", res["b_hat"], crc),
            ("genie", out.b, torch.ones_like(crc)),
        ):
            pv = torch.where(sched, torch.ones_like(power), power)
            pv[:, alloc] = _slot_pvalues(rx, out, rx.decoded_symbols(bits))
            known = ~sched | ok[:, None, None]  # where the tests apply
            maps[name] = torch.where(known, pv < alpha, unet)
            if name == "post":
                level = torch.where(unet, torch.full_like(pv, alpha_prior), alpha)
                maps["fused"] = torch.where(known, pv < level, unet)

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
    suffix = ("_refine" if refine else "") + ("_peaks" if PEAKS else "")
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


def table(paths) -> list[dict]:
    """F1 of every map per grid, over all cells of the replayed slots, for each seed's CSV, and
    the false-alarm rate of every map over the cells of the grid's benign slots."""
    import re

    out = []
    per = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))  # (split, map) -> seed -> counts
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            seed = int(re.search(r"_seed(\d+)", str(path)).group(1))
            c = per[(r["split"], r["map"])][seed]
            c[:3] += [int(r["tp"]), int(r["fp"]), int(r["fn"])]
            if r["attack"] == "none":
                c[3] += int(r["fp"])
    benign_cells = defaultdict(int)
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["attack"] == "none" and r["map"] == "unet":
                    benign_cells[(r["split"], path)] += int(r["slots"]) * 336
    for (split, name), seeds in sorted(per.items()):
        f1 = {s: 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1) for s, c in seeds.items()}
        cells = sum(v for (sp, _), v in benign_cells.items() if sp == split)
        fa = sum(c[3] for c in seeds.values()) / cells if cells else float("nan")
        out.append(
            {
                "split": split,
                "map": name,
                "seeds": len(f1),
                "f1_mean": float(np.mean(list(f1.values()))),
                "f1_min": min(f1.values()),
                "f1_max": max(f1.values()),
                "benign_fa": fa,
            }
        )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument(
        "--table", nargs="+", metavar="CSV", help="pool these per-seed CSVs into a table and stop"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-point", type=int, default=4, help="batches per grid point")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument(
        "--alpha-prior", type=float, default=0.1, help="test level on cells the U-Net flags"
    )
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--refine", action="store_true", help="passes refine the map (shrinkage jamming estimates)")
    p.add_argument("--peaks", action="store_true", help="also test the largest error of each cell")
    args = p.parse_args(argv)
    if args.table:
        rows = table(args.table)
        for r in rows:
            print(
                f"{r['split']:10s} {r['map']:6s} F1 {r['f1_mean']:.3f} "
                f"({r['f1_min']:.3f}-{r['f1_max']:.3f}, {r['seeds']} seeds)  "
                f"benign FA {100 * r['benign_fa']:.2f}%"
            )
        out = results_dir() / "decoded_map" / "table.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(out)
        return
    global PEAKS
    PEAKS = args.peaks
    run(
        args.split,
        args.seed,
        args.per_point,
        args.alpha,
        args.passes,
        args.refine,
        args.alpha_prior,
    )


if __name__ == "__main__":
    main()
