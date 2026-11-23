"""What decoding does to the bits of jammed and clean cells.

For the showcase batches (seed 7, 64 slots, TDL-C, 15 dB SNR, jammer on PRBs 10-13 of 6-17), and
for each receiver, this counts the coded bits that are wrong before decoding (the signs of the
receiver's LLRs against the transmitted bits) in jammed and in clean cells, and how much weight
the decoder gets on right and on wrong bits (mean |LLR| per bit of that kind of cell, split by
whether the bit is right). After decoding it counts the CRC failures and the wrong information
bits. It shows why the map-aware receiver decodes although its jammed symbols stay scattered:
their wrong bits reach the decoder with almost no weight.

    python -m jamloc.experiments.decoding_bits      # -> results/site/decoding_bits.csv
"""

import csv
from dataclasses import replace

import torch

from jamloc.paths import results_dir
from jamloc.site.cases import CASES
from jamloc.site.export import simulate_case

POINTS = (("partial", 3), ("partial", 9), ("data", 3), ("pilot", 3))
RECEIVERS = ("standard", "fused_soft", "true_soft")


def rows(case_name: str, jsr_db: float) -> list[dict]:
    case = replace(CASES[case_name], jsr_db=jsr_db, levels=())
    run = simulate_case(case)
    batch = run.batch
    with torch.no_grad():
        sent = batch.tx._tb_encoder(batch.b).reshape(batch.b.shape[0], -1).long()
    slots, num_bits = sent.shape
    start, width = case.alloc
    dmrs = case.config().dmrs_symbols[0]
    index = torch.arange(num_bits // 2)
    ordinal, subcarrier = index // (12 * width), index % (12 * width)
    symbol = ordinal + (ordinal >= dmrs).long()
    jammed = batch.labels[:, start + subcarrier // 12, symbol].repeat_interleave(2, dim=1)
    receivers = {"standard": batch.rx, **{k: run.receivers[k] for k in RECEIVERS[1:]}}
    out = []
    for name, rx in receivers.items():
        llr = rx["llr"].reshape(slots, -1)
        wrong = (llr > 0).long() != sent  # Sionna's LLRs are log P(1)/P(0)
        weight = llr.abs()
        crc = rx["tb_crc"].reshape(-1).bool()
        info_wrong = (rx["b_hat"].reshape(slots, -1) != batch.b.reshape(slots, -1)).float().mean()
        for cells, mask in (("jammed", jammed), ("clean", ~jammed)):
            if not mask.any():
                continue
            count = mask.sum()
            out.append(
                {
                    "attack": case_name,
                    "jsr_db": jsr_db,
                    "receiver": name,
                    "cells": cells,
                    "bits": int(count),
                    "wrong_before": float((wrong & mask).sum() / count),
                    "llr_right": float((weight * (~wrong & mask)).sum() / count),
                    "llr_wrong": float((weight * (wrong & mask)).sum() / count),
                    "crc_failures": int((~crc).sum()),
                    "slots": slots,
                    "info_bits_wrong_after": float(info_wrong),
                }
            )
    return out


def main() -> None:
    table = [row for case_name, jsr_db in POINTS for row in rows(case_name, jsr_db)]
    path = results_dir() / "site" / "decoding_bits.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(
            f"{row['attack']:8s} {row['jsr_db']:+.0f} dB {row['receiver']:10s} {row['cells']:6s} "
            f"wrong {100 * row['wrong_before']:5.1f}%  |LLR| right {row['llr_right']:6.2f} "
            f"wrong {row['llr_wrong']:6.2f}  CRC failures {row['crc_failures']}/{row['slots']}"
        )
    print(path)


if __name__ == "__main__":
    main()
