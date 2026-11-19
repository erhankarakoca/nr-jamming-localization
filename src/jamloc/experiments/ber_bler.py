"""BER and BLER of the unprotected link under each attack, versus JSR.

The jammer hits every scheduled PRB (worst-case overlap), so the curves show how much damage each
attack pattern does. Panel (b) re-plots BLER against the jammer's total energy in the slot,
relative to the UE's data energy, which shows how much more efficient pilot jamming is. Results
at 15 dB SNR go to ``ber_bler_{channel}.csv``, at other SNRs to ``ber_bler_{channel}_snr{x}.csv``.

``--benign`` instead sweeps the SNR without a jammer (``benign_{channel}.csv``): the SNR at which
MCS 5 reaches 10% BLER, the usual target of link adaptation, tells how much margin the link has
left at the SNR of the jamming experiments.

    python -m jamloc.experiments.ber_bler --channel A --snr-db 15
    python -m jamloc.experiments.ber_bler --channel C --benign
"""

import argparse
import csv
import math

import numpy as np

from jamloc.config import SimConfig
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.paths import results_dir
from jamloc.plotting import (
    ATTACK_STYLE,
    COLUMN_WIDTH,
    new_figure,
    paper_style,
    plot_bler,
    save,
    zero_error_handle,
)

# OFDM symbols the jammer occupies per PRB; the UE sends data on 13 of the 14 symbols
JAMMED_SYMBOLS = {"pilot": 1, "data": 13, "partial": 14}
UE_DATA_SYMBOLS = 13


def energy_db(attack: str, jsr_db: float) -> float:
    """Total jammer energy in the slot relative to the UE's data energy [dB]."""
    return jsr_db + 10 * math.log10(JAMMED_SYMBOLS[attack] / UE_DATA_SYMBOLS)


def count_errors(mc_fun, max_mc_iter: int, target_block_errors: int) -> dict:
    """Simulate batches until ``target_block_errors`` block errors or ``max_mc_iter`` batches.

    ``mc_fun()`` returns ``(b, b_hat)``; a block is one transport block (last axis = its bits).
    The counts are kept so that confidence intervals can be drawn later.
    """
    bit_errors = bits = block_errors = blocks = 0
    for _ in range(max_mc_iter):
        b, b_hat = mc_fun()
        errors = b != b_hat
        bit_errors += errors.sum().item()
        bits += errors.numel()
        block_errors += errors.any(-1).sum().item()
        blocks += errors[..., 0].numel()
        if block_errors >= target_block_errors:
            break
    return {"bit_errors": bit_errors, "bits": bits, "block_errors": block_errors, "blocks": blocks}


def simulate(channel, snr_db, jsr_dbs, batch_size, max_mc_iter, target_block_errors, seed):
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    rng = seed_everything(seed)

    def run(attack: str, jsr_db: float) -> dict:
        def mc_fun():
            sc = draw_scenario(
                rng,
                batch_size,
                cfg,
                channel,
                attacks=(ATTACKS.index(attack),),
                jsr_db=float(jsr_db),
                snr_db=snr_db,
                scheduling_aware=True,
            )
            out = link(sc, decode=True)
            return out.b, out.rx["b_hat"]

        c = count_errors(mc_fun, max_mc_iter, target_block_errors)
        return {**c, "ber": c["bit_errors"] / c["bits"], "bler": c["block_errors"] / c["blocks"]}

    rows = [
        {
            "attack": "none",
            "snr_db": float(snr_db),
            "jsr_db": math.nan,
            "energy_db": math.nan,
            **run("none", 0.0),
        }
    ]  # benign reference at the same SNR
    for attack in ("pilot", "data", "partial"):
        for j in jsr_dbs:
            rows.append(
                {
                    "attack": attack,
                    "snr_db": float(snr_db),
                    "jsr_db": float(j),
                    "energy_db": energy_db(attack, j),
                    **run(attack, j),
                }
            )
            print(
                f"[{channel}] {attack:8s} JSR {j:+5.1f} dB: BLER {rows[-1]['bler']:.4f} "
                f"({rows[-1]['block_errors']}/{rows[-1]['blocks']})",
                flush=True,
            )
    return rows


def simulate_benign(channel, snr_dbs, batch_size, max_mc_iter, target_block_errors, seed):
    """BLER without a jammer, versus SNR (random allocations)."""
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    rng = seed_everything(seed)
    rows = []
    for snr in snr_dbs:

        def mc_fun(snr=snr):
            sc = draw_scenario(rng, batch_size, cfg, channel, attacks=(0,), snr_db=float(snr))
            out = link(sc, decode=True)
            return out.b, out.rx["b_hat"]

        c = count_errors(mc_fun, max_mc_iter, target_block_errors)
        rows.append({"snr_db": float(snr), **c, "bler": c["block_errors"] / c["blocks"]})
        print(f"[{channel}] no jammer, SNR {snr:+5.1f} dB: BLER {rows[-1]['bler']:.4f}", flush=True)
    return rows


def path_for(channel: str, snr_db: float = 15.0):
    """CSV of the sweep at one SNR (15 dB keeps the original name)."""
    name = f"ber_bler_{channel}" if snr_db == 15.0 else f"ber_bler_{channel}_snr{snr_db:g}"
    return results_dir() / "ber_bler" / f"{name}.csv"


def read_rows(path, default_snr_db: float = 15.0) -> list[dict]:
    """Rows of a saved sweep. Older CSVs have no ``snr_db`` column; they get ``default_snr_db``."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = [
            {k: (v if k == "attack" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)
        ]
    for r in rows:
        r.setdefault("snr_db", default_snr_db)
    return rows


def plot(rows, channel):
    snr_db = rows[0]["snr_db"]
    fig, (ax_jsr, ax_energy) = new_figure(COLUMN_WIDTH, 2.6, ncols=2, sharey=True)
    for attack in ("pilot", "data", "partial"):
        sel = [r for r in rows if r["attack"] == attack]
        errors, blocks = [r["block_errors"] for r in sel], [r["blocks"] for r in sel]
        plot_bler(ax_jsr, [r["jsr_db"] for r in sel], errors, blocks, ATTACK_STYLE[attack], attack)
        plot_bler(ax_energy, [r["energy_db"] for r in sel], errors, blocks, ATTACK_STYLE[attack])
    ax_jsr.set(
        yscale="log",
        xlabel="JSR per jammed RE [dB]",
        ylabel="BLER",
        title=f"(a) TDL-{channel}, SNR {snr_db:g} dB",
    )
    ax_energy.set(xlabel="Slot energy ratio J/S [dB]", title="(b) Same, vs. energy")
    # Legend below the panels, so it never covers a curve
    handles = ax_jsr.get_legend_handles_labels()[0]
    handles.append(zero_error_handle())
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", choices=("A", "C"), default="A")
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument(
        "--jsr-db", type=float, nargs="+", default=[float(j) for j in np.arange(-20.0, 10.1, 2.5)]
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-mc-iter", type=int, default=200)
    p.add_argument("--target-block-errors", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true", help="redraw from the saved CSV")
    p.add_argument("--benign", action="store_true", help="BLER vs SNR without a jammer")
    p.add_argument("--snr-sweep", type=float, nargs="+", default=[float(s) for s in range(-2, 17)])
    args = p.parse_args(argv)

    out_dir = results_dir() / "ber_bler"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.benign:
        rows = simulate_benign(
            args.channel,
            args.snr_sweep,
            args.batch_size,
            args.max_mc_iter,
            args.target_block_errors,
            args.seed,
        )
        path = out_dir / f"benign_{args.channel}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")
        return
    stem = path_for(args.channel, args.snr_db).with_suffix("")

    if args.plot_only:
        rows = read_rows(stem.with_suffix(".csv"), args.snr_db)
    else:
        rows = simulate(
            args.channel,
            args.snr_db,
            args.jsr_db,
            args.batch_size,
            args.max_mc_iter,
            args.target_block_errors,
            args.seed,
        )
        with open(stem.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with paper_style():
        fig = plot(rows, args.channel)
    save(fig, stem)
    print(f"Wrote {stem}.csv/.pdf/.png")


if __name__ == "__main__":
    main()
