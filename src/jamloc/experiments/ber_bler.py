"""BER and BLER of the unprotected link under each attack, versus JSR.

The jammer hits every scheduled PRB (worst-case overlap), so the curves show how much damage each
attack pattern does. Panel (b) re-plots BLER against the jammer's total energy in the slot,
relative to the UE's data energy, which shows how much more efficient pilot jamming is.

    python -m jamloc.experiments.ber_bler --channel A --snr-db 15
"""

import argparse
import csv
import math

import numpy as np
import torch
from sionna.phy.utils import sim_ber

from jamloc.config import SimConfig
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.paths import results_dir
from jamloc.plotting import ATTACK_STYLE, COLUMN_WIDTH, new_figure, paper_style, save

# OFDM symbols the jammer occupies per PRB; the UE sends data on 13 of the 14 symbols
JAMMED_SYMBOLS = {"pilot": 1, "data": 13, "partial": 14}
UE_DATA_SYMBOLS = 13


def energy_db(attack: str, jsr_db: float) -> float:
    """Total jammer energy in the slot relative to the UE's data energy [dB]."""
    return jsr_db + 10 * math.log10(JAMMED_SYMBOLS[attack] / UE_DATA_SYMBOLS)


def simulate(channel, snr_db, jsr_dbs, batch_size, max_mc_iter, target_block_errors, seed):
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    rng = seed_everything(seed)

    def run(attack: str, sweep_db):
        # sim_ber sweeps its "ebno_db" argument; here that argument is the JSR
        def mc_fun(batch_size, ebno_db):
            sc = draw_scenario(
                rng,
                batch_size,
                cfg,
                channel,
                attacks=(ATTACKS.index(attack),),
                jsr_db=float(ebno_db),
                snr_db=snr_db,
                scheduling_aware=True,
            )
            out = link(sc, decode=True)
            return out.b, out.rx["b_hat"]

        ber, bler = sim_ber(
            mc_fun,
            torch.tensor(sweep_db, dtype=torch.float32),
            batch_size,
            max_mc_iter,
            num_target_block_errors=target_block_errors,
            early_stop=False,
            verbose=False,
        )
        return ber.cpu().numpy(), bler.cpu().numpy()

    rows = []
    ber, bler = run("none", [0.0])  # benign reference at the same SNR
    rows.append(
        {
            "attack": "none",
            "jsr_db": math.nan,
            "energy_db": math.nan,
            "ber": float(ber[0]),
            "bler": float(bler[0]),
        }
    )
    for attack in ("pilot", "data", "partial"):
        ber, bler = run(attack, jsr_dbs)
        for j, b1, b2 in zip(jsr_dbs, ber, bler):
            rows.append(
                {
                    "attack": attack,
                    "jsr_db": float(j),
                    "energy_db": energy_db(attack, j),
                    "ber": float(b1),
                    "bler": float(b2),
                }
            )
    return rows


def plot(rows, channel, snr_db):
    fig, (ax_jsr, ax_energy) = new_figure(COLUMN_WIDTH, 2.2, ncols=2, sharey=True)
    for attack in ("pilot", "data", "partial"):
        sel = [r for r in rows if r["attack"] == attack]
        # BLER = 0 cannot be drawn on a log axis; those points are left as gaps
        bler = np.array([r["bler"] if r["bler"] > 0 else np.nan for r in sel])
        style = ATTACK_STYLE[attack]
        ax_jsr.plot([r["jsr_db"] for r in sel], bler, label=attack, **style)
        ax_energy.plot([r["energy_db"] for r in sel], bler, **style)

    benign = next(r for r in rows if r["attack"] == "none")["bler"]
    if benign > 0:
        for ax in (ax_jsr, ax_energy):
            ax.axhline(benign, color="black", linestyle=":", linewidth=0.8, label="no jammer")

    ax_jsr.set(
        yscale="log",
        xlabel="JSR per jammed RE [dB]",
        ylabel="BLER",
        title=f"(a) TDL-{channel}, SNR {snr_db:g} dB",
    )
    ax_energy.set(xlabel="Slot energy ratio J/S [dB]", title="(b) Same, vs. energy")
    ax_jsr.legend(loc="lower right")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", choices=("A", "C"), default="A")
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument(
        "--jsr-db", type=float, nargs="+", default=[float(j) for j in np.arange(-20.0, 10.1, 2.5)]
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-mc-iter", type=int, default=20)
    p.add_argument("--target-block-errors", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true", help="redraw from the saved CSV")
    args = p.parse_args(argv)

    out_dir = results_dir() / "ber_bler"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"ber_bler_{args.channel}"

    if args.plot_only:
        with open(stem.with_suffix(".csv"), encoding="utf-8", newline="") as f:
            rows = [
                {k: (v if k == "attack" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)
            ]
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
        fig = plot(rows, args.channel, args.snr_db)
    save(fig, stem)
    print(f"Wrote {stem}.csv/.pdf/.png")


if __name__ == "__main__":
    main()
