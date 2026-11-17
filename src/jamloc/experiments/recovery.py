"""What the jamming map is good for: BLER with and without map-aware recovery.

The UE is scheduled on PRBs 6-17 (12 PRBs) and the jammer hits a block of 4 PRBs inside that
allocation. Every simulated batch is decoded by five receivers, on exactly the same slots:

* ``standard``      - Sionna's PUSCH receiver, no map (the unprotected link)
* ``lmmse_no_map``  - map-aware receiver with an empty map: shows what the LMMSE channel estimate
                      alone gives, so the gain of the map itself can be told apart
* ``map_p_e``       - map from the U-Net without the DMRS residual (power + EVM)
* ``map_p_r_e``     - map from the full U-Net (power + residual + EVM)
* ``true_map``      - the true jamming map (upper bound)

Detector maps are thresholded at the score that flags 1% of the non-jammed validation cells.
Two extra cases: slots without a jammer (what protection costs when false alarms erase clean
data), and pilot jamming of the whole allocation, where no clean pilot is left
(``standard`` and ``true_map`` only; by design they should match).

    python -m jamloc.experiments.recovery --channel A
"""

import argparse
import csv

import numpy as np
import torch
from sionna.phy import config

from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.features import extract
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import threshold_at_fa
from jamloc.model import select_inputs
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, plot_bler, save, zero_error_handle
from jamloc.recovery import MapAwareReceiver
from jamloc.train import load_model

ALLOC = (6, 12)  # UE on PRBs 6-17
JAM_SIZE = 4  # jammed PRBs, placed inside the allocation
RECEIVERS = ("standard", "lmmse_no_map", "map_p_e", "map_p_r_e", "true_map")
MAP_VARIANTS = {"map_p_e": "no_residual", "map_p_r_e": "fused"}
STYLE = {
    "standard": {"label": "No protection", "color": "#000000", "marker": "x", "linestyle": ":"},
    "lmmse_no_map": {
        "label": "LMMSE estimate, no map",
        "color": "#999999",
        "marker": "+",
        "linestyle": "--",
    },
    "map_p_e": {
        "label": "Map from U-Net P+E",
        "color": "#56B4E9",
        "marker": "v",
        "linestyle": "--",
    },
    "map_p_r_e": {
        "label": "Map from U-Net P+R+E",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "true_map": {"label": "True map", "color": "#009E73", "marker": "^", "linestyle": "-."},
}


def val_threshold(variant: str, seed: int = 0) -> float:
    """Logit threshold that flags 1% of the non-jammed validation cells."""
    d = load("val")
    model = load_model(variant, seed)
    features = torch.from_numpy(d["features"])
    with torch.no_grad():
        logits = torch.cat(
            [
                model(select_inputs(features[i].to(config.device), variant)).cpu()
                for i in torch.arange(len(features)).split(2048)
            ]
        )
    return threshold_at_fa(logits.numpy()[d["labels"] == 0], FA)


def detector_cells(batch, cfg, model, variant: str, thr: float) -> torch.Tensor:
    """Jamming map ``[B, 24, 14]`` predicted by a trained U-Net."""
    with torch.no_grad():
        return model(select_inputs(extract(batch, cfg), variant)) > thr


def _scenario(rng, n, channel, attack, jsr_db, snr_db, whole_allocation=False) -> Scenario:
    if whole_allocation:
        start, size = np.full(n, ALLOC[0]), np.full(n, ALLOC[1])
    else:
        start = rng.integers(ALLOC[0], ALLOC[0] + ALLOC[1] - JAM_SIZE + 1, n)
        size = np.full(n, JAM_SIZE)
    return Scenario(
        channel=channel,
        alloc_start=ALLOC[0],
        alloc_size=ALLOC[1],
        attack=torch.full((n,), ATTACKS.index(attack)),
        jam_start=torch.as_tensor(start),
        jam_size=torch.as_tensor(size),
        jsr_db=torch.full((n,), float(jsr_db)),
        snr_db=torch.full((n,), float(snr_db)),
    )


def simulate_point(link, detectors, scenario_fn, receivers, max_mc_iter, target_block_errors):
    """Bit and block errors of each receiver, all decoding the same simulated batches.

    A receiver stops once it has ``target_block_errors`` block errors (its estimate is precise
    enough); the others keep decoding further batches. So receivers that rarely fail get many
    slots without making the frequent failures expensive.
    """
    cfg = link.cfg
    alloc = slice(ALLOC[0], ALLOC[0] + ALLOC[1])
    counts = {rx: np.zeros(4) for rx in receivers}  # bit errors, bits, block errors, blocks
    active = list(receivers)
    map_aware = None
    for _ in range(max_mc_iter):
        if not active:
            break
        out = link(scenario_fn(), decode="standard" in active)
        map_aware = map_aware or MapAwareReceiver(out.tx, cfg)
        maps = {
            "lmmse_no_map": torch.zeros_like(out.labels[:, alloc]),
            "true_map": out.labels[:, alloc],
        }
        for rx, (model, variant, thr) in detectors.items():
            if rx in active:
                maps[rx] = detector_cells(out, cfg, model, variant, thr)[:, alloc]

        for rx in list(active):
            if rx == "standard":
                b_hat = out.rx["b_hat"]
            else:
                b_hat = map_aware(out.y_alloc, out.no, maps[rx])["b_hat"]
            errors = out.b != b_hat
            counts[rx] += [
                errors.sum().item(),
                errors.numel(),
                errors.any(-1).sum().item(),
                errors[..., 0].numel(),
            ]
            if counts[rx][2] >= target_block_errors:
                active.remove(rx)
    return counts


def simulate(channel, snr_db, jsr_dbs, batch_size, max_mc_iter, target_block_errors, seed):
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    detectors = {}
    for rx, variant in MAP_VARIANTS.items():
        model = load_model(variant, seed)
        detectors[rx] = (model, variant, val_threshold(variant, seed))
    rng = seed_everything(1000 + seed)

    # Clean slots first: what each receiver costs when there is no jammer (false alarms of a
    # detector map erase clean data and drop clean pilots)
    cases = [("none", False, RECEIVERS)]
    cases += [(attack, False, RECEIVERS) for attack in ("pilot", "data", "partial")]
    cases.append(("pilot", True, ("standard", "true_map")))  # limit case: no clean pilot left
    rows = []
    for attack, whole, receivers in cases:
        name = "pilot_whole_allocation" if whole else attack
        for jsr in jsr_dbs[:1] if attack == "none" else jsr_dbs:  # JSR is irrelevant then

            def scenario_fn(attack=attack, jsr=jsr, whole=whole):
                return _scenario(rng, batch_size, channel, attack, jsr, snr_db, whole)

            counts = simulate_point(
                link, detectors, scenario_fn, receivers, max_mc_iter, target_block_errors
            )
            for rx in receivers:
                bit_err, bits, block_err, blocks = counts[rx]
                rows.append(
                    {
                        "receiver": rx,
                        "attack": name,
                        "jsr_db": float("nan") if attack == "none" else float(jsr),
                        "snr_db": float(snr_db),
                        "ber": bit_err / bits,
                        "bler": block_err / blocks,
                        "block_errors": int(block_err),
                        "blocks": int(blocks),
                    }
                )
            print(
                f"[{channel}] {name:22s} JSR {jsr:+5.1f} dB: "
                + "  ".join(f"{rx} {counts[rx][2] / counts[rx][3]:.3f}" for rx in receivers),
                flush=True,
            )
    return rows


MAIN_RECEIVERS = ("standard", "map_p_r_e", "true_map")
ABLATION_RECEIVERS = ("standard", "lmmse_no_map", "map_p_e", "map_p_r_e")


def _protection_helps_from(rows, attack):
    """Smallest JSR from which the P+R+E map gives a lower BLER than no protection at every
    higher JSR (point estimates), or None."""

    def bler(rx):
        return {
            r["jsr_db"]: r["bler"] for r in rows if r["receiver"] == rx and r["attack"] == attack
        }

    std, mapped = bler("standard"), bler("map_p_r_e")
    start = None
    for j in sorted(std, reverse=True):
        if j in mapped and mapped[j] < std[j]:
            start = j
        else:
            break
    return start


def plot(rows, channel, receivers=MAIN_RECEIVERS, shade=True):
    fig, axes = new_figure(7.16, 2.9, ncols=3, sharey=True)
    floor = 1.0 / max(r["blocks"] for r in rows)
    for ax, attack in zip(axes, ("pilot", "data", "partial"), strict=True):
        start = _protection_helps_from(rows, attack) if shade else None
        if start is not None:
            hi = max(r["jsr_db"] for r in rows if r["attack"] == attack)
            ax.axvspan(start - 1.5, hi + 1.5, color="#009E73", alpha=0.08, lw=0)
            ax.text(start - 1.2, 0.5, "map helps", fontsize=6, color="#00704F", va="top")
        for rx in receivers:
            sel = [r for r in rows if r["receiver"] == rx and r["attack"] == attack]
            style = {k: v for k, v in STYLE[rx].items() if k != "label"}
            plot_bler(
                ax,
                [r["jsr_db"] for r in sel],
                [r["block_errors"] for r in sel],
                [r["blocks"] for r in sel],
                style,
                STYLE[rx]["label"],
            )
        ax.set(
            title=f"{attack} jamming",
            xlabel="JSR per jammed RE [dB]",
            yscale="log",
            ylim=(floor / 2, 1.5),
        )
    axes[0].set_ylabel(f"BLER (TDL-{channel}, SNR {rows[0]['snr_db']:g} dB)")
    handles = axes[0].get_legend_handles_labels()[0] + [zero_error_handle()]
    fig.legend(
        handles=handles, loc="outside lower center", ncols=min(len(handles), 4), frameon=False
    )
    benign = {r["receiver"]: r["bler"] for r in rows if r["attack"] == "none"}
    fig.suptitle(
        f"UE on {ALLOC[1]} PRBs, jammer on {JAM_SIZE} of them; bands: 95% confidence.\n"
        f"Without a jammer: BLER {benign.get('standard', float('nan')):.3f} unprotected, "
        f"{benign.get('map_p_r_e', float('nan')):.3f} with the P+R+E map"
    )
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", choices=("A", "C"), default="A")
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--jsr-db", type=float, nargs="+", default=[float(j) for j in range(-15, 10, 3)])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-mc-iter", type=int, default=100)
    p.add_argument("--target-block-errors", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="training seed of the detector models")
    p.add_argument("--plot-only", action="store_true", help="redraw from the saved CSV")
    args = p.parse_args(argv)

    out = results_dir() / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"recovery_{args.channel}"
    if args.plot_only:
        with open(stem.with_suffix(".csv"), encoding="utf-8", newline="") as f:
            text = ("receiver", "attack")
            rows = [
                {k: (v if k in text else float(v)) for k, v in r.items()} for r in csv.DictReader(f)
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
        main_fig = plot(rows, args.channel)  # the message: no protection vs map vs true map
        ablation = plot(
            rows, args.channel, ABLATION_RECEIVERS, shade=False
        )  # where gains come from
    save(main_fig, stem)
    save(ablation, stem.with_name(f"{stem.name}_ablation"))
    print(f"Wrote {stem}.csv and the main and ablation figures (.pdf/.png)")


if __name__ == "__main__":
    main()
