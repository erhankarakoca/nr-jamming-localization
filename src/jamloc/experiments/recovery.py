"""What the jamming map is good for: BLER with and without map-aware recovery.

The UE is scheduled on PRBs 6-17 (12 PRBs) and the jammer hits a block of 4 PRBs inside that
allocation. Every simulated batch is decoded by every receiver, on exactly the same slots
(``jamloc.recovery`` explains the modes):

* ``standard``    - Sionna's PUSCH receiver, no map (the unprotected link)
* ``lmmse``       - no map: the LMMSE interpolator of the map-aware receivers, nothing weighted.
                    Separates the gain of the better interpolator from the gain of the map.
* ``slot_ie``     - no map: noise plus interference measured once per slot (mean residual for the
                    pilots, mean excess energy for the data), what a receiver without localization
                    can do
* ``dmrs_ie``     - no map: interference detected and measured on the DMRS only (analytic CFAR
                    test on the residual), assumed to last the whole slot
* ``power_hard``  - map from a per-cell power test, pilots dropped and data erased (the classic
                    receiver of Li et al., MILCOM 2012)
* ``block_soft``  - map from the block test (one decision per PRB for its DMRS and one for its
                    data symbols, no learned parameters), soft: PRB resolution
* ``p_e_soft``    - map from the U-Net without the DMRS residual (power + EVM), soft
* ``p_r_e_hard``  - map from the full U-Net (power + residual + EVM), hard
* ``p_r_e_soft``  - map from the full U-Net, soft (the proposed receiver)
* ``true_hard`` / ``true_soft`` - the true jamming map (where the jammer is; its strength is still
                    estimated in the soft mode)

With the receiver held fixed (soft mode), ``slot_ie`` -> ``dmrs_ie`` -> ``block_soft`` ->
``p_r_e_soft`` -> ``true_soft`` changes only where the jamming is assumed to be: nowhere in
particular, per PRB from the DMRS, per PRB and part of the slot, per cell, exactly.

Detector maps are thresholded at the score that flags 1% of the non-jammed validation cells.
Two extra cases: slots without a jammer (what protection costs when false alarms hit clean
cells), and pilot jamming of the whole allocation, where no clean pilot is left. ``--jammer``
replaces the nominal jammer by a pulsed one (each data symbol jammed with probability 0.5) or a
hopping one (new PRB block on every OFDM symbol), where per-PRB decisions are too coarse in time.
``--seeds`` runs the detectors of several training seeds on different slots; the figures pool
their block errors.

    python -m jamloc.experiments.recovery --channel A
    python -m jamloc.experiments.recovery --channel C --jammer pulsed --seeds 0 1 2
"""

import argparse
import csv
import dataclasses

import numpy as np
import torch
from sionna.phy import config

from jamloc.baselines import block_test_scores
from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.features import extract
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import threshold_at_fa
from jamloc.model import select_inputs
from jamloc.noise import estimated_snr_db
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, plot_bler, save, zero_error_handle
from jamloc.recovery import MapAwareReceiver
from jamloc.train import load_model

ALLOC = (6, 12)  # UE on PRBs 6-17
JAM_SIZE = 4  # jammed PRBs, placed inside the allocation
# receiver: (where its map comes from, mode of the map-aware receiver)
RECEIVERS = {
    "standard": (None, None),
    "lmmse": (None, "lmmse"),
    "slot_ie": (None, "slot"),
    "dmrs_ie": (None, "dmrs"),
    "power_hard": ("power_threshold", "hard"),
    "block_soft": ("block_test", "soft"),
    "p_e_soft": ("no_residual", "soft"),
    "p_r_e_hard": ("fused", "hard"),
    "p_r_e_soft": ("fused", "soft"),
    "true_hard": ("true", "hard"),
    "true_soft": ("true", "soft"),
}
DETECTORS = ("power_threshold", "block_test", "no_residual", "fused")
# Jammers: the nominal one, or one whose support is not a block of whole PRB rows
JAMMERS = {
    "nominal": {},
    "pulsed": {"jammer_duty": 0.5},
    "hopping": {"jammer_hopping": True},
}
STYLE = {
    "standard": {"label": "No protection", "color": "#000000", "marker": "x", "linestyle": ":"},
    "lmmse": {
        "label": "LMMSE, no map",
        "color": "#777777",
        "marker": "1",
        "linestyle": ":",
    },
    "slot_ie": {
        "label": "Slot-wide interference est. (no map)",
        "color": "#BBBBBB",
        "marker": "2",
        "linestyle": "--",
    },
    "block_soft": {
        "label": "Block-test map (per PRB), soft",
        "color": "#882255",
        "marker": "P",
        "linestyle": "-.",
    },
    "dmrs_ie": {
        "label": "DMRS-only interference est. (no map)",
        "color": "#999999",
        "marker": "+",
        "linestyle": "--",
    },
    "power_hard": {
        "label": "Power-test map, hard (Li et al.)",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "-.",
    },
    "p_e_soft": {"label": "Map P+E, soft", "color": "#56B4E9", "marker": "v", "linestyle": "--"},
    "p_r_e_hard": {"label": "Map P+R+E, hard", "color": "#CC79A7", "marker": "D", "linestyle": ":"},
    "p_r_e_soft": {"label": "Map P+R+E, soft", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "true_hard": {"label": "True map, hard", "color": "#D55E00", "marker": "<", "linestyle": ":"},
    "true_soft": {"label": "True map, soft", "color": "#009E73", "marker": "^", "linestyle": "-."},
}


def _power_scores(features: torch.Tensor) -> torch.Tensor:
    """Per-cell power test: received power of each scheduled cell relative to the slot's median
    scheduled cell, -inf elsewhere (the same score as ``baselines.power_scores``).

    The threshold comes from validation slots of 5-25 dB SNR; without the per-slot reference it
    would be set by the strongest slots and flag almost nothing at the 15 dB used here.
    """
    from jamloc.features import FEATURE_NAMES

    power = 10 ** (features[:, FEATURE_NAMES.index("power_db")] / 10)
    alloc = features[:, FEATURE_NAMES.index("alloc_mask")] > 0
    level = torch.nanmedian(torch.where(alloc, power, torch.nan).flatten(1), dim=1).values
    relative = power / level[:, None, None]
    return torch.where(alloc, relative, torch.full_like(power, -torch.inf))


def _scores(detector: str, model, features: torch.Tensor) -> torch.Tensor:
    if detector == "power_threshold":
        return _power_scores(features)
    return model(select_inputs(features, detector))


def val_threshold(detector: str, model=None) -> float:
    """Score threshold that flags 1% of the non-jammed validation cells."""
    d = load("val")
    if detector == "block_test":
        scores = block_test_scores(d, d)
        return threshold_at_fa(scores[d["labels"] == 0], FA)
    features = torch.from_numpy(d["features"])
    with torch.no_grad():
        scores = torch.cat(
            [
                _scores(detector, model, features[i].to(config.device)).cpu()
                for i in torch.arange(len(features)).split(2048)
            ]
        )
    return threshold_at_fa(scores.numpy()[d["labels"] == 0], FA)


def detector_cells(batch, cfg, detector: str, model, thr: float) -> torch.Tensor:
    """Jamming map ``[B, 24, 14]`` of a detector. For the block test, ``model`` is the loaded
    ``val`` split whose benign slots give its empirical nulls."""
    with torch.no_grad():
        features = extract(batch, cfg)
        if detector != "block_test":
            return _scores(detector, model, features) > thr
        n, sc = len(features), batch.scenario
        d = {
            "features": features.cpu().numpy(),
            "alloc_start": np.full(n, sc.alloc_start),
            "alloc_size": np.full(n, sc.alloc_size),
            "snr_db": sc.snr_db.numpy(),
            "snr_db_est": estimated_snr_db(batch.y_alloc, batch.no, cfg).cpu().numpy(),
        }
        return torch.from_numpy(block_test_scores(d, model) > thr).to(features.device)


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
        maps = {"true": out.labels[:, alloc]}
        needed = {RECEIVERS[rx][0] for rx in active}
        for det, (model, thr) in detectors.items():
            if det in needed:
                maps[det] = detector_cells(out, cfg, det, model, thr)[:, alloc]

        for rx in list(active):
            source, mode = RECEIVERS[rx]
            if rx == "standard":
                b_hat = out.rx["b_hat"]
            else:
                cells = maps[source] if source else None
                b_hat = map_aware(out.y_alloc, out.no, cells, mode=mode)["b_hat"]
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


def simulate(
    channel,
    snr_db,
    jsr_dbs,
    batch_size,
    max_mc_iter,
    target_block_errors,
    seed,
    only=None,
    jammer="nominal",
):
    cfg = dataclasses.replace(SimConfig(), **JAMMERS[jammer])
    link = JammedPUSCHLink(cfg)
    detectors = {}
    for det in DETECTORS:
        if det == "power_threshold":
            model = None
        elif det == "block_test":
            model = load("val")  # its empirical nulls
        else:
            model = load_model(det, seed)
        detectors[det] = (model, val_threshold(det, model))
    rng = seed_everything(1000 + seed)

    # Clean slots first: what each receiver costs when there is no jammer (false alarms of a
    # detector map erase clean data and drop clean pilots)
    if jammer == "nominal":
        cases = [("none", False, tuple(RECEIVERS))]
        cases += [(attack, False, tuple(RECEIVERS)) for attack in ("pilot", "data", "partial")]
        # limit case: no clean pilot left
        cases.append(("pilot", True, ("standard", "true_hard", "true_soft")))
    else:  # a pulsed or hopping pilot attack is the nominal one (it has a single OFDM symbol)
        cases = [(attack, False, tuple(RECEIVERS)) for attack in ("data", "partial")]
    if only:  # re-run some receivers only
        cases = [(a, w, tuple(rx for rx in rxs if rx in only)) for a, w, rxs in cases]
        cases = [c for c in cases if c[2]]
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
                        "jammer": jammer,
                        "seed": seed,
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
                f"[{channel} {jammer} seed {seed}] {name:22s} JSR {jsr:+5.1f} dB: "
                + "  ".join(f"{rx} {counts[rx][2] / counts[rx][3]:.3f}" for rx in receivers),
                flush=True,
            )
    return rows


MAIN_RECEIVERS = ("standard", "dmrs_ie", "power_hard", "p_r_e_soft", "true_soft")
ABLATION_RECEIVERS = ("standard", "p_r_e_hard", "p_r_e_soft", "p_e_soft", "true_hard", "true_soft")
# The receiver held fixed (soft weighting), only the source of the map changes
SOURCE_RECEIVERS = ("lmmse", "slot_ie", "dmrs_ie", "block_soft", "p_r_e_soft", "true_soft")


def pool_seeds(rows) -> list[dict]:
    """Sum the errors of the runs with different training seeds (and slots) per receiver, attack
    and JSR: the BLER of the receiver with a detector trained from a random seed."""
    pooled = {}
    for r in rows:
        key = (r["receiver"], r.get("jammer", "nominal"), r["attack"], r["jsr_db"], r["snr_db"])
        if isinstance(r["jsr_db"], float) and np.isnan(r["jsr_db"]):
            key = (r["receiver"], r.get("jammer", "nominal"), r["attack"], "none", r["snr_db"])
        p = pooled.setdefault(key, {**r, "block_errors": 0, "blocks": 0, "bit_errors": 0.0})
        p["block_errors"] += int(r["block_errors"])
        p["blocks"] += int(r["blocks"])
    out = []
    for p in pooled.values():
        p["bler"] = p["block_errors"] / max(p["blocks"], 1)
        p.pop("bit_errors", None)
        p.pop("seed", None)
        out.append(p)
    return out


def _protection_helps_from(rows, attack):
    """Smallest JSR from which the soft P+R+E receiver gives a lower BLER than no protection at
    every higher JSR (point estimates), or None."""

    def bler(rx):
        return {
            r["jsr_db"]: r["bler"] for r in rows if r["receiver"] == rx and r["attack"] == attack
        }

    std, mapped = bler("standard"), bler("p_r_e_soft")
    start = None
    for j in sorted(std, reverse=True):
        if j in mapped and mapped[j] < std[j]:
            start = j
        else:
            break
    return start


def plot(rows, channel, receivers=MAIN_RECEIVERS, shade=True):
    attacks = [a for a in ("pilot", "data", "partial") if any(r["attack"] == a for r in rows)]
    fig, axes = new_figure(7.16 * len(attacks) / 3, 2.9, ncols=len(attacks), sharey=True)
    axes = np.atleast_1d(axes)
    floor = 1.0 / max(r["blocks"] for r in rows)
    for ax, attack in zip(axes, attacks, strict=True):
        start = _protection_helps_from(rows, attack) if shade else None
        if start is not None:
            hi = max(r["jsr_db"] for r in rows if r["attack"] == attack)
            ax.axvspan(start - 1.5, hi + 1.5, color="#009E73", alpha=0.08, lw=0)
            ax.text(start - 1.2, 0.5, "map helps", fontsize=6, color="#00704F", va="top")
        for rx in receivers:
            sel = [r for r in rows if r["receiver"] == rx and r["attack"] == attack]
            if not sel:
                continue
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
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    benign = {r["receiver"]: r["bler"] for r in rows if r["attack"] == "none"}
    jammer = rows[0].get("jammer", "nominal")
    title = f"UE on {ALLOC[1]} PRBs, {jammer} jammer on {JAM_SIZE} of them; bands: 95% confidence."
    if benign:
        title += (
            f"\nWithout a jammer: BLER {benign.get('standard', float('nan')):.3f} unprotected, "
            f"{benign.get('p_r_e_soft', float('nan')):.3f} with the soft P+R+E map"
        )
    fig.suptitle(title)
    return fig


def _read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        text = ("receiver", "attack", "jammer")
        return [
            {k: (v if k in text else float(v)) for k, v in r.items()} for r in csv.DictReader(f)
        ]


def rows_rx(rows) -> set:
    return {r["receiver"] for r in rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", choices=("A", "C"), default="A")
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--jsr-db", type=float, nargs="+", default=[float(j) for j in range(-15, 10, 3)])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-mc-iter", type=int, default=100)
    p.add_argument("--target-block-errors", type=int, default=100)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="training seeds of the detector models"
    )
    p.add_argument("--jammer", choices=list(JAMMERS), default="nominal")
    p.add_argument("--plot-only", action="store_true", help="redraw from the saved CSV")
    p.add_argument(
        "--receivers",
        nargs="+",
        choices=list(RECEIVERS),
        help="simulate only these receivers and replace their rows in the saved CSV",
    )
    args = p.parse_args(argv)

    out = results_dir() / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"recovery_{args.channel}"
    if args.jammer != "nominal":
        stem = stem.with_name(f"{stem.name}_{args.jammer}")
    if args.snr_db != 15.0:
        stem = stem.with_name(f"{stem.name}_snr{args.snr_db:g}")
    if args.plot_only:
        rows = _read_rows(stem.with_suffix(".csv"))
    else:
        rows = []
        for seed in args.seeds:
            rows += simulate(
                args.channel,
                args.snr_db,
                args.jsr_db,
                args.batch_size,
                args.max_mc_iter,
                args.target_block_errors,
                seed,
                args.receivers,
                args.jammer,
            )
        if args.receivers and stem.with_suffix(".csv").is_file():
            kept = [
                r
                for r in _read_rows(stem.with_suffix(".csv"))
                if r["receiver"] not in rows_rx(rows)
            ]
            rows = kept + rows
        with open(stem.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pooled = pool_seeds(rows)
    with paper_style():
        main_fig = plot(pooled, args.channel)  # the message: no protection vs map vs true map
        ablation = plot(
            pooled, args.channel, ABLATION_RECEIVERS, shade=False
        )  # where gains come from
        sources = plot(pooled, args.channel, SOURCE_RECEIVERS, shade=False)  # what the map adds
    save(main_fig, stem)
    save(ablation, stem.with_name(f"{stem.name}_ablation"))
    save(sources, stem.with_name(f"{stem.name}_sources"))
    print(f"Wrote {stem}.csv and the main, ablation and map-source figures (.pdf/.png)")


if __name__ == "__main__":
    main()
