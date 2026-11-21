"""Does the per-cell false-alarm rate hold at every SNR, and when the noise power is misestimated?

A cell-level detector runs hundreds of tests per slot, so its false-alarm rate has to stay at the
design value under the conditions it meets. Thresholds here are set the CFAR way: on the cells of
benign ``val`` slots (1% flagged), and the false-alarm rate is measured on benign ``test`` slots.

1. **SNR.** One threshold for all SNRs, or one per 4 dB bin of the SNR the gNB estimates from
   the slot (``snr_db_est``). The residual's p-value needs neither: its distribution does not
   depend on the SNR.
2. **Noise power off by a set error.** The test grids ``noise_m2`` ... ``noise_p2`` run the whole
   receiver chain (channel estimation, equalization, features) with a noise power that is
   ``delta`` dB off, and the thresholds stay those set with the correct noise power. For the
   residual the exact answer follows from the theory: ``P(R > t 10^(delta/10))``.
3. **Estimated noise power** (``--estimator``). The receiver estimates the noise power in every
   slot from the empty DMRS comb (``jamloc.noise``; splits ``test_nest`` and ``sched_nest``).
   Three ways to deploy a detector: as trained and calibrated with the true noise power; trained
   with the true noise power but calibrated on ``val_nest``; trained and calibrated on
   ``train_nest`` / ``val_nest`` (``train --data nest``). The estimate's own error is reported per
   attack: a jammer on the whole DMRS symbol of the whole allocation biases it upward.

    python -m jamloc.experiments.calibration
    python -m jamloc.experiments.calibration --estimator
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.baselines import block_test_scores, residual_cfar_scores
from jamloc.config import SimConfig
from jamloc.dataset import NOISE_ERRORS_DB, generate, load, noise_split
from jamloc.evaluate import DETECTOR_STYLE, FA, _logits, aggregate, available_seeds
from jamloc.features import FEATURE_NAMES
from jamloc.metrics import f1, threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.theory import ccdf, threshold, weights
from jamloc.train import checkpoint_path, load_model

DETECTORS = ("fused", "no_residual", "power", "fused_nj1", "block_test", "residual_cfar")
# Learned detectors: (variant, load_model options). fused_nj1 is the fused U-Net trained with a
# noise-estimation error drawn from +-1 dB (``train --noise-jitter-db 1``)
LEARNED = {
    "fused": ("fused", {}),
    "no_residual": ("no_residual", {}),
    "power": ("power", {}),
    "fused_nj1": ("fused", {"jitter": 1.0}),
}
THRESHOLD_EDGES_DB = (9.0, 13.0, 17.0, 21.0)  # SNR bins of the conditioned thresholds
PLOT_EDGES_DB = np.arange(5.0, 25.1, 2.5)  # SNR bins of the false-alarm curves
MISMATCH_DB = NOISE_ERRORS_DB
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def shift_noise(d: dict, delta_db: float, *, cfg: SimConfig | None = None) -> dict:
    """The split as seen with a noise estimate ``delta_db`` above the true noise power, applied to
    the stored features (first-order approximation, as in the jitter training of
    ``model.noise_shift``; the experiment itself uses the full-chain ``noise_*`` grids)."""
    if delta_db == 0:
        return d
    f = d["features"].copy()
    f[:, _IDX["power_db"]] -= delta_db
    f[:, _IDX["residual_db"]] -= delta_db * f[:, _IDX["dmrs_mask"]]
    f[:, _IDX["evm_db"]] -= delta_db * f[:, _IDX["data_mask"]]
    f[:, _IDX["comb_db"]][..., (cfg or SimConfig()).dmrs_symbols] -= delta_db
    return {**d, "features": f}


_NUM_SLOTS = None  # smaller generated grids (--num-slots, for smoke tests)


def _split(name: str) -> dict:
    if not (data_dir() / f"{name}.npz").is_file():
        generate(name, _NUM_SLOTS)
    return load(name)


def _snr(d: dict) -> np.ndarray:
    """SNR the thresholds are binned by: the gNB's estimate (older datasets: the true SNR)."""
    return d.get("snr_db_est", d["snr_db"])


def scores(name: str, d: dict, seed: int, val: dict) -> np.ndarray:
    if name == "block_test":
        return block_test_scores(d, val)
    if name == "residual_cfar":
        return residual_cfar_scores(d)
    variant, options = LEARNED[name]
    return _logits(load_model(variant, seed, **options), variant, torch.from_numpy(d["features"]))


def available(name: str, seeds) -> bool:
    if name not in LEARNED:
        return True
    variant, options = LEARNED[name]
    return all(checkpoint_path(variant, seed, **options).is_file() for seed in seeds)


def _thresholds(s_val: np.ndarray, val: dict) -> tuple[float, np.ndarray]:
    """Global threshold and one per SNR bin, from the cells of benign validation slots."""
    benign = val["attack"] == 0
    bins = np.digitize(_snr(val), THRESHOLD_EDGES_DB)
    overall = threshold_at_fa(s_val[benign], FA)
    per_bin = np.array(
        [
            threshold_at_fa(s_val[sel], FA) if sel.any() else overall  # empty bin: global
            for sel in (benign & (bins == k) for k in range(len(THRESHOLD_EDGES_DB) + 1))
        ]
    )
    return overall, per_bin


def _flags(s: np.ndarray, d: dict, thr: float | np.ndarray) -> np.ndarray:
    if np.ndim(thr) == 0:
        return s > thr
    return s > thr[np.digitize(_snr(d), THRESHOLD_EDGES_DB)][:, None, None]


def run(seeds, detectors=DETECTORS) -> tuple[list[dict], list[dict], list[dict]]:
    val, test = load("val"), load("test")
    benign = test["attack"] == 0
    labels = test["labels"].astype(bool)
    plot_bins = np.digitize(test["snr_db"], PLOT_EDGES_DB) - 1

    snr_rows, summary_rows, mismatch_rows = [], [], []
    for name in [n for n in detectors if available(n, seeds)]:
        for seed in seeds if name in LEARNED else seeds[:1] or [0]:
            s_val, s_test = scores(name, val, seed, val), scores(name, test, seed, val)
            scored = s_test > -np.inf
            thr_global, thr_bins = _thresholds(s_val, val)
            for how, thr in (("one threshold", thr_global), ("threshold per SNR bin", thr_bins)):
                pred = _flags(s_test, test, thr)
                for k in range(len(PLOT_EDGES_DB) - 1):
                    sel = benign & (plot_bins == k)
                    snr_rows.append(
                        {
                            "detector": name,
                            "seed": seed,
                            "thresholds": how,
                            "snr_db": (PLOT_EDGES_DB[k] + PLOT_EDGES_DB[k + 1]) / 2,
                            "fa": float(pred[sel][scored[sel]].mean()),
                        }
                    )
                summary_rows.append(
                    {
                        "detector": name,
                        "seed": seed,
                        "thresholds": how,
                        "fa": float(pred[benign][scored[benign]].mean()),
                        "pd": float(pred[labels].mean()),
                        "f1": f1(pred, labels),
                    }
                )
            for delta in MISMATCH_DB:
                d = _split(noise_split(delta))  # the whole receiver chain with this error
                s = scores(name, d, seed, val)
                pred, ok = s > thr_global, d["attack"] == 0
                lab = d["labels"].astype(bool)
                mismatch_rows.append(
                    {
                        "detector": name,
                        "seed": seed,
                        "delta_db": delta,
                        "fa": float(pred[ok][(s > -np.inf)[ok]].mean()),
                        "pd": float(pred[lab].mean()),
                        "f1": f1(pred, lab),
                    }
                )
            print(f"{name}, seed {seed} done", flush=True)
    return snr_rows, summary_rows, mismatch_rows


ESTIMATOR_DEPLOYMENTS = {
    # name: (detector, load options, validation split for the threshold)
    "fused (true N0)": ("fused", {}, "val"),
    "fused, recalibrated": ("fused", {}, "val_nest"),
    "fused, noise jitter": ("fused_nj1", {}, "val"),
    "fused, trained on estimate": ("fused", {"data": "nest"}, "val_nest"),
    "block test": ("block_test", {}, "val_nest"),
    "residual p-value": ("residual_cfar", {}, "val_nest"),
}


def run_estimator(seeds) -> tuple[list[dict], list[dict]]:
    """Detection with the per-slot noise estimate, and the estimate's own error."""
    vals = {name: _split(name) for name in ("val", "val_nest")}
    rows, error_rows = [], []
    for split in ("test_nest", "sched_nest"):
        d = _split(split)
        labels, benign = d["labels"].astype(bool), d["attack"] == 0
        for attack in np.unique(d["attack"]):
            e = d["noise_error_db"][d["attack"] == attack]
            error_rows.append(
                {
                    "split": split,
                    "attack": int(attack),
                    "median_db": float(np.median(e)),
                    "p05_db": float(np.quantile(e, 0.05)),
                    "p95_db": float(np.quantile(e, 0.95)),
                    "std_db": float(np.std(e)),
                }
            )
        for name, (det, options, val_name) in ESTIMATOR_DEPLOYMENTS.items():
            val = vals[val_name]
            variant = "fused" if det == "fused_nj1" else det
            opts = {"jitter": 1.0} if det == "fused_nj1" else options
            if det in ("block_test", "residual_cfar"):
                runs = [(seeds[0], scores(det, d, 0, val), scores(det, val, 0, val))]
            else:
                if not all(checkpoint_path(variant, s, **opts).is_file() for s in seeds):
                    continue
                runs = []
                for seed in seeds:
                    model = load_model(variant, seed, **opts)
                    f = torch.from_numpy
                    runs.append(
                        (
                            seed,
                            _logits(model, variant, f(d["features"])),
                            _logits(model, variant, f(val["features"])),
                        )
                    )
            for seed, s, s_val in runs:
                thr = threshold_at_fa(s_val[val["labels"] == 0], FA)
                pred, scored = s > thr, s > -np.inf
                neg = ~labels
                rows.append(
                    {
                        "split": split,
                        "deployment": name,
                        "seed": seed,
                        "fa_benign": float(pred[benign][scored[benign]].mean())
                        if benign.any()
                        else float("nan"),
                        "fa": float((pred & neg).sum() / neg.sum()),
                        "pd": float(pred[labels].mean()),
                        "f1": f1(pred, labels),
                    }
                )
            print(f"[estimator] {split} {name} done", flush=True)
    return rows, error_rows


def residual_theory_fa(delta_db: float, d: dict, fa: float = FA) -> float:
    """Exact false-alarm rate of the residual test at design false-alarm rate ``fa`` with a noise
    estimate off by ``delta_db``, averaged over the DMRS cells of the benign slots of ``d``."""
    cfg = SimConfig()
    benign = d["attack"] == 0
    total, count = 0.0, 0
    sizes = d["alloc_size"][benign]
    for n in np.unique(sizes):
        slots = int((sizes == n).sum())
        for pos in range(int(n)):
            t = threshold(int(n), pos, fa, cfg) * 10 ** (delta_db / 10)
            total += slots * float(ccdf(t, weights(int(n), pos, cfg=cfg)))
            count += slots
    return total / count


def plot(snr_mean, mismatch_mean, theory):
    fig, axes = new_figure(7.16, 2.9, ncols=3)
    for ax, how in zip(axes[:2], ("one threshold", "threshold per SNR bin"), strict=True):
        for name in DETECTORS:
            rows = [r for r in snr_mean if r["detector"] == name and r["thresholds"] == how]
            if not rows:
                continue
            style = DETECTOR_STYLE[name]
            ax.plot([r["snr_db"] for r in rows], [r["fa_mean"] for r in rows], **style)
        ax.axhline(FA, color="k", lw=0.8, ls=":")
        ax.set(title=how, xlabel="SNR [dB]", yscale="log", ylim=(1e-3, 1e-1))
    axes[0].set_ylabel("per-cell false alarm, benign slots")
    ax = axes[2]
    for name in DETECTORS:
        rows = sorted(
            (r for r in mismatch_mean if r["detector"] == name), key=lambda r: r["delta_db"]
        )
        if not rows:
            continue
        ax.plot([r["delta_db"] for r in rows], [r["fa_mean"] for r in rows], **DETECTOR_STYLE[name])
    ax.plot(MISMATCH_DB, theory, color="k", lw=2.5, alpha=0.3, label="residual, theory")
    ax.axhline(FA, color="k", lw=0.8, ls=":")
    ax.set(
        title="noise estimate off by $\\delta$",
        xlabel="$\\delta$: estimated / true noise [dB]",
        yscale="log",
        ylim=(1e-4, 1.0),
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path, text=("detector", "thresholds")):
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all trained")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--estimator", action="store_true", help="per-slot noise estimate (part 3)")
    p.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTORS,
        default=None,
        help="run only these and replace their rows in the existing CSVs",
    )
    p.add_argument("--num-slots", type=int, default=None, help="generate smaller grids (testing)")
    args = p.parse_args(argv)
    global _NUM_SLOTS
    _NUM_SLOTS = args.num_slots

    out = results_dir() / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    if args.estimator:
        rows, error_rows = run_estimator(args.seeds or available_seeds())
        _write(error_rows, out / "noise_estimate_error.csv")
        mean = aggregate(rows, ("fa_benign", "fa", "pd", "f1"))
        _write(mean, out / "noise_estimate.csv")
        for r in error_rows:
            print(
                f"{r['split']:10s} attack {r['attack']}: estimate error median "
                f"{r['median_db']:+.2f} dB, 5-95% [{r['p05_db']:+.2f}, {r['p95_db']:+.2f}] dB"
            )
        for r in mean:
            print(
                f"{r['split']:10s} {r['deployment']:28s} FA benign {r['fa_benign_mean']:.4f}  "
                f"FA {r['fa_mean']:.4f}  Pd {r['pd_mean']:.3f}  F1 {r['f1_mean']:.3f}"
            )
        return
    if not args.plot_only:
        snr_rows, summary_rows, mismatch_rows = run(
            args.seeds or available_seeds(), args.detectors or DETECTORS
        )
        val = load("val")
        # The measured residual curve uses the threshold calibrated on benign validation cells
        # (p = 10^-score), so the theory uses the same design false-alarm rate, averaged over the
        # allocations of each noise grid
        s_val = residual_cfar_scores(val)
        fa = 10 ** -threshold_at_fa(s_val[val["attack"] == 0], FA)
        theory = [
            residual_theory_fa(delta, _split(noise_split(delta)), fa) for delta in MISMATCH_DB
        ]
        mismatch = aggregate(mismatch_rows, ("fa", "pd", "f1"))
        for r in mismatch:
            r["residual_theory_fa"] = theory[MISMATCH_DB.index(r["delta_db"])]
        for rows, name in (
            (aggregate(snr_rows, "fa"), "fa_vs_snr.csv"),
            (aggregate(summary_rows, ("fa", "pd", "f1")), "snr_thresholds.csv"),
            (mismatch, "noise_mismatch.csv"),
        ):
            if args.detectors and (out / name).is_file():
                new = {r["detector"] for r in rows}
                kept = [r for r in _read(out / name) if r["detector"] not in new]
                for r in kept:  # the theory column is the same for every detector
                    if "residual_theory_fa" in r:
                        r["residual_theory_fa"] = theory[MISMATCH_DB.index(r["delta_db"])]
                rows = kept + rows
            _write(rows, out / name)
    snr_mean = _read(out / "fa_vs_snr.csv")
    mismatch_mean = _read(out / "noise_mismatch.csv")
    theory = [
        next(r["residual_theory_fa"] for r in mismatch_mean if r["delta_db"] == delta)
        for delta in MISMATCH_DB
    ]
    with paper_style():
        fig = plot(snr_mean, mismatch_mean, theory)
    save(fig, out / "calibration")
    for r in _read(out / "snr_thresholds.csv"):
        print(
            f"{r['detector']:14s} {r['thresholds']:22s} FA {r['fa_mean']:.4f}  "
            f"Pd {r['pd_mean']:.3f}  F1 {r['f1_mean']:.3f}"
        )
    for r in mismatch_mean:
        print(
            f"{r['detector']:14s} delta {r['delta_db']:+.1f} dB  FA {r['fa_mean']:.4f}  "
            f"Pd {r['pd_mean']:.3f}  F1 {r['f1_mean']:.3f}  (residual theory "
            f"{r['residual_theory_fa']:.4f})"
        )


if __name__ == "__main__":
    main()
