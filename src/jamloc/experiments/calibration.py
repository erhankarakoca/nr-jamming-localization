"""Does the per-cell false-alarm rate hold at every SNR, and when the noise power is misestimated?

A cell-level detector runs hundreds of tests per slot, so its false-alarm rate has to stay at the
design value under the conditions it meets. Thresholds here are set the CFAR way: on the cells of
benign ``val`` slots (1% flagged), and the false-alarm rate is measured on benign ``test`` slots.

1. **SNR.** One threshold for all SNRs, or one per 4 dB SNR bin (the gNB knows its SNR). The
   residual's p-value needs neither: its distribution does not depend on the SNR.
2. **Noise power.** Every feature is divided by the noise variance, so an estimate that is off by
   ``delta`` dB shifts P, R and E by ``-delta`` dB (inside their masks). This is applied to the
   stored test features, with thresholds from the correct noise power. It leaves out the second-
   order effect of the wrong noise variance on the LMMSE weights. For the residual the exact
   answer follows from the theory: ``P(R > t 10^(delta/10))``.

    python -m jamloc.experiments.calibration
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.baselines import block_test_scores, residual_cfar_scores
from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import DETECTOR_STYLE, FA, _logits, aggregate, available_seeds
from jamloc.features import FEATURE_NAMES
from jamloc.metrics import f1, threshold_at_fa
from jamloc.paths import results_dir
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
MISMATCH_DB = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def shift_noise(d: dict, delta_db: float) -> dict:
    """The split as seen with a noise estimate ``delta_db`` above the true noise power."""
    if delta_db == 0:
        return d
    f = d["features"].copy()
    f[:, _IDX["power_db"]] -= delta_db
    f[:, _IDX["residual_db"]] -= delta_db * f[:, _IDX["dmrs_mask"]]
    f[:, _IDX["evm_db"]] -= delta_db * f[:, _IDX["data_mask"]]
    return {**d, "features": f}


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
    bins = np.digitize(val["snr_db"], THRESHOLD_EDGES_DB)
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
    return s > thr[np.digitize(d["snr_db"], THRESHOLD_EDGES_DB)][:, None, None]


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
                s = s_test if delta == 0 else scores(name, shift_noise(test, delta), seed, val)
                pred = s > thr_global
                mismatch_rows.append(
                    {
                        "detector": name,
                        "seed": seed,
                        "delta_db": delta,
                        "fa": float(pred[benign][(s > -np.inf)[benign]].mean()),
                        "pd": float(pred[labels].mean()),
                        "f1": f1(pred, labels),
                    }
                )
            print(f"{name}, seed {seed} done", flush=True)
    return snr_rows, summary_rows, mismatch_rows


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
    p.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTORS,
        default=None,
        help="run only these and replace their rows in the existing CSVs",
    )
    args = p.parse_args(argv)

    out = results_dir() / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        snr_rows, summary_rows, mismatch_rows = run(
            args.seeds or available_seeds(), args.detectors or DETECTORS
        )
        test, val = load("test"), load("val")
        # The measured residual curve uses the threshold calibrated on benign validation cells
        # (p = 10^-score), so the theory uses the same design false-alarm rate
        s_val = residual_cfar_scores(val)
        fa = 10 ** -threshold_at_fa(s_val[val["attack"] == 0], FA)
        theory = [residual_theory_fa(delta, test, fa) for delta in MISMATCH_DB]
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
