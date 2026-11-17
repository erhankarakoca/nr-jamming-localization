"""How small can the U-Net be? Detection quality and CPU latency against network width.

The fused detector (power + residual + EVM) is trained with narrower first levels (``width``
channels, doubled at every level, so the parameter count grows with ``width**2``) and evaluated
like the main detectors: thresholds at 1% false alarm on ``val``, F1 and Pd on ``test``.
Latency is measured separately on the CPU (one thread, batch of one), because the training runs
on a GPU: a 30 kHz slot lasts 0.5 ms, so feature extraction plus the U-Net should fit in that.
It is reported for plain PyTorch and for a frozen graph of the same network (batch norm folded,
no Python between layers); for a network this small, Python's per-layer overhead dominates.

    python -m jamloc.experiments.size_sweep                  # train missing widths, evaluate
    python -m jamloc.experiments.size_sweep --latency        # CPU latency only (run on the CPU)
    python -m jamloc.experiments.size_sweep --plot-only
"""

import argparse
import csv

import numpy as np
import torch
from sionna.phy import config

from jamloc.dataset import load
from jamloc.evaluate import aggregate, evaluate_scores
from jamloc.model import VARIANTS, TinyUNet, select_inputs
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import DEFAULT_WIDTH, checkpoint_path, load_model, train

VARIANT = "fused"
WIDTHS = (4, 6, 8, DEFAULT_WIDTH)
SLOT_MS = 0.5  # one slot at 30 kHz subcarrier spacing


def num_params(width: int) -> int:
    model = TinyUNet(len(VARIANTS[VARIANT]), base=width)
    return sum(p.numel() for p in model.parameters())


def _logits(model, features: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(features)
    with torch.no_grad():
        out = [
            model(select_inputs(x[i].to(config.device), VARIANT)).cpu()
            for i in torch.arange(len(x)).split(2048)
        ]
    return torch.cat(out).numpy()


def evaluate_widths(widths, seeds, epochs: int) -> list[dict]:
    val, test = load("val"), load("test")
    rows = []
    for width in widths:
        for seed in seeds:
            if not checkpoint_path(VARIANT, seed, width).is_file():
                train(VARIANT, epochs=epochs, seed=seed, width=width)
            model = load_model(VARIANT, seed, width)
            scores = {"unet": (_logits(model, val["features"]), _logits(model, test["features"]))}
            pd_rows, f1_rows = evaluate_scores(scores, val, test)
            f1_by = {r["attack"]: r for r in f1_rows}

            def pd(attack, jsr, pd_rows=pd_rows):
                (r,) = [
                    r
                    for r in pd_rows
                    if r["attack"] == attack
                    and r["channel"] == "both"
                    and r["cells"] == "scheduled"
                    and r["jsr_db"] == jsr
                ]
                return r["pd"]

            rows.append(
                {
                    "width": width,
                    "params": num_params(width),
                    "seed": seed,
                    "f1_all": f1_by["all"]["f1"],
                    "f1_pilot": f1_by["pilot"]["f1"],
                    "f1_data": f1_by["data"]["f1"],
                    "fa": f1_by["all"]["fa_scored"],
                    "pd_pilot_m15": pd("pilot", -15.0),
                    "pd_pilot_m9": pd("pilot", -9.0),
                    "pd_data_m15": pd("data", -15.0),
                }
            )
            print(f"width {width:2d} seed {seed}: F1 {rows[-1]['f1_all']:.3f}", flush=True)
    return rows


def _frozen(model, x):
    """The same network as a frozen TorchScript graph: batch norm folded into the convolutions
    and no Python between the layers, closer to how a gNB would deploy it. (TorchScript is
    deprecated in recent PyTorch but still works on every platform.)"""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        graph = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(model, x)))
    for _ in range(5):  # the first calls specialize the graph
        graph(x)
    return graph


def measure_latency(widths, reps: int, alloc_prb: int = 24) -> list[dict]:
    from jamloc.config import SimConfig
    from jamloc.experiments.latency import _median_ms
    from jamloc.features import extract
    from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything

    previous_device, previous_threads = config.device, torch.get_num_threads()
    config.device = "cpu"
    torch.set_num_threads(1)
    try:
        cfg = SimConfig()
        scenario = draw_scenario(
            seed_everything(0), 1, cfg, "A", attacks=(1,), alloc=(0, alloc_prb)
        )
        batch = JammedPUSCHLink(cfg)(scenario)
        rows = []
        with torch.inference_mode():
            feature_ms = _median_ms(lambda: extract(batch, cfg), reps)
            x = select_inputs(extract(batch, cfg), VARIANT)
            for width in widths:
                model = TinyUNet(x.shape[1], base=width).eval()
                model_ms = _median_ms(lambda m=model: m(x), reps)
                frozen = _frozen(model, x)
                frozen_ms = _median_ms(lambda m=frozen: m(x), reps)
                rows.append(
                    {
                        "width": width,
                        "params": num_params(width),
                        "feature_ms": feature_ms,
                        "model_ms": model_ms,
                        "total_ms": feature_ms + model_ms,
                        "frozen_model_ms": frozen_ms,
                        "frozen_total_ms": feature_ms + frozen_ms,
                    }
                )
                print(
                    f"width {width:2d}: U-Net {model_ms:.3f} ms (frozen graph {frozen_ms:.3f} ms), "
                    f"total {rows[-1]['total_ms']:.3f} ms ({rows[-1]['frozen_total_ms']:.3f} ms)"
                )
    finally:
        config.device = previous_device
        torch.set_num_threads(previous_threads)
    return rows


def plot(quality: list[dict], latency: list[dict] | None):
    fig, axes = new_figure(7.16, 2.4, ncols=3 if latency else 2)
    mean = aggregate(quality, [k for k in quality[0] if k not in ("width", "params", "seed")])
    mean.sort(key=lambda r: r["params"])
    x = np.array([r["params"] for r in mean]) / 1e3
    for ax, keys, title in (
        (axes[0], (("f1_all", "all slots"), ("f1_pilot", "pilot jamming")), "Cell-level F1"),
        (
            axes[1],
            (("pd_pilot_m15", "pilot, JSR -15 dB"), ("pd_data_m15", "data, JSR -15 dB")),
            "Pd at 1% false alarm",
        ),
    ):
        for (key, label), marker in zip(keys, ("o", "s"), strict=True):
            m = np.array([r[f"{key}_mean"] for r in mean])
            s = np.array([r[f"{key}_std"] for r in mean])
            ax.errorbar(x, m, yerr=s, marker=marker, capsize=2, label=label)
        ax.set(xscale="log", xlabel="U-Net parameters [thousands]", title=title)
        ax.legend(loc="lower right")
    if latency:
        latency = sorted(latency, key=lambda r: r["params"])
        lx = np.array([r["params"] for r in latency]) / 1e3
        ax = axes[2]
        ax.plot(lx, [r["total_ms"] for r in latency], "o-", label="features + U-Net")
        ax.plot(lx, [r["model_ms"] for r in latency], "s--", label="U-Net only")
        if "frozen_total_ms" in latency[0]:
            ax.plot(lx, [r["frozen_total_ms"] for r in latency], "^-.", label="total, frozen graph")
        ax.axhline(SLOT_MS, color="k", lw=0.8, ls=":")
        ax.text(lx[0], SLOT_MS * 1.03, "one slot (0.5 ms)", fontsize=6, va="bottom")
        ax.set(
            xscale="log",
            xlabel="U-Net parameters [thousands]",
            ylabel="ms per slot, 1 CPU thread",
            title="CPU latency",
            ylim=(0, None),
        )
        ax.legend(loc="lower right")
    fig.suptitle("Fused detector (P+R+E) with narrower U-Nets; bars: std over training seeds")
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--latency", action="store_true", help="only measure CPU latency")
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    out = results_dir() / "size_sweep"
    out.mkdir(parents=True, exist_ok=True)
    quality_csv, latency_csv = out / "size_sweep.csv", out / "size_sweep_latency.csv"
    if args.latency:
        _write(measure_latency(args.widths, args.reps), latency_csv)
    elif not args.plot_only:
        _write(evaluate_widths(args.widths, args.seeds, args.epochs), quality_csv)
    if quality_csv.is_file():
        latency = _read(latency_csv) if latency_csv.is_file() else None
        with paper_style():
            fig = plot(_read(quality_csv), latency)
        save(fig, out / "size_sweep")
        print(f"Wrote {out / 'size_sweep'}.pdf/.png")


if __name__ == "__main__":
    main()
