"""What each step of the receiver costs for one slot (paper v12, the cost table).

The steps follow Fig. 1(b) of the paper. For each, the number of real multiply-accumulates for
the allocation of the paper (12 PRBs), counted from the sizes of its matrices, and the median CPU
time for one slot on one core, with one and with two DMRS symbols. LDPC decoding and the whole
receiver are timed for scale, and LDPC decoding is counted in edge messages instead.

The time is that of this PyTorch implementation, in which every step is a few calls with fixed
overhead, so it shows the order of the costs, not what an optimized gNB would need. On one core,
as ``experiments.latency`` explains, a 24 x 14 grid runs faster than with several threads.

* Steps 1 to 3 (LS values, residuals, pilot weights) project the LS values onto the delay
  subspace of dimension ``D``, about ``2 N_p D`` complex multiply-accumulates for ``N_p`` pilots.
* Step 4 (LMMSE) solves one ``N_p x N_p`` system per slot, because the pilot weights change from
  slot to slot, with ``1 + K`` right-hand sides for ``K`` subcarriers (the estimate and its error
  variance). A receiver with fixed weights would compute the filter once per allocation and then
  needs only ``K N_p`` per slot (the ``lmmse_fixed`` row). The prior has rank ``D_c`` (21 for
  12 PRBs), so the same estimate takes about ``N_p D_c^2 + K D_c^2`` in the delay subspace
  (``recovery.lmmse_lowrank``, the ``lmmse_lowrank`` row).
* Steps 5 and 6 (excess power, shrinkage) take a few operations per data resource element.
* The U-Net takes the multiply-accumulates of its convolutions, counted by hooks.

    python -m jamloc.experiments.complexity --reps 200

Output: ``results/complexity/complexity.csv`` and a markdown table on stdout.
"""

import argparse
import csv
import dataclasses

import torch
from sionna.phy import config
from torch import nn

from jamloc.config import SimConfig
from jamloc.experiments.latency import _median_ms
from jamloc.experiments.size_sweep import _frozen
from jamloc.features import delay_subspace, dmrs_ls, extract
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.model import TinyUNet, select_inputs
from jamloc.paths import results_dir
from jamloc.receiver import PUSCHReceiverChain
from jamloc.recovery import MapAwareReceiver, _prior_basis, lmmse_interpolation, lmmse_lowrank

ALLOC = (6, 12)  # the UE's PRBs in the paper
# (key, step numbers of Fig. 1(b), what it computes)
STEPS = (
    ("pilots", "1-3", "LS values, residuals and pilot weights"),
    ("lmmse", "4", "LMMSE channel estimate with pilot weights"),
    ("lmmse_lowrank", "4", "The same estimate in the delay subspace"),
    ("lmmse_fixed", "4", "LMMSE with a filter fixed per allocation (reference)"),
    ("shrinkage", "5-6", "Excess power and shrinkage"),
    ("llr", "7", "Equalization and LLRs"),
    ("decoding", "7", "LDPC decoding (reference)"),
    ("features", "det.", "Features of the U-Net"),
    ("unet", "det.", "U-Net forward pass"),
    ("unet_frozen", "det.", "U-Net forward pass, frozen graph"),
    ("receiver", "1-7", "Whole receiver without a map"),
    ("baseline", "", "Baseline receiver (reference)"),
)


def unet_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of the convolutions of ``model`` on the input ``x``."""
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.ConvTranspose2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            total += inputs[0].numel() * module.out_channels * k // module.groups
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            total += output.numel() * module.in_channels * k // module.groups

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total


def operation_counts(num_pilots: int, dim: int, num_sc: int, num_data: int, prior_dim: int) -> dict:
    """Real multiply-accumulates of steps 1 to 6 for ``num_pilots`` LS values (all DMRS symbols),
    a delay subspace of dimension ``dim`` per DMRS symbol, ``num_sc`` subcarriers, ``num_data``
    data resource elements and a channel prior of rank ``prior_dim``. A complex multiply-accumulate counts as four real ones. Step 5 takes
    about six per data resource element (|y|^2 and the subtractions) and step 6 about four (the
    sums of the cell estimates, their squares and the pull)."""
    n = num_pilots
    solve = n**3 / 3 + n**2 * (1 + num_sc) + 2 * num_sc * n  # LU, two solves, two products
    d = prior_dim
    # V_p^H W V_p and V_p^H W h, the D x D solve with 1 + K right-hand sides, the two products
    lowrank = n * d**2 + n * d + d**3 / 3 + d**2 * (1 + num_sc) + 2 * num_sc * d
    return {
        "pilots": 4 * 2 * n * dim,
        "lmmse": round(4 * solve),
        "lmmse_lowrank": round(4 * lowrank),
        "lmmse_fixed": 4 * num_sc * n,
        "shrinkage": 10 * num_data,
    }


def _slot(cfg: SimConfig):
    """One slot of QPSK at 8 dB SNR under a hopping jammer on pilots and data at +9 dB JSR,
    as in the paper."""
    seed_everything(0)
    sc = Scenario(
        channel="C",
        alloc_start=ALLOC[0],
        alloc_size=ALLOC[1],
        attack=torch.tensor([ATTACKS.index("partial")]),
        jam_start=torch.tensor([10]),
        jam_size=torch.tensor([4]),
        jsr_db=torch.tensor([9.0]),
        snr_db=torch.tensor([8.0]),
    )
    link = JammedPUSCHLink(dataclasses.replace(cfg, jammer_hopping=True))
    return link(sc)


def benchmark(reps: int, dmrs2: bool) -> list[dict]:
    """One row per step of ``STEPS`` with its operation count and median time in ms."""
    cfg = SimConfig(dmrs_additional_position=1) if dmrs2 else SimConfig()
    out = _slot(cfg)
    y, no = out.y_alloc, out.no
    num_sc = y.shape[-1]
    rx = MapAwareReceiver(out.tx, cfg)
    rx._pilot_weights, rx._cellwise = "all", "shrink"
    baseline = PUSCHReceiverChain(out.tx)
    interp = {"num_sc": num_sc, "scs": cfg.scs, "tau_max": cfg.tau_max}

    with torch.inference_mode():
        h_ls, pilot_sc, pilot_power = dmrs_ls(y, out.tx)
        clean, pilot_var, r, interfered = rx._pilots(h_ls, pilot_sc, pilot_power, no, None, "cell")
        h_sc, err_sc = lmmse_interpolation(h_ls, pilot_sc, clean, pilot_var, **interp)
        # The fixed filter: the estimate is linear in the LS values, so unit LS vectors give
        # its columns
        eye = torch.eye(h_ls.shape[1], dtype=h_ls.dtype)
        filt = lmmse_interpolation(
            eye,
            pilot_sc,
            torch.ones_like(eye, dtype=torch.bool),
            (no / pilot_power).expand(eye.shape[0]),
            **interp,
        )[0].T  # [num_sc, N_p]
        h_hat, err_var = rx._on_grid(h_sc, err_sc, y)
        llr = rx._detect(y, no, h_hat, err_var, h_sc, err_sc, None, r, interfered, "cell")[2]
        features = extract(out, cfg)
        x = select_inputs(features, "fused")
        model = TinyUNet(x.shape[1]).eval()
        frozen = _frozen(model, x)

        timed = {
            "pilots": lambda: rx._pilots(*dmrs_ls(y, out.tx), no, None, "cell"),
            "lmmse": lambda: lmmse_interpolation(h_ls, pilot_sc, clean, pilot_var, **interp),
            "lmmse_lowrank": lambda: lmmse_lowrank(h_ls, pilot_sc, clean, pilot_var, **interp),
            "lmmse_fixed": lambda: h_ls @ filt.T,
            "shrinkage": lambda: rx._jamming_variance(
                y, no, h_sc, err_sc, None, r, interfered, "cell"
            ),
            # Without the jamming power of steps 5 and 6, which the receiver adds before demapping
            "llr": lambda: rx.demapper(*rx.equalizer(y, h_hat, err_var, no)),
            "decoding": lambda: rx.decoder(rx.layer_demapper(llr)),
            "features": lambda: extract(out, cfg),
            "unet": lambda: model(x),
            "unet_frozen": lambda: frozen(x),
            "receiver": lambda: rx.call(
                y, no, None, decode=True, mode="cell", cellwise="shrink", pilot_weights="all"
            ),
            "baseline": lambda: baseline(y, no),
        }
        times = {key: _median_ms(fn, reps) for key, fn in timed.items()}

    num_dmrs = len(rx.dmrs_symbols)
    dim = delay_subspace(pilot_sc[: pilot_sc.numel() // num_dmrs], cfg.scs, cfg.tau_max).shape[1]
    prior_dim = _prior_basis(num_sc, cfg.scs, cfg.tau_max, 1 - 1e-9)[0].shape[1]
    counts = operation_counts(h_ls.shape[1], dim, num_sc, rx.data_sym.numel(), prior_dim)
    counts["unet"] = counts["unet_frozen"] = unet_macs(model, x)
    ldpc = rx.decoder._decoder
    edge_messages = 2 * ldpc._num_edges * ldpc._num_iter  # both directions, every iteration

    rows = []
    for key, steps, what in STEPS:
        rows.append(
            {
                "dmrs": num_dmrs,
                "step": key,
                "steps": steps,
                "what": what,
                "real_macs": counts.get(key, ""),
                "edge_messages": edge_messages if key == "decoding" else "",
                "ms": times[key],
                "pilots": h_ls.shape[1],
                "subspace_dim": dim,
                "prior_dim": prior_dim,
            }
        )
    return rows


def table(rows: list[dict]) -> str:
    """Markdown table with one column of counts and one of times per DMRS configuration."""
    by = {(r["step"], int(r["dmrs"])): r for r in rows}
    configs = sorted({int(r["dmrs"]) for r in rows})
    head = (
        "| Steps | What | " + " | ".join(f"MACs, {d} DMRS | ms, {d} DMRS" for d in configs) + " |"
    )
    lines = [head, "| --- | --- |" + " --- | --- |" * len(configs)]
    for key, steps, what in STEPS:
        cells = []
        for d in configs:
            r = by[(key, d)]
            count = r["real_macs"] or (
                f"{int(r['edge_messages']):,} msg" if r["edge_messages"] else ""
            )
            cells += [
                f"{int(count):,}" if isinstance(count, int) else str(count),
                f"{float(r['ms']):.3f}",
            ]
        lines.append(f"| {steps} | {what} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = p.parse_args(argv)

    previous_device, previous_threads = config.device, torch.get_num_threads()
    config.device = "cpu"  # this benchmark is about CPU time
    torch.set_num_threads(args.threads)
    try:
        rows = benchmark(args.reps, dmrs2=False) + benchmark(args.reps, dmrs2=True)
    finally:
        config.device = previous_device
        torch.set_num_threads(previous_threads)

    out = results_dir() / "complexity"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "complexity.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(table(rows))


if __name__ == "__main__":
    main()
