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
* ``p_r_e_atk_soft`` - map from the full U-Net trained on mixed attacks (``train_atk``: also
                    hopping and pilot-comb jammers), soft; only when that model is trained
* ``gen_soft`` / ``gen_shrink`` - map from the full U-Net of the general training (``train_gen``:
                    every jammer, channel and modulation of its mix), with the per-PRB estimate or
                    with shrinkage; only when that model is trained
* ``true_hard`` / ``true_soft`` - the true jamming map (where the jammer is; its strength is still
                    estimated in the soft mode)
* ``true_prb_soft`` - the true map coarsened to the block test's resolution (one decision for the
                    DMRS cell and one for all data cells of each PRB), soft: separates the value of
                    resolution from the quality of a detector
* ``lmmse_da`` / ``p_r_e_soft_da`` / ``true_soft_da`` - ``lmmse``, ``p_r_e_soft`` and ``true_soft``
                    with ``DATA_AIDED`` data-aided passes (``jamloc.recovery``): after a failed
                    decoding, the decoded symbols of the data cells the map calls clean become
                    pilots. ``lmmse_da`` has no map, so it also feeds back jammed data cells.

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

``--noise-estimate comb`` runs the whole link with the noise power the gNB estimates in every slot
from the empty DMRS comb (``jamloc.noise``) instead of the true one: the receivers, the features
and the detectors all use the estimate. The detectors are then the ones trained or calibrated on
that estimate (the fused U-Net of ``train_nest``, the block test and the power test calibrated on
``val_nest``); the networks never trained on it are left out.

Every run also keeps, per slot, whether each receiver failed to decode it
(``*_seed<s>_slots.npz``). All receivers decode the same slots, so two receivers can be compared
slot by slot: ``*_paired.csv`` gives the BLER difference of every pair of receivers at every point,
with a 95% interval for paired outcomes (``metrics.paired_difference_interval``). A receiver that
stopped early (100 block errors) is compared on the slots both receivers decoded.

    python -m jamloc.experiments.recovery --channel A
    python -m jamloc.experiments.recovery --channel C --jammer pulsed --seeds 0 1 2
    python -m jamloc.experiments.recovery --channel C --noise-estimate comb --seeds 0 1 2
    python -m jamloc.experiments.recovery --channel C --jam-prb 8 --jsr-db 3 9  # wider jammer

``--replay DIR`` repeats an earlier run whose per-seed CSVs are in ``DIR`` on exactly the same
slots, but decodes only the points and receivers the paper quotes (``QUOTED``), so that their
per-slot outcomes, and hence paired intervals, are available without decoding everything again.
Every point is simulated in the original order and with the original number of batches (the
random draws do not depend on which receivers decode); its counts must reproduce the earlier
CSV, which ``check_replay`` verifies.

    python -m jamloc.experiments.recovery --channel C --seeds 0 1 2 --unet nominal \\
        --replay results/recovery_pr9

The U-Net map comes from the general U-Net by default. Runs made before the paper v9 runs used
the U-Net trained on the nominal mix, and ``--unet nominal`` repeats them.
"""

import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch
from sionna.phy import config

from jamloc.baselines import block_test_scores, evm_scores, residual_scores
from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.features import extract
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import paired_difference_interval, threshold_at_fa
from jamloc.model import select_inputs
from jamloc.noise import NOISE_ESTIMATES, estimated_snr_db
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, plot_bler, save, zero_error_handle
from jamloc.recovery import MapAwareReceiver
from jamloc.train import checkpoint_path, load_model

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
    "p_r_e_atk_soft": ("fused_atk", "soft"),
    "true_hard": ("true", "hard"),
    "true_soft": ("true", "soft"),
    "true_prb_soft": ("true_prb", "soft"),
    "lmmse_da": (None, "lmmse"),
    "p_r_e_soft_da": ("fused", "soft"),
    "true_soft_da": ("true", "soft"),
    "cell_soft": (None, "cell"),
    "p_r_e_cell": ("fused", "soft"),
    "p_r_e_refine": ("fused", "soft"),
    "p_r_e_renoise": ("fused", "soft"),
    "true_renoise": ("true", "soft"),
    "cell_da": (None, "cell"),
    "true_cell": ("true", "soft"),
    "p_r_e_full": ("fused", "soft"),
    "true_full": ("true", "soft"),
    "cell_full": (None, "cell"),
    "p_r_e_cell_da": ("fused", "soft"),
    "p_r_e_refine_prb": ("fused", "soft"),
    "p_r_e_shrink": ("fused", "soft"),
    "p_r_e_shrink_da": ("fused", "soft"),
    "p_r_e_shrink_full": ("fused", "soft"),
    "true_shrink": ("true", "soft"),
    "true_shrink_full": ("true", "soft"),
    "cell_shrink": (None, "cell"),
    "cell_shrink_full": (None, "cell"),
    # Resolution study: how finely to estimate the jamming power, and with which map
    "nomap_prb": (None, "cell"),
    "nomap_prb2": (None, "cell"),
    "nomap_prb4": (None, "cell"),
    "nomap_cell": (None, "cell"),
    "nomap_shrink": (None, "cell"),
    "evm_prb": ("evm_threshold", "soft"),
    "evm_cell": ("evm_threshold", "soft"),
    "evm_shrink": ("evm_threshold", "soft"),
    "p_r_e_prb2": ("fused", "soft"),
    "p_r_e_prb4": ("fused", "soft"),
    "true_prb_shrink": ("true_prb", "soft"),
    # Map of the U-Net of the general training (experiments/general_training)
    "gen_soft": ("fused_gen", "soft"),
    "gen_shrink": ("fused_gen", "soft"),
}
# Data-aided passes after the first decoding (jamloc.recovery); three take most of the gain
DATA_AIDED = {
    "lmmse_da": 3,
    "p_r_e_soft_da": 3,
    "true_soft_da": 3,
    "p_r_e_refine": 3,
    "p_r_e_renoise": 3,
    "true_renoise": 3,
    "cell_da": 3,
    "p_r_e_full": 3,
    "true_full": 3,
    "cell_full": 3,
    "p_r_e_cell_da": 3,
    "p_r_e_refine_prb": 3,
    "p_r_e_shrink_da": 3,
    "p_r_e_shrink_full": 3,
    "cell_shrink_full": 3,
    "true_shrink_full": 3,
}
# Options of the map-aware receiver (jamloc.recovery.MapAwareReceiver.call)
OPTIONS = {
    "p_r_e_cell": {"cellwise": True},
    "true_cell": {"cellwise": True},
    "p_r_e_full": {"cellwise": True, "refine": True, "renoise": True},
    "true_full": {"cellwise": True, "refine": True, "renoise": True},
    "cell_full": {"renoise": True},
    "p_r_e_cell_da": {"cellwise": True},
    "p_r_e_refine_prb": {"refine": True},
    "p_r_e_shrink": {"cellwise": "shrink"},
    "p_r_e_shrink_da": {"cellwise": "shrink"},
    "p_r_e_shrink_full": {"cellwise": "shrink", "refine": True, "renoise": "auto"},
    "true_shrink": {"cellwise": "shrink"},
    "true_shrink_full": {"cellwise": "shrink", "refine": True, "renoise": "auto"},
    "cell_shrink": {"cellwise": "shrink"},
    "cell_shrink_full": {"cellwise": "shrink", "renoise": "auto"},
    "p_r_e_refine": {"cellwise": True, "refine": True},
    "p_r_e_renoise": {"renoise": True},
    "true_renoise": {"renoise": True},
    # No map: every pilot weighted by its residual and every data cell treated as flagged, with no
    # threshold or test
    "nomap_prb": {"cellwise": "prb", "pilot_weights": "all"},
    "nomap_prb2": {"cellwise": "prb", "pilot_weights": "all", "scale": 2.0},
    "nomap_prb4": {"cellwise": "prb", "pilot_weights": "all", "scale": 4.0},
    "nomap_cell": {"cellwise": True, "pilot_weights": "all"},
    "nomap_shrink": {"cellwise": "shrink", "pilot_weights": "all"},
    "evm_cell": {"cellwise": True},
    "evm_shrink": {"cellwise": "shrink"},
    "p_r_e_prb2": {"scale": 2.0},
    "p_r_e_prb4": {"scale": 4.0},
    "true_prb_shrink": {"cellwise": "shrink"},
    "gen_shrink": {"cellwise": "shrink"},
}
DETECTORS = (
    "power_threshold",
    "evm_threshold",
    "block_test",
    "no_residual",
    "fused",
    "fused_atk",
    "fused_gen",
)
NO_MODEL = ("power_threshold", "evm_threshold")  # thresholds of a score, nothing trained
# Detectors that are a fused U-Net trained on other data: (load_model options, validation split)
TRAINED_ON = {
    "fused_atk": ({"data": "atk"}, "val_atk"),
    "fused_gen": ({"data": "gen"}, "val_gen"),
}
# Which training the U-Net of the p_r_e_* receivers ("fused") comes from: (load_model options,
# validation split of its threshold). The general one (experiments/general_training) is the
# default; the nominal one reproduces the results before section 26 of docs/results.md.
UNETS = {"general": ({"data": "gen"}, "val_gen"), "nominal": ({}, "val")}
TRAIN_COMMANDS = {
    "general": "python -m jamloc.experiments.general_training",
    "nominal": "python -m jamloc.train --variant fused",
}


def unet_source(unet: str) -> tuple[dict, str]:
    """``load_model`` options and validation split of the U-Net trained as ``unet``."""
    return UNETS[unet]


# With the estimated noise power: the detectors that were trained or calibrated on it, with the
# validation split their thresholds (and the block test's nulls) come from
NEST_DETECTORS = {"power_threshold": "val_nest", "block_test": "val_nest", "fused": "val_nest"}
# The arms of the resolution study added to the whole-allocation case
RESOLUTION = tuple(rx for rx in RECEIVERS if rx.startswith(("nomap_", "evm_")))
RESOLUTION += ("p_r_e_prb2", "p_r_e_prb4", "true_prb_shrink")
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
    "lmmse_da": {
        "label": "LMMSE, no map, data-aided",
        "color": "#777777",
        "marker": "1",
        "linestyle": "-.",
    },
    "p_r_e_soft_da": {
        "label": "P+R+E map, soft, data-aided",
        "color": "#CC0000",
        "marker": "s",
        "linestyle": "-",
    },
    "true_soft_da": {
        "label": "True map, soft, data-aided",
        "color": "#000000",
        "marker": "d",
        "linestyle": "-",
    },
    "true_prb_soft": {
        "label": "True map per PRB, soft",
        "color": "#117733",
        "marker": "v",
        "linestyle": "--",
    },
    "cell_soft": {
        "label": "Per-cell estimate, no map",
        "color": "#44AA99",
        "marker": "h",
        "linestyle": "--",
    },
    "p_r_e_cell": {
        "label": "Map P+R+E, soft, per cell",
        "color": "#1F78B4",
        "marker": "o",
        "linestyle": "--",
    },
    "p_r_e_refine": {
        "label": "P+R+E map, passes refine map",
        "color": "#E31A1C",
        "marker": "*",
        "linestyle": "-",
    },
    "p_r_e_renoise": {
        "label": "P+R+E map, passes re-estimate N0",
        "color": "#FF7F00",
        "marker": "*",
        "linestyle": "-",
    },
    "true_renoise": {
        "label": "True map, passes re-estimate N0",
        "color": "#6A3D9A",
        "marker": "*",
        "linestyle": "-.",
    },
    "true_cell": {
        "label": "True map, soft, per cell",
        "color": "#00A087",
        "marker": "^",
        "linestyle": "--",
    },
    "p_r_e_full": {
        "label": "P+R+E map, per cell, refining passes",
        "color": "#B2182B",
        "marker": "*",
        "linestyle": "-",
    },
    "true_full": {
        "label": "True map, per cell, refining passes",
        "color": "#1A1A1A",
        "marker": "P",
        "linestyle": "-",
    },
    "cell_full": {
        "label": "Per-cell estimate, no map, passes",
        "color": "#35978F",
        "marker": "h",
        "linestyle": "-.",
    },
    "p_r_e_cell_da": {
        "label": "P+R+E map, per cell, channel passes",
        "color": "#4393C3",
        "marker": "o",
        "linestyle": "-.",
    },
    "p_r_e_refine_prb": {
        "label": "P+R+E map, per PRB, map-refining passes",
        "color": "#D6604D",
        "marker": "s",
        "linestyle": "-.",
    },
    "p_r_e_shrink": {
        "label": "P+R+E map, shrinkage",
        "color": "#762A83",
        "marker": "v",
        "linestyle": ":",
    },
    "p_r_e_shrink_da": {
        "label": "P+R+E map, shrinkage, data-aided",
        "color": "#762A83",
        "marker": "v",
        "linestyle": "--",
    },
    "p_r_e_shrink_full": {
        "label": "P+R+E map, shrinkage, refining passes",
        "color": "#762A83",
        "marker": "v",
        "linestyle": "-",
    },
    "true_shrink": {
        "label": "True map, shrinkage",
        "color": "#9970AB",
        "marker": "^",
        "linestyle": ":",
    },
    "true_shrink_full": {
        "label": "True map, shrinkage, refining passes",
        "color": "#9970AB",
        "marker": "^",
        "linestyle": "-",
    },
    "cell_shrink": {
        "label": "No map, shrinkage over each PRB row",
        "color": "#C2A5CF",
        "marker": "h",
        "linestyle": ":",
    },
    "cell_shrink_full": {
        "label": "No map, shrinkage, gated noise passes",
        "color": "#C2A5CF",
        "marker": "h",
        "linestyle": "-",
    },
    "cell_da": {
        "label": "Per-cell estimate, no map, data-aided",
        "color": "#44AA99",
        "marker": "h",
        "linestyle": "-",
    },
    "nomap_prb": {"label": "No map, per PRB", "color": "#B8B8B8", "marker": "s", "linestyle": ":"},
    "nomap_prb2": {
        "label": "No map, per PRB x2",
        "color": "#A0A0A0",
        "marker": "s",
        "linestyle": "--",
    },
    "nomap_prb4": {
        "label": "No map, per PRB x4",
        "color": "#888888",
        "marker": "s",
        "linestyle": "-.",
    },
    "nomap_cell": {
        "label": "No map, per cell",
        "color": "#5AAE61",
        "marker": "h",
        "linestyle": "--",
    },
    "nomap_shrink": {
        "label": "No map, shrinkage",
        "color": "#1B7837",
        "marker": "h",
        "linestyle": "-",
    },
    "evm_prb": {"label": "EVM map, per PRB", "color": "#FDB863", "marker": "D", "linestyle": ":"},
    "evm_cell": {
        "label": "EVM map, per cell",
        "color": "#E08214",
        "marker": "D",
        "linestyle": "--",
    },
    "evm_shrink": {
        "label": "EVM map, shrinkage",
        "color": "#B35806",
        "marker": "D",
        "linestyle": "-",
    },
    "p_r_e_prb2": {
        "label": "P+R+E map, per PRB x2",
        "color": "#6BAED6",
        "marker": "o",
        "linestyle": "--",
    },
    "p_r_e_prb4": {
        "label": "P+R+E map, per PRB x4",
        "color": "#3182BD",
        "marker": "o",
        "linestyle": "-.",
    },
    "true_prb_shrink": {
        "label": "True map per PRB, shrinkage",
        "color": "#117733",
        "marker": "v",
        "linestyle": "-",
    },
    "p_r_e_atk_soft": {
        "label": "Map P+R+E (mixed attacks), soft",
        "color": "#332288",
        "marker": "d",
        "linestyle": "-",
    },
    "gen_soft": {
        "label": "Map P+R+E (general training), per PRB",
        "color": "#882255",
        "marker": "P",
        "linestyle": "--",
    },
    "gen_shrink": {
        "label": "Map P+R+E (general training), shrinkage",
        "color": "#882255",
        "marker": "P",
        "linestyle": "-",
    },
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
    if detector in ("evm_threshold", "residual_threshold"):
        baseline = evm_scores if detector == "evm_threshold" else residual_scores
        return torch.from_numpy(baseline(features.cpu().numpy())).to(features.device)
    variant = "fused" if detector in TRAINED_ON else detector
    return model(select_inputs(features, variant))


def val_threshold(detector: str, model=None, val_split: str | None = None) -> float:
    """Score threshold that flags 1% of the non-jammed validation cells (of the validation split
    that belongs to the detector's training data, unless ``val_split`` is given)."""
    d = load(val_split or (TRAINED_ON[detector][1] if detector in TRAINED_ON else "val"))
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


def detector_cells(batch, cfg, detector: str, model, thr: float, *, return_scores=False):
    """Jamming map ``[B, 24, 14]`` of a detector. For the block test, ``model`` is the loaded
    ``val`` split whose benign slots give its empirical nulls. With ``return_scores``, also
    expose the evidence behind the map as a pair (raw scores, decisions)."""
    with torch.no_grad():
        features = extract(batch, cfg)
        if detector == "block_test":
            n, sc = len(features), batch.scenario
            d = {
                "features": features.cpu().numpy(),
                "alloc_start": np.full(n, sc.alloc_start),
                "alloc_size": np.full(n, sc.alloc_size),
                "snr_db": sc.snr_db.cpu().numpy(),
                "snr_db_est": estimated_snr_db(batch.y_alloc, batch.no, cfg).cpu().numpy(),
            }
            scores = torch.from_numpy(block_test_scores(d, model)).to(features.device)
        else:
            scores = _scores(detector, model, features)
        decisions = scores > thr
        return (scores, decisions) if return_scores else decisions


def coarsen_to_prb(cells: torch.Tensor, dmrs_symbols: int | tuple[int, ...]) -> torch.Tensor:
    """A map at the block test's resolution: preserve each DMRS cell and flag every data
    cell of the PRB if any of its data cells is flagged."""
    data = torch.ones(cells.shape[-1], dtype=torch.bool, device=cells.device)
    data[list(dmrs_symbols) if isinstance(dmrs_symbols, tuple) else dmrs_symbols] = False
    out = cells.clone()
    out[:, :, data] = cells[:, :, data].any(dim=-1, keepdim=True).expand(-1, -1, int(data.sum()))
    return out


def _scenario(
    rng, n, channel, attack, jsr_db, snr_db, whole_allocation=False, jam_size=JAM_SIZE
) -> Scenario:
    if whole_allocation:
        start, size = np.full(n, ALLOC[0]), np.full(n, ALLOC[1])
    else:
        start = rng.integers(ALLOC[0], ALLOC[0] + ALLOC[1] - jam_size + 1, n)
        size = np.full(n, jam_size)
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


def simulate_point(
    link, detectors, scenario_fn, receivers, max_mc_iter, target_block_errors, fixed=False
):
    """Bit and block errors of each receiver, all decoding the same simulated batches, and per
    receiver which of the slots it decoded failed.

    A receiver stops once it has ``target_block_errors`` block errors (its estimate is precise
    enough); the others keep decoding further batches. So receivers that rarely fail get many
    slots without making the frequent failures expensive. With ``fixed``, exactly
    ``max_mc_iter`` batches are simulated even when no receiver is left (a replay).
    """
    cfg = link.cfg
    alloc = slice(ALLOC[0], ALLOC[0] + ALLOC[1])
    counts = {rx: np.zeros(4) for rx in receivers}  # bit errors, bits, block errors, blocks
    failed = {rx: [] for rx in receivers}  # per slot: block error or not
    active = list(receivers)
    map_aware = None
    for _ in range(max_mc_iter):
        if not active and not fixed:
            break
        out = link(scenario_fn(), decode="standard" in active, receive=bool(active))
        if not active:
            continue
        map_aware = map_aware or MapAwareReceiver(out.tx, cfg)
        maps = {"true": out.labels[:, alloc]}
        maps["true_prb"] = coarsen_to_prb(maps["true"], cfg.dmrs_symbols)
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
                passes = DATA_AIDED.get(rx, 0)
                b_hat = map_aware(
                    out.y_alloc, out.no, cells, mode=mode, iterations=passes, **OPTIONS.get(rx, {})
                )["b_hat"]
            errors = out.b != b_hat
            block_errors = errors.any(-1).reshape(-1)
            counts[rx] += [
                errors.sum().item(),
                errors.numel(),
                block_errors.sum().item(),
                block_errors.numel(),
            ]
            failed[rx].append(block_errors.cpu().numpy())
            if counts[rx][2] >= target_block_errors:
                active.remove(rx)
    return counts, {rx: np.concatenate(f) for rx, f in failed.items() if f}


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
    noise_estimate="true",
    replay=None,
    attacks=None,
    jam_size=JAM_SIZE,
    mcs_index=None,
    unet="general",
):
    """Rows of block-error counts (one per receiver and point), and the per-slot outcomes keyed by
    ``slot_key``. ``replay = (batches, decode)``: simulate every point with ``batches[point]``
    batches, decoding only the receivers ``decode[point]``. ``attacks``: simulate only these cases
    (``"none"``, ``"pilot"``, ``"data"``, ``"partial"``, ``"pilot_whole_allocation"``).
    ``jam_size``: PRBs of the pilot, data and partial-band attacks (inside the allocation).
    ``unet``: the training of the ``p_r_e_*`` receivers' U-Net (``UNETS``)."""
    cfg = dataclasses.replace(SimConfig(), noise_estimate=noise_estimate, **JAMMERS[jammer])
    if mcs_index is not None:  # another MCS: the detectors were trained on the default one
        cfg = dataclasses.replace(cfg, mcs_index=mcs_index)
    link = JammedPUSCHLink(cfg)
    nest = noise_estimate != "true"
    detectors = {}
    for det in DETECTORS:
        if nest and det not in NEST_DETECTORS:
            continue  # never trained or calibrated on the estimated noise power
        val_split = NEST_DETECTORS[det] if nest else None
        if det in NO_MODEL:
            model = None
        elif det == "block_test":
            model = load(val_split or "val")  # its empirical nulls
        elif nest:
            model = load_model(det, seed, data="nest")
        elif det == "fused":
            options, val_split = unet_source(unet)
            if not checkpoint_path("fused", seed, **options).is_file():
                other = " or pass --unet nominal for the U-Net of the earlier results"
                raise FileNotFoundError(
                    f"no {unet} U-Net for seed {seed}. Train it with {TRAIN_COMMANDS[unet]}"
                    + (other if unet == "general" else "")
                )
            model = load_model("fused", seed, **options)
        elif det in TRAINED_ON:
            options, val_split = TRAINED_ON[det]
            if not checkpoint_path("fused", seed, **options).is_file():
                continue  # not trained: its receiver is left out
            if not (data_dir() / f"{val_split}.npz").is_file():
                continue
            model = load_model("fused", seed, **options)
        else:
            model = load_model(det, seed)
        detectors[det] = (model, val_threshold(det, model, val_split))
    rng = seed_everything(1000 + seed)

    # Clean slots first: what each receiver costs when there is no jammer (false alarms of a
    # detector map erase clean data and drop clean pilots)
    if jammer == "nominal":
        cases = [("none", False, tuple(RECEIVERS))]
        cases += [(attack, False, tuple(RECEIVERS)) for attack in ("pilot", "data", "partial")]
        # limit case: no clean pilot left
        whole = ("standard", "lmmse", "p_r_e_soft", "true_hard", "true_soft", "cell_soft")
        whole += ("p_r_e_cell", "true_cell", "p_r_e_shrink", "true_shrink", "cell_shrink")
        whole += RESOLUTION
        whole += tuple(DATA_AIDED)
        whole += ("gen_soft", "gen_shrink")
        cases.append(("pilot", True, whole))  # per PRB = exact here
    else:  # a pulsed or hopping pilot attack is the nominal one (it has a single OFDM symbol)
        cases = [(attack, False, tuple(RECEIVERS)) for attack in ("data", "partial")]
    # leave out receivers whose detector is not available (e.g. a model that was not trained)
    always = (None, "true", "true_prb")
    usable = {rx for rx, (src, _) in RECEIVERS.items() if src in always or src in detectors}
    cases = [(a, w, tuple(rx for rx in rxs if rx in usable)) for a, w, rxs in cases]
    if only and replay is None:  # re-run some receivers only
        cases = [(a, w, tuple(rx for rx in rxs if rx in only)) for a, w, rxs in cases]
        cases = [c for c in cases if c[2]]
    if attacks:
        cases = [c for c in cases if ("pilot_whole_allocation" if c[1] else c[0]) in attacks]
    rows, slots = [], {}
    for attack, whole, receivers in cases:
        name = "pilot_whole_allocation" if whole else attack
        for jsr in jsr_dbs[:1] if attack == "none" else jsr_dbs:  # JSR is irrelevant then

            def scenario_fn(attack=attack, jsr=jsr, whole=whole):
                return _scenario(rng, batch_size, channel, attack, jsr, snr_db, whole, jam_size)

            if replay is None:
                counts, failed = simulate_point(
                    link, detectors, scenario_fn, receivers, max_mc_iter, target_block_errors
                )
            else:
                point = tuple(slot_key(name, jsr, "").split(";")[:2])
                batches, decode = replay
                receivers = tuple(rx for rx in decode.get(point, ()) if rx in usable)
                counts, failed = simulate_point(
                    link,
                    detectors,
                    scenario_fn,
                    receivers,
                    batches[point],
                    target_block_errors,
                    fixed=True,
                )
            for rx, f in failed.items():
                slots[slot_key(name, jsr, rx)] = f
            for rx in receivers:
                bit_err, bits, block_err, blocks = counts[rx]
                rows.append(
                    {
                        "receiver": rx,
                        "jammer": jammer,
                        "noise_estimate": noise_estimate,
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
            if receivers:
                print(
                    f"[{channel} {jammer} {noise_estimate} seed {seed}] {name:22s} "
                    f"JSR {jsr:+5.1f} dB: "
                    + "  ".join(f"{rx} {counts[rx][2] / counts[rx][3]:.3f}" for rx in receivers),
                    flush=True,
                )
    return rows, slots


def slot_key(attack: str, jsr_db: float, receiver: str) -> str:
    """Name of a receiver's per-slot outcomes at one point in the ``*_slots.npz`` files."""
    jsr = "none" if attack == "none" else f"{float(jsr_db):g}"
    return f"{attack};{jsr};{receiver}"


def paired_rows(slot_files) -> list[dict]:
    """BLER difference of every pair of receivers at every point, on the same slots.

    Within each seed's file, the two receivers are compared on the slots both decoded (the shorter
    run is a prefix of the longer one); the seeds' slots are then pooled, as in ``pool_seeds``.
    """
    runs = [dict(np.load(path)) for path in slot_files]
    points = {}
    for run in runs:
        for key in run:
            attack, jsr, rx = key.split(";")
            points.setdefault((attack, jsr), set()).add(rx)
    rows = []
    for (attack, jsr), present in points.items():
        order = [rx for rx in RECEIVERS if rx in present]
        for i, first in enumerate(order):
            for second in order[i + 1 :]:
                table = np.zeros(4, dtype=np.int64)  # both fail, only first, only second, neither
                k1, k2 = (f"{attack};{jsr};{rx}" for rx in (first, second))
                for run in runs:
                    if k1 not in run or k2 not in run:
                        continue
                    m = min(len(run[k1]), len(run[k2]))
                    x, y = run[k1][:m], run[k2][:m]
                    table += [(x & y).sum(), (x & ~y).sum(), (~x & y).sum(), (~x & ~y).sum()]
                n = int(table.sum())
                if not n:
                    continue
                delta, lo, hi = paired_difference_interval(*table)
                rows.append(
                    {
                        "first": first,
                        "second": second,
                        "attack": attack,
                        "jsr_db": float("nan") if jsr == "none" else float(jsr),
                        "slots": n,
                        "only_first": int(table[1]),
                        "only_second": int(table[2]),
                        "bler_first": (table[0] + table[1]) / n,
                        "bler_second": (table[0] + table[2]) / n,
                        "delta": delta,
                        "delta_lo": lo,
                        "delta_hi": hi,
                    }
                )
    return rows


# Receivers of the paper's BLER table, and the points (attack, JSR) and receivers it quotes
TABLE_RECEIVERS = (
    "standard",
    "lmmse",
    "slot_ie",
    "dmrs_ie",
    "power_hard",
    "block_soft",
    "p_r_e_hard",
    "p_r_e_soft",
    "p_r_e_atk_soft",
    "true_prb_soft",
    "true_soft",
)
_WEAK = ("p_r_e_hard", "p_r_e_soft", "true_soft")  # what using the map costs at weak jamming
QUOTED = {
    ("nominal", 15.0): {
        ("none", "none"): ("standard", *_WEAK),
        **{(a, "-15"): _WEAK for a in ("pilot", "data", "partial")},
        ("partial", "3"): ("p_r_e_soft", "true_prb_soft", "true_soft"),
        **{(a, "9"): TABLE_RECEIVERS for a in ("pilot", "data", "partial")},
    },
    ("pulsed", 15.0): {(a, "9"): TABLE_RECEIVERS for a in ("data", "partial")},
    ("hopping", 15.0): {(a, "9"): TABLE_RECEIVERS for a in ("data", "partial")},
    ("nominal", 8.0): {
        ("none", "none"): ("standard", "lmmse", "p_r_e_soft", "true_soft"),
        **{
            (a, "3"): ("standard", "p_r_e_soft", "true_prb_soft", "true_soft")
            for a in ("data", "partial")
        },
    },
}


def batches_per_point(path, batch_size: int) -> dict:
    """Number of batches each point of an earlier per-seed CSV simulated: the receiver that
    decoded longest decoded every batch."""
    out = {}
    for r in _read_rows(path):
        key = tuple(slot_key(r["attack"], r["jsr_db"], "").split(";")[:2])
        out[key] = max(out.get(key, 0), r["blocks"] // batch_size)
    return out


def check_replay(rows, path) -> list[str]:
    """Rows of a replay whose counts differ from the earlier run (empty if it reproduced)."""
    old = {
        (r["receiver"], r["attack"], slot_key(r["attack"], r["jsr_db"], "")): r
        for r in _read_rows(path)
    }
    bad = []
    for r in rows:
        o = old.get((r["receiver"], r["attack"], slot_key(r["attack"], r["jsr_db"], "")))
        if o is None or (o["block_errors"], o["blocks"]) != (r["block_errors"], r["blocks"]):
            got = (r["block_errors"], r["blocks"])
            was = None if o is None else (o["block_errors"], o["blocks"])
            bad.append(f"{r['receiver']} {r['attack']} {r['jsr_db']}: {got} vs {was}")
    return bad


MAIN_RECEIVERS = ("standard", "dmrs_ie", "power_hard", "p_r_e_soft", "true_soft")
ABLATION_RECEIVERS = ("standard", "p_r_e_hard", "p_r_e_soft", "p_e_soft", "true_hard", "true_soft")
# The receiver held fixed (soft weighting), only the source of the map changes
SOURCE_RECEIVERS = (
    "lmmse",
    "slot_ie",
    "dmrs_ie",
    "block_soft",
    "p_r_e_soft",
    "p_r_e_atk_soft",
    "true_prb_soft",
    "true_soft",
)


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
        text = ("receiver", "attack", "jammer", "seed", "noise_estimate")
        counts = ("block_errors", "blocks")
        return [
            {
                k: v if k in text else int(float(v)) if k in counts else float(v)
                for k, v in r.items()
            }
            for r in csv.DictReader(f)
        ]


def rows_rx(rows) -> set:
    return {r["receiver"] for r in rows}


def _write_rows(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    p.add_argument(
        "--noise-estimate",
        choices=NOISE_ESTIMATES,
        default="true",
        help="noise power of the receiver: the true one, or estimated per slot (jamloc.noise)",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        choices=("none", "pilot", "data", "partial", "pilot_whole_allocation"),
        help="simulate only these cases",
    )
    p.add_argument(
        "--jam-prb",
        type=int,
        choices=range(1, ALLOC[1] + 1),
        metavar=f"1..{ALLOC[1]}",
        help=f"jammed PRBs of the nominal attacks (default {JAM_SIZE}); written to *_jam<N> files "
        "(experiments/width_sweep)",
    )
    p.add_argument("--plot-only", action="store_true", help="redraw from the saved CSV")
    p.add_argument(
        "--unet",
        choices=list(UNETS),
        default="general",
        help="training of the U-Net of the p_r_e_* receivers",
    )
    p.add_argument(
        "--replay",
        metavar="DIR",
        help="repeat the run whose per-seed CSVs are in DIR, decoding only the quoted points",
    )
    p.add_argument(
        "--receivers",
        nargs="+",
        choices=list(RECEIVERS),
        help="simulate only these receivers and replace their rows in the saved CSV",
    )
    p.add_argument(
        "--mcs", type=int, default=None, help="MCS index (table 1) instead of the default"
    )
    args = p.parse_args(argv)

    out = results_dir() / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"recovery_{args.channel}"
    if args.jammer != "nominal":
        stem = stem.with_name(f"{stem.name}_{args.jammer}")
    if args.snr_db != 15.0:
        stem = stem.with_name(f"{stem.name}_snr{args.snr_db:g}")
    if args.noise_estimate != "true":
        stem = stem.with_name(f"{stem.name}_nest")
    if args.jam_prb is not None:
        stem = stem.with_name(f"{stem.name}_jam{args.jam_prb}")
    if args.mcs is not None:
        stem = stem.with_name(f"{stem.name}_mcs{args.mcs}")
    original = stem  # the name of the run a replay repeats
    if args.replay:
        stem = stem.with_name(f"{stem.name}_replay")

    # Every seed has its own CSV, so seeds can run as parallel processes; the combined CSV and
    # the figures pool whatever seed files exist
    def seed_path(seed):
        return stem.with_name(f"{stem.name}_seed{seed}.csv")

    def slots_path(seed):
        return stem.with_name(f"{stem.name}_seed{seed}_slots.npz")

    if not args.plot_only:
        for seed in args.seeds:
            replay = None
            if args.replay:
                before = Path(args.replay) / f"{original.name}_seed{seed}.csv"
                replay = (
                    batches_per_point(before, args.batch_size),
                    QUOTED[args.jammer, args.snr_db],
                )
                # the JSR grid of the earlier run
                args.jsr_db = sorted(
                    {r["jsr_db"] for r in _read_rows(before) if r["jsr_db"] == r["jsr_db"]}
                )
            rows, slots = simulate(
                args.channel,
                args.snr_db,
                args.jsr_db,
                args.batch_size,
                args.max_mc_iter,
                args.target_block_errors,
                seed,
                args.receivers,
                args.jammer,
                args.noise_estimate,
                replay,
                args.attacks,
                JAM_SIZE if args.jam_prb is None else args.jam_prb,
                args.mcs,
                args.unet,
            )
            if args.replay:
                bad = check_replay(rows, before)
                print(f"replay of seed {seed}: " + ("; ".join(bad) or "all counts reproduced"))
            path = seed_path(seed)
            if args.receivers and path.is_file():  # keep the other receivers' rows
                new = rows_rx(rows)
                rows = [r for r in _read_rows(path) if r["receiver"] not in new] + rows
                if slots_path(seed).is_file():
                    slots = {**dict(np.load(slots_path(seed))), **slots}
            _write_rows(rows, path)
            np.savez_compressed(slots_path(seed), **slots)
    seed_files = sorted(stem.parent.glob(f"{stem.name}_seed*.csv"))
    if seed_files:
        rows = [r for path in seed_files for r in _read_rows(path)]
        _write_rows(rows, stem.with_suffix(".csv"))
    else:  # a CSV from before the per-seed files
        rows = _read_rows(stem.with_suffix(".csv"))
    pooled = pool_seeds(rows)
    slot_files = sorted(stem.parent.glob(f"{stem.name}_seed*_slots.npz"))
    if slot_files:
        _write_rows(paired_rows(slot_files), stem.with_name(f"{stem.name}_paired.csv"))
    if (
        args.jam_prb is not None
        or args.mcs is not None
        or not any(r["attack"] in ("pilot", "data", "partial") for r in pooled)
    ):
        # e.g. --attacks pilot_whole_allocation: the figures plot the nominal attacks only; a
        # jammer width other than the nominal one is plotted against width by width_sweep
        print(f"Wrote {stem}.csv and {stem.name}_paired.csv (no figure)")
        return
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
