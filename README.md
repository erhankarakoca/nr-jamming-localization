# Receiver-Informed Detection and Localization of Resource-Selective Jamming in the 5G NR Uplink

**[Project page](https://erhankarakoca.github.io/nr-jamming-localization/)** ·
[Results](docs/results.md) ·
[Reproduce](docs/reproducing.md) ·
[Proposal](docs/proposal.pdf) ·
[Open in Colab](https://colab.research.google.com/github/erhankarakoca/nr-jamming-localization/blob/main/notebooks/colab_setup.ipynb)

A jammer that knows the 5G NR frame structure can hit only the pilots (DMRS) or only the data of
an uplink slot. This project lets the gNB map such attacks in every slot, cell by cell over
PRBs × OFDM symbols, and then decode through them with the map. Everything is simulated in
[Sionna](https://nvlabs.github.io/sionna/) with the PyTorch backend.

![From the received grid to decoding with the jamming map](docs/img/system_views.png)
<sub>One simulated slot with partial-band jamming on 4 of the user's 12 PRBs: (a) received grid,
(b) channel estimate, (c) equalized symbols, (d) features, (e) the U-Net's map, (f) decoding with
the map.</sub>

## How it works

1. **Three clues per cell**, from what the receiver already computes: received power, the DMRS
   residual (the part of the pilot estimates outside the channel's delay subspace) and the EVM
   of the equalized symbols. With no jamming, the channel inside that subspace and white noise
   of known power, the residual has a known distribution, which gives an analytic threshold.
2. **A small U-Net** (66,553 parameters) turns the clues into a 24 PRB × 14 symbol jamming map.
3. **Decoding with the map**: the receiver estimates the channel from the clean pilots and gives
   the bits of jammed cells little weight. Optional data-aided passes reuse decoded data as
   extra pilots, which helps when every pilot is jammed.

## Results

| | Result | Conditions |
| --- | --- | --- |
| Detection | Cell-level F1 0.952 at 1.15% per-cell false alarms; a training-free block test reaches 0.868 | Test split (TDL-A and TDL-C, JSR −15 to +9 dB, SNR 5–25 dB), three seeds |
| Decoding | Block error rate 87.8% with the standard receiver, 0.02% with the map | Partial-band jamming at +9 dB JSR on 4 of 12 PRBs, TDL-C, 15 dB SNR |
| Jammer width | 0.06%, 2.1% and 38% with the map when 4, 6 and 8 of 12 PRBs are jammed (standard receiver: 53%, 79%, 92%) | Partial-band jamming at +3 dB JSR, TDL-C, 15 dB SNR |
| Speed | 0.68 ms per slot to compute the features and run the U-Net | One core of an Apple M1 Pro, 24-PRB allocation; channel estimation, equalization and decoding not included |

These come from a synchronized, single-antenna simulation with known noise power.
[docs/results.md](docs/results.md) gives the operating point of each result and the cases where
the method does less well: hopping jammers, estimated noise power, and jammers that cover most of
the allocation.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv; for Windows see docs/development.md
git clone https://github.com/erhankarakoca/nr-jamming-localization.git
cd nr-jamming-localization
uv sync --locked
uv run pytest
```

[`notebooks/01_link_walkthrough.ipynb`](notebooks/01_link_walkthrough.ipynb) steps through one
slot. To rebuild the paper's results on a GPU:

```bash
uv run python -m jamloc.experiments.paper_run --seeds 0 1 2
```

[docs/reproducing.md](docs/reproducing.md) lists every other experiment, and
[docs/development.md](docs/development.md) covers Colab, VS Code, the code map and the Overleaf
paper.

## Repository

| Path | Contents |
| --- | --- |
| `src/jamloc/` | Link simulation, features, U-Net, baselines and the map-aware receiver |
| `src/jamloc/experiments/` | One script per experiment and figure |
| `site/` | The project page |
| `tests/` | pytest suite |
| `notebooks/` | One-slot walkthrough and Colab setup |
| `docs/` | Results log, claims register, plans, proposal |

Datasets and checkpoints are not committed: the scripts rebuild them in `data/` and `results/`
from recorded seeds. CPU and GPU runs can draw different samples and train different checkpoints.

---

CMPE 581 project, Boğaziçi University, Fall 2026 · Erhan Karakoca
