# Receiver-Informed Detection and Localization of Resource-Selective Jamming in the 5G NR Uplink

CMPE 581 project (Boğaziçi University, Fall 2026), Erhan Karakoca.

Jammers that know the 5G NR frame structure can attack only the DMRS pilots or only the
data resources of an uplink slot. This project asks whether a gNB can detect such attacks
and **locate them on the PRB × OFDM-symbol grid** using measurements its receiver already
computes: received power, a DMRS pilot residual, and decision-directed EVM. These three
feature maps are fused by a small U-Net that outputs a per-cell jamming map. Everything is
simulated in [Sionna](https://nvlabs.github.io/sionna/) (PyTorch backend).

The full proposal is in [`docs/proposal.pdf`](docs/proposal.pdf).

## Setting up on a new computer

You need **git** with access to this repo (an SSH key on your GitHub account, or `gh auth login`)
and **[uv](https://docs.astral.sh/uv/)**:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
git clone git@github.com:erhankarakoca/nr-jamming-localization.git
cd nr-jamming-localization
uv sync          # creates .venv/ with the exact versions in uv.lock (downloads Python 3.12 if needed)
uv run pytest    # checks that Sionna + PyTorch work
```

## VS Code

Open the project folder in VS Code and install the recommended extensions when asked (Python,
Jupyter, Ruff, Google Colab). After `uv sync`, VS Code picks up `.venv` by itself. Tests appear
in the Testing panel, and files are formatted with Ruff on save.

## Running on a Colab GPU

Use [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) at the start of each Colab
session. It clones or updates the repo on the Colab machine and installs it into Colab's Python,
keeping Colab's CUDA build of PyTorch (no uv there). Sionna uses the GPU automatically.

- **From VS Code (Google Colab extension):** open the notebook, choose *Select Kernel → Colab →
  New Colab Server* and a GPU, then run the cells. Colab Secrets and Google Drive don't work
  through the extension yet ([googlecolab/colab-vscode#215](https://github.com/googlecolab/colab-vscode/issues/215)),
  so the notebook asks for your GitHub token and keeps files on the Colab machine. Download
  results before the session ends.
- **From the browser:**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erhankarakoca/nr-jamming-localization/blob/main/notebooks/colab_setup.ipynb)
  Set `BROWSER_COLAB = True` in the first cell to read the token from the Colab secret
  `GITHUB_TOKEN` and save data and results to Google Drive. The first time, Colab asks to access
  GitHub; allow private repositories.

Either way you need a fine-grained GitHub token limited to this repo (steps in the notebook).
Recommended split: write code locally and push, and use Colab for heavy jobs (dataset generation,
training).

## Day-to-day workflow

```bash
git pull                  # get work from the other machine
uv sync                   # pick up any dependency changes
uv run python <script>    # or activate: source .venv/bin/activate (macOS/Linux), .venv\Scripts\activate (Windows)
uv run jupyter lab        # notebooks
git add -A && git commit -m "..." && git push
```

Add a dependency with `uv add <package>` (dev-only tools: `uv add --dev <package>`) and commit
both `pyproject.toml` and `uv.lock`.

## How it works

One uplink slot is simulated as a chain of Sionna blocks, written out step by step like the Sionna
tutorials ([`notebooks/01_link_walkthrough.ipynb`](notebooks/01_link_walkthrough.ipynb) walks
through one slot):

```text
UE: PUSCH transmitter (n PRBs) -> place on 24-PRB carrier -> TDL channel --+
                                                                           +-> AWGN -> gNB
Jammer: Gaussian noise on attacked PRB x symbol cells -> TDL channel ------+

gNB: take scheduled PRBs -> LS channel estimation -> LMMSE equalizer -> demapper -> LDPC decoder
         |                        |                        |
         v                        v                        v
     power P              DMRS residual R              EVM E        (one value per PRB x symbol)
         +------------------------+------------------------+
                                  v
                 small U-Net (66k params) -> jamming map (24 PRB x 14 symbols)
```

| Module | What it does |
| --- | --- |
| `config.py` | Simulation parameters (proposal Table 1) |
| `ue.py` | PUSCH transmitter per allocation size; placing the allocation on the carrier |
| `jammer.py` | Attack patterns (pilot / data / partial-band) and the Gaussian jammer |
| `receiver.py` | Receiver chain: LS estimation, LMMSE equalization, demapping, decoding |
| `link.py` | The whole slot: UE, channels, jammer, noise, receiver |
| `features.py` | Power, DMRS residual and EVM maps, plus allocation masks |
| `baselines.py` | Threshold detectors (EVM, residual, residual exact p-value, per-cell power) and the training-free block test |
| `theory.py` | Exact statistics of the DMRS residual (one slot or averaged over K): threshold, Pd, leakage |
| `recovery.py` | Map-aware receiver: hard (drop/erase) and soft (weight by estimated jamming) |
| `model.py`, `train.py` | U-Net detector, per-cell MLP (ablation), training (optionally with noise-estimate jitter) |
| `dataset.py` | Train / validation / test / scheduling-aware splits with fixed seeds |
| `evaluate.py`, `metrics.py` | Pd at 1% per-cell false alarm vs JSR, cell-level F1 |
| `experiments/` | One script per experiment and figure (see below) |

### Reproducing the results

Every command writes into `data/` or `results/` (or the folders set by `JAMLOC_DATA_DIR` /
`JAMLOC_RESULTS_DIR`). Times are for a MacBook Pro M1 Pro, CPU only.

**Detection experiment (datasets, training, evaluation, recovery, figures).** One command does
it all and leaves `results/full_run.zip` (evaluation CSVs and plots, trained models with their
loss histories, recovery curves, explanatory figures and the gallery page). Training 4 variants x 3 seeds takes about 2.5 hours on a laptop CPU, so
it is meant for a Colab GPU: run [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb),
whose last cell is this command.

```bash
uv run python -m jamloc.experiments.full_run --seeds 0 1 2
```

The same steps one by one:

```bash
for s in train val test sched; do uv run python -m jamloc.dataset --split $s; done   # ~2 min
uv run python -m jamloc.train --variant all --epochs 15 --seeds 0 1 2
uv run python -m jamloc.evaluate --split test                   # mean +- std over seeds
uv run python -m jamloc.evaluate --split sched                  # scheduling-aware jammer
uv run python -m jamloc.evaluate --split test --snr-db 15 25    # one SNR range only
uv run python -m jamloc.experiments.recovery --channel A        # map-aware receiver vs none
uv run python -m jamloc.experiments.figures                     # explanatory figures + gallery
```

**Figures:** everything lands in `results/` as PDF (paper) and PNG (slides).
`results/figures/index.html` shows every figure with a caption saying what it shows and why;
open it in any browser.

On Windows PowerShell, run the dataset loop as
`foreach ($s in "train","val","test","sched") { uv run python -m jamloc.dataset --split $s }`.

**Second round** (`docs/plan-v2.md`): the residual's theory, unseen conditions, network size,
mixed training. All reuse the trained models and the `val` / `test` splits:

```bash
uv run python -m jamloc.experiments.residual_theory    # analytic threshold and Pd vs simulation
uv run python -m jamloc.experiments.generalization     # long delays, 30 m/s, QPSK jammer (GPU)
uv run python -m jamloc.experiments.size_sweep         # narrower U-Nets: train + evaluate (GPU)
uv run python -m jamloc.experiments.size_sweep --latency   # their CPU latency (run on the CPU)
uv run python -m jamloc.experiments.robust_training    # retrain on mixed channels (GPU)
```

**Third round** (`docs/plan-v3.md`): fair baselines, detection before damage, calibration,
multi-slot residual. All run on a laptop CPU (the jittered U-Net trains in about 15 min per seed):

```bash
uv run python -m jamloc.train --variant fused --arch pixel --epochs 15 --seeds 0 1 2   # per-cell MLP
uv run python -m jamloc.train --variant fused --epochs 15 --seeds 0 1 2 --noise-jitter-db 1
uv run python -m jamloc.evaluate --split test        # now also: block test, per-cell MLP, jittered U-Net
uv run python -m jamloc.experiments.detection_margin # slot-level Pd vs unprotected BLER (needs ber_bler)
uv run python -m jamloc.experiments.calibration      # false alarms vs SNR and vs noise-estimate error
uv run python -m jamloc.experiments.multislot        # residual averaged over K slots: theory vs checks
```

**Fourth round** (`docs/plan-v4.md`): smarter jammers that use the DMRS configuration (pilot comb,
symbol hopping) and retraining on a mix of attacks (about 45 min on a laptop CPU):

```bash
uv run python -m jamloc.experiments.generalization            # now also the comb and hopping grids
uv run python -m jamloc.experiments.attack_training --seeds 0 1 2
```

**Other figures** (run locally; latency should be measured on the target CPU):

```bash
uv run python -m jamloc.experiments.ber_bler --channel A        # BER/BLER vs JSR (~4 min each)
uv run python -m jamloc.experiments.ber_bler --channel C
uv run python -m jamloc.experiments.feature_maps                # example feature maps
uv run python -m jamloc.experiments.latency                     # CPU latency per slot, 1 thread
```

## Layout

| Path | Contents |
| --- | --- |
| `src/jamloc/` | Project package: simulation, jammer, features, model, evaluation |
| `tests/` | pytest tests (`uv run pytest`) |
| `notebooks/` | Exploration notebooks |
| `docs/` | Proposal, notes, figures for the report |
| `data/`, `results/` | Generated datasets, checkpoints, results. **Not in git.** |

## Data policy

Datasets (tens of thousands of simulated slots) and model checkpoints are **not committed**.
They are regenerated from the code and fixed random seeds, so any machine can rebuild them.
