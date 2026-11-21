# Receiver-Informed Detection and Localization of Resource-Selective Jamming in the 5G NR Uplink

CMPE 581 project (Boğaziçi University, Fall 2026), Erhan Karakoca.

Jammers that know the 5G NR frame structure can attack only the DMRS pilots or only the
data resources of an uplink slot. This project asks whether a gNB can detect such attacks
and **locate them on the PRB × OFDM-symbol grid**. The detector derives received power, a DMRS
delay-subspace residual, and decision-directed EVM from the received grid and equalizer output.
Thresholding the U-Net's scores gives a per-cell jamming map. Everything is simulated in
[Sionna](https://nvlabs.github.io/sionna/) (PyTorch backend).

The full proposal is in [`docs/proposal.pdf`](docs/proposal.pdf).

## Showcase site

The [showcase site](https://erhankarakoca.github.io/nr-jamming-localization/) follows a simulated
uplink slot from the received grid through decoding and presents selected results.

## Setting up on a new computer

You need **git**, GitHub SSH access for the clone command below, and **[uv](https://docs.astral.sh/uv/)**.
The project requests Python 3.12 locally.

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
uv sync --locked  # creates .venv/ using the committed uv.lock
uv run pytest     # includes a Sionna/PyTorch smoke test
```

## VS Code

Open the project folder in VS Code and install the recommended extensions when asked (Python,
Jupyter, Ruff, Google Colab). The workspace enables pytest discovery and configures Ruff to
format Python files on save. Select `.venv` if VS Code does not discover it.

## Running on a Colab GPU

Use [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) at the start of each Colab
session. It clones or updates the repo on the Colab machine and installs it into Colab's Python,
where it can reuse a compatible preinstalled PyTorch build. By default, Sionna uses CUDA when
a compatible GPU runtime is available.

- **From VS Code (Google Colab extension):** open the notebook, choose *Select Kernel → Colab →
  New Colab Server* and a GPU, then run the cells. The notebook's default path prompts if
  `GITHUB_TOKEN` is missing and keeps files on the Colab machine. Download results before the
  session ends. The extension supports `drive.mount()` from v0.2.1. Use the browser to read
  Colab Secrets. See the [extension notes](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds).
- **From the browser:**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erhankarakoca/nr-jamming-localization/blob/main/notebooks/colab_setup.ipynb)
  Set `BROWSER_COLAB = True` in the first cell to mount Google Drive for data and results.
  If `GITHUB_TOKEN` is missing from the environment, the notebook reads the Colab secret of
  the same name.

See the notebook for steps to create a fine-grained GitHub token limited to this repo.
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

gNB: received grid -> take scheduled PRBs -> LS estimator -> LMMSE equalizer -> demapper -> LDPC decoder

Full received grid       -> power P
Scheduled received grid  -> raw pilot LS -> delay-subspace residual R
Equalized scheduled data -> decision-directed EVM E

P, R, E + allocation, DMRS and data masks
  -> U-Net (66,553 parameters) -> threshold -> jamming map (24 PRBs x 14 symbols)
```

| Module | What it does |
| --- | --- |
| `config.py` | Simulation parameters (proposal Table 1) |
| `ue.py` | PUSCH transmitter per allocation size; placing the allocation on the carrier |
| `jammer.py` | Attack patterns (pilot / data / partial-band) and the Gaussian jammer |
| `receiver.py` | Receiver chain: LS estimation, LMMSE equalization, demapping, decoding |
| `link.py` | The whole slot: UE, channels, jammer, noise, receiver |
| `features.py` | Power, DMRS residual and EVM maps, allocation masks, and the empty-comb power |
| `baselines.py` | Threshold detectors (EVM, residual, residual exact p-value, per-cell power, empty comb) and the block test (no learned parameters) |
| `theory.py` | Ideal DMRS residual law for one slot or a K-slot average, with white Gaussian noise of known power and the channel in the retained delay subspace |
| `noise.py` | The noise power the receiver assumes: exact, off by a set error, or estimated per slot from the empty DMRS comb |
| `recovery.py` | Map-aware receiver: hard (drop/erase) and soft (weight by estimated jamming) |
| `model.py`, `train.py` | U-Net detector, per-cell MLP (ablation), training (optionally with noise-estimate jitter) |
| `dataset.py` | Train / validation / test / scheduling-aware splits with fixed seeds |
| `evaluate.py`, `metrics.py` | Pd vs JSR at thresholds targeting 1% of scored clean validation cells, and cell-level F1 |
| `paper.py` | Copies the paper's figures into the local clone of its Overleaf project |
| `diagrams.py` | Exports the draw.io diagrams in `diagrams/` (e.g. the system diagram) to PDF and PNG |
| `experiments/` | One script per experiment and figure (see below) |

### Reproducing the results

Experiment artifacts go in `data/` or `results/` by default. Override these paths with
`JAMLOC_DATA_DIR` or `JAMLOC_RESULTS_DIR`. Runtime depends on hardware and run settings.

**Detection experiment (datasets, training, evaluation, recovery, figures).** This command
trains four variants per seed and builds `results/full_run.zip` with evaluation CSVs and plots,
models and loss histories, recovery curves for both channels using the first detector seed, and
figures with the gallery page. The last cell of
[`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) runs it on Colab.

```bash
uv run python -m jamloc.experiments.full_run --seeds 0 1 2
```

These commands run selected stages. Use `full_run` for the reference settings, including its
training and validation simulation batch size.

```bash
for s in train val test sched; do uv run python -m jamloc.dataset --split $s; done
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
mixed training. Comparisons reuse nominal models or splits where applicable. The width sweep
and mixed-channel study train missing checkpoints on their respective datasets.

```bash
uv run python -m jamloc.experiments.residual_theory    # analytic threshold and Pd vs simulation
uv run python -m jamloc.experiments.generalization     # long delays, 30 m/s, QPSK jammer (GPU)
uv run python -m jamloc.experiments.size_sweep         # narrower U-Nets: train + evaluate (GPU)
uv run python -m jamloc.experiments.size_sweep --latency   # their CPU latency (run on the CPU)
uv run python -m jamloc.experiments.robust_training    # retrain on mixed channels (GPU)
```

**Third round** (`docs/plan-v3.md`): fair baselines, detection before damage, calibration,
multi-slot residual. These studies use the required datasets and checkpoints. The detection
margin also needs the corresponding unprotected BLER sweep.

```bash
uv run python -m jamloc.train --variant fused --arch pixel --epochs 15 --seeds 0 1 2   # per-cell MLP
uv run python -m jamloc.train --variant fused --epochs 15 --seeds 0 1 2 --noise-jitter-db 1
uv run python -m jamloc.evaluate --split test        # now also: block test, per-cell MLP, jittered U-Net
uv run python -m jamloc.experiments.detection_margin # slot-level Pd vs unprotected BLER (needs ber_bler)
uv run python -m jamloc.experiments.calibration      # false alarms vs SNR and vs noise-estimate error
uv run python -m jamloc.experiments.multislot        # residual averaged over K slots: theory vs checks
```

**Fourth round** (`docs/plan-v4.md`): evaluation on pilot-comb and hopping grids, and retraining
on a mix of attacks. Start with the nominal splits and checkpoints.

```bash
uv run python -m jamloc.experiments.generalization            # now also the comb and hopping grids
uv run python -m jamloc.experiments.attack_training --seeds 0 1 2
```

**Fifth round** (`docs/review-v6.md`, `docs/plan-v6.md`): the datasets gain a seventh feature
(empty-comb power), and most grid splits use 16-slot batches. On a Colab GPU, `paper_run`
generates missing data and checkpoints and runs the detection, theory, margin, figure and archive
stages. Run recovery, the DMRS comparison and target-CPU timing separately.

```bash
uv run python -m jamloc.experiments.paper_run --seeds 0 1 2
```

The recovery experiment uses CPU LMMSE solves. After `paper_run` has produced the models and
`val`, you can run seeds in separate processes.

```bash
uv run python -m jamloc.experiments.recovery --channel C --seeds 0 1 2              # nominal jammer
uv run python -m jamloc.experiments.recovery --channel C --jammer pulsed --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --channel C --jammer hopping --seeds 0 1 2
uv run python -m jamloc.experiments.recovery --channel C --snr-db 8 --seeds 0 1 2   # paper's low-SNR case
```

The jammer covers 4 of the 12 scheduled PRBs above. `--jam-prb N` moves it to N PRBs (runs go to
`recovery_C_jam<N>` files); `experiments.width_sweep` then plots BLER against the width, and the
detector's Pd against the width of the test split's jammer blocks (`results/width_sweep/`,
`docs/results.md` section 22). The runs read the checkpoints through a `models` link in that folder
(`results/width_sweep/models` pointing to `results/models`).

```bash
JAMLOC_RESULTS_DIR=results/width_sweep uv run python -m jamloc.experiments.recovery --channel C \
    --jsr-db -3 3 9 --seeds 0 --max-mc-iter 25 --jam-prb 8 --attacks pilot data partial \
    --receivers standard lmmse p_r_e_soft p_r_e_soft_da true_soft true_soft_da
uv run python -m jamloc.experiments.width_sweep            # detection part, then the BLER figure
```

To compare one and two DMRS symbols, reuse the nominal test data and checkpoints from
`paper_run`. The experiment generates missing two-DMRS splits and trains the corresponding
P+R+E and P+E models. It writes detection and BLER tables under `results/dmrs_count/`.

```bash
uv run python -m jamloc.experiments.dmrs_count --seeds 0 1 2 --batch-size 64 --batches 25
uv run python -m jamloc.experiments.dmrs_count --small
```

New in this round: `experiments.intervals` (bootstrap intervals over batches, paired differences,
PRB-level F1), `experiments.neighbor` (a synchronized neighbor-cell UE as legitimate
interference), `experiments.calibration --estimator` (per-slot noise estimate), `ber_bler
--benign` (BLER against SNR without a jammer) and `detection_margin --snr-db` (margin closer to
the MCS's working point).

**Other figures** (run locally; latency should be measured on the target CPU):

```bash
uv run python -m jamloc.experiments.ber_bler --channel A        # BER/BLER vs JSR
uv run python -m jamloc.experiments.ber_bler --channel C
uv run python -m jamloc.experiments.feature_maps                # example feature maps
uv run python -m jamloc.experiments.latency                     # CPU latency per slot, 1 thread
```

## The paper (Overleaf)

The paper lives in its own Overleaf project. Use its Git remote to exchange changes with the
browser editor. This repository generates the figures, and `python -m jamloc.paper` copies the
ones the paper includes into a local clone of the project.

**Once per computer.** Get a Git authentication token from Overleaf's Account Settings and
clone the project next to this repository. Use `git` as the username and the token as the
password. Configure a credential helper if you want it to store the token. Never put the token
in the URL or a file.

```bash
cd ..        # the folder that contains nr-jamming-localization
git clone https://git.overleaf.com/<project-id> nr-jamming-paper
```

**Every time.** Pull before editing and push commits to update Overleaf.

```bash
git -C ../nr-jamming-paper pull                               # latest text from Overleaf
uv run python -m jamloc.diagrams                              # draw.io diagrams -> results/figures
uv run python -m jamloc.paper                                 # copy the current figures
git -C ../nr-jamming-paper add -A
git -C ../nr-jamming-paper commit -m "Update figures"
git -C ../nr-jamming-paper push
```

Block diagrams are drawn in draw.io (`diagrams/*.drawio`; edit them in the desktop app or at
app.diagrams.net). `jamloc.diagrams` needs the draw.io desktop app for the export
(macOS: `brew install --cask drawio`). `jamloc.paper` looks for the clone in `../nr-jamming-paper`; give `--dest` or set
`JAMLOC_PAPER_DIR` if it is elsewhere. `paper/main.tex` in this repository is a frozen copy of the
draft from before the move; the Overleaf copy is the paper. `docs/writing-guide.md` collects the
style rules (mathematics, structure, tone), exemplar papers and the references added in the
November 2026 revision.

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
The scripts rebuild them from recorded seed ranges on compatible systems. CPU and GPU runs can
produce different samples and checkpoints.
