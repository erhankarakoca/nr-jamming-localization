# Development

## Setup

You need git, [uv](https://docs.astral.sh/uv/) and GitHub SSH access for the clone below (clone
over HTTPS if you only need to read). The project uses Python 3.12.

```bash
# uv on macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv on Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone git@github.com:erhankarakoca/nr-jamming-localization.git
cd nr-jamming-localization
uv sync --locked    # creates .venv/ from the committed uv.lock
uv run pytest       # includes a Sionna/PyTorch smoke test
```

**VS Code.** Open the folder and install the recommended extensions when asked (Python, Jupyter,
Ruff, Google Colab). The workspace enables pytest discovery and formats Python files with Ruff
on save. Select `.venv` if VS Code does not find it.

## Day to day

```bash
git pull                  # work from the other computer
uv sync                   # dependency changes
uv run python <script>    # or activate .venv
uv run jupyter lab        # notebooks
```

Add a dependency with `uv add <package>` (`uv add --dev <package>` for tools) and commit both
`pyproject.toml` and `uv.lock`. Write code locally and push; use Colab for the heavy jobs
(dataset generation, training).

## Colab GPU

Run [`notebooks/colab_setup.ipynb`](../notebooks/colab_setup.ipynb) at the start of each Colab
session. It clones or updates the repository on the Colab machine and installs it into Colab's
Python, reusing a compatible preinstalled PyTorch. Sionna uses CUDA when a GPU runtime is
available.

- **From VS Code (Google Colab extension):** open the notebook, choose *Select Kernel → Colab →
  New Colab Server* and a GPU, and run the cells. The notebook asks for `GITHUB_TOKEN` if it is
  missing and keeps files on the Colab machine, so download results before the session ends.
  The extension supports `drive.mount()` from v0.2.1; Colab Secrets need the browser. See the
  [extension notes](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds).
- **From the browser:**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erhankarakoca/nr-jamming-localization/blob/main/notebooks/colab_setup.ipynb)
  Set `BROWSER_COLAB = True` in the first cell to mount Google Drive for data and results. The
  notebook reads `GITHUB_TOKEN` from the Colab secret of that name if it is not in the
  environment.

The notebook also explains how to create a fine-grained GitHub token limited to this repository.

## The paper (Overleaf)

The paper lives in its own Overleaf project, reached through Overleaf's Git remote. This
repository draws the figures, and `python -m jamloc.paper` copies the ones the paper includes into
a local clone of that project.

**Once per computer:** create a Git authentication token in Overleaf's Account Settings and clone
the project next to this repository, with `git` as the username and the token as the password.
Use a credential helper to store it; never put the token in a URL or a file.

```bash
cd ..        # the folder that contains nr-jamming-localization
git clone https://git.overleaf.com/<project-id> nr-jamming-paper
```

**Every time:**

```bash
git -C ../nr-jamming-paper pull                               # latest text from Overleaf
uv run python -m jamloc.diagrams                              # draw.io diagrams -> results/figures
uv run python -m jamloc.paper                                 # copy the current figures
git -C ../nr-jamming-paper add -A
git -C ../nr-jamming-paper commit -m "Update figures"
git -C ../nr-jamming-paper push
```

`jamloc.paper` looks for the clone in `../nr-jamming-paper`; pass `--dest` or set
`JAMLOC_PAPER_DIR` if it is elsewhere. Block diagrams are draw.io files in `diagrams/` (edit them
in the desktop app or at app.diagrams.net); exporting them needs the desktop app
(macOS: `brew install --cask drawio`). `paper/main.tex` here is a frozen copy of the draft from
before the move to Overleaf. [`writing-guide.md`](writing-guide.md) collects the style rules,
exemplar papers and the references added in the November 2026 revision.

## Code map

One uplink slot is simulated as a chain of Sionna blocks;
[`notebooks/01_link_walkthrough.ipynb`](../notebooks/01_link_walkthrough.ipynb) steps through it.

```text
UE: PUSCH transmitter (n PRBs) -> place on 24-PRB carrier -> TDL channel --+
                                                                           +-> AWGN -> gNB
Jammer: Gaussian noise on attacked PRB x symbol cells -> TDL channel ------+

gNB: received grid -> scheduled PRBs -> LS estimator -> LMMSE equalizer -> demapper -> LDPC decoder

Full received grid       -> power P
Scheduled received grid  -> raw pilot LS -> delay-subspace residual R
Equalized scheduled data -> decision-directed EVM E

P, R, E + allocation, DMRS and data masks
  -> U-Net (66,553 parameters) -> threshold -> jamming map (24 PRBs x 14 symbols)
```

| Module (`src/jamloc/`) | What it does |
| --- | --- |
| `config.py` | Simulation parameters (proposal Table 1) |
| `ue.py` | PUSCH transmitter per allocation size; placing the allocation on the carrier |
| `jammer.py` | Attack patterns (pilot, data, partial-band) and the Gaussian jammer |
| `receiver.py` | LS estimation, LMMSE equalization, demapping, decoding |
| `link.py` | The whole slot: UE, channels, jammer, noise, receiver |
| `features.py` | Power, DMRS residual and EVM maps, allocation masks, empty-comb power |
| `baselines.py` | Threshold detectors (EVM, residual, exact p-value, per-cell power, empty comb) and the block test |
| `theory.py` | Ideal null law of the DMRS residual, for one slot or a K-slot average |
| `noise.py` | The noise power the receiver assumes: exact, off by a set error, or estimated from the empty DMRS comb |
| `recovery.py` | Map-aware receiver: hard (drop or erase) and soft (weight by the estimated jamming), data-aided passes |
| `model.py`, `train.py` | U-Net detector, per-cell MLP ablation, training |
| `dataset.py` | Train, validation, test and scheduling-aware splits with fixed seeds |
| `evaluate.py`, `metrics.py` | Pd against JSR at thresholds set for 1% false alarms on clean validation cells, cell-level F1 |
| `paper.py`, `diagrams.py` | Copy figures into the Overleaf clone; export the draw.io diagrams |
| `site/` | Export, build and deploy of the project page (`python -m jamloc.site`) |
| `experiments/` | One script per experiment and figure ([reproducing.md](reproducing.md)) |
