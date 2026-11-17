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
