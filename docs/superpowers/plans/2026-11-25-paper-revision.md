# Paper Revision v9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A six-page paper (`conference-v9.tex`) whose only U-Net is the general one, with results rerun on this GPU, smooth BLER curves, simple tables, short captions and two new explanatory figures.

**Architecture:** The recovery experiment gets a switch that makes its `p_r_e_*` receivers use the general U-Net, so the existing table and sweep tools (`experiments.resolution`) produce the paper's numbers unchanged. Two new figure functions in `experiments.figures` simulate one slot each. The paper text is rewritten by hand and reviewed by Codex.

**Tech Stack:** Python 3.12, PyTorch, Sionna 2.2 (uv), pytest, ruff, IEEEtran LaTeX, Overleaf over Git.

**Spec:** `docs/superpowers/specs/2026-11-25-paper-revision-design.md`

## Global Constraints

- The general U-Net is `load_model("fused", seed, data="gen")` with thresholds from `val_gen`.
- Every point of every table and curve runs until 200 block errors or 30 000 slots pooled over the three seeds, which is `--target-block-errors 70 --max-mc-iter 160` per seed (batches of 64).
- Six pages in IEEEtran conference format, no overfull lines, no undefined references.
- Prose has no colons, semicolons or "§". Commits and PRs carry no attribution lines.
- Code follows the Sionna tutorial style with short docstrings (memory `sionna-example-code-style`).

## Review Focus

- Running recovery with the general U-Net when its checkpoint does not exist should stop with a message that names the command which trains it, not with a bare `FileNotFoundError`. Test in Task 1.
- The general U-Net must take its threshold from `val_gen`, never from `val`. Test in Task 1.
- `--noise-estimate comb` keeps using the models trained on the estimated noise power, whatever `--unet` says. Test in Task 1.
- In the shrinkage figure, a row without any jammed cell must show every shrunk value close to the row mean. Test in Task 2.
- The EVM map is calibrated on QPSK only, so the 16-QAM runs must not include `evm_*` receivers. Checked in Task 4 by the command list.

---

### Task 1: Choose the U-Net of the recovery experiment

**Files:**
- Modify: `src/jamloc/experiments/recovery.py` (constants near `TRAINED_ON`, `simulate`, `val_threshold`, `main`)
- Test: `tests/test_experiments.py`

**Interfaces:**
- Produces: `UNETS = {"general": ({"data": "gen"}, "val_gen"), "nominal": ({}, "val")}`, `unet_source(unet: str) -> tuple[dict, str]`, `simulate(..., unet: str = "general")`, CLI `--unet {general,nominal}` with default `general`.

- [ ] **Step 1: Write the failing tests**

```python
def test_unet_source_picks_training_and_validation_split():
    from jamloc.experiments import recovery

    assert recovery.unet_source("general") == ({"data": "gen"}, "val_gen")
    assert recovery.unet_source("nominal") == ({}, "val")


def test_recovery_explains_a_missing_general_unet(tmp_path, monkeypatch):
    # no general checkpoint: the error names the command that trains it
    ...  # set JAMLOC_* to tmp, generate tiny train/val, train fused and no_residual for 1 epoch
    with pytest.raises(FileNotFoundError, match="general_training"):
        recovery.main(["--channel", "A", "--batch-size", "8", "--max-mc-iter", "1", "--jsr-db", "3"])
```

In `test_recovery_script_runs`, pass `--unet nominal`, since it trains only nominal models. Add one assertion that `--noise-estimate comb` with `--unet general` still loads the `nest` checkpoint (monkeypatch `load_model` and record its `data` argument).

- [ ] **Step 2:** Run `uv run pytest tests/test_experiments.py -k "unet or recovery_script" -v`. Expected: FAIL (`unet_source` missing).
- [ ] **Step 3:** Implement `unet_source` and thread `unet` through `simulate` and `main`. For `fused` (not under `nest`), load with the options of `unet_source(unet)` and take the threshold from its validation split. A missing checkpoint raises `FileNotFoundError` whose message says `python -m jamloc.experiments.general_training`. The `gen_*` receivers stay for reproducing section 26.
- [ ] **Step 4:** Run the same tests. Expected: PASS. Then `uv run ruff check src tests`.
- [ ] **Step 5:** Commit "Let the recovery experiment use the general U-Net by default".

### Task 2: Worked example of shrinkage

**Files:**
- Modify: `src/jamloc/experiments/figures.py` (`jamming_estimates`, new `shrinkage_example`, `FIGURES`)
- Test: `tests/test_figures.py`

**Interfaces:**
- Produces: `_estimates_without_map(cfg, attack: str, jsr_db: float, seed: int) -> dict` with keys `"truth"` (bool `[n, 14]`), `"prb"`, `"cell"`, `"shrink"` (float `[n, 14]` in units of the jammer power per RE, nan on DMRS cells). `shrinkage_example(jsr_db: float = 9.0, seed: int = 7) -> tuple[Figure, str]`, registered as `"shrinkage_example"`.

- [ ] **Step 1: Write the failing test**

```python
def test_shrinkage_lies_between_the_cell_and_its_row_mean():
    from jamloc.experiments.figures import CFG, _estimates_without_map, shrinkage_example

    est = _estimates_without_map(CFG, "data", 9.0, seed=7)
    cell, row, shrink = est["cell"], est["prb"], est["shrink"]
    ok = ~np.isnan(cell)
    lo, hi = np.minimum(cell, row), np.maximum(cell, row)
    assert np.all(shrink[ok] >= np.maximum(lo[ok], 0) - 1e-6)
    assert np.all(shrink[ok] <= np.maximum(hi[ok], 0) + 1e-6)
    clean_rows = ~est["truth"].any(axis=1)  # rows the block jammer does not touch
    assert np.allclose(shrink[clean_rows][ok[clean_rows]], row[clean_rows][ok[clean_rows]], atol=0.05)
    fig, caption = shrinkage_example()
    assert len(fig.axes) >= 2 and "shrink" in caption.lower()
```

- [ ] **Step 2:** Run `uv run pytest tests/test_figures.py -k shrinkage -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Move the estimate computation out of `jamming_estimates` into `_estimates_without_map` (no change to that figure's output). `shrinkage_example` has two panels on one column width, (a) one PRB row of a block data jammer and (b) one row of a hopping data jammer hit on 2 to 5 of its 13 data symbols. Each panel shows the jammed symbols shaded, each cell's own estimate, the row mean as a line and the shrunk values. The caption has two sentences.
- [ ] **Step 4:** Run the test and `uv run python -m jamloc.experiments.figures --only shrinkage_example jamming_estimates`, then open both PNGs. Expected: PASS, `jamming_estimates` looks as before.
- [ ] **Step 5:** Commit "Add a worked example of shrinkage".

### Task 3: Three maps of one slot

**Files:**
- Modify: `src/jamloc/experiments/figures.py` (new `attack_maps`, `FIGURES`)
- Test: `tests/test_figures.py` (`test_data_figures_and_gallery` trains tiny models already)

**Interfaces:**
- Consumes: `recovery.detector_cells`, `recovery.val_threshold`, `recovery.unet_source("general")` from Task 1.
- Produces: `attack_maps(jsr_db: float = 0.0, seed: int = 0) -> tuple[Figure, str]`, registered as `"attack_maps"`. It needs `fused_gen_seed0.pt`, `val_gen`, `val`, and is skipped with the usual message when they are missing.

- [ ] **Step 1:** Extend `test_data_figures_and_gallery` so it also generates tiny `train_gen` / `val_gen` splits, trains `fused` with `data="gen"` for 1 epoch, and asserts that `attack_maps.png` is written. Run it. Expected: FAIL.
- [ ] **Step 2:** Implement `attack_maps`. A pilot attack on PRBs 10 to 13 of the 12-PRB allocation at 0 dB JSR and 20 dB SNR, TDL-C. Three panels in the style of `attack_patterns` show the true jammed cells, the general U-Net's map and the EVM threshold map. The caption has two sentences.
- [ ] **Step 3:** Run the test and render the figure. Expected: PASS, and the EVM panel marks data symbols of PRBs 10 to 13 while the U-Net marks their DMRS cells.
- [ ] **Step 4:** Commit "Add the three maps of one slot".

### Task 4: Rerun the paper's results with the general U-Net

**Files:**
- Modify: `docs/reproducing.md` (new paragraph "Paper v9")
- Output (not in Git): `results/v9/{nom,whole,hop,pulse,qam16,qamhop,sweep_hop,sweep_nom,passes}/recovery`

- [ ] **Step 1:** Write the commands into `docs/reproducing.md`. Every run uses `--channel C --seeds $s --target-block-errors 70 --max-mc-iter 160` and a `models` link as in section 26.
  - QPSK at +9 dB (`nom` with `--attacks none pilot data partial`, `pulse` with `--jammer pulsed`, `hop` with `--jammer hopping`) and every pilot at +3 and +9 dB (`whole`), with `standard lmmse nomap_prb nomap_cell nomap_shrink evm_prb evm_cell evm_shrink p_r_e_soft p_r_e_cell p_r_e_shrink true_prb_soft true_prb_shrink true_soft true_cell true_shrink`.
  - 16-QAM (`--snr-db 18 --mcs 13`) at +3 and +9 dB on blocks (`qam16`) and under hopping (`qamhop`), with `standard nomap_prb nomap_cell nomap_shrink p_r_e_soft p_r_e_shrink true_soft true_cell true_shrink` (no `evm_*`).
  - Curves from -9 to +9 dB in 3 dB steps, hopping (`sweep_hop`) and pilot and partial-band blocks (`sweep_nom`), with `standard nomap_prb evm_shrink p_r_e_shrink nomap_shrink true_shrink`.
  - Data-aided passes (`passes`), `p_r_e_shrink p_r_e_shrink_full` on hopping partial-band at +9 dB and every pilot at +9 dB.
- [ ] **Step 2:** Run them in the background on the GPU. Expected: every run ends with "Wrote ...", no traceback.
- [ ] **Step 3:** Merge and pool: `resolution merge results/v9/merged/recovery results/v9/{nom,whole,hop,pulse,qam16,qamhop}/recovery`, `recovery --plot-only` for each jammer and for 16-QAM, `resolution table` (with and without `--qam`), `resolution sweep results/v9/sweep_hop results/v9/sweep_nom`.
- [ ] **Step 4:** Check the Monte Carlo rule. Every receiver at every point has at least 200 block errors or at least 30 000 slots. List any exception.
- [ ] **Step 5:** Commit the reproducing commands.

### Task 5: Write conference-v9.tex

**Files:**
- Create: `/home/user/nr-jamming-paper/conference-v9.tex` (from v8)
- Copy into `/home/user/nr-jamming-paper/figures/`: `shrinkage_example.pdf`, `attack_maps.pdf`, the new `resolution_sweep.pdf`

- [ ] **Step 1:** Apply the Codex writing review (`codex-writing-result.md` in the scratchpad) and `docs/writing-guide.md`. One U-Net, called "the U-Net". No mention of the nominal training.
- [ ] **Step 2:** Tables. The QPSK table keeps no map, EVM, U-Net and true map, each with per PRB and shrinkage, plus per cell only where it differs. The 16-QAM table keeps no map, U-Net and true map. Each caption is at most two sentences and says that BLER is the share of transport blocks that fail the CRC. Column names are plain words.
- [ ] **Step 3:** Results. One paragraph per table or figure, each giving the comparison, the main numbers and one interpretation. Remove the old Table III and its paragraph.
- [ ] **Step 4:** Add the two figures where the method is explained, shrinkage next to the shrinkage equation and the three maps next to the map paragraph.
- [ ] **Step 5:** Build. Expected: 6 pages, 0 errors, 0 undefined references, 0 overfull boxes.
- [ ] **Step 6:** Ask Codex to check every number in v9 against the pooled CSVs and the writing rules, fix what it finds, rebuild.
- [ ] **Step 7:** Commit and push to Overleaf.

### Task 6: Documentation of the v9 round

**Files:**
- Modify: `docs/results.md` (new section 27), `README.md` (results rows use the general U-Net)

- [ ] **Step 1:** Section 27 gives the v9 tables, the curves and the Monte Carlo rule, in plain sentences.
- [ ] **Step 2:** Run the full test suite on the GPU. Expected: all pass, 1 skipped.
- [ ] **Step 3:** Commit and push `mixed-training`.
