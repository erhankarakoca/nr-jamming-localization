# Implementation Plan: Map-Aware Recovery and Explanatory Figures

**Goal:** A receiver that uses the jamming map to repair channel estimates and erase corrupted
soft bits, its BLER comparison, and a set of explanatory figures with captions and a gallery page.

**Architecture:** `recovery.py` adds an LMMSE pilot-interpolation function and a `MapAwareReceiver`
built from the same Sionna blocks as `receiver.py`. `experiments/recovery.py` runs the five-receiver
comparison. `experiments/figures.py` holds one function per explanatory figure (each returns the
figure and its caption) and writes a gallery page. `full_run` runs both, so one Colab run produces
every number and figure.

**Tech stack:** Sionna 2.2 (PyTorch), PyTorch, NumPy, Matplotlib (via `jamloc.plotting`), pytest.

**Spec:** `docs/design/recovery-and-figures.md` (approved).

## Global Constraints

- Sionna tutorial style: one block per processing step; comments say *why*, not only what.
- Windows/Linux/macOS (CPU) and Colab (CUDA): every tensor on the device of its inputs or on
  `sionna.phy.config.device`; paths via `pathlib`; text files with `encoding="utf-8"`.
- Outputs only through `results_dir()`: `results/recovery/`, `results/figures/`.
- Figures through `jamloc.plotting` (`paper_style`, `new_figure`, `save`, Okabe-Ito colours with a
  marker and line style per curve); PDF and PNG.
- Every figure has a caption of 2–4 sentences: what it shows, how to read it, why it matters.
- Commit messages plain and human; no co-author lines.

## Review Focus

1. **Every DMRS cell flagged** (no clean pilot): no NaN; falls back to the standard estimate;
   decodes like the unprotected receiver. Test in Task 2.
2. **Jammed block at the allocation edge:** interpolation stays finite, error variance is high in
   the gap and low at clean pilots. Test in Task 2.
3. **False alarms in a detector map** (clean cells flagged): erasing a few clean cells must not
   break decoding at high SNR. Test in Task 2.
4. **Missing inputs for figures** (no models, no histories, no recovery CSV): those figures are
   skipped with a note and the gallery is still written. Test in Task 5.
5. **GPU device placement** in the interpolation (correlation matrix, masks): built on the input's
   device. Checked in review; the Colab run is the real test.

---

### Task 1: Training history

**Files:** modify `src/jamloc/train.py`; test `tests/test_model.py`.

**Produces:** `results_dir()/models/{variant}_seed{seed}_history.csv`, columns
`epoch, train_loss, val_loss` (train loss = mean over the epoch's batches).

- [ ] **Step 1:** extend `test_train_writes_checkpoint`: train `epochs=2`; the history CSV exists
  with 2 rows and those columns; `val_loss` values are positive floats.
- [ ] **Step 2:** run → fails (no history file).
- [ ] **Step 3:** implement; also print the epoch's mean train loss instead of the last batch's.
- [ ] **Step 4:** run → pass. **Step 5:** commit "Save training history per run".

### Task 2: Map-aware receiver

**Files:** create `src/jamloc/recovery.py`; test `tests/test_recovery.py`.

**Consumes:** `features.dmrs_ls(y_alloc, tx)`, `receiver.data_positions(tx)`, the blocks used in
`receiver.PUSCHReceiverChain`.

**Produces:**
- `lmmse_interpolation(h_ls, pilot_sc, clean, s2, num_sc, scs, tau_max) -> (h_hat, err_var)`:
  `h_ls` `[B, N_p]` complex, `pilot_sc` `[N_p]` long, `clean` `[B, N_p]` bool, `s2` `[B]`;
  returns `[B, num_sc]` complex and `[B, num_sc]` float. Per slot, LMMSE from the clean pilots
  only, with `R[k, k'] = exp(-jπΔf τ) sinc(Δf τ)`, `Δf = (k − k')·scs`:
  `h_hat = R[:, c] (R[c, c] + s2 I)⁻¹ h_ls[c]`, `err_var_k = 1 − R[k, c] (R[c, c] + s2 I)⁻¹ R[c, k]`.
  Batched without loops: unclean pilots are decoupled by zeroing their rows/columns of `R[p, p]`,
  putting 1 on their diagonal and 0 in `h_ls`.
- `MapAwareReceiver(tx)` (Sionna `Block`): `call(y, no, cells, decode=True) -> dict` with the
  same keys as `PUSCHReceiverChain` (`h_hat, err_var, x_hat, no_eff, llr, b_hat, tb_crc`);
  `cells` is the map over the allocation, bool `[B, n, 14]`. Steps: LS pilots → clean = DMRS cell of
  the pilot's PRB not flagged → `lmmse_interpolation` → broadcast over 14 symbols as Sionna's
  `h_hat`/`err_var` shape → slots with no clean pilot take Sionna's LS estimate instead → LMMSE
  equalizer → demapper → LLRs of data REs in flagged data cells set to 0 → layer demapper → decoder.

- [ ] **Step 1: failing tests** (`tests/test_recovery.py`, 12-PRB allocation at PRBs 6–17):
  - `test_interpolation_all_clean_recovers_channel`: TDL-A, SNR 40 dB, benign; all pilots clean →
    relative error of `h_hat` against the true channel (from `OFDMChannel(return_channel=True)`)
    < 0.05 on every subcarrier; `err_var` < 0.05.
  - `test_interpolation_gap_has_high_error_variance`: same slot, pilots of PRBs 4–7 of the
    allocation marked unclean → `err_var` mean in the gap > 5× mean on clean PRBs; all finite.
  - `test_interpolation_gap_at_edge_is_finite`: unclean PRBs 0–3 → `h_hat`, `err_var` finite,
    `err_var` ≤ 1 + 1e-4.
  - `test_empty_map_decodes_benign`: SNR 25 dB, benign, empty map → all CRC pass.
  - `test_few_false_alarms_still_decode`: SNR 25 dB, benign, map with 3 random data cells and 1
    DMRS cell flagged per slot → all CRC pass.
  - `test_erasure_zeroes_flagged_llrs`: data attack, map = labels → LLRs of REs in flagged cells are
    exactly 0 and the others are not all 0.
  - `test_true_map_beats_unprotected`: TDL-A, SNR 15 dB, 4-PRB block inside the allocation, B=64,
    JSR +6 dB: for pilot and for data attacks, BLER(map-aware, true map) < BLER(standard).
  - `test_no_clean_pilot_falls_back`: pilot attack on the whole allocation, JSR −10 dB, map = true
    map → `b_hat` identical to `PUSCHReceiverChain`'s; no NaN anywhere.
- [ ] **Step 2–4:** fail → implement → pass. **Step 5:** commit "Add map-aware receiver".

### Task 3: Recovery experiment

**Files:** create `src/jamloc/experiments/recovery.py`; test `tests/test_experiments.py`.

**Consumes:** Task 2; `train.load_model`, `model.select_inputs`, `features.extract`,
`metrics.threshold_at_fa`, `dataset.load("val")`.

**Produces:**
- `RECEIVERS = ("standard", "lmmse_no_map", "map_p_e", "map_p_r_e", "true_map")`.
- `val_threshold(variant, seed=0) -> float`: 1%-FA threshold of the variant's logits on `val`.
- `detector_cells(batch, cfg, model, variant, thr) -> BoolTensor [B, 24, 14]`.
- `main(argv)`: `--channel A|C`, `--snr-db 15`, `--jsr-db` (default −15…+9 step 3),
  `--batch-size 64`, `--max-mc-iter 20`, `--target-block-errors 200`, `--seed 0`. UE on PRBs 6–17;
  4-PRB jammer block at a random start in 6…14; attacks pilot/data/partial. Every Monte-Carlo
  batch is simulated once and decoded by all five receivers (shared draws, fair comparison); a
  point stops when every receiver has the target block errors or after `max-mc-iter` batches.
  Limit case: attack `pilot_whole_allocation` (scheduling-aware pilot jamming) for `standard` and
  `true_map` at the same JSRs. Writes `results/recovery/recovery_{channel}.csv`
  (`receiver, attack, jsr_db, snr_db, ber, bler, blocks`) and `recovery_{channel}.pdf/png` (one
  BLER-vs-JSR panel per attack, log y, zero BLER drawn as gaps).
- [ ] **Step 1:** `test_recovery_script_runs`: in `tmp_path`, generate tiny `val`, train `fused`
  and `no_residual` 1 epoch, run `main(["--channel", "A", "--batch-size", "8", "--max-mc-iter",
  "1", "--jsr-db", "-3", "3"])` → CSV has every receiver × {pilot, data, partial} × 2 JSRs, plus
  limit rows; PNG and PDF exist.
- [ ] **Step 2–4:** fail → implement → pass. **Step 5:** commit "Add recovery experiment".

### Task 4: Explanatory figures without trained models

**Files:** create `src/jamloc/experiments/figures.py`; test `tests/test_figures.py`.

**Produces:** functions returning `(Figure, caption: str)`:
- `attack_patterns()`: 1 × 4 schematic grids (benign, pilot, data, partial) — allocation shaded,
  DMRS row marked, jammed cells hatched; no simulation.
- `system_diagram()`: block diagram (boxes and arrows) of UE → channels → jammer → receiver steps
  → P/R/E → U-Net → map, and the map-aware receiver path.
- `residual_explained()`: TDL-C, SNR 20 dB, 12-PRB allocation, without and with pilot jamming on 4
  PRBs (JSR 0 dB): top row |LS estimate| per pilot with the subspace fit; bottom row R per PRB (dB)
  with the 0 dB line.
- `constellations()`: equalized symbols for benign / pilot / data jamming (JSR 0 dB, SNR 20 dB,
  jammer on 4 of 12 PRBs); jammed-PRB symbols and clean-PRB symbols in different colour+marker.
- `FIGURES`: name → function registry; `main(argv)` with `--only NAME...` writes
  `results/figures/{name}.pdf/png` and `results/figures/captions.json`.
- [ ] **Step 1:** `test_model_free_figures`: each of the four returns a Figure with ≥ 1 axes and a
  caption of ≥ 2 sentences; `main(["--only", "attack_patterns"])` in `tmp_path` writes PDF and PNG.
- [ ] **Step 2–4:** fail → implement → pass; look at each PNG. **Step 5:** commit
  "Add explanatory figures: attack patterns, system diagram, residual, constellations".

### Task 5: Figures from data and models, and the gallery

**Files:** modify `src/jamloc/experiments/figures.py`; test `tests/test_figures.py`.

**Produces:**
- `feature_distributions(split="test")`: rows P, R, E (dB); columns pilot, data, partial; jammed
  vs non-jammed scheduled cells at JSR −6…0 dB, step histograms (density) with distinct styles.
- `detector_outputs(split="test", seed=0)`: 3 example slots (pilot, data, partial, JSR −6 dB) ×
  columns true map, P+R+E, P+E, power only, EVM threshold, residual threshold; cells coloured hit /
  miss / false alarm with a legend; thresholds at 1% FA from `val`.
- `training_curves()`: train (dashed) and val (solid) loss per epoch, one panel per variant, one
  line per seed, from the history CSVs.
- A figure whose inputs are missing (no split file, no model, no history) is skipped with a printed
  note and a gallery entry saying how to produce it.
- `write_gallery()`: `results/figures/index.html` — every new figure plus the existing ones found in
  `results/` (BER/BLER, feature maps, Pd vs JSR, recovery), each as `<img>` with a relative path
  and its caption; plain HTML and CSS, no network, opens offline.
- [ ] **Step 1:** `test_data_figures_and_gallery`: tiny `val`/`test` + 1-epoch models + histories
  in `tmp_path` → the three functions return figures with captions; `main([])` writes
  `index.html` containing `attack_patterns.png` and `training_curves.png` and each caption.
  `test_gallery_without_models`: empty results dir → `main([])` still writes `index.html` and notes
  the skipped figures.
- [ ] **Step 2–4:** fail → implement → pass; look at each PNG. **Step 5:** commit
  "Add data-driven figures and a figure gallery".

### Task 6: Full run and docs

**Files:** modify `src/jamloc/experiments/full_run.py`, `README.md`, `notebooks/colab_setup.ipynb`
(markdown only); test `tests/test_experiments.py`.

- `full_run` after evaluation: recovery for A and C (`--small`: batch 8, 1 batch, JSR 0 only),
  then `figures.main([])`; the zip also contains `recovery/*` and `figures/*`.
- [ ] **Step 1:** extend `test_full_run_small`: zip contains `figures/index.html`,
  `recovery/recovery_A.csv`, `models/fused_seed0_history.csv`.
- [ ] **Step 2–4:** fail → implement → pass; README "Reproducing the results" lists the recovery
  and figures commands; notebook markdown says what the zip now contains.
- [ ] **Step 5:** commit "Run recovery and figures as part of the full run". Full suite green,
  merge to `main`, push (user then runs the Colab reference run; `docs/results.md` is updated from
  its zip afterwards).
