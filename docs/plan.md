# Implementation Plan: Resource-Level Jamming Localization in the 5G NR Uplink

**Goal:** For every simulated PUSCH slot, produce a PRB × OFDM-symbol jamming map from three
receiver features (power, DMRS residual, EVM), and show that fusing them localizes pilot-, data-
and partial-band jamming better than single-feature and threshold detectors.

**Architecture:** A Sionna link written in the style of the Sionna tutorials: separate blocks for
the UE transmitter, the two fading channels (UE and jammer), the jammer, LS channel estimation,
LMMSE equalization, demapping and decoding. The receiver's intermediate results feed a feature
extractor, whose maps go to threshold baselines and to a small U-Net trained with PyTorch.

**Tech stack:** Python 3.12, Sionna 2.2 (PyTorch backend, `sionna-no-rt`), PyTorch, NumPy,
Matplotlib, pytest. uv locally, `pip install -e .` on Colab.

**Spec:** `docs/proposal.pdf` and the decisions below.

**Time budget:** one month. Week 1 = Tasks 1–4, week 2 = Tasks 5–7, week 3 = Tasks 8–10,
week 4 = Tasks 11–12 plus slides and paper draft. The sliding-window BLER detector and the
mitigation demo are out of scope unless week 4 has time left.

## Global Constraints

- Carrier: 24 PRB (288 subcarriers), 30 kHz SCS, 14 OFDM symbols; single antenna at UE, gNB and
  jammer; single layer; DMRS type 1, length 1, one front-loaded symbol (symbol index 2).
- QPSK with NR LDPC at MCS index 5 (table 1). MCS 0–2 are rejected by Sionna's LDPC encoder.
- Allocation contiguous, ≥ 6 PRB, randomized; channels TDL-A and TDL-C, low mobility.
- Code runs unchanged on Windows, Linux, macOS (CPU) and Colab (CUDA). Tensors and models go on
  `sionna.phy.config.device`; never hard-code a device. Paths via `pathlib`; text files opened with
  `encoding="utf-8"`; scripts are `python -m jamloc.<module>` entry points under
  `if __name__ == "__main__":`.
- Generated data and results only through `jamloc.paths.data_dir()` / `results_dir()`; nothing
  generated is committed; everything reproducible from code + seeds.
- Code style follows the Sionna tutorials: one block per processing step, a model class whose
  `call()` reads top to bottom like the block diagram, short comments that name the step.

## Design decisions

These fill gaps in the proposal. Each can be changed later without touching the rest.

1. **Frequency-domain simulation.** Perfect sync and a cyclic prefix longer than the delay spread
   make the time-domain OFDM step equivalent, so we use Sionna's `OFDMChannel` on the resource
   grid. FFT size 512 (15.36 MHz sampling) stays the nominal numerology.
2. **Allocation.** Sionna's PUSCH always fills its whole bandwidth part, so the UE grid is built
   for `n` PRBs and placed into the 24-PRB carrier grid at the allocation offset. One allocation
   (start, size) is drawn per batch; everything else is drawn per slot.
3. **Channels.** UE and jammer see independent fading channels of the same TDL model with unit
   average power (`normalize_channel=True`). TDL-A: 100 ns delay spread; TDL-C: 300 ns;
   speed 0–3 m/s; carrier 3.5 GHz.
4. **SNR and JSR.** SNR = UE energy per data RE / noise variance (data REs have unit energy).
   JSR = jammer energy per jammed RE / UE energy per data RE. Total jammer energy per slot =
   JSR × number of jammed REs, stored so results can also be plotted against energy.
5. **Jammer.** Complex Gaussian on every subcarrier of the jammed cells. A cell is one PRB × one
   OFDM symbol. Attacks: `none` (benign), `pilot` (DMRS symbol only), `data` (all symbols except
   the DMRS symbol), `partial` (all 14 symbols). The jammed PRBs form a contiguous block of 2–12
   PRBs placed anywhere on the carrier (the jammer does not know the allocation); the
   scheduling-aware variant jams exactly the allocated PRBs. Labels mark jammed cells over all
   24 × 14 cells.
6. **Receiver.** `PUSCHLSChannelEstimator` (linear interpolation) → `LMMSEEqualizer` →
   `Demapper` (max-log) → `LayerDemapper` → `TBDecoder`. The gNB uses the thermal noise variance
   only; it does not know about the jammer. Decoding is optional (needed only for BER/BLER).
7. **Features** (one value per cell, noise-normalized so that a benign slot gives about 0 dB):
   - Power `P = 10·log10(mean |y|² / N0)` over the cell's 12 subcarriers. Defined on all cells.
   - DMRS residual `R`: raw LS estimates `y/x` at the DMRS pilots of the allocation are projected
     onto the channel subspace (eigenvectors of the frequency correlation of a uniform delay
     profile over `[0, τ_max]` with `τ_max = 3 µs`, keeping 1 − 10⁻⁶ of its energy). What is left
     is interference plus noise. Per PRB: mean residual power / expected noise residual
     `(N0/|x_p|²)·(N_p − L)/N_p`. Defined on DMRS-symbol cells of allocated PRBs.
   - EVM `E`: mean `|x̂ − Q(x̂)|²` over a cell's data REs, divided by the mean LMMSE effective
     noise variance there (`Q` = nearest QPSK point). Defined on data cells of allocated PRBs.
   - Masks: `alloc_mask` (scheduled PRB, all symbols), `dmrs_mask` (where R is defined),
     `data_mask` (where E is defined). R and E are stored in dB, clamped to [−10, 40], and set
     to 0 outside their masks.
   - Channel order: `power_db, residual_db, evm_db, alloc_mask, dmrs_mask, data_mask`.
8. **Detectors.** Same U-Net, different input channels; all variants get the masks:
   `fused` (P, R, E), `power` (P), `evm` (E), `no_residual` (P, E: ablation showing what R adds).
   Threshold baselines: EVM threshold (Örnek & Kartal, flags data cells) and residual threshold
   (flags DMRS cells).
9. **Metrics.** Thresholds are set on the validation set so that 1% of non-jammed cells are
   flagged. Pd = fraction of jammed cells flagged, per attack type and JSR, on the test set,
   reported for all jammed cells and for scheduled jammed cells only (cells outside the
   allocation carry no UE signal and are easy). Cell-level F1 at the same threshold.
10. **Dataset.** Train 40 000 slots, validation 5 000 (attack drawn uniformly from the four
    classes, JSR uniform in [−15, 10] dB, SNR uniform in [5, 25] dB, both channels). Test: JSR
    grid −15…+9 dB in 3 dB steps × three attacks × two channels × 500 slots, plus 2 000 benign
    slots per channel. Scheduling-aware test: same grid, 300 slots per point. Seed bases: train
    1 000 000, val 2 000 000, test 3 000 000, scheduling-aware 4 000 000; batch `i` uses seed
    `base + i`.

## Review Focus

1. **Jammer block partly or fully outside the allocation:** labels must still mark those cells,
   and R and E must be 0 there (they have no UE signal). Test in Task 5.
2. **Allocation at either carrier edge (start 0, or start + size = 24) and full-band allocation:**
   no off-by-one in placement and in feature-to-cell mapping. Tests in Tasks 1 and 5.
3. **Extreme operating points (SNR 0 dB with JSR +20 dB):** features must stay finite (no log of
   zero, no division by a tiny noise variance). Test in Task 5.
4. **Deep fades in benign slots:** the residual must not leak the UE's own signal at high SNR,
   or deep-fade slots read as jamming. Test in Task 5 (benign TDL-C at 30 dB SNR).
5. **Reproducibility:** the same seed gives the same slot on the same device. Scenario draws
   (allocation, attack, JSR, SNR, jammer PRBs) come from NumPy and match on every device;
   Gaussian samples on CUDA differ from CPU, so datasets are reproducible per device. Test in
   Task 4.

---

### Task 1: Carrier grid and UE transmitters

**Files:** create `src/jamloc/config.py`, `src/jamloc/ue.py`; test `tests/test_ue.py`.

**Interfaces (produces):**
- `config.SimConfig` (frozen dataclass): `num_prb=24`, `num_symbols=14`, `scs=30e3`,
  `mcs_index=5`, `min_alloc_prb=6`, `dmrs_symbol=2`, `carrier_frequency=3.5e9`,
  `delay_spread={"A": 100e-9, "C": 300e-9}`, `max_speed=3.0`, `tau_max=3e-6`;
  property `num_subcarriers` (= 288).
- `ue.make_pusch_config(cfg, num_prb) -> PUSCHConfig`
- `ue.UEPool(cfg)` with `.transmitter(num_prb) -> PUSCHTransmitter` (cached per size).
- `ue.place_on_carrier(x, start_prb, cfg) -> Tensor` with shape `[..., 288]`, zeros outside.
- `ue.take_allocation(y, start_prb, num_prb) -> Tensor` with shape `[..., 12*num_prb]`.
- `ue.carrier_resource_grid(cfg) -> sionna.phy.ofdm.ResourceGrid` (14 × 288, used by channels).

- [ ] **Step 1: Write failing tests**
  - `test_pusch_config_matches_table1`: for `n in (6, 24)`: `dmrs_symbol_indices == [2]`,
    `num_subcarriers == 12*n`, `tb.num_bits_per_symbol == 2`.
  - `test_place_and_take_roundtrip`: for `(start, n) in [(0, 6), (5, 8), (18, 6), (0, 24)]`:
    `take_allocation(place_on_carrier(x, start, cfg), start, n)` equals `x`, and the carrier grid
    is exactly zero outside subcarriers `[12*start, 12*(start+n))`.
  - `test_transmitter_cached`: `pool.transmitter(8) is pool.transmitter(8)`; output shape
    `[B, 1, 1, 14, 96]`.
- [ ] **Step 2:** `uv run pytest tests/test_ue.py` → fails (module missing).
- [ ] **Step 3:** Implement. `place_on_carrier` pads the last dimension; no copies per slot.
- [ ] **Step 4:** `uv run pytest tests/test_ue.py` → pass.
- [ ] **Step 5:** Commit "Add carrier grid and UE transmitter pool".

### Task 2: Jammer

**Files:** create `src/jamloc/jammer.py`; test `tests/test_jammer.py`.

**Interfaces (produces):**
- `jammer.ATTACKS = ("none", "pilot", "data", "partial")` (index = integer code).
- `jammer.cell_mask(attack, jam_start, jam_size, cfg) -> BoolTensor [B, 24, 14]`
  (`attack`, `jam_start`, `jam_size` are `LongTensor [B]`).
- `jammer.Jammer(cfg)` (Sionna `Block`): `call(cells: BoolTensor[B,24,14], jsr_db: Tensor[B])
  -> Tensor[B, 1, 1, 14, 288]` complex Gaussian with variance `10**(jsr_db/10)` on the REs of
  jammed cells, exactly 0 elsewhere.

- [ ] **Step 1: Write failing tests**
  - `test_attack_patterns`: start 4, size 5: `pilot` → only column 2 set, rows 4–8;
    `data` → rows 4–8, all columns except 2; `partial` → rows 4–8, all 14 columns;
    `none` → no cell set. Start 20, size 4 → rows 20–23 (carrier edge).
  - `test_jammer_power`: B=64, `partial`, full band, JSR 10 dB → mean `|z|²` over jammed REs
    within 5% of 10.0; unjammed REs exactly 0.
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** Implement. Expand the cell mask to REs with `repeat_interleave(12)` on the
  PRB axis.
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** Commit "Add resource-selective Gaussian jammer".

### Task 3: Receiver chain

**Files:** create `src/jamloc/receiver.py`; test `tests/test_receiver.py`.

**Interfaces (consumes):** `UEPool.transmitter` (Task 1).
**Interfaces (produces):**
- `receiver.PUSCHReceiverChain(tx)` (Sionna `Block`): `call(y, no, decode=True) -> dict` with
  `h_hat`, `err_var`, `x_hat [B, num_data]`, `no_eff [B, num_data]`, and, when decoding,
  `b_hat`, `tb_crc`. `y` is the allocation grid `[B, 1, 1, 14, 12n]`, `no` is `[B]`.
- `receiver.data_positions(tx) -> (symbol_idx, subcarrier_idx)` (LongTensors, length
  `num_data`): grid position of each entry of `x_hat`.

- [ ] **Step 1: Write failing tests**
  - `test_noiseless_identity`: `y = x` from the transmitter, `no = 1e-4` → `x_hat` matches
    `x[..., symbol_idx, subcarrier_idx]` (atol 1e-2) and `b_hat == b`.
  - `test_benign_tdl_high_snr_decodes`: TDL-A through `OFDMChannel` on the allocation grid,
    SNR 30 dB, B=16 → all `tb_crc` true.
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** Implement following `PUSCHReceiver.__init__` (estimator with
  `interpolation_type="lin"`, `StreamManagement(np.ones([1, 1], bool), 1)`), but with
  `LMMSEEqualizer` + `Demapper("maxlog", "qam", 2)` instead of `LinearDetector`, so `x_hat` and
  `no_eff` are visible.
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** Commit "Add step-by-step PUSCH receiver chain".

### Task 4: Jammed link and BER/BLER experiment

**Files:** create `src/jamloc/link.py`, `src/jamloc/experiments/__init__.py`,
`src/jamloc/experiments/ber_bler.py`; test `tests/test_link.py`.

**Interfaces (consumes):** Tasks 1–3.
**Interfaces (produces):**
- `link.Scenario` (dataclass): `channel: str` ("A" or "C"), `alloc_start: int`,
  `alloc_size: int`, per-slot tensors `attack`, `jam_start`, `jam_size` (long, `[B]`) and
  `jsr_db`, `snr_db` (float, `[B]`).
- `link.draw_scenario(rng, batch_size, cfg, channel, attacks=(0, 1, 2, 3),
  jsr_db=(-15.0, 10.0), snr_db=(5.0, 25.0), jam_size=(2, 12), scheduling_aware=False,
  alloc=None) -> Scenario`. Ranges are uniform; a float instead of a tuple fixes the value;
  `alloc=(start, size)` fixes the allocation.
- `link.seed_everything(seed) -> np.random.Generator` (sets `sionna.phy.config.seed`).
- `link.SlotBatch` (dataclass): `y` `[B, 14, 288]`, `no` `[B]`, `labels` `[B, 24, 14]` bool,
  `scenario`, `y_alloc` `[B, 1, 1, 14, 12n]`, `rx` (dict from Task 3), `b`, `tx` (the
  transmitter used).
- `link.JammedPUSCHLink(cfg)` (Sionna `Block`): `call(scenario, decode=False) -> SlotBatch`.
- `python -m jamloc.experiments.ber_bler --channel A --snr-db 15 --batch-size 64` writes
  `results_dir()/ber_bler/ber_bler_{channel}.csv` and `.png`: BER and BLER vs JSR for `pilot`,
  `data`, `partial`, jamming the whole allocation (worst-case overlap), via `sim_ber` with JSR
  as the sweep variable and `early_stop=False`; a second panel plots BLER against total
  jammer energy per slot.

- [ ] **Step 1: Write failing tests**
  - `test_labels_match_jammer`: labels equal `cell_mask(...)` for the drawn scenario.
  - `test_same_seed_same_output`: same seed twice → identical `y`.
  - `test_benign_high_snr_decodes`: SNR 25 dB, attack `none`, B=32, TDL-A → all CRC true.
  - `test_partial_jamming_breaks_link`: scheduling-aware `partial`, JSR +10 dB, SNR 25 dB →
    BLER > 0.5.
  - `test_jammer_outside_allocation_is_harmless`: allocation PRBs 0–7, jammer PRBs 12–20,
    `partial` JSR +10 dB, SNR 25 dB → all CRC true; labels set only on rows 12–20.
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** Implement. `call()` reads as: transmit → place on carrier → UE channel →
  jammer → jammer channel → add the two → AWGN → take allocation → receiver.
- [ ] **Step 4:** run → pass; run the experiment for channels A and C, then check the figure
  (expect pilot jamming to need the least total energy for a given BLER).
- [ ] **Step 5:** Commit "Add jammed PUSCH link and BER/BLER sweep".

### Task 5: Feature extraction

**Files:** create `src/jamloc/features.py`; test `tests/test_features.py`.

**Interfaces (consumes):** `SlotBatch` (Task 4), `data_positions` (Task 3).
**Interfaces (produces):**
- `features.FEATURE_NAMES` (channel order from design decision 7).
- `features.dmrs_ls(y_alloc, tx) -> (h_ls [B, N_p], pilot_sc LongTensor [N_p], pilot_power float)`
- `features.delay_subspace(pilot_sc, scs, tau_max, energy=1 - 1e-6) -> Tensor [N_p, L]`
  (orthonormal columns).
- `features.power_map(y, no, cfg) -> [B, 24, 14]` (linear, noise-normalized).
- `features.residual_map(batch, cfg) -> [B, 24, 14]` (linear; 0 outside `dmrs_mask`).
- `features.evm_map(batch, cfg) -> [B, 24, 14]` (linear; 0 outside `data_mask`).
- `features.masks(batch, cfg) -> (alloc_mask, dmrs_mask, data_mask)`, each `[B, 24, 14]` float.
- `features.extract(batch, cfg) -> Tensor [B, 6, 24, 14]` (float32).

- [ ] **Step 1: Write failing tests**
  - `test_delay_subspace`: orthonormal (`Bᴴ B ≈ I`), `L < N_p`; a noiseless TDL-C channel's
    pilot response keeps < 1e-4 of its energy outside the subspace.
  - `test_benign_residual_is_noise_level`: TDL-C, SNR 30 dB, attack `none`, B=64 → mean of
    linear R over `dmrs_mask` cells in [0.7, 1.4].
  - `test_benign_evm_is_noise_level`: TDL-A, SNR 15 dB → mean linear E over `data_mask` in
    [0.5, 1.5].
  - `test_pilot_attack_raises_residual_only_on_jammed_prbs`: scheduling-aware `pilot`, JSR
    0 dB → mean R on jammed PRBs > 5× benign mean; `data` attack at the same JSR → R on jammed
    PRBs within [0.7, 1.4], E on jammed data cells > 3.
  - `test_features_finite_and_masked`: SNR 0 dB, JSR +20 dB, jammer partly outside allocation,
    allocations `(0, 6)` and `(18, 6)` → all values finite; R and E are 0 outside their masks.
- [ ] **Step 2:** run → fails.
- [ ] **Step 3:** Implement as in design decision 7. Build the subspace with
  `torch.linalg.eigh` of `C[k, k'] = exp(-jπΔf τ) sinc(Δf τ)` (`Δf` = subcarrier distance in Hz,
  `τ = tau_max`), cached per allocation size.
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** Commit "Add power, DMRS residual and EVM feature maps".

### Task 6: Walkthrough notebook and feature-map figure

**Files:** create `notebooks/01_link_walkthrough.ipynb`,
`src/jamloc/experiments/feature_maps.py`; test `tests/test_experiments.py`.

- The notebook goes through one slot step by step, in Sionna tutorial style: config → transmitter
  → channel → jammer → receiver → features, plotting the grids. It is the presentation
  material.
- `python -m jamloc.experiments.feature_maps --snr-db 15 --jsr-db 0` writes
  `results_dir()/feature_maps.png`: rows = benign, pilot, data, partial; columns = label, P, R, E.
- [ ] **Step 1:** `test_feature_maps_script_runs`: with `JAMLOC_RESULTS_DIR` set to `tmp_path`,
  `main(["--batch-size", "4"])` creates `feature_maps.png`.
- [ ] **Step 2–4:** fail → implement → pass; check the figure against proposal Table 2.
- [ ] **Step 5:** Commit "Add link walkthrough notebook and feature-map figure".

### Task 7: Metrics and threshold baselines

**Files:** create `src/jamloc/metrics.py`, `src/jamloc/baselines.py`;
test `tests/test_metrics.py`.

**Interfaces (produces):**
- `metrics.threshold_at_fa(neg_scores: ndarray, fa=0.01) -> float`
- `metrics.detection_rate(pos_scores, thr) -> float`
- `metrics.f1(pred: ndarray[bool], label: ndarray[bool]) -> float`
- `baselines.evm_scores(features) -> ndarray [N, 24, 14]` (linear E on `data_mask`, −inf elsewhere)
- `baselines.residual_scores(features) -> ndarray [N, 24, 14]` (linear R on `dmrs_mask`,
  −inf elsewhere)

- [ ] **Step 1: Write failing tests**
  - `test_threshold_at_fa`: 100 000 uniform scores → threshold ≈ 0.99 (±0.005); flags ≈ 1%.
  - `test_f1`: perfect prediction → 1.0; all-false prediction with positives present → 0.0.
  - `test_masked_cells_never_flagged`: −inf scores are never above any finite threshold.
- [ ] **Step 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit "Add detection metrics and threshold baselines".

### Task 8: Dataset generation

**Files:** create `src/jamloc/dataset.py`; test `tests/test_dataset.py`.

**Interfaces (produces):**
- `dataset.generate(split, num_slots=None, batch_size=64) -> Path` writes
  `data_dir()/{split}.npz` with `features` float32 `[N, 6, 24, 14]`, `labels` uint8
  `[N, 24, 14]`, and per-slot `attack`, `jsr_db`, `snr_db`, `channel`, `alloc_start`,
  `alloc_size`, `jam_start`, `jam_size`, `scheduling_aware`. Splits and seeds from design
  decision 10: `train`, `val`, `test`, `sched`.
- `dataset.load(split) -> dict[str, ndarray]`.
- CLI: `python -m jamloc.dataset --split train [--num-slots N]`.

- [ ] **Step 1:** `test_generate_small`: `num_slots=128` into `tmp_path` → shapes as above,
  benign slots have all-zero labels, `train` and `test` features differ.
- [ ] **Step 2–4:** fail → implement → pass. Time the full `train` split on the Mac.
- [ ] **Step 5:** Commit "Add dataset generation".

### Task 9: U-Net and training

**Files:** create `src/jamloc/model.py`, `src/jamloc/train.py`; test `tests/test_model.py`.

**Interfaces (produces):**
- `model.VARIANTS = {"fused": [0,1,2,3,4,5], "power": [0,3,4,5], "evm": [2,3,4,5],
  "no_residual": [0,2,3,4,5]}` (indices into `FEATURE_NAMES`).
- `model.TinyUNet(in_channels, base=12)`: `forward([B, C, 24, 14]) -> logits [B, 24, 14]`.
  Two levels: pad symbols 14 → 16, (conv-BN-ReLU)×2 per level, channels 12 → 24 → 48,
  transposed-conv upsampling with skip connections, crop back to 24 × 14.
- `train.train(variant, epochs=30, batch_size=128, lr=1e-3, seed=0) -> Path` saves
  `results_dir()/models/{variant}.pt` (weights of the epoch with best validation loss).
  Loss `BCEWithLogitsLoss(pos_weight = negatives/positives in train)`, AdamW.
- CLI: `python -m jamloc.train --variant fused`.

- [ ] **Step 1: Write failing tests**
  - `test_param_count`: 10 000 ≤ params ≤ 100 000 for the `fused` variant.
  - `test_output_shape`: `[8, 6, 24, 14] → [8, 24, 14]`.
  - `test_overfits_tiny_batch`: on 16 random slots, loss drops by ≥ 50% in 100 steps.
- [ ] **Step 2–4:** fail → implement → pass; train all four variants (Mac CPU; Colab if slow).
- [ ] **Step 5:** Commit "Add U-Net detector and training script".

### Task 10: Evaluation

**Files:** create `src/jamloc/evaluate.py`; test `tests/test_evaluate.py`.

**Interfaces (produces):**
- `evaluate.score_all(split) -> dict[str, ndarray [N, 24, 14]]` for the four U-Net variants and
  the two threshold baselines.
- `evaluate.main(["--split", "test"])` writes `results_dir()/eval/{split}_pd_vs_jsr.csv`
  (columns `detector, attack, channel, jsr_db, cells, pd`, where `cells` is `all` or `scheduled`),
  `{split}_f1.csv`, and Pd-vs-JSR plots (one panel per attack).
- Thresholds come from `val` at 1% per-cell false alarm (design decision 9).

- [ ] **Step 1:** `test_oracle_detector`: on a tiny generated split with an oracle score (= label),
  Pd = 1 and F1 = 1; CSV has the columns above.
- [ ] **Step 2–4:** fail → implement → pass; run on the real `test` split.
- [ ] **Step 5:** Commit "Add evaluation of Pd at 1% false alarm and F1".

### Task 11: Robustness and latency

**Files:** create `src/jamloc/experiments/latency.py`; modify `src/jamloc/evaluate.py`;
test `tests/test_experiments.py`.

- Evaluate all detectors on the `sched` split (scheduling-aware jammer, never seen in training).
- `python -m jamloc.experiments.latency --reps 200` writes `results_dir()/latency.csv`: median
  CPU time per slot (batch 1) for feature extraction from receiver outputs, for the U-Net forward
  pass, and their sum, plus parameter counts per variant.
- [ ] **Step 1:** `test_latency_script_runs`: `main(["--reps", "5"])` writes `latency.csv` with
  one row per variant.
- [ ] **Step 2–4:** fail → implement → pass; run both on the Mac.
- [ ] **Step 5:** Commit "Add scheduling-aware robustness test and latency benchmark".

### Task 12: Results write-up

**Files:** modify `README.md`; create `docs/results.md`.

- [ ] `README.md`: a "How it works" section with the block diagram in text and the commands that
  reproduce every figure, in order.
- [ ] `docs/results.md`: the final figures and tables (BER/BLER, feature maps, Pd vs JSR,
  F1/size/latency, robustness) with two or three lines of interpretation each, ready to move into
  slides and the paper.
- [ ] Full test suite passes: `uv run pytest`. Commit "Add results summary".
