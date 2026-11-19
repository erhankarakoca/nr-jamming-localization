# Plan v6: fix the weak points of review v6

18 November 2026. Works through all twelve items of `docs/review-v6.md`. The review stays as
written; it is the baseline the work is measured against.

## How the work is tracked

- All work happens on the branch `review-v6`, merged through one pull request.
- The PR description has one row per review item, with its status (**fixed**, **partly fixed**,
  **not fixed**) and two columns:
  - **What we did:** the code, experiment, result and paper change, with the numbers.
  - **What is still missing:** what remains and why.
  The table is updated as items land; the same table goes into the status section of this file
  when the PR is merged.
- An item counts as fixed only when its acceptance check below holds. If an experiment gives an
  unwelcome result, the item is still fixed: the paper then reports that result.
- The paper is edited in the Overleaf clone and pushed to Overleaf as each item lands (Overleaf
  has no branches; its history can revert any change). The PR lists the Overleaf commits per item.

Decisions (18 November):

- Push paper edits to Overleaf as items land.
- Length: restructure the arc and remove the templated parts now; the 6-page cut is a separate
  last step once the new numbers are in, and the long version is kept as an extended version.
- New inputs (empty-comb energy, spoofing consistency feature) are first reported as ablation
  rows. One that clearly helps becomes part of the proposed detector; the old detector stays as a
  row.

PR description layout:

```text
Review v6: what is fixed and what is missing

| # | Review item | Status | What we did | What is still missing |
| 1 | Recovery does not isolate the map | fixed | receiver modes lmmse/slot/prb ...; BLER table ...; Sec. VI | - |
...
Overleaf commits: <hash> <subject> (items)
Reproduce: <commands>
```

## Workstreams

### W1. Does the receiver need the map, and its resolution? (review #1)

- **Receiver modes** (`recovery.py`), all with the same delay-LMMSE interpolator, equalizer,
  demapper and decoder:
  - `lmmse`: no map, no weighting. This isolates the interpolator.
  - `slot`: one noise-plus-interference estimate for the whole allocation, from the residual and
    the excess data energy. This is what an interference-aware receiver without a map does.
  - `prb`: per-PRB pilot and data estimates, from the block test's map, used softly.
  - `soft` with the U-Net map.
  - `soft` with the true map.
  - The existing `standard`, `dmrs` and hard modes stay as references.
- **Jammers.**
  - The three nominal attacks, where a per-PRB estimate should suffice.
  - Pulsed and hopping jammers, where resolution in time can matter.
- **Runs.** TDL-C and TDL-A at 15 dB, all three checkpoints, every receiver on the same slots.
  BLER is paired, with Wilson intervals. Low-BLER points run up to 10 000 slots, so that "below
  x%" claims rest on a bound.
- **Acceptance.** A table of BLER against map source, with the receiver held fixed, per attack.
  The paper states where the map and its resolution help, and where they do not.

### W2. Operating point and alarm rate (review #5)

- **Spare margin.** BLER of the unjammed link against SNR (MCS 5, TDL-A/C). This gives the SNR
  at which MCS 5 reaches 10% BLER, the usual link-adaptation target, and so the spare margin at
  15 dB.
- **Margin against spare margin.** Rerun the detection-before-damage margin at two lower SNRs
  (spare margin of about 3 and 6 dB), and the recovery comparison at the lowest one. Plot the
  detection margin against the spare link margin.
- **Alarm rate.** State that 1% slot false alarm means 20 alarms per second per UE. Generate
  10 000 more benign margin slots, so the slot-level operating point can go down to 0.1%. Point
  to multi-slot averaging (Corollary 2) for reporting.
- **Acceptance.** The margin is reported as a function of the link margin. The abstract claim
  names its conditions.

### W3. Noise power the receiver does not know (review #3)

- **Wrong N0 in the whole chain.** A link option `noise_error_db` makes the receiver and the
  features use N0 times 10^(e/10), drawn per slot. This replaces the feature-shift
  approximation.
- **A real estimator.** Per slot, from the empty DMRS comb (6 REs per PRB) and the unscheduled
  PRBs, with a robust median across PRBs. There is also a long-term version averaged over slots.
  The SNR for the SNR bins is estimated from the pilot power.
- **Experiments.**
  - The noise-mismatch curve with the full chain, for the nominal and the jitter-trained models.
  - The estimator's error with and without jamming. A scheduling-aware jammer that covers the
    whole carrier biases it.
  - F1 and false alarms with the estimator; the fused model is retrained on estimated N0 if
    needed.
- **Acceptance.** The paper says how N0 is obtained, gives the numbers with a realistic
  estimator, and states the persistent-jammer limitation.

### W4. Legitimate interference (review #4)

- **Interferer.** A new waveform `ue`: a neighbor-cell UE with QPSK data and its own DMRS
  sequence on the pilot comb, through its own channel.
- **Test split.** `neighbor`, with label 0 everywhere, swept over interference-to-noise ratio,
  block size, and scheduled versus unscheduled PRBs.
- **Experiment.** The flagged fraction against INR for every detector.
- **Paper.** Frame the output as an interference map. Discuss how intent could be separated:
  neighbor-cell DMRS sequences are known to the network, and gNBs coordinate over Xn. Cite the
  Ericsson application.
- **Acceptance.** A measured false-flag rate under legitimate interference, and the framing
  above.

### W5. Baselines and fair comparison (review #6)

- **Empty-comb test.** Store the per-PRB energy on the empty comb of the DMRS symbol in the
  datasets. Add an empty-comb threshold detector with its exact Gamma(6, 1/6) null, and a block
  test that uses it. Evaluate both on the nominal and the comb jammer.
- **Per-cell MLP with the U-Net's size.** About 66k parameters, three seeds, to separate spatial
  context from capacity.
- **PRB-level metric.** Score which PRBs had their DMRS or their data attacked. EVM-based
  detectors are then not penalized by the label definition alone.
- **Reporting.**
  - Every F1 is reported with the achieved false-alarm rate per attack.
  - "Training-free" becomes "no learned parameters; empirical EVM null".
- **Acceptance.** The detection table has the new baseline rows and per-attack false alarms.

### W6. Pilot spoofing (review #7)

- **Step 1: retrain.** Train the fused network on the mixed attacks plus spoofing (labels on the
  DMRS cells). Evaluate on whole-allocation and partial spoofing, at -9 dB and pooled, with each
  population named.
- **Step 2, only if step 1 still misplaces the attack.** Add a pilot-versus-data consistency
  input: the DMRS-to-data power ratio per PRB, or a decision-directed channel estimate from the
  data compared with the DMRS estimate.
- **Paper.** Move spoofing next to the residual statistics, as an identifiability limit. Keep
  the evaluation result. Point out that the slot alarm fires before the BLER rises.
- **Acceptance.** Spoofing localization is reported with and without spoofing in training, and
  the paper explains the limit.

### W7. Statistics (review #9)

- **More allocations.** Regenerate the test-type splits (test, sched, unseen grids, margin) with
  batches of 16 slots: about 31 allocations per 500-slot point instead of 4.
- **Uncertainty.** Bootstrap over batches, the allocation unit, for 95% intervals on F1 and P_d.
- **Comparisons.** Paired differences between detectors per seed; per-seed detection margins.
- **Paper.** Seed std is described as training variability, the bootstrap as sampling
  uncertainty. Differences inside the noise are no longer interpreted.
- **Acceptance.** Every compared number in the paper has an interval, or is a paired difference
  with one.

### W8. Simulation conditions (review #10)

- **Unnormalized channels.** A config flag for Sionna's per-slot channel normalization, and a
  test split with ensemble-normalized channels, evaluated as one more unseen condition.
- **Paper.** State the per-slot normalization, the frequency-domain model (no ISI or ICI, TDL-C
  tail beyond the CP), single antenna, grid alignment and simulation-only, in the abstract and the
  system model.

### W9. Paper: story, novelty, wording, length (review #2, #8, #11, #12)

- **Related work.** Rewrite it around the closest methods, each with what it observes, outputs
  and calibrates:
  - the Ericsson application;
  - Magiera and Rajchowski 2024;
  - Fadul and Reising 2024;
  - Ornek and Kartal 2023 and 2024;
  - Amini et al. 2025;
  - Sheng et al. 2014;
  - delay-domain noise estimation in DFT-based channel estimation;
  - Han et al. 2008.

  Codex does the literature search for the noise-estimation precedent and checks every DOI.
- **Contribution at its true size.**
  - The allocation-dependent per-PRB law, with edge effects and leakage.
  - The distinction between attacked and damaged cells.
  - Fusion for weak pilot jamming.

  The propositions are presented as an application of the Gaussian quadratic form, with shorter
  proofs.
- **U-Net framing (#8).** The network is the fusion that handles weak pilot jamming. It learns
  the attack structure it is trained on.
- **Wording (#11).** Every item in the review's list, plus the conditions of the CFAR claim in
  the abstract and conclusion.
- **Arc (#12).**
  1. Misplaced EVM.
  2. The residual and its limits, spoofing included.
  3. Localization on scheduled cells.
  4. A receiver decision that needs the map.

  The templated "Interpretation" paragraphs and the roadmap go.
- **Final check.** An independent review of the text, and of every number against the CSVs, as
  in review v6.

## Order

| Phase | Work | Where | Who |
| --- | --- | --- | --- |
| A | Branch, this plan, draft PR with the empty status table | | me |
| B | Code with tests: comb energy, batch-16 splits, noise options, channel flag, recovery modes (W1, W3, W5, W7, W8) | laptop | me |
| B' | In parallel: `ue` interferer and `neighbor` split (W4), 66k per-cell MLP, PRB metric and bootstrap tool (W5, W7) | git worktree | Codex |
| C | Regenerate all splits, retrain all models, mixed + spoofing training (W6) | Colab GPU | me |
| D | All evaluations, intervals, `docs/results.md` section 17 | laptop | me |
| E | Recovery redesign, BLER against SNR, margin at lower SNR (W1, W2) | Colab GPU | me |
| F | Paper text and figures (W9 and the paper parts of W1-W8); literature search | Overleaf | me, Codex (search) |
| G | Independent review of text and numbers; fixes; status table; PR ready for review | | Codex review, me |

Phases C to E replace every number in the paper, so the text (F) waits for them. Rough cost: two
days of work, about 3-4 GPU hours on Colab.

## Out of scope (stated as limitations, listed as "not fixed" in the PR)

Multiple receive antennas, over-the-air validation, asynchronous jammers in a time-domain link,
and independent training data per seed.

## Status (19 November 2026)

- [x] A. Branch `review-v6`, this plan, draft PR #9.
- [x] B / B'. Code with tests: noise options and per-slot estimator (`noise.py`), empty-comb
  feature and baselines, 16-slot grids with batch index and estimated SNR, new splits, recovery
  modes (`lmmse`, `slot`, block-test map, mixed-attack map, true map per PRB), margin at several
  SNRs and slot false-alarm targets, full-chain calibration; Codex: neighbor-cell UE, intervals
  and PRB-level F1.
- [x] C. Full rebuild on a Colab T4 (`experiments.paper_run`, 2 h 10 min, about 3 compute units).
- [x] D. Evaluations, intervals, `docs/results.md` section 17.
- [x] E. Recovery (3 seeds, nominal / pulsed / hopping, TDL-A and TDL-C, 8 dB SNR) on the M1 Pro;
  latency re-measured on an idle CPU.
- [x] F. Paper rewritten and pushed to Overleaf; independent audit (693 numbers checked) and fixes.
- [ ] G. Focused re-check of the finished sections; final PR table; 6-page cut as a separate step.

Results that changed the plan along the way:

- The empty-comb U-Net helps on nominal attacks but a pilot-comb jammer defeats it, so it stays an
  ablation (decision rule of this plan: adopt only if it clearly helps).
- Training with spoofing in the mix localizes whole-allocation spoofing, so the pilot-versus-data
  feature of W6 step 2 was not needed; partial spoofing stays unsolved and is reported.
- The block test's map fails under strong partial-band jamming because of its pilot-only rule; a
  stricter rule is a trade-off (overall F1 0.868 -> 0.864), so the rule stays and the paper says so.
