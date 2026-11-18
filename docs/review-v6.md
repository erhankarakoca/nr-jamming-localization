# Review v6: weak points of the paper as a whole

18 November 2026. Two independent reviews of the Overleaf draft (one on story, novelty and related
work with a literature search; one auditing the claims against the code and the saved CSVs) plus
my own read. Every finding below was checked against the code, the CSVs or the source it cites.

Verdict a demanding conference reviewer would give today: **weak reject**. The deciding reason:
the paper never shows that a PRB x symbol map changes a receiver decision compared with a coarser
interference estimate. The nominal numbers themselves reproduce from the saved results (30+
claims checked; the mismatches are listed in item 11).

Effort: **T** text only, **R** rescoring saved data, **S** new simulation.

## Ranked drawbacks

| # | Drawback | Severity | Effort |
| --- | --- | --- | --- |
| 1 | Recovery does not isolate the value of the map or of its resolution | major (deciding) | S |
| 2 | Novelty narrower than claimed; close precedent uncited; propositions are textbook-level | major | T |
| 3 | All features use the simulator's exact noise power and SNR | major | S |
| 4 | Benign world too clean: empty carrier, no other UEs, no inter-cell interference | major | T + S |
| 5 | Operating point flatters the results: ~9 dB spare link margin; 1% false alarm is 20 alarms/s | major | T + S |
| 6 | Baselines: an obvious simple detector is missing; coverage and capacity confounds | major | R + S |
| 7 | Pilot spoofing defeats the residual and wrecks the fused map | major | T or S |
| 8 | The U-Net's lead is modest and it overfits the attack structure | medium | T |
| 9 | Statistics: same-data seeds, allocation drawn once per batch, point-estimate margins | medium | R + S |
| 10 | Simulation conditions not stated (per-slot channel normalization, no ISI/ICI) | medium | T |
| 11 | Wording and numbers stronger than the evidence | minor | T |
| 12 | Length and presentation: about twice the page budget, one arc buried | major for submission | T |

## Details

**1. Recovery does not isolate the map.** Every map-aware receiver also switches from Sionna's
linear interpolation to the delay-covariance LMMSE filter, and there is no unweighted LMMSE
control, so part of the gain may come from the better interpolator. The block-test map is never
used softly, only the power map hard. The nominal attacks fill whole PRB rows, so a per-PRB
pilot/data interference estimate represents every nominal attack exactly; the paper's sentence
"recovering from data jamming requires the map's resolution in time" is untested. The P+E map does
as well as the fused one when used softly. One detector checkpoint, one scenario (TDL-C, 15 dB,
4 of 12 PRBs).
Fix: hold the receiver fixed and vary only the map source: none (LMMSE), slot-wide estimate,
per-PRB pilot/data estimate, block test (soft), U-Net (soft), true map; add pulsed and hopping
jammers, where resolution can matter; same slots, paired BLER with intervals, all checkpoints.

**2. Novelty.** Close prior work: an Ericsson patent application (Zhong, WO2022151467A1, 2022)
estimates jamming-plus-noise from the residual on known uplink pilots including DMRS, and tries to
separate jamming from inter-cell PUSCH interference; delay-domain noise estimation (taps beyond the
channel's delay support) is standard in DFT-based channel estimation; Sheng et al. (IEICE 2014) use
the delay-domain noise subspace; Han et al. (2008) detect jammed pilots after LS. Already in
`references.bib` but uncited: Magiera and Rajchowski 2024 (selective DMRS contamination on a
Sionna PUSCH), Fadul and Reising 2024 (learned SISO OFDM pilot-jamming defense), Ornek and Kartal
2024 (RB scheduling from EVM, an operational consumer of the map), Amini et al. 2025. The claim
that EVM maps were shown "without a detection rule" is too broad (Ornek 2023 thresholds RB EVM).
Propositions 1-2 are the standard Gaussian-quadratic-form result applied to a projector.
Defensible novelty: the allocation-dependent per-PRB null law with edge effects and leakage, on
scheduled PUSCH DMRS, and the attacked-versus-damaged-cell distinction. Fix: rewrite related work
around the closest methods, state the contribution at that size, consider calling the results a
lemma plus corollaries.

**3. Genie noise power.** `link.py:191-200` passes the true N0 to the receiver and the features;
the paper says "estimated noise power". SNR bins (block test, SNR-dependent thresholds) use the
simulated SNR. A 0.5 dB underestimate gives 15% false alarms, 1 dB gives 37%; the jitter training
shifts stored features rather than re-running the receiver. A gNB that estimates N0 from its DMRS
would also absorb a persistent low-level jammer into the estimate. Fix: state it now; then add a
real estimator (e.g. empty REs or a robust per-slot residual median), run it through receiver,
features, thresholds and recovery.

**4. Benign world.** Outside one UE's allocation the carrier is empty, and the benign class is
noise plus fading. In a loaded, synchronized network other UEs fill those PRBs, and neighbor-cell
PUSCH is grid-aligned, PRB-selective and noise-like at the residual: these features cannot tell it
from jamming. The map localizes interference, not intent. Fix: say so and frame the output as an
interference map; optionally measure false alarms with a co-channel neighbor-cell UE (its own
DMRS, its own channel).

**5. Operating point.** MCS 5 (QPSK, rate 0.37) decodes error-free at 15 dB SNR; data jamming on
all PRBs reaches 10% BLER near -6 dB JSR, so the link has about 9 dB to spare. Link adaptation
would remove most of it, and the "detection before damage" margin and the recovery results would
shrink. Also, 1% per-cell false alarm is 3.4 false cells per slot, and 1% slot false alarm is 20
false alarms per second per UE at 2000 slots/s: fine for a receiver that tolerates false flags
(the soft receiver), unusable for alarm reporting without multi-slot integration. Fix: rerun the
margin and recovery at an SNR near the MCS's working point (cheap), state the alarm rate, use
Corollary 2 for reporting.

**6. Baselines.** For the nominal pilot attack (whole DMRS symbol), the six empty comb REs per PRB
give a trivial power test with a known Gamma null; only the comb jammer evades it. This is the
first baseline a reviewer would ask for, and it also states the residual's real purpose. EVM-only
F1 = 0 on pilot attacks follows from the label definition; a PRB-level metric would show the
residual's value without that. Achieved false alarms differ per attack (fused 0.55% on pilot slots
vs block test 2.72%; fused 1.75% on data). The per-cell MLP has 4.8k parameters against 66k, so
the "spatial context" ablation also changes capacity. "Training-free" block test still fits the
EVM null on validation data and uses the true SNR bin.

**7. Spoofing.** Replaying the pilots over the whole allocation hides from the residual (0.96% vs
1.00%), costs the most link damage per unit energy, and drives the fused map to F1 0.03 with 27%
false alarms (0.018 and 41% at -9 dB alone); mixed-attack training does not help. It currently
sits at the end of the robustness section. Positive side: the slot alarm fires on 91-95% of slots
at -9 dB, before the BLER rises. Fix (T): move it next to the residual as an identifiability
limit. Fix (S): add a pilot-versus-data consistency feature (power ratio, or a decision-directed
channel re-estimate from the data compared with the DMRS estimate) and train with spoofing.

**8. U-Net value.** It beats the block test by 0.08 F1, mostly on pilot attacks; with hopping and
pulsed jammers it falls to 0.59/0.63 with 10-21% false alarms, below the per-cell MLP, until it is
retrained on mixed attacks. Combined with item 1, a reviewer may ask what the network is for.
Fix: frame the network as the fusion that handles weak pilot jamming and context, and say plainly
that it learns the attack structure it is trained on.

**9. Statistics.** The three seeds share the same train/val/test data, so the std is training
noise only. The allocation is drawn once per batch (`link.py:82-108`), so a 500-slot test point
has about four allocations; the paper says "random per slot". Some interpreted gaps are inside the
seed spread (partial-band P+R+E vs P+E: +0.0004 +- 0.0028; mixed-attack matched gain +0.0016 vs
std 0.0021). The 8.2 dB margin is read from seed-averaged curves; per seed it is 7.5-8.8 dB.
Fix: paired deltas, bootstrap over batches, more allocation draws (smaller batches) for the test
and margin sets.

**10. Unstated simulation conditions.** Both channels use Sionna's `normalize_channel=True`, which
scales every slot's UE and jammer channel to unit energy over the carrier: no slot-to-slot link
budget variation, and the JSR holds per slot, not only on average. The frequency-domain model has
no ISI or ICI; TDL-C's last tap (2.6 us) exceeds the 2.34 us normal CP. Single antenna, grid
alignment and simulation-only should be in the abstract, not only the conclusion.

**11. Wording and numbers.** "Estimated noise power" (it is the true one); "random per slot" (per
batch); "within 0.005" (0.0052); spoofing F1 0.03 / 27% sits in a -9 dB sentence but is the sweep
aggregate; "at least 8.2 dB" (point estimate); "below 0.2%" (observed, 5/3200 on TDL-A, upper 95%
limit 0.37%); "within one slot of CPU time" (only the 7.7k frozen model; the 66k model needs 0.75
ms); "closed-form detection probabilities" (they need Monte Carlo over the jammer gains); the
CFAR claim in abstract and conclusion needs the exact-subspace, known-N0 and benign-slot
conditions; the 12 ms coherence time is an estimate, not measured.

**12. Length and presentation.** About 10,400 words, 8 figures, 6 tables: roughly 11-13 IEEE
pages against 5-6. The repeated "Interpretation" paragraphs, the contribution inventory and the
roadmap read as templated. Suggested arc: misplaced EVM -> DMRS residual and its limits (incl.
spoofing) -> localization on scheduled cells -> a receiver decision that needs the map. Candidates
to cut or move to supplementary material: notation block, roadmap, loss display, eigenvalue table,
separate margin figure, most of the latency sweep, full robustness matrix, multi-slot corollary,
tail appendix.

## What reviewers would keep

- The attacked-versus-damaged distinction and the equalizer-error explanation.
- The residual theory's predictions (edge thresholds, leakage onto clean neighbors) matching the
  link simulation: more persuasive than the F1 headline.
- The weak-pilot gain from the residual: +0.19 in P_d at -15 dB, consistent in every seed pair.
- Hard versus soft use of a map.
- The candid negative results (spoofing, long delays, noise mismatch, hopping).

## Out of scope for this paper

Multiple receive antennas, over-the-air validation, asynchronous (off-grid) jammers in a
time-domain link. State them as limitations.
