# Related work and novelty check

November 2026. Sources: the 45 papers of the Zotero collection "CMPE 581 — 5G NR Uplink Jamming
Localization" (22 read in full), a database search (OpenAlex, 10 topic queries, 2018–2026) and a
separate web and literature search. 35 relevant works were not in Zotero. All 80 references are
in `docs/references.bib` (every DOI resolves; import the file into Zotero to add the new ones).
The per-paper audit with overlap ratings is in `docs/literature_audit.md`.

## Bottom line

None of the sources does the whole combination: a per-slot PRB × symbol map of the *attacked*
cells, estimated from receiver-internal features that include a delay-subspace DMRS residual,
and used to drive pilot exclusion and erasure in an NR receiver. But two of our four ideas are
not new on their own, and the paper must say so:

- **Localizing jamming in time and frequency is not new.** `li2012robust` (MILCOM 2012) detects
  corrupted tones per OFDM symbol with a power test at a fixed false-alarm rate;
  `ornek2024advanced` and `ornek2023securing` build symbol × resource-block EVM maps;
  `reusmuns2024senseoran` localizes interferers on spectrograms.
- **Using such a map in the receiver is not new.** `li2012robust` already erases corrupted
  pilots, interpolates the channel from the clean ones and adapts the LLRs before LDPC decoding
  (checked in its Section III). `kumar2019sdr` scales LLR reliability around localized
  interference; `rajchowski2024examination` proposes feeding resource-element exclusions to
  channel estimation as future work.

What is defensible and new in combination:

1. **Attacked cells versus damaged cells.** Our labels mark where the jammer transmitted. Under
   pilot jamming the damage (EVM) appears in data symbols that were never jammed, so EVM-based
   maps (`ornek2024advanced`) point at the wrong symbols. Showing this, and fixing it, is our
   sharpest contribution (figures `constellations`, `detector_outputs`, `feature_distributions`).
2. **The DMRS delay-subspace residual** as a per-PRB feature in a single-antenna NR receiver. The
   projection principle exists (`mehrabian2026robust`, multi-antenna, per-block decision;
   `wang2023sequential`, spatial sparsity against DMRS spoofing). Our construction (uniform delay
   profile, raw LS pilots, per-pilot noise normalization), its **exact statistics** (a weighted sum
   of exponentials that depends on neither the channel nor the SNR, so the false-alarm threshold
   is computed, not learned, and Pd and the leakage into neighbouring PRBs follow in closed form)
   and its role in detecting *weak* pilot jamming are the contribution.
3. **Fusion with matched ablations and low-false-alarm evaluation** (Pd at 1% per-cell false
   alarm, seeds, scheduling-aware attacker). Fusing features with a network alone is not new
   (`honkala2021deeprx` on NR grids, `neasamonisanthi2025interfo` for uplink interference,
   detection only, `ulbricht2024pyjama` for jamming in Sionna).
4. **Map-conditioned NR recovery** as an evaluation of the map's value: the map says *where*, the
   receiver estimates *how strong* from the flagged cells (pilot noise weighting in an LMMSE
   interpolator, interference-aware LLRs), compared with hard exclusion and erasure
   (`li2012robust`), with DMRS-only interference estimation (no map), and with the true map.
   Present it as an NR refinement of `li2012robust`, not as a new receiver idea.

**Claims to avoid:** "first jamming map", "first U-Net jamming detector", "first receiver that
uses localized interference knowledge", "a new scheduling-aware attack" (`wang2026downlink`,
`flores2023implementation`, `alaimo2025exploiting` already show such attackers).

**Suggested claim:** "We study receiver-internal feature fusion for PRB-by-symbol localization of
resource-selective jamming in the NR uplink. A delay-constrained DMRS residual complements
decision-directed EVM where pilot jamming displaces the damage into unjammed data symbols, and the
estimated map drives pilot exclusion, uncertainty-aware channel interpolation and data erasure."

## The closest works and how we differ

| Work | What it does | Our difference |
| --- | --- | --- |
| `li2012robust` (MILCOM 2012) | Per-symbol corrupted-tone detection (power test), pilot erasure, linear interpolation, robust LLRs, LDPC-coded OFDM | Learned map from three receiver features that also catches *weak* pilot jamming; NR PRB granularity; LMMSE interpolation with per-subcarrier uncertainty; evaluation with estimated vs true maps |
| `ornek2024advanced`, `ornek2023securing` (EVM per symbol × RB) | 5G *downlink* MATLAB link; per-symbol × RB EVM images under tone, chirp and pulse jamming, compared visually with spectrograms, BER and throughput. No detector: no threshold, false-alarm rate or Pd; the "ML alternative to spectrograms" is proposed, not evaluated | We turn the EVM map into a detector with a measured false-alarm rate (our EVM-threshold baseline) and show where it fails: labels mark attacked cells, and under pilot jamming EVM points at the unjammed data symbols |
| `mehrabian2026robust` (IEEE COMML 2026) | M = 16 antennas; projects the received pilot block onto the complement of the known pilot sequence, then R1 (largest singular value), R1-CFAR and a projected energy detector (PED); one binary decision per pilot interval. States that it cannot see jammers that are silent during pilots and suggests adding data-phase detectors | Single antenna: the user's channel is removed by a *delay* subspace instead of the pilot sequence, and the statistic is per PRB. Our residual threshold is the per-PRB analogue of their PED. We derive its exact null distribution and Pd (they derive R1/R1-CFAR by GLRT) and fuse it with data-phase features, which is the combination they suggest |
| `kavvousanos2026deep` (arXiv 2026) | 14k-parameter CNN cancels narrowband interference and outputs soft bits; erasure baseline | AWGN channel and multitone interference; no PRB × symbol support map, no pilot jamming or channel reconstruction |
| `wang2026downlink` (IEEE TIFS 2026) | DCI-sniffing smart PUSCH jamming and spatial suppression | Same attacker idea; we use single-antenna support inference and pilot/LLR processing instead of spatial separation |
| `kazemian2024investigating` (arXiv 2024) | Finds attacked subcarriers from loss of orthogonality, cancels the jamming | Frequency-only localization; we resolve symbols and separate pilot from data attacks |
| `wang2023sequential` (IEEE TVT 2022) | DMRS spoofing detection from spatial channel sparsity | Multi-antenna, spoofing, per-slot decision |

## What this changes in the plan

1. **Add the classic recovery baseline.** Run the recovery experiment with a `li2012robust`-style
   map (per-cell power test at 1% false alarm) and the same receiver. This shows what a *better
   map* buys, which is the real claim of the recovery section. Add erasure-only and
   interpolation-only ablations.
2. **Fix the latency claim.** At 30 kHz a slot lasts 0.5 ms, so 0.80 ms per slot on one core is
   not real time on one core. Report it honestly (two cores, or batching), and run the U-Net size
   sweep to find a smaller model below 0.5 ms in total.
3. ~~Get two full texts~~ (done, November 2026; summarized in the table above).
4. **Report the residual's limits.** With a truncated subspace the residual also contains channel
   leakage, and the projection spreads localized jamming to neighbouring PRBs. The
   generalization tests (longer delay spread than `tau_max` assumes) cover this.
5. **Terminology.** Say "PRB-by-symbol" resolution, not "resource-element" resolution.

## Related-work section for the paper (outline, about 0.6 column)

1. *Jamming of NR and network-level detection:* `pirayesh2022jamming` (survey),
   `lichtman20185g`, `flores2023implementation`, `wang2026downlink`; KPI and O-RAN detectors
   (`kilinc2022jade`, `neasamonisanthi2025interfo`, `reusmuns2024senseoran`) decide per slot or
   need many slots and do not localize.
2. *Receiver-level detection:* EVM maps (`ornek2022efficient`, `ornek2023securing`,
   `ornek2024advanced`), pilot-subspace and channel-structure detectors (`mehrabian2026robust`,
   `wang2023sequential`), learned receivers (`honkala2021deeprx`, `kavvousanos2026deep`).
   Gap: none separates attacked from damaged cells, so pilot jamming is mislocated.
3. *Mitigation with interference knowledge:* `li2012robust`, `kumar2019sdr`, robust channel
   estimation under pilot jamming (`clancy2011efficient`, `mendonca2024adversarial`). Our recovery
   evaluates how much a better map improves this classic receiver in NR.
