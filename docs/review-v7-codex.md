# Review v7 by Codex: does the paper need a fundamental change?

24 November 2026. Independent review by the Codex CLI, given the repo and the v7 drafts but not
our conclusions. This is model output. Its repo facts were checked against the code and results,
and every DOI below resolves. The summary and the decision are in [plan-v7.md](plan-v7.md).

- Labels: CONFIRMED = checked the cited source; for repository results, this means checked the recorded result, not rerun the simulation. SUSPECTED = judgment, hypothesis, novelty assessment, or effort estimate.
- Scope: no files edited, committed, or pushed. Raw datasets, checkpoints and result CSVs are absent from this checkout; README.md:76–77 documents that exclusion.

- SUSPECTED — Verdict: I would recommend weak reject for A and reject/weak reject for C at a demanding mainstream IEEE communications conference.
- SUSPECTED — You need a different central claim and a focused experiment, not a replacement simulator or a more complicated network. Sharper writing alone would improve readability but would not resolve the novelty problem.

- CONFIRMED — The old review is partly obsolete: LMMSE-only controls, soft block-test maps, matched false-alarm comparisons and estimated-noise recovery now exist. Do not repeat those as missing experiments. [docs/results.md sections 17.9, 18.1–18.4](results.md)
- CONFIRMED — Overlength is also no longer the main problem: the supplied compiled A and C PDFs contain four and five pages, respectively. Sources: paper/drafts/v7/draft-A.pdf; paper/drafts/v7/draft-C.pdf.
- CONFIRMED — A’s headline does not establish the learned map’s necessity: under nominal data jamming, both the soft block-test map and soft U-Net give 0.04% BLER. The distinctive nominal result is partial-band jamming: 7.7% versus 0.02%. [docs/results.md section 17.9](results.md)
- CONFIRMED — C’s residual improves weak-pilot Pd by 0.179, but overall F1 by only 0.006. The accepted context also acknowledges little corresponding BLER improvement. [docs/results.md section 17.1](results.md)
- CONFIRMED — Fine temporal resolution adds little for the tested nominal and pulsed patterns. Coarsening the true map adds approximately 52–57 percentage points of BLER under hopping. Yet v7 explicitly leaves hopping out. [docs/results.md section 18.3](results.md); paper/drafts/v7/draft.tex:6.
- SUSPECTED — That is the central missed opportunity: the drafts emphasize the weakest novelty arguments while omitting the experiment that most clearly motivates a different receiver representation.

- CONFIRMED — Broad priority claims are unavailable: [Li et al. [1]](https://www.merl.com/publications/docs/TR2012-084.pdf) already combine interference detection, pilot erasure, channel reconstruction and robust LLRs; [Örnek–Kartal [2]](https://research.itu.edu.tr/en/publications/an-advanced-evm-metric-based-jamming-signal-detection-method-for-/) already use symbol-by-RB EVM; [Mehrabian–Kaddoum [3]](https://pure.etsmtl.ca/en/publications/robust-jamming-detection-for-weak-and-strong-signal-powers-using-/) already offer pilot-subspace detection; [DeepRx [4]](https://arxiv.org/abs/2005.01494) already combines pilot/data observations for learned soft reception.
- SUSPECTED — A is the better starting draft because decoding is a meaningful outcome. C’s allocation-dependent residual law is useful supporting analysis, but too narrow to carry the paper’s significance.

- SUSPECTED — The six directions below are candidate contributions, not verified discoveries. “Higher odds” assumes the proposed result survives strong comparisons. Effort means one person’s working days, excluding substantial compute delays.

1. Keep almost everything: a bounded NR receiver integration study.

- SUSPECTED — Claim: “Receiver-derived interference maps enable reliable PUSCH decoding when used through measured interference variances, with substantially less weak-interference damage than erasure.”
- CONFIRMED precedent / SUSPECTED novelty — Closest: Li [1], Kumar [5], Ercan [6]. Soft interference-aware decoding is established; the remaining contribution would be a careful NR integration and operating-envelope study.
- CONFIRMED evidence — Hard-minus-soft penalties reach 3.9–12.2 percentage points at weak jamming; estimated-noise nominal recovery remains useful; width sweeps expose failure boundaries. docs/results.md sections 18.3–18.4, 22.
- SUSPECTED work — 3–5 days: add power-map-soft and continuously weighted classical receiver baselines; include the existing width and low-SNR evidence; reduce residual theory to supporting material.
- SUSPECTED risk/odds — Main objection: competent application engineering with limited new insight. Acceptance odds: similar. This is the cheapest defensible submission, not my recommendation for demanding reviewers.

2. When does symbol-level localization actually improve decoding?

- SUSPECTED — Claim: “Within-PRB temporal interference variation predicts when PRB-level reliability estimates fail, allowing the receiver to select the necessary estimation resolution.”
- CONFIRMED precedent / SUSPECTED novelty — Li [1] already handles partial-time interference; Kumar [5] already motivates local variance estimates. The potential contribution is a validated resolution-selection rule and quantified decoding penalty, not the existence of fine maps.
- CONFIRMED evidence — Large hopping penalties already exist; nominal/pulsed coarse-map penalties are small. docs/results.md sections 17.9, 18.3.
- CONFIRMED caution — Coarsening currently ORs data flags across a PRB, then averages excess energy over those flags. The comparison therefore changes variance pooling as well as support resolution. src/jamloc/experiments/recovery.py:268–275; src/jamloc/recovery.py:474–485.
- SUSPECTED work — 7–12 days: sweep hopping dwell time, duty cycle, coverage and several QPSK code rates; compare validation-optimized PRB-constant variances, symbol-level classical estimates and learned estimates; derive an observable coarse/fine selection rule.
- SUSPECTED risk/odds — Main objection: “Your coarse estimator is unnecessarily poor.” Acceptance odds: higher, provided the penalty survives that control.

3. Predict receiver reliability rather than jammer transmission support.

- SUSPECTED — Claim: “Separate estimates of pilot contamination and data reliability improve decoding and transfer across attack patterns better than optimizing an injection-support mask.”
- CONFIRMED precedent / SUSPECTED novelty — DeepRx [4], Ercan [6] and code-aware interference detection by Li [7] preclude generic “learning for decoding” novelty. The possible contribution is an interpretable two-output receiver model with a demonstrated segmentation-versus-decoding mismatch.
- CONFIRMED evidence — Weak-pilot localization gains coexist with little decoding gain; hard use can hurt despite correct support; near-perfect support cannot overcome broad interference. docs/results.md section 6, explicitly retaining its qualitative observation; sections 18.3, 22.
- SUSPECTED work — 12–20 days: replace binary outputs with pilot observation variance and data effective variance; train with channel-error/bit-likelihood losses; compare identical backbones and training populations against segmentation, classical estimates and a compact learned receiver.
- SUSPECTED risk/odds — Main objection: a restricted DeepRx with additional heads. Acceptance odds: higher only if the explicit decomposition gives measurable transfer or complexity benefits.
- CONFIRMED additional warning — Receiver-facing reliability diagnostics are themselves established territory; the newly located PACE-CE paper [15] explicitly uses residual and conditioning diagnostics. Its accessible abstract narrows the novelty claim further.

4. Joint noise calibration and interference estimation under contaminated references.

- SUSPECTED — Claim: “A receiver jointly estimates its noise floor and selective interference under an explicit clean-reference condition, preserving calibration and decoding when the usual DMRS-comb estimate is contaminated.”
- CONFIRMED precedent / SUSPECTED novelty — Noise-uncertainty limits and estimated-noise detector calibration are established by Tandra–Sahai [8] and Mariani et al. [9]; pilot-subspace CFAR also exists [3]. A generic ratio test or robust median would not suffice.
- CONFIRMED evidence — A −1 dB noise error produces 39.9% learned-detector false alarms; whole-allocation pilot jamming defeats estimated-noise recovery even with true support. docs/results.md sections 17.4, 18.4.
- SUSPECTED work — 12–20 days: formulate identifiability assumptions; combine comb, delay residual and reliable-data evidence; account for shared-sample dependence; test contamination fraction, noise drift and delay mismatch through the complete receiver.
- SUSPECTED risk/odds — Main objection: the noise and jammer powers are not separately identifiable under unrestricted contamination. Acceptance odds: higher if the assumptions and failure boundary are explicit; similar if this becomes another estimator ablation.

5. Partial-band pilot replay: detect consistency failure and repair it.

- SUSPECTED — Claim: “Cross-checked pilot/data channel consistency localizes and mitigates partial-band DMRS replay that remains difficult for delay-residual segmentation.”
- CONFIRMED precedent / SUSPECTED novelty — Wang et al. [11] already detect NR DMRS spoofing through spatial structure; Liu et al. [12] combine estimator differences with data-aided mitigation. The potential distinction is SISO partial-band localization with coded PUSCH recovery and no transmitter modification.
- CONFIRMED evidence — Full replay nearly preserves the residual null; spoof-trained localization reaches 0.871 on full replay but only 0.292 on partial replay. Decoder-feedback infrastructure already exists. docs/results.md sections 17.3, 21; src/jamloc/recovery.py:337–362.
- SUSPECTED work — 15–25 days: build independently checked data-derived channel estimates; prevent reuse of the same unreliable decisions as “validation”; sweep replay width, phase and strength; compare estimator-consistency baselines and report BLER.
- SUSPECTED risk/odds — Main objection: decision feedback reinforces the corrupted estimate; spatially unresolved channel ambiguities may remain. Acceptance odds: higher if partial replay is actually repaired, with greater project risk than direction 2.

6. A different framing: when “detection before damage” disappears.

- SUSPECTED — Claim: “Early-warning gains measured with fixed-MCS headroom do not persist uniformly under link adaptation, and receiver protection can remain valuable where standalone alarms are late.”
- CONFIRMED precedent / SUSPECTED novelty — Chiarello et al. [10] already study detectability versus communication damage; Rao et al. [13] already connect selective interference to link adaptation and retransmission latency. Novelty would require an adapted-PUSCH result and a useful intervention.
- CONFIRMED evidence — The minimum TDL-C warning margin changes from +8.6 dB at 15 dB SNR to −6.1 dB at 5 dB. docs/results.md sections 17.6, 18.2.
- SUSPECTED work — 6–10 days: select rates to meet the same benign BLER target; measure incremental damage, goodput and reporting delay at a common false-alarm budget; compare alarm-triggered protection with continuous soft protection.
- SUSPECTED risk/odds — Main objection: an obvious consequence of reduced headroom. Avoid defining damage as the same BLER already targeted without interference. Acceptance odds: similar; higher only with a useful quantitative rule or intervention.

- SUSPECTED — Ranking by likelihood of producing a convincing short paper with substantial reuse: 2 > 4 > 3 > 5 > 6 > 1.
- SUSPECTED — Recommendation: direction 2. It has the strongest existing effect, a clear practical question and modest implementation needs.
- SUSPECTED — First decision experiment: compare fine estimates against a well-tuned PRB-constant variance receiver on identical hopping slots. If that removes the large gap, abandon the resolution claim before investing in another draft.
- SUSPECTED — A suitable paper arc is: temporal interference variation → controlled resolution penalty → simple resolution rule → estimated-map decoding. Keep the U-Net as an implementation and the residual law as supporting analysis.

- CONFIRMED — Claims I believe are wrong or need correction:
- CONFIRMED — “Delays within τmax” does not guarantee the exact central null after eigenspace truncation. Exactness requires the channel vector in the retained span. C’s proposition states this correctly, but its contribution wording and A’s residual paragraph weaken the condition. paper/drafts/v7/draft.tex:114–115, 275–277; src/jamloc/features.py:101–117; src/jamloc/theory.py:11–14.
- CONFIRMED — The median does not “ignore” contamination below 50%; that is a breakdown-resistance property, not immunity to bias. The implemented median generally shifts under minority contamination. src/jamloc/noise.py:14–18, 55–59.
- CONFIRMED — The soft data-aided receiver does not exclude all flagged data cells. It admits sufficiently confident cells with finite interference variance and down-weights them. The exclusion explanation in docs/results.md section 22:1418–1420 contradicts src/jamloc/recovery.py:419–428, 474–485.
- CONFIRMED — “All decoding results” use four jammed PRBs is false alongside the width results quoted immediately afterward. paper/drafts/v7/draft.tex:507–510; docs/results.md section 22.
- CONFIRMED — The implemented power baseline is a median-normalized cell detector plus erasure, not a faithful reproduction of Li’s complete robust/dynamic-LLR receiver. Treat it as an adaptation, not evidence of beating that published receiver. src/jamloc/baselines.py:83–96; Li [1], sections III-A–III-C.
- CONFIRMED — An interval containing zero does not prove equivalence. “Consistent with the true-support receiver within the reported uncertainty” is defensible; guaranteed oracle performance is not. docs/results.md section 18.3.
- SUSPECTED — The width cliff is consistent with loss of usable coded information, but clean-bit counting alone is not a capacity bound. The current qualified explanation is reasonable; strengthening it into an impossibility theorem would be wrong. docs/results.md section 22.

- CONFIRMED — Search gaps and things not found:
- CONFIRMED — I did not obtain independently readable full texts of Örnek–Kartal [2], Mehrabian–Kaddoum [3] or Fadul–Reising [14]. Their accessible primary abstracts/records establish relevant overlap, not every implementation detail. docs/claims-register.md, “Evidence and checks,” reports earlier local access to the first two; those PDFs are absent here.
- SUSPECTED — I found no verified exact prior matching the proposed resolution-selection study or two-output task-trained receiver. This is a bounded search result, not proof of priority.
- CONFIRMED — I found an additional relevant publisher-indexed paper, PACE-CE [15], absent from docs/literature_audit.md and docs/references.bib. I checked its indexed abstract, not its full methods.
- CONFIRMED — I did not find saved results establishing that weak-pilot detection gains improve BLER, or that fine-map gains survive an optimized coarse-variance comparator. The recorded evidence establishes narrower comparisons. docs/results.md sections 6, 17.9, 18.3.
- CONFIRMED — The current source has unfinished citation notes. paper/drafts/v7/draft.tex:98–99.

- Full literature citations; confirmed scope is stated where access was limited:

- [1] Liangbin Li, Ramesh Annavajjala, Toshiaki Koike-Akino, Philip Orlik. “Robust Receiver Algorithms to Mitigate Partial-Band and Partial-Time Interference in LDPC-coded OFDM Systems.” MILCOM, 2012, pp. 1–6. DOI: [10.1109/MILCOM.2012.6415663](https://doi.org/10.1109/MILCOM.2012.6415663). Full author manuscript checked.
- [2] Cem Örnek, Mesut Kartal. “An Advanced EVM Metric Based Jamming Signal Detection Method for Next-Generation Communication Systems.” BlackSeaCom, 2024, pp. 308–311. DOI: [10.1109/BlackSeaCom61746.2024.10646255](https://doi.org/10.1109/BlackSeaCom61746.2024.10646255). Institutional abstract checked.
- [3] Amir Mehrabian, Georges Kaddoum. “Robust Jamming Detection for Weak and Strong Signal Powers Using Pilot Subspace Projection.” IEEE Communications Letters, vol. 30, 2026, pp. 2168–2172. DOI: [10.1109/LCOMM.2026.3697967](https://doi.org/10.1109/LCOMM.2026.3697967). Primary abstract checked.
- [4] Mikko Honkala, Dani Korpi, Janne M. J. Huttunen. “DeepRx: Fully Convolutional Deep Learning Receiver.” IEEE Transactions on Wireless Communications, vol. 20, no. 6, 2021, pp. 3925–3940. DOI: [10.1109/TWC.2021.3054520](https://doi.org/10.1109/TWC.2021.3054520). Author abstract checked.
- [5] Sumit Kumar, Florian Kaltenberger, Alejandro Ramirez, Bernhard Kloiber. “An SDR Implementation of WiFi Receiver for Mitigating Multiple Co-channel ZigBee Interferers.” EURASIP Journal on Wireless Communications and Networking, 2019, article 224. DOI: [10.1186/s13638-019-1512-3](https://doi.org/10.1186/s13638-019-1512-3). Publisher methods checked.
- [6] Furkan Ercan, Kevin Galligan, Ken R. Duffy, Muriel Médard, David Starobinski, Rabia Tuğçe Yazıcıgil. “A General Security Approach for Soft-information Decoding against Smart Bursty Jammers.” IEEE GLOBECOM Workshops, 2022, pp. 245–251. DOI: [10.1109/GCWKSHPS56602.2022.10008541](https://doi.org/10.1109/GCWKSHPS56602.2022.10008541). Author preprint checked.
- [7] Tao Li, Wai Ho Mow, Vincent K. N. Lau, Manhung Siu, Roger S. Cheng, Ross D. Murch. “Robust Joint Interference Detection and Decoding for OFDM-Based Cognitive Radio Systems With Unknown Interference.” IEEE JSAC, vol. 25, no. 3, 2007, pp. 566–575. DOI: [10.1109/JSAC.2007.070407](https://doi.org/10.1109/JSAC.2007.070407). Institutional abstract checked.
- [8] Rahul Tandra, Anant Sahai. “SNR Walls for Signal Detection.” IEEE Journal of Selected Topics in Signal Processing, vol. 2, no. 1, 2008, pp. 4–17. DOI: [10.1109/JSTSP.2007.914879](https://doi.org/10.1109/JSTSP.2007.914879). Paper abstract checked.
- [9] Andrea Mariani, Andrea Giorgetti, Marco Chiani. “Effects of Noise Power Estimation on Energy Detection for Cognitive Radio Applications.” IEEE Transactions on Communications, vol. 59, no. 12, 2011, pp. 3410–3420. DOI: [10.1109/TCOMM.2011.102011.100708](https://doi.org/10.1109/TCOMM.2011.102011.100708). Institutional abstract checked.
- [10] Leonardo Chiarello, Paolo Baracca, Karthik Upadhya, Saeed R. Khosravirad, Thorsten Wild. “Jamming Detection With Subcarrier Blanking for 5G and Beyond in Industry 4.0 Scenarios.” IEEE PIMRC, 2021. DOI: [10.1109/PIMRC50174.2021.9569600](https://doi.org/10.1109/PIMRC50174.2021.9569600). Author abstract checked.
- [11] Shao-Di Wang, Hui-Ming Wang, Chen Feng, Victor C. M. Leung. “Sequential Anomaly Detection Against Demodulation Reference Signal Spoofing in 5G NR.” IEEE Transactions on Vehicular Technology, vol. 72, no. 1, 2023, pp. 1291–1295. DOI: [10.1109/TVT.2022.3202745](https://doi.org/10.1109/TVT.2022.3202745). Author full text checked.
- [12] Delong Liu, Wei Wang, Yang Huang. “Pilot Spoofing Attack Detection and Channel Estimation for Secure Massive MIMO.” Electronics Letters, vol. 60, no. 21, 2024, e70074. DOI: [10.1049/ell2.70074](https://doi.org/10.1049/ell2.70074). Publisher methods checked.
- [13] Raghunandan M. Rao, Vuk Marojevic, Jeffrey H. Reed. “Analysis of Non-Pilot Interference on Link Adaptation and Latency in Cellular Networks.” IEEE VTC-Spring, 2019. DOI: [10.1109/VTCSpring.2019.8746307](https://doi.org/10.1109/VTCSpring.2019.8746307). Publisher/author abstracts checked.
- [14] Mohamed K. M. Fadul, Donald R. Reising. “Using Deep Learning to Combat Pilot Jamming in SISO OFDM-based Wireless Networks.” MILCOM, 2024, pp. 695–702. DOI: [10.1109/MILCOM61039.2024.10773873](https://doi.org/10.1109/MILCOM61039.2024.10773873). Citation and author institutional record checked; methods unresolved.
- [15] Peng Liu, Hao Zhang, Shizhe Tan, Pengfei Xue, Zhen Wang, Yuan An. “Identifiability-aware Pilot-domain Channel Estimation for V2X OFDM under Structured Pilot Jamming.” Vehicular Communications, vol. 62, 2026, article 101086. DOI: [10.1016/j.vehcom.2026.101086](https://doi.org/10.1016/j.vehcom.2026.101086). Publisher-indexed abstract checked; full methods unresolved.