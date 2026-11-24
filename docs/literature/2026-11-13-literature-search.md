<!-- Literature search run with Codex (web search) on 2026-11-13.
     Checked by Claude: all 25 DOIs / arXiv IDs resolve and the titles match exactly;
     the thesis and 3GPP links open. Technical claims about each paper are Codex's
     reading and have NOT been checked against the full texts. -->

# Resource-selective jamming detection/localization in the NR uplink

Search cutoff: **13 November 2026**. The main search covered 2018–2026, with priority to 2023–2026 and arXiv; Clancy (2011) and the explicitly requested Shahriar tutorial (2015) are historical exceptions. `papers.json` contains the full author lists and bibliographic records. An arXiv DOI identifies a preprint, not peer-reviewed publication.

**Main finding:** the specific fusion of received-power, pilot delay-subspace residual and decision-directed EVM maps for dense NR uplink localization was not found in the inspected literature. Several major ingredients and much of the broader claim already exist. The contribution should be framed around this particular fusion, its complementary behavior under selective attacks, and controlled localization results, rather than a general first claim.

## Task A — proposal reference audit

“Yes” verifies bibliographic identity, not every technical claim. For publisher sites displaying robot challenges, a DOI was opened with `Accept: application/vnd.citationstyles.csl+json`; the DOI resolver returned matching publisher-deposited Crossref metadata. This counts as opening and matching the DOI record. Full-text access is stated separately. Standards, software and the dissertation were checked against their official issuing sources. No DOI, issue number or PDF URL was inferred from a naming pattern.

| Ref. | Verified? | Exact citation / corrections and description check |
|---|---|---|
| 1 | Yes | **3GPP TS 38.211, “NR; Physical channels and modulation,” V19.5.0, Release 19, September 2026**; also ETSI TS 138 211 V19.5.0. Latest published branch in the [3GPP portal](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3213); [official full PDF](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/19.05.00_60/ts_138211v190500p.pdf). Maintained Release 18 is V18.10.0, Release 17 V17.11.0. DOI: none. Description matches a normative waveform source, not jamming-performance evidence. |
| 2 | Yes, metadata | **T. Charles Clancy**, ICC 2011, **pp. 1–5**; [DOI 10.1109/ICC.2011.5962467](https://doi.org/10.1109/ICC.2011.5962467). Title/year/venue match. **Original full text and OA copy not obtained.** The 2 dB / 7.5 dB wording is reproduced in indexed abstracts with **BER 0.4, QPSK, pilot density 1/8**; it was not independently verified from the original primary text. Treat the proposal's description as conditionally supported, not a universal NR gain. |
| 3 | Yes | **Cem Örnek and Mesut Kartal**, *Electronics* **12(24), 4948 (2023)**; [DOI / publisher](https://doi.org/10.3390/electronics12244948). Citation matches. Description broadly matches RB-resolved EVM and sensitivity comparisons, but it uses **maximum reference-based EVM per RB**, not the proposal's noise-normalized mean decision-directed distance. See C3 for the decision-rule limitation. |
| 4 | Yes, metadata | **Cem Örnek and Mesut Kartal**, *2024 IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom)*, **pp. 308–311**; [DOI 10.1109/BlackSeaCom61746.2024.10646255](https://doi.org/10.1109/BlackSeaCom61746.2024.10646255). Citation and Symbol-RB-EVM time-frequency representation match. The claim that this paper actually trains/evaluates ML classifiers is **not established by the accessible abstract**; distinguish proposing ML-suitable features from demonstrating classifiers. No OA conference PDF verified; the author's thesis provides related full-text method detail. |
| 5 | Yes, metadata | **Amir Mehrabian and Georges Kaddoum**, *IEEE Communications Letters* **30, 2168–2172 (2026)**; [DOI 10.1109/LCOMM.2026.3697967](https://doi.org/10.1109/LCOMM.2026.3697967). **The future-looking citation is real and its volume/pages match.** The accessible abstract supports pilot-orthogonal projection and rank-one jamming detectors. No OA full text found; exact statistics, threshold calibration and full waveform assumptions remain unverified. Silence during the observed pilots implies blindness to a subsequent data-only attack under this observation model; that is a structural inference. |
| 6 | Yes | **Neagin Neasamoni Santhi, Davide Villa, Michele Polese, Tommaso Melodia**, “InterfO-RAN: Real-Time In-band Cellular Uplink Interference Detection with GPU-Accelerated dApps,” *MobiHoc ’25*, **pp. 71–80**; [DOI 10.1145/3704413.3764454](https://doi.org/10.1145/3704413.3764454), [OA full text](https://arxiv.org/html/2507.23177v2). Description matches: CNN, PUSCH I/Q plus RAN features, binary interference indication, reported accuracy exceeding 91% and latency under 650 μs. These are its GPU deployment results, not a comparable CPU benchmark. |
| 7 | Yes | **Guillem Reus-Muns, Pratheek S. Upadhyaya, Utku Demir, Nathan Stephenson, Nasim Soltani, Vijay K. Shah, Kaushik R. Chowdhury**, “SenseORAN: O-RAN-Based Radar Detection in the CBRS Band,” *IEEE JSAC* **42(2), 326–338 (2024)**; [DOI 10.1109/JSAC.2023.3336152](https://doi.org/10.1109/JSAC.2023.3336152), [author PDF](https://genesys-lab.org/papers/SenseORAN-JSAC.pdf). Citation matches; online publication was in 2023, issue year 2024. YOLO on uplink spectrograms matches, specifically radar sensing overlapping LTE rather than Gaussian NR DMRS/data attacks. |
| 8 | Yes | **Marcele O. K. Mendonça, Paulo S. R. Diniz, Javier Maroto Morales, Pascal Frossard**, *IEEE Open Journal of Signal Processing* **5, 1031–1041 (2024)**; [DOI 10.1109/OJSP.2024.3453176](https://doi.org/10.1109/OJSP.2024.3453176), [OA PDF](https://orbilu.uni.lu/bitstream/10993/63459/1/Adversarial_Training_for_Jamming-Robust_Channel_Estimation_in_OFDM_Systems.pdf). Citation/title match; no bracketed description supplied. Full text concerns pilot attacks and robust channel estimation, not jammer-support segmentation. |
| 9 | Yes, corrected citation | Current package **Sionna 2.2.0** is verified. Use the official **software title “Sionna”**, version 2.2.0, and its recommended author/year fields, below. The original paper is **“Sionna: An Open-Source Library for Next-Generation Physical Layer Research,” arXiv:2203.11854 (2022)**, [DOI 10.48550/arXiv.2203.11854](https://doi.org/10.48550/arXiv.2203.11854). The proposal's descriptive title is not the paper's exact title. Current PHY/SYS are PyTorch; the original paper describes TensorFlow. |
| 10 | Yes, metadata | **Rosolino Alaimo, Ilenia Tinnirello, Domenico Garlisi**, *2025 IEEE International Conference on Omni-layer Intelligent Systems (COINS)*, **pp. 1–6**; [DOI 10.1109/COINS65080.2025.11125746](https://doi.org/10.1109/COINS65080.2025.11125746), [primary institutional abstract](https://iris.unipa.it/handle/10447/690723). Citation matches; no bracketed description supplied. It is directly relevant to leaked-DCI selective PUSCH noise injection. Repository PDF is restricted; OA full text not verified. |
| 11 | Yes | **Gian Marti and Christoph Studer**, ICC 2022, **pp. 1336–1342**; [DOI 10.1109/ICC45855.2022.9838542](https://doi.org/10.1109/ICC45855.2022.9838542), [OA PDF](https://arxiv.org/pdf/2201.08778). Description matches the data-only jammer motivation; it also studies pilot-only, barrage and sparse attacks. MAED performs joint mitigation/channel estimation/data detection in frequency-flat massive MU-MIMO, not per-cell OFDM support classification. |
| 12 | Yes, issuing repository | **Cem Örnek**, *A robust framework covering measures developed using EVM metric against jamming attacks in next-generation communication systems*, Ph.D. dissertation, **İstanbul Technical University, 2024**, advisor Mesut Kartal; [permanent record](https://hdl.handle.net/11527/25720), [OA PDF](https://polen.itu.edu.tr/server/api/core/bitstreams/e49c7292-5798-4c28-9f9e-3a7c3dfafe52/content). Citation matches; DOI not found. No bracketed description supplied. It contains EVM-vs-RB, Symbol-RB-EVM, contours and a separate TDOA-based geographic-localization component; do not confuse geographic and resource-grid localization. |

For reference 9, the [versioned official README](https://raw.githubusercontent.com/NVlabs/sionna/v2.2.0/README.md) requests:

```bibtex
@software{sionna,
  title = {Sionna},
  author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal
    and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo
    and Vem, Avinash and Keller, Alexander},
  note = {https://nvlabs.github.io/sionna/},
  year = {2022},
  version = {2.2.0}
}
```

The release is from 2026; **2022 is the project's recommended software-citation year**, not the release date. The original paper has a different seven-author list, including Nikolaus Binder, preserved separately in `papers.json`. Cite the software version for reproducibility and optionally the original paper for history. [Official release](https://github.com/NVlabs/sionna/releases/tag/v2.2.0), [PyTorch migration](https://github.com/NVlabs/sionna/releases/tag/v2.0.0).

## Task B — closest prior work, ranked

“Direct competitor” below means a close competitor to the localization task or feature representation; it does not mean that every proposed component has already been combined. These comparisons distinguish an attack's actual time-frequency support from the downstream cells whose decoding/channel estimates are damaged.

| Rank | Paper / proximity | Overlap and remaining difference |
|---|---|---|
| 1 | **Örnek & Kartal 2024; Örnek thesis 2024 — direct competitor** | Symbol-RB-EVM already constructs time-frequency information at essentially the proposed output granularity; the thesis stacks EVM-vs-RB across OFDM symbols and classifies threshold-derived time/frequency occupancy features with LDA/SVM/RF/KNN. The proposed additions are pilot-residual/power fusion, decision-directed noise normalization and a dense support predictor; the conference paper's classifier experiments still need full-text checking. [Conference record](https://doi.org/10.1109/BlackSeaCom61746.2024.10646255), [thesis, sections 3.2.4–3.3](https://hdl.handle.net/11527/25720). |
| 2 | **She et al., BBD-JCED, 2026 preprint — direct competitor** | A U-Net already predicts a jammer-contaminated STFT-bin mask before OFDM recovery; pilot LS estimates and joint channel/data recovery are used later. Its mask estimator takes real/imaginary STFT inputs, rather than the proposed pilot-residual and DD-EVM maps, and it is not the same selective PUSCH PRB-by-symbol evaluation. [Full text, section III-B, Eq. (6), loss (25)](https://arxiv.org/html/2606.28778v1). |
| 3 | **Ghosh et al., SPARC, 2024 — direct competitor for affected-PRB localization** | CNN/YOLOv8 spectrogram boxes describe interference in time and frequency and are mapped to PRBs for resource control. Dense symbolwise support and the proposed receiver-feature fusion differ from box-based sensing; identifying interfered PRBs itself is established. [Main publication](https://doi.org/10.1145/3696405), [author project](https://wcsng.ucsd.edu/sparc/). |
| 4 | **Santhi et al., InterfO-RAN, 2025 — partial overlap, strongest NR fusion precedent** | Fuses a PUSCH I/Q grid with RSSI, RSRP, SINR, MCS and code-block decoding statistics at the gNB. Its output is binary per processed slot, so the distinction is dense localization and the particular physics-based features, not the first fusion of receiver measurements. [sections 3–4](https://arxiv.org/html/2507.23177v2). |
| 5 | **Örnek & Kartal 2023 and 2022 — partial overlap / essential EVM baselines** | RB-resolved EVM and jammed-RB threshold decisions predate the project. Their reference-based maximum EVM should be compared explicitly with the proposed decision-directed, noise-normalized score; neither establishes the proposed three-map fusion. [2023 publisher](https://doi.org/10.3390/electronics12244948), [2022 precursor](https://doi.org/10.1109/MENACOMM57252.2022.9998310). |
| 6 | **Schösser et al., OFDMA benchmark, 2026 preprint — partial overlap** | Five jammer classes include a pilot-aware jammer; spectrograms, allocation maps and global/local strength metadata are provided. The demonstrated task is sample-level anomaly detection; designated pilot positions are filled with data symbols, so this does not implement a DMRS-residual/EVM receiver detector. The inspected revision is **9 November 2026**. [sections III-C, III-E, IV–V](https://arxiv.org/html/2606.02102v2). |
| 7 | **Šabanović et al., 2024 — partial overlap** | U-Net segments measured 5G PSD waterfalls and compares them with expected occupancy masks to find anomalies. It is external distributed SDR sensing with band-level context, not native PUSCH receiver features or selectively jammed pilot/data support. [Author full text](https://comnets.etit.tu-dortmund.de/storages/cni-etit/r/Research/Publications/2024/Sabanovic_2024_MandN/Sabanovic_MandN2024_AuthorsVersion.pdf). |
| 8 | **Reus-Muns et al., SenseORAN, 2024 — partial overlap** | YOLO-based O-RAN radar detection demonstrates time-frequency object detection from uplink spectrograms in the presence of cellular signals. Its target is radar pulses and CBRS coexistence, without the proposed Gaussian pilot/data jammer labels or receiver-residual fusion. [Author full text](https://genesys-lab.org/papers/SenseORAN-JSAC.pdf). |
| 9 | **Wang et al., 2023 — partial overlap** | Detects PUSCH DMRS spoofing using changes in spatial channel-estimate sparsity under CDL-D fading. Observations across 4 or 16 RBs are pooled into one decision; it uses multiple antennas and does not identify a symbol-by-PRB attack mask or data-only attacks. [sections III–IV](https://arxiv.org/pdf/2212.01356). |
| 10 | **Mehrabian & Kaddoum, 2026 — partial overlap / pilot-detector method source** | Pilot-orthogonal projection suppresses the legitimate user's component and exposes rank-one jammer structure. The accessible description supports a binary pilot-observation detector; no dense OFDM map or pilot/data-feature fusion was established, and full-text access is still a material gap. [DOI](https://doi.org/10.1109/LCOMM.2026.3697967). |
| 11 | **Altun et al., 2022 — partial overlap** | A variance detector identifies jammed subcarriers using the dynamically empty tones of OFDM with index modulation under full-/partial-band Gaussian reactive attacks. This is relevant frequency localization, but changes the waveform and does not fuse native NR pilot/data measurements or produce the proposed dense grid. [Publisher](https://www.sciencedirect.com/science/article/pii/S1874490722001860). |
| 12 | **Naseri et al., 2024 — partial overlap / architecture background** | An OFDM-aligned time-frequency transform and U-Net separate co-channel RF mixtures. The objective is signal reconstruction, not selective-jammer support labels; no pilot-residual/DD-EVM fusion is used. [Full text](https://rfchallenge.mit.edu/wp-content/uploads/2024/02/UNET_ICASSP2024-1.pdf). |

Other relevant receiver and attack precedents are included in `papers.json`:

- **Marti & Studer 2022 — partial overlap:** jointly mitigates pilot-only and data-only smart jammers using spatial structure, without dense support labels. **Mendonça et al. 2024 — partial overlap:** robust OFDM pilot-based channel estimation, without attack localization. [MAED](https://arxiv.org/pdf/2201.08778), [adversarial training](https://orbilu.uni.lu/handle/10993/63459).
- **Hoffmann et al. 2026 — partial overlap / baseline:** moving-window BLER detection and MCS control, without frequency resolution. **Gopal & Rouil, CORA, 3 November 2026 — partial overlap:** combines KPI deviations with learned causal detector adaptation, but the detection unit is the cell/report stream, not the resource grid. [BLER full text](https://arxiv.org/html/2609.18499v1), [CORA full text](https://arxiv.org/html/2611.04330).
- **Alaimo et al. 2025 and 2026 — threat-model background:** real DCI-guided, time-frequency selective uplink noise injection; the 2026 STORM-RJ preprint studies response/tuning latency. These support the scheduling-aware robustness scenario rather than the detector's novelty. [COINS record](https://iris.unipa.it/handle/10447/690723), [STORM-RJ](https://arxiv.org/abs/2606.14465).
- **Lichtman et al. 2018; Arjoune & Faruque 2020; Shahriar et al. 2015 — background:** protocol-aware attacks on NR/LTE and OFDM, including reference signals and uplink control/access channels. They motivate resource selectivity and power accounting, not the proposed learned localization. [NR assessment](https://arxiv.org/abs/1803.03845), [NR review](https://arxiv.org/abs/2009.05531), [OFDM tutorial](https://doi.org/10.1109/COMST.2014.2349883).

**Explicit verdict:** no inspected paper was verified to do the complete combination **dense PRB × OFDM-symbol localization of pilot-targeted versus data-targeted NR uplink Gaussian jamming using fused power, pilot delay-subspace residual and data DD-EVM maps**. This is a bounded-search result, not proof that no such paper exists. EVM-based time-frequency localization, U-Net jammer masks, affected-PRB detection and NR receiver-feature fusion already exist separately. The original broad “no prior work combines…” sentence is too easy to read more broadly than the evidence supports.

A defensible replacement is: *“We investigate whether fusing received-power, pilot delay-subspace residual and decision-directed EVM maps improves PRB-by-symbol localization of resource-selective Gaussian jamming in a SISO NR uplink. To our knowledge, this specific feature combination and pilot-/data-selective evaluation have not been reported in the literature we reviewed.”* Establish the contribution through single-/two-/three-feature ablations, fixed false-alarm calibration and generalization across attack/channel conditions. Exact classifier experiments in the 2024 EVM conference paper and the closed 2026 projection paper remain the highest-priority full-text gaps.

## Task C — simulator decisions supported by the literature

### C1. Define both local and total jammer strength

There is no single interchangeable “JSR.” Lichtman et al., section IV, distinguish received **J/S on attacked channel REs** from **J/S averaged over a whole frame**; the latter accounts for resource sparsity. Marti & Studer, Eqs. (25)–(26), separately define total received jammer energy relative to an average UE, and active-phase power, related by the jammer duty cycle. [NR power accounting](https://arxiv.org/pdf/1803.03845), [MAED definitions](https://arxiv.org/pdf/2201.08778).

For this simulator, define the reference allocation set \(\mathcal A\), attacked subset \(\mathcal J\), and desired received RE power \(P_S\). With uniform desired power and attack occupancy \(f=|\mathcal J|/|\mathcal A|\):

\[
\mathrm{JSR}_{\rm local}=P_{J,\rm on}/P_S,\qquad
\mathrm{JSR}_{\rm total}=f\,\mathrm{JSR}_{\rm local},
\]
\[
\mathrm{JSR}_{\rm total,dB}=\mathrm{JSR}_{\rm local,dB}+10\log_{10} f.
\]

These equations are a project-specific specialization of that accounting, not a quoted definition from one paper. With nonuniform fading/pilot boosting, compute the actual ratios from sums of desired/jammer received energy over explicitly named sets rather than using only an RE-count correction. Report both equal-on-RE-power and equal-total-energy comparisons; a pilot jammer gains concentrated power under the latter constraint. JSR and SJR have opposite dB signs. The 2026 OFDMA benchmark additionally distinguishes **global SJR** from **local jammer/(signal + noise)**; do not relabel its JSNR as JSR. [Benchmark section IV-B3, Eq. (1)](https://arxiv.org/html/2606.02102v2).

Published values are heterogeneous, not a standard range: MAED tests **25 dB total-energy Gaussian jamming** and **0 dB active-phase QPSK jamming**; Wang et al. use **0 dB received JSR** for DMRS spoofing. Örnek & Kartal compare **SJR 0 and +20 dB**, equivalent in sign to JSR 0 and −20 dB, although their exact selective-resource normalization is not established here. For this detection project, a starting local-JSR sweep of **−20 to +20 dB**, extending where curves have not reached their floor/ceiling, is a **recommended experimental choice**, not a claimed literature consensus. [MAED section IV-C–D](https://arxiv.org/pdf/2201.08778), [DMRS detector section IV](https://arxiv.org/pdf/2212.01356), [EVM study](https://doi.org/10.3390/electronics12244948).

### C2. Model the jammer path explicitly; do not infer it from “additive”

| Study | What is actually modeled | Implication |
|---|---|---|
| Marti & Studer 2022 | \(Y=HS+jw^T+N\); desired and jammer channels are i.i.d. Rayleigh, frequency-flat block fading. | It has a separate jammer-to-BS channel despite an additive received-signal equation. [section II, section IV-A](https://arxiv.org/pdf/2201.08778). |
| Wang et al. 2023 | 3GPP **CDL-D**, spatial channel structure, separately placed legitimate UE and attacker. | Detection conclusions depend on array/propagation diversity; not a SISO TDL result. [section IV](https://arxiv.org/pdf/2212.01356). |
| Örnek & Kartal 2023 | Desired-link baseline is **CDL-C**, urban macro NLOS, with MMSE equalization; received disturbance is written as jammer plus noise. | A separately faded jammer path is not established by that additive notation alone; independent jammer-channel settings could not be verified. [section 2 and Table 1](https://doi.org/10.3390/electronics12244948). |
| Mendonça et al. 2024 | Conventional perturbation/direct NN-input invasion are distinguished from an eavesdropping-assisted transmitted jammer. In the latter, Eqs. (12)–(15) pass the jammer through the modeled channel operator, with a correlated/proximity assumption rather than a general independently drawn jammer TDL channel. Simulations use **ITU Pedestrian A**. | Do not summarize every attack as independent fading, or every perturbation as over-the-air jamming. [section III-B–D, section V-A](https://orbilu.uni.lu/bitstream/10993/63459/1/Adversarial_Training_for_Jamming-Robust_Channel_Estimation_in_OFDM_Systems.pdf). |
| Alaimo et al. 2025/2026 | SDR/testbed RF transmission with practical synchronization and tuning constraints. | Physical propagation is present, but the accessible 2025 abstract does not establish a named synthetic fading law. [2025 record](https://iris.unipa.it/handle/10447/690723), [2026 preprint](https://arxiv.org/abs/2606.14465). |

**Recommended project model (an implementation choice):** draw separate UE–gNB and jammer–gNB TDL realizations, then form \(y_{k,l}=h_{S,k,l}x_{k,l}+h_{J,k,l}j_{k,l}+n_{k,l}\). Under sufficient CP, slot/grid alignment and no CFO, frequency-domain channel multiplication is an OFDM-channel implementation, not a channel-free shortcut. Setting \(h_J=1\) and injecting noise directly is a useful idealized ablation; label it explicitly. State whether strength is set before fading, on ensemble received power, or by rescaling each realization, since the last choice suppresses natural power variations. [OFDM channel reference](https://doi.org/10.1109/COMST.2014.2349883).

### C3. EVM-per-RB and Symbol-RB-EVM

The 2023 paper's Eqs. (13)–(19) distinguish individual-symbol, RMS and maximum EVM. Writing the same definition with explicit reference/observed notation:

\[
e_n^2=|z_n-x_n^{\rm ref}|^2,
\quad P_{\rm ref}=\frac{1}{N}\sum_{n=1}^{N}|x_n^{\rm ref}|^2,
\quad \mathrm{EVM}_n=\sqrt{e_n^2/P_{\rm ref}},
\]
\[
\mathrm{EVM}_{\max}=\max_n\mathrm{EVM}_n,
\quad \mathrm{EVM}_{\rm RMS}=\sqrt{\frac{1}{N}\sum_n\mathrm{EVM}_n^2}.
\]

Multiply by 100 for percent. **The selected EVM-vs-RB feature is the maximum EVM among the IQ symbols demodulated from that RB**, not RMS. The stated decision is an upper-threshold exceedance for the RB. No reproducible universal numeric threshold or specified 1% false-alarm calibration was found in the examined method prose. Native reference symbols are discussed; the laboratory also transfers correct transmitted IQ to a PC. It does not establish nearest-constellation hard decisions as the reference. [2023 paper](https://doi.org/10.3390/electronics12244948).

The thesis, **section 2.3, printed p. 31**, constructs the **Symbol-RB-EVM matrix** by accumulating EVM-vs-RB measurements across individual OFDM symbols; **section 2.4, p. 37**, discusses reference-symbol and threshold requirements. **section 3.2.4, pp. 83–85**, reports classifiers using threshold-derived occupancy features, but does not supply a reproducible threshold/dataset protocol in the examined section. The text does not fully resolve reference availability at every symbol with sparse DMRS, so a practical receiver must define this explicitly. [Thesis, Eqs. (2.17)–(2.19), sections 2.3–2.4 and 3.2.4](https://polen.itu.edu.tr/server/api/core/bitstreams/e49c7292-5798-4c28-9f9e-3a7c3dfafe52/content).

Consequently, label a nearest-constellation/noise-normalized baseline **“DD-EVM threshold, inspired by Örnek & Kartal”**, unless their exact reference-based maximum feature is reproduced as another baseline. For the project score, explicitly choose mean versus maximum squared distance and divide by **post-equalization effective noise variance** in the same symbol domain, including equalizer gain. Comparing equalized distances with unscaled pre-equalizer noise is inconsistent. The latter point is an algebraic implementation requirement, not a claim about the authors' detector.

### C4. Pilot projection: verified concept, unavailable exact published detector

The accessible Mehrabian–Kaddoum abstract describes projecting the pilot observation onto the complement of the known pilot subspace and using the resulting rank-one jammer structure. **Its exact test-statistic formula and threshold-setting procedure could not be verified.** No legitimate OA full text was located. Do not assign it an invented eigenvalue ratio, GLRT equation or nominal threshold. [DOI](https://doi.org/10.1109/LCOMM.2026.3697967).

A generic flat-channel pilot observation \(Y=h p^H+J+N\) permits \(R=YQ_p\), with \(Q_p=I-pp^H/(p^Hp)\), removing the desired rank-one pilot term. This is the projection principle, **not a reproduction of the paper's detector**. The proposed OFDM channel is frequency selective, so projecting only against one pilot sequence would generally fail to remove it.

For an implementable **project-derived OFDM residual-energy baseline**, let \(\hat h_p=F_{p,L}a+v\) collect LS pilot estimates, with clean-noise covariance \(C_v\). Whiten and project:

\[
z=C_v^{-1/2}\hat h_p,\quad A=C_v^{-1/2}F_{p,L},\quad
P_\perp=I-AA^\dagger,\quad T=\|P_\perp z\|^2.
\]

If the assumed tap subspace is exact, noise covariance is known and noise is circular complex Gaussian, then \(2T\sim\chi^2_{2d}\), \(d=N_p-\operatorname{rank}(A)>0\), under no jamming. An ideal **whole-pilot-block** 1% threshold is \(\tau=\tfrac12F^{-1}_{\chi^2_{2d}}(0.99)\). This distribution follows directly from the rank of the orthogonal projector; it is our derivation, not an attributed Mehrabian equation. With estimated noise, fractional-delay/subspace mismatch or local PRB energy slices, calibrate empirically instead of assuming the same chi-square law.

Residuals are available only at observed pilots. A pilot-silent jammer is unobservable to this baseline, and interference inside the assumed channel subspace may also be invisible. Projection across many PRBs can spread the residual beyond the original attacked tones; specify the projection window and local score rather than equating projected support with RF support.

### C5. Calibrate 1% per-cell false alarms on held-out clean data

An empirical null-distribution threshold is supported by the DMRS detector's use of a simulated clean similarity CDF; its chosen threshold **0.92 is for its own statistic**, not an NR-wide constant. [Wang et al., section III](https://arxiv.org/pdf/2212.01356). No 3GPP-prescribed 1% per-grid-cell jamming threshold was found.

For this project, fit \(\tau=\widehat F_{T\mid H_0}^{-1}(0.99)\) on **jammer-free calibration slots excluded from training and test**, then freeze it. For the network use the same calibration on its probability score. A single threshold gives an aggregate nominal 1% rate under that calibration mixture; use documented conditional thresholds or report stratified rates if DMRS/data cells, SNRs, masks or allocations have different null distributions. At evaluation report actual clean-test false alarms and detection probability, not just the requested operating point.

Count valid grid cells in the denominator and state how inactive/undefined features are masked. The pilot-only baseline has no observation on nonpilot cells: report this limitation explicitly rather than calling missing values successful negative decisions. A 1% per-cell probability is not a 1% probability of any false alarm in a slot; correlated cells also do not supply independent confidence-interval trials. Resample at least at the independent-slot/channel-realization level. These are evaluation-design recommendations, not a reproduced published CFAR protocol.

The optional BLER baseline has a concrete source: \(\mathrm{BLER}_{avg}=N^{-1}\sum_i\mathbb 1[\mathrm{TB}_i=\mathrm{NACK}]\), alarm when it exceeds \(\beta\). Tune \(N,\beta\) using clean calibration. Its transport-block/time-window output cannot supply genuine per-cell frequency localization, so report its slot/window metric separately. [Hoffmann et al., section II-B, Eq. (1)](https://arxiv.org/html/2609.18499v1).

### Two labeling choices to settle before simulation

1. **DMRS REs versus the whole DMRS-bearing symbol:** type-1 DMRS is comb-mapped. In a one-port, one-CDM-group example, it occupies six subcarriers per PRB in that symbol; attacking the whole symbol is a different support/power budget. Likewise “data symbols only” differs from “all data REs,” since the DMRS-bearing symbol may also contain data or reserved REs. Derive masks from the configured resource mapper and name the attack precisely. [TS 38.211, section 6.4.1.1.3](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/19.05.00_60/ts_138211v190500p.pdf).
2. **Direct support versus propagated damage:** corrupted pilots may elevate EVM on later unjammed data. The ground truth should identify injected jammer support if localization is the objective; an elevated receiver-error map is evidence, not itself ground truth. Report nonjammed-cell false positives inside attacked slots separately from clean-slot false alarms. This follows from the channel-estimation attack mechanism and is an experimental-design inference. [Pilot attack mechanism](https://orbilu.uni.lu/handle/10993/63459).

## Searched but not found / unresolved access

These are bounded negative results, not claims of nonexistence. General web discovery included publisher/arXiv indexing, author and university repositories, and citation chasing. IEEE/ACM links, arXiv pages/HTML/PDFs, ITU Polen and research records, ÉTS records, EPFL/ORBilu, 3GPP/ETSI, and NVIDIA official sources were opened as applicable. DOI metadata was independently checked through the resolver/Crossref. Google Scholar links appeared in records; a standalone exhaustive Scholar, Scopus or Web of Science search was **not** completed.

| Query / place | What was not found |
|---|---|
| `"interference" "pilot" "EVM" "segmentation"`; `"resource grid" "interference detection" "deep learning"`; `5G resource grid interference localization segmentation U-Net jammer time frequency PRB OFDM deep learning 2024 2025` | No verified paper with the exact power + pilot-delay-residual + DD-EVM localization fusion. Many broader spectrogram/classification papers did appear. |
| `OFDM "pilot residual" "jamming" detection`; `OFDM jamming detection "LLR" "resource"` | No verified dense NR pilot/data-localization competitor using those exact receiver features. This does not exclude differently named residual or reliability methods. |
| `"SenseORAN" site:arxiv.org`; `"SPARC: Spatio-Temporal Adaptive Resource Control" arxiv`; `"A U-Net Architecture" site:arxiv.org` | No arXiv copy verified for those records. Author/project PDFs and DOI metadata were used instead. |
| Exact Mehrabian–Kaddoum title, DOI, ÉTS repository, arXiv discovery, OpenAlex/Unpaywall checks | No OA full text; exact published statistic, threshold and complete simulation assumptions unresolved. Bibliographic record is verified. |
| Exact BlackSeaCom EVM title / DOI, IEEE and ITU publication record | No OA conference PDF. Related thesis full text, including classifier experiments, was obtained; those experiments' presence in that conference paper remains unverified. |
| Exact Clancy title with `filetype:pdf`, IEEE, author-publication discovery, DOI/OA lookup | No accessible original OA PDF. Do not treat later papers' bibliographies or repeated abstract snippets as a full-text check. |
| `"Exploiting DCI Leakage" "pdf"`, publisher and Palermo repository | Repository lists a restricted PDF; no public full-text link verified. A related 2026 arXiv preprint was found and is separately listed. |
| `"jamming" "PUCCH" "PRACH" detection 2024 2025`; `5G uplink "jamming detection" "SINR" "BLER" 2025 2026` | Surveys and KPI-level detectors were found; no inspected control/access-channel result established the proposed dense PUSCH pilot/data feature fusion. This was not an exhaustive per-channel systematic review. |

Additional discovery terms included `"interference" "semantic segmentation" "5G"`, `"interference localization" "U-Net" wireless`, `"interference" "YOLO" "O-RAN"`, `"jamming" "channel estimation" "U-Net"`, `"SPARC" "YOLOv8" PRB`, and exact-title searches for each of the 12 proposal references. Generic spectrum-segmentation/reconstruction leads were screened but omitted when farther from the claim than the shortlisted papers. The bibliography is deliberately below the 35-entry cap.
