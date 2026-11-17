# Plan v2: strengthening the contribution

November 2026, after the literature review (`docs/related_work.md`). The review showed that
time-frequency localization and map-aware erasure exist on their own (Li et al. 2012, Örnek &
Kartal 2024). This plan adds the pieces that make our combination defensible and measurable.

| # | Work | Why | Compute |
| --- | --- | --- | --- |
| 1 | **Theory of the DMRS residual.** Exact distribution of the per-PRB residual without jamming (a weighted sum of exponentials, independent of channel and SNR), the false-alarm threshold from it, and Pd under Gaussian jamming. A training-free "residual CFAR" detector. | Turns the residual from a heuristic feature into a detector with a provable false-alarm rate. Mehrabian & Kaddoum derive a similar statistic for multi-antenna pilots; ours is the single-antenna, frequency-domain counterpart and is per PRB. | CPU, minutes |
| 2 | **Recovery that cannot hurt.** The map says *where*; the receiver estimates *how strong* from the flagged cells and weights pilots and LLRs accordingly, instead of hard dropping and erasing. Baselines: Li et al.-style power-test map, and per-PRB interference estimation from the DMRS (what a gNB does today, without any map). | Removes the weak-jamming penalty of hard erasure (BLER 0.06 where no protection gives 0) and answers "what does the map add over standard interference estimation?". | Colab GPU |
| 3 | **Generalization.** Test the trained models on conditions they never saw: delay spread beyond `tau_max`, higher speed, a QPSK-modulated jammer. | Shows where the method breaks and why (the theory predicts the residual's channel leakage). | Colab GPU |
| 4 | **U-Net size sweep and honest latency.** Narrower networks, Pd/F1 against parameters and CPU time. A 30 kHz slot lasts 0.5 ms. | The 0.80 ms figure is above one slot on one core. | Colab GPU + Mac CPU |
| 5 | Results, figures, related work and README updated; paper draft. | | |

Order: 4 and 3 are launched on Colab first (they only need small code changes), 1 is done on the
CPU meanwhile, then 2 is implemented and run on Colab.
