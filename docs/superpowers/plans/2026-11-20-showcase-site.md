# Plan: showcase site and new paper figures

Implements `docs/superpowers/specs/2026-11-20-showcase-site-design.md`. Branch `showcase-build`,
one PR. Each task ends with its tests passing and a commit.

## Phase 1: exporter (`src/jamloc/site/`)

**1. Case definitions** (`site/cases.py`)

- A `Case` dataclass:
  - name, title;
  - channel, SNR, attack, JSR;
  - jammer variant (`nominal`, `pulsed`, `hopping`, `spoof`), noise mode (`true`, `comb`);
  - simulation seed, detector seed;
  - the maps to decode (detector names), the selection rule.
- `CASES`:
  - the ten cases of the design;
  - `fig3`: the paired benign/jammed realization, TDL-C, 20 dB, 0 dB JSR;
  - `walkthrough_flip`: case 2 conditioned on standard CRC fail and protected CRC pass.
- Test: every case resolves to a valid `SimConfig` and its detectors exist in the registry.

**2. One case to one slot** (`site/export.py: simulate_case`)

- **Simulate:**
  - one batch of 64 slots on the CPU with the case's seed;
  - the jammer variant through `SimConfig` (`jammer_duty`, `jammer_hopping`, waveform `spoof`);
  - the noise mode through `noise_estimate`.
- **Detector maps:** `experiments.recovery.detector_cells` with `val_threshold`, using the models of
  the case (nominal, `atk`, `atk2`, `nest`).
- **Receivers, all on the same `y_alloc`:**
  - the standard receiver;
  - `MapAwareReceiver` with mode soft or hard, for each map to decode, the true map and
    `coarsen_to_prb(true)`.
- **Selection:**
  - by default the slot with the median U-Net hit rate on the jammed cells;
  - `walkthrough_flip` picks the first slot where the standard CRC fails and the soft one passes,
    and records how often that happens in the batch.
- Test: on a tiny case, the receivers share `y_alloc`, the selected index is in range, and the
  flip frequency lies between 0 and 1.

**3. Payload** (`site/export.py: payload`)

- **Grids (24 x 14):**
  - P, R, E in dB (from `features.extract`), with -inf mapped to null;
  - the allocation;
  - the true map;
  - per detector: scores and decisions, with an unscored mask.
- **Per PRB:** `R_b` and its analytic threshold (`theory`).
- **Per pilot:** the LS estimate and its projection (`dmrs_ls`, `delay_subspace`), complex.
- **Per receiver:**
  - `x_hat` of the allocation (re, im, as float16-rounded lists);
  - per-point |LLR| (mean over the two bits);
  - the CRC.
- **Metadata:**
  - the parameters;
  - the git commit (`git rev-parse`, or "unknown");
  - the device;
  - the threshold sources;
  - the selection rule and its statistic.
- Test: the JSON schema holds, and the size of one case stays under 250 kB.

**4. JSR sweep** (case 10)

- The same seed for every JSR, so the channel, noise and jammer waveform repeat and only the
  power changes.
- Export the per-pilot residual and `R_b` for each JSR, -15 to +9 dB in 3 dB steps.
- Test: the benign part (JSR -inf) has equal LS estimates across the sweep.

**5. CLI** (`python -m jamloc.site export [--cases ...]`)

- Writes `results_dir()/site/data/<case>.json` and `manifest.json`.
- Commit.

## Phase 2: paper figures (`jamloc.experiments.figures`)

**6. Timing figure** `timing()`

- One column, matplotlib.
- The two strips of the design, plus the caption text in `captions.json`.
- Test: the figure renders.

**7. Fig. 3** `residual_explained()`, rewritten from case `fig3`

- Top: the squared per-pilot residual with PRB dividers and a bracket.
- Bottom: `R_b` in dB with benign and jammed slots and the stepped analytic threshold.
- The leaking neighbor is labeled.

**8. Fig. 6** `detector_outputs()`, rewritten

- Left: three aligned maps of case 2 (injected, EVM flags, U-Net P+R+E flags), with the truth
  outline, unscored cells hatched, and the DMRS marked.
- Right: a dot plot of Pd at the lowest pilot JSR per detector, with intervals from
  `results/intervals/test_pd_ci.csv` (seed mean).

**9. Into the paper**

- `python -m jamloc.paper` copies the figures.
- In the Overleaf clone:
  - add the timing figure to the system model with two sentences of text;
  - update the captions of Fig. 3 and Fig. 6;
  - commit and push.

## Phase 3: site (`site/` sources, `jamloc.site.build`, `jamloc.site.deploy`)

**10. Front end** (`site/`)

- **Pages:** `index.html`, `walkthrough.html`, `explore.html`, `constellations.html`,
  `animations.html`, `results.html`.
- **Styles:** `assets/style.css`, using the paper palette and readable on a phone.
- **Scripts:**
  - `assets/js/data.js`: loads the manifest and cases on demand;
  - `grid.js`: SVG 24 x 14 grid, layers, click a cell;
  - `constellation.js`: canvas, opacity from |LLR|;
  - `stepper.js`: walkthrough;
  - `animate.js`: hopping, pulsed and the JSR sweep;
  - `results.js`: Plotly, pinned version.
- No build tools.

**11. Results data** (`site/results.py`)

- Compact JSON from:
  - `eval/test_pd_vs_jsr.csv`;
  - `margin/margin_vs_snr_ci.csv`;
  - `recovery/*_replay_paired.csv` and the pooled recovery CSVs;
  - `intervals/*matched*`.
- Each series carries its population text, and the original CSVs are copied for download.

**12. Build** (`python -m jamloc.site build`)

- Copy `site/`, the data and the results JSON/CSVs into `results_dir()/site/build/`, and add
  `.nojekyll`.
- Test: every page builds, every referenced file exists, scripts come only from pinned CDN URLs.

**13. Preview**

- `python -m jamloc.site serve` runs `http.server` on the build directory.
- The author checks the pages in a browser.

**14. Deploy** (`python -m jamloc.site deploy`)

- A git worktree on the orphan branch `gh-pages`: replace its contents with the build, commit, push.
- Enabling Pages (source: `gh-pages`, root) is a one-time action, done after the author agrees.
- Add the URL to the README.

## Phase 4: review

**15. Independent review** of the exporter and the site against the design's anti-goals:

- numbers equal to the paper's CSVs;
- populations stated;
- selection rules honest;
- no real-time claim.

Fix what it finds.

**16. Status, handoff, PR**

- Update `docs/results.md` (a short note on the site and the export command), HANDOFF and memory.
- Open the PR, with the usual "what we did / what is still missing" table.

## Order and delegation

- Phases 1 and 2 first, by hand: they decide the data the site shows.
- The front end (task 10) can be written in parallel in a separate worktree from a fixed data
  schema (task 3), then integrated.
- Deploy only after the author has seen the preview.
