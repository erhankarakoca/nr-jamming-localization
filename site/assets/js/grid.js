import { batchText, conditions, detectors, loadCase, loadManifest, slotDetails } from "./data.js";
import { affectedPRBs, comparisonReceivers, constellations, crcList } from "./constellation.js";
import { element, options, reportError } from "./nav.js";

export const FEATURES = {
  power_db: "Power P",
  residual_db: "Pilot residual R",
  evm_db: "Error vector magnitude E",
};

let nextPattern = 0;

export function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

function featureScale(values) {
  const finite = values.flat().filter(Number.isFinite);
  if (!finite.length) return null;
  return [Math.floor(Math.min(...finite)), Math.ceil(Math.max(...finite))];
}

// The parula colormap of the paper figures (plotting.PARULA), interpolated linearly.
const PARULA = [
  [0.2422, 0.1504, 0.6603], [0.2803, 0.2782, 0.9221], [0.244, 0.4358, 0.9988],
  [0.154, 0.5902, 0.9218], [0.0297, 0.7082, 0.8163], [0.1938, 0.7758, 0.6251],
  [0.5044, 0.7993, 0.348], [0.8634, 0.7406, 0.1596], [0.9763, 0.9831, 0.0538],
];

function heatColor(value, [min, max]) {
  const level = max === min ? 0.5 : Math.min(1, Math.max(0, (value - min) / (max - min)));
  const position = level * (PARULA.length - 1);
  const i = Math.min(PARULA.length - 2, Math.floor(position));
  const t = position - i;
  const rgb = PARULA[i].map((v, k) => Math.round(255 * (v + t * (PARULA[i + 1][k] - v))));
  return `rgb(${rgb.join(",")})`;
}

const RED = "#cc0000";
const BLUE = "#0033cc";

function legendItem(label, type) {
  const item = element("li");
  const swatch = element("i", null, `swatch ${type}`);
  swatch.setAttribute("aria-hidden", "true");
  item.append(swatch, label);
  return item;
}

export function drawGrid(slot, {
  feature = "", detector = "", truth = true, injected = false, symbol = null,
  selected = null, onSelect = null, compact = false, label = "Carrier resource grid",
} = {}) {
  const { grids, metadata } = slot;
  const map = slot.detectors[detector];
  const values = grids[feature];
  const scale = values ? featureScale(values) : null;
  const rows = grids.allocation.length;
  const cols = grids.allocation[0].length;
  const dmrs = metadata.config.dmrs_symbol;
  const dx = 20;
  const dy = compact ? 6 : 8;
  const x0 = 28;
  const y0 = 14;
  const width = x0 + dx * cols + 4;
  const height = y0 + dy * rows + 22;
  const patternId = `unscored-${++nextPattern}`;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "visual-grid",
    role: onSelect ? "group" : "img", "aria-label": label,
  });
  const defs = svgNode("defs");
  const hatch = svgNode("pattern", {
    id: patternId, width: 5, height: 5, patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  hatch.append(svgNode("rect", { width: 5, height: 5, fill: "#fff" }));
  hatch.append(svgNode("line", { x1: 1, x2: 1, y1: 0, y2: 5, stroke: "#999", "stroke-width": 0.6 }));
  defs.append(hatch);
  svg.append(defs);

  for (let s = 0; s < cols; s++) {
    const x = x0 + dx * s + dx / 2;
    svg.append(svgNode("text", {
      x, y: y0 - 4, "font-size": 8, "text-anchor": "middle",
      ...(s === dmrs ? { "font-weight": 700 } : {}),
    }, s));
  }
  const cells = [];
  for (let prb = 0; prb < rows; prb++) {
    const y = y0 + dy * (rows - 1 - prb);
    if (prb % 6 === 0 || prb === rows - 1) {
      svg.append(svgNode("text", {
        x: x0 - 4, y: y + dy / 2, "font-size": 8, "text-anchor": "end",
        "dominant-baseline": "central",
      }, prb));
    }
    cells[prb] = [];
    for (let s = 0; s < cols; s++) {
      const x = x0 + dx * s;
      const allocated = Boolean(grids.allocation[prb][s]);
      const highlighted = symbol == null || symbol === s;
      const hit = truth && grids.truth[prb][s] && highlighted;
      const flag = Boolean(map?.decisions[prb][s]) && highlighted;
      const unscored = Boolean(map?.unscored[prb][s]);
      const value = values?.[prb][s];
      let fill = allocated ? "#d9d9d9" : "#fff";
      if (scale && Number.isFinite(value)) fill = heatColor(value, scale);
      if (unscored) fill = `url(#${patternId})`;
      if (hit && injected) fill = RED;
      if (flag && !values && !injected) fill = BLUE;

      const cell = svgNode("g", { opacity: highlighted ? 1 : 0.4 });
      cell.append(svgNode("rect", {
        x, y, width: dx, height: dy, fill, stroke: "#b3b3b3", "stroke-width": 0.3,
      }));
      if (flag && (values || injected)) {
        cell.append(svgNode("rect", {
          x: x + 2, y: y + 1.5, width: 5, height: dy - 3, fill: BLUE,
        }));
      }
      if (hit && !injected) {
        cell.append(svgNode("rect", {
          x: x + 0.6, y: y + 0.6, width: dx - 1.2, height: dy - 1.2,
          fill: "none", stroke: RED, "stroke-width": 1.1,
        }));
      }
      if (onSelect) {
        const active = selected?.[0] === prb && selected?.[1] === s;
        cell.setAttribute("class", "cell");
        cell.setAttribute("role", "button");
        cell.setAttribute("tabindex", active ? "0" : "-1");
        cell.setAttribute("aria-label", `PRB ${prb}, OFDM symbol ${s}, ${allocated ? "scheduled" : "unscheduled"}`);
        cell.setAttribute("aria-pressed", active);
        cell.append(svgNode("rect", {
          x: x + 1, y: y + 1, width: dx - 2, height: dy - 2, fill: "transparent",
          class: "cell-border",
        }));
        cell.addEventListener("click", () => choose(prb, s));
        cell.addEventListener("keydown", event => {
          const moves = { ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            choose(prb, s);
          } else if (Object.hasOwn(moves, event.key)) {
            event.preventDefault();
            const [dp, ds] = moves[event.key];
            choose(Math.max(0, Math.min(rows - 1, prb + dp)), Math.max(0, Math.min(cols - 1, s + ds)));
          }
        });
        cells[prb][s] = cell;
      }
      svg.append(cell);
    }
  }

  function choose(prb, s) {
    const old = svg.querySelector('[tabindex="0"]');
    old?.setAttribute("tabindex", "-1");
    old?.setAttribute("aria-pressed", "false");
    const cell = cells[prb][s];
    cell.setAttribute("tabindex", "0");
    cell.setAttribute("aria-pressed", "true");
    cell.focus({ preventScroll: true });
    onSelect(prb, s);
  }

  if (symbol != null) {
    svg.append(svgNode("rect", {
      x: x0 + dx * symbol, y: y0, width: dx, height: dy * rows,
      fill: "none", stroke: "#000", "stroke-width": 1,
    }));
  }
  const axisCenter = y0 + dy * rows / 2;
  svg.append(svgNode("rect", {
    x: x0, y: y0, width: dx * cols, height: dy * rows, fill: "none", stroke: "#000", "stroke-width": 0.5,
  }));
  svg.append(svgNode("text", {
    x: 7, y: axisCenter, transform: `rotate(-90 7 ${axisCenter})`,
    "text-anchor": "middle", "dominant-baseline": "central", "font-size": 8.5,
  }, "PRB"));
  svg.append(svgNode("text", {
    x: x0 + dx * cols / 2, y: y0 + dy * rows + 13, "text-anchor": "middle", "font-size": 8.5,
  }, `OFDM symbol (DMRS at ${dmrs})`));

  const wrap = element("div", null, "grid-wrap");
  const legend = element("ul", null, "legend");
  legend.append(legendItem("Allocation", "alloc"));
  if (truth) legend.append(legendItem("Jammed", injected ? "injected" : "truth"));
  if (map) {
    legend.append(legendItem(`${detectors[detector] ?? detector} flag`, "flag"));
    if (map.unscored.some(row => row.some(Boolean))) legend.append(legendItem("Unscored", "hatched"));
  }
  if (scale) legend.append(legendItem(`${FEATURES[feature]}, ${scale[0]} to ${scale[1]} dB`, "scale"));
  wrap.append(svg, legend);
  return wrap;
}

function definitionList(entries) {
  const list = element("dl", null, "inspect-values");
  for (const [name, value] of entries) list.append(element("dt", name), element("dd", value));
  return list;
}

// One line on what each recorded slot shows (the former constellation gallery).
const lessons = {
  benign: "Without jamming, the data points sit around the four QPSK positions.",
  pilot: "Pilot jamming spoils the channel estimate, so the data points of the jammed PRBs spread.",
  pilot_weak: "The pilot residual reveals interference that leaves the data points almost unchanged.",
  data: "The jammer hits the data symbols directly; the map lowers their bit reliabilities.",
  partial: "The jammer hits pilots and data, so both the channel estimate and the symbols suffer.",
  hopping: "The jammer changes PRBs during the slot. The U-Net trained on mixed attacks follows it; the one trained on fixed blocks flags PRBs outside the hop.",
  pulsed: "The jammer is on for a few symbols only; the decoder relies on the rest.",
  spoof: "The jammer imitates the pilots. The U-Net trained with spoofing flags pilot cells the nominal one misses.",
  pilot_whole_comb: "A failure case: jamming the pilots across the whole allocation inflates the noise estimate, and even the true map does not save the block.",
};

async function explorer(root) {
  const entries = (await loadManifest()).cases.filter(item => !["fig3", "jsr_sweep", "walkthrough_flip"].includes(item.name));
  const caseSelect = document.querySelector("#case-select");
  const feature = document.querySelector("#feature-select");
  const detector = document.querySelector("#detector-select");
  const truth = document.querySelector("#truth-toggle");
  const everyReceiver = document.querySelector("#all-receivers");
  const lesson = document.querySelector("#explore-lesson");
  const context = document.querySelector("#explore-context");
  const grid = document.querySelector("#explore-grid");
  const pointHost = document.querySelector("#explore-points");
  const details = document.querySelector("#cell-details");
  const batch = document.querySelector("#explore-batch");
  const unavailable = document.querySelector("#unavailable");
  let slot;
  let selected;
  let request = 0;

  const receivers = () => everyReceiver.checked ? Object.keys(slot.receivers) : comparisonReceivers(slot);

  function inspect(prb, symbol) {
    selected = [prb, symbol];
    const onAllocation = Boolean(slot.grids.allocation[prb][symbol]);
    const entries = [
      ["PRB, symbol", `${prb}, ${symbol}`],
      ["Allocation", onAllocation ? "Yes" : "No"],
      ["Jammed", slot.grids.truth[prb][symbol] ? "Yes" : "No"],
      ...Object.entries(FEATURES).map(([key, label]) => {
        const value = slot.grids[key][prb][symbol];
        return [label, value == null ? "—" : `${value} dB`];
      }),
    ];
    const decisions = Object.entries(slot.detectors).map(([key, map]) => {
      const score = map.scores[prb][symbol];
      const decision = map.decisions[prb][symbol] ? "flagged" : "clear";
      return [detectors[key] ?? key, map.unscored[prb][symbol] ? "not scored" : `${decision} (${score})`];
    });
    details.replaceChildren(definitionList(entries), element("h3", "Detectors"),
      definitionList(decisions), element("p", "Scores: logits for the U-Nets, −log₁₀ p for the block test, linear ratios for thresholds.", "context"));
    showPoints();
  }

  function showPoints() {
    if (!slot || !selected) return;
    const [prb] = selected;
    const names = receivers();
    const [start, count] = slot.metadata.alloc;
    if (prb >= start && prb < start + count) {
      pointHost.replaceChildren(element("p", `PRB ${prb}`, "context"), constellations(slot, names, [prb]));
    } else {
      pointHost.replaceChildren(element("p", "Pick a PRB inside the allocation to see its data symbols.", "context"),
        crcList(slot, names));
    }
    batch.textContent = batchText(slot, names);
  }

  function redraw() {
    if (!slot) return;
    grid.replaceChildren(drawGrid(slot, {
      feature: feature.value, detector: detector.value, truth: truth.checked, selected,
      onSelect: inspect, label: `Interactive grid for ${slot.metadata.title}`,
    }));
  }

  async function changeCase() {
    const current = ++request;
    root.setAttribute("aria-busy", "true");
    const name = caseSelect.value;
    const previousDetector = detector.options.length ? detector.value : null;
    slot = null;
    selected = null;
    for (const control of [feature, detector, truth]) control.disabled = true;
    context.textContent = "Loading the slot…";
    lesson.textContent = lessons[name] ?? "";
    for (const host of [grid, details, pointHost, batch, unavailable]) host.replaceChildren();
    try {
      const loaded = await loadCase(name);
      if (current !== request) return;
      slot = loaded;
      const maps = Object.keys(slot.detectors);
      options(detector, [["", "None"], ...maps.map(key => [key, detectors[key] ?? key])]);
      detector.value = previousDetector === "" || maps.includes(previousDetector) ? previousDetector : "fused";
      feature.disabled = truth.disabled = false;
      context.replaceChildren(conditions(slot), slotDetails(slot));
      if (slot.metadata.noise_estimate === "comb") {
        unavailable.textContent = "Only the U-Net map exists for this noise estimate.";
      }
      inspect(affectedPRBs(slot)[0], slot.metadata.config.dmrs_symbol);
      redraw();
    } catch (error) {
      if (current === request) reportError(grid, error);
    } finally {
      if (current === request) root.setAttribute("aria-busy", "false");
    }
  }

  options(caseSelect, entries.map(({ name, title }) => [name, title]));
  caseSelect.value = new URLSearchParams(location.search).get("case") ?? "pilot";
  if (!caseSelect.value) caseSelect.selectedIndex = 0;
  caseSelect.addEventListener("change", changeCase);
  for (const control of [feature, detector, truth]) control.addEventListener("change", redraw);
  everyReceiver.addEventListener("change", showPoints);
  await changeCase();
}

const explorerHost = document.querySelector("#explorer");
if (explorerHost) explorer(explorerHost).catch(error => reportError(explorerHost, error));

const overview = document.querySelector("#overview-grid");
if (overview) {
  loadCase("pilot").then(slot => {
    overview.replaceChildren(drawGrid(slot, {
      detector: "fused", compact: true, label: "U-Net flags and jammed pilot cells",
    }));
    overview.setAttribute("aria-busy", "false");
    document.querySelector("#overview-caption").textContent = conditions(slot);
  }).catch(error => reportError(overview, error));
}
