import { batchText, conditions, detectors, loadCase, loadManifest, percent, selectionText } from "./data.js";
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

function heatColor(value, [min, max]) {
  const level = max === min ? 0.5 : (value - min) / (max - min);
  const dark = [82, 105, 119];
  const light = [239, 243, 245];
  return `rgb(${light.map((v, i) => Math.round(v + level * (dark[i] - v))).join(",")})`;
}

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
  const dx = 23;
  const dy = compact ? 6 : 9;
  const x0 = 37;
  const y0 = 22;
  const width = x0 + dx * cols + 5;
  const height = y0 + dy * rows + 21;
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
  hatch.append(svgNode("rect", { width: 5, height: 5, fill: "#f5f7f8" }));
  hatch.append(svgNode("line", { x1: 1, x2: 1, y1: 0, y2: 5, stroke: "#879ca8" }));
  defs.append(hatch);
  svg.append(defs);

  for (let s = 0; s < cols; s++) {
    const x = x0 + dx * s + dx / 2;
    if (s === dmrs) {
      svg.append(svgNode("rect", {
        x: x - dx / 2, y: y0 - 16, width: dx, height: 15, rx: 2, fill: "#dcecf5",
      }));
    }
    svg.append(svgNode("text", {
      x, y: y0 - 5, "font-size": 12, "text-anchor": "middle",
    }, s));
  }
  const cells = [];
  for (let prb = 0; prb < rows; prb++) {
    const y = y0 + dy * (rows - 1 - prb);
    if (prb % 6 === 0 || prb === rows - 1) {
      svg.append(svgNode("text", {
        x: x0 - 5, y: y + dy / 2, "font-size": 11, "text-anchor": "end",
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
      let fill = allocated ? "#dbe5eb" : "#f5f7f8";
      if (scale && Number.isFinite(value)) fill = heatColor(value, scale);
      if (unscored) fill = `url(#${patternId})`;
      if (hit && injected) fill = "#f7c9ad";
      if (flag && !values && !injected) fill = "#0072b2";

      const cell = svgNode("g", { opacity: highlighted ? 1 : 0.46 });
      cell.append(svgNode("rect", {
        x, y, width: dx, height: dy, fill, stroke: "#fff", "stroke-width": 0.7,
      }));
      if (flag && (values || injected)) {
        cell.append(svgNode("rect", {
          x: x + 2, y: y + 1.5, width: 6, height: dy - 3, rx: 0.5, fill: "#0072b2",
        }));
      }
      if (hit) {
        cell.append(svgNode("rect", {
          x: x + 0.7, y: y + 0.7, width: dx - 1.4, height: dy - 1.4,
          fill: "none", stroke: "#d55e00", "stroke-width": 1.5,
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
      fill: "none", stroke: "#405863", "stroke-width": 1.2,
    }));
  }
  const axisCenter = y0 + dy * rows / 2;
  svg.append(svgNode("text", {
    x: 10, y: axisCenter, transform: `rotate(-90 10 ${axisCenter})`,
    "text-anchor": "middle", "dominant-baseline": "central", "font-size": 11,
  }, "PRB"));
  svg.append(svgNode("text", {
    x: x0 + dx * cols / 2, y: height - 5, "text-anchor": "middle", "font-size": 10,
  }, `OFDM symbol · DMRS at ${dmrs}`));

  const wrap = element("div", null, "grid-wrap");
  const legend = element("ul", null, "legend");
  legend.append(legendItem("Scheduled", "alloc"));
  if (truth) legend.append(legendItem("Injected", injected ? "injected" : "truth"));
  if (map) {
    legend.append(legendItem(`${detectors[detector] ?? detector} flag`, "flag"));
    if (map.unscored.some(row => row.some(Boolean))) legend.append(legendItem("Unscored", "hatched"));
  }
  if (scale) legend.append(legendItem(`${FEATURES[feature]} · ${scale[0]} to ${scale[1]} dB`, "scale"));
  wrap.append(svg, legend);
  return wrap;
}

function definitionList(entries) {
  const list = element("dl", null, "inspect-values");
  for (const [name, value] of entries) list.append(element("dt", name), element("dd", value));
  return list;
}

async function explorer(root) {
  const entries = (await loadManifest()).cases.filter(item => !["fig3", "jsr_sweep"].includes(item.name));
  const caseSelect = document.querySelector("#case-select");
  const feature = document.querySelector("#feature-select");
  const detector = document.querySelector("#detector-select");
  const truth = document.querySelector("#truth-toggle");
  const context = document.querySelector("#explore-context");
  const grid = document.querySelector("#explore-grid");
  const pointHost = document.querySelector("#explore-points");
  const details = document.querySelector("#cell-details");
  const batch = document.querySelector("#explore-batch");
  const unavailable = document.querySelector("#unavailable");
  let slot;
  let selected;
  let request = 0;

  function inspect(prb, symbol) {
    selected = [prb, symbol];
    const onAllocation = Boolean(slot.grids.allocation[prb][symbol]);
    const entries = [
      ["PRB", prb],
      ["OFDM symbol", symbol],
      ["Allocation", onAllocation ? "Scheduled" : "Unscheduled"],
      ["Injected interference", slot.grids.truth[prb][symbol] ? "Yes" : "No"],
      ...Object.entries(FEATURES).map(([key, label]) => {
        const value = slot.grids[key][prb][symbol];
        return [label, value == null ? "—" : `${value} dB`];
      }),
    ];
    const decisions = Object.entries(slot.detectors).map(([key, map]) => {
      const score = map.scores[prb][symbol];
      const decision = map.decisions[prb][symbol] ? "Flagged" : "Clear";
      return [detectors[key] ?? key, map.unscored[prb][symbol] ? "Unscored" : `${score} · ${decision}`];
    });
    details.replaceChildren(definitionList(entries), element("h3", "Detector scores and decisions"),
      definitionList(decisions), element("p", "Scores are logits for U-Nets, −log₁₀ p-values for the block test, and linear ratios for feature thresholds.", "context"));
    const title = element("p", `PRB ${prb} · ${onAllocation ? "all data symbols" : "outside the allocation"}`, "context");
    // A PRB may be scheduled even when the selected symbol has no data points.
    const [start, count] = slot.metadata.alloc;
    const names = Object.keys(slot.receivers);
    pointHost.replaceChildren(title);
    if (prb >= start && prb < start + count) {
      pointHost.append(constellations(slot, names, [prb]));
    } else {
      pointHost.append(element("p", "Select a scheduled PRB to see its equalized symbols.", "context"),
        crcList(slot, names));
    }
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
    for (const control of [feature, detector, truth]) control.disabled = true;
    context.textContent = "Loading the selected slot…";
    for (const host of [grid, details, pointHost, batch, unavailable]) host.replaceChildren();
    try {
      const loaded = await loadCase(name);
      if (current !== request) return;
      slot = loaded;
      const maps = Object.keys(slot.detectors);
      options(detector, [["", "None"], ...maps.map(key => [key, detectors[key] ?? key])]);
      detector.value = previousDetector === "" || maps.includes(previousDetector) ? previousDetector : "fused";
      feature.disabled = truth.disabled = false;
      context.textContent = `${conditions(slot)}. ${selectionText(slot)}`;
      if (slot.metadata.noise_estimate === "comb") {
        unavailable.textContent = "Other detector maps are unavailable with this noise estimate.";
      }
      const prb = affectedPRBs(slot)[0];
      inspect(prb, slot.metadata.config.dmrs_symbol);
      redraw();
      batch.textContent = batchText(slot);
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
  await changeCase();
}

const lessons = {
  benign: "Without jamming, the receiver constellations cluster around the expected QPSK positions.",
  pilot: "Pilot jamming distorts the channel estimate and spreads the data points on the affected PRBs.",
  pilot_weak: "The pilot residual can expose interference that leaves the data constellation close to its benign shape.",
  data: "Interference enters the data symbols directly. Soft weighting reduces their bit reliabilities.",
  partial: "Pilot and data interference affect both the channel estimate and the symbols decoded with it.",
  hopping: "The mixed-attack map follows changes in jammer position across symbols. The PRB-level true map combines data cells that experience different interference.",
  pulsed: "The jammer is active on selected symbols. Soft weighting lets the decoder rely more on the remaining observations.",
  spoof: "The spoof-trained network places flags on the pilot cells that the nominal network misses in this slot.",
  pilot_whole_comb: "The estimated noise power is inflated in this slot. Even the receiver using the true map fails its CRC.",
};

async function gallery(root) {
  const manifest = await loadManifest();
  const entries = manifest.cases.filter(entry => Object.hasOwn(lessons, entry.name));
  const rows = entries.map(entry => {
    const section = element("article", null, "panel");
    section.append(element("h2", entry.title), element("div", "Loading the recorded slot…"));
    return section;
  });
  root.replaceChildren(...rows);
  root.setAttribute("aria-busy", "false");

  await Promise.all(entries.map(async (entry, index) => {
    const host = rows[index].lastElementChild;
    try {
      const slot = await loadCase(entry.name);
      const names = comparisonReceivers(slot);
      const row = element("div", null, "gallery-row");
      row.append(drawGrid(slot, { injected: true, compact: true, label: `Injected cells for ${entry.title}` }),
        constellations(slot, names, affectedPRBs(slot)));
      const link = element("a", "Inspect this slot");
      link.href = `explore.html?case=${encodeURIComponent(entry.name)}`;
      host.replaceChildren(element("p", conditions(slot), "context"), element("p", lessons[entry.name]), row,
        element("p", selectionText(slot), "context gallery-selection"),
        element("p", batchText(slot, names), "context"), link);
    } catch (error) {
      reportError(host, error);
    }
  }));
}

const explorerHost = document.querySelector("#explorer");
if (explorerHost) explorer(explorerHost).catch(error => reportError(explorerHost, error));
const galleryHost = document.querySelector("#gallery");
if (galleryHost) gallery(galleryHost).catch(error => reportError(galleryHost, error));

const overview = document.querySelector("#overview-grid");
if (overview) {
  loadCase("pilot").then(slot => {
    overview.replaceChildren(drawGrid(slot, {
      detector: "fused", compact: true, label: "U-Net flags and injected pilot interference",
    }));
    overview.setAttribute("aria-busy", "false");
    document.querySelector("#overview-caption").textContent =
      `${conditions(slot)}. ${selectionText(slot)}`;
  }).catch(error => reportError(overview, error));
}
