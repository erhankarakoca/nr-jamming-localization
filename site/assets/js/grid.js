import { batchText, calibrationText, conditions, detectors, loadCase, loadManifest, selectionText } from "./data.js";
import { comparisonReceivers, constellations, crcList } from "./constellation.js";
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

export function featureScale(values) {
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

export function heatColor(value, [min, max]) {
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
  const dy = onSelect ? 10 : (compact ? 6 : 8);
  const x0 = 28;
  const y0 = 14;
  const width = x0 + dx * cols + 4;
  const height = y0 + dy * rows + 22;
  const patternId = `unscored-${++nextPattern}`;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "visual-grid",
    role: onSelect ? "group" : "img", "aria-label": label,
  });
  if (!onSelect) {
    const defs = svgNode("defs");
    const hatch = svgNode("pattern", {
      id: patternId, width: 5, height: 5, patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    hatch.append(svgNode("rect", { width: 5, height: 5, fill: "#fff" }));
    hatch.append(svgNode("line", { x1: 1, x2: 1, y1: 0, y2: 5, stroke: "#999", "stroke-width": 0.6 }));
    defs.append(hatch);
    svg.append(defs);
  }

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
      if (unscored && !onSelect) fill = `url(#${patternId})`;
      if (hit && injected) fill = RED;
      if (flag && !values && !injected && !onSelect) fill = BLUE;

      const cell = svgNode("g", { opacity: highlighted ? 1 : 0.4 });
      cell.append(svgNode("rect", {
        x, y, width: dx, height: dy, fill, stroke: "#b3b3b3", "stroke-width": 0.3,
      }));
      if (unscored && onSelect) {
        cell.append(svgNode("line", {
          x1: x + dx / 2 - 2.5, y1: y + dy - 2.5, x2: x + dx / 2 + 2.5, y2: y + 2.5,
          stroke: "#999", "stroke-width": 1, "stroke-linecap": "round",
        }));
      }
      if (flag && onSelect) {
        cell.append(svgNode("circle", {
          cx: x + dx / 2, cy: y + dy / 2, r: 2, fill: BLUE, stroke: "#fff", "stroke-width": 0.7,
        }));
      } else if (flag && (values || injected)) {
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
          x: x + 2, y: y + 2, width: dx - 4, height: dy - 4, fill: "transparent",
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
    cell.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "instant" });
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
  if (!onSelect || !scale) {
    legend.append(legendItem("Allocation", "alloc"));
  } else if (values.some(row => row.some(value => !Number.isFinite(value)))) {
    legend.append(legendItem("Scheduled cell without a feature value", "alloc"),
      legendItem("Unscheduled cell without a feature value", "empty"));
  }
  if (truth) legend.append(legendItem("Jammed", injected ? "injected" : "truth"));
  if (map) {
    legend.append(legendItem(`${detectors[detector] ?? detector} flag`, onSelect ? "flag-dot" : "flag"));
    if (map.unscored.some(row => row.some(Boolean))) {
      legend.append(onSelect ? legendItem("No detector score", "unscored") : legendItem("Unscored", "hatched"));
    }
  }
  if (scale) legend.append(legendItem(`${FEATURES[feature]}, ${scale[0]} to ${scale[1]} dB`, "scale"));
  const frame = onSelect ? element("div", null, "explore-grid-scroll") : null;
  frame?.append(svg);
  wrap.append(frame ?? svg, legend);
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

const explorerExclusions = new Set(["fig3", "jsr_sweep", "aided_flip", "walkthrough_flip", "partial_flip"]);

function initialCell(slot) {
  const { allocation, truth } = slot.grids;
  for (const [prb, row] of allocation.entries()) {
    const symbol = row.findIndex((scheduled, index) => scheduled && truth[prb][index]);
    if (symbol !== -1) return [prb, symbol];
  }
  const [start, count] = slot.metadata.alloc;
  return [start + Math.floor(count / 2), Math.floor(slot.metadata.config.num_symbols / 2)];
}

function missingFeature(key, scheduled, pilot) {
  if (!scheduled && key !== "power_db") return "outside the allocation";
  if (pilot && key === "evm_db") return "no EVM on pilot cells";
  if (!pilot && key === "residual_db") return "pilot residual is measured on pilot cells";
  return "no value recorded for this cell";
}

function missingScore(key, scheduled, pilot) {
  if (["evm_threshold", "residual_threshold", "residual_cfar"].includes(key)) {
    return missingFeature(key === "evm_threshold" ? "evm_db" : "residual_db", scheduled, pilot);
  }
  if (key === "power_threshold" && !scheduled) return "outside the allocation";
  if (key === "comb_threshold") {
    if (!scheduled) return "outside the allocation";
    if (!pilot) return "the empty comb is measured on pilot cells";
  }
  return "no score recorded for this cell";
}

async function explorer(root) {
  const entries = (await loadManifest()).cases.filter(item => !explorerExclusions.has(item.name));
  const caseSelect = document.querySelector("#case-select");
  const feature = document.querySelector("#feature-select");
  const detector = document.querySelector("#detector-select");
  const truth = document.querySelector("#truth-toggle");
  const everyReceiver = document.querySelector("#all-receivers");
  const lesson = document.querySelector("#explore-lesson");
  const context = document.querySelector("#explore-context");
  const grid = document.querySelector("#explore-grid");
  const pointsHeading = document.querySelector("#points-heading");
  const pointHost = document.querySelector("#explore-points");
  const details = document.querySelector("#cell-details");
  const batch = document.querySelector("#explore-batch");
  let slot;
  let selected;
  let request = 0;

  function receivers() {
    const names = comparisonReceivers(slot);
    return everyReceiver.checked ? [...new Set([...names, ...Object.keys(slot.receivers)])] : names;
  }

  function inspect(prb, symbol) {
    const changedPRB = selected?.[0] !== prb;
    selected = [prb, symbol];
    const onAllocation = Boolean(slot.grids.allocation[prb][symbol]);
    const pilot = symbol === slot.metadata.config.dmrs_symbol;
    const entries = [
      ["Location", `PRB ${prb}, OFDM symbol ${symbol}`],
      ["Allocation", onAllocation ? "Scheduled" : "Outside the allocation"],
      ["Jammed", slot.grids.truth[prb][symbol] ? "Yes" : "No"],
      ...Object.entries(FEATURES).map(([key, label]) => {
        const value = slot.grids[key][prb][symbol];
        const readout = Number.isFinite(value) ? `${value} dB` :
          `Unavailable · ${missingFeature(key, onAllocation, pilot)}`;
        return [label, readout];
      }),
    ];
    const decisions = Object.entries(slot.detectors).map(([key, map]) => {
      const label = detectors[key] ?? key;
      const score = map.scores[prb][symbol];
      if (map.unscored[prb][symbol] || !Number.isFinite(score)) {
        return [label, `Unscored · ${missingScore(key, onAllocation, pilot)}`];
      }
      const decision = map.decisions[prb][symbol] ? "Flagged" : "Clear";
      const cutoff = slot.metadata.thresholds[key]?.value;
      const limit = Number.isFinite(cutoff) ? `flags above ${cutoff}` : "cutoff missing from this export";
      return [label, `${decision} · score ${score} · ${limit}`];
    });
    details.replaceChildren(definitionList(entries), element("h3", "Detector decisions"), definitionList(decisions));
    if (changedPRB) showPoints();
  }

  function showPoints() {
    if (!slot || !selected) return;
    const [prb] = selected;
    const names = receivers();
    const [start, count] = slot.metadata.alloc;
    if (prb >= start && prb < start + count) {
      const symbols = slot.metadata.config.num_symbols - 1;
      pointsHeading.textContent = `Data symbols of PRB ${prb}, all ${symbols} data symbols`;
      pointHost.replaceChildren(constellations(slot, names, [prb]));
    } else {
      pointsHeading.textContent = `Data symbols of PRB ${prb}`;
      pointHost.replaceChildren(element("p", `PRB ${prb} is outside the allocation. Pick a scheduled PRB to see its data symbols.`),
        crcList(slot, names));
    }
    batch.textContent = batchText(slot, names);
  }

  function redraw() {
    if (!slot) return;
    const scroll = grid.querySelector(".explore-grid-scroll")?.scrollLeft ?? 0;
    const drawn = drawGrid(slot, {
      feature: feature.value, detector: detector.value, truth: truth.checked, selected,
      onSelect: inspect, label: `Interactive grid for ${slot.metadata.title}`,
    });
    grid.replaceChildren(drawn);
    drawn.querySelector(".explore-grid-scroll").scrollLeft = scroll;
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
    pointsHeading.textContent = "Equalized symbols";
    lesson.textContent = lessons[name] ?? "";
    for (const host of [grid, details, pointHost, batch]) host.replaceChildren();
    try {
      const loaded = await loadCase(name);
      if (current !== request) return;
      slot = loaded;
      const maps = Object.keys(slot.detectors);
      options(detector, [["", "None"], ...maps.map(key => [key, detectors[key] ?? key])]);
      detector.value = previousDetector === "" || maps.includes(previousDetector) ? previousDetector : "fused";
      feature.disabled = truth.disabled = false;
      const { batch_size, selection } = slot.metadata;
      const settings = element("details", null, "more");
      settings.append(element("summary", "Simulation and detector settings"), element("p", calibrationText(slot)));
      if (slot.metadata.noise_estimate === "comb") {
        settings.append(element("p", "The receiver estimated the noise power from the empty pilot comb."));
      }
      context.replaceChildren(element("p", conditions(slot)),
        element("p", `Slot ${selection.slot + 1} of ${batch_size} simulated slots. ${selectionText(slot)}`), settings);
      inspect(...initialCell(slot));
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
