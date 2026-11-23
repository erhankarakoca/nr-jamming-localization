import {
  batchText, calibrationText, detectors, loadCase, loadManifest, receiverName, selectionText, signedDb,
} from "./data.js";
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
  selected = null, onSelect = null, compact = false, label = "Carrier resource grid", scale: fixedScale = null,
} = {}) {
  const { grids, metadata } = slot;
  const map = slot.detectors[detector];
  const values = grids[feature];
  const scale = values ? (fixedScale ?? featureScale(values)) : null;
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

// One line on what each attack does to the slot.
const lessons = {
  pilot: "The jammer hits only the pilots (symbol 2). As its power rises, the data points of the jammed PRBs spread as well, because their channel estimate is wrong.",
  data: "The jammer hits the data symbols directly; the map tells the decoder to give their bits little weight.",
  partial: "The jammer hits pilots and data, so both the channel estimate and the symbols suffer.",
  hopping: "The jammer changes PRBs during the slot. The U-Net trained on mixed attacks follows it; the one trained on fixed blocks flags PRBs outside the hop.",
  pulsed: "The jammer is on for a few symbols only; the decoder relies on the rest.",
  spoof: "The jammer imitates the pilots. The U-Net trained with spoofing flags pilot cells the nominal one misses.",
  pilot_whole_comb: "The pilots of the whole allocation are jammed and the receiver estimates the noise power from the empty pilot comb, which the jammer inflates. At +3 dB even the true map does not save this slot.",
};

// Cases with a jammer-power slider; the others (benign, the walkthrough's slots) are shown elsewhere
const explorerExclusions = new Set(["benign", "pilot_weak", "fig3", "jsr_sweep", "aided_flip", "partial_flip"]);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const PLAY_STEP_MS = 1300;

// The slot at the slider's power: the case's own power is the document itself, every other level
// replaces the per-power fields and keeps the rest (allocation, settings, selection).
function atLevel(doc, index) {
  const level = doc.levels?.[index];
  if (!level?.grids) return doc;
  return {
    ...doc,
    grids: level.grids, detectors: level.detectors, receivers: level.receivers,
    x_hat: level.x_hat, abs_llr: level.abs_llr,
    metadata: { ...doc.metadata, jsr_db: level.jsr_db, batch_crc: level.batch_crc },
  };
}

const powerText = jsr => (jsr == null ? "Jammer off" : signedDb(jsr));
const tickText = jsr => (jsr == null ? "Off" : `${jsr > 0 ? "+" : ""}${String(jsr).replace("-", "−")}`);

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

// Receivers drawn in the chart of the batch's CRC failures, with their line styles
const CHART_STYLES = {
  standard: { color: "#1a1a1a", dash: "2 2", marker: "cross" },
  fused_soft: { color: BLUE, dash: "", marker: "circle" },
  fused_atk_soft: { color: "#d95f02", dash: "", marker: "square" },
  fused_atk2_soft: { color: "#d95f02", dash: "", marker: "square" },
  true_soft: { color: "#008000", dash: "5 2 1 2", marker: "open-triangle" },
};

function marker(kind, x, y, color) {
  if (kind === "circle") return svgNode("circle", { cx: x, cy: y, r: 2.6, fill: color });
  if (kind === "square") return svgNode("rect", { x: x - 2.4, y: y - 2.4, width: 4.8, height: 4.8, fill: color });
  if (kind === "open-triangle") {
    // Hollow, so a filled marker underneath (often the U-Net map at the same count) stays visible
    return svgNode("path", {
      d: `M ${x} ${y - 4.2} L ${x + 4.2} ${y + 3.2} L ${x - 4.2} ${y + 3.2} Z`, fill: "none", stroke: color, "stroke-width": 1.1,
    });
  }
  return svgNode("path", {
    d: `M ${x - 2.5} ${y - 2.5} L ${x + 2.5} ${y + 2.5} M ${x - 2.5} ${y + 2.5} L ${x + 2.5} ${y - 2.5}`,
    stroke: color, "stroke-width": 1.3,
  });
}

// CRC failures among the batch's slots at every jammer power, with the slider's power marked.
// Clicking a column moves the slider there.
function failureChart(doc, names, current, onPick) {
  const levels = doc.levels.map((level, index) => atLevel(doc, index).metadata);
  const slots = doc.metadata.batch_crc.standard.slots;
  const width = 400;
  const height = 196;
  const left = 44;
  const right = 392;
  const top = 10;
  const bottom = 150;
  const pitch = (right - left) / levels.length;
  const x = i => left + pitch * (i + 0.5);
  const y = value => bottom - (value / slots) * (bottom - top);
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "failure-chart", role: "img",
    "aria-label": `CRC failures among the ${slots} slots simulated at each jammer power. ` +
      names.map(name => `${receiverName(name)}: ${levels.map(m => m.batch_crc[name].failures).join(", ")}`).join("; "),
  });
  svg.append(svgNode("rect", {
    x: left + pitch * current, y: top, width: pitch, height: bottom - top, fill: "#ececec",
  }));
  for (const value of [0, slots / 4, slots / 2, (3 * slots) / 4, slots]) {
    svg.append(svgNode("line", { x1: left, x2: right, y1: y(value), y2: y(value), stroke: "#d9d9d9", "stroke-width": 0.5 }));
    svg.append(svgNode("text", { x: left - 5, y: y(value) + 4, "text-anchor": "end", "font-size": 11 }, value));
  }
  for (const [i, m] of levels.entries()) {
    svg.append(svgNode("text", {
      x: x(i), y: bottom + 15, "text-anchor": "middle", "font-size": 11, "font-weight": i === current ? 700 : 400,
    }, tickText(m.jsr_db)));
  }
  svg.append(svgNode("text", { x: (left + right) / 2, y: height - 8, "text-anchor": "middle", "font-size": 12 }, "JSR (dB)"));
  const mid = (top + bottom) / 2;
  svg.append(svgNode("text", {
    x: 12, y: mid, "text-anchor": "middle", "font-size": 12, transform: `rotate(-90 12 ${mid})`,
  }, "CRC failures"));
  svg.append(svgNode("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "none", stroke: "#000", "stroke-width": 0.6 }));
  for (const name of names) {
    const style = CHART_STYLES[name];
    const values = levels.map(m => m.batch_crc[name].failures);
    svg.append(svgNode("path", {
      d: values.map((v, i) => `${i ? "L" : "M"} ${x(i)} ${y(v)}`).join(" "),
      fill: "none", stroke: style.color, "stroke-width": 1.2, "stroke-dasharray": style.dash,
    }));
    for (const [i, v] of values.entries()) svg.append(marker(style.marker, x(i), y(v), style.color));
  }
  for (const [i, m] of levels.entries()) {
    const hit = svgNode("rect", {
      x: left + pitch * i, y: top, width: pitch, height: bottom - top + 20, fill: "transparent", class: "pick",
    });
    hit.append(svgNode("title", {}, powerText(m.jsr_db)));
    hit.addEventListener("click", () => onPick(i));
    svg.append(hit);
  }
  const legend = element("ul", null, "legend");
  for (const name of names) {
    const item = element("li");
    const style = CHART_STYLES[name];
    const key = svgNode("svg", { viewBox: "0 0 22 8", width: 22, height: 8, "aria-hidden": "true" });
    key.append(svgNode("line", { x1: 1, x2: 21, y1: 4, y2: 4, stroke: style.color, "stroke-width": 1.2, "stroke-dasharray": style.dash }));
    key.append(marker(style.marker, 11, 4, style.color));
    item.append(key, receiverName(name));
    legend.append(item);
  }
  const figure = element("figure");
  const caption = element("figcaption", `CRC failures among the ${slots} slots simulated at each power (shaded: the slider's power)`);
  figure.append(caption, svg, legend);
  return figure;
}

// What the jammer did to the slot at this power, what the detector flagged and how each
// receiver decoded the block.
function levelSummary(slot, detectorKey, names) {
  const { truth, allocation } = slot.grids;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const width = slot.metadata.alloc[1];
  const flags = slot.detectors[detectorKey]?.decisions;
  let jammed = 0, pilotPRBs = 0, data = 0, outside = 0, hits = 0, other = 0;
  for (const [prb, row] of truth.entries()) {
    for (const [s, hit] of row.entries()) {
      const flagged = Boolean(flags?.[prb][s]);
      if (hit) {
        jammed++;
        if (!allocation[prb][s]) outside++;
        else if (s === dmrs) pilotPRBs++;
        else data++;
        if (flagged) hits++;
      } else if (flagged) other++;
    }
  }
  const lines = [];
  const power = powerText(slot.metadata.jsr_db);
  if (slot.metadata.jsr_db == null) {
    lines.push(`${power}: the same slot without the jammer.`);
  } else {
    const parts = [
      pilotPRBs && `the pilots of ${pilotPRBs} of the user's ${width} PRBs`,
      data && `${data} of its data cells`,
      outside && `${outside} cells outside its allocation`,
    ].filter(Boolean);
    const list = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}` : parts[0];
    lines.push(`${power}: the jammer hits ${jammed} cells (PRB × symbol): ${list}.`);
  }
  const detector = detectors[detectorKey] ?? detectorKey;
  lines.push(jammed
    ? `${detector} flags ${hits} of the ${jammed} jammed cells and ${other} other ${other === 1 ? "cell" : "cells"}.`
    : `${detector} flags ${other} ${other === 1 ? "cell" : "cells"} (false alarms).`);
  const host = element("div");
  for (const text of lines) host.append(element("p", text));
  host.append(crcList(slot, names));
  return host;
}

async function explorer(root) {
  const entries = (await loadManifest()).cases.filter(item => !explorerExclusions.has(item.name));
  const caseSelect = document.querySelector("#case-select");
  const levelInput = document.querySelector("#level");
  const levelValue = document.querySelector("#level-value");
  const ticks = document.querySelector("#level-ticks");
  const play = document.querySelector("#level-play");
  const summary = document.querySelector("#level-summary");
  const chart = document.querySelector("#level-chart");
  const feature = document.querySelector("#feature-select");
  const detector = document.querySelector("#detector-select");
  const truth = document.querySelector("#truth-toggle");
  const everyReceiver = document.querySelector("#all-receivers");
  const pointSource = document.querySelector("#point-source");
  const lesson = document.querySelector("#explore-lesson");
  const context = document.querySelector("#explore-context");
  const grid = document.querySelector("#explore-grid");
  const pointsHeading = document.querySelector("#points-heading");
  const pointHost = document.querySelector("#explore-points");
  const details = document.querySelector("#cell-details");
  const batch = document.querySelector("#explore-batch");
  let doc;
  let slot;
  let level = 0;
  let selected;
  let request = 0;
  let motion = 0;
  let timer = null;

  function receivers() {
    const names = comparisonReceivers(slot);
    return everyReceiver.checked ? [...new Set([...names, ...Object.keys(slot.receivers)])] : names;
  }

  function chartReceivers() {
    return ["standard", "fused_soft", ...Object.keys(slot.receivers).filter(n => n.startsWith("fused_atk")), "true_soft"]
      .filter(name => slot.receivers[name] && CHART_STYLES[name]);
  }

  // The colour scale of a feature over every power, so colours compare across the slider
  function sharedScale(key) {
    if (!key) return null;
    const frames = doc.levels ? doc.levels.map((_, i) => atLevel(doc, i)) : [doc];
    return featureScale(frames.flatMap(frame => frame.grids[key]));
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
    if (changedPRB && pointSource.value === "prb") showPoints();
  }

  function showPoints(from = null) {
    if (!slot || !selected) return;
    const [prb] = selected;
    const names = receivers();
    const [start, count] = slot.metadata.alloc;
    const mine = ++motion;
    const animate = { from: reduceMotion ? null : from, alive: () => mine === motion };
    if (pointSource.value === "all") {
      pointsHeading.textContent = `Data symbols of the user's ${count} PRBs (${start}–${start + count - 1})`;
      pointHost.replaceChildren(constellations(slot, names, null, animate));
    } else if (prb >= start && prb < start + count) {
      pointsHeading.textContent = `Data symbols of PRB ${prb}`;
      pointHost.replaceChildren(constellations(slot, names, [prb], animate));
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
      scale: sharedScale(feature.value),
      onSelect: inspect, label: `Interactive grid for ${slot.metadata.title} at ${powerText(slot.metadata.jsr_db)}`,
    });
    grid.replaceChildren(drawn);
    drawn.querySelector(".explore-grid-scroll").scrollLeft = scroll;
    summary.replaceChildren(levelSummary(slot, detector.value || "fused", comparisonReceivers(slot)));
  }

  function remember() {
    const params = new URLSearchParams(location.search);
    params.set("case", caseSelect.value);
    const jsr = slot.metadata.jsr_db;
    params.set("jsr", jsr == null ? "off" : String(jsr));
    history.replaceState(null, "", `?${params}`);
  }

  function showLevel(index, { animate = true } = {}) {
    const previous = slot;
    level = index;
    slot = atLevel(doc, index);
    levelInput.value = String(index);
    levelValue.textContent = powerText(slot.metadata.jsr_db);
    levelInput.setAttribute("aria-valuetext", levelValue.textContent);
    for (const [i, tick] of [...ticks.children].entries()) tick.classList.toggle("current", i === index);
    redraw();
    inspect(...selected);
    showPoints(animate ? previous : null);
    chart.replaceChildren(failureChart(doc, chartReceivers(), index, pick));
    remember();
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
    play.setAttribute("aria-pressed", "false");
  }

  function pick(index) {
    stop();
    showLevel(index);
  }

  function togglePlay() {
    if (timer) return stop();
    play.textContent = "Pause";
    play.setAttribute("aria-pressed", "true");
    const last = doc.levels.length - 1;
    if (level === last) showLevel(0);
    timer = setInterval(() => {
      if (level >= last) return stop();
      showLevel(level + 1);
    }, PLAY_STEP_MS);
  }

  async function changeCase({ jsr } = {}) {
    stop();
    const current = ++request;
    root.setAttribute("aria-busy", "true");
    const name = caseSelect.value;
    const previousDetector = detector.options.length ? detector.value : null;
    doc = slot = null;
    selected = null;
    for (const control of [feature, detector, truth, levelInput, play]) control.disabled = true;
    context.textContent = "Loading the slot at every jammer power…";
    pointsHeading.textContent = "Equalized symbols";
    lesson.textContent = lessons[name] ?? "";
    for (const host of [grid, details, pointHost, batch, summary, chart]) host.replaceChildren();
    try {
      const loaded = await loadCase(name);
      if (current !== request) return;
      doc = loaded;
      const levels = doc.levels ?? [{ jsr_db: doc.metadata.jsr_db }];
      const own = levels.findIndex(entry => entry.jsr_db === doc.metadata.jsr_db);
      const wanted = jsr === undefined ? -1 :
        levels.findIndex(entry => (entry.jsr_db == null ? "off" : String(entry.jsr_db)) === jsr);
      level = wanted >= 0 ? wanted : own;
      slot = atLevel(doc, level);
      levelInput.max = String(levels.length - 1);
      ticks.replaceChildren(...levels.map((entry, i) => {
        const tick = element("span", tickText(entry.jsr_db));
        tick.style.setProperty("--at", String(levels.length > 1 ? i / (levels.length - 1) : 0.5));
        return tick;
      }));
      const maps = Object.keys(slot.detectors);
      options(detector, [["", "None"], ...maps.map(key => [key, detectors[key] ?? key])]);
      detector.value = previousDetector === "" || maps.includes(previousDetector) ? previousDetector : "fused";
      feature.disabled = truth.disabled = false;
      levelInput.disabled = play.disabled = levels.length < 2;
      const { batch_size, selection } = doc.metadata;
      const settings = element("details", null, "more");
      settings.append(element("summary", "Simulation and detector settings"),
        element("p", `Slot ${selection.slot + 1} of ${batch_size} simulated slots, chosen at ${signedDb(doc.metadata.jsr_db)}. ${selectionText(doc)}`),
        element("p", calibrationText(doc)),
        element("p", "Every power replays the same random draws (channel, noise, data and the jammer's waveform) and scales only the jammer's power, so the same slot is shown throughout. The chart counts the CRC failures of all the slots simulated at each power."));
      if (doc.metadata.noise_estimate === "comb") {
        settings.append(element("p", "The receiver estimated the noise power from the empty pilot comb."));
      }
      const m = doc.metadata;
      const noise = m.noise_estimate === "comb" ? " · estimated noise power" : "";
      context.replaceChildren(element("p", `TDL-${m.channel} · ${m.snr_db} dB SNR · the user on PRBs ${m.alloc[0]}–${m.alloc[0] + m.alloc[1] - 1}${noise}`), settings);
      selected = initialCell(doc);
      showLevel(level, { animate: false });
    } catch (error) {
      if (current === request) reportError(grid, error);
    } finally {
      if (current === request) root.setAttribute("aria-busy", "false");
    }
  }

  options(caseSelect, entries.map(({ name, title }) => [name, title]));
  const params = new URLSearchParams(location.search);
  caseSelect.value = params.get("case") ?? "partial";
  if (!caseSelect.value) caseSelect.selectedIndex = 0;
  caseSelect.addEventListener("change", () => changeCase());
  levelInput.addEventListener("input", () => pick(Number(levelInput.value)));
  play.addEventListener("click", togglePlay);
  for (const control of [feature, detector, truth]) control.addEventListener("change", redraw);
  everyReceiver.addEventListener("change", () => showPoints());
  pointSource.addEventListener("change", () => showPoints());
  await changeCase({ jsr: params.get("jsr") ?? undefined });
}

const explorerHost = document.querySelector("#explorer");
if (explorerHost) explorer(explorerHost).catch(error => reportError(explorerHost, error));
