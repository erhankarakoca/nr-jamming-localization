// "Results at a glance" on the overview: one question per card, a one-line answer in plain
// words, one small chart, and the conditions underneath. The charts read results.json, so every
// number shown is the one in the downloadable CSVs.
import { loadResults, percent, signedDb } from "./data.js";
import { svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";

const RED = "#cc0000";
const BLUE = "#0033cc";
const GREEN = "#008000";
const CYAN = "#0099cc";
const INK = "#1a1a1a";
const GREY = "#777777";

// A small line chart in SVG with the plot area at (left, top) - (right, bottom).
function frame({ width = 320, height = 210, left = 46, right = 312, top = 12, bottom = 168 } = {}) {
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "glance-chart" });
  return { svg, width, height, left, right, top, bottom };
}

function axes(f, { x, y, xTicks, yTicks, xLabel, yLabel, yFormat = String, xFormat = String }) {
  const { svg, left, right, top, bottom } = f;
  for (const t of yTicks) {
    svg.append(svgNode("line", { x1: left, x2: right, y1: y(t), y2: y(t), stroke: "#e6e6e6", "stroke-width": 0.6 }));
    svg.append(svgNode("text", { x: left - 5, y: y(t) + 3.5, "text-anchor": "end", "font-size": 10 }, yFormat(t)));
  }
  for (const t of xTicks) {
    svg.append(svgNode("text", { x: x(t), y: bottom + 14, "text-anchor": "middle", "font-size": 10 }, xFormat(t)));
  }
  svg.append(svgNode("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "none", stroke: "#000", "stroke-width": 0.7 }));
  svg.append(svgNode("text", { x: (left + right) / 2, y: bottom + 30, "text-anchor": "middle", "font-size": 10.5 }, xLabel));
  const mid = (top + bottom) / 2;
  svg.append(svgNode("text", { x: 11, y: mid, "text-anchor": "middle", "font-size": 10.5, transform: `rotate(-90 11 ${mid})` }, yLabel));
}

function line(f, points, { color, dash = "", marker = "circle", label }) {
  const d = points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  f.svg.append(svgNode("path", { d, fill: "none", stroke: color, "stroke-width": 1.6, "stroke-dasharray": dash }));
  for (const [px, py, open] of points) {
    const shape = marker === "square"
      ? svgNode("rect", { x: px - 3, y: py - 3, width: 6, height: 6 })
      : svgNode("circle", { cx: px, cy: py, r: 3.2 });
    shape.setAttribute("fill", open ? "#fff" : color);
    shape.setAttribute("stroke", color);
    shape.setAttribute("stroke-width", 1.2);
    if (label) shape.append(svgNode("title", {}, label));
    f.svg.append(shape);
  }
}

function legend(items) {
  const list = element("ul", null, "legend");
  for (const [label, color, dash] of items) {
    const item = element("li");
    const mark = element("i", null, "swatch line");
    mark.style.borderTopColor = color;
    if (dash) mark.style.borderTopStyle = "dashed";
    mark.setAttribute("aria-hidden", "true");
    item.append(mark, label);
    list.append(item);
  }
  return list;
}

function card(question, answer, chart, note, link) {
  const article = element("article", null, "glance-card");
  article.append(element("h3", question), element("p", answer, "glance-answer"), chart);
  const foot = element("p", note, "context");
  if (link) {
    const a = element("a", link[0]);
    a.href = link[1];
    foot.append(" ", a);
  }
  article.append(foot);
  return article;
}

// Q1: share of jammed pilot cells found against jammer power, the U-Net and two baselines.
function findsTheJammer(pd) {
  const rows = pd.rows.filter(r => r.attack === "pilot");
  const jsrs = [...new Set(rows.map(r => r.jsr_db))].sort((a, b) => a - b);
  const f = frame();
  const x = v => f.left + (v - jsrs[0]) / (jsrs.at(-1) - jsrs[0]) * (f.right - f.left);
  const y = v => f.bottom - v * (f.bottom - f.top);
  axes(f, {
    x, y, xTicks: jsrs.filter((_, i) => i % 2 === 0), yTicks: [0, 0.25, 0.5, 0.75, 1],
    xLabel: "Jammer power relative to the user (dB)", yLabel: "Jammed pilot cells found",
    yFormat: v => `${v * 100}%`, xFormat: v => String(v).replace("-", "−"),
  });
  const series = [
    ["fused", RED, "", "U-Net (power, residual, EVM)"],
    ["block_test", GREEN, "4 2", "Block test (no learning)"],
    ["evm_threshold", CYAN, "2 2", "EVM threshold"],
  ];
  for (const [name, color, dash, label] of series) {
    const pts = rows.filter(r => r.detector === name).sort((a, b) => a.jsr_db - b.jsr_db)
      .map(r => [x(r.jsr_db), y(r.pd_mean)]);
    if (pts.length) line(f, pts, { color, dash, label });
  }
  const weakest = rows.find(r => r.detector === "fused" && r.jsr_db === jsrs[0]);
  const wrap = element("div");
  wrap.append(f.svg, legend(series.map(([, c, d, l]) => [l, c, d])));
  return card("Does it find the jammer?",
    weakest ? `Even when the jammer is ${String(-jsrs[0])} dB weaker than the user, the U-Net finds ${percent(weakest.pd_mean, 0)} of the pilot cells it hit; an EVM threshold finds none.` : "",
    wrap, "Pilot jamming, TDL-A and TDL-C test slots, thresholds set for 1% false alarms on validation cells.",
    ["Detection results", "results.html#detection"]);
}

// Q2: blocks lost against jammer power, standard receiver vs the U-Net map, one panel per attack,
// on a log axis; a point with no error is drawn hollow at the 1-in-N level of its sample.
function savesTheLink(curves) {
  const attacks = [["pilot", "Pilot"], ["data", "Data"], ["partial", "Partial-band"]];
  const rows = curves.rows.filter(r => Number.isFinite(r.jsr_db) && attacks.some(([k]) => k === r.attack));
  const jsrs = [...new Set(rows.map(r => r.jsr_db))].sort((a, b) => a - b);
  const grid = element("div", null, "glance-panels");
  const floor = 1e-4;
  for (const [key, title] of attacks) {
    const f = frame({ width: 200, height: 190, left: 40, right: 192, top: 18, bottom: 148 });
    const x = v => f.left + (v - jsrs[0]) / (jsrs.at(-1) - jsrs[0]) * (f.right - f.left);
    const y = v => f.bottom - (Math.log10(Math.max(v, floor)) - Math.log10(floor)) / -Math.log10(floor) * (f.bottom - f.top);
    axes(f, {
      x, y, xTicks: [jsrs[0], 0, jsrs.at(-1)].filter(v => v >= jsrs[0]), yTicks: [1e-4, 1e-3, 1e-2, 1e-1, 1],
      xLabel: "Jammer power (dB)", yLabel: key === "pilot" ? "Blocks lost" : "",
      yFormat: v => (v >= 0.01 ? `${v * 100}%` : `${v * 100}%`), xFormat: v => String(v).replace("-", "−"),
    });
    f.svg.append(svgNode("text", { x: (f.left + f.right) / 2, y: 12, "text-anchor": "middle", "font-size": 11, "font-weight": 600 }, title));
    for (const [rx, color, dash, label] of [["standard", INK, "4 2", "Standard receiver"], ["p_r_e_soft", RED, "", "With the U-Net map"]]) {
      const pts = rows.filter(r => r.attack === key && r.receiver === rx).sort((a, b) => a.jsr_db - b.jsr_db)
        .map(r => [x(r.jsr_db), y(r.block_errors ? r.bler : 1 / r.blocks), r.block_errors === 0]);
      if (pts.length) line(f, pts, { color, dash, label });
    }
    grid.append(f.svg);
  }
  const wrap = element("div");
  const worst = Math.max(...rows.filter(r => r.receiver === "standard").map(r => r.bler));
  const best = Math.max(...rows.filter(r => r.receiver === "p_r_e_soft").map(r => r.bler));
  wrap.append(grid, legend([["Standard receiver", INK, true], ["With the U-Net map", RED, false]]));
  return card("Does the map save the link?",
    `Without the map, up to ${Math.round(100 * worst)} of every 100 blocks are lost. Decoding the same slots again with the map loses at most ${percent(best, 2)}, fewer than 1 in 1,000.`,
    wrap, "TDL-C, 15 dB SNR, the user on 12 PRBs and the jammer on 4, known noise power, three detector seeds pooled. Hollow points: no error seen, drawn at one error in the sample.",
    ["Recovery results", "results.html#recovery"]);
}

// Q3: the smallest detection margin per SNR, as bars above (warns first) or below (too late) zero.
function warnsEarly(margin) {
  const rows = margin.rows.filter(r => r.channel === "C" && r.slot_fa_target === 0.01).sort((a, b) => b.snr_db - a.snr_db);
  const f = frame({ width: 320, height: 200 });
  const lo = Math.min(-8, ...rows.map(r => r.margin_db));
  const hi = Math.max(10, ...rows.map(r => r.margin_db));
  const y = v => f.top + (hi - v) / (hi - lo) * (f.bottom - f.top);
  const step = (f.right - f.left) / rows.length;
  for (const t of [-5, 0, 5, 10]) {
    f.svg.append(svgNode("line", { x1: f.left, x2: f.right, y1: y(t), y2: y(t), stroke: t === 0 ? "#000" : "#e6e6e6", "stroke-width": t === 0 ? 0.9 : 0.6 }));
    f.svg.append(svgNode("text", { x: f.left - 5, y: y(t) + 3.5, "text-anchor": "end", "font-size": 10 }, String(t).replace("-", "−")));
  }
  rows.forEach((r, i) => {
    const cx = f.left + step * (i + 0.5);
    const v = r.margin_db;
    const bar = svgNode("rect", {
      x: cx - 18, width: 36, y: Math.min(y(0), y(v)), height: Math.abs(y(v) - y(0)),
      fill: v >= 0 ? BLUE : RED,
    });
    bar.append(svgNode("title", {}, `${r.snr_db} dB SNR: ${v.toFixed(1)} dB`));
    f.svg.append(bar);
    f.svg.append(svgNode("text", { x: cx, y: v >= 0 ? y(v) - 4 : y(v) + 12, "text-anchor": "middle", "font-size": 10 }, `${v.toFixed(1).replace("-", "−")} dB`));
    f.svg.append(svgNode("text", { x: cx, y: f.bottom + 14, "text-anchor": "middle", "font-size": 10 }, `${r.snr_db} dB`));
  });
  f.svg.append(svgNode("text", { x: (f.left + f.right) / 2, y: f.bottom + 30, "text-anchor": "middle", "font-size": 10.5 }, "Signal-to-noise ratio of the link"));
  const mid = (f.top + f.bottom) / 2;
  f.svg.append(svgNode("text", { x: 11, y: mid, "text-anchor": "middle", "font-size": 10.5, transform: `rotate(-90 11 ${mid})` }, "Warning ahead (dB)"));
  f.svg.append(svgNode("text", { x: f.right - 4, y: f.top + 12, "text-anchor": "end", "font-size": 9.5, fill: BLUE }, "warns before errors"));
  f.svg.append(svgNode("text", { x: f.left + 4, y: f.bottom - 5, "text-anchor": "start", "font-size": 9.5, fill: RED }, "below zero: too late"));
  const at15 = rows.find(r => r.snr_db === 15);
  const at5 = rows.find(r => r.snr_db === 5);
  return card("Does it warn before the link breaks?",
    at15 && at5 ? `At 15 dB SNR the alarm comes ${at15.margin_db.toFixed(1)} dB of jammer power before 10% of blocks fail; at 5 dB SNR, where the link has little spare margin, it comes too late.` : "",
    f.svg, "Margin = jammer power at which 10% of blocks fail, minus jammer power at which 90% of jammed slots are flagged; smallest over pilot, data and partial-band jamming, TDL-C, 1% false alarms per slot.",
    ["Margin results", "results.html#margin"]);
}

// Q4: every pilot jammed: standard, U-Net map, U-Net map with data-aided passes.
function noCleanPilot(section) {
  const rows = section.rows.filter(r => r.attack === "pilot_whole_allocation" && r.snr_db === 15);
  const bars = [["standard", "Standard", INK, false], ["p_r_e_soft", "U-Net map", RED, false], ["p_r_e_soft_da", "U-Net map + data-aided", RED, true]];
  const jsrs = [...new Set(rows.map(r => r.jsr_db))].sort((a, b) => a - b);
  const f = frame({ width: 320, height: 200 });
  const y = v => f.bottom - v * (f.bottom - f.top);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    f.svg.append(svgNode("line", { x1: f.left, x2: f.right, y1: y(t), y2: y(t), stroke: t ? "#e6e6e6" : "#000", "stroke-width": t ? 0.6 : 0.9 }));
    f.svg.append(svgNode("text", { x: f.left - 5, y: y(t) + 3.5, "text-anchor": "end", "font-size": 10 }, `${t * 100}%`));
  }
  const group = (f.right - f.left) / jsrs.length;
  jsrs.forEach((jsr, g) => {
    const x0 = f.left + g * group + group / 2 - 45;
    bars.forEach(([rx, label, color, filled], i) => {
      const row = rows.find(r => r.jsr_db === jsr && r.receiver === rx);
      if (!row) return;
      const rect = svgNode("rect", { x: x0 + i * 32, y: y(row.bler), width: 26, height: Math.max(0.6, f.bottom - y(row.bler)), fill: filled ? color : "#fff", stroke: color, "stroke-width": 1.4 });
      rect.append(svgNode("title", {}, `${label}: ${percent(row.bler, 1)}`));
      f.svg.append(rect);
      f.svg.append(svgNode("text", { x: x0 + i * 32 + 13, y: y(row.bler) - 3, "text-anchor": "middle", "font-size": 9 }, percent(row.bler, 0)));
    });
    f.svg.append(svgNode("text", { x: x0 + 45, y: f.bottom + 14, "text-anchor": "middle", "font-size": 10 }, `${signedDb(jsr)}`));
  });
  f.svg.append(svgNode("text", { x: (f.left + f.right) / 2, y: f.bottom + 30, "text-anchor": "middle", "font-size": 10.5 }, "Jammer power relative to the user"));
  const mid = (f.top + f.bottom) / 2;
  f.svg.append(svgNode("text", { x: 11, y: mid, "text-anchor": "middle", "font-size": 10.5, transform: `rotate(-90 11 ${mid})` }, "Blocks lost"));
  const wrap = element("div");
  const items = element("ul", null, "legend");
  for (const [, label, color, filled] of bars) {
    const li = element("li");
    const mark = element("i", null, "swatch");
    mark.style.background = filled ? color : "#fff";
    mark.style.borderColor = color;
    li.append(mark, label);
    items.append(li);
  }
  wrap.append(f.svg, items);
  const at9 = k => rows.find(r => r.jsr_db === 9 && r.receiver === k);
  return card("And when every pilot is jammed?",
    at9("p_r_e_soft") && at9("p_r_e_soft_da") && at9("true_soft") ? `No pilot is clean, so even the true map leaves ${percent(at9("true_soft").bler, 0)} of blocks lost at +9 dB. Using the decoded data as extra pilots cuts the loss with the U-Net map from ${percent(at9("p_r_e_soft").bler, 0)} to ${percent(at9("p_r_e_soft_da").bler, 0)}, but does not remove it.` : "",
    wrap, "Every pilot of the 12-PRB allocation jammed, TDL-C, 15 dB SNR, known noise power, three detector seeds pooled.",
    ["No-clean-pilot results", "results.html#no-clean-pilot"]);
}

async function glance(host) {
  const data = await loadResults();
  const cards = [];
  if (data.recovery_curves) cards.push(savesTheLink(data.recovery_curves));
  cards.push(findsTheJammer(data.pd), warnsEarly(data.margin));
  if (data.data_aided) cards.push(noCleanPilot(data.data_aided));
  host.replaceChildren(...cards);
  host.setAttribute("aria-busy", "false");
}

const host = document.querySelector("#glance");
if (host) glance(host).catch(error => reportError(host, error));
