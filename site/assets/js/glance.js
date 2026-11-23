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
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Each chart registers how to draw itself in; a card plays its charts once when it scrolls into
// view (with reduced motion they are simply drawn complete).
let growing = [];
function tween(duration, draw) {
  return new Promise(resolve => {
    const begin = performance.now();
    const frame = now => {
      const k = Math.min(1, (now - begin) / duration);
      draw(1 - (1 - k) ** 3);
      if (k < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

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
  const pathOf = pts => pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const path = svgNode("path", { d: pathOf(points), fill: "none", stroke: color, "stroke-width": 1.6, "stroke-dasharray": dash });
  f.svg.append(path);
  const shapes = points.map(([px, py, open]) => {
    const shape = marker === "square"
      ? svgNode("rect", { x: px - 3, y: py - 3, width: 6, height: 6 })
      : svgNode("circle", { cx: px, cy: py, r: 3.2 });
    shape.setAttribute("fill", open ? "#fff" : color);
    shape.setAttribute("stroke", color);
    shape.setAttribute("stroke-width", 1.2);
    if (label) shape.append(svgNode("title", {}, label));
    f.svg.append(shape);
    return shape;
  });
  growing.push(() => {
    // Draw the polyline up to a growing fraction of its length
    const seg = points.slice(1).map(([x, y], i) => Math.hypot(x - points[i][0], y - points[i][1]));
    const total = seg.reduce((a, b) => a + b, 0) || 1;
    shapes.forEach(sh => sh.setAttribute("opacity", 0));
    return tween(1100, t => {
      let left = t * total;
      const pts = [points[0]];
      for (let i = 0; i < seg.length && left > 0; i++) {
        const k = Math.min(1, left / (seg[i] || 1));
        const [x0, y0] = points[i];
        const [x1, y1] = points[i + 1];
        pts.push([x0 + (x1 - x0) * k, y0 + (y1 - y0) * k]);
        left -= seg[i];
      }
      path.setAttribute("d", pathOf(pts));
      let reached = 0;
      for (let i = 0; i < shapes.length; i++) {
        if (i === 0 || seg.slice(0, i).reduce((a, b) => a + b, 0) <= t * total + 0.5) reached = i;
        shapes[i].setAttribute("opacity", i <= reached ? 1 : 0);
      }
    });
  });
}

// A bar that grows from the baseline `base` to its height
function bar(f, attrs, base) {
  const rect = svgNode("rect", attrs);
  f.svg.append(rect);
  const { y, height } = attrs;
  growing.push(() => tween(900, t => {
    const top = y < base ? base - (base - y) * t : y;
    rect.setAttribute("y", Math.min(top, base));
    rect.setAttribute("height", Math.max(0.6, height * t));
  }));
  return rect;
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
  article.grow = growing;
  growing = [];
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

// Q1: share of jammed cells found against jammer power, the U-Net and two baselines, with a
// switch between the attacks.
const ATTACK_NAMES = { partial: "Partial-band", pilot: "Pilot", data: "Data" };
function findsTheJammer(pd) {
  const series = [
    ["fused", RED, "", "U-Net (power, residual, EVM)"],
    ["block_test", GREEN, "4 2", "Block test (no learning)"],
    ["evm_threshold", CYAN, "2 2", "EVM threshold"],
  ];
  const chartHost = element("div");
  const answer = element("p", null, "glance-answer");
  const note = element("span");
  function draw(attack, animate) {
    const rows = pd.rows.filter(r => r.attack === attack);
    const jsrs = [...new Set(rows.map(r => r.jsr_db))].sort((a, b) => a - b);
    const f = frame();
    const x = v => f.left + (v - jsrs[0]) / (jsrs.at(-1) - jsrs[0]) * (f.right - f.left);
    const y = v => f.bottom - v * (f.bottom - f.top);
    axes(f, {
      x, y, xTicks: jsrs.filter((_, i) => i % 2 === 0), yTicks: [0, 0.25, 0.5, 0.75, 1],
      xLabel: "Jammer power relative to the user (dB)", yLabel: "Jammed cells found",
      yFormat: v => `${v * 100}%`, xFormat: v => String(v).replace("-", "−"),
    });
    for (const [name, color, dash, label] of series) {
      const pts = rows.filter(r => r.detector === name).sort((a, b) => a.jsr_db - b.jsr_db)
        .map(r => [x(r.jsr_db), y(r.pd_mean)]);
      if (pts.length) line(f, pts, { color, dash, label });
    }
    const runs = growing;
    growing = [];
    chartHost.replaceChildren(f.svg);
    if (animate && !reduceMotion) runs.forEach(run => run());
    const at = name => rows.find(r => r.detector === name && r.jsr_db === jsrs[0]);
    const what = attack === "pilot" ? "pilot cells" : "cells";
    answer.textContent = at("fused") && at("evm_threshold") && at("block_test")
      ? `Even when the jammer is ${String(-jsrs[0])} dB weaker than the user, the U-Net finds ${percent(at("fused").pd_mean, 0)} of the jammed ${what}; the block test finds ${percent(at("block_test").pd_mean, 0)} and an EVM threshold ${at("evm_threshold").pd_mean < 0.005 ? "almost none" : percent(at("evm_threshold").pd_mean, 0)}.`
      : "";
    note.textContent = `${ATTACK_NAMES[attack]} jamming, TDL-A and TDL-C test slots, jammed cells inside the allocation; thresholds set for 1% false alarms on validation cells.`;
    return runs;
  }
  const toggle = element("div", null, "toggle small");
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", "Attack");
  const buttons = Object.entries(ATTACK_NAMES).map(([key, name]) => {
    const b = element("button", name);
    b.type = "button";
    b.addEventListener("click", () => {
      buttons.forEach(o => o.setAttribute("aria-pressed", String(o === b)));
      draw(key, true);
    });
    return b;
  });
  toggle.append(...buttons);
  buttons[0].setAttribute("aria-pressed", "true");
  const initial = draw("partial", false);
  const wrap = element("div");
  wrap.append(toggle, chartHost, legend(series.map(([, c, d, l]) => [l, c, d])));
  growing = initial;  // the card plays the first chart when it scrolls into view
  const article = card("Does it find the jammer?", "", wrap, "", ["Detection results", "results.html#detection"]);
  article.querySelector(".glance-answer").replaceWith(answer);
  article.querySelector(".context").prepend(note, " ");
  return article;
}

// Q2: blocks lost against jammer power, standard receiver vs the U-Net map, one panel per attack,
// on a log axis; a point with no error is drawn hollow at the 1-in-N level of its sample.
function savesTheLink(curves) {
  const attacks = [["partial", "Partial-band"], ["pilot", "Pilot"], ["data", "Data"]];
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
      xLabel: "Jammer power (dB)", yLabel: key === attacks[0][0] ? "Blocks lost" : "",
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
    `Without the map, up to ${Math.round(100 * worst)} of every 100 blocks are lost. Decoding again with the map loses at most ${percent(best, 2)}, fewer than 1 in 1,000.`,
    wrap, "TDL-C, 15 dB SNR, the user on 12 PRBs and the jammer on 4 of them (a third of its pilots and, for partial-band jamming, a third of its data cells), known noise power, three detector seeds pooled. Hollow points: no error seen, drawn at one error in the sample.",
    ["Recovery results", "results.html#recovery"]);
}

// Q2b: blocks lost against the jammer's width (2-12 of the user's 12 PRBs) at one jammer power,
// with switches for the attack and the power.
function howWide(sweep) {
  const rows = sweep.rows;
  const widths = [...new Set(rows.map(r => r.jam_prb))].sort((a, b) => a - b);
  const jsrs = [...new Set(rows.map(r => r.jsr_db))].sort((a, b) => a - b).filter(j => j > 0);
  const total = sweep.alloc_prb;
  const series = [
    ["standard", INK, "4 2", "Standard receiver", false],
    ["p_r_e_soft", RED, "", "With the U-Net map", false],
    ["p_r_e_soft_da", RED, "2 2", "U-Net map + data-aided passes", true],
  ];
  const chartHost = element("div");
  const answer = element("p", null, "glance-answer");
  const note = element("span");
  let attack = "partial";
  let jsr = jsrs.includes(3) ? 3 : jsrs[0];
  const floor = 1e-4;
  function draw(animate) {
    const f = frame({ width: 320, height: 210 });
    const x = v => f.left + (v - widths[0]) / (widths.at(-1) - widths[0]) * (f.right - f.left);
    const y = v => f.bottom - (Math.log10(Math.max(v, floor)) - Math.log10(floor)) / -Math.log10(floor) * (f.bottom - f.top);
    axes(f, {
      x, y, xTicks: widths, yTicks: [1e-4, 1e-3, 1e-2, 1e-1, 1],
      xLabel: `Jammed PRBs of the user's ${total}`, yLabel: "Blocks lost", yFormat: v => `${v * 100}%`,
    });
    for (const [rx, color, dash, label, open] of series) {
      const pts = rows.filter(r => r.attack === attack && r.jsr_db === jsr && r.receiver === rx)
        .sort((a, b) => a.jam_prb - b.jam_prb)
        .map(r => [x(r.jam_prb), y(r.block_errors ? r.bler : 1 / r.blocks), open || r.block_errors === 0]);
      if (pts.length) line(f, pts, { color, dash, label, marker: open ? "square" : "circle" });
    }
    const runs = growing;
    growing = [];
    chartHost.replaceChildren(f.svg);
    if (animate && !reduceMotion) runs.forEach(run => run());
    const at = (rx, w) => rows.find(r => r.attack === attack && r.jsr_db === jsr && r.receiver === rx && r.jam_prb === w);
    // The widest jammer up to which every narrower one also leaves the map-aware receiver under 1%
    let widest = null;
    for (const w of widths) {
      if (!(at("p_r_e_soft", w)?.bler < 0.01)) break;
      widest = w;
    }
    const narrow = at("standard", widths[0]);
    const what = attack === "pilot" ? "pilots" : attack === "data" ? "data cells" : "pilots and data cells";
    const why = attack === "pilot"
      ? "Wider than that, fewer clean pilots are left to carry the channel estimate across the jammed band."
      : "Wider than that, the clean cells soon carry too few coded bits for the code (rate about 0.37) to fill in the jammed ones.";
    answer.textContent = widest === null
      ? "At this jammer power even the narrowest jammer costs the map-aware receiver 1% of blocks or more."
      : widest === widths.at(-1)
        ? `Even with the jammer on all ${total} PRBs, the map keeps the loss under 1% at this jammer power.`
        : `With the jammer on up to ${widest} of the ${total} PRBs (${Math.round(100 * widest / total)}% of the ${what}), the map keeps the loss under 1%${narrow ? `; without it, ${percent(narrow.bler, 0)} of blocks are lost already at ${widths[0]} PRBs` : ""}. ${why}`;
    note.textContent = `${ATTACK_NAMES[attack]} jamming at ${signedDb(jsr)}, TDL-C, 15 dB SNR, the user on ${total} PRBs, known noise power, three detector seeds pooled.`;
    return runs;
  }
  const controls = element("div", null, "glance-controls");
  const switchOf = (labelText, entries, current, set) => {
    const group = element("div", null, "toggle small");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", labelText);
    const buttons = entries.map(([value, name]) => {
      const b = element("button", name);
      b.type = "button";
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => {
        buttons.forEach(o => o.setAttribute("aria-pressed", String(o === b)));
        set(value);
        draw(true);
      });
      return b;
    });
    group.append(...buttons);
    return group;
  };
  controls.append(
    switchOf("Attack", Object.entries(ATTACK_NAMES), attack, v => { attack = v; }),
    switchOf("Jammer power", jsrs.map(j => [j, signedDb(j)]), jsr, v => { jsr = v; }),
  );
  const initial = draw(false);
  const wrap = element("div");
  wrap.append(controls, chartHost, legend(series.map(([, c, d, l]) => [l, c, d])));
  growing = initial;
  const article = card("How wide can the jammer be?", "", wrap, "", ["Width results", "results.html#width"]);
  article.querySelector(".glance-answer").replaceWith(answer);
  article.querySelector(".context").prepend(note, " ");
  return article;
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
    const rect = bar(f, {
      x: cx - 18, width: 36, y: Math.min(y(0), y(v)), height: Math.abs(y(v) - y(0)),
      fill: v >= 0 ? BLUE : RED,
    }, y(0));
    rect.append(svgNode("title", {}, `${r.snr_db} dB SNR: ${v.toFixed(1)} dB`));
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
      const rect = bar(f, { x: x0 + i * 32, y: y(row.bler), width: 26, height: Math.max(0.6, f.bottom - y(row.bler)), fill: filled ? color : "#fff", stroke: color, "stroke-width": 1.4 }, f.bottom);
      rect.append(svgNode("title", {}, `${label}: ${percent(row.bler, 1)}`));
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

// Q5: a jammer the detector never saw: BLER under hopping jamming with the per-PRB estimate, the
// refining passes and the true map, and the F1 of the map before and after decoding.
function unseenJammer(recovery, after) {
  const rows = recovery.rows.filter(r => r.jammer === "hopping" && r.jsr_db === 9);
  const bars = [["p_r_e_soft", "U-Net map, per PRB", RED, false], ["p_r_e_full", "U-Net map, refining passes", RED, true], ["true_soft", "True map", INK, false]];
  const attacks = [["data", "Data"], ["partial", "Partial-band"]];
  const f = frame({ width: 320, height: 200 });
  const y = v => f.bottom - Math.min(v / 0.4, 1) * (f.bottom - f.top);
  for (const t of [0, 0.1, 0.2, 0.3, 0.4]) {
    f.svg.append(svgNode("line", { x1: f.left, x2: f.right, y1: y(t), y2: y(t), stroke: t ? "#e6e6e6" : "#000", "stroke-width": t ? 0.6 : 0.9 }));
    f.svg.append(svgNode("text", { x: f.left - 5, y: y(t) + 3.5, "text-anchor": "end", "font-size": 10 }, `${Math.round(t * 100)}%`));
  }
  const group = (f.right - f.left) / attacks.length;
  attacks.forEach(([attack, name], g) => {
    const x0 = f.left + g * group + group / 2 - 45;
    bars.forEach(([rx, label, color, filled], i) => {
      const row = rows.find(r => r.attack === attack && r.receiver === rx);
      if (!row) return;
      const rect = bar(f, { x: x0 + i * 32, y: y(row.bler), width: 26, height: Math.max(0.6, f.bottom - y(row.bler)), fill: filled ? color : "#fff", stroke: color, "stroke-width": 1.4 }, f.bottom);
      rect.append(svgNode("title", {}, `${label}: ${percent(row.bler, 2)}`));
      f.svg.append(svgNode("text", { x: x0 + i * 32 + 13, y: y(row.bler) - 3, "text-anchor": "middle", "font-size": 9 }, percent(row.bler, row.bler < 0.01 ? 1 : 0)));
    });
    f.svg.append(svgNode("text", { x: x0 + 45, y: f.bottom + 14, "text-anchor": "middle", "font-size": 10 }, name));
  });
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
  const at = (attack, rx) => rows.find(r => r.attack === attack && r.receiver === rx);
  const f1 = (grid, map) => after?.rows.find(r => r.split === grid && r.map === map)?.f1_mean;
  const located = f1("hopjam", "unet") != null && f1("hopjam", "post") != null
    ? ` Once the slot decodes, tests on the decoded data raise the map's F1 score from ${f1("hopjam", "unet").toFixed(2)} to ${f1("hopjam", "post").toFixed(2)}.` : "";
  return card("And a jammer the detector never saw?",
    at("data", "p_r_e_soft") && at("data", "p_r_e_full") ? `A jammer that hops between PRBs from symbol to symbol. Estimating its power cell by cell and letting the decoder refine the map cuts the loss from ${percent(at("data", "p_r_e_soft").bler, 0)} to ${percent(at("data", "p_r_e_full").bler, 0)}, as with the true map.${located}` : "",
    wrap, "+9 dB jammer power, 4 PRBs that hop over the carrier, TDL-C, 15 dB SNR, three detector seeds pooled. The U-Net was trained only on jammers that fill PRB rows.",
    ["Recovery results", "results.html#recovery"]);
}

async function glance(host) {
  const data = await loadResults();
  const cards = [];
  if (data.recovery_curves) cards.push(savesTheLink(data.recovery_curves));
  cards.push(findsTheJammer(data.pd));
  if (data.width_sweep) cards.push(howWide(data.width_sweep));
  cards.push(warnsEarly(data.margin));
  if (data.data_aided) cards.push(noCleanPilot(data.data_aided));
  if (data.recovery.rows.some(r => r.receiver === "p_r_e_full")) cards.push(unseenJammer(data.recovery, data.after_decoding));
  host.replaceChildren(...cards);
  host.setAttribute("aria-busy", "false");
  if (reduceMotion) return;
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      entry.target.grow?.forEach(run => run());
    }
  }, { threshold: 0.35 });
  cards.forEach(c => observer.observe(c));
}

const host = document.querySelector("#glance");
if (host) glance(host).catch(error => reportError(host, error));
