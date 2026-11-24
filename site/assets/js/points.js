// Views shared by the overview's teaser and animations: the user's resource grid with switchable
// layers, the equalized symbols of a receiver, and a constellation that morphs between two
// receivers. Every panel is drawn for the same 420 x 330 box, so panels side by side line up.
import { svgNode } from "./grid.js";

export const RED = "#cc0000";
export const BLUE = "#0033cc";
export const BOX = { width: 420, height: 330 };
const SUBCARRIERS_PER_PRB = 12;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Opacity that encodes how much the decoder is told to trust a bit (|LLR|), as on the other pages
export const reliabilityAlpha = llr => 0.12 + 0.83 * (1 - Math.exp(-(llr ?? 0) / 14));
export const PILOT = "#ecdca4";

// Grey of a cell whose bits the decoder trusts with mean |LLR| `llr`: near white for none, mid grey
// for full trust (light enough for the jam outlines and flags on top)
function trustGrey(llr) {
  const level = 1 - Math.exp(-(llr ?? 0) / 10);
  const v = Math.round(247 - (247 - 128) * level);
  return `rgb(${v},${v},${v})`;
}

// Equalized data symbols of one receiver (all of the allocation's PRBs, or only `prbs`), in the
// export's order, with their cell, bit reliability and whether the jammer hit that cell.
export function symbols(slot, receiver, prbs = null) {
  const rx = slot.receivers[receiver];
  const points = slot.x_hat[rx.x_hat];
  const llrs = slot.abs_llr[rx.abs_llr];
  const [start, width] = slot.metadata.alloc;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const rowLength = width * SUBCARRIERS_PER_PRB;
  const keep = prbs ? new Set(prbs) : null;
  const out = [];
  for (const [index, point] of points.entries()) {
    const prb = start + Math.floor((index % rowLength) / SUBCARRIERS_PER_PRB);
    if ((keep && !keep.has(prb)) || point.some(v => v == null)) continue;
    const ordinal = Math.floor(index / rowLength);
    const symbol = ordinal + Number(ordinal >= dmrs);
    const llr = llrs[index] ?? 0;
    out.push({
      index, prb, symbol, x: point[0], y: point[1], llr, alpha: reliabilityAlpha(llr),
      jammed: Boolean(slot.grids.truth[prb][symbol]),
    });
  }
  return out;
}

const popcount = digit => (digit & 1) + ((digit >> 1) & 1);

// Wrong coded bits per cell ("prb,symbol" -> count; two bits per subcarrier, so up to 24 in a
// cell) before decoding: the signs of the receiver's LLRs against the bits sent. Null without
// exported bits.
export function cellErrors(slot, receiver) {
  const raw = slot.bits?.[slot.receivers[receiver]?.raw];
  if (!raw || !slot.sent) return null;
  const errors = new Map();
  for (const p of symbols(slot, receiver)) {
    const key = `${p.prb},${p.symbol}`;
    errors.set(key, (errors.get(key) ?? 0) + popcount(Number(raw[p.index]) ^ Number(slot.sent[p.index])));
  }
  return errors;
}

// What decoding does with the bits: one small grid per receiver (the user's PRBs by OFDM symbols).
// Before decoding each cell is darker the more of its bits are wrong. After decoding, a receiver
// whose CRC passes has recovered every bit (the dark cells clear), and one whose CRC fails has
// lost the whole block (hatched). `flags` (one decisions grid or null per receiver) marks the cells
// a receiver's map tells the decoder to doubt. `set(stage, animate)` switches "before" / "after".
export function decodingPanel(slot, names, labels, { width = BOX.width, height = 250, flags = [] } = {}) {
  const [start, count] = slot.metadata.alloc;
  const cols = slot.grids.allocation[0].length;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const left = 30;
  const gap = 22;
  const top = 30;
  const bottom = 46;
  const gridW = (width - left - 6 - gap * (names.length - 1)) / names.length;
  const cw = gridW / cols;
  const ch = (height - top - bottom) / count;
  const gridH = ch * count;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "viz decoding-panel", role: "img" });
  const defs = svgNode("defs");
  const hatch = svgNode("pattern", { id: `lost-${Math.random().toString(36).slice(2)}`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  hatch.append(svgNode("rect", { width: 6, height: 6, fill: "#d9d9d9" }), svgNode("line", { x1: 1, x2: 1, y1: 0, y2: 6, stroke: "#8c8c8c", "stroke-width": 1.6 }));
  defs.append(hatch);
  svg.append(defs);
  const mid = top + gridH / 2;
  svg.append(svgNode("text", { x: 10, y: mid, "text-anchor": "middle", "font-size": 10, transform: `rotate(-90 10 ${mid})` }, "PRB"));
  for (const prb of [start, start + count - 1]) {
    svg.append(svgNode("text", { x: left - 4, y: top + (start + count - 1 - prb) * ch + ch / 2 + 3.5, "text-anchor": "end", "font-size": 9 }, prb));
  }
  const panels = names.map((name, n) => {
    const x0 = left + n * (gridW + gap);
    const errors = cellErrors(slot, name) ?? new Map();
    const passed = slot.receivers[name].crc;
    const wrong = [...errors.values()].reduce((a, b) => a + b, 0);
    svg.append(svgNode("text", { x: x0 + gridW / 2, y: top - 10, "text-anchor": "middle", "font-size": 12, "font-weight": 700 }, labels[n]));
    const base = svgNode("g");
    const errorLayer = svgNode("g", { opacity: 1 });
    const jam = svgNode("g");
    const doubted = svgNode("g", { opacity: 1 });
    const decisions = flags[n];
    for (let prb = start; prb < start + count; prb++) {
      for (let s = 0; s < cols; s++) {
        const x = x0 + s * cw;
        const y = top + (start + count - 1 - prb) * ch;
        base.append(svgNode("rect", { x, y, width: cw, height: ch, fill: s === dmrs ? PILOT : "#fff", stroke: "#e0e0e0", "stroke-width": 0.6 }));
        const bad = errors.get(`${prb},${s}`) ?? 0;
        if (bad) {
          errorLayer.append(svgNode("rect", { x: x + 0.5, y: y + 0.5, width: cw - 1, height: ch - 1, fill: "#1a1a1a", opacity: Math.min(1, 0.25 + bad / 10) }));
        }
        if (slot.grids.truth[prb][s]) {
          jam.append(svgNode("rect", { x: x + 0.4, y: y + 0.4, width: cw - 0.8, height: ch - 0.8, fill: "none", stroke: RED, "stroke-width": 1 }));
        }
        if (decisions?.[prb][s]) {
          doubted.append(svgNode("circle", { cx: x + cw / 2, cy: y + ch / 2, r: Math.min(cw, ch) * 0.22, fill: BLUE, stroke: "#fff", "stroke-width": 0.8 }));
        }
      }
    }
    const lost = svgNode("rect", { x: x0, y: top, width: gridW, height: gridH, fill: `url(#${hatch.id})`, opacity: 0 });
    const frame = svgNode("rect", { x: x0, y: top, width: gridW, height: gridH, fill: "none", stroke: "#000", "stroke-width": 0.8 });
    // A stamp across the grid after decoding: "all bits right" or "block lost"
    const stamp = svgNode("g", { opacity: 0 });
    const stampText = passed ? "All bits right" : "Block lost";
    const stampW = 12 + stampText.length * 6.6;
    stamp.append(
      svgNode("rect", { x: x0 + gridW / 2 - stampW / 2, y: mid - 11, width: stampW, height: 22, fill: "#fff", stroke: passed ? "#006600" : RED, "stroke-width": 1.5 }),
      svgNode("text", { x: x0 + gridW / 2, y: mid + 4.5, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: passed ? "#006600" : RED }, stampText),
    );
    svg.append(base, errorLayer, jam, doubted, lost, frame, stamp);
    const y1 = top + gridH + 16;
    const before = svgNode("text", { x: x0 + gridW / 2, y: y1, "text-anchor": "middle", "font-size": 11 },
      wrong ? `${wrong.toLocaleString()} wrong bits` : "no wrong bits");
    const after = svgNode("text", { x: x0 + gridW / 2, y: y1, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: passed ? "#006600" : RED, opacity: 0 },
      passed ? "CRC pass" : "CRC fail, sent again");
    const trust = decisions ? "flagged cells' bits doubted" : "every bit trusted";
    const where = svgNode("text", { x: x0 + gridW / 2, y: y1 + 14, "text-anchor": "middle", "font-size": 10, fill: "#555" }, trust);
    svg.append(before, after, where);
    return { name, passed, wrong, errorLayer, doubted, lost, stamp, before, after, where, trust };
  });
  let token = 0;
  function set(stage, animate = false) {
    const mine = ++token;
    const alive = () => mine === token;
    const done = stage === "after";
    const fade = (node, to, ms, delay = 0) => {
      if (!animate) return node.setAttribute("opacity", to);
      return new Promise(r => setTimeout(r, delay)).then(() => (alive() ? fadeTo(node, to, ms, alive) : null));
    };
    const jobs = [];
    for (const p of panels) {
      jobs.push(fade(p.errorLayer, done && p.passed ? 0 : 1, 1200));
      jobs.push(fade(p.lost, done && !p.passed ? 0.85 : 0, 900));
      jobs.push(fade(p.stamp, done ? 1 : 0, 400, done ? 900 : 0));
      jobs.push(fade(p.before, done ? 0 : 1, 300));
      jobs.push(fade(p.after, done ? 1 : 0, 300, done ? 900 : 0));
      jobs.push(fade(p.doubted, done ? 0 : 1, 600));
      p.where.textContent = done ? "after decoding" : p.trust;
    }
    svg.setAttribute("aria-label", panels.map((p, i) => (done
      ? `${labels[i]} after decoding, ${p.passed ? "CRC pass, every bit right" : "CRC fail, the block is lost and sent again"}`
      : `${labels[i]} before decoding, ${p.wrong} wrong bits`)).join(". "));
    return Promise.all(jobs.filter(Boolean));
  }
  set("before");
  return { svg, set, panels };
}

// Mean bit reliability per cell ("prb,symbol" -> mean |LLR|) of a receiver's symbols
export function cellTrust(points) {
  const sums = new Map();
  for (const p of points) {
    const key = `${p.prb},${p.symbol}`;
    const [s, n] = sums.get(key) ?? [0, 0];
    sums.set(key, [s + p.llr, n + 1]);
  }
  return new Map([...sums].map(([key, [s, n]]) => [key, s / n]));
}

export const mean = values => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
export function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2 : NaN;
}

// Tween `draw(t)` from t = 0 to 1 with an ease-in-out; instant with reduced motion. It stops
// without drawing again as soon as `alive()` turns false (the viewer moved to another step).
export function tween(duration, draw, alive = () => true) {
  if (reduceMotion || duration <= 0) {
    if (alive()) draw(1);
    return Promise.resolve();
  }
  return new Promise(resolve => {
    const begin = performance.now();
    const frame = now => {
      if (!alive()) return resolve();
      const k = Math.min(1, (now - begin) / duration);
      draw(k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);
      if (k < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

export function fadeTo(node, to, duration = 450, alive = () => true) {
  const from = Number(node.getAttribute("opacity") ?? 1);
  if (from === to) return Promise.resolve();
  return tween(duration, t => node.setAttribute("opacity", String(from + (to - from) * t)), alive);
}

// A share of the allocation in words ("a third", "half"), or as a percentage
export function shareText(fraction) {
  const words = [[1, "all"], [3 / 4, "three quarters"], [2 / 3, "two thirds"], [1 / 2, "half"],
    [1 / 3, "a third"], [1 / 4, "a quarter"], [1 / 6, "a sixth"], [0, "none"]];
  const match = words.find(([f]) => Math.abs(f - fraction) < 1e-9);
  return match ? match[1] : `${Math.round(100 * fraction)}%`;
}

// The user's allocation (PRBs bottom to top, OFDM symbols left to right) with layers the steps
// switch on: jammed cells, detector flags, and how much each receiver trusts each cell's bits.
export function allocationGrid(slot, { trust = {}, label = "The user's resource grid" } = {}) {
  const { width, height } = BOX;
  const [start, count] = slot.metadata.alloc;
  const cols = slot.grids.allocation[0].length;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const x0 = 50;
  const y0 = 24;
  const cw = (width - x0 - 8) / cols;
  const ch = (height - y0 - 34) / count;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "viz", role: "img", "aria-label": label });
  const cell = (prb, s) => ({ x: x0 + s * cw, y: y0 + (start + count - 1 - prb) * ch });
  const layer = (name, opacity = 0) => svgNode("g", { class: `layer ${name}`, opacity });

  const base = layer("base", 1);
  const layers = { jam: layer("jam"), flags: layer("flags") };
  for (const name of Object.keys(trust)) layers[`trust_${name}`] = layer(`trust ${name}`);
  for (let prb = start; prb < start + count; prb++) {
    for (let s = 0; s < cols; s++) {
      const { x, y } = cell(prb, s);
      base.append(svgNode("rect", {
        x, y, width: cw, height: ch, fill: s === dmrs ? PILOT : "#dedede", stroke: "#fff", "stroke-width": 1,
      }));
      for (const [name, values] of Object.entries(trust)) {
        const llr = values.get(`${prb},${s}`);
        if (llr == null) continue;
        layers[`trust_${name}`].append(svgNode("rect", {
          x, y, width: cw, height: ch, fill: trustGrey(llr), stroke: "#fff", "stroke-width": 1,
        }));
      }
      if (slot.grids.truth[prb][s]) {
        layers.jam.append(svgNode("rect", {
          x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: "none", stroke: RED, "stroke-width": 2,
        }));
      }
      if (slot.detectors.fused.decisions[prb][s]) {
        // A dot, so the cell's own shading (e.g. how much its bits are trusted) stays visible
        layers.flags.append(svgNode("circle", {
          cx: x + cw / 2, cy: y + ch / 2, r: Math.min(cw, ch) * 0.24, fill: BLUE, stroke: "#fff", "stroke-width": 1,
        }));
      }
    }
    if (count <= 16 || prb % 2 === 0) {
      svg.append(svgNode("text", { x: x0 - 5, y: cell(prb, 0).y + ch / 2 + 3.5, "text-anchor": "end", "font-size": 10 }, prb));
    }
  }
  for (let s = 0; s < cols; s++) {
    svg.append(svgNode("text", {
      x: x0 + s * cw + cw / 2, y: y0 - 6, "text-anchor": "middle", "font-size": 10, "font-weight": s === dmrs ? 700 : 400,
    }, s));
  }
  svg.append(svgNode("text", { x: x0 + (cols * cw) / 2, y: height - 8, "text-anchor": "middle", "font-size": 11 },
    `OFDM symbol (time); pilots on symbol ${dmrs}`));
  const mid = y0 + (count * ch) / 2;
  svg.append(svgNode("text", { x: 12, y: mid, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${mid})` },
    "PRB (frequency)"));
  // Trust layers go under the jam outlines and the flags
  const trustLayers = Object.keys(trust).map(name => layers[`trust_${name}`]);
  svg.append(base, ...trustLayers, layers.jam, layers.flags);
  svg.append(svgNode("rect", { x: x0, y: y0, width: cols * cw, height: count * ch, fill: "none", stroke: "#000", "stroke-width": 0.8 }));
  return { svg, layers, cell, cw, ch };
}

// A constellation in the shared box: a square plot area, axes at +-limit
export function constellationView(canvas, limit = 2.5) {
  const { width, height } = BOX;
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * density);
  canvas.height = Math.round(height * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(density, density);
  const size = height - 60;
  const left = (width - size) / 2 + 8;
  const top = 22;
  const scale = size / (2 * limit);
  return {
    ctx, width, height, left, top, size, limit,
    px: v => left + size / 2 + scale * v,
    py: v => top + size / 2 - scale * v,
  };
}

// One frame of the points morphing from `from` (t = 0) to `to` (t = 1), with the four QPSK points
// and the number of points that lie outside the axes in this frame.
export function drawFrame(view, from, to, t) {
  const { ctx, width, height, left, top, size, limit, px, py } = view;
  const serif = '"STIX Two Text", "Times New Roman", Times, serif';
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e6e6e6";
  ctx.lineWidth = 0.6;
  const ticks = [-limit, -limit / 2, 0, limit / 2, limit];
  for (const tick of ticks.slice(1, -1)) {
    ctx.beginPath();
    ctx.moveTo(px(tick), top);
    ctx.lineTo(px(tick), top + size);
    ctx.moveTo(left, py(tick));
    ctx.lineTo(left + size, py(tick));
    ctx.stroke();
  }
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 0.8;
  ctx.strokeRect(left, top, size, size);
  ctx.fillStyle = "#1a1a1a";
  ctx.font = `11px ${serif}`;
  for (const tick of [-limit, 0, limit]) {
    const label = String(tick).replace("-", "−");
    ctx.textAlign = "center";
    ctx.fillText(label, px(tick), top + size + 14);
    ctx.textAlign = "right";
    ctx.fillText(label, left - 5, py(tick) + 4);
  }
  ctx.textAlign = "center";
  ctx.font = `italic 12px ${serif}`;
  ctx.fillText("In-phase", left + size / 2, top + size + 30);
  ctx.save();
  ctx.translate(left - 30, top + size / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Quadrature", 0, 0);
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, size, size);
  ctx.clip();
  const ease = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  let outside = 0;
  // Clean points first, so the jammed ones stay visible on top
  for (const pass of [false, true]) {
    for (let i = 0; i < from.length; i++) {
      const a = from[i];
      if (a.jammed !== pass) continue;
      const b = to[i];
      const x = a.x + (b.x - a.x) * ease;
      const y = a.y + (b.y - a.y) * ease;
      if (Math.abs(x) > limit || Math.abs(y) > limit) outside++;
      ctx.globalAlpha = a.alpha + (b.alpha - a.alpha) * ease;
      ctx.fillStyle = a.jammed ? RED : BLUE;
      ctx.beginPath();
      ctx.arc(px(x), py(y), 1.6, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.2;
  for (const [x, y] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const cx = px(x / Math.SQRT2);
    const cy = py(y / Math.SQRT2);
    ctx.beginPath();
    ctx.moveTo(cx - 6, cy);
    ctx.lineTo(cx + 6, cy);
    ctx.moveTo(cx, cy - 6);
    ctx.lineTo(cx, cy + 6);
    ctx.stroke();
  }
  ctx.fillStyle = "#555";
  ctx.font = `11px ${serif}`;
  ctx.textAlign = "right";
  ctx.fillText(outside ? `${outside} of ${from.length} points outside the axes` : `${from.length} points`,
    left + size, top - 7);
  return outside;
}
