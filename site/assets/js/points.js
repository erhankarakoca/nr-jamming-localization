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
