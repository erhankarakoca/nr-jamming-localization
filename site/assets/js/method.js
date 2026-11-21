// "How it works": the method animated step by step on one recorded slot, and the data-aided
// passes animated on a slot whose every pilot is jammed. Each step is a state of two panels (the
// resource grid on the left, the channel or the symbols on the right); moving to a step shows all
// earlier states at once and animates only the new one. With reduced motion, states just switch.
import { conditions, loadCase, selectionText } from "./data.js";
import { affectedPRBs, crcBadge } from "./constellation.js";
import { featureScale, heatColor, svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";
import { constellationCanvas, drawFrame, symbols } from "./points.js";

const RED = "#cc0000";
const BLUE = "#0033cc";
const GREY = "#9a9a9a";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const abs = ([re, im]) => Math.hypot(re, im);
const sqErr = (est, ref, ks) => ks.reduce((sum, k) => sum + (est[k][0] - ref[k][0]) ** 2 + (est[k][1] - ref[k][1]) ** 2, 0) / ks.length;

function tween(duration, draw) {
  if (reduceMotion || duration <= 0) {
    draw(1);
    return Promise.resolve();
  }
  return new Promise(resolve => {
    const begin = performance.now();
    const frame = now => {
      const k = Math.min(1, (now - begin) / duration);
      draw(k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);
      if (k < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

// The allocation's resource grid with layers that the steps switch on: jammed cells, the three
// feature maps, and the U-Net flags.
function gridPanel(slot) {
  const [start, count] = slot.metadata.alloc;
  const cols = slot.grids.allocation[0].length;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const cw = 20;
  const ch = 11;
  const x0 = 34;
  const y0 = 16;
  const width = x0 + cols * cw + 6;
  const height = y0 + count * ch + 30;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "anim-grid", role: "img",
    "aria-label": "The user's resource grid: PRBs by OFDM symbols" });
  const layer = name => {
    const g = svgNode("g", { class: `layer ${name}`, opacity: 0 });
    return g;
  };
  const base = svgNode("g");
  const features = { power_db: layer("power"), residual_db: layer("residual"), evm_db: layer("evm") };
  const jam = layer("jam");
  const flags = layer("flags");
  for (let r = 0; r < count; r++) {
    const prb = start + count - 1 - r;
    for (let s = 0; s < cols; s++) {
      const x = x0 + s * cw;
      const y = y0 + r * ch;
      base.append(svgNode("rect", { x, y, width: cw, height: ch, fill: "#d9d9d9", stroke: "#fff", "stroke-width": 0.8 }));
      if (slot.grids.truth[prb][s]) {
        jam.append(svgNode("rect", { x: x + 0.8, y: y + 0.8, width: cw - 1.6, height: ch - 1.6, fill: "none", stroke: RED, "stroke-width": 1.8, class: "pulse" }));
      }
      if (slot.detectors.fused.decisions[prb][s]) {
        flags.append(svgNode("rect", { x: x + 3, y: y + 2.5, width: cw - 6, height: ch - 5, fill: BLUE, "data-col": s }));
      }
    }
    if (prb % 2 === 0) {
      svg.append(svgNode("text", { x: x0 - 5, y: y0 + r * ch + ch / 2 + 3.5, "text-anchor": "end", "font-size": 9 }, prb));
    }
  }
  for (const [key, g] of Object.entries(features)) {
    const values = slot.grids[key].slice(start, start + count);
    const scale = featureScale(values);
    for (let r = 0; r < count; r++) {
      const prb = start + count - 1 - r;
      for (let s = 0; s < cols; s++) {
        const v = slot.grids[key][prb][s];
        if (!Number.isFinite(v) || !scale) continue;
        g.append(svgNode("rect", { x: x0 + s * cw, y: y0 + r * ch, width: cw, height: ch, fill: heatColor(v, scale), stroke: "#fff", "stroke-width": 0.8 }));
      }
    }
  }
  svg.append(base, ...Object.values(features), jam, flags);
  for (let s = 0; s < cols; s++) {
    svg.append(svgNode("text", { x: x0 + s * cw + cw / 2, y: y0 - 4, "text-anchor": "middle", "font-size": 9, "font-weight": s === dmrs ? 700 : 400 }, s));
  }
  svg.append(svgNode("text", { x: x0 + cols * cw / 2, y: height - 8, "text-anchor": "middle", "font-size": 10 }, `OFDM symbol (pilots on symbol ${dmrs})`));
  const mid = y0 + count * ch / 2;
  svg.append(svgNode("text", { x: 9, y: mid, "text-anchor": "middle", "font-size": 10, transform: `rotate(-90 9 ${mid})` }, "PRB"));
  const label = svgNode("text", { x: width - 6, y: y0 - 4, "text-anchor": "end", "font-size": 9.5, fill: "#555" }, "");
  svg.append(label);
  return { svg, features, jam, flags, label };
}

// The channel across the allocation's subcarriers (magnitude), with the residual per PRB below.
function channelPanel(slot, channel, { residual = true } = {}) {
  const n = channel.num_sc;
  const width = 480;
  const left = 40;
  const right = 470;
  const top = 14;
  const bottom = residual ? 190 : 230;
  const height = residual ? 330 : 270;
  const all = [...channel.true.map(abs), ...channel.ls.map(abs), ...channel.standard.map(abs)];
  const ymax = Math.min(3, Math.ceil(Math.max(...all) * 2) / 2);
  const x = k => left + (k + 0.5) / n * (right - left);
  const y = v => bottom - Math.min(v, ymax) / ymax * (bottom - top);
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "anim-plot", role: "img",
    "aria-label": "Channel magnitude across the allocation's subcarriers" });
  for (let b = 0; b <= n / 12; b++) {
    svg.append(svgNode("line", { x1: x(b * 12 - 0.5), x2: x(b * 12 - 0.5), y1: top, y2: bottom, stroke: "#eee", "stroke-width": 0.6 }));
  }
  for (const v of [0, ymax / 2, ymax]) {
    svg.append(svgNode("text", { x: left - 5, y: y(v) + 3.5, "text-anchor": "end", "font-size": 10 }, v.toFixed(1)));
  }
  svg.append(svgNode("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "none", stroke: "#000", "stroke-width": 0.7 }));
  const midY = (top + bottom) / 2;
  svg.append(svgNode("text", { x: 11, y: midY, "text-anchor": "middle", "font-size": 10.5, transform: `rotate(-90 11 ${midY})` }, "|channel|"));
  const [start] = slot.metadata.alloc;
  const prbs = n / 12;
  for (let b = 0; b < prbs; b += 2) {
    svg.append(svgNode("text", { x: x(b * 12 + 6), y: bottom + 13, "text-anchor": "middle", "font-size": 9 }, start + b));
  }
  svg.append(svgNode("text", { x: (left + right) / 2, y: bottom + 26, "text-anchor": "middle", "font-size": 10 }, "PRB (12 subcarriers each)"));

  const path = values => values.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const trueLine = svgNode("path", { d: path(channel.true.map(abs)), fill: "none", stroke: GREY, "stroke-width": 3, opacity: 0 });
  const estimate = svgNode("path", { d: path(channel.standard.map(abs)), fill: "none", stroke: "#000", "stroke-width": 1.6, "stroke-dasharray": "5 3", opacity: 0 });
  const fit = svgNode("path", {
    d: channel.pilot_sc.map((k, i) => `${i ? "L" : "M"}${x(k).toFixed(1)},${y(abs(channel.fit[i])).toFixed(1)}`).join(" "),
    fill: "none", stroke: BLUE, "stroke-width": 1.2, opacity: 0,
  });
  const pilots = svgNode("g");
  const dots = channel.pilot_sc.map((k, i) => {
    const jammed = channel.pilot_jammed[i];
    const dot = svgNode("circle", { cx: x(k), cy: y(abs(channel.ls[i])), r: 3, fill: jammed ? RED : "#fff", stroke: jammed ? RED : BLUE, "stroke-width": 1.2, opacity: 0 });
    pilots.append(dot);
    return dot;
  });
  svg.append(trueLine, fit, estimate, pilots);

  let bars = [];
  if (residual) {
    const rTop = 238;
    const rBottom = 305;
    const values = channel.residual_prb.map(v => 10 * Math.log10(Math.max(v, 1e-3)));
    const thresholds = channel.threshold_prb.map(v => 10 * Math.log10(v));
    const lo = -5;
    const hi = Math.max(15, Math.ceil(Math.max(...values) / 5) * 5);
    const ry = v => rBottom - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * (rBottom - rTop);
    const barGroup = svgNode("g", { opacity: 0, class: "bars" });
    barGroup.append(svgNode("text", { x: left, y: rTop - 8, "font-size": 10 }, "Pilot residual per PRB (dB), dashed: 1% threshold in noise"));
    barGroup.append(svgNode("line", { x1: left, x2: right, y1: rBottom, y2: rBottom, stroke: "#000", "stroke-width": 0.7 }));
    bars = values.map((v, b) => {
      const jammed = channel.pilot_jammed.some((j, i) => j && Math.floor(channel.pilot_sc[i] / 12) === b);
      const bx = x(b * 12) - 2;
      const bw = x(b * 12 + 11) - x(b * 12) + 4;
      const rect = svgNode("rect", { x: bx, y: rBottom, width: bw, height: 0, fill: jammed ? RED : "#bbb", "data-y": ry(v) });
      rect.append(svgNode("title", {}, `PRB ${start + b}: ${v.toFixed(1)} dB`));
      barGroup.append(rect);
      barGroup.append(svgNode("line", { x1: bx - 1, x2: bx + bw + 1, y1: ry(thresholds[b]), y2: ry(thresholds[b]), stroke: "#000", "stroke-width": 1, "stroke-dasharray": "3 2" }));
      return rect;
    });
    svg.append(barGroup);
    bars.group = barGroup;
    bars.bottom = rBottom;
  }
  return { svg, trueLine, estimate, fit, dots, bars, path, channel };
}

function morphPath(panel, from, to) {
  const a = from.map(abs);
  const b = to.map(abs);
  return tween(1400, t => panel.estimate.setAttribute("d", panel.path(a.map((v, k) => v + (b[k] - v) * t))));
}

function stepper(host, steps, { autoplay = true, param = null } = {}) {
  const caption = host.querySelector(".anim-caption");
  const prev = host.querySelector(".anim-prev");
  const next = host.querySelector(".anim-next");
  const play = host.querySelector(".anim-play");
  const dots = host.querySelector(".anim-dots");
  const buttons = steps.map((step, i) => {
    const b = element("button", String(i + 1));
    b.type = "button";
    b.title = step.title;
    b.setAttribute("aria-label", `Step ${i + 1}: ${step.title}`);
    b.addEventListener("click", () => { stop(); go(i); });
    return b;
  });
  dots.replaceChildren(...buttons);
  let index = -1;
  let timer = null;
  let busy = Promise.resolve();

  async function go(i) {
    index = Math.max(0, Math.min(steps.length - 1, i));
    buttons.forEach((b, k) => b.setAttribute("aria-pressed", String(k === index)));
    prev.disabled = index === 0;
    next.disabled = index === steps.length - 1;
    caption.replaceChildren(element("h3", `${index + 1}. ${steps[index].title}`), element("p", steps[index].text));
    // Earlier steps are shown at once, the current one is animated
    steps.forEach((step, k) => k < index && step.show(false));
    for (let k = index + 1; k < steps.length; k++) steps[k].hide?.();
    busy = steps[index].show(true) ?? Promise.resolve();
    await busy;
  }
  function stop() {
    clearTimeout(timer);
    timer = null;
    play.textContent = "Play";
  }
  async function run() {
    play.textContent = "Pause";
    while (timer !== null && index < steps.length - 1) {
      await go(index + 1);
      await new Promise(resolve => { timer = setTimeout(resolve, 2600); });
    }
    stop();
  }
  prev.addEventListener("click", () => { stop(); go(index - 1); });
  next.addEventListener("click", () => { stop(); go(index + 1); });
  play.addEventListener("click", () => {
    if (timer !== null) return stop();
    if (index >= steps.length - 1) index = -1;
    timer = 0;
    run();
  });
  // ?step=N (or the given parameter) opens the animation at step N, e.g. to link a step in a talk
  const linked = Number(new URLSearchParams(location.search).get(param));
  if (param && linked >= 1) {
    steps.slice(0, linked - 1).forEach(step => step.show(false));
    go(linked - 1);
    return;
  }
  go(0);
  if (autoplay && !reduceMotion) {
    let started = false;
    new IntersectionObserver((entries, observer) => {
      if (started || !entries.some(e => e.isIntersecting)) return;
      started = true;
      observer.disconnect();
      timer = 0;
      setTimeout(run, 800);
    }, { threshold: 0.5 }).observe(host);
  }
}

const fade = (node, to, ms = 500) => {
  const from = Number(node.getAttribute("opacity") ?? 1);
  return tween(ms, t => node.setAttribute("opacity", String(from + (to - from) * t)));
};

async function methodAnimation(host) {
  const slot = await loadCase("walkthrough_flip");
  const channel = slot.channel;
  if (!channel) throw new Error("The recorded slot has no channel data yet.");
  const grid = gridPanel(slot);
  const plot = channelPanel(slot, channel);
  const canvas = element("canvas", null, "anim-canvas");
  canvas.hidden = true;
  const crc = element("p", null, "crc-line");
  host.querySelector(".anim-left").replaceChildren(grid.svg);
  host.querySelector(".anim-right").replaceChildren(plot.svg, canvas, crc);
  const view = constellationCanvas(canvas, 2);
  const prbs = affectedPRBs(slot);
  const before = symbols(slot, "standard", prbs);
  const after = symbols(slot, "fused_soft", prbs);
  const showPlot = () => { plot.svg.style.display = ""; canvas.hidden = true; crc.replaceChildren(); };
  const featureCycle = async animate => {
    const order = ["power_db", "residual_db", "evm_db"];
    const names = { power_db: "power P", residual_db: "pilot residual R", evm_db: "EVM E" };
    if (!animate) {
      order.forEach(k => grid.features[k].setAttribute("opacity", 0));
      return;
    }
    for (const key of order) {
      grid.label.textContent = names[key];
      await fade(grid.features[key], 1, 450);
      await new Promise(r => setTimeout(r, reduceMotion ? 0 : 700));
      await fade(grid.features[key], 0, 350);
    }
    grid.label.textContent = "";
  };

  // Numbers the step texts quote, taken from the slot itself
  const [allocStart, allocCount] = slot.metadata.alloc;
  const jammedCells = slot.grids.truth.flat().filter(Boolean).length;
  const flagged = slot.detectors.fused.decisions.flat().filter(Boolean).length;
  const hits = slot.grids.truth.flatMap((row, p) => row.map((t, s) => t && slot.detectors.fused.decisions[p][s])).filter(Boolean).length;
  const mapText = hits === jammedCells && flagged === jammedCells
    ? `Here the map matches the jammer exactly: all ${jammedCells} jammed pilot cells, and none of the data cells whose EVM rose only because of the wrong channel estimate.`
    : `Here the map flags ${hits} of the ${jammedCells} jammed cells and ${flagged - hits} other cells; the data cells whose EVM rose only because of the wrong channel estimate are mostly left alone.`;
  // How much closer the map-aware estimate gets on the jammed PRBs (squared error against the
  // true channel), quoted by step 7
  const dmrsSym = slot.metadata.config.dmrs_symbol;
  const jammedSc = slot.grids.truth.slice(allocStart, allocStart + allocCount)
    .flatMap((row, b) => (row[dmrsSym] ? Array.from({ length: 12 }, (_, k) => 12 * b + k) : []));
  const errBefore = sqErr(channel.standard, channel.true, jammedSc);
  const errAfter = sqErr(channel.map, channel.true, jammedSc);
  const cut = Math.round(100 * (1 - errAfter / errBefore));
  const steps = [
    {
      title: "A jammer hits only the pilots",
      text: `The user sends one slot on PRBs ${allocStart} to ${allocStart + allocCount - 1}. The jammer transmits only on the pilot (DMRS) symbol of ${jammedCells} PRBs, outlined in red. The grey line is the channel the receiver has to estimate.`,
      show: animate => { showPlot(); grid.jam.setAttribute("opacity", 1); grid.svg.classList.toggle("pulsing", animate); return animate ? fade(plot.trueLine, 1) : plot.trueLine.setAttribute("opacity", 1); },
      hide: () => { grid.jam.setAttribute("opacity", 0); plot.trueLine.setAttribute("opacity", 0); },
    },
    {
      title: "Each pilot gives one noisy look at the channel",
      text: "Dividing each received pilot by the known pilot symbol gives a least-squares estimate. Clean pilots (open circles) sit near the channel; the jammed ones (red) scatter far from it.",
      show: animate => {
        grid.svg.classList.remove("pulsing");
        if (!animate) return plot.dots.forEach(d => d.setAttribute("opacity", 1));
        return Promise.all(plot.dots.map((d, i) => new Promise(r => setTimeout(() => fade(d, 1, 250).then(r), reduceMotion ? 0 : i * 12))));
      },
      hide: () => plot.dots.forEach(d => d.setAttribute("opacity", 0)),
    },
    {
      title: "The standard receiver trusts every pilot",
      text: "Interpolating between all pilots, it follows the jammed ones (dashed line). The data symbols of those PRBs are then equalized with a wrong channel, so the damage spreads to symbols the jammer never touched.",
      show: animate => (animate ? fade(plot.estimate, 1, 700) : plot.estimate.setAttribute("opacity", 1)),
      hide: () => { plot.estimate.setAttribute("opacity", 0); plot.estimate.setAttribute("d", plot.path(channel.standard.map(abs))); },
    },
    {
      title: "A smooth channel cannot explain the jammed pilots",
      text: "A real channel varies slowly across frequency (its delays are short). Fitting such a channel (blue) to the pilots and measuring what is left gives the residual: about 0 dB in noise alone, far higher where the jammer was.",
      show: animate => {
        plot.fit.setAttribute("opacity", 1);
        plot.bars.group.setAttribute("opacity", 1);
        const grow = t => plot.bars.forEach(b => {
          const target = Number(b.getAttribute("data-y"));
          const yv = plot.bars.bottom + (target - plot.bars.bottom) * t;
          b.setAttribute("y", Math.min(yv, plot.bars.bottom));
          b.setAttribute("height", Math.abs(plot.bars.bottom - yv));
        });
        return animate ? tween(900, grow) : grow(1);
      },
      hide: () => { plot.fit.setAttribute("opacity", 0); plot.bars.group.setAttribute("opacity", 0); },
    },
    {
      title: "Power, residual and EVM go into a small U-Net",
      text: "For every cell the receiver measures the received power, the pilot residual and the error of the equalized data symbols (EVM). The U-Net, 66,553 parameters, reads the three maps of the whole slot together.",
      show: animate => featureCycle(animate),
    },
    {
      title: "The U-Net marks the jammed cells",
      text: `Blue cells are flagged. ${mapText}`,
      show: animate => {
        if (!animate) return grid.flags.setAttribute("opacity", 1);
        grid.flags.setAttribute("opacity", 1);
        const rects = [...grid.flags.children];
        rects.forEach(r => r.setAttribute("opacity", 0));
        return Promise.all(rects.map((r, i) => new Promise(res => setTimeout(() => fade(r, 1, 300).then(res), i * 120))));
      },
      hide: () => { grid.flags.setAttribute("opacity", 0); [...grid.flags.children].forEach(r => r.setAttribute("opacity", 1)); },
    },
    {
      title: "Decoding again, the flagged pilots count less",
      text: `The receiver processes the same buffered samples again. Each flagged pilot's noise variance is multiplied by its residual, so those pilots pull less on the estimate (red line). On the jammed PRBs its squared error against the channel falls by ${cut}%; a missed PRB and the strongest jammed pilots still pull it off in places.`,
      show: animate => {
        showPlot();
        plot.estimate.setAttribute("stroke", RED);
        plot.estimate.setAttribute("stroke-dasharray", "");
        const jammed = plot.dots.filter((_, i) => channel.pilot_jammed[i]);
        if (!animate) {
          jammed.forEach(d => d.setAttribute("opacity", 0.2));
          plot.estimate.setAttribute("d", plot.path(channel.map.map(abs)));
          return;
        }
        return Promise.all([...jammed.map(d => fade(d, 0.2, 700)), morphPath(plot, channel.standard, channel.map)]);
      },
      hide: () => {
        plot.estimate.setAttribute("stroke", "#000");
        plot.estimate.setAttribute("stroke-dasharray", "5 3");
        plot.dots.forEach((d, i) => channel.pilot_jammed[i] && d.getAttribute("opacity") !== "0" && d.setAttribute("opacity", 1));
      },
    },
    {
      title: "The symbols tighten and the block decodes",
      text: "The equalized data symbols of the jammed PRBs move back to the four QPSK points. The standard receiver failed its CRC on this slot; decoding with the map passes.",
      show: animate => {
        plot.svg.style.display = "none";
        canvas.hidden = false;
        crc.replaceChildren("Standard receiver: ", crcBadge(slot.receivers.standard.crc));
        const done = () => crc.replaceChildren("Standard: ", crcBadge(slot.receivers.standard.crc), "  →  with the map: ", crcBadge(slot.receivers.fused_soft.crc));
        if (!animate) {
          drawFrame(view, before, after, 1, 2);
          return done();
        }
        drawFrame(view, before, after, 0, 2);
        return new Promise(r => setTimeout(r, 700)).then(() => tween(1600, t => drawFrame(view, before, after, t, 2))).then(done);
      },
      hide: () => showPlot(),
    },
  ];
  stepper(host, steps, { param: "step" });
  host.querySelector(".anim-note").replaceChildren(`${conditions(slot)}. ${selectionText(slot)}`);
  host.setAttribute("aria-busy", "false");
}

// The data-aided passes on a slot whose every pilot is jammed: the estimate and the symbols after
// each pass, with that pass's CRC.
async function aidedAnimation(host) {
  const slot = await loadCase("aided_flip");
  const { channel, passes } = slot;
  if (!channel || !passes) throw new Error("The recorded slot has no data-aided passes yet.");
  const plot = channelPanel(slot, channel, { residual: false });
  plot.trueLine.setAttribute("opacity", 1);
  plot.dots.forEach(d => d.setAttribute("opacity", 0.35));
  plot.estimate.setAttribute("opacity", 1);
  plot.estimate.setAttribute("stroke", RED);
  plot.estimate.setAttribute("stroke-dasharray", "");
  const canvas = element("canvas", null, "anim-canvas");
  const crc = element("p", null, "crc-line");
  host.querySelector(".anim-left").replaceChildren(plot.svg);
  host.querySelector(".anim-right").replaceChildren(canvas, crc);
  const view = constellationCanvas(canvas, 2);
  const [start, width] = slot.metadata.alloc;
  const prbs = Array.from({ length: width }, (_, i) => start + i);
  // A pass's symbols are stored like a receiver's; build a pseudo-receiver entry per pass
  const passSymbols = passes.map((p, i) => {
    const name = `__pass${i}`;
    slot.receivers[name] = { x_hat: p.x_hat, abs_llr: p.abs_llr ?? slot.receivers.fused_soft.abs_llr, crc: p.crc };
    return symbols(slot, name, prbs);
  });
  let current = 0;
  const show = async (i, animate) => {
    const from = current;
    current = i;
    const badges = passes.slice(0, i + 1).map((p, k) => {
      const span = element("span", null, "pass-crc");
      span.append(k === 0 ? "first decoding " : `pass ${k} `, crcBadge(p.crc));
      return span;
    });
    crc.replaceChildren(...badges);
    if (!animate) {
      plot.estimate.setAttribute("d", plot.path(passes[i].h.map(abs)));
      drawFrame(view, passSymbols[i], passSymbols[i], 0, 2);
      return;
    }
    await Promise.all([
      morphPath(plot, passes[from].h, passes[i].h),
      tween(1400, t => drawFrame(view, passSymbols[from], passSymbols[i], t, 2)),
    ]);
  };
  const firstPass = passes.findIndex(p => p.crc);
  const steps = passes.map((p, i) => ({
    title: i === 0 ? "Every pilot is jammed: the first estimate is poor" : `Pass ${i}: decoded data cells act as pilots`,
    text: i === 0
      ? "All pilots are red: even down-weighted, they leave the estimate (red line) far from the channel (grey), and the block fails."
      : i <= firstPass || firstPass < 0
        ? `The decoder's soft output says which symbol each data cell carried and how sure it is. These cells join the pilots, the estimate moves towards the channel, and the slot is decoded again${p.crc ? ": the CRC passes." : "."}`
        : `The block already decoded at pass ${firstPass}, where the receiver would stop. This pass is shown only to watch the estimate settle onto the channel.`,
    show: animate => show(i, animate),
  }));
  stepper(host, steps, { param: "pass" });
  host.querySelector(".anim-note").replaceChildren(`${conditions(slot)}, every pilot of the allocation jammed. ${selectionText(slot)}`);
  host.setAttribute("aria-busy", "false");
}

for (const [run, id] of [[methodAnimation, "#method-anim"], [aidedAnimation, "#aided-anim"]]) {
  const host = document.querySelector(id);
  if (host) run(host).catch(error => reportError(host.querySelector(".anim-left") ?? host, error));
}
