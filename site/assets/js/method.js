// "How it works": the method on one recorded partial-band slot, step by step, and the data-aided
// passes on a slot whose every pilot is jammed. Every step is a complete state of the two panels
// (the resource grid on the left, the channel, the three clues or the symbols on the right), so
// any step can be reached from any other; only a move to the next step is animated.
import { conditions, loadCase, selectionText } from "./data.js";
import { crcBadge } from "./constellation.js";
import { featureScale, heatColor, svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";
import {
  BLUE, BOX, PILOT, RED, allocationGrid, cellErrors, cellTrust, constellationView, decodingPanel, drawFrame,
  fadeTo, mean, median, shareText, symbols, tween,
} from "./points.js";

const GREY = "#9a9a9a";
const PASS_GREEN = "#1b7f3b";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const abs = ([re, im]) => Math.hypot(re, im);
const sqErr = (est, ref, ks) => mean(ks.map(k => (est[k][0] - ref[k][0]) ** 2 + (est[k][1] - ref[k][1]) ** 2));
const dB = v => `${v >= 0 ? "" : "−"}${Math.abs(v).toFixed(1)} dB`;

// The channel across the allocation's subcarriers (magnitude): true channel, least-squares pilot
// estimates and one receiver's estimate. Pilots above the axis are drawn as triangles at its top.
function channelPanel(slot, channel, { jammedPRBs = [] } = {}) {
  const { width, height } = BOX;
  const n = channel.num_sc;
  const [start] = slot.metadata.alloc;
  const left = 50;
  const right = width - 10;
  const top = 22;
  const bottom = height - 58;
  const lines = [channel.true, channel.standard, channel.map].flatMap(v => v.map(abs));
  const ymax = Math.min(3, Math.max(1, Math.ceil(Math.max(...lines) * 1.15)));
  const x = k => left + ((k + 0.5) / n) * (right - left);
  const y = v => bottom - (Math.min(v, ymax) / ymax) * (bottom - top);
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "viz", role: "img",
    "aria-label": "Channel magnitude across the user's subcarriers" });
  for (const b of jammedPRBs) {
    svg.append(svgNode("rect", { x: x(12 * (b - start) - 0.5), y: top, width: x(12) - x(0), height: bottom - top, fill: "#fbe3e3" }));
  }
  for (let b = 0; b <= n / 12; b++) {
    svg.append(svgNode("line", { x1: x(b * 12 - 0.5), x2: x(b * 12 - 0.5), y1: top, y2: bottom, stroke: "#e6e6e6", "stroke-width": 0.6 }));
  }
  for (const v of [0, ymax / 2, ymax]) {
    svg.append(svgNode("text", { x: left - 5, y: y(v) + 3.5, "text-anchor": "end", "font-size": 10 }, String(v)));
  }
  for (let b = 0; b < n / 12; b++) {
    svg.append(svgNode("text", { x: x(b * 12 + 5.5), y: bottom + 14, "text-anchor": "middle", "font-size": 10 }, start + b));
  }
  svg.append(svgNode("text", { x: (left + right) / 2, y: bottom + 30, "text-anchor": "middle", "font-size": 11 }, "PRB (12 subcarriers each)"));
  const midY = (top + bottom) / 2;
  svg.append(svgNode("text", { x: 13, y: midY, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 13 ${midY})` }, "Channel magnitude"));
  svg.append(svgNode("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "none", stroke: "#000", "stroke-width": 0.8 }));

  const path = values => values.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const trueLine = svgNode("path", { d: path(channel.true.map(abs)), fill: "none", stroke: GREY, "stroke-width": 3.2, opacity: 0 });
  const ghost = svgNode("path", { d: path(channel.standard.map(abs)), fill: "none", stroke: "#000", "stroke-width": 1.2, "stroke-dasharray": "5 3", opacity: 0 });
  const estimate = svgNode("path", { d: path(channel.standard.map(abs)), fill: "none", stroke: "#000", "stroke-width": 1.8, "stroke-dasharray": "5 3", opacity: 0 });
  const pilots = svgNode("g", { opacity: 0 });
  let above = 0;
  const dots = channel.pilot_sc.map((k, i) => {
    const jammed = channel.pilot_jammed[i];
    const v = abs(channel.ls[i]);
    const style = { fill: jammed ? RED : "#fff", stroke: jammed ? RED : BLUE, "stroke-width": 1.2 };
    let dot;
    if (v > ymax) {
      above++;
      const cx = x(k);
      dot = svgNode("path", { d: `M${cx - 3.5},${top + 7} L${cx + 3.5},${top + 7} L${cx},${top + 1} Z`, ...style });
    } else {
      dot = svgNode("circle", { cx: x(k), cy: y(v), r: 3, ...style });
    }
    pilots.append(dot);
    return dot;
  });
  const aboveNote = svgNode("text", { x: right - 4, y: top - 7, "text-anchor": "end", "font-size": 10, fill: "#555", opacity: 0 },
    above ? `${above} pilot estimates above ${ymax.toFixed(1)}: triangles at the top` : "");
  svg.append(trueLine, ghost, pilots, estimate, aboveNote);
  return { svg, trueLine, ghost, estimate, pilots, dots, aboveNote, path };
}

// Power, residual and EVM of the allocation as three small maps, and the U-Net map, in a 2 x 2
// layout; each map is outlined where the jammer was.
function cluesPanel(slot, numbers) {
  const { width, height } = BOX;
  const [start, count] = slot.metadata.alloc;
  const cols = slot.grids.allocation[0].length;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "viz", role: "img",
    "aria-label": "The three measurements per cell and the U-Net map" });
  const panels = [
    ["power_db", "Power P", numbers.power_db],
    ["residual_db", "Pilot residual R", numbers.residual_db],
    ["evm_db", "EVM E", numbers.evm_db],
    ["map", "→ U-Net map", "blue is flagged, red outline is jammed"],
  ];
  const pw = width / 2;
  const ph = height / 2;
  const cw = (pw - 24) / cols;
  const ch = (ph - 48) / count;
  const groups = {};
  panels.forEach(([key, title, note], i) => {
    const ox = (i % 2) * pw + 12;
    const oy = Math.floor(i / 2) * ph + 4;
    const g = svgNode("g", { opacity: key === "map" ? 0 : 1 });
    g.append(svgNode("text", { x: ox, y: oy + 13, "font-size": 12, "font-weight": 600 }, title));
    const values = key === "map" ? null : slot.grids[key];
    const scale = values ? featureScale(values.slice(start, start + count)) : null;
    for (let prb = start; prb < start + count; prb++) {
      for (let s = 0; s < cols; s++) {
        const x = ox + s * cw;
        const y = oy + 20 + (start + count - 1 - prb) * ch;
        let fill = "#f2f2f2";
        if (key === "map") fill = slot.detectors.fused.decisions[prb][s] ? BLUE : "#dedede";
        else if (Number.isFinite(values[prb][s])) fill = heatColor(values[prb][s], scale);
        g.append(svgNode("rect", { x, y, width: cw, height: ch, fill, stroke: "#fff", "stroke-width": 0.4 }));
        if (slot.grids.truth[prb][s]) {
          g.append(svgNode("rect", { x: x + 0.4, y: y + 0.4, width: cw - 0.8, height: ch - 0.8, fill: "none", stroke: RED, "stroke-width": 0.9 }));
        }
      }
    }
    g.append(svgNode("text", { x: ox, y: oy + 20 + count * ch + 13, "font-size": 10, fill: "#444" }, note));
    groups[key] = g;
    svg.append(g);
  });
  return { svg, groups };
}

// Back / Play / Next, numbered steps and (optionally) a bar of stages above the panels. Each
// step's `render(animate, from)` draws its complete state; ?<param>=N opens step N.
function stepper(host, steps, { param, stages = null, dwell = 5200 } = {}) {
  const caption = host.querySelector(".anim-caption");
  const prev = host.querySelector(".anim-prev");
  const next = host.querySelector(".anim-next");
  const play = host.querySelector(".anim-play");
  const dots = host.querySelector(".anim-dots");
  const stageBar = host.querySelector(".anim-stages");
  let index = -1;
  let timer = null;
  const buttons = steps.map((step, i) => {
    const b = element("button", String(i + 1));
    b.type = "button";
    b.title = step.title;
    b.setAttribute("aria-label", `Step ${i + 1}: ${step.title}`);
    b.addEventListener("click", () => { stop(); go(i); });
    return b;
  });
  dots.replaceChildren(...buttons);
  const stageItems = (stages ?? []).map((name, s) => {
    const li = element("li");
    const b = element("button", name);
    b.type = "button";
    b.addEventListener("click", () => { stop(); go(steps.findIndex(step => step.stage === s)); });
    li.append(b);
    return li;
  });
  if (stageBar) stageBar.replaceChildren(...stageItems);

  async function go(i) {
    const from = index;
    index = Math.max(0, Math.min(steps.length - 1, i));
    buttons.forEach((b, k) => b.setAttribute("aria-pressed", String(k === index)));
    stageItems.forEach((li, s) => li.classList.toggle("current", steps[index].stage === s));
    prev.disabled = index === 0;
    next.disabled = index === steps.length - 1;
    caption.replaceChildren(element("h3", `${index + 1}. ${steps[index].title}`), element("p", steps[index].text));
    await steps[index].render(from === index - 1 && !reduceMotion, from);
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
      await new Promise(resolve => { timer = setTimeout(resolve, dwell); });
    }
    stop();
  }
  prev.addEventListener("click", () => { stop(); go(index - 1); });
  next.addEventListener("click", () => { stop(); go(index + 1); });
  play.addEventListener("click", () => {
    if (timer !== null) return stop();
    if (index >= steps.length - 1) go(0);
    timer = 0;
    run();
  });
  const linked = Number(new URLSearchParams(location.search).get(param));
  go(linked >= 1 ? linked - 1 : 0);
  if (!(linked >= 1) && !reduceMotion) {
    // Play once when the animation first comes into view
    let started = false;
    new IntersectionObserver((entries, observer) => {
      if (started || !entries.some(e => e.isIntersecting)) return;
      started = true;
      observer.disconnect();
      timer = 0;
      setTimeout(run, 900);
    }, { threshold: 0.6 }).observe(host.querySelector(".anim-stage"));
  }
}

function legendItems(items) {
  const list = element("ul", null, "legend");
  for (const [className, label, style] of items) {
    const li = element("li");
    const mark = element("i", null, `swatch ${className}`);
    if (style) Object.assign(mark.style, style);
    mark.setAttribute("aria-hidden", "true");
    li.append(mark, label);
    list.append(li);
  }
  return list;
}

async function methodAnimation(host) {
  const slot = await loadCase("partial_flip");
  const channel = slot.channel;
  if (!channel) throw new Error("The recorded slot has no channel data.");
  const [start, count] = slot.metadata.alloc;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const truth = slot.grids.truth;
  const flags = slot.detectors.fused.decisions;
  // The allocation's cells [prb, symbol, jammed] for which pick(prb, symbol, jammed) holds
  const cellsWhere = pick => Array.from({ length: count }, (_, b) => start + b)
    .flatMap(prb => truth[prb].map((t, s) => [prb, s, t])).filter(([p, s, t]) => pick(p, s, t));

  // Numbers the step texts quote, all taken from the slot
  const jammedPRBs = Array.from({ length: count }, (_, b) => start + b).filter(p => truth[p].some(Boolean));
  const jammedCells = cellsWhere((p, s, t) => t).length;
  const flagged = cellsWhere((p, s) => flags[p][s]).length;
  const hits = cellsWhere((p, s, t) => t && flags[p][s]).length;
  const before = symbols(slot, "standard");
  const after = symbols(slot, "fused_soft");
  const trust = { standard: cellTrust(before), map: cellTrust(after) };
  // Wrong bits before decoding, per receiver (null without exported bits)
  const wrongBits = name => {
    const errors = cellErrors(slot, name);
    return errors && [...errors.values()].reduce((a, b) => a + b, 0);
  };
  const errors = { standard: wrongBits("standard"), map: wrongBits("fused_soft") };
  const decodedView = errors.map != null;
  const llr = (points, jammed) => mean(points.filter(p => p.jammed === jammed).map(p => p.llr));
  const featureNumbers = {};
  for (const key of ["power_db", "residual_db", "evm_db"]) {
    const j = median(cellsWhere((p, s, t) => t).map(([p, s]) => slot.grids[key][p][s]));
    const c = median(cellsWhere((p, s, t) => !t).map(([p, s]) => slot.grids[key][p][s]));
    featureNumbers[key] = { jammed: j, clean: c, text: `median: jammed ${dB(j)}, clean ${dB(c)}` };
  }
  const jammedSc = jammedPRBs.flatMap(p => (truth[p][dmrs] ? Array.from({ length: 12 }, (_, k) => 12 * (p - start) + k) : []));
  const cleanSc = Array.from({ length: 12 * count }, (_, k) => k).filter(k => !jammedSc.includes(k));
  const errStd = sqErr(channel.standard, channel.true, jammedSc);
  const errStdClean = sqErr(channel.standard, channel.true, cleanSc);
  const errMap = sqErr(channel.map, channel.true, jammedSc);
  const cut = Math.round(100 * (1 - errMap / errStd));
  const ratio = Math.round(errStd / errStdClean);
  const flaggedPilot = channel.pilot_sc.map(k => Boolean(flags[start + Math.floor(k / 12)][dmrs]));
  const batch = slot.metadata.batch_crc;
  const range = jammedPRBs.length ? `${jammedPRBs[0]}–${jammedPRBs.at(-1)}` : "none";
  const share = `${jammedPRBs.length} of the user's ${count} PRBs`;
  const jammedShare = shareText(jammedPRBs.length / count);
  const cleanShare = shareText(1 - jammedPRBs.length / count);

  // The panels
  const grid = allocationGrid(slot, { trust, label: "The user's resource grid" });
  const plot = channelPanel(slot, channel, { jammedPRBs });
  const clues = cluesPanel(slot, {
    power_db: featureNumbers.power_db.text,
    residual_db: featureNumbers.residual_db.text,
    evm_db: featureNumbers.evm_db.text,
  });
  const canvas = element("canvas", null, "viz");
  canvas.setAttribute("role", "img");
  const view = constellationView(canvas, 2.5);
  const crc = element("p", null, "crc-line");
  const right = host.querySelector(".anim-right");
  host.querySelector(".anim-left").replaceChildren(grid.svg);
  const decoding = decodedView
    ? decodingPanel(slot, ["standard", "fused_soft"], ["Standard receiver", "With the jamming map"],
      { height: BOX.height, flags: [null, flags] })
    : null;
  right.replaceChildren(plot.svg, clues.svg, canvas, ...(decoding ? [decoding.svg] : []));
  const legend = host.querySelector(".anim-legend");
  let shownPoints = null;
  let token = 0;

  // A step's state: grid layers, which right-hand view, and what that view shows
  const states = [
    { stage: 0, jam: 1, view: "channel", trueLine: 1 },
    { stage: 1, jam: 1, view: "channel", trueLine: 1, pilots: "all", estimate: "standard" },
    { stage: 1, jam: 1, trust: "standard", view: "points", points: "standard", crc: ["standard"] },
    { stage: 2, jam: 1, view: "clues" },
    { stage: 3, jam: 1, flags: 1, view: "clues", map: 1 },
    { stage: 4, jam: 1, flags: 1, view: "channel", trueLine: 1, pilots: "flagged faded", estimate: "map", ghost: 1 },
    { stage: 4, jam: 1, flags: 1, trust: "map", view: "points", points: "map" },
    // (no trust shading on the left here: on the right, darker means more wrong bits)
    { stage: 4, jam: 1, flags: 1, trust: decodedView ? null : "map", view: decodedView ? "decoding" : "points", points: "map", crc: ["standard", "fused_soft"] },
  ];

  async function apply(i, animate, from) {
    const s = states[i];
    const was = states[from] ?? {};
    const mine = ++token;
    const alive = () => mine === token;
    const fade = (node, to, ms) => (animate ? fadeTo(node, to, ms, alive) : (node.setAttribute("opacity", to), null));
    const jobs = [];
    // Grid
    jobs.push(fade(grid.layers.jam, s.jam ? 1 : 0));
    jobs.push(fade(grid.layers.flags, s.flags ? 1 : 0, 700));
    for (const name of Object.keys(trust)) jobs.push(fade(grid.layers[`trust_${name}`], s.trust === name ? 1 : 0, 900));
    // Right-hand view
    plot.svg.style.visibility = s.view === "channel" ? "visible" : "hidden";
    clues.svg.style.visibility = s.view === "clues" ? "visible" : "hidden";
    canvas.style.visibility = s.view === "points" ? "visible" : "hidden";
    if (decoding) decoding.svg.style.visibility = s.view === "decoding" ? "visible" : "hidden";
    if (s.view === "decoding") {
      // Start before decoding, then decode
      decoding.set("before");
      if (animate) jobs.push(new Promise(r => setTimeout(r, 1200)).then(() => (alive() ? decoding.set("after", true) : null)));
      else decoding.set("after");
    }
    if (s.view === "channel") {
      jobs.push(fade(plot.trueLine, s.trueLine ? 1 : 0));
      plot.pilots.setAttribute("opacity", s.pilots ? 1 : 0);
      plot.aboveNote.setAttribute("opacity", s.pilots ? 1 : 0);
      plot.dots.forEach((dot, k) => {
        const faded = s.pilots === "flagged faded" && flaggedPilot[k];
        const target = faded ? 0.15 : 1;
        if (animate && was.pilots && was.pilots !== s.pilots) jobs.push(fadeTo(dot, target, 700, alive));
        else if (animate && !was.pilots && s.pilots) {
          dot.setAttribute("opacity", 0);
          jobs.push(new Promise(r => setTimeout(r, k * 10)).then(() => fadeTo(dot, target, 250, alive)));
        } else dot.setAttribute("opacity", target);
      });
      plot.ghost.setAttribute("opacity", s.ghost ? 0.45 : 0);
      plot.estimate.setAttribute("opacity", s.estimate ? 1 : 0);
      const red = s.estimate === "map";
      plot.estimate.setAttribute("stroke", red ? RED : "#000");
      plot.estimate.setAttribute("stroke-dasharray", red ? "" : "5 3");
      const target = (red ? channel.map : channel.standard).map(abs);
      if (animate && red && was.estimate === "standard") {
        const a = channel.standard.map(abs);
        jobs.push(tween(1500, t => plot.estimate.setAttribute("d", plot.path(a.map((v, k) => v + (target[k] - v) * t))), alive));
      } else plot.estimate.setAttribute("d", plot.path(target));
    }
    if (s.view === "clues") {
      jobs.push(fade(clues.groups.map, s.map ? 1 : 0, 700));
    }
    if (s.view === "points") {
      const to = s.points === "map" ? after : before;
      if (animate && was.view === "points" && shownPoints && shownPoints !== to) {
        const a = shownPoints;
        jobs.push(tween(1700, t => drawFrame(view, a, to, t), alive));
      } else drawFrame(view, to, to, 0);
      shownPoints = to;
      canvas.setAttribute("aria-label", `Equalized data symbols of the user's ${count} PRBs, ${s.points === "map" ? "decoded with the jamming map" : "standard receiver"}`);
    }
    // Legend and CRC line under the panels
    const items = [["alloc", "data cell"], ["pilot", "pilot cell (symbol 2)"], ["truth", "jammed"]];
    if (s.flags) items.push(["flag-dot", "flagged by the U-Net"]);
    if (s.trust) items.push(["trust", "darker means the decoder trusts the cell's bits more"]);
    if (s.view === "channel") {
      items.push(["line grey-line", "true channel"]);
      if (s.pilots) items.push(["circle", "clean pilot"], ["dot", "jammed pilot"]);
      if (s.estimate === "standard") items.push(["dash", "standard estimate"]);
      if (s.estimate === "map") items.push(["dash faint", "standard estimate"], ["line red-line", "estimate with the map"]);
    }
    if (s.view === "decoding") {
      items.push(["errors", "darker means more of the cell's bits are wrong"], ["lost", "block lost"]);
    } else if (s.view === "points") items.push(["dot blue-dot", "clean cell"], ["dot", "jammed cell (fainter means less trusted)"]);
    const crcLine = element("p", null, "crc-line");
    if (s.crc) {
      s.crc.forEach((name, k) => {
        if (k) crcLine.append("  →  ");
        crcLine.append(name === "standard" ? "Standard receiver " : "With the jamming map ", crcBadge(slot.receivers[name].crc));
      });
    }
    legend.replaceChildren(legendItems(items), crcLine);
    await Promise.all(jobs.filter(Boolean));
  }

  const steps = [
    {
      title: "One uplink slot, and a jammer on part of it",
      text: `The user sends one slot on PRBs ${start}–${start + count - 1}, which is ${count} PRBs (frequency) by 14 OFDM symbols (time). Symbol ${dmrs} carries pilots, symbols the receiver knows and uses to learn the channel (grey line, right). The other 13 symbols carry data. The jammer sends noise on every symbol of PRBs ${range} (red). That is ${share}, so ${jammedShare} of the pilots and ${jammedShare} of the data cells.`,
    },
    {
      title: "Standard receiver. The jammed pilots pull the channel estimate",
      text: `Dividing each received pilot by the known pilot symbol gives one look at the channel. Clean pilots (open circles) sit on it, and jammed pilots (red) scatter. The standard receiver interpolates between all of them (dashed), so on PRBs ${range} its squared error is ${ratio} times its error on the clean PRBs.`,
    },
    {
      title: "Standard receiver. Every bit is trusted, and the block fails",
      text: `It equalizes the data and passes each bit to the decoder with a confidence (darker cells, left). The jammed cells' symbols are scattered (red points), yet their bits get almost the confidence of clean ones (mean |LLR| ${llr(before, true).toFixed(0)} against ${llr(before, false).toFixed(0)}). The decoder is misled and the CRC check fails.`,
    },
    {
      title: "Three clues per cell",
      text: `From the received slot, its pilots and the standard receiver's equalized symbols, the receiver measures three things for every cell. The received power P is the signal plus anything else, relative to the noise. On the pilot symbol, the residual R says how far the pilots of a PRB are from a smooth channel. On the data symbols, the EVM E says how far the equalized symbols are from the four QPSK points. On the jammed cells the power rises by only ${(featureNumbers.power_db.jammed - featureNumbers.power_db.clean).toFixed(1)} dB, the residual by ${(featureNumbers.residual_db.jammed - featureNumbers.residual_db.clean).toFixed(0)} dB and the EVM by ${(featureNumbers.evm_db.jammed - featureNumbers.evm_db.clean).toFixed(0)} dB (medians).`,
    },
    {
      title: "A small U-Net turns the clues into a map",
      text: `A U-Net with 66,553 parameters reads the three maps of the whole slot and scores each cell. Cells scoring above a cutoff, set so that 1% of clean cells are flagged, are marked (blue). ${hits === jammedCells && flagged === jammedCells ? `Here it flags all ${jammedCells} jammed cells and no other cell.` : `Here it flags ${hits} of the ${jammedCells} jammed cells and ${flagged - hits} other cells.`}`,
    },
    {
      title: "Decoding again with the channel from the clean pilots",
      text: `The receiver decodes the same buffered samples again, now with the map. Pilots in flagged cells get a noise variance scaled up by their residual, so they barely count (faded), and the estimate across PRBs ${range} is interpolated from the clean pilots around them (red line). Its squared error there falls by ${cut}%.`,
    },
    {
      title: "Decoding again, with jammed data bits counting less",
      text: `For each flagged data cell the receiver adds the jammer power it measured there to the noise variance. Those bits reach the decoder with almost no confidence (mean |LLR| ${llr(after, true).toFixed(1)}, against ${llr(after, false).toFixed(0)} for clean bits), and the jammed cells turn light on the left. The decoder fills them in from the clean ${cleanShare} of the block.`,
    },
    {
      title: decodedView ? "The decoder rebuilds the jammed bits" : "The block decodes",
      text: decodedView
        ? `Both receivers hand the decoder wrong bits in the jammed cells (dark, right), ${errors.standard} for the standard receiver and ${errors.map} with the map. The map does not fix them. It only tells the decoder to doubt them. The code's redundancy then lets the decoder rebuild them from the clean ${cleanShare} of the block, so every bit comes out right and the CRC passes. The standard receiver gave the same kind of wrong bits full confidence, so decoding fails and the whole block is lost. Over all ${batch.standard.slots} slots simulated under these conditions, the standard receiver lost ${batch.standard.failures} blocks and the receiver with the map ${batch.fused_soft.failures}.`
        : `Same received samples, one small network and one more decoding, and the CRC passes. Over all ${batch.standard.slots} slots simulated under these conditions, the standard receiver failed ${batch.standard.failures} and the receiver with the map ${batch.fused_soft.failures}.`,
    },
  ].map((step, i) => ({ ...step, stage: states[i].stage, render: (animate, from) => apply(i, animate, from) }));

  stepper(host, steps, {
    param: "step",
    stages: ["The slot", "Standard receiver", "Three clues", "U-Net map", "Decoding with the map"],
  });
  host.querySelector(".anim-note").replaceChildren(
    `${conditions(slot)}, partial-band jamming of PRBs ${range}. ${selectionText(slot)}`);
  host.setAttribute("aria-busy", "false");
}

// The user's grid during the data-aided passes: the jammed pilot column, and each data cell shaded
// green by how sure the decoder was of its symbols, which is how much it counts as a pilot.
function passGrid(slot) {
  const { width, height } = BOX;
  const [start, count] = slot.metadata.alloc;
  const cols = slot.metadata.config.num_symbols;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const x0 = 50;
  const y0 = 24;
  const cw = (width - x0 - 8) / cols;
  const ch = (height - y0 - 34) / count;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "viz", role: "img",
    "aria-label": "The user's resource grid with the cells used as pilots" });
  const green = [];
  for (let b = 0; b < count; b++) {
    const prb = start + b;
    green[b] = [];
    for (let s = 0; s < cols; s++) {
      const x = x0 + s * cw;
      const y = y0 + (count - 1 - b) * ch;
      svg.append(svgNode("rect", { x, y, width: cw, height: ch, fill: s === dmrs ? PILOT : "#e6e6e6", stroke: "#fff", "stroke-width": 1 }));
      if (s === dmrs) {
        svg.append(svgNode("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: "none", stroke: RED, "stroke-width": 2 }));
      } else {
        const cell = svgNode("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: PASS_GREEN, opacity: 0 });
        green[b][s] = cell;
        svg.append(cell);
      }
    }
    if (count <= 16 || b % 2 === 0) {
      svg.append(svgNode("text", { x: x0 - 5, y: y0 + (count - 1 - b) * ch + ch / 2 + 3.5, "text-anchor": "end", "font-size": 10 }, prb));
    }
  }
  for (let s = 0; s < cols; s++) {
    svg.append(svgNode("text", { x: x0 + s * cw + cw / 2, y: y0 - 6, "text-anchor": "middle", "font-size": 10, "font-weight": s === dmrs ? 700 : 400 }, s));
  }
  svg.append(svgNode("text", { x: x0 + (cols * cw) / 2, y: height - 8, "text-anchor": "middle", "font-size": 11 }, `OFDM symbol (time); pilots on symbol ${dmrs}`));
  const mid = y0 + (count * ch) / 2;
  svg.append(svgNode("text", { x: 12, y: mid, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${mid})` }, "PRB (frequency)"));
  svg.append(svgNode("rect", { x: x0, y: y0, width: cols * cw, height: count * ch, fill: "none", stroke: "#000", "stroke-width": 0.8 }));
  // One frame between two certainty grids (null: no data cell used yet)
  const paint = (from, to, t) => {
    for (let b = 0; b < count; b++) {
      for (let s = 0; s < cols; s++) {
        if (!green[b][s]) continue;
        const a = from?.[b][s] ?? 0;
        const z = to?.[b][s] ?? 0;
        green[b][s].setAttribute("opacity", String(a + (z - a) * t));
      }
    }
  };
  return { svg, paint };
}

// The data-aided passes on a slot whose every pilot is jammed, in three aligned panels: which cells
// act as pilots, the channel estimate, and the equalized symbols, after the first attempt and
// after each pass, with that pass's CRC.
async function aidedAnimation(host) {
  const slot = await loadCase("aided_flip");
  const { channel, passes } = slot;
  if (!channel || !passes) throw new Error("The recorded slot has no data-aided passes.");
  const [start, width] = slot.metadata.alloc;
  const grid = passGrid(slot);
  const plot = channelPanel(slot, channel, { jammedPRBs: Array.from({ length: width }, (_, i) => start + i) });
  plot.trueLine.setAttribute("opacity", 1);
  plot.pilots.setAttribute("opacity", 1);
  plot.aboveNote.setAttribute("opacity", 1);
  plot.dots.forEach(d => d.setAttribute("opacity", 0.35));
  plot.estimate.setAttribute("opacity", 1);
  plot.estimate.setAttribute("stroke", RED);
  plot.estimate.setAttribute("stroke-dasharray", "");
  const canvas = element("canvas", null, "viz");
  canvas.setAttribute("role", "img");
  host.querySelector(".anim-grid").replaceChildren(grid.svg);
  host.querySelector(".anim-left").replaceChildren(plot.svg);
  host.querySelector(".anim-right").replaceChildren(canvas);
  const view = constellationView(canvas, 2.5);
  // A pass's symbols are stored like a receiver's; build a pseudo-receiver entry per pass
  const passSymbols = passes.map((p, i) => {
    const name = `__pass${i}`;
    slot.receivers[name] = { x_hat: p.x_hat, abs_llr: p.abs_llr ?? slot.receivers.fused_soft.abs_llr, crc: p.crc };
    return symbols(slot, name);
  });
  const legend = host.querySelector(".anim-legend");
  const firstPass = passes.findIndex(p => p.crc);
  const estimate = i => passes[i].h.map(abs);
  const errors = passes.map(p => sqErr(p.h, channel.true, Array.from({ length: channel.num_sc }, (_, k) => k)));
  const error = v => (v < 0.001 ? "below 0.001" : v.toFixed(3));
  let token = 0;
  const render = async (i, animate, from) => {
    const mine = ++token;
    const alive = () => mine === token;
    const crcLine = element("p", null, "crc-line");
    passes.slice(0, i + 1).forEach((p, k) => {
      const span = element("span", null, "pass-crc");
      span.append(k === 0 ? "first attempt " : `pass ${k} `, crcBadge(p.crc));
      crcLine.append(span);
    });
    legend.replaceChildren(legendItems([
      ["pilot", "pilot cell (all jammed)"], ["pass-green", "data cell used as a pilot (darker means the decoder is surer)"],
      ["line grey-line", "true channel"], ["line red-line", "estimate"], ["dot", "jammed pilot"], ["dot blue-dot", "data symbol"],
    ]), crcLine);
    if (!animate || from < 0) {
      grid.paint(null, passes[i].certainty, 1);
      plot.estimate.setAttribute("d", plot.path(estimate(i)));
      drawFrame(view, passSymbols[i], passSymbols[i], 0);
      return;
    }
    const a = estimate(from);
    const b = estimate(i);
    await Promise.all([
      tween(1400, t => grid.paint(passes[from].certainty, passes[i].certainty, t), alive),
      tween(1400, t => plot.estimate.setAttribute("d", plot.path(a.map((v, k) => v + (b[k] - v) * t))), alive),
      tween(1400, t => drawFrame(view, passSymbols[from], passSymbols[i], t), alive),
    ]);
  };
  const steps = passes.map((p, i) => ({
    title: i === 0 ? "Every pilot is jammed, so the first estimate is poor" : `Pass ${i}`,
    text: i === 0
      ? `All the pilots (yellow column, outlined red) are jammed. Even down-weighted, they leave the estimate (red line) far from the true channel (grey), with a mean squared error of ${error(errors[0])}. The data points spread and the CRC fails.`
      : i <= firstPass || firstPass < 0
        ? `The decoder's output says which symbol each data cell most likely carried, and how sure it is. Dividing a cell's received samples by those symbols gives one more look at the channel, so the data cells now count as extra pilots (green, darker when the decoder is surer, left out when it is nearly unsure). With them the estimate moves towards the channel (error ${error(errors[i])}) and the slot is decoded again.${p.crc ? " The CRC passes, so the receiver stops here." : ""}`
        : `The block already decoded at pass ${firstPass}, so the receiver stops there. This pass is shown only to watch the estimate settle (error ${error(errors[i])}).`,
    render: (animate, from) => render(i, animate, from),
  }));
  stepper(host, steps, { param: "pass", dwell: 4600 });
  host.querySelector(".anim-note").replaceChildren(`${conditions(slot)}, every pilot of the allocation jammed. ${selectionText(slot)}`);
  host.setAttribute("aria-busy", "false");
}

// A toy code (3 data bits, 3 check bits) showing how the decoder fills in jammed bits when the map
// marks them unknown, and why trusting them wrongly makes every check disagree. Plain illustration,
// not simulation data.
function checksAnimation(host) {
  const GREEN = "#006600";
  const sent = [1, 0, 1, 1, 1, 0];                  // d1 d2 d3, c1 = d1^d2, c2 = d2^d3, c3 = d1^d3
  const names = [["d", 1], ["d", 2], ["d", 3], ["c", 1], ["c", 2], ["c", 3]];
  const jammed = [1, 5];                            // d2 and c3
  const checks = [[0, 1, 3], [1, 2, 4], [0, 2, 5]];  // the bits of c1, c2, c3
  const checkText = ["c1 = d1 ⊕ d2", "c2 = d2 ⊕ d3", "c3 = d1 ⊕ d3"];
  const width = 780;
  const height = 330;
  const boxX = i => 200 + i * 52 + (i >= 3 ? 16 : 0);
  const box = 46;
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "checks-viz", role: "img" });
  const sub = (g, x, y, [letter, n], size, attrs = {}) => {
    const t = svgNode("text", { x, y, "text-anchor": "middle", "font-size": size, "font-style": "italic", ...attrs }, letter);
    t.append(svgNode("tspan", { "baseline-shift": "sub", "font-size": size * 0.7, "font-style": "normal" }, n));
    g.append(t);
  };
  const rows = ["map", "standard"].map((kind, r) => {
    const top = 52 + r * 150;
    const g = svgNode("g");
    g.append(svgNode("text", { x: 16, y: top + 20, "font-size": 15, "font-weight": 700 }, kind === "map" ? "With the jamming map" : "Standard receiver"));
    g.append(svgNode("text", { x: 16, y: top + 38, "font-size": 12, fill: "#555" }, kind === "map" ? "jammed bits marked unknown" : "jammed bits trusted"));
    g.append(svgNode("text", { x: boxX(1) + box / 2, y: top - 26, "text-anchor": "middle", "font-size": 11, fill: "#555" }, "data bits"));
    g.append(svgNode("text", { x: boxX(4) + box / 2, y: top - 26, "text-anchor": "middle", "font-size": 11, fill: "#555" }, "check bits"));
    const bits = sent.map((_, i) => {
      const x = boxX(i);
      sub(g, x + box / 2, top - 7, names[i], 13);
      const rect = svgNode("rect", { x, y: top, width: box, height: box, rx: 5, class: "bit-box" });
      const value = svgNode("text", { x: x + box / 2, y: top + box / 2 + 8, "text-anchor": "middle", "font-size": 24, class: "bit-value" });
      g.append(rect, value);
      return { rect, value };
    });
    const pills = checkText.map((text, k) => {
      const x = 200 + k * 112;
      const y = top + 64;
      const rect = svgNode("rect", { x, y, width: 104, height: 26, rx: 13, class: "check-pill" });
      const label = svgNode("text", { x: x + 46, y: y + 17.5, "text-anchor": "middle", "font-size": 12.5 }, text);
      const mark = svgNode("text", { x: x + 93, y: y + 18, "text-anchor": "middle", "font-size": 15, "font-weight": 700 });
      g.append(rect, label, mark);
      return { rect, mark };
    });
    const outcome = svgNode("g", { class: "checks-outcome" });
    const badge = svgNode("rect", { x: 572, y: top + 6, width: 196, height: 34, rx: 4, fill: "#fff", "stroke-width": 1.5 });
    const line1 = svgNode("text", { x: 670, y: top + 28, "text-anchor": "middle", "font-size": 14, "font-weight": 700 });
    const line2 = svgNode("text", { x: 670, y: top + 60, "text-anchor": "middle", "font-size": 12, fill: "#555" });
    outcome.append(badge, line1, line2);
    g.append(outcome);
    svg.append(g);
    return { kind, bits, pills, outcome, badge, line1, line2 };
  });

  // Complete state of step i: what each bit shows, which check is being read, the checks' verdicts
  function render(i) {
    const active = [null, null, null, 0, 2, 1][i];
    for (const row of rows) {
      const map = row.kind === "map";
      row.bits.forEach(({ rect, value }, b) => {
        const hit = i >= 1 && jammed.includes(b);
        let text = String(sent[b]);
        let state = "";
        if (hit && i === 1) { text = "≈"; state = "noise"; }
        if (hit && i >= 2) {
          if (map) {
            const solved = (b === 1 && i >= 3) || (b === 5 && i >= 4);
            text = solved ? String(sent[b]) : "?";
            state = solved ? "solved" : "unknown";
          } else { text = "1"; state = "wrong"; }      // read wrongly, and trusted
        }
        if (!map && i === 5 && b === 2) { text = "0"; state = "wrong"; }   // the misled decoder flips d3
        value.textContent = text;
        rect.setAttribute("class", `bit-box ${hit ? "jammed" : ""} ${active != null && checks[active].includes(b) ? "reading" : ""}`);
        value.setAttribute("class", `bit-value ${state}`);
      });
      const order = [0, 2, 1];                        // c1, then c3, then c2 are read
      row.pills.forEach(({ rect, mark }, k) => {
        const read = i >= 3 && order.indexOf(k) <= i - 3;
        const verdict = read ? (map ? "ok" : "bad") : "";
        rect.setAttribute("class", `check-pill ${verdict} ${active === k ? "reading" : ""}`);
        mark.textContent = read ? (map ? "✓" : "✗") : "";
        mark.setAttribute("fill", map ? GREEN : RED);
      });
      const done = i === 5;
      row.outcome.setAttribute("opacity", done ? 1 : 0);
      row.badge.setAttribute("stroke", map ? GREEN : RED);
      row.line1.setAttribute("fill", map ? GREEN : RED);
      row.line1.textContent = map ? "All bits right · CRC pass" : "Wrong data · CRC fail";
      row.line2.textContent = map ? "data 1 0 1, as sent" : "data 1 1 0 instead of 1 0 1";
    }
    svg.setAttribute("aria-label", `Toy code, step ${i + 1} of 6`);
    return Promise.resolve();
  }

  host.querySelector(".checks-box").replaceChildren(svg);
  const steps = [
    { title: "Six bits", text: "A toy code. The phone sends 3 data bits (d) and 3 check bits (c). Each check bit is the XOR of two data bits, as written under the boxes." },
    { title: "The jammer hits two bits", text: "d2 and c3 arrive buried in noise (red outline)." },
    { title: "Unknown, or wrongly trusted", text: "With the map the receiver marks them unknown (?). The standard receiver reads them anyway. Both come out wrong (red), and it trusts them." },
    { title: "Reading check c1 = d1 ⊕ d2", text: "With the map, 1 = 1 ⊕ ? means d2 must be 0. For the standard receiver 1 ⊕ 1 = 0, but c1 is 1, which is a conflict." },
    { title: "Reading check c3 = d1 ⊕ d3", text: "With the map, 1 ⊕ 1 = 0, so the missing c3 is 0. The standard receiver gets 0 too, but it read 1, which is another conflict." },
    { title: "The result", text: "With the map, check c2 agrees too: every bit is back and the CRC passes. For the standard receiver every check conflicts. It cannot tell which bits to blame, settles on wrong data, and the CRC fails. The real code works the same way with thousands of bits and checks." },
  ].map((step, i) => ({ ...step, render: () => render(i) }));
  stepper(host, steps, { param: "bits", dwell: 3600 });
  host.setAttribute("aria-busy", "false");
}

const checksHost = document.querySelector("#checks-anim");
if (checksHost) {
  try { checksAnimation(checksHost); } catch (error) { reportError(checksHost, error); }
}

for (const [run, id] of [[methodAnimation, "#method-anim"], [aidedAnimation, "#aided-anim"]]) {
  const host = document.querySelector(id);
  if (host) run(host).catch(error => reportError(host.querySelector(".anim-left") ?? host, error));
}
