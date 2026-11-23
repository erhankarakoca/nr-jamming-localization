// "How it works": the method on one recorded partial-band slot, step by step, and the data-aided
// passes on a slot whose every pilot is jammed. Every step is a complete state of the two panels
// (the resource grid on the left, the channel, the three clues or the symbols on the right), so
// any step can be reached from any other; only a move to the next step is animated.
import { conditions, loadCase, selectionText } from "./data.js";
import { crcBadge } from "./constellation.js";
import { featureScale, heatColor, svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";
import {
  BLUE, BOX, RED, allocationGrid, bitErrors, cellTrust, constellationView, decodedSymbols, drawFrame,
  fadeTo, mean, median, shareText, symbols, tween,
} from "./points.js";

const GREY = "#9a9a9a";
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
    ["map", "→ U-Net map", "blue: flagged, red outline: jammed"],
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
  // After decoding: the map-aware receiver's points on the symbols its decoded block gives
  const decoded = decodedSymbols(slot, "fused_soft");
  const errors = { standard: bitErrors(slot, "standard", before), map: bitErrors(slot, "fused_soft", after) };
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
  right.replaceChildren(plot.svg, clues.svg, canvas);
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
    { stage: 4, jam: 1, flags: 1, trust: "map", view: "points", points: decoded ? "decoded" : "map", crc: ["standard", "fused_soft"] },
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
      const to = { map: after, decoded, standard: before }[s.points] ?? before;
      if (animate && was.view === "points" && shownPoints && shownPoints !== to) {
        const a = shownPoints;
        jobs.push(tween(1700, t => drawFrame(view, a, to, t), alive));
      } else drawFrame(view, to, to, 0);
      shownPoints = to;
      canvas.setAttribute("aria-label", s.points === "decoded"
        ? `The user's ${count} PRBs after decoding with the jamming map: every data symbol on the symbol that was sent`
        : `Equalized data symbols of the user's ${count} PRBs, ${s.points === "map" ? "decoded with the jamming map" : "standard receiver"}`);
    }
    // Legend and CRC line under the panels
    const items = [["alloc", "data cell"], ["pilot", "pilot cell (symbol 2)"], ["truth", "jammed"]];
    if (s.flags) items.push(["flag-dot", "flagged by the U-Net"]);
    if (s.trust) items.push(["trust", "darker: decoder trusts the cell's bits more"]);
    if (s.view === "channel") {
      items.push(["line grey-line", "true channel"]);
      if (s.pilots) items.push(["circle", "clean pilot"], ["dot", "jammed pilot"]);
      if (s.estimate === "standard") items.push(["dash", "standard estimate"]);
      if (s.estimate === "map") items.push(["dash faint", "standard estimate"], ["line red-line", "estimate with the map"]);
    }
    if (s.points === "decoded") {
      items.push(["dot blue-dot", "clean cell"], ["dot", "jammed cell"], ["dot black-dot", "decoded wrongly"]);
    } else if (s.view === "points") items.push(["dot blue-dot", "clean cell"], ["dot", "jammed cell (fainter: less trusted)"]);
    const crcLine = element("p", null, "crc-line");
    if (s.crc) {
      s.crc.forEach((name, k) => {
        if (k) crcLine.append("  →  ");
        crcLine.append(name === "standard" ? "Standard receiver: " : "With the jamming map: ", crcBadge(slot.receivers[name].crc));
      });
    }
    legend.replaceChildren(legendItems(items), crcLine);
    await Promise.all(jobs.filter(Boolean));
  }

  const steps = [
    {
      title: "One uplink slot, and a jammer on part of it",
      text: `The user sends one slot on PRBs ${start}–${start + count - 1}: ${count} PRBs (frequency) by 14 OFDM symbols (time). Symbol ${dmrs} carries pilots, symbols the receiver knows and uses to learn the channel (grey line, right); the other 13 symbols carry data. The jammer sends noise on every symbol of PRBs ${range} (red): ${share}, so ${jammedShare} of the pilots and ${jammedShare} of the data cells.`,
    },
    {
      title: "Standard receiver: the jammed pilots pull the channel estimate",
      text: `Dividing each received pilot by the known pilot symbol gives one look at the channel. Clean pilots (open circles) sit on it; jammed pilots (red) scatter. The standard receiver interpolates between all of them (dashed), so on PRBs ${range} its squared error is ${ratio} times its error on the clean PRBs.`,
    },
    {
      title: "Standard receiver: every bit is trusted, and the block fails",
      text: `It equalizes the data and passes each bit to the decoder with a confidence (darker cells, left). The jammed cells' symbols are scattered (red points), yet their bits get almost the confidence of clean ones: mean |LLR| ${llr(before, true).toFixed(0)} against ${llr(before, false).toFixed(0)}. The decoder is misled and the CRC check fails.`,
    },
    {
      title: "Three clues per cell",
      text: `From the received slot, its pilots and the standard receiver's equalized symbols, the receiver measures for every cell: the received power P (signal plus anything else, relative to the noise), on the pilot symbol the residual R (how far the pilots of a PRB are from a smooth channel), and on the data symbols the EVM E (how far the equalized symbols are from the four QPSK points). On the jammed cells the power rises by only ${(featureNumbers.power_db.jammed - featureNumbers.power_db.clean).toFixed(1)} dB, the residual by ${(featureNumbers.residual_db.jammed - featureNumbers.residual_db.clean).toFixed(0)} dB and the EVM by ${(featureNumbers.evm_db.jammed - featureNumbers.evm_db.clean).toFixed(0)} dB (medians).`,
    },
    {
      title: "A small U-Net turns the clues into a map",
      text: `A U-Net with 66,553 parameters reads the three maps of the whole slot and scores each cell. Cells scoring above a cutoff, set so that 1% of clean cells are flagged, are marked (blue). ${hits === jammedCells && flagged === jammedCells ? `Here it flags all ${jammedCells} jammed cells and no other cell.` : `Here it flags ${hits} of the ${jammedCells} jammed cells and ${flagged - hits} other cells.`}`,
    },
    {
      title: "Decoding again: the channel from the clean pilots",
      text: `The receiver decodes the same buffered samples again, now with the map. Pilots in flagged cells get a noise variance scaled up by their residual, so they barely count (faded), and the estimate across PRBs ${range} is interpolated from the clean pilots around them (red line). Its squared error there falls by ${cut}%.`,
    },
    {
      title: "Decoding again: jammed data bits count less",
      text: `For each flagged data cell the receiver adds the jammer power it measured there to the noise variance. Those bits reach the decoder with almost no confidence (mean |LLR| ${llr(after, true).toFixed(1)}, against ${llr(after, false).toFixed(0)} for clean bits; the jammed cells turn light on the left), so the decoder fills them in from the clean ${cleanShare} of the block.`,
    },
    {
      title: decoded ? "The decoder rebuilds the jammed bits" : "The block decodes",
      text: decoded
        ? `The map did not repair the jammed symbols: ${errors.map.jammedBefore} of their bits still pointed the wrong way, but with almost no weight. The decoder rebuilt them from the clean ${cleanShare} of the block, so after decoding every point lands on the symbol that was sent (no black points) and the CRC passes. The standard receiver passed ${errors.standard.before} wrong bits at full weight; its decoded block still differed from the one sent in ${errors.standard.after} bits and its CRC failed. Over all ${batch.standard.slots} slots simulated under these conditions, the standard receiver failed ${batch.standard.failures} and the receiver with the map ${batch.fused_soft.failures}.`
        : `Same received samples, one small network and one more decoding: the CRC passes. Over all ${batch.standard.slots} slots simulated under these conditions, the standard receiver failed ${batch.standard.failures} and the receiver with the map ${batch.fused_soft.failures}.`,
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

// The data-aided passes on a slot whose every pilot is jammed: the estimate and the symbols after
// each pass, with that pass's CRC.
async function aidedAnimation(host) {
  const slot = await loadCase("aided_flip");
  const { channel, passes } = slot;
  if (!channel || !passes) throw new Error("The recorded slot has no data-aided passes.");
  const [start, width] = slot.metadata.alloc;
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
  let token = 0;
  const render = async (i, animate, from) => {
    const mine = ++token;
    const alive = () => mine === token;
    const crcLine = element("p", null, "crc-line");
    passes.slice(0, i + 1).forEach((p, k) => {
      const span = element("span", null, "pass-crc");
      span.append(k === 0 ? "first decoding " : `pass ${k} `, crcBadge(p.crc));
      crcLine.append(span);
    });
    legend.replaceChildren(legendItems([
      ["line grey-line", "true channel"], ["dot", "jammed pilot (all of them)"], ["line red-line", "estimate"],
      ["dot blue-dot", "data symbol"],
    ]), crcLine);
    if (!animate || from < 0) {
      plot.estimate.setAttribute("d", plot.path(estimate(i)));
      drawFrame(view, passSymbols[i], passSymbols[i], 0);
      return;
    }
    const a = estimate(from);
    const b = estimate(i);
    await Promise.all([
      tween(1400, t => plot.estimate.setAttribute("d", plot.path(a.map((v, k) => v + (b[k] - v) * t))), alive),
      tween(1400, t => drawFrame(view, passSymbols[from], passSymbols[i], t), alive),
    ]);
  };
  const steps = passes.map((p, i) => ({
    title: i === 0 ? "Every pilot is jammed: the first estimate is poor" : `Pass ${i}: decoded data cells act as pilots`,
    text: i === 0
      ? `All pilots are red: even down-weighted, they leave the estimate (red line) far from the channel (grey; mean squared error ${errors[0].toFixed(2)}), and the block fails.`
      : i <= firstPass || firstPass < 0
        ? `The decoder's soft output says which symbol each data cell probably carried and how sure it is. The cells it is sure of join the pilots, the estimate moves towards the channel (mean squared error ${errors[i].toFixed(2)}), and the slot is decoded again${p.crc ? ": the CRC passes, and the receiver stops here." : "."}`
        : `The block already decoded at pass ${firstPass}, where the receiver stops. This pass is shown only to watch the estimate settle (mean squared error ${errors[i].toFixed(2)}).`,
    render: (animate, from) => render(i, animate, from),
  }));
  stepper(host, steps, { param: "pass", dwell: 4200 });
  host.querySelector(".anim-note").replaceChildren(`${conditions(slot)}, every pilot of the allocation jammed. ${selectionText(slot)}`);
  host.setAttribute("aria-busy", "false");
}

for (const [run, id] of [[methodAnimation, "#method-anim"], [aidedAnimation, "#aided-anim"]]) {
  const host = document.querySelector(id);
  if (host) run(host).catch(error => reportError(host.querySelector(".anim-left") ?? host, error));
}
