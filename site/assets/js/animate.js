import { conditions, loadCase, selectionText, signedDb } from "./data.js";
import { drawGrid, svgNode } from "./grid.js";
import { element, options, reportError } from "./nav.js";

const timingDetectors = [
  ["fused", "U-Net, block-trained"],
  ["fused_atk", "U-Net, mixed attacks"],
];

async function timing(host) {
  const select = document.querySelector("#timing-case");
  const detector = document.querySelector("#timing-detector");
  const comparison = document.querySelector("#timing-comparison");
  const play = document.querySelector("#timing-play");
  const step = document.querySelector("#timing-next");
  const count = document.querySelector("#symbol");
  const caption = document.querySelector("#timing-caption");
  let slot = null;
  let index = 0;
  let timer = null;
  let request = 0;

  function pause() {
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
  }

  function show() {
    const cols = slot.grids.allocation[0].length;
    const pilot = index === slot.metadata.config.dmrs_symbol ? " · DMRS" : "";
    count.textContent = `Symbol ${index} of 0–${cols - 1}${pilot}`;
    host.replaceChildren(drawGrid(slot, {
      detector: detector.value, injected: true, symbol: index,
      label: `${slot.metadata.title}, symbol ${index}`,
    }));
  }

  function advance() {
    index = (index + 1) % slot.grids.allocation[0].length;
    show();
  }

  async function change() {
    const current = ++request;
    pause();
    play.disabled = step.disabled = detector.disabled = true;
    comparison.hidden = true;
    host.setAttribute("aria-busy", "true");
    count.textContent = caption.textContent = "";
    host.textContent = "Loading the recorded slot…";
    try {
      const loaded = await loadCase(select.value);
      if (current !== request) return;
      slot = loaded;
      const available = timingDetectors.filter(([key]) => Object.hasOwn(slot.detectors, key));
      options(detector, available);
      detector.disabled = available.length < 2;
      comparison.hidden = select.value !== "hopping" || detector.disabled;
      index = 0;
      show();
      caption.textContent = `${conditions(slot)}. ${selectionText(slot)}`;
      play.disabled = step.disabled = false;
    } catch (error) {
      if (current === request) reportError(host, error);
    } finally {
      if (current === request) host.setAttribute("aria-busy", "false");
    }
  }

  select.disabled = false;
  select.addEventListener("change", change);
  detector.addEventListener("change", show);
  step.addEventListener("click", () => {
    pause();
    advance();
  });
  play.addEventListener("click", () => {
    if (timer !== null) {
      pause();
    } else {
      play.textContent = "Pause";
      timer = setInterval(advance, 550);
    }
  });
  await change();
}

function line(svg, x1, y1, x2, y2, attrs) {
  svg.append(svgNode("line", { x1, y1, x2, y2, ...attrs }));
}

function chart({
  values, benign, threshold, xValues, xTitle, yTitle, yTicks, jam, subcarriers = false,
}) {
  const width = 450;
  const height = 184;
  const left = 34;
  const right = 438;
  const top = 29;
  const bottom = 144;
  const start = xValues[0];
  const end = xValues.at(-1);
  const pitch = (right - left) / (end - start + 1);
  const x = value => left + pitch * (value - start + 0.5);
  const min = yTicks[0];
  const max = yTicks.at(-1);
  const y = value => bottom - (value - min) / (max - min) * (bottom - top);
  const svg = svgNode("svg", {
    class: "sweep-chart", viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": `${yTitle} by ${xTitle}, with a paired benign reference${threshold ? " and the analytic threshold" : ""}`,
  });
  const [low, high] = jam;
  svg.append(svgNode("rect", {
    x: x(low) - pitch / 2, y: top, width: (high - low + 1) * pitch, height: bottom - top,
    fill: "#fbe8da",
  }));
  for (const value of yTicks) {
    const yy = y(value);
    line(svg, left, yy, right, yy, { stroke: "#dbe4e9", "stroke-width": 0.75 });
    svg.append(svgNode("text", {
      x: left - 5, y: yy + 4, "text-anchor": "end", "font-size": 12,
    }, String(value).replace("-", "−")));
  }
  const step = subcarriers ? 24 : 2;
  for (let value = start; value <= end; value += step) {
    svg.append(svgNode("text", {
      x: x(value), y: bottom + 15, "text-anchor": "middle", "font-size": 12,
    }, value));
  }
  line(svg, left, bottom, right, bottom, { stroke: "#8197a3" });
  line(svg, left, top, left, bottom, { stroke: "#8197a3" });
  svg.append(svgNode("text", {
    x: (left + right) / 2, y: height - 8, "text-anchor": "middle", "font-size": 12,
  }, xTitle), svgNode("text", {
    x: left, y: 16, "font-size": 13,
  }, yTitle));

  if (threshold) {
    const path = threshold.map((value, i) => {
      const edge = x(xValues[i]) - pitch / 2;
      return `${i === 0 ? `M ${edge} ${y(value)}` : `V ${y(value)}`} H ${edge + pitch}`;
    }).join(" ");
    svg.append(svgNode("path", {
      d: path, fill: "none", stroke: "#475c68", "stroke-width": 1.4, "stroke-dasharray": "3 2",
    }));
  }
  for (const [i, value] of values.entries()) {
    const xx = x(xValues[i]);
    const hit = xValues[i] >= low && xValues[i] <= high;
    const radius = subcarriers ? 1.9 : 3;
    if (value != null) {
      const dot = svgNode("circle", {
        cx: xx, cy: y(value), r: radius, fill: hit ? "#d55e00" : "#0072b2",
      });
      dot.append(svgNode("title", {}, `${xValues[i]} · ${signedDb(Number(value.toFixed(1)))}`));
      svg.append(dot);
    }
    if (benign[i] != null) {
      svg.append(svgNode("circle", {
        cx: xx, cy: y(benign[i]), r: radius, fill: "none", stroke: "#263f4e", "stroke-width": 0.8,
      }));
    }
  }
  return svg;
}

function powerDb(value, reference = 1) {
  return value > 0 ? 10 * Math.log10(value / reference) : null;
}

function axisTicks(values, step) {
  const finite = values.filter(Number.isFinite);
  const low = Math.floor((Math.min(...finite) - 1) / step) * step;
  const high = Math.ceil((Math.max(...finite) + 1) / step) * step;
  return Array.from({ length: (high - low) / step + 1 }, (_, i) => low + i * step);
}

function sweepLegend() {
  const list = element("ul", null, "legend");
  for (const [label, style] of [
    ["Jammer on, affected PRBs", "injected"], ["Jammer on, other PRBs", "flag"],
    ["Paired benign slot", "circle"], ["Analytic threshold", "dash"],
  ]) {
    const item = element("li");
    const mark = element("i", null, `swatch ${style}`);
    mark.setAttribute("aria-hidden", "true");
    item.append(mark, label);
    list.append(item);
  }
  return list;
}

async function sweep(host) {
  const slot = await loadCase("jsr_sweep");
  const slider = document.querySelector("#jsr");
  const output = document.querySelector("#jsr-value");
  const [start] = slot.metadata.alloc;
  const [jamStart, jamCount] = slot.metadata.jam;
  const pilotJam = [12 * (jamStart - start), 12 * (jamStart + jamCount - start) - 1];
  const prbJam = [jamStart, jamStart + jamCount - 1];
  // The link uses N0 = 10^(-SNR/10). dmrs_ls divides by the occupied pilots,
  // whose squared amplitude is 2 for this transmitter's two CDM groups.
  const lsNoise = 10 ** (-slot.metadata.snr_db / 10) / 2;
  const threshold = slot.prb.threshold.map(value => powerDb(value));
  const all = [slot.benign, ...slot.sweep].map(frame => ({
    jsr: frame.jsr_db,
    subcarriers: frame.pilots.subcarrier,
    pilotDb: frame.pilots.residual_power.map(value => powerDb(value, lsNoise)),
    prbDb: frame.r_b.map(value => powerDb(value)),
  }));
  const [benign, ...frames] = all;
  const pilotTicks = axisTicks(all.flatMap(frame => frame.pilotDb), 10);
  const prbTicks = axisTicks([...threshold, ...all.flatMap(frame => frame.prbDb)], 5);

  function show() {
    const frame = frames[Number(slider.value)];
    output.textContent = signedDb(frame.jsr);
    slider.setAttribute("aria-valuetext", output.textContent);
    const perPilot = chart({
      values: frame.pilotDb, benign: benign.pilotDb,
      xValues: frame.subcarriers, xTitle: "Subcarrier within allocation",
      yTitle: "Pilot residual power / LS noise variance (dB)",
      yTicks: pilotTicks, jam: pilotJam, subcarriers: true,
    });
    const perPRB = chart({
      values: frame.prbDb, benign: benign.prbDb, threshold,
      xValues: frame.prbDb.map((_, i) => start + i), xTitle: "Carrier PRB", yTitle: "Rᵦ (dB)",
      yTicks: prbTicks, jam: prbJam,
    });
    host.replaceChildren(perPilot, perPRB, sweepLegend(), element("p",
      `TDL-${slot.metadata.channel} · ${slot.metadata.snr_db} dB SNR · ${signedDb(frames[0].jsr)} to ${signedDb(frames.at(-1).jsr)} JSR. The analytic threshold targets a 1% false-alarm rate per scheduled pilot cell. ${selectionText(slot)}`,
      "context",
    ));
  }

  slider.max = frames.length - 1;
  slider.value = Math.max(0, frames.findIndex(frame => frame.jsr === slot.metadata.jsr_db));
  slider.disabled = false;
  slider.addEventListener("input", show);
  show();
  host.setAttribute("aria-busy", "false");
}

const timingHost = document.querySelector("#timing-grid");
timing(timingHost).catch(error => reportError(timingHost, error));
const sweepHost = document.querySelector("#sweep");
sweep(sweepHost).catch(error => reportError(sweepHost, error));
