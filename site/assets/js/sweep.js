import { loadCase, signedDb, slotDetails } from "./data.js";
import { svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";

const RED = "#cc0000";
const BLUE = "#0033cc";

function line(svg, x1, y1, x2, y2, attrs) {
  svg.append(svgNode("line", { x1, y1, x2, y2, ...attrs }));
}

// A plain plot in the style of the paper's residual figure: the benign slot as hollow blue
// circles, the jammed slot in red, the jammed block shaded grey and the threshold dashed.
function chart({
  values, benign, threshold, xValues, xTitle, yTitle, yTicks, jam, subcarriers = false,
}) {
  const width = 400;
  const height = 170;
  const left = 34;
  const right = 392;
  const top = 10;
  const bottom = 134;
  const start = xValues[0];
  const end = xValues.at(-1);
  const pitch = (right - left) / (end - start + 1);
  const x = value => left + pitch * (value - start + 0.5);
  const min = yTicks[0];
  const max = yTicks.at(-1);
  const y = value => bottom - (value - min) / (max - min) * (bottom - top);
  // yTitle is a list of [text, subscript] pieces, so R_b can be typeset.
  const yText = yTitle.map(([text]) => text).join("");
  const svg = svgNode("svg", {
    class: "sweep-chart", viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": `${yText} by ${xTitle}, with the same slot without jamming${threshold ? " and the 1% threshold" : ""}`,
  });
  const [low, high] = jam;
  svg.append(svgNode("rect", {
    x: x(low) - pitch / 2, y: top, width: (high - low + 1) * pitch, height: bottom - top, fill: "#e6e6e6",
  }));
  for (const value of yTicks) {
    const yy = y(value);
    line(svg, left, yy, right, yy, { stroke: "#d9d9d9", "stroke-width": 0.5 });
    svg.append(svgNode("text", {
      x: left - 4, y: yy + 3, "text-anchor": "end", "font-size": 9,
    }, String(value).replace("-", "−")));
  }
  const step = subcarriers ? 24 : 2;
  for (let value = start; value <= end; value += step) {
    svg.append(svgNode("text", {
      x: x(value), y: bottom + 11, "text-anchor": "middle", "font-size": 9,
    }, value));
  }
  svg.append(svgNode("rect", {
    x: left, y: top, width: right - left, height: bottom - top, fill: "none", stroke: "#000", "stroke-width": 0.6,
  }));
  svg.append(svgNode("text", {
    x: (left + right) / 2, y: height - 8, "text-anchor": "middle", "font-size": 9.5,
  }, xTitle));
  const label = svgNode("text", {
    x: 9, y: (top + bottom) / 2, "text-anchor": "middle", "font-size": 9.5,
    transform: `rotate(-90 9 ${(top + bottom) / 2})`,
  });
  for (const [text, subscript] of yTitle) {
    label.append(subscript
      ? svgNode("tspan", { "baseline-shift": "sub", "font-size": 7, "font-style": "italic" }, text)
      : svgNode("tspan", text === "R" ? { "font-style": "italic" } : {}, text));
  }
  svg.append(label);

  if (threshold) {
    const path = threshold.map((value, i) => {
      const edge = x(xValues[i]) - pitch / 2;
      return `${i === 0 ? `M ${edge} ${y(value)}` : `V ${y(value)}`} H ${edge + pitch}`;
    }).join(" ");
    svg.append(svgNode("path", {
      d: path, fill: "none", stroke: "#000", "stroke-width": 0.9, "stroke-dasharray": "4 2",
    }));
  }
  const radius = subcarriers ? 1.6 : 2.6;
  for (const [i, value] of values.entries()) {
    const xx = x(xValues[i]);
    if (benign[i] != null) {
      svg.append(svgNode("circle", {
        cx: xx, cy: y(benign[i]), r: radius, fill: "none", stroke: BLUE, "stroke-width": 0.7,
      }));
    }
    if (value != null) {
      const dot = svgNode("circle", { cx: xx, cy: y(value), r: radius, fill: RED });
      dot.append(svgNode("title", {}, `${xValues[i]} · ${signedDb(Number(value.toFixed(1)))}`));
      svg.append(dot);
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
    ["Jammed slot", "dot"], ["Same slot, no jammer", "circle"],
    ["Jammed PRBs", "alloc"], ["1% threshold", "dash"],
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

  // The per-pilot view sits in a disclosure that stays open while the slider moves.
  const perPRBHost = element("div");
  const perPilotHost = element("div");
  const perPilot = element("details", null, "more");
  const note = element("p", "Residual power of each pilot over the noise variance of its least-squares estimate; ");
  const r = element("var", "R");
  r.append(element("sub", "b"));
  note.append(r, " combines the pilots of PRB ", element("var", "b"), ".");
  perPilot.append(element("summary", "Per pilot subcarrier"), note, perPilotHost);
  const caption = element("div", null, "context");
  const m = slot.metadata;
  caption.append(`TDL-${m.channel} · ${m.snr_db} dB SNR · JSR from ${signedDb(frames[0].jsr)} to ${signedDb(frames.at(-1).jsr)}`,
    slotDetails(slot));
  host.replaceChildren(perPRBHost, sweepLegend(), perPilot, caption);

  function show() {
    const frame = frames[Number(slider.value)];
    output.textContent = signedDb(frame.jsr);
    slider.setAttribute("aria-valuetext", output.textContent);
    perPRBHost.replaceChildren(chart({
      values: frame.prbDb, benign: benign.prbDb, threshold,
      xValues: frame.prbDb.map((_, i) => start + i), xTitle: "PRB", yTitle: [["R"], ["b", true], [" (dB)"]],
      yTicks: prbTicks, jam: prbJam,
    }));
    perPilotHost.replaceChildren(chart({
      values: frame.pilotDb, benign: benign.pilotDb,
      xValues: frame.subcarriers, xTitle: "Pilot subcarrier within the allocation",
      yTitle: [["Pilot residual (dB)"]], yTicks: pilotTicks, jam: pilotJam, subcarriers: true,
    }));
  }

  slider.max = frames.length - 1;
  slider.value = Math.max(0, frames.findIndex(frame => frame.jsr === slot.metadata.jsr_db));
  slider.disabled = false;
  slider.addEventListener("input", show);
  show();
  host.setAttribute("aria-busy", "false");
}

const sweepHost = document.querySelector("#sweep");
sweep(sweepHost).catch(error => reportError(sweepHost, error));
