// The overview page: a teaser that decodes one slot with and without the jamming map, and a strip
// of the method's intermediate maps for another slot. Both read the exported slot format, so a
// live simulator could feed them the same way.
import { conditions, loadCase, selectionText, slotDetails } from "./data.js";
import { affectedPRBs, crcBadge } from "./constellation.js";
import { drawGrid, featureScale, heatColor, svgNode } from "./grid.js";
import { element, reportError } from "./nav.js";
import { constellationCanvas, drawFrame, symbols } from "./points.js";

const RED = "#cc0000";
const BLUE = "#0033cc";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function jammedRange(slot) {
  const dmrs = slot.metadata.config.dmrs_symbol;
  const prbs = slot.grids.truth.flatMap((row, prb) => (row[dmrs] ? [prb] : []));
  return prbs.length ? `PRBs ${prbs[0]}–${prbs.at(-1)}` : "no PRB";
}

async function teaser() {
  const host = document.querySelector("#teaser-grid");
  const slot = await loadCase("walkthrough_flip");
  host.replaceChildren(drawGrid(slot, {
    detector: "fused", compact: true, label: "Jammed pilot cells and the U-Net flags",
  }));
  host.setAttribute("aria-busy", "false");

  const limit = 2;
  const prbs = affectedPRBs(slot);
  const standard = symbols(slot, "standard", prbs);
  const mapped = symbols(slot, "fused_soft", prbs);
  const canvas = document.querySelector("#teaser-canvas");
  const view = constellationCanvas(canvas, limit);
  const buttons = { standard: document.querySelector("#show-standard"), map: document.querySelector("#show-map") };
  const crc = document.querySelector("#teaser-crc");
  let t = 0;
  let frame = null;

  function label(target) {
    const name = target ? "fused_soft" : "standard";
    const text = target ? "With the jamming map: " : "Standard receiver: ";
    crc.replaceChildren(text, crcBadge(slot.receivers[name].crc));
    buttons.standard.setAttribute("aria-pressed", String(!target));
    buttons.map.setAttribute("aria-pressed", String(Boolean(target)));
    canvas.setAttribute("aria-label", `Equalized data symbols of ${jammedRange(slot)}, ${target ? "decoded with the jamming map" : "standard receiver"}`);
  }

  function go(target) {
    cancelAnimationFrame(frame);
    label(target);
    if (reduceMotion) {
      t = target;
      drawFrame(view, standard, mapped, t, limit);
      return;
    }
    const begin = performance.now();
    const t0 = t;
    const step = now => {
      const k = Math.min(1, (now - begin) / 900);
      t = t0 + (target - t0) * k;
      drawFrame(view, standard, mapped, t, limit);
      if (k < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
  }

  buttons.standard.addEventListener("click", () => go(0));
  buttons.map.addEventListener("click", () => go(1));
  label(0);
  drawFrame(view, standard, mapped, 0, limit);

  // Play the switch once when the teaser first comes into view
  let played = false;
  new IntersectionObserver((entries, observer) => {
    if (played || !entries.some(e => e.isIntersecting)) return;
    played = true;
    observer.disconnect();
    setTimeout(() => go(1), 1200);
  }, { threshold: 0.6 }).observe(canvas);

  const caption = document.querySelector("#teaser-caption");
  caption.replaceChildren(
    `${conditions(slot)}. The jammer hits only the pilot (DMRS) symbol of ${jammedRange(slot)}. `,
    "With its channel estimate spoiled, the standard receiver fails the CRC; decoding the same ",
    "received samples again with the U-Net map passes. Blue: clean cells, red: jammed cells, darker: ",
    "more reliable bits. ",
    element("span", selectionText(slot), "context"),
  );
}

// One small map of the allocation (PRBs on the vertical axis, OFDM symbols on the horizontal one)
function miniGrid(slot, { values = null, flags = null, truth = false, label }) {
  const [start, count] = slot.metadata.alloc;
  const cols = slot.grids.allocation[0].length;
  const cw = 10;
  const ch = 7;
  const scale = values ? featureScale(values.slice(start, start + count)) : null;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${cols * cw} ${count * ch}`, class: "mini-grid", role: "img", "aria-label": label,
  });
  for (let r = 0; r < count; r++) {
    const prb = start + count - 1 - r;
    for (let s = 0; s < cols; s++) {
      const value = values?.[prb][s];
      let fill = "#fff";
      if (scale && Number.isFinite(value)) fill = heatColor(value, scale);
      else if (values) fill = "#f0f0f0";
      if (flags?.[prb][s]) fill = BLUE;
      svg.append(svgNode("rect", {
        x: s * cw, y: r * ch, width: cw, height: ch, fill, stroke: "#c8c8c8", "stroke-width": 0.3,
      }));
      if (truth && slot.grids.truth[prb][s]) {
        svg.append(svgNode("rect", {
          x: s * cw + 0.7, y: r * ch + 0.7, width: cw - 1.4, height: ch - 1.4,
          fill: "none", stroke: RED, "stroke-width": 1.2,
        }));
      }
    }
  }
  svg.append(svgNode("rect", {
    x: 0, y: 0, width: cols * cw, height: count * ch, fill: "none", stroke: "#000", "stroke-width": 0.6,
  }));
  return { svg, scale };
}

function panel(title, body, note) {
  const fig = element("figure", null, "strip-panel");
  fig.append(element("figcaption", title), body);
  if (note) fig.append(element("p", note, "panel-note"));
  return fig;
}

async function methodStrip() {
  const host = document.querySelector("#method-strip");
  const slot = await loadCase("pilot");
  const range = scale => (scale ? `${scale[0]} to ${scale[1]} dB` : "");
  const panels = [];
  for (const [key, title, note] of [
    ["power_db", "(a) Received power P", "signal and jammer together"],
    ["residual_db", "(b) Pilot residual R", "only on the DMRS symbol"],
    ["evm_db", "(c) EVM E", "spoiled estimate shows on the data"],
  ]) {
    const { svg, scale } = miniGrid(slot, { values: slot.grids[key], label: title });
    panels.push(panel(title, svg, `${range(scale)}; ${note}`));
  }
  const map = miniGrid(slot, { flags: slot.detectors.fused.decisions, truth: true, label: "U-Net map" });
  panels.push(panel("(d) U-Net map", map.svg, "blue: flagged, red: jammed"));

  const canvas = element("canvas");
  const view = constellationCanvas(canvas, 2);
  const prbs = affectedPRBs(slot);
  const points = symbols(slot, "fused_soft", prbs);
  drawFrame(view, points, points, 0, 2);
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Equalized symbols of the jammed PRBs, decoded with the map");
  panels.push(panel("(e) Decoding with the map", canvas, "flagged pilots and bits weigh less"));

  host.replaceChildren(...panels);
  host.setAttribute("aria-busy", "false");
  const caption = document.querySelector("#method-caption");
  caption.replaceChildren(
    `${conditions(slot)}, pilot jamming of ${jammedRange(slot)}. Each panel shows the UE's PRBs ${slot.metadata.alloc[0]}–${slot.metadata.alloc[0] + slot.metadata.alloc[1] - 1} (bottom to top) and OFDM symbols 0–13 (left to right). `,
    slotDetails(slot),
  );
}

for (const [run, id] of [[teaser, "#teaser-grid"], [methodStrip, "#method-strip"]]) {
  const host = document.querySelector(id);
  if (host) run().catch(error => reportError(host, error));
}
