// The overview's teaser: one recorded partial-band slot decoded twice, by the standard receiver
// and again with the jamming map. The grid shows where the jammer was, what the U-Net flagged and
// how much the decoder trusts each cell; the constellation morphs between the two decodings.
import { conditions, loadCase } from "./data.js";
import { crcBadge } from "./constellation.js";
import { element, reportError } from "./nav.js";
import {
  allocationGrid, cellTrust, constellationView, drawFrame, fadeTo, shareText, symbols, tween,
} from "./points.js";

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function legend(items) {
  const list = element("ul", null, "legend");
  for (const [className, label] of items) {
    const li = element("li");
    const mark = element("i", null, `swatch ${className}`);
    mark.setAttribute("aria-hidden", "true");
    li.append(mark, label);
    list.append(li);
  }
  return list;
}

async function teaser() {
  const slot = await loadCase("partial_flip");
  const [start, count] = slot.metadata.alloc;
  const jammedPRBs = Array.from({ length: count }, (_, b) => start + b)
    .filter(p => slot.grids.truth[p].some(Boolean));
  const range = jammedPRBs.length ? `PRBs ${jammedPRBs[0]}–${jammedPRBs.at(-1)}` : "no PRB";
  const before = symbols(slot, "standard");
  const after = symbols(slot, "fused_soft");
  const grid = allocationGrid(slot, {
    trust: { standard: cellTrust(before), map: cellTrust(after) },
    label: `The user's PRBs ${start}–${start + count - 1} over one slot; jammed: ${range}`,
  });
  const gridHost = document.querySelector("#teaser-grid");
  gridHost.replaceChildren(grid.svg);
  gridHost.setAttribute("aria-busy", "false");

  const canvas = document.querySelector("#teaser-canvas");
  const view = constellationView(canvas, 2.5);
  const buttons = { standard: document.querySelector("#show-standard"), map: document.querySelector("#show-map") };
  const crc = document.querySelector("#teaser-crc");
  let t = 0;
  let token = 0;

  function label(mapped) {
    const name = mapped ? "fused_soft" : "standard";
    crc.replaceChildren(mapped ? "With the jamming map: " : "Standard receiver: ",
      crcBadge(slot.receivers[name].crc));
    buttons.standard.setAttribute("aria-pressed", String(!mapped));
    buttons.map.setAttribute("aria-pressed", String(mapped));
    canvas.setAttribute("aria-label", `Equalized data symbols of the user's ${count} PRBs, ${mapped ? "decoded with the jamming map" : "standard receiver"}`);
  }

  async function go(target, animate = !reduceMotion) {
    const mine = ++token;
    const alive = () => mine === token;
    label(Boolean(target));
    const layers = grid.layers;
    const jobs = [
      animate ? fadeTo(layers.flags, target, 700, alive) : layers.flags.setAttribute("opacity", target),
      animate ? fadeTo(layers.trust_map, target, 900, alive) : layers.trust_map.setAttribute("opacity", target),
      animate ? fadeTo(layers.trust_standard, 1 - target, 900, alive) : layers.trust_standard.setAttribute("opacity", 1 - target),
    ];
    if (!animate) {
      t = target;
      drawFrame(view, before, after, t);
      return;
    }
    const t0 = t;
    jobs.push(tween(1100, k => { t = t0 + (target - t0) * k; drawFrame(view, before, after, t); }, alive));
    await Promise.all(jobs.filter(Boolean));
  }

  buttons.standard.addEventListener("click", () => go(0));
  buttons.map.addEventListener("click", () => go(1));
  grid.layers.jam.setAttribute("opacity", 1);
  go(0, false);
  document.querySelector("#teaser-legend").replaceChildren(legend([
    ["pilot", "pilot cells (symbol 2)"], ["truth", "jammed"], ["flag-dot", "flagged by the U-Net"],
    ["trust", "darker: the decoder trusts the cell's bits more"],
    ["dot blue-dot", "clean cell"], ["dot", "jammed cell"],
  ]));

  // Play the switch once when the teaser first comes into view
  if (!reduceMotion) {
    let played = false;
    new IntersectionObserver((entries, observer) => {
      if (played || !entries.some(e => e.isIntersecting)) return;
      played = true;
      observer.disconnect();
      setTimeout(() => go(1), 1500);
    }, { threshold: 0.6 }).observe(canvas);
  }

  const share = shareText(jammedPRBs.length / count);
  const counts = slot.metadata.batch_crc;
  document.querySelector("#teaser-caption").replaceChildren(
    `${conditions(slot)}. The jammer sends noise on every symbol of ${range} (red): ${share} of the user's pilots and ${share} of its data cells. `,
    "The standard receiver estimates the channel from every pilot and trusts every cell's bits (dark); the CRC fails. ",
    "Decoding the same received samples again with the map: the U-Net flags the jammed cells (blue), the receiver estimates the channel from the clean pilots and gives the jammed cells' bits little weight (light); the CRC passes. ",
    element("span", `Of ${counts.standard.slots} slots simulated under these conditions, the standard receiver failed ${counts.standard.failures} and the receiver with the map ${counts.fused_soft.failures || "none"}; this is the first slot the map recovers.`, "context"),
  );
}

const host = document.querySelector("#teaser-grid");
if (host) teaser().catch(error => reportError(host, error));
