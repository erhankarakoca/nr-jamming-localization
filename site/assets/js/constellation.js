import { receiverName } from "./data.js";
import { element } from "./nav.js";
import { symbols } from "./points.js";

const AXIS_LIMIT = 4;
const INSET = 20;
const SIZE = 160;
const SPAN = SIZE - 2 * INSET;
const SCALE = SPAN / (2 * AXIS_LIMIT);
export function crcBadge(passed) {
  return element("span", passed ? "CRC pass" : "CRC fail", `badge ${passed ? "pass" : "fail"}`);
}

function plotReceiverName(name) {
  if (name === "standard") return "Standard receiver";
  if (name === "fused_soft") return "Map-aware receiver";
  return receiverName(name);
}

export function crcList(slot, names) {
  const list = element("ul", null, "badge-list");
  for (const name of names) {
    const item = element("li", `${plotReceiverName(name)} `);
    item.append(crcBadge(slot.receivers[name].crc));
    list.append(item);
  }
  return list;
}

export function affectedPRBs(slot) {
  const [start, count] = slot.metadata.alloc;
  const scheduled = Array.from({ length: count }, (_, i) => start + i);
  const hit = scheduled.filter(prb => slot.grids.truth[prb].some(Boolean));
  return hit.length ? hit : scheduled;
}

export function comparisonReceivers(slot) {
  return ["standard", "fused_soft", ...Object.keys(slot.receivers).filter(name =>
    name.startsWith("fused_atk") || (slot.metadata.name === "hopping" && name === "true_prb_soft") ||
    (["hopping", "pilot_whole_comb"].includes(slot.metadata.name) && name === "true_soft"),
  )];
}

function draw(slot, name, prbs) {
  const receiver = slot.receivers[name];
  const label = plotReceiverName(name);
  const canvas = element("canvas");
  canvas.setAttribute("role", "img");
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = canvas.height = Math.round(SIZE * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(canvas.width / SIZE, canvas.height / SIZE);
  const xOf = x => SIZE / 2 + SCALE * x;
  const yOf = y => SIZE / 2 - SCALE * y;

  ctx.lineWidth = 0.6;
  ctx.strokeStyle = "#000";
  ctx.strokeRect(INSET, INSET, SPAN, SPAN);
  ctx.strokeStyle = "#d9d9d9";
  ctx.beginPath();
  ctx.moveTo(INSET, yOf(0));
  ctx.lineTo(SIZE - INSET, yOf(0));
  ctx.moveTo(xOf(0), INSET);
  ctx.lineTo(xOf(0), SIZE - INSET);
  ctx.stroke();

  ctx.font = '10px "STIX Two Text", "Times New Roman", Times, serif';
  ctx.fillStyle = "#000";
  ctx.textAlign = "center";
  for (const tick of [-AXIS_LIMIT, 0, AXIS_LIMIT]) {
    ctx.fillText(String(tick).replace("-", "−"), xOf(tick), SIZE - 8);
    ctx.textAlign = "right";
    ctx.fillText(String(tick).replace("-", "−"), INSET - 3, yOf(tick) + 3);
    ctx.textAlign = "center";
  }
  ctx.font = 'italic 10px "STIX Two Text", "Times New Roman", Times, serif';
  ctx.fillText("I", SIZE - INSET / 2, SIZE - 8);
  ctx.fillText("Q", INSET / 2 - 2, INSET - 6);

  ctx.save();
  ctx.beginPath();
  ctx.rect(INSET, INSET, SPAN, SPAN);
  ctx.clip();
  let shown = 0;
  let outside = 0;
  for (const { x, y, alpha, jammed } of symbols(slot, name, prbs)) {
    if (Math.abs(x) > AXIS_LIMIT || Math.abs(y) > AXIS_LIMIT) {
      outside++;
      continue;
    }
    shown++;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = jammed ? "#cc0000" : "#0033cc";
    ctx.beginPath();
    ctx.arc(xOf(x), yOf(y), 1.1, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  const extent = `I and Q axes from −${AXIS_LIMIT} to +${AXIS_LIMIT}.`;
  const outsideLabel = `${outside.toLocaleString()} ${outside === 1 ? "point" : "points"} outside the axes`;
  const description = `${label}, ${shown.toLocaleString()} points on the plot. ${extent} ${outsideLabel}.`;
  canvas.setAttribute("aria-label", description);
  canvas.textContent = description;

  const figure = element("figure");
  const caption = element("figcaption");
  caption.append(element("strong", label), crcBadge(receiver.crc), element("span", outsideLabel, "context"));
  figure.append(canvas, caption);
  return figure;
}

export function constellations(slot, names, prbs) {
  const group = element("div", null, "constellation-set");
  for (const name of names) group.append(draw(slot, name, prbs));
  return group;
}
