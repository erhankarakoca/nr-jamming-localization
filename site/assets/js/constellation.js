import { receiverName } from "./data.js";
import { element } from "./nav.js";
import { symbols, tween } from "./points.js";

const AXIS_LIMIT = 4;
const INSET = 20;
const SIZE = 160;
const SPAN = SIZE - 2 * INSET;
const SCALE = SPAN / (2 * AXIS_LIMIT);
export function crcBadge(passed) {
  return element("span", passed ? "CRC pass" : "CRC fail", `badge ${passed ? "pass" : "fail"}`);
}

export function plotReceiverName(name) {
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

// One frame of the points moving from `from` (t = 0) to `to` (t = 1); points pair up by their
// position in the slot, so the same data symbol moves when the jammer's power changes.
function paint(ctx, to, from, t) {
  const xOf = x => SIZE / 2 + SCALE * x;
  const yOf = y => SIZE / 2 - SCALE * y;
  const label = v => String(v).replace("-", "−");
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, SIZE, SIZE);
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
    ctx.fillText(label(tick), xOf(tick), SIZE - 8);
    ctx.textAlign = "right";
    ctx.fillText(label(tick), INSET - 3, yOf(tick) + 3);
    ctx.textAlign = "center";
  }
  ctx.font = 'italic 10px "STIX Two Text", "Times New Roman", Times, serif';
  ctx.fillText("I", SIZE - INSET / 2, SIZE - 8);
  ctx.fillText("Q", INSET / 2 - 2, INSET - 6);

  ctx.save();
  ctx.beginPath();
  ctx.rect(INSET, INSET, SPAN, SPAN);
  ctx.clip();
  // Clean points first, so the jammed ones stay visible on top
  for (const jammed of [false, true]) {
    for (const point of to) {
      if (point.jammed !== jammed) continue;
      const before = from?.get(point.index) ?? point;
      const x = before.x + (point.x - before.x) * t;
      const y = before.y + (point.y - before.y) * t;
      if (Math.abs(x) > AXIS_LIMIT + 0.5 || Math.abs(y) > AXIS_LIMIT + 0.5) continue;
      ctx.globalAlpha = before.alpha + (point.alpha - before.alpha) * t;
      ctx.fillStyle = point.jammed ? "#cc0000" : "#0033cc";
      ctx.beginPath();
      ctx.arc(xOf(x), yOf(y), 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

// `from` (optional): the same slot at another jammer power; the points then move into place.
// `alive` stops that movement when a newer drawing replaces this one.
function draw(slot, name, prbs, { from = null, alive = () => true } = {}) {
  const receiver = slot.receivers[name];
  const label = plotReceiverName(name);
  const canvas = element("canvas");
  canvas.setAttribute("role", "img");
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = canvas.height = Math.round(SIZE * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(canvas.width / SIZE, canvas.height / SIZE);
  const to = symbols(slot, name, prbs);
  const previous = from?.receivers[name] ? new Map(symbols(from, name, prbs).map(p => [p.index, p])) : null;
  paint(ctx, to, previous, previous ? 0 : 1);
  if (previous) tween(450, t => paint(ctx, to, previous, t), alive);
  const outside = to.filter(({ x, y }) => Math.abs(x) > AXIS_LIMIT || Math.abs(y) > AXIS_LIMIT).length;
  const shown = to.length - outside;
  const extent = `I and Q axes from −${AXIS_LIMIT} to +${AXIS_LIMIT}.`;
  const outsideLabel = `${outside.toLocaleString()} ${outside === 1 ? "point" : "points"} outside the axes`;
  const description = `${label}, ${shown.toLocaleString()} points on the plot. ${extent} ${outsideLabel}.`;
  canvas.setAttribute("aria-label", description);
  canvas.textContent = description;

  const figure = element("figure");
  const caption = element("figcaption");
  caption.append(element("strong", label), crcBadge(receiver.crc));
  if (outside) caption.append(element("span", outsideLabel, "context"));
  figure.append(canvas, caption);
  return figure;
}

export function constellations(slot, names, prbs, motion = {}) {
  const group = element("div", null, "constellation-set");
  for (const name of names) group.append(draw(slot, name, prbs, motion));
  return group;
}
