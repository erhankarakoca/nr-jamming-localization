import { receiverName } from "./data.js";
import { element } from "./nav.js";

const AXIS_LIMIT = 4;
const INSET = 29;
const SIZE = 240;
const SPAN = SIZE - 2 * INSET;
const SCALE = SPAN / (2 * AXIS_LIMIT);
const SUBCARRIERS_PER_PRB = 12;

export function crcBadge(passed) {
  return element("span", passed ? "CRC pass" : "CRC fail", `badge ${passed ? "pass" : "fail"}`);
}

export function crcList(slot, names) {
  const list = element("ul", null, "badge-list");
  for (const name of names) {
    const item = element("li", `${receiverName(name)} `);
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
  const [start, width] = slot.metadata.alloc;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const points = slot.x_hat[receiver.x_hat];
  const llrs = slot.abs_llr[receiver.abs_llr];
  const label = receiverName(name);
  const canvas = element("canvas");
  canvas.setAttribute("role", "img");
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = canvas.height = Math.round(SIZE * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(canvas.width / SIZE, canvas.height / SIZE);
  const xOf = x => SIZE / 2 + SCALE * x;
  const yOf = y => SIZE / 2 - SCALE * y;

  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(INSET, INSET, SPAN, SPAN);
  ctx.strokeStyle = "#a6b7c1";
  ctx.strokeRect(INSET, INSET, SPAN, SPAN);
  ctx.beginPath();
  ctx.moveTo(INSET, yOf(0));
  ctx.lineTo(SIZE - INSET, yOf(0));
  ctx.moveTo(xOf(0), INSET);
  ctx.lineTo(xOf(0), SIZE - INSET);
  ctx.stroke();

  ctx.font = "11px system-ui";
  ctx.fillStyle = "#455965";
  ctx.textAlign = "center";
  for (const tick of [-AXIS_LIMIT, 0, AXIS_LIMIT]) {
    ctx.fillText(tick, xOf(tick), SIZE - 14);
    ctx.fillText(tick, 16, yOf(tick) + 4);
  }
  ctx.fillText("I", SIZE - 9, SIZE - 14);
  ctx.fillText("Q", 16, 17);

  ctx.save();
  ctx.beginPath();
  ctx.rect(INSET, INSET, SPAN, SPAN);
  ctx.clip();
  const selected = new Set(prbs);
  const rowLength = width * SUBCARRIERS_PER_PRB;
  let shown = 0;
  let outside = 0;
  for (const [index, point] of points.entries()) {
    const prb = start + Math.floor((index % rowLength) / SUBCARRIERS_PER_PRB);
    if (!selected.has(prb) || point.some(v => v == null)) continue;
    const ordinal = Math.floor(index / rowLength);
    const symbol = ordinal + Number(ordinal >= dmrs);
    if (point.some(v => Math.abs(v) > AXIS_LIMIT)) {
      outside++;
      continue;
    }
    shown++;
    ctx.globalAlpha = 0.14 + 0.76 * (1 - Math.exp(-(llrs[index] ?? 0) / 14));
    ctx.fillStyle = slot.grids.truth[prb][symbol] ? "#d55e00" : "#254c63";
    ctx.beginPath();
    ctx.arc(xOf(point[0]), yOf(point[1]), 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  const extent = `I and Q axes from −${AXIS_LIMIT} to +${AXIS_LIMIT}.`;
  const description = `${label}, ${shown.toLocaleString()} points on the plot. ${extent}`;
  canvas.setAttribute("aria-label", outside ? `${description} ${outside} outside the axes.` : description);
  canvas.textContent = description;

  const figure = element("figure");
  const caption = element("figcaption");
  caption.append(element("strong", label), crcBadge(receiver.crc));
  if (outside) caption.append(element("span", `${outside.toLocaleString()} outside axes`, "context"));
  figure.append(canvas, caption);
  return figure;
}

export function constellations(slot, names, prbs) {
  const group = element("div", null, "constellation-set");
  for (const name of names) group.append(draw(slot, name, prbs));
  return group;
}
