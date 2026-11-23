import { receiverName } from "./data.js";
import { element } from "./nav.js";
import { bitErrors, decodedSymbols, symbols, tween } from "./points.js";

const AXIS_LIMIT = 4;
// After decoding every point sits near one of the four QPSK symbols, so the axes zoom in
const DECODED_LIMIT = 1.6;
const INSET = 20;
const SIZE = 160;
const SPAN = SIZE - 2 * INSET;
const limitOf = stage => (stage === "after" ? DECODED_LIMIT : AXIS_LIMIT);
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

// One frame of the points moving from `from` (t = 0) to `to` (t = 1); points pair up by their
// position in the slot, so the same data symbol moves when the jammer's power changes. The axes
// run from -limit to +limit, the limit itself moving between the two drawings' limits.
function paint(ctx, to, from, t, [fromLimit, toLimit], ordered) {
  const limit = fromLimit + (toLimit - fromLimit) * t;
  const scale = SPAN / (2 * limit);
  const xOf = x => SIZE / 2 + scale * x;
  const yOf = y => SIZE / 2 - scale * y;
  const label = v => String(Number(v.toFixed(1))).replace("-", "−");
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
  for (const tick of [-limit, 0, limit]) {
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
  // Clean points first, so the jammed ones stay visible on top (or in slot order, so the
  // decoded clouds show their mix of jammed and clean symbols)
  for (const jammed of ordered ? [null] : [false, true]) {
    for (const point of to) {
      if (jammed !== null && point.jammed !== jammed) continue;
      const before = from?.get(point.index) ?? point;
      const x = before.x + (point.x - before.x) * t;
      const y = before.y + (point.y - before.y) * t;
      if (Math.abs(x) > limit + 0.5 || Math.abs(y) > limit + 0.5) continue;
      const alpha = before.alpha + (point.alpha - before.alpha) * t;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = point.jammed ? "#cc0000" : "#0033cc";
      ctx.beginPath();
      ctx.arc(xOf(x), yOf(y), 1.1, 0, Math.PI * 2);
      ctx.fill();
      // A symbol the decoder got wrong turns black once the points have landed (same size and
      // place, so the share of black matches the share of wrong symbols)
      if (point.wrong && t > 0.6) {
        ctx.globalAlpha = (t - 0.6) / 0.4;
        ctx.fillStyle = "#000";
        ctx.fill();
      }
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

// The points of one receiver at one decoding stage: as received ("before") or placed on the
// symbols its decoded block maps to ("after"; falls back to the received points without bits)
function pointsFor(slot, name, prbs, stage) {
  return (stage === "after" && decodedSymbols(slot, name, prbs)) || symbols(slot, name, prbs);
}

function ledger(slot, name, points) {
  const errors = bitErrors(slot, name, points);
  if (!errors) return null;
  const line = element("span", null, "ledger");
  line.title = `Coded bits of the shown symbols whose value differs from the one sent, out of ${errors.total.toLocaleString()}`;
  line.append("Wrong bits ", element("b", errors.before.toLocaleString()), " → ",
    element("b", errors.after.toLocaleString(), errors.after ? "bad" : "good"), " after decoding");
  return line;
}

// `stage`: "before" or "after" decoding. `from` (optional, { slot, stage }): the previous drawing
// (another jammer power or the other stage); the points then move into place. `alive` stops that
// movement when a newer drawing replaces this one.
function draw(slot, name, prbs, { from = null, alive = () => true, stage = "before" } = {}) {
  const receiver = slot.receivers[name];
  const label = plotReceiverName(name);
  const canvas = element("canvas");
  canvas.setAttribute("role", "img");
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = canvas.height = Math.round(SIZE * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(canvas.width / SIZE, canvas.height / SIZE);
  const to = pointsFor(slot, name, prbs, stage);
  const previous = from?.slot?.receivers[name]
    ? new Map(pointsFor(from.slot, name, prbs, from.stage).map(p => [p.index, p])) : null;
  const limits = [limitOf(previous ? from.stage : stage), limitOf(stage)];
  const ordered = stage === "after";
  paint(ctx, to, previous, previous ? 0 : 1, limits, ordered);
  if (previous) tween(from.stage === stage ? 450 : 1100, t => paint(ctx, to, previous, t, limits, ordered), alive);
  const outside = to.filter(({ x, y }) => Math.abs(x) > limitOf(stage) || Math.abs(y) > limitOf(stage)).length;
  const shown = to.length - outside;
  const extent = `I and Q axes from −${limitOf(stage)} to +${limitOf(stage)}.`;
  const outsideLabel = `${outside.toLocaleString()} ${outside === 1 ? "point" : "points"} outside the axes`;
  const wrong = to.filter(p => p.wrong).length;
  const description = stage === "after"
    ? `${label} after decoding: each of ${to.length.toLocaleString()} symbols sits on the symbol the decoded block gives; ${wrong.toLocaleString()} of them differ from the symbol sent.`
    : `${label}, ${shown.toLocaleString()} points on the plot. ${extent} ${outsideLabel}.`;
  canvas.setAttribute("aria-label", description);
  canvas.textContent = description;

  const figure = element("figure");
  const caption = element("figcaption");
  caption.append(element("strong", label), crcBadge(receiver.crc));
  const count = ledger(slot, name, to);
  if (count) caption.append(count);
  if (stage === "before" && outside) caption.append(element("span", outsideLabel, "context"));
  figure.append(canvas, caption);
  return figure;
}

export function constellations(slot, names, prbs, motion = {}) {
  const group = element("div", null, "constellation-set");
  for (const name of names) group.append(draw(slot, name, prbs, motion));
  return group;
}
