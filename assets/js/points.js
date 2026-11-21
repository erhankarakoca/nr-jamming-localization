// Equalized-symbol helpers shared by the overview's animations: the symbols of one receiver on
// some PRBs, a square canvas, and one frame of a constellation morphing between two receivers.
const RED = "#cc0000";
const BLUE = "#0033cc";
const SUBCARRIERS_PER_PRB = 12;

// Equalized data symbols of one receiver on the given PRBs, in the export's order, with the
// opacity that encodes bit reliability (as on the other pages) and whether the cell was jammed.
export function symbols(slot, receiver, prbs) {
  const rx = slot.receivers[receiver];
  const points = slot.x_hat[rx.x_hat];
  const llrs = slot.abs_llr[rx.abs_llr];
  const [start, width] = slot.metadata.alloc;
  const dmrs = slot.metadata.config.dmrs_symbol;
  const rowLength = width * SUBCARRIERS_PER_PRB;
  const keep = new Set(prbs);
  const out = [];
  for (const [index, point] of points.entries()) {
    const prb = start + Math.floor((index % rowLength) / SUBCARRIERS_PER_PRB);
    if (!keep.has(prb) || point.some(v => v == null)) continue;
    const ordinal = Math.floor(index / rowLength);
    const symbol = ordinal + Number(ordinal >= dmrs);
    out.push({
      index, x: point[0], y: point[1],
      alpha: 0.15 + 0.8 * (1 - Math.exp(-(llrs[index] ?? 0) / 14)),
      jammed: Boolean(slot.grids.truth[prb][symbol]),
    });
  }
  return out;
}

export function constellationCanvas(canvas, limit) {
  const css = 320;
  const density = Math.min(3, window.devicePixelRatio || 1);
  canvas.width = canvas.height = Math.round(css * density);
  const ctx = canvas.getContext("2d");
  ctx.scale(density, density);
  const inset = 28;
  const span = css - 2 * inset;
  const at = v => css / 2 + (span / (2 * limit)) * v;
  return { ctx, css, inset, span, at };
}

export function drawFrame(view, from, to, t, limit) {
  const { ctx, css, inset, span, at } = view;
  const serif = '"STIX Two Text", "Times New Roman", Times, serif';
  ctx.clearRect(0, 0, css, css);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, css, css);
  ctx.strokeStyle = "#e6e6e6";
  ctx.lineWidth = 0.6;
  for (const tick of [-limit / 2, limit / 2]) {
    ctx.beginPath();
    ctx.moveTo(at(tick), inset);
    ctx.lineTo(at(tick), css - inset);
    ctx.moveTo(inset, css - at(tick));
    ctx.lineTo(css - inset, css - at(tick));
    ctx.stroke();
  }
  ctx.strokeStyle = "#bbb";
  ctx.beginPath();
  ctx.moveTo(at(0), inset);
  ctx.lineTo(at(0), css - inset);
  ctx.moveTo(inset, at(0));
  ctx.lineTo(css - inset, at(0));
  ctx.stroke();
  ctx.strokeStyle = "#000";
  ctx.strokeRect(inset, inset, span, span);
  ctx.fillStyle = "#000";
  ctx.font = `11px ${serif}`;
  ctx.textAlign = "center";
  for (const tick of [-limit, 0, limit]) {
    const label = String(tick).replace("-", "−");
    ctx.fillText(label, at(tick), css - inset + 14);
    ctx.textAlign = "right";
    ctx.fillText(label, inset - 4, css - at(tick) + 4);
    ctx.textAlign = "center";
  }
  ctx.font = `italic 12px ${serif}`;
  ctx.fillText("I", css - inset / 2, css - inset + 14);
  ctx.fillText("Q", inset - 10, inset - 8);

  ctx.save();
  ctx.beginPath();
  ctx.rect(inset, inset, span, span);
  ctx.clip();
  const ease = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  for (let i = 0; i < from.length; i++) {
    const a = from[i];
    const b = to[i];
    const x = a.x + (b.x - a.x) * ease;
    const y = a.y + (b.y - a.y) * ease;
    ctx.globalAlpha = a.alpha + (b.alpha - a.alpha) * ease;
    ctx.fillStyle = a.jammed ? RED : BLUE;
    ctx.beginPath();
    ctx.arc(at(x), css - at(y), 1.7, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
  // The four QPSK points
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  for (const [x, y] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const cx = at(x / Math.SQRT2);
    const cy = css - at(y / Math.SQRT2);
    ctx.beginPath();
    ctx.moveTo(cx - 5, cy);
    ctx.lineTo(cx + 5, cy);
    ctx.moveTo(cx, cy - 5);
    ctx.lineTo(cx, cy + 5);
    ctx.stroke();
  }
}
