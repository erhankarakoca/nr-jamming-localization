import { detectors, loadCase, loadResults, percent, receiverName, signedDb } from "./data.js";
import { element, options, reportError } from "./nav.js";

const attacks = { all: "All attacks", pilot: "Pilot", data: "Data", partial: "Partial-band" };
const jammers = { nominal: "Nominal", pulsed: "Pulsed", hopping: "Hopping" };
const detectorName = name => detectors[name] ?? name;
const decimal = (value, digits = 3) => value == null ? "—" : value.toFixed(digits).replace("-", "−");
const interval = (lo, hi, format = decimal) => lo == null && hi == null ? "—" : `[${format(lo)}, ${format(hi)}]`;
const percentagePoints = value => decimal(value == null ? null : value * 100, 2);
const integer = value => value.toLocaleString("en-US");

function table(caption, labels, rows) {
  const node = element("table");
  node.createCaption().textContent = caption;
  const head = node.createTHead().insertRow();
  labels.forEach(label => {
    const cell = element("th", label);
    cell.scope = "col";
    head.append(cell);
  });
  const body = node.createTBody();
  rows.forEach(row => {
    const tr = body.insertRow();
    row.forEach((value, i) => {
      const cell = element(i === 0 ? "th" : "td");
      if (i === 0) cell.scope = "row";
      cell.append(element("span", value, i ? "numeric" : null));
      tr.append(cell);
    });
  });
  const scroll = element("div", null, "table-scroll");
  scroll.setAttribute("tabindex", "0");
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", caption);
  scroll.append(node);
  return scroll;
}

function sectionContent(id, section) {
  const host = document.getElementById(id);
  host.querySelector(".population").textContent = section.population;
  const links = section.downloads.map(path => {
    const link = element("a", path.split("/").at(-1));
    link.href = path;
    link.download = "";
    return link;
  });
  host.querySelector(".downloads").replaceChildren(...links);
}

// The detector styles of the paper's Pd figure (evaluate.DETECTOR_STYLE): colour, hollow
// marker and dash, so a curve looks the same on the site and in the paper.
const STYLE = {
  fused: ["#cc0000", "circle-open", "solid"],
  no_residual: ["#0033cc", "square-open", "dash"],
  power: ["#000000", "triangle-up-open", "dashdot"],
  evm: ["#cc00cc", "triangle-down-open", "dot"],
  evm_threshold: ["#0099cc", "triangle-right-open", "dot"],
  residual_threshold: ["#009e73", "x-thin-open", "dash"],
  power_threshold: ["#666666", "star-open", "dot"],
  residual_cfar: ["#8b4513", "x-thin-open", "dashdot"],
  block_test: ["#008000", "diamond-open", "solid"],
  fused_nj1: ["#332288", "diamond-open", "dot"],
  fused_pixel: ["#44aa99", "hexagon-open", "dash"],
  fused_pixel_wide: ["#117733", "hexagon2-open", "dash"],
  fused_comb: ["#aa4499", "pentagon-open", "solid"],
};
const OTHER = ["#8b4513", "cross-thin-open", "longdash"];
const priority = ["fused", "no_residual", "block_test", "evm_threshold", "residual_threshold"];
// The chart has the detectors of the paper's figure (evaluate.PLOT_DETECTORS); the table all.
const PLOTTED = ["fused", "no_residual", "power", "evm", "block_test", "evm_threshold", "residual_cfar", "power_threshold"];
const SHOWN = new Set(["fused", "no_residual", "block_test", "evm_threshold"]);
// The receivers of the short recovery table; the full table is under "How this was measured".
const KEY_RECEIVERS = ["standard", "lmmse", "block_soft", "p_r_e_soft", "p_r_e_atk_soft", "true_soft"];
const SERIF = '"STIX Two Text", "Times New Roman", Times, serif';
const axis = title => ({
  title: { text: title, standoff: 6 }, showline: true, linecolor: "#000", linewidth: 0.8, mirror: "ticks",
  ticks: "inside", ticklen: 4, tickcolor: "#000", gridcolor: "#e6e6e6", zeroline: false, automargin: true,
});

function renderPlot(id, traces, x, y, extras = {}) {
  const host = document.getElementById(id);
  if (!window.Plotly) {
    host.textContent = "Chart unavailable. The data table and CSV are available below.";
    host.classList.add("context");
    host.nextElementSibling.open = true;
    return;
  }
  const layout = {
    autosize: true,
    height: 320,
    margin: { l: 52, r: 12, t: 8, b: 40, autoexpand: true },
    paper_bgcolor: "#fff",
    plot_bgcolor: "#fff",
    font: { family: SERIF, size: 12, color: "#000" },
    xaxis: axis(x),
    yaxis: axis(y),
    // Legend above the plot: below it, many entries collide with the x-axis title on a phone
    legend: {
      orientation: "h", x: 0, y: 1.01, xanchor: "left", yanchor: "bottom", groupclick: "togglegroup",
      font: { size: 11 },
    },
    hovermode: "closest",
    ...extras,
  };
  window.Plotly.react(host, traces, layout, {
    responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d"],
  }).catch(error => reportError(host, error));
}

function detection(section) {
  sectionContent("detection", section);
  const picker = document.querySelector("#pd-attack");
  const target = document.querySelector("#pd-table");
  const levels = [...new Set(section.rows.map(row => row.attack))];
  const names = [...new Set(section.rows.map(row => row.detector))];
  const ordered = [...priority.filter(name => names.includes(name)), ...names.filter(name => !priority.includes(name))];
  options(picker, levels.map(key => [key, attacks[key] ?? key]));

  function show() {
    const key = picker.value;
    const rows = section.rows.filter(r => r.attack === key);
    const traces = [];
    PLOTTED.filter(name => names.includes(name)).forEach(name => {
      const series = rows.filter(r => r.detector === name).sort((a, b) => a.jsr_db - b.jsr_db);
      if (!series.length) return;
      const [color, symbol, dash] = STYLE[name] ?? OTHER;
      traces.push({
        x: series.map(r => r.jsr_db), y: series.map(r => r.pd_mean),
        customdata: series.map(r => [r.pd_std, r.num_cells, r.n_seeds]),
        type: "scatter", mode: "lines+markers", legendgroup: name, name: detectorName(name),
        line: { color, dash, width: 1.2 }, marker: { symbol, size: 7, color, line: { width: 1.1, color } },
        visible: SHOWN.has(name) ? true : "legendonly",
        hovertemplate: "%{x} dB JSR<br>Pd %{y:.3f} ± %{customdata[0]:.3f} (SD over seeds)<br>%{customdata[1]} jammed cells · %{customdata[2]} seeds<extra>%{fullData.name}</extra>",
      });
    });
    renderPlot("pd-chart", traces, "JSR (dB)", "Pd", {
      yaxis: { ...axis("Detection probability"), range: [-0.02, 1.02] },
    });
    target.replaceChildren(table(`${attacks[key] ?? key} jamming on the scheduled cells`,
      ["Detector", "JSR", "Mean Pd ± SD", "Jammed cells", "Seeds"],
      ordered.flatMap(name => rows.filter(r => r.detector === name)
        .sort((a, b) => a.jsr_db - b.jsr_db)
        .map(r => [detectorName(name), signedDb(r.jsr_db), `${decimal(r.pd_mean)} ± ${decimal(r.pd_std)}`, integer(r.num_cells), r.n_seeds])),
    ));
  }
  picker.addEventListener("change", show);
  show();
}

function marginEstimate(row) {
  return `${row.point_any_attack_censored ? "≤ " : ""}${decimal(row.margin_db, 2)}`;
}

function marginInterval(row) {
  const lower = row.margin_lo_db == null ? "−∞" : decimal(row.margin_lo_db, 2);
  const upper = row.margin_hi_db == null ? "−∞" : decimal(row.margin_hi_db, 2);
  return `[${lower}, ${upper}]`;
}

function margins(section) {
  sectionContent("margin", section);
  const traces = [];
  const annotations = [];
  const ordered = [...section.rows].sort((a, b) =>
    a.channel.localeCompare(b.channel) || b.slot_fa_target - a.slot_fa_target || a.snr_db - b.snr_db,
  );
  for (const channel of new Set(ordered.map(r => r.channel))) {
    const data = ordered.filter(r => r.channel === channel);
    const color = channel === "C" ? "#cc0000" : "#0033cc";
    for (const target of new Set(data.map(r => r.slot_fa_target))) {
      const rows = data.filter(r => r.slot_fa_target === target);
      const key = `${channel}-${target}`;
      const marker = target >= 0.01 ? "circle" : "diamond";
      const line = { color, dash: target >= 0.01 ? "solid" : "dash" };
      const xOf = row => row.snr_db;
      const finite = rows.filter(r => r.margin_lo_db != null && r.margin_hi_db != null);
      traces.push({
        type: "scatter", mode: "lines+markers", x: rows.map(xOf), y: rows.map(r => r.margin_db),
        name: `TDL-${channel} · ${percent(target)} slot FA`, legendgroup: key, line,
        marker: {
          size: 7, color, symbol: rows.map(r => r.point_any_attack_censored ? `${marker}-open` : marker),
          line: { width: 1.1, color },
        },
        text: rows.map(r => `${r.snr_db} dB SNR<br>Margin ${marginEstimate(r)} dB<br>95% interval ${marginInterval(r)} dB`),
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      }, {
        type: "scatter", mode: "markers", x: finite.map(xOf),
        y: finite.map(r => (r.margin_lo_db + r.margin_hi_db) / 2),
        legendgroup: key, showlegend: false, hoverinfo: "skip",
        marker: { size: 0.1, color },
        error_y: {
          type: "data", array: finite.map(r => (r.margin_hi_db - r.margin_lo_db) / 2),
          color, thickness: 1.25, width: 3,
        },
      });
      for (const r of rows.filter(row => row.margin_lo_db == null && row.margin_hi_db != null)) {
        // The cap is the finite endpoint. A downward triangle marks the unbounded side.
        traces.push({
          type: "scatter", mode: "markers", x: [xOf(r), xOf(r)], y: [r.margin_hi_db, r.margin_hi_db - 0.65],
          marker: { symbol: ["line-ew", "triangle-down"], color, size: [7, 7], line: { width: 1.3, color } },
          legendgroup: key, showlegend: false, hoverinfo: "skip",
        }, {
          type: "scatter", mode: "lines", x: [xOf(r), xOf(r)], y: [r.margin_hi_db, r.margin_hi_db - 0.65],
          line: { color, width: 1.25 }, legendgroup: key, showlegend: false, hoverinfo: "skip",
        });
      }
    }
  }
  if (ordered.some(r => r.margin_lo_db == null)) {
    annotations.push({
      xref: "paper", yref: "paper", x: 0.01, y: 0.99, xanchor: "left", yanchor: "top", showarrow: false,
      text: "▼ Lower interval bound unbounded",
      font: { size: 10, color: "#555" },
    });
  }
  renderPlot("margin-chart", traces, "SNR (dB)", "Smallest margin (dB)", {
    xaxis: { ...axis("SNR (dB)"), tickvals: [...new Set(ordered.map(r => r.snr_db))].sort((a, b) => a - b) },
    yaxis: { ...axis("Smallest margin (dB)"), zeroline: true, zerolinecolor: "#000", zerolinewidth: 0.6 },
    annotations,
  });
  document.querySelector("#margin-table").replaceChildren(table("Fused U-Net margin in dB",
    ["Channel", "Slot FA target", "SNR", "Estimate", "95% interval", "Draws"],
    ordered.map(r => [`TDL-${r.channel}`, percent(r.slot_fa_target), `${r.snr_db} dB`,
      marginEstimate(r), marginInterval(r), integer(r.boot_draws)]),
  ));
}

function recovery(section) {
  sectionContent("recovery", section);
  const picker = document.querySelector("#recovery-jammer");
  options(picker, [...new Set(section.rows.map(r => r.jammer))].map(key => [key, jammers[key] ?? key]));

  function show() {
    const key = picker.value;
    const rows = section.rows.filter(r => r.jammer === key);
    const groups = [...new Set(rows.map(r => r.attack))];
    const receivers = [...new Set(rows.map(r => r.receiver))];
    const cells = (names, counts) => names.map(name => [receiverName(name), ...groups.map(group => {
      const row = rows.find(r => r.receiver === name && r.attack === group);
      if (!row) return "—";
      return counts ? `${percent(row.bler, 2)} (${integer(row.block_errors)}/${integer(row.blocks)})` : percent(row.bler, 2);
    })]);
    const labels = ["Receiver", ...groups.map(group => attacks[group] ?? group)];
    document.querySelector("#recovery-table").replaceChildren(table(`BLER, ${jammers[key] ?? key} jammer`,
      labels, cells(KEY_RECEIVERS.filter(name => receivers.includes(name)), false)));
    document.querySelector("#recovery-all").replaceChildren(table("All receivers: BLER (block errors/blocks)",
      labels, cells(receivers, true)));
    const pairs = section.paired.filter(r => r.jammer === key);
    document.querySelector("#recovery-paired").replaceChildren(table(
      "First receiver minus second in percentage points",
      ["First receiver", "Second receiver", "Attack", "Common slots", "Difference", "95% interval"],
      pairs.map(r => [receiverName(r.first), receiverName(r.second), attacks[r.attack] ?? r.attack, integer(r.slots),
        percentagePoints(r.delta), interval(r.delta_lo, r.delta_hi, percentagePoints)]),
    ));
  }
  picker.addEventListener("change", show);
  show();
}

function matched(section) {
  sectionContent("matched", section);
  const picker = document.querySelector("#matched-attack");
  const groups = [...new Set(section.rows.map(r => r.attack))].sort();
  options(picker, groups.map(key => [key, attacks[key] ?? key]));
  const target = document.querySelector("#matched-table");

  function show() {
    const key = picker.value;
    const rows = section.rows.filter(r => r.attack === key).sort((a, b) => b.f1 - a.f1);
    const numberAndCI = (row, field, format = decimal) => row[field] == null ? "—" :
      `${format(row[field])} ${interval(row[field + "_lo"], row[field + "_hi"], format)}`;
    const jsr = rows[0]?.jsr_db;
    const faintest = jsr == null ? "at the weakest JSR of each attack" : `at ${signedDb(jsr)} JSR`;
    target.replaceChildren(table(`${attacks[key] ?? key} · mean and 95% interval`,
      ["Detector", "Cell F1", "Scored cell FA", `Scheduled cell Pd ${faintest}`],
      rows.map(r => [detectorName(r.detector), numberAndCI(r, "f1"),
        numberAndCI(r, "fa_scored", v => percent(v, 2)), numberAndCI(r, "pd")]),
    ));
    document.querySelector("#matched-paired").replaceChildren(table("First detector minus second",
      ["First detector", "Second detector", "Metric", "Difference", "95% interval"],
      section.paired.filter(r => r.attack === key).map(r => [detectorName(r.first), detectorName(r.second),
        r.metric === "pd" ? `Pd ${r.jsr_db == null ? faintest : `at ${signedDb(r.jsr_db)} JSR`}` : "F1",
        decimal(r.delta), interval(r.delta_lo, r.delta_hi)]),
    ));
  }
  picker.addEventListener("change", show);
  show();
}

function finding(title, value, note, link) {
  const article = element("article");
  const heading = element("h3");
  if (link) {
    const anchor = element("a", title);
    anchor.href = link;
    heading.append(anchor);
  } else {
    heading.textContent = title;
  }
  article.append(heading);
  if (value) article.append(element("strong", value));
  article.append(element("p", note));
  return article;
}

function dmrsRow(section, count, detector) {
  return section.rows.find(r => r.dmrs_count === count && r.detector === detector);
}

function home(host, data) {
  const pilot = data.pd.rows.filter(r => r.detector === "fused" && r.attack === "pilot")
    .sort((a, b) => a.jsr_db - b.jsr_db)[0];
  const margin = data.margin.rows.find(r => r.channel === "C" && r.snr_db === 15 && r.slot_fa_target === 0.01);
  const rows = data.recovery.rows.filter(r => r.jammer === "nominal" && r.attack === "partial" && r.snr_db === 15);
  const standard = rows.find(r => r.receiver === "standard");
  const protectedRow = rows.find(r => r.receiver === "p_r_e_soft");
  const [one, two] = [1, 2].map(count => dmrsRow(data.dmrs, count, "fused"));
  const cards = [];
  if (pilot) cards.push(finding("Weak pilot jamming", percent(pilot.pd_mean),
    `of the jammed cells detected at ${signedDb(pilot.jsr_db)} JSR, with 1% false alarms per cell.`, "results.html#detection"));
  if (margin) cards.push(finding("Detection margin", `${decimal(margin.margin_db, 1)} dB`,
    `between 90% slot detection and 10% block errors, worst attack, TDL-C at ${margin.snr_db} dB SNR.`, "results.html#margin"));
  if (standard && protectedRow) cards.push(finding("Partial-band BLER", `${percent(standard.bler)} → ${percent(protectedRow.bler, 2)}`,
    `standard receiver → decoding with the U-Net map, at ${signedDb(standard.jsr_db)} JSR.`, "results.html#recovery"));
  if (one && two) cards.push(finding("Second DMRS symbol", `${percent(one.pd_pilot_m15)} → ${percent(two.pd_pilot_m15)}`,
    "weak pilot jamming detected with one → two DMRS symbols.", "results.html#dmrs"));
  host.replaceChildren(...cards);
}

function dmrs(section) {
  sectionContent("dmrs", section);
  const bler = (count, jammer, receiver) => {
    const row = section.recovery.find(r => r.dmrs_count === count && r.jammer_dmrs === jammer &&
      r.attack === "pilot" && r.receiver === receiver);
    return row ? `${percent(row.bler, 2)}` : "—";
  };
  const value = (count, detector, field) => {
    const row = dmrsRow(section, count, detector);
    return row ? percent(row[field]) : "—";
  };
  const rows = [[1, "all", "1"], [2, "all", "2"], [2, "first", "2, jammer on the first only"]].map(
    ([count, jammer, label]) => [label,
      jammer === "all" ? value(count, "fused", "f1_pilot") : "—",
      jammer === "all" ? value(count, "fused", "pd_pilot_m15") : "—",
      jammer === "all" ? value(count, "block_test", "f1_pilot") : "—",
      bler(count, jammer, "standard"), bler(count, jammer, "lmmse"), bler(count, jammer, "p_r_e_soft")],
  );
  document.querySelector("#dmrs-table").replaceChildren(table("Pilot jamming",
    ["DMRS symbols", "U-Net F1", "U-Net Pd, −15 dB", "Block test F1", "BLER standard, +9 dB",
      "BLER LMMSE", "BLER with U-Net map"], rows));
}

function hitCount(slot, detector) {
  return slot.grids.truth.flatMap((row, prb) => row.map((hit, symbol) =>
    hit && slot.detectors[detector].decisions[prb][symbol],
  )).filter(Boolean).length;
}

async function failures(host, data) {
  const [spoof, whole] = await Promise.all(["spoof", "pilot_whole_comb"].map(loadCase));
  const jammed = spoof.grids.truth.flat().filter(Boolean).length;
  const crc = whole.metadata.batch_crc.true_soft;
  const hoppingRows = data.recovery.rows.filter(r => r.jammer === "hopping" && r.attack === "data");
  const block = hoppingRows.find(r => r.receiver === "p_r_e_soft");
  const mixed = hoppingRows.find(r => r.receiver === "p_r_e_atk_soft");
  host.replaceChildren(
    finding("Pilot spoofing", `${hitCount(spoof, "fused")} of ${jammed}`,
      `spoofed pilot cells flagged by the nominal U-Net in this slot; ${hitCount(spoof, "fused_atk2")} of ${jammed} after training with spoofing.`,
      "explore.html?case=spoof"),
    finding("Whole-allocation pilot jamming", `${crc.failures} of ${crc.slots}`,
      "blocks fail even with the true map: the jammer inflates the noise estimate taken from the empty pilot comb.",
      "explore.html?case=pilot_whole_comb"),
    ...(block && mixed ? [finding("Hopping jammer", `${percent(block.bler)} → ${percent(mixed.bler)}`,
      `BLER on data jamming at ${signedDb(block.jsr_db)} JSR with the block-trained map → the map trained on mixed attacks.`,
      "explore.html?case=hopping")] : []),
  );
}

const homeHost = document.querySelector("#home-results");
if (homeHost) loadResults().then(data => home(homeHost, data)).catch(error => reportError(homeHost, error));

const resultsHost = document.querySelector("#paper-results");
if (resultsHost) {
  loadResults().then(data => {
    detection(data.pd);
    margins(data.margin);
    recovery(data.recovery);
    dmrs(data.dmrs);
    matched(data.matched);
    resultsHost.setAttribute("aria-busy", "false");
    const target = document.querySelector("#failure-results");
    failures(target, data).catch(error => reportError(target, error));
  }).catch(error => reportError(resultsHost, error));
}
