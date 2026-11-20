import { conditions, detectors, loadCase, loadResults, percent, receiverName, signedDb } from "./data.js";
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

const palette = [
  ["#0072b2", "0,114,178"], ["#4e5f69", "78,95,105"], ["#d55e00", "213,94,0"],
  ["#4654a5", "70,84,165"], ["#8b5b2e", "139,91,46"], ["#397166", "57,113,102"],
];
const priority = ["fused", "no_residual", "block_test", "residual_threshold", "evm_threshold"];

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
    height: 410,
    margin: { l: 56, r: 20, t: 12, b: 47, autoexpand: true },
    paper_bgcolor: "#fff",
    plot_bgcolor: "#fff",
    font: { family: "system-ui, sans-serif", size: 12, color: "#142632" },
    xaxis: { title: x, gridcolor: "#e2eaee", automargin: true },
    yaxis: { title: y, gridcolor: "#e2eaee", automargin: true },
    // Legend above the plot: below it, many entries collide with the x-axis title on a phone
    legend: { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom", groupclick: "togglegroup" },
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
    ordered.forEach((name, index) => {
      const series = rows.filter(r => r.detector === name).sort((a, b) => a.jsr_db - b.jsr_db);
      if (!series.length) return;
      const [color, rgb] = palette[index % palette.length];
      const visible = index < 4 ? true : "legendonly";
      const x = series.map(r => r.jsr_db);
      const band = {
        x, mode: "lines", legendgroup: name, showlegend: false, visible,
        hoverinfo: "skip", line: { width: 0 }, type: "scatter",
      };
      traces.push(
        { ...band, y: series.map(r => Math.max(0, r.pd_mean - r.pd_std)) },
        { ...band, y: series.map(r => Math.min(1, r.pd_mean + r.pd_std)), fill: "tonexty", fillcolor: `rgba(${rgb},0.12)` },
        {
          x, y: series.map(r => r.pd_mean), customdata: series.map(r => [r.pd_std, r.num_cells, r.n_seeds]),
          type: "scatter", mode: "lines+markers", legendgroup: name, name: detectorName(name),
          line: { color, dash: index < palette.length ? "solid" : "dash" }, visible,
          hovertemplate: "%{x} dB JSR<br>%{y:.3f} mean Pd ± %{customdata[0]:.3f} SD<br>%{customdata[1]} jammed scheduled cells · %{customdata[2]} seeds<extra>%{fullData.name}</extra>",
        },
      );
    });
    renderPlot("pd-chart", traces, "JSR (dB)", "Detection probability", {
      yaxis: { title: "Detection probability", range: [-0.025, 1.025], tickformat: ".0%", automargin: true },
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
    const color = channel === "C" ? "#0072b2" : "#d55e00";
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
          size: 8, color, symbol: rows.map(r => r.point_any_attack_censored ? `${marker}-open` : marker),
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
      xref: "paper", yref: "paper", x: 0.02, y: 0.03, xanchor: "left", showarrow: false,
      text: "▼ Lower interval bound unbounded",
      font: { size: 10, color: "#4b616d" },
    });
  }
  renderPlot("margin-chart", traces, "SNR (dB)", "Smallest margin (dB)", {
    xaxis: { title: "SNR (dB)", tickvals: [...new Set(ordered.map(r => r.snr_db))].sort((a, b) => a - b), automargin: true },
    yaxis: { title: "Smallest margin (dB)", zeroline: true, zerolinecolor: "#637682", automargin: true },
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
    const cells = receivers.map(name => [receiverName(name), ...groups.map(group => {
      const row = rows.find(r => r.receiver === name && r.attack === group);
      return row ? `${percent(row.bler, 2)} · ${integer(row.block_errors)}/${integer(row.blocks)}` : "—";
    })]);
    document.querySelector("#recovery-table").replaceChildren(table(
      `${jammers[key] ?? key} jammer · BLER and errors / blocks`,
      ["Receiver", ...groups.map(group => attacks[group] ?? group)], cells,
    ));
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
  article.append(element("h3", title));
  if (value) article.append(element("strong", value));
  article.append(element("p", note));
  if (link) {
    const anchor = element("a", "Inspect the slot");
    anchor.href = link;
    article.append(anchor);
  }
  return article;
}

function home(host, data) {
  const pilot = data.pd.rows.filter(r => r.detector === "fused" && r.attack === "pilot")
    .sort((a, b) => a.jsr_db - b.jsr_db)[0];
  const margin = data.margin.rows.find(r => r.channel === "C" && r.snr_db === 15 && r.slot_fa_target === 0.01);
  const rows = data.recovery.rows.filter(r => r.jammer === "nominal" && r.attack === "partial" && r.snr_db === 15);
  const standard = rows.find(r => r.receiver === "standard");
  const protectedRow = rows.find(r => r.receiver === "p_r_e_soft");
  const cards = [];
  if (pilot) cards.push(finding("Weak pilot jamming", percent(pilot.pd_mean),
    `U-Net mean detection at ${signedDb(pilot.jsr_db)} JSR on ${integer(pilot.num_cells)} jammed scheduled cells in the TDL-A and TDL-C test split, over ${pilot.n_seeds} training seeds. Validation targets 1% per-cell false alarms.`));
  if (margin) cards.push(finding("Smallest detection margin", `${decimal(margin.margin_db, 1)} dB`,
    `Over the pilot, data, and partial-band attacks on scheduled PRBs at ${margin.snr_db} dB SNR on TDL-C. The U-Net targets ${percent(margin.slot_fa_target)} false alarms per benign slot. The 95% interval is ${marginInterval(margin)} dB.`));
  if (standard && protectedRow) cards.push(finding("Partial-band recovery", `${percent(standard.bler)} → ${percent(protectedRow.bler, 2)}`,
    `Pooled BLER from standard to U-Net soft at ${signedDb(standard.jsr_db)} JSR on TDL-C at ${standard.snr_db} dB SNR with known noise power. The counts are ${integer(standard.block_errors)}/${integer(standard.blocks)} and ${integer(protectedRow.block_errors)}/${integer(protectedRow.blocks)}.`));
  host.replaceChildren(...cards);
}

function hitCount(slot, detector) {
  return slot.grids.truth.flatMap((row, prb) => row.map((hit, symbol) =>
    hit && slot.detectors[detector].decisions[prb][symbol],
  )).filter(Boolean).length;
}

async function failures(host, data) {
  const [spoof, whole, hopping] = await Promise.all(["spoof", "pilot_whole_comb", "hopping"].map(loadCase));
  const jammed = spoof.grids.truth.flat().filter(Boolean).length;
  const noise = whole.metadata;
  const crc = noise.batch_crc.true_soft;
  const hoppingRows = data.recovery.rows.filter(r => r.jammer === "hopping" && r.attack === "data");
  const block = hoppingRows.find(r => r.receiver === "p_r_e_soft");
  const mixed = hoppingRows.find(r => r.receiver === "p_r_e_atk_soft");
  const hoppingNote = block && mixed
    ? `At ${signedDb(block.jsr_db)} JSR on TDL-C at ${block.snr_db} dB SNR, the soft receiver using the block-trained map had ${percent(block.bler)} BLER (${integer(block.block_errors)}/${integer(block.blocks)}) on hopping data jamming. The mixed-attack map had ${percent(mixed.bler)} (${integer(mixed.block_errors)}/${integer(mixed.blocks)}). Both rates pool the recovery runs over training seeds with known noise power.`
    : `${conditions(hopping)}. The block-trained network flagged ${hitCount(hopping, "fused")} of the injected cells in the selected slot. The mixed-attack network flagged ${hitCount(hopping, "fused_atk")}.`;
  host.replaceChildren(
    finding("Pilot spoofing", null,
      `${conditions(spoof)}. In the slot closest to the nominal U-Net’s median hit rate, its map flagged ${hitCount(spoof, "fused")} of ${jammed} injected pilot cells. The spoof-trained map flagged ${hitCount(spoof, "fused_atk2")} of ${jammed} on that slot.`,
      "explore.html?case=spoof"),
    finding("Whole-allocation pilot jamming", null,
      `${conditions(whole)}. The receiver decoded with the true map and failed in ${crc.failures} of ${crc.slots} slots in this batch. In the first such slot, the comb gave a noise variance of ${noise.noise_variance} against a simulated variance of ${noise.true_noise_variance}.`,
      "explore.html?case=pilot_whole_comb"),
    finding("Hopping with block-trained detection", null, hoppingNote, "explore.html?case=hopping"),
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
    matched(data.matched);
    resultsHost.setAttribute("aria-busy", "false");
    const target = document.querySelector("#failure-results");
    failures(target, data).catch(error => reportError(target, error));
  }).catch(error => reportError(resultsHost, error));
}
