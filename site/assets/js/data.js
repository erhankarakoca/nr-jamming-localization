const dataRoot = new URL("../../data/", import.meta.url);
const pending = new Map();

function readJSON(url) {
  if (!pending.has(url.href)) {
    const request = fetch(url).then(response => {
      if (!response.ok) throw new Error(`Could not load ${url.pathname} (HTTP ${response.status}).`);
      return response.json();
    }).catch(error => {
      pending.delete(url.href);
      throw error;
    });
    pending.set(url.href, request);
  }
  return pending.get(url.href);
}

function checkVersion(document) {
  if (document.schema_version !== 2) throw new Error("Expected showcase data schema version 2.");
  return document;
}

export function loadManifest() {
  return readJSON(new URL("manifest.json", dataRoot)).then(checkVersion);
}

export async function loadCase(name) {
  const manifest = await loadManifest();
  const entry = manifest.cases.find(item => item.name === name);
  if (!entry) throw new Error(`The manifest does not contain the ${name} case.`);
  return readJSON(new URL(entry.file, dataRoot)).then(checkVersion);
}

export function loadResults() {
  return readJSON(new URL("results.json", dataRoot));
}

export const detectors = {
  fused: "U-Net P+R+E",
  no_residual: "U-Net P+E",
  power: "U-Net P",
  evm: "U-Net E",
  fused_comb: "U-Net P+R+E+C",
  fused_nj1: "U-Net P+R+E, noise jitter",
  fused_pixel: "Per-cell MLP",
  fused_pixel_wide: "Wide per-cell MLP",
  fused_atk: "U-Net, mixed attacks",
  fused_atk2: "U-Net, trained with spoofing",
  block_test: "Block test",
  block_test_comb: "Block test with empty comb",
  evm_threshold: "EVM threshold",
  residual_threshold: "Residual threshold",
  residual_cfar: "Analytic residual test",
  power_threshold: "Power threshold",
  comb_threshold: "Empty-comb threshold",
};

const receiverNames = {
  standard: "Standard",
  fused_soft: "U-Net, soft",
  fused_hard: "U-Net, hard",
  fused_atk_soft: "Mixed-attack U-Net, soft",
  fused_atk2_soft: "Spoof-trained U-Net, soft",
  true_soft: "True map, soft",
  true_prb_soft: "True map per PRB, soft",
  p_r_e_soft: "U-Net, soft",
  p_r_e_hard: "U-Net, hard",
  p_r_e_atk_soft: "Mixed-attack U-Net, soft",
  lmmse: "LMMSE interpolation",
  slot_ie: "Slot-wide estimate",
  dmrs_ie: "DMRS only",
  block_soft: "Block test, soft",
  power_hard: "Power test, hard",
};

export function receiverName(name) {
  return receiverNames[name] ?? name.replaceAll("_", " ");
}

export function percent(value, digits = 1) {
  return value == null ? "—" : (100 * value).toFixed(digits).replace(/\.0+$/, "") + "%";
}

export function signedDb(value) {
  return value == null ? "No jamming" : `${value > 0 ? "+" : ""}${String(value).replace("-", "−")} dB`;
}

export function calibrationText(slot) {
  const m = slot.metadata;
  const calibration = m.thresholds.fused;
  return `Simulation seed ${m.seed}, detector seed ${m.detector_seed} (${calibration.training_data}). The nominal U-Net cutoff is ${calibration.value}, calibrated on ${calibration.split} at a ${percent(calibration.false_alarm_rate)} per-cell false-alarm target`;
}

export function conditions(slot) {
  const m = slot.metadata;
  const jsr = m.jsr_db == null ? "no jamming" : `${signedDb(m.jsr_db)} JSR`;
  const noise = m.noise_estimate === "comb" ? "estimated noise power" : "known noise power";
  return `TDL-${m.channel} · ${m.snr_db} dB SNR · ${jsr} · ${noise}. ${calibrationText(slot)}`;
}

export function selectionText(slot) {
  const { selection: s, batch_size: size } = slot.metadata;
  if (s.rule === "first_crc_flip") {
    return `First slot where standard decoding failed and U-Net soft decoding passed. This happens in ${s.flip_count} of ${size} slots in its batch.`;
  }
  if (s.rule === "first_true_soft_failure") {
    return `First slot where the soft receiver failed with the true map. This happens in ${s.failure_count} of ${size} slots in its batch.`;
  }
  if (s.hit_rate == null) return `The first of ${size} benign slots is shown because there are no jammed cells.`;
  return `This slot has the U-Net hit rate closest to the median over ${size} batch slots, with the earliest slot used on a tie. The slot hit rate is ${percent(s.hit_rate)} and the batch median is ${percent(s.batch_median)} over jammed carrier cells.`;
}

export function batchText(slot, names = Object.keys(slot.receivers)) {
  const counts = slot.metadata.batch_crc;
  const size = counts[names[0]].slots;
  const summary = names.map(name => `${receiverName(name)} failed ${counts[name].failures}`);
  return `In this batch of ${size} slots, ${summary.join(", ")}.`;
}
