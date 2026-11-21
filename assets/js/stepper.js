import { batchText, conditions, loadCase, selectionText, slotDetails } from "./data.js";
import { affectedPRBs, constellations, crcList } from "./constellation.js";
import { drawGrid } from "./grid.js";
import { element, reportError } from "./nav.js";

const receivers = ["standard", "fused_soft"];

const steps = [
  {
    title: "Received slot",
    text: "Each cell is one PRB on one OFDM symbol. Grey is the user's allocation; red marks the cells the jammer transmits on, here the pilot (DMRS) symbol of four PRBs.",
    view: slot => drawGrid(slot, { injected: true, label: "Resource grid with the jammed pilot cells" }),
  },
  {
    title: "Power P",
    text: "Received energy per cell over the noise power. Signal and jammer both add to it, so it says little on its own.",
    view: slot => drawGrid(slot, { feature: "power_db", label: "Received power by resource cell" }),
  },
  {
    title: "Pilot residual R",
    text: "What is left of the pilot channel estimate after fitting a smooth channel. Noise leaves about 0 dB; a jammer leaves much more.",
    view: slot => drawGrid(slot, { feature: "residual_db", label: "Pilot residual by resource cell" }),
  },
  {
    title: "EVM E",
    text: "How far the equalized data points lie from QPSK, over the noise. A spoiled pilot estimate raises it on every data symbol of the PRB.",
    view: slot => drawGrid(slot, { feature: "evm_db", label: "Error vector magnitude by resource cell" }),
  },
  {
    title: "U-Net map",
    text: "The U-Net reads P, R, and E of the whole slot and flags cells (blue). Red outlines are the jammed cells.",
    view: slot => drawGrid(slot, { detector: "fused", label: "U-Net flags and jammed cells" }),
  },
  {
    title: "Decoding with the map",
    text: "The receiver down-weights the flagged pilots and lowers the reliability of flagged data bits. Data points of the jammed PRBs:",
    view: slot => constellations(slot, receivers, affectedPRBs(slot)),
  },
  {
    title: "CRC",
    text: "Did the decoded block pass its CRC?",
    view: slot => {
      const group = element("div");
      group.append(crcList(slot, receivers), element("p", batchText(slot, receivers), "context"));
      return group;
    },
  },
];

async function walkthrough(host) {
  const slot = await loadCase("pilot");
  const back = document.querySelector("#previous");
  const next = document.querySelector("#next");
  const buttons = steps.map((step, i) => {
    const button = element("button", String(i + 1));
    button.type = "button";
    button.title = step.title;
    button.setAttribute("aria-label", `Step ${i + 1}: ${step.title}`);
    button.addEventListener("click", () => show(i));
    return button;
  });
  document.querySelector("#step-buttons").replaceChildren(...buttons);
  let index = 0;

  function show(i) {
    index = i;
    const step = steps[index];
    back.disabled = index === 0;
    next.disabled = index === steps.length - 1;
    buttons.forEach((button, k) => button.setAttribute("aria-pressed", String(k === index)));
    const text = element("div");
    text.append(element("h2", `${index + 1}. ${step.title}`, "step-title"), element("p", step.text, "step-text"));
    const row = element("div", null, "slot-row");
    row.append(step.view(slot), text);
    host.replaceChildren(row);
  }

  back.addEventListener("click", () => show(index - 1));
  next.addEventListener("click", () => show(index + 1));
  document.querySelector("#walkthrough-selection").replaceChildren(conditions(slot), slotDetails(slot));
  show(0);
  host.setAttribute("aria-busy", "false");
}

async function recoveredSlot(host) {
  const slot = await loadCase("walkthrough_flip");
  host.replaceChildren(
    element("p", `${conditions(slot)}. ${selectionText(slot)}`, "context"),
    constellations(slot, receivers, affectedPRBs(slot)),
    element("p", batchText(slot, receivers), "context"),
  );
  host.setAttribute("aria-busy", "false");
}

const host = document.querySelector("#walkthrough");
walkthrough(host).catch(error => reportError(host, error));
const flip = document.querySelector("#flip");
recoveredSlot(flip).catch(error => reportError(flip, error));
