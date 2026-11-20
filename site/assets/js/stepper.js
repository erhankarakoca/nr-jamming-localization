import { batchText, conditions, loadCase, selectionText } from "./data.js";
import { affectedPRBs, constellations, crcList } from "./constellation.js";
import { drawGrid } from "./grid.js";
import { element, reportError } from "./nav.js";

const receivers = ["standard", "fused_soft"];

const steps = [
  {
    title: "Receive the slot",
    text: "Each cell is a physical resource block (PRB) on one OFDM symbol. The orange cells show injected interference on the demodulation reference signal (DMRS), which the receiver uses as a pilot for channel estimation.",
    view: slot => drawGrid(slot, { injected: true, label: "Resource grid with injected pilot interference" }),
  },
  {
    title: "Measure power P",
    text: "The receiver measures the average received energy in each cell relative to the noise power. Both the signal and the interference contribute to this measurement.",
    view: slot => drawGrid(slot, { feature: "power_db", label: "Received power by resource cell" }),
  },
  {
    title: "Measure the pilot residual R",
    text: "The receiver projects the pilot channel estimates onto the subspace of expected channel responses. The energy left over, normalized by its expected value in noise, gives the residual on the pilot symbol.",
    view: slot => drawGrid(slot, { feature: "residual_db", label: "Pilot residual by resource cell" }),
  },
  {
    title: "Measure error vector magnitude E",
    text: "On data symbols, the receiver measures how far equalized points lie from the nearest ideal constellation points, relative to the expected noise. A corrupted pilot estimate can increase this measure on other symbols.",
    view: slot => drawGrid(slot, { feature: "evm_db", label: "Error vector magnitude by resource cell" }),
  },
  {
    title: "Flag cells with the U-Net",
    text: "The U-Net combines power, residual, and error vector magnitude across the buffered slot. Blue shows its flags and orange outlines the injected cells.",
    view: slot => drawGrid(slot, { detector: "fused", label: "U-Net flags and injected interference" }),
  },
  {
    title: "Reprocess the affected PRBs",
    text: "The protected receiver uses the map to reweight pilots and adjust the reliability of the data bits. These fixed-axis plots show the equalized data symbols from the same affected PRBs, with stronger bit reliabilities drawn more opaquely.",
    view: slot => constellations(slot, receivers, affectedPRBs(slot)),
  },
  {
    title: "Check the decoded block",
    text: "The CRC checks the transport block after decoding. The badges give the outcomes for this slot and the failure counts cover the full case batch.",
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
  const play = document.querySelector("#walk-play");
  const count = document.querySelector("#step-count");
  let index = 0;
  let timer = null;

  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
  }

  function show() {
    const step = steps[index];
    count.textContent = `Step ${index + 1} of ${steps.length} · ${step.title}`;
    back.disabled = index === 0;
    next.disabled = index === steps.length - 1;
    if (next.disabled) stop();
    host.replaceChildren(element("h2", step.title), element("p", step.text), step.view(slot));
  }

  function move(delta) {
    index += delta;
    show();
  }

  back.addEventListener("click", () => move(-1));
  next.addEventListener("click", () => move(1));
  play.addEventListener("click", () => {
    if (timer !== null) {
      stop();
      return;
    }
    if (index === steps.length - 1) {
      index = 0;
      show();
    }
    play.textContent = "Pause";
    timer = setInterval(() => move(1), 2600);
  });
  document.querySelector("#walkthrough-selection").textContent =
    `${conditions(slot)}. ${selectionText(slot)}`;
  play.disabled = false;
  show();
  host.setAttribute("aria-busy", "false");
}

async function recoveredSlot(host) {
  const slot = await loadCase("walkthrough_flip");
  host.replaceChildren(
    element("p", conditions(slot), "context"),
    element("p", selectionText(slot)),
    constellations(slot, receivers, affectedPRBs(slot)),
    element("p", batchText(slot, receivers), "context"),
  );
  host.setAttribute("aria-busy", "false");
}

const host = document.querySelector("#walkthrough");
walkthrough(host).catch(error => reportError(host, error));
const flip = document.querySelector("#flip");
recoveredSlot(flip).catch(error => reportError(flip, error));
