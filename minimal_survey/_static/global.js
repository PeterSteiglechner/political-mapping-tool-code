// Shared slider logic for all belief-rating pages.
// Every <input class="likert-slider"> must be moved before the page can be submitted.

const LIKERT_LABELS = [
  "Strongly disagree",
  "Disagree",
  "Somewhat disagree",
  "Neutral",
  "Somewhat agree",
  "Agree",
  "Strongly agree",
];

function getLikertLabel(value) {
  const idx = Math.min(6, Math.floor(((parseFloat(value) + 100) / 200) * 7));
  return LIKERT_LABELS[idx];
}

document.addEventListener("DOMContentLoaded", function () {
  const sliders = document.querySelectorAll(".likert-slider");
  if (sliders.length === 0) return;

  const nextButton = document.querySelector(".otree-btn-next");
  const touched = new Set();

  function refreshButton() {
    if (!nextButton) return;
    const ok = touched.size === sliders.length;
    nextButton.disabled = !ok;
    nextButton.style.opacity = ok ? "1" : "0.5";
  }

  sliders.forEach(function (slider) {
    slider.addEventListener("input", function () {
      touched.add(slider.id);
      const color = slider.dataset.color || "#555";
      const textcolor = slider.dataset.textcolor || "white";

      const label = document.getElementById("label_" + slider.id);
      if (label) {
        label.innerHTML =
          "<span class='pill' style='background-color:" + color +
          ";color:" + textcolor + ";'>" + getLikertLabel(slider.value) + "</span>";
      }

      // coloured track from the midpoint to the current value
      const val = parseFloat(slider.value);
      const min = parseFloat(slider.min);
      const max = parseFloat(slider.max);
      const mid = (min + max) / 2;
      const l = val < mid ? 50 + ((val - mid) / (mid - min)) * 50 : 50;
      const r = val < mid ? 50 : 50 + ((val - mid) / (max - mid)) * 50;
      slider.style.background =
        "linear-gradient(to right, #e0e0e0 0%, #e0e0e0 " + l + "%, " +
        color + " " + l + "%, " + color + " " + r + "%, " +
        "#e0e0e0 " + r + "%, #e0e0e0 100%)";

      refreshButton();
    });
  });

  refreshButton();
});