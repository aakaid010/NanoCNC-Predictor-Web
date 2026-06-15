/* ============================================================
   NanoCNC Predictor — frontend logic
   ============================================================ */

// Use the same origin so the app works on localhost, Render, or any host.
// Override with window.NANOCNC_API_BASE if you need a separate API host.
const API_BASE =
  (typeof window !== "undefined" && window.NANOCNC_API_BASE) ||
  window.location.origin;

const SLIDER_INPUTS = [
  { slider: "acidConc",   input: "acidConcInput",   min: 8,   max: 75   },
  { slider: "tempC",      input: "tempCInput",      min: 25,  max: 100  },
  { slider: "timeMin",    input: "timeMinInput",    min: 1,   max: 900  },
];

const TYPICAL_RANGES = {
  acidConc: { min: 55, max: 65, hardMin: 8, hardMax: 75 },
  tempC:    { min: 40, max: 65, hardMin: 25, hardMax: 100 },
  timeMin:  { min: 30, max: 120, hardMin: 1, hardMax: 900 },
};

/* ----------------- DOM ----------------- */
const $ = (id) => document.getElementById(id);

const els = {
  celluloseGroup: $("celluloseGroup"),
  acidConc:       $("acidConc"),
  acidConcInput:  $("acidConcInput"),
  tempC:          $("tempC"),
  tempCInput:     $("tempCInput"),
  timeMin:        $("timeMin"),
  timeMinInput:   $("timeMinInput"),
  predictBtn:     $("predictBtn"),
  warningBanner:  $("warningBanner"),
  resultsCard:    $("resultsCard"),
  lengthValue:    $("lengthValue"),
  crystallinityValue: $("crystallinityValue"),
  lengthNote:     $("lengthNote"),
  crystallinityNote: $("crystallinityNote"),
  modelUsed:      $("modelUsed"),
  modelBadge:     $("modelBadge"),
  modelFile:      $("modelFile"),
  uploadBtn:      $("uploadBtn"),
  uploadStatus:   $("uploadStatus"),
  toast:          $("toast"),
};

/* ============================================================
   Slider / number input sync
   ============================================================ */
function updateSliderFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.setProperty("--fill", `${pct}%`);
}

function syncSliderAndInput(sliderId, inputId) {
  const slider = document.getElementById(sliderId);
  const input = document.getElementById(inputId);

  if (!slider || !input) return;

  // Slider -> number
  slider.addEventListener("input", () => {
    input.value = slider.value;
    updateSliderFill(slider);
    validateRanges();
  });

  // Number -> slider (clamped to slider min/max)
  input.addEventListener("input", () => {
    let v = parseFloat(input.value);
    if (Number.isNaN(v)) return;
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    if (v < min) v = min;
    if (v > max) v = max;
    slider.value = v;
    updateSliderFill(slider);
    validateRanges();
  });

  // Commit on blur so values just outside the typed range get clamped visibly
  input.addEventListener("blur", () => {
    let v = parseFloat(input.value);
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    if (Number.isNaN(v)) v = (min + max) / 2;
    if (v < min) v = min;
    if (v > max) v = max;
    input.value = v;
    slider.value = v;
    updateSliderFill(slider);
    validateRanges();
  });

  updateSliderFill(slider);
}

/* ============================================================
   Range validation (yellow warning banner)
   ============================================================ */
function validateRanges() {
  const acidConc = parseFloat(els.acidConc.value);
  const tempC = parseFloat(els.tempC.value);
  const timeMin = parseFloat(els.timeMin.value);

  const outOfRange =
    acidConc < TYPICAL_RANGES.acidConc.hardMin || acidConc > TYPICAL_RANGES.acidConc.hardMax ||
    tempC    < TYPICAL_RANGES.tempC.hardMin    || tempC    > TYPICAL_RANGES.tempC.hardMax    ||
    timeMin  < TYPICAL_RANGES.timeMin.hardMin  || timeMin  > TYPICAL_RANGES.timeMin.hardMax;

  if (outOfRange) {
    els.warningBanner.classList.remove("hidden");
  } else {
    els.warningBanner.classList.add("hidden");
  }

  return { acidConc, tempC, timeMin, outOfRange };
}

/* ============================================================
   Count-up animation
   ============================================================ */
function countUpAnimation(element, targetValue, decimals = 0, duration = 800) {
  const start = performance.now();
  const startValue = 0;

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function frame(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = easeOutCubic(progress);
    const current = startValue + (targetValue - startValue) * eased;
    element.textContent = current.toFixed(decimals);
    if (progress < 1) {
      requestAnimationFrame(frame);
    } else {
      element.textContent = targetValue.toFixed(decimals);
    }
  }

  requestAnimationFrame(frame);
}

/* ============================================================
   Predict
   ============================================================ */
async function predict() {
  const { acidConc, tempC, timeMin } = validateRanges();
  const celluloseGroup = els.celluloseGroup.value;

  // Lock UI
  els.predictBtn.disabled = true;
  els.predictBtn.classList.add("loading");
  els.celluloseGroup.disabled = true;
  els.acidConc.disabled = true;
  els.acidConcInput.disabled = true;
  els.tempC.disabled = true;
  els.tempCInput.disabled = true;
  els.timeMin.disabled = true;
  els.timeMinInput.disabled = true;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cellulose_group: celluloseGroup,
        acid_conc_wt_percent: acidConc,
        temp_c: tempC,
        time_min: timeMin,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let errMsg = `Server returned ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson && errJson.error) errMsg = errJson.error;
      } catch (_) { /* ignore */ }
      throw new Error(errMsg);
    }

    const data = await response.json();
    handlePredictionResult(data);
  } catch (err) {
    const message =
      err.name === "AbortError"
        ? "Prediction request timed out after 10 seconds."
        : "Could not connect to prediction server. Make sure Flask backend is running on port 5000.";
    showToast(message, "error");
    console.error("Prediction error:", err);
  } finally {
    els.predictBtn.disabled = false;
    els.predictBtn.classList.remove("loading");
    els.celluloseGroup.disabled = false;
    els.acidConc.disabled = false;
    els.acidConcInput.disabled = false;
    els.tempC.disabled = false;
    els.tempCInput.disabled = false;
    els.timeMin.disabled = false;
    els.timeMinInput.disabled = false;
  }
}

function handlePredictionResult(data) {
  const length = Number(data.cnc_length_nm) || 0;
  const crystallinity = Number(data.crystallinity_percent) || 0;
  const modelUsed = data.model_used || "Unknown";
  const note = data.confidence_note || "";

  // Show results card (fade-in via CSS animation)
  els.resultsCard.classList.remove("hidden");
  // Force reflow so the animation replays cleanly
  void els.resultsCard.offsetWidth;
  els.resultsCard.style.animation = "none";
  void els.resultsCard.offsetWidth;
  els.resultsCard.style.animation = "";

  // Decimals: 0 if integer, else up to 2
  const lengthDecimals = Number.isInteger(length) ? 0 : 2;
  const crystalDecimals = Number.isInteger(crystallinity) ? 0 : 2;

  countUpAnimation(els.lengthValue, length, lengthDecimals, 800);
  countUpAnimation(els.crystallinityValue, crystallinity, crystalDecimals, 800);

  // Interpretation notes
  els.lengthNote.textContent = interpretLength(length);
  els.crystallinityNote.textContent = interpretCrystallinity(crystallinity);

  els.modelUsed.textContent = modelUsed;
  if (note) {
    showToast(note, "success");
  }

  // Scroll into view smoothly
  els.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function interpretLength(nm) {
  if (nm < 200) return "Ultrashort nanocrystals";
  if (nm <= 500) return "Typical range";
  return "Elongated nanocrystals";
}

function interpretCrystallinity(pct) {
  if (pct > 75) return "Highly crystalline";
  if (pct >= 60) return "Moderately crystalline";
  return "Low crystallinity";
}

/* ============================================================
   Model info / boot
   ============================================================ */
async function loadModelInfo() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE}/model-info`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json();

    if (info.demo_mode) {
      els.modelBadge.textContent = "Demo Mode · Thesis Research Tool";
      els.modelBadge.classList.add("demo");
    } else {
      const r2avg = ((Number(info.r2_length) || 0) + (Number(info.r2_crystallinity) || 0)) / 2;
      els.modelBadge.textContent = `${info.model_name} · R² ${r2avg.toFixed(2)}`;
      els.modelBadge.classList.remove("demo");
    }
  } catch (err) {
    els.modelBadge.textContent = "Demo Mode · Server offline";
    els.modelBadge.classList.add("demo");
    console.warn("Could not reach backend:", err);
  }
}

/* ============================================================
   Upload model
   ============================================================ */
async function uploadModel() {
  const file = els.modelFile.files[0];
  if (!file) {
    setUploadStatus("Please select a .pkl file first.", "error");
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pkl")) {
    setUploadStatus("Only .pkl files are accepted.", "error");
    return;
  }

  setUploadStatus("Uploading…", "");
  els.uploadBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("model", file);

    const res = await fetch(`${API_BASE}/upload-model`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || `Upload failed (${res.status})`);
    }

    setUploadStatus(`✓ ${data.message}`, "success");
    showToast("Model loaded — future predictions will use the trained ML model.", "success");
    await loadModelInfo();
  } catch (err) {
    setUploadStatus(`✗ ${err.message}`, "error");
    showToast(err.message, "error");
  } finally {
    els.uploadBtn.disabled = false;
  }
}

function setUploadStatus(msg, kind) {
  els.uploadStatus.textContent = msg;
  els.uploadStatus.className = "upload-status" + (kind ? ` ${kind}` : "");
}

/* ============================================================
   Toast
   ============================================================ */
let toastTimer = null;
function showToast(message, kind = "") {
  els.toast.textContent = message;
  els.toast.className = "toast" + (kind ? ` ${kind}` : "");
  // Trigger transition
  void els.toast.offsetWidth;
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.classList.remove("show");
  }, 4500);
}

/* ============================================================
   Boot
   ============================================================ */
function boot() {
  // Wire slider <-> input
  for (const { slider, input } of SLIDER_INPUTS) {
    syncSliderAndInput(slider, input);
  }

  // Predict button
  els.predictBtn.addEventListener("click", predict);

  // Enter key in any input triggers predict
  document.querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        predict();
      }
    });
  });

  // Upload
  els.uploadBtn.addEventListener("click", uploadModel);

  // Initial range check + model info load
  validateRanges();
  loadModelInfo();
}

document.addEventListener("DOMContentLoaded", boot);
