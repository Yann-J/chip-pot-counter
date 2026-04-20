import * as ort from "onnxruntime-web";

const STORAGE_KEY = "chip-pot-counter-config-v1";
const ORT_WASM_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// In local Vite dev, some setups can serve ORT's internal wasm files
// through a fallback MIME type. Force a stable CDN path that serves
// `application/wasm` so ORT can initialize consistently.
ort.env.wasm.wasmPaths = ORT_WASM_DIST;

/** @typedef {{ modelUrl: string; classesUrl: string; inputSize: number; minConfidence: number; maxFps: number; chipValues: Record<string, number> }} AppConfig */

function defaultConfig() {
  return {
    modelUrl: "models/model.onnx",
    classesUrl: "models/classes.json",
    inputSize: 640,
    minConfidence: 0.4,
    maxFps: 8,
    chipValues: {},
  };
}

/** @returns {AppConfig} */
function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultConfig();
    const parsed = JSON.parse(raw);
    return {
      ...defaultConfig(),
      ...parsed,
      chipValues: parsed.chipValues ?? {},
    };
  } catch {
    return defaultConfig();
  }
}

/** @param {AppConfig} cfg */
function saveConfig(cfg) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

/** @param {string} cls */
function classHue(cls) {
  let h = 0;
  for (let i = 0; i < cls.length; i++) h = (h * 31 + cls.charCodeAt(i)) >>> 0;
  return h % 360;
}

const el = {
  status: document.getElementById("status-line"),
  video: /** @type {HTMLVideoElement} */ (document.getElementById("camera")),
  canvas: /** @type {HTMLCanvasElement} */ (document.getElementById("overlay")),
  classCounts: document.getElementById("class-counts"),
  potTotal: document.getElementById("pot-total"),
  btnToggleCapture: document.getElementById("btn-toggle-capture"),
  btnSettings: document.getElementById("btn-settings"),
  backdrop: document.getElementById("modal-backdrop"),
  dialog: /** @type {HTMLDialogElement} */ (
    document.getElementById("settings-dialog")
  ),
  form: /** @type {HTMLFormElement} */ (
    document.getElementById("settings-form")
  ),
  cfgKey: /** @type {HTMLInputElement} */ (document.getElementById("cfg-key")),
  cfgModel: /** @type {HTMLInputElement} */ (
    document.getElementById("cfg-model")
  ),
  cfgVersion: /** @type {HTMLInputElement} */ (
    document.getElementById("cfg-version")
  ),
  cfgConfidence: /** @type {HTMLInputElement} */ (
    document.getElementById("cfg-confidence")
  ),
  cfgFps: /** @type {HTMLInputElement} */ (document.getElementById("cfg-fps")),
  chipRows: document.getElementById("chip-rows"),
  btnAddClass: document.getElementById("btn-add-class"),
  btnCancel: document.getElementById("btn-cancel-settings"),
  btnSave: document.getElementById("btn-save-settings"),
};

const ctx = el.canvas.getContext("2d");
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

/** @type {AppConfig} */
let config = loadConfig();
/** @type {Set<string>} */
const discoveredClasses = new Set(Object.keys(config.chipValues));

/** @type {ort.InferenceSession | null} */
let session = null;
/** @type {Record<number, string>} */
let classNames = {};
/** @type {string | null} */
let modelConfigKey = null;

let lastInferTime = 0;
let inferBusy = false;
let capturePaused = false;
/** @type {Record<string, number>} */
let lastCounts = {};
let rafId = 0;

function setStatus(text) {
  el.status.textContent = text;
}

function renderCaptureButton() {
  const icon = el.btnToggleCapture.querySelector("i");
  if (!icon) return;
  icon.className = capturePaused ? "bi bi-play-fill" : "bi bi-pause-fill";
  el.btnToggleCapture.setAttribute(
    "aria-label",
    capturePaused ? "Resume capture" : "Pause capture",
  );
  el.btnToggleCapture.title = capturePaused
    ? "Resume capture"
    : "Pause capture";
}

function setCapturePaused(paused, reason = "manual") {
  const changed = capturePaused !== paused;
  capturePaused = paused;
  renderCaptureButton();

  if (!changed) return;
  if (capturePaused) {
    if (!el.video.paused) {
      el.video.pause();
    }
    if (reason === "settings") {
      setStatus("Capture paused while settings are open.");
    } else {
      setStatus("Capture paused. Resume when you are ready.");
    }
    return;
  }
  if (el.video.srcObject && el.video.paused) {
    el.video.play().catch(() => {
      setStatus("Capture resumed, but video preview needs a user gesture.");
    });
  }
  setStatus("Capture resumed.");
}

function configFingerprint() {
  return `${config.modelUrl}|${config.classesUrl}|${config.inputSize}`;
}

function resolveAssetUrl(value) {
  const trimmed = value.trim();
  if (!trimmed) return "";
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return new URL(trimmed, window.location.href).toString();
}

async function loadClassNames(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Unable to fetch classes file (${resp.status}).`);
  }
  const data = await resp.json();
  /** @type {Record<number, string>} */
  const out = {};
  for (const [k, v] of Object.entries(data)) {
    const idx = parseInt(k, 10);
    if (Number.isFinite(idx) && typeof v === "string") out[idx] = v;
  }
  return out;
}

function stopInference() {
  session = null;
  modelConfigKey = null;
}

async function ensureModel() {
  const fp = configFingerprint();
  if (!config.modelUrl?.trim() || !config.classesUrl?.trim()) {
    setStatus("Open settings and configure model and classes URLs.");
    return;
  }

  if (session && modelConfigKey === fp) return;

  stopInference();
  setStatus("Loading model…");
  const modelUrl = resolveAssetUrl(config.modelUrl);
  const classesUrl = resolveAssetUrl(config.classesUrl);

  try {
    classNames = await loadClassNames(classesUrl);
    Object.values(classNames).forEach((name) => discoveredClasses.add(name));
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    modelConfigKey = fp;
    setStatus("Model ready. Point the camera at the pot.");
  } catch (err) {
    session = null;
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Model failed to load: ${msg}`);
  }
}

function isNormalizedBbox(b) {
  if (!b) return false;
  const vals = [b.x, b.y, b.width, b.height];
  return vals.every((v) => typeof v === "number" && v >= 0 && v <= 1.0001);
}

function hslToRgb(h, s, l) {
  const hh = ((h % 360) + 360) % 360;
  const ss = Math.max(0, Math.min(1, s));
  const ll = Math.max(0, Math.min(1, l));
  const c = (1 - Math.abs(2 * ll - 1)) * ss;
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1));
  const m = ll - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (hh < 60) [r, g, b] = [c, x, 0];
  else if (hh < 120) [r, g, b] = [x, c, 0];
  else if (hh < 180) [r, g, b] = [0, c, x];
  else if (hh < 240) [r, g, b] = [0, x, c];
  else if (hh < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * @typedef {{
 *  class?: string;
 *  confidence?: number;
 *  bbox?: { x: number; y: number; width: number; height: number };
 *  maskCoeffs?: Float32Array;
 * }} Detection
 */

/**
 * @param {Detection} pred
 * @param {ort.Tensor} proto
 * @param {number} inputSize
 * @param {number} videoWidth
 * @param {number} videoHeight
 */
function renderMaskForDetection(
  pred,
  proto,
  inputSize,
  videoWidth,
  videoHeight,
) {
  const bbox = pred.bbox;
  const coeffs = pred.maskCoeffs;
  if (!bbox || !coeffs || !maskCtx) return;
  const protoDims = proto.dims;
  if (protoDims.length !== 4) return;
  const maskChannels = protoDims[1];
  const maskHeight = protoDims[2];
  const maskWidth = protoDims[3];
  if (coeffs.length !== maskChannels) return;
  if (!(proto.data instanceof Float32Array)) return;
  const protoData = proto.data;

  const x0 = Math.max(0, Math.floor(bbox.x));
  const y0 = Math.max(0, Math.floor(bbox.y));
  const x1 = Math.min(videoWidth, Math.ceil(bbox.x + bbox.width));
  const y1 = Math.min(videoHeight, Math.ceil(bbox.y + bbox.height));
  const boxWidth = x1 - x0;
  const boxHeight = y1 - y0;
  if (boxWidth <= 1 || boxHeight <= 1) return;

  const scale = Math.min(inputSize / videoWidth, inputSize / videoHeight);
  const padX = (inputSize - videoWidth * scale) / 2;
  const padY = (inputSize - videoHeight * scale) / 2;

  maskCanvas.width = boxWidth;
  maskCanvas.height = boxHeight;
  const imageData = maskCtx.createImageData(boxWidth, boxHeight);
  const out = imageData.data;
  const hue = classHue(pred.class ?? "unknown");
  const rgb = hslToRgb(hue, 0.85, 0.52);

  for (let y = 0; y < boxHeight; y++) {
    const videoY = y0 + y;
    const modelY = videoY * scale + padY;
    const py = Math.max(
      0,
      Math.min(maskHeight - 1, Math.floor((modelY / inputSize) * maskHeight)),
    );
    for (let x = 0; x < boxWidth; x++) {
      const videoX = x0 + x;
      const modelX = videoX * scale + padX;
      const px = Math.max(
        0,
        Math.min(maskWidth - 1, Math.floor((modelX / inputSize) * maskWidth)),
      );
      let logit = 0;
      for (let c = 0; c < maskChannels; c++) {
        logit +=
          coeffs[c] *
          protoData[c * maskHeight * maskWidth + py * maskWidth + px];
      }
      if (sigmoid(logit) < 0.5) continue;
      const i = (y * boxWidth + x) * 4;
      out[i] = rgb.r;
      out[i + 1] = rgb.g;
      out[i + 2] = rgb.b;
      out[i + 3] = 85;
    }
  }

  maskCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(maskCanvas, x0, y0);
}

/** @param {Detection[]} preds */
function drawOverlay(preds, vw, vh, proto, inputSize) {
  el.canvas.width = vw;
  el.canvas.height = vh;
  ctx.clearRect(0, 0, vw, vh);
  const minConf = config.minConfidence;

  for (const p of preds) {
    if ((p.confidence ?? 0) < minConf) continue;
    renderMaskForDetection(p, proto, inputSize, vw, vh);
  }

  for (const p of preds) {
    if ((p.confidence ?? 0) < minConf) continue;
    const b = p.bbox;
    if (!b) continue;
    const x = b.x;
    const y = b.y;
    const cls = p.class ?? "unknown";
    const hue = classHue(cls);
    const label = `${cls} ${((p.confidence ?? 0) * 100).toFixed(0)}%`;
    ctx.font = "14px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = `hsl(${hue} 30% 18%)`;
    ctx.fillRect(x, Math.max(0, y - 20), tw + 8, 20);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(label, x + 4, Math.max(14, y - 5));
  }
}

/** @param {{ class?: string; confidence?: number; bbox?: { x: number; y: number; width: number; height: number } }[]} preds */
function summarizeFrame(preds) {
  const minConf = config.minConfidence;
  /** @type {Record<string, number>} */
  const counts = {};
  let total = 0;

  for (const p of preds) {
    if ((p.confidence ?? 0) < minConf) continue;
    const cls = p.class ?? "unknown";
    counts[cls] = (counts[cls] ?? 0) + 1;
    discoveredClasses.add(cls);
  }

  for (const [cls, n] of Object.entries(counts)) {
    const unit = config.chipValues[cls] ?? 0;
    total += n * unit;
  }

  lastCounts = counts;
  renderCounts(counts, total);
}

function renderCounts(counts, total) {
  el.classCounts.innerHTML = "";
  const classes = Object.keys(counts).sort();
  if (classes.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No chips above confidence threshold.";
    el.classCounts.appendChild(li);
  } else {
    for (const cls of classes) {
      const li = document.createElement("li");
      const unit = config.chipValues[cls] ?? 0;
      const sub = unit ? ` × ${unit} = ${(counts[cls] * unit).toFixed(2)}` : "";
      li.innerHTML = `<span class="cc-name">${escapeHtml(cls)}</span><span class="cc-num">${counts[cls]}${sub ? `<span class="cc-sub">${escapeHtml(sub)}</span>` : ""}</span>`;
      el.classCounts.appendChild(li);
    }
  }

  el.potTotal.textContent = Number.isFinite(total) ? total.toFixed(2) : "—";
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

function nms(preds, threshold = 0.5) {
  const sorted = [...preds].sort(
    (a, b) => (b.confidence ?? 0) - (a.confidence ?? 0),
  );
  /** @type {Detection[]} */
  const kept = [];
  for (const p of sorted) {
    if (!p.bbox) continue;
    const overlaps = kept.some(
      (k) => k.class === p.class && k.bbox && iou(k.bbox, p.bbox) > threshold,
    );
    if (!overlaps) kept.push(p);
  }
  return kept;
}

function extractDetections(
  output,
  minConfidence,
  inputSize,
  videoWidth,
  videoHeight,
  maskChannels,
) {
  const dims = output.dims;
  if (dims.length < 3) return [];
  const data = output.data;
  if (!(data instanceof Float32Array)) return [];

  let channels;
  let candidates;
  let channelsFirst = false;
  if (dims[1] < dims[2]) {
    channels = dims[1];
    candidates = dims[2];
    channelsFirst = true;
  } else {
    candidates = dims[1];
    channels = dims[2];
  }
  const classesCount = Object.keys(classNames).length;
  const boxOffset = 4;
  const coeffOffset = boxOffset + classesCount;
  if (channels < coeffOffset + maskChannels) return [];

  const scale = Math.min(inputSize / videoWidth, inputSize / videoHeight);
  const padX = (inputSize - videoWidth * scale) / 2;
  const padY = (inputSize - videoHeight * scale) / 2;

  /** @type {Detection[]} */
  const detections = [];
  for (let i = 0; i < candidates; i++) {
    const at = (c) => {
      if (channelsFirst) return data[c * candidates + i];
      return data[i * channels + c];
    };
    const cx = at(0);
    const cy = at(1);
    const w = at(2);
    const h = at(3);
    let clsIdx = -1;
    let bestScore = 0;
    for (let c = 0; c < classesCount; c++) {
      const score = at(boxOffset + c);
      if (score > bestScore) {
        bestScore = score;
        clsIdx = c;
      }
    }
    if (bestScore < minConfidence || clsIdx < 0) continue;

    const x0 = (cx - w / 2 - padX) / scale;
    const y0 = (cy - h / 2 - padY) / scale;
    const x1 = (cx + w / 2 - padX) / scale;
    const y1 = (cy + h / 2 - padY) / scale;

    const x = Math.max(0, Math.min(videoWidth, x0));
    const y = Math.max(0, Math.min(videoHeight, y0));
    const bx = Math.max(0, Math.min(videoWidth, x1) - x);
    const by = Math.max(0, Math.min(videoHeight, y1) - y);
    if (bx < 2 || by < 2) continue;

    detections.push({
      class: classNames[clsIdx] ?? `class_${clsIdx}`,
      confidence: bestScore,
      bbox: { x, y, width: bx, height: by },
      maskCoeffs: new Float32Array(
        Array.from({ length: maskChannels }, (_, idx) => at(coeffOffset + idx)),
      ),
    });
  }
  return nms(detections, 0.5);
}

function preprocessFrame(video, inputSize) {
  const offscreen = document.createElement("canvas");
  offscreen.width = inputSize;
  offscreen.height = inputSize;
  const c = offscreen.getContext("2d");
  if (!c) throw new Error("Canvas context unavailable.");
  c.fillStyle = "#000";
  c.fillRect(0, 0, inputSize, inputSize);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(inputSize / vw, inputSize / vh);
  const drawW = Math.round(vw * scale);
  const drawH = Math.round(vh * scale);
  const dx = Math.floor((inputSize - drawW) / 2);
  const dy = Math.floor((inputSize - drawH) / 2);
  c.drawImage(video, 0, 0, vw, vh, dx, dy, drawW, drawH);
  const pixels = c.getImageData(0, 0, inputSize, inputSize).data;
  const chw = new Float32Array(3 * inputSize * inputSize);
  const channelSize = inputSize * inputSize;
  for (let i = 0; i < channelSize; i++) {
    const p = i * 4;
    chw[i] = pixels[p] / 255;
    chw[channelSize + i] = pixels[p + 1] / 255;
    chw[channelSize * 2 + i] = pixels[p + 2] / 255;
  }
  return new ort.Tensor("float32", chw, [1, 3, inputSize, inputSize]);
}

async function inferFrame() {
  if (!session || el.video.readyState < 2) return;
  const vw = el.video.videoWidth;
  const vh = el.video.videoHeight;
  if (!vw || !vh) return;
  const inputSize = Math.max(32, config.inputSize || 640);
  const tensor = preprocessFrame(el.video, inputSize);
  const feeds = { [session.inputNames[0]]: tensor };
  const outputs = await session.run(feeds);
  const outputList = session.outputNames
    .map((name) => outputs[name])
    .filter(Boolean);
  const detectionOutput = outputList.find((t) => t.dims.length === 3);
  const protoOutput = outputList.find((t) => t.dims.length === 4);
  if (!detectionOutput || !protoOutput) {
    throw new Error(
      "Expected YOLOv8-seg outputs (detections + mask prototypes).",
    );
  }
  const preds = extractDetections(
    detectionOutput,
    config.minConfidence,
    inputSize,
    vw,
    vh,
    protoOutput.dims[1],
  );
  drawOverlay(preds, vw, vh, protoOutput, inputSize);
  summarizeFrame(preds);
}

function loop(time) {
  rafId = requestAnimationFrame(loop);
  if (!session || inferBusy || capturePaused) return;
  const maxFps = Math.max(1, Math.min(30, config.maxFps || 8));
  const interval = 1000 / maxFps;
  if (time - lastInferTime < interval) return;
  lastInferTime = time;
  inferBusy = true;
  inferFrame()
    .catch((err) => {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Inference error: ${msg}`);
    })
    .finally(() => {
      inferBusy = false;
    });
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    el.video.srcObject = stream;
    await el.video.play();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Camera error: ${msg}`);
  }
}

function openSettings() {
  setCapturePaused(true, "settings");
  el.cfgKey.value = config.modelUrl;
  el.cfgModel.value = config.classesUrl;
  el.cfgVersion.value = String(config.inputSize);
  el.cfgConfidence.value = String(config.minConfidence);
  el.cfgFps.value = String(config.maxFps);
  renderChipRows();
  el.backdrop.classList.remove("hidden");
  el.dialog.showModal();
}

function closeSettings() {
  el.dialog.close();
  el.backdrop.classList.add("hidden");
}

function renderChipRows() {
  el.chipRows.innerHTML = "";
  const names = new Set([
    ...Object.keys(config.chipValues),
    ...discoveredClasses,
  ]);
  if (names.size === 0) {
    names.add("");
  }
  for (const name of [...names].sort()) {
    el.chipRows.appendChild(createChipRow(name, config.chipValues[name] ?? 0));
  }
}

/**
 * @param {string} className
 * @param {number} value
 */
function createChipRow(className, value) {
  const row = document.createElement("div");
  row.className = "chip-row";
  row.innerHTML = `
    <input type="text" class="chip-class" placeholder="class name" value="${escapeHtml(className)}" />
    <input type="number" class="chip-value" min="0" step="0.01" value="${Number.isFinite(value) ? value : 0}" />
    <button type="button" class="row-remove" aria-label="Remove row">×</button>
  `;
  row.querySelector(".row-remove")?.addEventListener("click", () => {
    row.remove();
    if (!el.chipRows.querySelector(".chip-row")) {
      el.chipRows.appendChild(createChipRow("", 0));
    }
  });
  return row;
}

function readChipValuesFromForm() {
  /** @type {Record<string, number>} */
  const out = {};
  for (const row of el.chipRows.querySelectorAll(".chip-row")) {
    const cls = /** @type {HTMLInputElement} */ (
      row.querySelector(".chip-class")
    ).value.trim();
    const valRaw = /** @type {HTMLInputElement} */ (
      row.querySelector(".chip-value")
    ).value;
    const val = parseFloat(valRaw);
    if (cls) out[cls] = Number.isFinite(val) ? val : 0;
  }
  return out;
}

function applySettingsFromForm() {
  config.modelUrl = el.cfgKey.value.trim();
  config.classesUrl = el.cfgModel.value.trim();
  config.inputSize = Math.max(32, parseInt(el.cfgVersion.value, 10) || 640);
  config.minConfidence = Math.min(
    1,
    Math.max(0, parseFloat(el.cfgConfidence.value) || 0),
  );
  config.maxFps = Math.min(30, Math.max(1, parseInt(el.cfgFps.value, 10) || 8));
  config.chipValues = readChipValuesFromForm();
  for (const k of Object.keys(config.chipValues)) discoveredClasses.add(k);
  saveConfig(config);
}

el.btnAddClass.addEventListener("click", () => {
  el.chipRows.appendChild(createChipRow("", 0));
});

el.btnSettings.addEventListener("click", () => openSettings());

el.btnToggleCapture.addEventListener("click", () => {
  setCapturePaused(!capturePaused, "manual");
});

el.btnCancel.addEventListener("click", () => closeSettings());

el.backdrop.addEventListener("click", () => closeSettings());

el.dialog.addEventListener("close", () => {
  el.backdrop.classList.add("hidden");
});

el.btnSave.addEventListener("click", async (e) => {
  e.preventDefault();
  applySettingsFromForm();
  closeSettings();
  await ensureModel();
});

el.form.addEventListener("submit", (e) => e.preventDefault());

async function init() {
  cancelAnimationFrame(rafId);
  stopInference();
  config = loadConfig();
  await startCamera();
  await ensureModel();
  renderCaptureButton();
  rafId = requestAnimationFrame(loop);
}

init();
