import { InferenceEngine, CVImage } from "inferencejs";

const STORAGE_KEY = "chip-pot-counter-config-v1";

/** @typedef {{ publishableKey: string; modelId: string; modelVersion: number; minConfidence: number; maxFps: number; chipValues: Record<string, number> }} AppConfig */

function defaultConfig() {
  return {
    publishableKey: "",
    modelId: "",
    modelVersion: 1,
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

/** @type {AppConfig} */
let config = loadConfig();
/** @type {Set<string>} */
const discoveredClasses = new Set(Object.keys(config.chipValues));

/** @type {InferenceEngine | null} */
let inferEngine = null;
/** @type {string | null} */
let workerId = null;
/** @type {string | null} */
let workerConfigKey = null;

let lastInferTime = 0;
let inferBusy = false;
/** @type {Record<string, number>} */
let lastCounts = {};
let rafId = 0;

function setStatus(text) {
  el.status.textContent = text;
}

function configFingerprint() {
  return `${config.modelId}|${config.modelVersion}|${config.publishableKey}`;
}

async function stopInference() {
  if (inferEngine && workerId) {
    try {
      await inferEngine.stopWorker(workerId);
    } catch {
      /* ignore */
    }
  }
  workerId = null;
  workerConfigKey = null;
}

async function ensureWorker() {
  const fp = configFingerprint();
  if (!config.publishableKey?.trim() || !config.modelId?.trim()) {
    setStatus("Open settings and set your publishable key and model slug.");
    return;
  }

  if (workerId && workerConfigKey === fp) return;

  await stopInference();
  setStatus("Loading model…");

  if (!inferEngine) inferEngine = new InferenceEngine();

  try {
    workerId = await inferEngine.startWorker(
      config.modelId.trim(),
      Number(config.modelVersion) || 1,
      config.publishableKey.trim(),
    );
    workerConfigKey = fp;
    setStatus("Model ready. Point the camera at the pot.");
  } catch (err) {
    workerId = null;
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Model failed to load: ${msg}`);
  }
}

function isNormalizedBbox(b) {
  if (!b) return false;
  const vals = [b.x, b.y, b.width, b.height];
  return vals.every((v) => typeof v === "number" && v >= 0 && v <= 1.0001);
}

/** @param {{ class?: string; confidence?: number; bbox?: { x: number; y: number; width: number; height: number } }[]} preds */
function drawOverlay(preds, vw, vh) {
  el.canvas.width = vw;
  el.canvas.height = vh;
  ctx.clearRect(0, 0, vw, vh);
  const minConf = config.minConfidence;

  for (const p of preds) {
    if ((p.confidence ?? 0) < minConf) continue;
    const b = p.bbox;
    if (!b) continue;
    const norm = isNormalizedBbox(b);
    const x = norm ? b.x * vw : b.x;
    const y = norm ? b.y * vh : b.y;
    const w = norm ? b.width * vw : b.width;
    const h = norm ? b.height * vh : b.height;
    const cls = p.class ?? "unknown";
    const hue = classHue(cls);
    ctx.strokeStyle = `hsl(${hue} 85% 52%)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = `hsla(${hue} 85% 40% / 0.35)`;
    ctx.fillRect(x, y, w, h);
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

async function inferFrame() {
  if (!inferEngine || !workerId || el.video.readyState < 2) return;
  const vw = el.video.videoWidth;
  const vh = el.video.videoHeight;
  if (!vw || !vh) return;

  const cvImg = new CVImage(el.video);
  try {
    const preds = await inferEngine.infer(workerId, cvImg);
    drawOverlay(preds, vw, vh);
    summarizeFrame(preds);
  } finally {
    cvImg.dispose();
  }
}

function loop(time) {
  rafId = requestAnimationFrame(loop);
  if (!workerId || inferBusy) return;
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
  el.cfgKey.value = config.publishableKey;
  el.cfgModel.value = config.modelId;
  el.cfgVersion.value = String(config.modelVersion);
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
  config.publishableKey = el.cfgKey.value.trim();
  config.modelId = el.cfgModel.value.trim();
  config.modelVersion = Math.max(1, parseInt(el.cfgVersion.value, 10) || 1);
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

el.btnCancel.addEventListener("click", () => closeSettings());

el.backdrop.addEventListener("click", () => closeSettings());

el.btnSave.addEventListener("click", async (e) => {
  e.preventDefault();
  applySettingsFromForm();
  closeSettings();
  await ensureWorker();
});

el.form.addEventListener("submit", (e) => e.preventDefault());

async function init() {
  cancelAnimationFrame(rafId);
  await stopInference();
  config = loadConfig();
  await startCamera();
  await ensureWorker();
  rafId = requestAnimationFrame(loop);
}

init();
