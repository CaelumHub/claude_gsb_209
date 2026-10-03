/* common.js — shared helpers: toasts, formatting, canvas renderers. */

/* ---------------------------------------------------------------- toasts */

function toast(msg, type = "info", ms = 3400) {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  const t = document.createElement("div");
  t.className = "toast " + type;
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/* -------------------------------------------------------------- formatting */

function fmtTime(s) {
  if (s == null || isNaN(s)) return "--:--";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

function fmtBytes(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(n >= 10 ? 1 : 2) + " " + units[i];
}

function fmtHz(f) { return f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : f.toFixed(0) + " Hz"; }

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function $(sel, root) { return (root || document).querySelector(sel); }
function $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

/* ----------------------------------------------------------- canvas utils */

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* Inferno-style colormap for spectrograms / heatmaps. */
function colormap(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, 11, 12, 43], [0.25, 74, 16, 92], [0.50, 173, 55, 60],
    [0.75, 240, 137, 33], [1.00, 252, 255, 164],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0 || 1);
      return [r0 + (r1 - r0) * k, g0 + (g1 - g0) * k, b0 + (b1 - b0) * k];
    }
  }
  return [252, 255, 164];
}

function colorStyle(t) {
  const [r, g, b] = colormap(t);
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

/* ------------------------------------------------------------- waveform */

function drawWaveform(canvas, env, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const mins = env.mins || [], maxs = env.maxs || [];
  ctx.clearRect(0, 0, w, h);
  if (!mins.length) return;

  const color = opts.color || getComputedStyle(document.documentElement)
    .getPropertyValue("--accent").trim() || "#58a6ff";
  const mid = h / 2;
  const step = w / mins.length;

  // selection highlight
  if (opts.selection) {
    const [a, b] = opts.selection;
    ctx.fillStyle = "rgba(88,166,255,0.15)";
    ctx.fillRect(a * w, 0, (b - a) * w, h);
  }
  if (opts.playhead != null) {
    ctx.fillStyle = "rgba(63,185,80,0.7)";
    ctx.fillRect(opts.playhead * w - 1, 0, 2, h);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < mins.length; i++) {
    const x = i * step;
    const yMin = mid - (maxs[i] || 0) * mid * 0.95;
    const yMax = mid - (mins[i] || 0) * mid * 0.95;
    ctx.moveTo(x, yMin);
    ctx.lineTo(x, yMax);
  }
  ctx.stroke();

  // centre line
  ctx.strokeStyle = "rgba(139,148,158,0.35)";
  ctx.beginPath();
  ctx.moveTo(0, mid); ctx.lineTo(w, mid);
  ctx.stroke();
}

/* ----------------------------------------------------------- spectrogram */

function drawSpectrogram(canvas, spec, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const data = spec.data || [];
  const rows = data.length;
  ctx.clearRect(0, 0, w, h);
  if (!rows) return;

  const img = ctx.createImageData(w, h);
  const cols = data[0].length;
  // dB normalisation: assume floor -100, ceiling 0.
  const dbMin = opts.dbMin ?? -100, dbMax = opts.dbMax ?? 0;
  for (let y = 0; y < h; y++) {
    const rowIdx = Math.floor((h - 1 - y) / h * rows);
    const row = data[rowIdx];
    for (let x = 0; x < w; x++) {
      const colIdx = Math.floor(x / w * cols);
      const v = row[colIdx];
      const t = (v - dbMin) / (dbMax - dbMin);
      const [r, g, b] = colormap(t);
      const idx = (y * w + x) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* -------------------------------------------------------------- line plot */

function drawLinePlot(canvas, seriesList, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 12, t: 10, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  let yMin = Infinity, yMax = -Infinity, xMax = 0;
  for (const s of seriesList) {
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
      xMax = Math.max(xMax, s.data.length);
    }
  }
  if (!isFinite(yMin)) return;
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const range = yMax - yMin;

  // grid
  ctx.strokeStyle = "rgba(139,148,158,0.15)";
  ctx.fillStyle = "rgba(139,148,158,0.6)";
  ctx.font = "10px sans-serif";
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + ph * (g / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    const val = yMax - range * (g / 4);
    ctx.fillText(fmtVal(val), 4, y + 3);
  }

  const xAt = (i) => pad.l + (xMax <= 1 ? 0 : i / (xMax - 1) * pw);
  const yAt = (v) => pad.t + (1 - (v - yMin) / range) * ph;

  for (const s of seriesList) {
    ctx.strokeStyle = s.color || "#58a6ff";
    ctx.lineWidth = s.width || 1.4;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i++) {
      const x = xAt(i), y = yAt(s.data[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function fmtVal(v) {
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "k";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}

/* --------------------------------------------------------------- heatmap */

function drawHeatmap(canvas, matrix, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const rows = matrix.length;
  ctx.clearRect(0, 0, w, h);
  if (!rows) return;
  const img = ctx.createImageData(w, h);
  const cols = matrix[0].length;
  for (let y = 0; y < h; y++) {
    const r = Math.floor(y / h * rows);
    const row = matrix[r];
    for (let x = 0; x < w; x++) {
      const c = Math.floor(x / w * cols);
      const [cr, cg, cb] = colormap(row[c]);
      const idx = (y * w + x) * 4;
      img.data[idx] = cr; img.data[idx + 1] = cg; img.data[idx + 2] = cb; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ----------------------------------------------------------- play helpers */

function playFile(id, name) {
  const url = API.fileUrl(id);
  const audio = new Audio(url);
  audio.play().catch(() => toast("无法播放该音频", "error"));
  toast("播放 " + name);
  return audio;
}
