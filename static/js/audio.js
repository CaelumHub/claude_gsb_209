/* audio.js — Web Audio helpers: base64 WAV playback and microphone capture. */

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function playBase64Wav(b64) {
  const buf = base64ToArrayBuffer(b64);
  const blob = new Blob([buf], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.onended = () => URL.revokeObjectURL(url);
  audio.play().catch(() => toast("无法播放预览", "error"));
  return audio;
}

/**
 * Capture microphone audio and deliver PCM frames for local visualisation and
 * remote real-time analysis.
 *
 *   startMic({ onFrame, onLevel }).then(stopFn => ...)
 *
 *   onLevel(samples)  — every animation frame (local waveform / spectrum)
 *   onFrame(samples, sr) — throttled (~20 Hz) for the backend analysis POST
 */
function startMic({ onFrame, onLevel } = {}) {
  return new Promise((resolve, reject) => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      reject(new Error("当前浏览器不支持麦克风采集"));
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      const sr = ctx.sampleRate;
      const N = 1024;
      const samples = new Array(N);
      let raf = 0;
      let lastSend = 0;

      const loop = (ts) => {
        analyser.getFloatTimeDomainData(buf);
        for (let i = 0; i < N; i++) samples[i] = buf[i * 2];
        if (onLevel) onLevel(samples, sr);
        if (onFrame && ts - lastSend > 50) {
          lastSend = ts;
          onFrame(samples.slice(), sr);
        }
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);

      resolve(() => {
        cancelAnimationFrame(raf);
        stream.getTracks().forEach((t) => t.stop());
        try { ctx.close(); } catch (_) { /* noop */ }
      });
    }).catch((e) => reject(new Error("无法访问麦克风: " + e.message)));
  });
}

/* Small local helpers shared by realtime visualisers. */
function drawLiveWave(canvas, samples, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;
  const color = opts.color || "#3fb950";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  const step = w / samples.length;
  for (let i = 0; i < samples.length; i++) {
    const x = i * step;
    const y = mid - samples[i] * mid;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawLiveSpectrum(canvas, samples, opts = {}) {
  // FFT of the windowed samples, drawn as a bar spectrum.
  const N = 512;
  const win = new Array(N);
  for (let i = 0; i < N; i++) {
    win[i] = samples[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  }
  const re = win.slice(), im = new Array(N).fill(0);
  jsfft(re, im, false);
  const mags = new Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    mags[i] = Math.log10(1 + Math.hypot(re[i], im[i]) * 10);
  }
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const barW = w / mags.length;
  const max = Math.max(...mags);
  const color = opts.color || "#bc8cff";
  ctx.fillStyle = color;
  for (let i = 0; i < mags.length; i++) {
    const bh = (mags[i] / max) * h;
    ctx.fillRect(i * barW, h - bh, Math.max(1, barW - 1), bh);
  }
}

/* Iterative radix-2 FFT in JS for the live spectrum (self-contained). */
function jsfft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
