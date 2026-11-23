// Token / channel covariance heat maps for the Latent view (data from /api/runs/<id>/cov/<step>).
// Diverging colour: cyan negative, dark zero, amber positive, on a symmetric scale set by the
// largest diagonal entry, so a whitened latent reads as a bright diagonal on a dark field.
import { ACCENT, ACCENT2 } from './palette.js';

export async function fetchCov(runId, step) {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/cov/${step}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

const hex = (h) => { const s = h.replace('#', ''); return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16)); };
function rgbOf(css) {                         // accepts #rrggbb or rgb()/hsl() via a canvas probe
  if (/^#[0-9a-f]{6}$/i.test(css)) return hex(css);
  const c = document.createElement('canvas').getContext('2d'); c.fillStyle = css; return hex(c.fillStyle);
}

/**
 * Draw an n×n matrix into `canvas`; returns stats for a caption.
 * `readout` (optional element) shows the value under the pointer.
 */
export function drawCov(canvas, M, n, readout, label) {
  const pos = rgbOf(ACCENT), neg = rgbOf(ACCENT2), bg = [14, 18, 26];
  let scale = 0, diag = 0, off = 0;
  for (let i = 0; i < n; i++) { scale = Math.max(scale, Math.abs(M[i * n + i])); diag += M[i * n + i]; }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) off += Math.abs(M[i * n + j]);
  scale = scale || 1;
  const img = new ImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const v = Math.max(-1, Math.min(1, M[i] / scale));
    const t = Math.sqrt(Math.abs(v));          // sqrt: keep weak off-diagonal structure visible
    const col = v >= 0 ? pos : neg;
    for (let k = 0; k < 3; k++) img.data[4 * i + k] = Math.round(bg[k] + (col[k] - bg[k]) * t);
    img.data[4 * i + 3] = 255;
  }
  canvas.width = n; canvas.height = n;
  canvas.getContext('2d').putImageData(img, 0, 0);
  canvas.onpointermove = (e) => {
    if (!readout) return;
    const r = canvas.getBoundingClientRect();
    const j = Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n));
    const i = Math.min(n - 1, Math.floor(((e.clientY - r.top) / r.height) * n));
    readout.textContent = `${label} ${i} × ${j}: ${M[i * n + j].toFixed(3)}`;
  };
  canvas.onpointerleave = () => { if (readout) readout.textContent = ''; };
  const meanDiag = diag / n, meanOff = off / (n * n - n);
  return { meanDiag, meanOff, ratio: meanOff / (Math.abs(meanDiag) || 1) };
}
