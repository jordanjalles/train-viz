// LineChart: a dependency-free SVG line chart with
//   - spring-damped y-range (math/time spring) so live appends and log/linear
//     toggles glide instead of snapping
//   - draw-in animation for new series (math/time easing)
//   - hover crosshair with a per-series readout
//   - log-scale toggle, multi-series with per-series colour
//   - drag-to-zoom on x (mouse/pen; touch keeps page scroll), × or double-click resets
//   - expanded view: the ⤢ button, or a drag on an inline chart, opens a large copy in a
//     modal <dialog>. The copy follows live data, zooms further by drag (touch too), and
//     closes with Esc, the close button or a backdrop click. The inline chart keeps its range.
//
// Data: setSeries([{ id, label, color, points: [[x, y], ...], dash? }])

import { spring, easing } from 'math/time';
import { remap, clamp } from 'math';
import { MUTED } from './palette.js';

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(v);
  return ticks;
}
/** Number for labels and readouts; `extra` adds decimals (the hover guide uses 2). */
const fmt = (v, extra = 0) => {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2 + extra);
  if (a >= 100) return v.toFixed(0 + extra);
  if (a >= 10) return v.toFixed(1 + extra);
  if (a >= 1) return v.toFixed(2 + extra);
  return v.toFixed(3 + extra);
};
const fmtX = (v) => (v >= 1000 ? `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}k` : String(Math.round(v)));

export class LineChart {
  constructor(el, { title = '', log = false, xLabel = 'step', height = 180, yMin = null, yMax = null, smooth = false, xFmt = fmtX, xName = 'step', inModal = false } = {}) {
    this.xFmt = xFmt; this.xName = xName;   // x need not be steps (e.g. noise level)
    this.el = el; this.title = title; this.log = log; this.xLabel = xLabel; this.height = height;
    this.fixed = { yMin, yMax }; this.smooth = smooth;
    this.inModal = inModal;   // the large copy inside the dialog: drag zooms in place
    this.series = [];
    this.pad = inModal ? { l: 58, r: 16, t: 12, b: 30 } : { l: 46, r: 12, t: 10, b: 24 };
    this.ylo = spring.create(0); this.yhi = spring.create(1);
    this._first = true;
    this._mirror = null;      // the open modal copy, if any
    el.classList.add('chart');
    el.innerHTML = `
      <div class="chart-head">
        <span class="chart-title">${title}</span>
        <span class="chart-readout"></span>
        <button class="chart-unzoom" title="zoom out to the full range" aria-label="zoom out to the full range" hidden>×</button>
        <button class="chart-log ${log ? 'on' : ''}" title="log scale" aria-pressed="${log}">log</button>
        ${inModal ? '' : '<button class="chart-expand" title="open larger (or drag across the chart to open a range)" aria-label="open chart larger">⤢</button>'}
      </div>`;
    if (!inModal) el.querySelector('.chart-expand').addEventListener('click', () => this.openModal(this.zoom));
    this.readout = el.querySelector('.chart-readout');
    this.logBtn = el.querySelector('.chart-log');
    this.logBtn.addEventListener('click', () => { this.log = !this.log; this.logBtn.classList.toggle('on', this.log); this.logBtn.setAttribute('aria-pressed', this.log); this.retarget(); });
    this.svg = svgEl('svg', { class: 'chart-svg' });
    el.appendChild(this.svg);
    this.gGrid = svgEl('g', { class: 'chart-grid' });
    this.gLines = svgEl('g', { class: 'chart-lines' });
    this.gHover = svgEl('g', { class: 'chart-hover' });
    this.gSel = svgEl('g', { class: 'chart-sel' });
    // lines are clipped to the plot area so a zoomed x-range doesn't spill over the axes
    this.clipId = `clip-${Math.random().toString(36).slice(2)}`;
    const defs = svgEl('defs'); this.clipRect = svgEl('rect');
    const cp = svgEl('clipPath', { id: this.clipId }); cp.append(this.clipRect); defs.append(cp);
    this.gLines.setAttribute('clip-path', `url(#${this.clipId})`);
    this.svg.append(defs, this.gGrid, this.gLines, this.gHover, this.gSel);
    this.zoom = null;   // {lo, hi} in data x, or null for the full range
    this.unzoomBtn = el.querySelector('.chart-unzoom');
    this.unzoomBtn.addEventListener('click', () => this.setZoom(null));
    this._bindZoom();
    this.W = 0;
    this.ro = new ResizeObserver(() => { this.W = el.clientWidth; this.render(); });
    this.ro.observe(el);
    this.svg.addEventListener('pointermove', (e) => this.hover(e));
    this.svg.addEventListener('pointerleave', () => this.hover(null));
    this._raf = null; this._last = performance.now();
    this.legend = document.createElement('div'); this.legend.className = 'chart-legend';
    el.appendChild(this.legend);
  }

  destroy() { this.ro.disconnect(); if (this._raf) cancelAnimationFrame(this._raf); this._mirror?.dialog.close(); }

  /** Large copy of this chart in a modal dialog, optionally zoomed to {lo, hi}. */
  openModal(zoom = null) {
    if (this._mirror) return;
    const dialog = document.createElement('dialog');
    dialog.className = 'chart-modal';
    dialog.setAttribute('aria-label', this.title || 'chart');
    const close = document.createElement('button');
    close.className = 'chart-modal-close'; close.textContent = '×';
    close.setAttribute('aria-label', 'close'); close.title = 'close (Esc)';
    const head = document.createElement('div'); head.className = 'chart-modal-head';
    const h = document.createElement('h2'); h.textContent = this.title;
    const hint = document.createElement('span'); hint.className = 'chart-modal-hint';
    hint.textContent = 'drag to zoom further · double-click or × for the full range · Esc to close';
    head.append(h, hint, close);
    const body = document.createElement('div');
    dialog.append(head, body);
    document.body.append(dialog);
    // the svg is drawn 1:1 in pixels, so its CSS height must equal the chart height
    const H = Math.round(clamp(window.innerHeight * 0.62, 260, 760));
    const big = new LineChart(body, { title: '', log: this.log, height: H, yMin: this.fixed.yMin, yMax: this.fixed.yMax, smooth: this.smooth, xFmt: this.xFmt, xName: this.xName, inModal: true });
    big.svg.style.height = `${H}px`;
    big.setSeries(this.series.map((s) => ({ ...s })));
    if (zoom) big.setZoom(zoom);
    this._mirror = { dialog, chart: big };
    close.addEventListener('click', () => dialog.close());
    // a click on the backdrop lands on the dialog element itself, outside its content box.
    // Ignore clicks in the first 300 ms: the drag that opened the dialog may still end in one.
    const openedAt = performance.now();
    dialog.addEventListener('click', (e) => {
      if (e.target !== dialog || performance.now() - openedAt < 300) return;
      const r = dialog.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
    });
    dialog.addEventListener('close', () => {
      big.destroy(); dialog.remove(); this._mirror = null;
      document.documentElement.classList.remove('chart-modal-open');
    });
    document.documentElement.classList.add('chart-modal-open');
    dialog.showModal();
    close.focus();
  }

  setZoom(z) {
    this.zoom = z;
    this.unzoomBtn.hidden = !z;
    this.retarget();
  }

  /** data x under a clientX */
  _dataX(clientX) {
    const rect = this.svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * this.W;
    return remap(clamp(px, this.pad.l, this.W - this.pad.r), this.pad.l, this.W - this.pad.r, this.xlo, this.xhi);
  }

  _bindZoom() {
    let drag = null;
    const band = () => {
      this.gSel.replaceChildren();
      if (!drag) return;
      const a = this.sx(Math.min(drag.x0, drag.x1)), b = this.sx(Math.max(drag.x0, drag.x1));
      this.gSel.append(svgEl('rect', { x: a, y: this.pad.t, width: Math.max(0, b - a), height: this.height - this.pad.t - this.pad.b, class: 'zoom-band' }));
    };
    this.svg.addEventListener('pointerdown', (e) => {
      // inline, touch keeps page scroll; in the modal there is nothing to scroll, so touch zooms too
      if ((e.pointerType === 'touch' && !this.inModal) || e.button !== 0) return;
      drag = { x0: this._dataX(e.clientX), x1: this._dataX(e.clientX), cx: e.clientX };
      this.svg.setPointerCapture(e.pointerId);
    });
    this.svg.addEventListener('pointermove', (e) => { if (drag) { drag.x1 = this._dataX(e.clientX); band(); } });
    const end = (e) => {
      if (!drag) return;
      const d = drag; drag = null; band();
      if (e.type === 'pointerup' && Math.abs(e.clientX - d.cx) > 6) {
        const range = { lo: Math.min(d.x0, d.x1), hi: Math.max(d.x0, d.x1) };
        // inline: the selection opens the large view; the inline chart keeps its range
        if (this.inModal) this.setZoom(range); else this.openModal(range);
      }
    };
    this.svg.addEventListener('pointerup', end); this.svg.addEventListener('pointercancel', end);
    this.svg.addEventListener('dblclick', () => { if (this.zoom) this.setZoom(null); });
  }

  ty(v) { return this.log ? Math.log10(Math.max(v, 1e-9)) : v; }

  setSeries(series) {
    const prevIds = new Set(this.series.map((s) => s.id));
    this.series = series.filter((s) => s.points?.length);
    for (const s of this.series) s._new = !prevIds.has(s.id);
    // series with hideLegend (pinned-run overlays) are identified by colour elsewhere; hover still names them
    this.legend.replaceChildren(...this.series.filter((s) => !s.hideLegend).map((s) => {
      const b = document.createElement('span'); b.className = 'legend-item';
      b.innerHTML = `<i style="background:${s.color}"></i>${s.label}`; return b;
    }));
    this.retarget();
    this._mirror?.chart.setSeries(series.map((s) => ({ ...s })));   // live data reaches the large view too
  }

  retarget() {
    let lo = Infinity, hi = -Infinity, xlo = Infinity, xhi = -Infinity;
    const z = this.zoom;
    for (const s of this.series) for (const [x, y] of s.points) {
      if (!Number.isFinite(y)) continue;
      if (z && (x < z.lo || x > z.hi)) continue;   // y-range from what is in view
      const v = this.ty(y);
      if (v < lo) lo = v; if (v > hi) hi = v;
      if (x < xlo) xlo = x; if (x > xhi) xhi = x;
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (this.fixed.yMin != null) lo = this.ty(this.fixed.yMin);
    if (this.fixed.yMax != null) hi = this.ty(this.fixed.yMax);
    if (hi - lo < 1e-12) { hi = lo + 1; lo -= 1; }
    const m = (hi - lo) * 0.06;
    if (z) { xlo = z.lo; xhi = z.hi; }
    this.xlo = xlo; this.xhi = xhi === xlo ? xlo + 1 : xhi;
    this.targetLo = lo - m; this.targetHi = hi + m;
    if (this._first) { this.ylo.value = this.targetLo; this.yhi.value = this.targetHi; this._first = false; }
    this.animate();
  }

  animate() {
    if (this._raf) return;
    this._last = performance.now();
    const step = (now) => {
      const dt = Math.min(0.1, (now - this._last) / 1000); this._last = now;
      spring.damp(this.ylo, this.targetLo, 0.35, dt);
      spring.damp(this.yhi, this.targetHi, 0.35, dt);
      this.render();
      const settled = Math.abs(this.ylo.value - this.targetLo) < 1e-6 && Math.abs(this.yhi.value - this.targetHi) < 1e-6 && Math.abs(this.ylo.velocity) < 1e-6;
      this._raf = settled ? null : requestAnimationFrame(step);
      if (settled) { this.ylo.value = this.targetLo; this.yhi.value = this.targetHi; this.render(); }
    };
    this._raf = requestAnimationFrame(step);
  }

  sx(x) { return remap(x, this.xlo, this.xhi, this.pad.l, this.W - this.pad.r); }
  sy(v) { return remap(this.ty(v), this.ylo.value, this.yhi.value, this.height - this.pad.b, this.pad.t); }

  render() {
    if (!this.W) return;
    const { W, height: H, pad } = this;
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    Object.entries({ x: pad.l, y: 0, width: Math.max(0, W - pad.l - pad.r), height: H }).forEach(([k, v]) => this.clipRect.setAttribute(k, v));
    // grid
    this.gGrid.replaceChildren();
    const lo = this.ylo.value, hi = this.yhi.value;
    const ticks = this.log ? niceTicks(lo, hi, 3).map((t) => 10 ** t) : niceTicks(lo, hi, 4);
    for (const t of ticks) {
      const y = this.sy(t);
      if (y < pad.t - 1 || y > H - pad.b + 1) continue;
      this.gGrid.append(svgEl('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, class: 'grid-line' }));
      const label = svgEl('text', { x: pad.l - 6, y: y + 3, class: 'grid-label', 'text-anchor': 'end' });
      label.textContent = fmt(t); this.gGrid.append(label);
    }
    if (this.xhi > this.xlo) for (const t of niceTicks(this.xlo, this.xhi, 5)) {
      const x = this.sx(t);
      const label = svgEl('text', { x, y: H - 6, class: 'grid-label', 'text-anchor': 'middle' });
      label.textContent = this.xFmt(t); this.gGrid.append(label);
    }
    // lines
    this.gLines.replaceChildren();
    // Draw order: overlays (pinned runs, earlier evals: the series without a legend entry) first,
    // this run's own series last, so this run is on top and never hidden under a pinned run.
    const drawOrder = [...this.series.filter((s) => s.hideLegend), ...this.series.filter((s) => !s.hideLegend)];
    for (const s of drawOrder) {
      let d = '', pen = false;
      for (const [x, y] of s.points) {
        if (!Number.isFinite(y)) { pen = false; continue; }
        d += `${pen ? 'L' : 'M'}${this.sx(x).toFixed(1)},${clamp(this.sy(y), -50, H + 50).toFixed(1)}`; pen = true;
      }
      const p = svgEl('path', { d, class: 'series-line', style: `stroke:${s.color}${s.dash ? ';stroke-dasharray:4 3' : ''}${s.width ? `;stroke-width:${s.width}` : ''}` });
      if (s._new) { p.classList.add('draw-in'); s._new = false; }
      this.gLines.append(p);
      if (s.points.length <= 40 || s.dots) for (const [x, y] of s.points) {
        if (!Number.isFinite(y) || x < this.xlo || x > this.xhi) continue;
        this.gLines.append(svgEl('circle', { cx: this.sx(x), cy: this.sy(y), r: 2.2, style: `fill:${s.color}` }));
      }
    }
    if (this._hoverX != null) this.hover({ clientX: this._hoverX, clientY: this._hoverY });
  }

  hover(e) {
    this.gHover.replaceChildren();
    if (!e) { this._hoverX = null; this._hoverY = null; this.readout.textContent = ''; return; }
    this._hoverX = e.clientX; this._hoverY = e.clientY ?? null;
    const rect = this.svg.getBoundingClientRect();
    // horizontal guide at the pointer's height, with its y value on the axis: read a level
    // across the whole chart, the way the vertical line reads a step
    if (this._hoverY != null && rect.height) {
      const py = ((this._hoverY - rect.top) / rect.height) * this.height;
      if (py >= this.pad.t && py <= this.height - this.pad.b) {
        this.gHover.append(svgEl('line', { x1: this.pad.l, x2: this.W - this.pad.r, y1: py, y2: py, class: 'hover-line' }));
        const v = remap(py, this.height - this.pad.b, this.pad.t, this.ylo.value, this.yhi.value);
        // at the right end, with two more decimals than the axis labels
        const label = svgEl('text', { x: this.W - this.pad.r - 4, y: py - 3, class: 'grid-label hover-y', 'text-anchor': 'end' });
        label.textContent = fmt(this.log ? 10 ** v : v, 2);
        this.gHover.append(label);
      }
    }
    const px = ((e.clientX - rect.left) / rect.width) * this.W;
    const x = remap(px, this.pad.l, this.W - this.pad.r, this.xlo, this.xhi);
    const parts = [];
    let xs = null;
    for (const s of this.series) {
      let best = null, bd = Infinity;
      for (const p of s.points) { if (p[0] < this.xlo || p[0] > this.xhi) continue; const d = Math.abs(p[0] - x); if (d < bd) { bd = d; best = p; } }
      if (!best) continue;
      xs = best[0];
      parts.push(`<i style="background:${s.color}"></i>${s.label} <b>${fmt(best[1])}</b>`);
      this.gHover.append(svgEl('circle', { cx: this.sx(best[0]), cy: this.sy(best[1]), r: 3.5, class: 'hover-dot', style: `stroke:${s.color}` }));
    }
    if (xs != null) {
      const hx = this.sx(xs);
      this.gHover.prepend(svgEl('line', { x1: hx, x2: hx, y1: this.pad.t, y2: this.height - this.pad.b, class: 'hover-line' }));
      this.readout.innerHTML = `<span class="muted">${this.xName} ${this.xFmt(xs)}</span> ${parts.join(' ')}`;
    }
  }
}

/** Tiny inline sparkline path (no axes) for cards. */
export function sparklinePath(points, W = 120, H = 28, log = true) {
  const ys = points.map(([, y]) => (log ? Math.log10(Math.max(y, 1e-9)) : y)).filter(Number.isFinite);
  if (ys.length < 2) return '';
  const lo = Math.min(...ys), hi = Math.max(...ys);
  return points.map(([, y], i) => {
    const v = log ? Math.log10(Math.max(y, 1e-9)) : y;
    const x = (i / (points.length - 1)) * W;
    const yy = hi > lo ? H - 2 - ((v - lo) / (hi - lo)) * (H - 4) : H / 2;
    return `${i ? 'L' : 'M'}${x.toFixed(1)},${yy.toFixed(1)}`;
  }).join('');
}

export { easing, fmt, fmtX, MUTED };
