// GPU telemetry: ring gauges in the top bar plus a larger panel on the Live
// view. Values are spring-damped (math/time) so the 2 s samples read as a
// continuous signal instead of jumping.

import { spring, easing } from 'math/time';
import { clamp, remapClamp } from 'math';
import { ACCENT, ACCENT2, DANGER, OK, css } from './palette.js';

const gauges = new Map(); // key -> {state, el, ...}

function ring(key, label, unit, color, max) {
  const el = document.createElement('div');
  el.className = 'gauge';
  el.innerHTML = `
    <svg viewBox="0 0 44 44" class="gauge-ring">
      <circle cx="22" cy="22" r="18" class="gauge-track"/>
      <circle cx="22" cy="22" r="18" class="gauge-arc" style="stroke:${color}"/>
    </svg>
    <div class="gauge-text"><b class="gauge-value">–</b><span class="gauge-label">${label}</span></div>`;
  const arc = el.querySelector('.gauge-arc');
  const value = el.querySelector('.gauge-value');
  const circ = 2 * Math.PI * 18;
  arc.style.strokeDasharray = `${circ}`;
  arc.style.strokeDashoffset = `${circ}`;
  const g = { key, label, unit, color, max, el, arc, value, circ, state: spring.create(0), target: 0, text: '–' };
  gauges.set(key, g);
  return g;
}

export function mountStrip(container) {
  container.replaceChildren(
    ring('util', 'util', '%', ACCENT2, 100).el,
    ring('mem', 'mem', 'GiB', ACCENT, 1).el,
    ring('temp', 'temp', '°C', OK, 95).el,
    ring('power', 'pwr', 'W', css(300, 0.7, 0.66), 1).el,
  );
}

let latest = null;
export function update(sample) {
  latest = sample;
  const util = gauges.get('util'), mem = gauges.get('mem'), temp = gauges.get('temp'), pwr = gauges.get('power');
  if (!util) return;
  util.target = (sample.util ?? 0) / 100; util.text = sample.util == null ? '–' : `${Math.round(sample.util)}%`;
  const memGiB = (sample.memUsed ?? 0) / 1024;
  const memTot = sample.memTotal ? sample.memTotal / 1024 : 128; // GB10: 128 GB unified
  mem.target = memGiB / memTot; mem.text = `${memGiB.toFixed(1)}G`;
  temp.target = (sample.temp ?? 0) / 95; temp.text = sample.temp == null ? '–' : `${Math.round(sample.temp)}°`;
  temp.arc.style.stroke = sample.temp > 80 ? DANGER : sample.temp > 70 ? ACCENT : OK;
  const lim = sample.powerLimit || 140;
  pwr.target = (sample.power ?? 0) / lim; pwr.text = sample.power == null ? '–' : `${Math.round(sample.power)}W`;
}

export function latestSample() { return latest; }

let lastT = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  for (const g of gauges.values()) {
    spring.damp(g.state, clamp(g.target, 0, 1), 0.5, dt);
    g.arc.style.strokeDashoffset = `${g.circ * (1 - g.state.value)}`;
    if (g.value.textContent !== g.text) g.value.textContent = g.text;
  }
}
requestAnimationFrame(tick);

// ---- Live-section instrument cluster --------------------------------------
// Two speedometers (util, memory) each with a filled history strip beneath,
// a zoned temperature dial, and power as a small stat. Built once, updated
// in place, spring-damped like the strip gauges.

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
const polar = (cx, cy, r, deg) => [cx + r * Math.cos((deg * Math.PI) / 180), cy + r * Math.sin((deg * Math.PI) / 180)];
/** SVG arc path from angle a0 to a1 (degrees, clockwise, screen coords). */
function arcPath(cx, cy, r, a0, a1) {
  const [x0, y0] = polar(cx, cy, r, a0), [x1, y1] = polar(cx, cy, r, a1);
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M${x0.toFixed(2)},${y0.toFixed(2)} A${r},${r} 0 ${large} 1 ${x1.toFixed(2)},${y1.toFixed(2)}`;
}
const A0 = 135, SWEEP = 270; // speedometer: 7:30 → 4:30 o'clock

class Speedo {
  constructor({ label, unit, color, max, zones = null, strip = true }) {
    this.max = max; this.color = color; this.unit = unit; this.zones = zones;
    this.state = spring.create(0); this.target = 0;
    const W = 96, cx = 48, cy = 50, r = 38;
    this.el = document.createElement('div'); this.el.className = 'speedo';
    const svg = svgEl('svg', { viewBox: `0 0 ${W} 92`, class: 'speedo-svg' });
    if (zones) {
      // background split into coloured zones (fractions of max)
      let a = A0;
      for (const z of zones) { const a1 = A0 + SWEEP * (z.to / max); svg.append(svgEl('path', { d: arcPath(cx, cy, r, a, a1), class: 'speedo-zone', style: `stroke:${z.color}` })); a = a1; }
    } else svg.append(svgEl('path', { d: arcPath(cx, cy, r, A0, A0 + SWEEP), class: 'speedo-track' }));
    this.arc = svgEl('path', { d: arcPath(cx, cy, r, A0, A0 + SWEEP), class: 'speedo-arc', style: `stroke:${zones ? 'transparent' : color}` });
    this.len = (SWEEP / 360) * 2 * Math.PI * r;
    this.arc.style.strokeDasharray = `${this.len}`; this.arc.style.strokeDashoffset = `${this.len}`;
    this.marker = svgEl('circle', { r: 4, class: 'speedo-marker', style: `fill:${color}` });
    // ticks
    for (let i = 0; i <= 4; i++) { const a = A0 + (SWEEP * i) / 4; const [x0, y0] = polar(cx, cy, r + 6, a), [x1, y1] = polar(cx, cy, r + 9, a); svg.append(svgEl('line', { x1: x0, y1: y0, x2: x1, y2: y1, class: 'speedo-tick' })); }
    this.value = svgEl('text', { x: cx, y: cy + 4, class: 'speedo-value', 'text-anchor': 'middle' });
    this.unitEl = svgEl('text', { x: cx, y: cy + 18, class: 'speedo-unit', 'text-anchor': 'middle' }); this.unitEl.textContent = unit;
    this.labelEl = svgEl('text', { x: cx, y: 90, class: 'speedo-label', 'text-anchor': 'middle' }); this.labelEl.textContent = label;
    svg.append(this.arc, this.marker, this.value, this.unitEl, this.labelEl);
    this.el.append(svg);
    this.cx = cx; this.cy = cy; this.r = r;
    if (strip) { this.strip = svgEl('svg', { viewBox: '0 0 96 22', class: 'speedo-strip', preserveAspectRatio: 'none' }); this.area = svgEl('path', { class: 'strip-area', style: `fill:${color}` }); this.line = svgEl('path', { class: 'strip-line', style: `stroke:${color}` }); this.strip.append(this.area, this.line); this.el.append(this.strip); }
    this.text = '–';
  }
  set(v, text) { this.target = clamp((v ?? 0) / this.max, 0, 1); this.text = text; }
  history(vals) {
    if (!this.strip || !vals.length) return;
    const n = vals.length, W = 96, H = 22;
    const pts = vals.map((v, i) => [((i / Math.max(1, n - 1)) * W).toFixed(1), (H - 1 - clamp((v ?? 0) / this.max, 0, 1) * (H - 2)).toFixed(1)]);
    const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x},${y}`).join('');
    this.line.setAttribute('d', d);
    this.area.setAttribute('d', `${d} L${W},${H} L0,${H} Z`);
  }
  tick(dt) {
    spring.damp(this.state, this.target, 0.5, dt);
    const f = this.state.value;
    this.arc.style.strokeDashoffset = `${this.len * (1 - f)}`;
    const [mx, my] = polar(this.cx, this.cy, this.r, A0 + SWEEP * f);
    this.marker.setAttribute('cx', mx.toFixed(2)); this.marker.setAttribute('cy', my.toFixed(2));
    if (this.value.textContent !== this.text) this.value.textContent = this.text;
  }
}

export class GpuPanel {
  constructor(el) {
    this.el = el; el.classList.add('gpu-cluster');
    this.util = new Speedo({ label: 'utilisation', unit: '%', color: ACCENT2, max: 100 });
    this.mem = new Speedo({ label: 'memory', unit: 'GiB', color: ACCENT, max: 128 });
    this.temp = new Speedo({ label: 'temperature', unit: '°C', color: '#fff', max: 100, strip: false, zones: [{ to: 60, color: OK }, { to: 78, color: ACCENT }, { to: 100, color: DANGER }] });
    this.head = document.createElement('div'); this.head.className = 'gpu-cluster-head';
    this.stats = document.createElement('div'); this.stats.className = 'gpu-cluster-stats';
    el.replaceChildren(this.head, Object.assign(document.createElement('div'), { className: 'gpu-cluster-dials' }), this.stats);
    el.children[1].append(this.util.el, this.mem.el, this.temp.el);
    this._last = performance.now(); this._alive = true;
    const loop = (now) => { if (!this._alive) return; requestAnimationFrame(loop); const dt = Math.min(0.1, (now - this._last) / 1000); this._last = now; for (const g of [this.util, this.mem, this.temp]) g.tick(dt); };
    requestAnimationFrame(loop);
  }
  destroy() { this._alive = false; }
  /** Append one live sample (from the SSE stream) and redraw. */
  push(sample) {
    this._hist = this._hist || [];
    this._hist.push(sample); if (this._hist.length > 300) this._hist.shift();
    this.update(this._hist);
  }
  update(history) {
    this._hist = history ? [...history] : this._hist;
    if (!history?.length) { this.head.innerHTML = '<span class="muted">No GPU samples yet.</span>'; return; }
    const last = history.at(-1);
    const memTot = last.memTotal ? last.memTotal / 1024 : 128;
    this.mem.max = memTot; this.mem.unitEl.textContent = `of ${Math.round(memTot)} GiB`;
    this.util.set(last.util, last.util == null ? '–' : `${Math.round(last.util)}`);
    this.mem.set((last.memUsed ?? 0) / 1024, ((last.memUsed ?? 0) / 1024).toFixed(1));
    this.temp.set(last.temp, last.temp == null ? '–' : `${Math.round(last.temp)}`);
    this.temp.marker.style.fill = last.temp > 78 ? DANGER : last.temp > 60 ? ACCENT : OK;
    this.util.history(history.map((s) => s.util));
    this.mem.history(history.map((s) => (s.memUsed ?? 0) / 1024));
    const mins = Math.round(((last.t - history[0].t) / 60000) * 10) / 10;
    this.head.replaceChildren(
      Object.assign(document.createElement('span'), { className: 'eyebrow', textContent: 'GPU' }),
      Object.assign(document.createElement('h3'), { textContent: last.name || 'GPU' }),
      Object.assign(document.createElement('span'), { className: 'muted mono', style: 'font-size:11px;margin-left:auto', title: 'the history strips under the dials span this long; values refresh every 2 s', textContent: `strips: ${mins} min` }));
    const procs = (last.procs || []).map((p) => `<span class="chip">${p.name} <b>${p.pid}</b>${p.memMiB != null ? ` ${(p.memMiB / 1024).toFixed(1)} GiB` : ''}</span>`).join('');
    this.stats.innerHTML = `<span class="chip">power <b>${last.power == null ? '–' : Math.round(last.power) + ' W'}</b></span><span class="chip">sm clock <b>${last.clockSm == null ? '–' : last.clockSm + ' MHz'}</b></span>${procs || '<span class="chip muted">no compute processes</span>'}`;
  }
}

/** Big sparkline panel (legacy): util + power over the last N samples. */
export function renderHistory(el, history) {
  if (!history?.length) { el.innerHTML = '<p class="muted">No GPU samples yet.</p>'; return; }
  const W = 600, H = 120, pad = 6;
  const n = history.length;
  const path = (key, max) => history.map((s, i) => {
    const x = pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
    const y = H - pad - remapClamp(s[key] ?? 0, 0, max, 0, H - 2 * pad);
    return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const last = history[n - 1];
  const procs = (last.procs || []).map((p) => `<span class="chip">${p.name} <b>${p.pid}</b> ${p.memMiB != null ? (p.memMiB / 1024).toFixed(1) + ' GiB' : ''}</span>`).join('');
  el.innerHTML = `
    <div class="gpu-hist-head">
      <div><span class="eyebrow">GPU</span><h3>${last.name || 'GPU'}</h3></div>
      <div class="gpu-hist-legend"><i style="background:${ACCENT2}"></i>util <i style="background:${ACCENT}"></i>power <i style="background:${OK}"></i>temp</div>
    </div>
    <svg viewBox="0 0 ${W} ${H}" class="gpu-hist" preserveAspectRatio="none">
      <path d="${path('util', 100)}" style="stroke:${ACCENT2}"/>
      <path d="${path('power', last.powerLimit || 140)}" style="stroke:${ACCENT}"/>
      <path d="${path('temp', 100)}" style="stroke:${OK}"/>
    </svg>
    <div class="gpu-procs">${procs || '<span class="muted">no compute processes</span>'}</div>`;
}

export { easing };
