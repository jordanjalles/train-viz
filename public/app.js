// train-viz frontend shell: data store, SSE live updates, hash routing, and
// the views (Runs with the live hero, run Details, Latent). Rendering is plain
// DOM; charts, the latent point cloud, covariances and mobile gestures are own modules.

import * as gpu from './gpu.js';
import { LineChart, sparklinePath, fmt, fmtX } from './charts.js';
import { LatentView, fetchPca } from './constellation.js';
import { fetchCov, drawCov } from './covariance.js';
import { seriesColor, depthColor, NEUTRAL, PIN_COLORS, ROUTE, ACCENT } from './palette.js';
import { zoomable, fullscreenStage, isTouch } from './mobile.js';
import { viewExperiments, viewExperiment } from './experiments.js';
import { viewReports } from './reports.js';
import { clamp } from 'math';

// ---------------------------------------------------------------- state
const state = {
  roots: [], runs: [], byId: new Map(), full: new Map(),
  pinned: new Set(JSON.parse(localStorage.getItem('tv.pinned') || '[]')),
  filter: { root: null, status: null },
  // run list controls (per viewer, remembered): search text, sort, "has latents", show archived
  list: (() => { try { return { q: '', sort: 'newest', latents: false, archived: false, ...JSON.parse(localStorage.getItem('tv.list') || '{}') }; } catch { return { q: '', sort: 'newest', latents: false, archived: false }; } })(),
  // curation shared by every viewer (server state/run-state.json): kept (★) and archived runs
  kept: new Set(), archived: new Set(),
  view: null, cleanup: [], runHead: null,
};
// A smoke test: `smoke` as a word of the run name (smoke-x, x-smoke--job, smoke2_x), or config.smoke. Hidden from "All runs" by default.
export const isSmoke = (r) => /(^|[-_])smoke\d*([-_]|$)/i.test(r.name || '') || r.config?.smoke === true;
const saveListPrefs = () => { try { localStorage.setItem('tv.list', JSON.stringify(state.list)); } catch { /* ignore */ } };
function setRunState(s) { state.kept = new Set(s?.kept || []); state.archived = new Set(s?.archived || []); }
/** Keep / unkeep / archive / unarchive on the server; the SSE 'state' event updates every viewer. */
async function curate(op, ids) {
  const r = await fetch('/api/state', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ op, ids }) });
  const d = await r.json();
  if (!r.ok) { toast(d.error || 'failed'); return null; }
  setRunState(d.state);
  for (const l of listeners) l('run', null);
  return d;
}
// Routes: "#" → Runs (top level: live, pinned, all). "#run/<id>/<details|latent>" → one run.
const runHash = (id, sub = 'details') => `#run/${encodeURIComponent(id)}/${sub}`;
const $ = (sel, root = document) => root.querySelector(sel);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
};
const html = (s) => { const t = document.createElement('template'); t.innerHTML = s.trim(); return t.content; };
const toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2600); };
const ago = (t) => { if (!t) return '–'; const s = (Date.now() - t) / 1000; return s < 60 ? `${Math.round(s)}s ago` : s < 3600 ? `${Math.round(s / 60)}m ago` : s < 86400 ? `${(s / 3600).toFixed(1)}h ago` : `${(s / 86400).toFixed(1)}d ago`; };
// Run names: notebook runs are "<YYYY-MM-DD>_<HHMMSS>_<micro>_<title>", and runs mirrored from
// the 3090 add "--<job id>". The title is shown alone; the start time goes in a subheader.
const NAME_RE = /^(?:(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})(?:_\d+)?_)?(.+?)(?:--((?:nb-)?[0-9a-f]{20,}))?$/;
function parseRunName(r) {
  const m = NAME_RE.exec(r.name) || [];
  let when = null;
  if (m[1]) {
    const p = m.slice(1, 7).map(Number);
    // the stamp is the training machine's local time; a run may declare it was UTC (config.name_utc)
    when = r.config?.name_utc ? new Date(Date.UTC(p[0], p[1] - 1, p[2], p[3], p[4], p[5])) : new Date(p[0], p[1] - 1, p[2], p[3], p[4], p[5]);
  } else if (r.startedAt) when = new Date(r.startedAt);
  return { title: m[7] || r.name, when, job: m[8] || null };
}
const runTitle = (r) => parseRunName(r).title;
/** "Fri, Oct 2 · 3:15 PM" (with the year when it is not this year). */
function runWhen(r) {
  const { when } = parseRunName(r);
  if (!when || Number.isNaN(when.getTime())) return '';
  const sameYear = when.getFullYear() === new Date().getFullYear();
  const day = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric', ...(sameYear ? {} : { year: 'numeric' }) }).format(when);
  const time = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(when);
  return `${day} · ${time}`;
}
const runTip = (r) => [runWhen(r), r.name].filter(Boolean).join('\n');   // hover: time and the full id
const shortName = (r) => { const t = runTitle(r); return t.length > 42 ? t.slice(0, 40) + '…' : t; };
// per-viewer preferences (numbers), kept in localStorage
const pref = (k, d) => { const v = parseFloat(localStorage.getItem(k)); return Number.isFinite(v) ? v : d; };
const setPref = (k, v) => localStorage.setItem(k, String(v));
// Orbit camera (yaw / pitch / zoom / auto-rotate) survives run jumps and reloads.
function saveCamera(v, key) {
  try { localStorage.setItem(key, JSON.stringify({ yaw: v.yawT, pitch: v.pitchT, dist: v.distT, auto: v.autoRotate })); } catch { /* ignore */ }
}
function restoreCamera(v, key) {
  try {
    const c = JSON.parse(localStorage.getItem(key) || 'null'); if (!c) return;
    v.yawT = c.yaw; v.pitchT = c.pitch; v.distT = c.dist; v.autoRotate = !!c.auto;
    v.yaw.value = c.yaw; v.pitch.value = c.pitch; v.dist.value = c.dist; // start there, no fly-in
    v.idleSince = performance.now();
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------- data
// The run list is large (~0.6 MB gzipped). Fetch it once; the event stream keeps it current after
// that. A dropped stream may have missed events, so the next view fetches the list again.
let runsFresh = false;
async function loadRuns() {
  if (runsFresh && state.runs?.length) return;
  const r = await (await fetch('/api/runs')).json();
  runsFresh = true;
  state.roots = r.roots; state.runs = r.runs;
  state.byId = new Map(r.runs.map((x) => [x.id, x]));
  setRunState(r.state);
}
async function loadFull(id) {
  if (state.full.has(id)) return state.full.get(id);
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error('run not found');
  const f = await r.json(); state.full.set(id, f); return f;
}
/** Same rule as lib/runs.js appendRecords: a repeated or earlier step means the log was rewritten
 *  from there (resumed run), so drop everything from that step before appending. */
function appendRecords(arr, fresh) {
  for (const r of fresh) {
    if (typeof r.step === 'number' && arr.length && r.step === arr.at(-1).step) { arr[arr.length - 1] = { ...arr.at(-1), ...r }; continue; }   // same step: more fields of it
    if (typeof r.step === 'number' && arr.length && r.step < arr.at(-1).step) {
      let i = arr.length;
      while (i > 0 && arr[i - 1].step >= r.step) i--;
      arr.length = i;
    }
    arr.push(r);
  }
  return arr;
}
function upsertSummary(s) {
  const i = state.runs.findIndex((x) => x.id === s.id);
  if (i >= 0) state.runs[i] = s; else state.runs.unshift(s);
  state.byId.set(s.id, s);
  const f = state.full.get(s.id);
  if (f) Object.assign(f, { status: s.status, recons: s.recons, emaRecons: s.emaRecons, zSteps: s.zSteps, robustPngs: s.robustPngs, images: s.images, updatedAt: s.updatedAt, description: s.description });
}

const listeners = new Set(); // view refreshers: fn(kind, payload)
function connectSSE() {
  const es = new EventSource('/api/events');
  es.addEventListener('error', () => { runsFresh = false; });
  es.addEventListener('gpu', (e) => {
    const s = JSON.parse(e.data); gpu.update(s);
    for (const l of listeners) l('gpu', s);
  });
  es.addEventListener('run', (e) => { upsertSummary(JSON.parse(e.data)); for (const l of listeners) l('run', null); });
  // a continuation folded into its first run: it is no longer a run of its own
  es.addEventListener('run-removed', (e) => {
    const { id } = JSON.parse(e.data);
    state.runs = state.runs.filter((x) => x.id !== id); state.byId.delete(id);
    for (const l of listeners) l('run', null);
  });
  es.addEventListener('state', (e) => { setRunState(JSON.parse(e.data)); for (const l of listeners) l('run', null); });
  es.addEventListener('records', (e) => {
    const d = JSON.parse(e.data);
    const s = state.byId.get(d.id);
    if (s) {
      const last = d.metrics.at(-1); if (last) { s.last = last; s.step = last.step; }
      const ev = d.metrics.findLast((m) => 'test_clean' in m); if (ev) s.lastEval = ev;
      const sp = d.metrics.findLast((m) => typeof m.sps === 'number'); if (sp) s.sps = sp.sps;
      s.status = d.status; s.updatedAt = d.updatedAt; s.nMetrics += d.metrics.length;
      if (last && typeof last.clean === 'number') {
        while (s.spark.length && s.spark.at(-1)[0] >= last.step) s.spark.pop();   // rewritten log (resume)
        s.spark.push([last.step, last.clean]); if (s.spark.length > 60) s.spark.splice(0, s.spark.length - 60); }
    }
    const f = state.full.get(d.id);
    if (f) { appendRecords(f.metrics, d.metrics); appendRecords(f.diag, d.diag); appendRecords(f.robust ||= [], d.robust || []); f.status = d.status; }
    for (const l of listeners) l('records', d);
  });
  es.onerror = () => { /* EventSource auto-reconnects */ };
}

// ---------------------------------------------------------------- health
function health(r) { return healthOf(r.lastEval || r.last || {}, r.status, r.step); }
function healthOf(ev, status, step) {
  const out = [];
  if (status === 'diverged' || ev.nan) out.push(['warn', 'NaN']);
  if (typeof ev.recon_div === 'number') out.push(ev.recon_div < 0.15 ? ['warn', `decoder collapse · div ${fmt(ev.recon_div)}`] : ['ok', `image-dependent · div ${fmt(ev.recon_div)}`]);
  if (typeof ev.z_eff_rank === 'number') out.push(ev.z_eff_rank < 40 ? ['warn', `rank collapse · ${Math.round(ev.z_eff_rank)}`] : ['info', `eff rank ${Math.round(ev.z_eff_rank)}`]);
  if (typeof ev.z_dead_frac === 'number' && ev.z_dead_frac > 0.5 && step > 0) out.push(['warn', `dead dims ${Math.round(ev.z_dead_frac * 100)}%`]);
  return out;
}
const toBadges = (list) => list.map(([k, t]) => h('span', { class: `badge ${k}` }, t));
const badges = (r) => toBadges(health(r));
/** Failure-reason badges for a specific step: the eval record at (or just before) that step. */
function badgesAtStep(full, step) {
  const evals = full.metrics.filter((m) => 'test_clean' in m || m.nan);
  let ev = null;
  for (const m of evals) if (m.step <= step) ev = m;
  if (!ev) return [h('span', { class: 'badge info' }, 'no eval at this step')];
  const list = healthOf(ev, full.status, ev.step);
  if (!list.some(([k]) => k === 'warn')) list.unshift(['ok', 'healthy']);
  return toBadges(list);
}

// Epochs: STL10 train+unlabeled is 105,000 images; harness runs use batch 64 with drop_last,
// notebook runs record their own batch_size in the config.
const TRAIN_IMAGES = 105000;
function epochLabel(r) {
  const bs = r.config?.batch_size || 64;
  const perEpoch = Math.floor(TRAIN_IMAGES / bs);
  const ep = (r.step || 0) / perEpoch;
  const tot = r.totalSteps ? ` / ${(r.totalSteps / perEpoch).toFixed(0)}` : '';
  return h('div', { class: 'epoch-sub mono', title: `${perEpoch.toLocaleString()} steps per epoch (batch ${bs})` }, `epoch ${ep.toFixed(1)}${tot}`);
}

// ---------------------------------------------------------------- router
const SUBS = ['details', 'latent'];
function route() {
  const parts = location.hash.slice(1).split('/');
  // legacy hashes from earlier versions
  const legacy = { runs: 'details', latent: 'latent', pca: 'latent' }[parts[0]];
  if (legacy && parts.length > 1) { location.replace(runHash(decodeURIComponent(parts.slice(1).join('/')), legacy)); return; }
  for (const c of state.cleanup) c(); state.cleanup = []; listeners.clear(); state.runHead = null;
  const el = $('#view'); el.replaceChildren(); el.scrollTop = 0;
  if (parts[0] === 'run' && parts.length >= 2) {
    // the Latent tab used to be called Constellation
    if (parts.at(-1) === 'constellation') { location.replace(runHash(decodeURIComponent(parts.slice(1, -1).join('/')), 'latent')); return; }
    const sub = SUBS.includes(parts.at(-1)) ? parts.at(-1) : 'details';
    const id = decodeURIComponent(parts.slice(1, SUBS.includes(parts.at(-1)) ? -1 : undefined).join('/'));
    viewRunPage(el, id, sub);
  } else if (parts[0] === 'experiments') { setNav(topTabs('experiments')); viewExperiments(el); }
  else if (parts[0] === 'experiment' && parts.length >= 2) { setNav(topTabs('experiments')); viewExperiment(el, decodeURIComponent(parts.slice(1).join('/')), state.cleanup); }
  else if (parts[0] === 'reports') { setNav(topTabs('reports')); viewReports(el); }
  else viewRuns(el);
}
window.addEventListener('hashchange', route);

// ---------------------------------------------------------------- card
/** Identity colour for a pinned run, used ONLY where lines/chips of several runs share a chart. */
const runColor = (id) => { const i = [...state.pinned].indexOf(id); return i < 0 ? NEUTRAL : PIN_COLORS[i % PIN_COLORS.length]; };
/** Cards and headers carry no run colour at all. */
const runTint = () => NEUTRAL;
/** Footer of a run: image size and module shapes from config.hparams (user, 2026-10-09: shapes belong here, not in the run name). */
export function shapeFooter(r) {
  const c = r.config || {}, hp = c.hparams || {};
  if (!String(c.run_type || '').startsWith('pixel')) return '';
  const parts = [`${hp.IMAGE_SIZE ?? 128} px`, `${hp.num_blocks ?? 8} blocks`, `width ${hp.emb_dim ?? 256}`, `${hp.n_heads ?? 8} heads`];
  if (c.n_params) parts.push(`${(c.n_params / 1e6).toFixed(1)}M`);
  if (c.dataset && c.dataset !== 'imagenet') parts.push(c.dataset);
  return parts.join(' · ');
}
function runCard(r, { compact = false } = {}) {
  const col = runTint(r.id);
  const pct = r.totalSteps ? (100 * (r.step || 0)) / r.totalSteps : 0;
  const last = r.last || {};
  const card = h('div', { class: `card ${state.pinned.has(r.id) ? 'pinned' : ''} ${state.archived.has(r.id) ? 'archived' : ''}`, style: `--run-color:${col}`, onclick: () => (location.hash = runHash(r.id)) },
    h('div', { class: 'row spread' }, h('span', { class: `status ${r.status}` }, r.status), h('span', { class: 'muted mono', style: 'font-size:11px' }, ago(r.updatedAt))),
    h('div', { class: 'card-title', title: runTip(r) }, shortName(r)),
    runWhen(r) ? h('div', { class: 'run-when' }, runWhen(r)) : null,
    r.description ? h('div', { class: 'card-desc', title: r.description }, r.description) : null,
    h('div', { class: 'card-meta' }, h('span', {}, state.roots.find((x) => x.id === r.root)?.label || r.root), r.kind === 'harness' ? h('span', {}, `${fmtX(r.step || 0)} / ${r.totalSteps ? fmtX(r.totalSteps) : '?'}`) : null),
  );
  if (r.kind === 'harness') {
    card.append(
      h('div', { class: 'progress' }, h('i', { style: `width:${pct}%` })),
      html(`<svg class="card-spark" viewBox="0 0 120 28" preserveAspectRatio="none"><path d="${sparklinePath(r.spark || [])}"/></svg>`),
      h('div', { class: 'card-stats' },
        stat(fmt(r.lastEval?.test_clean), 'test mse'), stat(fmt(last.view), 'view'), stat(fmt(r.lastEval?.z_eff_rank), 'eff rank')),
      h('div', { class: 'row' }, ...badges(r)),
    );
  } else {
    const c = r.config || {};
    card.append(h('div', { class: 'card-stats' }, stat(c.lr ?? '–', 'lr'), stat(c.ae_emb_dim ?? '–', 'width'), stat(c.num_blocks ?? '–', 'blocks')),
      h('div', { class: 'row' }, h('span', { class: 'badge info' }, r.images?.length ? `${r.images.length} live plots` : 'metadata only')));
  }
  if (shapeFooter(r)) card.append(h('div', { class: 'card-foot', title: 'image size and module shapes (config.hparams)' }, shapeFooter(r)));
  if (!compact) {
    const pin = h('button', { class: `pin ${state.pinned.has(r.id) ? 'on' : ''}`, title: 'pin for comparison', onclick: (e) => { e.stopPropagation(); togglePin(r.id); pin.classList.toggle('on'); card.classList.toggle('pinned'); card.style.setProperty('--run-color', runTint(r.id)); } }, '◆');
    card.append(pin, ...curationButtons(r.id, 'card-act'));
  }
  return card;
}
const stat = (v, l) => h('div', { class: 'stat' }, h('b', {}, v ?? '–'), h('span', {}, l));
/** Keep (★) and archive (⊘ / restore) buttons for a run; `cls` styles them for a card or a header. */
function curationButtons(id, cls) {
  const kept = state.kept.has(id), archived = state.archived.has(id);
  const stop = (fn) => (e) => { e.stopPropagation(); e.preventDefault(); fn(); };
  const keep = h('button', { class: `${cls} keep ${kept ? 'on' : ''}`, title: kept ? 'kept: shown at the top and never bulk-archived (click to un-keep)' : 'keep this run', 'aria-label': kept ? 'un-keep' : 'keep',
    onclick: stop(() => curate(kept ? 'unkeep' : 'keep', [id])) }, kept ? '★' : '☆');
  const arch = h('button', { class: `${cls} arch ${archived ? 'on' : ''}`, title: archived ? 'archived: hidden from the default list (click to restore)' : 'archive: hide from the default list (nothing is deleted)', 'aria-label': archived ? 'restore' : 'archive',
    onclick: stop(async () => {
      if (kept && !archived) { toast('kept runs are not archived; un-keep it first'); return; }
      const d = await curate(archived ? 'unarchive' : 'archive', [id]);
      if (d && !archived) toast('archived; "show archived" brings it back');
    }) }, archived ? '↺' : '⊘');
  return [arch, keep];
}
function togglePin(id) {
  if (state.pinned.has(id)) state.pinned.delete(id); else state.pinned.add(id);
  localStorage.setItem('tv.pinned', JSON.stringify([...state.pinned]));
}

// ---------------------------------------------------------------- Live section (top of Runs)
async function renderLiveSection(el) {
  const live = state.runs.filter((r) => r.status === 'live');
  const hero = live[0] || null;
  const gpuEl = h('div', { class: 'panel' });
  const gpuPanel = new gpu.GpuPanel(gpuEl);
  state.cleanup.push(() => gpuPanel.destroy());
  // Every live run gets a full panel with charts. The first sits beside the GPU panel.
  const heroes = live.map((r) => liveRunPanel(r));
  const latentPanel = h('div', { class: 'panel mini-latent' });
  const showLatent = !!hero?.zSteps?.length;
  el.append(
    h('div', { class: 'section-head' }, h('h2', {}, 'Live'), h('span', { class: 'muted' }, live.length ? `${live.length} training now` : 'nothing training right now')),
    hero ? h('div', { class: 'hero3' }, heroes[0].panel, h('div', { class: 'hero-side' }, gpuEl, showLatent ? latentPanel : null)) : gpuEl,
    ...heroes.slice(1).map((x) => h('div', { class: 'hero3', style: 'margin-top:16px' }, x.panel)),
  );
  // mini constellation of the live run's latest latent snapshot
  let mini = null, miniStep = null;
  const renderMini = async () => {
    const r = state.byId.get(hero.id); const step = r?.zSteps?.at(-1);
    if (step == null || step === miniStep) return;
    if (!mini) {
      const canvas = h('canvas');
      const stage = h('div', { class: 'latent-stage' }, canvas);
      const caption = h('div', { class: 'mini-caption' });
      latentPanel.replaceChildren(h('div', { class: 'row spread' }, h('span', { class: 'eyebrow', style: 'margin:0' }, 'latent'), h('a', { href: runHash(hero.id, 'latent'), class: 'back' }, 'open →')), stage, caption);
      try { mini = new LatentView(canvas); } catch { latentPanel.replaceChildren(h('div', { class: 'empty' }, 'WebGL2 unavailable')); return; }
      fullscreenStage(stage, mini);
      mini.size = 1.6; mini.mode = pref('tv.pca.mode', 0); mini.autoRotate = true;
      latentPanel._caption = caption;
      state.cleanup.push(() => mini.destroy());
    }
    try {
      const { xyz, norms, chans, header } = await fetchPca(hero.id, step);
      mini.setSnapshot(xyz, norms, header, chans); miniStep = step;
      latentPanel._caption.replaceChildren(h('span', {}, `step ${fmtX(step)}`), h('span', {}, `PC1–3 ${(100 * header.explained.reduce((a, b) => a + b, 0)).toFixed(0)}%`));
    } catch (e) { latentPanel._caption.textContent = e.message; }
  };
  const renderGpu = async () => { const g = await (await fetch('/api/gpu')).json(); gpuPanel.update(g.history); };
  await Promise.all(heroes.map((x) => x.render())); renderGpu();
  if (showLatent) renderMini();
  const liveKey = () => state.runs.filter((r) => r.status === 'live').map((r) => r.id).sort().join('|');
  const shownKey = liveKey();
  listeners.add((kind, d) => {
    // A run started or stopped: rebuild the page so every live run has its own panel.
    if (kind === 'run' && liveKey() !== shownKey) { route(); return; }
    if (kind === 'gpu') gpuPanel.push(d); // every 2 s sample, straight from the stream
    for (const x of heroes) x.onEvent(kind, d);
    if (kind === 'run' && showLatent) renderMini();
  });
}

/** Full panel for one live run: status, step, losses, latent health, newest reconstruction. */
function liveRunPanel(run) {
  const id = run.id;
  const panel = h('div', { class: 'panel hero-run' });
  const charts = {};
  const current = () => state.byId.get(id) || run;
  const render = async () => {
    const r = current();
    panel.style.setProperty('--run-color', runTint(r.id));
    panel.replaceChildren(
      h('div', { class: 'row spread' }, h('span', { class: `status ${r.status}` }, r.status === 'live' ? 'training now' : r.status),
        h('div', { class: 'row', style: 'gap:10px' }, h('span', { class: 'muted mono', style: 'font-size:11px' }, `updated ${ago(r.updatedAt)}`), h('a', { href: runHash(r.id), class: 'badge info' }, 'open →'))),
      h('a', { href: runHash(r.id), title: runTip(r) }, h('h1', { class: 'mono' }, runTitle(r)),
        runWhen(r) ? h('div', { class: 'run-when' }, `started ${runWhen(r)}`) : null),
      r.kind === 'harness' ? h('div', { class: 'hero-big hero-stats' }) : null,
      h('div', { class: 'progress' }, h('i', { style: `width:${r.totalSteps ? (100 * r.step) / r.totalSteps : 0}%` })),
      h('div', { class: 'row hero-badges' }),
      h('div', { class: 'grid cols-2 hero-charts' }),
      h('div', { class: 'recon-strip hero-recon' }),
    );
    updateStats();
    if (r.kind === 'harness') {
      const f = await loadFull(r.id);
      const cwrap = $('.hero-charts', panel);
      charts.loss = new LineChart(h('div'), { title: 'losses', log: true, height: 120 }); cwrap.append(charts.loss.el);
      charts.rank = new LineChart(h('div'), { title: 'latent health', height: 120 }); cwrap.append(charts.rank.el);
      state.cleanup.push(() => { charts.loss.destroy(); charts.rank.destroy(); });
      updateCharts(f);
      updateRecon(f);
    } else if (r.images?.length) {
      $('.hero-recon', panel).append(...r.images.filter((n) => /recon_clean|objective_recent/.test(n)).map((n) => zoomable(h('img', { src: `/api/runs/${encodeURIComponent(r.id)}/image/${n}?t=${r.updatedAt}`, alt: n }))));
    }
  };
  /** Live parts: status, step, ETA, losses, progress, health badges. Cheap; runs on every event. */
  const updateStats = () => {
    const r = current(); const last = r.last || {};
    const st = $('.status', panel); if (st) { st.className = `status ${r.status}`; st.textContent = r.status === 'live' ? 'training now' : r.status; }
    const stats = $('.hero-stats', panel);
    if (stats) {
      const sps = r.sps ?? last.sps; const eta = r.status === 'live' && sps && r.totalSteps ? `${Math.round((r.totalSteps - r.step) / sps / 60)} min left` : '';
      stats.replaceChildren(
        h('div', { class: 'num-wrap' }, h('div', { class: 'num' }, fmtX(r.step || 0), h('small', {}, `/ ${r.totalSteps ? fmtX(r.totalSteps) : '?'} steps`)), epochLabel(r)),
        h('div', {}, h('span', { class: 'eyebrow' }, 'throughput'), h('b', { class: 'mono' }, sps ? `${sps.toFixed(2)} it/s` : '–'), h('div', { class: 'muted', style: 'font-size:12px' }, eta)),
        h('div', {}, h('span', { class: 'eyebrow', title: 'reconstruction MSE on test images (last eval). The train clean loss is noised on some steps.' }, 'test mse'), h('b', { class: 'mono' }, fmt(r.lastEval?.test_clean))),
        h('div', {}, h('span', { class: 'eyebrow' }, 'view loss'), h('b', { class: 'mono' }, fmt(last.view))));
    }
    const p = $('.progress i', panel); if (p) p.style.width = `${r.totalSteps ? (100 * (r.step || 0)) / r.totalSteps : 0}%`;
    const b = $('.hero-badges', panel); if (b) b.replaceChildren(...badges(r));
  };
  const updateCharts = (f) => {
    const m = f.metrics;
    charts.loss.setSeries(TRAIN_KEYS.map((k, i) => ({ id: k, label: k, color: seriesColor(i), points: m.filter((r) => k in r).map((r) => [r.step, r[k]]) })).filter((s) => s.points.length));
    const ev = m.filter((r) => 'z_eff_rank' in r);
    // Notebook runs log test_clean without z_eff_rank, so test points get their own filter.
    const tests = m.filter((r) => typeof r.test_clean === 'number');
    charts.rank.setSeries([
      { id: 'rank', label: 'z eff rank', color: seriesColor(0), points: ev.map((r) => [r.step, r.z_eff_rank]), dots: true },
      { id: 'div', label: 'recon div ×100', color: seriesColor(1), points: ev.map((r) => [r.step, r.recon_div * 100]), dots: true },
      { id: 'test', label: 'test mse ×1000', color: seriesColor(2), points: tests.map((r) => [r.step, r.test_clean * 1000]), dots: true },
    ].filter((s) => s.points.length));
  };
  const updateRecon = (f) => {
    const wrap = $('.hero-recon', panel); if (!wrap) return;
    const step = f.recons.at(-1);
    if (step == null) return;
    const src = `/api/runs/${encodeURIComponent(f.id)}/recon/${step}`;
    if (wrap.dataset.src === src) return;
    wrap.dataset.src = src;
    const kind = current().reconKinds?.[step];
    wrap.replaceChildren(h('span', { class: 'eyebrow' }, `reconstruction · step ${fmtX(step)} (top: input, bottom: output)${kind === 'reeval' ? ' · re-rendered from checkpoint' : ''}`), zoomable(h('img', { src, alt: 'reconstruction' })));
  };
  const onEvent = (kind, d) => {
    if (kind === 'records' && d.id === id) {
      updateStats();
      const f = state.full.get(id); if (f && charts.loss) updateCharts(f);
    }
    if (kind === 'run') { updateStats(); const f = state.full.get(id); if (f) updateRecon(f); }
  };
  return { panel, render, onEvent };
}

// ---------------------------------------------------------------- Runs (top level)
async function viewRuns(el) {
  await loadRuns();
  setNav(topTabs('runs'));
  const liveSec = h('section', { class: 'section', style: 'margin-top:0' });
  const pinnedGrid = h('div', { class: 'grid cards' });
  const pinnedHead = h('div', { class: 'section-head' });
  const pinnedSec = h('section', { class: 'section' }, pinnedHead, pinnedGrid);
  const keptGrid = h('div', { class: 'grid cards' });
  const keptHead = h('div', { class: 'section-head' });
  const keptSec = h('section', { class: 'section' }, keptHead, keptGrid);
  const filters = h('div', { class: 'filters' });
  const grid = h('div', { class: 'grid cards' });
  const allHead = h('h2', {}, 'All runs');
  // search box lives outside render() so typing never loses focus
  const search = h('input', { class: 'run-search', type: 'search', placeholder: 'search name, description, flags…', value: state.list.q, 'aria-label': 'search runs' });
  search.addEventListener('input', () => { state.list.q = search.value; saveListPrefs(); render(); });
  const sortSel = h('select', { 'aria-label': 'sort runs' }, h('option', { value: 'newest' }, 'newest first'), h('option', { value: 'best' }, 'best test mse'));
  sortSel.value = state.list.sort;
  sortSel.addEventListener('change', () => { state.list.sort = sortSel.value; saveListPrefs(); render(); });
  const listChips = h('div', { class: 'filters' });
  const tools = h('div', { class: 'list-tools' }, search, sortSel, listChips);
  el.append(liveSec, pinnedSec, keptSec, h('section', { class: 'section' }, h('div', { class: 'section-head spread', style: 'align-items:center' }, allHead, filters), tools, grid));
  renderLiveSection(liveSec);
  const FAILED = new Set(['killed', 'failed', 'stale', 'diverged', 'timeout']);
  // searchable text per run: name, description, config flags as key=value
  const hay = (r) => [r.name, r.description || '', ...Object.entries(r.config || {}).filter(([, v]) => typeof v !== 'object').map(([k, v]) => `${k}=${v}`)].join(' ').toLowerCase();
  const matches = (r) => {
    const terms = state.list.q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return true;
    const s = hay(r);
    return terms.every((t) => s.includes(t));
  };
  const sorted = (list) => state.list.sort === 'best'
    ? [...list].sort((a, b) => (a.bestTest ?? Infinity) - (b.bestTest ?? Infinity))
    : list;   // server order: newest first
  const render = () => {
    const kept = state.runs.filter((r) => state.kept.has(r.id) && matches(r));
    keptSec.hidden = !kept.length;
    keptHead.replaceChildren(h('h2', {}, '★ Kept'), h('span', { class: 'muted' }, 'never bulk-archived'));
    keptGrid.replaceChildren(...sorted(kept).map((r) => runCard(r)));
    const pins = [...state.pinned].filter((id) => state.byId.has(id));
    pinnedSec.hidden = !pins.length;
    pinnedHead.replaceChildren(h('h2', {}, 'Pinned'), h('span', { class: 'muted' }, 'overlaid in every run\'s charts'),
      pins.length > 1 ? h('a', { href: runHash(pins[0]), class: 'badge info' }, 'open overlay →') : null,
      h('button', { class: 'filter', style: 'margin-left:auto', onclick: () => { state.pinned.clear(); localStorage.setItem('tv.pinned', '[]'); render(); } }, 'unpin all'));
    pinnedGrid.replaceChildren(...pins.map((id) => runCard(state.byId.get(id))));
    const statuses = [...new Set(state.runs.map((r) => r.status))];
    filters.replaceChildren(
      ...state.roots.map((rt) => h('button', { class: `filter ${state.filter.root === rt.id ? 'on' : ''}`, onclick: () => { state.filter.root = state.filter.root === rt.id ? null : rt.id; render(); } }, rt.label)),
      ...statuses.map((s) => h('button', { class: `filter ${state.filter.status === s ? 'on' : ''}`, onclick: () => { state.filter.status = state.filter.status === s ? null : s; render(); } }, s)),
    );
    const nArchived = state.runs.filter((r) => state.archived.has(r.id)).length;
    // bulk archive: failed/killed/stale runs that are not kept, live, pinned or already archived
    const failed = state.runs.filter((r) => FAILED.has(r.status) && !state.kept.has(r.id) && !state.archived.has(r.id) && !state.pinned.has(r.id));
    listChips.replaceChildren(
      h('button', { class: `filter ${state.list.latents ? 'on' : ''}`, title: 'only runs with latent snapshots (Latent tab)', onclick: () => { state.list.latents = !state.list.latents; saveListPrefs(); render(); } }, 'has latents'),
      h('button', { class: `filter ${state.list.archived ? 'on' : ''}`, onclick: () => { state.list.archived = !state.list.archived; saveListPrefs(); render(); } }, `show archived (${nArchived})`),
      h('button', { class: `filter ${state.list.smoke ? 'on' : ''}`, title: 'smoke tests are hidden by default; nothing is deleted', onclick: () => { state.list.smoke = !state.list.smoke; saveListPrefs(); render(); } }, `show smoke tests (${state.runs.filter(isSmoke).length})`),
      failed.length ? h('button', { class: 'filter danger', title: 'hide these from the default list; nothing is deleted, and "show archived" brings them back',
        onclick: async () => {
          const names = failed.slice(0, 12).map((r) => `  ${r.status.padEnd(8)} ${r.name}`).join('\n');
          if (!confirm(`Archive ${failed.length} failed / killed / stale runs?\nKept (★) and pinned runs are skipped. Nothing is deleted.\n\n${names}${failed.length > 12 ? `\n  … and ${failed.length - 12} more` : ''}`)) return;
          const d = await curate('archive', failed.map((r) => r.id));
          if (d) toast(`archived ${d.changed} runs`);
        } }, `archive failed (${failed.length})`) : null,
    );
    const list = state.runs.filter((r) => !state.pinned.has(r.id) && !state.kept.has(r.id)
      && (state.list.archived || !state.archived.has(r.id))
      && (state.list.smoke || !isSmoke(r))
      && (!state.list.latents || r.zSteps?.length)
      && (!state.filter.root || r.root === state.filter.root) && (!state.filter.status || r.status === state.filter.status)
      && matches(r));
    allHead.textContent = state.list.q || state.list.latents || state.filter.root || state.filter.status ? `Runs (${list.length})` : `All runs (${list.length})`;
    grid.replaceChildren(...(list.length ? sorted(list).map((r) => runCard(r)) : [h('div', { class: 'empty' }, 'nothing matches')]));
  };
  render();
  listeners.add((kind) => { if (kind !== 'gpu') render(); });
}

/** Top-level pages in the top bar: Runs ("#"), Experiments ("#experiments", groups of comparable runs) and Reports ("#reports", custom pages). */
const topTabs = (active) => [{ sub: true, href: '#', label: 'Runs', active: active === 'runs' }, { sub: true, href: '#experiments', label: 'Experiments', active: active === 'experiments' },
  { sub: true, href: '#reports', label: 'Reports', active: active === 'reports' }];
/** Top-bar navigation: crumbs + sub-tabs for the current page. */
function setNav(items) {
  const nav = $('#nav');
  nav.replaceChildren(...items.map((it) => it.sub
    ? h('a', { href: it.href, class: `tab ${it.active ? 'active' : ''}` }, it.label)
    : h('a', { href: it.href, class: 'crumb' }, it.label)));
  nav.hidden = !items.length;
}

// ---------------------------------------------------------------- Run detail
const TRAIN_KEYS = ['clean', 'lap', 'reg', 'view']; // colour rank: primary → quaternary
const EVAL_GROUPS = [
  { title: 'test reconstruction', keys: ['test_clean', 'test_clean_jit'], log: true },
  { title: 'output diversity (recon std / image std)', keys: ['recon_div', 'recon_div_jit'] },
  { title: 'latent effective rank', keys: ['z_eff_rank', 'z_rank_1pct', 'rank_chan', 'rank_tok', 'z_pr_rank'] },
  { title: 'latent scale', keys: ['z_std_mean', 'z_std_min', 'z_dead_frac'], log: true },
  { title: 'decoder stream rms', keys: ['dec_patch_rms', 'dec_lat_rms', 'dec_patch_img_sd'], log: true },
  { title: 'laplacian ratio (high-freq energy, recon / image)', keys: ['lap_ratio'] },
  // Is the latent organised by color? From 2,048 fixed test images and their 4 x 4 thumbnails (runner sink).
  { title: 'color spread among latent neighbors / over all images (1 = neighbors are as mixed as random images)', keys: ['nn_color_ratio', 'prior_nn_color_ratio'], labels: { nn_color_ratio: 'around test images', prior_nn_color_ratio: 'around random prior points' } },
  { title: 'share of the variance of z that the thumbnail explains (linear fit)', keys: ['z_from_color_r2'] },
  { title: 'z distances to the prior: neighbor batches / random batches (1 = a neighborhood looks like the whole latent)', keys: ['knn_reg_ratio'], log: true },
];
// Evals of the pixel-space diffusion runs (ImageNet-Unify).
const PIXEL_EVAL = [
  { title: 'test loss (fixed images, t and noise)', keys: ['test_clean', 'train_clean_eval'], log: true },
  { title: 'x-prediction MSE at fixed noise levels (t = 0.1 … 0.9)', keys: ['x_mse_t01', 'x_mse_t03', 'x_mse_t05', 'x_mse_t07', 'x_mse_t09'], log: true },
  { title: 'x-prediction MSE, mean over the noise levels', keys: ['test_x_mse'], log: true },
  { title: 'learning rate (read from the optimizer at every log)', keys: ['lr'], log: true },
  { title: 'XM gap (loss of all candidates − loss of the kept ones)', keys: ['xm_gap'] },
  // detail of the 12 fixed-seed previews: from the trained weights, and from the moving average of the weights (cfg ema)
  { title: 'sample sharpness of the fixed seeds (real images ≈ 0.093): trained weights vs averaged weights', keys: ['prev_sharp', 'ema_prev_sharp'], labels: { prev_sharp: 'trained weights', ema_prev_sharp: 'averaged weights (EMA)' } },
  // black/white high contrast (user, 2026-10-24: self-teaching on self-flow pushes samples to black and white). Real values:
  // 2,400 validation images, the job's 128 px resize; a set of 12 real images spans pixel std 0.24-0.30, black 0.4-8.5 % (5-95 %).
  { title: 'sample contrast of the fixed seeds: pixel std (real images ≈ 0.270)', keys: ['prev_contrast', 'ema_prev_contrast'], labels: { prev_contrast: 'trained weights', ema_prev_contrast: 'averaged weights (EMA)' } },
  { title: 'black / white pixels of the fixed seeds, EMA: share below 0.02 / above 0.98 (real images ≈ 0.031 / 0.027)', keys: ['ema_prev_dark', 'ema_prev_light'], labels: { ema_prev_dark: 'black (< 0.02)', ema_prev_light: 'white (> 0.98)' } },
  // generative scores of samples from fixed seeds against real validation images (gen_eval.py); lower is better
  // *_ema: the same score from the averaged weights (gen_metrics.jsonl / gen_fix.jsonl, scored after the fact)
  { title: 'FID (10,000 samples, 16 euler steps; real against real ≈ 7)', keys: ['fid', 'fid_ema'], labels: { fid: 'trained weights', fid_ema: 'averaged weights (EMA)' }, log: true },
  { title: 'KID × 1000 (same samples; real against real ≈ 0)', keys: ['kid', 'kid_ema'], labels: { kid: 'trained weights', kid_ema: 'averaged weights (EMA)' }, log: true },
  { title: 'fd_ae: Fréchet distance in the latent space of the 128 px AE', keys: ['fd_ae_z', 'fd_ae_pool', 'fd_ae_z_ema', 'fd_ae_pool_ema'], labels: { fd_ae_z: 'flattened z (real against real ≈ 162)', fd_ae_pool: 'token mean and std (real against real ≈ 0.01)', fd_ae_z_ema: 'flattened z, EMA', fd_ae_pool_ema: 'token mean and std, EMA' }, log: true },
];
// Curve charts: x is not the train step. Each eval record carries one curve as flat keys
// `<prefix>1`, `<prefix>2`, ... (x = the number). `history`: earlier evals as a dark→light ramp
// under the latest eval; without it only the latest eval is drawn, one line per family.
const intTick = (f) => (v) => (Number.isInteger(v) ? f(v) : '');
const LOOP_CURVES = [
  { title: 'quality vs loops: eval loss by loop count (latest eval)', xName: 'loops', xFmt: intTick(String), log: true,
    families: [{ prefix: 'loopq_dense_', label: 'dense eval' }, { prefix: 'loopq_sparse_', label: 'sparse eval' }] },
  { title: 'recurrent update size: RMS(h_next − h) / RMS(h) per loop transition', xName: 'transition', xFmt: intTick((v) => `${v}→${v + 1}`), log: true, history: true,
    families: [{ prefix: 'upd_', label: 'update size' }] },
  { title: 'recurrent state similarity: cos(h_k, h_k+1) per loop transition', xName: 'transition', xFmt: intTick((v) => `${v}→${v + 1}`), history: true,
    families: [{ prefix: 'cos_', label: 'cosine' }] },
  { title: 'XSA effect: share of the attention output that XSA removes, per loop', xName: 'loop', xFmt: intTick(String), history: true,
    families: [{ prefix: 'xsa_', label: 'removed share' }] },
];
/**
 * Run types: what a run page shows. A run names its type in config.run_type; older runs are
 * recognised by the metric keys they logged. Every chart is also hidden while none of its
 * series has data, so a type lists what MAY exist and the page shows what DOES.
 *   train   charts of per-step training metrics       evals   charts of eval metrics over steps
 *   curves  curve charts (see LOOP_CURVES)            robust / diag   the AE-only sections
 */
const RUN_TYPES = {
  ae: { label: 'autoencoder', train: [{ title: 'losses (this run)', keys: TRAIN_KEYS, log: true },
    // raw distances to the Gaussian prior, no weights: of z itself, and of the projection head's output (PROJ_SWD runs)
    { title: 'distance to the Gaussian prior: z itself (z_*) and the projection of z', keys: ['proj_swd', 'proj_sigreg', 'proj_inv', 'view_z', 'cluster_reg', 'reg', 'cluster_tight', 'z_swd', 'z_bwd', 'z_c_swd', 'z_t_swd', 'z_reg_sum'], labels: { proj_swd: 'projection swd (in the loss)', proj_sigreg: 'projection SIGReg (in the loss)', proj_inv: 'projection invariance (in the loss)', view_z: 'view loss in z space (monitor)', cluster_reg: 'weighted z distances on a neighbor batch', cluster_tight: 'neighbor distance / typical distance', reg: 'weighted z distances on a train batch', z_reg_sum: 'z sum of four' }, log: true },
    // regularizer power experiment (regx/README.md): every 1,000 steps on fixed held-out images
    { title: 'latent churn: RMS change of z over 100 steps (held-out images)', keys: ['rx_churn'], labels: { rx_churn: 'churn' } },
    { title: 'held-out regularizer terms (the noise floor of swd and bwd is 0.0039 each)', keys: ['rx_g_swd', 'rx_g_bwd', 'rx_g_c_swd', 'rx_g_t_swd'], labels: { rx_g_swd: 'swd', rx_g_bwd: 'bwd', rx_g_c_swd: 'c_swd', rx_g_t_swd: 't_swd' }, log: true },
    { title: 'encoder gradient norms on a fixed batch', keys: ['rx_g_view', 'rx_g_reg'], labels: { rx_g_view: 'view', rx_g_reg: 'regularizer' }, log: true },
    { title: 'z effective rank (held-out images)', keys: ['rx_z_rank'], labels: { rx_z_rank: 'rank' }, log: true }], evals: EVAL_GROUPS, curves: [], robust: true, diag: true,
    testLabel: 'test mse', trainLabel: 'train clean', reconTitle: 'Reconstructions over training', reconNote: 'top row: input · bottom row: output · drag to scrub' },
  pixel: { label: 'pixel diffusion', train: [{ title: 'train loss (this run)', keys: ['clean', 'loss_all'], log: true }], evals: PIXEL_EVAL, curves: [],
    testLabel: 'test loss', trainLabel: 'train loss', reconTitle: 'Samples over training', reconNote: 'fixed seeds · drag to scrub' },
  pixel_pyramid: { label: 'pixel diffusion, pyramid readouts',
    train: [{ title: 'train loss of the final head (pinned runs dashed)', keys: ['clean'], log: true },
      { title: 'train loss per head: low-res readouts, final head, and their sum', keys: ['R32', 'R64', 'clean', 'total'], labels: { R32: 'R32 (early readout)', R64: 'R64 (later readout)', clean: 'final 128', total: 'sum' }, log: true },
      { title: 'hidden-state teaching loss: mse(student block, target block), train and test', keys: ['teach', 'test_teach'], labels: { teach: 'train (in the loss)', test_teach: 'test, one forward pass' }, log: true },
      // cfg selfteach (2026-10-21, scaling sweep): normalized distance of the output crop to the clean crop in the frozen judge
      { title: 'self-teaching: output crop to clean crop in the frozen judge (normalized mse, before the weight)', keys: ['st_out', 'st_noised'], need: 'st_out', labels: { st_out: 'output term (in the loss)', st_noised: 'noised term' }, log: true },
      { title: 'self-teaching distance measured on fixed test images (every run of the sweep, baseline too): one-pass output crop to clean crop', keys: ['st_eval_b3', 'st_eval_b6'], need: 'st_eval_b3', labels: { st_eval_b3: 'judge block 3', st_eval_b6: 'judge block 6' }, log: true },
      { title: 'train x-MSE by noise level of the input (0 = clean, 1 = pure noise)', keys: ['train_x_mse_n1', 'train_x_mse_n3', 'train_x_mse_n5', 'train_x_mse_n7', 'train_x_mse_n9'],
        labels: { train_x_mse_n1: 'noise 0.1–0.2', train_x_mse_n3: 'noise 0.3–0.4', train_x_mse_n5: 'noise 0.5–0.6', train_x_mse_n7: 'noise 0.7–0.8', train_x_mse_n9: 'noise 0.9–1.0' }, log: true },
      { title: 'mean noise level of the train inputs (0.5 without a timestep shift)', keys: ['noise_mean'] },
      // cfg wreg: the 4 Wasserstein regularizers of the AE notebook on the residual stream after one block (raw, before the weight)
      { title: 'Wasserstein regularizers on the residual stream: distance to N(0, 1), before the weight', keys: ['wreg_swd', 'wreg_bwd', 'wreg_c_swd', 'wreg_t_swd'],
        labels: { wreg_swd: 'sliced (flat)', wreg_bwd: 'basis (across images)', wreg_c_swd: 'channel-sliced', wreg_t_swd: 'token-sliced' }, log: true },
      { title: 'regularized residual stream: mean and std of all values (the prior has 0 and 1)', keys: ['wreg_z_mean', 'wreg_z_std'], labels: { wreg_z_mean: 'mean', wreg_z_std: 'std' } },
      { title: 'transport step at 100% noise: x-MSE with the Hungarian pairing and with the fixed pairing', keys: ['ot_loss', 'ot_loss_fixed'], labels: { ot_loss: 'Hungarian pairing', ot_loss_fixed: 'fixed pairing (output i, image i)' }, log: true },
      { title: 'transport step: share of the loss saved by the pairing, and share of outputs that changed partner', keys: ['ot_gain', 'ot_moved'], labels: { ot_gain: 'loss saved', ot_moved: 'changed partner' } },
      { title: 'gradient norm: transport step (with its weight) and normal step', keys: ['ot_grad_norm', 'grad_norm_normal'], labels: { ot_grad_norm: 'transport step', grad_norm_normal: 'normal step' }, log: true },
      { title: 'outputs from 100% noise: std over the batch (0 = every output is the same image)', keys: ['ot_out_std', 'ot_prev_std'], labels: { ot_out_std: 'train batch', ot_prev_std: 'preview seeds' } },
      { title: '2-step distillation: mse between the 1-step and the 2-step output', keys: ['d2_gap'] },
      { title: '2-step distillation: mse to the clean image (train inputs)', keys: ['d2_err_a', 'd2_err_b'], labels: { d2_err_a: '1-step output', d2_err_b: '2-step output' } },
      { title: 'Class conditioning: mse of 256 test images at noise level 0.95', keys: ['cond_err_true', 'cond_err_none', 'cond_err_wrong'], labels: { cond_err_true: 'true class', cond_err_none: 'no class', cond_err_wrong: 'class of another image' } },
      { title: 'Class conditioning: share of the error that the class removes', keys: ['cond_gain'] },
      // cfg msa (2026-10-15): three forwards per step (dense, split 42/22, sparse 32 single + 16 pairs) of the same weights
      { title: 'MSA: flow loss of the final head per forward (dense = the model that samples)', keys: ['clean', 'msa_final_split', 'msa_final_sparse'], need: 'msa_final_split', labels: { clean: 'dense', msa_final_split: 'split', msa_final_sparse: 'sparse' }, log: true },
      { title: 'MSA: alignment terms (normalized mse, student block 3 to the detached target)', keys: ['msa_d6s3', 'msa_d6p3', 'msa_s6p3'], need: 'msa_d6s3', labels: { msa_d6s3: 'dense 6 > split 3', msa_d6p3: 'dense 6 > sparse 3', msa_s6p3: 'split 6 > sparse 3' }, log: true },
      { title: 'MSA: mean t per forward (t = 1 is the clean image)', keys: ['msa_t_dense', 'msa_t_split', 'msa_t_sparse'], need: 'msa_t_dense', labels: { msa_t_dense: 'dense', msa_t_split: 'split', msa_t_sparse: 'sparse' } },
      // cfg corrector (2026-10-13): blocks 1..split = base denoiser, the others = a corrector of its image, applied k times
      { title: 'Corrector: train loss of the base denoiser and of the corrector (1 pass), and their sum', keys: ['base', 'clean', 'total'], need: 'base', labels: { base: 'base denoiser', clean: 'corrector, 1 pass', total: 'sum' }, log: true },
      { title: 'Corrector: test loss of the base denoiser and of the corrector (same images, t and noise)', keys: ['test_base', 'test_clean'], need: 'test_base', labels: { test_base: 'base denoiser', test_clean: 'corrector, 1 pass' }, log: true },
      { title: 'Corrector: plain x-MSE on the train batch', keys: ['corr_base_xmse', 'corr_final_xmse'], labels: { corr_base_xmse: 'base denoiser', corr_final_xmse: 'corrector, 1 pass' }, log: true },
      { title: 'Corrector: test x-MSE after k corrections (noise levels of training; k = 0 is the base denoiser)', keys: ['corr_mse_k0', 'corr_mse_k1', 'corr_mse_k2', 'corr_mse_k4', 'corr_mse_k8'], labels: { corr_mse_k0: 'k = 0', corr_mse_k1: 'k = 1 (trained)', corr_mse_k2: 'k = 2', corr_mse_k4: 'k = 4', corr_mse_k8: 'k = 8' }, log: true },
      { title: 'Corrector: test x-MSE after k corrections at noise level 0.9', keys: ['corr_mse_n90_k0', 'corr_mse_n90_k1', 'corr_mse_n90_k2', 'corr_mse_n90_k4', 'corr_mse_n90_k8'], labels: { corr_mse_n90_k0: 'k = 0', corr_mse_n90_k1: 'k = 1', corr_mse_n90_k2: 'k = 2', corr_mse_n90_k4: 'k = 4', corr_mse_n90_k8: 'k = 8' }, log: true },
      { title: 'Corrector: test x-MSE after k corrections at noise level 0.5', keys: ['corr_mse_n50_k0', 'corr_mse_n50_k1', 'corr_mse_n50_k2', 'corr_mse_n50_k4', 'corr_mse_n50_k8'], labels: { corr_mse_n50_k0: 'k = 0', corr_mse_n50_k1: 'k = 1', corr_mse_n50_k2: 'k = 2', corr_mse_n50_k4: 'k = 4', corr_mse_n50_k8: 'k = 8' }, log: true },
      { title: 'Corrector: test x-MSE after k corrections at noise level 0.3', keys: ['corr_mse_n30_k0', 'corr_mse_n30_k1', 'corr_mse_n30_k2', 'corr_mse_n30_k4', 'corr_mse_n30_k8'], labels: { corr_mse_n30_k0: 'k = 0', corr_mse_n30_k1: 'k = 1', corr_mse_n30_k2: 'k = 2', corr_mse_n30_k4: 'k = 4', corr_mse_n30_k8: 'k = 8' }, log: true },
      { title: 'MoE: how far the experts moved apart (distance to their mean, relative)', keys: ['moe_spread'] },
      { title: 'MoE router: mean noise level of the tokens of the expert with the cleanest and with the noisiest tokens (last MoE block)', keys: ['moe_noise_low', 'moe_noise_high'], labels: { moe_noise_low: 'cleanest expert', moe_noise_high: 'noisiest expert' } },
      { title: 'MoE router: mean probability of the chosen expert (1 / experts = no preference)', keys: ['moe_router_p'] },
      // cfg moe route "top2" (2026-10-18): usage share per expert per window (balanced 1 / 8), load balancing, clones
      ...[2, 4, 6, 8].map((b) => ({ title: `MoE top-2, block ${b}: share of the train assignments per expert, per window (balanced 0.125)`,
        keys: [0, 1, 2, 3, 4, 5, 6, 7].map((e) => `moe_use_b${b}_e${e}`), labels: Object.fromEntries([0, 1, 2, 3, 4, 5, 6, 7].map((e) => [`moe_use_b${b}_e${e}`, `expert ${e}`])) })),
      { title: 'MoE top-2: load-balancing term (sum over the MoE layers, balanced = number of layers)', keys: ['moe_aux'] },
      { title: 'MoE top-2: experts replaced by a clone, so far', keys: ['moe_replacements'] },
      { title: 'MoE top-2: train loss jump at a clone (mean of 200 steps after - 200 before)', keys: ['moe_jump_b2', 'moe_jump_b4', 'moe_jump_b6', 'moe_jump_b8'],
        labels: { moe_jump_b2: 'block 2', moe_jump_b4: 'block 4', moe_jump_b6: 'block 6', moe_jump_b8: 'block 8' } },
      { title: 'Noise transport: mean squared pixel distance from an image to its noise sample', keys: ['nx_dist', 'nx_dist_fixed'], labels: { nx_dist: 'paired', nx_dist_fixed: 'unpaired' } },
      { title: 'Noise transport: share of the noise samples that changed image', keys: ['nx_moved'] },
      { title: 'XM transport: share of the weighted loss saved against a random assignment of the seeds', keys: ['xm_gain'], labels: { xm_gain: 'loss saved' } },
      { title: 'XM transport: share of the x-MSE saved, by noise level (n0 nearly clean, n9 nearly pure noise)', keys: ['xm_gain_n0', 'xm_gain_n2', 'xm_gain_n4', 'xm_gain_n6', 'xm_gain_n8', 'xm_gain_n9'] }],
    evals: [...PIXEL_EVAL, { title: 'latent snapshot (fixed clean test images): std over images and participation rank', keys: ['z_std_mean', 'z_pr_rank'], labels: { z_std_mean: 'std over images', z_pr_rank: 'participation rank' }, log: true },
      { title: 'test loss per head (same images, t and noise)', keys: ['test_R32', 'test_R64', 'test_clean'], labels: { test_R32: 'R32', test_R64: 'R64', test_clean: 'final 128' }, log: true }],
    // layer distillation with several pairs: the distance of every student block to its target (block 0 = patch embedding)
    curves: [{ title: 'layer distillation: distance of each student block to its target block (0 = patch embedding)', xName: 'student block', xFmt: intTick(String), log: true, history: true,
      families: [{ prefix: 'teach_layer_', label: 'distance to target' }] },
      { title: 'train x-MSE in 10 bins of the noise level (bin 0 = nearly clean, bin 9 = nearly pure noise)', xName: 'noise level bin', xFmt: intTick(String), log: true, history: true,
        families: [{ prefix: 'train_x_mse_n', label: 'train x-MSE' }] }],
    testLabel: 'test loss', trainLabel: 'train loss', reconTitle: 'Samples over training', reconNote: 'fixed seeds, final head · drag to scrub' },
  pixel_loop: { label: 'pixel diffusion, looped middle',
    train: [{ title: 'train loss (this run): total = 0.5 · mean(L1..L3) + 0.5 · L4', keys: ['clean'], log: true },
      { title: 'per-loop supervision: train flow loss after each loop', keys: ['L1', 'L2', 'L3', 'L4', 'L_avg123', 'clean'], labels: { L_avg123: 'avg(L1:L3)', clean: 'total' }, log: true },
      { title: 'last-loop loss (L4) in sparse and in dense steps', keys: ['L4_sparse', 'L4_dense'], log: true }],
    evals: PIXEL_EVAL, curves: LOOP_CURVES,
    testLabel: 'test loss', trainLabel: 'train loss', reconTitle: 'Samples over training', reconNote: 'fixed seeds, dense, trained loop count · drag to scrub' },
  flow: { label: 'latent flow', train: [{ title: 'train flow loss (this run)', keys: ['clean', 'sparse_flow_loss', 'dense_flow_loss'], labels: { clean: 'flow loss' }, log: true },
    { title: 'learning rate (read from the optimizer)', keys: ['lr'] }],
    evals: [{ title: 'test flow loss (dense, all validation latents)', keys: ['test_clean', 'train_clean_eval'], log: true },
      { title: 'XM gap (loss of all candidates − loss of the kept ones)', keys: ['xm_gap'] },
      // generative scores of decoded samples; *_recon = the AE's own ceiling: decoded latents of real validation images
      { title: 'FID (10,000 samples, 16 euler steps) and the AE ceiling', keys: ['fid', 'fid_recon'], labels: { fid: 'FID of samples', fid_recon: 'AE ceiling (decoded real latents)' }, log: true },
      { title: 'KID × 1000 and the AE ceiling', keys: ['kid', 'kid_recon'], labels: { kid: 'KID of samples', kid_recon: 'AE ceiling' }, log: true },
      { title: 'fd_ae: Fréchet distance of decoded samples in the latent space of the 128 px reference AE', keys: ['fd_ae_z', 'fd_ae_z_recon', 'fd_ae_pool'], labels: { fd_ae_z: 'flattened z', fd_ae_z_recon: 'AE ceiling, flattened z', fd_ae_pool: 'token mean and std' }, log: true },
      { title: 'fd_z_own: Fréchet distance of the sampled latents to real latents, in this run\'s own AE', keys: ['fd_z_own'], log: true }], curves: [],
    testLabel: 'test loss', trainLabel: 'train loss', reconTitle: 'Samples over training', reconNote: 'fixed seeds, decoded by the AE · drag to scrub' },
  // lejepa-wreg (2026-11-09): self-supervised encoder, loss = (1 − λ) · align + λ · reg; evals = frozen linear probe on native 384-d features
  ssl: { label: 'self-supervised encoder (LeJEPA-style)',
    train: [{ title: 'train loss: (1 − λ) · align + λ · reg', keys: ['clean'], labels: { clean: 'total' }, log: true },
      { title: 'loss terms before the weights: alignment (views to their mean) and regularizer', keys: ['align', 'reg'], labels: { align: 'alignment', reg: 'regularizer (SIGReg, or weighted Wasserstein sum)' }, log: true },
      { title: 'Wasserstein terms per view, before the 0.05 weights (arm B)', keys: ['w_swd', 'w_bwd', 'w_bwd_coord'], need: 'w_swd', labels: { w_swd: 'sliced: 1024 directions of the 384-d cloud', w_bwd: 'basis, per image: its 384 values vs N(0, 1)', w_bwd_coord: 'basis, per coordinate across images (monitor, not in the loss)' }, log: true },
      { title: 'learning rate (cosine to the absolute floor 1e-3)', keys: ['lr'] },
      { title: 'throughput: images per second (source images) and data wait per step (ms)', keys: ['img_s', 'data_wait_ms'], labels: { img_s: 'images / s', data_wait_ms: 'data wait ms' } },
      { title: 'peak GPU memory (GiB)', keys: ['gpu_mem_gib'] },
      // fixed 512 unaugmented images, eval mode, every drift_every epochs; change since the previous snapshot
      { title: 'latent drift of native features: displacement and pairwise-distance change since the previous snapshot', keys: ['drift_native_disp', 'drift_native_pdist', 'drift_native_pdist_shape'], need: 'drift_native_disp', labels: { drift_native_disp: 'displacement / norm', drift_native_pdist: 'pairwise distances, relative', drift_native_pdist_shape: 'pairwise distances, scale-free' }, log: true },
      { title: 'latent drift of projector outputs (arm A)', keys: ['drift_proj_disp', 'drift_proj_pdist', 'drift_proj_pdist_shape'], need: 'drift_proj_disp', labels: { drift_proj_disp: 'displacement / norm', drift_proj_pdist: 'pairwise distances, relative', drift_proj_pdist_shape: 'pairwise distances, scale-free' }, log: true },
      { title: 'effective rank of the drift images (native; projector for A)', keys: ['drift_native_erank', 'drift_proj_erank'], need: 'drift_native_erank', labels: { drift_native_erank: 'native 384-d', drift_proj_erank: 'projector output' } }],
    evals: [{ title: 'frozen linear probe top-1 on native 384-d features (A also: its 512-d map output)', keys: ['probe_acc', 'probe_acc_emb512'], labels: { probe_acc: 'native 384-d', probe_acc_emb512: 'A: 512-d map output' } },
      { title: 'collapse check: covariance effective rank of native features (probe-eval images)', keys: ['feat_erank'] },
      { title: 'collapse check: per-dimension std of native features (mean and min)', keys: ['feat_std_mean', 'feat_std_min'], labels: { feat_std_mean: 'mean', feat_std_min: 'min' }, log: true },
      { title: 'each embedding over its own 384 values: |mean| and std (B targets 0 and 1; LayerNorm arms sit near 0 and 1 by construction)', keys: ['feat_img_mean', 'feat_img_std'], labels: { feat_img_mean: '|mean|', feat_img_std: 'std' } }],
    curves: [], testLabel: 'probe error', trainLabel: 'train loss', reconTitle: 'Images over training', reconNote: 'this run type writes no images' },
};
function runType(f) {
  if (RUN_TYPES[f.config?.run_type]) return f.config.run_type;
  const has = (k) => (f.metrics || []).some((r) => typeof r[k] === 'number');
  if (has('L1') || has('loopq_dense_1')) return 'pixel_loop';
  if (has('R32') || has('test_R32')) return 'pixel_pyramid';
  if (has('sparse_flow_loss')) return 'flow';
  if (has('test_x_mse')) return 'pixel';
  return 'ae';
}
/** Run page shell: persistent run header + Details / Latent sub-tabs. */
async function viewRunPage(el, id, sub) {
  let f;
  try { await loadRuns(); f = await loadFull(id); } catch { setNav([]); el.append(h('div', { class: 'empty' }, `run ${id} not found`)); return; }
  const others = [...state.pinned].filter((p) => p !== id && state.byId.has(p) && state.byId.get(p).kind === 'harness');
  const subtab = (s, label) => ({ href: runHash(id, s), label, sub: true, active: s === sub });
  setNav([{ href: '#', label: '← runs' }, subtab('details', 'Details'), subtab('latent', 'Latent')]);
  const head = h('div', { class: 'panel detail-head run-head', style: `--run-color:${seriesColor(0)}` });
  const body = h('div');
  el.append(head, body);
  const renderHead = () => {
    const r = state.byId.get(id) || f; const last = r.last || {};
    const T = RUN_TYPES[runType(f)];
    head.replaceChildren(
      h('div', { class: 'detail-title' },
        h('div', { class: 'title-block', title: runTip(r) }, h('h1', {}, runTitle(r)),
          runWhen(r) ? h('div', { class: 'run-when' }, `started ${runWhen(r)}`) : null),
        h('span', { class: `status ${r.status}` }, r.status),
        h('button', { class: `filter ${state.pinned.has(id) ? 'on' : ''}`, onclick: () => { togglePin(id); renderHead(); } }, state.pinned.has(id) ? '◆ pinned' : '◇ pin'),
        ...curationButtons(id, 'filter').reverse(),
        h('div', { class: 'subtabs', style: 'margin-left:auto' }, ...SUBS.map((s) => h('a', { href: runHash(id, s), class: s === sub ? 'active' : '' }, s[0].toUpperCase() + s.slice(1))))),
      // a description may name pages of this site (/multires.html) or full URLs: they become links
      ...(r.description ? [h('p', { class: 'run-desc' }, ...r.description.split(/(https?:\/\/[^\s)]+|\/[\w.-]+\.html)/)
        .map((part, i) => (i % 2 ? h('a', { href: part, target: '_blank', rel: 'noopener' }, part) : part)))] : []),   // replaceChildren would print a null
      ...(shapeFooter(r) ? [h('p', { class: 'card-foot' }, shapeFooter(r))] : []),
      r.kind === 'harness' ? h('div', { class: 'hero-big' },
        h('div', { class: 'num-wrap' }, h('div', { class: 'num' }, fmtX(r.step || 0), h('small', {}, `/ ${r.totalSteps ? fmtX(r.totalSteps) : '?'}`)), epochLabel(r)),
        // test MSE is the comparable number; train clean is noised on some steps (latent noise) so it reads higher
        h('div', {}, h('span', { class: 'eyebrow', title: `${T.testLabel} on fixed test data, eval at step ${fmtX(r.lastEval?.step ?? 0)}` }, T.testLabel), h('b', { class: 'mono' }, fmt(r.lastEval?.test_clean))),
        h('div', {}, h('span', { class: 'eyebrow', title: 'training loss, mean of the last logged window' }, T.trainLabel), h('b', { class: 'mono' }, fmt(last.clean))),
        ...(typeof last.view === 'number' ? [h('div', {}, h('span', { class: 'eyebrow' }, 'view'), h('b', { class: 'mono' }, fmt(last.view)))] : []),
        h('div', {}, h('span', { class: 'eyebrow' }, 'it/s'), h('b', { class: 'mono' }, (r.sps ?? last.sps) ? (r.sps ?? last.sps).toFixed(2) : '–')),
        h('div', {}, h('span', { class: 'eyebrow' }, 'started'), h('b', { class: 'mono' }, ago(r.startedAt))),
      ) : h('div', { class: 'muted' }, `notebook run · started ${r.meta?.started_at || '?'} · history.pt lives on the notebook owner's account`),
      h('div', { class: 'progress' }, h('i', { style: `width:${r.totalSteps ? (100 * r.step) / r.totalSteps : 0}%` })),
      h('div', { class: 'row' }, ...badges(r)),
      h('div', { class: 'row switcher' },
        h('span', { class: 'muted', style: 'font-size:11px' }, 'jump to'),
        // stable list: live runs, then pinned runs — the current one stays in place, greyed and inert
        ...state.runs.filter((x) => x.status === 'live').map((x) => x.id === id
          ? h('span', { class: 'chip here' }, h('span', { class: 'status live' }, ''), h('b', {}, shortName(x)))
          // a pinned run keeps its chart colour on the border while it is live
          : h('a', { href: runHash(x.id, sub), class: 'chip live', title: runTip(x), ...(state.pinned.has(x.id) ? { style: `border-color:${runColor(x.id)}` } : {}) },
            h('span', { class: 'status live' }, ''), state.pinned.has(x.id) ? 'vs ' : '', h('b', {}, shortName(x)))),
        ...[...state.pinned].filter((p) => state.byId.has(p) && !(state.byId.get(p).status === 'live')).map((o) => o === id
          ? h('span', { class: 'chip here' }, '◆ ', h('b', {}, shortName(state.byId.get(o))))
          : h('a', { href: runHash(o, sub), class: 'chip', style: `border-color:${runColor(o)}`, title: 'pinned run' }, 'vs ', h('b', {}, shortName(state.byId.get(o)))))),
    );
  };
  renderHead();
  state.runHead = renderHead;
  listeners.add((kind, d) => { if (kind === 'run' || (kind === 'records' && d.id === id)) renderHead(); });
  if (sub === 'latent') return viewConstellation(body, id);
  return viewRunDetails(body, id, f, others);
}

async function viewRunDetails(el, id, f, others) {
  // This run is cyan in every chart; pinned runs keep their identity colours.
  const col = seriesColor(0);
  const otherFull = await Promise.all(others.map(loadFull));
  const renderHead = () => state.runHead?.();

  if (f.kind === 'notebook') {
    if (f.images?.length) el.append(h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Live plots from the notebook')),
      h('div', { class: 'grid cols-2' }, ...f.images.map((n) => h('div', { class: 'panel' }, h('span', { class: 'eyebrow' }, n.replace('latest_', '').replace('.png', '')), zoomable(h('img', { src: `/api/runs/${encodeURIComponent(id)}/image/${n}?t=${f.updatedAt}`, alt: n, style: 'width:100%;border-radius:8px' })))))));
    el.append(configSection(f, otherFull));
    return;
  }

  // charts
  const charts = [];
  const trainGrid = h('div', { class: 'grid cols-2' });
  const evalGrid = h('div', { class: 'grid cols-3' });
  const reconSec = h('div', { class: 'section' });   // filled by the recon timeline below; sits right under the losses
  el.append(h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Training losses'), h('span', { class: 'muted' }, 'every 100 steps')), trainGrid),
    reconSec,
    h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Evaluation'), h('span', { class: 'muted' }, `every ${f.config?.eval_every || '?'} steps`)), evalGrid));
  const mk = (grid, opts) => { const c = new LineChart(h('div'), opts); grid.append(c.el); charts.push(c); return c; };
  // The charts of this run's type (RUN_TYPES). A chart stays hidden while it has no data.
  const T = RUN_TYPES[runType(f)];
  const trainCharts = T.train.map((g) => [g, mk(trainGrid, { title: g.title, log: !!g.log })]);
  // Cross-run comparison uses the test metric, not the train `clean` loss: training schemes differ
  // (e.g. DISTORT never noises the clean branch), so train losses are not comparable across runs.
  const cmpChart = mk(trainGrid, { title: others.length ? `${T.testLabel} vs pinned (colours as in the pills above)` : 'throughput (it/s)', log: !!others.length });
  const evalCharts = T.evals.map((g) => [g, mk(evalGrid, { title: g.title, log: !!g.log })]);
  const curveCharts = T.curves.map((g) => [g, mk(evalGrid, { title: g.title, log: !!g.log, xFmt: g.xFmt, xName: g.xName })]);
  state.cleanup.push(() => charts.forEach((c) => c.destroy()));

  const series = (m, k, color, label = k, dash = false) => ({ id: label, label, color, dash, points: m.filter((r) => typeof r[k] === 'number').map((r) => [r.step, r[k]]) });
  const show = (c, s) => { c.setSeries(s); c.el.hidden = !s.some((x) => x.points.length); };
  // one curve of an eval record: [[1, r.<prefix>1], [2, r.<prefix>2], ...]
  const curveOf = (r, prefix) => { const p = []; for (let n = 1; typeof r[`${prefix}${n}`] === 'number'; n++) p.push([n, r[`${prefix}${n}`]]); return p; };
  const updateCharts = () => {
    // g.need: the chart belongs to runs that log that key (the corrector charts share 'clean' and 'total' with every pyramid run)
    for (const [g, c] of trainCharts) {
      const s = g.need && !f.metrics.some((r) => typeof r[g.need] === 'number') ? [] : g.keys.map((k, i) => series(f.metrics, k, seriesColor(i), g.labels?.[k] || k)).filter((x) => x.points.length);
      // the first train chart also draws the pinned runs' train loss (dashed, pill colours): only comparable for the same training scheme
      if (s.length && g === T.train[0]) for (const o of otherFull) s.push({ ...series(o.metrics, g.keys[0], runColor(o.id), `${g.keys[0]} · ${shortName(o)}`, true), hideLegend: true });
      show(c, s);
    }
    // pinned overlays: colour-matched to the jump-to pills above, so no legend entry (hover still names them)
    if (others.length) cmpChart.setSeries([{ ...series(f.metrics, 'test_clean', col, 'test_clean · this run'), dots: true }, ...otherFull.map((o) => ({ ...series(o.metrics, 'test_clean', runColor(o.id), shortName(o)), dots: true, hideLegend: true }))]);
    else cmpChart.setSeries([series(f.metrics, 'sps', seriesColor(0), 'it/s')]);
    for (const [g, c] of evalCharts) {
      const s = g.keys.map((k, i) => ({ ...series(f.metrics, k, seriesColor(i), g.labels?.[k] || k), dots: true })).filter((x) => x.points.length);
      if (s.length) for (const o of otherFull) s.push(...g.keys.slice(0, 1).map((k) => ({ ...series(o.metrics, k, runColor(o.id), `${k} · ${shortName(o)}`, true), dots: true, hideLegend: true })));
      show(c, s);
    }
    for (const [g, c] of curveCharts) {
      const s = [];
      g.families.forEach((fam, i) => {
        const evals = f.metrics.filter((r) => typeof r[`${fam.prefix}1`] === 'number');
        if (!evals.length) return;
        const latest = evals.at(-1);
        if (g.history) {
          // up to 8 evenly spaced earlier evals, dark (early) to light (late); the latest eval in amber
          const n = Math.min(8, evals.length - 1);
          const idx = [...new Set(Array.from({ length: n }, (_, j) => Math.round((j * (evals.length - 2)) / Math.max(1, n - 1))))];
          idx.forEach((j, q) => s.push({ id: `${fam.prefix}${evals[j].step}`, label: `step ${fmtX(evals[j].step)}`, color: depthColor(q, idx.length + 1), width: 1.2, hideLegend: true, points: curveOf(evals[j], fam.prefix) }));
          s.push({ id: `${fam.prefix}latest`, label: `${fam.label} · step ${fmtX(latest.step)}`, color: ACCENT, width: 2.5, dots: true, points: curveOf(latest, fam.prefix) });
        } else s.push({ id: fam.prefix, label: `${fam.label} · step ${fmtX(latest.step)}`, color: seriesColor(i), dots: true, points: curveOf(latest, fam.prefix) });
      });
      show(c, s);
    }
  };
  updateCharts();

  // Timelines of the sample grids. One for the grids from the trained weights (recon_*.png). A run that
  // also writes grids from the averaged weights (emaprev_*.png: same seeds, same layout) gets a second
  // one directly below. Both show the same step: moving one slider moves the other.
  const curRun = () => state.byId.get(id) || f;
  let selStep = null;   // the selected step, shared by the timelines; null = follow the newest grid
  const timelines = [];
  const mkTimeline = (sec, { title, note, field, route, alt, empty }) => {
    const tl = h('div', { class: 'panel timeline' });
    const head = h('h2', {}, title);
    sec.append(h('div', { class: 'section-head' }, head, h('span', { class: 'muted' }, note)), tl);
    const img = zoomable(h('img', { alt }));
    const slider = h('input', { type: 'range', min: 0, max: 0, value: 0 });
    const label = h('div', { class: 'row spread mono', style: 'font-size:12px' });
    tl.append(img, slider, label);
    const render = () => {
      const r = curRun(); const steps = r[field] || [];
      if (!steps.length) {
        img.hidden = true; slider.hidden = true;
        label.replaceChildren(h('span', { class: 'muted' }, r.status === 'live' ? 'no snapshots yet' : empty));
        return;
      }
      img.hidden = false; slider.hidden = false;
      let i = steps.length - 1;   // the newest grid, or the grid nearest to the selected step
      if (selStep !== null) for (let j = 0; j < steps.length; j++) if (Math.abs(steps[j] - selStep) < Math.abs(steps[i] - selStep)) i = j;
      slider.max = steps.length - 1; slider.value = i;
      const src = `/api/runs/${encodeURIComponent(id)}/${route}/${steps[i]}`;
      if (img.getAttribute('src') !== src) img.src = src;
      const kind = field === 'recons' ? r.reconKinds?.[steps[i]] : null;
      label.replaceChildren(h('span', {}, `step ${fmtX(steps[i])}`,
        kind === 'reeval' ? h('span', { class: 'badge info', style: 'margin-left:8px' }, 're-rendered from checkpoint') : null,
        selStep !== null && steps[i] !== selStep ? h('span', { class: 'muted', style: 'margin-left:8px' }, `no grid at step ${fmtX(selStep)}: nearest shown`) : null),
      h('span', { class: 'muted' }, `${i + 1} / ${steps.length}`));
    };
    slider.addEventListener('input', () => {
      const steps = curRun()[field] || [];
      selStep = +slider.value === +slider.max ? null : steps[+slider.value];
      timelines.forEach((t) => t.render());
    });
    const t = { head, render };
    timelines.push(t);
    return t;
  };
  const rawTimeline = mkTimeline(reconSec, { title: T.reconTitle, note: T.reconNote, field: 'recons', route: 'recon', alt: 'reconstruction',
    empty: 'no reconstruction grids in this run folder yet' });
  let emaSec = null;
  // A live run can write its first EMA grid while the page is open: the second timeline is made then.
  const showRecons = () => {
    if (!emaSec && curRun().emaRecons?.length) {
      const decay = f.metrics.findLast((r) => typeof r.ema_preview_decay === 'number')?.ema_preview_decay ?? 0.9998;
      rawTimeline.head.textContent = `${T.reconTitle}: trained weights`;
      emaSec = h('div', { class: 'section' });
      reconSec.after(emaSec);
      mkTimeline(emaSec, { title: `${T.reconTitle}: averaged weights (EMA ${decay})`, note: 'same seeds and step as the grid above · the two sliders move together',
        field: 'emaRecons', route: 'emarecon', alt: 'samples from the averaged weights', empty: 'no snapshots' });
    }
    timelines.forEach((t) => t.render());
  };
  showRecons();

  // latent-noise robustness, as the notebook's plot_z_robustness: recon MSE when z is lerped
  // toward fixed Gaussian noise (t = 0 … 0.95), on fixed test images, at every eval step.
  const robSec = h('div', { class: 'section' });
  // AE sections: shown for AE runs, and for any run that has such data
  const robWanted = () => !!(T.robust || (f.robust || []).length);
  robSec.hidden = !robWanted();
  el.append(robSec);
  const robCurve = new LineChart(h('div'), { title: 'recon MSE vs latent noise t (dark = early, light = late, amber = selected step)', log: true, xFmt: (v) => v.toFixed(2), xName: 'noise' });
  const robTime = new LineChart(h('div'), { title: 'recon MSE at each noise level over training (dark = low noise)', log: true });
  // edge sharpness (t = 0 decode): MSE on the outer 12px ring / MSE on the interior; 1 = edges as good as the middle
  const edgeChart = new LineChart(h('div'), { title: 'edge / interior recon MSE (outer 12px ring; 1 = edges as sharp as the middle)' });
  charts.push(robCurve, robTime, edgeChart);
  const robImg = zoomable(h('img', { alt: 'robustness strip' }));
  const robSlider = h('input', { type: 'range', min: 0, max: 0, value: 0 });
  const robLabel = h('div', { class: 'row spread mono', style: 'font-size:12px' });
  const robBody = h('div', {}, h('div', { class: 'grid cols-2' }, robCurve.el, robTime.el, edgeChart.el),
    h('div', { class: 'panel timeline', style: 'margin-top:12px' }, robImg, robSlider, robLabel));
  const robEmpty = h('div', { class: 'panel muted' }, 'No robustness sweeps for this run yet. Runs started before the sweep was added (2026-09-27) get them from their checkpoints via robust_backfill.py, one per checkpoint.');
  robSec.append(
    h('div', { class: 'section-head' }, h('h2', {}, 'Latent noise robustness'), h('span', { class: 'muted' }, 'decode lerp(z, noise, t) · recon MSE vs input · drag to pick a step')),
    robEmpty, robBody);
  let robSel = null;   // index into the sweep list; null = follow the latest
  const sweeps = (run) => {   // one sweep per step (a live sweep wins over a checkpoint backfill)
    const by = new Map();
    for (const r of run.robust || []) if (r.mse?.length && !(by.has(r.step) && r.source === 'ckpt')) by.set(r.step, r);
    return [...by.values()].sort((a, b) => a.step - b.step);
  };
  const curve = (r) => r.fracs.map((t, k) => [t, r.mse[k]]);
  const renderRobust = () => {
    const R = sweeps(f);
    robEmpty.hidden = !!R.length; robBody.hidden = !R.length;
    if (!R.length) return;
    const i = robSel == null ? R.length - 1 : Math.min(robSel, R.length - 1);
    // ≤12 evenly spaced steps as a dark→light ramp, plus the selected step in amber
    const n = Math.min(12, R.length);
    const idx = [...new Set(Array.from({ length: n }, (_, j) => Math.round((j * (R.length - 1)) / Math.max(1, n - 1))))].filter((j) => j !== i);
    const s = idx.map((j, q) => ({ id: `s${R[j].step}`, label: `step ${fmtX(R[j].step)}`, color: depthColor(q, idx.length + 1), width: 1.2, hideLegend: true, points: curve(R[j]) }));
    for (const o of otherFull) { const Ro = sweeps(o); if (Ro.length) s.push({ id: `o${o.id}`, label: `${shortName(o)} · step ${fmtX(Ro.at(-1).step)}`, color: runColor(o.id), dash: true, dots: true, hideLegend: true, points: curve(Ro.at(-1)) }); }
    s.push({ id: `sel${R[i].step}`, label: `step ${fmtX(R[i].step)}`, color: ACCENT, width: 2.5, dots: true, points: curve(R[i]) });
    robCurve.setSeries(s);
    robTime.setSeries(R[0].fracs.map((t, k) => ({ id: `t${t}`, label: `t ${t.toFixed(2)}`, color: depthColor(k, R[0].fracs.length), dots: R.length <= 20, points: R.map((r) => [r.step, r.mse[k]]) })));
    const edge = (rs) => rs.filter((r) => typeof r.edge_ratio === 'number').map((r) => [r.step, r.edge_ratio]);
    edgeChart.setSeries([{ id: 'edge', label: 'edge ratio · this run', color: seriesColor(0), dots: true, points: edge(R) },
      ...otherFull.map((o) => ({ id: `e${o.id}`, label: shortName(o), color: runColor(o.id), dash: true, dots: true, hideLegend: true, points: edge(sweeps(o)) }))]);
    edgeChart.el.hidden = !edge(R).length && !otherFull.some((o) => edge(sweeps(o)).length);
    robSlider.max = R.length - 1; robSlider.value = i;
    // strip PNG: the selected step, else the nearest earlier step that has one
    const pngs = (state.byId.get(id) || f).robustPngs || [];
    const png = [...pngs].reverse().find((p) => p <= R[i].step);
    robImg.hidden = png == null;
    if (png != null) robImg.src = `/api/runs/${encodeURIComponent(id)}/robust/${png}`;
    robLabel.replaceChildren(h('span', {}, `step ${fmtX(R[i].step)}`, R[i].source === 'ckpt' ? h('span', { class: 'badge info', style: 'margin-left:8px' }, 'from checkpoint') : null),
      h('span', { class: 'muted' }, R[i].mse.map((m, k) => `t${R[i].fracs[k]}: ${fmt(m)}`).join('  ')));
  };
  robSlider.addEventListener('input', () => { robSel = +robSlider.value === +robSlider.max ? null : +robSlider.value; renderRobust(); });
  renderRobust();

  // diag: per-block attention routing at the latest diag step
  const diagWrap = h('div', { class: 'panel' });
  const renderDiag = () => {
    const d = f.diag.at(-1);
    if (!d) { diagWrap.replaceChildren(h('span', { class: 'muted' }, 'no diagnostics recorded for this run')); return; }
    const blocks = [...new Set(Object.keys(d).filter((k) => /^b\d+_/.test(k)).map((k) => k.split('_')[0]))].sort((a, b) => +a.slice(1) - +b.slice(1));
    diagWrap.replaceChildren(
      h('div', { class: 'row spread' }, h('div', {}, h('span', { class: 'eyebrow' }, `decoder attention routing · step ${fmtX(d.step)}`), h('h3', {}, 'where each block\'s patch tokens attend: ', h('i', { style: `color:${ROUTE.z}` }, 'z'), ' · ', h('i', { style: `color:${ROUTE.slot}` }, 'slots'), ' · ', h('i', { style: `color:${ROUTE.patch}` }, 'patches'))),
        h('span', { class: 'muted mono', style: 'font-size:11px' }, 'patch img-sd = how much the patch stream differs across images (0 → input-independent)')),
      h('div', { style: 'height:10px' }),
      h('div', { class: 'diag-blocks' }, ...blocks.map((b) => {
        const z = d[`${b}_att_z`] ?? 0, s = d[`${b}_att_slot`] ?? 0, p = d[`${b}_att_patch`] ?? 0, tot = z + s + p || 1;
        return h('div', { class: 'block' }, h('span', {}, b), h('div', { class: 'bar' }, h('i', { style: `width:${(100 * z) / tot}%;background:${ROUTE.z}` }), h('i', { style: `width:${(100 * s) / tot}%;background:${ROUTE.slot}` }), h('i', { style: `width:${(100 * p) / tot}%;background:${ROUTE.patch}` })),
          h('span', {}, `img-sd ${fmt(d[`${b}_patch_imgsd`])}`));
      })),
    );
  };
  const diagChart = mk(h('div'), { title: 'patch-stream image dependence per block (dark = early, light = late)', log: true });
  const renderDiagChart = () => {
    const d0 = f.diag[0]; if (!d0) return;
    const blocks = [...new Set(Object.keys(d0).filter((k) => /_patch_imgsd$/.test(k)).map((k) => k.split('_')[0]))];
    diagChart.setSeries(blocks.map((b, i) => series(f.diag, `${b}_patch_imgsd`, depthColor(i, blocks.length), b)));
  };
  // z drift: ||z(step) - z(previous diag)|| / ||z(previous diag)|| on a fixed batch, i.e. latent
  // churn per diagnostics interval. Intervals differ between runs, so the title states this run's.
  const every = f.config?.diag_every;
  const driftChart = mk(h('div'), { title: `z drift — relative change of z per diag interval${every ? ` (${fmtX(every)} steps)` : ''}` });
  const renderDriftChart = () => {
    const s = [series(f.diag, 'z_drift', seriesColor(0), 'z drift · this run')];
    for (const o of otherFull) s.push({ ...series(o.diag || [], 'z_drift', runColor(o.id), `z drift · ${shortName(o)}`, true), hideLegend: true });
    driftChart.setSeries(s);
  };
  renderDiag(); renderDiagChart(); renderDriftChart();
  const diagSec = h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Diagnostics')), h('div', { class: 'grid cols-2' }, diagWrap, diagChart.el, driftChart.el));
  const diagWanted = () => !!(T.diag || (f.diag || []).length);
  diagSec.hidden = !diagWanted();
  el.append(diagSec);
  el.append(configSection(f, otherFull));

  listeners.add((kind, d) => {
    if (kind === 'records' && d.id === id) { renderHead(); updateCharts(); robSec.hidden = !robWanted(); diagSec.hidden = !diagWanted(); if (d.diag.length) { renderDiag(); renderDiagChart(); renderDriftChart(); } if (d.robust?.length) renderRobust(); }
    if (kind === 'run') { renderHead(); showRecons(); renderRobust(); }
  });
}

function configSection(f, others = []) {
  const c = f.config || {}; if (!Object.keys(c).length) return h('div');
  const base = others[0]?.config;
  return h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Config'),
    f.configSource === 'queue.txt' ? h('span', { class: 'badge info' }, 'rebuilt from queue.txt — config.json was archived') : null,
    base ? h('span', { class: 'muted' }, `amber = differs from ${shortName(others[0])}`) : null),
    h('div', { class: 'panel kv' }, ...Object.entries(c).filter(([, v]) => typeof v !== 'object').map(([k, v]) => h('div', {}, h('span', {}, k), h('b', { class: base && JSON.stringify(base[k]) !== JSON.stringify(v) ? 'diff' : '' }, String(v))))));
}

// ---------------------------------------------------------------- run pickers
/**
 * Resolve which run a latent viewer shows: the hash's run, else the current
 * run. If the current run has no snapshots, say so instead of silently
 * switching to another run; offer the ones that do.
 */
function pickRun(candidates, id, el, view) {
  const r = state.byId.get(id);
  if (r?.zSteps?.length) return r;
  el.append(h('div', { class: 'panel empty' },
    h('div', {}, h('b', { class: 'mono' }, r?.name || id), ' has no latent snapshots (z_*.pt)', r?.kind === 'notebook' ? ' — notebook runs need the sink cell to save them' : ''),
    h('div', { style: 'height:12px' }), h('span', { class: 'muted' }, 'runs with snapshots:'),
    h('div', { class: 'row', style: 'justify-content:center;margin-top:8px' }, ...candidates.slice(0, 8).map((c) => h('a', { href: runHash(c.id, view), class: 'chip' }, shortName(c))))));
  return null;
}
/** Run <select> with pinned runs in their own group at the top. Call .refresh(currentId) after pinning. */
function runSelect(candidates, currentId) {
  const sel = h('select');
  sel.refresh = (cur) => {
    // several runs can share a title, so the menu keeps the start time beside it
    const opt = (r) => h('option', { value: r.id, selected: r.id === cur ? '' : null }, [runTitle(r), runWhen(r)].filter(Boolean).join('  ·  '));
    const pinned = candidates.filter((r) => state.pinned.has(r.id));
    const rest = candidates.filter((r) => !state.pinned.has(r.id));
    sel.replaceChildren(
      ...(pinned.length ? [h('optgroup', { label: '◆ pinned' }, ...pinned.map(opt)), h('optgroup', { label: 'all runs' }, ...rest.map(opt))] : rest.map(opt)));
  };
  sel.refresh(currentId);
  return sel;
}
/** Pin toggle bound to a run getter; re-renders the select's groups. */
function pinButton(getRun, sel) {
  const b = h('button', { class: 'filter', title: 'pin for comparison', onclick: () => { togglePin(getRun().id); sync(); sel.refresh(getRun().id); state.runHead?.(); } });
  const sync = () => { const on = state.pinned.has(getRun().id); b.classList.toggle('on', on); b.textContent = on ? '◆ pinned' : '◇ pin'; };
  sync(); b.sync = sync;
  return b;
}

// ---------------------------------------------------------------- Latent view (PCA cloud + covariances)
async function viewConstellation(el, id) {
  await loadRuns();
  const candidates = state.runs.filter((r) => r.zSteps?.length);
  if (!candidates.length) { el.append(h('div', { class: 'empty' }, 'No latent snapshots (z_*.pt) found in any run yet.')); return; }
  let run = pickRun(candidates, id, el, 'latent');
  if (!run) return;
  const canvas = h('canvas');
  const overlay = h('div', { class: 'latent-overlay' });
  const stage = h('div', { class: 'latent-stage' }, canvas, overlay, h('div', { class: 'latent-hint' }, isTouch() ? 'tap to expand' : 'drag to orbit · wheel to zoom · double-click to reset'));
  const side = h('div', { class: 'latent-side' });
  el.append(h('div', { class: 'latent-layout' }, stage, side));
  // covariance matrices under the cloud: how the image-to-image variation co-varies across
  // tokens (T×T) and across channels (C×C); a whitened latent reads as a bright diagonal
  const covReadout = h('span', { class: 'mono muted cov-readout' });
  const mkCov = (title, sub) => {
    const canvas = h('canvas', { class: 'cov-canvas' });
    const cap = h('div', { class: 'mono muted cov-cap' });
    return { canvas, cap, el: h('div', { class: 'panel cov-panel' }, h('div', { class: 'row spread' }, h('b', {}, title), h('span', { class: 'muted' }, sub)), canvas, cap) };
  };
  const tokCov = mkCov('token covariance', 'T × T · over images & channels');
  const chanCov = mkCov('channel covariance', 'C × C · over images & tokens');
  el.append(h('div', { class: 'section cov-section' },
    h('div', { class: 'section-head spread' }, h('h2', {}, 'Covariance'), covReadout),
    h('div', { class: 'grid cov-grid' }, tokCov.el, chanCov.el)));
  const drawCovs = async (step) => {
    try {
      const c = await fetchCov(run.id, step);
      for (const [box, M, n, label] of [[tokCov, c.tok, c.tokens, 'token'], [chanCov, c.chan, c.channels, 'channel']]) {
        const st = drawCov(box.canvas, M, n, covReadout, label);
        box.cap.textContent = `step ${fmtX(step)} · mean diag ${st.meanDiag.toFixed(2)} · mean |off-diag| ${st.meanOff.toFixed(3)} (${(100 * st.ratio).toFixed(1)}% of diag)`;
      }
    } catch (e) { tokCov.cap.textContent = chanCov.cap.textContent = `covariance: ${e.message}`; }
  };
  let view;
  try { view = new LatentView(canvas); } catch (e) { stage.replaceChildren(h('div', { class: 'empty' }, `WebGL2 unavailable: ${e.message}`)); return; }
  fullscreenStage(stage, view);
  restoreCamera(view, 'tv.pca.cam');
  state.cleanup.push(() => { saveCamera(view, 'tv.pca.cam'); view.destroy(); });

  const runSel = runSelect(candidates, run.id);
  const pinBtn = pinButton(() => run, runSel);
  const stepSlider = h('input', { type: 'range', min: 0, max: run.zSteps.length - 1, value: run.zSteps.length - 1 });
  const stepLabel = h('span', { class: 'mono' });
  const play = h('button', { class: 'play' }, '▶ play');
  const modeSeg = h('div', { class: 'seg' }, ...[['token', 'hue = token index'], ['channel', 'hue = channel with the largest |z| in that token vector'], ['norm', 'brightness ramp = token vector length']].map(([m, tip], i) => h('button', { class: i === view.mode ? 'on' : '', title: tip, onclick: (e) => { view.mode = i; setPref('tv.pca.mode', i); for (const b of modeSeg.children) b.classList.remove('on'); e.target.classList.add('on'); } }, m)));
  view.size = pref('tv.pca.size', 2.2); view.mode = pref('tv.pca.mode', 0);
  const size = h('input', { type: 'range', min: 0.6, max: 5, step: 0.1, value: view.size, oninput: (e) => { view.size = +e.target.value; setPref('tv.pca.size', view.size); } });
  const focusLabel = h('span', { class: 'mono muted' }, 'all images');
  // the focused image itself (runs that save z_images.pt, or harness runs via the STL-10 test file)
  const thumb = h('img', { class: 'focus-thumb', alt: 'focused image' });
  thumb.hidden = true;
  thumb.addEventListener('error', () => { thumb.hidden = true; setTok(-1); });
  // Patch <-> token: the picture is cut into a g × g grid, g = sqrt(tokens), row-major from the
  // top left. Hovering a patch marks its token point in the cloud; hovering a token point of the
  // focused image marks its patch. A token count that is not a perfect square switches this off.
  const TOK_TIP = 'token index, row and column all count from 0; row 0 is the top row, column 0 is the left column';
  const patchRing = h('div', { class: 'patch-ring' });
  patchRing.hidden = true;
  const tokReadout = h('span', { class: 'mono muted', title: TOK_TIP });
  tokReadout.hidden = true;
  const thumbBox = h('div', { class: 'focus-pic' }, thumb, patchRing);
  const grid = () => { const g = Math.round(Math.sqrt(view.tokens)); return view.count && g > 1 && g * g === view.tokens ? g : 0; };
  // the displayed picture inside the <img> border box, in layout px (object-fit: cover, centred)
  const picBox = () => {
    const bw = thumb.clientWidth, bh = thumb.clientHeight, nw = thumb.naturalWidth || bw, nh = thumb.naturalHeight || bh;
    const s = Math.max(bw / nw, bh / nh);
    return { x: thumb.clientLeft + (bw - nw * s) / 2, y: thumb.clientTop + (bh - nh * s) / 2, w: nw * s, h: nh * s };
  };
  const setTok = (t) => {
    const g = grid(), on = g > 0 && t >= 0 && t < g * g && !thumb.hidden && view.focusImage >= 0;
    view.hoverToken = on ? t : -1;
    patchRing.hidden = !on;
    tokReadout.hidden = thumb.hidden;
    if (!on) { tokReadout.textContent = g ? 'point at a patch to see its token' : `${view.tokens} tokens: not a square grid, no patch map`; return; }
    const row = Math.floor(t / g), col = t % g, b = picBox();
    Object.assign(patchRing.style, { left: `${thumb.offsetLeft + b.x + col * b.w / g}px`, top: `${thumb.offsetTop + b.y + row * b.h / g}px`, width: `${b.w / g}px`, height: `${b.h / g}px` });
    tokReadout.textContent = `token ${t} · row ${row}, column ${col}`;
  };
  const tokAt = (e) => {
    const g = grid(); if (!g) return -1;
    const r = thumb.getBoundingClientRect(), b = picBox();
    const kx = r.width / (thumb.offsetWidth || 1) || 1, ky = r.height / (thumb.offsetHeight || 1) || 1;   // CSS transforms, if any
    const col = Math.floor(((e.clientX - r.left) / kx - b.x) / b.w * g), row = Math.floor(((e.clientY - r.top) / ky - b.y) / b.h * g);
    return col < 0 || col >= g || row < 0 || row >= g ? -1 : row * g + col;
  };
  thumb.addEventListener('pointermove', (e) => setTok(tokAt(e)));
  thumb.addEventListener('pointerdown', (e) => setTok(tokAt(e)));          // touch: a tap marks the patch
  thumb.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') setTok(-1); });
  const tapAway = (e) => { if (e.pointerType === 'touch' && e.target !== thumb) setTok(-1); };   // ... until the next tap elsewhere
  document.addEventListener('pointerdown', tapAway);
  state.cleanup.push(() => document.removeEventListener('pointerdown', tapAway));
  view.onHoverToken = (t) => setTok(t);
  const setFocus = (i) => {
    view.focusImage = i; focus.value = i;
    focusLabel.textContent = i < 0 ? 'all images' : `image #${i}`;
    thumb.hidden = i < 0 || !run.zImages;
    if (!thumb.hidden) thumb.src = `/api/runs/${encodeURIComponent(run.id)}/zimg/${i}`;
    setTok(-1);
  };
  const focus = h('input', { type: 'range', min: -1, max: 1023, step: 1, value: -1, oninput: (e) => setFocus(+e.target.value) });
  // click a spot in the cloud: focus the image whose tokens centre there; empty space clears it
  view.onPick = (i) => setFocus(i);
  // What the 3-D view draws: latent token points and image sprites, each with its own switch
  // (remembered per browser). Images are drawn at the centre of their tokens.
  const check = (label, on) => { const box = h('input', { type: 'checkbox' }); box.checked = on; return { box, el: h('label', { class: 'control-check' }, box, ` ${label}`) }; };
  const ptsToggle = check('latent points', pref('tv.pca.showPoints', 1) > 0);
  const imgToggle = check('images (at the centre of their tokens)', false);
  view.showPoints = ptsToggle.box.checked;
  ptsToggle.box.addEventListener('change', () => { view.showPoints = ptsToggle.box.checked; setPref('tv.pca.showPoints', view.showPoints ? 1 : 0); });
  const imgSize = h('input', { type: 'range', min: 0.04, max: 0.5, step: 0.01, value: pref('tv.pca.imgSize', 0.14), 'aria-label': 'image size' });
  view.imageSize = +imgSize.value;
  imgSize.addEventListener('input', () => { view.imageSize = +imgSize.value; setPref('tv.pca.imgSize', view.imageSize); });
  const imgSizeRow = h('div', { class: 'control' }, 'image size', imgSize);
  imgToggle.el.hidden = imgSizeRow.hidden = !run.zImages;
  const showCtl = h('div', { class: 'control' }, 'show', ptsToggle.el, imgToggle.el);
  let atlasFor = null;
  const setImages = async (on) => {
    imgToggle.box.checked = on; view.showImages = on; setPref('tv.pca.showImages', on ? 1 : 0);
    imgSizeRow.hidden = !on || !run.zImages;
    if (!on || atlasFor === run.id) return;
    try {
      const r = await fetch(`/api/runs/${encodeURIComponent(run.id)}/zatlas/48`);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
      const n = +r.headers.get('x-count'), cols = +r.headers.get('x-cols');
      view.setImageAtlas(await createImageBitmap(await r.blob()), n, cols);
      atlasFor = run.id;
    } catch (err) { toast(`images: ${err.message}`); imgToggle.box.checked = false; view.showImages = false; imgSizeRow.hidden = true; }
  };
  imgToggle.box.addEventListener('change', () => setImages(imgToggle.box.checked));
  if (run.zImages && pref('tv.pca.showImages', 0) > 0) setImages(true);   // restore the last choice
  else imgSizeRow.hidden = true;
  localStorage.removeItem('tv.pca.dimAlpha'); // was a tuning slider; 40% is now the default
  // Which three principal components span the 3-D view: 1–3 (strongest), 4–6, 7–9, ...
  // The server clamps to what the snapshot has and reports it back as header.pcFrom / header.maxPc.
  let pcFrom = pref('tv.pca.from', 0);
  const pcName = (f) => `PC${f + 1}–${f + 3}`;
  const pcSel = h('select', { 'aria-label': 'principal components shown' });
  const setPcOptions = (maxPc) => {
    const groups = Math.max(1, Math.floor(maxPc / 3));
    if (pcSel.options.length === groups) { pcSel.value = String(pcFrom); return; }
    pcSel.replaceChildren(...Array.from({ length: groups }, (_, g) => h('option', { value: String(3 * g) }, pcName(3 * g))));
    pcSel.value = String(pcFrom);
  };
  setPcOptions(pcFrom + 3);
  pcSel.addEventListener('change', () => { pcFrom = +pcSel.value; setPref('tv.pca.from', pcFrom); load(+stepSlider.value); });
  const varTitle = h('span', { class: 'eyebrow' }, `explained variance (${pcName(pcFrom)})`);
  const varBars = h('div', { class: 'var-bars' });
  const info = h('div', { class: 'kv' });
  side.append(
    h('div', { class: 'panel', style: 'display:flex;flex-direction:column;gap:12px' },
      h('div', { class: 'control' }, 'switch run', runSel, h('div', { class: 'control-row' }, pinBtn)),
      h('div', { class: 'control' }, h('div', { class: 'row spread' }, 'snapshot step', stepLabel), stepSlider, h('div', { class: 'row' }, play)),
      h('div', { class: 'control', title: 'Lower groups show weaker directions. Each group is rescaled to unit RMS radius, so its % of variance is the only measure of its strength.' }, 'principal components', pcSel),
      h('div', { class: 'control' }, 'colour by', modeSeg),
      h('div', { class: 'control' }, 'point size', size),
      showCtl,
      imgSizeRow,
      h('div', { class: 'control' }, h('div', { class: 'row spread' }, 'focus one image', focusLabel), focus, thumbBox, tokReadout),
    ),
    h('div', { class: 'panel' }, varTitle, varBars, h('div', { style: 'height:10px' }), info),
    h('div', { class: 'panel muted', style: 'font-size:12px;line-height:1.5' }, 'Each point is one latent token of one eval image (N × T of them), projected from C channels to 3-D by PCA (PCs 1–3 by default; pick a weaker group above) and rescaled to unit RMS radius. A healthy latent fills a blob; rank collapse flattens it to a sheet or a line. Decoder collapse leaves it healthy while reconstructions go grey — the badge and the run page cover that side.'),
  );

  let loading = null, playing = false, timer = null;
  const PLAY_MS = 1600;
  // Small snapshot cache so play mode can prefetch the next step: network time then never
  // shows up as an extra pause (noticeable on the phone over the private network).
  const cache = new Map();
  const getPca = (step) => {
    const key = `${run.id}@${step}#${pcFrom}`;
    if (!cache.has(key)) {
      cache.set(key, fetchPca(run.id, step, pcFrom).catch((e) => { cache.delete(key); throw e; }));
      while (cache.size > 6) cache.delete(cache.keys().next().value);
    }
    return cache.get(key);
  };
  const load = async (i) => {
    const step = run.zSteps[i]; stepLabel.textContent = `step ${fmtX(step)}`;
    const want = `${step}#${pcFrom}`;
    if (loading === want) return; loading = want;
    if (playing) getPca(run.zSteps[(i + 1) % run.zSteps.length]).catch(() => {});
    try {
      const { xyz, norms, chans, header } = await getPca(step);
      if (loading !== want) return;
      const f = header.pcFrom ?? 0;
      pcFrom = f;   // the server may have clamped it (fewer channels than the remembered group)
      setPcOptions(header.maxPc ?? 3);
      varTitle.textContent = `explained variance (${pcName(f)})`;
      view.setSnapshot(xyz, norms, header, chans);
      setTok(view.hoverToken);   // the token count can change with the snapshot
      drawCovs(step);
      focus.max = header.n - 1;
      const full = await loadFull(run.id).catch(() => null);
      varBars.replaceChildren(...header.explained.map((e, k) => h('div', {}, h('span', {}, `PC${f + k + 1}`), h('i', { style: `width:${clamp(100 * e, 0, 100)}%` }), h('span', {}, `${(100 * e).toFixed(1)}%`))));
      info.replaceChildren(...Object.entries({ images: header.n, tokens: header.tokens, channels: header.channels, points: header.n * header.tokens, 'rms scale': view.rms.toExponential(2), dtype: header.dtype }).map(([k, v]) => h('div', {}, h('span', {}, k), h('b', {}, v))));
      overlay.replaceChildren(h('span', {}, runTitle(run)), h('b', {}, `step ${fmtX(step)}`), h('span', { class: 'muted' }, `${(header.n * header.tokens).toLocaleString()} tokens · ${pcName(f)} explain ${(100 * header.explained.reduce((a, b) => a + b, 0)).toFixed(0)}%`),
        h('div', { class: 'row', style: 'margin-top:6px' }, ...(full ? badgesAtStep(full, step) : [])));
    } catch (e) { toast(`constellation: ${e.message}`); }
  };
  stepSlider.addEventListener('input', () => load(+stepSlider.value));
  play.addEventListener('click', () => {
    playing = !playing; play.classList.toggle('on', playing); play.textContent = playing ? '■ stop' : '▶ play';
    clearInterval(timer);
    // while playing, each morph spans the whole interval so motion never stalls between steps
    view.morphDur = playing ? PLAY_MS : 900;
    if (playing) {
      getPca(run.zSteps[(+stepSlider.value + 1) % run.zSteps.length]).catch(() => {});
      timer = setInterval(() => { const n = run.zSteps.length; stepSlider.value = (+stepSlider.value + 1) % n; load(+stepSlider.value); }, PLAY_MS);
    }
  });
  state.cleanup.push(() => clearInterval(timer));
  runSel.addEventListener('change', () => { location.hash = runHash(runSel.value, 'latent'); });
  load(+stepSlider.value);
  listeners.add((kind) => {
    if (kind !== 'run') return;
    const fresh = state.byId.get(run.id); if (!fresh || fresh.zSteps.length === run.zSteps.length) return;
    const atEnd = +stepSlider.value === +stepSlider.max;
    run = fresh; stepSlider.max = run.zSteps.length - 1;
    if (atEnd && !playing) { stepSlider.value = stepSlider.max; load(+stepSlider.value); }
  });
}

// ---------------------------------------------------------------- boot
gpu.mountStrip($('#gpu-strip'));
fetch('/api/gpu').then((r) => r.json()).then((g) => { if (g.latest) gpu.update(g.latest); }).catch(() => {});
connectSSE();
route();
