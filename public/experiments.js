// Experiments view: groups of directly comparable runs (MLflow's "experiment"), read from
// /api/experiments (experiments.json + per-run scores). Phone first: a card list, and per
// experiment a short header, a runs × score-steps table and one chart of the chosen metric.
//   #experiments          → list
//   #experiment/<id>      → one experiment (?m=<metric> is not used; the metric switch is per view)

import { LineChart } from './charts.js';
import { NEUTRAL, PIN_COLORS } from './palette.js';
import { reportsFor } from './reports.js';

const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v != null && v !== false) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
};
const runHash = (id) => `#run/${encodeURIComponent(id)}/details`;
export const expHash = (id) => `#experiment/${encodeURIComponent(id)}`;
const kStep = (s) => (s >= 1000 ? `${+(s / 1000).toFixed(1)}k` : String(s));
const pref = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const setPref = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } };

// A metric is "lower is better", or, with `real`, "closer to the real-image value is better". kid_ema is stored ×1000 already.
// The real values: 2,400 ImageNet validation images at the jobs' 128 px resize (see PIXEL_EVAL in app.js).
export const METRICS = [
  { key: 'fid_ema', label: 'EMA FID' },
  { key: 'fid', label: 'raw FID' },
  { key: 'kid_ema', label: 'KID ×1000' },
  { key: 'fd_ae_z_ema', label: 'fd_ae z' },
  { key: 'test_clean', label: 'test loss', log: true },
  { key: 'probe_err', label: 'probe top-1 error' },
  { key: 'ema_prev_contrast', label: 'contrast (pixel std)', real: 0.270 },
  { key: 'ema_prev_dark', label: 'black pixels', real: 0.031 },
  { key: 'ema_prev_light', label: 'white pixels', real: 0.027 },
];
const metricInfo = (key) => METRICS.find((m) => m.key === key) || METRICS[0];
/** Badness of a value: the value itself, or its distance to the real-image value. */
export const badness = (m, v) => (m.real == null ? v : Math.abs(v - m.real));
/** Gap class of run vs baseline: 'better' / 'worse' / ''. */
export function gapClass(m, v, b) {
  const d = badness(m, v) - badness(m, b);
  return d < 0 ? 'better' : d > 0 ? 'worse' : '';
}

export function fmtScore(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  return a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(3);
}
export function fmtGap(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v), s = v > 0 ? '+' : v < 0 ? '−' : '±';
  return s + (a >= 1 ? a.toFixed(1) : a.toPrecision(2));
}

/** step → value map of one run for a metric. */
export const valuesOf = (r, key) => new Map(r.scores?.[key] || []);
export const baselineOf = (e) => e.runs.find((r) => r.role === 'baseline') || e.runs.find((r) => r.run === e.baseline) || null;

/** The metric an experiment is judged by: its primary_metric when the runs have it, else the first metric with data. */
export function primaryKey(e) {
  const has = (k) => e.runs.some((r) => r.scores?.[k]?.length);
  if (METRICS.some((m) => m.key === e.primary_metric) && has(e.primary_metric)) return e.primary_metric;
  return METRICS.map((m) => m.key).find(has) || (METRICS.some((m) => m.key === e.primary_metric) ? e.primary_metric : 'fid_ema');
}

/** Gap of run r to the baseline at their last common step: {step, gap} or null. */
export function gapAtLastCommon(r, base, key) {
  if (!base || r === base) return null;
  const a = valuesOf(r, key), b = valuesOf(base, key);
  let step = -1;
  for (const s of a.keys()) if (b.has(s) && s > step) step = s;
  return step < 0 ? null : { step, gap: a.get(step) - b.get(step) };
}

/** Headline of an experiment: best non-baseline value at the last step the baseline shares with any run, and its gap. */
export function headline(e, key = primaryKey(e)) {
  const base = baselineOf(e);
  const others = e.runs.filter((r) => r !== base && r.scores?.[key]?.length);
  if (!others.length) return null;
  const bv = base ? valuesOf(base, key) : new Map();
  let S = -1;
  for (const r of others) for (const s of valuesOf(r, key).keys()) if (bv.has(s) && s > S) S = s;
  if (S < 0) for (const r of others) S = Math.max(S, ...valuesOf(r, key).keys());
  let best = null;
  for (const r of others) { const v = valuesOf(r, key).get(S); if (v != null && (!best || v < best.value)) best = { run: r, value: v }; }
  if (!best) return null;
  return { ...best, step: S, key, gap: bv.has(S) ? best.value - bv.get(S) : null };
}

/** Treatment equal to the baseline at every common step (relative 5e-4, "to 3 digits"): the treatment was probably not applied. */
export function identicalToBaseline(r, base, key) {
  if (!base || r === base) return false;
  const a = valuesOf(r, key), b = valuesOf(base, key);
  let n = 0;
  for (const [s, v] of a) if (b.has(s)) { n++; const w = b.get(s); if (Math.abs(v - w) > 5e-4 * Math.max(Math.abs(v), Math.abs(w), 1e-12)) return false; }
  return n > 0;
}

/** Arms of an experiment: runs whose names differ only in the seed. True when every arm has one seed. */
export function singleSeed(e) {
  const arms = new Map();
  for (const r of e.runs) {
    const arm = (r.name || r.run).replace(/(^|[-_])seed\d+/, '$1seed#').replace(/--[0-9a-f]{20,}$/, '');
    arms.set(arm, (arms.get(arm) || 0) + 1);
  }
  return [...arms.values()].every((n) => n <= 1);
}

/** Table columns: the registry's score steps for the primary metric, else the steps with a value (at most 10). */
export function columnsFor(e, key) {
  if (key === e.primary_metric && e.score_steps?.length) return [...e.score_steps].sort((a, b) => a - b);
  const steps = [...new Set(e.runs.flatMap((r) => (r.scores?.[key] || []).map(([s]) => s)))].sort((a, b) => a - b);
  if (steps.length <= 10) return steps;
  return [...new Set(Array.from({ length: 10 }, (_, i) => steps[Math.round((i * (steps.length - 1)) / 9)]))];
}

/** Colour of each run, in registry order: the baseline grey, the others the pin colours (palette.js) in turn. */
function colorsFor(e) {
  const base = baselineOf(e);
  let i = 0;
  return new Map(e.runs.map((r) => [r, r === base ? NEUTRAL : PIN_COLORS[i++ % PIN_COLORS.length]]));
}

const statusPill = (s) => h('span', { class: `status ${s === 'running' ? 'live' : s}` }, s);
const fmtDate = (d) => (d ? String(d).slice(0, 10) : '–');

async function fetchExperiments() {
  const r = await fetch('/api/experiments');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- list
export async function viewExperiments(el) {
  let data;
  try { data = await fetchExperiments(); } catch (err) { el.append(h('div', { class: 'empty' }, `could not load experiments: ${err.message}`)); return; }
  const all = [...data.experiments].sort((a, b) => String(b.created || '').localeCompare(String(a.created || '')) || (b.updatedAt || 0) - (a.updatedAt || 0));
  let q = pref('tv.exp.q', ''), status = pref('tv.exp.status', '');
  const search = h('input', { class: 'run-search', type: 'search', placeholder: 'search title, question, runs…', value: q, 'aria-label': 'search experiments' });
  const chips = h('div', { class: 'filters', role: 'group', 'aria-label': 'filter by status' });
  const head = h('h2', {});
  const grid = h('div', { class: 'grid cards exp-cards' });
  el.append(h('section', { class: 'section', style: 'margin-top:0' },
    h('div', { class: 'section-head spread', style: 'align-items:center' }, head, data.error ? h('span', { class: 'badge warn' }, `registry: ${data.error}`) : null),
    h('div', { class: 'list-tools' }, search, chips), grid));
  const hay = (e) => [e.title, e.question, e.verdict, e.id, ...e.runs.flatMap((r) => [r.label, r.run, r.note])].filter(Boolean).join(' ').toLowerCase();
  const render = () => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    const list = all.filter((e) => (!status || e.liveStatus === status) && terms.every((t) => hay(e).includes(t)));
    head.textContent = `Experiments (${list.length})`;
    chips.replaceChildren(...['running', 'stale', 'done'].map((s) => h('button', {
      class: `filter ${status === s ? 'on' : ''}`, 'aria-pressed': String(status === s),
      onclick: () => { status = status === s ? '' : s; setPref('tv.exp.status', status); render(); },
    }, `${s} (${all.filter((e) => e.liveStatus === s).length})`)));
    grid.replaceChildren(...(list.length ? list.map(expCard) : [h('div', { class: 'empty' }, 'nothing matches')]));
  };
  search.addEventListener('input', () => { q = search.value; setPref('tv.exp.q', q); render(); });
  render();
}

function expCard(e) {
  const key = primaryKey(e), hl = headline(e, key);
  const machines = [...new Set(e.runs.map((r) => r.machine).filter(Boolean))].join(' + ');
  return h('a', { class: 'card exp-card', href: expHash(e.id) },
    h('div', { class: 'card-title' }, e.title || e.id),
    h('div', { class: 'card-meta' }, statusPill(e.liveStatus), h('span', {}, fmtDate(e.created)), h('span', {}, `${e.runs.length} runs`), machines ? h('span', {}, machines) : null),
    hl ? h('div', { class: 'exp-headline' },
      h('span', { class: 'muted' }, `best ${metricInfo(key).label} `), h('b', { class: 'mono' }, fmtScore(hl.value)),
      hl.gap != null ? h('span', { class: `mono exp-gap ${hl.gap < 0 ? 'better' : hl.gap > 0 ? 'worse' : ''}` }, ` ${fmtGap(hl.gap)} vs baseline`) : null,
      h('span', { class: 'muted' }, ` @${kStep(hl.step)} · ${hl.run.label || hl.run.name}`))
      : h('div', { class: 'muted exp-headline' }, e.primary_metric === 'none' ? 'no score metric' : 'no scores yet'));
}

// ---------------------------------------------------------------- one experiment
/** A field of the header: one line; a long one folds into <details> with its first words as the summary. */
function headerLine(label, text) {
  if (!text) return null;
  const t = String(text);
  if (t.length <= 140) return h('div', { class: 'exp-field' }, h('b', {}, `${label}: `), t);
  return h('details', { class: 'exp-field' }, h('summary', {}, h('b', {}, `${label}: `), `${t.slice(0, 110).replace(/\s+\S*$/, '')}…`), h('p', {}, t));
}

export async function viewExperiment(el, id, cleanup) {
  let data;
  try { data = await fetchExperiments(); } catch (err) { el.append(h('div', { class: 'empty' }, `could not load experiments: ${err.message}`)); return; }
  const e = data.experiments.find((x) => x.id === id);
  if (!e) { el.append(h('div', { class: 'empty' }, `experiment ${id} not found · `, h('a', { href: '#experiments' }, 'all experiments'))); return; }
  const base = baselineOf(e);
  const rows = base ? [base, ...e.runs.filter((r) => r !== base)] : e.runs;
  const colors = colorsFor(e);
  const available = METRICS.filter((m) => e.runs.some((r) => r.scores?.[m.key]?.length));
  let key = primaryKey(e);

  el.append(h('div', { class: 'exp-head' },
    h('a', { href: '#experiments', class: 'muted exp-back' }, '← experiments'),
    h('h1', { class: 'exp-title' }, e.title || e.id),
    h('div', { class: 'card-meta' }, statusPill(e.liveStatus), h('span', {}, fmtDate(e.created)), h('span', {}, `${e.runs.length} runs`),
      h('span', {}, `primary: ${e.primary_metric || '–'}`)),
    headerLine('Question', e.question), headerLine('Falsifier', e.falsifier), headerLine('Verdict', e.verdict),
    // registry `pages`: [{title, href}], write-ups and other sub-pages of this experiment, plus the
    // public/reports.json pages that name it (a page in both shows once, with its reports.json title)
    ...[...new Map([...(e.pages || []), ...(await reportsFor(e.id))].map((p) => [p.href, p])).values()]
      .map((p) => h('p', { class: 'exp-page-link' }, h('a', { href: p.href }, `${p.title} →`))),
    h('p', { class: 'exp-page-link' }, h('a', { href: '#reports', class: 'muted' }, 'all reports →'))));

  const switcher = h('div', { class: 'filters exp-metrics', role: 'group', 'aria-label': 'metric' });
  const tableSec = h('div', {});
  const chartBox = h('div', {});
  el.append(h('section', { class: 'section' }, h('div', { class: 'section-head spread', style: 'align-items:center;flex-wrap:wrap' }, h('h2', {}, 'Results'), switcher), tableSec, chartBox));
  const chart = new LineChart(h('div'), { title: '', height: 220 });
  chartBox.append(chart.el);
  cleanup.push(() => chart.destroy());

  if (e.planned_runs?.length) {
    el.append(h('section', { class: 'section exp-planned' }, h('h2', {}, 'Planned'),
      h('ul', {}, ...e.planned_runs.map((p) => h('li', {}, p.id ? h('a', { href: runHash(p.id) }, p.name) : p.name, p.status ? h('span', { class: 'muted' }, ` · ${p.status}`) : ' · not started')))));
  }
  if (e.sources?.length) el.append(h('p', { class: 'muted exp-sources' }, `Sources: ${e.sources.join(' · ')}`));

  const render = () => {
    const m = metricInfo(key);
    switcher.replaceChildren(...(available.length ? available : [m]).map((x) => h('button', {
      class: `filter ${x.key === key ? 'on' : ''}`, 'aria-pressed': String(x.key === key), onclick: () => { key = x.key; render(); },
    }, x.label)));
    tableSec.replaceChildren(resultsTable(e, rows, base, key, colors), caveats(e, base, key));
    chart.title = `${m.label} vs step`;
    chart.el.querySelector('.chart-title').textContent = `${m.label} vs step · baseline dashed grey`;
    if (chart.log !== !!m.log) chart.logBtn.click();
    const series = rows.map((r) => ({
      id: r.run, label: r.label || r.name, color: colors.get(r), dash: r === base, dots: true,
      points: (r.scores?.[key] || []).map(([s, v]) => [s, v]),
    }));
    if (m.real != null) {
      // the real-image value as a flat dotted line over the steps that have data
      const steps = series.flatMap((s) => s.points.map(([x]) => x));
      if (steps.length) series.push({ id: '__real', label: `real images (${m.real})`, color: NEUTRAL, dash: true, points: [[Math.min(...steps), m.real], [Math.max(...steps), m.real]] });
    }
    chart.setSeries(series);
    chart.el.hidden = !rows.some((r) => r.scores?.[key]?.length);
  };
  render();
}

function resultsTable(e, rows, base, key, colors) {
  const cols = columnsFor(e, key);
  const vals = new Map(rows.map((r) => [r, valuesOf(r, key)]));
  const m = metricInfo(key);
  const best = cols.map((s) => {
    const vs = rows.map((r) => vals.get(r).get(s)).filter((v) => v != null);
    return vs.length ? vs.reduce((a, b) => (badness(m, b) < badness(m, a) ? b : a)) : null;
  });
  const tbl = h('table', { class: 'exp-table' },
    h('caption', { class: 'sr-only' }, `${m.label} per run and step; ${m.real == null ? 'lower is better' : `closer to the real-image value ${m.real} is better`}; best per step in bold`),
    h('thead', {}, h('tr', {}, h('th', { scope: 'col', class: 'exp-run' }, 'Run'), ...cols.map((s) => h('th', { scope: 'col' }, kStep(s))), h('th', { scope: 'col' }, 'vs baseline'))),
    h('tbody', {}, ...rows.map((r) => {
      const v = vals.get(r);
      const chips = [];
      if (r.missing) chips.push(h('span', { class: 'badge warn' }, 'run not found'));
      if (r.running) chips.push(h('span', { class: 'badge ok' }, `running · ${kStep(r.step ?? 0)}`));
      if (base && r !== base && r.machine && base.machine && r.machine !== base.machine) chips.push(h('span', { class: 'badge info' }, r.machine));
      if (identicalToBaseline(r, base, key)) chips.push(h('span', { class: 'badge warn', title: 'equal to the baseline to 3 digits at every shared step: check that the treatment was applied' }, '= baseline: bug?'));
      const link = r.missing ? h('span', {}, r.label || r.run) : h('a', { href: runHash(r.shownId || r.run), title: r.name }, r.label || r.name);
      const sub = [r === base ? 'baseline' : r.role, r.machine, r.msPerStep ? `${r.msPerStep.toFixed(0)} ms/step` : null].filter(Boolean).join(' · ');
      const g = gapAtLastCommon(r, base, key);
      return h('tr', { class: r === base ? 'is-base' : '' },
        h('th', { scope: 'row', class: 'exp-run' },
          h('div', { class: 'exp-run-name' }, h('i', { class: 'swatch', style: `background:${colors.get(r)}`, 'aria-hidden': 'true' }), link),
          h('div', { class: 'exp-run-sub', title: r.description || r.note || '' }, sub),
          chips.length ? h('div', { class: 'exp-chips' }, chips) : null),
        ...cols.map((s, i) => {
          const x = v.get(s);
          if (x == null) return h('td', { class: 'muted' }, r.running && (r.step ?? 0) < s ? '…' : '–');
          return h('td', { class: 'mono' }, x === best[i] ? h('b', {}, fmtScore(x)) : fmtScore(x));
        }),
        h('td', { class: `mono exp-gap ${g ? gapClass(m, v.get(g.step), v.get(g.step) - g.gap) : ''}` },
          r === base ? h('span', { class: 'muted' }, 'baseline') : g ? [fmtGap(g.gap), h('span', { class: 'muted exp-at' }, ` @${kStep(g.step)}`)] : h('span', { class: 'muted' }, '–')));
    })));
  return h('div', { class: 'exp-table-wrap', role: 'region', 'aria-label': `results table, ${m.label}`, tabindex: '0' }, tbl);
}

function caveats(e, base, key) {
  const c = [];
  if (singleSeed(e)) c.push(['info', 'one seed per arm: small gaps may be noise']);
  const otherMachine = base && e.runs.some((r) => r !== base && r.machine && r.machine !== base.machine);
  if (otherMachine) c.push(['info', `baseline on ${base.machine}, some runs elsewhere (data copy may differ)`]);
  const nLive = e.runs.filter((r) => r.running).length;
  if (nLive) c.push(['ok', `${nLive} still running`]);
  const same = e.runs.filter((r) => identicalToBaseline(r, base, key));
  if (same.length) c.push(['warn', `${same.length} equal to the baseline to 3 digits: check the treatment`]);
  if (!base) c.push(['warn', 'no baseline in the registry']);
  if (key === 'fid') c.push(['info', 'raw weights: swings about ±15 between checkpoints']);
  const m = metricInfo(key);
  if (m.real != null) c.push(['info', `12 fixed seeds per run; real images ${m.real} (sets of 12 real images vary widely)`]);
  return c.length ? h('div', { class: 'exp-chips exp-caveats' }, c.map(([k, t]) => h('span', { class: `badge ${k}` }, t))) : null;
}
