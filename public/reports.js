// Reports view: the custom pages under public/ (write-ups, sample grids, notebook code pages), from
// public/reports.json. #reports → one list, newest first, grouped by kind, with search.
// Each card links the page, the experiments it belongs to and related pages.

import { expHash } from './experiments.js';

const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v != null && v !== false) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
};
const pref = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const setPref = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } };

const KINDS = [
  { kind: 'report', label: 'Results and readouts' },
  { kind: 'code', label: 'Notebook code pages' },
  { kind: 'dashboard', label: 'Dashboards' },
];

let cache = null;
/** The registry, fetched once per page load. */
export async function fetchReports() {
  if (!cache) cache = fetch('/reports.json', { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
  return cache;
}

/** Reports that name experiment `id`, newest first (the experiment page lists them). */
export async function reportsFor(id) {
  try { return (await fetchReports()).pages.filter((p) => p.experiments?.includes(id)); } catch { return []; }
}

export async function viewReports(el) {
  let data, titles = new Map();
  try { data = await fetchReports(); } catch (err) { el.append(h('div', { class: 'empty' }, `could not load reports: ${err.message}`)); return; }
  try { titles = new Map((await (await fetch('/api/experiments')).json()).experiments.map((e) => [e.id, e.title || e.id])); } catch { /* links show ids */ }
  const byHref = new Map(data.pages.map((p) => [p.href, p]));
  let q = pref('tv.rep.q', '');
  const search = h('input', { class: 'run-search', type: 'search', placeholder: 'search titles, summaries, experiments…', value: q, 'aria-label': 'search reports' });
  const body = h('div', {});
  el.append(h('section', { class: 'section', style: 'margin-top:0' },
    h('div', { class: 'section-head' }, h('h2', {}, 'Reports')),
    h('div', { class: 'list-tools' }, search), body));
  const hay = (p) => [p.title, p.summary, p.href, ...(p.experiments || []), ...(p.experiments || []).map((id) => titles.get(id))].filter(Boolean).join(' ').toLowerCase();
  const card = (p) => h('article', { class: 'card rep-card' },
    h('a', { class: 'card-title rep-title', href: p.href }, p.title),
    h('div', { class: 'card-meta' }, h('span', {}, p.date || '–'), h('span', {}, p.href.replace(/^\//, ''))),
    p.summary ? h('p', { class: 'rep-summary' }, p.summary) : null,
    p.experiments?.length ? h('div', { class: 'rep-links' }, h('span', { class: 'muted' }, 'Experiment: '),
      ...p.experiments.map((id, i) => [i ? ' · ' : '', h('a', { href: expHash(id) }, titles.get(id) || id)])) : null,
    p.related?.length ? h('div', { class: 'rep-links' }, h('span', { class: 'muted' }, 'Related: '),
      ...p.related.map((r, i) => [i ? ' · ' : '', h('a', { href: r }, byHref.get(r)?.title || r)])) : null);
  const render = () => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    const list = data.pages.filter((p) => terms.every((t) => hay(p).includes(t))).sort((a, b) => String(b.date).localeCompare(String(a.date)));
    body.replaceChildren(...KINDS.map(({ kind, label }) => {
      const ps = list.filter((p) => p.kind === kind);
      return ps.length ? h('div', { class: 'rep-group' }, h('h3', { class: 'rep-kind' }, `${label} (${ps.length})`), h('div', { class: 'grid cards rep-cards' }, ...ps.map(card))) : null;
    }).filter(Boolean));
    if (!body.children.length) body.append(h('div', { class: 'empty' }, 'nothing matches'));
  };
  search.addEventListener('input', () => { q = search.value; setPref('tv.rep.q', q); render(); });
  render();
}
