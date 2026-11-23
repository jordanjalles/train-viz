// Experiments: groups of directly comparable runs (MLflow's "experiment"), from experiments.json at the
// project root (written by scripts/experiment_add.py). GET /api/experiments returns the registry with
// each run enriched from the RunStore: scores per step, status, latest step, ms/step, machine, and
// line 1 of DESCRIPTION.md. Read-only, like the rest of the server.

import fs from 'node:fs';
import path from 'node:path';
import { scoreRecords } from './runs.js';

// Keys shown per step. gen_metrics.jsonl (+ the plain-sampler rows of gen_fix.jsonl, see scoreRecords)
// holds the *_ema scores; metrics.jsonl holds the raw-weight `fid` and the test loss.
// ema_prev_*: contrast stats of the 12 fixed-seed previews from the EMA weights (image_detail in the pixel jobs).
export const SCORE_KEYS = ['fid_ema', 'kid_ema', 'fd_ae_z_ema', 'fd_ae_pool_ema', 'fid', 'test_clean', 'probe_err','ema_prev_contrast', 'ema_prev_dark', 'ema_prev_light'];
const MAX_POINTS = 80;   // a dense key (test_clean every 1k steps) is thinned to this many points

const statKey = (file) => { try { const s = fs.statSync(file); return `${s.mtimeMs}:${s.size}`; } catch { return '-'; } };

/** Machine of a run from its root (the 3090 feed, Colab runs in the shared sink, else the Spark). */
export function machineOf(run) {
  if (run.root === 'wsl3090') return '3090';
  if (run.config?.backend === 'colab') return 'Colab';
  return 'Spark';
}

/** First non-empty line of a description, without markdown heading marks. */
export function firstLine(text) {
  const l = String(text || '').split('\n').map((s) => s.replace(/^#+\s*/, '').trim()).find(Boolean);
  return l ? l.slice(0, 300) : null;
}

/** Evenly thin points to at most n, always keeping the first and the last. */
export function thin(points, n = MAX_POINTS) {
  if (points.length <= n) return points;
  const out = [];
  for (let i = 0; i < n; i++) out.push(points[Math.round((i * (points.length - 1)) / (n - 1))]);
  return out;
}

/** {key: [[step, value], ...]} from metric records and side (score) records; a side record wins at an equal step. */
export function scoreSeries(records, side, keepSteps = []) {
  const keep = new Set(keepSteps);
  const out = {};
  for (const k of SCORE_KEYS) {
    const byStep = new Map();
    for (const r of records) if (typeof r[k] === 'number' && Number.isFinite(r[k]) && typeof r.step === 'number') byStep.set(r.step, r[k]);
    for (const r of side) if (typeof r[k] === 'number' && Number.isFinite(r[k])) byStep.set(r.step, r[k]);
    if (!byStep.size) continue;
    const pts = [...byStep.entries()].sort((a, b) => a[0] - b[0]);
    // thinning must not drop the registry's score steps
    out[k] = pts.length > MAX_POINTS ? [...new Map([...thin(pts), ...pts.filter(([s]) => keep.has(s))].sort((a, b) => a[0] - b[0])).entries()] : pts;
  }
  return out;
}

export class ExperimentStore {
  constructor(file, store) {
    this.file = file; this.store = store;
    this.reg = null; this.regKey = null;
    this.runCache = new Map();   // run id -> {key, value}
  }

  registry() {
    const k = statKey(this.file);
    if (k !== this.regKey) {
      try { this.reg = JSON.parse(fs.readFileSync(this.file, 'utf8')); this.regKey = k; }
      catch (e) { if (!this.reg) this.reg = { version: 1, experiments: [], error: e.message }; }   // mid-write: keep the last good copy
    }
    return this.reg;
  }

  /** Enriched view of one run id, cached on the record counts and score-file stats of the run and its stages. */
  runInfo(id, scoreSteps) {
    const raw = this.store.get(id);
    if (!raw) return { missing: true };
    const run = this.store.shown(raw);
    const parts = [run, ...(run.stages || [])];
    const key = [run.id, run.status, run.description?.length ?? 0, scoreSteps.join(','),
      ...parts.map((r) => `${r.metrics?.records.length ?? 0}:${r.metrics?.offset ?? 0}:${r.status}:${statKey(path.join(r.dir, 'gen_metrics.jsonl'))}:${statKey(path.join(r.dir, 'gen_fix.jsonl'))}`)].join('|');
    const hit = this.runCache.get(id);
    if (hit?.key === key) return hit.value;
    const records = this.store.records(run);
    // scores of every stage of a continuation chain; a later stage wins at an equal step
    const side = [];
    for (const r of parts) side.push(...scoreRecords(r.dir));
    const head = (run.stages || []).reduce((h, s) => ((s.updatedAt || 0) >= (h.updatedAt || 0) ? s : h), run);
    const sps = records.slice(-50).findLast((r) => typeof r.sps === 'number')?.sps;
    const scores = scoreSeries(records, side, scoreSteps);
    const step = Math.max(records.at(-1)?.step ?? -Infinity, ...Object.values(scores).map((p) => p.at(-1)[0]));
    const nameSeed = /(?:^|[-_])seed(\d+)/.exec(run.name)?.[1];
    const value = {
      shownId: run.id !== id ? run.id : undefined,   // a continuation stage: its records live under the first run
      name: run.name, root: run.root, machine: machineOf(run),
      status: head.status, running: head.status === 'live',
      step: Number.isFinite(step) ? step : null,
      totalSteps: head.config?.steps ?? run.config?.steps ?? null,
      msPerStep: sps > 0 ? 1000 / sps : null,
      seed: run.config?.seed ?? run.config?.hparams?.seed ?? (nameSeed != null ? Number(nameSeed) : null),
      description: firstLine(run.description),
      updatedAt: Math.max(run.updatedAt || 0, head.updatedAt || 0) || null,
      scores,
    };
    this.runCache.set(id, { key, value });
    return value;
  }

  list() {
    const reg = this.registry();
    const experiments = (reg.experiments || []).map((e) => {
      const steps = Array.isArray(e.score_steps) ? e.score_steps : [];
      const runs = (e.runs || []).map((r) => ({ ...r, ...this.runInfo(r.run, steps) }));
      const planned = (e.planned_runs || []).map((name) => {
        const found = [...this.store.runs.values()].find((r) => r.name === name || r.id === name);
        return { name, id: found ? this.store.shown(found).id : null, status: found ? found.status : null };
      });
      const anyLive = runs.some((r) => r.running);
      const updatedAt = Math.max(0, ...runs.map((r) => r.updatedAt || 0)) || null;
      // "running" only while a run of it is live; a registry "running" without a live run shows as stale
      const liveStatus = anyLive ? 'running' : e.status === 'running' ? 'stale' : e.status;
      return { ...e, runs, planned_runs: e.planned_runs ? planned : undefined, liveStatus, updatedAt };
    });
    return { version: reg.version, error: reg.error, experiments };
  }
}
