// Run discovery + live tailing.
//
// Two kinds of run directories are understood:
//
//  * harness  — stl10-selfteach style: runs/<name>/{config.json, metrics.jsonl,
//               diag.jsonl, log.txt, recon_NNNNNN.png, z_NNNNNN.pt}
//  * notebook — the Jupyter run manager's training_runs/ae/<run_id>/metadata.json
//               (history.pt lives on the notebook owner's account and is not
//               visible here; we show metadata + any PNGs in <run_id>/latest/)
//
// Roots are configured in config.json at the project root. Each root is
// scanned periodically (new runs) and JSONL files are tailed with fs.watch,
// with a polling fallback for bind-mounted volumes where inotify is unreliable.

import { EventEmitter } from 'node:events';
import { colabRunStatus } from './colab.js';
import fs from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

// Slow runs (256 px on the Spark: ~2 min per 100-step record, longer across a test) must not
// flip to stale between records.
const LIVE_WINDOW_MS = 10 * 60 * 1000;

function safeJson(line) {
  try { return JSON.parse(line); } catch { return null; }
}

/**
 * Append records in step order. A record whose step is not after the last one means the log was
 * rewritten from that point (a resumed run keeps its history up to the resume step, then appends),
 * so everything from that step on is dropped first. Without this a resume duplicated the history
 * and charts drew a line from the restart point back to step 0. Same rule in public/app.js.
 */
export function appendRecords(arr, fresh) {
  for (const r of fresh) {
    // A record with the SAME step as the last one adds fields to it (a sink may write the train record,
    // the eval record and the latent statistics of one step as three lines). Only an EARLIER step means a rewrite.
    if (typeof r.step === 'number' && arr.length && r.step === arr.at(-1).step) { arr[arr.length - 1] = { ...arr.at(-1), ...r }; continue; }
    if (typeof r.step === 'number' && arr.length && typeof arr.at(-1).step === 'number' && r.step < arr.at(-1).step) {
      let i = arr.length;
      while (i > 0 && arr[i - 1].step >= r.step) i--;
      arr.length = i;
    }
    arr.push(r);
  }
  return arr;
}

class JsonlTail {
  constructor(file) {
    this.file = file;
    this.records = [];
    this.offset = 0;
    this.partial = '';
  }
  /** Read whatever was appended since last time. Returns the new records. */
  readNew() {
    let st;
    try { st = fs.statSync(this.file); } catch { return []; }
    if (st.size < this.offset) { // truncated / rewritten
      this.offset = 0; this.records = []; this.partial = '';
    }
    if (st.size === this.offset) return [];
    const fd = fs.openSync(this.file, 'r');
    try {
      const buf = Buffer.alloc(st.size - this.offset);
      fs.readSync(fd, buf, 0, buf.length, this.offset);
      this.offset = st.size;
      const text = this.partial + buf.toString('utf8');
      const lines = text.split('\n');
      this.partial = lines.pop();
      const fresh = [];
      for (const l of lines) {
        if (!l.trim()) continue;
        const r = safeJson(l);
        if (r) fresh.push(r);
      }
      appendRecords(this.records, fresh);
      return fresh;
    } finally {
      fs.closeSync(fd);
    }
  }
}

export class RunStore extends EventEmitter {
  constructor(roots, { scanMs = 5000, pollMs = 1500 } = {}) {
    super();
    this.roots = roots;
    this.runs = new Map(); // id -> run
    this.scanMs = scanMs;
    this.pollMs = pollMs;
  }

  async start() {
    await this.scan();
    this._scanTimer = setInterval(() => this.scan().catch((e) => this.emit('error', e)), this.scanMs);
    this._pollTimer = setInterval(() => this.poll(), this.pollMs);
    this._scanTimer.unref(); this._pollTimer.unref();
  }

  idFor(root, name) {
    return `${root.id}/${name}`;
  }

  async scan() {
    const added = [];
    for (const root of this.roots) {
      let names;
      try { names = await readdir(root.path, { withFileTypes: true }); } catch { continue; }
      for (const d of names) {
        if (!d.isDirectory() || d.name.startsWith('.') || d.name.startsWith('_')) continue; // _archive_invalid etc.
        const id = this.idFor(root, d.name);
        // A continuation stage folded into its original run (FOLDED_INTO marker): the original
        // run's directory carries its records with shifted steps, so this one is not a run.
        if (fs.existsSync(path.join(root.path, d.name, 'FOLDED_INTO'))) {
          const old = this.runs.get(id);
          if (old) { for (const w of old.watchers || []) w.close(); this.runs.delete(id); this.emit('run-removed', { id }); }
          continue;
        }
        if (this.runs.has(id)) { await this.refreshFiles(this.runs.get(id)); continue; }
        const run = await this.load(root, d.name);
        if (run) { this.runs.set(id, run); added.push(run); }
      }
    }
    this.relink();
    for (const run of added) if (!run.foldedInto) this.emit('run-added', this.summary(run));
  }

  /**
   * Continuations. A run whose config has `resumed_from.job` continues the run of that runner job
   * (config `runner_job_id`, same root) from a checkpoint. It is a stage of that first run, not a
   * run: it leaves the list, and the first run shows the records and files of all its stages.
   * A chain (a continuation of a continuation) folds into the first run in chain order.
   * `resumed_from.separate: true` keeps a run by itself (a branch, not a continuation).
   */
  relink() {
    const byJob = new Map();
    for (const r of this.runs.values()) if (r.kind === 'harness' && r.config?.runner_job_id) byJob.set(`${r.root}/${r.config.runner_job_id}`, r);
    const parentOf = (r) => {
      const f = r.kind === 'harness' ? r.config?.resumed_from : null;
      const p = f?.job && !f.separate ? byJob.get(`${r.root}/${f.job}`) : null;
      return p && p !== r ? p : null;
    };
    const stages = new Map();
    for (const r of this.runs.values()) {
      let first = r, depth = 0;
      for (let p; depth < 64 && (p = parentOf(first)); depth++) first = p;
      r.stageDepth = depth;
      const into = depth ? first.id : null;
      if (into !== (r.foldedInto ?? null)) {
        if (into && r.listed) this.emit('run-removed', { id: r.id });
        r.foldedInto = into;
      }
      if (depth) { if (!stages.has(first.id)) stages.set(first.id, []); stages.get(first.id).push(r); }
    }
    for (const r of this.runs.values()) {
      const now = (stages.get(r.id) || []).sort((a, b) => (a.stageDepth - b.stageDepth) || ((a.startedAt || 0) - (b.startedAt || 0)) || a.name.localeCompare(b.name));
      const before = r.stages || [];
      if (now.length !== before.length || now.some((x, i) => x !== before[i])) {
        r.stages = now; r.merged = null;
        if (!r.listed) continue;
        this.emit('run-updated', this.summary(r));
        // open pages hold the first run's records: send them what a new stage has already written
        for (const st of now) if (!before.includes(st) && st.metrics.records.length) {
          this.emit('records', { id: r.id, metrics: st.metrics.records, diag: [], robust: [], status: st.status, updatedAt: st.updatedAt });
        }
      }
    }
  }

  /** The run a record or file change is shown under: the first run of a continuation chain. */
  shown(run) { return (run.foldedInto && this.runs.get(run.foldedInto)) || run; }

  /** Metric records of a run with the records of its continuation stages appended (appendRecords rule). */
  records(run) {
    if (!run.metrics) return [];
    if (!run.stages?.length) return run.metrics.records;
    const key = [run, ...run.stages].map((r) => `${r.metrics.records.length}:${r.metrics.offset}`).join('|');
    if (run.merged?.key !== key) {
      const all = run.metrics.records.slice();
      for (const s of run.stages) appendRecords(all, s.metrics.records);
      run.merged = { key, all };
    }
    return run.merged.all;
  }

  /** Reconstruction and latent files of a run and its stages. A later stage wins at an equal step. */
  files(run) {
    const recon = {}, ema = {}, z = {};
    for (const r of [run, ...(run.stages || [])]) {
      for (const [step, f] of Object.entries(r.reconFiles || {})) recon[step] = { ...f, path: path.join(r.dir, f.file) };
      for (const [step, f] of Object.entries(r.emaReconFiles || {})) ema[step] = { ...f, path: path.join(r.dir, f.file) };
      for (const step of r.zSteps || []) z[step] = path.join(r.dir, `z_${String(step).padStart(6, '0')}.pt`);
    }
    return { recon, ema, z };
  }
  reconFile(run, step) { return this.files(run).recon[step] || null; }
  emaReconFile(run, step) { return this.files(run).ema[step] || null; }
  zFile(run, step) { return this.files(run).z[step] || null; }

  async load(root, name) {
    const dir = path.join(root.path, name);
    const id = this.idFor(root, name);
    const run = {
      id, name, root: root.id, kind: root.kind, dir,
      config: null, meta: null, metrics: null, diag: null,
      recons: [], emaRecons: [], zSteps: [], robustPngs: [], images: [], status: 'unknown', startedAt: null, updatedAt: null,
    };
    if (root.kind === 'harness') {
      try { run.config = JSON.parse(await readFile(path.join(dir, 'config.json'), 'utf8')); } catch { /* not a run dir */ }
      if (!run.config && !fs.existsSync(path.join(dir, 'metrics.jsonl'))) return null;
      if (!run.config) { // config.json archived (clobbered by an analysis script): rebuild from the harness queue
        const q = await queueConfig(path.join(root.path, '..', 'queue.txt'), name);
        if (q) { run.config = q; run.configSource = 'queue.txt'; }
      }
      run.metrics = new JsonlTail(path.join(dir, 'metrics.jsonl'));
      // BRANCHED_FROM.json {"run": NAME, "step": N}: a branch of run NAME (same root) from its checkpoint of step N.
      // full() shows the parent's records up to step N before this run's own, so the charts have no gap.
      try { run.branchedFrom = JSON.parse(await readFile(path.join(dir, 'BRANCHED_FROM.json'), 'utf8')); } catch { /* no branch */ }
      run.diag = new JsonlTail(path.join(dir, 'diag.jsonl'));
      // latent-noise robustness sweeps: train.py at each eval, or robust_backfill.py from checkpoints
      run.robust = new JsonlTail(path.join(dir, 'robust.jsonl'));
      run.metrics.readNew(); run.diag.readNew(); run.robust.readNew();
      run.watchers = [];
      for (const t of [run.metrics, run.diag, run.robust]) {
        try {
          const w = fs.watch(t.file, { persistent: false }, () => this.pollRun(run));
          w.on('error', () => {});
          run.watchers.push(w);
        } catch { /* file may not exist yet; polling covers it */ }
      }
    } else if (root.kind === 'notebook') {
      try { run.meta = JSON.parse(await readFile(path.join(dir, 'metadata.json'), 'utf8')); } catch { return null; }
      run.config = run.meta.config || null;
    }
    await this.refreshFiles(run);
    return run;
  }

  async refreshFiles(run) {
    let names = [];
    try { names = await readdir(run.dir); } catch { return; }
    // Reconstruction grids: live `recon_NNNNNN.png`, or — where those were
    // archived as invalid and re-rendered from a checkpoint by reeval.py —
    // `reeval_ckpt_NNNNNN.png`. A live snapshot wins when both exist.
    // `emaprev_NNNNNN.png`: the grid of the same step and seeds, sampled from the averaged (EMA) weights.
    const reconFiles = {}, emaReconFiles = {}, zs = [], rob = [];
    for (const n of names) {
      let m;
      if ((m = /^recon_(\d+)\.png$/.exec(n))) reconFiles[parseInt(m[1], 10)] = { file: n, kind: 'live' };
      else if ((m = /^emaprev_(\d+)\.png$/.exec(n))) emaReconFiles[parseInt(m[1], 10)] = { file: n };
      else if ((m = /^reeval_ckpt_(\d+)\.png$/.exec(n))) { const s = parseInt(m[1], 10); if (!reconFiles[s]) reconFiles[s] = { file: n, kind: 'reeval' }; }
      else if ((m = /^z_(\d+)\.pt$/.exec(n))) zs.push(parseInt(m[1], 10));
      else if ((m = /^robust_(\d+)\.png$/.exec(n))) rob.push(parseInt(m[1], 10));
    }
    const recons = Object.keys(reconFiles).map(Number).sort((a, b) => a - b);
    zs.sort((a, b) => a - b); rob.sort((a, b) => a - b);
    run.reconFiles = reconFiles;
    run.reconKinds = Object.fromEntries(recons.map((s) => [s, reconFiles[s].kind]));
    const emaRecons = Object.keys(emaReconFiles).map(Number).sort((a, b) => a - b);
    run.emaReconFiles = emaReconFiles;
    let changed = recons.length !== run.recons.length || emaRecons.length !== run.emaRecons.length || zs.length !== run.zSteps.length || rob.length !== run.robustPngs.length;
    // Optional experiment description: DESCRIPTION.md in the run dir (written at launch; editable
    // any time), else a `description` field in the notebook metadata or harness config.
    let desc = null;
    if (names.includes('DESCRIPTION.md')) {
      try { desc = (await readFile(path.join(run.dir, 'DESCRIPTION.md'), 'utf8')).trim().slice(0, 4000); } catch { /* ignore */ }
    }
    desc = desc || run.meta?.description || run.config?.description || null;
    if (desc !== (run.description ?? null)) { run.description = desc; changed = true; }
    run.recons = recons; run.emaRecons = emaRecons; run.zSteps = zs; run.robustPngs = rob;
    // Thumbnails of the images behind the z snapshots, for the Latent view's focus control:
    // z_images.pt (uint8 N×3×H×W, written next to the snapshots), else a root-level STL-10
    // test file for harness runs (their snapshots are the first test images, in order).
    const root = this.roots.find((r) => r.id === run.root);
    const zImg = names.includes('z_images.pt') ? 'file' : (zs.length && root?.stl10TestBin ? 'stl10' : null);
    if (zImg !== (run.zImageSource ?? null)) { run.zImageSource = zImg; changed = true; }
    // config.json is rewritten when a run is resumed with new flags (e.g. extended STEPS); re-read it
    // on change, or a resumed run keeps its old length and shows as done once it passes it.
    if (run.kind === 'harness' && names.includes('config.json')) {
      try {
        const m = (await stat(path.join(run.dir, 'config.json'))).mtimeMs;
        if (m !== run.configMtime) {
          if (run.configMtime !== undefined) {
            run.config = JSON.parse(await readFile(path.join(run.dir, 'config.json'), 'utf8'));
            run.configSource = 'config.json';
            changed = true;
          }
          run.configMtime = m;
        }
      } catch { /* mid-write or unreadable: keep the previous config */ }
    }
    const prevUpdated = run.updatedAt, prevStatus = run.status;

    if (run.kind === 'notebook') {
      // PNGs from the notebook's live plotting live in <images root>/<run_id>/latest/
      const imgDir = run.imageDir ?? (run.imageDir = this.findImageDir(run));
      if (imgDir) {
        try {
          const files = (await readdir(imgDir)).filter((f) => f.endsWith('.png')).sort();
          const st = await stat(imgDir);
          run.images = files;
          run.updatedAt = st.mtimeMs;
        } catch { run.images = []; }
      }
      run.startedAt = run.meta?.started_at ? Date.parse(run.meta.started_at) : null;
      run.status = run.meta?.status === 'running' && run.updatedAt && Date.now() - run.updatedAt < LIVE_WINDOW_MS ? 'live'
        : run.meta?.status === 'running' ? 'stale' : (run.meta?.status || 'unknown');
    } else {
      await this.refreshHarnessStatus(run);
    }
    if (run.kind === 'notebook' && run.updatedAt !== prevUpdated) changed = true;
    if (run.status !== prevStatus) changed = true;
    if (changed && run.listed) this.emit('run-updated', this.summary(this.shown(run)));
  }

  findImageDir(run) {
    const root = this.roots.find((r) => r.id === run.root);
    if (!root?.images) return null;
    const p = path.join(root.images, run.name, 'latest');
    return fs.existsSync(p) ? p : null;
  }

  async refreshHarnessStatus(run) {
    let mtime = 0;
    for (const f of ['metrics.jsonl', 'log.txt']) {
      try { const st = await stat(path.join(run.dir, f)); mtime = Math.max(mtime, st.mtimeMs); } catch { /* ignore */ }
    }
    try { const st = await stat(path.join(run.dir, 'config.json')); run.startedAt = st.mtimeMs; } catch { /* ignore */ }
    run.updatedAt = mtime || run.updatedAt;
    let tail = '';
    try {
      const fd = fs.openSync(path.join(run.dir, 'log.txt'), 'r');
      const st = fs.fstatSync(fd);
      const n = Math.min(st.size, 4096);
      const b = Buffer.alloc(n);
      fs.readSync(fd, b, 0, n, st.size - n);
      fs.closeSync(fd);
      tail = b.toString('utf8');
    } catch { /* no log */ }
    const last = run.metrics.records.at(-1);
    const exit = /exit=(\d+)/.exec(tail);
    if (last?.nan) run.status = 'diverged';
    else if (/^DONE/m.test(tail) || (run.config?.steps && last?.step >= run.config.steps)) run.status = 'done';
    // 137 = SIGKILL (docker rm -f / manual stop), 124 = the harness's RUN_TIMEOUT
    else if (exit) run.status = exit[1] === '0' ? 'done' : exit[1] === '124' ? 'timeout' : exit[1] === '137' ? 'killed' : 'failed';
    else if (Date.now() - mtime < LIVE_WINDOW_MS) run.status = 'live';
    else run.status = 'stale';
    // The restricted 3090 collector owns this status file; experiment logs cannot override it.
    if (run.root === 'wsl3090') {
      try {
        const runner = JSON.parse(await readFile(path.join(run.dir, 'runner-status.json'), 'utf8'));
        const states = { queued: 'queued', running: 'live', done: 'done', failed: 'failed', cancelled: 'killed', interrupted: 'killed' };
        if (runner.source === '3090-wsl' && states[runner.status]) {
          run.status = states[runner.status];
          if (['queued', 'running'].includes(runner.status) && Date.now() - runner.collected_at * 1000 > LIVE_WINDOW_MS) run.status = 'stale';
        }
        // Notebook runs mirrored through the feed are not jobs: one that stops writing has just
        // ended (the kernel was stopped or moved on), which is not a failure. finish_run() -> done.
        if (runner.source === '3090-notebook') {
          run.status = { running: 'live', done: 'done', interrupted: 'ended' }[runner.status] || run.status;
          if (runner.status === 'running' && Date.now() - runner.collected_at * 1000 > LIVE_WINDOW_MS) run.status = 'ended';
        }
      } catch { /* collector not yet available: retain normal status inference */ }
    }
    if (run.root === 'shared' && run.config?.backend === 'colab') {
      try {
        const collector = JSON.parse(await readFile(path.join(run.dir, 'runner-status.json'), 'utf8'));
        const status = colabRunStatus(collector);
        if (status) { run.status = status; return; }
      } catch { /* no collector status yet */ }
    }
    // Any sink run that wrote its FINISHED marker ended normally, whatever the root.
    if (run.status === 'stale' && fs.existsSync(path.join(run.dir, 'FINISHED'))) run.status = 'done';
    // Spark-kernel notebook runs (sink writes straight into this root): same rule.
    if (run.root === 'nbsink' && !['live', 'diverged'].includes(run.status)) {
      run.status = fs.existsSync(path.join(run.dir, 'FINISHED')) ? 'done' : 'ended';
    }
  }

  poll() {
    for (const run of this.runs.values()) if (run.kind === 'harness') this.pollRun(run);
  }

  /** Runs whose records carry no `sps` (the notebook sink, 3090 feed): derive steps/s from when
   *  their records arrive here, over the last few arrivals, and put it on the newest record. */
  deriveThroughput(run, fresh) {
    const train = fresh.filter((r) => typeof r.step === 'number' && !('test_clean' in r));
    const last = train.at(-1);
    if (!last || typeof last.sps === 'number') return;
    const now = Date.now();
    const seen = (run.arrivals ||= []);
    // first batch after a (re)start is the file's history, and a lower step is a resume: no rate
    if (!run.arrivalsPrimed || (seen.length && last.step <= seen.at(-1).step)) {
      run.arrivalsPrimed = true; seen.length = 0; seen.push({ t: now, step: last.step }); return;
    }
    seen.push({ t: now, step: last.step });
    if (seen.length > 5) seen.shift();
    const first = seen[0], dt = (now - first.t) / 1000;
    if (dt >= 5) last.sps = (last.step - first.step) / dt;
  }

  pollRun(run) {
    const m = run.metrics.readNew();
    if (m.length) this.deriveThroughput(run, m);
    const d = run.diag.readNew();
    const rb = run.robust.readNew();
    if (m.length || d.length || rb.length) {
      this.refreshHarnessStatus(run).then(() => {
        this.emit('records', { id: this.shown(run).id, metrics: m, diag: d, robust: rb, status: run.status, updatedAt: run.updatedAt });
      });
    }
  }

  summary(run) {
    if (!run.foldedInto) run.listed = true;
    const records = this.records(run);
    // the stage that wrote last decides state and length (a stage cancelled before it started stays behind)
    const head = (run.stages || []).reduce((h, s) => ((s.updatedAt || 0) >= (h.updatedAt || 0) ? s : h), run);
    const files = run.stages?.length ? this.files(run) : null;
    const nums = (o) => Object.keys(o).map(Number).sort((x, y) => x - y);
    const recons = files ? nums(files.recon) : run.recons;
    const last = records.at(-1) || null;
    const lastEval = records.findLast((r) => 'test_clean' in r) || null;
    let bestTest = null;   // lowest test MSE seen (for sorting the run list)
    for (const r of records) if (typeof r.test_clean === 'number' && (bestTest === null || r.test_clean < bestTest)) bestTest = r.test_clean;
    const config = run.stages?.length && run.config
      ? { ...run.config, steps: head.config?.steps ?? run.config.steps, stages: run.stages.map((s) => ({ name: s.name, job: s.config?.runner_job_id ?? null, from_step: s.config?.resumed_from?.steps ?? null, status: s.status })) }
      : run.config;
    return {
      id: run.id, name: run.name, root: run.root, kind: run.kind, status: head.status,
      startedAt: run.startedAt, updatedAt: Math.max(run.updatedAt || 0, head.updatedAt || 0) || null,
      step: last?.step ?? null, totalSteps: config?.steps ?? null,
      last, lastEval, bestTest,
      // newest steps/s, logged by the run or derived from record arrival (deriveThroughput)
      sps: records.slice(-50).findLast((r) => typeof r.sps === 'number')?.sps ?? null,
      config, meta: run.meta,
      recons, reconKinds: files ? Object.fromEntries(recons.map((st) => [st, files.recon[st].kind])) : (run.reconKinds || {}),
      emaRecons: files ? nums(files.ema) : run.emaRecons,   // steps with an `emaprev_*.png` grid (averaged weights)
      zSteps: files ? nums(files.z) : run.zSteps, robustPngs: run.robustPngs, images: run.images,
      zImages: !!run.zImageSource,
      configSource: run.configSource || 'config.json',
      description: run.description || null,
      nMetrics: records.length, nDiag: run.diag?.records.length ?? 0,
      // cheap sparkline: the `clean` loss subsampled to ≤60 points
      spark: run.metrics ? spark(records, 'clean', 60) : [],
    };
  }

  list() {
    return [...this.runs.values()].filter((r) => !r.foldedInto).map((r) => this.summary(r)).sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0));
  }

  get(id) { return this.runs.get(id) || null; }

  full(id) {
    const run = this.get(id);
    if (!run) return null;
    return { ...this.summary(run), metrics: [...this.branchPrefix(run), ...mergeBySteps(this.records(run), scoreRecords(run.dir))], diag: run.diag?.records ?? [], robust: run.robust?.records ?? [] };
  }

  /** The parent's records (with its generative scores) up to the branch step, for a run with BRANCHED_FROM.json. Chains follow.
   *  The Colab JiT chain is ~43 links (one folder per VM session, many with no records), hence the high cap. */
  branchPrefix(run, depth = 0) {
    const b = run.branchedFrom;
    const parent = b && typeof b.step === 'number' && depth < 200 ? this.get(this.idFor({ id: run.root }, b.run)) : null;
    if (!parent || parent === run) return [];
    const own = mergeBySteps(this.records(parent), scoreRecords(parent.dir));
    return [...this.branchPrefix(parent, depth + 1), ...own.filter((r) => typeof r.step === 'number' && r.step <= b.step)];
  }
}

/**
 * The generative scores of a run folder: gen_metrics.jsonl, and the plain-sampler rows (gen_fix "{}") of gen_fix.jsonl,
 * which hold the same score (EMA weights, plain sampler) for later checkpoints. gen_metrics.jsonl wins at an equal step.
 */
export function scoreRecords(dir) {
  const plain = readSideRecords(path.join(dir, 'gen_fix.jsonl'), (r) => r.gen_fix === '{}');
  const main = readSideRecords(path.join(dir, 'gen_metrics.jsonl'));
  const have = new Set(main.map((r) => r.step));
  return [...main, ...plain.filter((r) => !have.has(r.step))];
}

/**
 * Records of a side file (gen_metrics.jsonl: generative scores of checkpoints, written after the
 * fact and in any order). They must not go into metrics.jsonl: a record with an earlier step there
 * means "the log was rewritten from here" (appendRecords). A later line for a step wins.
 */
export function readSideRecords(file, keep = () => true) {
  let text;
  try { text = fs.readFileSync(file, 'utf8'); } catch { return []; }
  const byStep = new Map();
  for (const line of text.split('\n')) {
    const r = line.trim() && safeJson(line);
    if (r && typeof r.step === 'number' && keep(r)) byStep.set(r.step, { ...(byStep.get(r.step) || {}), ...r });
  }
  return [...byStep.values()];
}

/** A copy of `records` with the side records merged in: into the last record of the same step, else inserted in step order. */
export function mergeBySteps(records, side) {
  if (!side.length) return records;
  const out = records.slice();
  const lastIndex = new Map();
  out.forEach((r, i) => { if (typeof r.step === 'number') lastIndex.set(r.step, i); });
  const extra = [];
  for (const s of side) {
    if (lastIndex.has(s.step)) out[lastIndex.get(s.step)] = { ...out[lastIndex.get(s.step)], ...s };
    else extra.push(s);
  }
  if (!extra.length) return out;
  return [...out, ...extra].map((r, i) => [r, i]).sort((a, b) => ((a[0].step ?? 0) - (b[0].step ?? 0)) || (a[1] - b[1])).map((x) => x[0]);
}

/**
 * Flags for a run from the harness's queue.txt ("<name> KEY=VAL ..."; a leading
 * '#' is a done/disabled marker). Mapped to train.py's config keys where the
 * names differ; everything else is lower-cased. Numbers are parsed.
 */
const FLAG_KEYS = { STEPS: 'steps', EVAL_EVERY: 'eval_every', VIEW_W: 'view_weight', WW: 'ww', LR: 'lr', DIAG_EVERY: 'diag_every', CKPT_EVERY: 'ckpt_every' };
async function queueConfig(file, name) {
  let text;
  try { text = await readFile(file, 'utf8'); } catch { return null; }
  for (const line of text.split('\n')) {
    const parts = line.replace(/^#+/, '').trim().split(/\s+/);
    if (parts[0] !== name) continue;
    const cfg = { run: name, steps: 6000, eval_every: 1000 }; // train.py defaults
    for (const kv of parts.slice(1)) {
      const i = kv.indexOf('='); if (i < 0) continue;
      const k = kv.slice(0, i), v = kv.slice(i + 1);
      const num = Number(v);
      cfg[FLAG_KEYS[k] || k.toLowerCase()] = v !== '' && Number.isFinite(num) ? num : v;
    }
    return cfg;
  }
  return null;
}

function spark(records, key, n) {
  const pts = records.filter((r) => typeof r[key] === 'number');
  if (pts.length <= n) return pts.map((r) => [r.step, r[key]]);
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = pts[Math.floor((i * (pts.length - 1)) / (n - 1))];
    out.push([r.step, r[key]]);
  }
  return out;
}
