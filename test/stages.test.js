import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { RunStore } from '../lib/runs.js';

function runDir(root, name, config, records, files = []) {
  const d = path.join(root, name);
  fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'config.json'), JSON.stringify(config));
  fs.writeFileSync(path.join(d, 'metrics.jsonl'), records.map((r) => JSON.stringify(r)).join('\n') + '\n');
  for (const f of files) fs.writeFileSync(path.join(d, f), 'x');
  return d;
}
const recs = (from, to) => { const a = []; for (let s = from; s <= to; s += 100) a.push({ step: s, clean: 1 / (s + 1) }); return a; };

test('a continuation chain is one run: records, files, state and length come from all stages', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-stages-'));
  runDir(root, 'first--a', { steps: 600, runner_job_id: 'a' }, recs(0, 600), ['recon_000600.png', 'emaprev_000600.png', 'z_000600.pt']);
  // stage b ran to 900 and was cancelled; its checkpoint is at 800. Stage c continues from 800.
  runDir(root, 'first_more--b', { steps: 1400, runner_job_id: 'b', resumed_from: { job: 'a', steps: 600 } },
    [{ step: 600, test_clean: 0.5 }, ...recs(700, 900).map((r) => ({ ...r, stage: 'b' }))], ['recon_000800.png', 'recon_000900.png', 'z_000800.pt']);
  const dc = runDir(root, 'first--c', { steps: 1400, runner_job_id: 'c', resumed_from: { job: 'b', steps: 800 } },
    recs(800, 1000).map((r) => ({ ...r, stage: 'c' })), ['recon_000800.png', 'emaprev_000800.png']);
  runDir(root, 'branch--d', { steps: 700, runner_job_id: 'd', resumed_from: { job: 'a', steps: 600, separate: true } }, recs(600, 700));
  const store = new RunStore([{ id: 'r', path: root, kind: 'harness' }]);
  await store.scan();
  const list = store.list();
  assert.deepEqual(list.map((r) => r.name).sort(), ['branch--d', 'first--a']);
  const one = store.full('r/first--a');
  assert.deepEqual(one.metrics.map((r) => r.step), recs(0, 1000).map((r) => r.step));
  assert.equal(one.metrics.find((r) => r.step === 600).test_clean, 0.5);                  // same step: fields merge
  assert.deepEqual(one.metrics.filter((r) => r.step >= 800).map((r) => r.stage), ['c', 'c', 'c']);   // the later stage rewrites from 800
  assert.equal(one.totalSteps, 1400);
  assert.deepEqual(one.recons, [600, 800, 900]);
  assert.deepEqual(one.zSteps, [600, 800]);
  assert.equal(store.reconFile(store.get('r/first--a'), 800).path, path.join(dc, 'recon_000800.png'));
  // EMA sample grids (emaprev_*.png) are a list of their own, also across stages; a run without them has an empty list
  assert.deepEqual(one.emaRecons, [600, 800]);
  assert.equal(store.emaReconFile(store.get('r/first--a'), 800).path, path.join(dc, 'emaprev_000800.png'));
  assert.equal(store.emaReconFile(store.get('r/first--a'), 900), null);
  assert.deepEqual(store.full('r/branch--d').emaRecons, []);
  assert.equal(one.config.stages.length, 2);
  // records that a stage writes later are sent under the first run
  const seen = [];
  store.on('records', (d) => seen.push(d.id));
  fs.appendFileSync(path.join(dc, 'metrics.jsonl'), JSON.stringify({ step: 1100, clean: 0.1 }) + '\n');
  store.poll();
  await new Promise((r) => setTimeout(r, 50));
  assert.deepEqual(seen, ['r/first--a']);
  assert.equal(store.full('r/first--a').metrics.at(-1).step, 1100);
  fs.rmSync(root, { recursive: true });
});
