// gen_metrics.jsonl (generative scores written after the fact) is merged into a run's records by step.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { mergeBySteps, readSideRecords, appendRecords } from '../lib/runs.js';

test('side records merge into the record of their step and never drop history', () => {
  const records = appendRecords([], [{ step: 0, test_clean: 0.3 }, { step: 100, clean: 0.2 }, { step: 200, clean: 0.1, test_clean: 0.15 }, { step: 300, clean: 0.09 }]);
  const merged = mergeBySteps(records, [{ step: 200, fid: 50 }, { step: 100, fid: 80 }, { step: 250, fid: 40 }]);
  assert.deepEqual(merged.map((r) => r.step), [0, 100, 200, 250, 300]);
  assert.deepEqual(merged[2], { step: 200, clean: 0.1, test_clean: 0.15, fid: 50 });
  assert.deepEqual(merged[1], { step: 100, clean: 0.2, fid: 80 });
  assert.deepEqual(merged[3], { step: 250, fid: 40 });
  assert.equal(records.length, 4);                       // the tail's own array is untouched
  assert.equal(records[2].fid, undefined);
  assert.equal(mergeBySteps(records, []), records);
});

test('the side file is read by step, later lines win, bad lines are skipped', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'side-'));
  const file = path.join(dir, 'gen_metrics.jsonl');
  fs.writeFileSync(file, '{"step": 50000, "fid": 90}\nnot json\n{"step": 25000, "fid": 120, "kid": 100}\n{"step": 50000, "fid": 88, "kid": 70}\n');
  const side = readSideRecords(file);
  assert.deepEqual(side.find((r) => r.step === 50000), { step: 50000, fid: 88, kid: 70 });
  assert.equal(side.length, 2);
  assert.deepEqual(readSideRecords(path.join(dir, 'missing.jsonl')), []);
});
