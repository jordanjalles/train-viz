import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreSeries, firstLine, thin, machineOf } from '../lib/experiments.js';

test('scoreSeries: side records win at an equal step, keys without data are left out', () => {
  const records = [{ step: 100, fid: 130, test_clean: 0.5 }, { step: 200, fid: 120 }];
  const side = [{ step: 200, fid_ema: 118 }, { step: 100, fid_ema: 125, fid: 131 }];
  const s = scoreSeries(records, side);
  assert.deepEqual(s.fid, [[100, 131], [200, 120]]);
  assert.deepEqual(s.fid_ema, [[100, 125], [200, 118]]);
  assert.deepEqual(s.test_clean, [[100, 0.5]]);
  assert.equal(s.kid_ema, undefined);
});

test('scoreSeries: thinning a dense key keeps the score steps', () => {
  const records = Array.from({ length: 500 }, (_, i) => ({ step: i * 1000, test_clean: 1 / (i + 1) }));
  const s = scoreSeries(records, [], [123000]);
  assert.ok(s.test_clean.length <= 81);
  assert.ok(s.test_clean.some(([st]) => st === 123000));
  assert.deepEqual(s.test_clean.at(-1)[0], 499000);
});

test('thin keeps first and last', () => {
  const p = Array.from({ length: 10 }, (_, i) => [i, i]);
  assert.deepEqual(thin(p, 3), [[0, 0], [5, 5], [9, 9]]);
});

test('firstLine strips headings and blank lines; machineOf maps roots', () => {
  assert.equal(firstLine('\n# What this run tests\nmore'), 'What this run tests');
  assert.equal(firstLine(null), null);
  assert.equal(machineOf({ root: 'wsl3090' }), '3090');
  assert.equal(machineOf({ root: 'shared', config: { backend: 'colab' } }), 'Colab');
  assert.equal(machineOf({ root: 'shared' }), 'Spark');
});
