import { test } from 'node:test';
import assert from 'node:assert/strict';
import { appendRecords } from '../lib/runs.js';

const steps = (a) => a.map((r) => r.step);

test('plain appends keep order', () => {
  const a = appendRecords([], [{ step: 0 }, { step: 100 }]);
  appendRecords(a, [{ step: 200 }]);
  assert.deepEqual(steps(a), [0, 100, 200]);
});

test('a resumed run re-sending its history does not duplicate it', () => {
  // live history 0..300, then the rewritten file is re-read from the top: 0..200 kept, resume at 200
  const a = appendRecords([], [0, 100, 200, 300].map((step) => ({ step })));
  appendRecords(a, [0, 100, 200, 300, 400].map((step) => ({ step, v: 'new' })));
  assert.deepEqual(steps(a), [0, 100, 200, 300, 400]);
  assert.ok(a.every((r) => r.v === 'new'));
});

test('a resume that rewinds past the last record drops everything from the resume step on', () => {
  const a = appendRecords([], [0, 100, 200, 300].map((step) => ({ step })));
  appendRecords(a, [{ step: 200, v: 'resumed' }, { step: 300, v: 'resumed' }]);
  assert.deepEqual(steps(a), [0, 100, 200, 300]);
  assert.equal(a.at(-1).v, 'resumed');
});
