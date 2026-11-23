// node --test test/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadTensor } from '../lib/torchpt.js';

const SAMPLE = process.env.TRAIN_VIZ_SAMPLE_PT
  || '<data-root>/workspaces/stl10-selfteach/runs/E1_hp11_slices1024/z_000000.pt';

test('loads a real z snapshot', { skip: !fs.existsSync(SAMPLE) && 'no sample .pt on this host' }, async () => {
  const t = await loadTensor(SAMPLE);
  assert.equal(t.dtype, 'f16');
  assert.equal(t.shape.length, 3);
  assert.equal(t.data.length, t.shape.reduce((a, b) => a * b, 1));
  let finite = 0;
  for (let i = 0; i < t.data.length; i += 997) if (Number.isFinite(t.data[i])) finite++;
  assert.ok(finite > 0);
});
