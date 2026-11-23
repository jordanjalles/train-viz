// 4-D uint8 tensors (z_images.pt thumbnails) and 3-D fp16 snapshots both parse.
// Uses real files from the Spark ImageNet run when present; skips otherwise.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { loadTensor } from '../lib/torchpt.js';

const dir = '<data-root>/outputs/training-runs/2026-10-02_221527_524159_ae_25M_selfteach_imagenet_128_maskcrop_ww03';
const have = fs.existsSync(path.join(dir, 'z_images.pt'));

test('uint8 N×3×H×W thumbnails', { skip: !have && 'no z_images.pt here' }, async () => {
  const t = await loadTensor(path.join(dir, 'z_images.pt'));
  assert.equal(t.dtype, 'u8');
  assert.deepEqual(t.shape, [512, 3, 64, 64]);
  assert.ok(t.data instanceof Uint8Array);
  assert.equal(t.data.length, 512 * 3 * 64 * 64);
});

test('fp16 N×T×C snapshot still parses', { skip: !have && 'no snapshots here' }, async () => {
  const f = fs.readdirSync(dir).find((n) => /^z_\d+\.pt$/.test(n));
  const t = await loadTensor(path.join(dir, f));
  assert.equal(t.dtype, 'f16');
  assert.deepEqual(t.shape, [512, 48, 48]);
});
