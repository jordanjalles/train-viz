// alignAxes: a mirrored / axis-swapped copy of a cloud must come back to the reference exactly.
import { test } from 'node:test';
import assert from 'node:assert/strict';

// constellation.js imports 'math' via the browser import map; load just the pure function.
const src = await import('node:fs').then((fs) => fs.readFileSync(new URL('../public/constellation.js', import.meta.url), 'utf8'));
const body = src.slice(src.indexOf('const PERMS'), src.indexOf('function compile('));
const { alignAxes } = await import(`data:text/javascript,${encodeURIComponent(body)}`);

function cloud(n, seed = 1) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5;
  const p = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { p[3 * i] = 3 * rnd(); p[3 * i + 1] = 2 * rnd(); p[3 * i + 2] = rnd(); }
  return p;
}

test('undoes sign flips and axis swaps', () => {
  const ref = cloud(500);
  const moved = new Float32Array(ref.length);
  for (let i = 0; i < 500; i++) {                  // x -> -z, y -> x, z -> -y
    moved[3 * i] = ref[3 * i + 1]; moved[3 * i + 1] = -ref[3 * i + 2]; moved[3 * i + 2] = -ref[3 * i];
  }
  alignAxes(moved, ref);
  for (let i = 0; i < ref.length; i++) assert.ok(Math.abs(moved[i] - ref[i]) < 1e-6);
});

test('leaves an already-aligned cloud untouched', () => {
  const ref = cloud(200, 7);
  const same = Float32Array.from(ref);
  const r = alignAxes(same, ref);
  assert.deepEqual(r.perm, [0, 1, 2]);
  assert.deepEqual(r.sign, [1, 1, 1]);
});
