import { test } from 'node:test';
import assert from 'node:assert/strict';
import { covariances } from '../lib/cov.js';

function gauss(n, seed = 3) {
  let s = seed; const u = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
  return out;
}

test('white noise gives ~identity token and channel covariance', () => {
  const N = 2000, T = 8, C = 6;
  const { tok, chan } = covariances(gauss(N * T * C), N, T, C);
  for (let a = 0; a < T; a++) for (let b = 0; b < T; b++) assert.ok(Math.abs(tok[a * T + b] - (a === b ? 1 : 0)) < 0.06);
  for (let a = 0; a < C; a++) for (let b = 0; b < C; b++) assert.ok(Math.abs(chan[a * C + b] - (a === b ? 1 : 0)) < 0.06);
});

test('a duplicated channel shows as off-diagonal channel covariance; mean offset is removed', () => {
  const N = 2000, T = 4, C = 3;
  const z = gauss(N * T * C, 11);
  for (let i = 0; i < N * T; i++) { z[i * C + 2] = z[i * C + 0]; z[i * C + 1] += 5; }   // ch2 = ch0; ch1 offset
  const { chan } = covariances(z, N, T, C);
  assert.ok(chan[0 * C + 2] > 0.9 && chan[2 * C + 0] > 0.9);
  assert.ok(Math.abs(chan[1 * C + 1] - 1) < 0.08);                                    // offset centred away
});
