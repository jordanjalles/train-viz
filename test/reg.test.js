import { test } from 'node:test';
import assert from 'node:assert/strict';
import { regTerms } from '../lib/reg.js';

function randn(n, seed = 7) {
  let a = seed; const r = () => { a = (a * 16807) % 2147483647; return a / 2147483647; };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  return out;
}

test('a standard-normal volume scores near zero on every term', () => {
  const N = 256, T = 8, C = 8;
  const r = regTerms(randn(N * T * C), N, T, C, { swd_square: 0, swd_slices: 16, ww: 1 });
  for (const [k, v] of Object.entries(r.terms)) assert.ok(v < 0.05, `${k}=${v}`);
  assert.equal(r.bwdMap.length, T * C); assert.equal(r.chanMarg.length, C); assert.equal(r.tokMarg.length, T);
});

test('a collapsed volume (all zeros) scores ≈ 1 on the batch term', () => {
  const N = 256, T = 8, C = 8;
  const r = regTerms(new Float32Array(N * T * C), N, T, C, { swd_square: 0, swd_slices: 16, ww: 1 });
  assert.ok(Math.abs(r.terms.bwd - 1) < 0.15, `bwd=${r.terms.bwd}`); // E[x²] of N(0,1)
  assert.ok(r.terms.chan > 0.5 && r.terms.tok > 0.5 && r.terms.flat > 0.5);
});

test('a scaled volume is penalised more on every term than the unit one', () => {
  const N = 256, T = 8, C = 8;
  const z = randn(N * T * C); const z3 = z.map((v) => v * 3);
  const a = regTerms(z, N, T, C, {}), b = regTerms(z3, N, T, C, {});
  for (const k of Object.keys(a.terms)) assert.ok(b.terms[k] > a.terms[k] * 5, k);
});
