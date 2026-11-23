import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pca, pcaBasis } from '../lib/pca.js';

test('pca recovers a planted dominant direction', () => {
  const n = 2000, c = 8;
  const data = new Float32Array(n * c);
  let s = 1;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  for (let i = 0; i < n; i++) {
    const big = rnd() * 20;
    for (let j = 0; j < c; j++) data[i * c + j] = rnd() + (j === 3 ? big : 0);
  }
  const p = pca(data, n, c, 3);
  assert.equal(p.projected.length, n * 3);
  assert.ok(p.explained[0] > 0.9, `explained[0]=${p.explained[0]}`);
  assert.ok(p.explained[0] >= p.explained[1] && p.explained[1] >= p.explained[2]);
});

test('pcaBasis serves a lower group of PCs that matches the full decomposition', () => {
  const n = 3000, c = 9;
  const data = new Float32Array(n * c);
  let s = 7;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  // channel j has std ∝ (c - j): PCs come out in channel order, each well separated
  for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) data[i * c + j] = rnd() * (c - j);
  const all = pca(data, n, c, 6);
  const b = pcaBasis(data, n, c);
  const low = b.project(3, 3);   // PCs 4–6, asked for first
  b.project(0, 3);
  assert.equal(low.projected.length, n * 3);
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(low.explained[k] - all.explained[3 + k]) < 1e-9);
  for (let i = 0; i < 50; i++) for (let k = 0; k < 3; k++) assert.ok(Math.abs(low.projected[i * 3 + k] - all.projected[i * 6 + 3 + k]) < 1e-4);
  assert.ok(low.explained[0] < all.explained[2] && low.explained[2] > 0);
  assert.equal(b.project(c - 1, 3).projected.length, n);   // asking past c returns only what exists
});
