// Wasserstein regularisers of stl10-selfteach/train.py, re-evaluated on a
// saved z snapshot (N, T, C) against a seeded standard-normal prior of the
// same shape. Mirrors the four terms exactly (sorted-projection W2²):
//
//   bwd       sort along the batch axis, per (t, c)             "batch axis"
//   swd_axis  channel-sliced: rows = N·T, project C → n slices  "channel axis"
//   swd_tok   token-sliced:   rows = N·C, project T → n slices  "token axis"
//   swd_flat  whole volume:   rows = N, project T·C → n slices  "independent of axes"
//
// Also returns per-position marginal maps for the display: W2² per (t, c)
// (the bwd map), and the unsliced per-channel / per-token marginal W2².
// Estimates use the eval batch (N images), not the train batch, so they are
// comparable across steps of one run but not equal to the logged `reg`.

// --- seeded RNG (mulberry32) + Box–Muller normals --------------------------
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function randn(n, seed) {
  const r = rng(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = 1 - r(), v = r(), m = Math.sqrt(-2 * Math.log(u));
    out[i] = m * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** mean over columns of (sort(col(z)) - sort(col(p)))², for a (rows × cols) row-major pair. */
function sortedW2(z, p, rows, cols, perCol = null) {
  const a = new Float32Array(rows), b = new Float32Array(rows);
  let total = 0;
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) { a[r] = z[r * cols + c]; b[r] = p[r * cols + c]; }
    a.sort(); b.sort();
    let s = 0;
    for (let r = 0; r < rows; r++) { const d = a[r] - b[r]; s += d * d; }
    s /= rows;
    if (perCol) perCol[c] = s;
    total += s;
  }
  return total / cols;
}

/** Random unit directions (dim × n) as in F.normalize(randn(dim, n), dim=0). */
function slices(dim, n, seed) {
  const s = randn(dim * n, seed);
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += s[i * n + j] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i++) s[i * n + j] /= norm;
  }
  return s;
}

/** Sliced W2² of (rows × dim) z vs p through n random directions. */
function slicedW2(z, p, rows, dim, n, seed) {
  const s = slices(dim, n, seed);
  const pz = new Float32Array(rows * n), pp = new Float32Array(rows * n);
  for (let r = 0; r < rows; r++) {
    const zo = r * dim, po = r * n;
    for (let j = 0; j < n; j++) {
      let az = 0, ap = 0;
      for (let i = 0; i < dim; i++) { const w = s[i * n + j]; az += z[zo + i] * w; ap += p[zo + i] * w; }
      pz[po + j] = az; pp[po + j] = ap;
    }
  }
  return sortedW2(pz, pp, rows, n);
}

/** Transpose (N, T, C) → (N, C, T). */
function swapTC(z, N, T, C) {
  const out = new Float32Array(z.length);
  for (let n = 0; n < N; n++) for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) out[(n * C + c) * T + t] = z[(n * T + t) * C + c];
  return out;
}

/**
 * @param z Float32Array (N·T·C) row-major
 * @param cfg run config: { swd_square, swd_slices, ww, swd_flat_on }
 */
export function regTerms(z, N, T, C, cfg = {}, { seed = 1234, maxFlatSlices = 64 } = {}) {
  const sq = (x) => (cfg.swd_square ? x * x : x);
  const prior = randn(z.length, seed);

  // bwd: rows = N (batch), cols = T·C, keep the per-(t,c) map
  const bwdMap = new Float32Array(T * C);
  const bwdRaw = sortedW2(z, prior, N, T * C, bwdMap);

  // channel-sliced: rows = N·T, dim = C
  const chanRaw = slicedW2(z, prior, N * T, C, 32, seed + 1);
  const chanMarg = new Float32Array(C);
  sortedW2(z, prior, N * T, C, chanMarg);

  // token-sliced: rows = N·C, dim = T (on the transposed volume)
  const zt = swapTC(z, N, T, C), pt = swapTC(prior, N, T, C);
  const tokRaw = slicedW2(zt, pt, N * C, T, 32, seed + 2);
  const tokMarg = new Float32Array(T);
  sortedW2(zt, pt, N * C, T, tokMarg);

  // flat: rows = N, dim = T·C, n = swd_slices (capped for CPU time)
  const nFlat = Math.min(cfg.swd_slices || 32, maxFlatSlices);
  const flatRaw = slicedW2(z, prior, N, T * C, nFlat, seed + 3);

  const ww = cfg.ww ?? 1, flatOn = cfg.swd_flat_on ?? 1;
  const terms = { bwd: sq(bwdRaw), chan: sq(chanRaw), tok: sq(tokRaw), flat: sq(flatRaw) };
  const reg = ww * (flatOn * terms.flat + terms.bwd + terms.chan + terms.tok);

  // volume stats for the display
  let mean = 0, m2 = 0, amax = 0;
  for (let i = 0; i < z.length; i++) { const v = z[i]; mean += v; m2 += v * v; if (Math.abs(v) > amax) amax = Math.abs(v); }
  mean /= z.length; const std = Math.sqrt(Math.max(0, m2 / z.length - mean * mean));

  return {
    terms, raw: { bwd: bwdRaw, chan: chanRaw, tok: tokRaw, flat: flatRaw }, reg,
    weighted: { flat: ww * flatOn * terms.flat, bwd: ww * terms.bwd, chan: ww * terms.chan, tok: ww * terms.tok },
    flatSlices: nFlat, squared: !!cfg.swd_square, ww,
    bwdMap: Array.from(bwdMap), chanMarg: Array.from(chanMarg), tokMarg: Array.from(tokMarg),
    stats: { mean, std, absMax: amax },
  };
}
