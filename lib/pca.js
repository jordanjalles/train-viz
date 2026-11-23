// PCA of an (n × c) row-major Float32Array, via the c×c covariance and power
// iteration with deflation. c is small (≤ a few hundred), n can be tens of
// thousands. Deterministic, dependency-free.
//
// pcaBasis() builds the covariance once and finds components lazily, so a
// caller can ask for PCs 1–3 first and PCs 7–9 later without redoing the
// n·c² covariance pass.

export function pcaBasis(data, n, c) {
  const mean = new Float64Array(c);
  for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) mean[j] += data[i * c + j];
  for (let j = 0; j < c; j++) mean[j] /= n;

  // covariance (upper triangle, then mirror)
  const cov = new Float64Array(c * c);
  const row = new Float64Array(c);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < c; j++) row[j] = data[i * c + j] - mean[j];
    for (let a = 0; a < c; a++) {
      const ra = row[a];
      if (ra === 0) continue;
      for (let b = a; b < c; b++) cov[a * c + b] += ra * row[b];
    }
  }
  for (let a = 0; a < c; a++) for (let b = a; b < c; b++) {
    cov[a * c + b] /= Math.max(1, n - 1);
    cov[b * c + a] = cov[a * c + b];
  }
  let trace = 0;
  for (let a = 0; a < c; a++) trace += cov[a * c + a];

  const comps = [];
  const eigen = [];
  const v = new Float64Array(c), w = new Float64Array(c);
  /** Find components until there are k of them (capped at c). cov is deflated in place. */
  const extend = (k) => {
    k = Math.min(k, c);
    for (let comp = comps.length; comp < k; comp++) {
      // deterministic start vector
      for (let j = 0; j < c; j++) v[j] = 1 + 0.01 * ((j * 7919 + comp * 104729) % 13);
      let lambda = 0;
      for (let it = 0; it < 200; it++) {
        for (let a = 0; a < c; a++) {
          let s = 0;
          for (let b = 0; b < c; b++) s += cov[a * c + b] * v[b];
          w[a] = s;
        }
        let norm = 0;
        for (let a = 0; a < c; a++) norm += w[a] * w[a];
        norm = Math.sqrt(norm) || 1;
        let diff = 0;
        for (let a = 0; a < c; a++) { const nv = w[a] / norm; diff += Math.abs(nv - v[a]); v[a] = nv; }
        lambda = norm;
        if (diff < 1e-9) break;
      }
      // orient consistently: largest-magnitude loading positive
      let big = 0;
      for (let a = 1; a < c; a++) if (Math.abs(v[a]) > Math.abs(v[big])) big = a;
      if (v[big] < 0) for (let a = 0; a < c; a++) v[a] = -v[a];
      comps.push(Float64Array.from(v));
      eigen.push(lambda);
      // deflate
      for (let a = 0; a < c; a++) for (let b = 0; b < c; b++) cov[a * c + b] -= lambda * v[a] * v[b];
    }
  };
  /** Rows projected onto components [from, from + k): Float32Array (n × k), plus their eigenvalues. */
  const project = (from, k) => {
    extend(from + k);
    k = Math.max(0, Math.min(k, comps.length - from));
    const out = new Float32Array(n * k);
    for (let i = 0; i < n; i++) {
      for (let comp = 0; comp < k; comp++) {
        const u = comps[from + comp];
        let s = 0;
        for (let j = 0; j < c; j++) s += (data[i * c + j] - mean[j]) * u[j];
        out[i * k + comp] = s;
      }
    }
    const eigenvalues = eigen.slice(from, from + k);
    return { projected: out, eigenvalues, explained: eigenvalues.map((e) => (trace > 0 ? e / trace : 0)) };
  };
  return { mean: Array.from(mean), trace, extend, project };
}

export function pca(data, n, c, k = 3) {
  const b = pcaBasis(data, n, c);
  return { ...b.project(0, k), mean: b.mean };
}
