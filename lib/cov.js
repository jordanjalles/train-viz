// Token and channel covariance of a latent snapshot z (N images × T tokens × C channels),
// computed on z centred across images, i.e. how the image-to-image variation co-varies:
//   tok[a][b]  = mean over (n, c) of zc[n,a,c] · zc[n,b,c]        (T × T)
//   chan[a][b] = mean over (n, t) of zc[n,t,a] · zc[n,t,b]        (C × C)
// A whitened latent gives ≈ identity for both; strong off-diagonal blocks are
// tokens / channels that move together (redundancy, rank collapse along that axis).
export function covariances(data, n, tok, c) {
  const per = tok * c;
  const mean = new Float64Array(per);
  for (let i = 0; i < n; i++) for (let j = 0; j < per; j++) mean[j] += data[i * per + j];
  for (let j = 0; j < per; j++) mean[j] /= n;
  const zc = new Float32Array(n * per);
  for (let i = 0; i < n; i++) for (let j = 0; j < per; j++) zc[i * per + j] = data[i * per + j] - mean[j];

  const tokCov = new Float64Array(tok * tok);
  const chanCov = new Float64Array(c * c);
  for (let i = 0; i < n; i++) {
    const o = i * per;
    for (let a = 0; a < tok; a++) {
      const ra = o + a * c;
      for (let b = a; b < tok; b++) {          // token pair: dot over channels
        const rb = o + b * c; let s = 0;
        for (let k = 0; k < c; k++) s += zc[ra + k] * zc[rb + k];
        tokCov[a * tok + b] += s;
      }
    }
    for (let t = 0; t < tok; t++) {
      const r = o + t * c;
      for (let a = 0; a < c; a++) {             // channel pair: outer product per token
        const va = zc[r + a];
        for (let b = a; b < c; b++) chanCov[a * c + b] += va * zc[r + b];
      }
    }
  }
  const out = (m, d, norm) => {
    const f = new Float32Array(d * d);
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) f[a * d + b] = f[b * d + a] = m[a * d + b] / norm;
    return f;
  };
  return { tok: out(tokCov, tok, n * c), chan: out(chanCov, c, n * tok) };
}

