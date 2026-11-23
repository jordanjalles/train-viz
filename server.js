#!/usr/bin/env node
// train-viz — live visualizer for the AE training runs on this DGX Spark.
// Zero runtime dependencies. Binds to loopback (or whatever config.json
// says); expose it over the private network with `a private proxy`, never 0.0.0.0.

import http from 'node:http';
import fs from 'node:fs';
import zlib from 'node:zlib';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RunStore } from './lib/runs.js';
import { ExperimentStore } from './lib/experiments.js';
import { GpuPoller } from './lib/gpu.js';
import { loadTensor } from './lib/torchpt.js';
import { regTerms } from './lib/reg.js';
import { pcaBasis } from './lib/pca.js';
import { covariances } from './lib/cov.js';
import { RunState } from './lib/state.js';
import { encodePng, resizeRgb } from './lib/png.js';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');
const config = JSON.parse(await readFile(path.join(ROOT, 'config.json'), 'utf8'));
const HOST = process.env.TRAIN_VIZ_HOST || config.host || '127.0.0.1';
const PORT = parseInt(process.env.TRAIN_VIZ_PORT || config.port || 8600, 10);

const roots = config.roots.filter((r) => fs.existsSync(r.path));
const store = new RunStore(roots);
const gpu = new GpuPoller({ intervalMs: 2000 });
const runState = new RunState(path.join(ROOT, 'state', 'run-state.json'));   // kept / archived runs
const experiments = new ExperimentStore(path.join(ROOT, 'experiments.json'), store);   // groups of comparable runs
store.on('error', (e) => console.error('[runs]', e.message));

// ---- SSE -------------------------------------------------------------------
const clients = new Set();
function broadcast(event, data) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) res.write(msg);
}
store.on('records', (d) => broadcast('records', d));
store.on('run-added', (d) => broadcast('run', d));
store.on('run-updated', (d) => broadcast('run', d));
store.on('run-removed', (d) => broadcast('run-removed', d));
gpu.on('sample', (s) => broadcast('gpu', s));

// ---- latent cache ----------------------------------------------------------
// One entry per (run, step): the decoded (N, T, C) volume plus the Wasserstein
// terms evaluated on it. Requests then slice K images out of it.
const zCache = new Map(); // "id@step" -> {t, n, tok, c, reg}
const Z_CACHE_MAX = 8;
const zPending = new Map();

// ---- z image thumbnails ----------------------------------------------------
// The images behind a run's z snapshots: one PNG per image for the Latent view's
// focus control, and an atlas of all of them for the 3-D image sprites.
const zImgCache = new Map(); // run id -> {mtime, t} for z_images.pt
const atlasCache = new Map(); // "id@cell" -> {key, png, n, cols, cell}
async function zImageCount(run) {
  if (run.zImageSource === 'file') { await zImage(run, 0); return zImgCache.get(run.id).t.shape[0]; }
  if (run.zImageSource === 'stl10') {
    const bin = roots.find((r) => r.id === run.root)?.stl10TestBin;
    const n = Math.floor((await fs.promises.stat(bin)).size / STL_BYTES);
    return Math.min(n, 1024);   // harness snapshots hold the first 4 test batches of 256
  }
  return 0;
}
async function zAtlas(run, cell) {
  const key = `${run.id}@${cell}`;
  const n = await zImageCount(run);
  const stamp = run.zImageSource === 'file' ? zImgCache.get(run.id).mtime : 'stl10';
  const hit = atlasCache.get(key);
  if (hit && hit.stamp === stamp && hit.n === n) return hit;
  const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
  const W = cols * cell, H = rows * cell, rgb = Buffer.alloc(W * H * 3);
  for (let i = 0; i < n; i++) {
    const img = await zImage(run, i); if (!img) continue;
    const t = resizeRgb(img.rgb, img.w, img.h, cell);
    const ox = (i % cols) * cell, oy = Math.floor(i / cols) * cell;
    for (let y = 0; y < cell; y++) t.copy(rgb, 3 * ((oy + y) * W + ox), 3 * y * cell, 3 * (y + 1) * cell);
  }
  const out = { stamp, n, cols, cell, png: encodePng(W, H, rgb) };
  atlasCache.set(key, out);
  if (atlasCache.size > 6) atlasCache.delete(atlasCache.keys().next().value);
  return out;
}
const STL_SIDE = 96, STL_BYTES = 3 * 96 * 96;
async function zImage(run, i) {
  if (run.zImageSource === 'file') {
    const file = path.join(run.dir, 'z_images.pt');
    const mtime = (await fs.promises.stat(file)).mtimeMs;
    let c = zImgCache.get(run.id);
    if (!c || c.mtime !== mtime) {
      c = { mtime, t: await loadTensor(file) };
      zImgCache.set(run.id, c);
      if (zImgCache.size > 4) zImgCache.delete(zImgCache.keys().next().value);
    }
    const [n, ch, hgt, wid] = c.t.shape;
    if (c.t.dtype !== 'u8' || ch !== 3 || i >= n) return null;
    const plane = hgt * wid, base = i * 3 * plane, rgb = Buffer.alloc(3 * plane);
    for (let p = 0; p < plane; p++) for (let k = 0; k < 3; k++) rgb[3 * p + k] = c.t.data[base + k * plane + p];
    return { w: wid, h: hgt, rgb };
  }
  if (run.zImageSource === 'stl10') {
    // STL-10 binary: per image 3 channels of 96×96, each stored column-major
    const bin = roots.find((r) => r.id === run.root)?.stl10TestBin;
    const fh = await fs.promises.open(bin, 'r');
    try {
      const src = Buffer.alloc(STL_BYTES);
      const { bytesRead } = await fh.read(src, 0, STL_BYTES, i * STL_BYTES);
      if (bytesRead < STL_BYTES) return null;
      const rgb = Buffer.alloc(STL_BYTES);
      for (let x = 0; x < STL_SIDE; x++) for (let y = 0; y < STL_SIDE; y++) for (let k = 0; k < 3; k++) {
        rgb[3 * (y * STL_SIDE + x) + k] = src[k * STL_SIDE * STL_SIDE + x * STL_SIDE + y];
      }
      return { w: STL_SIDE, h: STL_SIDE, rgb };
    } finally { await fh.close(); }
  }
  return null;
}
async function latentVolume(run, step) {
  const key = `${run.id}@${step}`;
  if (zCache.has(key)) { const v = zCache.get(key); zCache.delete(key); zCache.set(key, v); return v; }
  if (zPending.has(key)) return zPending.get(key);
  const p = (async () => {
    const t = await loadTensor(store.zFile(run, step));
    const [n, tok, c] = t.shape.length === 3 ? t.shape : [t.shape[0], 1, t.shape[1]];
    const t0 = Date.now();
    const reg = regTerms(t.data, n, tok, c, run.config || {});
    const v = { t, n, tok, c, reg, ms: Date.now() - t0 };
    zCache.set(key, v);
    if (zCache.size > Z_CACHE_MAX) zCache.delete(zCache.keys().next().value);
    return v;
  })();
  zPending.set(key, p);
  try { return await p; } finally { zPending.delete(key); }
}
/** u32 length + JSON header (4-byte padded) + raw float32 payloads. */
function packBody(header, ...arrays) {
  const json = Buffer.from(JSON.stringify(header), 'utf8');
  const jsonLen = Math.ceil(json.length / 4) * 4;
  const total = arrays.reduce((a, x) => a + x.byteLength, 0);
  const body = Buffer.alloc(4 + jsonLen + total);
  body.writeUInt32LE(json.length, 0); json.copy(body, 4);
  let off = 4 + jsonLen;
  for (const x of arrays) { Buffer.from(x.buffer, x.byteOffset, x.byteLength).copy(body, off); off += x.byteLength; }
  return body;
}
const PCA_MAX_PC = 48;   // deepest component the latent view can ask for (power iteration is O(c²) per PC)
/** PCA constellation: every (image, token) row of the volume → 3-D on PCs from+1..from+3 (0-based from), plus its L2 norm. */
function latentPca(v, step, from = 0) {
  if (!v.pca) {
    const { t, n, tok, c } = v;
    const rows = n * tok;
    const basis = pcaBasis(t.data, rows, c);
    const norms = new Float32Array(rows);
    const domChan = new Float32Array(rows); // channel that is furthest from its own mean, in its own std, in each token vector
    // Per-channel mean and std over all rows. A residual stream can hold one channel with a large constant offset
    // (2026-10-11: channel 796 of 800, mean -146 against a median RMS of 1.9). Raw |z| then names that channel in 98%
    // of the tokens and the norm shows only the offset. Both are taken from the centred values, as the PCA is.
    const mean = new Float64Array(c), sd = new Float64Array(c);
    for (let i = 0; i < rows; i++) for (let j = 0; j < c; j++) mean[j] += t.data[i * c + j];
    for (let j = 0; j < c; j++) mean[j] /= rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < c; j++) { const d = t.data[i * c + j] - mean[j]; sd[j] += d * d; }
    for (let j = 0; j < c; j++) sd[j] = Math.sqrt(sd[j] / rows) || 1;
    for (let i = 0; i < rows; i++) {
      let s = 0, best = 0, bestAbs = -1;
      for (let j = 0; j < c; j++) { const d = t.data[i * c + j] - mean[j]; s += d * d; const a = Math.abs(d) / sd[j]; if (a > bestAbs) { bestAbs = a; best = j; } }
      norms[i] = Math.sqrt(s); domChan[i] = best;
    }
    v.pca = { basis, norms, domChan, groups: new Map() };   // groups: from → { projected, explained, eigenvalues }
  }
  const { t, n, tok, c } = v;
  const maxPc = Math.min(c, PCA_MAX_PC);
  from = Math.max(0, Math.min(from | 0, maxPc - 3));
  if (!v.pca.groups.has(from)) v.pca.groups.set(from, v.pca.basis.project(from, 3));
  const g = v.pca.groups.get(from);
  const header = { step, n, tokens: tok, channels: c, dtype: t.dtype, pcFrom: from, maxPc, explained: g.explained, eigenvalues: g.eigenvalues, layout: 'u32 len, json, float32 xyz[n*tokens], float32 norm[n*tokens], float32 dominantChannel[n*tokens]' };
  return packBody(header, g.projected, v.pca.norms, v.pca.domChan);
}
/** Slice K images (evenly strided) out of the volume for the point cube. */
function latentSlice(v, step, K) {
  const { t, n, tok, c, reg } = v;
  K = Math.max(1, Math.min(n, K || 64));
  const stride = Math.max(1, Math.floor(n / K));
  const per = tok * c;
  const header = { step, n, tokens: tok, channels: c, dtype: t.dtype, images: K, stride, reg, regMs: v.ms, layout: 'uint32 jsonLength, json header (padded to 4 bytes), float32 z[images][tokens][channels]' };
  // Body = u32 length + JSON header (padded to a 4-byte boundary) + float32 values.
  // The header carries the heat-map arrays, too big for an HTTP header.
  const json = Buffer.from(JSON.stringify(header), 'utf8');
  const jsonLen = Math.ceil(json.length / 4) * 4;
  const body = Buffer.alloc(4 + jsonLen + K * per * 4);
  body.writeUInt32LE(json.length, 0);
  json.copy(body, 4);
  const out = new Float32Array(body.buffer, body.byteOffset + 4 + jsonLen, K * per);
  for (let k = 0; k < K; k++) out.set(t.data.subarray(k * stride * per, (k * stride + 1) * per), k * per);
  return { body };
}

// ---- helpers ---------------------------------------------------------------
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.ico': 'image/x-icon' };
// Over the private network to a phone, bytes are the cost (2026-11-11: /api/runs was 3.1 MB raw, 0.6 MB gzipped).
// Gzip any text body over 1 kB when the client accepts it.
function sendBody(res, code, headers, body) {
  const gzip = body.length > 1024 && /\bgzip\b/.test(res.req?.headers['accept-encoding'] || '');
  const out = gzip ? zlib.gzipSync(body, { level: 4 }) : body;
  res.writeHead(code, { ...headers, 'content-length': out.length, vary: 'accept-encoding', ...(gzip ? { 'content-encoding': 'gzip' } : {}) });
  res.end(out);
}
function json(res, code, data) {
  sendBody(res, code, { 'content-type': 'application/json', 'cache-control': 'no-store' }, Buffer.from(JSON.stringify(data)));
}
const TEXT = new Set(['.html', '.js', '.css', '.json', '.svg']);
function sendFile(res, file, extraHeaders = {}) {
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); res.end('not found'); return; }
    const type = MIME[path.extname(file)] || 'application/octet-stream';
    // a validator lets the phone reuse an unchanged file: one small 304 instead of the whole file
    const etag = `"${st.size.toString(36)}-${Math.floor(st.mtimeMs).toString(36)}"`;
    if (res.req?.headers['if-none-match'] === etag) { res.writeHead(304, { etag, ...extraHeaders }); return res.end(); }
    const headers = { 'content-type': type, etag, ...extraHeaders };
    if (TEXT.has(path.extname(file))) {
      return fs.readFile(file, (e, buf) => (e ? (res.writeHead(500), res.end()) : sendBody(res, 200, headers, buf)));
    }
    res.writeHead(200, { ...headers, 'content-length': st.size });
    fs.createReadStream(file).pipe(res);
  });
}
function runFrom(res, id) {
  const run = store.get(id);
  if (!run) json(res, 404, { error: 'unknown run' });
  return run;
}

// ---- server ----------------------------------------------------------------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const p = url.pathname;
  try {
    if (p === '/api/experiments') {     // ~600 kB of JSON for 500 runs: gzip it for the phone
      const body = JSON.stringify(experiments.list());
      const gzip = /\bgzip\b/.test(req.headers['accept-encoding'] || '');
      const out = gzip ? zlib.gzipSync(body) : Buffer.from(body);
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': out.length, 'cache-control': 'no-store', vary: 'accept-encoding', ...(gzip ? { 'content-encoding': 'gzip' } : {}) });
      return res.end(out);
    }
    if (p === '/api/runs') return json(res, 200, { roots: roots.map(({ id, label, kind }) => ({ id, label, kind })), runs: store.list(), state: runState });
    if (p === '/api/state') {
      if (req.method === 'GET') return json(res, 200, runState);
      if (req.method !== 'POST') return json(res, 405, { error: 'GET or POST' });
      let body = '';
      for await (const chunk of req) { body += chunk; if (body.length > 256 * 1024) return json(res, 413, { error: 'too large' }); }
      let r;
      try { const { op, ids } = JSON.parse(body); r = runState.apply(op, ids); } catch (e) { return json(res, 400, { error: e.message }); }
      broadcast('state', runState);
      return json(res, 200, { ...r, state: runState });
    }
    if (p === '/api/gpu') return json(res, 200, { available: gpu.available, latest: gpu.latest, history: gpu.history });
    if (p === '/api/events') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive', 'x-accel-buffering': 'no' });
      res.write(`event: hello\ndata: ${JSON.stringify({ t: Date.now() })}\n\n`);
      clients.add(res);
      const ping = setInterval(() => res.write(': ping\n\n'), 15000);
      req.on('close', () => { clearInterval(ping); clients.delete(res); });
      return;
    }
    // Run ids are "<root>/<name>"; clients may send the slash literally or as %2F.
    const m = /^\/api\/runs\/(.+?)(?:\/(recon|emarecon|robust|image|zimg|zatlas|z|pca|cov)\/([^/]+))?$/.exec(p);
    if (m) {
      const id = decodeURIComponent(m[1]);
      const sub = m[2], arg = m[3] && decodeURIComponent(m[3]);
      if (!sub) {
        const full = store.full(id);
        return full ? json(res, 200, full) : json(res, 404, { error: 'unknown run' });
      }
      const run = runFrom(res, id); if (!run) return;
      if (sub === 'recon') {
        if (!/^\d+$/.test(arg)) return json(res, 400, { error: 'bad step' });
        const rf = store.reconFile(run, parseInt(arg, 10));
        if (!rf) return json(res, 404, { error: 'no reconstruction at that step' });
        return sendFile(res, rf.path, { 'cache-control': 'max-age=3600', 'x-recon-kind': rf.kind });
      }
      if (sub === 'emarecon') {        // the grid of that step from the averaged (EMA) weights: emaprev_NNNNNN.png
        if (!/^\d+$/.test(arg)) return json(res, 400, { error: 'bad step' });
        const ef = store.emaReconFile(run, parseInt(arg, 10));
        if (!ef) return json(res, 404, { error: 'no EMA sample grid at that step' });
        return sendFile(res, ef.path, { 'cache-control': 'max-age=3600' });
      }
      if (sub === 'robust') {          // latent-noise robustness strip (input + decodes at each noise level)
        const step = parseInt(arg, 10);
        if (!/^\d+$/.test(arg) || !run.robustPngs?.includes(step)) return json(res, 404, { error: 'no robustness strip at that step' });
        return sendFile(res, path.join(run.dir, `robust_${String(step).padStart(6, '0')}.png`), { 'cache-control': 'max-age=3600' });
      }
      if (sub === 'image') {
        if (!run.imageDir || !/^[A-Za-z0-9_.-]+\.png$/.test(arg)) { res.writeHead(404); return res.end(); }
        return sendFile(res, path.join(run.imageDir, arg), { 'cache-control': 'no-store' });
      }
      if (sub === 'zimg') {            // z snapshot image #i as PNG
        if (!/^\d+$/.test(arg) || !run.zImageSource) return json(res, 404, { error: 'no image thumbnails for this run' });
        const img = await zImage(run, parseInt(arg, 10));
        if (!img) return json(res, 404, { error: 'no thumbnail for that image' });
        const png = encodePng(img.w, img.h, img.rgb);
        res.writeHead(200, { 'content-type': 'image/png', 'content-length': png.length, 'cache-control': 'max-age=3600' });
        return res.end(png);
      }
      if (sub === 'zatlas') {          // all z snapshot images in one PNG grid, `cell` px each
        const cell = parseInt(arg, 10);
        if (!/^\d+$/.test(arg) || cell < 16 || cell > 96 || !run.zImageSource) return json(res, 404, { error: 'no image atlas for this run' });
        const a = await zAtlas(run, cell);
        res.writeHead(200, { 'content-type': 'image/png', 'content-length': a.png.length, 'x-count': a.n, 'x-cols': a.cols, 'x-cell': a.cell, 'cache-control': 'no-cache' });
        return res.end(a.png);
      }
      if (sub === 'cov') {             // token + channel covariance of z (centred across images)
        const step = parseInt(arg, 10);
        if (!store.zFile(run, step)) return json(res, 404, { error: 'no latent snapshot at that step' });
        const v = await latentVolume(run, step);
        if (!v.cov) { const t0 = Date.now(); v.cov = { ...covariances(v.t.data, v.n, v.tok, v.c), ms: Date.now() - t0 }; }
        return json(res, 200, { step, n: v.n, tokens: v.tok, channels: v.c, ms: v.cov.ms, tok: Array.from(v.cov.tok), chan: Array.from(v.cov.chan) });
      }
      if (sub === 'z' || sub === 'pca') {
        const step = parseInt(arg, 10);
        if (!store.zFile(run, step)) return json(res, 404, { error: 'no latent snapshot at that step' });
        const vol = await latentVolume(run, step);
        const body = sub === 'pca' ? latentPca(vol, step, parseInt(url.searchParams.get('pc') || '1', 10) - 1) : latentSlice(vol, step, parseInt(url.searchParams.get('images') || '64', 10)).body;
        res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': body.length, 'cache-control': 'max-age=3600' });
        return res.end(body);
      }
    }
    if (p.startsWith('/api/')) return json(res, 404, { error: 'no such endpoint' });

    // static
    let file = path.normalize(path.join(PUBLIC, p === '/' ? 'index.html' : p));
    if (!file.startsWith(PUBLIC)) { res.writeHead(403); return res.end(); }
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.js');
    return sendFile(res, file, { 'cache-control': p.startsWith('/vendor/') ? 'max-age=86400' : 'no-cache' });
  } catch (e) {
    console.error(req.method, p, e);
    if (!res.headersSent) json(res, 500, { error: e.message });
    else res.end();
  }
});

await store.start();
gpu.start();
server.listen(PORT, HOST, () => {
  console.log(`train-viz listening on http://${HOST}:${PORT}  roots: ${roots.map((r) => r.id).join(', ') || 'none'}`);
});
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { server.close(); process.exit(0); });
