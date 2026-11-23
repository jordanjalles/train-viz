// Latent constellation: every latent token of every eval image (N × T points
// of C dims) projected to 3-D by PCA on the server, drawn as a WebGL2 point
// cloud: the *shape* of the latent distribution. math does the camera (quat
// orbit, mat4 perspective/lookAt) and the motion (spring-damped orbit + zoom,
// eased morph between snapshots). Mouse: drag orbit, wheel zoom, double-click
// reset. Touch (only once setTouch(true)): drag orbit, pinch zoom, double-tap reset.

import { vec3, mat4, quat, clamp } from 'math';
import { spring } from 'math/time';
import { ramp, ACCENT, ACCENT2 } from './palette.js';

const VS = `#version 300 es
precision highp float;
in vec3 a_p0; in vec3 a_p1; in float a_norm; in float a_index; in float a_chan;
uniform mat4 u_mvp; uniform float u_t; uniform float u_size; uniform float u_mode;
uniform float u_tokens; uniform float u_n; uniform float u_channels; uniform float u_normMax; uniform float u_dpr;
uniform float u_camDist; uniform float u_radius; uniform float u_fade; uniform float u_focusScale; uniform float u_dimAlpha;
uniform float u_focusImage; uniform float u_pointAlpha; uniform float u_hot;
out vec3 v_col; out float v_alpha; out float v_hot;
vec3 hsv2rgb(vec3 c){ vec4 K=vec4(1.,2./3.,1./3.,3.); vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www); return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y); }
vec3 rampc(float t){ // indigo → cyan → amber, mirrors palette.ramp
  float h = (250. - 210.*t)/360.; float l = 0.28+0.45*t; return hsv2rgb(vec3(h, 0.85, 0.35+0.65*l)); }
void main(){
  vec3 p = mix(a_p0, a_p1, u_t);
  vec4 clip = u_mvp * vec4(p, 1.0);
  gl_Position = clip;
  float tok = mod(a_index, u_tokens); float img = floor(a_index / u_tokens);
  float nrm = a_norm / max(u_normMax, 1e-6);
  if (u_mode < 0.5)      v_col = hsv2rgb(vec3(tok / u_tokens, 0.75, 0.95));
  else if (u_mode < 1.5) v_col = hsv2rgb(vec3(a_chan / u_channels, 0.75, 0.95));
  else                   v_col = rampc(clamp(nrm, 0., 1.));
  bool focusing = u_focusImage >= 0.0;
  bool mine = focusing && abs(img - u_focusImage) < 0.5;
  float focus = !focusing ? 1.0 : (mine ? 1.0 : u_dimAlpha);
  // depth cue relative to the cloud's own extent (independent of zoom): nearest face 1.0, farthest 1 - u_fade
  float depthT = clamp((clip.w - (u_camDist - u_radius)) / (2.0 * u_radius), 0.0, 1.0);
  // the focused image's tokens are drawn full-alpha and enlarged (u_focusScale) so 32 points read against 32k
  v_alpha = (mine ? 1.0 : (0.35 + 0.65 * clamp(nrm, 0., 1.)) * focus * (1.0 - u_fade * depthT)) * u_pointAlpha;
  float dist = max(clip.w, 0.1);
  // u_focusScale is an AREA multiplier (gl_PointSize is a diameter), and the rest keep their size
  float sizeMul = mine ? sqrt(u_focusScale) : 1.0;
  gl_PointSize = u_dpr * u_size * (0.7 + 0.9 * clamp(nrm, 0., 1.)) * sizeMul * 3.0 / dist;
  // the hovered token (second pass, one point): opaque, 2.4x wider, never under 16 CSS px
  v_hot = u_hot;
  if (u_hot > 0.5) { v_alpha = 1.0; gl_PointSize = max(gl_PointSize * 2.4, 16.0 * u_dpr); }
}`;
const FS = `#version 300 es
precision highp float;
in vec3 v_col; in float v_alpha; in float v_hot; out vec4 o;
void main(){ vec2 d = gl_PointCoord - 0.5; float r2 = dot(d,d); if (r2 > 0.25) discard;
  if (v_hot > 0.5) { // solid disc in the point colour, black line, white ring: reads on any background
    float r = sqrt(r2); o = vec4(r < 0.27 ? v_col : (r < 0.36 ? vec3(0.0) : vec3(1.0)), 1.0); return; }
  float a = smoothstep(0.25, 0.0, r2); o = vec4(v_col * a * v_alpha, a * v_alpha); }`;

// Image sprites: one textured square per eval image, at the mean 3-D position of its tokens.
// The atlas holds every image in a cols × rows grid; a sprite's size is in world units, so
// it shrinks with distance like the cloud does.
const SVS = `#version 300 es
precision highp float;
in vec3 a_c0; in vec3 a_c1; in float a_img;
uniform mat4 u_mvp; uniform float u_t; uniform float u_px; uniform float u_maxPt;
uniform float u_focusImage; uniform float u_cols;
out vec2 v_cell; out float v_alpha;
void main(){
  vec4 clip = u_mvp * vec4(mix(a_c0, a_c1, u_t), 1.0);
  bool focusing = u_focusImage >= 0.0;
  bool mine = focusing && abs(a_img - u_focusImage) < 0.5;
  gl_PointSize = clamp(u_px / max(clip.w, 0.05) * (mine ? 2.5 : 1.0), 2.0, u_maxPt);
  if (mine) clip.z = -0.999 * clip.w;   // the focused image is drawn in front of the rest
  gl_Position = clip;
  v_cell = vec2(mod(a_img, u_cols), floor(a_img / u_cols));
  v_alpha = (!focusing || mine) ? 1.0 : 0.3;
}`;
const SFS = `#version 300 es
precision highp float;
in vec2 v_cell; in float v_alpha; uniform sampler2D u_atlas; uniform vec2 u_grid; out vec4 o;
void main(){ vec3 c = texture(u_atlas, (v_cell + gl_PointCoord) / u_grid).rgb; o = vec4(c * v_alpha, v_alpha); }`;

/** Morph easing: 25% linear + 75% sine ease-in-out. Speed at each snapshot is ~0.25x the
 *  average, so the change of direction between consecutive morphs happens while the cloud
 *  is nearly still (no jerk), yet it never fully stops; in play mode each morph spans the
 *  whole interval, so there is no dwell either. */
export const softEase = (t) => 0.25 * t + 0.75 * (0.5 - 0.5 * Math.cos(Math.PI * t));

const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/** In place: permute and sign-flip the 3 axes of pts (rows × 3) to best match ref (same rows):
 *  maximise Σ_b |<pts[:, perm[b]], ref[:, b]>| over the 6 axis orders, then flip each axis
 *  whose inner product with its reference axis is negative. Pure relabelling of PCA axes —
 *  the cloud's shape is unchanged; only the arbitrary sign/order choices are made consistent. */
export function alignAxes(pts, ref) {
  const rows = pts.length / 3;
  const c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];            // c[a][b] = <pts axis a, ref axis b>
  for (let i = 0; i < rows; i++) {
    const o = 3 * i;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) c[a][b] += pts[o + a] * ref[o + b];
  }
  let best = PERMS[0], bestScore = -1;
  for (const p of PERMS) {
    const score = Math.abs(c[p[0]][0]) + Math.abs(c[p[1]][1]) + Math.abs(c[p[2]][2]);
    if (score > bestScore) { bestScore = score; best = p; }
  }
  const sign = best.map((a, b) => (c[a][b] < 0 ? -1 : 1));
  const tmp = [0, 0, 0];
  for (let i = 0; i < rows; i++) {
    const o = 3 * i;
    for (let b = 0; b < 3; b++) tmp[b] = sign[b] * pts[o + best[b]];
    pts[o] = tmp[0]; pts[o + 1] = tmp[1]; pts[o + 2] = tmp[2];
  }
  return { perm: best, sign };
}

function compile(gl, type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class LatentView {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.u = Object.fromEntries(['u_mvp', 'u_t', 'u_size', 'u_mode', 'u_tokens', 'u_n', 'u_channels', 'u_normMax', 'u_dpr', 'u_focusImage', 'u_camDist', 'u_radius', 'u_fade', 'u_focusScale', 'u_dimAlpha', 'u_pointAlpha', 'u_hot'].map((n) => [n, gl.getUniformLocation(prog, n)]));
    // image sprites (off until an atlas is loaded and showImages is set)
    const sp = gl.createProgram();
    gl.attachShader(sp, compile(gl, gl.VERTEX_SHADER, SVS));
    gl.attachShader(sp, compile(gl, gl.FRAGMENT_SHADER, SFS));
    gl.linkProgram(sp);
    if (!gl.getProgramParameter(sp, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(sp));
    this.sp = sp;
    this.su = Object.fromEntries(['u_mvp', 'u_t', 'u_px', 'u_maxPt', 'u_focusImage', 'u_cols', 'u_atlas', 'u_grid'].map((n) => [n, gl.getUniformLocation(sp, n)]));
    this.sbuf = { c0: gl.createBuffer(), c1: gl.createBuffer(), img: gl.createBuffer() };
    this.svao = gl.createVertexArray();
    gl.bindVertexArray(this.svao);
    for (const [name, buf, size] of [['a_c0', this.sbuf.c0, 3], ['a_c1', this.sbuf.c1, 3], ['a_img', this.sbuf.img, 1]]) {
      const loc = gl.getAttribLocation(sp, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
    this.maxPoint = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.atlas = null;          // {tex, cols, rows, n}
    this.showImages = false;
    this.showPoints = true;     // the latent token points
    this.imageSize = 0.14;      // sprite edge in world units (the cloud has unit RMS radius)
    this.c0 = null; this.c1 = null;
    this.onPick = null;         // click on the cloud: fn(image index, or -1 for empty space)
    this.hoverToken = -1;       // token of the focused image to mark in the cloud (-1 = none)
    this.onHoverToken = null;   // mouse over a token point of the focused image: fn(token index, or -1)
    this.fade = 0.2; this.radius = 1.8; // cloud is normalised to unit RMS radius; ~1.8 spans the bulk
    this.focusScale = 4;   // focused image's tokens: area multiplier
    this.dimAlpha = 0.2;   // everything else while focusing: alpha multiplier
    this.a = Object.fromEntries(['a_p0', 'a_p1', 'a_norm', 'a_index', 'a_chan'].map((n) => [n, gl.getAttribLocation(prog, n)]));
    this.vao = gl.createVertexArray();
    this.buf = { p0: gl.createBuffer(), p1: gl.createBuffer(), norm: gl.createBuffer(), index: gl.createBuffer(), chan: gl.createBuffer() };
    gl.bindVertexArray(this.vao);
    const bind = (name, buf, size) => { gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(this.a[name]); gl.vertexAttribPointer(this.a[name], size, gl.FLOAT, false, 0, 0); };
    bind('a_p0', this.buf.p0, 3); bind('a_p1', this.buf.p1, 3); bind('a_norm', this.buf.norm, 1); bind('a_index', this.buf.index, 1); bind('a_chan', this.buf.chan, 1);
    gl.bindVertexArray(null);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    // axes: 3 lines
    this.axisProg = null;

    this.count = 0; this.tokens = 1; this.n = 1; this.normMax = 1;
    this.p0 = null; this.p1 = null;
    this.mode = 0; this.size = 2.2; this.focusImage = -1;
    this.morph = { t: 1, start: 0, dur: 900 };
    this.morphDur = 900;   // ms per snapshot morph; the play loop sets it to its interval

    // camera
    this.yaw = spring.create(0.6); this.pitch = spring.create(0.35); this.dist = spring.create(3.2);
    this.yawT = 0.6; this.pitchT = 0.35; this.distT = 3.2;
    this.autoRotate = true; this.idleSince = performance.now();
    this.q = quat.create(); this.qy = quat.create(); this.qx = quat.create();
    this.eye = vec3.create(); this.center = vec3.fromValues(0, 0, 0); this.up = vec3.fromValues(0, 1, 0);
    this.view = mat4.create(); this.proj = mat4.create(); this.mvp = mat4.create();
    this.explained = [];
    this.onFrame = null;

    this._bindInput();
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas);
    this.resize();
    this._last = performance.now();
    this._raf = requestAnimationFrame((t) => this.frame(t));
  }

  destroy() { cancelAnimationFrame(this._raf); this.ro.disconnect(); }

  /** Image atlas for the sprites: an ImageBitmap/HTMLImageElement holding n images in a cols-wide grid. */
  setImageAtlas(image, n, cols) {
    const gl = this.gl;
    const tex = this.atlas?.tex || gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.atlas = { tex, n, cols, rows: Math.ceil(n / cols) };
  }

  /** Per-image mean of the token positions: (rows × 3) image-major -> (n × 3). */
  _centroids(pts) {
    const n = this.n, T = this.tokens, out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      let x = 0, y = 0, z = 0;
      for (let t = 0; t < T; t++) { const o = 3 * (i * T + t); x += pts[o]; y += pts[o + 1]; z += pts[o + 2]; }
      out[3 * i] = x / T; out[3 * i + 1] = y / T; out[3 * i + 2] = z / T;
    }
    return out;
  }

  /** Image whose token centre is nearest the given client point, within maxPx; else -1. */
  pickImage(clientX, clientY, maxPx = 22) {
    if (!this.c1) return -1;
    const r = this.canvas.getBoundingClientRect();
    const t = softEase(this.morph.t), m = this.mvp;
    let best = -1, bd = maxPx * maxPx;
    for (let i = 0; i < this.n; i++) {
      const x = this.c0[3 * i] + (this.c1[3 * i] - this.c0[3 * i]) * t;
      const y = this.c0[3 * i + 1] + (this.c1[3 * i + 1] - this.c0[3 * i + 1]) * t;
      const z = this.c0[3 * i + 2] + (this.c1[3 * i + 2] - this.c0[3 * i + 2]) * t;
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0.05) continue;
      const sx = r.left + ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * r.width;
      const sy = r.top + (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * r.height;
      const d = (sx - clientX) ** 2 + (sy - clientY) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  /** Token of the focused image whose point is nearest the given client point, within maxPx; else -1. */
  pickToken(clientX, clientY, maxPx = 12) {
    const f = this.focusImage, T = this.tokens;
    if (!this.p1 || !this.showPoints || f < 0 || f >= this.n) return -1;
    const r = this.canvas.getBoundingClientRect();
    const t = softEase(this.morph.t), m = this.mvp, a = this.p0, b = this.p1;
    let best = -1, bd = maxPx * maxPx;
    for (let k = 0; k < T; k++) {
      const o = 3 * (f * T + k);
      const x = a[o] + (b[o] - a[o]) * t, y = a[o + 1] + (b[o + 1] - a[o + 1]) * t, z = a[o + 2] + (b[o + 2] - a[o + 2]) * t;
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0.05) continue;
      const sx = r.left + ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * r.width;
      const sy = r.top + (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * r.height;
      const d = (sx - clientX) ** 2 + (sy - clientY) ** 2;
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.dpr = dpr;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  /** Touch input is off by default so an inline canvas never traps page
   *  scrolling on phones; the mobile fullscreen view switches it on. Mouse
   *  and pen always work. */
  setTouch(on) { this.touchOn = on; }

  resetCamera() { this.yawT = 0.6; this.pitchT = 0.35; this.distT = 3.2; this.autoRotate = true; }

  _bindInput() {
    const c = this.canvas;
    const pts = new Map();          // active pointers: id -> {x, y}
    let pinch = null;               // {d, dist} at pinch start
    let lastTap = 0;
    const accepts = (e) => e.pointerType !== 'touch' || this.touchOn;
    const spread = () => { const [a, b] = [...pts.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };
    let down = null;                // first pointer of a gesture: a click if it barely moved
    c.addEventListener('pointerdown', (e) => {
      if (!accepts(e)) return;
      if (pts.size === 0) down = { x: e.clientX, y: e.clientY, moved: false };
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY }); c.setPointerCapture(e.pointerId); this.autoRotate = false;
      if (pts.size === 2) pinch = { d: spread(), dist: this.distT };
      if (e.pointerType === 'touch' && pts.size === 1) {        // double-tap resets, like dblclick
        const now = performance.now();
        if (now - lastTap < 300) this.resetCamera();
        lastTap = now;
      }
    });
    // mouse hover (no button down) over a token point of the focused image reports that token
    let hov = -1;
    const hover = (k) => { if (k !== hov) { hov = k; if (this.onHoverToken) this.onHoverToken(k); } };
    c.addEventListener('pointerleave', () => hover(-1));
    c.addEventListener('pointermove', (e) => {
      if (pts.size === 0 && e.pointerType !== 'touch') hover(this.pickToken(e.clientX, e.clientY));
      const p = pts.get(e.pointerId); if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY; this.idleSince = performance.now();
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) down.moved = true;
      if (pts.size >= 2 && pinch) {                            // pinch: spread apart = zoom in
        this.distT = clamp(pinch.dist * pinch.d / Math.max(spread(), 1), 0.6, 12);
      } else if (pts.size === 1) {                             // one finger / mouse: orbit
        this.yawT -= dx * 0.006; this.pitchT = clamp(this.pitchT - dy * 0.006, -1.4, 1.4);   // drag up tilts the top away
      }
    });
    const end = (e) => {
      const single = pts.size === 1;
      pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
      // a click (not a drag or pinch) picks the image whose tokens centre nearest the pointer
      if (e.type === 'pointerup' && single && down && !down.moved && this.onPick) this.onPick(this.pickImage(e.clientX, e.clientY));
      if (pts.size === 0) down = null;
    };
    c.addEventListener('pointerup', end); c.addEventListener('pointercancel', end);
    c.addEventListener('wheel', (e) => { e.preventDefault(); this.distT = clamp(this.distT * Math.exp(e.deltaY * 0.0012), 0.6, 12); this.idleSince = performance.now(); }, { passive: false });
    c.addEventListener('dblclick', () => this.resetCamera());
  }

  /** Load a snapshot: xyz Float32Array (rows*3), norms Float32Array (rows), header. */
  setSnapshot(xyz, norms, header, chans = null) {
    const rows = norms.length;
    // normalise to unit RMS radius so snapshots at wildly different z scales are comparable
    let s = 0; for (let i = 0; i < xyz.length; i++) s += xyz[i] * xyz[i];
    const rms = Math.sqrt(s / rows) || 1;
    const pts = new Float32Array(xyz.length);
    for (let i = 0; i < xyz.length; i++) pts[i] = xyz[i] / rms;
    this.rms = rms;
    let nm = 0; for (let i = 0; i < rows; i++) if (norms[i] > nm) nm = norms[i];
    // sort-free: keep server order (image-major, token-minor)
    const gl = this.gl;
    const same = this.p1 && this.p1.length === pts.length;
    // PCA axes are only defined up to sign (and near-equal PCs can swap order), so a new
    // snapshot can come back mirrored. Points correspond across snapshots (same eval images
    // and tokens, same order), so re-orient it to best match the previous one.
    if (same) alignAxes(pts, this.p1);
    const from = same ? this._currentPositions() : pts;
    this.p0 = from; this.p1 = pts;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.p0); gl.bufferData(gl.ARRAY_BUFFER, this.p0, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.p1); gl.bufferData(gl.ARRAY_BUFFER, this.p1, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.norm); gl.bufferData(gl.ARRAY_BUFFER, norms, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.chan); gl.bufferData(gl.ARRAY_BUFFER, chans || new Float32Array(rows), gl.DYNAMIC_DRAW);
    this.channels = header.channels || 1;
    if (this.count !== rows) {
      const idx = new Float32Array(rows); for (let i = 0; i < rows; i++) idx[i] = i;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.index); gl.bufferData(gl.ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    }
    this.count = rows; this.tokens = header.tokens; this.n = header.n; this.normMax = nm || 1;
    this.explained = header.explained || [];
    // sprite positions: token centre of each image, morphing with the cloud
    this.c0 = this._centroids(from); this.c1 = this._centroids(pts);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sbuf.c0); gl.bufferData(gl.ARRAY_BUFFER, this.c0, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sbuf.c1); gl.bufferData(gl.ARRAY_BUFFER, this.c1, gl.DYNAMIC_DRAW);
    const ids = new Float32Array(this.n); for (let i = 0; i < this.n; i++) ids[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sbuf.img); gl.bufferData(gl.ARRAY_BUFFER, ids, gl.STATIC_DRAW);
    this.morph = { t: 0, start: performance.now(), dur: this.p0 === pts ? 1 : this.morphDur };
  }

  _currentPositions() {
    const t = softEase(this.morph.t);
    const out = new Float32Array(this.p1.length);
    for (let i = 0; i < out.length; i++) out[i] = this.p0[i] + (this.p1[i] - this.p0[i]) * t;
    return out;
  }

  frame(now) {
    this._raf = requestAnimationFrame((t) => this.frame(t));
    if (document.hidden || !this.canvas.isConnected) return;
    const dt = Math.min(0.1, (now - this._last) / 1000); this._last = now;
    if (this.autoRotate || now - this.idleSince > 8000) this.yawT += dt * 0.12;
    spring.damp(this.yaw, this.yawT, 0.25, dt);
    spring.damp(this.pitch, this.pitchT, 0.25, dt);
    spring.damp(this.dist, this.distT, 0.3, dt);
    this.morph.t = clamp((now - this.morph.start) / this.morph.dur, 0, 1);

    // camera: eye = rotate (0,0,dist) by yaw about Y then pitch about X
    quat.setAxisAngle(this.qy, this.up, this.yaw.value);
    quat.setAxisAngle(this.qx, vec3.fromValues(1, 0, 0), this.pitch.value);
    quat.multiply(this.q, this.qy, this.qx);
    vec3.set(this.eye, 0, 0, this.dist.value);
    vec3.transformQuat(this.eye, this.eye, this.q);
    mat4.lookAt(this.view, this.eye, this.center, this.up);
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    mat4.perspectiveNO(this.proj, 0.9, aspect, 0.05, 100);
    mat4.multiply(this.mvp, this.proj, this.view);

    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count) return;
    const sprites = this.showImages && this.atlas && this.atlas.n >= this.n;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.u.u_mvp, false, this.mvp);
    gl.uniform1f(this.u.u_t, softEase(this.morph.t));
    gl.uniform1f(this.u.u_size, this.size);
    gl.uniform1f(this.u.u_mode, this.mode);
    gl.uniform1f(this.u.u_tokens, this.tokens);
    gl.uniform1f(this.u.u_n, this.n);
    gl.uniform1f(this.u.u_channels, this.channels || 1);
    gl.uniform1f(this.u.u_normMax, this.normMax);
    gl.uniform1f(this.u.u_dpr, this.dpr || 1);
    gl.uniform1f(this.u.u_camDist, this.dist.value); gl.uniform1f(this.u.u_radius, this.radius); gl.uniform1f(this.u.u_fade, this.fade);
    gl.uniform1f(this.u.u_focusScale, this.focusScale); gl.uniform1f(this.u.u_dimAlpha, this.dimAlpha);
    gl.uniform1f(this.u.u_focusImage, this.focusImage);
    gl.uniform1f(this.u.u_pointAlpha, 1.0);
    gl.uniform1f(this.u.u_hot, 0.0);
    if (this.showPoints) gl.drawArrays(gl.POINTS, 0, this.count);   // points and images toggle separately
    gl.bindVertexArray(null);
    if (sprites) {
      gl.useProgram(this.sp);
      gl.bindVertexArray(this.svao);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.atlas.tex);
      gl.uniform1i(this.su.u_atlas, 0);
      gl.uniformMatrix4fv(this.su.u_mvp, false, this.mvp);
      gl.uniform1f(this.su.u_t, softEase(this.morph.t));
      // world size -> pixels: canvas height / (2 tan(fov/2)), fov 0.9 rad
      gl.uniform1f(this.su.u_px, this.imageSize * this.canvas.height / (2 * Math.tan(0.45)));
      gl.uniform1f(this.su.u_maxPt, this.maxPoint);
      gl.uniform1f(this.su.u_focusImage, this.focusImage);
      gl.uniform1f(this.su.u_cols, this.atlas.cols);
      gl.uniform2f(this.su.u_grid, this.atlas.cols, this.atlas.rows);
      gl.drawArrays(gl.POINTS, 0, this.n);
      gl.disable(gl.DEPTH_TEST);
      gl.bindVertexArray(null);
    }
    const ht = this.hoverToken, fi = this.focusImage;
    if (this.showPoints && fi >= 0 && fi < this.n && ht >= 0 && ht < this.tokens) {   // hovered token: one point, on top of points and sprites
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.uniform1f(this.u.u_hot, 1.0);
      gl.drawArrays(gl.POINTS, fi * this.tokens + ht, 1);
      gl.bindVertexArray(null);
    }
    if (this.onFrame) this.onFrame(this);
  }
}

/** Fetch the PCA projection of a snapshot onto PCs from+1..from+3 (0-based from): { xyz, norms, header }. */
export async function fetchPca(runId, step, from = 0) {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/pca/${step}?v=2&pc=${from + 1}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  const buf = await r.arrayBuffer();
  const jsonLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  const off = 4 + Math.ceil(jsonLen / 4) * 4;
  const rows = header.n * header.tokens;
  const hasChans = buf.byteLength >= off + rows * 20;
  return { xyz: new Float32Array(buf, off, rows * 3), norms: new Float32Array(buf, off + rows * 12, rows), chans: hasChans ? new Float32Array(buf, off + rows * 16, rows) : null, header };
}

export { ramp, ACCENT, ACCENT2 };
