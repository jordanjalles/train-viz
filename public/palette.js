// Palette built with math/color: a warm accent, a cool accent, and a
// deterministic hue wheel for run/series colours. Everything returns CSS
// strings, computed once and memoised.

import { color, hsl } from 'math/color';
import { mulberry32 } from 'math/random';

const tmpC = color.create();
const tmpH = hsl.create();

export function css(h, s, l) {
  // math's HSL hue lives in [0, 1); callers here pass degrees.
  hsl.set(tmpH, (((h % 360) + 360) % 360) / 360, s, l);
  hsl.toColor(tmpC, tmpH);
  return color.toCSS(tmpC);
}

export const ACCENT = css(36, 0.86, 0.65);     // amber
export const ACCENT2 = css(192, 0.86, 0.64);   // cyan
export const DANGER = css(4, 0.85, 0.62);
export const OK = css(150, 0.6, 0.55);
export const MUTED = css(220, 0.12, 0.55);

// Series colours inside one chart, in rank order: teal (primary), green
// (secondary), orange (tertiary), then on around the hue wheel in the same
// direction — magenta, violet, blue. Even lightness so no series shouts.
const SERIES_HUES = [192, 150, 28, 320, 270, 225];
const SERIES = SERIES_HUES.map((hh) => css(hh, 0.78, 0.64));
export function seriesColor(i) { return SERIES[i % SERIES.length]; }
/** Ordered ramp for depth/index (e.g. decoder block 0..n): one hue, lightness rising. */
export function depthColor(i, n) { const t = n > 1 ? i / (n - 1) : 0; return css(192, 0.5 + 0.35 * t, 0.36 + 0.42 * t); }
export const NEUTRAL = css(220, 0.12, 0.62);
// Pinned-run identity colours, assigned in pin order (never cyan — that is
// always "the run you are looking at").
export const PIN_COLORS = [css(36, 0.86, 0.65), css(300, 0.7, 0.68), css(150, 0.6, 0.6), css(12, 0.8, 0.66), css(260, 0.7, 0.72), css(80, 0.6, 0.6)];
// Semantic colours for the latent axes / Wasserstein terms and the decoder routing sources.
export const AXIS = { tok: css(300, 0.7, 0.68), chan: css(150, 0.6, 0.6), bwd: css(36, 0.86, 0.65), flat: css(192, 0.86, 0.64) };
export const ROUTE = { z: css(192, 0.86, 0.64), slot: css(36, 0.86, 0.65), patch: css(300, 0.7, 0.68) };

// Run colours: a hash of the run id → hue, so a run keeps its colour across
// reloads and across views, with a golden-angle spread for neighbours.
const runHueCache = new Map();
export function runHue(id) {
  if (runHueCache.has(id)) return runHueCache.get(id);
  let seed = 2166136261;
  for (let i = 0; i < id.length; i++) { seed ^= id.charCodeAt(i); seed = Math.imul(seed, 16777619) >>> 0; }
  const rng = mulberry32.create(seed);
  const h = Math.floor(mulberry32.sample(rng) * 360);
  runHueCache.set(id, h);
  return h;
}
export function runColor(id, l = 0.66, s = 0.78) { return css(runHue(id), s, l); }

/** Sequential ramp (t∈[0,1]) from deep indigo through cyan to amber — for heatmaps. */
export function ramp(t) {
  t = Math.min(1, Math.max(0, t));
  const h = 250 - 210 * t;          // 250 → 40
  const l = 0.28 + 0.45 * t;
  const s = 0.7 + 0.2 * Math.sin(Math.PI * t);
  return css(h, s, l);
}
