import { srgbToLinear, linearToSrgb } from './colorspace.js';
export { fromColorInput, setFromColorInput } from './parse.js';

/** Create a new Color initialized to black [0, 0, 0]. */
function create() {
    return [0, 0, 0];
}
/** Create a new Color with the given linear r, g, b values. */
function fromValues(r, g, b) {
    return [r, g, b];
}
/** Create a new Color that is a copy of `c`. */
function clone(c) {
    return [c[0], c[1], c[2]];
}
/** Copy the values from `src` into `out`. Returns `out`. */
function copy(out, src) {
    out[0] = src[0];
    out[1] = src[1];
    out[2] = src[2];
    return out;
}
/** Set the linear r, g, b components of `out` directly. Returns `out`. */
function set(out, r, g, b) {
    out[0] = r;
    out[1] = g;
    out[2] = b;
    return out;
}
/** Set all three channels of `out` to the same linear value `s` (a gray). Returns `out`. */
function setScalar(out, s) {
    out[0] = s;
    out[1] = s;
    out[2] = s;
    return out;
}
/**
 * Set `out` from an sRGB gamma-encoded [r, g, b] array with values in [0, 1].
 * Converts from sRGB gamma space to linear. Returns `out`.
 */
function setFromSRGB(out, srgb) {
    out[0] = srgbToLinear(srgb[0]);
    out[1] = srgbToLinear(srgb[1]);
    out[2] = srgbToLinear(srgb[2]);
    return out;
}
/** Create a new Color from an sRGB gamma-encoded [r, g, b] array with values in [0, 1]. */
function fromSRGB(srgb) {
    return setFromSRGB(create(), srgb);
}
/** Write the sRGB gamma-encoded [r, g, b] of a linear Color into `out` (values [0, 1]). */
function toSRGB(out, c) {
    out[0] = linearToSrgb(c[0]);
    out[1] = linearToSrgb(c[1]);
    out[2] = linearToSrgb(c[2]);
    return out;
}
/** Create a CSS `rgb(...)` string in sRGB gamma space (for HTML/canvas use). */
function toCSS(c) {
    return `rgb(${to255(c[0])}, ${to255(c[1])}, ${to255(c[2])})`;
}
/** Convert to a 0xRRGGBB integer in sRGB gamma space. */
function toHex(c) {
    return (to255(c[0]) << 16) | (to255(c[1]) << 8) | to255(c[2]);
}
/** Convert to a 6-digit sRGB hex string without a leading '#', e.g. 'ff8800'. */
function toHexString(c) {
    return toHex(c).toString(16).padStart(6, '0');
}
/** Add `a + b` component-wise into `out`. Returns `out`. */
function add(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
}
/** Add scalar `s` to each channel of `a` into `out`. Returns `out`. */
function addScalar(out, a, s) {
    out[0] = a[0] + s;
    out[1] = a[1] + s;
    out[2] = a[2] + s;
    return out;
}
/** Subtract `a - b` component-wise into `out`. Returns `out`. */
function sub(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
}
/** Multiply `a * b` component-wise into `out` (tinting). Returns `out`. */
function multiply(out, a, b) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
    return out;
}
/** Scale each channel of `a` by `s` into `out` (brightness). Returns `out`. */
function multiplyScalar(out, a, s) {
    out[0] = a[0] * s;
    out[1] = a[1] * s;
    out[2] = a[2] * s;
    return out;
}
/** Linearly interpolate from `a` to `b` by `t` into `out` (physically-correct blend). Returns `out`. */
function lerp(out, a, b, t) {
    out[0] = a[0] + (b[0] - a[0]) * t;
    out[1] = a[1] + (b[1] - a[1]) * t;
    out[2] = a[2] + (b[2] - a[2]) * t;
    return out;
}
/** Clamp each channel of `c` to [0, 1] into `out`. Returns `out`. */
function clamp(out, c) {
    out[0] = clamp01(c[0]);
    out[1] = clamp01(c[1]);
    out[2] = clamp01(c[2]);
    return out;
}
/** Whether `a` and `b` are equal, within an optional per-channel `epsilon` (default exact). */
function equals(a, b, epsilon = 0) {
    return Math.abs(a[0] - b[0]) <= epsilon && Math.abs(a[1] - b[1]) <= epsilon && Math.abs(a[2] - b[2]) <= epsilon;
}
/** Relative luminance in [0, 1] (Rec. 709 weights, on linear light). */
function luminance(c) {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
/** linear channel -> clamped sRGB byte [0, 255]. */
function to255(c) {
    return Math.max(0, Math.min(255, Math.round(linearToSrgb(c) * 255)));
}

export { add, addScalar, clamp, clone, copy, create, equals, fromSRGB, fromValues, lerp, luminance, multiply, multiplyScalar, set, setFromSRGB, setScalar, sub, toCSS, toHex, toHexString, toSRGB };
//# sourceMappingURL=color.js.map
