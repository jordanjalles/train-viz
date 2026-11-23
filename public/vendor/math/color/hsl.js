import { linearToSrgb, srgbToLinear } from './colorspace.js';

/** Create a new HSL initialized to [0, 0, 0] (black). */
function create() {
    return [0, 0, 0];
}
/** Create a new HSL with the given h, s, l values (all in [0, 1]). */
function fromValues(h, s, l) {
    return [h, s, l];
}
/** Create a new HSL that is a copy of `a`. */
function clone(a) {
    return [a[0], a[1], a[2]];
}
/** Copy the values from `src` into `out`. Returns `out`. */
function copy(out, src) {
    out[0] = src[0];
    out[1] = src[1];
    out[2] = src[2];
    return out;
}
/** Set the h, s, l components of `out` directly. Returns `out`. */
function set(out, h, s, l) {
    out[0] = h;
    out[1] = s;
    out[2] = l;
    return out;
}
/** Write the HSL of a linear Color into `out`. Returns `out`. */
function fromColor(out, c) {
    // linear -> sRGB gamma; HSL is defined on gamma-encoded sRGB
    const r = linearToSrgb(c[0]);
    const g = linearToSrgb(c[1]);
    const b = linearToSrgb(c[2]);
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    let h = 0;
    let s = 0;
    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        if (max === r)
            h = (g - b) / d + (g < b ? 6 : 0);
        else if (max === g)
            h = (b - r) / d + 2;
        else
            h = (r - g) / d + 4;
        h /= 6;
    }
    out[0] = h;
    out[1] = s;
    out[2] = l;
    return out;
}
/** Write the linear Color of an HSL into `out`. Returns `out`. */
function toColor(out, a) {
    const h = a[0];
    const s = a[1];
    const l = a[2];
    if (s === 0) {
        const v = srgbToLinear(l);
        out[0] = v;
        out[1] = v;
        out[2] = v;
        return out;
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    out[0] = srgbToLinear(hue(p, q, h + 1 / 3));
    out[1] = srgbToLinear(hue(p, q, h));
    out[2] = srgbToLinear(hue(p, q, h - 1 / 3));
    return out;
}
/**
 * Interpolate from `a` to `b` by `t` into `out`, taking the shortest path around
 * the hue wheel (so e.g. 350°→10° passes through 0°, not all the way back).
 * Returns `out`.
 */
function lerp(out, a, b, t) {
    let dh = b[0] - a[0];
    if (dh > 0.5)
        dh -= 1;
    else if (dh < -0.5)
        dh += 1;
    let h = a[0] + dh * t;
    h -= Math.floor(h); // wrap into [0, 1)
    out[0] = h;
    out[1] = a[1] + (b[1] - a[1]) * t;
    out[2] = a[2] + (b[2] - a[2]) * t;
    return out;
}
/**
 * Offset `a` by (dh, ds, dl) into `out`: hue wraps into [0, 1), saturation and
 * lightness are clamped to [0, 1]. Returns `out`.
 */
function offset(out, a, dh, ds, dl) {
    let h = a[0] + dh;
    h -= Math.floor(h);
    out[0] = h;
    out[1] = clamp01(a[1] + ds);
    out[2] = clamp01(a[2] + dl);
    return out;
}
function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
function hue(p, q, t) {
    if (t < 0)
        t += 1;
    if (t > 1)
        t -= 1;
    if (t < 1 / 6)
        return p + (q - p) * 6 * t;
    if (t < 1 / 2)
        return q;
    if (t < 2 / 3)
        return p + (q - p) * (2 / 3 - t) * 6;
    return p;
}

export { clone, copy, create, fromColor, fromValues, lerp, offset, set, toColor };
//# sourceMappingURL=hsl.js.map
