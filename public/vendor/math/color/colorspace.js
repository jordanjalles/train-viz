// Color-space conversions (pure functions — no global working-space state).
//
// math's Color is stored in linear sRGB. Two kinds of conversion live here:
//   - transfer functions: encode/decode a channel's gamma (sRGB <-> linear)
//   - gamut conversions:   move between primaries (linear sRGB <-> linear Display-P3)
// Display-P3 uses the same sRGB transfer curve, so only the primaries differ.
/** Convert a single sRGB gamma-encoded channel [0, 1] to linear light [0, 1]. */
function srgbToLinear(c) {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
/** Convert a single linear light channel [0, 1] to sRGB gamma-encoded [0, 1]. */
function linearToSrgb(c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
}
/**
 * Convert a linear-sRGB Color to linear Display-P3 primaries, into `out`. Returns `out`.
 * (Both spaces share the sRGB transfer curve; this changes only the primaries.)
 */
function linearSrgbToLinearDisplayP3(out, c) {
    const r = c[0];
    const g = c[1];
    const b = c[2];
    out[0] = 0.8224621 * r + 0.177538 * g;
    out[1] = 0.0331941 * r + 0.9668058 * g;
    out[2] = 0.0170827 * r + 0.0723974 * g + 0.9105199 * b;
    return out;
}
/**
 * Convert a linear Display-P3 Color to linear-sRGB primaries, into `out`. Returns `out`.
 * Colors outside the sRGB gamut yield channels outside [0, 1] — clamp if needed.
 */
function linearDisplayP3ToLinearSrgb(out, c) {
    const r = c[0];
    const g = c[1];
    const b = c[2];
    out[0] = 1.2249401 * r - 0.2249404 * g;
    out[1] = -0.0420569 * r + 1.0420571 * g;
    out[2] = -0.0196376 * r - 0.0786361 * g + 1.0982735 * b;
    return out;
}

export { linearDisplayP3ToLinearSrgb, linearSrgbToLinearDisplayP3, linearToSrgb, srgbToLinear };
//# sourceMappingURL=colorspace.js.map
