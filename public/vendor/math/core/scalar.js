const EPSILON = 0.000001;
/**
 * Symmetric round
 * see https://www.npmjs.com/package/round-half-up-symmetric#user-content-detailed-background
 *
 * @param a value to round
 */
function round(a) {
    if (a >= 0)
        return Math.round(a);
    return a % 0.5 === 0 ? Math.floor(a) : Math.round(a);
}
/**
 * Tests whether or not the arguments have approximately the same value, within an absolute
 * or relative tolerance of glMatrix.EPSILON (an absolute tolerance is used for values less
 * than or equal to 1.0, and a relative tolerance is used for larger values)
 *
 * @param a The first number to test.
 * @param b The second number to test.
 * @returns True if the numbers are approximately equal, false otherwise.
 */
function equals(a, b, epsilon = EPSILON) {
    return Math.abs(a - b) <= epsilon * Math.max(1.0, Math.abs(a), Math.abs(b));
}
/**
 * Ease-in-out, goes to -Infinite before 0 and Infinite after 1
 *
 * https://www.desmos.com/calculator/vsnmlaljdu
 *
 * @param t
 * @returns
 */
function fade(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}
/**
 *
 * Returns the result of linearly interpolating between input A and input B by input T.
 *
 * @param v0
 * @param v1
 * @param t
 * @returns
 */
function lerp(v0, v1, t) {
    return v0 * (1 - t) + v1 * t;
}
/**
 *
 * Quadratic interpolation through three values using Lagrange interpolation.
 * Passes through v0 at t=0, v1 at t=0.5, and v2 at t=1.
 *
 * @param v0 value at t=0
 * @param v1 value at t=0.5
 * @param v2 value at t=1
 * @param t
 * @returns
 */
function lagrange(v0, v1, v2, t) {
    return 2 * (t - 1) * (t - 0.5) * v0 - 4 * (t - 1) * t * v1 + 2 * (t - 0.5) * t * v2;
}
/**
 *
 * Binomial coefficient, "n choose k": the number of ways to choose k items
 * from a set of n without order or repetition.
 *
 * @param n total number of items
 * @param k number of items to choose
 * @returns the binomial coefficient, or 0 when k is out of range
 */
function binomial(n, k) {
    if (k < 0 || k > n)
        return 0;
    // C(n, k) === C(n, n - k), so iterate over the smaller of the two
    const j = Math.min(k, n - k);
    let coeff = 1;
    // interleave multiply/divide to keep intermediates small and integer-exact
    for (let i = 0; i < j; i++) {
        coeff = (coeff * (n - i)) / (i + 1);
    }
    return Math.round(coeff);
}
/**
 * Clamp a value between min and max
 */
const clamp = (value, min, max) => {
    return Math.max(min, Math.min(max, value));
};
/**
 * Loops `t` so that it is never larger than `length` and never smaller than 0.
 */
function repeat(t, length) {
    return clamp(t - Math.floor(t / length) * length, 0, length);
}
/**
 * Remaps a number from one range to another.
 */
function remap(number, inLow, inHigh, outLow, outHigh) {
    const scale = (number - inLow) / (inHigh - inLow);
    return outLow + scale * (outHigh - outLow);
}
/**
 * Remaps a number from one range to another, clamping the result to the output range.
 */
function remapClamp(value, inLow, inHigh, outLow, outHigh) {
    const scale = (value - inLow) / (inHigh - inLow);
    const remapped = outLow + scale * (outHigh - outLow);
    return Math.max(outLow, Math.min(outHigh, remapped));
}

export { EPSILON, binomial, clamp, equals, fade, lagrange, lerp, remap, remapClamp, repeat, round };
//# sourceMappingURL=scalar.js.map
