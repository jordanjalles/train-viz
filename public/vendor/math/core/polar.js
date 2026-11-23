import { equals as equals$1, lerp as lerp$1 } from './scalar.js';
import { wrapAngle } from './angle.js';

/**
 * Creates a new polar coordinate at r=1, theta=0
 *
 * @returns a new Polar
 */
function create() {
    return [1, 0];
}
/**
 * Creates a new Polar initialized with the given values
 *
 * @param r radial distance
 * @param theta angle from +X, counter-clockwise (radians)
 * @returns a new Polar
 */
function fromValues(r, theta) {
    return [r, theta];
}
/**
 * Creates a new Polar initialized with values from an existing one
 *
 * @param a the source Polar
 * @returns a new Polar
 */
function clone(a) {
    return [a[0], a[1]];
}
/**
 * Copies values from one Polar to another
 *
 * @param out the receiving Polar
 * @param a the source Polar
 * @returns out
 */
function copy(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    return out;
}
/**
 * Sets the components of a Polar
 *
 * @param out the receiving Polar
 * @param r radial distance
 * @param theta angle from +X, counter-clockwise (radians)
 * @returns out
 */
function set(out, r, theta) {
    out[0] = r;
    out[1] = theta;
    return out;
}
/**
 * Sets r=1, preserving the angle. No-op on the angle if r is already zero.
 *
 * @param out the receiving Polar
 * @param a the source Polar
 * @returns out
 */
function normalize(out, a) {
    out[0] = 1;
    out[1] = a[1];
    return out;
}
/**
 * Scales the radial distance r by a scalar
 *
 * @param out the receiving Polar
 * @param a the source Polar
 * @param s scalar to multiply r by
 * @returns out
 */
function scale(out, a, s) {
    out[0] = a[0] * s;
    out[1] = a[1];
    return out;
}
/**
 * Rotates a Polar by an angle (in radians), wrapping theta into (-pi, pi].
 *
 * @param out the receiving Polar
 * @param a the source Polar
 * @param rad the angle to add to theta
 * @returns out
 */
function rotate(out, a, rad) {
    out[0] = a[0];
    out[1] = wrapAngle(a[1] + rad);
    return out;
}
/**
 * Linearly interpolates between two Polar coordinates, taking the shortest
 * angular path for theta.
 *
 * @param out the receiving Polar
 * @param a the first operand
 * @param b the second operand
 * @param t interpolation factor in [0, 1]
 * @returns out
 */
function lerp(out, a, b, t) {
    out[0] = lerp$1(a[0], b[0], t);
    out[1] = a[1] + wrapAngle(b[1] - a[1]) * t;
    return out;
}
/**
 * Sets a Polar from Cartesian Vec2 coordinates:
 *   r     = sqrt(x^2 + y^2)
 *   theta = atan2(y, x)
 *
 * @param out the receiving Polar
 * @param v the source Vec2
 * @returns out
 */
function setFromVec2(out, v) {
    const x = v[0];
    const y = v[1];
    out[0] = Math.sqrt(x * x + y * y);
    out[1] = out[0] === 0 ? 0 : Math.atan2(y, x);
    return out;
}
/** @alias setFromVec2 */
const fromVec2 = setFromVec2;
/**
 * Converts polar coordinates to a Cartesian Vec2:
 *   x = r * cos(theta)
 *   y = r * sin(theta)
 *
 * @param out the receiving Vec2
 * @param a the source Polar
 * @returns out
 */
function toVec2(out, a) {
    const r = a[0];
    const theta = a[1];
    out[0] = r * Math.cos(theta);
    out[1] = r * Math.sin(theta);
    return out;
}
/**
 * Returns the smallest angle (in radians) between two polar directions,
 * ignoring r. Range [0, pi].
 *
 * @param a the first Polar
 * @param b the second Polar
 * @returns angle in radians in [0, pi]
 */
function angleTo(a, b) {
    return Math.abs(wrapAngle(b[1] - a[1]));
}
/**
 * Returns the straight-line (chord) distance between two polar coordinates,
 * via the law of cosines.
 *
 * @param a the first Polar
 * @param b the second Polar
 * @returns the Euclidean distance between the two points
 */
function distance(a, b) {
    const ra = a[0];
    const rb = b[0];
    const d = ra * ra + rb * rb - 2 * ra * rb * Math.cos(b[1] - a[1]);
    return Math.sqrt(Math.max(0, d));
}
/**
 * Returns true if two Polar coordinates are approximately equal,
 * within an absolute/relative tolerance of EPSILON.
 *
 * @param a the first Polar
 * @param b the second Polar
 * @returns true if approximately equal
 */
function equals(a, b) {
    return equals$1(a[0], b[0]) && equals$1(a[1], b[1]);
}
/**
 * Returns true if two Polar coordinates are exactly equal (===).
 *
 * @param a the first Polar
 * @param b the second Polar
 * @returns true if exactly equal
 */
function exactEquals(a, b) {
    return a[0] === b[0] && a[1] === b[1];
}
/**
 * Returns a string representation of a Polar
 *
 * @param a the source Polar
 * @returns string representation
 */
function str(a) {
    return `Polar(${a[0]}, ${a[1]})`;
}

export { angleTo, clone, copy, create, distance, equals, exactEquals, fromValues, fromVec2, lerp, normalize, rotate, scale, set, setFromVec2, str, toVec2 };
//# sourceMappingURL=polar.js.map
