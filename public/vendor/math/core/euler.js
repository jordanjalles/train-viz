import { fromEuler, create as create$1 } from './quat.js';
import { EPSILON, clamp } from './scalar.js';

const DEG2RAD = Math.PI / 180;
/**
 * Creates a new Euler with default values (0, 0, 0, 'xyz').
 */
function create() {
    return [0, 0, 0, 'xyz'];
}
/**
 * Creates a new Euler from the given values.
 * @param x The x rotation in radians.
 * @param y The y rotation in radians.
 * @param z The z rotation in radians.
 * @param order The order of rotation.
 * @returns A new Euler.
 */
function fromValues(x, y, z, order) {
    return [x, y, z, order];
}
/**
 * Sets a given Euler from the given values.
 * @param x The x rotation in radians.
 * @param y The y rotation in radians.
 * @param z The z rotation in radians.
 * @param order The order of rotation.
 * @returns The output Euler.
 */
function set(out, x, y, z, order) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = order;
    return out;
}
/**
 * Sets Euler angle radians from given degrees
 * @param out The output Euler.
 * @param x The x rotation in degrees.
 * @param y The y rotation in degrees.
 * @param z The z rotation in degrees.
 * @param order The order of rotation.
 * @returns The output Euler.
 */
function fromDegrees(out, x, y, z, order) {
    out[0] = x * DEG2RAD;
    out[1] = y * DEG2RAD;
    out[2] = z * DEG2RAD;
    out[3] = order;
    return out;
}
/**
 * Sets the Euler angles from a rotation matrix.
 * @param out The output Euler.
 * @param rotationMatrix The input rotation matrix.
 * @param order The order of the Euler angles.
 * @returns The output Euler.
 */
function fromRotationMat4(out, rotationMatrix, order = out[3] || 'xyz') {
    return fromRotationMatrixValues(out, rotationMatrix[0], rotationMatrix[4], rotationMatrix[8], rotationMatrix[1], rotationMatrix[5], rotationMatrix[9], rotationMatrix[2], rotationMatrix[6], rotationMatrix[10], order);
}
/**
 * Sets the Euler angles from the elements of a rotation matrix (column-major, so `mNM` is row N, column M).
 * Shared by {@link fromRotationMat4} and {@link fromQuat} so neither has to allocate or read back a scratch matrix.
 */
function fromRotationMatrixValues(out, m11, m12, m13, m21, m22, m23, m31, m32, m33, order) {
    switch (order) {
        case 'xyz':
            out[1] = Math.asin(clamp(m13, -1, 1));
            if (Math.abs(m13) < 0.9999999) {
                out[0] = Math.atan2(-m23, m33);
                out[2] = Math.atan2(-m12, m11);
            }
            else {
                out[0] = Math.atan2(m32, m22);
                out[2] = 0;
            }
            break;
        case 'yxz':
            out[0] = Math.asin(-clamp(m23, -1, 1));
            if (Math.abs(m23) < 0.9999999) {
                out[1] = Math.atan2(m13, m33);
                out[2] = Math.atan2(m21, m22);
            }
            else {
                out[1] = Math.atan2(-m31, m11);
                out[2] = 0;
            }
            break;
        case 'zxy':
            out[0] = Math.asin(clamp(m32, -1, 1));
            if (Math.abs(m32) < 0.9999999) {
                out[1] = Math.atan2(-m31, m33);
                out[2] = Math.atan2(-m12, m22);
            }
            else {
                out[1] = 0;
                out[2] = Math.atan2(m21, m11);
            }
            break;
        case 'zyx':
            out[1] = Math.asin(-clamp(m31, -1, 1));
            if (Math.abs(m31) < 0.9999999) {
                out[0] = Math.atan2(m32, m33);
                out[2] = Math.atan2(m21, m11);
            }
            else {
                out[0] = 0;
                out[2] = Math.atan2(-m12, m22);
            }
            break;
        case 'yzx':
            out[2] = Math.asin(clamp(m21, -1, 1));
            if (Math.abs(m21) < 0.9999999) {
                out[0] = Math.atan2(-m23, m22);
                out[1] = Math.atan2(-m31, m11);
            }
            else {
                out[0] = 0;
                out[1] = Math.atan2(m13, m33);
            }
            break;
        case 'xzy':
            out[2] = Math.asin(-clamp(m12, -1, 1));
            if (Math.abs(m12) < 0.9999999) {
                out[0] = Math.atan2(m32, m22);
                out[1] = Math.atan2(m13, m11);
            }
            else {
                out[0] = Math.atan2(-m23, m33);
                out[1] = 0;
            }
            break;
        default:
            console.warn(`encountered an unknown order: ${order}`);
    }
    out[3] = order;
    return out;
}
/**
 * Returns whether or not the euler angles have exactly the same elements in the same position (when compared with ===)
 *
 * @param a The first euler.
 * @param b The second euler.
 * @returns True if the euler angles are equal, false otherwise.
 */
function exactEquals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}
/**
 * Returns whether or not the euler angles have approximately the same elements in the same position.
 *
 * @param a The first euler.
 * @param b The second euler.
 * @returns True if the euler angles are equal, false otherwise.
 */
function equals(a, b) {
    const a0 = a[0];
    const a1 = a[1];
    const a2 = a[2];
    const b0 = b[0];
    const b1 = b[1];
    const b2 = b[2];
    return (Math.abs(a0 - b0) <= EPSILON * Math.max(1.0, Math.abs(a0), Math.abs(b0)) &&
        Math.abs(a1 - b1) <= EPSILON * Math.max(1.0, Math.abs(a1), Math.abs(b1)) &&
        Math.abs(a2 - b2) <= EPSILON * Math.max(1.0, Math.abs(a2), Math.abs(b2)) &&
        a[3] === b[3]);
}
/**
 * Sets the Euler angles from a quaternion.
 * @param out The output Euler.
 * @param q The input quaternion. Assumed to be normalized (unit length).
 * @param order The order of the Euler.
 * @returns The output Euler
 */
function fromQuat(out, q, order) {
    // compute the rotation matrix elements directly from the quaternion
    const x = q[0];
    const y = q[1];
    const z = q[2];
    const w = q[3];
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const yx = y * x2;
    const yy = y * y2;
    const zx = z * x2;
    const zy = z * y2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    return fromRotationMatrixValues(out, 1 - yy - zz, yx - wz, zx + wy, yx + wz, 1 - xx - zz, zy - wx, zx - wy, zy + wx, 1 - xx - yy, order);
}
const _reorderQuaternion = /*@__PURE__*/ create$1();
/**
 * Reorders the Euler based on the specified order.
 * @param out The output Euler.
 * @param a The input Euler.
 * @param order The order of the Euler.
 * @returns The output Euler.
 */
function reorder(out, a, order) {
    fromEuler(_reorderQuaternion, a);
    fromQuat(out, _reorderQuaternion, order);
    return out;
}

export { create, equals, exactEquals, fromDegrees, fromQuat, fromRotationMat4, fromValues, reorder, set };
//# sourceMappingURL=euler.js.map
