const TAU = Math.PI * 2;
const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;
/**
 * Converts Degrees To Radians
 *
 * @param a Angle in Degrees
 */
function degreesToRadians(degrees) {
    return degrees * DEGREES_TO_RADIANS;
}
/**
 * Converts Radians To Degrees
 *
 * @param a Angle in Radians
 */
function radiansToDegrees(radians) {
    return radians * RADIANS_TO_DEGREES;
}
/**
 * Wraps an angle (in radians) into the range (-π, π].
 *
 * @param a the angle to wrap
 * @returns the angle wrapped into (-π, π]
 */
function wrapAngle(a) {
    return a - TAU * Math.floor((a + Math.PI) / TAU);
}
/**
 * Calculates the shortest signed difference between two angles (in radians).
 *
 * @param current the current angle
 * @param target the target angle
 * @returns the shortest signed difference in (-π, π]
 */
function deltaAngle(current, target) {
    const diff = target - current;
    const delta = diff - TAU * Math.floor(diff / TAU);
    return delta > Math.PI ? delta - TAU : delta;
}

export { DEGREES_TO_RADIANS, RADIANS_TO_DEGREES, degreesToRadians, deltaAngle, radiansToDegrees, wrapAngle };
//# sourceMappingURL=angle.js.map
