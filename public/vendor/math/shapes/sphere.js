/**
 * Creates a new sphere with a default center 0,0,0 and radius 1
 * @returns A new sphere.
 */
function create() {
    return { center: [0, 0, 0], radius: 1 };
}
/**
 * Returns true if a point lies inside (or on the surface of) the sphere.
 *
 * @param sphere the sphere
 * @param point the point to test
 * @returns true if the point is within the sphere's radius
 */
function containsPoint(sphere, point) {
    const dx = point[0] - sphere.center[0];
    const dy = point[1] - sphere.center[1];
    const dz = point[2] - sphere.center[2];
    return dx * dx + dy * dy + dz * dz <= sphere.radius * sphere.radius;
}

export { containsPoint, create };
//# sourceMappingURL=sphere.js.map
