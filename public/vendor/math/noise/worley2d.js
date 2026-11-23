import { createPermutation } from './permutation.js';

/**
 * Creates a 2D Worley noise generator with the given seed.
 *
 * @param seed The seed value for the noise generator
 * @returns A generator to pass to {@link sample}
 */
function create(seed) {
    return createPermutation(seed);
}
/**
 * Samples 2D Worley (cellular) noise: the Euclidean distance to the nearest of a
 * set of feature points, one scattered per unit cell. Gives organic cell / scale
 * / caustic patterns; values are roughly in [0, 1] (0 right at a feature point,
 * rising between them). Threshold it for cracks, cells, or region masks.
 *
 * @param generator A generator created with {@link create}
 * @param x X coordinate
 * @param y Y coordinate
 * @returns The distance to the nearest feature point (F1)
 */
function sample({ perm }, x, y) {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    let f1 = Infinity; // squared nearest distance
    // check the point's cell and its 8 neighbours; the nearest feature point can
    // only be in one of these. The row (y) hashes are shared by the three cells
    // in a row, so hash them once per row.
    for (let gy = -1; gy <= 1; gy++) {
        const cy = iy + gy;
        const py = perm[cy & 255];
        const qy = perm[(cy + 37) & 255];
        for (let gx = -1; gx <= 1; gx++) {
            const cx = ix + gx;
            // a hashed feature-point offset in [0, 1) per axis, decorrelated
            const rx = perm[(cx & 255) + py] / 256;
            const ry = perm[((cx + 71) & 255) + qy] / 256;
            const dx = cx + rx - x;
            const dy = cy + ry - y;
            const d2 = dx * dx + dy * dy;
            if (d2 < f1)
                f1 = d2;
        }
    }
    return Math.sqrt(f1);
}

export { create, sample };
//# sourceMappingURL=worley2d.js.map
