import { deltaAngle } from '../core/angle.js';
import { coef, coefficients } from './spring-core.js';

// A scalar spring (damped harmonic oscillator), integrated with the exact
// analytic solution — unconditionally stable at any frame delta, and
// behaviourally equivalent to Unity SmoothDamp / OG maath `damp` for the
// critical case. Two dials:
//   smoothTime    — roughly how long to reach the target; omega = 2 / smoothTime
//   dampingRatio  — the character: 1 = critical (no overshoot), <1 bouncy, >1 sluggish
// `damp` is `update` pinned to dampingRatio = 1.
//
// State lives in a `Spring<number>` the caller allocates once and passes back
// each frame; `update`/`damp` mutate `value` and `velocity` in place.
/** Creates a scalar spring at `value`, at rest. */
const create = (value = 0) => ({ value, velocity: 0 });
/**
 * Springs `state.value` toward `target`, mutating `state` in place. Returns it.
 *
 * @param state spring state, mutated in place
 * @param target goal value
 * @param smoothTime approximate time to reach the target; smaller is faster
 * @param dampingRatio 1 = critically damped (no overshoot), <1 bouncy, >1 sluggish
 * @param delta frame delta, for refresh-rate independence
 * @returns state
 */
function update(state, target, smoothTime, dampingRatio, delta) {
    coefficients(smoothTime, dampingRatio, delta);
    const d = state.value - target;
    const v = state.velocity;
    state.value = target + coef.pp * d + coef.pv * v;
    state.velocity = coef.vp * d + coef.vv * v;
    return state;
}
/**
 * Critically-damped {@link update} (dampingRatio = 1): moves toward `target` as
 * fast as possible without overshooting.
 */
function damp(state, target, smoothTime, delta) {
    return update(state, target, smoothTime, 1, delta);
}
/**
 * Like {@link damp}, but takes the shortest angular path to `target` (radians),
 * wrapping across the ±π seam rather than unwinding the long way around.
 */
function dampAngle(state, target, smoothTime, delta) {
    return damp(state, state.value + deltaAngle(state.value, target), smoothTime, delta);
}
/**
 * Converts a SwiftUI-style `response` — the spring's natural period, in seconds
 * — to the `smoothTime` that `update`/`damp` consume. `dampingRatio` is the
 * orthogonal second dial and is passed to `update` unchanged.
 *
 * Yields the same stiffness/damping as SwiftUI's `Spring(response:dampingFraction:)`:
 *
 * ```ts
 * spring3.update(state, target, spring.fromResponse(0.5), 0.3, delta); // bouncy, ~0.5s period
 * ```
 */
const fromResponse = (response) => response / Math.PI;

export { create, damp, dampAngle, fromResponse, update };
//# sourceMappingURL=spring.js.map
