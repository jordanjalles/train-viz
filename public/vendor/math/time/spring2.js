import { coefficients, coef } from './spring-core.js';

// A Vec2 spring. See `spring` (scalar) for the model; the same four coefficients
// are applied to each component. `damp` is `update` pinned to dampingRatio = 1.
/** Creates a Vec2 spring at `value` (copied), at rest. */
const create = (value = [0, 0]) => ({ value: [value[0], value[1]], velocity: [0, 0] });
/**
 * Springs `state.value` toward `target`, mutating `state` in place. Returns it.
 * @param dampingRatio 1 = critically damped (no overshoot), <1 bouncy, >1 sluggish
 */
function update(state, target, smoothTime, dampingRatio, delta) {
    coefficients(smoothTime, dampingRatio, delta);
    const val = state.value;
    const vel = state.velocity;
    let d = val[0] - target[0];
    let v = vel[0];
    val[0] = target[0] + coef.pp * d + coef.pv * v;
    vel[0] = coef.vp * d + coef.vv * v;
    d = val[1] - target[1];
    v = vel[1];
    val[1] = target[1] + coef.pp * d + coef.pv * v;
    vel[1] = coef.vp * d + coef.vv * v;
    return state;
}
/** Critically-damped Vec2 spring (dampingRatio = 1). See {@link update}. */
function damp(state, target, smoothTime, delta) {
    return update(state, target, smoothTime, 1, delta);
}

export { create, damp, update };
//# sourceMappingURL=spring2.js.map
