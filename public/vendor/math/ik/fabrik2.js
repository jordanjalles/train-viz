import { distance, subtract, squaredLength, copy, normalize, squaredDistance, rotate, signedAngle } from '../core/vec2.js';

// FABRIK (Forward And Backward Reaching Inverse Kinematics) for 2D chains.
//
// The 2D counterpart of `fabrik3`, same shape and same two passes. What differs is the constraint
// model: with no axis to hinge about, a joint is a wedge - how far the bone may swing clockwise and
// anticlockwise of a baseline, either the previous bone (`LOCAL`) or a fixed world direction
// (`GLOBAL`, which pins the bone's absolute heading).
//
// The constraint model follows Caliko, the reference implementation accompanying Aristidou &
// Lasenby (2011), "FABRIK: A fast, iterative solver for the Inverse Kinematics problem".
/** What a joint's clockwise and anticlockwise limits are measured from. */
var ConstraintCoordinateSystem;
(function (ConstraintCoordinateSystem) {
    /** The direction of the previous bone, so the limits bound how far this bone may bend. */
    ConstraintCoordinateSystem[ConstraintCoordinateSystem["LOCAL"] = 0] = "LOCAL";
    /** A fixed world direction, so the limits bound this bone's absolute heading. */
    ConstraintCoordinateSystem[ConstraintCoordinateSystem["GLOBAL"] = 1] = "GLOBAL";
})(ConstraintCoordinateSystem || (ConstraintCoordinateSystem = {}));
/**
 * How the first bone in a chain is constrained.
 *
 * The first bone has no bone before it, so it is constrained against a direction held on the chain.
 * The `LOCAL_` types only mean anything for a chain in a {@link Structure2}, where
 * {@link solveStructure} resolves them against the bone the chain hangs off.
 */
var BaseboneConstraintType;
(function (BaseboneConstraintType) {
    /** The first bone may point anywhere. */
    BaseboneConstraintType[BaseboneConstraintType["NONE"] = 0] = "NONE";
    /** Constrained against a fixed world direction. */
    BaseboneConstraintType[BaseboneConstraintType["GLOBAL_ABSOLUTE"] = 1] = "GLOBAL_ABSOLUTE";
    /** Constrained against the host bone's direction, so the chain bends relative to what it hangs off. */
    BaseboneConstraintType[BaseboneConstraintType["LOCAL_RELATIVE"] = 2] = "LOCAL_RELATIVE";
    /**
     * Constrained against a direction expressed in the host bone's frame, where **+Y** is the host
     * bone's own direction. (`fabrik3`'s equivalent uses +Z, so a rig ported between the two needs
     * its axes rotated.)
     */
    BaseboneConstraintType[BaseboneConstraintType["LOCAL_ABSOLUTE"] = 3] = "LOCAL_ABSOLUTE";
})(BaseboneConstraintType || (BaseboneConstraintType = {}));
/** Which end of a host bone a connected chain hangs off. */
var BoneConnectionPoint;
(function (BoneConnectionPoint) {
    /** The host bone's start - the joint it shares with the bone before it. */
    BoneConnectionPoint[BoneConnectionPoint["START"] = 0] = "START";
    /** The host bone's end. */
    BoneConnectionPoint[BoneConnectionPoint["END"] = 1] = "END";
})(BoneConnectionPoint || (BoneConnectionPoint = {}));
// the most iterations a solve runs before giving up on reaching the target
const DEFAULT_MAX_ITERATIONS = 20;
// how close the effector must get before a solve stops early. in world units, so scale it with the
// scene - the default suits a chain a few units long
const DEFAULT_SOLVE_DISTANCE_THRESHOLD = 0.01;
// how little an iteration may improve the solve distance before it counts as stalled. a constrained
// chain can reach a pose no further iteration improves on while still short of the threshold, and
// without this it would burn every remaining iteration going nowhere. two orders of magnitude below
// the distance threshold, so a solve still closing in on the target is never mistaken for a stalled
// one
const DEFAULT_MIN_ITERATION_CHANGE = 1e-4;
// below this squared length a vector carries no usable direction - 1e-12 in length terms, well above
// a normalize's noise floor and well below anything meaningful. every test against it also checks
// for NaN, so a value poisoned upstream takes the same fallback instead of spreading down the chain
const DEGENERATE_SQUARED_LENGTH = 1e-24;
/**
 * Normalizes `axis` into `out`, leaving `out` untouched if `axis` has no direction to give.
 *
 * A zero axis cannot be normalized, and storing one turns every constraint that reads it into NaN.
 * `out` always starts as a valid unit vector, so keeping it is the safe fallback.
 */
function setUnitAxis(out, axis) {
    if (!hasDirection(squaredLength(axis)))
        return out;
    return normalize(out, axis);
}
/**
 * Whether a squared length is a usable magnitude: a real number, and not effectively zero.
 *
 * NaN counts as unusable. A value poisoned upstream then takes the same fallback as a zero-length
 * one, instead of spreading down the chain and leaving it stuck.
 */
function hasDirection(squaredLength) {
    if (Number.isNaN(squaredLength))
        return false;
    return squaredLength >= DEGENERATE_SQUARED_LENGTH;
}
/** Creates an unconstrained joint. */
function createJoint2() {
    return {
        clockwise: Math.PI,
        anticlockwise: Math.PI,
        coordinateSystem: ConstraintCoordinateSystem.LOCAL,
        globalAxis: [0, 1],
    };
}
/** Creates an empty chain with a fixed base at the origin and no basebone constraint. */
function createChain2() {
    return {
        bones: [],
        length: 0,
        base: [0, 0],
        fixedBase: true,
        baseboneConstraintType: BaseboneConstraintType.NONE,
        baseboneAxis: [0, 1],
        baseboneWorldAxis: [0, 1],
        baseboneClockwise: Math.PI,
        baseboneAnticlockwise: Math.PI,
        embeddedTarget: [0, 0],
        useEmbeddedTarget: false,
        maxIterations: DEFAULT_MAX_ITERATIONS,
        solveDistanceThreshold: DEFAULT_SOLVE_DISTANCE_THRESHOLD,
        minIterationChange: DEFAULT_MIN_ITERATION_CHANGE,
        solveDistance: Number.POSITIVE_INFINITY,
        bestSolution: [],
    };
}
/**
 * Appends a bone spanning `start` to `end`, copying both.
 *
 * The bone's length is taken from the distance between them. When this is the first bone the
 * chain's base is moved to `start`.
 *
 * @param chain the chain to append to
 * @param start the bone's start point
 * @param end the bone's end point
 * @param joint the bone's joint, or a fresh unconstrained one if omitted
 * @returns the appended bone
 */
function addBone(chain, start, end, joint = createJoint2()) {
    const bone = {
        start: [start[0], start[1]],
        end: [end[0], end[1]],
        length: distance(start, end),
        joint,
    };
    if (chain.bones.length === 0) {
        chain.base[0] = start[0];
        chain.base[1] = start[1];
    }
    chain.bones.push(bone);
    chain.length += bone.length;
    // four numbers of best-pose scratch per bone, so `solve` never allocates
    chain.bestSolution.push(0, 0, 0, 0);
    return bone;
}
const _addConsecutive_end = [0, 0];
/**
 * Appends a bone starting where the chain currently ends, running `length` along `direction`.
 *
 * @param chain the chain to append to, which must already have at least one bone
 * @param direction the direction to extend in, assumed to be unit length
 * @param length the length of the new bone
 * @param joint the bone's joint, or a fresh unconstrained one if omitted
 * @returns the appended bone
 */
function addConsecutiveBone(chain, direction, length, joint = createJoint2()) {
    const previous = chain.bones[chain.bones.length - 1];
    _addConsecutive_end[0] = previous.end[0] + direction[0] * length;
    _addConsecutive_end[1] = previous.end[1] + direction[1] * length;
    return addBone(chain, previous.end, _addConsecutive_end, joint);
}
/**
 * Prepends a bone at the base end, extending the chain backward.
 *
 * This is what a follower grows with: driven by {@link forward} alone the base end is the tail.
 *
 * `joint` becomes the joint of the bone that was previously first, because that is the junction the
 * new bone creates. The new first bone's own joint is unused, as always.
 *
 * @param chain the chain to prepend to, which must already have at least one bone
 * @param direction the direction the new bone points, from its own start toward the existing chain. Assumed to be unit length
 * @param length the length of the new bone
 * @param joint the joint for the junction this creates, or a fresh unconstrained one if omitted
 * @returns the prepended bone
 */
function addBoneAtBase(chain, direction, length, joint = createJoint2()) {
    const first = chain.bones[0];
    // the bone that was first now sits at index 1, so it is the one whose joint governs the new
    // junction. its old joint was the unused basebone slot
    first.joint = joint;
    const bone = {
        start: [first.start[0] - direction[0] * length, first.start[1] - direction[1] * length],
        end: [first.start[0], first.start[1]],
        length,
        joint: createJoint2(),
    };
    chain.bones.unshift(bone);
    chain.length += length;
    chain.base[0] = bone.start[0];
    chain.base[1] = bone.start[1];
    chain.bestSolution.push(0, 0, 0, 0);
    return bone;
}
/**
 * Sets a joint's limits relative to the previous bone's direction, which is the usual case.
 *
 * @param joint the joint to configure
 * @param clockwise how far the bone may swing clockwise of the previous bone, in radians, clamped to [0, PI]
 * @param anticlockwise how far it may swing anticlockwise, in radians, clamped to [0, PI]
 * @returns the joint
 */
function setLocalJoint(joint, clockwise, anticlockwise) {
    joint.coordinateSystem = ConstraintCoordinateSystem.LOCAL;
    joint.clockwise = clampAngle(clockwise);
    joint.anticlockwise = clampAngle(anticlockwise);
    return joint;
}
/**
 * Sets a joint's limits relative to a fixed world direction, pinning the bone's absolute heading
 * however the bones before it move.
 *
 * @param joint the joint to configure
 * @param axis the world direction the limits are measured from, assumed to be unit length
 * @param clockwise how far the bone may swing clockwise of `axis`, in radians, clamped to [0, PI]
 * @param anticlockwise how far it may swing anticlockwise, in radians, clamped to [0, PI]
 * @returns the joint
 */
function setGlobalJoint(joint, axis, clockwise, anticlockwise) {
    joint.coordinateSystem = ConstraintCoordinateSystem.GLOBAL;
    joint.clockwise = clampAngle(clockwise);
    joint.anticlockwise = clampAngle(anticlockwise);
    setUnitAxis(joint.globalAxis, axis);
    return joint;
}
/**
 * Constrains the first bone to a wedge about `axis`.
 *
 * @param chain the chain to configure
 * @param type which space `axis` is expressed in
 * @param axis the baseline direction, assumed to be unit length. Ignored for {@link BaseboneConstraintType.LOCAL_RELATIVE}, which uses the host bone's direction
 * @param clockwise how far the first bone may swing clockwise of the baseline, in radians, clamped to [0, PI]
 * @param anticlockwise how far it may swing anticlockwise, in radians, clamped to [0, PI]
 * @returns the chain
 */
function setBaseboneConstraint(chain, type, axis, clockwise, anticlockwise) {
    chain.baseboneConstraintType = type;
    chain.baseboneClockwise = clampAngle(clockwise);
    chain.baseboneAnticlockwise = clampAngle(anticlockwise);
    setUnitAxis(chain.baseboneAxis, axis);
    copy(chain.baseboneWorldAxis, chain.baseboneAxis);
    return chain;
}
/**
 * Moves the chain's pinned base, without moving the bones.
 *
 * The next {@link backward} or {@link solve} pulls the chain to it.
 */
function setBaseLocation(chain, base) {
    chain.base[0] = base[0];
    chain.base[1] = base[1];
    return chain;
}
/**
 * Lays the chain out straight from its base along `direction`, discarding the current pose.
 *
 * A dead-straight chain is the worst starting pose for {@link solve} - see the note there. Bend
 * `direction` slightly between bones instead if the chain will be solved cold.
 */
function straighten(chain, direction) {
    const bones = chain.bones;
    let x = chain.base[0];
    let y = chain.base[1];
    for (let i = 0; i < bones.length; i++) {
        const bone = bones[i];
        bone.start[0] = x;
        bone.start[1] = y;
        x += direction[0] * bone.length;
        y += direction[1] * bone.length;
        bone.end[0] = x;
        bone.end[1] = y;
    }
    return chain;
}
/**
 * Writes the end effector's position - the last bone's end - into `out`.
 *
 * A chain with no bones has its base as its effector, matching {@link solve}, which treats an empty
 * chain as a supported state rather than an error.
 */
function getEffector(out, chain) {
    const count = chain.bones.length;
    return copy(out, count === 0 ? chain.base : chain.bones[count - 1].end);
}
/** Writes the unit direction of bone `index`, from its start toward its end, into `out`. */
function getBoneDirection(out, chain, index) {
    const bone = chain.bones[index];
    subtract(out, bone.end, bone.start);
    if (!hasDirection(squaredLength(out))) {
        return copy(out, UP);
    }
    return normalize(out, out);
}
/** The angle of bone `index`, in radians, measured counter-clockwise from the +X axis. */
function getBoneAngle(chain, index) {
    const bone = chain.bones[index];
    return Math.atan2(bone.end[1] - bone.start[1], bone.end[0] - bone.start[0]);
}
/** Whether `target` is within reach of the chain's base, so a solve can place the effector exactly on it. */
function isReachable(chain, target) {
    return squaredDistance(chain.base, target) <= chain.length * chain.length;
}
/**
 * The forward pass: snaps the end effector onto `target` and drags the rest of the chain after it.
 *
 * A non-finite target leaves the chain untouched rather than poisoning it.
 *
 * The base is not pinned, so the whole chain moves. Run alone once per frame this is a follower
 * rather than a solver - a rope or tentacle whose head leads. Pair it with {@link backward}, or use
 * {@link solve}, to keep the base put.
 *
 * @param chain the chain to move, mutated in place
 * @param target where the end effector should go
 * @returns the chain
 */
function forward(chain, target) {
    const bones = chain.bones;
    const count = bones.length;
    if (count === 0)
        return chain;
    // a non-finite target would be written straight into the effector and spread down the chain
    if (!Number.isFinite(target[0]) || !Number.isFinite(target[1]))
        return chain;
    // snap the effector onto the target. the rest of the pass follows from it
    const effector = bones[count - 1];
    effector.end[0] = target[0];
    effector.end[1] = target[1];
    // the outer-to-inner direction of the bone one step further out, which is what a joint on this
    // bone is constrained against on this pass. undefined for the effector itself.
    let hasReference = false;
    for (let i = count - 1; i >= 0; i--) {
        const bone = bones[i];
        const start = bone.start;
        const end = bone.end;
        // this bone's outer-to-inner direction
        let dx = start[0] - end[0];
        let dy = start[1] - end[1];
        const squaredLength = dx * dx + dy * dy;
        if (!hasDirection(squaredLength)) {
            // the two ends coincide, so there is no direction to preserve. carry on along the bone
            // further out, or straight up if this is the effector
            dx = hasReference ? _pass_reference[0] : UP[0];
            dy = hasReference ? _pass_reference[1] : UP[1];
        }
        else {
            const inverseLength = 1 / Math.sqrt(squaredLength);
            dx *= inverseLength;
            dy *= inverseLength;
        }
        constrainForward(chain, i, dx, dy, hasReference);
        dx = _pass_direction[0];
        dy = _pass_direction[1];
        const x = end[0] + dx * bone.length;
        const y = end[1] + dy * bone.length;
        start[0] = x;
        start[1] = y;
        // the previous bone shares this point
        if (i > 0) {
            const previousEnd = bones[i - 1].end;
            previousEnd[0] = x;
            previousEnd[1] = y;
        }
        _pass_reference[0] = dx;
        _pass_reference[1] = dy;
        hasReference = true;
    }
    return chain;
}
/**
 * The backward pass: pins the base and pushes each bone outward from it. The basebone constraint
 * is applied here.
 *
 * @param chain the chain to move, mutated in place
 * @param base where the base should go, used only when `chain.fixedBase` is set
 * @returns the chain
 */
function backward(chain, base) {
    const bones = chain.bones;
    const count = bones.length;
    if (count === 0)
        return chain;
    if (!Number.isFinite(base[0]) || !Number.isFinite(base[1]))
        return chain;
    if (chain.fixedBase) {
        const start = bones[0].start;
        start[0] = base[0];
        start[1] = base[1];
    }
    let hasReference = false;
    for (let i = 0; i < count; i++) {
        const bone = bones[i];
        const start = bone.start;
        const end = bone.end;
        // this bone's inner-to-outer direction
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        const squaredLength = dx * dx + dy * dy;
        if (!hasDirection(squaredLength)) {
            dx = hasReference ? _pass_reference[0] : UP[0];
            dy = hasReference ? _pass_reference[1] : UP[1];
        }
        else {
            const inverseLength = 1 / Math.sqrt(squaredLength);
            dx *= inverseLength;
            dy *= inverseLength;
        }
        constrainBackward(chain, i, dx, dy);
        dx = _pass_direction[0];
        dy = _pass_direction[1];
        // when the base is free, the first bone swings about its end rather than its start
        if (i === 0 && !chain.fixedBase) {
            start[0] = end[0] - dx * bone.length;
            start[1] = end[1] - dy * bone.length;
        }
        const x = start[0] + dx * bone.length;
        const y = start[1] + dy * bone.length;
        end[0] = x;
        end[1] = y;
        if (i < count - 1) {
            const nextStart = bones[i + 1].start;
            nextStart[0] = x;
            nextStart[1] = y;
        }
        _pass_reference[0] = dx;
        _pass_reference[1] = dy;
        hasReference = true;
    }
    return chain;
}
/**
 * One full FABRIK iteration - {@link forward} then {@link backward}.
 *
 * @param chain the chain to move, mutated in place
 * @param target where the end effector should go
 * @returns the distance from the effector to `target` afterwards
 */
function iterate(chain, target) {
    if (chain.bones.length === 0)
        return Number.POSITIVE_INFINITY;
    forward(chain, target);
    backward(chain, chain.base);
    return distance(chain.bones[chain.bones.length - 1].end, target);
}
/**
 * Solves the chain for `target`, iterating until it is close enough, stops improving, or runs out
 * of attempts.
 *
 * A constrained chain does not converge monotonically, so the best pose seen is kept and restored.
 *
 * A dead-straight chain is the worst case to solve cold: every direction lies on one line, leaving
 * nothing sideways to bend it with. Give it a slight bend if the first solve matters.
 *
 * @param chain the chain to solve, mutated in place
 * @param target where the end effector should go
 * @returns the distance from the effector to `target`, also stored as `chain.solveDistance`
 */
function solve(chain, target) {
    const count = chain.bones.length;
    if (count === 0) {
        chain.solveDistance = Number.POSITIVE_INFINITY;
        return chain.solveDistance;
    }
    // in case bones were pushed onto the chain directly rather than through addBone
    if (chain.bestSolution.length < count * 4) {
        chain.bestSolution.length = count * 4;
        chain.bestSolution.fill(0);
    }
    let best = Number.POSITIVE_INFINITY;
    let previous = Number.POSITIVE_INFINITY;
    for (let i = 0; i < chain.maxIterations; i++) {
        const distance = iterate(chain, target);
        if (distance < best) {
            best = distance;
            saveSolution(chain);
            if (distance <= chain.solveDistanceThreshold)
                break;
        }
        else if (Math.abs(distance - previous) < chain.minIterationChange) {
            // ground to a halt - more iterations will not improve on what we already have
            break;
        }
        previous = distance;
    }
    if (best !== Number.POSITIVE_INFINITY) {
        restoreSolution(chain);
    }
    chain.solveDistance = best;
    return best;
}
/** Creates an empty structure. */
function createStructure2() {
    return { chains: [], connections: [] };
}
/**
 * Adds a chain that hangs off nothing, solving directly for the structure's target.
 *
 * @returns the index of the added chain
 */
function addChain(structure, chain) {
    structure.chains.push(chain);
    structure.connections.push({ hostChain: -1, hostBone: 0, point: BoneConnectionPoint.END });
    return structure.chains.length - 1;
}
/**
 * Adds a chain whose base is pinned to one end of a bone in a chain already in the structure.
 *
 * Add the host first: {@link solveStructure} walks chains in insertion order, so a chain always
 * sees its host already posed.
 *
 * @param structure the structure to add to
 * @param chain the chain to add
 * @param hostChain the index of the chain to hang off
 * @param hostBone the index of the bone within that chain
 * @param point which end of the host bone to attach to
 * @returns the index of the added chain
 */
function connectChain(structure, chain, hostChain, hostBone, point) {
    structure.chains.push(chain);
    structure.connections.push({ hostChain, hostBone, point });
    return structure.chains.length - 1;
}
/**
 * Solves every chain in the structure.
 *
 * A connected chain has its base moved onto its host bone's connection point and any `LOCAL_`
 * basebone constraint resolved against the host bone's direction first. A chain with
 * `useEmbeddedTarget` set solves for its own `embeddedTarget` instead.
 *
 * @param structure the structure to solve, mutated in place
 * @param target the target for every chain that does not use an embedded target
 */
function solveStructure(structure, target) {
    const chains = structure.chains;
    for (let i = 0; i < chains.length; i++) {
        const chain = chains[i];
        const connection = structure.connections[i];
        if (connection.hostChain >= 0) {
            const host = chains[connection.hostChain];
            const hostBone = host.bones[connection.hostBone];
            setBaseLocation(chain, connection.point === BoneConnectionPoint.START ? hostBone.start : hostBone.end);
            // being connected IS what fixes this chain's base - a connected chain with a free base
            // would drift away from the host it is supposed to hang off, so the flag is not the
            // caller's to clear here
            chain.fixedBase = true;
            const type = chain.baseboneConstraintType;
            if (type === BaseboneConstraintType.LOCAL_RELATIVE) {
                // the baseline is the host bone itself, so the chain bends relative to what it hangs off
                getBoneDirection(chain.baseboneWorldAxis, host, connection.hostBone);
            }
            else if (type === BaseboneConstraintType.LOCAL_ABSOLUTE) {
                // the baseline is expressed in the host bone's frame, so turn it by however far the
                // host bone is turned from straight up
                getBoneDirection(_structure_direction, host, connection.hostBone);
                rotate(chain.baseboneWorldAxis, chain.baseboneAxis, ORIGIN, signedAngle(UP, _structure_direction));
            }
        }
        solve(chain, chain.useEmbeddedTarget ? chain.embeddedTarget : target);
    }
}
/* internals */
const UP = [0, 1];
const ORIGIN = [0, 0];
const _structure_direction = [0, 0];
// the direction the current pass is constraining against - the previous bone's, whichever way the
// pass is walking - and the constrained result handed back from `constrain*`
const _pass_reference = [0, 0];
const _pass_direction = [0, 0];
const _constrain_baseline = [0, 0];
function clampAngle(radians) {
    return radians < 0 ? 0 : radians > Math.PI ? Math.PI : radians;
}
/**
 * Clamps the direction `(x, y)` into the wedge reaching `clockwise` one way and `anticlockwise` the
 * other from `baseline`. Writes `_pass_direction`.
 */
function constrainToWedge(x, y, baseline, clockwise, anticlockwise) {
    _pass_direction[0] = x;
    _pass_direction[1] = y;
    if (clockwise >= Math.PI && anticlockwise >= Math.PI)
        return;
    // anticlockwise is positive
    const signed = signedAngle(baseline, _pass_direction);
    if (signed > anticlockwise) {
        rotate(_pass_direction, baseline, ORIGIN, anticlockwise);
    }
    else if (signed < -clockwise) {
        rotate(_pass_direction, baseline, ORIGIN, -clockwise);
    }
}
/**
 * Constrains a bone's outer-to-inner direction during the forward pass.
 *
 * A `GLOBAL` joint pins this bone's own heading. A `LOCAL` joint limits the bend between two bones,
 * and that angle belongs to the joint between the two, which walking inward is the next bone's.
 *
 * The effector needs none of its own: its pair is clamped when the bone behind it is processed.
 *
 * Writes the result into `_pass_direction`, which starts out holding the unconstrained direction,
 * so a branch that does not apply can simply return.
 */
function constrainForward(chain, index, x, y, hasReference) {
    // start from the unconstrained direction. each branch below narrows it if it applies
    _pass_direction[0] = x;
    _pass_direction[1] = y;
    // a GLOBAL joint pins this bone's heading whatever its neighbours do. this pass works in
    // outer-to-inner directions, so the baseline is flipped to match
    if (index > 0) {
        const joint = chain.bones[index].joint;
        if (joint.coordinateSystem === ConstraintCoordinateSystem.GLOBAL) {
            _constrain_baseline[0] = -joint.globalAxis[0];
            _constrain_baseline[1] = -joint.globalAxis[1];
            constrainToWedge(x, y, _constrain_baseline, joint.clockwise, joint.anticlockwise);
            return;
        }
    }
    // the wedge shared with the bone one step further out. the end effector has no such bone
    if (!hasReference)
        return;
    const outer = chain.bones[index + 1].joint;
    if (outer.coordinateSystem !== ConstraintCoordinateSystem.LOCAL)
        return;
    if (outer.clockwise >= Math.PI && outer.anticlockwise >= Math.PI)
        return;
    // this pass measures the pair the other way round - from the outer bone back to this one - and a
    // wedge is not symmetric, so the two limits trade places
    constrainToWedge(x, y, _pass_reference, outer.anticlockwise, outer.clockwise);
}
/**
 * Constrains a bone's inner-to-outer direction during the backward pass.
 *
 * Writes the result into `_pass_direction`, which starts out holding the unconstrained direction,
 * so a branch that does not apply can simply return.
 */
function constrainBackward(chain, index, x, y) {
    // start from the unconstrained direction. each branch below narrows it if it applies
    _pass_direction[0] = x;
    _pass_direction[1] = y;
    if (index === 0) {
        if (chain.baseboneConstraintType === BaseboneConstraintType.NONE)
            return;
        constrainToWedge(x, y, chain.baseboneWorldAxis, chain.baseboneClockwise, chain.baseboneAnticlockwise);
        return;
    }
    const joint = chain.bones[index].joint;
    if (joint.clockwise >= Math.PI && joint.anticlockwise >= Math.PI)
        return;
    const baseline = joint.coordinateSystem === ConstraintCoordinateSystem.GLOBAL ? joint.globalAxis : _pass_reference;
    constrainToWedge(x, y, baseline, joint.clockwise, joint.anticlockwise);
}
function saveSolution(chain) {
    const bones = chain.bones;
    const solution = chain.bestSolution;
    for (let i = 0; i < bones.length; i++) {
        const bone = bones[i];
        const offset = i * 4;
        solution[offset] = bone.start[0];
        solution[offset + 1] = bone.start[1];
        solution[offset + 2] = bone.end[0];
        solution[offset + 3] = bone.end[1];
    }
}
function restoreSolution(chain) {
    const bones = chain.bones;
    const solution = chain.bestSolution;
    for (let i = 0; i < bones.length; i++) {
        const bone = bones[i];
        const offset = i * 4;
        bone.start[0] = solution[offset];
        bone.start[1] = solution[offset + 1];
        bone.end[0] = solution[offset + 2];
        bone.end[1] = solution[offset + 3];
    }
}

export { BaseboneConstraintType, BoneConnectionPoint, ConstraintCoordinateSystem, addBone, addBoneAtBase, addChain, addConsecutiveBone, backward, connectChain, createChain2, createJoint2, createStructure2, forward, getBoneAngle, getBoneDirection, getEffector, isReachable, iterate, setBaseLocation, setBaseboneConstraint, setGlobalJoint, setLocalJoint, solve, solveStructure, straighten };
//# sourceMappingURL=fabrik2.js.map
