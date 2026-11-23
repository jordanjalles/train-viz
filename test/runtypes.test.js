// Run types of the run page (public/app.js): which charts a run gets, from its config or its metric keys.
// app.js needs a DOM, so this test evaluates only its run-type block.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');
const a = src.indexOf('const TRAIN_KEYS');
const b = src.indexOf('/** Run page shell');
assert.ok(a > 0 && b > a);
const ctx = vm.createContext({});
vm.runInContext(`${src.slice(a, b)}\nglobalThis.out = { RUN_TYPES, runType, LOOP_CURVES };`, ctx);
const { RUN_TYPES, runType, LOOP_CURVES } = ctx.out;
const run = (config, ...records) => ({ config, metrics: records });

test('a run type comes from config.run_type, else from the metric keys', () => {
  assert.equal(runType(run({ run_type: 'pixel_loop' })), 'pixel_loop');
  assert.equal(runType(run({ run_type: 'no such type' }, { step: 1, clean: 0.1, view: 0.2 })), 'ae');
  assert.equal(runType(run(null, { step: 100, clean: 0.1, lap: 0.2, view: 0.3 }, { step: 100, test_clean: 0.01, z_eff_rank: 30 })), 'ae');
  assert.equal(runType(run({}, { step: 100, clean: 0.05 }, { step: 2000, test_clean: 0.04, test_x_mse: 0.01, x_mse_t05: 0.008 })), 'pixel');
  assert.equal(runType(run({}, { step: 100, clean: 1.2, sparse_flow_loss: 1.3, dense_flow_loss: 1.1, xm_gap: 0.02 })), 'flow');
  assert.equal(runType(run({}, { step: 100, clean: 0.05, L1: 0.06, L4: 0.05 }, { step: 2000, test_x_mse: 0.01, loopq_dense_1: 0.05 })), 'pixel_loop');
  assert.equal(runType(run({}, )), 'ae');
});

test('every type lists charts with keys, and only the AE type has the AE sections', () => {
  for (const [name, T] of Object.entries(RUN_TYPES)) {
    for (const g of [...T.train, ...T.evals]) assert.ok(g.title && g.keys.length, `${name}: ${g.title}`);
    for (const k of ['testLabel', 'trainLabel', 'reconTitle', 'reconNote']) assert.ok(T[k], `${name}.${k}`);
    assert.equal(!!T.robust, name === 'ae');
    assert.equal(!!T.diag, name === 'ae');
  }
  // no pixel or flow chart asks for an AE-only metric
  const aeOnly = ['lap', 'reg', 'view', 'z_eff_rank', 'recon_div', 'z_std_mean', 'dec_patch_rms', 'lap_ratio'];
  // except: the unify pyramid job logs its own latent snapshot as z_std_mean + z_pr_rank (unify3090/job/main.py)
  const alsoLogs = { pixel_pyramid: ['z_std_mean'] };
  for (const name of ['pixel', 'pixel_loop', 'pixel_pyramid', 'flow']) for (const g of [...RUN_TYPES[name].train, ...RUN_TYPES[name].evals]) for (const k of g.keys) assert.ok(!aeOnly.includes(k) || alsoLogs[name]?.includes(k), `${name}: ${k}`);
});

test('the looped type has the five loop charts', () => {
  const T = RUN_TYPES.pixel_loop;
  assert.deepEqual([...T.train.find((g) => g.keys.includes('L1')).keys], ['L1', 'L2', 'L3', 'L4', 'L_avg123', 'clean']);
  assert.deepEqual([...T.curves.map((c) => c.families.map((f) => f.prefix).join('+'))], ['loopq_dense_+loopq_sparse_', 'upd_', 'cos_', 'xsa_']);
  assert.equal(T.curves, LOOP_CURVES);
  assert.equal(LOOP_CURVES[1].xFmt(2), '2→3');
  assert.equal(LOOP_CURVES[1].xFmt(2.5), '');
  assert.equal(LOOP_CURVES[0].xFmt(6), '6');
});
