import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { RunState } from '../lib/state.js';

const tmpFile = () => path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'tvstate-')), 'state', 'run-state.json');

test('keep / archive / restore round trip, persisted across instances', () => {
  const f = tmpFile();
  const s = new RunState(f);
  assert.deepEqual(s.apply('archive', ['a', 'b']), { changed: 2, skipped: [] });
  assert.deepEqual(s.apply('keep', ['c']), { changed: 1, skipped: [] });
  const s2 = new RunState(f);
  assert.deepEqual(s2.toJSON(), { kept: ['c'], archived: ['a', 'b'] });
  s2.apply('unarchive', ['a']);
  assert.deepEqual(new RunState(f).toJSON().archived, ['b']);
});

test('kept runs are never archived; keeping an archived run restores it', () => {
  const s = new RunState(tmpFile());
  s.apply('keep', ['k']);
  assert.deepEqual(s.apply('archive', ['k', 'x']), { changed: 1, skipped: ['k'] });
  s.apply('keep', ['x']);
  assert.deepEqual(s.toJSON(), { kept: ['k', 'x'], archived: [] });
});

test('bad input is rejected and changes nothing', () => {
  const s = new RunState(tmpFile());
  assert.throws(() => s.apply('delete', ['a']));
  assert.throws(() => s.apply('archive', 'a'));
  assert.throws(() => s.apply('archive', [1]));
  assert.deepEqual(s.toJSON(), { kept: [], archived: [] });
});
