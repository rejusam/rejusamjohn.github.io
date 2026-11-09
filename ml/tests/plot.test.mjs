import test from 'node:test';
import assert from 'node:assert/strict';
import { linearScale, clamp, pathFrom } from '../assets/plot.js';

test('linearScale maps and inverts', () => {
  const s = linearScale([0, 10], [50, 550]);
  assert.equal(s(5), 300);
  assert.equal(s.invert(300), 5);
});

test('ticks are nice', () => {
  const t = linearScale([0, 1], [0, 1]).ticks(5);
  const want = [0, 0.2, 0.4, 0.6, 0.8, 1];
  assert.equal(t.length, want.length);
  t.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12));
});

test('clamp keeps drag inside domain', () => {
  assert.equal(clamp(12, 0, 10), 10);
  assert.equal(clamp(-1, 0, 10), 0);
});

test('pathFrom skips non-finite', () => {
  const id = (v) => v;
  const d = pathFrom([[0, 0], [1, NaN], [2, 2]], id, id);
  assert.ok(!d.includes('NaN'));
  assert.ok(d.startsWith('M'));
});
