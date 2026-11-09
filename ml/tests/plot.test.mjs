import test from 'node:test';
import assert from 'node:assert/strict';
import { linearScale, clamp, pathFrom, fmt } from '../assets/plot.js';

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

test('tick labels use a true minus sign', () => {
  assert.equal(fmt(-2), '\u22122');
  assert.equal(fmt(-0.5), '\u22120.5');
  assert.equal(fmt(3), '3');
  assert.equal(fmt(0), '0');
  const labels = linearScale([-1.8, 1.8], [0, 100]).ticks(5).map(fmt);
  assert.deepEqual(labels, ['\u22121', '0', '1']);
});
