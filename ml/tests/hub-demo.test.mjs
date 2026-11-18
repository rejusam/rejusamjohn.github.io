import test from 'node:test';
import assert from 'node:assert/strict';
import { DATA, lossAt, gradAt, bestW, descend } from '../assets/hub-demo.js';

test('the eight points are centred on x = 0', () => {
  assert.equal(DATA.x.length, 8);
  assert.equal(DATA.x.reduce((a, b) => a + b, 0), 0);
});

test('bestW is the least-squares slope through the origin', () => {
  const sxy = DATA.x.reduce((s, x, i) => s + x * DATA.y[i], 0);
  const sxx = DATA.x.reduce((s, x) => s + x * x, 0);
  assert.equal(bestW(), sxy / sxx);
  // The gradient is zero there, and the loss is lowest there.
  assert.ok(Math.abs(gradAt(bestW())) < 1e-12);
  for (const dw of [-0.01, 0.01, -1, 1]) assert.ok(lossAt(bestW() + dw) > lossAt(bestW()));
});

test('gradAt matches a central difference of lossAt', () => {
  for (const w of [-1.5, -1, 0, 0.7, 2.4]) {
    const h = 1e-5;
    const fd = (lossAt(w + h) - lossAt(w - h)) / (2 * h);
    assert.ok(Math.abs(fd - gradAt(w)) < 1e-6, `w = ${w}`);
  }
});

test('the run starts at w = -1 and the loss falls at every step', () => {
  const run = descend();
  assert.equal(run[0].w, -1);
  assert.equal(run.length, 19);
  for (let k = 1; k < run.length; k++) assert.ok(run[k].loss < run[k - 1].loss, `step ${k}`);
});

test('the run ends at the best slope to two decimals, as the figure shows it', () => {
  const run = descend();
  const last = run[run.length - 1];
  assert.equal(last.w.toFixed(2), bestW().toFixed(2));
  assert.equal(last.loss.toFixed(2), lossAt(bestW()).toFixed(2));
});

test('each step moves w by eta times the gradient', () => {
  const run = descend({ eta: 0.03 });
  for (let k = 1; k < run.length; k++) {
    assert.ok(Math.abs(run[k].w - (run[k - 1].w - 0.03 * run[k - 1].grad)) < 1e-12);
  }
});
