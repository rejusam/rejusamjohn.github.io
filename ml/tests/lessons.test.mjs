// Tests for the data and the numbers each lesson's stage shows.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import * as lesson1 from '../what-is-a-model/stage.js';

test('lesson1 data is reproducible', () => {
  const a = lesson1.lessonData();
  const b = lesson1.lessonData();
  assert.equal(a.xs.length, 20);
  assert.equal(a.ys.length, 20);
  assert.deepEqual(a.xs, b.xs);
  assert.deepEqual(a.ys, b.ys);
  for (const x of a.xs) assert.ok(x >= 1 && x <= 10, `x out of range: ${x}`);
  for (const y of a.ys) assert.ok(y >= 0 && y <= 100, `y out of range: ${y}`);
});

test('lesson1 fitted slope is near the generating slope', () => {
  const { xs, ys } = lesson1.lessonData();
  const fit = lesson1.fitRule(xs, ys);
  assert.ok(fit.ok);
  assert.ok(Math.abs(fit.b - 5) < 1.5, `slope ${fit.b}`);
});

test('lesson1 stage numbers agree with the fitted rule', () => {
  const { xs, ys } = lesson1.lessonData();
  const fit = lesson1.fitRule(xs, ys);
  // Least squares residuals sum to zero, and the fit beats any other line.
  const s = lesson1.missSummary(xs, ys, fit);
  assert.ok(Math.abs(s.residuals.reduce((t, e) => t + e, 0)) < 1e-9);
  const worse = lesson1.missSummary(xs, ys, { a: fit.a + 1, b: fit.b });
  assert.ok(worse.mse > s.mse);
  assert.equal(lesson1.predict(fit, 0), fit.a);
  assert.ok(s.maxAbs >= Math.abs(s.residuals[s.worst]) - 1e-12);
  assert.equal(lesson1.fmt(-2, 1), '−2.0');
  assert.equal(lesson1.fmt(34.264, 1), '34.3');
});
