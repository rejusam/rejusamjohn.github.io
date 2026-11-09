import {test} from 'node:test';
import assert from 'node:assert/strict';
import {svmFit, kktViolation, hinge} from '../assets/svm.js';
import {blobs2} from '../assets/classify.js';
import {dot, norm} from '../assets/maths.js';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);
const pm = y => y.map(v => (v === 1 ? 1 : -1));
const margins = (X, y, w, b) => X.map((x, i) => y[i] * (dot(w, x) + b));

const separable = () => {
  const {X, y} = blobs2({n0: 20, n1: 20, mu0: [-2, -1.5], mu1: [2, 1.5], sd: 0.7, seed: 21});
  return {X, y: pm(y)};
};
const overlapping = () => {
  const {X, y} = blobs2({n0: 25, n1: 25, mu0: [-1, -0.5], mu1: [1, 0.5], sd: 1.2, seed: 22});
  return {X, y: pm(y)};
};

test('svmFit hand example', () => {
  const X = [[1, 1], [2, 2], [-1, -1], [-2, -2]], y = [1, 1, -1, -1];
  const r = svmFit(X, y, {C: 1e6});
  assert.equal(r.ok, true);
  near(r.w[0], 0.5, 1e-6); near(r.w[1], 0.5, 1e-6);
  near(r.b, 0, 1e-6);
  assert.deepEqual(r.sv, [0, 2]);
  near(r.margin, 2 * Math.SQRT2, 1e-6);
  const h = hinge(X, y, r.w, r.b);
  h.forEach(v => near(v, 0, 1e-6));
  assert.deepEqual(hinge([[0, 0]], [1], [0.5, 0.5], 0), [1]);
});

test('svmFit works in three dimensions', () => {
  const X = [[1, 1, 2], [2, 2, 3], [-1, -1, 2], [-2, -2, 3]], y = [1, 1, -1, -1];
  const r = svmFit(X, y, {C: 1e6});
  assert.equal(r.ok, true);
  assert.equal(r.w.length, 3);
  near(r.w[0], 0.5, 1e-6); near(r.w[1], 0.5, 1e-6); near(r.w[2], 0, 1e-6);
  near(r.b, 0, 1e-6);
  assert.ok(r.kkt < 1e-6);
});

test('svmFit satisfies KKT', () => {
  const {X, y} = separable(), C = 10;
  const r = svmFit(X, y, {C});
  assert.equal(r.ok, true);
  assert.ok(r.kkt < 1e-6, `kkt ${r.kkt}`);
  near(kktViolation(X, y, C, r.alpha, r.b), r.kkt, 1e-15);
  near(r.alpha.reduce((s, a, i) => s + a * y[i], 0), 0, 1e-9);
  r.alpha.forEach(a => assert.ok(a >= 0 && a <= C));
  // w is the weighted sum of the training points
  const w = [0, 0];
  X.forEach((x, i) => { w[0] += r.alpha[i] * y[i] * x[0]; w[1] += r.alpha[i] * y[i] * x[1]; });
  near(w[0], r.w[0], 1e-12); near(w[1], r.w[1], 1e-12);
});

test('svmFit satisfies KKT at the C slider\'s ends', () => {
  const {X, y} = overlapping();
  for (const C of [0.01, 100]) {
    const r = svmFit(X, y, {C});
    assert.equal(r.ok, true, `C ${C}`);
    assert.ok(r.kkt < 1e-6, `C ${C}: kkt ${r.kkt}`);
    r.alpha.forEach(a => assert.ok(a >= 0 && a <= C));
  }
  // the data really overlap: some point sits inside the margin even at C = 100
  const r = svmFit(X, y, {C: 100});
  assert.ok(Math.max(...hinge(X, y, r.w, r.b)) > 1);
});

test('large C reproduces the hard margin', () => {
  const {X, y} = separable();
  const a = svmFit(X, y, {C: 1e6}), b = svmFit(X, y, {C: 1e8});
  assert.ok(a.ok && b.ok);
  near(a.w[0], b.w[0], 1e-6); near(a.w[1], b.w[1], 1e-6);
  near(a.b, b.b, 1e-6);
  margins(X, y, a.w, a.b).forEach(m => assert.ok(m >= 1 - 1e-6, `y f = ${m}`));
});

test('moving a non-support vector changes nothing', () => {
  const {X, y} = separable(), C = 1e6;
  const r = svmFit(X, y, {C});
  const m = margins(X, y, r.w, r.b);
  const k = m.reduce((best, v, i) => (r.alpha[i] === 0 && v > m[best] ? i : best),
    r.alpha.findIndex(a => a === 0));
  assert.ok(k >= 0 && r.alpha[k] === 0);
  const u = r.w.map(v => v / norm(r.w));
  const X2 = X.map(x => [...x]);
  X2[k] = X[k].map((v, j) => v + y[k] * 1.5 * u[j]);
  const s = svmFit(X2, y, {C});
  assert.equal(s.ok, true);
  near(s.w[0], r.w[0], 1e-7); near(s.w[1], r.w[1], 1e-7);
  near(s.b, r.b, 1e-7);
});

test('svmFit returns ok:false and finite values when maxIter is hit', () => {
  const {X, y} = overlapping();
  const r = svmFit(X, y, {C: 1, maxIter: 1});
  assert.equal(r.ok, false);
  assert.equal(r.iters, 1);
  for (const v of [...r.w, r.b, ...r.alpha, r.margin, r.kkt]) assert.ok(Number.isFinite(v), `${v}`);
});
