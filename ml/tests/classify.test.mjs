import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  blobs2, sigmoid, logLoss, logisticFit, logisticGD, knnPredict,
  accuracy, confusion, gaussianNB, nbPosterior
} from '../assets/classify.js';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);

test('blobs2 is reproducible and ordered', () => {
  const o = {n0: 4000, n1: 4000, mu0: [0, 0], mu1: [3, -2], sd: 1, rho: 0.6, seed: 5};
  const a = blobs2(o), b = blobs2(o);
  assert.deepEqual(a, b);
  assert.deepEqual(a.y, [...Array(4000).fill(0), ...Array(4000).fill(1)]);
  const c1 = a.X.slice(4000);
  const mx = c1.reduce((s, r) => s + r[0], 0) / 4000, my = c1.reduce((s, r) => s + r[1], 0) / 4000;
  near(mx, 3, 0.05); near(my, -2, 0.05);
  let sxy = 0, sxx = 0, syy = 0;
  for (const r of c1) { sxy += (r[0] - mx) * (r[1] - my); sxx += (r[0] - mx) ** 2; syy += (r[1] - my) ** 2; }
  near(sxy / Math.sqrt(sxx * syy), 0.6, 0.05);
});

test('sigmoid is stable and symmetric', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.equal(sigmoid(800), 1);
  assert.equal(sigmoid(-800), 0);
  for (const z of [-30, -2, 0.3, 7]) near(sigmoid(z) + sigmoid(-z), 1, 1e-15);
});

test('logLoss is finite at p = 0 and 1', () => {
  const l = logLoss([1], [0]);
  assert.ok(Number.isFinite(l));
  near(l, -Math.log(1e-15), 1e-9);
  near(logLoss([1, 0], [0.5, 0.5]), Math.LN2, 1e-15);
  assert.ok(Number.isFinite(logLoss([0], [1])));
});

const fitData = () => blobs2({n0: 50, n1: 50, mu0: [0, 0], mu1: [1.5, 1], sd: 1, seed: 11});
const lossAt = (X, y, w0, w) => logLoss(y, X.map(r => sigmoid(w0 + r[0] * w[0] + r[1] * w[1])));

test('logisticFit reaches zero gradient', () => {
  const {X, y} = fitData();
  const f = logisticFit(X, y);
  assert.equal(f.ok, true);
  assert.ok(f.gradNorm < 1e-8);
  const g = [0, 0, 0];
  X.forEach((r, i) => {
    const e = sigmoid(f.w0 + r[0] * f.w[0] + r[1] * f.w[1]) - y[i];
    g[0] += e; g[1] += e * r[0]; g[2] += e * r[1];
  });
  assert.ok(Math.max(...g.map(v => Math.abs(v / X.length))) < 1e-8);
  const base = lossAt(X, y, f.w0, f.w);
  for (const d of [0.01, -0.01]) {
    assert.ok(base < lossAt(X, y, f.w0 + d, f.w));
    assert.ok(base < lossAt(X, y, f.w0, [f.w[0] + d, f.w[1]]));
    assert.ok(base < lossAt(X, y, f.w0, [f.w[0], f.w[1] + d]));
  }
});

test('logisticGD approaches the Newton fit', () => {
  const {X, y} = fitData();
  const f = logisticFit(X, y);
  const path = logisticGD(X, y, {eta: 0.5, steps: 20000});
  assert.equal(path.length, 20001);
  const last = path[path.length - 1];
  near(last.w0, f.w0, 1e-4); near(last.w[0], f.w[0], 1e-4); near(last.w[1], f.w[1], 1e-4);
  for (let i = 1; i < path.length; i++) assert.ok(path[i].loss <= path[i - 1].loss + 1e-15);
});

test('logisticFit flags separable data', () => {
  const f = logisticFit([[0], [1], [2], [3]], [0, 0, 1, 1]);
  assert.equal(f.ok, false);
  assert.equal(f.reason, 'separable');
  assert.ok(Number.isFinite(f.w0) && f.w.every(Number.isFinite));
});

test('knnPredict breaks distance ties by lower index', () => {
  const r = knnPredict([[1, 0], [-1, 0], [0, 1]], [1, 0, 0], [0, 0], 1);
  assert.deepEqual(r.idx, [0]);
  assert.equal(r.cls, 1);
});

test('knnPredict with k=1 recovers each training label', () => {
  const {X, y} = blobs2({n0: 30, n1: 30, mu0: [0, 0], mu1: [1, 1], sd: 1, seed: 3});
  X.forEach((x, i) => assert.equal(knnPredict(X, y, x, 1).cls, y[i]));
});

test('confusion counts add up', () => {
  const c = confusion([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]);
  assert.equal(c.tp, 2); assert.equal(c.fn, 1); assert.equal(c.tn, 1); assert.equal(c.fp, 1);
  near(c.accuracy, 0.6, 1e-15); near(c.precision, 2 / 3, 1e-15); near(c.recall, 2 / 3, 1e-15);
  assert.equal(accuracy([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]), 0.6);
  assert.equal(confusion([1, 0], [0, 0]).precision, null);
  assert.equal(confusion([0, 0], [0, 1]).recall, null);
});

test('nbPosterior matches a hand Bayes calculation', () => {
  const m = {prior: [0.6, 0.4], mu: [[0], [2]], vari: [[1], [1]]};
  near(nbPosterior(m, [1])[1], 0.4, 1e-12);
});

test('gaussianNB uses MLE variance', () => {
  const m = gaussianNB([[1], [3], [10], [12]], [0, 0, 1, 1]);
  near(m.vari[0][0], 1, 1e-15);
  near(m.mu[0][0], 2, 1e-15);
  assert.deepEqual(m.prior, [0.5, 0.5]);
});

test('nbPosterior is stable far from both classes', () => {
  const m = {prior: [0.5, 0.5], mu: [[0, 0], [2, 2]], vari: [[1, 1], [1, 1]]};
  const p = nbPosterior(m, [50, -50]);
  assert.ok(p.every(Number.isFinite));
  near(p[0] + p[1], 1, 1e-12);
});
