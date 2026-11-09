import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  blobs2, sigmoid, logLoss, logisticFit, logisticGD, knnPredict,
  accuracy, confusion, gaussianNB, nbPosterior,
  gini, bestSplit, growTree, predictTree, treeStats, bootstrapIdx, forest, stumpFit, boost
} from '../assets/classify.js';
import {mulberry32, centralDiff, mean} from '../assets/maths.js';
import {sineData} from '../assets/sine-data.js';

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

test('gini values', () => {
  near(gini([5, 5]), 0.5, 1e-15);
  assert.equal(gini([10, 0]), 0);
  assert.equal(gini([]), 0);
  near(gini([1, 3]), 0.375, 1e-15);
});

test('bestSplit equals brute force', () => {
  const {X, y} = blobs2({n0: 30, n1: 30, mu0: [0, 0], mu1: [1.2, 0.8], sd: 1, seed: 5});
  const idx = X.map((_, i) => i), n = idx.length;
  let best = null;
  for (let f = 0; f < 2; f++) {
    const v = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
    for (let k = 0; k + 1 < v.length; k++) {
      const t = (v[k] + v[k + 1]) / 2, c = [[0, 0], [0, 0]];
      X.forEach((r, i) => c[r[f] < t ? 0 : 1][y[i]]++);
      const imp = (c[0][0] + c[0][1]) / n * gini(c[0]) + (c[1][0] + c[1][1]) / n * gini(c[1]);
      if (!best || imp < best.imp - 1e-12) best = {f, t, imp};
    }
  }
  const s = bestSplit(X, y, idx);
  assert.equal(s.feature, best.f);
  near(s.threshold, best.t, 1e-12);
  near(s.impurity, best.imp, 1e-12);
  assert.equal(s.left.length + s.right.length, n);
  assert.ok(s.left.every(i => X[i][s.feature] < s.threshold));
  assert.ok(s.right.every(i => X[i][s.feature] >= s.threshold));
  assert.equal(bestSplit(X, y, idx.filter(i => y[i] === 0)), null);
});

test('growTree with no depth limit fits the training data', () => {
  const {X, y} = blobs2({n0: 40, n1: 40, mu0: [0, 0], mu1: [1, 1], sd: 1, seed: 8});
  const seen = new Map();
  X.forEach((r, i) => { const k = r.join(','); assert.ok(!seen.has(k) || seen.get(k) === y[i]); seen.set(k, y[i]); });
  const tree = growTree(X, y);
  X.forEach((r, i) => assert.equal(predictTree(tree, r), y[i]));
});

test('growTree respects maxDepth', () => {
  const {X, y} = blobs2({n0: 60, n1: 60, mu0: [0, 0], mu1: [1, 1], sd: 1, seed: 3});
  for (let d = 1; d <= 4; d++) {
    const st = treeStats(growTree(X, y, {maxDepth: d}));
    assert.ok(st.depth <= d, `depth ${st.depth} > ${d}`);
    assert.ok(st.leaves >= 2);
  }
});

test('bootstrap leaves out about 1/e', () => {
  const n = 100000, idx = bootstrapIdx(n, mulberry32(11));
  assert.equal(idx.length, n);
  near(new Set(idx).size / n, 1 - (1 - 1 / n) ** n, 0.005);
});

test('forest prefix property', () => {
  const {X, y} = blobs2({n0: 40, n1: 40, mu0: [0, 0], mu1: [1.2, 0.8], sd: 1, seed: 2});
  const big = forest(X, y, {B: 7, mtry: 1, seed: 4}), small = forest(X, y, {B: 3, mtry: 1, seed: 4});
  for (let i = 0; i < 10; i++) assert.equal(big.votes(X[i], 3), small.votes(X[i]));
});

test('growTree with mtry keeps searching when the sampled features give no split', () => {
  // Feature 0 is constant, so it can never split; feature 1 separates the classes.
  const X = [[1, 0], [1, 1], [1, 2], [1, 3]], y = [0, 0, 1, 1];
  // rng() = 0 always samples feature 0.
  const tree = growTree(X, y, {mtry: 1, rng: () => 0});
  assert.equal(tree.leaf, false);
  assert.equal(tree.feature, 1);
  assert.equal(tree.threshold, 1.5);
  X.forEach((r, i) => assert.equal(predictTree(tree, r), y[i]));
  // The fallback takes no extra draws: one draw per node (root and two leaves).
  let draws = 0;
  growTree(X, y, {mtry: 1, rng: () => { draws++; return 0; }});
  assert.equal(draws, 3);
  // A node that no feature can split is still a leaf.
  const flat = growTree([[1, 1], [1, 1]], [0, 1], {mtry: 1, rng: () => 0});
  assert.equal(flat.leaf, true);
});

test('forest OOB matches a direct count with a Set per tree', () => {
  const {X, y} = blobs2({n0: 30, n1: 30, mu0: [0, 0], mu1: [1, 1], sd: 1, seed: 12});
  const f = forest(X, y, {B: 9, mtry: 1, seed: 3});
  for (const b of [1, 4, 9]) {
    const o = f.oob(b);
    X.forEach((r, i) => {
      const ts = f.trees.slice(0, b).filter((_, t) => !f.samples[t].includes(i));
      if (ts.length === 0) return assert.equal(o.pred[i], null);
      assert.equal(o.pred[i], ts.filter(t => predictTree(t, r) === 1).length / ts.length > 0.5 ? 1 : 0);
    });
  }
});

test('forest OOB uses only trees that left the row out', () => {
  const {X, y} = blobs2({n0: 40, n1: 40, mu0: [0, 0], mu1: [1.2, 0.8], sd: 1, seed: 6});
  const f = forest(X, y, {B: 15, maxDepth: 3, mtry: 1, seed: 9}), o = f.oob();
  let covered = 0, right = 0;
  X.forEach((r, i) => {
    const ts = f.trees.filter((_, t) => !f.samples[t].includes(i));
    if (ts.length === 0) { assert.equal(o.pred[i], null); return; }
    const frac = ts.filter(t => predictTree(t, r) === 1).length / ts.length;
    assert.equal(o.pred[i], frac > 0.5 ? 1 : 0);
    covered++; if (o.pred[i] === y[i]) right++;
  });
  assert.equal(o.covered, covered);
  near(o.accuracy, right / covered, 1e-15);
});

test('stumpFit finds the least-squares split', () => {
  const xs = [1, 2, 3, 4, 5, 6], r = [1, 1.2, 0.8, 5, 5.2, 4.8];
  const s = stumpFit(xs, r);
  near(s.threshold, 3.5, 1e-12); near(s.left, 1, 1e-12); near(s.right, 5, 1e-12);
  near(s.sse, 0.16, 1e-12);
});

test('boost residuals are the negative gradient', () => {
  const {xs: x, ys: y} = sineData({n: 30, seed: 4}), eta = 0.3;
  const b = boost(x, y, {M: 6, eta});
  for (let m = 1; m <= 6; m++) {
    const res = x.map((xi, i) => {
      const F = b.predict(xi, m - 1);
      const res = -centralDiff(f => 0.5 * (y[i] - f) ** 2, F);
      near(res, y[i] - F, 1e-8);
      return res;
    });
    const st = b.stumps[m - 1];
    near(st.left, mean(res.filter((_, i) => x[i] < st.threshold)), 1e-10);
    near(st.right, mean(res.filter((_, i) => x[i] >= st.threshold)), 1e-10);
  }
});

test('boost train MSE never rises', () => {
  const {xs: x, ys: y} = sineData({n: 40, seed: 9});
  for (const eta of [0.1, 1, 1.5]) {
    const b = boost(x, y, {M: 200, eta});
    assert.equal(b.trainMSE.length, 201);
    for (let m = 1; m <= 200; m++) assert.ok(b.trainMSE[m] <= b.trainMSE[m - 1] + 1e-12, `eta ${eta} m ${m}`);
  }
});
