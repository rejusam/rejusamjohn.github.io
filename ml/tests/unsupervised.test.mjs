import {test} from 'node:test';
import assert from 'node:assert/strict';
import {inertia, kmeans, cov2, pca2, projVar, perpMSE} from '../assets/unsupervised.js';
import {blobs2} from '../assets/classify.js';
import {mulberry32, randn} from '../assets/maths.js';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);

// 3 blobs, 20 points each, centres (0,0), (6,0), (3,5), sd 0.8, seed 31.
function threeBlobs() {
  const rng = mulberry32(31);
  const X = [];
  for (const [cx, cy] of [[0, 0], [6, 0], [3, 5]])
    for (let i = 0; i < 20; i++) X.push([cx + 0.8 * randn(rng), cy + 0.8 * randn(rng)]);
  return X;
}
const initFrom = X => [X[0], X[20], X[40]].map(p => p.slice());

test('kmeans inertia never rises', () => {
  const X = threeBlobs();
  const r = kmeans(X, initFrom(X), {});
  const vals = r.states.filter(s => s.inertia !== null).map(s => s.inertia);
  assert.ok(vals.length >= 3);
  for (let i = 1; i < vals.length; i++) assert.ok(vals[i] <= vals[i - 1] + 1e-12);
  assert.equal(r.states[0].phase, 'init');
  assert.equal(r.states[0].labels, null);
  assert.equal(r.states[0].inertia, null);
});

test('kmeans reaches a fixed point', () => {
  const X = threeBlobs();
  const r = kmeans(X, initFrom(X), {});
  assert.ok(r.converged);
  const r2 = kmeans(X, r.centres, {});
  assert.deepEqual(r2.labels, r.labels);
  assert.deepEqual(r2.centres, r.centres);
  near(r2.inertia, r.inertia, 1e-12);
});

test("kmeans keeps an empty cluster's centre", () => {
  const X = threeBlobs();
  const far = [1000, 1000];
  const r = kmeans(X, [X[0].slice(), X[20].slice(), far], {});
  assert.deepEqual(r.centres[2], far);
  for (const c of r.centres) for (const v of c) assert.ok(Number.isFinite(v));
  assert.ok(Number.isFinite(r.inertia));
  const upd = r.states.find(s => s.phase === 'update');
  assert.deepEqual(upd.centres[2], far);
});

test('kmeans hand example', () => {
  const X = [[0, 0], [0, 1], [10, 0], [10, 1]];
  const r = kmeans(X, [[0, 0], [10, 0]], {});
  assert.deepEqual(r.centres, [[0, 0.5], [10, 0.5]]);
  assert.deepEqual(r.labels, [0, 0, 1, 1]);
  near(r.inertia, 1, 1e-12);
  near(inertia(X, r.centres, r.labels), 1, 1e-12);
});

test('cov2 uses n-1', () => {
  // x = 1,2,3 ; y = 2,4,9: mean (2, 5); dx = -1,0,1; dy = -3,-1,4
  const S = cov2([[1, 2], [2, 4], [3, 9]]);
  near(S[0][0], 1, 1e-15);
  near(S[1][1], 26 / 2, 1e-15);
  near(S[0][1], 7 / 2, 1e-15);
  assert.equal(S[0][1], S[1][0]);
});

function corrData() {
  return blobs2({n0: 300, n1: 0, mu0: [1, -1], mu1: [0, 0], sd: [2, 1], rho: 0.8, seed: 9}).X;
}

test('pca2 eigen pairs', () => {
  const X = corrData();
  const {S, gammas, vecs} = pca2(X);
  for (let k = 0; k < 2; k++) {
    const v = vecs[k];
    near(S[0][0] * v[0] + S[0][1] * v[1], gammas[k] * v[0], 1e-10);
    near(S[1][0] * v[0] + S[1][1] * v[1], gammas[k] * v[1], 1e-10);
    near(Math.hypot(v[0], v[1]), 1, 1e-12);
  }
  near(gammas[0] + gammas[1], S[0][0] + S[1][1], 1e-12);
  near(vecs[0][0] * vecs[1][0] + vecs[0][1] * vecs[1][1], 0, 1e-12);
  assert.ok(gammas[0] >= gammas[1]);
});

test('pca2 handles sxy = 0 and the sign convention', () => {
  const r = pca2([[-2, -1], [-2, 1], [2, -1], [2, 1]]);
  assert.deepEqual(r.vecs[0], [1, 0]);
  assert.deepEqual(r.vecs[1].map(Math.abs), [0, 1]);
  assert.ok(r.vecs[1][1] > 0);
  const e = r.explained;
  near(e[0] + e[1], 1, 1e-12);
  const q = pca2(corrData());
  for (const v of q.vecs) {
    const big = Math.abs(v[0]) >= Math.abs(v[1]) ? v[0] : v[1];
    assert.ok(big > 0);
  }
});

test('projection variance peaks at v1', () => {
  const X = blobs2({n0: 500, n1: 0, mu0: [0, 0], mu1: [0, 0], sd: 1, rho: 0.8, seed: 4}).X;
  const {gammas, vecs} = pca2(X);
  let best = -1, bestT = 0;
  for (let i = 0; i < 3600; i++) {
    const t = (i / 3600) * Math.PI;
    const v = projVar(X, t);
    if (v > best) { best = v; bestT = t; }
  }
  let a = Math.atan2(vecs[0][1], vecs[0][0]);
  a = ((a % Math.PI) + Math.PI) % Math.PI;
  const d = Math.min(Math.abs(bestT - a), Math.PI - Math.abs(bestT - a));
  assert.ok(d < 0.001, `angle diff ${d}`);
  near(best, gammas[0], 1e-6);
});

test('perpendicular error is the variance not captured', () => {
  const X = corrData();
  const n = X.length;
  const {S} = pca2(X);
  for (const t of [0, 0.4, 1.1, 2.0, 2.9])
    near(perpMSE(X, t) * n / (n - 1) + projVar(X, t), S[0][0] + S[1][1], 1e-10);
});
