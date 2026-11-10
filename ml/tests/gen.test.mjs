import test from 'node:test';
import assert from 'node:assert/strict';
import {gauss1Fit, gauss2Fit, blobs3, BLOBS, spiral2D, gmmInit, gmmEM, gmmLogLik, gmmSample,
  ddpmSchedule, qSample, timeFeatures, inputFeatures, LESSON_SPIRAL, denoiserInit, denoiserForward, denoiserPreacts,
  denoiserGrad, denoiserFlatten, denoiserUnflatten, ddpmTrainer, ddpmStep, ddpmSample,
  nearestDist} from '../assets/gen.js';
import {gradCheck, spiralData} from '../assets/nn.js';
import {mulberry32, randn} from '../assets/maths.js';

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) < tol, `${msg ?? ''} ${a} vs ${b}`);

test('gaussian MLE by hand', () => {
  // mean 4, squared deviations 4 + 0 + 4 + 0 = 8, divided by n = 4 (population)
  const g = gauss1Fit([2, 4, 6, 4]);
  assert.equal(g.mu, 4);
  near(g.s2, 2, 1e-12);
  const g1 = gauss1Fit([1, 3, 5, 7]);
  assert.equal(g1.mu, 4);
  near(g1.s2, 5, 1e-12); // (9 + 1 + 1 + 9) / 4
  const g2 = gauss2Fit([[0, 0], [2, 0], [0, 2], [2, 2]]);
  assert.deepEqual(g2.mu, [1, 1]);
  near(g2.S[0][0], 1, 1e-12); near(g2.S[1][1], 1, 1e-12);
  near(g2.S[0][1], 0, 1e-12); near(g2.S[1][0], 0, 1e-12);
  const g3 = gauss2Fit([[0, 0], [1, 1], [2, 2]]);
  near(g3.S[0][1], 2 / 3, 1e-12); // (1 + 0 + 1) / 3
});

test('data: blobs3 and spiral2D are seeded; spiral2D is standardised', () => {
  assert.deepEqual(blobs3({n: 30, seed: 271}), blobs3({n: 30, seed: 271}));
  assert.equal(blobs3({n: 30, seed: 271}).length, 30);
  assert.equal(BLOBS.length, 3);
  const S = spiral2D({n: 400, seed: 305});
  assert.equal(S.length, 400);
  for (const k of [0, 1]) {
    const v = S.map(p => p[k]);
    const m = v.reduce((a, b) => a + b) / v.length;
    const s2 = v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length;
    near(m, 0, 1e-12); near(s2, 1, 1e-12);
  }
  // turns and noise pass through: same shape up to the standardising scale
  const A = spiral2D({n: 40, seed: 1, turns: 1, noise: 0.1}), raw = spiralData({n: 40, seed: 1, turns: 1, noise: 0.1}).X;
  const r = (A[1][0] - A[0][0]) / (raw[1][0] - raw[0][0]);
  near((A[5][0] - A[0][0]) / (raw[5][0] - raw[0][0]), r, 1e-9);
  assert.notDeepEqual(A, spiral2D({n: 40, seed: 1}));
});

test('gmmInit picks distinct data points and equal weights', () => {
  const X = blobs3({n: 60, seed: 271});
  const p = gmmInit(X, 5, 7);
  assert.deepEqual(p.pi, [0.2, 0.2, 0.2, 0.2, 0.2]);
  const keys = p.mu.map(m => m.join(','));
  assert.equal(new Set(keys).size, 5);
  for (const m of p.mu) assert.ok(X.some(x => x[0] === m[0] && x[1] === m[1]));
  assert.deepEqual(p, gmmInit(X, 5, 7));
});

test('EM never lowers the log-likelihood', () => {
  const X = blobs3({n: 300, seed: 271});
  for (let K = 1; K <= 5; K++) {
    const {logLik} = gmmEM(X, gmmInit(X, K, 271), 50);
    assert.equal(logLik.length, 51);
    for (let i = 0; i < 50; i++) {
      assert.ok(Number.isFinite(logLik[i + 1]), `K ${K} i ${i}`);
      assert.ok(logLik[i + 1] >= logLik[i] - 1e-9, `K ${K} i ${i}: ${logLik[i]} -> ${logLik[i + 1]}`);
    }
  }
});

test('gmmEM logLik[0] is the starting log-likelihood and the last matches the fit', () => {
  const X = blobs3({n: 300, seed: 271});
  const init = gmmInit(X, 3, 271);
  const r = gmmEM(X, init, 10);
  near(r.logLik[0], gmmLogLik(X, init), 1e-12);
  near(r.logLik[10], gmmLogLik(X, r.params), 1e-12);
  // K = 1 after one step is the plain Gaussian MLE
  const one = gmmEM(X, gmmInit(X, 1, 1), 1).params;
  const g = gauss2Fit(X);
  near(one.mu[0][0], g.mu[0], 1e-12); near(one.S[0][0][1], g.S[0][1], 1e-12);
});

test('EM recovers three blobs', () => {
  const X = blobs3({n: 300, seed: 271});
  const {params} = gmmEM(X, gmmInit(X, 3, 271), 50);
  for (const b of BLOBS) {
    const d = Math.min(...params.mu.map(m => Math.hypot(m[0] - b.mu[0], m[1] - b.mu[1])));
    assert.ok(d < 0.3, `centre ${b.mu}: ${d}`);
  }
  near(params.pi.reduce((a, c) => a + c), 1, 1e-12);
});

test('gmmSample is reproducible and has the fitted mean', () => {
  const X = blobs3({n: 300, seed: 271});
  const {params} = gmmEM(X, gmmInit(X, 3, 271), 50);
  const A = gmmSample(params, 20000, 9);
  assert.deepEqual(A, gmmSample(params, 20000, 9));
  for (const k of [0, 1]) {
    const want = params.pi.reduce((s, p, j) => s + p * params.mu[j][k], 0);
    const got = A.reduce((s, p) => s + p[k], 0) / A.length;
    near(got, want, 0.05, `axis ${k}`);
  }
});

test('schedule: abar is the running product and abar[T] < 0.01', () => {
  const s = ddpmSchedule(50);
  assert.equal(s.beta.length, 51);
  assert.equal(s.abar[0], 1);
  near(s.beta[1], 1e-3, 1e-15); near(s.beta[50], 0.2, 1e-15);
  near(s.beta[2] - s.beta[1], (0.2 - 1e-3) / 49, 1e-15);
  let p = 1;
  for (let t = 1; t <= 50; t++) {
    near(s.alpha[t], 1 - s.beta[t], 1e-15);
    p *= s.alpha[t];
    near(s.abar[t], p, 1e-15);
  }
  assert.ok(s.abar[50] < 0.01, `abar[T] ${s.abar[50]}`);
});

test('qSample has the right mean and variance', () => {
  const s = ddpmSchedule(50), t = 10, x0 = [1.5, -0.8];
  const rng = mulberry32(31), n = 20000;
  const xs = Array.from({length: n}, () => qSample(x0, t, [randn(rng), randn(rng)], s));
  for (const k of [0, 1]) {
    const v = xs.map(p => p[k]);
    const m = v.reduce((a, b) => a + b) / n;
    const s2 = v.reduce((a, b) => a + (b - m) ** 2, 0) / n;
    near(m, Math.sqrt(s.abar[t]) * x0[k], 0.03, `mean ${k}`);
    near(s2, 1 - s.abar[t], 0.03, `variance ${k}`);
  }
});

test('timeFeatures by hand', () => {
  const f = timeFeatures(25, 50);
  assert.equal(f.length, 7);
  near(f[0], 0.5, 1e-15);
  near(f[1], 1, 1e-15); near(f[2], 0, 1e-15);   // sin, cos of pi/2
  near(f[3], 0, 1e-15); near(f[4], -1, 1e-15);  // of pi
  near(f[5], 0, 1e-15); near(f[6], 1, 1e-15);   // of 2 pi
});

test('ddpmStep matches Algorithm 2 by hand', () => {
  const s = ddpmSchedule(50);
  const stub = () => [0.3, -0.2];
  const x = [1, 0.5], z = [0.1, -0.4];
  for (const t of [10, 1]) {
    const b = s.beta[t], a = 1 - b;
    let ab = 1;
    for (let u = 1; u <= t; u++) ab *= s.alpha[u];
    const sig = t > 1 ? Math.sqrt(b) : 0;
    const want = [0, 1].map(k => (x[k] - b / Math.sqrt(1 - ab) * stub()[k]) / Math.sqrt(a) + sig * z[k]);
    const got = ddpmStep(stub, x, t, z, s);
    near(got[0], want[0], 1e-12); near(got[1], want[1], 1e-12);
  }
  // a real net and a function wrapping it agree
  const net = denoiserInit({hidden: [8, 8], seed: 3});
  const f = (p, t) => denoiserForward(net, p, timeFeatures(t, 50));
  assert.deepEqual(ddpmStep(net, x, 7, z, s), ddpmStep(f, x, 7, z, s));
});

test('inputFeatures by hand', () => {
  const f = inputFeatures(Math.PI / 2, 0);
  const want = [Math.PI / 2, 0,
    1, 0, 0, 1,    // k = 1: sin(pi/2), cos(pi/2), sin 0, cos 0
    0, -1, 0, 1,   // k = 2: sin(pi), cos(pi)
    0, 1, 0, 1];   // k = 4: sin(2 pi), cos(2 pi)
  assert.equal(f.length, 14);
  want.forEach((w, i) => near(f[i], w, 1e-15, `feature ${i}`));
  // the network's first layer sees exactly these, then the time features
  const net = denoiserInit({hidden: [3], seed: 1});
  assert.equal(net.sizes[0], 14 + 7);
});

const smallBatch = () => {
  const rng = mulberry32(77);
  return Array.from({length: 3}, (_, i) => ({
    x: [randn(rng), randn(rng)], tf: timeFeatures(5 + 15 * i, 50), eps: [randn(rng), randn(rng)]
  }));
};

test('denoiser gradient check', () => {
  const net = denoiserInit({hidden: [6, 5], seed: 41});
  const batch = smallBatch();
  // no ReLU kinks within 1e-4, so central differences are valid
  for (const r of batch) {
    for (const z of denoiserPreacts(net, r.x, r.tf).flat()) assert.ok(Math.abs(z) > 1e-4, `kink ${z}`);
  }
  const g = denoiserGrad(net, batch);
  const lossAt = p => denoiserGrad(denoiserUnflatten(net, Float64Array.from(p)), batch).loss;
  const rel = gradCheck(lossAt, Array.from(denoiserFlatten(net)), Array.from(g.grad));
  assert.ok(rel < 1e-6, `rel ${rel}`);
  // loss is the batch mean of the squared error summed over both axes
  const want = batch.reduce((s, r) => {
    const o = denoiserForward(net, r.x, r.tf);
    return s + (o[0] - r.eps[0]) ** 2 + (o[1] - r.eps[1]) ** 2;
  }, 0) / 3;
  near(g.loss, want, 1e-12);
});

test('denoiser init: He scale, zero biases, flatten round trip', () => {
  const net = denoiserInit({hidden: [64, 64], seed: 5});
  assert.deepEqual(net.sizes, [21, 64, 64, 2]);
  const v = denoiserFlatten(net);
  assert.ok(v instanceof Float64Array);
  assert.equal(v.length, 21 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2);
  assert.deepEqual(denoiserFlatten(denoiserUnflatten(net, v)), v);
  v[0] = 99;
  assert.notEqual(denoiserFlatten(net)[0], 99);
  // second layer weights: variance near 2 / 64
  const W2 = net.params.slice(21 * 64 + 64, 21 * 64 + 64 + 64 * 64);
  const m2 = W2.reduce((s, w) => s + w * w, 0) / W2.length;
  near(m2, 2 / 64, 0.004);
  assert.ok(net.params.slice(21 * 64, 21 * 64 + 64).every(b => b === 0));
});

const spiralSetup = () => ({X: spiral2D({n: 400, seed: 305}), sched: ddpmSchedule(50)});

test('ddpm trainer deterministic', () => {
  const {X, sched} = spiralSetup();
  const mk = () => ddpmTrainer({net: denoiserInit({hidden: [16, 16], seed: 9}), X, sched,
    batch: 32, hyper: {eta: 3e-3}, seed: 11});
  const a = mk().step(120), b = mk().step(120);
  assert.equal(a.iter, 120);
  assert.deepEqual(a.batchLoss, b.batchLoss);
  assert.deepEqual(a.evalLoss, b.evalLoss);
  assert.equal(a.evalLoss.length, 2); // at 0 and 100
  assert.deepEqual(denoiserFlatten(a.net), denoiserFlatten(b.net));
  assert.equal(a.diverged, false);
});

test('reset repeats', () => {
  const {X, sched} = spiralSetup();
  const tr = ddpmTrainer({net: denoiserInit({hidden: [16, 16], seed: 9}), X, sched,
    batch: 32, hyper: {eta: 3e-3}, seed: 11});
  const first = tr.step(50);
  const loss = first.batchLoss.slice(), w = denoiserFlatten(first.net);
  const again = tr.reset();
  assert.equal(again.iter, 0);
  tr.step(50);
  assert.deepEqual(tr.state.batchLoss, loss);
  assert.deepEqual(denoiserFlatten(tr.state.net), w);
});

test('ddpmSample is seeded, keeps the path and starts from noise', () => {
  const sched = ddpmSchedule(50);
  const net = denoiserInit({hidden: [8, 8], seed: 2});
  const a = ddpmSample(net, 20, sched, 4, {keepPath: true});
  assert.deepEqual(a, ddpmSample(net, 20, sched, 4, {keepPath: true}));
  assert.equal(a.path.length, 51);
  assert.deepEqual(a.path[50], a.X);
  const rng = mulberry32(4);
  assert.deepEqual(a.path[0][0], [randn(rng), randn(rng)]);
  assert.equal(ddpmSample(net, 20, sched, 4).path, undefined);
});

test('nearestDist by hand', () => {
  const A = [[0, 0], [3, 4]], B = [[0, 1], [3, 0]];
  // from A: 1 and 4; from B: 1 and 3
  near(nearestDist(A, B), 2.5, 1e-15);
  near(nearestDist(B, A), 2, 1e-15);
  assert.equal(nearestDist(A, A), 0);
});

test('trained diffusion samples sit far closer to the spiral than noise does', () => {
  const X = spiral2D(LESSON_SPIRAL), sched = ddpmSchedule(50);
  assert.deepEqual(LESSON_SPIRAL, {n: 400, seed: 305, turns: 1, noise: 0.1});
  const tr = ddpmTrainer({net: denoiserInit({hidden: [64, 64], seed: 311}), X, sched,
    batch: 128, hyper: {eta: 3e-3}, seed: 312});
  const s = tr.step(6000);
  assert.equal(s.diverged, false);
  const {X: S, path} = ddpmSample(s.net, 400, sched, 313, {keepPath: true});
  const noise = path[0]; // the very points the samples started from
  const prec = nearestDist(S, X), cov = nearestDist(X, S);
  const prec0 = nearestDist(noise, X), cov0 = nearestDist(X, noise);
  // measured: samples 0.079 / 0.062, noise 0.222 / 0.112
  assert.ok(prec < 0.6 * prec0, `precision ${prec} vs noise ${prec0}`);
  assert.ok(cov < 0.8 * cov0, `coverage ${cov} vs noise ${cov0}`);
});
