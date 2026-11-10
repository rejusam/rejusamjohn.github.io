import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  xorData, circlesData, spiralData, ACT, mlpInit, mlpForward, bceLogit,
  mlpBackward, batchGrad, predictProb, accuracyNet, flatten, unflatten,
  gradCheck, optInit, optStep, trainer
} from '../assets/nn.js';
import {mulberry32, randn} from '../assets/maths.js';
import {sigmoid} from '../assets/classify.js';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);

test('generators are reproducible', () => {
  assert.deepEqual(xorData({n: 50, seed: 1}), xorData({n: 50, seed: 1}));
  assert.deepEqual(circlesData({n: 50, seed: 2}), circlesData({n: 50, seed: 2}));
  assert.deepEqual(spiralData({n: 50, seed: 3}), spiralData({n: 50, seed: 3}));
  assert.notDeepEqual(xorData({n: 50, seed: 1}), xorData({n: 50, seed: 2}));
  // xor labels follow the corner rule, recomputed from the same stream
  const {X, y} = xorData({n: 40, sd: 0.35, seed: 9});
  const rng = mulberry32(9);
  for (let i = 0; i < 40; i++) {
    const c = Math.floor(rng() * 4);
    const sx = c % 2 ? 1 : -1, sy = c >> 1 ? 1 : -1;
    const px = sx + 0.35 * randn(rng), py = sy + 0.35 * randn(rng);
    assert.deepEqual(X[i], [px, py]);
    assert.equal(y[i], sx !== sy ? 1 : 0);
  }
  const circ = circlesData({n: 60, seed: 4});
  assert.equal(circ.X.length, 60);
  assert.ok(circ.y.slice(0, 30).every(v => v === 0));
  assert.ok(circ.y.slice(30).every(v => v === 1));
  const sp = spiralData({n: 60, seed: 5});
  assert.equal(sp.X.length, 60);
  assert.equal(sp.y.filter(v => v === 1).length, 30);
});

test('bceLogit is stable', () => {
  for (const z of [800, -800]) for (const y of [0, 1]) assert.ok(Number.isFinite(bceLogit(y, z)));
  near(bceLogit(1, 0), Math.LN2, 1e-15);
  for (const z of [-3, 0.5, 4]) near(bceLogit(1, z), -Math.log(sigmoid(z)), 1e-12);
});

test('mlpForward by hand', () => {
  const net = mlpInit([2, 2, 1], {hidden: 'tanh', seed: 1});
  net.W[0] = [[0.5, -0.3], [0.2, 0.8]];
  net.b[0] = [0.1, -0.2];
  net.W[1] = [[1.5, -0.7]];
  net.b[1] = [0.05];
  const x = [0.4, -1.2];
  const h1 = Math.tanh(0.5 * 0.4 + -0.3 * -1.2 + 0.1);
  const h2 = Math.tanh(0.2 * 0.4 + 0.8 * -1.2 - 0.2);
  const out = 1.5 * h1 - 0.7 * h2 + 0.05;
  const r = mlpForward(net, x);
  near(r.out[0], out, 1e-15);
  assert.deepEqual(r.as[0], x);
  assert.equal(r.zs.length, 2);
  assert.equal(r.as.length, 3);
});

const randomX = (seed, n = 2) => { const r = mulberry32(seed); return Array.from({length: n}, () => randn(r)); };

test('mlpBackward passes a gradient check', () => {
  for (const hidden of ['sigmoid', 'tanh']) {
    for (const loss of ['bce', 'mse']) {
      const net = mlpInit([2, 3, 3, 1], {hidden, seed: 7});
      const x = randomX(11), y = 1;
      const {dW, db} = mlpBackward(net, x, y, {loss});
      const grad = flatten({...net, W: dW, b: db});
      const f = p => mlpBackward(unflatten(net, p), x, y, {loss}).loss;
      assert.ok(gradCheck(f, flatten(net), grad) < 1e-6, `${hidden}/${loss}`);
    }
  }
});

test('gradCheck passes for relu away from kinks', () => {
  const net = mlpInit([2, 3, 3, 1], {hidden: 'relu', init: 'he', seed: 8});
  let x, s = 20;
  for (;; s++) {
    x = randomX(s);
    const {zs} = mlpForward(net, x);
    if (zs.slice(0, -1).every(z => z.every(v => Math.abs(v) >= 1e-4))) break;
  }
  const {dW, db} = mlpBackward(net, x, 1);
  const grad = flatten({...net, W: dW, b: db});
  const f = p => mlpBackward(unflatten(net, p), x, 1).loss;
  assert.ok(gradCheck(f, flatten(net), grad) < 1e-6);
});

test('batchGrad adds weight decay to weights only', () => {
  const net = mlpInit([2, 3, 1], {seed: 5});
  const {X, y} = xorData({n: 12, seed: 6});
  const idx = [0, 3, 5, 7, 11];
  const g0 = batchGrad(net, X, y, idx, {lambda: 0});
  const g1 = batchGrad(net, X, y, idx, {lambda: 0.3});
  for (let l = 0; l < net.W.length; l++) {
    for (let j = 0; j < net.W[l].length; j++) {
      for (let i = 0; i < net.W[l][j].length; i++) {
        near(g1.dW[l][j][i] - g0.dW[l][j][i], 0.3 * net.W[l][j][i], 1e-15);
      }
    }
  }
  assert.deepEqual(g1.db, g0.db);
  assert.equal(g1.loss, g0.loss);
  // mean of per-example gradients
  let s = 0;
  for (const i of idx) s += mlpBackward(net, X[i], y[i]).dW[0][1][0];
  near(g0.dW[0][1][0], s / idx.length, 1e-12);
});

test('flatten round-trips', () => {
  const net = mlpInit([2, 4, 3, 1], {seed: 3});
  assert.deepEqual(unflatten(net, flatten(net)), net);
  assert.equal(flatten(net).length, 2 * 4 + 4 + 4 * 3 + 3 + 3 + 1);
  assert.equal(predictProb(net, [0, 0]), sigmoid(mlpForward(net, [0, 0]).out[0]));
});

test('adam first step by hand', () => {
  const params = [1], grad = [0.5];
  const state = optInit('adam', 1);
  const before = JSON.stringify({params, grad, state});
  const r = optStep('adam', {eta: 0.1}, state, params, grad);
  near(r.params[0], 1 - 0.1 * 0.5 / (0.5 + 1e-8), 1e-12);
  assert.equal(JSON.stringify({params, grad, state}), before);
});

test('momentum and rmsprop by hand', () => {
  const gs = [[0.4], [-0.2]];
  // momentum: v = 0.9 v + g; p - eta v
  let st = optInit('momentum', 1), p = [1];
  let v = 0, pe = 1;
  for (const g of gs) {
    ({params: p, state: st} = optStep('momentum', {eta: 0.1, beta: 0.9}, st, p, g));
    v = 0.9 * v + g[0]; pe -= 0.1 * v;
    near(p[0], pe, 1e-14);
  }
  // rmsprop
  st = optInit('rmsprop', 1); p = [1];
  let s = 0; pe = 1;
  for (const g of gs) {
    ({params: p, state: st} = optStep('rmsprop', {eta: 0.01, beta2: 0.9}, st, p, g));
    s = 0.9 * s + 0.1 * g[0] * g[0]; pe -= 0.01 * g[0] / (Math.sqrt(s) + 1e-8);
    near(p[0], pe, 1e-14);
  }
  // sgd
  const r = optStep('sgd', {eta: 0.5}, optInit('sgd', 1), [2], [0.4]);
  near(r.params[0], 1.8, 1e-15);
});

test('optimiser defaults match the brief', () => {
  const p = [1, -2], g = [0.3, -0.7];
  const explicit = {eta: 0.1, beta: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8};
  for (const kind of ['momentum', 'rmsprop', 'adam']) {
    const a = optStep(kind, {eta: 0.1}, optInit(kind, 2), p, g);
    const b = optStep(kind, explicit, optInit(kind, 2), p, g);
    assert.deepEqual(a, b, kind);
  }
});

const makeTrainer = (extra = {}) => {
  const {X, y} = xorData({n: 200, seed: 3});
  const net = mlpInit([2, 4, 1], {hidden: 'tanh', seed: 4});
  return trainer({net, X, y, batch: 20, kind: 'adam', hyper: {eta: 0.05}, seed: 4, ...extra});
};

test('trainer is deterministic', () => {
  const a = makeTrainer(), b = makeTrainer();
  a.step(300); b.step(300);
  assert.deepEqual(a.state.batchLoss, b.state.batchLoss);
  assert.equal(a.state.batchLoss.length, 300);
});

test('trainer learns xor', () => {
  const {X, y} = xorData({n: 200, seed: 3});
  const t = makeTrainer();
  const st = t.step(100 * 10);
  assert.equal(st.epoch, 100);
  assert.equal(st.epochLoss.length, 101);
  assert.ok(accuracyNet(st.net, X, y) >= 0.95);
  assert.ok(st.epochLoss.at(-1) < st.epochLoss[0]);
});

test('trainer flags divergence and keeps finite params', () => {
  const {X, y} = circlesData({n: 100, seed: 2});
  // tanh units saturate, so their loss stays bounded; ReLU steps can blow up
  const net = mlpInit([2, 4, 1], {hidden: 'relu', init: 'he', seed: 1});
  const t = trainer({net, X, y, batch: 10, kind: 'sgd', hyper: {eta: 1e3}, seed: 1});
  const st = t.step(500);
  assert.equal(st.diverged, true);
  assert.ok(flatten(st.net).every(Number.isFinite));
  const n = st.iter;
  t.step(10);
  assert.equal(t.state.iter, n);
});

test('reset repeats the run', () => {
  const t = makeTrainer();
  t.step(50);
  const first = t.state.batchLoss.slice();
  t.reset();
  assert.equal(t.state.iter, 0);
  t.step(50);
  assert.deepEqual(t.state.batchLoss, first);
});

test('mlpInit sets bias zero and scales by init rule', () => {
  const a = mlpInit([2, 5, 1], {seed: 1}), b = mlpInit([2, 5, 1], {seed: 1});
  assert.deepEqual(a, b);
  assert.ok(a.b.flat().every(v => v === 0));
  const c = mlpInit([2, 5, 1], {seed: 1, init: 0});
  assert.ok(c.W.flat(2).every(v => v === 0));
});
