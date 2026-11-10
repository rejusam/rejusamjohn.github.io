import test from 'node:test';
import assert from 'node:assert/strict';
import {attnInit, attnForward, attnLoss, attnBackward, attnFlatten, attnUnflatten,
  attnParamCount, layerNorm, scaledDot, attnTrainer} from '../assets/attn.js';
import {gradCheck} from '../assets/nn.js';
import {softmax} from '../assets/text.js';
import {echoData} from '../assets/rnn.js';
import {mulberry32, randn} from '../assets/maths.js';

const matVec = (W, rows, cols, x) =>
  Array.from({length: rows}, (_, r) => x.reduce((s, xi, c) => s + W[r * cols + c] * xi, 0));

test('attention by hand', () => {
  const m = attnInit({vocab: 3, len: 3, d: 2, heads: 1, ln: false, mlp: 0, seed: 1});
  const E = [0.5, -1, 1, 0.25, -0.5, 0.75];
  const P = [0.1, 0.2, -0.3, 0.1, 0, -0.2];
  const Wq = [1, 0.5, -0.5, 1], Wk = [0.8, -0.2, 0.3, 1.1], Wv = [1, -1, 0.5, 2];
  const Wo = [0.7, 0.1, -0.4, 0.9], Wout = [1, 0, 0, 1, 1, -1], bout = [0.1, -0.2, 0.3];
  m.theta.set([...E, ...P, ...Wq, ...Wk, ...Wv, ...Wo, ...Wout, ...bout]);
  const ids = [2, 0, 1];
  const x = ids.map((id, t) => [E[id * 2] + P[t * 2], E[id * 2 + 1] + P[t * 2 + 1]]);
  const q = x.map(v => matVec(Wq, 2, 2, v)), k = x.map(v => matVec(Wk, 2, 2, v));
  const v = x.map(u => matVec(Wv, 2, 2, u));
  const A = q.map(qt => softmax(k.map(ks => (qt[0] * ks[0] + qt[1] * ks[1]) / Math.SQRT2)));
  const o = A.map(row => [0, 1].map(j => row.reduce((s, a, sI) => s + a * v[sI][j], 0)));
  const h = x.map((xt, t) => matVec(Wo, 2, 2, o[t]).map((z, j) => z + xt[j]));
  const logits = h.map(ht => matVec(Wout, 3, 2, ht).map((z, j) => z + bout[j]));
  const f = attnForward(m, ids);
  for (let t = 0; t < 3; t++) {
    for (let s = 0; s < 3; s++) assert.ok(Math.abs(f.A[0][0][t][s] - A[t][s]) < 1e-14);
    for (let j = 0; j < 3; j++) assert.ok(Math.abs(f.logits[t][j] - logits[t][j]) < 1e-14);
  }
});

test('rows of A sum to 1 and causal rows are zero above the diagonal', () => {
  const m = attnInit({vocab: 6, len: 7, d: 8, heads: 2, causal: true, mlp: 8, blocks: 2, seed: 4});
  const {A} = attnForward(m, [1, 5, 0, 3, 3, 2, 4]);
  for (const blk of A) {
    for (const head of blk) {
      head.forEach((row, t) => {
        assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12);
        row.forEach((a, s) => {
          if (s > t) assert.equal(a, 0);
          else assert.ok(a > 0);
        });
      });
    }
  }
  const nc = attnInit({vocab: 6, len: 7, d: 8, heads: 2, mlp: 8, seed: 4});
  for (const row of attnForward(nc, [1, 5, 0, 3]).A[0][1]) {
    assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.ok(row.every(a => a > 0));
  }
});

test('scaled scores have variance near 1', () => {
  const rng = mulberry32(17), dk = 64, n = 20000;
  const s = [];
  for (let i = 0; i < n; i++) {
    const q = Array.from({length: dk}, () => randn(rng));
    const k = Array.from({length: dk}, () => randn(rng));
    s.push(scaledDot(q, k));
  }
  const mu = s.reduce((a, b) => a + b, 0) / n;
  const v = s.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (n - 1);
  assert.ok(Math.abs(v - 1) < 0.05, `variance ${v}`);
});

test('attention without positions is permutation equivariant', () => {
  const m = attnInit({vocab: 7, len: 6, d: 8, heads: 2, mlp: 12, seed: 8});
  m.P.fill(0);
  const ids = [3, 0, 6, 2, 2, 5], perm = [4, 1, 5, 0, 3, 2];
  const a = attnForward(m, ids).logits;
  const b = attnForward(m, perm.map(i => ids[i])).logits;
  perm.forEach((src, t) => {
    for (let j = 0; j < 7; j++) assert.ok(Math.abs(b[t][j] - a[src][j]) < 1e-12);
  });
});

test('layerNorm output has mean 0 and population variance near 1', () => {
  const rng = mulberry32(3);
  for (const d of [2, 8, 32]) {
    const x = Array.from({length: d}, () => 3 + 2 * randn(rng));
    const {y, mu, inv} = layerNorm(x, new Array(d).fill(1), new Array(d).fill(0));
    const xm = x.reduce((a, b) => a + b, 0) / d;
    const xv = x.reduce((a, b) => a + (b - xm) * (b - xm), 0) / d;
    assert.ok(Math.abs(mu - xm) < 1e-12);
    assert.ok(Math.abs(inv - 1 / Math.sqrt(xv + 1e-5)) < 1e-12);
    const ym = y.reduce((a, b) => a + b, 0) / d;
    const yv = y.reduce((a, b) => a + (b - ym) * (b - ym), 0) / d;
    assert.ok(Math.abs(ym) < 1e-9);
    assert.ok(Math.abs(yv - xv / (xv + 1e-5)) < 1e-9);
  }
});

// ---- gradient checks ----------------------------------------------------------------------

function checkGrad(cfg, ids, targets, mask) {
  const m = attnInit(cfg);
  const {cache} = attnForward(m, ids);
  for (const b of cache.blocks) {
    if (b.hpre) assert.ok(b.hpre.every(z => Math.abs(z) > 1e-4), 'a ReLU input sits on its kink');
  }
  const {grads, loss} = attnBackward(m, ids, targets, mask);
  assert.ok(Math.abs(loss - attnLoss(m, ids, targets, mask)) < 1e-14);
  const f = v => attnLoss(attnUnflatten(m, v), ids, targets, mask);
  return gradCheck(f, attnFlatten(m), Array.from(grads.theta));
}

test('gradient check: echo (causal, 1 head, no LN, no MLP, logit)', () => {
  const {seqs, targets, mask} = echoData({n: 1, T: 12, k: 3, seed: 31});
  const err = checkGrad({vocab: 2, len: 12, d: 4, heads: 1, causal: true, ln: false, mlp: 0,
    out: 'logit', seed: 231}, seqs[0], targets[0], mask[0]);
  assert.ok(err < 1e-6, `rel err ${err}`);
});

test('gradient check: reversal (non-causal, 2 heads, LN, MLP 16, softmax)', () => {
  const ids = [3, 1, 4, 1, 5, 9, 2, 6];
  const err = checkGrad({vocab: 10, len: 8, d: 8, heads: 2, ln: true, mlp: 16, seed: 232},
    ids, ids.slice().reverse());
  assert.ok(err < 1e-6, `rel err ${err}`);
});

test('gradient check: language model (causal, 2 heads, LN, MLP 16, 2 blocks, softmax)', () => {
  const text = [7, 0, 14, 3, 27, 9, 12, 0, 21, 5, 18, 2, 11];
  const err = checkGrad({vocab: 28, len: 12, d: 8, heads: 2, causal: true, ln: true, mlp: 16,
    blocks: 2, seed: 233}, text.slice(0, 12), text.slice(1));
  assert.ok(err < 1e-6, `rel err ${err}`);
});

test('cross-entropy finite at ±800 logits', () => {
  const m = attnInit({vocab: 3, len: 4, d: 4, seed: 5});
  m.Wout.fill(0);
  m.bout.set([800, -800, 0]);
  for (const target of [0, 1, 2]) {
    const {loss, grads} = attnBackward(m, [0, 1, 2], [target, target, target]);
    assert.ok(Number.isFinite(loss));
    assert.ok(grads.theta.every(Number.isFinite));
  }
  assert.ok(Math.abs(attnLoss(m, [0], [1]) - 1600) < 1e-9);
  assert.ok(attnLoss(m, [0], [0]) < 1e-300);
  const b = attnInit({vocab: 2, len: 2, d: 4, out: 'logit', seed: 5});
  b.Wout.fill(0);
  for (const [z, y, want] of [[800, 0, 800], [-800, 1, 800], [800, 1, 0], [-800, 0, 0]]) {
    b.bout[0] = z;
    const {loss, grads} = attnBackward(b, [0, 1], [y, y]);
    assert.ok(Math.abs(loss - want) < 1e-9);
    assert.ok(grads.theta.every(Number.isFinite));
  }
});

test('mask leaves out positions and an all-zero mask gives loss 0', () => {
  const m = attnInit({vocab: 2, len: 5, d: 4, causal: true, out: 'logit', seed: 6});
  const ids = [1, 0, 1, 1, 0], y = [0, 1, 1, 0, 1];
  const full = attnForward(m, ids).logits;
  const bce = (t, z) => Math.max(z, 0) - t * z + Math.log1p(Math.exp(-Math.abs(z)));
  const want = (bce(y[2], full[2]) + bce(y[4], full[4])) / 2;
  assert.ok(Math.abs(attnLoss(m, ids, y, [0, 0, 1, 0, 1]) - want) < 1e-14);
  const {loss, grads} = attnBackward(m, ids, y, [0, 0, 0, 0, 0]);
  assert.equal(loss, 0);
  assert.ok(grads.theta.every(g => g === 0));
});

// ---- trainer ------------------------------------------------------------------------------

const echoCfg = {vocab: 2, len: 12, d: 8, heads: 1, causal: true, ln: false, mlp: 0,
  out: 'logit', seed: 21};
const echoTrainer = (extra = {}) => attnTrainer({model: attnInit(echoCfg),
  data: echoData({n: 40, T: 12, k: 2, seed: 22}), batch: 8, kind: 'adam',
  hyper: {eta: 0.01}, seed: 9, ...extra});

test('trainer deterministic', () => {
  const a = echoTrainer(), b = echoTrainer();
  a.step(25); b.step(25);
  assert.equal(a.state.epoch, 5);
  assert.deepEqual(attnFlatten(a.state.model), attnFlatten(b.state.model));
  assert.deepEqual(a.state.batchLoss, b.state.batchLoss);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
  assert.ok(a.state.epochLoss[5] < a.state.epochLoss[0]);
});

test('reset repeats the run', () => {
  const t = echoTrainer();
  const data = echoData({n: 40, T: 12, k: 2, seed: 22});
  const first = data.seqs.reduce((s, ids, i) =>
    s + attnLoss(attnInit(echoCfg), ids, data.targets[i], data.mask[i]), 0) / 40;
  assert.equal(t.state.epochLoss[0], first);
  t.step(17);
  const snap = {p: attnFlatten(t.state.model), b: t.state.batchLoss.slice(), e: t.state.epochLoss.slice()};
  t.reset();
  assert.equal(t.state.iter, 0);
  assert.deepEqual(t.state.epochLoss, [first]);
  t.step(17);
  assert.deepEqual(attnFlatten(t.state.model), snap.p);
  assert.deepEqual(t.state.batchLoss, snap.b);
  assert.deepEqual(t.state.epochLoss, snap.e);
});

test('divergence flag', () => {
  const t = attnTrainer({model: attnInit({...echoCfg, ln: true, mlp: 8}),
    data: echoData({n: 40, T: 12, k: 2, seed: 22}), batch: 8, kind: 'sgd',
    hyper: {eta: 1e9}, seed: 9});
  t.step(50);
  assert.ok(t.state.diverged);
  assert.ok(attnFlatten(t.state.model).every(Number.isFinite));
  const n = t.state.iter;
  t.step(5);
  assert.equal(t.state.iter, n);
});

test('sampler mode draws batches from the stream and evaluates on the held-out set', () => {
  const cfg = {vocab: 5, len: 6, d: 8, heads: 2, causal: true, mlp: 8, seed: 41};
  const text = Array.from({length: 200}, (_, i) => (i * 7 + (i >> 3)) % 5);
  const win = (rng, n) => {
    const seqs = [], targets = [];
    for (let i = 0; i < n; i++) {
      const s = Math.floor(rng() * (text.length - 7));
      seqs.push(text.slice(s, s + 6));
      targets.push(text.slice(s + 1, s + 7));
    }
    return {seqs, targets};
  };
  const heldOut = win(mulberry32(99), 10);
  const mk = () => attnTrainer({model: attnInit(cfg), data: {heldOut}, batch: 4,
    sampler: rng => win(rng, 4), evalEvery: 10, hyper: {eta: 0.02}, seed: 7});
  const a = mk(), b = mk();
  a.step(30); b.step(30);
  assert.deepEqual(a.state.evalIter, [0, 10, 20, 30]);
  assert.equal(a.state.evalLoss.length, 4);
  assert.deepEqual(a.state.evalLoss, b.state.evalLoss);
  assert.ok(a.state.evalLoss[3] < a.state.evalLoss[0]);
  const snap = a.state.batchLoss.slice();
  a.reset();
  assert.deepEqual(a.state.evalIter, [0]);
  a.step(30);
  assert.deepEqual(a.state.batchLoss, snap);
});

test('param count equals flatten length', () => {
  const cfgs = [
    {vocab: 2, len: 12, d: 4, causal: true, ln: false, out: 'logit', seed: 1},
    {vocab: 10, len: 8, d: 8, heads: 2, mlp: 16, seed: 1},
    {vocab: 28, len: 32, d: 32, heads: 2, causal: true, mlp: 64, blocks: 2, seed: 1}
  ];
  const formula = ({vocab, len, d, ln = true, mlp = 0, blocks = 1, out}) => {
    const lnP = ln ? 2 * d : 0, o = out === 'logit' ? 1 : vocab;
    const blk = lnP + 4 * d * d + (mlp > 0 ? lnP + 2 * mlp * d + mlp + d : 0);
    return vocab * d + len * d + blocks * blk + lnP + o * d + o;
  };
  for (const cfg of cfgs) {
    const m = attnInit(cfg);
    assert.equal(attnParamCount(m), attnFlatten(m).length);
    assert.equal(attnParamCount(m), formula(cfg));
    assert.deepEqual(attnFlatten(attnUnflatten(m, attnFlatten(m))), attnFlatten(m));
  }
});

test('init draws from one seeded stream; LN gains 1, biases 0', () => {
  const cfg = {vocab: 5, len: 4, d: 4, heads: 2, mlp: 6, seed: 12};
  assert.deepEqual(attnFlatten(attnInit(cfg)), attnFlatten(attnInit(cfg)));
  const m = attnInit(cfg), b = m.blocks[0];
  for (const v of [b.g1, b.g2, m.gf]) assert.ok(v.every(x => x === 1));
  for (const v of [b.b1, b.b2, b.c1, b.c2, m.bf, m.bout]) assert.ok(v.every(x => x === 0));
  const rng = mulberry32(12);
  assert.equal(m.E[0], 0.5 * randn(rng));
});
