import {test} from 'node:test';
import assert from 'node:assert/strict';
import {echoData, rnnInit, rnnForward, bptt, rnnFlatten, rnnUnflatten} from '../assets/rnn.js';
import {gradCheck, bceLogit} from '../assets/nn.js';
import {sigmoid} from '../assets/classify.js';

const near = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b} (tol ${tol})`);

test('echo targets are the bit k steps back', () => {
  for (const k of [1, 3]) {
    const {seqs, targets, mask} = echoData({n: 20, T: 12, k, seed: 5});
    assert.equal(seqs.length, 20);
    seqs.forEach((xs, i) => {
      assert.equal(xs.length, 12);
      assert.ok(xs.every(b => b === 0 || b === 1));
      for (let t = 0; t < 12; t++) {
        assert.equal(mask[i][t], t >= k ? 1 : 0);
        if (t >= k) assert.equal(targets[i][t], xs[t - k]);
      }
    });
  }
  assert.deepEqual(echoData({n: 5, k: 2, seed: 1}), echoData({n: 5, k: 2, seed: 1}));
});

test('rnnForward by hand', () => {
  const net = {nh: 2, Wx: [0.5, -1], Wh: [[0.2, -0.3], [0.1, 0.4]], bh: [0.1, -0.2], Wy: [1, -2], by: 0.3};
  const xs = [1, 0, 1];
  const {hs, zs} = rnnForward(net, xs);
  let h = [0, 0];
  assert.deepEqual(hs[0], [0, 0]);
  xs.forEach((x, t) => {
    const n = [
      Math.tanh(0.5 * x + 0.2 * h[0] - 0.3 * h[1] + 0.1),
      Math.tanh(-1 * x + 0.1 * h[0] + 0.4 * h[1] - 0.2)
    ];
    near(hs[t + 1][0], n[0], 1e-15); near(hs[t + 1][1], n[1], 1e-15);
    near(zs[t], n[0] - 2 * n[1] + 0.3, 1e-15);
    h = n;
  });
  assert.equal(hs.length, 4);
  assert.equal(zs.length, 3);
});

test('bptt passes a gradient check', () => {
  const net = rnnInit({nh: 4, seed: 21});
  const {seqs, targets, mask} = echoData({n: 1, T: 12, k: 2, seed: 21});
  const {grads, loss} = bptt(net, seqs[0], targets[0], mask[0]);
  assert.ok(Number.isFinite(loss) && loss > 0);
  const f = v => bptt(rnnUnflatten(net, v), seqs[0], targets[0], mask[0]).loss;
  const worst = gradCheck(f, rnnFlatten(net), rnnFlatten({nh: 4, ...grads}));
  assert.ok(worst < 1e-6, `worst ${worst}`);
});

test('all-zero mask gives zero loss and zero grads', () => {
  const net = rnnInit({nh: 3, seed: 2});
  const r = bptt(net, [1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]);
  assert.equal(r.loss, 0);
  assert.ok(rnnFlatten({nh: 3, ...r.grads}).every(v => v === 0));
  assert.ok(r.lagNorms.every(Number.isFinite));
});

test('flatten and unflatten round trip', () => {
  const net = rnnInit({nh: 3, seed: 8});
  assert.deepEqual(rnnUnflatten(net, rnnFlatten(net)), net);
});

test('lag norms by hand at j = 0 and 1', () => {
  const net = rnnInit({nh: 4, seed: 21});
  const {seqs, targets, mask} = echoData({n: 1, T: 12, k: 2, seed: 21});
  const xs = seqs[0], ys = targets[0], T = 12;
  const {lagNorms} = bptt(net, xs, ys, mask[0]);
  const {hs, zs} = rnnForward(net, xs);
  assert.equal(lagNorms.length, T);
  const g0 = net.Wy.map(w => (sigmoid(zs[T - 1]) - ys[T - 1]) * w);
  near(lagNorms[0], Math.hypot(...g0), 1e-12);
  const a = g0.map((v, i) => v * (1 - hs[T][i] ** 2));
  const g1 = net.Wh[0].map((_, c) => net.Wh.reduce((s, row, i) => s + row[c] * a[i], 0));
  near(lagNorms[1], Math.hypot(...g1), 1e-12);
  // the loss at T alone is bceLogit(y_T, z_T); sanity that it is finite
  assert.ok(Number.isFinite(bceLogit(ys[T - 1], zs[T - 1])));
});

test('small recurrent weights make lag norms shrink', () => {
  const net = rnnInit({nh: 8, seed: 3, scale: 0.5});
  const {seqs, targets, mask} = echoData({n: 1, T: 12, k: 3, seed: 3});
  const {lagNorms} = bptt(net, seqs[0], targets[0], mask[0]);
  assert.ok(lagNorms[8] < lagNorms[0] * 1e-2, `${lagNorms[8]} vs ${lagNorms[0]}`);
});
