import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  convOutSize, conv2d, maxPool, shapeImages, cnnInit, cnnForward,
  cnnBackward, cnnFlatten, cnnUnflatten, cnnParamCount, denseParamCount
} from '../assets/conv.js';
import {gradCheck, bceLogit} from '../assets/nn.js';

const grid = (n, fn) => Array.from({length: n}, (_, r) => Array.from({length: n}, (_, c) => fn(r, c)));

test('convOutSize matches conv2d', () => {
  for (const n of [8, 12]) for (const f of [1, 3, 5]) for (const s of [1, 2]) for (const p of [0, 1]) {
    const out = conv2d(grid(n, (r, c) => r + c), grid(f, () => 1), {stride: s, pad: p});
    const m = convOutSize(n, f, s, p);
    assert.equal(out.length, m);
    assert.equal(out[0].length, m);
    assert.equal(m, Math.floor((n + 2 * p - f) / s) + 1);
  }
});

test('conv2d by hand', () => {
  const img = grid(4, (r, c) => 4 * r + c);
  const W = [[1, 0, -1], [1, 0, -1], [1, 0, -1]];
  const out = conv2d(img, W, {});
  // each output is the left column sum minus the right column sum: -2 per row, 3 rows
  assert.deepEqual(out, [[-6, -6], [-6, -6]]);
  // not flipped: a non-symmetric filter reads the patch as written
  const out2 = conv2d(grid(3, (r, c) => 3 * r + c), [[1, 0, 0], [0, 0, 0], [0, 0, 0]], {});
  assert.deepEqual(out2, [[0]]);
  const out3 = conv2d(grid(3, (r, c) => 3 * r + c), [[0, 0, 0], [0, 0, 0], [0, 0, 1]], {});
  assert.deepEqual(out3, [[8]]);
  // bias and zero padding
  const out4 = conv2d([[1, 2], [3, 4]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], {pad: 1, bias: 0.5});
  assert.deepEqual(out4, [[10.5, 10.5], [10.5, 10.5]]);
});

test('maxPool picks maxima and records where', () => {
  const img = [[1, 3, 2, 2], [4, 0, 2, 1], [5, 5, 0, 0], [5, 1, 0, -1]];
  const {out, arg} = maxPool(img, 2);
  assert.deepEqual(out, [[4, 2], [5, 0]]);
  assert.deepEqual(arg, [[[1, 0], [0, 2]], [[2, 0], [2, 2]]]); // ties: first in row-major order
  assert.throws(() => maxPool([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2));
});

test('shapeImages reproducible and in range', () => {
  const a = shapeImages({n: 20, seed: 5});
  assert.deepEqual(a, shapeImages({n: 20, seed: 5}));
  assert.notDeepEqual(a, shapeImages({n: 20, seed: 6}));
  assert.equal(a.X.length, 20);
  a.y.forEach((v, i) => assert.equal(v, i % 2));
  for (const img of a.X) {
    assert.equal(img.length, 8);
    for (const row of img) for (const v of row) assert.ok(v >= 0 && v <= 1);
  }
  const clean = shapeImages({n: 6, noise: 0, seed: 1});
  clean.X.forEach((img, i) => {
    const sum = img.flat().reduce((s, v) => s + v, 0);
    assert.equal(sum, i % 2 ? 8 : 5);
  });
});

function pick(seed) {
  const net = cnnInit({seed});
  const {X, y} = shapeImages({n: 40, seed: 3});
  for (let i = 0; i < X.length; i++) {
    const fw = cnnForward(net, X[i]);
    const pre = fw.maps.flat(2);
    if (pre.some(v => Math.abs(v) < 1e-4)) continue;
    let tie = false;
    for (const m of fw.relu) for (let r = 0; r < 3 && !tie; r++) for (let c = 0; c < 3; c++) {
      const w = [m[2 * r][2 * c], m[2 * r][2 * c + 1], m[2 * r + 1][2 * c], m[2 * r + 1][2 * c + 1]].sort((p, q) => q - p);
      if (w[0] - w[1] < 1e-9) { tie = true; break; }
    }
    if (!tie) return {net, img: X[i], y: y[i], i};
  }
  throw new Error('no clean image');
}

test('cnnBackward passes a gradient check', () => {
  const {net, img, y} = pick(12);
  const {grads, loss} = cnnBackward(net, img, y);
  assert.ok(Number.isFinite(loss));
  const f = v => {
    const z = cnnForward(cnnUnflatten(net, v), img).z;
    return bceLogit(y, z);
  };
  const rel = gradCheck(f, cnnFlatten(net), cnnFlatten(grads));
  assert.ok(rel < 1e-6, `rel ${rel}`);
});

test('forward shapes and flatten round trip', () => {
  const net = cnnInit({seed: 4});
  const {X} = shapeImages({n: 2, seed: 1});
  const fw = cnnForward(net, X[0]);
  assert.equal(fw.maps.length, 2);
  assert.equal(fw.maps[0].length, 6);
  assert.equal(fw.pooled[0].length, 3);
  assert.equal(fw.feat.length, 18);
  assert.ok(Number.isFinite(fw.z));
  const v = cnnFlatten(net);
  assert.deepEqual(cnnFlatten(cnnUnflatten(net, v)), v);
  assert.deepEqual(cnnInit({seed: 4}), net);
});

test('param counts', () => {
  assert.equal(cnnParamCount(cnnInit({})), 2 * 9 + 2 + 18 + 1);
  assert.equal(cnnFlatten(cnnInit({})).length, 39);
  assert.equal(denseParamCount(64, 18), 64 * 18 + 18 + 18 + 1);
});
