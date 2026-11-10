// Convolution numerics for the ML Explorer: cross-correlation, max pooling,
// a seeded shape-image generator and a tiny CNN with backpropagation.
// Pure functions on plain arrays. Images are arrays of rows, img[r][c].
// Convolution here is cross-correlation: the filter is not flipped.
import {mulberry32, randn} from './maths.js';
import {sigmoid} from './classify.js';
import {bceLogit} from './nn.js';

// Output side of a convolution: floor((n + 2p - f) / s) + 1.
export const convOutSize = (n, f, s = 1, p = 0) => Math.floor((n + 2 * p - f) / s) + 1;

// out[i][j] = bias + sum_{a,b} W[a][b] * img[i*s + a - p][j*s + b - p], zero outside the image.
export function conv2d(img, W, {stride = 1, pad = 0, bias = 0} = {}) {
  const n = img.length, f = W.length;
  const m = convOutSize(n, f, stride, pad);
  const out = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < m; j++) {
      let s = bias;
      for (let a = 0; a < f; a++) {
        const r = i * stride + a - pad;
        if (r < 0 || r >= n) continue;
        for (let b = 0; b < f; b++) {
          const c = j * stride + b - pad;
          if (c < 0 || c >= n) continue;
          s += W[a][b] * img[r][c];
        }
      }
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

// Non-overlapping max pooling (stride = size). arg[r][c] = [i, j], the input position of the
// maximum; on ties the first in row-major order wins. Sides must be divisible by size.
export function maxPool(img, size = 2) {
  const n = img.length;
  if (n % size !== 0 || img[0].length % size !== 0) throw new Error('maxPool: side not divisible by size');
  const out = [], arg = [];
  for (let r = 0; r < n / size; r++) {
    const orow = [], arow = [];
    for (let c = 0; c < img[0].length / size; c++) {
      let best = -Infinity, bi = 0, bj = 0;
      for (let a = 0; a < size; a++) for (let b = 0; b < size; b++) {
        const v = img[r * size + a][c * size + b];
        if (v > best) { best = v; bi = r * size + a; bj = c * size + b; }
      }
      orow.push(best); arow.push([bi, bj]);
    }
    out.push(orow); arg.push(arow);
  }
  return {out, arg};
}

// Shape images, size x size. Class alternates: even i is class 0, odd i is class 1.
//   Class 0: a plus sign, the centre pixel and its four neighbours (up, down, left, right) = 1.
//   Class 1: a hollow 3x3 square, the eight pixels around the centre = 1 (centre stays 0).
// Draw order from one mulberry32(seed) stream, per image: (1) the centre row, then (2) the
// centre column, each 1 + floor(u * (size - 2)), uniform over positions that keep the whole 3x3
// shape inside the image; (3) one randn per pixel in row-major order. Pixel = shape (1) or
// background (0) + noise * randn, clipped to [0, 1].
export function shapeImages({n, size = 8, noise = 0.15, seed}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const cr = 1 + Math.floor(rng() * (size - 2));
    const cc = 1 + Math.floor(rng() * (size - 2));
    const img = [];
    for (let r = 0; r < size; r++) {
      const row = [];
      for (let c = 0; c < size; c++) {
        const dr = Math.abs(r - cr), dc = Math.abs(c - cc);
        const on = cls === 0 ? dr + dc === 0 || dr + dc === 1 : Math.max(dr, dc) === 1;
        const v = (on ? 1 : 0) + noise * randn(rng);
        row.push(Math.min(1, Math.max(0, v)));
      }
      img.push(row);
    }
    X.push(img); y.push(cls);
  }
  return {X, y};
}

// ---- tiny CNN: conv (valid, stride 1) -> ReLU -> maxPool 2 -> flatten -> one dense logit ----
// net = {size, f, filters: number[k][f][f], fb: number[k], dw: number[k * pooled^2], db: number}.
// Flatten order (cnnFlatten, and the same for grads): filters (filter, row, column), fb, dw, db.
// The feature vector is the pooled maps concatenated, filter by filter, row-major.

// Weights from one mulberry32(seed) stream in a fixed order: all filter weights (filter, row,
// column) first, then the dense weights. Filters ~ N(0, 2/9) (variance; He for fan-in 9, so
// sd sqrt(2/9)), dense ~ N(0, 1/n_flat) (variance, sd sqrt(1/n_flat)). Biases 0.
export function cnnInit({size = 8, filters = 2, f = 3, seed} = {}) {
  const rng = mulberry32(seed);
  const side = (size - f + 1) / 2;
  const nFlat = filters * side * side;
  const sdF = Math.sqrt(2 / (f * f)), sdD = Math.sqrt(1 / nFlat);
  const F = [];
  for (let k = 0; k < filters; k++) {
    F.push(Array.from({length: f}, () => Array.from({length: f}, () => sdF * randn(rng))));
  }
  const dw = Array.from({length: nFlat}, () => sdD * randn(rng));
  return {size, f, filters: F, fb: new Array(filters).fill(0), dw, db: 0};
}

export function cnnForward(net, img) {
  const maps = net.filters.map((W, k) => conv2d(img, W, {bias: net.fb[k]}));
  const relu = maps.map(m => m.map(row => row.map(v => Math.max(0, v))));
  const pools = relu.map(m => maxPool(m, 2));
  const pooled = pools.map(p => p.out);
  const args = pools.map(p => p.arg);
  const feat = pooled.flat(2);
  let z = net.db;
  for (let i = 0; i < feat.length; i++) z += net.dw[i] * feat[i];
  return {maps, relu, pooled, args, feat, z};
}

// BCE on the logit for label y. grads has the same shape as net's parameters.
export function cnnBackward(net, img, y) {
  const fw = cnnForward(net, img);
  const loss = bceLogit(y, fw.z);
  const dz = sigmoid(fw.z) - y;
  const k = net.filters.length, f = net.f;
  const side = fw.pooled[0].length;
  const grads = {
    size: net.size, f,
    filters: net.filters.map(W => W.map(row => row.map(() => 0))),
    fb: new Array(k).fill(0),
    dw: fw.feat.map(v => dz * v),
    db: dz
  };
  for (let m = 0; m < k; m++) {
    for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) {
      const [i, j] = fw.args[m][r][c];
      if (!(fw.maps[m][i][j] > 0)) continue; // ReLU gate (the max of a pool window is the ReLU of the max)
      const d = dz * net.dw[m * side * side + r * side + c];
      grads.fb[m] += d;
      for (let a = 0; a < f; a++) for (let b = 0; b < f; b++) {
        grads.filters[m][a][b] += d * img[i + a][j + b];
      }
    }
  }
  return {grads, loss};
}

export function cnnFlatten(net) {
  return [...net.filters.flat(2), ...net.fb, ...net.dw, net.db];
}

// Returns a new net with the same size and shape, parameters taken from v in cnnFlatten order.
export function cnnUnflatten(net, v) {
  const f = net.f;
  let p = 0;
  const filters = net.filters.map(W => W.map(row => row.map(() => v[p++])));
  const fb = net.fb.map(() => v[p++]);
  const dw = net.dw.map(() => v[p++]);
  return {size: net.size, f, filters, fb, dw, db: v[p]};
}

export const cnnParamCount = net => cnnFlatten(net).length;

// One hidden layer and a logit: nIn*nHidden + nHidden (biases) + nHidden (output weights) + 1.
export const denseParamCount = (nIn, nHidden) => nIn * nHidden + nHidden + nHidden + 1;
