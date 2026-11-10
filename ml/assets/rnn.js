// Recurrent network numerics for the ML Explorer: a seeded "echo the bit from
// k steps ago" task and a small tanh RNN with backpropagation through time.
// Pure functions on plain arrays. Inputs x_t are scalar bits, t = 1..T.
import {mulberry32, randn} from './maths.js';
import {sigmoid} from './classify.js';
import {bceLogit} from './nn.js';

// n random bit sequences of length T. The target at step t is the input bit
// k steps earlier; the mask is 1 where such a bit exists (t >= k, zero-based).
export function echoData({n, T = 12, k, seed}) {
  const rng = mulberry32(seed);
  const seqs = [], targets = [], mask = [];
  for (let i = 0; i < n; i++) {
    const xs = [];
    for (let t = 0; t < T; t++) xs.push(rng() < 0.5 ? 1 : 0);
    seqs.push(xs);
    targets.push(xs.map((_, t) => (t >= k ? xs[t - k] : 0)));
    mask.push(xs.map((_, t) => (t >= k ? 1 : 0)));
  }
  return {seqs, targets, mask};
}

// One random stream, order Wx, Wh (row-major), Wy. Wh ~ N(0, scale^2/nh).
export function rnnInit({nh = 8, seed, scale = 1}) {
  const rng = mulberry32(seed);
  const Wx = Array.from({length: nh}, () => randn(rng));
  const sh = scale / Math.sqrt(nh);
  const Wh = Array.from({length: nh}, () => Array.from({length: nh}, () => sh * randn(rng)));
  const Wy = Array.from({length: nh}, () => randn(rng) / Math.sqrt(nh));
  return {nh, Wx, Wh, bh: new Array(nh).fill(0), Wy, by: 0};
}

// h_0 = 0; h_t = tanh(Wx x_t + Wh h_{t-1} + bh); z_t = Wy . h_t + by (logit).
// hs has T + 1 entries (hs[0] = h_0); zs has T entries (zs[t] pairs with hs[t + 1]).
export function rnnForward(net, xs) {
  const {nh, Wx, Wh, bh, Wy, by} = net;
  const hs = [new Array(nh).fill(0)], zs = [];
  for (let t = 0; t < xs.length; t++) {
    const prev = hs[t], h = new Array(nh);
    for (let i = 0; i < nh; i++) {
      let s = Wx[i] * xs[t] + bh[i];
      for (let j = 0; j < nh; j++) s += Wh[i][j] * prev[j];
      h[i] = Math.tanh(s);
    }
    hs.push(h);
    let z = by;
    for (let i = 0; i < nh; i++) z += Wy[i] * h[i];
    zs.push(z);
  }
  return {hs, zs};
}

// Backpropagation through time for one sequence. Loss is the mean binary
// cross-entropy over masked steps (0 with zero grads if nothing is masked).
// grads has the shape of net (without nh).
//
// lagNorms[j] = || dL_T / dh_{T-j} || for j = 0..T-1, where L_T is the
// per-step cross-entropy at the final step T alone (not divided by the mask
// count, and computed even if the mask is 0 there). h_{T-j} is the hidden
// state j steps before the last; h_0 is not included, so lagNorms has length T.
// Recursion: g_0 = (sigma(z_T) - y_T) Wy, g_{j+1} = Wh^T ((1 - h_{T-j}^2) * g_j).
export function bptt(net, xs, ys, mask) {
  const {nh, Wh, Wy} = net;
  const T = xs.length;
  const {hs, zs} = rnnForward(net, xs);
  let M = 0;
  for (let t = 0; t < T; t++) M += mask[t] ? 1 : 0;
  const grads = {
    Wx: new Array(nh).fill(0),
    Wh: Array.from({length: nh}, () => new Array(nh).fill(0)),
    bh: new Array(nh).fill(0), Wy: new Array(nh).fill(0), by: 0
  };
  let loss = 0;
  if (M > 0) {
    let dhNext = new Array(nh).fill(0); // gradient reaching h_t from step t+1
    for (let t = T - 1; t >= 0; t--) {
      const h = hs[t + 1], hp = hs[t];
      const m = mask[t] ? 1 : 0;
      const dz = m ? (sigmoid(zs[t]) - ys[t]) / M : 0;
      if (m) loss += bceLogit(ys[t], zs[t]) / M;
      const da = new Array(nh);
      for (let i = 0; i < nh; i++) {
        grads.Wy[i] += dz * h[i];
        da[i] = (Wy[i] * dz + dhNext[i]) * (1 - h[i] * h[i]);
      }
      grads.by += dz;
      for (let i = 0; i < nh; i++) {
        grads.Wx[i] += da[i] * xs[t];
        grads.bh[i] += da[i];
        for (let j = 0; j < nh; j++) grads.Wh[i][j] += da[i] * hp[j];
      }
      dhNext = new Array(nh).fill(0);
      for (let j = 0; j < nh; j++) for (let i = 0; i < nh; i++) dhNext[j] += Wh[i][j] * da[i];
    }
  }
  const lagNorms = [];
  let g = Wy.map(w => (sigmoid(zs[T - 1]) - ys[T - 1]) * w);
  for (let j = 0; j < T; j++) {
    lagNorms.push(Math.hypot(...g));
    const h = hs[T - j];
    const a = g.map((v, i) => v * (1 - h[i] * h[i]));
    g = new Array(nh).fill(0);
    for (let c = 0; c < nh; c++) for (let i = 0; i < nh; i++) g[c] += Wh[i][c] * a[i];
  }
  return {grads, loss, lagNorms};
}

// Flat parameter vector, order Wx, Wh (row-major), bh, Wy, by.
export function rnnFlatten(net) {
  return [...net.Wx, ...net.Wh.flat(), ...net.bh, ...net.Wy, net.by];
}

export function rnnUnflatten(net, v) {
  const nh = net.nh;
  let p = 0;
  const take = n => { const a = v.slice(p, p + n); p += n; return a; };
  const Wx = take(nh);
  const Wh = Array.from({length: nh}, () => take(nh));
  const bh = take(nh), Wy = take(nh), by = v[p];
  return {nh, Wx, Wh, bh, Wy, by};
}
