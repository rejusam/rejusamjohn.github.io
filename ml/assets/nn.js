// Neural network numerics for the ML Explorer: data, a small MLP with
// backpropagation, optimisers and a seeded mini-batch trainer.
// Pure functions on plain arrays (the trainer alone holds state).
// W[l][j][i] maps unit i of layer l to unit j of layer l+1; sigma is the sigmoid.
import {mulberry32, randn} from './maths.js';
import {sigmoid} from './classify.js';

// ---- data ------------------------------------------------------------

// Four Gaussian clumps at (+-1, +-1); class 1 where the corner's signs differ.
export function xorData({n, sd = 0.35, seed}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rng() * 4);
    const sx = c % 2 ? 1 : -1, sy = c >> 1 ? 1 : -1;
    const px = sx + sd * randn(rng);
    const py = sy + sd * randn(rng);
    X.push([px, py]);
    y.push(sx !== sy ? 1 : 0);
  }
  return {X, y};
}

// Inner ring (class 0, first n/2 rows) and outer ring (class 1).
export function circlesData({n, rIn = 1, rOut = 2.2, noise = 0.25, seed}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const cls = i < half ? 0 : 1;
    const theta = 2 * Math.PI * rng();
    const r = (cls ? rOut : rIn) + noise * randn(rng);
    X.push([r * Math.cos(theta), r * Math.sin(theta)]);
    y.push(cls);
  }
  return {X, y};
}

// Two interleaved spiral arms, rows alternating by arm blocks (arm 0 first).
export function spiralData({n, turns = 1.5, noise = 0.2, seed}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  const per = n / 2;
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < per; i++) {
      const t = i / per;
      const r = 0.3 + 2.7 * t;
      const th = 2 * Math.PI * turns * t + Math.PI * c;
      const px = r * Math.cos(th) + noise * randn(rng);
      const py = r * Math.sin(th) + noise * randn(rng);
      X.push([px, py]);
      y.push(c);
    }
  }
  return {X, y};
}

// ---- activations -----------------------------------------------------

// relu.df(0) = 0 by convention; step has zero gradient everywhere.
export const ACT = {
  sigmoid: {f: sigmoid, df: z => { const s = sigmoid(z); return s * (1 - s); }},
  tanh: {f: Math.tanh, df: z => { const t = Math.tanh(z); return 1 - t * t; }},
  relu: {f: z => (z > 0 ? z : 0), df: z => (z > 0 ? 1 : 0)},
  step: {f: z => (z > 0 ? 1 : 0), df: () => 0}
};

// ---- network ---------------------------------------------------------

// init: 'xavier' (variance 1/n_in), 'he' (2/n_in) or a number (a standard deviation).
export function mlpInit(sizes, {hidden = 'tanh', init = 'xavier', seed} = {}) {
  const rng = mulberry32(seed);
  const W = [], b = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nIn = sizes[l];
    const s = typeof init === 'number' ? init
      : Math.sqrt((init === 'he' ? 2 : 1) / nIn);
    W.push(Array.from({length: sizes[l + 1]}, () =>
      Array.from({length: nIn}, () => s * randn(rng))));
    b.push(new Array(sizes[l + 1]).fill(0));
  }
  return {sizes: sizes.slice(), hidden, W, b};
}

// zs[l] and as[l+1] belong to layer l+1; the last layer is linear.
export function mlpForward(net, x) {
  const L = net.W.length;
  const g = ACT[net.hidden].f;
  const zs = [], as = [x.slice()];
  for (let l = 0; l < L; l++) {
    const prev = as[l];
    const z = net.W[l].map((row, j) => {
      let s = net.b[l][j];
      for (let i = 0; i < row.length; i++) s += row[i] * prev[i];
      return s;
    });
    zs.push(z);
    as.push(l === L - 1 ? z.slice() : z.map(g));
  }
  return {zs, as, out: zs[L - 1].slice()};
}

// Binary cross-entropy on a logit; the same quantity as L3's log-loss.
export const bceLogit = (y, z) => Math.max(z, 0) - y * z + Math.log1p(Math.exp(-Math.abs(z)));

const target = (y, k) => (Array.isArray(y) ? y[k] : y);

function exampleLoss(out, y, loss) {
  let s = 0;
  for (let k = 0; k < out.length; k++) {
    const t = target(y, k);
    s += loss === 'mse' ? 0.5 * (out[k] - t) * (out[k] - t) : bceLogit(t, out[k]);
  }
  return s;
}

// One example. dW and db have the shapes of net.W and net.b.
export function mlpBackward(net, x, y, {loss = 'bce'} = {}) {
  const L = net.W.length;
  const {zs, as, out} = mlpForward(net, x);
  const dg = ACT[net.hidden].df;
  const dW = new Array(L), db = new Array(L);
  let delta = out.map((o, k) => (loss === 'mse' ? o - target(y, k) : sigmoid(o) - target(y, k)));
  for (let l = L - 1; l >= 0; l--) {
    dW[l] = delta.map(d => as[l].map(a => d * a));
    db[l] = delta.slice();
    if (l > 0) {
      const next = new Array(as[l].length).fill(0);
      for (let j = 0; j < delta.length; j++) {
        const row = net.W[l][j];
        for (let i = 0; i < next.length; i++) next[i] += row[i] * delta[j];
      }
      delta = next.map((v, i) => v * dg(zs[l - 1][i]));
    }
  }
  return {dW, db, loss: exampleLoss(out, y, loss)};
}

const zerosLike = net => ({
  dW: net.W.map(M => M.map(r => r.map(() => 0))),
  db: net.b.map(v => v.map(() => 0))
});

// Mean gradient over idx; lambda * W is added to dW only (not to biases).
export function batchGrad(net, X, y, idx, {loss = 'bce', lambda = 0} = {}) {
  const {dW, db} = zerosLike(net);
  let total = 0;
  for (const n of idx) {
    const g = mlpBackward(net, X[n], y[n], {loss});
    total += g.loss;
    for (let l = 0; l < dW.length; l++) {
      for (let j = 0; j < dW[l].length; j++) {
        for (let i = 0; i < dW[l][j].length; i++) dW[l][j][i] += g.dW[l][j][i];
        db[l][j] += g.db[l][j];
      }
    }
  }
  const m = idx.length;
  for (let l = 0; l < dW.length; l++) {
    for (let j = 0; j < dW[l].length; j++) {
      for (let i = 0; i < dW[l][j].length; i++) {
        dW[l][j][i] = dW[l][j][i] / m + lambda * net.W[l][j][i];
      }
      db[l][j] /= m;
    }
  }
  return {dW, db, loss: total / m};
}

export const predictProb = (net, x) => sigmoid(mlpForward(net, x).out[0]);

export function accuracyNet(net, X, y) {
  let ok = 0;
  for (let i = 0; i < X.length; i++) ok += (predictProb(net, X[i]) > 0.5 ? 1 : 0) === y[i] ? 1 : 0;
  return ok / X.length;
}

// W then b, layer by layer, row-major.
export function flatten(net) {
  const v = [];
  for (let l = 0; l < net.W.length; l++) {
    for (const row of net.W[l]) for (const w of row) v.push(w);
    for (const c of net.b[l]) v.push(c);
  }
  return v;
}

export function unflatten(net, v) {
  let p = 0;
  const W = [], b = [];
  for (let l = 0; l < net.W.length; l++) {
    W.push(net.W[l].map(row => row.map(() => v[p++])));
    b.push(net.b[l].map(() => v[p++]));
  }
  return {...net, sizes: net.sizes.slice(), W, b};
}

// Largest relative gap between backprop and central differences.
export function gradCheck(f, params, grad, {h = 1e-5} = {}) {
  let worst = 0;
  for (let i = 0; i < params.length; i++) {
    const up = params.slice(), dn = params.slice();
    up[i] += h; dn[i] -= h;
    const num = (f(up) - f(dn)) / (2 * h);
    const rel = Math.abs(grad[i] - num) / Math.max(1e-8, Math.abs(grad[i]) + Math.abs(num));
    if (rel > worst) worst = rel;
  }
  return worst;
}

// ---- optimisers ------------------------------------------------------

export function optInit(kind, nParams) {
  const z = () => new Array(nParams).fill(0);
  if (kind === 'momentum') return {v: z()};
  if (kind === 'rmsprop') return {s: z()};
  if (kind === 'adam') return {t: 0, m: z(), v: z()};
  return {};
}

// hyper: eta, beta (momentum), beta2 (rmsprop; adam), beta1, eps. Never mutates inputs.
export function optStep(kind, hyper, state, params, grad) {
  const {eta = 0.01, eps = 1e-8} = hyper;
  const beta = hyper.beta ?? 0.9, beta1 = hyper.beta1 ?? 0.9;
  const beta2 = hyper.beta2 ?? (kind === 'adam' ? 0.999 : 0.9);
  if (kind === 'momentum') {
    const v = state.v.map((vi, i) => beta * vi + grad[i]);
    return {params: params.map((p, i) => p - eta * v[i]), state: {v}};
  }
  if (kind === 'rmsprop') {
    const s = state.s.map((si, i) => beta2 * si + (1 - beta2) * grad[i] * grad[i]);
    return {params: params.map((p, i) => p - eta * grad[i] / (Math.sqrt(s[i]) + eps)), state: {s}};
  }
  if (kind === 'adam') {
    const t = state.t + 1;
    const m = state.m.map((mi, i) => beta1 * mi + (1 - beta1) * grad[i]);
    const v = state.v.map((vi, i) => beta2 * vi + (1 - beta2) * grad[i] * grad[i]);
    const c1 = 1 - Math.pow(beta1, t), c2 = 1 - Math.pow(beta2, t);
    return {
      params: params.map((p, i) => p - eta * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps)),
      state: {t, m, v}
    };
  }
  return {params: params.map((p, i) => p - eta * grad[i]), state: {...state}};
}

// ---- trainer ---------------------------------------------------------

const DIVERGED = 1e6;

function meanLoss(net, X, y, loss) {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += exampleLoss(mlpForward(net, X[i]).out, y[i], loss);
  return s / X.length;
}

// Mini-batch training with a seeded shuffle each epoch. The returned state
// object is updated in place by step(); reset() gives a fresh one.
export function trainer({net, X, y, batch, kind = 'sgd', hyper = {}, seed, lambda = 0, loss = 'bce'}) {
  const start = unflatten(net, flatten(net));
  const n = X.length;
  let rng, order, cursor, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    order = null;
    cursor = 0;
    params = flatten(start);
    opt = optInit(kind, params.length);
    const first = meanLoss(start, X, y, loss);
    return {net: unflatten(start, params), epoch: 0, iter: 0, batchLoss: [],
      epochLoss: [first], diverged: !Number.isFinite(first)};
  };

  const shuffle = () => {
    order = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    cursor = 0;
  };

  const api = {state: fresh()};

  const one = () => {
    const s = api.state;
    if (cursor === 0) shuffle();
    const idx = order.slice(cursor, cursor + batch);
    const g = batchGrad(s.net, X, y, idx, {loss, lambda});
    if (!Number.isFinite(g.loss) || Math.abs(g.loss) > DIVERGED) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, flatten({...s.net, W: g.dW, b: g.db}));
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.net = unflatten(s.net, params);
    s.batchLoss.push(g.loss);
    s.iter++;
    cursor += batch;
    if (cursor >= n) {
      cursor = 0;
      s.epoch++;
      const el = meanLoss(s.net, X, y, loss);
      s.epochLoss.push(el);
      if (!Number.isFinite(el) || Math.abs(el) > DIVERGED) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}
