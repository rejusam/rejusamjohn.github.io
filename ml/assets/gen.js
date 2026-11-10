// Generative model numerics for the ML Explorer: Gaussian fits, Gaussian
// mixtures fitted by EM, and a small 2-D diffusion model (DDPM) with its own
// flat-array ReLU denoiser. Pure functions on plain arrays (the trainer alone
// holds state). Natural logs throughout; pi_k is a mixture weight.
import {mulberry32, randn, mean, variance} from './maths.js';
import {cov2} from './unsupervised.js';
import {spiralData, optInit, optStep} from './nn.js';

// ---- single Gaussians (maximum likelihood, population variance: divide by n) ----

export function gauss1Fit(xs) {
  return {mu: mean(xs), s2: variance(xs, {sample: false})};
}

export function gauss2Fit(X) {
  const n = X.length;
  const mu = [mean(X.map(p => p[0])), mean(X.map(p => p[1]))];
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of X) {
    const dx = p[0] - mu[0], dy = p[1] - mu[1];
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  return {mu, S: [[sxx / n, sxy / n], [sxy / n, syy / n]]};
}

// ---- data ------------------------------------------------------------------------

// The three blobs: centre mu and covariance S of each. Point i comes from blob i mod 3.
export const BLOBS = [
  {mu: [-2, -1], S: [[0.5, 0.2], [0.2, 0.4]]},
  {mu: [2, -1], S: [[0.4, -0.15], [-0.15, 0.6]]},
  {mu: [0, 2.2], S: [[0.7, 0], [0, 0.25]]}
];

// Lower Cholesky factor of a 2x2 covariance.
function chol2(S) {
  const l11 = Math.sqrt(S[0][0]), l21 = S[1][0] / l11;
  return [l11, l21, Math.sqrt(S[1][1] - l21 * l21)];
}

function gaussDraw(mu, S, rng) {
  const [a, b, c] = chol2(S), z1 = randn(rng), z2 = randn(rng);
  return [mu[0] + a * z1, mu[1] + b * z1 + c * z2];
}

export function blobs3({n, seed}) {
  const rng = mulberry32(seed);
  return Array.from({length: n}, (_, i) => gaussDraw(BLOBS[i % 3].mu, BLOBS[i % 3].S, rng));
}

// Both arms of spiralData, labels dropped, each axis standardised to mean 0 and
// population SD 1 (divide by n) using this set. turns and noise pass through to
// spiralData (its defaults when left out).
export function spiral2D({n, seed, turns, noise}) {
  const {X} = spiralData({n, seed, turns, noise});
  const st = [0, 1].map(k => {
    const v = X.map(p => p[k]);
    return [mean(v), Math.sqrt(variance(v, {sample: false}))];
  });
  return X.map(p => [(p[0] - st[0][0]) / st[0][1], (p[1] - st[1][0]) / st[1][1]]);
}

// ---- Gaussian mixtures -------------------------------------------------------------

const LOG2PI = Math.log(2 * Math.PI);

// log N(x | mu, S) in two dimensions.
function logGauss2(x, mu, S) {
  const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  const dx = x[0] - mu[0], dy = x[1] - mu[1];
  const q = (S[1][1] * dx * dx - (S[0][1] + S[1][0]) * dx * dy + S[0][0] * dy * dy) / det;
  return -LOG2PI - 0.5 * Math.log(det) - 0.5 * q;
}

// Means: K distinct data points picked by the seeded rng. Covariances: the data's
// sample covariance (cov2). Weights: 1/K each.
export function gmmInit(X, K, seed) {
  const rng = mulberry32(seed), used = new Set(), mu = [];
  while (mu.length < K) {
    const i = Math.floor(rng() * X.length);
    if (used.has(i)) continue;
    used.add(i);
    mu.push([X[i][0], X[i][1]]);
  }
  const C = cov2(X);
  return {pi: new Array(K).fill(1 / K), mu, S: mu.map(() => [[C[0][0], C[0][1]], [C[1][0], C[1][1]]])};
}

// E-step: responsibilities R[i][k] and the mean log-likelihood per point (log-sum-exp).
function eStep(X, {pi, mu, S}) {
  const K = pi.length, R = [];
  let total = 0;
  for (const x of X) {
    const l = new Array(K);
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      l[k] = Math.log(pi[k]) + logGauss2(x, mu[k], S[k]);
      if (l[k] > m) m = l[k];
    }
    let s = 0;
    for (let k = 0; k < K; k++) s += Math.exp(l[k] - m);
    const lse = m + Math.log(s);
    total += lse;
    R.push(l.map(v => Math.exp(v - lse)));
  }
  return {R, logLik: total / X.length};
}

// M-step: maximum-likelihood weights, means and covariances given R (no ridge).
function mStep(X, R, K) {
  const n = X.length, pi = [], mu = [], S = [];
  for (let k = 0; k < K; k++) {
    let nk = 0, mx = 0, my = 0;
    for (let i = 0; i < n; i++) { nk += R[i][k]; mx += R[i][k] * X[i][0]; my += R[i][k] * X[i][1]; }
    mx /= nk; my /= nk;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      const dx = X[i][0] - mx, dy = X[i][1] - my, r = R[i][k];
      sxx += r * dx * dx; sxy += r * dx * dy; syy += r * dy * dy;
    }
    pi.push(nk / n);
    mu.push([mx, my]);
    S.push([[sxx / nk, sxy / nk], [sxy / nk, syy / nk]]);
  }
  return {pi, mu, S};
}

// Mean log-likelihood per point (natural log).
export function gmmLogLik(X, params) {
  return eStep(X, params).logLik;
}

// logLik[0] is before the first iteration; logLik[i] after i iterations.
export function gmmEM(X, params, iters) {
  const logLik = [];
  let p = params;
  for (let it = 0; it < iters; it++) {
    const e = eStep(X, p);
    logLik.push(e.logLik);
    p = mStep(X, e.R, p.pi.length);
  }
  logLik.push(gmmLogLik(X, p));
  return {params: p, logLik};
}

export function gmmSample({pi, mu, S}, n, seed) {
  const rng = mulberry32(seed), out = [];
  for (let i = 0; i < n; i++) {
    const u = rng();
    let k = 0, c = pi[0];
    while (u >= c && k < pi.length - 1) c += pi[++k];
    out.push(gaussDraw(mu[k], S[k], rng));
  }
  return out;
}

// ---- diffusion: schedule and forward process ----------------------------------------

// Linear beta_1..beta_T; arrays indexed 1..T (index 0 unused, abar[0] = 1).
export function ddpmSchedule(T = 50, {beta1 = 1e-3, betaT = 0.2} = {}) {
  const beta = new Array(T + 1).fill(0), alpha = new Array(T + 1).fill(1), abar = new Array(T + 1).fill(1);
  for (let t = 1; t <= T; t++) {
    beta[t] = T === 1 ? beta1 : beta1 + (betaT - beta1) * (t - 1) / (T - 1);
    alpha[t] = 1 - beta[t];
    abar[t] = abar[t - 1] * alpha[t];
  }
  return {T, beta, alpha, abar};
}

// x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps.
export function qSample(x0, t, eps, sched) {
  const a = Math.sqrt(sched.abar[t]), b = Math.sqrt(1 - sched.abar[t]);
  return [a * x0[0] + b * eps[0], a * x0[1] + b * eps[1]];
}

// [t/T, sin(pi t/T 2^j), cos(pi t/T 2^j) for j = 0, 1, 2]: 7 values.
export function timeFeatures(t, T) {
  const u = t / T, f = [u];
  for (let j = 0; j < 3; j++) {
    const a = Math.PI * u * (1 << j);
    f.push(Math.sin(a), Math.cos(a));
  }
  return f;
}

// ---- denoiser: a flat-array ReLU MLP ---------------------------------------------------
// Input (x, y, ...timeFeatures) = 9 values, output the predicted noise (2 values).
// params holds, layer by layer, W (out x in, row-major) then b (out).

const N_IN = 9;

export function denoiserInit({hidden = [64, 64], seed}) {
  const sizes = [N_IN, ...hidden, 2];
  let n = 0;
  for (let l = 0; l < sizes.length - 1; l++) n += sizes[l + 1] * sizes[l] + sizes[l + 1];
  const params = new Float64Array(n), rng = mulberry32(seed);
  let o = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    const fin = sizes[l], fout = sizes[l + 1], sd = Math.sqrt(2 / fin); // He init
    for (let i = 0; i < fout * fin; i++) params[o++] = sd * randn(rng);
    o += fout; // biases start at 0
  }
  return {sizes, params};
}

export const denoiserFlatten = net => Float64Array.from(net.params);
export const denoiserUnflatten = (net, v) => ({sizes: net.sizes, params: Float64Array.from(v)});

// Forward pass keeping every layer's input (acts) and pre-activation (pre).
function forward(net, x, tf) {
  const {sizes, params} = net, L = sizes.length - 1;
  let a = Float64Array.of(x[0], x[1], ...tf);
  const acts = [a], pre = [];
  let o = 0;
  for (let l = 0; l < L; l++) {
    const fin = sizes[l], fout = sizes[l + 1], z = new Float64Array(fout), bo = o + fout * fin;
    for (let j = 0; j < fout; j++) {
      let s = params[bo + j];
      const r = o + j * fin;
      for (let i = 0; i < fin; i++) s += params[r + i] * a[i];
      z[j] = s;
    }
    o = bo + fout;
    pre.push(z);
    if (l < L - 1) {
      a = new Float64Array(fout);
      for (let j = 0; j < fout; j++) a[j] = z[j] > 0 ? z[j] : 0;
      acts.push(a);
    }
  }
  return {acts, pre};
}

export function denoiserForward(net, x, tf) {
  const out = forward(net, x, tf).pre.at(-1);
  return [out[0], out[1]];
}

// Hidden-layer pre-activations (for checking ReLU kinks).
export function denoiserPreacts(net, x, tf) {
  return forward(net, x, tf).pre.slice(0, -1).map(z => Array.from(z));
}

// batch: rows {x: [2], tf: [7], eps: [2]}. Loss = batch mean of ||eps - eps_hat||^2.
export function denoiserGrad(net, batch) {
  const {sizes, params} = net, L = sizes.length - 1, n = batch.length;
  const grad = new Float64Array(params.length);
  const offs = [];
  let o = 0;
  for (let l = 0; l < L; l++) { offs.push(o); o += sizes[l + 1] * sizes[l] + sizes[l + 1]; }
  let loss = 0;
  for (const row of batch) {
    const {acts, pre} = forward(net, row.x, row.tf);
    const out = pre[L - 1];
    const e0 = out[0] - row.eps[0], e1 = out[1] - row.eps[1];
    loss += e0 * e0 + e1 * e1;
    let delta = Float64Array.of(2 * e0 / n, 2 * e1 / n);
    for (let l = L - 1; l >= 0; l--) {
      const fin = sizes[l], fout = sizes[l + 1], a = acts[l], wo = offs[l], bo = wo + fout * fin;
      for (let j = 0; j < fout; j++) {
        const d = delta[j];
        if (d === 0) continue;
        grad[bo + j] += d;
        const r = wo + j * fin;
        for (let i = 0; i < fin; i++) grad[r + i] += d * a[i];
      }
      if (l > 0) {
        const prev = new Float64Array(fin), zp = pre[l - 1];
        for (let i = 0; i < fin; i++) {
          if (zp[i] <= 0) continue;
          let s = 0;
          for (let j = 0; j < fout; j++) s += params[wo + j * fin + i] * delta[j];
          prev[i] = s;
        }
        delta = prev;
      }
    }
  }
  return {loss: loss / n, grad};
}

// ---- diffusion: training and sampling --------------------------------------------------

const DIVERGED = 1e6, EVAL_N = 512, EVAL_EVERY = 100;

// Training rows: x0 drawn from X with replacement, t uniform on 1..T, eps ~ N(0, I).
function drawRows(X, sched, n, rng) {
  const rows = [];
  for (let b = 0; b < n; b++) {
    const x0 = X[Math.floor(rng() * X.length)];
    const t = 1 + Math.floor(rng() * sched.T);
    const eps = [randn(rng), randn(rng)];
    rows.push({x: qSample(x0, t, eps, sched), tf: timeFeatures(t, sched.T), eps});
  }
  return rows;
}

// One step: a fresh batch, the MSE between the net's noise guess and the true
// noise, one optimiser step. evalLoss is measured every 100 steps (and at 0) on a
// fixed batch of 512 rows drawn with seed + 1. state is updated in place.
export function ddpmTrainer({net, X, sched, batch = 128, kind = 'adam', hyper = {eta: 1e-3}, seed}) {
  const start = denoiserUnflatten(net, net.params);
  const evalRows = drawRows(X, sched, EVAL_N, mulberry32(seed + 1));
  const evalLoss = n => denoiserGrad(n, evalRows).loss;
  let rng, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    params = denoiserFlatten(start);
    opt = optInit(kind, params.length);
    const first = evalLoss(start);
    return {net: denoiserUnflatten(start, params), iter: 0, batchLoss: [], evalLoss: [first],
      diverged: !Number.isFinite(first)};
  };

  const api = {state: fresh()};

  const one = () => {
    const s = api.state;
    const g = denoiserGrad(s.net, drawRows(X, sched, batch, rng));
    if (!Number.isFinite(g.loss) || g.loss > DIVERGED) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, g.grad);
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.net = {sizes: s.net.sizes, params};
    s.batchLoss.push(g.loss);
    s.iter++;
    if (s.iter % EVAL_EVERY === 0) {
      const el = evalLoss(s.net);
      s.evalLoss.push(el);
      if (!Number.isFinite(el) || el > DIVERGED) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}

// Ho, Jain & Abbeel (2020), Algorithm 2, one step from t to t - 1:
// x' = (1/sqrt(alpha_t)) (x - (beta_t / sqrt(1 - abar_t)) eps_hat) + sigma_t z,
// sigma_t^2 = beta_t, and z is ignored at t = 1. net is a denoiser or a function (x, t) -> eps_hat.
export function ddpmStep(net, x, t, z, sched) {
  const e = typeof net === 'function' ? net(x, t) : denoiserForward(net, x, timeFeatures(t, sched.T));
  const b = sched.beta[t], c = b / Math.sqrt(1 - sched.abar[t]), r = 1 / Math.sqrt(sched.alpha[t]);
  const sig = t > 1 ? Math.sqrt(b) : 0;
  return [r * (x[0] - c * e[0]) + sig * z[0], r * (x[1] - c * e[1]) + sig * z[1]];
}

// Start from n points of N(0, I) and step t = T ... 1. path[k] holds the points
// after k steps (path[0] is the starting noise, path[T] the samples).
export function ddpmSample(net, n, sched, seed, {keepPath = false} = {}) {
  const rng = mulberry32(seed);
  let X = Array.from({length: n}, () => [randn(rng), randn(rng)]);
  const path = keepPath ? [X] : null;
  for (let t = sched.T; t >= 1; t--) {
    X = X.map(x => ddpmStep(net, x, t, t > 1 ? [randn(rng), randn(rng)] : [0, 0], sched));
    if (keepPath) path.push(X);
  }
  return keepPath ? {X, path} : {X};
}

// Mean over a in A of the distance from a to its nearest b in B.
export function nearestDist(A, B) {
  let total = 0;
  for (const a of A) {
    let best = Infinity;
    for (const b of B) {
      const d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
      if (d < best) best = d;
    }
    total += Math.sqrt(best);
  }
  return total / A.length;
}
