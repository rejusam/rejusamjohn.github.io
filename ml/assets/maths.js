// Numerics for the ML Explorer. Pure functions on plain number arrays.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const uniform = (rng, a, b) => a + (b - a) * rng();

export const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;

export function median(xs) {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function variance(xs, {sample = true} = {}) {
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - (sample ? 1 : 0));
}

export const sd = (xs, opts) => Math.sqrt(variance(xs, opts));

export function zscores(xs, opts) {
  const m = mean(xs), s = sd(xs, opts);
  return xs.map(x => (x - m) / s);
}

export const residuals = (ys, yhat) => ys.map((y, i) => y - yhat[i]);
export const mse = (ys, yhat) => mean(residuals(ys, yhat).map(e => e * e));
export const mae = (ys, yhat) => mean(residuals(ys, yhat).map(Math.abs));

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = a => Math.sqrt(dot(a, a));
export const euclidean = (a, b) => norm(a.map((x, i) => x - b[i]));

export const centralDiff = (f, x, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);

export function olsFit(xs, ys) {
  const mx = mean(xs), my = mean(ys);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
    syy += (ys[i] - my) ** 2;
  }
  if (sxx === 0) return {ok: false, reason: 'no-x-spread'};
  const slope = sxy / sxx, intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return {ok: true, slope, intercept, r2};
}

// Least squares by Householder QR on X, augmented with sqrt(lambda) rows for ridge.
export function lstsq(X, y, {lambda = 0, penalise = null} = {}) {
  const p = X[0].length;
  const A = X.map(r => [...r]), b = [...y];
  if (lambda > 0) {
    const s = Math.sqrt(lambda);
    for (let j = 0; j < p; j++) {
      if (penalise ? penalise[j] : j > 0) {
        const row = new Array(p).fill(0);
        row[j] = s;
        A.push(row);
        b.push(0);
      }
    }
  }
  const m = A.length;
  if (m < p) return {ok: false, coef: new Array(p).fill(NaN)};
  for (let k = 0; k < p; k++) {
    let nrm = 0;
    for (let i = k; i < m; i++) nrm += A[i][k] ** 2;
    nrm = Math.sqrt(nrm);
    if (nrm === 0) continue;
    const alpha = A[k][k] > 0 ? -nrm : nrm;
    const v = new Array(m).fill(0);
    for (let i = k; i < m; i++) v[i] = A[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] ** 2;
    if (vv === 0) continue;
    for (let j = k; j < p; j++) {
      let d = 0;
      for (let i = k; i < m; i++) d += v[i] * A[i][j];
      const f = 2 * d / vv;
      for (let i = k; i < m; i++) A[i][j] -= f * v[i];
    }
    let d = 0;
    for (let i = k; i < m; i++) d += v[i] * b[i];
    const f = 2 * d / vv;
    for (let i = k; i < m; i++) b[i] -= f * v[i];
  }
  const coef = new Array(p).fill(0);
  let ok = true;
  const scale = Math.max(...Array.from({length: p}, (_, k) => Math.abs(A[k][k])), 1e-300);
  for (let k = p - 1; k >= 0; k--) {
    let s = b[k];
    for (let j = k + 1; j < p; j++) s -= A[k][j] * coef[j];
    if (Math.abs(A[k][k]) < 1e-12 * scale) { ok = false; coef[k] = 0; }
    else coef[k] = s / A[k][k];
  }
  return {ok, coef};
}

export const polyDesign = (xs, degree) =>
  xs.map(x => Array.from({length: degree + 1}, (_, j) => x ** j));

// Assumes the caller has already scaled x into [-1, 1].
export const polyFit = (xs, ys, degree, lambda = 0) =>
  lstsq(polyDesign(xs, degree), ys, {lambda});

export const polyEval = (coef, x) => coef.reduceRight((s, c) => s * x + c, 0);

export function gdRun({grad, init, eta, steps, tol = 1e-10, limit = 1e6}) {
  let w = [...init];
  const path = [w];
  for (let t = 0; t < steps; t++) {
    const g = grad(w);
    const next = w.map((v, i) => v - eta * g[i]);
    if (!next.every(Number.isFinite) || norm(next) > limit) return {path, status: 'diverged'};
    path.push(next);
    const step = norm(next.map((v, i) => v - w[i]));
    w = next;
    if (step < tol) return {path, status: 'converged'};
  }
  return {path, status: 'max-steps'};
}

export const mseGrad1D = (xs, ys) => ([w]) =>
  [(2 / xs.length) * xs.reduce((s, x, i) => s + (w * x - ys[i]) * x, 0)];

export const mseGrad2D = (xs, ys) => ([b, w]) => {
  const n = xs.length;
  let gb = 0, gw = 0;
  for (let i = 0; i < n; i++) {
    const e = b + w * xs[i] - ys[i];
    gb += e;
    gw += e * xs[i];
  }
  return [2 * gb / n, 2 * gw / n];
};

export function standardise(xs) {
  const mu = mean(xs), sigma = sd(xs, {sample: false});
  return {z: xs.map(x => (x - mu) / sigma), mu, sigma};
}

export function trainTestSplit(n, testFraction, rng) {
  const idx = Array.from({length: n}, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.round(n * testFraction);
  return {train: idx.slice(nTest), test: idx.slice(0, nTest)};
}

// Eigenvalues of [[a, b], [b, c]], largest first.
export function eig2Sym([[a, b], [, c]]) {
  const m = (a + c) / 2, r = Math.hypot((a - c) / 2, b);
  return [m + r, m - r];
}
