// Linear soft-margin SVM for the ML Explorer. Pure functions on plain arrays.
// X is rows of numbers (any dimension), y is -1 or +1, f(x) = w·x + b.
import {dot, norm} from './maths.js';

const weights = (X, y, alpha) => {
  const w = new Array(X[0].length).fill(0);
  X.forEach((x, i) => {
    if (alpha[i] !== 0) for (let j = 0; j < w.length; j++) w[j] += alpha[i] * y[i] * x[j];
  });
  return w;
};

export const hinge = (X, y, w, b) => X.map((x, i) => Math.max(0, 1 - y[i] * (dot(w, x) + b)));

// Largest breach of the KKT conditions: α = 0 needs y f ≥ 1,
// 0 < α < C needs y f = 1, α = C needs y f ≤ 1.
export function kktViolation(X, y, C, alpha, b) {
  const w = weights(X, y, alpha);
  let worst = 0;
  X.forEach((x, i) => {
    const yf = y[i] * (dot(w, x) + b), a = alpha[i];
    const v = a <= 0 ? 1 - yf : a >= C ? yf - 1 : Math.abs(yf - 1);
    if (v > worst) worst = v;
  });
  return worst;
}

// Dual problem: maximise Σα − ½ΣΣ αi αj yi yj xi·xj subject to
// 0 ≤ α ≤ C and Σ α y = 0. SMO: each step moves the most violating
// pair (i, j) along the line that keeps Σ α y fixed, solved exactly and
// clipped to the box. Stops when the violation gap is at most tol.
export function svmFit(X, y, {C, tol = 1e-9, maxIter = 100000}) {
  const n = X.length;
  const K = X.map(a => X.map(b => dot(a, b)));
  const alpha = new Array(n).fill(0);
  // G is the gradient of the minimised dual, (Qα)k − 1 with Qkl = yk yl Kkl.
  const freshG = () => X.map((_, k) => {
    let s = 0;
    for (let l = 0; l < n; l++) if (alpha[l] !== 0) s += alpha[l] * y[l] * K[k][l];
    return y[k] * s - 1;
  });
  let G = freshG();
  // i can move up the line, j down it; -y G is y − w·x, the bias each point asks for.
  const isUp = k => (y[k] === 1 ? alpha[k] < C : alpha[k] > 0);
  const isLow = k => (y[k] === 1 ? alpha[k] > 0 : alpha[k] < C);
  const select = () => {
    let i = -1, j = -1, m = -Infinity, M = Infinity;
    for (let k = 0; k < n; k++) {
      const v = -y[k] * G[k];
      if (isUp(k) && v > m) { m = v; i = k; }
      if (isLow(k) && v < M) { M = v; j = k; }
    }
    return {i, j, gap: m - M};
  };

  let iters = 0, ok = false;
  for (;;) {
    let {i, j, gap} = select();
    if (!(gap > tol)) {
      G = freshG();  // clear rounding drift before accepting
      ({i, j, gap} = select());
      if (!(gap > tol)) { ok = true; break; }
    }
    if (iters >= maxIter) break;
    // αi += yi t, αj −= yj t; along that line the minimised dual is lowest at t = gap / (Kii + Kjj − 2Kij).
    const eta = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], 1e-12);
    const ti = y[i] === 1 ? C - alpha[i] : alpha[i];
    const tj = y[j] === 1 ? alpha[j] : C - alpha[j];
    const t = Math.min(gap / eta, ti, tj);
    const set = (k, v, hit) => { alpha[k] = hit ? (v > C / 2 ? C : 0) : Math.min(C, Math.max(0, v)); };
    set(i, alpha[i] + y[i] * t, t === ti);
    set(j, alpha[j] - y[j] * t, t === tj);
    for (let k = 0; k < n; k++) G[k] += y[k] * t * (K[k][i] - K[k][j]);
    iters++;
  }

  const w = weights(X, y, alpha);
  // b: average over free support vectors, else the middle of the feasible interval.
  let sum = 0, nFree = 0, lo = -Infinity, hi = Infinity;
  X.forEach((x, k) => {
    const v = y[k] - dot(w, x);
    if (alpha[k] > 0 && alpha[k] < C) { sum += v; nFree++; }
    if (isUp(k)) lo = Math.max(lo, v);
    if (isLow(k)) hi = Math.min(hi, v);
  });
  const b = nFree ? sum / nFree
    : Number.isFinite(lo) && Number.isFinite(hi) ? (lo + hi) / 2
    : Number.isFinite(lo) ? lo : Number.isFinite(hi) ? hi : 0;
  const sv = alpha.flatMap((a, k) => (a > 1e-8 ? [k] : []));
  // With w = 0 (no steps taken, or one class only) there is no margin: null.
  const wn = norm(w);
  return {ok, w, b, alpha, sv, margin: wn > 0 ? 2 / wn : null, kkt: kktViolation(X, y, C, alpha, b), iters};
}
