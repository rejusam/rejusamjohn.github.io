// Classification numerics for the ML Explorer: k-NN, logistic regression, Naive Bayes.
// Pure functions on plain arrays. X is rows of numbers, y is 0 or 1.
import {mulberry32, randn, euclidean, mean} from './maths.js';

// Two Gaussian clouds; class 0 rows first. sd is a number or [sx, sy].
export function blobs2({n0, n1, mu0, mu1, sd, rho = 0, seed}) {
  const rng = mulberry32(seed);
  const [sx, sy] = Array.isArray(sd) ? sd : [sd, sd];
  const c = Math.sqrt(1 - rho * rho);
  const X = [], y = [];
  const draw = (mu, cls) => {
    const z1 = randn(rng), z2 = randn(rng);
    X.push([mu[0] + sx * z1, mu[1] + sy * (rho * z1 + c * z2)]);
    y.push(cls);
  };
  for (let i = 0; i < n0; i++) draw(mu0, 0);
  for (let i = 0; i < n1; i++) draw(mu1, 1);
  return {X, y};
}

// Never overflows: only exp of a non-positive number is taken.
export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// Mean log-loss, with p clipped so the answer is always finite.
export function logLoss(y, p) {
  const lo = 1e-15, hi = 1 - 1e-15;
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const q = Math.min(hi, Math.max(lo, p[i]));
    s -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q);
  }
  return s / y.length;
}

const probs = (X, y, w0, w) => X.map(r => sigmoid(w0 + r.reduce((s, v, j) => s + v * w[j], 0)));

// Solve A x = b by Gaussian elimination with partial pivoting; null if singular.
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-14) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// Newton (IRLS) on the mean log-loss. Separable data has no finite optimum.
export function logisticFit(X, y, {maxIter = 100, tol = 1e-10} = {}) {
  const n = X.length, d = X[0].length;
  let beta = new Array(d + 1).fill(0);
  const split = b => [b[0], b.slice(1)];
  const finite = b => b.every(Number.isFinite);
  for (let it = 0; it <= maxIter; it++) {
    const [w0, w] = split(beta);
    const p = probs(X, y, w0, w);
    const g = new Array(d + 1).fill(0);
    const H = Array.from({length: d + 1}, () => new Array(d + 1).fill(0));
    for (let i = 0; i < n; i++) {
      const row = [1, ...X[i]], e = p[i] - y[i], s = p[i] * (1 - p[i]);
      for (let a = 0; a <= d; a++) {
        g[a] += e * row[a] / n;
        for (let b = 0; b <= d; b++) H[a][b] += s * row[a] * row[b] / n;
      }
    }
    const gradNorm = Math.hypot(...g);
    // Perfect training accuracy means the classes are separable: no finite optimum.
    const perfect = p.every((q, i) => (q > 0.5 ? 1 : 0) === y[i] && q !== 0.5);
    if (Math.max(...beta.map(Math.abs)) > 1e6 || logLoss(y, p) < 1e-12 || (perfect && it > 0))
      return {ok: false, reason: 'separable', w0, w};
    if (gradNorm < tol) return {ok: true, w0, w, iters: it, gradNorm};
    if (it === maxIter) break;
    const step = solve(H, g);
    if (!step || !finite(step)) return {ok: false, reason: 'no-converge', w0, w};
    beta = beta.map((v, k) => v - step[k]);
  }
  const [w0, w] = split(beta);
  return {ok: false, reason: 'no-converge', w0, w};
}

// Batch gradient descent on the mean log-loss. Returns every state, start included.
export function logisticGD(X, y, {eta, steps, w0 = 0, w = null}) {
  const n = X.length, d = X[0].length;
  let b0 = w0, b = w ? [...w] : new Array(d).fill(0);
  const out = [];
  for (let t = 0; t <= steps; t++) {
    const p = probs(X, y, b0, b);
    out.push({w0: b0, w: [...b], loss: logLoss(y, p)});
    if (t === steps) break;
    let g0 = 0;
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const e = p[i] - y[i];
      g0 += e;
      for (let j = 0; j < d; j++) g[j] += e * X[i][j];
    }
    b0 -= eta * g0 / n;
    b = b.map((v, j) => v - eta * g[j] / n);
  }
  return out;
}

// k nearest by Euclidean distance; ties go to the lower index. Majority vote (k odd).
export function knnPredict(Xtr, ytr, x, k) {
  const idx = Xtr.map((r, i) => [euclidean(r, x), i])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .slice(0, k).map(p => p[1]);
  const votes1 = idx.reduce((s, i) => s + ytr[i], 0);
  return {cls: votes1 > k / 2 ? 1 : 0, votes1, idx};
}

export const accuracy = (y, yhat) => y.filter((v, i) => v === yhat[i]).length / y.length;

export function confusion(y, yhat) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  y.forEach((v, i) => {
    if (yhat[i] === 1) v === 1 ? tp++ : fp++;
    else v === 1 ? fn++ : tn++;
  });
  return {
    tp, fp, tn, fn,
    accuracy: (tp + tn) / y.length,
    precision: tp + fp === 0 ? null : tp / (tp + fp),
    recall: tp + fn === 0 ? null : tp / (tp + fn)
  };
}

// Per-class prior, feature means and variances (MLE: divide by the class count).
export function gaussianNB(X, y) {
  const d = X[0].length, prior = [], mu = [], vari = [];
  for (const c of [0, 1]) {
    const rows = X.filter((_, i) => y[i] === c), m = rows.length;
    prior.push(m / X.length);
    const mc = Array.from({length: d}, (_, j) => rows.reduce((s, r) => s + r[j], 0) / m);
    mu.push(mc);
    vari.push(mc.map((u, j) => rows.reduce((s, r) => s + (r[j] - u) ** 2, 0) / m));
  }
  return {prior, mu, vari};
}

// Posterior over the two classes, in log space with log-sum-exp.
export function nbPosterior(model, x) {
  const lp = [0, 1].map(c => {
    let s = Math.log(model.prior[c]);
    x.forEach((v, j) => {
      const va = model.vari[c][j];
      s += -0.5 * Math.log(2 * Math.PI * va) - (v - model.mu[c][j]) ** 2 / (2 * va);
    });
    return s;
  });
  const m = Math.max(...lp), z = m + Math.log(lp.reduce((s, v) => s + Math.exp(v - m), 0));
  return lp.map(v => Math.exp(v - z));
}

// ---- Trees, forests, boosting ----

// Gini impurity of class counts: 1 - sum (c/n)^2. Empty gives 0.
export function gini(counts) {
  const n = counts.reduce((s, c) => s + c, 0);
  return n === 0 ? 0 : 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0);
}

const countsOf = (y, idx) => {
  const c = [0, 0];
  for (const i of idx) c[y[i]]++;
  return c;
};

// Best split on the rows in idx. Thresholds sit midway between consecutive distinct values;
// left is x < threshold. Lowest weighted Gini wins; ties go to the lower feature, then the
// lower threshold. null if the node is pure or no split gains more than 1e-12.
// minLeaf (default 1) rejects splits that leave fewer rows on either side.
export function bestSplit(X, y, idx, {features = null, minLeaf = 1} = {}) {
  const n = idx.length, tot = countsOf(y, idx), parent = gini(tot);
  if (n < 2 || tot[0] === 0 || tot[1] === 0) return null;
  const feats = features ?? X[0].map((_, j) => j);
  let best = null;
  for (const f of feats) {
    const order = [...idx].sort((a, b) => X[a][f] - X[b][f]);
    const cl = [0, 0];
    for (let k = 0; k + 1 < n; k++) {
      cl[y[order[k]]]++;
      const a = X[order[k]][f], b = X[order[k + 1]][f];
      if (a === b) continue;
      const nl = k + 1, nr = n - nl;
      if (nl < minLeaf || nr < minLeaf) continue;
      const cr = [tot[0] - cl[0], tot[1] - cl[1]];
      const imp = (nl * gini(cl) + nr * gini(cr)) / n;
      if (!best || imp < best.impurity - 1e-12) best = {feature: f, threshold: (a + b) / 2, impurity: imp};
    }
  }
  if (!best || parent - best.impurity <= 1e-12) return null;
  const left = idx.filter(i => X[i][best.feature] < best.threshold);
  const right = idx.filter(i => X[i][best.feature] >= best.threshold);
  return {...best, gain: parent - best.impurity, left, right};
}

// Grow a tree. Leaf class is the majority, a tie goes to class 0.
// With mtry and rng, each split looks at mtry features drawn without replacement.
export function growTree(X, y, {maxDepth = Infinity, minLeaf = 1, mtry = null, rng = null, idx = null} = {}) {
  const d = X[0].length;
  const grow = (rows, depth) => {
    const counts = countsOf(y, rows);
    const node = {leaf: true, counts, cls: counts[1] > counts[0] ? 1 : 0, depth, n: rows.length};
    if (depth >= maxDepth) return node;
    let features = null;
    if (mtry !== null && mtry < d) {
      const pool = Array.from({length: d}, (_, j) => j);
      for (let k = 0; k < mtry; k++) {
        const j = k + Math.floor(rng() * (d - k));
        [pool[k], pool[j]] = [pool[j], pool[k]];
      }
      features = pool.slice(0, mtry).sort((a, b) => a - b);
    }
    const s = bestSplit(X, y, rows, {features, minLeaf});
    if (!s) return node;
    return {...node, leaf: false, feature: s.feature, threshold: s.threshold,
      left: grow(s.left, depth + 1), right: grow(s.right, depth + 1)};
  };
  return grow(idx ?? X.map((_, i) => i), 0);
}

export function predictTree(node, x) {
  while (!node.leaf) node = x[node.feature] < node.threshold ? node.left : node.right;
  return node.cls;
}

export function treeStats(node) {
  if (node.leaf) return {depth: node.depth, leaves: 1};
  const a = treeStats(node.left), b = treeStats(node.right);
  return {depth: Math.max(a.depth, b.depth), leaves: a.leaves + b.leaves};
}

// n draws with replacement.
export const bootstrapIdx = (n, rng) => Array.from({length: n}, () => Math.floor(rng() * n));

// Random forest. One rng stream, in tree order (bootstrap, then that tree's feature draws),
// so the first b trees of a B-forest are the b-forest. A vote tie goes to class 0.
export function forest(X, y, {B, maxDepth = Infinity, mtry = null, seed}) {
  const rng = mulberry32(seed), trees = [], samples = [];
  for (let t = 0; t < B; t++) {
    const idx = bootstrapIdx(X.length, rng);
    samples.push(idx);
    trees.push(growTree(X, y, {maxDepth, mtry, rng, idx}));
  }
  const votes = (x, b = B) => trees.slice(0, b).reduce((s, t) => s + predictTree(t, x), 0) / b;
  const predict = (x, b = B) => votes(x, b) > 0.5 ? 1 : 0;
  const oob = (b = B) => {
    const pred = [];
    let right = 0, covered = 0;
    X.forEach((x, i) => {
      const vs = [];
      for (let t = 0; t < b; t++) if (!samples[t].includes(i)) vs.push(predictTree(trees[t], x));
      if (vs.length === 0) { pred.push(null); return; }
      const p = mean(vs) > 0.5 ? 1 : 0;
      pred.push(p); covered++;
      if (p === y[i]) right++;
    });
    return {pred, accuracy: covered ? right / covered : null, covered};
  };
  return {trees, samples, votes, predict, oob};
}

// Best one-split fit to r by least squares: left is x < threshold, each side takes its mean.
export function stumpFit(xs, r) {
  const n = xs.length, order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const total = r.reduce((s, v) => s + v, 0);
  let bestK = -1, bestScore = -Infinity, sl = 0;
  for (let k = 0; k + 1 < n; k++) {
    sl += r[order[k]];
    if (xs[order[k]] === xs[order[k + 1]]) continue;
    const score = sl * sl / (k + 1) + (total - sl) ** 2 / (n - k - 1);
    if (score > bestScore + 1e-12) { bestScore = score; bestK = k; }
  }
  if (bestK < 0) {
    const m = mean(r);
    return {threshold: -Infinity, left: m, right: m, sse: r.reduce((s, v) => s + (v - m) ** 2, 0)};
  }
  const threshold = (xs[order[bestK]] + xs[order[bestK + 1]]) / 2;
  const L = r.filter((_, i) => xs[i] < threshold), R = r.filter((_, i) => xs[i] >= threshold);
  const left = mean(L), right = mean(R);
  const sse = L.reduce((s, v) => s + (v - left) ** 2, 0) + R.reduce((s, v) => s + (v - right) ** 2, 0);
  return {threshold, left, right, sse};
}

// Gradient boosting with stumps on squared error. F0 is the mean; each round fits a stump
// to the residuals and adds eta times it. trainMSE[0] is F0 alone.
export function boost(xs, ys, {M, eta}) {
  const f0 = mean(ys), stumps = [], F = ys.map(() => f0);
  const mseNow = () => mean(ys.map((v, i) => (v - F[i]) ** 2));
  const trainMSE = [mseNow()];
  for (let m = 0; m < M; m++) {
    const s = stumpFit(xs, ys.map((v, i) => v - F[i]));
    stumps.push(s);
    xs.forEach((x, i) => { F[i] += eta * (x < s.threshold ? s.left : s.right); });
    trainMSE.push(mseNow());
  }
  const predict = (x, m = M) =>
    stumps.slice(0, m).reduce((s, st) => s + eta * (x < st.threshold ? st.left : st.right), f0);
  return {f0, stumps, predict, trainMSE};
}
