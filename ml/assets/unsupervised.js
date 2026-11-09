// k-means and 2D PCA numerics. Pure functions, no DOM.
import {eig2Sym} from './maths.js';

const d2 = (p, c) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;

// Sum (not mean) of squared distances to the assigned centres.
export function inertia(X, centres, labels) {
  return X.reduce((s, p, i) => s + d2(p, centres[labels[i]]), 0);
}

// Nearest centre; ties go to the lower index.
function assign(X, centres) {
  return X.map(p => {
    let best = 0, bd = d2(p, centres[0]);
    for (let k = 1; k < centres.length; k++) {
      const d = d2(p, centres[k]);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  });
}

// Mean of the assigned points; an empty cluster keeps its old centre.
function update(X, centres, labels) {
  const sums = centres.map(() => [0, 0]), counts = centres.map(() => 0);
  X.forEach((p, i) => { sums[labels[i]][0] += p[0]; sums[labels[i]][1] += p[1]; counts[labels[i]]++; });
  return centres.map((c, k) => counts[k] ? [sums[k][0] / counts[k], sums[k][1] / counts[k]] : c.slice());
}

// Lloyd's algorithm from the given centres. Stops when an assign step changes no label.
export function kmeans(X, init, {maxIter = 100} = {}) {
  let centres = init.map(c => c.slice());
  const states = [{phase: 'init', centres: centres.map(c => c.slice()), labels: null, inertia: null}];
  let labels = null, iters = 0, converged = false;
  while (iters < maxIter) {
    const next = assign(X, centres);
    iters++;
    const same = labels !== null && next.every((l, i) => l === labels[i]);
    labels = next;
    states.push({phase: 'assign', centres: centres.map(c => c.slice()), labels: labels.slice(), inertia: inertia(X, centres, labels)});
    if (same) { converged = true; break; }
    centres = update(X, centres, labels);
    states.push({phase: 'update', centres: centres.map(c => c.slice()), labels: labels.slice(), inertia: inertia(X, centres, labels)});
  }
  return {states, centres, labels, inertia: inertia(X, centres, labels), iters, converged};
}

const meanXY = X => {
  const n = X.length;
  return [X.reduce((s, p) => s + p[0], 0) / n, X.reduce((s, p) => s + p[1], 0) / n];
};

// Sample covariance (n - 1).
export function cov2(X) {
  const [mx, my] = meanXY(X), n = X.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of X) {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  return [[sxx / (n - 1), sxy / (n - 1)], [sxy / (n - 1), syy / (n - 1)]];
}

// Unit vector with its largest-magnitude component positive.
function signed(v) {
  const [a, b] = v, h = Math.hypot(a, b);
  let u = [a / h, b / h];
  if (Math.abs(u[0]) >= Math.abs(u[1]) ? u[0] < 0 : u[1] < 0) u = [-u[0], -u[1]];
  return u;
}

export function pca2(X) {
  const S = cov2(X), mean = meanXY(X);
  const [[sxx, sxy], [, syy]] = S;
  const gammas = eig2Sym(S);
  let v1;
  if (sxy === 0) v1 = sxx >= syy ? [1, 0] : [0, 1];
  else {
    // (S - gI)v = 0 gives v = (sxy, g - sxx) or (g - syy, sxy); take the longer.
    const a = [sxy, gammas[0] - sxx], b = [gammas[0] - syy, sxy];
    v1 = Math.hypot(...a) >= Math.hypot(...b) ? a : b;
  }
  v1 = signed(v1);
  const vecs = [v1, signed([-v1[1], v1[0]])];
  const tot = gammas[0] + gammas[1];
  return {mean, S, gammas, vecs, explained: [gammas[0] / tot, gammas[1] / tot]};
}

// Sample variance (n - 1) of the projections onto (cos t, sin t).
export function projVar(X, theta) {
  const [mx, my] = meanXY(X), c = Math.cos(theta), s = Math.sin(theta);
  return X.reduce((a, p) => a + ((p[0] - mx) * c + (p[1] - my) * s) ** 2, 0) / (X.length - 1);
}

// Mean (1/n) squared perpendicular distance to the line through the mean at angle theta.
export function perpMSE(X, theta) {
  const [mx, my] = meanXY(X), c = Math.cos(theta), s = Math.sin(theta);
  return X.reduce((a, p) => a + (-(p[0] - mx) * s + (p[1] - my) * c) ** 2, 0) / X.length;
}
