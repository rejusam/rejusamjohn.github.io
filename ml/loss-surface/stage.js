// Lesson 8, "Many features and the loss surface": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, uniform, randn, mean, mse, olsFit, gdRun, mseGrad2D, standardise, eig2Sym } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, tween } from '../assets/plot.js';

export const SEED = 13;
export const N = 30;

// For each point in turn: x uniform between 100 and 300, then
// y = 50 + 0.9x + 15 × a standard normal draw.
export function lossData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = uniform(rng, 100, 300);
    xs.push(x);
    ys.push(50 + 0.9 * x + 15 * randn(rng));
  }
  return { xs, ys };
}

// The model is ŷ = a + w·x. Its MSE as a function of (a, w).
export const lossAt = (xs, ys, a, w) => mse(ys, xs.map((x) => a + w * x));

// The Hessian of the MSE in (a, w): (2/n)·[[n, Σx], [Σx, Σx²]].
export function hessian(xs) {
  const n = xs.length;
  const sx = xs.reduce((s, x) => s + x, 0);
  const sxx = xs.reduce((s, x) => s + x * x, 0);
  return [[2, (2 / n) * sx], [(2 / n) * sx, (2 / n) * sxx]];
}

// The two curvatures (eigenvalues of H) and the condition number.
export function curvatures(xs) {
  const [gmax, gmin] = eig2Sym(hessian(xs));
  return { gmax, gmin, kappa: gmax / gmin };
}

// The lowest MSE, at the least-squares line for these inputs.
export function minLoss(xs, ys) {
  const o = olsFit(xs, ys);
  return lossAt(xs, ys, o.intercept, o.slope);
}

// Both runs start from the line ŷ = 0, which is (0, 0) in either set of units.
export const START = [0, 0];
// Both runs use the same rule for the learning rate: η = 1.8 / γmax of their
// own bowl. Anything below 2 / γmax is stable.
export const ETA_FACTOR = 1.8;
export const etaFor = (xs) => ETA_FACTOR / curvatures(xs).gmax;
// "Close enough": the extra loss above the lowest has fallen below one
// millionth of its value at the start.
export const TOL = 1e-6;
// Steps drawn and animated.
export const K = 40;

// The extra loss after k steps of gradient descent from START, in closed form.
// Along each eigen-direction j of H the distance to the bottom is multiplied
// by q_j = 1 − η·γ_j every step, and the extra loss is ½ Σ γ_j c_j² q_j^(2k).
function eigenParts(xs, ys, eta) {
  const H = hessian(xs);
  const [g1, g2] = eig2Sym(H);
  const o = olsFit(xs, ys);
  const d0 = [START[0] - o.intercept, START[1] - o.slope];
  // Unit eigenvector for g1; the other is at right angles to it.
  let v = [H[0][1], g1 - H[0][0]];
  if (Math.hypot(v[0], v[1]) < 1e-12) v = [1, 0];
  const nv = Math.hypot(v[0], v[1]);
  const e1 = [v[0] / nv, v[1] / nv];
  const e2 = [-e1[1], e1[0]];
  const c1 = d0[0] * e1[0] + d0[1] * e1[1];
  const c2 = d0[0] * e2[0] + d0[1] * e2[1];
  return [
    { g: g1, c: c1, q: 1 - eta * g1, e: e1 },
    { g: g2, c: c2, q: 1 - eta * g2, e: e2 },
  ];
}
export function closedExcess(xs, ys, eta, k) {
  return eigenParts(xs, ys, eta).reduce((s, p) => s + 0.5 * p.g * p.c * p.c * Math.abs(p.q) ** (2 * k), 0);
}
// The smallest k with extra loss ≤ TOL × the starting extra loss. The extra
// loss only falls (for a stable η), so doubling then halving finds it.
export function stepsToTol(xs, ys, eta) {
  const target = TOL * closedExcess(xs, ys, eta, 0);
  const done = (k) => closedExcess(xs, ys, eta, k) <= target;
  let hi = 1;
  while (!done(hi)) hi *= 2;
  let lo = hi / 2;
  if (hi === 1) return done(0) ? 0 : 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (done(mid)) hi = mid;
    else lo = mid;
  }
  return hi;
}

// Standardised fit ŷ = a_z + w_z·z, z = (x − μ)/σ, back in the original units.
export const mapBack = ([a2, w2], mu, sigma) => ({ a: a2 - (w2 * mu) / sigma, w: w2 / sigma });

// The first K steps of a run, for drawing.
export const runPath = (xs, ys, eta, steps = K) => gdRun({ grad: mseGrad2D(xs, ys), init: START, eta, steps, tol: 0 }).path;

// Gradient descent on standardised x, run until its steps are tiny, then
// mapped back to the original units.
export function standardisedFit() {
  const { xs, ys } = lossData();
  const { z, mu, sigma } = standardise(xs);
  const r = gdRun({ grad: mseGrad2D(z, ys), init: START, eta: etaFor(z), steps: 10000, tol: 1e-12 });
  const last = r.path[r.path.length - 1];
  return { ...mapBack(last, mu, sigma), a2: last[0], w2: last[1], status: r.status, steps: r.path.length - 1 };
}

// Everything about the two runs that the page quotes.
export function runFacts() {
  const { xs, ys } = lossData();
  const { z, mu, sigma } = standardise(xs);
  const make = (data) => {
    const c = curvatures(data);
    const eta = ETA_FACTOR / c.gmax;
    const o = olsFit(data, ys);
    return { ...c, eta, steps: stepsToTol(data, ys, eta), star: [o.intercept, o.slope], lStar: minLoss(data, ys), e0: closedExcess(data, ys, eta, 0) };
  };
  return { raw: make(xs), std: make(z), mu, sigma };
}

// The plot windows. The raw window is far taller in w than it is in a, per
// unit; the standardised window uses the same scale on both axes.
export const RAW_A = [-10, 125];
export const RAW_W = [-0.1, 2.3];
// Contour levels: extra loss above the lowest, the same in both plots.
export const LEVELS = [25, 100, 400, 1600, 6400, 25600];

const PX0 = 56;
const PX1 = 372;
const PY0 = 290;
const PY1 = 26;

export function stdWindow(facts = runFacts()) {
  const [a2, w2] = facts.std.star;
  const ha = 1.12 * Math.max(Math.abs(a2 - START[0]), Math.abs(w2 - START[1]));
  const hw = (ha * (PY0 - PY1)) / (PX1 - PX0);
  return { A: [a2 - ha, a2 + ha], W: [w2 - hw, w2 + hw] };
}

// The MSE on a grid of (a, w) values: vals[j][i] is the loss at (as[i], ws[j]).
export function lossGrid(xs, ys, A, W, nx, ny) {
  const as = Array.from({ length: nx + 1 }, (_, i) => A[0] + ((A[1] - A[0]) * i) / nx);
  const ws = Array.from({ length: ny + 1 }, (_, j) => W[0] + ((W[1] - W[0]) * j) / ny);
  const vals = ws.map((w) => as.map((a) => lossAt(xs, ys, a, w)));
  return { as, ws, vals };
}

// Marching squares: the line segments where the gridded loss crosses level.
export function contourSegments({ as, ws, vals }, level) {
  const segs = [];
  const cross = (p, q, vp, vq) => {
    const t = (level - vp) / (vq - vp);
    return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
  };
  for (let j = 0; j < ws.length - 1; j++) {
    for (let i = 0; i < as.length - 1; i++) {
      const P = [[as[i], ws[j]], [as[i + 1], ws[j]], [as[i + 1], ws[j + 1]], [as[i], ws[j + 1]]];
      const V = [vals[j][i], vals[j][i + 1], vals[j + 1][i + 1], vals[j + 1][i]];
      const up = V.map((v) => v >= level);
      // Crossings on edges 0 (bottom), 1 (right), 2 (top), 3 (left).
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const f = (e + 1) % 4;
        pts.push(up[e] !== up[f] ? cross(P[e], P[f], V[e], V[f]) : null);
      }
      const hits = pts.filter(Boolean);
      if (hits.length === 2) segs.push(hits);
      else if (hits.length === 4) {
        const centre = (V[0] + V[1] + V[2] + V[3]) / 4 >= level;
        if (centre === up[0]) segs.push([pts[0], pts[1]], [pts[2], pts[3]]);
        else segs.push([pts[3], pts[0]], [pts[1], pts[2]]);
      }
    }
  }
  return segs;
}

// The gradient of the MSE at (a, w): −(2/n)·Σ eᵢ·(1, xᵢ).
export const gradAt = (xs, ys, a, w) => mseGrad2D(xs, ys)([a, w]);

// The step direction −∇L drawn as an arrow of fixed length in pixels, or
// null where the gradient is zero. X and Y are the plot's scales.
export const ARROW_PX = 42;
export function arrowVec(xs, ys, a, w, X, Y) {
  const [ga, gw] = gradAt(xs, ys, a, w);
  if (Math.hypot(ga, gw) < 1e-9) return null;
  // A move of (−ga, −gw) in (a, w), in pixels.
  const dx = X(a - ga * 1e-9) - X(a);
  const dy = Y(w - gw * 1e-9) - Y(w);
  const len = Math.hypot(dx, dy);
  if (!(len > 0)) return null;
  return [(ARROW_PX * dx) / len, (ARROW_PX * dy) / len];
}

// ---------------------------------------------------------------- formatting

const minus = (s) => s.replace('-', '−');
// Fixed decimals with a true minus sign and thousands commas.
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  const [int, frac] = s.split('.');
  const grouped = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (v < 0 && !zero ? '−' : '') + grouped + (frac ? `.${frac}` : '');
}
// Significant figures, without trailing zeros.
export const sig = (v, p = 3) => minus(String(Number(v.toPrecision(p))));
// Losses: commas for big ones, powers of ten for tiny ones.
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function fmtLoss(v) {
  if (v >= 100) return fmt(v, 1);
  if (v >= 0.01) return sig(v, 3);
  if (v <= 0) return '0';
  const e = Math.floor(Math.log10(v));
  const m = v / 10 ** e;
  return `${m.toFixed(1)} × 10${String(e).split('').map((ch) => SUP[ch]).join('')}`;
}
// Step counts: exact when small, "about … million" when huge.
export function fmtCount(k) {
  if (k >= 1e6) return `about ${sig(k / 1e6, 3)} million`;
  return fmt(k, 0);
}
// Learning rates as plain decimals to three significant figures.
export function fmtEta(eta) {
  if (eta >= 0.001) return sig(eta, 3);
  const dp = -Math.floor(Math.log10(eta)) + 2;
  return eta.toFixed(dp);
}

// The read-out rows for each step. state: { a, w } for the handle (steps 1–2),
// k for the step reached in a run (steps 3–4), view 'raw' or 'std' (step 4).
export const A_DEFAULT = 100;
export const W_DEFAULT = 0.4;
export function readout(step, state = {}, facts = runFacts(), data = lossData()) {
  const { xs, ys } = data;
  const a = state.a ?? A_DEFAULT;
  const w = state.w ?? W_DEFAULT;
  const k = clamp(state.k ?? K, 0, K);
  if (step === 0) return [`a = ${fmt(a, 1)}`, `w = ${fmt(w, 3)}`, `MSE = ${fmtLoss(lossAt(xs, ys, a, w))}`];
  if (step === 1) {
    const [ga, gw] = gradAt(xs, ys, a, w);
    return [`a = ${fmt(a, 1)}, w = ${fmt(w, 3)}`, `∂L/∂a = ${fmt(ga, 1)}`, `∂L/∂w = ${fmt(gw, 1)}`];
  }
  const runRows = (raw) => {
    const f = raw ? facts.raw : facts.std;
    const dat = raw ? xs : standardise(xs).z;
    const p = runPath(dat, ys, f.eta, k)[k];
    const ex = lossAt(dat, ys, p[0], p[1]) - f.lStar;
    const sub = raw ? '' : '_z';
    return [
      `${raw ? 'raw x' : 'standardised'}, η = ${fmtEta(f.eta)}`,
      `step ${k} of ${K}`,
      `a${sub} = ${fmt(p[0], raw ? 3 : 1)}, w${sub} = ${fmt(p[1], raw ? 3 : 2)}`,
      `extra loss = ${fmtLoss(ex)}`,
    ];
  };
  if (step === 2) return runRows(true);
  if (step === 3) return runRows(state.view === 'raw');
  if (step === 4) return [`raw x: κ = ${fmt(facts.raw.kappa, 0)}`, `standardised: κ = ${sig(facts.std.kappa, 6)}`];
  const fit = standardisedFit();
  const o = olsFit(xs, ys);
  return ['gradient descent:', `a = ${fmt(fit.a, 2)}, w = ${fmt(fit.w, 4)}`, 'least squares:', `a = ${fmt(o.intercept, 2)}, w = ${fmt(o.slope, 4)}`];
}

// Values quoted in the page text, all computed from the data.
export function pageValues() {
  const { xs, ys } = lossData();
  const f = runFacts();
  const o = olsFit(xs, ys);
  const fit = standardisedFit();
  const rawK = runPath(xs, ys, f.raw.eta)[K];
  return {
    n: String(xs.length),
    mu: fmt(f.mu, 2),
    sigma: fmt(f.sigma, 2),
    'ols-a': fmt(o.intercept, 2),
    'ols-w': fmt(o.slope, 4),
    'l-star': fmt(f.raw.lStar, 1),
    e0: fmt(f.raw.e0, 0),
    'tol-loss': sig(TOL * f.raw.e0, 2),
    'gmax-raw': fmt(f.raw.gmax, 0),
    'gmin-raw': sig(f.raw.gmin, 3),
    'kappa-raw': fmt(f.raw.kappa, 0),
    'gmax-std': sig(f.std.gmax, 6),
    'gmin-std': sig(f.std.gmin, 6),
    'kappa-std': sig(f.std.kappa, 6),
    'eta-raw': fmtEta(f.raw.eta),
    'eta-std': fmtEta(f.std.eta),
    'q-fast': sig(1 - f.raw.eta * f.raw.gmax, 3),
    'q-slow': (1 - f.raw.eta * f.raw.gmin).toFixed(7),
    'raw-steps': fmtCount(f.raw.steps),
    'std-steps': fmtCount(f.std.steps),
    k: String(K),
    'raw-a-k': fmt(rawK[0], 3),
    'raw-w-k': fmt(rawK[1], 3),
    'raw-ex-k': fmt(lossAt(xs, ys, rawK[0], rawK[1]) - f.raw.lStar, 1),
    'std-a-star': fmt(f.std.star[0], 2),
    'std-w-star': fmt(f.std.star[1], 2),
    'fit-a': fmt(fit.a, 2),
    'fit-w': fmt(fit.w, 4),
  };
}

// Words for numbers in the live region.
const say = (s) => s.replace(/−/g, 'minus ').replace(/_z/g, ' z');

// ---------------------------------------------------------------- the stage

const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE_C = '#334155';
const HALO = { stroke: SURFACE, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

const SX0 = 430;
const SX1 = 588;
const SY0 = 290;
const SY1 = 158;
const BOX = { x: 398, y: 8, row: 20 };
const LOG_LO = -4;
const LOG_HI = 5;

// Writes text into an SVG text node, drawing "a_z" and "w_z" with a lowered,
// smaller z. dy moves are used because Firefox ignores baseline-shift.
function subText(node, str) {
  const doc = node.ownerDocument;
  node.replaceChildren();
  const parts = str.split(/_z/);
  parts.forEach((part, i) => {
    if (i > 0) {
      const z = doc.createElementNS('http://www.w3.org/2000/svg', 'tspan');
      z.setAttribute('dy', '0.3em');
      z.setAttribute('font-size', '75%');
      z.textContent = 'z';
      node.appendChild(z);
    }
    if (part) {
      const t = doc.createElementNS('http://www.w3.org/2000/svg', 'tspan');
      if (i > 0) t.setAttribute('dy', '-0.3em');
      t.textContent = part;
      node.appendChild(t);
    }
  });
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const data = lossData();
  const { xs, ys } = data;
  const facts = runFacts();
  const { z } = standardise(xs);
  const fit = standardisedFit();
  const ols = olsFit(xs, ys);
  const rawPath = runPath(xs, ys, facts.raw.eta);
  const stdPath = runPath(z, ys, facts.std.eta);
  const exOf = (dat, f) => (p) => lossAt(dat, ys, p[0], p[1]) - f.lStar;
  const rawEx = rawPath.map(exOf(xs, facts.raw));
  const stdEx = stdPath.map(exOf(z, facts.std));
  const SW = stdWindow(facts);

  let ha = A_DEFAULT;
  let hw = W_DEFAULT;
  let kShown = K;
  let view = 'std';

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A contour map of the loss over the intercept a and the slope w, with the path gradient descent takes, and a small chart beside it that changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l8-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);
  const sclip = el('clipPath', { id: 'l8-side' }, defs);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1 }, sclip);
  const marker = el('marker', { id: 'l8-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 z', fill: STRONG }, marker);
  const markerM = el('marker', { id: 'l8-arrow-m', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 z', fill: MUTED }, markerM);

  // One contour plot: axes, contour lines from the gridded MSE, the bottom.
  function contourPlot(dat, A, W, nx, ny, labels, caption, lowBelow) {
    const X = linearScale(A, [PX0, PX1]);
    const Y = linearScale(W, [PY0, PY1]);
    const g = el('g', {}, svg);
    axes(g, X, Y, labels);
    const grid = lossGrid(dat, ys, A, W, nx, ny);
    const lStar = minLoss(dat, ys);
    const cg = el('g', { 'clip-path': 'url(#l8-plot)' }, g);
    LEVELS.forEach((c, idx) => {
      let d = '';
      for (const [p, q] of contourSegments(grid, lStar + c)) {
        d += `M${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}L${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`;
      }
      // Inner (lower) rings brighter.
      el('path', { d, fill: 'none', stroke: INK, 'stroke-width': 1.1, 'stroke-linecap': 'round', opacity: (0.85 - idx * 0.1).toFixed(2) }, cg);
    });
    const o = olsFit(dat, ys);
    el('rect', { x: X(o.intercept) - 4.5, y: Y(o.slope) - 4.5, width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, g);
    const lowLabel = el('text', { class: 'stage-text', x: X(o.intercept) - (lowBelow ? 0 : 8), y: Y(o.slope) + (lowBelow ? 24 : 18), 'text-anchor': lowBelow ? 'middle' : 'end', text: 'lowest', ...HALO }, g);
    lowLabel.style.fill = VIOLET;
    el('text', { class: 'value', x: PX0, y: 14, text: caption }, g);
    return { g, X, Y, star: [o.intercept, o.slope], lowLabel };
  }

  const raw = contourPlot(xs, RAW_A, RAW_W, 160, 300, { xLabel: 'intercept a', yLabel: 'slope w' }, 'raw x: axes not to scale');
  const std = contourPlot(z, SW.A, SW.W, 120, 100, { xLabel: 'intercept a_z', yLabel: 'slope w_z' }, 'standardised x: equal axes', true);

  // Start label, on both plots.
  for (const P of [raw, std]) {
    el('circle', { cx: P.X(START[0]), cy: P.Y(START[1]), r: 4, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, P.g);
    el('text', { class: 'stage-text', x: P.X(START[0]) + 8, y: P.Y(START[1]) - 8, text: 'start', ...HALO }, P.g).style.fill = STRONG;
  }

  // Step 5: the steep and gentle directions through the bottom of the raw bowl.
  const dirG = el('g', {}, svg);
  {
    const parts = eigenParts(xs, ys, facts.raw.eta);
    const bx = raw.X(raw.star[0]);
    const by = raw.Y(raw.star[1]);
    [[parts[0], 'steep', 70], [parts[1], 'gentle', 120]].forEach(([p, name, len]) => {
      const dx = raw.X(raw.star[0] + p.e[0] * 1e-6) - bx;
      const dy = raw.Y(raw.star[1] + p.e[1] * 1e-6) - by;
      const l = Math.hypot(dx, dy);
      const ux = dx / l;
      const uy = dy / l;
      el('line', { x1: bx - ux * len, y1: by - uy * len, x2: bx + ux * len, y2: by + uy * len, stroke: STRONG, 'stroke-width': 2, 'marker-start': 'url(#l8-arrow)', 'marker-end': 'url(#l8-arrow)' }, dirG);
      const tx = bx + ux * (len + 6) * (uy < -0.5 ? 1 : 1);
      const ty = by + uy * (len + 6);
      el('text', { class: 'stage-text', x: tx + (Math.abs(ux) > 0.5 ? 0 : 8), y: ty + (Math.abs(ux) > 0.5 ? -8 : 4), 'text-anchor': Math.abs(ux) > 0.5 ? 'end' : 'start', text: name, ...HALO }, dirG).style.fill = STRONG;
    });
  }

  // The path of a run, drawn as dots joined by dashed arcs.
  const trailG = el('g', { 'clip-path': 'url(#l8-plot)' }, svg);
  const ball = el('circle', { r: 6.5, fill: CYAN, stroke: BG, 'stroke-width': 2 }, svg);

  // Steps 1–2: a handle for (a, w), dragged with the pointer; the sliders
  // below do the same from the keyboard.
  const arrowG = el('g', {}, svg);
  const upArrow = el('line', { stroke: MUTED, 'stroke-width': 1.8, 'stroke-dasharray': '4 3', 'marker-end': 'url(#l8-arrow-m)' }, arrowG);
  const upLabel = el('text', { class: 'stage-text', text: 'gradient: uphill', ...HALO }, arrowG);
  upLabel.style.fill = MUTED;
  const downArrow = el('line', { stroke: STRONG, 'stroke-width': 2.4, 'marker-end': 'url(#l8-arrow)' }, arrowG);
  const downLabel = el('text', { class: 'stage-text', text: 'step: downhill', ...HALO }, arrowG);
  downLabel.style.fill = STRONG;
  const handleG = el('g', {}, svg);
  const knob = el('circle', { r: 7, fill: CYAN, stroke: BG, 'stroke-width': 2 }, handleG);
  const grab = el('circle', { class: 'handle', r: 18, fill: 'transparent', 'aria-hidden': 'true' }, handleG);

  // Side panel: the data and the line ŷ = a + w·x (steps 1, 2 and 6).
  const DX = linearScale([80, 320], [SX0, SX1]);
  const DY = linearScale([0, 400], [SY0, SY1]);
  const scatG = el('g', {}, svg);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1, fill: 'none', stroke: LINE_C }, scatG);
  xs.forEach((x, i) => el('circle', { cx: DX(x), cy: DY(ys[i]), r: 2.4, fill: MUTED }, scatG));
  const scatLine = el('line', { stroke: CYAN, 'stroke-width': 2, 'clip-path': 'url(#l8-side)' }, scatG);
  const olsLine = el('line', { x1: DX(80), x2: DX(320), y1: DY(ols.intercept + ols.slope * 80), y2: DY(ols.intercept + ols.slope * 320), stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 5', 'clip-path': 'url(#l8-side)' }, scatG);
  const scatTitle = el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'line ŷ = a + w·x' }, scatG);
  const scatSub = el('text', { class: 'value', x: SX0, y: SY1 - 28, text: 'cyan: this run' }, scatG);
  el('text', { class: 'stage-text', x: SX0, y: SY0 + 16, text: '100' }, scatG);
  el('text', { class: 'stage-text', x: SX1, y: SY0 + 16, 'text-anchor': 'end', text: '300' }, scatG);
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 16, 'text-anchor': 'middle', text: 'x' }, scatG);
  el('line', { x1: DX(100), x2: DX(100), y1: SY0, y2: SY0 + 4, stroke: LINE_C }, scatG);
  el('line', { x1: DX(300), x2: DX(300), y1: SY0, y2: SY0 + 4, stroke: LINE_C }, scatG);

  // Side panel: extra loss at each step, log scale (steps 3 and 4).
  const KX = linearScale([0, K], [SX0, SX1]);
  const LY = linearScale([LOG_LO, LOG_HI], [SY0, SY1]);
  const lyOf = (v) => LY(clamp(Math.log10(Math.max(v, 1e-300)), LOG_LO - 0.5, LOG_HI));
  const exG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE_C }, exG);
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE_C }, exG);
  [[4, '10⁴'], [0, '1'], [-4, '10⁻⁴']].forEach(([e, t]) => {
    el('line', { x1: SX0 - 4, x2: SX0, y1: LY(e), y2: LY(e), stroke: LINE_C }, exG);
    el('text', { class: 'stage-text', x: SX0 - 7, y: LY(e) + 4, 'text-anchor': 'end', text: t }, exG);
  });
  const tolY = lyOf(TOL * facts.raw.e0);
  el('line', { x1: SX0, x2: SX1, y1: tolY, y2: tolY, stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '2 3' }, exG);
  el('text', { class: 'stage-text', x: SX1, y: tolY - 5, 'text-anchor': 'end', text: 'target', ...HALO }, exG).style.fill = MUTED;
  el('text', { class: 'stage-text', x: SX0, y: SY0 + 16, 'text-anchor': 'middle', text: '0' }, exG);
  el('text', { class: 'stage-text', x: SX1, y: SY0 + 16, 'text-anchor': 'middle', text: String(K) }, exG);
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 16, 'text-anchor': 'middle', text: 'step' }, exG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'extra loss (log)' }, exG);
  const exPlot = el('g', { 'clip-path': 'url(#l8-side)' }, exG);
  const rawLine = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 1.8, 'stroke-dasharray': '5 4' }, exPlot);
  const stdLine = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 1.8 }, exPlot);
  const rawTag = el('text', { class: 'stage-text', 'text-anchor': 'end', text: 'raw', ...HALO }, exG);
  rawTag.style.fill = VIOLET;
  // The standardised run's key sits in the empty lower left of the chart.
  const stdTag = el('g', {}, exG);
  el('line', { x1: SX0 + 6, x2: SX0 + 20, y1: LY(-3.2), y2: LY(-3.2), stroke: CYAN, 'stroke-width': 1.8 }, stdTag);
  el('text', { class: 'stage-text', x: SX0 + 24, y: LY(-3.2) + 4, text: 'standardised' }, stdTag).style.fill = CYAN;

  // Side panel: the curvature table (step 5).
  const tabG = el('g', {}, svg);
  {
    const cols = [SX0 + 40, SX0 + 98, SX1];
    const rowsT = [
      ['', 'raw', 'stand.'],
      ['γmax', fmt(facts.raw.gmax, 0), sig(facts.std.gmax, 6)],
      ['γmin', sig(facts.raw.gmin, 3), sig(facts.std.gmin, 6)],
      ['κ', fmt(facts.raw.kappa, 0), sig(facts.std.kappa, 6)],
    ];
    rowsT.forEach((r, i) => {
      const y = SY1 + 22 + i * 30;
      el('text', { class: 'stage-text', x: SX0, y, text: r[0] }, tabG);
      el('text', { class: 'stage-text', x: cols[1], y, 'text-anchor': 'end', text: r[1] }, tabG).style.fill = i === 0 ? MUTED : STRONG;
      el('text', { class: 'stage-text', x: cols[2], y, 'text-anchor': 'end', text: r[2] }, tabG).style.fill = i === 0 ? MUTED : STRONG;
    });
    el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'curvatures' }, tabG);
  }

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE_C }, box);
  const rows = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));
  // The standardised axis labels carry a subscript z.
  for (const t of std.g.querySelectorAll('.axes text')) if (/_z/.test(t.textContent)) subText(t, t.textContent);

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, extra, lo, hi, stepv, val) => {
    const label = make('label');
    const input = make('input', { type: 'range', min: lo, max: hi, step: stepv, value: val });
    input.style.width = '130px';
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    label.append(name, make('span', { className: 'vh', textContent: extra }), ' ', input, ' ', out);
    return { label, input, out };
  };
  const handleGroup = make('div');
  const aS = slider('a', ', the intercept', RAW_A[0], RAW_A[1], 0.5, A_DEFAULT);
  const wS = slider('w', ', the slope', RAW_W[0], RAW_W[1], 0.005, W_DEFAULT);
  handleGroup.append(aS.label, wS.label);
  const runGroup = make('div');
  const runBtn = make('button', { type: 'button', textContent: 'Run again' });
  runGroup.append(runBtn);
  const viewGroup = make('div');
  const rawBtn = make('button', { type: 'button', textContent: 'Raw x' });
  const stdBtn = make('button', { type: 'button', textContent: 'Standardised x' });
  viewGroup.append(rawBtn, stdBtn);
  controls.append(handleGroup, runGroup, viewGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  let runToken = 0;
  const LAYERS = { raw: 0, std: 0, dir: 0, trail: 0, arrow: 0, handle: 0, scat: 0, ex: 0, tab: 0 };
  const target = (i) => {
    if (i === 0) return { ...LAYERS, raw: 1, handle: 1, scat: 1 };
    if (i === 1) return { ...LAYERS, raw: 1, handle: 1, arrow: 1, scat: 1 };
    if (i === 2) return { ...LAYERS, raw: 1, trail: 1, ex: 1 };
    if (i === 3) return view === 'raw' ? { ...LAYERS, raw: 1, trail: 1, ex: 1 } : { ...LAYERS, std: 1, trail: 1, ex: 1 };
    if (i === 4) return { ...LAYERS, raw: 1, dir: 1, tab: 1 };
    return { ...LAYERS, std: 1, trail: 1, scat: 1 };
  };
  let s = target(0);

  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };

  // Which run is on show.
  const runIsRaw = () => step === 2 || (step === 3 && view === 'raw');
  const curPlot = () => (runIsRaw() ? raw : std);
  const curPath = () => (runIsRaw() ? rawPath : stdPath);

  // A hop from p to q. On the raw map it is bowed to one side so hops that
  // retrace each other stay apart; on the round bowl every hop lies on one
  // line through the start and the bottom, so it is drawn straight.
  const hopD = (P, p, q) => {
    const x0 = P.X(p[0]);
    const y0 = P.Y(p[1]);
    const x1 = P.X(q[0]);
    const y1 = P.Y(q[1]);
    const len = Math.hypot(x1 - x0, y1 - y0) || 1;
    let nx = -(y1 - y0) / len;
    let ny = (x1 - x0) / len;
    if (nx - ny < 0) { nx = -nx; ny = -ny; }
    const bow = P === std ? 0 : 6 + 0.15 * len;
    return { x0, y0, x1, y1, qx: (x0 + x1) / 2 + nx * bow, qy: (y0 + y1) / 2 + ny * bow };
  };

  function drawTrail(upto) {
    trailG.replaceChildren();
    const P = curPlot();
    const path = curPath();
    for (let k = 1; k <= upto; k++) {
      const h = hopD(P, path[k - 1], path[k]);
      el('path', { d: `M${h.x0.toFixed(2)},${h.y0.toFixed(2)} Q${h.qx.toFixed(2)},${h.qy.toFixed(2)} ${h.x1.toFixed(2)},${h.y1.toFixed(2)}`, fill: 'none', stroke: CYAN, 'stroke-width': 1.3, 'stroke-dasharray': '4 3', opacity: 0.75 }, trailG);
    }
    for (let k = 0; k < upto; k++) el('circle', { cx: P.X(path[k][0]), cy: P.Y(path[k][1]), r: 3.5, fill: CYAN, opacity: 0.4 }, trailG);
  }

  function drawEx() {
    const kRaw = step === 2 || (step === 3 && view === 'raw') ? kShown : K;
    const kStd = step === 3 && view === 'std' ? kShown : step === 3 ? K : -1;
    const line = (list, upto) => {
      let d = '';
      for (let k = 0; k <= upto; k++) d += `${k ? 'L' : 'M'}${KX(k).toFixed(2)},${lyOf(list[k]).toFixed(2)}`;
      return d;
    };
    rawLine.setAttribute('d', line(rawEx, kRaw));
    set(rawTag, { x: SX1 - 2, y: lyOf(rawEx[K]) - 7 });
    if (kStd >= 0) {
      stdLine.setAttribute('d', line(stdEx, kStd));
      stdLine.style.display = '';
      stdTag.style.display = '';
    } else {
      stdLine.style.display = 'none';
      stdTag.style.display = 'none';
    }
  }

  function render() {
    op(raw.g, s.raw);
    op(std.g, s.std);
    op(dirG, s.dir);
    op(trailG, s.trail);
    op(arrowG, s.arrow);
    op(handleG, s.handle);
    op(scatG, s.scat);
    op(exG, s.ex);
    op(tabG, s.tab);
    // In step 2 the arrow labels need the room; the violet square stays.
    raw.lowLabel.style.display = step === 1 ? 'none' : '';

    // Handle and arrows (steps 1–2).
    const hx = raw.X(ha);
    const hy = raw.Y(hw);
    set(knob, { cx: hx, cy: hy });
    set(grab, { cx: hx, cy: hy });
    const v = arrowVec(xs, ys, ha, hw, raw.X, raw.Y);
    if (v) {
      downArrow.style.display = '';
      upArrow.style.display = '';
      set(downArrow, { x1: hx + v[0] * 0.2, y1: hy + v[1] * 0.2, x2: hx + v[0], y2: hy + v[1] });
      set(upArrow, { x1: hx - v[0] * 0.2, y1: hy - v[1] * 0.2, x2: hx - v[0], y2: hy - v[1] });
      // Labels beside the arrow tips, kept inside the plot.
      const place = (label, sx, sy) => {
        const tx = hx + sx * 1.15;
        const ty = hy + sy * 1.15;
        const right = tx < (PX0 + PX1) / 2;
        label.setAttribute('text-anchor', right ? 'start' : 'end');
        set(label, { x: clamp(tx + (right ? 8 : -8), PX0 + 4, PX1 - 4), y: clamp(ty + 4, PY1 + 12, PY0 - 6) });
      };
      place(downLabel, v[0], v[1]);
      place(upLabel, -v[0], -v[1]);
    } else {
      downArrow.style.display = 'none';
      upArrow.style.display = 'none';
    }

    // Path and ball.
    if (step === 2 || step === 3 || step === 5) {
      const upto = step === 5 ? K : kShown;
      drawTrail(upto);
      const p = curPath()[upto];
      set(ball, { cx: curPlot().X(p[0]), cy: curPlot().Y(p[1]) });
      ball.style.display = '';
    } else {
      ball.style.display = 'none';
    }
    drawEx();

    // Side panel line.
    const la = step === 5 ? fit.a : ha;
    const lw = step === 5 ? fit.w : hw;
    set(scatLine, { x1: DX(80), x2: DX(320), y1: DY(la + lw * 80), y2: DY(la + lw * 320) });
    olsLine.style.display = step === 5 ? '' : 'none';
    scatTitle.textContent = step === 5 ? 'dashed: Lesson 6' : 'line ŷ = a + w·x';
    scatSub.style.display = step === 5 ? '' : 'none';

    // Read-out.
    const texts = readout(step, { a: ha, w: hw, k: kShown, view }, facts, data);
    rows.forEach((r, k) => {
      subText(r, texts[k] || '');
      r.style.display = k < texts.length ? '' : 'none';
    });
    let bw = 186;
    try {
      const lens = rows.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((x) => x > 0)) bw = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const boxX = Math.min(BOX.x, 598 - bw);
    set(boxRect, { x: boxX, y: BOX.y, width: bw, height: 16 + BOX.row * texts.length });
    rows.forEach((r, k) => set(r, { x: boxX + 12, y: BOX.y + 8 + BOX.row * k + BOX.row / 2 + 5 }));

    aS.out.textContent = fmt(ha, 1);
    wS.out.textContent = fmt(hw, 3);
    const L = fmtLoss(lossAt(xs, ys, ha, hw));
    aS.input.setAttribute('aria-valuetext', say(`a ${fmt(ha, 1)}, MSE ${L}`));
    wS.input.setAttribute('aria-valuetext', say(`w ${fmt(hw, 3)}, MSE ${L}`));
    rawBtn.setAttribute('aria-pressed', String(view === 'raw'));
    stdBtn.setAttribute('aria-pressed', String(view === 'std'));
  }

  function animateTo(to, ms) {
    const my = ++token;
    const from = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = from[key] + (to[key] - from[key]) * kt;
      s = next;
      render();
    }, { reduced });
  }

  // --- Words for the live region.
  const spoken = (i) => {
    const r = readout(i, { a: ha, w: hw, k: kShown, view }, facts, data);
    if (i === 0) return say(`${r.join(', ')}.`);
    if (i === 1) return say(`${r.join(', ')}. The step goes the opposite way to the gradient, downhill.`);
    if (i === 2 || i === 3) {
      const isRaw = i === 2 || view === 'raw';
      const f = isRaw ? facts.raw : facts.std;
      return say(`${r.join(', ')}. To reach the target takes ${fmtCount(f.steps)} steps.`);
    }
    if (i === 4) return say(`Raw x: steepest curvature ${fmt(facts.raw.gmax, 0)}, gentlest ${sig(facts.raw.gmin, 3)}, condition number ${fmt(facts.raw.kappa, 0)}. Standardised x: both curvatures ${sig(facts.std.gmax, 6)}, condition number ${sig(facts.std.kappa, 6)}.`);
    return say(`Mapped back to the original units, gradient descent gives a ${fmt(fit.a, 2)} and w ${fmt(fit.w, 4)}. Least squares gives a ${fmt(ols.intercept, 2)} and w ${fmt(ols.slope, 4)}.`);
  };
  const sayLater = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction.
  const setHandle = (a, w) => {
    ha = clamp(Math.round(a * 2) / 2, RAW_A[0], RAW_A[1]);
    hw = clamp(Math.round(w * 200) / 200, RAW_W[0], RAW_W[1]);
    aS.input.value = String(ha);
    wS.input.value = String(hw);
    render();
    sayLater(spoken(step));
  };
  aS.input.addEventListener('input', () => setHandle(Number(aS.input.value), hw));
  wS.input.addEventListener('input', () => setHandle(ha, Number(wS.input.value)));
  // Pointer drag on the plot (keyboard users have the sliders).
  {
    let active = false;
    const toUser = (e) => {
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      return pt.matrixTransform(ctm.inverse());
    };
    const move = (e) => {
      const p = toUser(e);
      if (p) setHandle(raw.X.invert(clamp(p.x, PX0, PX1)), raw.Y.invert(clamp(p.y, PY1, PY0)));
    };
    grab.style.touchAction = 'none';
    grab.addEventListener('pointerdown', (e) => {
      if (step > 1) return;
      active = true;
      grab.setPointerCapture(e.pointerId);
      e.preventDefault();
      move(e);
    });
    grab.addEventListener('pointermove', (e) => { if (active) move(e); });
    const end = (e) => {
      active = false;
      if (grab.hasPointerCapture && grab.hasPointerCapture(e.pointerId)) grab.releasePointerCapture(e.pointerId);
    };
    grab.addEventListener('pointerup', end);
    grab.addEventListener('pointercancel', end);
  }

  // Play a run: the ball hops through the first K steps.
  async function playRun(animate) {
    const my = ++runToken;
    if (!animate || reduced) {
      kShown = K;
      render();
      return;
    }
    const P = curPlot();
    const path = curPath();
    kShown = 0;
    render();
    for (let k = 1; k <= K; k++) {
      const h = hopD(P, path[k - 1], path[k]);
      // eslint-disable-next-line no-await-in-loop
      await tween(k < 8 ? 320 : 90, (t) => {
        if (my !== runToken) return;
        const u = 1 - t;
        set(ball, { cx: u * u * h.x0 + 2 * u * t * h.qx + t * t * h.x1, cy: u * u * h.y0 + 2 * u * t * h.qy + t * t * h.y1 });
      }, { reduced });
      if (my !== runToken) return;
      kShown = k;
      render();
    }
    announce(spoken(step));
  }
  runBtn.addEventListener('click', () => { playRun(true); });
  const setView = (v) => {
    if (view === v) return;
    view = v;
    runToken++;
    kShown = K;
    animateTo(target(step), reduced ? 0 : 300);
    render();
    announce(spoken(step));
  };
  rawBtn.addEventListener('click', () => setView('raw'));
  stdBtn.addEventListener('click', () => setView('std'));

  // Sentences on the page that quote values from the data (hidden until filled).
  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const node of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...node.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) node.hidden = false;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      runToken++;
      kShown = K;
      if (i === 3) view = 'std';
      showGroup(handleGroup, i <= 1);
      showGroup(runGroup, i === 2 || i === 3);
      showGroup(viewGroup, i === 3);
      announce(spoken(i));
      const to = target(i);
      const fade = () => {
        if (!animate) {
          token++;
          s = { ...to };
          render();
          return Promise.resolve();
        }
        return animateTo(to, 450);
      };
      if ((i === 2 || i === 3) && animate && !reduced) {
        kShown = 0;
        return fade().then(() => playRun(true));
      }
      return fade();
    },
  };
}
