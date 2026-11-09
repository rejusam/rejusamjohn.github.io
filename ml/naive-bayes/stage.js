// Lesson L3.3, "Naive Bayes": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-naive-bayes.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, randn, mean } from '../assets/maths.js';
import { gaussianNB, nbPosterior, accuracy } from '../assets/classify.js';
import { linearScale, clamp, el, fmt, axes, tween, classMarker, regionMap, draggable, trianglePoints } from '../assets/plot.js';

export const SEED_TRAIN = 61;
export const SEED_CORR = 62;
export const SEED_TEST = 63;
export const SEED_CORR_TEST = 64;
const SHAPE = { mu0: [2, 2], mu1: [4.5, 3.5], sd0: 1, sd1: 0.7 };
export const RHO = 0.85;
// The band used for counting in step 1: 3 ≤ feature 1 < 4.
export const BIN = [3, 4];
// Prior slider values for class 1: 0.10, 0.15, ..., 0.90.
export const PRIORS = Array.from({ length: 17 }, (_, i) => Number((0.1 + 0.05 * i).toFixed(2)));
export const X1_0 = 3.5;
export const Q0 = [3.5, 2.5];
export const X_DOM = [-1, 7];
export const Y_DOM = [-1.5, 5.5];
// Region map cells: square, 4 user units a side.
export const GRID = [64, 56];

// Two classes, class 0 rows first, all draws from one mulberry32 stream.
// Each class has its own SD s; feature 2 has correlation rho with feature 1.
// With sd0 = sd1 this is exactly blobs2 from classify.js.
export function classData({ n0, n1, mu0, mu1, sd0, sd1, rho = 0, seed }) {
  const rng = mulberry32(seed);
  const c = Math.sqrt(1 - rho * rho);
  const X = [];
  const y = [];
  const draw = (mu, s, cls) => {
    const z1 = randn(rng);
    const z2 = randn(rng);
    X.push([mu[0] + s * z1, mu[1] + s * (rho * z1 + c * z2)]);
    y.push(cls);
  };
  for (let i = 0; i < n0; i++) draw(mu0, sd0, 0);
  for (let i = 0; i < n1; i++) draw(mu1, sd1, 1);
  return { X, y };
}

export const trainData = () => classData({ n0: 60, n1: 40, ...SHAPE, seed: SEED_TRAIN });
export const testData = () => classData({ n0: 600, n1: 400, ...SHAPE, seed: SEED_TEST });
export const corrTrain = () => classData({ n0: 60, n1: 40, ...SHAPE, rho: RHO, seed: SEED_CORR });
export const corrTest = () => classData({ n0: 600, n1: 400, ...SHAPE, rho: RHO, seed: SEED_CORR_TEST });

const TR = trainData();
const TE = testData();
const CTR = corrTrain();
const CTE = corrTest();
const M = gaussianNB(TR.X, TR.y);
const CM = gaussianNB(CTR.X, CTR.y);
export const model = () => M;
export const corrModel = () => CM;

// The same model with the prior of class 1 set by hand.
export const withPrior = (m, p1) => ({ ...m, prior: [1 - p1, p1] });
// Feature 1 only.
export const oneD = (m) => ({ prior: m.prior, mu: m.mu.map((r) => [r[0]]), vari: m.vari.map((r) => [r[0]]) });

export const normalPdf = (x, mu, v) => Math.exp(-((x - mu) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
// P(c) × p(x | c) for each class, one feature.
export const weighted = (m1, x) => [0, 1].map((c) => m1.prior[c] * normalPdf(x, m1.mu[c][0], m1.vari[c][0]));

// Where the posterior of class 1 is 1/2 on feature 1: the log-odds is a x² + b x + c.
export function boundaries1D(p1) {
  const [m0, m1] = [M.mu[0][0], M.mu[1][0]];
  const [v0, v1] = [M.vari[0][0], M.vari[1][0]];
  const a = 1 / (2 * v0) - 1 / (2 * v1);
  const b = m1 / v1 - m0 / v0;
  const c = Math.log(p1 / (1 - p1)) - 0.5 * Math.log(v1 / v0) - m1 * m1 / (2 * v1) + m0 * m0 / (2 * v0);
  if (a === 0) return [-c / b];
  const disc = b * b - 4 * a * c;
  if (disc < 0) return [];
  // The stable form of the quadratic formula.
  const q = -0.5 * (b + Math.sign(b) * Math.sqrt(disc));
  return [q / a, c / q].sort((u, v) => u - v);
}

// Region map cells on the 2D plot, row 0 at the top (as plot.js lays them out).
let cellsCache = null;
function cells() {
  if (!cellsCache) {
    cellsCache = [];
    for (let j = 0; j < GRID[1]; j++) {
      for (let i = 0; i < GRID[0]; i++) {
        cellsCache.push([
          X_DOM[0] + ((i + 0.5) / GRID[0]) * (X_DOM[1] - X_DOM[0]),
          Y_DOM[1] + ((j + 0.5) / GRID[1]) * (Y_DOM[0] - Y_DOM[1]),
        ]);
      }
    }
  }
  return cellsCache;
}
// Number of cells predicted class 1, with the prior of class 1 set to p1.
export function regionCount(p1) {
  const m = withPrior(M, p1);
  return cells().filter((x) => nbPosterior(m, x)[1] >= 0.5).length;
}

export function binCounts() {
  let n0 = 0;
  let n1 = 0;
  TR.X.forEach((x, i) => {
    if (x[0] >= BIN[0] && x[0] < BIN[1]) TR.y[i] ? n1++ : n0++;
  });
  return { n0, n1, post1: n1 / (n0 + n1) };
}

// The rule built from the true shapes: bivariate normal log density.
const trueLogDen = (x, mu, s, rho) => {
  const a = (x[0] - mu[0]) / s;
  const b = (x[1] - mu[1]) / s;
  const qf = (a * a - 2 * rho * a * b + b * b) / (1 - rho * rho);
  return -Math.log(2 * Math.PI * s * s * Math.sqrt(1 - rho * rho)) - qf / 2;
};
const trueRule = (x, rho) =>
  (Math.log(0.4) + trueLogDen(x, SHAPE.mu1, SHAPE.sd1, rho) > Math.log(0.6) + trueLogDen(x, SHAPE.mu0, SHAPE.sd0, rho) ? 1 : 0);

// Test accuracy, mean confidence (the posterior of the class picked) and the true-shape rule's accuracy.
const calibCache = new Map();
export function calib(which, p1 = null) {
  const key = `${which}:${p1}`;
  if (!calibCache.has(key)) {
    const corr = which === 'corr';
    const te = corr ? CTE : TE;
    const m = p1 === null ? (corr ? CM : M) : withPrior(corr ? CM : M, p1);
    const post = te.X.map((x) => nbPosterior(m, x));
    calibCache.set(key, {
      acc: accuracy(te.y, post.map((p) => (p[1] >= 0.5 ? 1 : 0))),
      conf: mean(post.map((p) => Math.max(p[0], p[1]))),
      bayes: accuracy(te.y, te.X.map((x) => trueRule(x, corr ? RHO : 0))),
    });
  }
  return calibCache.get(key);
}

// Sample correlation of the two features within one class.
export function sampleCorr({ X, y }, cls) {
  const rows = X.filter((_, i) => y[i] === cls);
  const m0 = mean(rows.map((r) => r[0]));
  const m1 = mean(rows.map((r) => r[1]));
  let sxy = 0, sxx = 0, syy = 0;
  for (const r of rows) {
    sxy += (r[0] - m0) * (r[1] - m1);
    sxx += (r[0] - m0) ** 2;
    syy += (r[1] - m1) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
// A density or product: three decimals, or "< 0.001" when smaller.
export const dens = (v) => (v >= 0.0005 ? v.toFixed(3) : '< 0.001');
const p2 = (v) => v.toFixed(2);

// The parts of the 2D posterior at x: P(c), p(x₁|c), p(x₂|c) for each class.
export function parts(x, m = M) {
  return [0, 1].map((c) => ({
    prior: m.prior[c],
    d1: normalPdf(x[0], m.mu[c][0], m.vari[c][0]),
    d2: normalPdf(x[1], m.mu[c][1], m.vari[c][1]),
  }));
}

export function readout(step, { prior1 = 0.4, x1 = X1_0, q = Q0, data = 'corr' } = {}) {
  if (step === 0) {
    const b = binCounts();
    return [`3 ≤ x₁ < 4`, `circles (class 0): ${b.n0}`, `triangles (class 1): ${b.n1}`, `P(1 | band) = ${b.n1}/${b.n0 + b.n1} = ${p2(b.post1)}`];
  }
  if (step === 1 || step === 2) {
    const p1 = step === 1 ? 0.4 : prior1;
    const m1 = oneD(withPrior(M, p1));
    const w = weighted(m1, x1);
    const post = nbPosterior(m1, [x1])[1];
    if (step === 1) return [`x₁ = ${p2(x1)}`, `0.60 × p(x₁|0) = ${dens(w[0])}`, `0.40 × p(x₁|1) = ${dens(w[1])}`, `P(1 | x₁) = ${p2(post)}`];
    return [`P(class 1) = ${p2(p1)}`, `P(class 0) = ${p2(1 - p1)}`, `boundary: x₁ = ${p2(boundaries1D(p1)[0])}`, `P(1 | x₁ = ${p2(x1)}) = ${p2(post)}`];
  }
  if (step === 3 || step === 4) {
    const m = step === 3 ? M : withPrior(M, prior1);
    const post = nbPosterior(m, q)[1];
    if (step === 3) {
      const pr = parts(q, m);
      const line = (c) => `class ${c}: ${p2(pr[c].prior)}×${dens(pr[c].d1)}×${dens(pr[c].d2)}`;
      return [`new case (${p2(q[0])}, ${p2(q[1])})`, line(0), line(1), `P(class 1 | x) = ${p2(post)}`];
    }
    return [`P(class 1) = ${p2(prior1)}`, `new case: P(1 | x) = ${p2(post)}`, `test accuracy: ${pct(calib('main', prior1).acc)}`];
  }
  const c = calib(data);
  return [data === 'corr' ? 'correlated features' : 'independent features', `test accuracy: ${pct(c.acc)}`, `mean confidence: ${pct(c.conf)}`, `true-shape rule: ${pct(c.bayes)}`];
}

export const PIN_KEYS = ['b1', 'b1-far', 'b1-double', 'x1-post', 'q-post', 'main-acc', 'main-conf', 'corr-acc', 'corr-conf', 'corr-bayes', 'm0x', 's1x'];
export const PAGE_PIN = {
  'b1': '3.62', 'b1-far': '8.2', 'b1-double': '3.30', 'x1-post': '0.34', 'q-post': '0.28',
  'main-acc': '96.7%', 'main-conf': '97.5%', 'corr-acc': '90.1%', 'corr-conf': '95.7%', 'corr-bayes': '93.8%',
  'm0x': '2.13', 's1x': '0.53',
};
// Three training values, rounded, so a change to the generator cannot pass unnoticed.
export const PINNED = [1.9411667106, 4.7514455942, 4.8928076773];

export function pageValues() {
  const b = binCounts();
  const main = calib('main');
  const corr = calib('corr');
  const sdv = (c, j) => p2(Math.sqrt(M.vari[c][j]));
  return {
    'bin-n0': String(b.n0),
    'bin-n1': String(b.n1),
    'bin-n': String(b.n0 + b.n1),
    'bin-post': p2(b.post1),
    'bin-share0': p2(b.n0 / TR.X.length),
    'bin-share1': p2(b.n1 / TR.X.length),
    'n-test': String(TE.X.length),
    'b1': boundaries1D(0.4)[0].toFixed(2),
    'b1-far': boundaries1D(0.4)[1].toFixed(1),
    'b1-double': boundaries1D(0.8)[0].toFixed(2),
    'x1-post': p2(nbPosterior(oneD(M), [X1_0])[1]),
    'q-post': p2(nbPosterior(M, Q0)[1]),
    'm0x': p2(M.mu[0][0]), 'm0y': p2(M.mu[0][1]),
    'm1x': p2(M.mu[1][0]), 'm1y': p2(M.mu[1][1]),
    's0x': sdv(0, 0), 's0y': sdv(0, 1),
    's1x': sdv(1, 0), 's1y': sdv(1, 1),
    'main-acc': pct(main.acc),
    'main-conf': pct(main.conf),
    'corr-acc': pct(corr.acc),
    'corr-conf': pct(corr.conf),
    'corr-bayes': pct(corr.bayes),
    'corr-gap': ((corr.conf - corr.acc) * 100).toFixed(1),
    'corr-cost': ((corr.bayes - corr.acc) * 100).toFixed(1),
    'corr-r0': p2(sampleCorr(CTR, 0)),
    'corr-r1': p2(sampleCorr(CTR, 1)),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';

// One feature (steps 1–3): bars and curves above, the rug, the posterior below.
const X1_DOM = [-1, 7.5];
const AX0 = 56;
const AX1 = 380;
const D_TOP = 40;
const D_BASE = 168;
const D_DOM = [0, 0.7];
const RUG = [182, 196];
const P_TOP = 222;
const P_BASE = 290;
// Two features (steps 4–6): 32 user units per feature unit on both axes.
const SX0 = 56;
const SX1 = 312;
const SY0 = 290;
const SY1 = 66;
const BOX_Y = 8;
const ROW = 20;

// Share of all training cases per unit of feature 1, in bins [k, k + 1).
function barShares() {
  const out = [];
  for (let k = X1_DOM[0]; k < 7; k++) {
    const n = [0, 0];
    TR.X.forEach((x, i) => { if (x[0] >= k && x[0] < k + 1) n[TR.y[i]]++; });
    out.push({ k, share: n.map((v) => v / TR.X.length) });
  }
  return out;
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let prior1 = 0.4;
  let x1 = X1_0;
  let q = Q0.slice();
  let data = 'corr';

  const X1 = linearScale(X1_DOM, [AX0, AX1]);
  const DY = linearScale(D_DOM, [D_BASE, D_TOP]);
  const PY = linearScale([0, 1], [P_BASE, P_TOP]);
  const X2 = linearScale(X_DOM, [SX0, SX1]);
  const Y2 = linearScale(Y_DOM, [SY0, SY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Steps 1 to 3: one feature. Bars and two curves show how common each class is along feature 1, the cyan solid curve for class 0 and the violet dashed curve for class 1, with the probability of class 1 plotted beneath. Steps 4 to 6: two features, class 0 as cyan circles and class 1 as violet triangles, with oval outlines for each class, a new case you can move, and shaded regions with a line along the boundary.');

  const defs = el('defs', {}, svg);
  const hatch = el('pattern', { id: 'nb-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  el('rect', { width: 5, height: 5, fill: VIOLET, 'fill-opacity': 0.18 }, hatch);
  el('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: VIOLET, 'stroke-width': 2 }, hatch);
  const clip = el('clipPath', { id: 'nb-plot' }, defs);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1 }, clip);

  // ---- One feature.
  const view1 = el('g', {}, svg);
  const tick = (g, x1a, y1a, x2a, y2a) => el('line', { x1: x1a, y1: y1a, x2: x2a, y2: y2a, stroke: LINE }, g);
  const label = (g, x, y, text, anchor = 'start') => el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, text }, g);
  tick(view1, AX0, D_BASE, AX1, D_BASE);
  tick(view1, AX0, D_BASE, AX0, D_TOP);
  for (const v of [0, 0.2, 0.4, 0.6]) {
    tick(view1, AX0 - 4, DY(v), AX0, DY(v));
    label(view1, AX0 - 7, DY(v) + 4, fmt(v), 'end');
  }
  label(view1, AX0 + 6, D_TOP + 2, 'share of all cases per unit of x₁');
  tick(view1, AX0, P_BASE, AX1, P_BASE);
  tick(view1, AX0, P_BASE, AX0, P_TOP);
  for (const v of [0, 0.5, 1]) {
    tick(view1, AX0 - 4, PY(v), AX0, PY(v));
    label(view1, AX0 - 7, PY(v) + 4, fmt(v), 'end');
  }
  for (const t of X1.ticks(8)) {
    tick(view1, X1(t), P_BASE, X1(t), P_BASE + 5);
    label(view1, X1(t), P_BASE + 18, fmt(t), 'middle');
  }
  label(view1, (AX0 + AX1) / 2, P_BASE + 38, 'feature 1, x₁', 'middle');
  label(view1, AX0 + 6, P_TOP - 6, 'P(class 1 | x₁)');
  el('line', { x1: AX0, x2: AX1, y1: PY(0.5), y2: PY(0.5), stroke: LINE, 'stroke-dasharray': '2 3' }, view1);

  const barsG = el('g', {}, view1);
  const bw = (X1(1) - X1(0)) * 0.36;
  for (const b of barShares()) {
    const mid = X1(b.k + 0.5);
    b.share.forEach((sh, c) => {
      if (!sh) return;
      el('rect', {
        x: c ? mid + 1 : mid - 1 - bw, y: DY(sh), width: bw, height: D_BASE - DY(sh),
        fill: c ? 'url(#nb-hatch)' : CYAN, 'fill-opacity': c ? 1 : 0.3, stroke: c ? VIOLET : CYAN, 'stroke-width': 1,
      }, barsG);
    });
  }
  const band = el('rect', {
    x: X1(BIN[0]), y: D_TOP + 10, width: X1(BIN[1]) - X1(BIN[0]), height: RUG[1] - 2 - D_TOP,
    fill: STRONG, 'fill-opacity': 0.06, stroke: STRONG, 'stroke-dasharray': '4 3',
  }, view1);

  const rugG = el('g', {}, view1);
  const jit = mulberry32(5);
  TR.X.forEach((x, i) => classMarker(rugG, X1(x[0]), RUG[TR.y[i]] + (jit() - 0.5) * 6, TR.y[i], { r: 3.5 }));

  const curvesG = el('g', {}, view1);
  const curve0 = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.2 }, curvesG);
  const curve1 = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 2.2, 'stroke-dasharray': '6 4' }, curvesG);
  const postCurve = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 2 }, curvesG);
  const bLine = el('line', { y1: D_TOP, y2: P_BASE, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '5 3' }, curvesG);
  const bText = el('text', { class: 'stage-text', y: P_TOP - 6, 'text-anchor': 'middle', text: 'boundary' }, curvesG);
  bText.style.fill = STRONG;
  const caseG = el('g', {}, view1);
  const caseLine = el('line', { y1: D_TOP, y2: P_BASE, stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '1 3' }, caseG);
  const dot0 = el('circle', { r: 4, fill: BG, stroke: CYAN, 'stroke-width': 2 }, caseG);
  const dot1 = el('path', { fill: BG, stroke: VIOLET, 'stroke-width': 2 }, caseG);
  const dotP = el('circle', { r: 4, fill: STRONG }, caseG);

  // ---- Two features.
  const view2 = el('g', {}, svg);
  const regionG = el('g', {}, view2);
  const region = regionMap(regionG, X2, Y2, () => 0.5, { nx: GRID[0], ny: GRID[1], mode: 'prob' });
  axes(view2, X2, Y2, { xLabel: 'feature 1, x₁', yLabel: 'feature 2, x₂' });
  const contourG = el('g', { 'clip-path': 'url(#nb-plot)' }, view2);
  const ptsMain = el('g', {}, view2);
  TR.X.forEach((x, i) => classMarker(ptsMain, X2(x[0]), Y2(x[1]), TR.y[i], { r: 4.5 }));
  const ptsCorr = el('g', {}, view2);
  CTR.X.forEach((x, i) => classMarker(ptsCorr, X2(x[0]), Y2(x[1]), CTR.y[i], { r: 4.5 }));
  const drawContours = (m) => {
    contourG.replaceChildren();
    for (const c of [0, 1]) {
      for (const k of [1, 2]) {
        el('ellipse', {
          cx: X2(m.mu[c][0]), cy: Y2(m.mu[c][1]),
          rx: k * Math.sqrt(m.vari[c][0]) * (X2(1) - X2(0)), ry: k * Math.sqrt(m.vari[c][1]) * (Y2(0) - Y2(1)),
          fill: 'none', stroke: c ? VIOLET : CYAN, 'stroke-width': k === 1 ? 2 : 1.4,
          'stroke-dasharray': c ? '6 4' : null, 'stroke-opacity': k === 1 ? 1 : 0.7,
        }, contourG);
      }
    }
  };

  const queryG = el('g', {}, view2);
  const focusRing = el('circle', { r: 14, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, queryG);
  const diamond = el('path', { fill: BG, stroke: STRONG, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, queryG);
  const handle = el('circle', {
    class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'button',
    'aria-roledescription': 'movable point',
  }, queryG);
  handle.style.outline = 'none';
  let focused = false;

  // Key, above the plots.
  const keyG = el('g', {}, svg);
  classMarker(keyG, AX0 + 6, 16, 0, { r: 5 });
  el('line', { x1: AX0 + 16, x2: AX0 + 34, y1: 16, y2: 16, stroke: CYAN, 'stroke-width': 2.2 }, keyG);
  label(keyG, AX0 + 40, 20, 'class 0');
  classMarker(keyG, AX0 + 116, 16, 1, { r: 5 });
  el('line', { x1: AX0 + 126, x2: AX0 + 144, y1: 16, y2: 16, stroke: VIOLET, 'stroke-width': 2.2, 'stroke-dasharray': '6 4' }, keyG);
  label(keyG, AX0 + 150, 20, 'class 1');
  const keyNew = el('g', {}, keyG);
  el('path', { d: diamondD(AX0 + 232, 16, 6), fill: BG, stroke: STRONG, 'stroke-width': 1.8 }, keyNew);
  label(keyNew, AX0 + 243, 20, 'new case');

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (text, hidden, attrs) => {
    const g = make('div');
    const lab = make('label');
    const input = make('input', { type: 'range', ...attrs });
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    lab.append(text, make('span', { className: 'vh', textContent: hidden }), ' ', input, ' ', out);
    g.append(lab);
    return { g, input, out };
  };
  const xs = slider('x₁', ', feature 1 of the new case', { min: X1_DOM[0], max: X1_DOM[1], step: 0.05, value: x1 });
  const ps = slider('P(class 1)', ', the prior of class 1', { min: 0.1, max: 0.9, step: 0.05, value: prior1 });
  const dataGroup = make('div');
  const DATA = ['main', 'corr'];
  const dataBtns = DATA.map((d) => make('button', { type: 'button', textContent: d === 'main' ? 'Independent features' : 'Correlated features' }));
  dataGroup.append(...dataBtns);
  controls.append(xs.g, ps.g, dataGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // ---- State.
  let step = 0;
  let token = 0;
  let s = { v1: 1, v2: 0, bars: 1, band: 1, curves: 0, case: 0, contours: 0, region: 0, query: 0 };
  const target = (i) => ({
    v1: i <= 2 ? 1 : 0,
    v2: i >= 3 ? 1 : 0,
    bars: i === 0 ? 1 : i <= 2 ? 0.3 : 0,
    band: i === 0 ? 1 : 0,
    curves: i === 1 || i === 2 ? 1 : 0,
    case: i === 1 || i === 2 ? 1 : 0,
    contours: i === 3 || i === 5 ? 1 : 0,
    region: i >= 4 ? 1 : 0,
    query: i === 3 || i === 4 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const curPrior = () => (step === 2 || step === 4 ? prior1 : 0.4);
  const curData = () => (step === 5 ? data : 'main');
  const curModel2 = () => (curData() === 'corr' ? CM : withPrior(M, curPrior()));

  let regionKey = '';
  let contourKey = '';
  let pending = false;
  function drawRegion() {
    const key = `${curData()}:${curPrior()}`;
    if (key === regionKey) return;
    regionKey = key;
    const m = curModel2();
    region.update((a, b) => nbPosterior(m, [a, b])[1]);
  }
  function queueRegion() {
    if (pending) return;
    pending = true;
    const run = () => { pending = false; drawRegion(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  }

  const P = (v) => Number(v.toFixed(2));
  function render() {
    op(view1, s.v1);
    op(view2, s.v2);
    op(barsG, s.bars);
    op(band, s.band);
    op(curvesG, s.curves);
    op(caseG, s.case);
    op(contourG, s.contours);
    op(regionG, s.region);
    op(queryG, s.query);
    op(keyNew, s.query);

    if (step <= 2) {
      const m1 = oneD(withPrior(M, curPrior()));
      const n = 240;
      const pts = Array.from({ length: n + 1 }, (_, k) => X1_DOM[0] + ((X1_DOM[1] - X1_DOM[0]) * k) / n);
      const path = (f, Y) => pts.map((x, k) => `${k ? 'L' : 'M'}${P(X1(x))},${P(Y(f(x)))}`).join('');
      curve0.setAttribute('d', path((x) => weighted(m1, x)[0], DY));
      curve1.setAttribute('d', path((x) => weighted(m1, x)[1], DY));
      postCurve.setAttribute('d', path((x) => nbPosterior(m1, [x])[1], PY));
      const bx = X1(boundaries1D(curPrior())[0]);
      bLine.setAttribute('x1', P(bx));
      bLine.setAttribute('x2', P(bx));
      bText.setAttribute('x', P(bx));
      const w = weighted(m1, x1);
      const cx = X1(x1);
      caseLine.setAttribute('x1', P(cx));
      caseLine.setAttribute('x2', P(cx));
      dot0.setAttribute('cx', P(cx));
      dot0.setAttribute('cy', P(DY(w[0])));
      dot1.setAttribute('d', triD(cx, DY(w[1]), 4.5));
      dotP.setAttribute('cx', P(cx));
      dotP.setAttribute('cy', P(PY(nbPosterior(m1, [x1])[1])));
    } else {
      queueRegion();
      const ck = curData();
      if (ck !== contourKey) {
        contourKey = ck;
        drawContours(ck === 'corr' ? CM : M);
      }
      op(ptsMain, ck === 'main' ? 1 : 0);
      op(ptsCorr, ck === 'corr' ? 1 : 0);
      const qx = X2(q[0]);
      const qy = Y2(q[1]);
      diamond.setAttribute('d', diamondD(qx, qy, 8));
      for (const n of [handle, focusRing]) {
        n.setAttribute('cx', P(qx));
        n.setAttribute('cy', P(qy));
      }
      focusRing.style.display = focused ? '' : 'none';
      handle.setAttribute('aria-label', `New case at feature 1 = ${q[0].toFixed(1)}, feature 2 = ${q[1].toFixed(1)}. Arrow keys move it.`);
    }

    const texts = readout(step, { prior1, x1, q, data });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let w = 200;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min((step <= 2 ? AX1 : SX1) + 14, 598 - w);
    boxRect.setAttribute('x', bx);
    // In the 2D view the box sits below the key, beside the plot.
    const by = step <= 2 ? BOX_Y : BOX_Y + 30;
    boxRect.setAttribute('y', by);
    boxRect.setAttribute('width', w.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', bx + 12);
      r.setAttribute('y', by + 8 + ROW * i + ROW / 2 + 5);
    });

    xs.out.textContent = x1.toFixed(2);
    xs.input.setAttribute('aria-valuetext', `feature 1 = ${x1.toFixed(2)}`);
    ps.out.textContent = prior1.toFixed(2);
    ps.input.setAttribute('aria-valuetext', `P(class 1) = ${prior1.toFixed(2)}, P(class 0) = ${(1 - prior1).toFixed(2)}`);
    dataBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(DATA[i] === data)));
  }

  function animateTo(to, ms) {
    const my = ++token;
    const start = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * kt;
      s = next;
      render();
    }, { reduced });
  }

  const spoken = (i) => {
    const r = readout(i, { prior1, x1, q, data });
    if (i === 0) {
      const b = binCounts();
      return `One hundred training cases: sixty circles, class 0, and forty triangles, class 1. Between feature 1 = 3 and 4 there are ${b.n0} circles and ${b.n1} triangles, so the share of class 1 there is ${b.n1} out of ${b.n0 + b.n1}, ${p2(b.post1)}.`;
    }
    if (i === 1 || i === 2) {
      const m1 = oneD(withPrior(M, curPrior()));
      const post = p2(nbPosterior(m1, [x1])[1]);
      return `Prior of class 1 ${p2(curPrior())}. Boundary at feature 1 = ${p2(boundaries1D(curPrior())[0])}; class 1, the triangles, is predicted to its right. At feature 1 = ${p2(x1)} the probability of class 1 is ${post}.`;
    }
    if (i === 3 || i === 4) {
      const m = i === 3 ? M : withPrior(M, prior1);
      const post = nbPosterior(m, q)[1];
      return `New case at feature 1 = ${q[0].toFixed(1)}, feature 2 = ${q[1].toFixed(1)}. Probability of class 1, the triangles: ${p2(post)}. Predicted class ${post >= 0.5 ? '1, triangle' : '0, circle'}.${i === 4 ? ` ${r[2]}.` : ''}`;
    }
    return `${r[0]}. ${r[1]}, ${r[2]}, ${r[3]}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  const setQuery = (a, b) => {
    q = [clamp(a, X_DOM[0], X_DOM[1]), clamp(b, Y_DOM[0], Y_DOM[1])];
    render();
    say(spoken(step));
  };
  draggable(handle, {
    bounds: { x0: SX0, x1: SX1, y0: SY0, y1: SY1 },
    onDrag: (px, py) => setQuery(X2.invert(px), Y2.invert(py)),
  });
  handle.addEventListener('focus', () => {
    let kb = true;
    try { kb = handle.matches(':focus-visible'); } catch { /* keep the ring */ }
    focused = kb;
    render();
  });
  handle.addEventListener('blur', () => { focused = false; render(); });

  xs.input.addEventListener('input', () => {
    x1 = clamp(Number(xs.input.value), X1_DOM[0], X1_DOM[1]);
    render();
    say(spoken(step));
  });
  ps.input.addEventListener('input', () => {
    prior1 = clamp(Math.round(Number(ps.input.value) * 20) / 20, 0.1, 0.9);
    render();
    say(spoken(step));
  });
  dataBtns.forEach((b, i) => b.addEventListener('click', () => {
    data = DATA[i];
    render();
    announce(spoken(5));
  }));

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) p.hidden = false;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      showGroup(xs.g, i === 1 || i === 2);
      showGroup(ps.g, i === 2 || i === 4);
      showGroup(dataGroup, i === 5);
      const live = i === 3 || i === 4;
      handle.setAttribute('tabindex', live ? 0 : -1);
      if (!live && doc.activeElement === handle) handle.blur();
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      return animateTo(to, 500);
    },
  };
}

function diamondD(cx, cy, r) {
  const P = (v) => Number(v.toFixed(2));
  return `M${P(cx)},${P(cy - r)}L${P(cx + r)},${P(cy)}L${P(cx)},${P(cy + r)}L${P(cx - r)},${P(cy)}Z`;
}

function triD(cx, cy, r) {
  return trianglePoints(cx, cy, r).map(([x, y], k) => `${k ? 'L' : 'M'}${Number(x.toFixed(2))},${Number(y.toFixed(2))}`).join('') + 'Z';
}
