// Lesson 10, "Overfitting and regularisation": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mse, mean, variance, lstsq, polyDesign, polyEval } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, tween } from '../assets/plot.js';
import { sineData, truth } from '../assets/sine-data.js';

export const N_TRAIN = 15;
export const SEED_TRAIN = 3;
export const N_TEST = 200;
export const SEED_TEST = 4;
export const SD = 0.25;
export const MAX_DEGREE = 12;
export const STEP_DEGREES = [1, 3, 12];
// Step 5: fits to many other training sets of 15 points.
export const BV_SEED0 = 101;
export const BV_SETS = 200;
export const BV_DRAW = 20;
export const BV_DEGREES = [1, 3, 7];
export const X0 = 0.5;
// Step 6: λ = 10^(−6 + t/10) for t = 0 … 70.
export const SLIDER_MAX = 70;
export const T_START = 0;

export const trainData = () => sineData({ n: N_TRAIN, sd: SD, seed: SEED_TRAIN });
export const testData = () => sineData({ n: N_TEST, sd: SD, seed: SEED_TEST });

// Ridge on the polynomial features x, x², …, x^p. The page's objective is
//   J(θ) = (1/n)·Σ eᵢ² + λ·Σ_{j≥1} θⱼ²,
// while lstsq minimises Σ eᵢ² + λ_code·Σ_{j≥1} θⱼ² (a sum, not a mean), so
// λ_code = n·λ gives the same minimiser. θ₀, the intercept, is not penalised.
export function ridgeFit(xs, ys, p, lambda = 0) {
  const r = lstsq(polyDesign(xs, p), ys, { lambda: xs.length * lambda });
  if (!r.ok) throw new Error('polynomial fit failed');
  return r.coef;
}

export function ridgeObjective(coef, xs, ys, lambda) {
  const fit = mse(ys, xs.map((x) => polyEval(coef, x)));
  return fit + lambda * coef.slice(1).reduce((s, c) => s + c * c, 0);
}

const scoreOf = (coef, d) => mse(d.ys, d.xs.map((x) => polyEval(coef, x)));

export function degreeTable() {
  const tr = trainData();
  const te = testData();
  return Array.from({ length: MAX_DEGREE + 1 }, (_, p) => {
    const coef = ridgeFit(tr.xs, tr.ys, p);
    return { degree: p, coef, train: scoreOf(coef, tr), test: scoreOf(coef, te) };
  });
}

export function bestDegree() {
  const rows = degreeTable();
  return rows.reduce((a, b) => (b.test < a.test ? b : a)).degree;
}

// Test MSE on the test points that lie inside the training points' range.
export function insideTest(p) {
  const tr = trainData();
  const te = testData();
  const lo = Math.min(...tr.xs);
  const hi = Math.max(...tr.xs);
  const idx = te.xs.map((x, i) => i).filter((i) => te.xs[i] >= lo && te.xs[i] <= hi);
  const coef = ridgeFit(tr.xs, tr.ys, p);
  return { n: idx.length, mse: mse(idx.map((i) => te.ys[i]), idx.map((i) => polyEval(coef, te.xs[i]))) };
}

// Midpoint of the widest gap between neighbouring training x values.
export function gapMid() {
  const xs = [...trainData().xs].sort((a, b) => a - b);
  let best = 1;
  for (let i = 2; i < xs.length; i++) if (xs[i] - xs[i - 1] > xs[best] - xs[best - 1]) best = i;
  return (xs[best] + xs[best - 1]) / 2;
}

export function lambdaFromSlider(t) {
  const k = -60 + Math.round(t);
  // Whole powers of ten exactly, so the ends are exactly 1e-6 and 10.
  if (k % 10 === 0) return Number(`1e${k / 10}`);
  return 10 ** (k / 10);
}
export const sliderFromLambda = (lam) => Math.round((Math.log10(lam) + 6) * 10);

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const UNSUP = Object.fromEntries(Object.entries(SUP).map(([k, v]) => [v, k]));

// λ for display: two significant figures; below 0.01 as a × 10⁻ᵏ.
export function lambdaText(lam) {
  if (lam >= 0.01) return lam.toPrecision(2);
  let k = Math.floor(Math.log10(lam));
  let a = Number((lam / 10 ** k).toPrecision(2));
  if (a >= 10) { a /= 10; k += 1; }
  const exp = String(k).split('').map((c) => SUP[c]).join('');
  return `${a.toFixed(1)} × 10${exp}`;
}
// The number a display string stands for (used by the tests).
export function lambdaValue(text) {
  const m = /^([\d.]+) × 10(.+)$/.exec(text);
  if (!m) return Number(text);
  const k = Number([...m[2]].map((c) => UNSUP[c]).join(''));
  return Number(m[1]) * 10 ** k;
}
// Words for screen readers.
export function lambdaSpoken(lam) {
  const t = lambdaText(lam);
  const m = /^([\d.]+) × 10(.+)$/.exec(t);
  if (!m) return t;
  const k = [...m[2]].map((c) => UNSUP[c]).join('').replace('-', 'minus ');
  return `${m[1]} times 10 to the ${k}`;
}

export function ridgeTable(p = MAX_DEGREE) {
  const tr = trainData();
  const te = testData();
  return Array.from({ length: SLIDER_MAX + 1 }, (_, t) => {
    const lambda = lambdaFromSlider(t);
    const coef = ridgeFit(tr.xs, tr.ys, p, lambda);
    return { t, lambda, coef, train: scoreOf(coef, tr), test: scoreOf(coef, te), maxAbs: Math.max(...coef.slice(1).map(Math.abs)) };
  });
}

// Fits of degree p to BV_SETS fresh training sets (seeds seed0 … seed0 + 199);
// bias² and variance of their predictions at X0 (population variance over the
// sets). The squared gap of the average, (f − mean)², overstates bias² by
// about variance/BV_SETS on average, so the estimate subtracts that; when the
// gap is under twice that floor, bias² is reported as too small to measure.
const bvCache = new Map();
export function biasVariance(p, seed0 = BV_SEED0) {
  const key = `${p}:${seed0}`;
  if (bvCache.has(key)) return bvCache.get(key);
  const fits = [];
  for (let k = 0; k < BV_SETS; k++) {
    const d = sineData({ n: N_TRAIN, sd: SD, seed: seed0 + k });
    fits.push(ridgeFit(d.xs, d.ys, p));
  }
  const preds = fits.map((c) => polyEval(c, X0));
  const m = mean(preds);
  const gap2 = (truth(X0) - m) ** 2;
  const v = variance(preds, { sample: false });
  const floor = v / BV_SETS;
  const out = {
    degree: p, fits, preds, mean: m,
    gap2,
    variance: v,
    floor,
    bias2: Math.max(0, gap2 - floor),
    measurable: gap2 > 2 * floor,
    sigma2: SD * SD,
  };
  bvCache.set(key, out);
  return out;
}

// Other blocks of 200 training sets, for the spread of degree 7's variance.
export const BV_BLOCKS = [101, 301, 501, 701, 901, 1101];
export function varianceSpread(p) {
  const vs = BV_BLOCKS.map((b) => biasVariance(p, b).variance);
  return { lo: Math.min(...vs), hi: Math.max(...vs) };
}

// MSE for display: 3 decimals below 10, 1 below 1,000, then whole with commas.
export function fmtMSE(v) {
  if (v < 10) return v.toFixed(3);
  if (v < 1000) return v.toFixed(1);
  return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
// Bias² to two significant figures, or "too small" below the noise floor.
export const fmtBias = (b, long = false) => (b.measurable ? Number(b.bias2.toPrecision(2)).toString() : long ? 'too small to measure with 200 sets' : 'too small');
// Variance: two significant figures below 1; one, with "about", from 1 up.
export const fmtVar = (v) => (v >= 1 ? `about ${Number(v.toPrecision(1))}` : v.toPrecision(2));
// Round to one significant figure, for a quoted range.
export const sig1 = (v) => String(Number(v.toPrecision(1)));
// A large number to two significant figures, with commas.
export const round2 = (v) => Number(v.toPrecision(2)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
}

// Step numbers: 0 to 2 show degrees 1, 3, 12; 3 the slider; 4 bias–variance; 5 ridge.
export function readout(step, { degree = 3, t = T_START, bv = 1 } = {}) {
  if (step <= 3) {
    const p = step < 3 ? STEP_DEGREES[step] : degree;
    const r = degreeTable()[p];
    return [`degree ${p}`, `train MSE: ${fmtMSE(r.train)}`, `test MSE: ${fmtMSE(r.test)}`];
  }
  if (step === 4) {
    const b = biasVariance(bv);
    return [`degree ${bv}, at x₀ = ${X0}`, `bias²: ${fmtBias(b)}`, `variance: ${fmtVar(b.variance)}`, `σ²: ${b.sigma2}`];
  }
  const r = ridgeTable()[t];
  return [`λ = ${lambdaText(r.lambda)}`, `train MSE: ${fmtMSE(r.train)}`, `test MSE: ${fmtMSE(r.test)}`, `largest |θⱼ|: ${fmt(r.maxAbs)}`];
}

export function pageValues() {
  const rows = degreeTable();
  const best = bestDegree();
  const ridge = ridgeTable();
  const rb = ridge.reduce((a, b) => (b.test < a.test ? b : a));
  const tr = trainData();
  const inside = insideTest(MAX_DEGREE);
  const inside3 = insideTest(3);
  const spread7 = varianceSpread(7);
  const out = {
    'best-degree': String(best),
    'best-test': fmtMSE(rows[best].test),
    'train-xmax': fmt(Math.max(...tr.xs)),
    'inside-n': String(inside.n),
    'inside-12': fmtMSE(inside.mse),
    'inside-3': fmtMSE(inside3.mse),
    'd12-maxabs': round2(Math.max(...rows[12].coef.slice(1).map(Math.abs))),
    'gap-mid': fmt(gapMid(), 1),
    'bv7-var-lo': sig1(spread7.lo),
    'bv7-var-hi': sig1(spread7.hi),
    'ridge-start': lambdaText(ridge[0].lambda),
    'ridge-start-test': fmtMSE(ridge[0].test),
    'ridge-best-lambda': lambdaText(rb.lambda),
    'ridge-best-test': fmtMSE(rb.test),
    'ridge-end-test': fmtMSE(ridge[ridge.length - 1].test),
    'bv-sets': String(BV_SETS),
    'bv-draw': String(BV_DRAW),
    'bv-x0': String(X0),
  };
  for (const p of [0, 1, 3, 12]) {
    out[`d${p}-train`] = fmtMSE(rows[p].train);
    out[`d${p}-test`] = fmtMSE(rows[p].test);
  }
  for (const p of BV_DEGREES) {
    const b = biasVariance(p);
    out[`bv${p}-bias2`] = fmtBias(b, true);
    out[`bv${p}-var`] = fmtVar(b.variance);
  }
  return out;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const FONT = "'JetBrains Mono', monospace";

const PX0 = 56;
const PX1 = 340;
const PY0 = 290;
const PY1 = 20;
const Y_DOM = [-1.5, 1.5];
const RX0 = 372;
const BOX_Y = 8;
const ROW = 20;
// The MSE chart, log scale.
const CX0 = 420;
const CX1 = 586;
const CY0 = 286;
const CY1 = 132;
const LOG_DOM = [-2.5, 5];

const supPow = (k) => (k === 0 ? '1' : k === 1 ? '10' : `10${String(k).split('').map((c) => SUP[c]).join('')}`);
// Label for the level 10^k on the log MSE axis.
export const logTick = (k) => (k < 0 ? `0.${'0'.repeat(-k - 1)}1` : supPow(k));

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const tr = trainData();
  const te = testData();
  const rows = degreeTable();
  const ridge = ridgeTable();
  const GRID = Array.from({ length: 241 }, (_, i) => -1 + i / 120);
  let degree = 3;
  let t = T_START;
  let bvDeg = 1;

  const X = linearScale([-1, 1], [PX0, PX1]);
  const Y = linearScale(Y_DOM, [PY0, PY1]);
  const curveD = (f) => pathFrom(GRID.map((x) => [x, clamp(f(x), -50, 50)]), X, Y);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Fifteen training points around a wave, a fitted polynomial curve, and a chart of training and test error. The read-out on the right changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l10-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  axes(svg, X, Y, { xLabel: 'x', yLabel: 'y' });
  const testG = el('g', {}, svg);
  te.xs.forEach((x, i) => el('rect', { x: X(x) - 2, y: Y(clamp(te.ys[i], ...Y_DOM)) - 2, width: 4, height: 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1, opacity: 0.45 }, testG));

  const curvesG = el('g', { 'clip-path': 'url(#l10-plot)' }, svg);
  const spagG = el('g', {}, curvesG);
  const spag = Array.from({ length: BV_DRAW }, () => el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 1.1, opacity: 0.4 }, spagG));
  const meanPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.6 }, spagG);
  const truthPath = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 1.8, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round', d: curveD(truth) }, curvesG);
  const fitPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, curvesG);
  const x0G = el('g', {}, svg);
  el('line', { x1: X(X0), x2: X(X0), y1: PY1, y2: PY0, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, x0G);
  el('text', { class: 'stage-text', x: X(X0) + 5, y: PY0 - 8, text: `x₀ = ${X0}` }, x0G);

  const trainG = el('g', {}, svg);
  tr.xs.forEach((x, i) => el('circle', { cx: X(x), cy: Y(tr.ys[i]), r: 4.5, fill: CYAN, stroke: BG, 'stroke-width': 1.2 }, trainG));

  // Key, top-left of the plot.
  const keyG = el('g', {}, svg);
  el('rect', { x: PX0 + 6, y: PY1 + 2, width: 132, height: 44, rx: 4, fill: SURFACE, 'fill-opacity': 0.9 }, keyG);
  el('circle', { cx: PX0 + 20, cy: PY1 + 16, r: 4.5, fill: CYAN }, keyG);
  el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 20, text: 'training (15)' }, keyG);
  el('rect', { x: PX0 + 18, y: PY1 + 32, width: 4, height: 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1 }, keyG);
  const kt = el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 38, text: 'test (200)' }, keyG);
  kt.style.fill = VIOLET;
  const key5 = el('g', {}, svg);
  el('rect', { x: PX0 + 6, y: PY1 + 2, width: 160, height: 62, rx: 4, fill: SURFACE, 'fill-opacity': 0.9 }, key5);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 16, y2: PY1 + 16, stroke: CYAN, 'stroke-width': 1.1, opacity: 0.6 }, key5);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 20, text: `${BV_DRAW} of ${BV_SETS} fits` }, key5);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 34, y2: PY1 + 34, stroke: CYAN, 'stroke-width': 2.6 }, key5);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 38, text: 'their average' }, key5);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 52, y2: PY1 + 52, stroke: STRONG, 'stroke-width': 1.8, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round' }, key5);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 56, text: 'true curve' }, key5);

  // --- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Chart: MSE (log scale) against degree, or against λ in step 6.
  const chartG = el('g', {}, svg);
  const LY = linearScale(LOG_DOM, [CY0, CY1]);
  const ly = (v) => LY(clamp(Math.log10(v), ...LOG_DOM));
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, chartG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, chartG);
  for (const k of [-2, 0, 2, 4]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: LY(k), y2: LY(k), stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: LY(k) + 4, 'text-anchor': 'end', 'font-family': FONT, text: logTick(k) }, chartG);
  }
  el('text', { class: 'stage-text', x: CX0 - 4, y: CY1 - 10, 'text-anchor': 'middle', text: 'MSE' }, chartG);
  // Degree axis.
  const degAxis = el('g', {}, chartG);
  const DX = linearScale([0, MAX_DEGREE], [CX0 + 8, CX1 - 4]);
  for (const p of [0, 3, 6, 9, 12]) {
    el('line', { x1: DX(p), x2: DX(p), y1: CY0, y2: CY0 + 4, stroke: LINE }, degAxis);
    el('text', { class: 'stage-text', x: DX(p), y: CY0 + 16, 'text-anchor': 'middle', text: String(p) }, degAxis);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'degree' }, degAxis);
  const degMarker = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, degAxis);
  el('path', { d: pathFrom(rows.map((r) => [r.degree, Math.log10(r.train)]), DX, LY), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, degAxis);
  el('path', { d: pathFrom(rows.map((r) => [r.degree, clamp(Math.log10(r.test), ...LOG_DOM)]), DX, LY), fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, degAxis);
  rows.forEach((r) => {
    el('circle', { cx: DX(r.degree), cy: ly(r.train), r: 3, fill: CYAN }, degAxis);
    el('rect', { x: DX(r.degree) - 3, y: ly(r.test) - 3, width: 6, height: 6, fill: BG, stroke: VIOLET, 'stroke-width': 1.5 }, degAxis);
  });
  // λ axis (step 6).
  const lamAxis = el('g', {}, chartG);
  const LX = linearScale([0, SLIDER_MAX], [CX0 + 8, CX1 - 4]);
  for (const k of [-6, -4, -2, 0]) {
    const tt = (k + 6) * 10;
    el('line', { x1: LX(tt), x2: LX(tt), y1: CY0, y2: CY0 + 4, stroke: LINE }, lamAxis);
    el('text', { class: 'stage-text', x: LX(tt), y: CY0 + 16, 'text-anchor': 'middle', text: supPow(k) }, lamAxis);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'λ (degree 12)' }, lamAxis);
  const lamMarker = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, lamAxis);
  el('path', { d: pathFrom(ridge.map((r) => [r.t, Math.log10(r.train)]), LX, LY), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, lamAxis);
  el('path', { d: pathFrom(ridge.map((r) => [r.t, clamp(Math.log10(r.test), ...LOG_DOM)]), LX, LY), fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, lamAxis);
  const lamDotA = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.2 }, lamAxis);
  const lamDotB = el('rect', { width: 8, height: 8, fill: BG, stroke: VIOLET, 'stroke-width': 1.8 }, lamAxis);
  // Chart key.
  const ckey = el('g', {}, chartG);
  el('line', { x1: CX0 + 36, x2: CX0 + 52, y1: CY1 - 14, y2: CY1 - 14, stroke: CYAN, 'stroke-width': 2 }, ckey);
  const ckA = el('text', { class: 'stage-text', x: CX0 + 56, y: CY1 - 10, text: 'train' }, ckey);
  ckA.style.fill = CYAN;
  el('line', { x1: CX0 + 108, x2: CX0 + 124, y1: CY1 - 14, y2: CY1 - 14, stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, ckey);
  const ckB = el('text', { class: 'stage-text', x: CX0 + 128, y: CY1 - 10, text: 'test' }, ckey);
  ckB.style.fill = VIOLET;

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const degGroup = make('div');
  const degLabel = make('label');
  const degInput = make('input', { type: 'range', min: 0, max: MAX_DEGREE, step: 1, value: degree });
  const degOut = make('output');
  degOut.setAttribute('aria-hidden', 'true');
  degLabel.append('Degree ', degInput, ' ', degOut);
  degGroup.append(degLabel);

  const bvGroup = make('div');
  const bvBtns = BV_DEGREES.map((p) => make('button', { type: 'button', textContent: `Degree ${p}` }));
  bvGroup.append(...bvBtns);

  const lamGroup = make('div');
  const lamLabel = make('label');
  const lamInput = make('input', { type: 'range', min: 0, max: SLIDER_MAX, step: 1, value: T_START });
  lamInput.style.width = '150px';
  const lamOut = make('output');
  lamOut.setAttribute('aria-hidden', 'true');
  lamLabel.append('λ', make('span', { className: 'vh', textContent: ', the penalty strength' }), ' ', lamInput, ' ', lamOut);
  lamGroup.append(lamLabel);
  controls.append(degGroup, bvGroup, lamGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { fit: 0, spag: 0, x0: 0, test: 0, train: 0, deg: 0, lam: 0, key5: 0 };
  let s = { ...LAYERS, fit: 1, test: 1, train: 1, deg: 1 };
  const target = (i) => {
    if (i <= 3) return { ...LAYERS, fit: 1, test: 1, train: 1, deg: 1 };
    if (i === 4) return { ...LAYERS, spag: 1, x0: 1, key5: 1 };
    return { ...LAYERS, fit: 1, test: 1, train: 1, lam: 1 };
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const curDegree = () => (step < 3 ? STEP_DEGREES[step] : degree);

  function render() {
    let coef;
    if (step === 5) coef = ridge[t].coef;
    else coef = rows[curDegree()].coef;
    set(fitPath, { d: curveD((x) => polyEval(coef, x)) });
    op(fitPath, s.fit);
    op(testG, s.test);
    op(trainG, s.train);
    op(keyG, 1 - s.key5);
    op(key5, s.key5);
    op(truthPath, s.spag);
    op(x0G, s.x0);

    const bv = biasVariance(bvDeg);
    spag.forEach((path, k) => set(path, { d: curveD((x) => polyEval(bv.fits[k], x)) }));
    set(meanPath, { d: curveD((x) => mean(bv.fits.map((c) => polyEval(c, x)))) });
    op(spagG, s.spag);

    op(chartG, Math.max(s.deg, s.lam));
    op(degAxis, s.deg);
    op(lamAxis, s.lam);
    set(degMarker, { x1: DX(curDegree()), x2: DX(curDegree()) });
    set(lamMarker, { x1: LX(t), x2: LX(t) });
    set(lamDotA, { cx: LX(t), cy: ly(ridge[t].train) });
    set(lamDotB, { x: LX(t) - 4, y: ly(ridge[t].test) - 4 });

    const texts = readout(step, { degree, t, bv: bvDeg });
    rowsT.forEach((r, k) => {
      r.textContent = texts[k] || '';
      r.style.display = k < texts.length ? '' : 'none';
    });
    let w = 200;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0 + 4, 598 - w);
    set(boxRect, { x: bx, y: BOX_Y, width: w, height: 16 + ROW * texts.length });
    rowsT.forEach((r, k) => set(r, { x: bx + 12, y: BOX_Y + 8 + ROW * k + ROW / 2 + 5 }));

    degOut.textContent = String(degree);
    degInput.setAttribute('aria-valuetext', `degree ${degree}`);
    lamOut.textContent = lambdaText(ridge[t].lambda);
    lamInput.setAttribute('aria-valuetext', `lambda ${lambdaSpoken(ridge[t].lambda)}`);
    bvBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(BV_DEGREES[k] === bvDeg)));
  }

  function animateTo(to, ms) {
    const my = ++token;
    const start = { ...s };
    return tween(ms, (kt2) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * kt2;
      s = next;
      render();
    }, { reduced });
  }

  const spoken = (i) => {
    if (i <= 3) {
      const p = curDegree();
      return `Degree ${p}. Training MSE ${fmtMSE(rows[p].train)}, test MSE ${fmtMSE(rows[p].test)}.`;
    }
    if (i === 4) {
      const b = biasVariance(bvDeg);
      return `Degree ${bvDeg}, fitted to ${BV_SETS} training sets. At x nought ${X0}: bias squared ${fmtBias(b, true)}, variance ${fmtVar(b.variance)}.`;
    }
    const r = ridge[t];
    return `Degree 12, lambda ${lambdaSpoken(r.lambda)}. Training MSE ${fmtMSE(r.train)}, test MSE ${fmtMSE(r.test)}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  degInput.addEventListener('input', () => {
    degree = clamp(Math.round(Number(degInput.value)), 0, MAX_DEGREE);
    render();
    say(spoken(3));
  });
  lamInput.addEventListener('input', () => {
    t = clamp(Math.round(Number(lamInput.value)), 0, SLIDER_MAX);
    render();
    say(spoken(5));
  });
  bvBtns.forEach((b, k) => b.addEventListener('click', () => {
    bvDeg = BV_DEGREES[k];
    render();
    announce(spoken(4));
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
      showGroup(degGroup, i === 3);
      showGroup(bvGroup, i === 4);
      showGroup(lamGroup, i === 5);
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
