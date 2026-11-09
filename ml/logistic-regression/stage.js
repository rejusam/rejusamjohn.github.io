// Lesson 12, "Logistic regression": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-logistic-regression.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, uniform, olsFit } from '../assets/maths.js';
import { blobs2, sigmoid, logLoss, logisticFit, logisticGD, confusion, accuracy } from '../assets/classify.js';
import { linearScale, clamp, el, axes, tween, classMarker, regionMap, fmt } from '../assets/plot.js';

export const SEED = 51;
export const SEED_2D = 52;
export const N = 30;
export const ETA = 0.25;
export const GD_STEPS = 3000;
// Threshold slider: 0.05 to 0.95 in steps of 0.05.
export const THRESHOLDS = Array.from({ length: 19 }, (_, i) => Number(((i + 1) * 0.05).toFixed(2)));
// The 2D probability map's grid.
export const GRID = [60, 56];
// Where the sliders start in steps 2 and 3.
export const START = { b: -2, w: 0.4 };

// Hours x ~ U(0, 10); the student passes (y = 1) when rng() < σ(−5 + x).
let D1 = null;
export function data1D() {
  if (!D1) {
    const rng = mulberry32(SEED);
    const x = [];
    const y = [];
    for (let i = 0; i < N; i++) {
      const xi = uniform(rng, 0, 10);
      x.push(xi);
      y.push(rng() < sigmoid(-5 + xi) ? 1 : 0);
    }
    D1 = { x, y };
  }
  return { x: [...D1.x], y: [...D1.y] };
}

export const data2D = () => blobs2({ n0: 40, n1: 40, mu0: [2, 2], mu1: [4, 3.5], sd: 1, seed: SEED_2D });

const { x: XS, y: YS } = data1D();
const B2 = data2D();

// The least-squares straight line through the 0/1 outcomes, its values at 0 and
// 10 hours, and where it crosses 0 and 1.
export function olsLine() {
  const { slope, intercept } = olsFit(XS, YS);
  return {
    slope, intercept,
    at0: intercept, at10: intercept + 10 * slope,
    cross0: -intercept / slope, cross1: (1 - intercept) / slope,
  };
}

let F1 = null;
let F2 = null;
export const fit1D = () => (F1 ??= logisticFit(XS.map((v) => [v]), YS));
export const fit2D = () => (F2 ??= logisticFit(B2.X, B2.y));

export const probs = (b, w) => XS.map((v) => sigmoid(b + w * v));
export const fitProbs = () => probs(fit1D().w0, fit1D().w[0]);

// Each student's penalty: −ln of the probability given to what really happened,
// clipped exactly as logLoss clips it.
export const penalties = (b, w) => probs(b, w).map((p, i) => logLoss([YS[i]], [p]));
export const meanLoss = (b, w) => logLoss(YS, probs(b, w));

// The largest penalty; a tie goes to the lower index.
export function worstPoint(b, w) {
  const pen = penalties(b, w);
  let i = 0;
  pen.forEach((v, k) => { if (v > pen[i]) i = k; });
  return { i, x: XS[i], y: YS[i], p: probs(b, w)[i], pen: pen[i] };
}

let GD = null;
export const gdPath = () => (GD ??= logisticGD(XS.map((v) => [v]), YS, { eta: ETA, steps: GD_STEPS }));

// First step whose loss is within tol of the minimum.
export function gdWithin(tol) {
  const f = fit1D();
  const min = meanLoss(f.w0, f.w[0]);
  return gdPath().findIndex((s) => s.loss - min < tol);
}

// Predict 1 (pass) when p̂ ≥ t, using the fitted model.
export const confusionAt = (t) => confusion(YS, fitProbs().map((p) => (p >= t ? 1 : 0)));

// p̂ = t where b + w·x = ln(t / (1 − t)).
export function cutoffHours(t) {
  const f = fit1D();
  return (Math.log(t / (1 - t)) - f.w0) / f.w[0];
}

const p2 = (r) => { const f = fit2D(); return sigmoid(f.w0 + f.w[0] * r[0] + f.w[1] * r[1]); };
// The first slider move up where precision falls: [t before, t after].
export function precisionDip() {
  for (let k = 1; k < THRESHOLDS.length; k++) {
    if (confusionAt(THRESHOLDS[k]).precision < confusionAt(THRESHOLDS[k - 1]).precision) return [THRESHOLDS[k - 1], THRESHOLDS[k]];
  }
  return null;
}

export const acc2D = () => accuracy(B2.y, B2.X.map((r) => (p2(r) >= 0.5 ? 1 : 0)));

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
// Fixed decimals with a true minus sign, and never "−0.00".
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}

export function readout(step, { b = START.b, w = START.w, t = 0.5, gd = 0 } = {}) {
  if (step === 0) {
    const o = olsLine();
    const n1 = YS.filter((v) => v === 1).length;
    return [`${N} students`, `passed (1): ${n1}`, `failed (0): ${N - n1}`, `line at 0 h: ${num(o.at0, 2)}`, `line at 10 h: ${num(o.at10, 2)}`];
  }
  if (step === 1 || step === 2) {
    const rows = [`b = ${num(b, 1)}`, `w = ${num(w, 2)}`];
    if (step === 1) {
      rows.push(w === 0 ? `flat: p̂ = ${num(sigmoid(b), 2)}` : `p̂ = 0.5 at ${num(-b / w, 2)} h`);
    } else {
      const wp = worstPoint(b, w);
      rows.push(`mean log-loss: ${num(meanLoss(b, w), 3)}`, `largest: ${num(wp.pen, 2)}`, `  at ${num(wp.x, 2)} h, y = ${wp.y}`);
    }
    return rows;
  }
  if (step === 3) {
    const s = gdPath()[gd];
    const f = fit1D();
    return [`step ${gd} of ${GD_STEPS}`, `loss: ${num(s.loss, 5)}`, `b = ${num(s.w0, 2)}`, `w = ${num(s.w[0], 3)}`, `minimum: ${num(meanLoss(f.w0, f.w[0]), 5)}`];
  }
  if (step === 4) {
    const c = confusionAt(t);
    return [`threshold t = ${num(t, 2)}`, `cut-off: ${num(cutoffHours(t), 2)} h`, `precision: ${c.precision === null ? 'none' : pct(c.precision)}`, `recall: ${pct(c.recall)}`, `accuracy: ${pct(c.accuracy)}`];
  }
  const f = fit2D();
  return [`w₁ = ${num(f.w[0], 2)}`, `w₂ = ${num(f.w[1], 2)}`, `b = ${num(f.w0, 2)}`, `accuracy: ${pct(acc2D())}`];
}

export function pageValues() {
  const o = olsLine();
  const f = fit1D();
  const b = f.w0;
  const w = f.w[0];
  const wp = worstPoint(b, w);
  const n1 = YS.filter((v) => v === 1).length;
  const c5 = confusionAt(0.5);
  const c8 = confusionAt(0.8);
  const f2 = fit2D();
  const dip = precisionDip();
  return {
    'n': String(N),
    'n-pass': String(n1),
    'n-fail': String(N - n1),
    'ols-0': num(o.at0, 2),
    'ols-10': num(o.at10, 2),
    'ols-cross0': num(o.cross0, 1),
    'ols-cross1': num(o.cross1, 1),
    'fit-b': num(b, 2),
    'fit-w': num(w, 2),
    'fit-mid': num(-b / w, 2),
    'fit-loss': num(meanLoss(b, w), 3),
    'loss-0': num(meanLoss(0, 0), 3),
    'worst-x': num(wp.x, 2),
    'worst-p': num(wp.p, 2),
    'worst-pen': num(wp.pen, 2),
    'newton-iters': String(f.iters),
    'gd-eta': String(ETA),
    'gd-steps': String(GD_STEPS),
    'gd-within': String(gdWithin(1e-3)),
    'cm-t': num(0.5, 2),
    'cm-tp': String(c5.tp),
    'cm-fp': String(c5.fp),
    'cm-tn': String(c5.tn),
    'cm-fn': String(c5.fn),
    'acc-50': pct(c5.accuracy),
    'prec-50': pct(c5.precision),
    'rec-50': pct(c5.recall),
    'cut-50': num(cutoffHours(0.5), 2),
    'prec-80': pct(c8.precision),
    'rec-80': pct(c8.recall),
    'tp-80': String(c8.tp),
    'n-2d': String(B2.X.length),
    'w1': num(f2.w[0], 2),
    'w2': num(f2.w[1], 2),
    'b2': num(f2.w0, 2),
    'acc-2d': pct(acc2D()),
    'pen-half': num(-Math.log(0.5), 2),
    'pen-hundredth': num(-Math.log(0.01), 1),
    'pen-max': String(PEN_MAX),
    'grid': `${GRID[0]} × ${GRID[1]}`,
    'dip-t0': num(dip[0], 2),
    'dip-t1': num(dip[1], 2),
    'dip-p0': pct(confusionAt(dip[0]).precision),
    'dip-p1': pct(confusionAt(dip[1]).precision),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const SURFACE = '#1e293b';
const LINE = '#334155';
const FONT = "'JetBrains Mono', monospace";

// The 1D plot: hours 0 to 10 across; p̂ up, with room above 1 and below 0.
const PX0 = 56;
const PX1 = 380;
const PY1 = 34;
const BOTTOM_FULL = 286;
const BOTTOM_STRIP = 180;
const P_DOM = [-0.3, 1.3];
// The penalty strip under it (step 3).
const SB = 300;
const ST = 222;
export const PEN_MAX = 3;
// Read-out box and loss chart on the right.
const RX0 = 404;
const BOX_Y = 8;
const ROW = 20;
const CX0 = 452;
const CX1 = 590;
const CY0 = 286;
const CY1 = 164;
const LOSS_DOM = [0.3, 0.7];
// The 2D plot: 30 user units per feature unit on both axes.
const QX0 = 56;
const QX1 = 326;
const QY0 = 286;
const QY1 = 34;
const X2_DOM = [-1, 8];
const Y2_DOM = [-1, 7.4];

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const fit = fit1D();
  const ols = olsLine();
  const path = gdPath();
  let b = START.b;
  let w = START.w;
  let t = 0.5;
  let gd = 0;

  const HX = linearScale([0, 10], [PX0, PX1]);
  const PEN = linearScale([0, PEN_MAX], [SB, ST]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Thirty students: hours studied across, and whether each passed. Fails are cyan circles at 0, passes violet triangles at 1. Curves show the predicted chance of passing. In the last step, two features with a shaded probability map and a straight boundary.');

  const defs = el('defs', {}, svg);
  const clip1 = el('clipPath', { id: 'l12-one' }, defs);
  const clipRect = el('rect', { x: PX0, y: PY1 - 30, width: PX1 - PX0 }, clip1);
  const clip2 = el('clipPath', { id: 'l12-two' }, defs);
  el('rect', { x: QX0, y: QY1, width: QX1 - QX0, height: QY0 - QY1 }, clip2);

  // ---- The 1D view.
  const oneG = el('g', {}, svg);
  const bandsG = el('g', {}, oneG);
  const bandHi = el('rect', { x: PX0, width: PX1 - PX0, fill: MUTED, 'fill-opacity': 0.07 }, bandsG);
  const bandLo = el('rect', { x: PX0, width: PX1 - PX0, fill: MUTED, 'fill-opacity': 0.07 }, bandsG);
  const bandTextHi = el('text', { class: 'stage-text', x: PX1 - 6, 'text-anchor': 'end', text: 'above 1: not a probability' }, bandsG);
  const bandTextLo = el('text', { class: 'stage-text', x: PX1 - 6, 'text-anchor': 'end', text: 'below 0: not a probability' }, bandsG);
  const axesG = el('g', {}, oneG);
  const guidesG = el('g', {}, oneG);
  const plotG = el('g', { 'clip-path': 'url(#l12-one)' }, oneG);
  const olsPath = el('path', { fill: 'none', stroke: INK, 'stroke-width': 2, 'stroke-dasharray': '7 4' }, plotG);
  const newtonPath = el('path', { fill: 'none', stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '4 4' }, plotG);
  const curvePath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, plotG);
  const threshG = el('g', {}, plotG);
  const tLine = el('line', { x1: PX0, x2: PX1, stroke: STRONG, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, threshG);
  const cutLine = el('line', { stroke: STRONG, 'stroke-width': 1.6 }, threshG);
  const tText = el('text', { class: 'stage-text', 'text-anchor': 'end' }, threshG);
  const ringsG = el('g', {}, oneG);
  // Outside the clip, so the labels stay whole at the slider's ends.
  const sideG = el('g', {}, oneG);
  const sideL = el('text', { class: 'stage-text', 'text-anchor': 'end', text: 'predict fail' }, sideG);
  const sideR = el('text', { class: 'stage-text', text: 'predict pass' }, sideG);
  const pointsG = el('g', {}, oneG);
  const curveKey = el('g', {}, oneG);

  // Penalty strip.
  const stripG = el('g', {}, oneG);
  const stripAxes = el('g', {}, stripG);
  const barsG = el('g', {}, stripG);

  // Key above the plot.
  const keyG = el('g', {}, svg);
  classMarker(keyG, PX0 + 6, 16, 0, { r: 5 });
  const key0 = el('text', { class: 'stage-text', x: PX0 + 16, y: 20 }, keyG);
  classMarker(keyG, PX0 + 126, 16, 1, { r: 5 });
  const key1 = el('text', { class: 'stage-text', x: PX0 + 136, y: 20 }, keyG);

  // ---- The 2D view.
  const twoG = el('g', {}, svg);
  const QX = linearScale(X2_DOM, [QX0, QX1]);
  const QY = linearScale(Y2_DOM, [QY0, QY1]);
  const f2 = fit2D();
  const region = regionMap(twoG, QX, QY, (a, c) => sigmoid(f2.w0 + f2.w[0] * a + f2.w[1] * c), { nx: GRID[0], ny: GRID[1], mode: 'prob' });
  // The exact straight boundary is drawn below instead of the grid's staircase.
  region.g.querySelector('.boundary').style.display = 'none';
  axes(twoG, QX, QY, { xLabel: 'feature 1', yLabel: 'feature 2' });
  const linesG = el('g', { 'clip-path': 'url(#l12-two)' }, twoG);
  const level = (q) => {
    // w₁x₁ + w₂x₂ + b = ln(q / (1 − q)), drawn across the plot's width.
    const c = Math.log(q / (1 - q));
    const at = (a) => (c - f2.w0 - f2.w[0] * a) / f2.w[1];
    return { x1: QX(X2_DOM[0]), y1: QY(at(X2_DOM[0])), x2: QX(X2_DOM[1]), y2: QY(at(X2_DOM[1])) };
  };
  for (const q of [0.1, 0.9]) el('line', { ...level(q), stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '2 3' }, linesG);
  el('line', { ...level(0.5), stroke: STRONG, 'stroke-width': 2 }, linesG);
  const pts2 = el('g', {}, twoG);
  B2.X.forEach((r, i) => classMarker(pts2, QX(r[0]), QY(r[1]), B2.y[i], { r: 4.5 }));
  // Line key, on the right under the read-out.
  const lineKey = el('g', {}, twoG);
  el('line', { x1: RX0 + 8, x2: RX0 + 30, y1: 140, y2: 140, stroke: STRONG, 'stroke-width': 2 }, lineKey);
  el('text', { class: 'stage-text', x: RX0 + 36, y: 144, text: 'p̂ = 0.5' }, lineKey);
  el('line', { x1: RX0 + 8, x2: RX0 + 30, y1: 162, y2: 162, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '2 3' }, lineKey);
  el('text', { class: 'stage-text', x: RX0 + 36, y: 166, text: 'p̂ = 0.1, 0.9' }, lineKey);

  // ---- Loss chart (step 4).
  const chartG = el('g', {}, svg);
  const SX = linearScale([0, GD_STEPS], [CX0, CX1]);
  const LY = linearScale(LOSS_DOM, [CY0, CY1]);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, chartG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, chartG);
  for (const v of [0.3, 0.5, 0.7]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: LY(v), y2: LY(v), stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: LY(v) + 4, 'text-anchor': 'end', text: v.toFixed(1) }, chartG);
  }
  for (const v of [0, 1500, 3000]) {
    el('line', { x1: SX(v), x2: SX(v), y1: CY0, y2: CY0 + 4, stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: SX(v), y: CY0 + 16, 'text-anchor': v === 3000 ? 'end' : 'middle', text: String(v) }, chartG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'step' }, chartG);
  el('text', { class: 'stage-text', x: CX0 - 40, y: CY1 - 12, text: 'mean log-loss' }, chartG);
  const minLoss = meanLoss(fit.w0, fit.w[0]);
  el('line', { x1: CX0, x2: CX1, y1: LY(minLoss), y2: LY(minLoss), stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, chartG);
  el('text', { class: 'stage-text', x: CX1, y: LY(minLoss) - 6, 'text-anchor': 'end', text: 'Newton minimum' }, chartG);
  const trace = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, chartG);

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, hint, min, max, stepv, value) => {
    const label = make('label');
    const input = make('input', { type: 'range', min, max, step: stepv, value });
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    label.append(name, make('span', { className: 'vh', textContent: hint }), ' ', input, ' ', out);
    return { label, input, out };
  };
  const bw = make('div');
  const bS = slider('b', ', the intercept', -10, 4, 0.1, b);
  const wS = slider('w', ', the weight on hours', -1, 2.5, 0.05, w);
  bw.append(bS.label, wS.label);
  const fitGroup = make('div');
  const fitBtn = make('button', { type: 'button', textContent: 'Fit' });
  fitGroup.append(fitBtn);
  const tGroup = make('div');
  const tS = slider('t', ', the threshold', 0.05, 0.95, 0.05, t);
  tGroup.append(tS.label);
  controls.append(bw, fitGroup, tGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // ---- State. Layers fade; strip moves the 1D plot's bottom edge.
  let step = 0;
  let token = 0;
  let fitToken = 0;
  let s = { one: 1, two: 0, ols: 1, curve: 0, newton: 0, strip: 0, chart: 0, thresh: 0 };
  const target = (i) => ({
    one: i < 5 ? 1 : 0,
    two: i === 5 ? 1 : 0,
    ols: i === 0 ? 1 : 0,
    curve: i >= 1 && i <= 4 ? 1 : 0,
    newton: i === 3 ? 1 : 0,
    strip: i === 2 ? 1 : 0,
    chart: i === 3 ? 1 : 0,
    thresh: i === 4 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };

  const curveParams = () => {
    if (step === 3) return [path[gd].w0, path[gd].w[0]];
    if (step === 4) return [fit.w0, fit.w[0]];
    return [b, w];
  };
  const P = (v) => Number(v.toFixed(2));
  const sigPath = (bb, ww, Y) => {
    let d = '';
    for (let k = 0; k <= 200; k++) {
      const x = k / 20;
      d += `${k ? 'L' : 'M'}${P(HX(x))},${P(Y(sigmoid(bb + ww * x)))}`;
    }
    return d;
  };

  function drawAxes(Y, bottom) {
    axesG.replaceChildren();
    const font = { fill: MUTED, 'font-family': FONT, 'font-size': 11 };
    el('line', { x1: PX0, x2: PX1, y1: bottom, y2: bottom, stroke: LINE }, axesG);
    el('line', { x1: PX0, x2: PX0, y1: bottom, y2: PY1, stroke: LINE }, axesG);
    const labels = s.strip < 0.5;
    for (let h = 0; h <= 10; h += 2) {
      el('line', { x1: HX(h), x2: HX(h), y1: bottom, y2: bottom + 5, stroke: LINE }, axesG);
      if (labels) el('text', { x: HX(h), y: bottom + 18, 'text-anchor': 'middle', ...font, text: String(h) }, axesG);
    }
    if (labels) el('text', { x: (PX0 + PX1) / 2, y: bottom + 38, 'text-anchor': 'middle', ...font, 'font-size': 12, text: 'hours studied' }, axesG);
    for (const v of [0, 0.5, 1]) {
      el('line', { x1: PX0 - 5, x2: PX0, y1: Y(v), y2: Y(v), stroke: LINE }, axesG);
      el('text', { x: PX0 - 9, y: Y(v) + 4, 'text-anchor': 'end', ...font, text: fmt(v) }, axesG);
    }
    const cy = (Y(0) + Y(1)) / 2;
    el('text', { x: 14, y: cy, transform: `rotate(-90 14 ${P(cy)})`, 'text-anchor': 'middle', ...font, 'font-size': 12, text: step === 0 ? 'pass = 1, fail = 0' : 'p̂, chance of passing' }, axesG);
    guidesG.replaceChildren();
    for (const v of [0, 1]) el('line', { x1: PX0, x2: PX1, y1: Y(v), y2: Y(v), stroke: LINE, 'stroke-dasharray': '2 3' }, guidesG);
  }

  function drawStrip(bb, ww) {
    stripAxes.replaceChildren();
    barsG.replaceChildren();
    if (s.strip < 0.001) return;
    const font = { fill: MUTED, 'font-family': FONT, 'font-size': 11 };
    el('text', { class: 'stage-text', x: PX0, y: ST - 14, text: 'penalty: −ln p̂ of what happened' }, stripAxes);
    el('line', { x1: PX0, x2: PX1, y1: SB, y2: SB, stroke: LINE }, stripAxes);
    el('line', { x1: PX0, x2: PX0, y1: SB, y2: ST, stroke: LINE }, stripAxes);
    for (const v of [0, 1, 2, 3]) {
      el('line', { x1: PX0 - 5, x2: PX0, y1: PEN(v), y2: PEN(v), stroke: LINE }, stripAxes);
      el('text', { x: PX0 - 9, y: PEN(v) + 4, 'text-anchor': 'end', ...font, text: String(v) }, stripAxes);
    }
    for (let h = 0; h <= 10; h += 2) {
      el('line', { x1: HX(h), x2: HX(h), y1: SB, y2: SB + 5, stroke: LINE }, stripAxes);
      el('text', { x: HX(h), y: SB + 17, 'text-anchor': 'middle', ...font, text: String(h) }, stripAxes);
    }
    el('text', { x: (PX0 + PX1) / 2 + 60, y: SB + 34, 'text-anchor': 'middle', ...font, 'font-size': 12, text: 'hours studied' }, stripAxes);
    const pen = penalties(bb, ww);
    const worst = worstPoint(bb, ww).i;
    pen.forEach((v, i) => {
      const top = PEN(Math.min(v, PEN_MAX));
      const x = HX(XS[i]);
      el('rect', { x: P(x - 1.5), y: P(top), width: 3, height: P(SB - top), fill: YS[i] ? 'var(--mk-1)' : 'var(--mk-0)', 'fill-opacity': 0.75 }, barsG);
      classMarker(barsG, x, top - 5, YS[i], { r: 3.5 });
      // Off the scale: a double bar marks the cut.
      if (v > PEN_MAX) el('path', { d: `M${P(x - 4)},${P(top + 6)}h8M${P(x - 4)},${P(top + 9)}h8`, stroke: STRONG, 'stroke-width': 1.2 }, barsG);
      if (i === worst) el('circle', { cx: P(x), cy: P(top - 5), r: 8, fill: 'none', stroke: STRONG, 'stroke-width': 1.4 }, barsG);
    });
  }

  function render() {
    const bottom = BOTTOM_FULL + (BOTTOM_STRIP - BOTTOM_FULL) * s.strip;
    const Y = linearScale(P_DOM, [bottom, PY1]);
    op(oneG, s.one);
    op(keyG, 1);
    op(twoG, s.two);
    key0.textContent = step === 5 ? 'class 0' : 'failed (0)';
    key1.textContent = step === 5 ? 'class 1' : 'passed (1)';
    clipRect.setAttribute('height', P(bottom - PY1 + 30));
    drawAxes(Y, bottom);

    bandHi.setAttribute('y', P(PY1 - 4));
    bandHi.setAttribute('height', P(Y(1) - PY1 + 4));
    bandLo.setAttribute('y', P(Y(0)));
    bandLo.setAttribute('height', P(bottom - Y(0)));
    bandTextHi.setAttribute('y', P(PY1 + 8));
    bandTextLo.setAttribute('y', P(bottom - 6));
    op(bandsG, s.ols);
    op(bandTextLo, s.ols * (1 - s.strip));

    olsPath.setAttribute('d', `M${P(HX(0))},${P(Y(ols.at0))}L${P(HX(10))},${P(Y(ols.at10))}`);
    op(olsPath, s.ols);
    const [cb, cw] = curveParams();
    curvePath.setAttribute('d', sigPath(cb, cw, Y));
    op(curvePath, s.curve);
    newtonPath.setAttribute('d', sigPath(fit.w0, fit.w[0], Y));
    op(newtonPath, s.newton);

    // Points: fails at 0, passes at 1.
    pointsG.replaceChildren();
    XS.forEach((x, i) => classMarker(pointsG, HX(x), Y(YS[i]), YS[i], { r: 5 }));

    // Threshold: the line p̂ = t, the hours cut-off, rings on wrong predictions.
    const cut = cutoffHours(t);
    tLine.setAttribute('y1', P(Y(t)));
    tLine.setAttribute('y2', P(Y(t)));
    const cx = HX(clamp(cut, -0.5, 10.5));
    cutLine.setAttribute('x1', P(cx));
    cutLine.setAttribute('x2', P(cx));
    cutLine.setAttribute('y1', P(PY1 - 4));
    cutLine.setAttribute('y2', P(bottom));
    tText.setAttribute('x', PX1 - 4);
    tText.setAttribute('y', P(Y(t) - 5));
    tText.textContent = `t = ${num(t, 2)}`;
    sideL.setAttribute('x', P(cx - 6));
    sideR.setAttribute('x', P(cx + 6));
    // Clear of the curve: it is below 0.5 left of the cut-off and above it to the right.
    sideL.setAttribute('y', P(Y(0.8)));
    sideR.setAttribute('y', P(Y(0.2)));
    op(threshG, s.thresh);
    op(sideG, s.thresh);
    ringsG.replaceChildren();
    if (s.thresh > 0.001) {
      const p = fitProbs();
      XS.forEach((x, i) => {
        if ((p[i] >= t ? 1 : 0) !== YS[i]) el('circle', { cx: P(HX(x)), cy: P(Y(YS[i])), r: 9, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, ringsG);
      });
    }
    op(ringsG, s.thresh);

    // Curve key, top right of the plot area, in steps 4.
    curveKey.replaceChildren();
    if (s.newton > 0.001) {
      el('line', { x1: PX0 + 230, x2: PX0 + 250, y1: 16, y2: 16, stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '4 4' }, curveKey);
      el('text', { class: 'stage-text', x: PX0 + 256, y: 20, text: 'Newton' }, curveKey);
    }
    op(curveKey, s.newton);

    op(stripG, s.strip);
    drawStrip(b, w);

    op(chartG, s.chart);
    let d = '';
    for (let k = 0; k <= gd; k += k < 100 ? 1 : 10) d += `${k ? 'L' : 'M'}${P(SX(k))},${P(LY(path[k].loss))}`;
    d += `L${P(SX(gd))},${P(LY(path[gd].loss))}`;
    trace.setAttribute('d', d);

    // Read-out.
    const texts = readout(step, { b, w, t, gd });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let wd = 180;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) wd = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0 + 4, 598 - wd);
    boxRect.setAttribute('x', bx);
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', wd.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', bx + 12);
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    bS.out.textContent = num(b, 1);
    wS.out.textContent = num(w, 2);
    tS.out.textContent = num(t, 2);
    bS.input.setAttribute('aria-valuetext', `b = ${num(b, 1)}`);
    wS.input.setAttribute('aria-valuetext', `w = ${num(w, 2)}`);
    tS.input.setAttribute('aria-valuetext', `threshold ${num(t, 2)}`);
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

  // Updates the confusion matrix in the step 5 card.
  const cells = {};
  for (const key of ['cm-t', 'cm-tp', 'cm-fp', 'cm-tn', 'cm-fn']) cells[key] = doc.querySelector(`main.lesson [data-value="${key}"]`);
  function fillMatrix() {
    const c = confusionAt(t);
    const v = { 'cm-t': num(t, 2), 'cm-tp': String(c.tp), 'cm-fp': String(c.fp), 'cm-tn': String(c.tn), 'cm-fn': String(c.fn) };
    for (const [k, node] of Object.entries(cells)) if (node) node.textContent = v[k];
  }

  const spoken = (i) => {
    if (i === 0) return `Thirty students: ${YS.filter((v) => v === 1).length} passed, shown as triangles at 1, and the rest failed, circles at 0. The least-squares straight line through them is ${num(ols.at0, 2)} at 0 hours and ${num(ols.at10, 2)} at 10 hours, outside 0 to 1.`;
    if (i === 1) return `Sigmoid curve with b = ${num(b, 1)} and w = ${num(w, 2)}. ${w === 0 ? `It is flat at ${num(sigmoid(b), 2)}.` : `It crosses 0.5 at ${num(-b / w, 2)} hours.`}`;
    if (i === 2) {
      const wp = worstPoint(b, w);
      return `b = ${num(b, 1)}, w = ${num(w, 2)}. Mean log-loss ${num(meanLoss(b, w), 3)}. The largest penalty, ${num(wp.pen, 2)}, belongs to the student at ${num(wp.x, 2)} hours who ${wp.y ? 'passed' : 'failed'}.`;
    }
    if (i === 3) return `Gradient descent, step ${gd} of ${GD_STEPS}. Mean log-loss ${num(path[gd].loss, 5)}; the minimum is ${num(minLoss, 5)}.`;
    if (i === 4) {
      const c = confusionAt(t);
      return `Threshold ${num(t, 2)}: predict a pass from ${num(cutoffHours(t), 2)} hours. ${c.tp} true positives, ${c.fp} false positives, ${c.tn} true negatives, ${c.fn} false negatives. Precision ${c.precision === null ? 'undefined' : pct(c.precision)}, recall ${pct(c.recall)}.`;
    }
    return `Two features, ${B2.X.length} cases, 40 circles of class 0 and 40 triangles of class 1. Shading shows the predicted probability of class 1; the straight boundary is where it equals 0.5. Accuracy ${pct(acc2D())}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  bS.input.addEventListener('input', () => { b = clamp(Number(bS.input.value), -10, 4); render(); say(spoken(step)); });
  wS.input.addEventListener('input', () => { w = clamp(Number(wS.input.value), -1, 2.5); render(); say(spoken(step)); });
  tS.input.addEventListener('input', () => {
    t = clamp(Math.round(Number(tS.input.value) * 20) / 20, 0.05, 0.95);
    render();
    fillMatrix();
    say(spoken(4));
  });

  function runFit() {
    const my = ++fitToken;
    gd = 0;
    render();
    return tween(3000, (kt) => {
      if (my !== fitToken || step !== 3) return;
      gd = Math.round(GD_STEPS * kt);
      render();
    }, { reduced }).then(() => {
      if (my !== fitToken || step !== 3) return;
      gd = GD_STEPS;
      render();
      fitBtn.textContent = 'Fit again';
      announce(`Gradient descent finished: after ${GD_STEPS} steps the mean log-loss is ${num(path[gd].loss, 5)}, against the minimum ${num(minLoss, 5)} that Newton's method reached in ${fit.iters} steps.`);
    });
  }
  fitBtn.addEventListener('click', () => { runFit(); });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      fitToken++;
      showGroup(bw, i === 1 || i === 2);
      showGroup(fitGroup, i === 3);
      showGroup(tGroup, i === 4);
      if (i !== 3) {
        gd = 0;
        fitBtn.textContent = 'Fit';
      }
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
