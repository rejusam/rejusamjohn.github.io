// Lesson 7, "Gradient descent": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from Lesson 6's data.

import { gdRun, mseGrad1D, mean } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, tween } from '../assets/plot.js';
import { regressionData, fitLine } from '../linear-regression/stage.js';

// Lesson 6's 25 points, with the mean of x taken off every x and the mean of
// y taken off every y. The model is ŷ = w·x on these centred values.
export function centredData() {
  const { xs, ys } = regressionData();
  const xbar = mean(xs);
  const ybar = mean(ys);
  return { cx: xs.map((x) => x - xbar), cy: ys.map((y) => y - ybar), xbar, ybar };
}

// On centred data, L(w) = mean((y − w·x)²) = a(w − w*)² + c, where
// a = mean(x²), w* = Σxy / Σx² (the least-squares slope) and c = L(w*).
export function lossParams(cx, cy) {
  const n = cx.length;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxx += cx[i] * cx[i];
    sxy += cx[i] * cy[i];
    syy += cy[i] * cy[i];
  }
  const wStar = sxy / sxx;
  return { a: sxx / n, wStar, c: (syy - wStar * sxy) / n };
}

export const loss = ({ a, wStar, c }, w) => a * (w - wStar) ** 2 + c;
// L′(w) = 2a(w − w*).
export const lossSlope = ({ a, wStar }, w) => 2 * a * (w - wStar);
// The same slope computed straight from the data, as gdRun uses it.
export const gradOf = (d) => mseGrad1D(d.cx, d.cy);
// Back on the original scale: ŷ = ȳ + w(x − x̄), so the intercept is ȳ − w·x̄.
export const intercept = (d, w) => d.ybar - w * d.xbar;

// Where every run starts, and the plot ranges.
export const W0 = -0.5;
export const W_DOMAIN = [-1, 4];
export const L_HI = 45;
// Steps drawn in steps 4 and 5 of the lesson.
export const K = 25;

// The learning rate: a fixed default for steps 3 and 4, and a log slider
// from ETA_MIN to 2.5/a for step 5.
export const ETA_DEFAULT = 0.03;
export const ETA_MIN = 0.001;
export const SLIDER_MAX = 1000;
export function etaFromSlider(t, a) {
  const hi = 2.5 / a;
  if (t <= 0) return ETA_MIN;
  if (t >= SLIDER_MAX) return hi;
  return ETA_MIN * (hi / ETA_MIN) ** (t / SLIDER_MAX);
}
export function sliderFromEta(eta, a) {
  const hi = 2.5 / a;
  return Math.round((SLIDER_MAX * Math.log(eta / ETA_MIN)) / Math.log(hi / ETA_MIN));
}

// Each step multiplies the distance to w* by r = 1 − 2aη.
export const rate = (eta, a) => 1 - 2 * a * eta;
export function regime(eta, a) {
  const r = rate(eta, a);
  if (Math.abs(r + 1) < 1e-12) return 'never';
  if (r < -1) return 'diverges';
  if (r < 0 && Math.abs(r) >= 1e-12) return 'zigzag';
  return 'steady';
}
export const REGIME_LABEL = {
  steady: 'steady',
  zigzag: 'zig-zags but settles',
  never: 'never settles',
  diverges: 'diverges',
};

// K steps of gradient descent from W0. gdRun stops early, with the path so
// far, if w runs away, so nothing returned is ever NaN or infinite.
export function runPath(eta, d, steps = K) {
  return gdRun({ grad: gradOf(d), init: [W0], eta, steps, tol: 0 });
}

// Run until the steps are tiny.
export function converge(d, eta) {
  const r = gdRun({ grad: gradOf(d), init: [W0], eta, steps: 100000, tol: 1e-10 });
  return { w: r.path[r.path.length - 1][0], steps: r.path.length - 1, status: r.status };
}

// The ball is drawn at w, or at the edge of the plot if w is off it.
export function ballView(w) {
  const [lo, hi] = W_DOMAIN;
  const cw = clamp(w, lo, hi);
  return { w: cw, clamped: cw !== w };
}

// Which way to move w to go downhill: +1 right, −1 left, 0 flat.
export function downhillDir(slope, tol = 0.005) {
  if (Math.abs(slope) < tol) return 0;
  return slope < 0 ? 1 : -1;
}

// Step 6's inset: a different, made-up bumpy curve, not this lesson's loss.
// It has two dips; the one on the left is lower.
export const bumpy = (u) => 0.25 * u ** 4 - u * u + 0.3 * u + 2;
export const bumpyPrime = (u) => u ** 3 - 2 * u + 0.3;
export const U_DOMAIN = [-2.2, 2.2];
export const U0 = 2;
export const BUMPY_ETA = 0.05;
export function bumpyRun() {
  const r = gdRun({ grad: ([u]) => [bumpyPrime(u)], init: [U0], eta: BUMPY_ETA, steps: 10000, tol: 1e-10 });
  return { u: r.path[r.path.length - 1][0], path: r.path.map(([u]) => u), status: r.status };
}
// The lowest point of the bumpy curve, by a grid search then Newton's method.
export function bumpyLowest() {
  let best = U_DOMAIN[0];
  for (let u = U_DOMAIN[0]; u <= U_DOMAIN[1]; u += 0.001) if (bumpy(u) < bumpy(best)) best = u;
  for (let k = 0; k < 20; k++) best -= bumpyPrime(best) / (3 * best * best - 2);
  return best;
}

// Fixed decimals with a true minus sign.
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}
// Learning rates to three significant figures, without trailing zeros.
export const fmtEta = (eta) => String(Number(eta.toPrecision(3)));

const DIR_ROW = { 1: 'negative: go right', [-1]: 'positive: go left', 0: 'zero: flat here' };

// The read-out rows for each step, as shown on the stage.
// state: { d, eta, w (the current w), trail (the ws of the current run) }.
export function readout(step, { d, eta, w, trail }) {
  const p = lossParams(d.cx, d.cy);
  const m = lossSlope(p, w);
  if (step === 0) return [`w = ${fmt(w)}`, `L(w) = ${fmt(loss(p, w))}`];
  if (step === 1) return [`w = ${fmt(w)}`, `slope = ${fmt(m)}`, DIR_ROW[downhillDir(m)]];
  if (step === 2) return [`η = ${fmtEta(eta)}`, `w = ${fmt(w)}`, `slope = ${fmt(m)}`, `next w = ${fmt(w - eta * m)}`];
  if (step === 3) {
    const k = trail.length - 1;
    const last = trail[k];
    const rows = [`step ${k} of ${K}`, `w = ${fmt(last)}`];
    rows.push(k > 0 ? `last step: ${fmt(Math.abs(last - trail[k - 1]), 3)}` : 'no step yet');
    rows.push(`Lesson 6 b̂ = ${fmt(p.wStar)}`);
    return rows;
  }
  if (step === 4) {
    const reg = regime(eta, p.a);
    const last = trail[trail.length - 1];
    const off = ballView(last).clamped;
    return [
      `η = ${fmtEta(eta)}`,
      REGIME_LABEL[reg],
      `settles if η < ${fmtEta(1 / p.a)}`,
      off ? 'w: off the chart' : `w after ${trail.length - 1}: ${fmt(last)}`,
    ];
  }
  const dir = downhillDir(m);
  return [`w = ${fmt(w)}`, `slope = ${fmt(m)}`, dir === 0 ? 'the lowest point' : DIR_ROW[dir]];
}

// Values quoted in the page text, all computed from the data.
export function pageValues() {
  const d = centredData();
  const p = lossParams(d.cx, d.cy);
  const { xs, ys } = regressionData();
  const ols = fitLine(xs, ys);
  const conv = converge(d, ETA_DEFAULT);
  const b = bumpyRun();
  const low = bumpyLowest();
  return {
    n: String(d.cx.length),
    xbar: fmt(d.xbar),
    ybar: fmt(d.ybar),
    a: fmt(p.a),
    'w-star': fmt(p.wStar, 4),
    'loss-min': fmt(p.c),
    'loss-w0': fmt(loss(p, W0)),
    w0: fmt(W0),
    'eta-default': fmtEta(ETA_DEFAULT),
    'eta-min': fmtEta(ETA_MIN),
    'eta-half': fmtEta(1 / (2 * p.a)),
    'eta-one': fmtEta(1 / p.a),
    'eta-max': fmtEta(2.5 / p.a),
    'r-default': fmt(rate(ETA_DEFAULT, p.a)),
    k: String(K),
    'w-conv': fmt(conv.w, 4),
    'conv-steps': String(conv.steps),
    'b-hat': fmt(ols.b, 4),
    'int-conv': fmt(intercept(d, conv.w), 4),
    'a-hat': fmt(ols.a, 4),
    'bump-stop': fmt(b.u),
    'bump-stop-h': fmt(bumpy(b.u)),
    'bump-low': fmt(low),
    'bump-low-h': fmt(bumpy(low)),
  };
}

// Words for numbers in the live region.
const say2 = (v, dp = 2) => {
  const s = fmt(v, dp);
  return s.startsWith('−') ? `minus ${s.slice(1)}` : s;
};

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

const PX0 = 56;
const PX1 = 372;
const PY0 = 290;
const PY1 = 20;
const SX0 = 430;
const SX1 = 588;
const SY0 = 290;
const SY1 = 158;
const BOX = { x: 398, y: 8, row: 20 };
// Half the drawn length of the tangent, in pixels.
const TAN_PX = 62;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const d = centredData();
  const p = lossParams(d.cx, d.cy);
  const L = (w) => loss(p, w);
  const conv = converge(d, ETA_DEFAULT);
  const full3 = runPath(ETA_DEFAULT, d).path.map(([w]) => w);

  let wDrag = W0;
  let trail2 = [W0];
  let trail3 = full3.slice();
  let eta = ETA_DEFAULT;
  let run4 = runPath(eta, d);
  let ballW = W0;

  const X = linearScale(W_DOMAIN, [PX0, PX1]);
  const Y = linearScale([0, L_HI], [PY0, PY1]);
  // Off-plot values are squeezed before drawing, so no coordinate is ever huge.
  const Xc = (w) => X(clamp(w, -50, 50));
  const Yc = (v) => Y(clamp(v, -1e4, 1e4));

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A U-shaped loss curve over the weight w, with a ball for the current w, and a small chart beside it that changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l7-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);
  const sclip = el('clipPath', { id: 'l7-side' }, defs);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1 }, sclip);
  const marker = el('marker', { id: 'l7-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 z', fill: STRONG }, marker);

  axes(svg, X, Y, { xLabel: 'weight w', yLabel: 'loss L(w)' });

  // --- Main plot.
  const ws = Array.from({ length: 161 }, (_, j) => W_DOMAIN[0] + (j / 160) * (W_DOMAIN[1] - W_DOMAIN[0]));
  const plotG = el('g', { 'clip-path': 'url(#l7-plot)' }, svg);
  el('path', { d: pathFrom(ws.map((w) => [w, L(w)]), X, Y), fill: 'none', stroke: INK, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, plotG);

  // The bottom of the bowl.
  const minG = el('g', {}, svg);
  el('line', { x1: X(p.wStar), x2: X(p.wStar), y1: Y(p.c), y2: PY0, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, minG);
  el('text', { class: 'stage-text', x: X(p.wStar) + 6, y: PY0 - 6, text: 'w* (lowest)', ...HALO }, minG);

  // Past positions and the hops between them.
  const trailG = el('g', { 'clip-path': 'url(#l7-plot)' }, svg);

  // The tangent at the ball and its slope.
  const tanG = el('g', {}, svg);
  const tanLine = el('line', { stroke: VIOLET, 'stroke-width': 2.2, 'stroke-linecap': 'round', 'clip-path': 'url(#l7-plot)' }, tanG);
  const tanLabel = el('text', { class: 'stage-text', 'text-anchor': 'middle', ...HALO }, tanG);
  tanLabel.style.fill = VIOLET;

  // Step 2: which way to move w, drawn just above the w axis.
  const arrowG = el('g', {}, svg);
  const guide = el('line', { stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '2 3' }, arrowG);
  const arrow = el('line', { stroke: STRONG, 'stroke-width': 2.2, 'marker-end': 'url(#l7-arrow)' }, arrowG);
  const arrowLabel = el('text', { class: 'stage-text', text: 'downhill', ...HALO }, arrowG);
  arrowLabel.style.fill = STRONG;

  // Step 5: the warning when the steps grow.
  const noteG = el('g', {}, svg);
  const cx0 = (PX0 + PX1) / 2;
  el('text', { class: 'note', x: cx0, y: 110, 'text-anchor': 'middle', text: 'Diverged: the steps grow', ...HALO }, noteG).style.fill = STRONG;
  el('text', { class: 'note', x: cx0, y: 128, 'text-anchor': 'middle', text: 'instead of shrinking', ...HALO }, noteG).style.fill = STRONG;

  const ball = el('circle', { r: 7, fill: CYAN, stroke: BG, 'stroke-width': 2 }, svg);
  const offLabel = el('text', { class: 'stage-text', text: '', ...HALO }, svg);
  offLabel.style.fill = STRONG;

  // --- Side panel: the centred data and the line ŷ = w·x (steps 1–3).
  const m = Math.max(...d.cx.map(Math.abs)) * 1.15;
  const my = Math.max(...d.cy.map(Math.abs)) * 1.15;
  const SXs = linearScale([-m, m], [SX0, SX1]);
  const SYs = linearScale([-my, my], [SY0, SY1]);
  const scatG = el('g', {}, svg);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1, fill: 'none', stroke: LINE_C }, scatG);
  el('line', { x1: SX0, x2: SX1, y1: SYs(0), y2: SYs(0), stroke: LINE_C }, scatG);
  el('line', { x1: SXs(0), x2: SXs(0), y1: SY0, y2: SY1, stroke: LINE_C }, scatG);
  d.cx.forEach((x, i) => el('circle', { cx: SXs(x), cy: SYs(d.cy[i]), r: 2.6, fill: MUTED }, scatG));
  const scatLine = el('line', { stroke: CYAN, 'stroke-width': 2, 'clip-path': 'url(#l7-side)' }, scatG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'line ŷ = w·x' }, scatG);

  // --- Side panel: w at each step (steps 4 and 5).
  const KX = linearScale([0, K], [SX0, SX1]);
  const KY = linearScale(W_DOMAIN, [SY0, SY1]);
  const iterG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE_C }, iterG);
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE_C }, iterG);
  el('line', { x1: SX0, x2: SX1, y1: KY(p.wStar), y2: KY(p.wStar), stroke: VIOLET, 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, iterG);
  el('text', { class: 'stage-text', x: SX1, y: KY(p.wStar) - 6, 'text-anchor': 'end', text: 'w*', ...HALO }, iterG).style.fill = VIOLET;
  el('text', { class: 'stage-text', x: SX0, y: SY0 + 16, 'text-anchor': 'middle', text: '0' }, iterG);
  el('text', { class: 'stage-text', x: SX1, y: SY0 + 16, 'text-anchor': 'middle', text: String(K) }, iterG);
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 16, 'text-anchor': 'middle', text: 'step' }, iterG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'w at each step' }, iterG);
  const iterPlot = el('g', { 'clip-path': 'url(#l7-side)' }, iterG);
  const iterPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 1.6 }, iterPlot);
  const iterDots = el('g', {}, iterPlot);

  // --- Side panel: a different, bumpy curve (step 6).
  const b = bumpyRun();
  const bLow = bumpyLowest();
  const UX = linearScale(U_DOMAIN, [SX0, SX1]);
  const UY = linearScale([0, 4], [SY0, SY1]);
  const bumpG = el('g', {}, svg);
  el('rect', { x: SX0, y: SY1, width: SX1 - SX0, height: SY0 - SY1, fill: 'none', stroke: LINE_C, 'stroke-dasharray': '4 3' }, bumpG);
  const us = Array.from({ length: 121 }, (_, j) => U_DOMAIN[0] + (j / 120) * (U_DOMAIN[1] - U_DOMAIN[0]));
  el('path', { d: pathFrom(us.map((u) => [u, bumpy(u)]), UX, UY), fill: 'none', stroke: INK, 'stroke-width': 2 }, bumpG);
  b.path.slice(0, 12).forEach((u) => el('circle', { cx: UX(u), cy: UY(bumpy(u)), r: 3, fill: CYAN, opacity: 0.4 }, bumpG));
  el('circle', { cx: UX(b.u), cy: UY(bumpy(b.u)), r: 5.5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, bumpG);
  el('text', { class: 'stage-text', x: UX(b.u), y: UY(bumpy(b.u)) - 16, 'text-anchor': 'end', text: 'stops here', ...HALO }, bumpG);
  el('rect', { x: UX(bLow) - 4.5, y: UY(bumpy(bLow)) - 4.5, width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, bumpG);
  el('text', { class: 'stage-text', x: UX(bLow), y: UY(bumpy(bLow)) - 12, 'text-anchor': 'middle', text: 'lower dip', ...HALO }, bumpG).style.fill = VIOLET;
  el('text', { class: 'value', x: SX0, y: SY1 - 28, text: 'a different curve,' }, bumpG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'not this loss' }, bumpG);

  // --- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE_C }, box);
  const rows = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const wGroup = make('div');
  const wLabel = make('label');
  const wInput = make('input', { type: 'range', min: W_DOMAIN[0], max: W_DOMAIN[1], step: 0.01, value: W0 });
  wInput.style.width = '140px';
  const wOut = make('output');
  wOut.setAttribute('aria-hidden', 'true');
  wLabel.append('w', make('span', { className: 'vh', textContent: ', the weight' }), ' ', wInput, ' ', wOut);
  wGroup.append(wLabel);
  const stepGroup = make('div');
  const stepBtn = make('button', { type: 'button', textContent: 'Take a step' });
  const againBtn = make('button', { type: 'button', textContent: 'Start again' });
  stepGroup.append(stepBtn, againBtn);
  const runGroup = make('div');
  const runBtn = make('button', { type: 'button', textContent: 'Run again' });
  runGroup.append(runBtn);
  const etaGroup = make('div');
  const etaLabel = make('label');
  const etaInput = make('input', { type: 'range', min: 0, max: SLIDER_MAX, step: 1, value: sliderFromEta(ETA_DEFAULT, p.a) });
  etaInput.style.width = '150px';
  const etaOut = make('output');
  etaOut.setAttribute('aria-hidden', 'true');
  etaLabel.append('Learning rate η', make('span', { className: 'vh', textContent: ', on a log scale' }), ' ', etaInput, ' ', etaOut);
  etaGroup.append(etaLabel);
  controls.append(wGroup, stepGroup, runGroup, etaGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  let hopToken = 0;
  const LAYERS = { tan: 0, arrow: 0, min: 0, trail: 0, scat: 0, iter: 0, bump: 0 };
  const target = (i) => {
    if (i === 0) return { ...LAYERS, scat: 1 };
    if (i === 1) return { ...LAYERS, tan: 1, arrow: 1, scat: 1 };
    if (i === 2) return { ...LAYERS, tan: 1, min: 1, trail: 1, scat: 1 };
    if (i === 3) return { ...LAYERS, min: 1, trail: 1, iter: 1 };
    if (i === 4) return { ...LAYERS, min: 1, trail: 1, iter: 1 };
    return { ...LAYERS, min: 1, bump: 1 };
  };
  let s = target(0);

  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };

  // The trail being shown, and the w the read-out describes.
  const currentTrail = () => (step === 2 ? trail2 : step === 3 ? trail3 : step === 4 ? run4.path.map(([w]) => w) : [ballW]);
  const readW = () => (step <= 1 ? wDrag : step === 5 ? conv.w : currentTrail()[currentTrail().length - 1]);

  const hopD = (w0, w1) => {
    const x0 = Xc(w0);
    const y0 = Yc(L(w0));
    const x1 = Xc(w1);
    const y1 = Yc(L(w1));
    const lift = 16 + 0.12 * Math.abs(x1 - x0);
    return { x0, y0, x1, y1, qx: (x0 + x1) / 2, qy: Math.min(y0, y1) - lift };
  };

  function drawTrail(list, upto) {
    trailG.replaceChildren();
    for (let k = 1; k < upto; k++) {
      const h = hopD(list[k - 1], list[k]);
      el('path', { d: `M${h.x0.toFixed(2)},${h.y0.toFixed(2)} Q${h.qx.toFixed(2)},${h.qy.toFixed(2)} ${h.x1.toFixed(2)},${h.y1.toFixed(2)}`, fill: 'none', stroke: CYAN, 'stroke-width': 1.4, 'stroke-dasharray': '4 3', opacity: 0.8 }, trailG);
    }
    for (let k = 0; k < upto - 1; k++) el('circle', { cx: Xc(list[k]), cy: Yc(L(list[k])), r: 4.5, fill: CYAN, opacity: 0.35 }, trailG);
  }

  function drawIter(list) {
    const pts = list.map((w, k) => [k, clamp(w, W_DOMAIN[0] - 0.2, W_DOMAIN[1] + 0.2)]);
    iterPath.setAttribute('d', pathFrom(pts, KX, KY));
    iterDots.replaceChildren();
    pts.forEach(([k, w]) => el('circle', { cx: KX(k), cy: KY(w), r: 2.6, fill: CYAN }, iterDots));
  }

  // Draws everything except the ball's in-flight position (see placeBall).
  function render() {
    const w = readW();
    const list = currentTrail();
    if (step >= 2 && step <= 4) drawTrail(list, list.length);
    op(trailG, s.trail);
    op(minG, s.min);

    // Tangent and slope label at the ball.
    const bv = ballView(w);
    const slope = lossSlope(p, bv.w);
    const bx = X(bv.w);
    const by = Y(L(bv.w));
    const dx = X(bv.w + 1) - X(bv.w);
    const dy = Y(L(bv.w) + slope) - Y(L(bv.w));
    const len = Math.hypot(dx, dy);
    set(tanLine, { x1: bx - (TAN_PX * dx) / len, y1: by - (TAN_PX * dy) / len, x2: bx + (TAN_PX * dx) / len, y2: by + (TAN_PX * dy) / len });
    tanLabel.textContent = `slope = ${fmt(slope)}`;
    set(tanLabel, { x: clamp(bx, PX0 + 60, PX1 - 60), y: Math.max(by - 18, PY1 + 2) });
    op(tanG, s.tan);

    // The direction arrow.
    const dir = downhillDir(slope);
    const ay = PY0 - 14;
    set(guide, { x1: bx, x2: bx, y1: by + 8, y2: PY0 });
    if (dir === 0) {
      arrow.style.display = 'none';
      arrowLabel.textContent = 'flat';
      arrowLabel.setAttribute('text-anchor', 'middle');
      set(arrowLabel, { x: bx, y: ay - 6 });
    } else {
      arrow.style.display = '';
      set(arrow, { x1: bx, x2: bx + dir * 46, y1: ay, y2: ay });
      arrowLabel.textContent = 'downhill';
      arrowLabel.setAttribute('text-anchor', dir > 0 ? 'start' : 'end');
      set(arrowLabel, { x: bx + dir * 52, y: ay + 4 });
    }
    op(arrowG, s.arrow);

    // The divergence warning (step 5 only).
    const diverging = step === 4 && regime(eta, p.a) === 'diverges';
    noteG.style.display = diverging ? '' : 'none';

    // Side panels.
    const sw = step <= 2 ? w : conv.w;
    set(scatLine, { x1: SXs(-m), x2: SXs(m), y1: SYs(-m * sw), y2: SYs(m * sw) });
    op(scatG, s.scat);
    drawIter(step === 4 ? run4.path.map(([v]) => v) : trail3);
    op(iterG, s.iter);
    op(bumpG, s.bump);

    // Read-out box.
    const texts = readout(step, { d, eta: step === 2 || step === 3 ? ETA_DEFAULT : eta, w, trail: list });
    rows.forEach((r, k) => {
      r.textContent = texts[k] || '';
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

    wOut.textContent = fmt(wDrag);
    wInput.setAttribute('aria-valuetext', `w ${say2(wDrag)}, loss ${say2(L(wDrag))}, slope ${say2(lossSlope(p, wDrag))}`);
    etaOut.textContent = fmtEta(eta);
    etaInput.setAttribute('aria-valuetext', `η ${fmtEta(eta)}, ${REGIME_LABEL[regime(eta, p.a)]}`);
  }

  // The ball, at w or partway along a hop.
  function placeBall(w, hop = null, t = 1) {
    const bv = ballView(w);
    let x = X(bv.w);
    let y = Y(L(bv.w));
    if (hop && t < 1) {
      const u = 1 - t;
      x = u * u * hop.x0 + 2 * u * t * hop.qx + t * t * hop.x1;
      y = u * u * hop.y0 + 2 * u * t * hop.qy + t * t * hop.y1;
      x = clamp(x, PX0, PX1);
      y = clamp(y, PY1, PY0);
    }
    set(ball, { cx: x, cy: y });
    if (bv.clamped && !(hop && t < 1)) {
      const right = bv.w === W_DOMAIN[1];
      offLabel.textContent = right ? 'off the chart →' : '← off the chart';
      offLabel.setAttribute('text-anchor', right ? 'end' : 'start');
      set(offLabel, { x: right ? x - 12 : x + 12, y: y + 22 });
      offLabel.style.display = '';
    } else {
      offLabel.style.display = 'none';
    }
  }

  function draw() {
    render();
    placeBall(readW());
  }

  function animateTo(to, ms) {
    const my = ++token;
    const from = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = from[key] + (to[key] - from[key]) * kt;
      s = next;
      draw();
    }, { reduced });
  }

  // One hop of the ball from w0 to w1, drawn along its arc.
  function hop(w0, w1, ms, my) {
    const h = hopD(w0, w1);
    if (reduced || ms === 0) return Promise.resolve();
    return tween(ms, (kt) => {
      if (my !== hopToken) return;
      placeBall(w1, h, kt);
    }, { reduced });
  }

  // --- Words for the live region.
  const spoken = (i) => {
    const w = readW();
    if (i === 0) return `w ${say2(w)}. The line through the centred points has slope ${say2(w)}, and its loss is ${say2(L(w))}.`;
    if (i === 1) {
      const sl = lossSlope(p, w);
      const dir = downhillDir(sl);
      return `At w ${say2(w)} the slope is ${say2(sl)}: ${dir === 0 ? 'flat, at the bottom' : dir > 0 ? 'negative, so downhill is to the right' : 'positive, so downhill is to the left'}.`;
    }
    if (i === 2) {
      const sl = lossSlope(p, w);
      return `w ${say2(w)}, slope ${say2(sl)}. The next step goes to ${say2(w - ETA_DEFAULT * sl)}.`;
    }
    if (i === 3) {
      const k = trail3.length - 1;
      return `After ${k} steps, w is ${say2(trail3[k])}. The least-squares slope from Lesson 6 is ${say2(p.wStar)}.`;
    }
    if (i === 4) {
      const list = run4.path.map(([v]) => v);
      const last = list[list.length - 1];
      const reg = regime(eta, p.a);
      const where = ballView(last).clamped ? 'w has run off the chart' : `w is ${say2(last)}`;
      return `Learning rate ${fmtEta(eta)}: ${REGIME_LABEL[reg]}. After ${list.length - 1} steps ${where}.${reg === 'diverges' ? ' Diverged: the steps grow instead of shrinking.' : ''}`;
    }
    return `The ball rests at the bottom, w ${say2(conv.w)}, where the slope is 0. Beside it, a different bumpy curve: there, gradient descent stops in the higher dip.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction.
  wInput.addEventListener('input', () => {
    wDrag = clamp(Number(wInput.value), W_DOMAIN[0], W_DOMAIN[1]);
    draw();
    say(spoken(step));
  });

  stepBtn.addEventListener('click', () => {
    const my = ++hopToken;
    const w0 = trail2[trail2.length - 1];
    const w1 = w0 - ETA_DEFAULT * lossSlope(p, w0);
    hop(w0, w1, 450, my).then(() => {
      if (my !== hopToken) return;
      trail2 = [...trail2, w1];
      draw();
      announce(spoken(2));
    });
  });
  againBtn.addEventListener('click', () => {
    hopToken++;
    trail2 = [W0];
    draw();
    announce(`Back to the start. ${spoken(2)}`);
  });

  async function run3(animate) {
    const my = ++hopToken;
    if (!animate || reduced) {
      trail3 = full3.slice();
      draw();
      return;
    }
    trail3 = [full3[0]];
    draw();
    for (let k = 1; k < full3.length; k++) {
      // eslint-disable-next-line no-await-in-loop
      await hop(full3[k - 1], full3[k], k < 8 ? 380 : 120, my);
      if (my !== hopToken) return;
      trail3 = full3.slice(0, k + 1);
      draw();
    }
    announce(spoken(3));
  }
  runBtn.addEventListener('click', () => { run3(true); });

  etaInput.addEventListener('input', () => {
    eta = etaFromSlider(Number(etaInput.value), p.a);
    run4 = runPath(eta, d);
    draw();
    say(spoken(4));
  });

  // Sentences on the page that quote values from the data (hidden until filled).
  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const el_ of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...el_.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) el_.hidden = false;
  }

  draw();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(draw, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      hopToken++;
      showGroup(wGroup, i <= 1);
      showGroup(stepGroup, i === 2);
      showGroup(runGroup, i === 3);
      showGroup(etaGroup, i === 4);
      if (i === 5) ballW = conv.w;
      announce(spoken(i));
      const to = target(i);
      const fade = () => {
        if (!animate) {
          token++;
          s = { ...to };
          draw();
          return Promise.resolve();
        }
        return animateTo(to, 450);
      };
      if (i === 3) {
        if (animate && !reduced) {
          trail3 = [full3[0]];
          return fade().then(() => run3(true));
        }
        trail3 = full3.slice();
      }
      return fade();
    },
  };
}
