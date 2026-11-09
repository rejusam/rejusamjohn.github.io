// L3.7, "Gradient boosting": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-gradient-boosting.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mse } from '../assets/maths.js';
import { boost } from '../assets/classify.js';
import { linearScale, clamp, el, axes, pathFrom, tween } from '../assets/plot.js';
import { sineData } from '../assets/sine-data.js';

export const N_TRAIN = 40;
export const SEED_TRAIN = 91;
export const N_TEST = 300;
export const SEED_TEST = 92;
export const SD = 0.25;
export const M_MAX = 300;
// The η slider's four positions; step 4 uses η = 0.1. Steps 1–3 add each
// stump in full, η = 1, so each change is easy to see.
export const ETAS = [0.05, 0.1, 0.3, 1];
export const ETA_EARLY = 1;
export const M_START = 20;
// Step 6 looks at the model after this many rounds with η = 0.1.
export const STEP6_M = 10;

export const trainData = () => sineData({ n: N_TRAIN, sd: SD, seed: SEED_TRAIN });
export const testData = () => sineData({ n: N_TEST, sd: SD, seed: SEED_TEST });

const hOf = (s, x) => (x < s.threshold ? s.left : s.right);

// One boosting run of M_MAX rounds per η, done once and kept. train[m] and
// test[m] are the MSEs of F_m, the model after m rounds (m = 0 is the mean).
const runs = new Map();
export function run(eta) {
  if (runs.has(eta)) return runs.get(eta);
  const tr = trainData();
  const te = testData();
  const b = boost(tr.xs, tr.ys, { M: M_MAX, eta });
  const F = te.xs.map(() => b.f0);
  const test = [mse(te.ys, F)];
  for (const s of b.stumps) {
    te.xs.forEach((x, i) => { F[i] += eta * hOf(s, x); });
    test.push(mse(te.ys, F));
  }
  const out = { eta, boost: b, train: b.trainMSE, test };
  runs.set(eta, out);
  return out;
}

// The round with the lowest test MSE (the first, if tied).
export function bestM(eta) {
  const t = run(eta).test;
  let best = 0;
  for (let m = 1; m < t.length; m++) if (t[m] < t[best]) best = m;
  return best;
}

// rᵢ = yᵢ − F_m(xᵢ) on the training points.
export function residuals(eta, m) {
  const { xs, ys } = trainData();
  const b = run(eta).boost;
  return xs.map((x, i) => ys[i] - b.predict(x, m));
}

// The model after m rounds as corner points of a step path on [−1, 1].
export function stepPoints(eta, m) {
  const b = run(eta).boost;
  const cuts = [...new Set(b.stumps.slice(0, m).map((s) => s.threshold).filter((t) => t > -1 && t < 1))].sort((a, c) => a - c);
  const edges = [-1, ...cuts, 1];
  const pts = [];
  for (let k = 0; k + 1 < edges.length; k++) {
    const v = b.predict((edges[k] + edges[k + 1]) / 2, m);
    pts.push([edges[k], v], [edges[k + 1], v]);
  }
  return pts;
}

// A single stump as a step path on [−1, 1].
export function stumpPoints(s) {
  if (!(s.threshold > -1 && s.threshold < 1)) return [[-1, s.right], [1, s.right]];
  return [[-1, s.left], [s.threshold, s.left], [s.threshold, s.right], [1, s.right]];
}

export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
}
export const fmtMSE = (v) => v.toFixed(3);
const etaText = (e) => String(e);

// Step numbers: 0–2 use η = 1 and rounds 1–2; 3 the rounds slider at η = 0.1;
// 4 adds the η slider; 5 the residuals after STEP6_M rounds at η = 0.1.
export function readout(step, { m = M_START, eta = 1 } = {}) {
  const r1 = run(ETA_EARLY);
  const [s1, s2] = r1.boost.stumps;
  if (step === 0) {
    return [`F₀ = ȳ = ${fmt(r1.boost.f0)}`, `one stump: split x = ${fmt(s1.threshold)}`, `levels ${fmt(r1.boost.f0 + s1.left)} and ${fmt(r1.boost.f0 + s1.right)}`, `train MSE: ${fmtMSE(r1.train[0])} → ${fmtMSE(r1.train[1])}`];
  }
  if (step === 1) {
    return ['after one stump (η = 1)', `train MSE: ${fmtMSE(r1.train[1])}`, '(mean of squared sticks)'];
  }
  if (step === 2) {
    return [`h₂: split x = ${fmt(s2.threshold)}`, `levels ${fmt(s2.left)} and ${fmt(s2.right)}`, 'F₂ = F₁ + 1 × h₂', `train MSE: ${fmtMSE(r1.train[1])} → ${fmtMSE(r1.train[2])}`];
  }
  if (step === 5) {
    const e = ETAS[1];
    const s = run(e).boost.stumps[STEP6_M];
    return [`after ${STEP6_M} rounds (η = ${e})`, 'rᵢ = −∂L/∂F at each xᵢ', `next stump: split x = ${fmt(s.threshold)}`];
  }
  const e = step === 3 ? ETAS[1] : ETAS[clamp(eta, 0, ETAS.length - 1)];
  const r = run(e);
  const mm = clamp(m, 0, M_MAX);
  const rows = [`η = ${etaText(e)}, M = ${mm}`, `train MSE: ${fmtMSE(r.train[mm])}`, `test MSE: ${fmtMSE(r.test[mm])}`];
  if (step === 4) rows.push(`lowest test: M = ${bestM(e)}`);
  return rows;
}

export const PIN_KEYS = ['ybar', 'mse0', 'split1', 'mse1', 'split2', 'mse2', 'start-train', 'start-test', 'best-m', 'best-test', 'end-test', 'end-train', 'best-m-0', 'best-m-2', 'best-m-3', 'best-test-3'];
export const PAGE_PIN = {
  'ybar': '0.12', 'mse0': '0.553', 'split1': '−0.01', 'mse1': '0.144', 'split2': '0.66', 'mse2': '0.124',
  'start-train': '0.124', 'start-test': '0.131', 'best-m': '111', 'best-test': '0.097', 'end-test': '0.110',
  'end-train': '0.025', 'best-m-0': '207', 'best-m-2': '34', 'best-m-3': '16', 'best-test-3': '0.106',
};

export function pageValues() {
  const r1 = run(ETA_EARLY);
  const [s1, s2] = r1.boost.stumps;
  const e = ETAS[1];
  const r = run(e);
  const best = bestM(e);
  const s6 = r.boost.stumps[STEP6_M];
  const out = {
    'ybar': fmt(r1.boost.f0),
    'mse0': fmtMSE(r1.train[0]),
    'split1': fmt(s1.threshold),
    'left1': fmt(r1.boost.f0 + s1.left),
    'right1': fmt(r1.boost.f0 + s1.right),
    'mse1': fmtMSE(r1.train[1]),
    'split2': fmt(s2.threshold),
    'h2-left': fmt(s2.left),
    'h2-right': fmt(s2.right),
    'mse2': fmtMSE(r1.train[2]),
    'start-train': fmtMSE(r.train[M_START]),
    'start-test': fmtMSE(r.test[M_START]),
    'best-m': String(best),
    'best-test': fmtMSE(r.test[best]),
    'best-train': fmtMSE(r.train[best]),
    'end-test': fmtMSE(r.test[M_MAX]),
    'end-train': fmtMSE(r.train[M_MAX]),
    'split6': fmt(s6.threshold),
  };
  ETAS.forEach((eta, k) => {
    out[`best-m-${k}`] = String(bestM(eta));
    out[`best-test-${k}`] = fmtMSE(run(eta).test[bestM(eta)]);
  });
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

const PX0 = 56;
const PX1 = 340;
const PY0 = 290;
const PY1 = 20;
const Y_DOM = [-1.6, 1.6];
const RX0 = 372;
const BOX_Y = 8;
const ROW = 20;
// The right-hand panel: residuals (steps 2, 3, 6) or the MSE chart (4, 5).
const CX0 = 420;
const CX1 = 586;
const CY0 = 286;
const CY1 = 152;
const R_DOM = [-1.5, 1.5];
const LOG_DOM = [-3, 0];

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const tr = trainData();
  const te = testData();
  const r1 = run(ETA_EARLY);
  let m = M_START;
  let etaK = 1;

  const X = linearScale([-1, 1], [PX0, PX1]);
  const Y = linearScale(Y_DOM, [PY0, PY1]);
  const yc = (v) => Y(clamp(v, ...Y_DOM));

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Forty training points around a wave and the boosted model, a staircase of flat steps. On the right, a read-out and either a plot of the residuals or a chart of training and test error against the number of rounds.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'gb-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  axes(svg, X, Y, { xLabel: 'x', yLabel: 'y' });
  const testG = el('g', {}, svg);
  te.xs.forEach((x, i) => el('rect', { x: X(x) - 2, y: yc(te.ys[i]) - 2, width: 4, height: 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1, opacity: 0.45 }, testG));

  const plotG = el('g', { 'clip-path': 'url(#gb-plot)' }, svg);
  const meanLine = el('line', { x1: PX0, x2: PX1, y1: Y(r1.boost.f0), y2: Y(r1.boost.f0), stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round' }, plotG);
  const splitLine = el('line', { y1: PY1, y2: PY0, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, plotG);
  const sticksG = el('g', {}, plotG);
  const sticks = tr.xs.map(() => el('line', { stroke: STRONG, 'stroke-width': 1.2, opacity: 0.75 }, sticksG));
  const prevPath = el('path', { fill: 'none', stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, plotG);
  const fitPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.4, 'stroke-linejoin': 'round' }, plotG);

  const trainG = el('g', {}, svg);
  tr.xs.forEach((x, i) => el('circle', { cx: X(x), cy: yc(tr.ys[i]), r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.2 }, trainG));

  // Keys, top-left of the plot.
  const keyFit = el('g', {}, svg);
  el('rect', { x: PX0 + 6, y: PY1 + 2, width: 150, height: 44, rx: 4, fill: SURFACE, 'fill-opacity': 0.9 }, keyFit);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 16, y2: PY1 + 16, stroke: CYAN, 'stroke-width': 2.4 }, keyFit);
  const keyFitT = el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 20, text: 'model' }, keyFit);
  const keyMean = el('g', {}, keyFit);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 34, y2: PY1 + 34, stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round' }, keyMean);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 38, text: 'mean, F₀' }, keyMean);
  const keyPrev = el('g', {}, keyFit);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 34, y2: PY1 + 34, stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, keyPrev);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 38, text: 'before, F₁' }, keyPrev);
  const keyTest = el('g', {}, keyFit);
  el('rect', { x: PX0 + 18, y: PY1 + 32, width: 4, height: 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1 }, keyTest);
  const kt = el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 38, text: `test (${N_TEST})` }, keyTest);
  kt.style.fill = VIOLET;
  const keySticks = el('g', {}, keyFit);
  el('line', { x1: PX0 + 20, x2: PX0 + 20, y1: PY1 + 28, y2: PY1 + 40, stroke: STRONG, 'stroke-width': 1.2 }, keySticks);
  el('text', { class: 'stage-text', x: PX0 + 34, y: PY1 + 38, text: 'residual' }, keySticks);

  // --- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Residual panel.
  const resG = el('g', {}, svg);
  const RX = linearScale([-1, 1], [CX0, CX1]);
  const RY = linearScale(R_DOM, [CY0, CY1]);
  const ry = (v) => RY(clamp(v, ...R_DOM));
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, resG);
  for (const k of [-1, 0, 1]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: RY(k), y2: RY(k), stroke: LINE }, resG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: RY(k) + 4, 'text-anchor': 'end', text: k < 0 ? `−${-k}` : String(k) }, resG);
  }
  el('line', { x1: CX0, x2: CX1, y1: RY(0), y2: RY(0), stroke: LINE }, resG);
  for (const k of [-1, 0, 1]) el('text', { class: 'stage-text', x: RX(k), y: CY0 + 16, 'text-anchor': 'middle', text: k < 0 ? `−${-k}` : String(k) }, resG);
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'x' }, resG);
  const resTitle = el('text', { class: 'stage-text', x: CX0, y: CY1 - 26, text: 'residuals rᵢ' }, resG);
  const resSticks = tr.xs.map((x) => el('line', { x1: RX(x), x2: RX(x), y1: RY(0), stroke: STRONG, 'stroke-width': 1.1, opacity: 0.75 }, resG));
  const resDots = tr.xs.map((x) => el('circle', { cx: RX(x), r: 2.6, fill: CYAN, stroke: BG, 'stroke-width': 0.8 }, resG));
  const stumpPath = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 2.2, 'stroke-dasharray': '6 3' }, resG);
  const stumpKey = el('g', {}, resG);
  el('line', { x1: CX0, x2: CX0 + 16, y1: CY1 - 12, y2: CY1 - 12, stroke: VIOLET, 'stroke-width': 2.2, 'stroke-dasharray': '6 3' }, stumpKey);
  const skT = el('text', { class: 'stage-text', x: CX0 + 22, y: CY1 - 8, text: 'next stump' }, stumpKey);
  skT.style.fill = VIOLET;

  // --- MSE chart (log scale) against rounds.
  const chartG = el('g', {}, svg);
  const MX = linearScale([0, M_MAX], [CX0 + 8, CX1 - 4]);
  const LY = linearScale(LOG_DOM, [CY0, CY1]);
  const ly = (v) => LY(clamp(Math.log10(v), ...LOG_DOM));
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, chartG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, chartG);
  for (const k of [-3, -2, -1, 0]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: LY(k), y2: LY(k), stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: LY(k) + 4, 'text-anchor': 'end', text: k === 0 ? '1' : `0.${'0'.repeat(-k - 1)}1` }, chartG);
  }
  for (const v of [0, 100, 200, 300]) {
    el('line', { x1: MX(v), x2: MX(v), y1: CY0, y2: CY0 + 4, stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: MX(v), y: CY0 + 16, 'text-anchor': 'middle', text: String(v) }, chartG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'rounds M' }, chartG);
  el('text', { class: 'stage-text', x: CX0 - 4, y: CY1 - 12, 'text-anchor': 'middle', text: 'MSE' }, chartG);
  const ckey = el('g', {}, chartG);
  el('line', { x1: CX0 + 30, x2: CX0 + 46, y1: CY1 - 16, y2: CY1 - 16, stroke: CYAN, 'stroke-width': 2 }, ckey);
  const ckA = el('text', { class: 'stage-text', x: CX0 + 50, y: CY1 - 12, text: 'train' }, ckey);
  ckA.style.fill = CYAN;
  el('line', { x1: CX0 + 100, x2: CX0 + 116, y1: CY1 - 16, y2: CY1 - 16, stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, ckey);
  const ckB = el('text', { class: 'stage-text', x: CX0 + 120, y: CY1 - 12, text: 'test' }, ckey);
  ckB.style.fill = VIOLET;
  const trainCurve = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, chartG);
  const testCurve = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, chartG);
  const minG = el('g', {}, chartG);
  const minLine = el('line', { stroke: VIOLET, 'stroke-width': 1.2, 'stroke-dasharray': '1 3', 'stroke-linecap': 'round' }, minG);
  const minRing = el('circle', { r: 5.5, fill: 'none', stroke: VIOLET, 'stroke-width': 1.8 }, minG);
  const minText = el('text', { class: 'stage-text' }, minG);
  minText.style.fill = VIOLET;
  const mMarker = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, chartG);
  const dotA = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.2 }, chartG);
  const dotB = el('rect', { width: 8, height: 8, fill: BG, stroke: VIOLET, 'stroke-width': 1.8 }, chartG);
  const ROUNDS = Array.from({ length: M_MAX + 1 }, (_, k) => k);
  const curveFor = (arr) => pathFrom(ROUNDS.map((k) => [k, clamp(Math.log10(arr[k]), ...LOG_DOM)]), MX, LY);
  const curves = ETAS.map((e) => ({ train: curveFor(run(e).train), test: curveFor(run(e).test), best: bestM(e) }));

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const mGroup = make('div');
  const mLabel = make('label');
  const mInput = make('input', { type: 'range', min: 0, max: M_MAX, step: 1, value: m });
  mInput.style.width = '160px';
  const mOut = make('output');
  mOut.setAttribute('aria-hidden', 'true');
  mLabel.append('M', make('span', { className: 'vh', textContent: ', the number of rounds' }), ' ', mInput, ' ', mOut);
  mGroup.append(mLabel);
  const eGroup = make('div');
  const eLabel = make('label');
  const eInput = make('input', { type: 'range', min: 0, max: ETAS.length - 1, step: 1, value: etaK });
  eInput.style.width = '90px';
  const eOut = make('output');
  eOut.setAttribute('aria-hidden', 'true');
  eLabel.append('η', make('span', { className: 'vh', textContent: ', the learning rate' }), ' ', eInput, ' ', eOut);
  eGroup.append(eLabel);
  controls.append(mGroup, eGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { mean: 0, split: 0, sticks: 0, prev: 0, test: 0, res: 0, stump: 0, chart: 0 };
  let s = { ...LAYERS, mean: 1, split: 1 };
  const target = (i) => {
    if (i === 0) return { ...LAYERS, mean: 1, split: 1 };
    if (i === 1) return { ...LAYERS, sticks: 1, res: 1 };
    if (i === 2) return { ...LAYERS, prev: 1, res: 1, stump: 1 };
    if (i === 3 || i === 4) return { ...LAYERS, test: 1, chart: 1 };
    return { ...LAYERS, sticks: 1, res: 1, stump: 1 };
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  // Which run and round the left plot shows, and the stump (if any) on the right.
  const view = () => {
    if (step <= 2) return { eta: ETA_EARLY, m: step === 2 ? 2 : 1, prevM: 1, resM: 1, stump: step === 2 ? r1.boost.stumps[1] : null };
    if (step === 5) return { eta: ETAS[1], m: STEP6_M, resM: STEP6_M, stump: run(ETAS[1]).boost.stumps[STEP6_M] };
    return { eta: step === 3 ? ETAS[1] : ETAS[etaK], m };
  };

  function render() {
    const v = view();
    const b = run(v.eta).boost;
    set(fitPath, { d: pathFrom(stepPoints(v.eta, v.m), X, Y) });
    if (v.prevM !== undefined) set(prevPath, { d: pathFrom(stepPoints(v.eta, v.prevM), X, Y) });
    op(prevPath, s.prev);
    op(meanLine, s.mean);
    const t1 = r1.boost.stumps[0].threshold;
    set(splitLine, { x1: X(t1), x2: X(t1) });
    op(splitLine, s.split);
    op(testG, s.test);
    // Sticks from each point to the model it is a residual of.
    const resM = v.resM ?? v.m;
    tr.xs.forEach((x, i) => set(sticks[i], { x1: X(x), x2: X(x), y1: yc(tr.ys[i]), y2: yc(b.predict(x, resM)) }));
    op(sticksG, s.sticks);
    op(keyMean, s.mean);
    op(keyPrev, s.prev);
    op(keyTest, s.test);
    op(keySticks, s.sticks);
    keyFitT.textContent = step === 0 ? 'one stump, F₁' : step === 2 ? 'after, F₂' : 'model';

    if (v.resM !== undefined) {
      tr.xs.forEach((x, i) => {
        const r = tr.ys[i] - b.predict(x, v.resM);
        set(resSticks[i], { y2: ry(r) });
        set(resDots[i], { cy: ry(r) });
      });
      resTitle.textContent = step === 5 ? 'rᵢ = −gradient' : 'residuals rᵢ of F₁';
    }
    if (v.stump) set(stumpPath, { d: pathFrom(stumpPoints(v.stump), RX, RY) });
    op(resG, s.res);
    op(stumpPath, s.stump);
    op(stumpKey, s.stump);

    const cv = curves[step === 3 ? 1 : etaK];
    const rr = run(step === 3 ? ETAS[1] : ETAS[etaK]);
    set(trainCurve, { d: cv.train });
    set(testCurve, { d: cv.test });
    set(mMarker, { x1: MX(m), x2: MX(m) });
    set(dotA, { cx: MX(m), cy: ly(rr.train[m]) });
    set(dotB, { x: MX(m) - 4, y: ly(rr.test[m]) - 4 });
    const bx = MX(cv.best);
    const by = ly(rr.test[cv.best]);
    set(minLine, { x1: bx, x2: bx, y1: by + 6, y2: CY0 });
    set(minRing, { cx: bx, cy: by });
    set(minText, { x: CX1, y: by - 12 });
    minText.setAttribute('text-anchor', 'end');
    minText.textContent = `lowest: M = ${cv.best}`;
    op(minG, step === 4 ? 1 : 0);
    op(chartG, s.chart);

    const texts = readout(step, { m, eta: etaK });
    rowsT.forEach((r, k) => {
      r.textContent = texts[k] || '';
      r.style.display = k < texts.length ? '' : 'none';
    });
    let w = 214;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bxr = Math.min(RX0 + 4, 598 - w);
    set(boxRect, { x: bxr, y: BOX_Y, width: w, height: 16 + ROW * texts.length });
    rowsT.forEach((r, k) => set(r, { x: bxr + 12, y: BOX_Y + 8 + ROW * k + ROW / 2 + 5 }));

    mOut.textContent = String(m);
    mInput.setAttribute('aria-valuetext', `${m} rounds`);
    eOut.textContent = etaText(ETAS[etaK]);
    eInput.setAttribute('aria-valuetext', `eta ${etaText(ETAS[etaK])}`);
  }

  function animateTo(to, ms) {
    const my = ++token;
    const start = { ...s };
    return tween(ms, (k) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * k;
      s = next;
      render();
    }, { reduced });
  }

  const spoken = (i) => {
    if (i === 0) return `The mean, then one stump: split at x ${pageValues().split1}, levels ${pageValues().left1} and ${pageValues().right1}. Training MSE falls from ${fmtMSE(r1.train[0])} to ${fmtMSE(r1.train[1])}.`;
    if (i === 1) return `Residual sticks from each point to the one-stump model. Training MSE ${fmtMSE(r1.train[1])}.`;
    if (i === 2) return `A second stump fitted to the residuals, split at x ${pageValues().split2}. Added in full, training MSE falls to ${fmtMSE(r1.train[2])}.`;
    if (i === 5) return `After ${STEP6_M} rounds with eta 0.1, the residuals are the negative gradient. The next stump splits at x ${pageValues().split6}.`;
    const e = i === 3 ? ETAS[1] : ETAS[etaK];
    const rr = run(e);
    const extra = i === 4 ? ` Lowest test MSE at ${bestM(e)} rounds, ${fmtMSE(rr.test[bestM(e)])}.` : '';
    return `Eta ${etaText(e)}, ${m} rounds. Training MSE ${fmtMSE(rr.train[m])}, test MSE ${fmtMSE(rr.test[m])}.${extra}`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  mInput.addEventListener('input', () => {
    m = clamp(Math.round(Number(mInput.value)), 0, M_MAX);
    render();
    say(spoken(step));
  });
  eInput.addEventListener('input', () => {
    etaK = clamp(Math.round(Number(eInput.value)), 0, ETAS.length - 1);
    render();
    say(spoken(step));
  });

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
      showGroup(mGroup, i === 3 || i === 4);
      showGroup(eGroup, i === 4);
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
