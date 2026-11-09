// Lesson 4, "Measuring error": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, uniform, randn, olsFit, mean, median, mse, mae } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, fmt as tickFmt, tween, draggable } from '../assets/plot.js';

export const SEED = 5;
export const N = 11;
// The point you can drag in step 5 (the third one generated).
export const CHOSEN = 2;
// Its miss can be dragged to at most KMAX times its starting size.
export const KMAX = 4;
export const Y_LO = 0;
export const Y_HI = 14;
// The range of the constant guess c in step 6.
export const C_LO = 0;
export const C_HI = 12;
export const C_START = 3;

// 11 made-up points. For each point in turn: x ~ Uniform(0, 10), then
// y = 2 + 0.8x + a standard normal draw.
export function lessonData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = uniform(rng, 0, 10);
    xs.push(x);
    ys.push(2 + 0.8 * x + randn(rng));
  }
  return { xs, ys };
}

// The least-squares line ŷ = a + b·x. It is fitted once and then stays fixed.
export function fitLine(xs, ys) {
  const r = olsFit(xs, ys);
  if (!r.ok) throw new Error('lesson data has no spread in x');
  return { a: r.intercept, b: r.slope };
}

export const predict = ({ a, b }, x) => a + b * x;

// Residuals e = y − ŷ against a line.
export const residualsOf = (xs, ys, line) => ys.map((y, i) => y - predict(line, xs[i]));

// Everything the stage says about a set of residuals.
export function lossSummary(es) {
  const n = es.length;
  let above = 0;
  let below = 0;
  let nAbove = 0;
  let nBelow = 0;
  for (const e of es) {
    if (e > 0) { above += e; nAbove++; } else if (e < 0) { below += e; nBelow++; }
  }
  const zeros = new Array(n).fill(0);
  const m = mse(es, zeros);
  return {
    n,
    sum: es.reduce((s, e) => s + e, 0),
    above,
    below,
    nAbove,
    nBelow,
    sumAbs: es.reduce((s, e) => s + Math.abs(e), 0),
    sumSq: es.reduce((s, e) => s + e * e, 0),
    mae: mae(es, zeros),
    mse: m,
    rmse: Math.sqrt(m),
  };
}

// The losses of one constant guess c for every point.
export const mseAt = (ys, c) => mse(ys, ys.map(() => c));
export const maeAt = (ys, c) => mae(ys, ys.map(() => c));

// Where the chosen point may be dragged: its miss stays within KMAX times
// its starting size, and the point stays on the stage.
export function dragRange(xs, ys, line, i = CHOSEN) {
  const yhat = predict(line, xs[i]);
  const e0 = ys[i] - yhat;
  const span = KMAX * Math.abs(e0);
  return { yhat, e0, lo: Math.max(Y_LO, yhat - span), hi: Math.min(Y_HI, yhat + span) };
}

// How many times its starting size a miss is.
export const kFactor = (e, e0) => Math.abs(e) / Math.abs(e0);

// Fixed decimals with a true minus sign.
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

// Same, with a plus sign on positive values.
export function signed(v, dp = 2) {
  const s = fmt(v, dp);
  return s.startsWith('−') || Number(Math.abs(v).toFixed(dp)) === 0 ? s : `+${s}`;
}

// The read-out rows for each step, as shown on the stage.
// state: { xs, ys (current), line, start (starting ys), c }.
export function readout(step, { xs, ys, line, start, c }) {
  const es = residualsOf(xs, ys, line);
  const L = lossSummary(es);
  const n = L.n;
  if (step === 0) {
    const yhat = predict(line, xs[CHOSEN]);
    return ['the ringed point', `y = ${fmt(ys[CHOSEN])}`, `ŷ = ${fmt(yhat)}`, `e = y − ŷ = ${signed(es[CHOSEN])}`];
  }
  if (step === 1) {
    return [`${L.nAbove} above: ${signed(L.above)}`, `${L.nBelow} below: ${signed(L.below)}`, `sum of e: ${fmt(L.sum)}`];
  }
  if (step === 2) return [`sum of |e|: ${fmt(L.sumAbs)}`, `MAE = ${fmt(L.sumAbs)}/${n} = ${fmt(L.mae)}`];
  if (step === 3) return [`sum of e²: ${fmt(L.sumSq)}`, `MSE = ${fmt(L.sumSq)}/${n} = ${fmt(L.mse)}`];
  if (step === 4) {
    const L0 = lossSummary(residualsOf(xs, start, line));
    const e0 = start[CHOSEN] - predict(line, xs[CHOSEN]);
    const k = kFactor(es[CHOSEN], e0);
    return [`miss: ×${fmt(k)}`, `miss²: ×${fmt(k * k)}`, `MAE: ${fmt(L0.mae)} → ${fmt(L.mae)}`, `MSE: ${fmt(L0.mse)} → ${fmt(L.mse)}`];
  }
  const m = mseAt(ys, c);
  return [`c = ${fmt(c)}`, `MAE = ${fmt(maeAt(ys, c))}`, `MSE = ${fmt(m)}`, `RMSE = ${fmt(Math.sqrt(m))}`];
}

// Values quoted in the page text, all from the starting data.
export function pageValues(xs, ys) {
  const line = fitLine(xs, ys);
  const L = lossSummary(residualsOf(xs, ys, line));
  const my = mean(ys);
  const md = median(ys);
  const r = dragRange(xs, ys, line);
  return {
    n: String(L.n),
    'n-above': String(L.nAbove),
    'n-below': String(L.nBelow),
    above: signed(L.above),
    below: signed(L.below),
    'sum-abs': fmt(L.sumAbs),
    mae: fmt(L.mae),
    'sum-sq': fmt(L.sumSq),
    mse: fmt(L.mse),
    rmse: fmt(L.rmse),
    e0: signed(r.e0),
    'e0-3': signed(3 * r.e0),
    'e0-sq': fmt(r.e0 ** 2),
    'e0-3-sq': fmt((3 * r.e0) ** 2),
    mean: fmt(my),
    median: fmt(md),
    'mse-at-mean': fmt(mseAt(ys, my)),
    'mae-at-median': fmt(maeAt(ys, md)),
    'mse-at-median': fmt(mseAt(ys, md)),
    'mae-at-mean': fmt(maeAt(ys, my)),
  };
}

// Words for numbers in the live region.
const say2 = (v) => {
  const s = fmt(v);
  return s.startsWith('−') ? `minus ${s.slice(1)}` : s;
};
const saySigned = (v) => {
  const s = fmt(v);
  if (s.startsWith('−')) return `minus ${s.slice(1)}`;
  return Number(s) === 0 ? s : `plus ${s}`;
};

// ---------------------------------------------------------------- the stage

const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: '#1e293b', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

// Main plot on the left; the side panel on the right holds the read-out box
// and a small chart for each step.
const PX0 = 56;
const PX1 = 330;
const PY0 = 290;
const PY1 = 20;
const SX0 = 386;
const SX1 = 588;
const SY0 = 290;
const SY1 = 126;
const BOX = { x: 362, y: 8, row: 20 };
const R = 5;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const { xs, ys: start } = lessonData();
  const line = fitLine(xs, start);
  const n = xs.length;
  const range = dragRange(xs, start, line);
  const yMean = mean(start);
  const yMedian = median(start);
  let ys = start.slice();
  let c = C_START;

  const X = linearScale([0, 10], [PX0, PX1]);
  const Y = linearScale([Y_LO, Y_HI], [PY0, PY1]);
  const unit = PY0 - Y(1);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Scatter plot of 11 points with a fixed straight line and a vertical stick from each point to the line, beside a small chart that changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l4-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: 356 - PX0, height: PY0 - PY1 }, clip);
  const clipLine = el('clipPath', { id: 'l4-line' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clipLine);

  axes(svg, X, Y, { xLabel: 'x', yLabel: 'y' });

  // --- Main plot layers.
  const squaresG = el('g', { 'clip-path': 'url(#l4-plot)' }, svg);
  const sq = xs.map(() => el('rect', { fill: CYAN, 'fill-opacity': 0.1, stroke: CYAN, 'stroke-width': 1 }, squaresG));
  const refsG = el('g', {}, svg);
  const meanLine = el('line', { x1: PX0, x2: PX1, stroke: CYAN, 'stroke-width': 1.4, opacity: 0.85 }, refsG);
  const medianLine = el('line', { x1: PX0, x2: PX1, stroke: VIOLET, 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }, refsG);
  const meanLabel = el('text', { class: 'value', x: PX0 + 6, ...HALO }, refsG);
  const medianLabel = el('text', { class: 'value', x: PX0 + 6, ...HALO }, refsG);
  meanLabel.style.fill = CYAN;
  medianLabel.style.fill = VIOLET;
  const lineG = el('g', { 'clip-path': 'url(#l4-line)' }, svg);
  const fitted = el('line', { stroke: STRONG, 'stroke-width': 2.2, 'stroke-linecap': 'round' }, lineG);
  const cLabel = el('text', { class: 'value', x: PX1 + 6, ...HALO }, svg);
  const sticksG = el('g', {}, svg);
  const sticks = xs.map(() => el('line', { 'stroke-linecap': 'round' }, sticksG));
  const ring = el('circle', { r: R + 5, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, svg);
  const focusRing = el('circle', { r: R + 9, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, svg);
  const dotsG = el('g', {}, svg);
  const order = [...xs.keys()].filter((i) => i !== CHOSEN).concat(CHOSEN);
  const dots = [];
  for (const i of order) dots[i] = el('circle', { class: 'dot', r: R, stroke: BG, 'stroke-width': 1.5 }, dotsG);
  const handle = el('circle', {
    class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'slider',
    'aria-label': 'The ringed point’s y value',
    'aria-valuemin': fmt(range.lo), 'aria-valuemax': fmt(range.hi),
  }, svg);
  handle.style.outline = 'none';

  // --- Side panel: read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: '#1e293b', 'fill-opacity': 0.92, stroke: LINE }, box);
  const rows = [0, 1, 2, 3].map(() => ({ sample: el('line', {}, box), text: el('text', { class: 'value' }, box), off: 12 }));
  const SAMPLES = {
    abs: { stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' },
    sq: { stroke: CYAN, 'stroke-width': 2.2 },
  };
  const ROW_SAMPLES = [[], [], [], [], ['abs', 'sq'], [null, 'abs', null, 'sq']];

  // --- Side panel, step 2: the residuals laid end to end.
  const walkG = el('g', {}, svg);
  const WALK = { up: 446, down: 516, base: SY0 - 14 };
  el('line', { x1: WALK.up - 26, x2: WALK.down + 26, y1: WALK.base, y2: WALK.base, stroke: MUTED, 'stroke-width': 1 }, walkG);
  el('text', { class: 'stage-text', x: WALK.up - 32, y: WALK.base + 4, 'text-anchor': 'end', text: '0' }, walkG);
  el('text', { class: 'stage-text', x: WALK.up, y: WALK.base + 22, 'text-anchor': 'middle', text: 'above' }, walkG);
  el('text', { class: 'stage-text', x: WALK.down, y: WALK.base + 22, 'text-anchor': 'middle', text: 'below' }, walkG);
  const walkLink = el('line', { stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '1 4', 'stroke-linecap': 'round' }, walkG);
  const walkSegs = xs.map(() => el('line', { 'stroke-width': 3 }, walkG));
  const walkJoints = xs.map(() => el('line', { stroke: BG, 'stroke-width': 2 }, walkG));
  const walkEnd = el('circle', { r: 4.5, fill: BG, stroke: STRONG, 'stroke-width': 2 }, walkG);

  // --- Side panel, steps 3 and 4: one bar per point, |e| turning into e².
  const byX = [...xs.keys()].sort((i, j) => xs[i] - xs[j]);
  const es0 = residualsOf(xs, start, line);
  const L0 = lossSummary(es0);
  const barMax = Math.max(1, ...es0.map((e) => e * e), ...es0.map((e) => Math.abs(e)));
  // The box has only two rows in these steps, so the bars can use more height.
  const BAR_TOP = 96;
  const B = linearScale([0, barMax], [SY0, BAR_TOP]);
  const barsG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE }, barsG);
  const barTicks = [];
  for (let t = 0; t <= barMax; t += 1) {
    el('line', { x1: SX0 - 4, x2: SX0, y1: B(t), y2: B(t), stroke: LINE }, barsG);
    if (t % 2 === 0) barTicks.push({ t, text: el('text', { class: 'stage-text', x: SX0 - 8, y: B(t) + 4, 'text-anchor': 'end', text: tickFmt(t) }, barsG) });
  }
  const barCap = el('text', { class: 'value', x: SX0, y: SY1 - 8 }, barsG);
  const slot = (SX1 - SX0) / n;
  const bars = byX.map(() => el('rect', { width: Math.min(13, slot - 5) }, barsG));
  const barRef = el('line', { x1: SX0, x2: SX1 }, barsG);
  // The average's name sits left of the chart, where no bar can cover it.
  const barRefLabel = el('text', { class: 'stage-text', x: SX0 - 8, 'text-anchor': 'end', ...HALO }, barsG);

  // --- Side panel, step 5: one miss k times as big.
  const K = linearScale([0, KMAX], [SX0, SX1]);
  const KY = linearScale([0, KMAX * KMAX], [SY0, SY1]);
  const kG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE }, kG);
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE }, kG);
  for (let k = 0; k <= KMAX; k++) {
    el('line', { x1: K(k), x2: K(k), y1: SY0, y2: SY0 + 4, stroke: LINE }, kG);
    el('text', { class: 'stage-text', x: K(k), y: SY0 + 17, 'text-anchor': 'middle', text: tickFmt(k) }, kG);
  }
  for (let v = 0; v <= KMAX * KMAX; v += 4) {
    el('line', { x1: SX0 - 4, x2: SX0, y1: KY(v), y2: KY(v), stroke: LINE }, kG);
    el('text', { class: 'stage-text', x: SX0 - 8, y: KY(v) + 4, 'text-anchor': 'end', text: tickFmt(v) }, kG);
  }
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 35, 'text-anchor': 'middle', text: 'miss × k' }, kG);
  const ks = Array.from({ length: 81 }, (_, j) => (j / 80) * KMAX);
  el('path', { d: pathFrom(ks.map((k) => [k, k]), K, KY), ...SAMPLES.abs, fill: 'none' }, kG);
  el('path', { d: pathFrom(ks.map((k) => [k, k * k]), K, KY), ...SAMPLES.sq, fill: 'none' }, kG);
  const kGuide = el('line', { y1: SY0, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, kG);
  const kDotAbs = el('rect', { width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, kG);
  const kDotSq = el('circle', { r: 5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, kG);
  const kLabelSq = el('text', { class: 'stage-text', x: K(2.55) - 6, y: KY(2.55 * 2.55), 'text-anchor': 'end', text: '× k²', ...HALO }, kG);
  const kLabelAbs = el('text', { class: 'stage-text', x: SX1, y: KY(KMAX) - 10, 'text-anchor': 'end', text: '× k', ...HALO }, kG);
  kLabelSq.style.fill = CYAN;
  kLabelAbs.style.fill = VIOLET;

  // --- Side panel, step 6: MAE and RMSE for every constant guess c.
  const cs = [];
  for (let j = 0; j <= 240; j++) cs.push(C_LO + (j / 240) * (C_HI - C_LO));
  // Include each y value exactly, so MAE's corners are drawn sharp.
  for (const y of start) if (y > C_LO && y < C_HI) cs.push(y);
  cs.sort((p, q) => p - q);
  const curveMax = Math.max(...cs.map((v) => Math.max(maeAt(start, v), Math.sqrt(mseAt(start, v)))));
  const CX = linearScale([C_LO, C_HI], [SX0, SX1]);
  const CY = linearScale([0, Math.ceil(curveMax / 2) * 2], [SY0, SY1]);
  const cG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE }, cG);
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE }, cG);
  for (const t of CX.ticks(3)) {
    el('line', { x1: CX(t), x2: CX(t), y1: SY0, y2: SY0 + 4, stroke: LINE }, cG);
    el('text', { class: 'stage-text', x: CX(t), y: SY0 + 17, 'text-anchor': 'middle', text: tickFmt(t) }, cG);
  }
  for (const t of CY.ticks(4)) {
    el('line', { x1: SX0 - 4, x2: SX0, y1: CY(t), y2: CY(t), stroke: LINE }, cG);
    el('text', { class: 'stage-text', x: SX0 - 8, y: CY(t) + 4, 'text-anchor': 'end', text: tickFmt(t) }, cG);
  }
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 35, 'text-anchor': 'middle', text: 'constant guess c' }, cG);
  const maeCurve = el('path', { ...SAMPLES.abs, fill: 'none' }, cG);
  const rmseCurve = el('path', { ...SAMPLES.sq, fill: 'none' }, cG);
  const cGuide = el('line', { y1: SY0, y2: SY1, stroke: STRONG, 'stroke-width': 1.2 }, cG);
  const cDotMae = el('rect', { width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, cG);
  const cDotRmse = el('circle', { r: 5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, cG);

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const tripleBtn = make('button', { type: 'button', textContent: 'Make its miss 3 times as big' });
  const backBtn = make('button', { type: 'button', textContent: 'Put it back' });
  const dragGroup = make('div');
  dragGroup.append(tripleBtn, backBtn);
  const cGroup = make('div');
  const cLabelEl = make('label');
  const cInput = make('input', { type: 'range', min: C_LO, max: C_HI, step: 0.01, value: C_START });
  cInput.style.width = '120px';
  const cOut = make('output');
  cOut.setAttribute('aria-hidden', 'true');
  cLabelEl.append('c', make('span', { className: 'vh', textContent: ', the constant guess' }), ' ', cInput, ' ', cOut);
  const meanBtn = make('button', { type: 'button', textContent: 'Use the mean' });
  const medianBtn = make('button', { type: 'button', textContent: 'Use the median' });
  cGroup.append(cLabelEl, meanBtn, medianBtn);
  controls.append(dragGroup, cGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Layer opacities and the line's shape animate between steps.
  let step = 0;
  let token = 0;
  let valueToken = 0;
  let focused = false;
  const LAYERS = { ring: 0, squares: 0, walk: 0, bars: 0, barSq: 0, kchart: 0, cchart: 0, refs: 0, toC: 0 };
  let s = { ...LAYERS, ring: 1 };

  function target(i) {
    if (i === 0) return { ...LAYERS, ring: 1 };
    if (i === 1) return { ...LAYERS, walk: 1 };
    if (i === 2) return { ...LAYERS, bars: 1 };
    if (i === 3) return { ...LAYERS, squares: 1, bars: 1, barSq: 1 };
    if (i === 4) return { ...LAYERS, squares: 1, ring: 1, kchart: 1 };
    return { ...LAYERS, cchart: 1, refs: 1, toC: 1 };
  }

  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };

  function setRow(k, text, key) {
    const r = rows[k];
    r.text.textContent = text || '';
    r.off = key ? 40 : 12;
    r.sample.style.display = key ? '' : 'none';
    for (const a of ['stroke', 'stroke-width', 'stroke-dasharray']) r.sample.removeAttribute(a);
    if (key) for (const [a, v] of Object.entries(SAMPLES[key])) r.sample.setAttribute(a, v);
  }

  function render() {
    const u = s.toC;
    const base = (x) => (1 - u) * predict(line, x) + u * c;
    const es = residualsOf(xs, ys, line);

    // The line: the fitted line, turning flat at height c in step 6.
    set(fitted, { x1: X(0), x2: X(10), y1: Y(base(0)), y2: Y(base(10)) });
    cLabel.textContent = 'c';
    set(cLabel, { y: Y(c) + 4 });
    op(cLabel, s.cchart);

    // Sticks from each point to the line (or to c).
    sticks.forEach((st, i) => {
      const e = ys[i] - base(xs[i]);
      set(st, { x1: X(xs[i]), x2: X(xs[i]), y1: Y(ys[i]), y2: Y(base(xs[i])) });
      let stroke = MUTED;
      let width = 1.6;
      let dash = null;
      if (step === 1) {
        stroke = e >= 0 ? INK : MUTED;
        dash = e >= 0 ? null : '4 3';
        width = 2;
      } else if (step === 2) {
        stroke = VIOLET;
        dash = e >= 0 ? null : '4 3';
        width = 2;
      } else if ((step === 0 || step === 4) && i === CHOSEN) {
        stroke = STRONG;
        width = 2.6;
      }
      st.setAttribute('stroke', stroke);
      st.setAttribute('stroke-width', width);
      if (dash) st.setAttribute('stroke-dasharray', dash);
      else st.removeAttribute('stroke-dasharray');
    });

    // A square on each stick: its side is the miss, so its area is the miss squared.
    sq.forEach((r, i) => {
      const side = Math.abs(es[i]) * unit;
      const px = X(xs[i]);
      const right = px + side <= 352;
      set(r, {
        x: right ? px : px - side,
        y: Math.min(Y(ys[i]), Y(predict(line, xs[i]))),
        width: side,
        height: side,
      });
      const bright = step !== 4 || i === CHOSEN;
      r.setAttribute('opacity', bright ? 1 : 0.3);
    });
    op(squaresG, s.squares);

    // Mean and median of y (step 6).
    set(meanLine, { y1: Y(yMean), y2: Y(yMean) });
    set(medianLine, { y1: Y(yMedian), y2: Y(yMedian) });
    // Names only: the values are in the text and in the box after "Use the mean" or "Use the median".
    meanLabel.textContent = 'mean';
    medianLabel.textContent = 'median';
    set(meanLabel, { y: Y(yMean) - 5 });
    // Below its line, so it never sits on the mean line just above.
    set(medianLabel, { y: Y(yMedian) + 13 });
    op(refsG, s.refs);

    // Dots, the ring, and the drag handle on the chosen point.
    xs.forEach((x, i) => set(dots[i], { cx: X(x), cy: Y(ys[i]) }));
    const hx = X(xs[CHOSEN]);
    const hy = Y(ys[CHOSEN]);
    set(ring, { cx: hx, cy: hy });
    op(ring, s.ring);
    set(handle, { cx: hx, cy: hy });
    set(focusRing, { cx: hx, cy: hy });
    focusRing.style.display = focused && step === 4 ? '' : 'none';
    const k = kFactor(es[CHOSEN], range.e0);
    handle.setAttribute('aria-valuenow', fmt(ys[CHOSEN]));
    handle.setAttribute('aria-valuetext', `y ${say2(ys[CHOSEN])}, miss ${saySigned(es[CHOSEN])}, ${fmt(k)} times its starting size`);

    // Step 2: positive residuals stacked upwards, then negative ones back down.
    let h = 0;
    xs.forEach((_, i) => {
      if (es[i] > 0) {
        set(walkSegs[i], { x1: WALK.up, x2: WALK.up, y1: WALK.base - h * unit, y2: WALK.base - (h + es[i]) * unit });
        walkSegs[i].setAttribute('stroke', INK);
        walkSegs[i].removeAttribute('stroke-dasharray');
        h += es[i];
        set(walkJoints[i], { x1: WALK.up - 5, x2: WALK.up + 5, y1: WALK.base - h * unit, y2: WALK.base - h * unit });
      }
    });
    const top = h;
    set(walkLink, { x1: WALK.up, x2: WALK.down, y1: WALK.base - top * unit, y2: WALK.base - top * unit });
    xs.forEach((_, i) => {
      if (es[i] < 0) {
        set(walkSegs[i], { x1: WALK.down, x2: WALK.down, y1: WALK.base - h * unit, y2: WALK.base - (h + es[i]) * unit });
        walkSegs[i].setAttribute('stroke', MUTED);
        walkSegs[i].setAttribute('stroke-dasharray', '4 3');
        h += es[i];
        set(walkJoints[i], { x1: WALK.down - 5, x2: WALK.down + 5, y1: WALK.base - h * unit, y2: WALK.base - h * unit });
      }
      const on = es[i] !== 0;
      walkSegs[i].style.display = on ? '' : 'none';
      walkJoints[i].style.display = on ? '' : 'none';
    });
    set(walkEnd, { cx: WALK.down, cy: WALK.base - h * unit });
    op(walkG, s.walk);

    // Steps 3–4: bars of |e| turning into bars of e².
    const v = s.barSq;
    const L = lossSummary(es);
    byX.forEach((i, kx) => {
      const val = (1 - v) * Math.abs(es[i]) + v * es[i] * es[i];
      const w = Math.min(13, slot - 5);
      set(bars[kx], { x: SX0 + (kx + 0.5) * slot - w / 2, y: B(val), height: SY0 - B(val) });
      const absLook = v < 0.5;
      bars[kx].setAttribute('fill', absLook ? VIOLET : CYAN);
      bars[kx].setAttribute('fill-opacity', 0.25);
      bars[kx].setAttribute('stroke', absLook ? VIOLET : CYAN);
      bars[kx].setAttribute('stroke-width', 1.4);
      if (absLook && es[i] < 0) bars[kx].setAttribute('stroke-dasharray', '3 2');
      else bars[kx].removeAttribute('stroke-dasharray');
    });
    const ref = (1 - v) * L.mae + v * L.mse;
    set(barRef, { y1: B(ref), y2: B(ref) });
    for (const [a, val] of Object.entries(v < 0.5 ? SAMPLES.abs : SAMPLES.sq)) barRef.setAttribute(a, val);
    if (v >= 0.5) barRef.removeAttribute('stroke-dasharray');
    barRefLabel.textContent = v < 0.5 ? 'MAE' : 'MSE';
    barRefLabel.style.fill = v < 0.5 ? VIOLET : CYAN;
    set(barRefLabel, { y: B(ref) + 4 });
    // Hide a tick label the name would sit on.
    for (const { t, text } of barTicks) text.style.display = Math.abs(B(t) - B(ref)) < 14 ? 'none' : '';
    barCap.textContent = v < 0.5 ? 'each |e|' : 'each e²';
    set(barCap, { y: BAR_TOP - 8 });
    op(barsG, s.bars);

    // Step 5: the chosen miss is k times its start; its square k² times.
    const kk = Math.min(k, KMAX);
    set(kGuide, { x1: K(kk), x2: K(kk), y2: KY(kk * kk) });
    set(kDotAbs, { x: K(kk) - 4.5, y: KY(kk) - 4.5 });
    set(kDotSq, { cx: K(kk), cy: KY(kk * kk) });
    op(kG, s.kchart);

    // Step 6: MAE and RMSE of a constant guess, for every c.
    maeCurve.setAttribute('d', pathFrom(cs.map((t) => [t, maeAt(ys, t)]), CX, CY));
    rmseCurve.setAttribute('d', pathFrom(cs.map((t) => [t, Math.sqrt(mseAt(ys, t))]), CX, CY));
    set(cGuide, { x1: CX(c), x2: CX(c) });
    set(cDotMae, { x: CX(c) - 4.5, y: CY(maeAt(ys, c)) - 4.5 });
    set(cDotRmse, { cx: CX(c), cy: CY(Math.sqrt(mseAt(ys, c))) });
    op(cG, s.cchart);

    // Read-out box.
    const texts = readout(step, { xs, ys, line, start, c });
    texts.forEach((t, kr) => setRow(kr, t, ROW_SAMPLES[step][kr]));
    const nRows = texts.length;
    rows.forEach((r, kr) => {
      const on = kr < nRows;
      r.text.style.display = on ? '' : 'none';
      if (!on) r.sample.style.display = 'none';
    });
    let w = 200;
    try {
      const lens = rows.slice(0, nRows).map((r) => r.text.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...rows.slice(0, nRows).map((r, kr) => r.off + lens[kr])) + 12;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(BOX.x, 598 - w);
    const bh = 16 + BOX.row * nRows;
    set(boxRect, { x: bx, y: BOX.y, width: w, height: bh });
    rows.forEach((r, kr) => {
      const yc = BOX.y + 8 + BOX.row * kr + BOX.row / 2;
      set(r.text, { x: bx + r.off, y: yc + 5 });
      set(r.sample, { x1: bx + 12, x2: bx + 32, y1: yc, y2: yc });
    });

    cOut.textContent = fmt(c);
    cInput.setAttribute('aria-valuetext', `c = ${fmt(c)}`);
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
    const es = residualsOf(xs, ys, line);
    const L = lossSummary(es);
    if (i === 0) {
      return `The ringed point has y ${say2(ys[CHOSEN])}. The line predicts ${say2(predict(line, xs[CHOSEN]))}. ` +
        `Its residual is ${saySigned(es[CHOSEN])}, so the point is ${es[CHOSEN] >= 0 ? 'above' : 'below'} the line.`;
    }
    if (i === 1) {
      return `${L.nAbove} points above the line, with residuals adding to ${saySigned(L.above)}. ` +
        `${L.nBelow} below, adding to ${saySigned(L.below)}. All ${L.n} add to ${say2(L.sum)}.`;
    }
    if (i === 2) return `The sizes of the misses add up to ${say2(L.sumAbs)}. Divided by ${L.n}, the MAE is ${say2(L.mae)}.`;
    if (i === 3) return `The squared misses add up to ${say2(L.sumSq)}. Divided by ${L.n}, the MSE is ${say2(L.mse)}.`;
    if (i === 4) {
      const k = kFactor(es[CHOSEN], range.e0);
      return `The ringed point's miss is ${fmt(k)} times its starting size, and its square ${fmt(k * k)} times. ` +
        `MAE ${say2(L0.mae)} to ${say2(L.mae)}. MSE ${say2(L0.mse)} to ${say2(L.mse)}.`;
    }
    const m = mseAt(ys, c);
    return `Constant guess c ${say2(c)}. MAE ${say2(maeAt(ys, c))}. MSE ${say2(m)}, so RMSE ${say2(Math.sqrt(m))}. ` +
      `MSE is lowest at the mean, ${say2(yMean)}; MAE at the median, ${say2(yMedian)}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction: the chosen point (step 5).
  const clampY = (y) => clamp(y, range.lo, range.hi);
  const setY = (y, { speak = true } = {}) => {
    valueToken++;
    ys = ys.slice();
    ys[CHOSEN] = clampY(y);
    render();
    if (speak) say(spoken(4));
  };
  const animateY = (to, done) => {
    const my = ++valueToken;
    const from = ys[CHOSEN];
    tween(reduced ? 0 : 700, (kt) => {
      if (my !== valueToken) return;
      ys = ys.slice();
      ys[CHOSEN] = from + (to - from) * kt;
      render();
    }, { reduced }).then(() => {
      if (my !== valueToken) return;
      ys = ys.slice();
      ys[CHOSEN] = to;
      render();
      announce(done());
    });
  };

  handle.addEventListener('keydown', (e) => {
    const y = ys[CHOSEN];
    const map = {
      ArrowUp: y + 0.1, ArrowRight: y + 0.1, ArrowDown: y - 0.1, ArrowLeft: y - 0.1,
      PageUp: y + 1, PageDown: y - 1, Home: range.lo, End: range.hi,
    };
    if (!(e.key in map)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    setY(map[e.key]);
  });
  handle.addEventListener('focus', () => {
    let kb = true;
    try { kb = handle.matches(':focus-visible'); } catch { /* keep the ring */ }
    focused = kb;
    render();
  });
  handle.addEventListener('blur', () => { focused = false; render(); });
  draggable(handle, {
    bounds: { x0: X(xs[CHOSEN]), x1: X(xs[CHOSEN]), y0: Y(range.hi), y1: Y(range.lo) },
    onDrag: (_, py) => setY(Y.invert(py)),
  });

  tripleBtn.addEventListener('click', () => animateY(range.yhat + 3 * range.e0, () => `Miss made 3 times as big. ${spoken(4)}`));
  backBtn.addEventListener('click', () => animateY(start[CHOSEN], () => `Point put back. ${spoken(4)}`));

  // --- Interaction: the constant guess (step 6).
  const setC = (v, { fromSlider = false } = {}) => {
    c = clamp(v, C_LO, C_HI);
    if (!fromSlider) cInput.value = String(c);
    render();
  };
  cInput.addEventListener('input', () => {
    setC(Number(cInput.value), { fromSlider: true });
    say(spoken(5));
  });
  let cToken = 0;
  const animateC = (to) => {
    const my = ++cToken;
    const from = c;
    tween(reduced ? 0 : 600, (kt) => {
      if (my !== cToken) return;
      setC(from + (to - from) * kt);
    }, { reduced }).then(() => {
      if (my !== cToken) return;
      setC(to);
      announce(spoken(5));
    });
  };
  meanBtn.addEventListener('click', () => animateC(yMean));
  medianBtn.addEventListener('click', () => animateC(yMedian));

  // Sentences on the page that quote values from the starting data (hidden until filled).
  const filled = pageValues(xs, start);
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) p.hidden = false;
  }

  render();
  // Text widths change once the mono font arrives.
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      const prev = step;
      step = i;
      // The point goes back when you leave step 5, so every other step shows the starting data.
      if (prev === 4 && i !== 4) {
        valueToken++;
        ys = start.slice();
      }
      showGroup(dragGroup, i === 4);
      showGroup(cGroup, i === 5);
      handle.setAttribute('tabindex', i === 4 ? 0 : -1);
      handle.style.display = i === 4 ? '' : 'none';
      if (i !== 4 && doc.activeElement === handle) handle.blur();
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      return animateTo(to, 700);
    },
  };
}
