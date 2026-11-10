// Lesson L4.1, "The artificial neuron": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-neurons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { blobs2, logisticFit } from '../assets/classify.js';
import { xorData, ACT, mlpForward } from '../assets/nn.js';
import { linearScale, clamp, el, axes, classMarker, regionMap, networkDiagram } from '../assets/plot.js';

export const SEED_BLOBS = 143;
export const SEED_XOR = 142;
export const N_XOR = 80;
// Every slider (w₁, w₂, b) runs over this range.
export const RANGE = { min: -4, max: 4, step: 0.05 };
// Where the sliders start, and the point the first two steps follow.
export const START = { w: [1, -0.5], b: 0.5 };
export const HIGHLIGHT = 31;
// Angles tried by the line search: 0 to π in 360 steps.
export const ANGLES = 360;
// The region maps' grid.
export const GRID = [54, 50];

export const blobsData = () => blobs2({ n0: 30, n1: 30, mu0: [-1, -1], mu1: [1.2, 1], sd: 0.8, seed: SEED_BLOBS });
export const xorSet = () => xorData({ n: N_XOR, seed: SEED_XOR });

const BL = blobsData();
const XR = xorSet();

// The activations, in the order the picker shows them (from nn.js; ReLU′(0) = 0 there).
export const ACT_KEYS = ['step', 'sigmoid', 'tanh', 'relu'];
export const ACTS = { step: ACT.step, sigmoid: ACT.sigmoid, tanh: ACT.tanh, relu: ACT.relu };
const ACT_NAME = { step: 'step', sigmoid: 'sigmoid', tanh: 'tanh', relu: 'ReLU' };

// One neuron: z = w₁x₁ + w₂x₂ + b, then a = g(z).
export function neuron(w, b, x, g = 'sigmoid') {
  const z = w[0] * x[0] + w[1] * x[1] + b;
  return { z, a: ACTS[g].f(z) };
}

// A slider moved: replace only that weight (0: w₁, 1: w₂, 2: b), clamped to
// the range, and keep the other two exactly as they were.
export function applySlider({ w, b }, which, value) {
  const v = Math.min(RANGE.max, Math.max(RANGE.min, value));
  if (which === 0) return { w: [v, w[1]], b };
  if (which === 1) return { w: [w[0], v], b };
  return { w: w.slice(), b: v };
}

let FIT = null;
export const fit = () => (FIT ??= logisticFit(BL.X, BL.y));

// Share of points on the right side: predict class 1 when z ≥ 0 (σ(z) ≥ 0.5).
export function accuracyLine(X, y, w, b) {
  let right = 0;
  X.forEach((x, i) => { if ((neuron(w, b, x).z >= 0 ? 1 : 0) === y[i]) right++; });
  return right / X.length;
}

// For one direction (cos θ, sin θ): sort the points by their position along it
// and try every cut between neighbours (and beyond both ends), with class 1 on
// either side. Returns the best count and where it was found.
function sweep(X, y, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const p = X.map((x, i) => ({ v: x[0] * c + x[1] * s, y: y[i] })).sort((a, b) => a.v - b.v);
  const n = p.length;
  const ones = y.reduce((t, v) => t + v, 0);
  let best = { right: -1 };
  let below1 = 0;
  for (let k = 0; k <= n; k++) {
    if (k > 0) below1 += p[k - 1].y;
    // Skip a cut between two equal positions: no line passes between them.
    if (k > 0 && k < n && p[k].v === p[k - 1].v) continue;
    const offset = k === 0 ? p[0].v - 1 : k === n ? p[n - 1].v + 1 : (p[k - 1].v + p[k].v) / 2;
    // side +1: class 1 above the cut; side −1: class 1 below it.
    const up = (k - below1) + (ones - below1);
    for (const [side, right] of [[1, up], [-1, n - up]]) {
      if (right > best.right) best = { right, offset, side };
    }
  }
  return best;
}

// Best accuracy of any line on a grid of ANGLES directions, each with every cut.
export function bestLinear(X, y, angles = ANGLES) {
  let best = { right: -1 };
  for (let k = 0; k < angles; k++) {
    const theta = (Math.PI * k) / angles;
    const r = sweep(X, y, theta);
    if (r.right > best.right) best = { ...r, theta };
  }
  return { acc: best.right / X.length, theta: best.theta, offset: best.offset, side: best.side };
}

// The same search over every distinct way a line can split the points: the
// order along a direction only changes where two points line up, so one
// direction between each pair of neighbouring critical angles covers them all.
export function bestLinearExact(X, y) {
  const crit = [];
  for (let i = 0; i < X.length; i++) {
    for (let j = i + 1; j < X.length; j++) {
      let t = Math.atan2(X[j][1] - X[i][1], X[j][0] - X[i][0]) + Math.PI / 2;
      t = ((t % Math.PI) + Math.PI) % Math.PI;
      crit.push(t);
    }
  }
  crit.sort((a, b) => a - b);
  let best = 0;
  for (let k = 0; k < crit.length; k++) {
    const next = k + 1 < crit.length ? crit[k + 1] : crit[0] + Math.PI;
    best = Math.max(best, sweep(X, y, (crit[k] + next) / 2).right);
  }
  return best / X.length;
}

// Two step neurons, set by hand: h₁ fires when x₁ − x₂ > 1, h₂ when x₂ − x₁ > 1;
// the output fires when either does.
export const HAND = {
  sizes: [2, 2, 1],
  hidden: 'step',
  W: [[[1, -1], [-1, 1]], [[1, 1]]],
  b: [[-1, -1], [-0.5]],
};
export const handPredict = (x) => (mlpForward(HAND, x).out[0] > 0 ? 1 : 0);
export const handAccuracy = () => XR.y.filter((v, i) => v === handPredict(XR.X[i])).length / XR.X.length;

// Points per quadrant: (−,−), (+,−), (−,+), (+,+).
export function xorGroups() {
  const g = [0, 0, 0, 0];
  XR.X.forEach(([a, c]) => { g[(a > 0 ? 1 : 0) + (c > 0 ? 2 : 0)]++; });
  return g;
}

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
// Fixed decimals with a true minus sign, and never "−0.00".
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
// Two significant figures, for values inside the diagram's units.
const sig2 = (v) => Number(v.toPrecision(2));

let BEST = null;
const best = () => (BEST ??= bestLinear(XR.X, XR.y));

export function readout(step, { w = START.w, b = START.b, act = 'sigmoid' } = {}) {
  const pt = BL.X[HIGHLIGHT];
  const wRow = `w₁ = ${num(w[0], 2)}, w₂ = ${num(w[1], 2)}`;
  const flat = w[0] === 0 && w[1] === 0;
  if (step === 0) {
    const { z } = neuron(w, b, pt);
    return [`x₁ = ${num(pt[0], 2)}`, `x₂ = ${num(pt[1], 2)}`, wRow, `b = ${num(b, 2)}`, `z = ${num(z, 2)}`];
  }
  if (step === 1) {
    const { z, a } = neuron(w, b, pt, act);
    return [`g: ${ACT_NAME[act]}`, `z = ${num(z, 2)}`, `a = g(z) = ${num(a, 2)}`, `slope g′(z) = ${num(ACTS[act].df(z), 2)}`];
  }
  if (step === 2) {
    const { a } = neuron(w, b, pt, 'sigmoid');
    return [wRow, `b = ${num(b, 2)}`, `point's p̂ = ${num(a, 2)}`, `accuracy: ${pct(accuracyLine(BL.X, BL.y, w, b))}`];
  }
  if (step === 3) {
    return [wRow, `b = ${num(b, 2)}`, `accuracy: ${pct(accuracyLine(BL.X, BL.y, w, b))}`, flat ? 'no line: z = b' : 'boundary: a line'];
  }
  if (step === 4) {
    return [wRow, `b = ${num(b, 2)}`, `this line: ${pct(accuracyLine(XR.X, XR.y, w, b))}`, `best line: ${pct(best().acc)}`];
  }
  return ['weights set by hand', `two lines: ${pct(handAccuracy())}`, `best one line: ${pct(best().acc)}`];
}

export function pageValues() {
  const f = fit();
  const pt = BL.X[HIGHLIGHT];
  const z = neuron(START.w, START.b, pt).z;
  const v = {
    'n-blobs': String(BL.X.length),
    'n-xor': String(XR.X.length),
    'pt-x1': num(pt[0], 2),
    'pt-x2': num(pt[1], 2),
    'pt-z': num(z, 2),
    'start-w1': num(START.w[0], 2),
    'start-w2': num(START.w[1], 2),
    'start-b': num(START.b, 2),
    'fit-w1': num(f.w[0], 2),
    'fit-w2': num(f.w[1], 2),
    'fit-b': num(f.w0, 2),
    'fit-acc': pct(accuracyLine(BL.X, BL.y, f.w, f.w0)),
    'xor-best': pct(best().acc),
    'xor-angles': String(ANGLES),
    'hand-acc': pct(handAccuracy()),
    'xor-ones': String(XR.y.filter((v) => v === 1).length),
    'xor-wrong': String(Math.round((1 - best().acc) * XR.X.length)),
    'xor-major': pct(Math.max(...[0, 1].map((c) => XR.y.filter((v) => v === c).length)) / XR.y.length),
    'grid': `${GRID[0]} × ${GRID[1]}`,
  };
  for (const g of ACT_KEYS) v[`pt-${g}`] = num(ACTS[g].f(z), 2);
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The plot on the left: blobs on −3..3, XOR on −2..2, the activation curve on z −5..5.
const QX0 = 56;
const QX1 = 326;
const QY0 = 286;
const QY1 = 34;
const BLOB_DOM = [-3, 3];
const XOR_DOM = [-2, 2];
const Z_DOM = [-5, 5];
// The diagram and read-out on the right.
const NET_BOX = { x: 412, y: 66, w: 148, h: 76 };
const RX0 = 404;
const BOX_Y = 172;
const ROW = 20;

export function createStage(svg, { controls, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const f = fit();
  const pt = BL.X[HIGHLIGHT];
  let w = START.w.slice();
  let b = START.b;
  let act = 'sigmoid';
  let step = 0;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A single neuron with two inputs. On the left, two-feature data (cyan circles class 0, violet triangles class 1) or the activation curve; on the right, the neuron diagram and its numbers.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l41-plot' }, defs);
  el('rect', { x: QX0, y: QY1, width: QX1 - QX0, height: QY0 - QY1 }, clip);
  const P = (v) => Number(v.toFixed(2));

  // The exact line w·x + b = 0 across a plot, or null when w = 0.
  const lineEnds = (ww, bb, X, Y) => {
    const [lo, hi] = X.domain;
    if (ww[0] === 0 && ww[1] === 0) return null;
    if (Math.abs(ww[1]) >= Math.abs(ww[0])) {
      const at = (a) => -(bb + ww[0] * a) / ww[1];
      return { x1: P(X(lo)), y1: P(clamp(Y(at(lo)), -2000, 2000)), x2: P(X(hi)), y2: P(clamp(Y(at(hi)), -2000, 2000)) };
    }
    const at = (c) => -(bb + ww[1] * c) / ww[0];
    const [ylo, yhi] = Y.domain;
    return { x1: P(clamp(X(at(ylo)), -2000, 2000)), y1: P(Y(ylo)), x2: P(clamp(X(at(yhi)), -2000, 2000)), y2: P(Y(yhi)) };
  };
  const setLine = (node, ends) => {
    if (!ends) { node.style.display = 'none'; return; }
    node.style.display = '';
    for (const [k, v] of Object.entries(ends)) node.setAttribute(k, v);
  };
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };

  // ---- Blobs view.
  const BX = linearScale(BLOB_DOM, [QX0, QX1]);
  const BY = linearScale(BLOB_DOM, [QY0, QY1]);
  const blobG = el('g', {}, svg);
  const blobRegion = regionMap(blobG, BX, BY, (a, c) => ACTS.sigmoid.f(neuron(w, b, [a, c]).z), { nx: GRID[0], ny: GRID[1], mode: 'prob' });
  blobRegion.g.querySelector('.boundary').style.display = 'none';
  axes(blobG, BX, BY, { xLabel: 'x₁', yLabel: 'x₂' });
  const blobLineG = el('g', { 'clip-path': 'url(#l41-plot)' }, blobG);
  const blobLine = el('line', { stroke: STRONG, 'stroke-width': 2 }, blobLineG);
  const blobPts = el('g', {}, blobG);
  BL.X.forEach((x, i) => classMarker(blobPts, BX(x[0]), BY(x[1]), BL.y[i], { r: 4.5 }));
  const ring = el('circle', { cx: P(BX(pt[0])), cy: P(BY(pt[1])), r: 10, fill: 'none', stroke: STRONG, 'stroke-width': 1.6 }, blobG);

  // ---- XOR view.
  const XX = linearScale(XOR_DOM, [QX0, QX1]);
  const XY = linearScale(XOR_DOM, [QY0, QY1]);
  const xorG = el('g', {}, svg);
  const xorRegionG = el('g', {}, xorG);
  const xorRegion = regionMap(xorRegionG, XX, XY, (a, c) => ACTS.sigmoid.f(neuron(w, b, [a, c]).z), { nx: GRID[0], ny: GRID[1], mode: 'prob' });
  xorRegion.g.querySelector('.boundary').style.display = 'none';
  const handRegionG = el('g', {}, xorG);
  const handRegion = regionMap(handRegionG, XX, XY, (a, c) => handPredict([a, c]), { nx: GRID[0], ny: GRID[1], mode: 'class' });
  handRegion.g.querySelector('.boundary').style.display = 'none';
  axes(xorG, XX, XY, { xLabel: 'x₁', yLabel: 'x₂' });
  const xorLineG = el('g', { 'clip-path': 'url(#l41-plot)' }, xorG);
  const xorLine = el('line', { stroke: STRONG, 'stroke-width': 2 }, xorLineG);
  const handLines = HAND.W[0].map((row, j) => el('line', { ...lineEnds(row, HAND.b[0][j], XX, XY), stroke: STRONG, 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }, xorLineG));
  const handText = [
    el('text', { class: 'stage-text', x: P(XX(-0.6)), y: P(XY(-1.88)), text: 'line 1' }, xorLineG),
    el('text', { class: 'stage-text', x: P(XX(1.12)), y: P(XY(1.82)), text: 'line 2' }, xorLineG),
  ];
  const xorPts = el('g', {}, xorG);
  XR.X.forEach((x, i) => classMarker(xorPts, XX(x[0]), XY(x[1]), XR.y[i], { r: 4.5 }));

  // ---- Activation view.
  const actG = el('g', {}, svg);
  const actAxes = el('g', {}, actG);
  const actPlot = el('g', { 'clip-path': 'url(#l41-plot)' }, actG);
  const actCurve = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, actPlot);
  const guideV = el('line', { stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, actPlot);
  const guideH = el('line', { stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, actPlot);
  const actDot = el('circle', { r: 5.5, fill: STRONG }, actPlot);
  const actNote = el('text', { class: 'stage-text', x: QX1 - 4, y: QY1 + 14, 'text-anchor': 'end' }, actG);

  // ---- Key above the plot.
  const keyG = el('g', {}, svg);
  classMarker(keyG, QX0 + 6, 16, 0, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 16, y: 20, text: 'class 0' }, keyG);
  classMarker(keyG, QX0 + 120, 16, 1, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 130, y: 20, text: 'class 1' }, keyG);

  // ---- Diagrams on the right.
  const net1G = el('g', {}, svg);
  const net1 = networkDiagram(net1G, [2, 1], { box: NET_BOX, r: 16, labels: ['inputs', 'neuron'] });
  const midY = NET_BOX.y + NET_BOX.h / 2;
  const midX = NET_BOX.x + NET_BOX.w / 2;
  el('text', { class: 'stage-text', x: midX, y: NET_BOX.y + 12, 'text-anchor': 'middle', text: 'w₁' }, net1G);
  el('text', { class: 'stage-text', x: midX, y: NET_BOX.y + NET_BOX.h - 2, 'text-anchor': 'middle', text: 'w₂' }, net1G);
  el('text', { class: 'stage-text', x: NET_BOX.x + NET_BOX.w, y: midY + 36, 'text-anchor': 'middle', text: '+ b' }, net1G);
  el('text', { class: 'stage-text', x: NET_BOX.x - 22, y: NET_BOX.y + NET_BOX.h / 4 + 5, 'text-anchor': 'end', text: 'x₁' }, net1G);
  el('text', { class: 'stage-text', x: NET_BOX.x - 22, y: NET_BOX.y + (3 * NET_BOX.h) / 4 + 5, 'text-anchor': 'end', text: 'x₂' }, net1G);
  const outName = el('text', { class: 'stage-text', x: NET_BOX.x + NET_BOX.w + 22, y: midY + 5 }, net1G);
  const net2G = el('g', {}, svg);
  const net2 = networkDiagram(net2G, [2, 2, 1], { box: NET_BOX, r: 14, labels: ['inputs', 'hidden', 'output'] });
  net2.update({ weights: HAND.W });

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, hint, value) => {
    const label = make('label');
    const input = make('input', { type: 'range', min: RANGE.min, max: RANGE.max, step: RANGE.step, value });
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    label.append(name, make('span', { className: 'vh', textContent: hint }), ' ', input, ' ', out);
    return { label, input, out };
  };
  const sliders = make('div');
  const w1S = slider('w₁', ', the weight on x₁', w[0]);
  const w2S = slider('w₂', ', the weight on x₂', w[1]);
  const bS = slider('b', ', the bias', b);
  sliders.append(w1S.label, w2S.label, bS.label);
  const actGroup = make('div');
  actGroup.setAttribute('role', 'group');
  actGroup.setAttribute('aria-label', 'Activation');
  const actBtns = ACT_KEYS.map((k) => {
    const btn = make('button', { type: 'button', textContent: ACT_NAME[k] });
    btn.addEventListener('click', () => { act = k; render(); say(spoken(1)); });
    actGroup.append(btn);
    return [k, btn];
  });
  const fitGroup = make('div');
  const fitBtn = make('button', { type: 'button', textContent: 'Fit' });
  fitGroup.append(fitBtn);
  controls.append(sliders, actGroup, fitGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  function drawAct() {
    actAxes.replaceChildren();
    const ZX = linearScale(Z_DOM, [QX0, QX1]);
    const AY = linearScale(act === 'relu' ? [-1, 5] : [-1.25, 1.25], [QY0, QY1]);
    axes(actAxes, ZX, AY, { xLabel: 'z' });
    // Inside the plot's empty top-left corner, clear of the tick labels on a phone.
    el('text', { class: 'stage-text', x: QX0 + 8, y: QY1 + 16, text: `a = ${ACT_NAME[act]}(z)` }, actAxes);
    el('line', { x1: QX0, x2: QX1, y1: P(AY(0)), y2: P(AY(0)), stroke: LINE, 'stroke-dasharray': '2 3' }, actAxes);
    const g = ACTS[act].f;
    let d = '';
    if (act === 'step') {
      d = `M${P(ZX(-5))},${P(AY(0))}L${P(ZX(0))},${P(AY(0))}M${P(ZX(0))},${P(AY(1))}L${P(ZX(5))},${P(AY(1))}`;
    } else {
      for (let k = 0; k <= 200; k++) {
        const z = -5 + k / 20;
        d += `${k ? 'L' : 'M'}${P(ZX(z))},${P(AY(g(z)))}`;
      }
    }
    actCurve.setAttribute('d', d);
    const { z, a } = neuron(w, b, pt, act);
    const inside = z >= Z_DOM[0] && z <= Z_DOM[1];
    for (const node of [guideV, guideH, actDot]) show(node, inside);
    actNote.textContent = inside ? '' : 'z is off the plot';
    if (inside) {
      guideV.setAttribute('x1', P(ZX(z)));
      guideV.setAttribute('x2', P(ZX(z)));
      guideV.setAttribute('y1', P(AY(0)));
      guideV.setAttribute('y2', P(AY(a)));
      guideH.setAttribute('x1', QX0);
      guideH.setAttribute('x2', P(ZX(z)));
      guideH.setAttribute('y1', P(AY(a)));
      guideH.setAttribute('y2', P(AY(a)));
      actDot.setAttribute('cx', P(ZX(z)));
      actDot.setAttribute('cy', P(AY(a)));
    }
  }

  function render() {
    const blobs = step === 0 || step === 2 || step === 3;
    const xor = step === 4 || step === 5;
    show(blobG, blobs);
    show(xorG, xor);
    show(actG, step === 1);
    show(keyG, step !== 1);
    show(blobRegion.g, step === 3);
    show(blobLineG, step === 2 || step === 3);
    show(ring, step <= 2);
    show(xorRegionG, step === 4);
    show(handRegionG, step === 5);
    show(xorLine, step === 4);
    handLines.forEach((n) => show(n, step === 5));
    handText.forEach((n) => show(n, step === 5));
    show(net1G, step <= 4);
    show(net2G, step === 5);

    if (step === 3) blobRegion.update((a, c) => ACTS.sigmoid.f(neuron(w, b, [a, c]).z));
    if (step === 4) xorRegion.update((a, c) => ACTS.sigmoid.f(neuron(w, b, [a, c]).z));
    setLine(blobLine, lineEnds(w, b, BX, BY));
    setLine(xorLine, step === 4 ? lineEnds(w, b, XX, XY) : null);
    if (step === 1) drawAct();

    // Diagram: link width and dash from the weights; values in the units for steps 1 to 3.
    const { z, a } = neuron(w, b, pt, step === 1 ? act : 'sigmoid');
    const out = step === 0 ? z : a;
    net1.update({ weights: [[w]], values: step <= 2 ? [[sig2(pt[0]), sig2(pt[1])], [sig2(out)]] : [[], []] });
    outName.textContent = step === 0 ? 'z' : step === 1 ? 'a' : step === 2 ? 'p̂' : '';

    const texts = readout(step, { w, b, act });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let wd = 190;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) wd = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0, 598 - wd);
    boxRect.setAttribute('x', P(bx));
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', P(wd));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', P(bx + 12));
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    w1S.out.textContent = num(w[0], 2);
    w2S.out.textContent = num(w[1], 2);
    bS.out.textContent = num(b, 2);
    w1S.input.setAttribute('aria-valuetext', `w₁ = ${num(w[0], 2)}`);
    w2S.input.setAttribute('aria-valuetext', `w₂ = ${num(w[1], 2)}`);
    bS.input.setAttribute('aria-valuetext', `b = ${num(b, 2)}`);
    for (const [k, btn] of actBtns) btn.setAttribute('aria-pressed', String(k === act));
  }

  const spoken = (i) => {
    const { z, a } = neuron(w, b, pt, i === 1 ? act : 'sigmoid');
    if (i === 0) return `The ringed point has x₁ = ${num(pt[0], 2)} and x₂ = ${num(pt[1], 2)}. With w₁ = ${num(w[0], 2)}, w₂ = ${num(w[1], 2)} and b = ${num(b, 2)}, its weighted sum z is ${num(z, 2)}.`;
    if (i === 1) return `Activation ${ACT_NAME[act]}: z = ${num(z, 2)} gives a = ${num(a, 2)}.`;
    if (i === 2) return `With a sigmoid the neuron's output is the probability of class 1: ${num(a, 2)} for the ringed point. Accuracy at p̂ = 0.5: ${pct(accuracyLine(BL.X, BL.y, w, b))}.`;
    if (i === 3) return `The shading shows σ(z) over the plane; the boundary z = 0 is a straight line. Accuracy ${pct(accuracyLine(BL.X, BL.y, w, b))}.`;
    if (i === 4) return `XOR data, ${XR.X.length} points in four groups. This line scores ${pct(accuracyLine(XR.X, XR.y, w, b))}; the best any straight line can do is ${pct(best().acc)}.`;
    return `Two step neurons with hand-set weights draw two parallel lines; the output neuron fires when either fires. Accuracy ${pct(handAccuracy())}, against ${pct(best().acc)} for the best single line.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // Only the weight whose slider moved is read back, so exact fitted values in
  // the other two survive (the inputs snap them to the 0.05 grid).
  const onSlide = (which) => () => {
    ({ w, b } = applySlider({ w, b }, which, Number([w1S, w2S, bS][which].input.value)));
    render();
    say(spoken(step));
  };
  [w1S, w2S, bS].forEach((s, k) => s.input.addEventListener('input', onSlide(k)));

  fitBtn.addEventListener('click', () => {
    w = f.w.slice();
    b = f.w0;
    w1S.input.value = String(w[0]);
    w2S.input.value = String(w[1]);
    bS.input.value = String(b);
    render();
    announce(`Fitted: w₁ = ${num(w[0], 2)}, w₂ = ${num(w[1], 2)}, b = ${num(b, 2)}, the logistic regression answer. Accuracy ${pct(accuracyLine(BL.X, BL.y, w, b))}.`);
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i) {
      step = i;
      showGroup(sliders, i <= 4);
      showGroup(actGroup, i === 1);
      showGroup(fitGroup, i === 2);
      render();
      announce(spoken(i));
      return Promise.resolve();
    },
  };
}
