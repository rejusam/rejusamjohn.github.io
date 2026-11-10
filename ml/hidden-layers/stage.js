// Lesson L4.2, "Hidden layers: bending the boundary": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-hidden-layers.test.mjs);
// the DOM is only touched inside createStage. Every number on the stage is read
// from the network being trained at that moment; the numbers written in the
// page's text come from the same seeded runs (storyValues, checked by the tests).

import { logisticFit, sigmoid } from '../assets/classify.js';
import { circlesData, mlpInit, mlpForward, trainer, accuracyNet, bceLogit } from '../assets/nn.js';
import { linearScale, clamp, el, axes, classMarker, regionMap, regionCells, networkDiagram, trianglePoints } from '../assets/plot.js';
import { createTrainLoop, trainButtons } from '../assets/train-loop.js';

export const SEED_TRAIN = 151;
export const SEED_TEST = 152;
export const SEED_INIT = 155;
export const SEED_SHUFFLE = 154;
export const N_TRAIN = 200;
export const N_TEST = 400;
export const ETA = 0.03;
export const BATCH = 20;
export const EPOCHS = 150;
// One epoch is one pass through the training points in batches of 20.
export const ITERS_PER_EPOCH = Math.ceil(N_TRAIN / BATCH);
export const MAX_ITER = EPOCHS * ITERS_PER_EPOCH;
export const H_CHOICES = [1, 2, 3, 4, 8];
// Steps per frame at most: slow at first so the early bending can be watched.
export const PACE = 1;
export const paceAt = (iter) => (iter < 200 ? PACE : iter < 400 ? 2 : 6);
// The plot's range in both directions, the zoomed-out range, and the far ring.
export const DOM = [-3.2, 3.2];
export const FAR_DOM = [-8, 8];
export const FAR_R = 7;
export const GRID = [54, 50];
// The loss curve's vertical range (every run starts below it).
export const LOSS_MAX = 0.8;

export const trainSet = () => circlesData({ n: N_TRAIN, seed: SEED_TRAIN });
export const testSet = () => circlesData({ n: N_TEST, seed: SEED_TEST });

const TR = trainSet();
const TE = testSet();

// ---- one neuron (step 1): logistic regression, as in the previous lesson

let FIT = null;
export const fitOne = () => (FIT ??= logisticFit(TR.X, TR.y));
export function oneAccuracy(X, y) {
  const f = fitOne();
  let right = 0;
  X.forEach((x, i) => { if ((f.w0 + f.w[0] * x[0] + f.w[1] * x[1] >= 0 ? 1 : 0) === y[i]) right++; });
  return right / X.length;
}

// The best accuracy of any straight line: for each of 360 directions, sort the
// points along it and try every cut, with class 1 on either side.
export function bestLine(X, y, angles = 360) {
  let best = 0;
  const ones = y.reduce((t, v) => t + v, 0);
  for (let k = 0; k < angles; k++) {
    const c = Math.cos((Math.PI * k) / angles);
    const s = Math.sin((Math.PI * k) / angles);
    const p = X.map((x, i) => ({ v: x[0] * c + x[1] * s, y: y[i] })).sort((a, b) => a.v - b.v);
    let below1 = 0;
    for (let m = 0; m <= p.length; m++) {
      if (m > 0) below1 += p[m - 1].y;
      if (m > 0 && m < p.length && p[m].v === p[m - 1].v) continue;
      const up = (m - below1) + (ones - below1);
      best = Math.max(best, up, p.length - up);
    }
  }
  return best / X.length;
}

// ---- the [2, H, 1] network and its seeded run

export function makeRun(H) {
  const net = mlpInit([2, H, 1], { hidden: 'tanh', init: 'xavier', seed: SEED_INIT });
  return trainer({ net, X: TR.X, y: TR.y, batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
}

export const finished = (t) => t.state.iter >= MAX_ITER || t.state.diverged;

// k more optimiser steps, never past 150 epochs.
export function advance(t, k) {
  const left = MAX_ITER - t.state.iter;
  if (left > 0) t.step(Math.min(k, left));
  return t.state;
}

// One animation frame: the loop offers n steps; the page takes at most paceAt.
export const frameStep = (t, n) => advance(t, Math.min(n, paceAt(t.state.iter)));

const RUNS = new Map();
export function trainedNet(H) {
  if (!RUNS.has(H)) {
    const t = makeRun(H);
    advance(t, MAX_ITER);
    RUNS.set(H, t);
  }
  return RUNS.get(H).state.net;
}

export const rowOf = (t) => ({ train: accuracyNet(t.state.net, TR.X, TR.y), test: accuracyNet(t.state.net, TE.X, TE.y) });
export const resultsRow = (H) => (trainedNet(H), rowOf(RUNS.get(H)));

// Mean binary cross-entropy (log-loss) over the training points.
export function trainLoss(net) {
  let s = 0;
  for (let i = 0; i < TR.X.length; i++) s += bceLogit(TR.y[i], mlpForward(net, TR.X[i]).out[0]);
  return s / TR.X.length;
}

// p̂ for a [2, H, 1] tanh network without building arrays (for the shading).
export function fastProb(net) {
  const W1 = net.W[0];
  const b1 = net.b[0];
  const w2 = net.W[1][0];
  const b2 = net.b[1][0];
  const H = W1.length;
  return (x, y) => {
    let z = b2;
    for (let j = 0; j < H; j++) z += w2[j] * Math.tanh(W1[j][0] * x + W1[j][1] * y + b1[j]);
    return sigmoid(z);
  };
}

// Each point at its hidden activations (a₁, …, a_H).
export const hiddenPoints = (net, X) => X.map((x) => mlpForward(net, x).as[1]);

// The output neuron alone, reading the hidden activations: σ(w⁽²⁾·a + b⁽²⁾) > 0.5.
export function outputAccuracyHidden(net, X, y) {
  const w = net.W[1][0];
  const b = net.b[1][0];
  let right = 0;
  hiddenPoints(net, X).forEach((a, i) => {
    const z = a.reduce((s, v, j) => s + w[j] * v, b);
    if ((sigmoid(z) > 0.5 ? 1 : 0) === y[i]) right++;
  });
  return right / X.length;
}

// p̂ at K evenly spaced points on the circle of radius r.
export function farRing(net, r, K) {
  const p = fastProb(net);
  return Array.from({ length: K }, (_, k) => p(r * Math.cos((2 * Math.PI * k) / K), r * Math.sin((2 * Math.PI * k) / K)));
}

// Along each of K directions from the centre, the first distance where p̂
// reaches 0.5 (scan in steps of 0.01, then bisect).
export function boundaryRadii(net, K, rMax = DOM[1]) {
  const p = fastProb(net);
  const radii = [];
  for (let k = 0; k < K; k++) {
    const c = Math.cos((2 * Math.PI * k) / K);
    const s = Math.sin((2 * Math.PI * k) / K);
    const f = (r) => p(r * c, r * s) - 0.5;
    let lo = 0;
    let hi = NaN;
    for (let r = 0.01; r <= rMax; r += 0.01) {
      if (f(r) >= 0) { hi = r; break; }
      lo = r;
    }
    if (Number.isNaN(hi)) { radii.push(NaN); continue; }
    for (let it = 0; it < 60; it++) {
      const m = (lo + hi) / 2;
      if (f(m) >= 0) hi = m; else lo = m;
    }
    radii.push((lo + hi) / 2);
  }
  return { radii, min: Math.min(...radii), max: Math.max(...radii) };
}

// Marching squares on a grid of values at cell centres (row j = 0 first).
// Returns segments [[gx, gy], [gx, gy]] in grid units, where cell (i, j) spans
// [i, i + 1] × [j, j + 1], so its centre is at (i + 0.5, j + 0.5).
export function contour(vals, nx, ny, level = 0.5) {
  const v = (i, j) => vals[j * nx + i];
  const segs = [];
  const cross = (i1, j1, i2, j2) => {
    const a = v(i1, j1);
    const b = v(i2, j2);
    const t = (level - a) / (b - a);
    return [i1 + 0.5 + t * (i2 - i1), j1 + 0.5 + t * (j2 - j1)];
  };
  for (let j = 0; j + 1 < ny; j++) {
    for (let i = 0; i + 1 < nx; i++) {
      const a = v(i, j) >= level;
      const b = v(i + 1, j) >= level;
      const c = v(i + 1, j + 1) >= level;
      const d = v(i, j + 1) >= level;
      if (a === b && b === c && c === d) continue;
      const top = a !== b ? cross(i, j, i + 1, j) : null;
      const right = b !== c ? cross(i + 1, j, i + 1, j + 1) : null;
      const bottom = d !== c ? cross(i, j + 1, i + 1, j + 1) : null;
      const left = a !== d ? cross(i, j, i, j + 1) : null;
      if (top && right && bottom && left) {
        const centre = (v(i, j) + v(i + 1, j) + v(i + 1, j + 1) + v(i, j + 1)) / 4 >= level;
        if (centre === a) segs.push([top, right], [bottom, left]);
        else segs.push([left, top], [right, bottom]);
      } else {
        const pts = [top, right, bottom, left].filter(Boolean);
        segs.push([pts[0], pts[1]]);
      }
    }
  }
  return segs;
}

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
// Fixed decimals with a true minus sign, and never "−0.00".
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}

// The results table: H, then train and test accuracy once that run has done 150 epochs.
export const tableRows = (done) => H_CHOICES.map((H) => {
  const r = done.get(H);
  return [String(H), r ? pct(r.train) : '–', r ? pct(r.test) : '–'];
});

// The run's state in words: "training…" only while the loop runs; "press Train"
// before the first step; "not finished: press Train" when paused part-way.
export function trainStatus({ running = false, iter = 0, done = false }) {
  if (done) return 'finished';
  if (running) return 'training…';
  return iter === 0 ? 'press Train' : 'not finished: press Train';
}

// What the read-out needs from a run, all from its current weights.
export function info(t, H, running = false) {
  const net = t.state.net;
  return {
    H,
    running,
    iter: t.state.iter,
    epoch: t.state.epoch,
    loss: trainLoss(net),
    train: accuracyNet(net, TR.X, TR.y),
    test: accuracyNet(net, TE.X, TE.y),
    hidden: H === 2 ? outputAccuracyHidden(net, TR.X, TR.y) : null,
    far: Math.min(...farRing(net, FAR_R, 360)),
    done: finished(t),
  };
}

export function readout(step, s) {
  if (step === 0) {
    return ['one neuron', `train: ${pct(oneAccuracy(TR.X, TR.y))}`, `test: ${pct(oneAccuracy(TE.X, TE.y))}`];
  }
  const head = `H = ${s.H} · epoch ${s.epoch}`;
  const status = trainStatus(s);
  if (step === 4) {
    return [head, `loss: ${num(s.loss, 3)}`, 'train accuracy:', `line on (a₁,a₂): ${pct(s.hidden ?? 0)}`, `network: ${pct(s.train)}`, status];
  }
  if (step === 5) {
    const far = `min p̂, r = ${FAR_R}: ${num(s.far, 2)}`;
    if (!s.done) return [head, `test: ${pct(s.test)}`, far, status];
    return [head, `train: ${pct(s.train)}`, `test: ${pct(s.test)}`, far, status];
  }
  return [head, `loss: ${num(s.loss, 3)}`, `train: ${pct(s.train)}`, `test: ${pct(s.test)}`, status];
}

// Run t to the end. Returns the epoch during which test accuracy first reaches
// 95% (checked after every step): iteration k belongs to epoch ⌈k / 10⌉.
export function first95Epoch(t) {
  let firstIter = null;
  while (!finished(t)) {
    advance(t, 1);
    if (firstIter === null && accuracyNet(t.state.net, TE.X, TE.y) >= 0.95) firstIter = t.state.iter;
  }
  return firstIter === null ? null : Math.ceil(firstIter / ITERS_PER_EPOCH);
}

// Filled into the page at run time. The one-neuron fit's two values wait until
// step 1 is first shown (fitValues), so a deep link to a later step skips the fit.
export const fitValues = () => ({
  'one-train': pct(oneAccuracy(TR.X, TR.y)),
  'one-test': pct(oneAccuracy(TE.X, TE.y)),
});
export const pageValues = () => ({ ...baseValues(), ...fitValues() });
export function baseValues() {
  return {
    'n-train': String(TR.X.length),
    'n-test': String(TE.X.length),
    'epochs': String(EPOCHS),
    'batch': String(BATCH),
    'iters': String(ITERS_PER_EPOCH),
    'eta': String(ETA),
    'far-r': String(FAR_R),
    'grid': `${GRID[0]} × ${GRID[1]}`,
    'max-r': num(maxRadius(), 1),
  };
}

// The furthest any training or test point lies from the centre.
export function maxRadius() {
  let m = 0;
  for (const x of [...TR.X, ...TE.X]) m = Math.max(m, Math.hypot(x[0], x[1]));
  return m;
}

// Distance of each hidden unit's line w_j·x + b_j = 0 from the centre.
export const lineDistances = (net) => net.W[0].map((w, j) => Math.abs(net.b[0][j]) / Math.hypot(w[0], w[1]));

// From the seeded training runs. Written in the page and checked by the tests;
// the page does not train on load, so these are not recomputed in the browser.
export function storyValues() {
  const v = {};
  for (const H of H_CHOICES) {
    const r = resultsRow(H);
    v[`h${H}-train`] = pct(r.train);
    v[`h${H}-test`] = pct(r.test);
  }
  const t = makeRun(4);
  v['h4-first95'] = String(first95Epoch(t));
  v['best-line'] = pct(bestLine(TR.X, TR.y));
  v['h4-loss0'] = num(t.state.epochLoss[0], 2);
  v['h4-loss'] = num(t.state.epochLoss[EPOCHS], 3);
  const n2 = trainedNet(2);
  v['h2-hidden'] = pct(outputAccuracyHidden(n2, TR.X, TR.y));
  const n4 = trainedNet(4);
  const d = lineDistances(n4);
  v['line-min'] = num(Math.min(...d), 1);
  v['line-max'] = num(Math.max(...d), 1);
  v['far-min'] = num(Math.min(...farRing(n4, FAR_R, 360)), 2);
  const r = boundaryRadii(n4, 360);
  v['r-min'] = num(r.min, 1);
  v['r-max'] = num(r.max, 1);
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The square plot on the left, the loss curve (or diagram, or table) top right,
// the read-out under it.
const QX0 = 60;
const QX1 = 312;
const QY0 = 286;
const QY1 = 34;
const LX0 = 420;
const LX1 = 590;
const LY0 = 124;
const LY1 = 40;
const NET_BOX = { x: 424, y: 50, w: 150, h: 92 };
const RX0 = 388;
const BOX_Y = 172;
const ROW = 20;
const TAB_Y = 46;
const TAB_ROW = 21;
const TAB_X = [416, 504, 588];
const HID = [-1.15, 1.15];

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  // Step 1's one-neuron fit is made when that step is first shown (about 1 ms).
  let fitShown = false;
  // Nothing is drawn until the engine shows a step, so a deep link draws only that step.
  let shown = false;
  let H = 4;
  const runs = new Map();
  const run = (h) => {
    if (!runs.has(h)) runs.set(h, makeRun(h));
    return runs.get(h);
  };
  const curH = () => (step === 3 ? H : step === 4 ? 2 : 4);
  const cur = () => run(curH());

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Two rings of points, cyan circles class 0 inside and violet triangles class 1 outside, shaded by the network\'s p̂ with its boundary drawn; on the right the training loss by epoch and the network\'s numbers.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l42-plot' }, defs);
  el('rect', { x: QX0, y: QY1, width: QX1 - QX0, height: QY0 - QY1 }, clip);
  const P = (v) => Number(v.toFixed(2));
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };

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

  // A region map whose staircase is hidden, with a smooth p̂ = 0.5 contour on top.
  const smoothMap = (parent, X, Y, prob) => {
    const cells = regionCells(GRID[0], GRID[1], X, Y);
    const vals = new Array(cells.length);
    let k = 0;
    const record = (fn) => (x, y) => { const p = fn(x, y); vals[k++] = p; return p; };
    const map = regionMap(parent, X, Y, record(prob), { nx: GRID[0], ny: GRID[1], mode: 'prob' });
    map.g.querySelector('.boundary').style.display = 'none';
    const lineG = el('g', { 'clip-path': 'url(#l42-plot)' }, parent);
    const path = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 2, 'stroke-linejoin': 'round' }, lineG);
    const { w, h, px: left, py: top } = cells[0];
    const draw = () => {
      path.setAttribute('d', contour(vals, GRID[0], GRID[1], 0.5)
        .map(([[a, b], [c, d]]) => `M${P(left + a * w)},${P(top + b * h)}L${P(left + c * w)},${P(top + d * h)}`).join(''));
    };
    draw();
    return {
      update(fn) { k = 0; map.update(record(fn)); draw(); },
      g: map.g,
      lineG,
    };
  };

  const plotPoints = (parent, X, Y, pts, ys, r) => pts.map((x, i) => classMarker(parent, P(X(x[0])), P(Y(x[1])), ys[i], { r }));
  const movePoint = (node, cls, cx, cy, r) => {
    if (cls === 1) {
      node.setAttribute('d', trianglePoints(cx, cy, r).map(([x, y], k) => `${k ? 'L' : 'M'}${P(x)},${P(y)}`).join('') + 'Z');
    } else {
      node.setAttribute('cx', P(cx));
      node.setAttribute('cy', P(cy));
    }
  };

  // ---- Input view (steps 1 to 4).
  const IX = linearScale(DOM, [QX0, QX1]);
  const IY = linearScale(DOM, [QY0, QY1]);
  const inG = el('g', {}, svg);
  // The four shaded maps (about 6 ms of drawing each) are built when their step
  // is first shown, each in a group made now so the layering stays the same.
  const oneMapG = el('g', {}, inG);
  const netMapG = el('g', {}, inG);
  let netMap = null;
  axes(inG, IX, IY, { xLabel: 'x₁', yLabel: 'x₂' });
  const unitG = el('g', { 'clip-path': 'url(#l42-plot)' }, inG);
  const unitLines = Array.from({ length: 8 }, () => el('line', { stroke: MUTED, 'stroke-width': 1.3, 'stroke-dasharray': '6 4' }, unitG));
  const inPts = el('g', {}, inG);
  plotPoints(inPts, IX, IY, TR.X, TR.y, 3.4);

  // ---- Hidden view (step 5): each point at (a₁, a₂).
  const HX = linearScale(HID, [QX0, QX1]);
  const HY = linearScale(HID, [QY0, QY1]);
  const hidG = el('g', {}, svg);
  const hidMapG = el('g', {}, hidG);
  let hidRegion = null;
  axes(hidG, HX, HY, { xLabel: 'a₁, hidden unit 1', yLabel: 'a₂' });
  const hidLineG = el('g', { 'clip-path': 'url(#l42-plot)' }, hidG);
  const hidLine = el('line', { stroke: STRONG, 'stroke-width': 2 }, hidLineG);
  const hidPtsG = el('g', {}, hidG);
  const hidPts = plotPoints(hidPtsG, HX, HY, TR.X.map(() => [0, 0]), TR.y, 3.4);

  // ---- Zoomed-out view (step 6).
  const FX = linearScale(FAR_DOM, [QX0, QX1]);
  const FY = linearScale(FAR_DOM, [QY0, QY1]);
  const farG = el('g', {}, svg);
  const farMapG = el('g', {}, farG);
  let farMap = null;
  axes(farG, FX, FY, { xLabel: 'x₁', yLabel: 'x₂' });
  el('rect', { x: P(FX(DOM[0])), y: P(FY(DOM[1])), width: P(FX(DOM[1]) - FX(DOM[0])), height: P(FY(DOM[0]) - FY(DOM[1])), fill: 'none', stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }, farG);
  el('circle', { cx: P(FX(0)), cy: P(FY(0)), r: P(FX(FAR_R) - FX(0)), fill: 'none', stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '2 4' }, farG);
  el('text', { class: 'stage-text', x: P(FX(DOM[0]) + 4), y: P(FY(DOM[1]) - 6), text: 'steps 1–5' }, farG);
  el('text', { class: 'stage-text', x: P(FX(FAR_R * Math.SQRT1_2) + 4), y: P(FY(FAR_R * Math.SQRT1_2) - 4), text: `r = ${FAR_R}` }, farG);
  const farPts = el('g', {}, farG);
  TR.X.forEach((x, i) => classMarker(farPts, P(FX(x[0])), P(FY(x[1])), TR.y[i], { r: 1.8 }));

  // ---- Key above the plot.
  const keyG = el('g', {}, svg);
  classMarker(keyG, QX0 + 6, 16, 0, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 16, y: 20, text: 'class 0' }, keyG);
  classMarker(keyG, QX0 + 120, 16, 1, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 130, y: 20, text: 'class 1' }, keyG);

  // ---- Loss curve, top right.
  const LXs = linearScale([0, EPOCHS], [LX0, LX1]);
  const LYs = linearScale([0, LOSS_MAX], [LY0, LY1]);
  const lossG = el('g', {}, svg);
  axes(lossG, LXs, LYs, { ticks: 3 });
  el('text', { class: 'stage-text', x: LX0 + 6, y: LY1 - 8, text: 'training loss' }, lossG);
  el('text', { class: 'stage-text', x: LX1, y: LY0 + 36, 'text-anchor': 'end', text: 'epoch' }, lossG);
  const lossPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, lossG);

  // ---- Diagram (step 3), top right.
  const netG = el('g', {}, svg);
  const diagram = networkDiagram(netG, [2, 4, 1], { box: NET_BOX, r: 9, labels: ['inputs', 'hidden', 'output'] });

  // ---- Results table (step 4), top right.
  const tabG = el('g', {}, svg);
  ['H', 'train', 'test'].forEach((t, c) => el('text', { class: 'stage-text', x: TAB_X[c], y: TAB_Y, 'text-anchor': c ? 'end' : 'start', fill: MUTED, text: t }, tabG));
  el('line', { x1: TAB_X[0], x2: TAB_X[2], y1: TAB_Y + 6, y2: TAB_Y + 6, stroke: LINE }, tabG);
  const tabCells = H_CHOICES.map((_, r) => [0, 1, 2].map((c) => el('text', { class: 'value', x: TAB_X[c] + (c ? 0 : 14), y: TAB_Y + TAB_ROW * (r + 1) + 4, 'text-anchor': c ? 'end' : 'start' }, tabG)));
  const tabMark = el('text', { class: 'value', x: TAB_X[0], y: 0, text: '▸' }, tabG);

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4, 5].map(() => el('text', { class: 'value' }, box));

  // ---- Controls: the hidden-units slider (step 4) and Train / Pause / Reset.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const hGroup = make('div');
  const hLabel = make('label');
  const hInput = make('input', { type: 'range', min: 0, max: H_CHOICES.length - 1, step: 1, value: H_CHOICES.indexOf(H) });
  const hOut = make('output');
  hOut.setAttribute('aria-hidden', 'true');
  hLabel.append('hidden units', make('span', { className: 'vh', textContent: ', H' }), ' ', hInput, ' ', hOut);
  hGroup.append(hLabel);
  const trainGroup = make('div');
  controls.append(hGroup, trainGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  let narrow = false;
  const measure = () => {
    const wd = svg.clientWidth || (svg.getBoundingClientRect ? svg.getBoundingClientRect().width : 600);
    narrow = wd > 0 && wd < 500;
  };
  let frames = 0;

  function drawMaps(t) {
    const net = t.state.net;
    if (step >= 1 && step <= 3) {
      if (netMap) netMap.update(fastProb(net));
      else netMap = smoothMap(netMapG, IX, IY, fastProb(net));
    }
    if (step === 5) {
      if (farMap) farMap.update(fastProb(net));
      else farMap = smoothMap(farMapG, FX, FY, fastProb(net));
    }
    if (step === 4) {
      const w = net.W[1][0];
      const b = net.b[1][0];
      const prob = (a1, a2) => sigmoid(b + w[0] * a1 + w[1] * a2);
      if (hidRegion) hidRegion.update(prob);
      else {
        hidRegion = regionMap(hidMapG, HX, HY, prob, { nx: GRID[0], ny: GRID[1], mode: 'prob' });
        hidRegion.g.querySelector('.boundary').style.display = 'none';
      }
      setLine(hidLine, lineEnds(w, b, HX, HY));
    }
  }

  function render(full = true) {
    if (!shown) return;
    if (step === 0 && !fitShown) {
      fitShown = true;
      const f = fitOne();
      smoothMap(oneMapG, IX, IY, (a, c) => sigmoid(f.w0 + f.w[0] * a + f.w[1] * c));
      fill(fitValues());
    }
    const t = step === 0 ? null : cur();
    show(inG, step <= 3);
    show(hidG, step === 4);
    show(farG, step === 5);
    show(oneMapG, step === 0);
    show(netMapG, step >= 1 && step <= 3);
    show(lossG, step === 1 || step === 4 || step === 5);
    show(netG, step === 2);
    show(tabG, step === 3);

    if (t) {
      const net = t.state.net;
      // Shading: every frame on a wide screen, every 4th on a narrow one.
      frames++;
      if (full || !narrow || frames % 4 === 0) drawMaps(t);
      // Each hidden unit's line w_j·x + b_j = 0 (steps 3 and 4).
      unitLines.forEach((node, j) => {
        const on = (step === 2 || step === 3) && j < net.W[0].length;
        setLine(node, on ? lineEnds(net.W[0][j], net.b[0][j], IX, IY) : null);
      });
      if (step === 4) {
        hiddenPoints(net, TR.X).forEach((a, i) => movePoint(hidPts[i], TR.y[i], HX(a[0]), HY(a[1]), 3.4));
      }
      if (step === 2) diagram.update({ weights: net.W });
      const el2 = t.state.epochLoss;
      lossPath.setAttribute('d', el2.map((v, e) => `${e ? 'L' : 'M'}${P(LXs(e))},${P(LYs(Math.min(v, LOSS_MAX)))}`).join(''));
    } else {
      unitLines.forEach((node) => setLine(node, null));
    }

    if (step === 3) {
      const done = new Map();
      for (const h of H_CHOICES) if (runs.has(h) && finished(runs.get(h))) done.set(h, rowOf(runs.get(h)));
      tableRows(done).forEach((cells, r) => cells.forEach((text, c) => { tabCells[r][c].textContent = text; }));
      tabMark.setAttribute('y', TAB_Y + TAB_ROW * (H_CHOICES.indexOf(H) + 1) + 4);
      hOut.textContent = String(H);
      hInput.setAttribute('aria-valuetext', `H = ${H} hidden units`);
    }

    const texts = readout(step, t ? info(t, curH(), loop.running()) : null);
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let wd = 200;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.length && lens.every((len) => len > 0)) wd = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0, 598 - wd);
    const by = step === 0 ? QY1 + 10 : BOX_Y;
    boxRect.setAttribute('x', P(bx));
    boxRect.setAttribute('y', by);
    boxRect.setAttribute('width', P(wd));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', P(bx + 12));
      r.setAttribute('y', by + 8 + ROW * i + ROW / 2 + 5);
    });
  }

  const spoken = (i) => {
    if (i === 0) return `One sigmoid neuron fitted as logistic regression draws a straight line that gets ${pct(oneAccuracy(TR.X, TR.y))} of the training points and ${pct(oneAccuracy(TE.X, TE.y))} of the test points right.`;
    const s = info(cur(), curH(), loop.running());
    const where = `${s.H} hidden units, epoch ${s.epoch} of ${EPOCHS}: training loss ${num(s.loss, 3)}, train accuracy ${pct(s.train)}, test accuracy ${pct(s.test)}.`;
    if (i === 1) return `Press Train to train the network. ${where}`;
    if (i === 2) return `Each hidden unit's line is drawn dashed. ${where}`;
    if (i === 3) return `Choose the number of hidden units and press Train. ${where}`;
    if (i === 4) return `Each training point drawn at its two hidden activations. A straight line there gets ${pct(s.hidden ?? 0)}, the same as the network. ${where}`;
    return `Zoomed out to ${FAR_DOM[1]} in each direction. ${where} The lowest p̂ on the circle of radius ${FAR_R} is ${num(s.far, 2)}.`;
  };

  // ---- The training loop.
  const loop = createTrainLoop({
    step: (n) => frameStep(cur(), n),
    render: () => render(false),
    done: () => finished(cur()),
    onStop: () => {
      buttons.sync();
      render(true);
      if (finished(cur())) announce(`Finished ${EPOCHS} epochs. ${spoken(step)}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { cur().reset(); },
    reduced,
    runToEnd: () => { advance(cur(), MAX_ITER); },
    render: () => { render(true); announce(spoken(step)); },
  });

  // Pausing does not call render, so redraw the read-out after any Train/Pause/Reset click.
  trainGroup.addEventListener('click', () => { if (step >= 1) render(true); });

  hInput.addEventListener('input', () => {
    buttons.stop();
    H = H_CHOICES[Number(hInput.value)] ?? 4;
    render(true);
    announce(spoken(step));
  });

  function fill(vals) {
    for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
      const val = vals[span.dataset.value];
      if (val !== undefined) span.textContent = val;
    }
  }
  fill(baseValues());

  measure();
  if (typeof ResizeObserver === 'function') new ResizeObserver(measure).observe(svg);
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(() => render(true), () => {});

  return {
    count: 6,
    show(i) {
      buttons.stop();
      step = i;
      shown = true;
      showGroup(hGroup, i === 3);
      showGroup(trainGroup, i >= 1);
      controls.classList.toggle('has-train', i >= 1);
      render(true);
      announce(spoken(i));
      return Promise.resolve();
    },
  };
}
