// Lesson L4.4, "Training a network": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-training-networks.test.mjs);
// the DOM is only touched inside createStage. Every number on the stage is read
// from the network being trained at that moment; the numbers written in the
// page's text come from the same seeded runs (storyValues, checked by the tests).

import { mulberry32 } from '../assets/maths.js';
import { sigmoid } from '../assets/classify.js';
import { spiralData, mlpInit, mlpForward, trainer, accuracyNet, bceLogit } from '../assets/nn.js';
import { linearScale, el, axes, classMarker, regionMap, regionCells } from '../assets/plot.js';
import { createTrainLoop, trainButtons } from '../assets/train-loop.js';
import { contour, pct, num } from '../hidden-layers/stage.js';

export { pct, num };

export const SEED_TRAIN = 171;
export const SEED_TEST = 172;
export const SEED_INIT = 173;
export const SEED_SHUFFLE = 174;
export const SEED_SMALL = 175;
export const N_TRAIN = 160;
export const N_TEST = 400;
export const N_SMALL = 60;
export const SIZES = [2, 16, 16, 1];
export const BIG_SIZES = [2, 64, 64, 1];
export const ETA = 0.01;

// Step 1: three batch sizes, each run for the same 1600 examples (10 epochs).
export const BATCHES = [1, 8, 160];
export const NOISE_EXAMPLES = 1600;
export const NOISE_MAX = 2.4;

// Steps 2 and 3: the spiral run.
export const BATCH = 16;
export const EPOCHS = 300;
export const ITERS_PER_EPOCH = N_TRAIN / BATCH;
export const MAX_ITER = EPOCHS * ITERS_PER_EPOCH;
// Step 2 shows one batch at a time: one optimiser step every SLOW_FRAMES frames.
export const SLOW_FRAMES = 10;
// Step 3: slow at first so the early bending can be watched.
export const paceAt = (iter) => (iter < 300 ? 2 : iter < 800 ? 4 : 10);

// Steps 4 and 5: the big network on the small training set.
export const BIG_BATCH = 20;
export const BIG_EPOCHS = 400;
export const BIG_ITERS = N_SMALL / BIG_BATCH;
export const BIG_MAX = BIG_EPOCHS * BIG_ITERS;
export const LAMBDA = 0.001;

// The plot's range in both directions, shading grids, and the loss charts' top.
export const DOM = [-3.8, 3.8];
export const GRID = [54, 50];
export const BIG_GRID = [40, 36];
export const LOSS_MAX = 0.8;

export const trainSet = () => spiralData({ n: N_TRAIN, seed: SEED_TRAIN });
export const testSet = () => spiralData({ n: N_TEST, seed: SEED_TEST });
export const smallSet = () => spiralData({ n: N_SMALL, seed: SEED_SMALL });
export const startNet = () => mlpInit(SIZES, { hidden: 'tanh', init: 'xavier', seed: SEED_INIT });
export const bigStartNet = () => mlpInit(BIG_SIZES, { hidden: 'tanh', init: 'xavier', seed: SEED_INIT });

const TR = trainSet();
const TE = testSet();
const SM = smallSet();

// Three significant figures with a true minus sign.
export function sig3(v) {
  if (v === 0) return '0.00';
  const s = Number(v.toPrecision(3)) === 0 ? '0.00' : v.toPrecision(3);
  const t = s.includes('e') ? String(Number(s)) : s;
  return t.startsWith('-') ? '−' + t.slice(1) : t;
}

// The output logit of a tanh network, with the weights copied into flat arrays
// once so the shading and the test loss need no allocations per point.
export function makeLogit(net) {
  const L = net.W.length;
  const Ws = net.W.map((M) => Float64Array.from(M.flat()));
  const bs = net.b.map((v) => Float64Array.from(v));
  const outs = net.b.map((v) => new Float64Array(v.length));
  const nIn = net.W.map((M) => M[0].length);
  const inp = new Float64Array(2);
  return (x, y) => {
    inp[0] = x;
    inp[1] = y;
    let prev = inp;
    for (let l = 0; l < L; l++) {
      const W = Ws[l];
      const b = bs[l];
      const out = outs[l];
      const m = nIn[l];
      for (let j = 0; j < out.length; j++) {
        let s = b[j];
        const o = j * m;
        for (let i = 0; i < m; i++) s += W[o + i] * prev[i];
        out[j] = l === L - 1 ? s : Math.tanh(s);
      }
      prev = out;
    }
    return prev[0];
  };
}

// Mean binary cross-entropy (log-loss) over a data set, from the fast logit.
export function meanLossFast(net, D) {
  const f = makeLogit(net);
  let s = 0;
  for (let i = 0; i < D.X.length; i++) s += bceLogit(D.y[i], f(D.X[i][0], D.X[i][1]));
  return s / D.X.length;
}

// Mean log-loss over the training points, computed the trainer's way.
function trainLoss(net) {
  let s = 0;
  for (let i = 0; i < TR.X.length; i++) s += bceLogit(TR.y[i], mlpForward(net, TR.X[i]).out[0]);
  return s / TR.X.length;
}

export function meanAbsChange(ys) {
  let s = 0;
  for (let i = 1; i < ys.length; i++) s += Math.abs(ys[i] - ys[i - 1]);
  return s / (ys.length - 1);
}

// ---- step 1: the same 1600 examples in batches of 1, 8 and 160

const NOISE = new Map();
// The loss on each step's batch, measured just before that step, plotted at the
// number of examples seen once that batch is used.
export function noiseRun(b) {
  if (!NOISE.has(b)) {
    const t = trainer({ net: startNet(), X: TR.X, y: TR.y, batch: b, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
    const steps = NOISE_EXAMPLES / b;
    t.step(steps);
    const ys = t.state.batchLoss.slice();
    NOISE.set(b, {
      b, steps, ys,
      xs: ys.map((_, i) => (i + 1) * b),
      noise: meanAbsChange(ys),
      end: t.state.epochLoss[NOISE_EXAMPLES / N_TRAIN],
    });
  }
  return NOISE.get(b);
}

export const noiseInfo = (b) => noiseRun(b);

// ---- steps 2 and 3: the spiral run

export function makeRun() {
  return trainer({ net: startNet(), X: TR.X, y: TR.y, batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
}

export const finished = (t) => t.state.iter >= MAX_ITER || t.state.diverged;

// k more optimiser steps, never past 300 epochs.
export function advance(t, k) {
  const left = MAX_ITER - t.state.iter;
  if (left > 0) t.step(Math.min(k, left));
  return t.state;
}

// One animation frame of step 3: the loop offers n steps; the page takes at most paceAt.
export const frameStep = (t, n) => advance(t, Math.min(n, paceAt(t.state.iter)));

// One animation frame of step 2: a single step every SLOW_FRAMES frames.
export function slowPacer() {
  let f = 0;
  return (t) => {
    f++;
    if (f >= SLOW_FRAMES) {
      f = 0;
      advance(t, 1);
    }
    return t.state;
  };
}

// The trainer's shuffle, replayed: epoch e's order of the 160 points.
const ORDERS = [];
let orderRng = null;
export function epochOrder(e) {
  if (!orderRng) orderRng = mulberry32(SEED_SHUFFLE);
  while (ORDERS.length <= e) {
    const o = Array.from({ length: N_TRAIN }, (_, i) => i);
    for (let i = N_TRAIN - 1; i > 0; i--) {
      const j = Math.floor(orderRng() * (i + 1));
      [o[i], o[j]] = [o[j], o[i]];
    }
    ORDERS.push(o);
  }
  return ORDERS[e];
}

// The points used by optimiser step k (k = 1, 2, …).
export function batchAt(k) {
  const e = Math.floor((k - 1) / ITERS_PER_EPOCH);
  const p = (k - 1) % ITERS_PER_EPOCH;
  return epochOrder(e).slice(p * BATCH, (p + 1) * BATCH);
}

// Finished epochs, and how many of the latest epoch's batches are done (all 10
// once an epoch has just ended, so the count never reads "0 of 10" mid-run).
export const epochInfo = (iter) => ({
  epoch: Math.floor(iter / ITERS_PER_EPOCH),
  batch: iter === 0 ? 0 : ((iter - 1) % ITERS_PER_EPOCH) + 1,
});

export function spiralInfo(t) {
  const net = t.state.net;
  return {
    iter: t.state.iter,
    ...epochInfo(t.state.iter),
    loss: trainLoss(net),
    train: accuracyNet(net, TR.X, TR.y),
    test: accuracyNet(net, TE.X, TE.y),
    done: finished(t),
  };
}

// ---- steps 4 and 5: the big network, with the test loss after every epoch

export function makeBig(lambda) {
  const b = {
    lambda,
    t: trainer({ net: bigStartNet(), X: SM.X, y: SM.y, batch: BIG_BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE, lambda }),
    test: [],
    reset() { b.t.reset(); b.test = [meanLossFast(b.t.state.net, TE)]; },
  };
  b.test.push(meanLossFast(b.t.state.net, TE));
  return b;
}

export const bigFinished = (b) => b.t.state.iter >= BIG_MAX || b.t.state.diverged;

// k more optimiser steps, recording the test loss whenever an epoch ends.
export function bigAdvance(b, k) {
  let left = Math.min(k, BIG_MAX - b.t.state.iter);
  while (left > 0 && !b.t.state.diverged) {
    const m = Math.min(left, BIG_ITERS - (b.t.state.iter % BIG_ITERS));
    b.t.step(m);
    left -= m;
    if (b.t.state.iter % BIG_ITERS === 0) b.test.push(meanLossFast(b.t.state.net, TE));
  }
  return b.t.state;
}

// The epoch with the lowest test loss (the first, if tied).
export function earlyStop(test) {
  let e = 0;
  for (let i = 1; i < test.length; i++) if (test[i] < test[e]) e = i;
  return { epoch: e, value: test[e] };
}

export function bigInfo(b) {
  const e = b.t.state.epoch;
  return {
    lambda: b.lambda,
    epoch: e,
    train: b.t.state.epochLoss[e],
    test: b.test[e],
    best: earlyStop(b.test),
    done: bigFinished(b),
  };
}

// The current run, plus the final test loss of each weight-decay setting that has finished.
export function wdInfo(b, runs) {
  const end = (r) => (r && bigFinished(r) ? r.test[BIG_EPOCHS] : null);
  return { ...bigInfo(b), end0: end(runs[0]), endW: end(runs[LAMBDA]) };
}

const NOT_DONE = 'not finished: press Train';

export function readout(step, s) {
  if (step === 0) {
    return [
      `batch size ${s.b}${s.b === N_TRAIN ? ' (all)' : ''}`,
      `${N_TRAIN / s.b} steps per epoch`,
      `${s.steps} steps in ${NOISE_EXAMPLES / N_TRAIN} epochs`,
      `mean |change|: ${sig3(s.noise)}`,
    ];
  }
  if (step === 1) {
    return [`steps: ${s.iter}`, `epochs done: ${s.epoch}`, `batch ${s.batch} of ${ITERS_PER_EPOCH} done`, `train loss: ${num(s.loss, 3)}`];
  }
  if (step === 2) {
    const rows = [`epoch ${s.epoch} of ${EPOCHS}`, `train loss: ${num(s.loss, 3)}`, `train: ${pct(s.train)}`, `test: ${pct(s.test)}`];
    return s.done ? rows : [...rows, NOT_DONE];
  }
  if (step === 3) {
    const rows = [`epoch ${s.epoch} of ${BIG_EPOCHS}`, `train loss: ${num(s.train, 3)}`, `test loss: ${num(s.test, 3)}`, `best test: ${num(s.best.value, 3)} @ ${s.best.epoch}`];
    return s.done ? rows : [...rows, NOT_DONE];
  }
  if (step === 4) {
    const end = (v) => (v === null ? 'not finished' : num(v, 3));
    return [
      `λ = ${s.lambda} · epoch ${s.epoch}`,
      `train loss: ${num(s.train, 3)}`,
      `test loss: ${num(s.test, 3)}`,
      'final test loss:',
      `λ = 0: ${end(s.end0)}`,
      `λ = ${LAMBDA}: ${end(s.endW)}`,
    ];
  }
  return [];
}

// Weights and biases in a fully connected network with these layer sizes.
export const paramCount = (sizes) => sizes.slice(1).reduce((s, m, l) => s + m * sizes[l] + m, 0);

// Filled into the page at run time (constants of the runs).
export function pageValues() {
  return {
    'n-train': String(N_TRAIN),
    'n-test': String(N_TEST),
    'n-small': String(N_SMALL),
    'batch': String(BATCH),
    'iters': String(ITERS_PER_EPOCH),
    'epochs': String(EPOCHS),
    'eta': String(ETA),
    'noise-ex': String(NOISE_EXAMPLES),
    'big-batch': String(BIG_BATCH),
    'big-iters': String(BIG_ITERS),
    'big-epochs': String(BIG_EPOCHS),
    'lambda': String(LAMBDA),
    'params': String(paramCount(SIZES)),
    'big-params': String(paramCount(BIG_SIZES)),
  };
}

// From the seeded training runs. Written in the page and checked by the tests;
// the page does not train on load, so these are not recomputed in the browser.
let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  for (const b of BATCHES) {
    const r = noiseRun(b);
    v[`noise-${b}`] = sig3(r.noise);
    v[`end-${b}`] = num(r.end, 3);
  }
  const t = makeRun();
  advance(t, MAX_ITER);
  v['sp-train'] = pct(accuracyNet(t.state.net, TR.X, TR.y));
  v['sp-test'] = pct(accuracyNet(t.state.net, TE.X, TE.y));
  v['sp-loss0'] = num(t.state.epochLoss[0], 2);
  v['sp-loss'] = num(t.state.epochLoss[EPOCHS], 3);
  const g = makeBig(0);
  bigAdvance(g, BIG_MAX);
  const best = earlyStop(g.test);
  const tl = g.t.state.epochLoss;
  v['big-best-epoch'] = String(best.epoch);
  v['big-best'] = num(best.value, 3);
  v['big-end'] = num(g.test[BIG_EPOCHS], 3);
  v['big-rise'] = `${Math.round((g.test[BIG_EPOCHS] / best.value - 1) * 100)}%`;
  v['big-train-best'] = num(tl[best.epoch], 3);
  v['big-train-end'] = num(tl[BIG_EPOCHS], 3);
  let ups = 0;
  for (let e = 1; e <= best.epoch; e++) if (g.test[e] > g.test[e - 1]) ups++;
  v['big-ups'] = String(ups);
  v['big-train-acc'] = pct(accuracyNet(g.t.state.net, SM.X, SM.y));
  v['big-test-acc'] = pct(accuracyNet(g.t.state.net, TE.X, TE.y));
  const w = makeBig(LAMBDA);
  bigAdvance(w, BIG_MAX);
  const bw = earlyStop(w.test);
  v['wd-end'] = num(w.test[BIG_EPOCHS], 3);
  v['wd-best'] = num(bw.value, 3);
  v['wd-rise'] = `${Math.round((w.test[BIG_EPOCHS] / bw.value - 1) * 100)}%`;
  v['wd-train-end'] = num(w.t.state.epochLoss[BIG_EPOCHS], 3);
  v['wd-train-acc'] = pct(accuracyNet(w.t.state.net, SM.X, SM.y));
  v['wd-test-acc'] = pct(accuracyNet(w.t.state.net, TE.X, TE.y));
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The square plot on the left (or step 1's chart in the same place), the loss
// chart top right, the read-out under it.
const QX0 = 60;
const QX1 = 312;
const QY0 = 286;
const QY1 = 34;
const LX0 = 420;
const LX1 = 590;
const LY0 = 124;
const LY1 = 40;
const RX0 = 388;
const BOX_Y = 172;
const ROW = 20;

const BATCH_STYLE = {
  1: { stroke: CYAN, 'stroke-width': 0.8 },
  8: { stroke: VIOLET, 'stroke-width': 1.4, 'stroke-dasharray': '5 3' },
  160: { stroke: STRONG, 'stroke-width': 2.4 },
};

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let pick = 1;
  let lam = 0;
  const spiral = makeRun();
  const bigs = new Map();
  const big = (l) => {
    if (!bigs.has(l)) bigs.set(l, makeBig(l));
    return bigs.get(l);
  };
  const curBig = () => big(step === 4 ? lam : 0);
  const usesBig = () => step === 3 || step === 4;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Two interleaved spirals of points, cyan circles class 0 and violet triangles class 1, shaded by the network\'s p̂ with its boundary drawn; on the right loss curves and the network\'s numbers.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l44-plot' }, defs);
  el('rect', { x: QX0, y: QY1, width: QX1 - QX0, height: QY0 - QY1 }, clip);
  const P = (v) => Number(v.toFixed(2));
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const IX = linearScale(DOM, [QX0, QX1]);
  const IY = linearScale(DOM, [QY0, QY1]);

  // A region map whose staircase is hidden, with a smooth p̂ = 0.5 contour on top.
  const smoothMap = (parent, grid) => {
    const cells = regionCells(grid[0], grid[1], IX, IY);
    const vals = new Array(cells.length);
    let k = 0;
    const record = (fn) => (x, y) => { const p = fn(x, y); vals[k++] = p; return p; };
    const map = regionMap(parent, IX, IY, () => 0.5, { nx: grid[0], ny: grid[1], mode: 'prob' });
    map.g.querySelector('.boundary').style.display = 'none';
    const path = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 2, 'stroke-linejoin': 'round', 'clip-path': 'url(#l44-plot)' }, parent);
    const { w, h, px: left, py: top } = cells[0];
    return {
      update(net) {
        const f = makeLogit(net);
        k = 0;
        map.update(record((x, y) => sigmoid(f(x, y))));
        path.setAttribute('d', contour(vals, grid[0], grid[1], 0.5)
          .map(([[a, b], [c, d]]) => `M${P(left + a * w)},${P(top + b * h)}L${P(left + c * w)},${P(top + d * h)}`).join(''));
      },
    };
  };

  // ---- Key above the plot.
  const keyG = el('g', {}, svg);
  classMarker(keyG, QX0 + 6, 16, 0, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 16, y: 20, text: 'class 0' }, keyG);
  classMarker(keyG, QX0 + 120, 16, 1, { r: 5 });
  el('text', { class: 'stage-text', x: QX0 + 130, y: 20, text: 'class 1' }, keyG);

  // ---- Step 1: the batch-loss chart in the plot's place.
  const NX = linearScale([0, NOISE_EXAMPLES], [QX0, QX1 + 20]);
  const NY = linearScale([0, NOISE_MAX], [QY0, QY1]);
  const noiseG = el('g', {}, svg);
  axes(noiseG, NX, NY, { ticks: 4, xLabel: 'examples seen', yLabel: 'loss on the batch' });
  const curveG = el('g', {}, noiseG);
  const curves = new Map();
  for (const b of BATCHES) {
    const g = el('g', {}, curveG);
    const path = el('path', { fill: 'none', 'stroke-linejoin': 'round', ...BATCH_STYLE[b] }, g);
    curves.set(b, { g, path, drawn: false });
  }
  const legG = el('g', {}, noiseG);
  BATCHES.forEach((b, k) => {
    const y = QY1 + 6 + 21 * k;
    el('line', { x1: RX0 + 4, x2: RX0 + 30, y1: y, y2: y, ...BATCH_STYLE[b], 'stroke-width': Math.max(1.4, BATCH_STYLE[b]['stroke-width']) }, legG);
    if (b === N_TRAIN) el('circle', { cx: RX0 + 17, cy: y, r: 3, fill: STRONG }, legG);
    el('text', { class: 'stage-text', x: RX0 + 38, y: y + 4, text: b === N_TRAIN ? 'batch 160 (all)' : `batch ${b}` }, legG);
  });
  const drawCurve = (b) => {
    const c = curves.get(b);
    if (c.drawn) return;
    const r = noiseRun(b);
    c.path.setAttribute('d', r.ys.map((v, i) => `${i ? 'L' : 'M'}${P(NX(r.xs[i]))},${P(NY(v))}`).join(''));
    if (b === N_TRAIN) r.ys.forEach((v, i) => el('circle', { cx: P(NX(r.xs[i])), cy: P(NY(v)), r: 3, fill: STRONG }, c.g));
    c.drawn = true;
  };

  // ---- Steps 2 and 3: the spiral network.
  const spG = el('g', {}, svg);
  const spMap = smoothMap(el('g', {}, spG), GRID);
  axes(spG, IX, IY, { xLabel: 'x₁', yLabel: 'x₂' });
  const spPts = el('g', {}, spG);
  TR.X.forEach((x, i) => classMarker(spPts, P(IX(x[0])), P(IY(x[1])), TR.y[i], { r: 3.4 }));
  const ringG = el('g', {}, spG);
  const rings = Array.from({ length: BATCH }, () => el('circle', { r: 7, fill: 'none', stroke: STRONG, 'stroke-width': 1.6 }, ringG));

  // ---- Steps 4 and 5: the big network on 60 points.
  const bgG = el('g', {}, svg);
  const bgMap = smoothMap(el('g', {}, bgG), BIG_GRID);
  axes(bgG, IX, IY, { xLabel: 'x₁', yLabel: 'x₂' });
  const bgPts = el('g', {}, bgG);
  SM.X.forEach((x, i) => classMarker(bgPts, P(IX(x[0])), P(IY(x[1])), SM.y[i], { r: 4 }));

  // ---- Step 2: this epoch's batches, one cell each.
  const epG = el('g', {}, svg);
  el('text', { class: 'stage-text', x: LX0 - 20, y: LY1 + 4, text: 'this epoch\'s batches' }, epG);
  const CELL = 16;
  const cells = Array.from({ length: ITERS_PER_EPOCH }, (_, k) => el('rect', { x: LX0 - 20 + k * (CELL + 2), y: LY1 + 16, width: CELL, height: CELL, rx: 2, stroke: MUTED, 'stroke-width': 1 }, epG));

  // ---- Loss chart, top right (steps 3 to 5).
  const lossG = el('g', {}, svg);
  const LYs = linearScale([0, LOSS_MAX], [LY0, LY1]);
  let LXs = linearScale([0, EPOCHS], [LX0, LX1]);
  const lossAxes = el('g', {}, lossG);
  const drawLossAxes = (E) => {
    LXs = linearScale([0, E], [LX0, LX1]);
    lossAxes.replaceChildren();
    axes(lossAxes, LXs, LYs, { ticks: 3 });
  };
  drawLossAxes(EPOCHS);
  const lossTitle = el('text', { class: 'stage-text', x: LX0 - 30, y: LY1 - 10, text: 'training loss' }, lossG);
  el('text', { class: 'stage-text', x: LX1, y: LY0 + 36, 'text-anchor': 'end', text: 'epoch' }, lossG);
  const otherPath = el('path', { fill: 'none', stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '1.5 3' }, lossG);
  const testPath = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, lossG);
  const trainPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, lossG);
  const bestLine = el('line', { y1: LY0, y2: LY1, stroke: STRONG, 'stroke-width': 1.2, 'stroke-dasharray': '2 3' }, lossG);
  const bestDot = el('circle', { r: 3.5, fill: VIOLET, stroke: STRONG, 'stroke-width': 1 }, lossG);
  const curvePath = (ys) => ys.map((v, e) => `${e ? 'L' : 'M'}${P(LXs(e))},${P(LYs(Math.min(v, LOSS_MAX)))}`).join('');

  // ---- Step 6: the knobs.
  const knobG = el('g', {}, svg);
  const KX = [40, 200, 330];
  ['knob', 'on this page', 'watch for'].forEach((t, c) => el('text', { x: KX[c], y: 60, fill: MUTED, style: 'font-size: 15px', text: t }, knobG));
  el('line', { x1: KX[0], x2: 570, y1: 70, y2: 70, stroke: LINE }, knobG);
  [
    ['batch size', String(BATCH), 'small: noisy steps'],
    ['epochs', String(EPOCHS), 'many: memorising'],
    ['η', String(ETA), 'large: unsteady'],
    ['λ', String(LAMBDA), 'large: underfitting'],
  ].forEach((row, r) => row.forEach((t, c) => el('text', { x: KX[c], y: 104 + 42 * r, fill: STRONG, style: 'font-size: 18px', text: t }, knobG)));
  el('text', { x: KX[0], y: 290, fill: MUTED, style: 'font-size: 15px', text: 'judge them on held-back points, not the training loss' }, knobG);

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4, 5].map(() => el('text', { class: 'value' }, box));

  // ---- Controls: batch picker (step 1), weight decay (step 5), Train / Pause / Reset.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const pickGroup = make('div');
  pickGroup.setAttribute('role', 'group');
  pickGroup.setAttribute('aria-label', 'batch size');
  const pickLabel = make('span', { textContent: 'batch size' });
  const pickBtns = BATCHES.map((b) => make('button', { type: 'button', textContent: b === N_TRAIN ? '160 (all)' : String(b) }));
  pickGroup.append(pickLabel, ...pickBtns);
  const wdGroup = make('div');
  const wdBtn = make('button', { type: 'button', textContent: `weight decay λ = ${LAMBDA}` });
  wdGroup.append(wdBtn);
  const trainGroup = make('div');
  controls.append(pickGroup, wdGroup, trainGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  let narrow = false;
  const measure = () => {
    const wd = svg.clientWidth || (svg.getBoundingClientRect ? svg.getBoundingClientRect().width : 600);
    narrow = wd > 0 && wd < 500;
  };
  let frames = 0;

  function info() {
    if (step === 0) return noiseInfo(pick);
    if (step === 1 || step === 2) return spiralInfo(spiral);
    if (step === 3) return bigInfo(curBig());
    if (step === 4) return wdInfo(curBig(), { 0: bigs.get(0), [LAMBDA]: bigs.get(LAMBDA) });
    return null;
  }

  function render(full = true) {
    show(noiseG, step === 0);
    show(spG, step === 1 || step === 2);
    show(bgG, usesBig());
    show(keyG, step >= 1 && step <= 4);
    show(epG, step === 1);
    show(lossG, step >= 2 && step <= 4);
    show(knobG, step === 5);
    show(box, step <= 4);
    frames++;

    if (step === 0) {
      for (const b of BATCHES) drawCurve(b);
      // Fixed order (batch 1 at the back); the picked curve is opaque, the others faint.
      for (const b of BATCHES) curves.get(b).g.style.opacity = b === pick ? 1 : b === 1 ? 0.22 : 0.45;
      pickBtns.forEach((btn, k) => btn.setAttribute('aria-pressed', String(BATCHES[k] === pick)));
    }
    if (step === 1 || step === 2) {
      const net = spiral.state.net;
      if (full || !narrow || frames % 4 === 0) spMap.update(net);
      const it = spiral.state.iter;
      const idx = step === 1 && it > 0 ? batchAt(it) : [];
      rings.forEach((r, k) => {
        const i = idx[k];
        show(r, i !== undefined);
        if (i !== undefined) { r.setAttribute('cx', P(IX(TR.X[i][0]))); r.setAttribute('cy', P(IY(TR.X[i][1]))); }
      });
      const done = epochInfo(it).batch;
      cells.forEach((c, k) => { c.setAttribute('fill', k < done ? CYAN : 'none'); c.setAttribute('fill-opacity', k < done ? 0.7 : 0); });
      if (step === 2) {
        trainPath.setAttribute('d', curvePath(spiral.state.epochLoss));
        testPath.setAttribute('d', '');
        otherPath.setAttribute('d', '');
        show(bestLine, false);
        show(bestDot, false);
      }
    }
    if (usesBig()) {
      const b = curBig();
      if (full || frames % (narrow ? 12 : 6) === 0) bgMap.update(b.t.state.net);
      trainPath.setAttribute('d', curvePath(b.t.state.epochLoss));
      testPath.setAttribute('d', curvePath(b.test));
      const other = step === 4 ? bigs.get(lam ? 0 : LAMBDA) : null;
      otherPath.setAttribute('d', other && other.test.length > 1 ? curvePath(other.test) : '');
      const best = earlyStop(b.test);
      const on = b.test.length > 1;
      show(bestLine, on);
      show(bestDot, on);
      if (on) {
        bestLine.setAttribute('x1', P(LXs(best.epoch)));
        bestLine.setAttribute('x2', P(LXs(best.epoch)));
        bestDot.setAttribute('cx', P(LXs(best.epoch)));
        bestDot.setAttribute('cy', P(LYs(best.value)));
      }
      wdBtn.setAttribute('aria-pressed', String(lam === LAMBDA));
    }

    const texts = step <= 4 ? readout(step, info()) : [];
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let wd = 200;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) wd = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0, 598 - wd);
    const by = step === 0 ? QY1 + 78 : step === 1 ? LY1 + 60 : BOX_Y;
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
    const s = info();
    if (i === 0) return `Batch size ${s.b}: ${s.steps} steps in ${NOISE_EXAMPLES} examples; the batch loss changes by ${sig3(s.noise)} from one step to the next on average.`;
    if (i === 1) return `Press Train to run slowly. ${s.iter} steps, ${s.epoch} epochs done, batch ${s.batch} of ${ITERS_PER_EPOCH} in this epoch. Training loss ${num(s.loss, 3)}.`;
    if (i === 2) return `Epoch ${s.epoch} of ${EPOCHS}: training loss ${num(s.loss, 3)}, train accuracy ${pct(s.train)}, test accuracy ${pct(s.test)}.`;
    if (i === 3) return `Big network, epoch ${s.epoch} of ${BIG_EPOCHS}: training loss ${num(s.train, 3)}, test loss ${num(s.test, 3)}, lowest test loss ${num(s.best.value, 3)} at epoch ${s.best.epoch}.`;
    if (i === 4) return `Weight decay λ = ${s.lambda}, epoch ${s.epoch} of ${BIG_EPOCHS}: training loss ${num(s.train, 3)}, test loss ${num(s.test, 3)}.`;
    return 'The four knobs: batch size, epochs, learning rate and weight decay.';
  };

  const stepFns = { 1: slowPacer(), 2: (t, n) => frameStep(t, n) };
  const isDone = () => (usesBig() ? bigFinished(curBig()) : finished(spiral));

  // ---- The training loop.
  const loop = createTrainLoop({
    step: (n) => {
      if (usesBig()) bigAdvance(curBig(), n);
      else if (stepFns[step]) stepFns[step](spiral, n);
    },
    render: () => render(false),
    done: isDone,
    onStop: () => {
      buttons.sync();
      render(true);
      if (isDone()) announce(`Finished. ${spoken(step)}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { if (usesBig()) curBig().reset(); else spiral.reset(); },
    reduced,
    runToEnd: () => { if (usesBig()) bigAdvance(curBig(), BIG_MAX); else advance(spiral, MAX_ITER); },
    render: () => { render(true); announce(spoken(step)); },
  });

  pickBtns.forEach((btn, k) => btn.addEventListener('click', () => {
    pick = BATCHES[k];
    render(true);
    announce(spoken(step));
  }));
  wdBtn.addEventListener('click', () => {
    buttons.stop();
    lam = lam ? 0 : LAMBDA;
    render(true);
    announce(spoken(step));
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  measure();
  if (typeof ResizeObserver === 'function') new ResizeObserver(measure).observe(svg);
  render(true);
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(() => render(true), () => {});

  return {
    count: 6,
    show(i) {
      buttons.stop();
      step = i;
      showGroup(pickGroup, i === 0);
      showGroup(wdGroup, i === 4);
      showGroup(trainGroup, i >= 1 && i <= 4);
      controls.classList.toggle('has-train', i >= 1 && i <= 4);
      if (i === 2) { drawLossAxes(EPOCHS); lossTitle.textContent = 'training loss'; }
      if (i === 3 || i === 4) { drawLossAxes(BIG_EPOCHS); lossTitle.textContent = 'train — / test - -'; }
      render(true);
      announce(spoken(i));
      return Promise.resolve();
    },
  };
}
