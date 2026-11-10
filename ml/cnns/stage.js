// Lesson L4.8, "Convolutional networks": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-cnns.test.mjs); the
// DOM is only touched inside createStage. Every map is cnnForward from conv.js,
// every training run is cnnTrainer below (or nn.js's trainer for the dense net),
// and every number in the page's text comes from pageValues or storyValues.

import { mulberry32 } from '../assets/maths.js';
import { sigmoid } from '../assets/classify.js';
import { shapeImages, cnnInit, cnnForward, cnnBackward, cnnFlatten, cnnUnflatten, cnnParamCount, denseParamCount } from '../assets/conv.js';
import { mlpInit, trainer, accuracyNet, optInit, optStep, bceLogit } from '../assets/nn.js';
import { el, pixelFill, classMarker } from '../assets/plot.js';
import { createTrainLoop, trainButtons } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const SEED_TRAIN = 201;
export const SEED_TEST = 202;
export const SEED_INIT = 203;
export const SEED_SHUFFLE = 204;
export const SEED_DENSE = 205;
export const N_TRAIN = 200;
export const N_TEST = 400;
export const ETA = 0.02;
export const BATCH = 20;
export const EPOCHS = 30;
export const ITERS = (EPOCHS * N_TRAIN) / BATCH;
export const DENSE_HIDDEN = 18;
export const SIDE = 8;

// The example shown in steps 2 to 4 by default (a cross), and the cross and square of step 5.
export const EXAMPLE = 0;
export const CROSS = 0;
export const SQUARE = 1;
export const SHOWN = 12;

export const trainSet = () => shapeImages({ n: N_TRAIN, seed: SEED_TRAIN });
export const testSet = () => shapeImages({ n: N_TEST, seed: SEED_TEST });
export const startNet = () => cnnInit({ seed: SEED_INIT });
export const denseStart = () => mlpInit([SIDE * SIDE, DENSE_HIDDEN, 1], { hidden: 'relu', init: 'he', seed: SEED_DENSE });
export const flat = (img) => img.flat();

const TR = trainSet();
const TE = testSet();

// Where the shape of image i is centred: the first two draws of its stretch of
// the generator, as in shapeImages (one randn per pixel follows; randn takes two
// uniforms per call).
export function shapeCentre(i) {
  const rng = mulberry32(SEED_TRAIN);
  let rc = null;
  for (let k = 0; k <= i; k++) {
    const r = 1 + Math.floor(rng() * (SIDE - 2));
    const c = 1 + Math.floor(rng() * (SIDE - 2));
    if (k === i) rc = [r, c];
    for (let q = 0; q < 2 * SIDE * SIDE; q++) rng();
  }
  return rc;
}

// ---- the CNN trainer ------------------------------------------------------------

const DIVERGED = 1e6;

export function cnnMeanLoss(net, X, y) {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += bceLogit(y[i], cnnForward(net, X[i]).z);
  return s / X.length;
}

export function cnnAccuracy(net, X, y) {
  let ok = 0;
  for (let i = 0; i < X.length; i++) ok += ((cnnForward(net, X[i]).z > 0 ? 1 : 0) === y[i]) ? 1 : 0;
  return ok / X.length;
}

// The mean over idx of each image's gradient (flat, cnnFlatten order), and the mean loss.
export function batchGradCnn(net, X, y, idx) {
  let grad = null;
  let loss = 0;
  for (const k of idx) {
    const b = cnnBackward(net, X[k], y[k]);
    const g = cnnFlatten(b.grads);
    if (!grad) grad = g;
    else for (let q = 0; q < g.length; q++) grad[q] += g[q];
    loss += b.loss;
  }
  const m = idx.length;
  return { grad: grad.map((v) => v / m), loss: loss / m };
}

// Mini-batch training with the same rules as nn.js's trainer: a Fisher–Yates
// shuffle from mulberry32(seed) at the start of each epoch, batches taken in
// order, a last partial batch kept, epochLoss[0] the mean loss before any step.
// step(k) updates api.state in place; reset() gives a fresh state.
export function cnnTrainer({ net, X, y, batch, kind = 'adam', hyper = {}, seed }) {
  const start = cnnUnflatten(net, cnnFlatten(net));
  const n = X.length;
  let rng, order, cursor, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    order = null;
    cursor = 0;
    params = cnnFlatten(start);
    opt = optInit(kind, params.length);
    const first = cnnMeanLoss(start, X, y);
    return { net: cnnUnflatten(start, params), epoch: 0, iter: 0, batchLoss: [], epochLoss: [first], diverged: !Number.isFinite(first) };
  };

  const shuffle = () => {
    order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    cursor = 0;
  };

  const api = { state: fresh() };

  const one = () => {
    const s = api.state;
    if (cursor === 0) shuffle();
    const idx = order.slice(cursor, cursor + batch);
    const g = batchGradCnn(s.net, X, y, idx);
    if (!Number.isFinite(g.loss) || Math.abs(g.loss) > DIVERGED) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, g.grad);
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.net = cnnUnflatten(s.net, params);
    s.batchLoss.push(g.loss);
    s.iter++;
    cursor += batch;
    if (cursor >= n) {
      cursor = 0;
      s.epoch++;
      const el2 = cnnMeanLoss(s.net, X, y);
      s.epochLoss.push(el2);
      if (!Number.isFinite(el2) || Math.abs(el2) > DIVERGED) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}

export const makeTrainer = () => cnnTrainer({ net: startNet(), X: TR.X, y: TR.y, batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
const flatTR = () => TR.X.map(flat);
const flatTE = () => TE.X.map(flat);
export const makeDenseTrainer = () => trainer({ net: denseStart(), X: flatTR(), y: TR.y, batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });

// The page's live run: the trainer plus the test accuracy at the end of every epoch.
// advance(k) takes at most k steps and never runs past epoch 30.
export function makeLive() {
  const trainer1 = makeTrainer();
  const live = { trainer: trainer1, testAcc: [cnnAccuracy(trainer1.state.net, TE.X, TE.y)] };
  live.advance = (k) => {
    for (let i = 0; i < k && trainer1.state.iter < ITERS && !trainer1.state.diverged; i++) {
      const e = trainer1.state.epoch;
      trainer1.step(1);
      if (trainer1.state.epoch > e) live.testAcc.push(cnnAccuracy(trainer1.state.net, TE.X, TE.y));
    }
  };
  live.reset = () => {
    trainer1.reset();
    live.testAcc = [live.testAcc[0]];
  };
  live.done = () => trainer1.state.iter >= ITERS || trainer1.state.diverged;
  return live;
}

// ---- the end-of-run results for steps 5 and 6, computed on demand ---------------

export const RUN_KINDS = ['cnn', 'dense'];
const RUNS = new Map();
const JOBS = new Map();

// k more steps of this run; true once its result is ready. The page calls this in
// small slices; the slicing does not change the result.
export function runWork(kind, k) {
  if (RUNS.has(kind)) return true;
  if (kind === 'cnn') {
    if (!JOBS.has(kind)) JOBS.set(kind, makeLive());
    const live = JOBS.get(kind);
    live.advance(k);
    if (!live.done()) return false;
    const s = live.trainer.state;
    RUNS.set(kind, { losses: s.epochLoss.slice(), testAcc: live.testAcc.slice(), net: s.net, trainAcc: cnnAccuracy(s.net, TR.X, TR.y), diverged: s.diverged });
  } else {
    if (!JOBS.has(kind)) JOBS.set(kind, makeDenseTrainer());
    const t = JOBS.get(kind);
    t.step(Math.max(0, Math.min(k, ITERS - t.state.iter)));
    if (t.state.iter < ITERS && !t.state.diverged) return false;
    const s = t.state;
    RUNS.set(kind, { losses: s.epochLoss.slice(), net: s.net, trainAcc: accuracyNet(s.net, flatTR(), TR.y), testAcc: accuracyNet(s.net, flatTE(), TE.y), diverged: s.diverged });
  }
  JOBS.delete(kind);
  return true;
}
export const runReady = (kind) => RUNS.has(kind);
export function runResult(kind) {
  runWork(kind, Infinity);
  return RUNS.get(kind);
}
const readyRuns = () => (RUN_KINDS.every(runReady) ? { cnn: RUNS.get('cnn'), dense: RUNS.get('dense') } : null);

// ---- pooling and shifts ----------------------------------------------------------

// Move every pixel d columns right; the columns that come in on the left are 0.
export function shiftRight(img, d = 1) {
  return img.map((row) => row.map((_, c) => (c >= d ? row[c - d] : 0)));
}

// Total change as a share of the total: Σ|a − b| / Σ|a|.
export function mapChange(a, b) {
  const A = a.flat(2);
  const B = b.flat(2);
  let d = 0;
  let s = 0;
  for (let i = 0; i < A.length; i++) { d += Math.abs(A[i] - B[i]); s += Math.abs(A[i]); }
  return s > 0 ? d / s : 0;
}

export function shiftInfo(net, img) {
  const a = cnnForward(net, img);
  const b = cnnForward(net, shiftRight(img, 1));
  const P = a.pooled.flat(2);
  const Q = b.pooled.flat(2);
  let same = 0;
  for (let i = 0; i < P.length; i++) if (P[i] === Q[i]) same++;
  return { a, b, raw: mapChange(a.relu, b.relu), pooled: mapChange(a.pooled, b.pooled), same };
}

// ReLU cells that are not the maximum of their 2 × 2 pooling window: [filter, row, column].
export function nonMaxCells(fw) {
  const out = [];
  fw.relu.forEach((m, k) => m.forEach((row, i) => row.forEach((_, j) => {
    const [ai, aj] = fw.args[k][i >> 1][j >> 1];
    if (ai !== i || aj !== j) out.push([k, i, j]);
  })));
  return out;
}

// ---- what the filters became -------------------------------------------------------

export function filterMaps(net) {
  const fc = cnnForward(net, TR.X[CROSS]);
  const fs = cnnForward(net, TR.X[SQUARE]);
  const argmax = (m) => {
    let best = -Infinity;
    let at = [0, 0];
    m.forEach((row, i) => row.forEach((v, j) => { if (v > best) { best = v; at = [i, j]; } }));
    return at;
  };
  return {
    cross: fc, square: fs,
    crossMax: fc.relu.map((m) => Math.max(...m.flat())),
    squareMax: fs.relu.map((m) => Math.max(...m.flat())),
    squareAt: fs.relu.map(argmax),
    crossP: sigmoid(fc.z), squareP: sigmoid(fs.z),
  };
}

export const firstAbove = (accs, level) => accs.findIndex((a) => a >= level);

// ---- formatting ---------------------------------------------------------------------

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
// Accuracy as a percentage, up to two decimals (exact for 200 or 400 images).
export const pct = (v) => `${Math.round(v * 10000) / 100}%`;
export const pctWhole = (v) => `${Math.round(v * 100)}%`;
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const cls = (i) => (TR.y[i] ? 'square' : 'cross');

// ---- read-outs -------------------------------------------------------------------------

export const gridInfo = () => ({ n: SHOWN });
export function layerInfo(k) {
  const fw = cnnForward(startNet(), TR.X[k]);
  return { k, cls: cls(k), z: fw.z, p: sigmoid(fw.z) };
}
export const poolInfo = (k) => ({ k, ...shiftInfo(startNet(), TR.X[k]) });
export function liveInfo(live, k, running) {
  const s = live.trainer.state;
  return { k, epoch: s.epoch, loss: s.epochLoss.at(-1), acc: live.testAcc.at(-1), running, diverged: s.diverged };
}
export const filtersInfo = (runs) => (runs ? { ...filterMaps(runs.cnn.net) } : null);
export function compareInfo(runs) {
  if (!runs) return null;
  return { cnnParams: cnnParamCount(runs.cnn.net), denseParams: denseParamCount(SIDE * SIDE, DENSE_HIDDEN), cnn: runs.cnn, dense: runs.dense };
}

export function readout(step, s) {
  if (step === 0) return [`${s.n} of the ${N_TRAIN} training images`, 'circle: cross (class 0)', 'triangle: square (class 1)'];
  if (step === 1) return [`example ${s.k + 1}, a ${s.cls}`, '8×8 → 6×6×2 → 3×3×2 → 18 → 1', `starting filters: p(square) = ${num(s.p, 2)}`];
  if (step === 2) return [`example ${s.k + 1}, shifted right 1 pixel`, `maps after ReLU change by ${pctWhole(s.raw)}`, `pooled maps change by ${pctWhole(s.pooled)}`];
  if (step === 3) {
    const status = s.diverged ? 'diverged' : s.running && s.epoch < EPOCHS ? 'training…' : s.epoch === 0 ? 'press Train' : s.epoch < EPOCHS ? 'not finished: press Train' : 'finished';
    return [`epoch ${s.epoch} of ${EPOCHS} · ${status}`, `training loss ℒ: ${num(s.loss, 3)}`, `test accuracy: ${pct(s.acc)}`];
  }
  if (step === 4) {
    if (!s) return ['computing the end of the run…'];
    return ['largest value after ReLU', `filter 1: cross ${num(s.crossMax[0], 2)} · square ${num(s.squareMax[0], 2)}`, `filter 2: cross ${num(s.crossMax[1], 2)} · square ${num(s.squareMax[1], 2)}`];
  }
  if (step === 5) {
    if (!s) return ['computing both runs…'];
    return [`CNN: ${count(s.cnnParams)} weights and biases · test ${pct(s.cnn.testAcc.at(-1))}`, `dense: ${count(s.denseParams)} weights and biases · test ${pct(s.dense.testAcc)}`, `training accuracy: CNN ${pct(s.cnn.trainAcc)} · dense ${pct(s.dense.trainAcc)}`];
  }
  return [];
}

// ---- values written into the page ---------------------------------------------------

// Filled into the page at run time (constants of the two architectures).
export function pageValues() {
  const nC = cnnParamCount(startNet());
  const nD = denseParamCount(SIDE * SIDE, DENSE_HIDDEN);
  return {
    'cnn-params': String(nC),
    'dense-params': count(nD),
    'dense-ratio': String(Math.round(nD / nC)),
    'conv-w': String(startNet().filters.length * 9),
    'dense-in': count(SIDE * SIDE * DENSE_HIDDEN),
  };
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const sn = startNet();
  const sh = shiftInfo(sn, TR.X[EXAMPLE]);
  v['shift-raw'] = pctWhole(sh.raw);
  v['shift-pooled'] = pctWhole(sh.pooled);
  v['pool-same'] = String(sh.same);
  v['start-p'] = num(sigmoid(cnnForward(sn, TR.X[EXAMPLE]).z), 2);
  const c = runResult('cnn');
  const d = runResult('dense');
  v['cnn-start-loss'] = num(c.losses[0], 3);
  v['cnn-end-loss'] = num(c.losses[EPOCHS], 3);
  v['cnn-start-acc'] = pct(c.testAcc[0]);
  v['cnn-test'] = pct(c.testAcc[EPOCHS]);
  const f = firstAbove(c.testAcc, 0.95);
  v['first-95'] = String(f);
  v['first-95-acc'] = pct(c.testAcc[f]);
  v['before-95-acc'] = pct(c.testAcc[f - 1]);
  const [f1, f2] = c.net.filters;
  v['f1-centre'] = num(f1[1][1], 2);
  v['f2-centre'] = num(f2[1][1], 2);
  v['f1-side-l'] = num(f1[1][0], 2);
  v['f1-side-r'] = num(f1[1][2], 2);
  v['f1-bias'] = num(c.net.fb[0], 2);
  v['dw-positive'] = String(c.net.dw.filter((w) => w > 0).length);
  const m = filterMaps(c.net);
  for (const k of [0, 1]) {
    v[`f${k + 1}-cross`] = num(m.crossMax[k], 2);
    v[`f${k + 1}-square`] = num(m.squareMax[k], 2);
  }
  v['sq-prob'] = num(m.squareP, 3);
  v['cr-prob'] = num(m.crossP, 3);
  v['dense-test'] = pct(d.testAcc);
  v['dense-train'] = pct(d.trainAcc);
  v['dense-wrong'] = String(Math.round((1 - d.testAcc) * N_TEST));
  v['dense-end-loss'] = num(d.losses[EPOCHS], 3);
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [292, 314, 336];

// A grid of cells for an image, a map or a filter. 'grey': v in [0, 1] is the ink.
// 'signed': cyan for positive, violet for negative, opacity |v| / scale, and a minus
// glyph on every negative cell that reads as negative at two decimals. 'numbers'
// prints each value in its cell (filters).
function cellGrid(parent, rows, cols, { x, y, cell, kind = 'grey', numbers = false }) {
  const g = el('g', { class: 'pixels' }, parent);
  const rects = [];
  const marks = [];
  const texts = [];
  for (let a = 0; a < rows; a++) for (let b = 0; b < cols; b++) rects.push(el('rect', { x: x + b * cell, y: y + a * cell, width: cell, height: cell, opacity: 0 }, g));
  if (kind === 'signed' && !numbers) {
    for (let a = 0; a < rows; a++) for (let b = 0; b < cols; b++) marks.push(el('text', { class: 'px-sign', x: x + (b + 0.5) * cell, y: y + (a + 0.5) * cell, 'text-anchor': 'middle', text: '−', visibility: 'hidden' }, g));
  }
  if (numbers) {
    for (let a = 0; a < rows; a++) {
      for (let b = 0; b < cols; b++) {
        const t = el('text', { class: 'stage-text', x: x + (b + 0.5) * cell, y: y + (a + 0.5) * cell + 5, 'text-anchor': 'middle', 'font-size': 13, ...HALO }, g);
        t.style.fill = STRONG;
        texts.push(t);
      }
    }
  }
  el('rect', { class: 'px-frame', x, y, width: cols * cell, height: rows * cell }, g);
  const update = (vals, scale) => {
    const flatV = vals.flat();
    let max = scale;
    if (max === undefined) { max = 0; for (const v of flatV) max = Math.max(max, Math.abs(v)); }
    flatV.forEach((v0, k) => {
      const v = Number.isFinite(v0) ? v0 : 0;
      if (kind === 'grey') {
        const f = pixelFill(v, 'grey');
        rects[k].setAttribute('class', f.cls);
        rects[k].setAttribute('opacity', P(f.opacity));
        return;
      }
      const f = pixelFill(v, { signed: max });
      if (numbers) {
        rects[k].setAttribute('class', '');
        const shown = num(v, 2);
        rects[k].setAttribute('fill', shown === '0.00' ? LINE : v < 0 ? VIOLET : CYAN);
        rects[k].setAttribute('stroke', LINE);
        rects[k].setAttribute('opacity', 1);
        rects[k].setAttribute('fill-opacity', shown === '0.00' ? 0.4 : P(0.12 + 0.6 * f.opacity));
        texts[k].textContent = shown;
        return;
      }
      rects[k].setAttribute('class', f.cls);
      rects[k].setAttribute('opacity', P(f.opacity));
      if (marks.length) marks[k].setAttribute('visibility', v < 0 && num(v, 2) !== '0.00' ? 'visible' : 'hidden');
    });
  };
  return { g, update };
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let example = EXAMPLE;
  let live = null;
  const start = startNet();

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': 12, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  const markerLabel = (parent, x, y, c) => {
    classMarker(parent, x + 6, y - 4, c);
    label(parent, x + 16, y, c ? 'square' : 'cross');
  };

  // ---- step 1: twelve training images.
  const gridG = el('g', {}, svg);
  for (let k = 0; k < SHOWN; k++) {
    const gx = 24 + (k % 6) * 94;
    const gy = 30 + Math.floor(k / 6) * 124;
    cellGrid(gridG, SIDE, SIDE, { x: gx, y: gy, cell: 8 }).update(TR.X[k]);
    markerLabel(gridG, gx, gy + 84, TR.y[k]);
    label(gridG, gx + 64, gy - 6, String(k + 1), 'end');
  }

  // ---- step 2: the layers, with the starting filters.
  const layG = el('g', {}, svg);
  label(layG, 16, 22, 'image');
  label(layG, 141, 22, 'W', 'middle');
  label(layG, 214, 22, 'conv', 'middle');
  label(layG, 308, 22, 'ReLU', 'middle');
  label(layG, 387, 22, 'pool', 'middle');
  label(layG, 429, 22, '18', 'middle');
  label(layG, 446, 22, '');
  const layImg = cellGrid(layG, SIDE, SIDE, { x: 16, y: 80, cell: 12 });
  const layW = [57, 157].map((y) => cellGrid(layG, 3, 3, { x: 120, y, cell: 14, kind: 'signed' }));
  const layConv = [36, 136].map((y) => cellGrid(layG, 6, 6, { x: 172, y, cell: 14, kind: 'signed' }));
  const layRelu = [36, 136].map((y) => cellGrid(layG, 6, 6, { x: 266, y, cell: 14, kind: 'signed' }));
  const layPool = [51, 151].map((y) => cellGrid(layG, 3, 3, { x: 360, y, cell: 18, kind: 'signed' }));
  const layFeat = cellGrid(layG, 18, 1, { x: 424, y: 47, cell: 9, kind: 'signed' });
  const layOut = [label(layG, 446, 112, 'output', 'start'), label(layG, 446, 136, '', 'start', STRONG), label(layG, 446, 160, '', 'start', STRONG)];
  const layEx = label(layG, 16, 72, '');

  // ---- step 3: pooling, the example and the example shifted right by one pixel.
  const poolG = el('g', {}, svg);
  const poolRows = [40, 170].map((y, r) => {
    const t = label(poolG, 16, y - 10, '');
    if (r === 0) {
      label(poolG, 120, y - 10, 'after ReLU');
      label(poolG, 310, y - 10, 'pooled');
    }
    const img = cellGrid(poolG, SIDE, SIDE, { x: 16, y, cell: 10 });
    const maps = [120, 204].map((x) => cellGrid(poolG, 6, 6, { x, y: y + 4, cell: 12, kind: 'signed' }));
    const pools = [310, 380].map((x) => cellGrid(poolG, 3, 3, { x, y: y + 10, cell: 20, kind: 'signed' }));
    // 2 × 2 pooling windows, and a ring on each window's largest cell.
    const rings = [];
    [120, 204].forEach((x) => {
      for (const k of [2, 4]) {
        el('line', { x1: x + k * 12, y1: y + 4, x2: x + k * 12, y2: y + 76, stroke: MUTED, 'stroke-width': 1.25 }, poolG);
        el('line', { x1: x, y1: y + 4 + k * 12, x2: x + 72, y2: y + 4 + k * 12, stroke: MUTED, 'stroke-width': 1.25 }, poolG);
      }
      const rs = [];
      for (let q = 0; q < 9; q++) rs.push(el('rect', { width: 8, height: 8, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, poolG));
      rings.push(rs);
    });
    return { t, img, maps, pools, rings, y };
  });

  // ---- step 4: the live run. Both filters with their weights, and the example's maps.
  const liveG = el('g', {}, svg);
  label(liveG, 16, 28, 'filter 1');
  label(liveG, 184, 28, 'filter 2');
  const liveW = [16, 184].map((x) => cellGrid(liveG, 3, 3, { x, y: 40, cell: 52, kind: 'signed', numbers: true }));
  const liveEx = label(liveG, 352, 28, '');
  const liveImg = cellGrid(liveG, SIDE, SIDE, { x: 352, y: 40, cell: 8 });
  label(liveG, 430, 136, 'map 1');
  label(liveG, 514, 136, 'map 2');
  const liveMaps = [430, 514].map((x) => cellGrid(liveG, 6, 6, { x, y: 40, cell: 12, kind: 'signed' }));
  const liveZero = [430, 514].map((x) => label(liveG, x + 36, 81, 'all 0', 'middle'));
  label(liveG, 430, 158, 'after ReLU');

  // ---- step 5: the trained filters on a cross and a square.
  const endG = el('g', {}, svg);
  [250, 350].forEach((x, c) => cellGrid(endG, SIDE, SIDE, { x, y: 24, cell: 8 }).update(TR.X[c === 0 ? CROSS : SQUARE]));
  markerLabel(endG, 250, 18, 0);
  markerLabel(endG, 350, 18, 1);
  const endRows = [100, 178].map((y, k) => {
    label(endG, 140, y + 38, `filter ${k + 1}`, 'end');
    const W = cellGrid(endG, 3, 3, { x: 150, y, cell: 22, kind: 'signed' });
    const maps = [250, 350].map((x) => cellGrid(endG, 6, 6, { x, y, cell: 11, kind: 'signed' }));
    const zero = [250, 350].map((x) => label(endG, x + 33, y + 38, 'all 0', 'middle'));
    return { W, maps, zero };
  });
  const endP = [250, 350].map((x) => label(endG, x, 262, '', 'start', STRONG));
  label(endG, 140, 262, 'p(square)', 'end');
  const endWait = label(endG, 300, 160, 'computing…', 'middle');

  // ---- step 6: weights and test accuracy, CNN and dense.
  const cmpG = el('g', {}, svg);
  const BX = 120;
  const BW = 400;
  label(cmpG, 20, 34, 'weights and biases');
  label(cmpG, 20, 160, 'accuracy on the 400 test images');
  const bar = (y, fill, dash) => el('rect', { x: BX, y, height: 22, width: 0, fill, 'fill-opacity': 0.55, stroke: fill, 'stroke-width': 1.5, 'stroke-dasharray': dash }, cmpG);
  const cmp = [
    { name: 'CNN', y: 48, fill: CYAN, dash: 'none' },
    { name: 'dense', y: 82, fill: VIOLET, dash: '4 3' },
    { name: 'CNN', y: 174, fill: CYAN, dash: 'none' },
    { name: 'dense', y: 208, fill: VIOLET, dash: '4 3' },
  ].map((b) => {
    label(cmpG, BX - 10, b.y + 16, b.name, 'end', STRONG);
    return { rect: bar(b.y, b.fill, b.dash), text: label(cmpG, BX + 6, b.y + 16, '', 'start', STRONG), y: b.y };
  });
  el('line', { x1: BX, y1: 170, x2: BX, y2: 234, stroke: MUTED }, cmpG);
  el('line', { x1: BX, y1: 44, x2: BX, y2: 108, stroke: MUTED }, cmpG);

  // ---- read-out rows, under every picture.
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED));
  rowsT.forEach((t) => t.setAttribute('font-size', 13));

  // ---- controls: the example picker and Train / Pause / Reset.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const exGroup = make('div');
  exGroup.setAttribute('role', 'group');
  exGroup.setAttribute('aria-label', 'example image');
  const exLab = make('label');
  const exSel = make('select');
  for (let k = 0; k < SHOWN; k++) exSel.append(make('option', { value: String(k), textContent: `${k + 1}, ${cls(k)}` }));
  exLab.append(make('span', { textContent: 'example' }), exSel);
  exGroup.append(exLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the network');
  const trainTag = make('span', { textContent: 'network' });
  trainTag.setAttribute('aria-hidden', 'true');
  Object.assign(trainTag.style, { color: MUTED, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', marginRight: '6px' });
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  trainGroup.append(trainTag);
  controls.append(exGroup, trainGroup);

  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };

  function info() {
    if (step === 0) return gridInfo();
    if (step === 1) return layerInfo(example);
    if (step === 2) return poolInfo(example);
    if (step === 3) return liveInfo(getLive(), example, loop.running());
    if (step === 4) return filtersInfo(readyRuns());
    return compareInfo(readyRuns());
  }

  const exName = (k) => `example ${k + 1}, a ${cls(k)}`;

  function render() {
    show(gridG, step === 0);
    show(layG, step === 1);
    show(poolG, step === 2);
    show(liveG, step === 3);
    show(endG, step === 4);
    show(cmpG, step === 5);
    const s = info();
    const img = TR.X[example];

    if (step === 1) {
      const fw = cnnForward(start, img);
      layEx.textContent = `example ${example + 1}`;
      layImg.update(img);
      start.filters.forEach((W, k) => {
        layW[k].update(W);
        layConv[k].update(fw.maps[k]);
        layRelu[k].update(fw.relu[k], Math.max(...fw.maps[k].flat().map(Math.abs)));
        layPool[k].update(fw.pooled[k], Math.max(...fw.maps[k].flat().map(Math.abs)));
      });
      layFeat.update(fw.feat.map((v) => [v]));
      layOut[1].textContent = `z = ${num(fw.z, 2)}`;
      layOut[2].textContent = `p(square) = ${num(sigmoid(fw.z), 2)}`;
    }

    if (step === 2) {
      const pair = [s.a, s.b];
      const imgs = [img, shiftRight(img, 1)];
      poolRows.forEach((row, r) => {
        row.t.textContent = r === 0 ? `example ${example + 1}` : 'shifted right 1';
        row.img.update(imgs[r]);
        const fw = pair[r];
        for (let k = 0; k < 2; k++) {
          const scale = Math.max(...s.a.relu[k].flat(), ...s.b.relu[k].flat());
          row.maps[k].update(fw.relu[k], scale);
          row.pools[k].update(fw.pooled[k], scale);
          const x0 = k === 0 ? 120 : 204;
          fw.args[k].flat().forEach(([i, j], q) => {
            const ring = row.rings[k][q];
            ring.setAttribute('x', x0 + j * 12 + 2);
            ring.setAttribute('y', row.y + 4 + i * 12 + 2);
            show(ring, fw.relu[k][i][j] > 0);
          });
        }
      });
    }

    if (step === 3) {
      const net = getLive().trainer.state.net;
      const fw = cnnForward(net, img);
      net.filters.forEach((W, k) => liveW[k].update(W));
      liveEx.textContent = `example ${example + 1}`;
      liveImg.update(img);
      const scale = Math.max(1e-9, ...fw.relu.flat(2));
      fw.relu.forEach((m, k) => {
        liveMaps[k].update(m, scale);
        show(liveZero[k], Math.max(...m.flat()) === 0);
      });
    }

    if (step === 4) {
      show(endWait, !s);
      endRows.forEach((r) => { show(r.W.g, !!s); r.maps.forEach((m) => show(m.g, !!s)); r.zero.forEach((z) => show(z, false)); });
      endP.forEach((t) => show(t, !!s));
      if (s) {
        const net = RUNS.get('cnn').net;
        const scale = Math.max(1e-9, ...s.square.relu.flat(2), ...s.cross.relu.flat(2));
        endRows.forEach((r, k) => {
          r.W.update(net.filters[k]);
          r.maps[0].update(s.cross.relu[k], scale);
          r.maps[1].update(s.square.relu[k], scale);
          show(r.zero[0], s.crossMax[k] === 0);
          show(r.zero[1], s.squareMax[k] === 0);
        });
        endP[0].textContent = num(s.crossP, 3);
        endP[1].textContent = num(s.squareP, 3);
      }
    }

    if (step === 5) {
      const vals = s ? [s.cnnParams, s.denseParams, s.cnn.testAcc.at(-1), s.dense.testAcc] : [0, 0, 0, 0];
      cmp.forEach((b, q) => {
        const frac = q < 2 ? vals[q] / denseParamCount(SIDE * SIDE, DENSE_HIDDEN) : vals[q];
        const w = P(BW * frac);
        b.rect.setAttribute('width', w);
        b.text.setAttribute('x', P(BX + w + 8));
        b.text.textContent = s ? (q < 2 ? count(vals[q]) : pct(vals[q])) : '';
      });
    }

    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Twelve 8 by 8 training images in two rows: crosses (class 0, circle marker) and hollow squares (class 1, triangle marker), alternating. ${rest}.`;
    if (step === 1) return `${exName(example)}, passed through the starting network: two 3 by 3 filters give two 6 by 6 maps, ReLU sets negatives to 0, 2 by 2 max-pooling gives two 3 by 3 maps, their 18 values feed one output. ${rest}.`;
    if (step === 2) return `${exName(example)} and the same image shifted right by one pixel, each with its two maps after ReLU and the two pooled maps. ${rest}.`;
    if (step === 3) {
      const ws = getLive().trainer.state.net.filters.map((W, k) => `filter ${k + 1}, row by row: ${W.map((row) => row.map((w) => num(w, 2)).join(', ')).join('; ')}`).join('. ');
      return `The two 3 by 3 filters with their current weights, and the maps they give on ${exName(example)}. ${ws}. ${rest}.`;
    }
    if (step === 4) return `The trained filters, and their maps after ReLU on a cross and on a square. ${rest}.`;
    return `Bars comparing the number of weights and the test accuracy of the CNN and the dense network. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  // ---- step 4's live run.
  const loop = createTrainLoop({
    // At most two steps a frame, so the filters can be watched changing (about five seconds).
    step: (n) => getLive().advance(Math.min(2, n)),
    render,
    done: () => getLive().done(),
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done()) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => getLive().reset(),
    reduced,
    runToEnd: () => getLive().advance(ITERS),
    render: () => { render(); announce(spoken()); },
  });
  // Pausing does not call render, so redraw the read-out after any Train/Pause/Reset click.
  trainGroup.addEventListener('click', () => { if (step === 3) { render(); announce(spoken()); } });

  // ---- steps 5 and 6: the end of the CNN run and the dense run, computed when
  // first needed, in slices of about 6 ms per animation frame.
  let job = null;
  const cancelRuns = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const computeRuns = () => {
    if (job !== null || RUN_KINDS.every(runReady)) return;
    const tick = () => {
      job = null;
      if (step < 4) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const k = RUN_KINDS.find((x) => !runReady(x));
        if (k === undefined) break;
        runWork(k, 2);
      }
      if (RUN_KINDS.every(runReady)) { render(); announce(spoken()); } else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

  exSel.addEventListener('change', () => {
    example = Number(exSel.value) || 0;
    render();
    announce(spoken());
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
      buttons.stop();
      step = i;
      if (i >= 4) computeRuns(); else cancelRuns();
      exGroup.style.display = i >= 1 && i <= 3 ? '' : 'none';
      trainGroup.style.display = i === 3 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 3);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
