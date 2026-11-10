// Lesson L4.9, "Recurrent networks": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-rnns.test.mjs); the
// DOM is only touched inside createStage. Every forward pass is rnnForward from
// rnn.js, every gradient is bptt, every training run is rnnTrainer below, and
// every number in the page's text comes from pageValues or storyValues.

import { mulberry32 } from '../assets/maths.js';
import { sigmoid } from '../assets/classify.js';
import { echoData, rnnInit, rnnForward, bptt, rnnFlatten, rnnUnflatten } from '../assets/rnn.js';
import { optInit, optStep } from '../assets/nn.js';
import { el, pixelFill, tween } from '../assets/plot.js';
import { createTrainLoop, trainButtons } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const SEED_TRAIN = 211;
export const SEED_TEST = 212;
export const SEED_INIT = 213;
export const SEED_SHUFFLE = 214;
export const N_TRAIN = 200;
export const N_TEST = 200;
export const T = 12;
export const NH = 8;
export const ETA = 0.02;
export const BATCH = 10;
// The budget: every run on the page is 10 epochs, 200 batches of 10 sequences.
export const EPOCHS = 10;
export const ITERS = (EPOCHS * N_TRAIN) / BATCH;
export const K_LIVE = 2;
export const K_CHOICES = [1, 2, 4, 8];
// The test stream drawn in steps 2 and 3, and how many lines step 5 draws.
export const EXAMPLE = 0;
export const LAG_SHOWN = 12;
export const SEQ_SHOWN = 3;

const DATA = new Map();
const cached = (key, make) => {
  if (!DATA.has(key)) DATA.set(key, make());
  return DATA.get(key);
};
export const trainSet = (k) => cached(`tr${k}`, () => echoData({ n: N_TRAIN, T, k, seed: SEED_TRAIN }));
export const testSet = (k) => cached(`te${k}`, () => echoData({ n: N_TEST, T, k, seed: SEED_TEST }));
export const startNet = () => rnnInit({ nh: NH, seed: SEED_INIT });
export const maskedCount = (k) => N_TEST * (T - k);

// ---- predictions, accuracy and loss --------------------------------------------

// p̂ = σ(z) > 0.5 exactly when z > 0; p̂ = 0.5 predicts 0.
export const predictBit = (z) => (z > 0 ? 1 : 0);

// Masked accuracy: over every masked step (t ≥ k, zero-based) of every sequence,
// how many predicted bits equal the target.
export function maskedAccuracy(net, D) {
  let ok = 0;
  let n = 0;
  for (let i = 0; i < D.seqs.length; i++) {
    const { zs } = rnnForward(net, D.seqs[i]);
    for (let t = 0; t < zs.length; t++) {
      if (!D.mask[i][t]) continue;
      n++;
      if (predictBit(zs[t]) === D.targets[i][t]) ok++;
    }
  }
  return { ok, n };
}

// The mean over sequences of each sequence's mean masked cross-entropy.
export function rnnMeanLoss(net, D) {
  let s = 0;
  for (let i = 0; i < D.seqs.length; i++) s += bptt(net, D.seqs[i], D.targets[i], D.mask[i]).loss;
  return s / D.seqs.length;
}

// The mean over idx of each sequence's gradient (flat, rnnFlatten order), and the mean loss.
export function batchGradRnn(net, D, idx) {
  let grad = null;
  let loss = 0;
  for (const q of idx) {
    const b = bptt(net, D.seqs[q], D.targets[q], D.mask[q]);
    const g = rnnFlatten(b.grads);
    if (!grad) grad = g;
    else for (let j = 0; j < g.length; j++) grad[j] += g[j];
    loss += b.loss;
  }
  const m = idx.length;
  return { grad: grad.map((v) => v / m), loss: loss / m };
}

// ---- the trainer -----------------------------------------------------------------

const DIVERGED = 1e6;

// Mini-batch training with the same rules as nn.js's trainer: a Fisher–Yates
// shuffle of the sequences from mulberry32(seed) at the start of each epoch,
// batches taken in order, a last partial batch kept, epochLoss[0] the mean loss
// before any step. step(k) updates api.state in place; reset() gives a fresh state.
export function rnnTrainer({ net, data, batch, kind = 'adam', hyper = {}, seed }) {
  const start = rnnUnflatten(net, rnnFlatten(net));
  const n = data.seqs.length;
  let rng, order, cursor, opt, params;

  const fresh = () => {
    rng = mulberry32(seed);
    order = null;
    cursor = 0;
    params = rnnFlatten(start);
    opt = optInit(kind, params.length);
    const first = rnnMeanLoss(start, data);
    return { net: rnnUnflatten(start, params), epoch: 0, iter: 0, batchLoss: [], epochLoss: [first], diverged: !Number.isFinite(first) };
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
    const g = batchGradRnn(s.net, data, idx);
    if (!Number.isFinite(g.loss) || Math.abs(g.loss) > DIVERGED) { s.diverged = true; return; }
    const r = optStep(kind, hyper, opt, params, g.grad);
    if (!r.params.every(Number.isFinite)) { s.diverged = true; return; }
    params = r.params;
    opt = r.state;
    s.net = rnnUnflatten(s.net, params);
    s.batchLoss.push(g.loss);
    s.iter++;
    cursor += batch;
    if (cursor >= n) {
      cursor = 0;
      s.epoch++;
      const e = rnnMeanLoss(s.net, data);
      s.epochLoss.push(e);
      if (!Number.isFinite(e) || Math.abs(e) > DIVERGED) s.diverged = true;
    }
  };

  api.step = (k = 1) => {
    for (let i = 0; i < k && !api.state.diverged; i++) one();
    return api.state;
  };
  api.reset = () => { api.state = fresh(); return api.state; };
  return api;
}

export const makeTrainer = (k) => rnnTrainer({ net: startNet(), data: trainSet(k), batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });

// Step 3's live run at k = 2: the trainer plus the masked test count after every
// batch. advance(k) takes at most k batches and never runs past the budget.
export function makeLive() {
  const te = testSet(K_LIVE);
  const tr = makeTrainer(K_LIVE);
  const live = { trainer: tr, n: maskedCount(K_LIVE), testOk: [maskedAccuracy(tr.state.net, te).ok] };
  live.advance = (k) => {
    for (let i = 0; i < k && tr.state.iter < ITERS && !tr.state.diverged; i++) {
      tr.step(1);
      live.testOk.push(maskedAccuracy(tr.state.net, te).ok);
    }
  };
  live.reset = () => {
    tr.reset();
    live.testOk = [live.testOk[0]];
  };
  live.done = () => tr.state.iter >= ITERS || tr.state.diverged;
  return live;
}

// ---- step 4: one run per k, computed on demand ------------------------------------

const RUNS = new Map();
const JOBS = new Map();

// k more batches of this k's run; true once its result is ready. The page calls
// this in small slices; the slicing does not change the result.
export function runWork(k, steps) {
  if (RUNS.has(k)) return true;
  if (!JOBS.has(k)) {
    const t = makeTrainer(k);
    JOBS.set(k, { t, testOk: [maskedAccuracy(t.state.net, testSet(k)).ok] });
  }
  const job = JOBS.get(k);
  const { t } = job;
  for (let i = 0; i < steps && t.state.iter < ITERS && !t.state.diverged; i++) {
    const e = t.state.epoch;
    t.step(1);
    if (t.state.epoch > e) job.testOk.push(maskedAccuracy(t.state.net, testSet(k)).ok);
  }
  if (t.state.iter < ITERS && !t.state.diverged) return false;
  const s = t.state;
  RUNS.set(k, { k, losses: s.epochLoss.slice(), testOk: job.testOk.slice(), n: maskedCount(k), net: s.net, trainAcc: maskedAccuracy(s.net, trainSet(k)), diverged: s.diverged });
  JOBS.delete(k);
  return true;
}
export const runReady = (k) => RUNS.has(k);
export function runResult(k) {
  runWork(k, Infinity);
  return RUNS.get(k);
}
const readyRuns = () => (K_CHOICES.every(runReady) ? Object.fromEntries(K_CHOICES.map((k) => [k, RUNS.get(k)])) : null);

// ---- step 5: gradient size by distance back in time ------------------------------

// For each test sequence, bptt's lagNorms: ‖∂ℒ_T/∂h_{T−j}‖ for j = 0..T−1, where
// ℒ_T is the last step's cross-entropy alone. lines: the first LAG_SHOWN
// sequences; typical: the geometric mean over all of them at each j.
export function lagInfo(net, D) {
  const all = D.seqs.map((xs, i) => bptt(net, xs, D.targets[i], D.mask[i]).lagNorms);
  const typical = [];
  for (let j = 0; j < T; j++) {
    let s = 0;
    for (const row of all) s += Math.log(row[j]);
    typical.push(Math.exp(s / all.length));
  }
  return { lines: all.slice(0, LAG_SHOWN), typical };
}

let LAG = null;
export const lagReady = () => LAG !== null;
// The plotted network: the starting network, on the k = 2 test targets.
export function lagPlot() {
  if (!LAG) LAG = lagInfo(startNet(), testSet(K_LIVE));
  return LAG;
}

// Whole decades of the log axis that hold every plotted value.
export function lagRange(info) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of [...info.typical, ...info.lines.flat()]) {
    const e = Math.log10(v);
    lo = Math.min(lo, e);
    hi = Math.max(hi, e);
  }
  return [Math.floor(lo), Math.ceil(hi)];
}

// The largest factor by which W stretches a vector (its spectral norm), by power
// iteration on WᵀW from a fixed start; v is the unit vector it stretches most.
export function spectralNorm(W) {
  const n = W[0].length;
  let v = Array.from({ length: n }, (_, i) => 1 + i / n);
  const mul = (x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));
  const mulT = (u) => Array.from({ length: n }, (_, j) => W.reduce((s, row, i) => s + row[j] * u[i], 0));
  for (let it = 0; it < 2000; it++) {
    const w = mulT(mul(v));
    const len = Math.hypot(...w);
    if (!(len > 0)) break;
    v = w.map((x) => x / len);
  }
  return { norm: Math.hypot(...mul(v)), v };
}

// ---- small helpers ---------------------------------------------------------------

export const firstAt = (ok, n, level) => ok.findIndex((v) => v / n >= level);
// The first index from which every later entry is n (all right), or −1.
export function allRightFrom(ok, n) {
  let i = ok.length;
  while (i > 0 && ok[i - 1] === n) i--;
  return i < ok.length ? i : -1;
}

// ---- formatting ------------------------------------------------------------------

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
// Two significant figures, plain decimals (for values from 1e-6 up).
export function sig2(v) {
  const d = Math.max(0, 1 - Math.floor(Math.log10(Math.abs(v))));
  return num(Number(v.toPrecision(2)), d);
}
// A count as a percentage to one decimal, ".0" dropped; integer arithmetic, so exact.
export function pctCount(ok, n) {
  const t = Math.round((ok * 1000) / n);
  return `${t % 10 === 0 ? String(t / 10) : (t / 10).toFixed(1)}%`;
}
export const lossText = (v) => (v >= 0.01 ? num(v, 3) : sig2(v));
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

// ---- read-outs ---------------------------------------------------------------------

export function seqInfo(k) {
  const D = trainSet(k);
  return { k, seqs: D.seqs.slice(0, SEQ_SHOWN), targets: D.targets.slice(0, SEQ_SHOWN), mask: D.mask.slice(0, SEQ_SHOWN) };
}

// The starting network reading test stream EXAMPLE, up to step t (0 = nothing read yet).
export function loopInfo(t) {
  const D = testSet(K_LIVE);
  const xs = D.seqs[EXAMPLE];
  const { hs, zs } = rnnForward(startNet(), xs);
  return { t, xs, hs, zs, ps: zs.map(sigmoid), targets: D.targets[EXAMPLE], mask: D.mask[EXAMPLE] };
}

export function liveInfo(live, running) {
  const s = live.trainer.state;
  return { iter: s.iter, epoch: s.epoch, ok: live.testOk.at(-1), n: live.n, loss: s.epochLoss.at(-1), running, diverged: s.diverged, done: live.done() };
}

export function budgetInfo(k, runs) {
  if (!runs) return null;
  const r = runs[k];
  return { k, ok: r.testOk[EPOCHS], n: r.n, trainAcc: r.trainAcc, loss: r.losses[EPOCHS], testOk: r.testOk, runs };
}

export function readout(step, s) {
  if (step === 0) return [`${SEQ_SHOWN} of the ${N_TRAIN} training streams`, `task: say the bit from k = ${s.k} steps ago`, `no target for the first ${s.k} steps (–)`];
  if (step === 1) {
    if (s.t === 0) return ['before the first step: h₀ = 0', 'the same cell reads one bit per step', 'starting network, not trained'];
    const tgt = s.mask[s.t - 1] ? String(s.targets[s.t - 1]) : '–';
    return [`test stream ${EXAMPLE + 1}, step t = ${s.t} of ${T}`, `hₜ: ${NH} numbers, from xₜ and hₜ₋₁`, `starting network: p̂ = ${num(s.ps[s.t - 1], 2)} · target ${tgt}`];
  }
  if (step === 2) {
    const status = s.diverged ? 'diverged' : s.running && !s.done ? 'training…' : s.iter === 0 ? 'press Train' : !s.done ? 'not finished: press Train' : 'finished';
    return [`batch ${s.iter} of ${ITERS} · ${status}`, `masked test accuracy: ${pctCount(s.ok, s.n)}`, `training loss ℒ (epoch ${s.epoch}): ${lossText(s.loss)}`];
  }
  if (step === 3) {
    if (!s) return ['computing the four runs…'];
    return [`k = ${s.k} · ${EPOCHS} epochs from the same start`, `test: ${pctCount(s.ok, s.n)} (${count(s.ok)} of ${count(s.n)} masked steps)`, `train: ${pctCount(s.trainAcc.ok, s.trainAcc.n)} · training loss ℒ: ${lossText(s.loss)}`];
  }
  if (step === 4) {
    const t = s.typical;
    return ['starting network · typical gradient size', `at the last step: ${sig2(t[0])}`, `${T - 1} steps back: ${sig2(t[T - 1])} (${Math.round(t[0] / t[T - 1])} times smaller)`];
  }
  if (step === 5) return ['a sketch of the idea, not a trained network', 'plain RNN: h goes through Wₕ and tanh', 'gated cell: c can pass on unchanged'];
  return [];
}

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  return {
    'n-params': String(rnnFlatten(startNet()).length),
    'masked-k2': count(maskedCount(2)),
    'masked-k8': count(maskedCount(8)),
  };
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const live = makeLive();
  while (!live.done()) live.advance(ITERS);
  const ok = live.testOk;
  const n = live.n;
  const L = live.trainer.state.epochLoss;
  v['start-acc'] = pctCount(ok[0], n);
  const f = firstAt(ok, n, 0.95);
  v['first-95'] = String(f);
  v['first-95-acc'] = pctCount(ok[f], n);
  v['before-95-acc'] = pctCount(ok[f - 1], n);
  v['all-right'] = String(allRightFrom(ok, n));
  v['end-acc'] = pctCount(ok.at(-1), n);
  v['start-loss'] = lossText(L[0]);
  v['end-loss'] = lossText(L[EPOCHS]);
  const runs = Object.fromEntries(K_CHOICES.map((k) => [k, runResult(k)]));
  for (const k of K_CHOICES) v[`acc-k${k}`] = pctCount(runs[k].testOk[EPOCHS], runs[k].n);
  v['k4-perfect'] = String(runs[4].testOk.findIndex((x) => x === runs[4].n));
  v['k8-ok'] = String(runs[8].testOk[EPOCHS]);
  v['k8-n'] = String(runs[8].n);
  v['k8-train'] = pctCount(runs[8].trainAcc.ok, runs[8].trainAcc.n);
  v['k8-start'] = pctCount(runs[8].testOk[0], runs[8].n);
  v['k8-best'] = pctCount(Math.max(...runs[8].testOk), runs[8].n);
  const t = lagPlot().typical;
  v['lag0'] = sig2(t[0]);
  v['lag11'] = sig2(t[T - 1]);
  v['lag-ratio'] = String(Math.round(t[0] / t[T - 1]));
  v['lag2-ratio'] = String(Math.round(t[0] / t[2]));
  v['lag8-ratio'] = String(Math.round(t[0] / t[8]));
  const t8 = lagInfo(runs[8].net, testSet(8)).typical;
  v['k8-lag0'] = sig2(t8[0]);
  v['k8-lag11'] = sig2(t8[T - 1]);
  v['wh-norm'] = num(spectralNorm(startNet().Wh).norm, 2);
  v['loop-p-last'] = num(loopInfo(T).ps[T - 1], 2);
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
const FONT = "'JetBrains Mono', monospace";

// Columns of the unrolled pictures (steps 1 to 3): t = 1..12.
const CX0 = 96;
const CW = 41;
const colX = (t) => CX0 + (t - 1) * CW;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let loopT = T;
  let kSel = 3;
  let live = null;
  let token = 0;

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  const defs = el('defs', {}, svg);
  const marker = el('marker', { id: 'rnn-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, marker);
  const arrow = (parent, x1, y1, x2, y2, extra = {}) => el('line', { x1, y1, x2, y2, stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#rnn-arrow)', ...extra }, parent);

  // A bit in a box: "1" on a cyan tint, "0" on none, "–" when there is no target (null),
  // empty when not read yet (undefined).
  const bitCell = (parent, x, y, w = 28, h = 24) => {
    const r = el('rect', { x, y, width: w, height: h, rx: 3, fill: CYAN, 'fill-opacity': 0, stroke: LINE }, parent);
    const t = label(parent, x + w / 2, y + h / 2 + 5, '', 'middle', STRONG, 13);
    return {
      set(v) {
        r.setAttribute('fill-opacity', v === 1 ? 0.3 : 0);
        t.textContent = v === undefined ? '' : v === null ? '–' : String(v);
        t.style.fill = v === null ? MUTED : STRONG;
      },
    };
  };

  // A column of NH cells for a hidden state, tinted by sign and size (tanh is in (−1, 1)).
  const hCol = (parent0, x, y, cell = 11, w = 18) => {
    const parent = el('g', { class: 'pixels' }, parent0);
    const rects = [];
    const marks = [];
    for (let i = 0; i < NH; i++) {
      rects.push(el('rect', { x, y: y + i * cell, width: w, height: cell, opacity: 0 }, parent));
      marks.push(el('text', { class: 'px-sign', x: x + w / 2, y: y + (i + 0.5) * cell, 'text-anchor': 'middle', text: '−', visibility: 'hidden' }, parent));
    }
    const frame = el('rect', { class: 'px-frame', x, y, width: w, height: NH * cell, fill: 'none', stroke: LINE }, parent);
    return {
      frame,
      set(h) {
        rects.forEach((r, i) => {
          const v = h && Number.isFinite(h[i]) ? h[i] : 0;
          const f = pixelFill(v, { signed: 1 });
          r.setAttribute('class', f.cls);
          r.setAttribute('opacity', h ? P(f.opacity) : 0);
          marks[i].setAttribute('visibility', h && v < 0 && num(v, 2) !== '0.00' && f.opacity > 0.3 ? 'visible' : 'hidden');
        });
      },
    };
  };

  // ---- step 1: three training streams and their k = 2 targets.
  const seqG = el('g', {}, svg);
  for (let t = 1; t <= T; t++) label(seqG, colX(t) + 14, 18, String(t), 'middle');
  label(seqG, 16, 18, 't');
  const seqRows = [0, 1, 2].map((r) => {
    const y = 42 + r * 82;
    label(seqG, 16, y - 4, `stream ${r + 1}`, 'start', STRONG);
    label(seqG, 16, y + 17, 'x');
    label(seqG, 16, y + 45, 'target');
    const xs = [];
    const ys = [];
    for (let t = 1; t <= T; t++) {
      xs.push(bitCell(seqG, colX(t), y));
      ys.push(bitCell(seqG, colX(t), y + 28));
    }
    return { xs, ys, y };
  });
  // One link per stream: the target at step t is the input at t − 2.
  const LINK_T = [5, 8, 11];
  LINK_T.forEach((t, r) => {
    const y = seqRows[r].y;
    for (const [cx, cy] of [[colX(t - K_LIVE), y], [colX(t), y + 28]]) el('rect', { x: cx - 2, y: cy - 2, width: 32, height: 28, rx: 4, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, seqG);
    arrow(seqG, colX(t - K_LIVE) + 14, y + 24, colX(t) + 12, y + 28, { stroke: CYAN, 'stroke-width': 1.5 });
  });

  // ---- step 2: the loop, unrolled over twelve steps, with the starting network.
  const loopG = el('g', {}, svg);
  for (let t = 1; t <= T; t++) label(loopG, colX(t) + 14, 18, String(t), 'middle');
  label(loopG, 16, 18, 't');
  label(loopG, 16, 44, 'xₜ', 'start', STRONG, 13);
  label(loopG, 16, 120, 'hₜ', 'start', STRONG, 13);
  label(loopG, 16, 186, 'p̂ₜ', 'start', STRONG, 13);
  label(loopG, 16, 214, 'target', 'start', STRONG, 13);
  const H_Y = 70;
  const h0 = hCol(loopG, 60, H_Y);
  h0.set(new Array(NH).fill(0));
  label(loopG, 69, H_Y + NH * 11 + 14, 'h₀', 'middle');
  const loopCols = [];
  for (let t = 1; t <= T; t++) {
    const x = colX(t);
    const xb = bitCell(loopG, x, 26);
    arrow(loopG, x + 14, 52, x + 14, H_Y - 2);
    arrow(loopG, t === 1 ? 78 : x - 18, H_Y + 44, x + 3, H_Y + 44);
    const h = hCol(loopG, x + 5, H_Y);
    const p = label(loopG, x + 14, 186, '', 'middle', STRONG, 12);
    const yb = bitCell(loopG, x, 196);
    loopCols.push({ xb, h, p, yb });
  }
  const loopNow = el('rect', { y: 22, width: 36, height: 204, rx: 4, fill: 'none', stroke: CYAN, 'stroke-width': 1.5 }, loopG);
  label(loopG, 16, 250, 'the same weights are used at every step', 'start', MUTED, 12);

  // ---- step 3: the live run. The example stream with the current predictions, and the accuracy curve.
  const liveG = el('g', {}, svg);
  label(liveG, 16, 36, 'xₜ', 'start', STRONG, 13);
  label(liveG, 16, 64, 'target', 'start', STRONG, 13);
  label(liveG, 16, 90, 'p̂ₜ', 'start', STRONG, 13);
  const liveCols = [];
  for (let t = 1; t <= T; t++) {
    const x = colX(t);
    const xb = bitCell(liveG, x, 18, 28, 22);
    const yb = bitCell(liveG, x, 46, 28, 22);
    const p = label(liveG, x + 14, 90, '', 'middle', STRONG, 12);
    const m = label(liveG, x + 14, 108, '', 'middle', STRONG, 13);
    liveCols.push({ xb, yb, p, m });
  }
  const AX0 = CX0;
  const AX1 = 584;
  const AY0 = 252; // 40%
  const AY1 = 140; // 100%
  const ax = (i) => AX0 + (i / ITERS) * (AX1 - AX0);
  const ay = (a) => AY0 + ((a - 0.4) / 0.6) * (AY1 - AY0);
  for (const [a, txt] of [[1, '100%'], [0.5, '50%']]) {
    el('line', { x1: AX0, x2: AX1, y1: P(ay(a)), y2: P(ay(a)), stroke: LINE, 'stroke-dasharray': a === 0.5 ? '5 4' : '2 4' }, liveG);
    label(liveG, AX0 - 8, P(ay(a)) + 4, txt, 'end');
  }
  label(liveG, AX1, P(ay(0.5)) - 6, 'chance', 'end');
  el('line', { x1: AX0, x2: AX0, y1: AY0, y2: AY1 - 6, stroke: LINE }, liveG);
  el('line', { x1: AX0, x2: AX1, y1: AY0, y2: AY0, stroke: LINE }, liveG);
  for (const i of [0, 100, 200]) label(liveG, P(ax(i)), AY0 + 16, String(i), 'middle');
  label(liveG, 16, AY0 + 16, 'batch');
  label(liveG, AX0, AY1 - 14, 'masked test accuracy, by batch');
  const livePath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, liveG);
  const liveDot = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, liveG);

  // ---- step 4: one run per k. Bars of the final masked test accuracy, and the chosen k by epoch.
  const budG = el('g', {}, svg);
  label(budG, 16, 26, `masked test accuracy after ${EPOCHS} epochs`);
  const BX0 = 70;
  const BW = 220;
  const bx = (a) => BX0 + a * BW;
  const budRows = K_CHOICES.map((k, q) => {
    const y = 48 + q * 46;
    const name = label(budG, BX0 - 10, y + 20, `k = ${k}`, 'end', STRONG, 13);
    const rect = el('rect', { x: BX0, y, height: 28, width: 0, fill: CYAN, 'fill-opacity': 0.35, stroke: CYAN, 'stroke-width': 1 }, budG);
    const val = label(budG, BX0 + 6, y + 19, '', 'start', STRONG, 12);
    return { name, rect, val, y };
  });
  el('line', { x1: bx(0.5), x2: bx(0.5), y1: 40, y2: 230, stroke: MUTED, 'stroke-dasharray': '5 4' }, budG);
  label(budG, bx(0.5), 248, 'chance 50%', 'middle');
  el('line', { x1: BX0, x2: BX0, y1: 40, y2: 230, stroke: LINE }, budG);
  const CX0b = 430;
  const CX1b = 584;
  const CY0b = 220;
  const CY1b = 60;
  const cxb = (e) => CX0b + (e / EPOCHS) * (CX1b - CX0b);
  const cyb = (a) => CY0b + ((a - 0.4) / 0.6) * (CY1b - CY0b);
  const budTitle = label(budG, CX0b, 26, '', 'start', STRONG);
  for (const [a, txt] of [[1, '100%'], [0.5, '50%']]) {
    el('line', { x1: CX0b, x2: CX1b, y1: P(cyb(a)), y2: P(cyb(a)), stroke: LINE, 'stroke-dasharray': a === 0.5 ? '5 4' : '2 4' }, budG);
    label(budG, CX0b - 6, P(cyb(a)) + 4, txt, 'end');
  }
  el('line', { x1: CX0b, x2: CX0b, y1: CY0b, y2: CY1b - 6, stroke: LINE }, budG);
  el('line', { x1: CX0b, x2: CX1b, y1: CY0b, y2: CY0b, stroke: LINE }, budG);
  for (const e of [0, 5, 10]) label(budG, P(cxb(e)), CY0b + 16, String(e), 'middle');
  label(budG, (CX0b + CX1b) / 2, CY0b + 34, 'epoch', 'middle');
  const budPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2 }, budG);
  const budDots = Array.from({ length: EPOCHS + 1 }, () => el('circle', { r: 3, fill: CYAN }, budG));
  const budWait = label(budG, 300, 150, 'computing…', 'middle');

  // ---- step 5: gradient size by steps back, log scale, the starting network.
  const lagG = el('g', {}, svg);
  const LX0 = 110;
  const LX1 = 560;
  const LY0 = 236;
  const LY1 = 40;
  const lx = (j) => LX0 + (j / (T - 1)) * (LX1 - LX0);
  const lagAxes = el('g', {}, lagG);
  const lagThin = Array.from({ length: LAG_SHOWN }, () => el('path', { d: '', fill: 'none', stroke: MUTED, 'stroke-width': 1, 'stroke-opacity': 0.55, 'stroke-dasharray': '3 3' }, lagG));
  const lagThick = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, lagG);
  const lagDots = Array.from({ length: T }, () => el('circle', { r: 3.5, fill: CYAN, stroke: BG, 'stroke-width': 1 }, lagG));
  for (let j = 0; j < T; j++) label(lagG, P(lx(j)), LY0 + 16, String(j), 'middle');
  label(lagG, (LX0 + LX1) / 2, LY0 + 34, 'steps back from the last step, T − t', 'middle');
  const ycx = 26;
  const ycy = (LY0 + LY1) / 2;
  const yl = label(lagG, ycx, ycy, 'gradient size, log scale', 'middle');
  yl.setAttribute('transform', `rotate(-90 ${ycx} ${ycy})`);
  el('line', { x1: 380, x2: 404, y1: 30, y2: 30, stroke: CYAN, 'stroke-width': 2.5 }, lagG);
  label(lagG, 410, 34, `typical (all ${N_TEST})`, 'start', STRONG);
  el('line', { x1: 380, x2: 404, y1: 50, y2: 50, stroke: MUTED, 'stroke-dasharray': '3 3' }, lagG);
  label(lagG, 410, 54, `one test stream (${LAG_SHOWN})`);
  let lagDrawn = false;
  const drawLag = () => {
    if (lagDrawn) return;
    lagDrawn = true;
    const info = lagPlot();
    const [lo, hi] = lagRange(info);
    const ly = (v) => LY0 + ((Math.log10(v) - lo) / (hi - lo)) * (LY1 - LY0);
    for (let e = lo; e <= hi; e++) {
      const y = P(LY0 + ((e - lo) / (hi - lo)) * (LY1 - LY0));
      el('line', { x1: LX0, x2: LX1, y1: y, y2: y, stroke: LINE, 'stroke-dasharray': '2 4' }, lagAxes);
      label(lagAxes, LX0 - 8, y + 4, e === 0 ? '1' : `10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`, 'end');
    }
    el('line', { x1: LX0, x2: LX0, y1: LY0, y2: LY1, stroke: LINE }, lagAxes);
    const d = (row) => row.map((v, j) => `${j ? 'L' : 'M'}${P(lx(j))},${P(ly(v))}`).join('');
    info.lines.forEach((row, i) => lagThin[i].setAttribute('d', d(row)));
    lagThick.setAttribute('d', d(info.typical));
    info.typical.forEach((v, j) => { lagDots[j].setAttribute('cx', P(lx(j))); lagDots[j].setAttribute('cy', P(ly(v))); });
  };

  // ---- step 6: a sketch of a plain cell and a gated cell.
  const gateG = el('g', {}, svg);
  const box = (x, y, w, h, text) => {
    el('rect', { x, y, width: w, height: h, rx: 5, fill: 'none', stroke: MUTED }, gateG);
    label(gateG, x + w / 2, y + h / 2 + 4, text, 'middle', STRONG);
  };
  label(gateG, 16, 30, 'plain RNN', 'start', STRONG, 13);
  label(gateG, 72, 74, 'hₜ₋₁', 'middle', STRONG);
  arrow(gateG, 94, 70, 128, 70);
  box(130, 52, 140, 36, 'Wₕ, then tanh');
  arrow(gateG, 270, 70, 302, 70);
  label(gateG, 318, 74, 'hₜ', 'middle', STRONG);
  arrow(gateG, 334, 70, 366, 70);
  box(368, 52, 140, 36, 'Wₕ, then tanh');
  arrow(gateG, 508, 70, 536, 70);
  label(gateG, 556, 74, 'hₜ₊₁', 'middle', STRONG);
  label(gateG, 130, 110, 'every step multiplies by Wₕ and squashes', 'start');

  label(gateG, 16, 144, 'gated cell (the LSTM idea)', 'start', STRONG, 13);
  label(gateG, 72, 186, 'cₜ₋₁', 'middle', STRONG);
  el('line', { x1: 96, x2: 536, y1: 182, y2: 182, stroke: CYAN, 'stroke-width': 2 }, gateG);
  el('path', { d: 'M530,177 L540,182 L530,187 Z', fill: CYAN }, gateG);
  label(gateG, 556, 186, 'cₜ', 'middle', STRONG);
  const node = (cx, text) => {
    el('circle', { cx, cy: 182, r: 13, fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, gateG);
    label(gateG, cx, 187, text, 'middle', STRONG, 14);
  };
  node(220, '×');
  node(400, '+');
  arrow(gateG, 220, 226, 220, 197);
  label(gateG, 220, 240, 'forget gate', 'middle');
  label(gateG, 220, 256, '(0 to 1)', 'middle');
  arrow(gateG, 400, 226, 400, 197);
  label(gateG, 400, 240, 'input gate × new', 'middle');
  label(gateG, 400, 256, '(gate 0 to 1)', 'middle');
  label(gateG, 300, 164, 'no Wₕ, no tanh on this path', 'middle');

  // ---- read-out rows, under every picture.
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- controls: the step slider (step 2), Train / Pause / Reset (step 3), the k slider (step 4).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const tGroup = make('div');
  const tLab = make('label');
  const tIn = make('input', { type: 'range', min: '0', max: String(T), step: '1', value: String(T) });
  const tOut = make('output');
  tOut.setAttribute('aria-hidden', 'true');
  tLab.append('step t', ' ', tIn, ' ', tOut);
  tGroup.append(tLab);
  const kGroup = make('div');
  const kLab = make('label');
  const kIn = make('input', { type: 'range', min: '0', max: String(K_CHOICES.length - 1), step: '1', value: String(kSel) });
  const kOut = make('output');
  kOut.setAttribute('aria-hidden', 'true');
  kLab.append('steps back, k', ' ', kIn, ' ', kOut);
  kGroup.append(kLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the network');
  const trainTag = make('span', { textContent: 'network, k = 2' });
  trainTag.setAttribute('aria-hidden', 'true');
  Object.assign(trainTag.style, { color: MUTED, fontFamily: FONT, fontSize: '12px', marginRight: '6px' });
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  trainGroup.append(trainTag);
  controls.append(tGroup, trainGroup, kGroup);
  const setValueText = () => {
    tOut.textContent = String(loopT);
    tIn.setAttribute('aria-valuetext', loopT === 0 ? 'before the first step' : `step ${loopT} of ${T}`);
    kOut.textContent = String(K_CHOICES[kSel]);
    kIn.setAttribute('aria-valuetext', `k = ${K_CHOICES[kSel]}`);
  };

  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };

  function info() {
    if (step === 0) return seqInfo(K_LIVE);
    if (step === 1) return loopInfo(loopT);
    if (step === 2) return liveInfo(getLive(), loop.running());
    if (step === 3) return budgetInfo(K_CHOICES[kSel], readyRuns());
    if (step === 4) return lagPlot();
    return null;
  }

  function render() {
    show(seqG, step === 0);
    show(loopG, step === 1);
    show(liveG, step === 2);
    show(budG, step === 3);
    show(lagG, step === 4);
    show(gateG, step === 5);
    setValueText();
    const wd = svg.clientWidth || 600;
    const pSize = wd > 0 && wd < 500 ? '13px' : '';
    for (const c of [...loopCols, ...liveCols]) c.p.style.fontSize = pSize;
    const s = info();

    if (step === 0) {
      seqRows.forEach((row, r) => {
        row.xs.forEach((c, t) => c.set(s.seqs[r][t]));
        row.ys.forEach((c, t) => c.set(s.mask[r][t] ? s.targets[r][t] : null));
      });
    }

    if (step === 1) {
      loopCols.forEach((c, i) => {
        const on = i < s.t;
        c.xb.set(on ? s.xs[i] : undefined);
        c.h.set(on ? s.hs[i + 1] : null);
        c.p.textContent = on ? num(s.ps[i], 2) : '';
        c.yb.set(!on ? undefined : s.mask[i] ? s.targets[i] : null);
      });
      show(loopNow, s.t > 0);
      if (s.t > 0) loopNow.setAttribute('x', colX(s.t) - 4);
    }

    if (step === 2) {
      const lv = getLive();
      const D = testSet(K_LIVE);
      const xs = D.seqs[EXAMPLE];
      const zs = rnnForward(lv.trainer.state.net, xs).zs;
      liveCols.forEach((c, i) => {
        c.xb.set(xs[i]);
        const masked = D.mask[EXAMPLE][i];
        c.yb.set(masked ? D.targets[EXAMPLE][i] : null);
        c.p.textContent = num(sigmoid(zs[i]), 2);
        const right = predictBit(zs[i]) === D.targets[EXAMPLE][i];
        c.m.textContent = masked ? (right ? '✓' : '✗') : '';
        c.m.style.fill = right ? CYAN : VIOLET;
      });
      const pts = lv.testOk.map((v, i) => `${i ? 'L' : 'M'}${P(ax(i))},${P(ay(v / lv.n))}`).join('');
      livePath.setAttribute('d', pts);
      liveDot.setAttribute('cx', P(ax(lv.testOk.length - 1)));
      liveDot.setAttribute('cy', P(ay(lv.testOk.at(-1) / lv.n)));
    }

    if (step === 3) {
      show(budWait, !s);
      budRows.forEach((row, q) => {
        const k = K_CHOICES[q];
        const r = s ? s.runs[k] : null;
        const a = r ? r.testOk[EPOCHS] / r.n : 0;
        row.rect.setAttribute('width', P(a * BW));
        row.val.setAttribute('x', P(BX0 + a * BW + 6));
        row.val.textContent = r ? pctCount(r.testOk[EPOCHS], r.n) : '';
        const sel = q === kSel;
        row.rect.setAttribute('fill-opacity', sel ? 0.6 : 0.25);
        row.rect.setAttribute('stroke-width', sel ? 2.5 : 1);
        row.name.textContent = sel ? `▸ k = ${k}` : `k = ${k}`;
        row.name.style.fill = sel ? STRONG : MUTED;
      });
      budTitle.textContent = `k = ${K_CHOICES[kSel]}, by epoch`;
      if (s) {
        budPath.setAttribute('d', s.testOk.map((v, e) => `${e ? 'L' : 'M'}${P(cxb(e))},${P(cyb(v / s.n))}`).join(''));
        s.testOk.forEach((v, e) => { budDots[e].setAttribute('cx', P(cxb(e))); budDots[e].setAttribute('cy', P(cyb(v / s.n))); });
      }
      show(budPath, !!s);
      budDots.forEach((d) => show(d, !!s));
    }

    if (step === 4) drawLag();

    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Three training streams of 12 bits, each with its target row: the target at step t is the input bit two steps earlier, and the first two steps have no target. ${rest}.`;
    if (step === 1) return `The recurrent network unrolled over 12 steps: each step reads one bit, updates the 8-number hidden state from the bit and the previous state, and gives a probability. ${rest}.`;
    if (step === 2) {
      const lv = getLive();
      return `Test stream 1 with its targets and the network's current probabilities, each marked right or wrong; below, the masked test accuracy after each of the ${lv.testOk.length - 1} batches so far. ${rest}.`;
    }
    if (step === 3) {
      if (!s) return `Computing the four runs. ${rest}.`;
      const bars = K_CHOICES.map((k) => `k = ${k}: ${pctCount(s.runs[k].testOk[EPOCHS], s.runs[k].n)}`).join(', ');
      return `Bars of masked test accuracy after ${EPOCHS} epochs: ${bars}. A dashed line marks chance, 50%. On the right, k = ${s.k} by epoch. ${rest}.`;
    }
    if (step === 4) return `A log-scale plot of the size of the gradient of the last step's loss with respect to each earlier hidden state, against steps back, for the starting network: one thick line for the typical size over ${N_TEST} test streams and thin dashed lines for ${LAG_SHOWN} single streams. ${rest}.`;
    return `A sketch: in a plain RNN the hidden state passes through Wₕ and tanh at every step; in a gated cell a separate cell state runs along a path with only a multiply by a forget gate and an add, so it can pass on unchanged. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  // ---- step 3's live run: one batch per frame, about three seconds for the 200.
  const loop = createTrainLoop({
    step: () => getLive().advance(1),
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
  trainGroup.addEventListener('click', () => { if (step === 2) { render(); announce(spoken()); } });

  // ---- step 4's four runs, computed when first needed, in slices of about 6 ms per frame.
  let job = null;
  const cancelRuns = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const computeRuns = () => {
    if (job !== null || K_CHOICES.every(runReady)) return;
    const tick = () => {
      job = null;
      if (step !== 3) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const k = K_CHOICES.find((x) => !runReady(x));
        if (k === undefined) break;
        runWork(k, 4);
      }
      if (K_CHOICES.every(runReady)) { render(); announce(spoken()); } else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

  // ---- step 2: the sequence is read one step at a time.
  const readAlong = (animate) => {
    const my = ++token;
    if (!animate || reduced) {
      loopT = T;
      tIn.value = String(T);
      render();
      return Promise.resolve();
    }
    return tween(900, (f) => {
      if (my !== token || step !== 1) return;
      loopT = Math.round(f * T);
      tIn.value = String(loopT);
      render();
    }, { reduced });
  };

  tIn.addEventListener('input', () => {
    token++;
    loopT = Number(tIn.value) || 0;
    render();
    announce(spoken());
  });
  kIn.addEventListener('input', () => {
    kSel = Number(kIn.value) || 0;
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
    show(i, animate) {
      buttons.stop();
      token++;
      step = i;
      if (i === 3) computeRuns(); else cancelRuns();
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      showGroup(tGroup, i === 1);
      showGroup(kGroup, i === 3);
      trainGroup.style.display = i === 2 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 2);
      const done = i === 1 ? readAlong(animate) : (render(), Promise.resolve());
      announce(spoken());
      return done;
    },
  };
}
