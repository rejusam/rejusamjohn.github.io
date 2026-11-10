// Lesson L4.5, "Optimisers: momentum and Adam": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-optimisers.test.mjs);
// the DOM is only touched inside createStage. Every path on the valley is made
// by optStep from nn.js, the same code the network uses, and every number in the
// page's text comes from these runs (storyValues, checked by the tests).

import { circlesData, mlpInit, trainer, accuracyNet, optInit, optStep } from '../assets/nn.js';
import { linearScale, el, axes, tween, trianglePoints } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- the valley ---------------------------------------------------------

// ℒ(u, v) = ½(u² + 25v²): curvature 1 along the valley (u), 25 across it (v),
// so κ = 25. The (u, v) axes are turned by ANGLE degrees from the plot's (w₁, w₂).
export const ANGLE = 30;
export const CURV = [1, 25];
export const START_UV = [-4, 2.5];
export const TARGET = 1e-3;
export const MAX_STEPS = 2000;
export const DIVERGED_AT = 1e6;
// Plain gradient descent is stable for η < 2 / (largest curvature).
export const ETA_LIMIT = 2 / CURV[1];

export function makeValley(angle = ANGLE) {
  const c = Math.cos((angle * Math.PI) / 180);
  const s = Math.sin((angle * Math.PI) / 180);
  const toUV = ([a, b]) => [c * a + s * b, -s * a + c * b];
  const toW = ([u, v]) => [c * u - s * v, s * u + c * v];
  return {
    angle,
    toUV,
    toW,
    loss(w) { const [u, v] = toUV(w); return 0.5 * (CURV[0] * u * u + CURV[1] * v * v); },
    grad(w) { const [u, v] = toUV(w); return toW([CURV[0] * u, CURV[1] * v]); },
    start: toW(START_UV),
  };
}

export const VALLEY = makeValley(ANGLE);
const VALLEYS = new Map([[ANGLE, VALLEY]]);

// One optimiser on the valley from the start, step by step, until ℒ < TARGET,
// the loss passes DIVERGED_AT (flagged, nothing non-finite kept) or max steps.
const RUNS = new Map();
export function valleyRun(kind, hyper, { max = MAX_STEPS, angle = ANGLE } = {}) {
  const key = `${kind}|${hyper.eta}|${hyper.beta ?? ''}|${max}|${angle}`;
  if (RUNS.has(key)) return RUNS.get(key);
  if (!VALLEYS.has(angle)) VALLEYS.set(angle, makeValley(angle));
  const V = VALLEYS.get(angle);
  let p = V.start.slice();
  let st = optInit(kind, 2);
  const path = [p];
  const losses = [V.loss(p)];
  let reached = null;
  let diverged = false;
  for (let k = 1; k <= max; k++) {
    const r = optStep(kind, hyper, st, p, V.grad(p));
    const L = V.loss(r.params);
    if (!r.params.every(Number.isFinite) || !Number.isFinite(L)) { diverged = true; break; }
    p = r.params;
    st = r.state;
    path.push(p);
    losses.push(L);
    if (L > DIVERGED_AT) { diverged = true; break; }
    if (L < TARGET) { reached = k; break; }
  }
  const run = { kind, eta: hyper.eta, beta: hyper.beta, path, losses, reached, diverged };
  RUNS.set(key, run);
  return run;
}

// How many times a path crosses the valley floor (v changes sign).
export function floorCrossings(run) {
  let n = 0;
  for (let k = 1; k < run.path.length; k++) if (Math.sign(VALLEY.toUV(run.path[k])[1]) !== Math.sign(VALLEY.toUV(run.path[k - 1])[1])) n++;
  return n;
}

// How far each weight moved on the first step.
export const firstMove = (run) => [0, 1].map((i) => Math.abs(run.path[1][i] - run.path[0][i]));

export const KINDS = ['sgd', 'momentum', 'rmsprop', 'adam'];
export const LABEL = { sgd: 'GD', momentum: 'Momentum', rmsprop: 'RMSProp', adam: 'Adam' };
// 1, 2 and 5 times powers of ten from 0.001 to 1.
export const GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];

// The η on the grid that reaches ℒ < TARGET in the fewest steps (ties: the first).
export function tuneValley(kind) {
  const table = GRID.map((eta) => {
    const r = valleyRun(kind, { eta });
    return { eta, reached: r.reached, diverged: r.diverged };
  });
  let best = null;
  for (const row of table) if (row.reached !== null && (best === null || row.reached < best.reached)) best = row;
  return { eta: best.eta, steps: best.reached, table };
}

// The values tuneValley picks (the tests check them).
export const VALLEY_ETA = { sgd: 0.05, momentum: 0.02, rmsprop: 0.1, adam: 1 };

// Step 1: the η slider, 0.005 to 0.09 in steps of 0.005.
export const GD_ETAS = Array.from({ length: 18 }, (_, i) => Number(((i + 1) * 0.005).toFixed(3)));
export const GD_DEFAULT = 0.07;
// Step 2: momentum's grid η with three values of β.
export const MOM_ETA = 0.02;
export const BETAS = [0, 0.5, 0.9];

export const gdInfo = (eta) => ({ eta, run: valleyRun('sgd', { eta }) });
export const momInfo = (beta) => ({ beta, run: valleyRun('momentum', { eta: MOM_ETA, beta }), reached: valleyRun('momentum', { eta: MOM_ETA, beta }).reached });
export const kindInfo = (kind) => ({ kind, run: valleyRun(kind, { eta: VALLEY_ETA[kind] }) });

// Step 5: the four runs, each with its grid η.
export const raceRuns = () => Object.fromEntries(KINDS.map((k) => [k, valleyRun(k, { eta: VALLEY_ETA[k] })]));
export const raceLength = (runs) => Math.max(...KINDS.map((k) => runs[k].reached));
export function raceOrder() {
  const runs = raceRuns();
  return KINDS.map((kind) => ({ kind, steps: runs[kind].reached })).sort((a, b) => a.steps - b.steps);
}
// Where each run is after k steps; a run that has reached the target stays put.
// running is the loop's activity(): false, 'training' or 'computing'.
export function raceInfo(runs, k, running = false) {
  const at = {};
  const loss = {};
  const done = {};
  for (const kind of KINDS) {
    const r = runs[kind];
    const j = Math.min(k, r.path.length - 1);
    at[kind] = r.path[j];
    loss[kind] = r.losses[j];
    done[kind] = r.reached !== null && k >= r.reached;
  }
  return { k, at, loss, done, runs, running, end: k >= raceLength(runs) };
}

// ---- the network (step 6) ------------------------------------------------

export const SEED_TRAIN = 151;
export const SEED_TEST = 152;
export const SEED_INIT = 155;
export const SEED_SHUFFLE = 154;
export const N_TRAIN = 200;
export const N_TEST = 400;
export const SIZES = [2, 4, 1];
export const BATCH = 20;
export const NET_EPOCHS = 100;
export const NET_ITERS = (NET_EPOCHS * N_TRAIN) / BATCH;
export const LOSS_MAX = 0.8;
// The grid η with the lowest training loss averaged over epochs 1 to 100 (the tests check them).
export const NET_ETA = { sgd: 1, momentum: 0.5, rmsprop: 0.05, adam: 0.1 };

export const trainSet = () => circlesData({ n: N_TRAIN, seed: SEED_TRAIN });
export const testSet = () => circlesData({ n: N_TEST, seed: SEED_TEST });
export const startNet = () => mlpInit(SIZES, { hidden: 'tanh', init: 'xavier', seed: SEED_INIT });

const TR = trainSet();
const TE = testSet();

const makeNet = (kind, eta) => trainer({ net: startNet(), X: TR.X, y: TR.y, batch: BATCH, kind, hyper: { eta }, seed: SEED_SHUFFLE });

// The training loss averaged over epochs 1 to 100: low when a run gets down fast and stays down.
export const meanAfterStart = (losses) => losses.slice(1).reduce((a, v) => a + v, 0) / (losses.length - 1);

const NETS = new Map();
const JOBS = new Map();

// k more optimiser steps of this optimiser's run; true once its result is ready.
// The page calls this in small slices; the slicing does not change the result.
export function netWork(kind, k) {
  if (NETS.has(kind)) return true;
  if (!JOBS.has(kind)) JOBS.set(kind, makeNet(kind, NET_ETA[kind]));
  const t = JOBS.get(kind);
  t.step(Math.max(0, Math.min(k, NET_ITERS - t.state.iter)));
  if (t.state.iter < NET_ITERS && !t.state.diverged) return false;
  const losses = t.state.epochLoss.slice();
  NETS.set(kind, { kind, eta: NET_ETA[kind], losses, final: losses[NET_EPOCHS], mean: meanAfterStart(losses), test: accuracyNet(t.state.net, TE.X, TE.y) });
  JOBS.delete(kind);
  return true;
}
export const netReady = (kind) => NETS.has(kind);
export function netRun(kind) {
  netWork(kind, Infinity);
  return NETS.get(kind);
}
export const netInfo = () => ({ nets: Object.fromEntries(KINDS.map((k) => [k, NETS.get(k) || null])) });

// How many times the training loss goes up from one epoch to the next.
export const ups = (losses) => losses.slice(1).filter((v, i) => v > losses[i]).length;

// The same run with every batch holding all 200 points (a noise-free gradient).
export function fullBatchUps(kind) {
  const t = trainer({ net: startNet(), X: TR.X, y: TR.y, batch: N_TRAIN, kind, hyper: { eta: NET_ETA[kind] }, seed: SEED_SHUFFLE });
  t.step(NET_EPOCHS);
  return ups(t.state.epochLoss);
}

// The grid search behind NET_ETA (used by the tests, not by the page).
export function tuneNet(kind) {
  const table = GRID.map((eta) => {
    const t = makeNet(kind, eta);
    t.step(NET_ITERS);
    return { eta, loss: t.state.diverged ? null : meanAfterStart(t.state.epochLoss) };
  });
  let best = null;
  for (const row of table) if (row.loss !== null && (best === null || row.loss < best.loss)) best = row;
  return { eta: best.eta, loss: best.loss, table };
}

// ---- formatting and read-outs -------------------------------------------

export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
const T3 = 'ℒ < 10⁻³';

function runRows(run) {
  if (run.diverged) return ['Diverged: the steps grow', 'instead of shrinking'];
  if (run.reached === null) return [`ℒ not below 10⁻³`, `in ${MAX_STEPS} steps`];
  return [`steps to ${T3}: ${run.reached}`];
}

export function readout(step, s) {
  if (step === 0) {
    const flip = s.eta * CURV[1] > 1;
    return [`plain GD, η = ${s.eta}`, ...runRows(s.run), s.run.diverged ? '' : flip ? 'across: flips each step' : 'across: no flip'].filter(Boolean);
  }
  if (step === 1) return [`momentum, η = ${MOM_ETA}`, `β = ${s.beta}`, ...runRows(s.run)];
  if (step === 2 || step === 3) {
    const m = firstMove(s.run);
    return [`${LABEL[s.kind]}, η = ${s.run.eta}`, ...runRows(s.run), `step 1: each weight ±${num(m[0], 2)}`];
  }
  if (step === 4) {
    const head = s.k === 0 ? 'all four start together' : `step ${s.k}`;
    return [head, ...KINDS.map((k) => {
      const name = `${LABEL[k]} η ${s.runs[k].eta}`;
      return s.done[k] ? `${name}: at ${s.runs[k].reached}` : `${name}: ℒ ${num(s.loss[k], 2)}`;
    }), trainStatus({ running: s.running, iter: s.k, done: s.end })];
  }
  if (step === 5) {
    return ['train loss · test acc.', ...KINDS.map((k) => {
      const r = s.nets[k];
      return r ? `${LABEL[k]}: ${num(r.final, 3)} · ${pct(r.test)}` : `${LABEL[k]}: computing…`;
    })];
  }
  return [];
}

// Filled into the page at run time (constants).
export function pageValues() {
  const [a, b] = VALLEY.start;
  return {
    l0: String(VALLEY.loss(VALLEY.start)),
    'w-start': `(${num(a, 2)}, ${num(b, 2)})`,
    'eta-limit': String(ETA_LIMIT),
    'gd-eta': String(GD_DEFAULT),
    'mom-eta': String(MOM_ETA),
    'max-steps': String(MAX_STEPS),
    'eta-sgd': String(VALLEY_ETA.sgd),
    'eta-momentum': String(VALLEY_ETA.momentum),
    'eta-rmsprop': String(VALLEY_ETA.rmsprop),
    'eta-adam': String(VALLEY_ETA.adam),
    'net-eta-sgd': String(NET_ETA.sgd),
    'net-eta-momentum': String(NET_ETA.momentum),
    'net-eta-rmsprop': String(NET_ETA.rmsprop),
    'net-eta-adam': String(NET_ETA.adam),
  };
}

// From the runs; written in the page and checked by the tests.
let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  v['gd-default'] = String(valleyRun('sgd', { eta: GD_DEFAULT }).reached);
  v['gd-limit-loss'] = num(Math.min(...valleyRun('sgd', { eta: ETA_LIMIT }).losses), 1);
  v['gd-aligned'] = String(valleyRun('sgd', { eta: GD_DEFAULT }, { angle: 0 }).reached);
  v['mom-0'] = String(momInfo(0).reached);
  v['mom-05'] = String(momInfo(0.5).reached);
  v['mom-09'] = String(momInfo(0.9).reached);
  v['mom-cross'] = String(momInfo(0.9).run.path.findIndex((p) => VALLEY.toUV(p)[0] > 0));
  v['mom-swings'] = String(floorCrossings(momInfo(0.9).run));
  const rms = kindInfo('rmsprop').run;
  const adam = kindInfo('adam').run;
  v['rms-steps'] = String(rms.reached);
  v['rms-first'] = num(firstMove(rms)[0], 2);
  v['rms-aligned'] = String(valleyRun('rmsprop', { eta: VALLEY_ETA.rmsprop }, { angle: 0 }).reached);
  v['adam-steps'] = String(adam.reached);
  v['adam-first'] = num(firstMove(adam)[0], 2);
  const runs = raceRuns();
  for (const k of KINDS) v[`race-${k}`] = String(runs[k].reached);
  v['gd-fine'] = String(valleyRun('sgd', { eta: 0.075 }).reached);
  v['adam-2'] = String(valleyRun('adam', { eta: 2 }).reached);
  for (const k of KINDS) {
    const r = netRun(k);
    v[`net-${k}`] = num(r.final, 3);
    v[`acc-${k}`] = pct(r.test);
  }
  v['rms-full-ups'] = String(fullBatchUps('rmsprop'));
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

// The plot (same scale on both axes), and the panel on the right.
const PX0 = 50;
const PX1 = 360;
const PY0 = 300;
const PY1 = 24;
const W1 = [-5.4, 2.2];
const W2_LO = -4.4;
const W2 = [W2_LO, W2_LO + ((PY0 - PY1) * (W1[1] - W1[0])) / (PX1 - PX0)];
const RX0 = 384;
const ROW = 20;
// Contour levels of ℒ.
const LEVELS = [0.25, 1, 4, 16, 64];

const STYLE = {
  sgd: { stroke: STRONG, dash: '2 3', width: 1.6 },
  momentum: { stroke: CYAN, dash: '', width: 1.8 },
  rmsprop: { stroke: MUTED, dash: '7 4', width: 1.6 },
  adam: { stroke: VIOLET, dash: '4 2 1 2', width: 1.8 },
};

// A marker of about the same area for each optimiser: circle, square, diamond, triangle.
function marker(parent, kind, r = 5.5) {
  const st = { fill: STYLE[kind].stroke, stroke: BG, 'stroke-width': 1.5 };
  if (kind === 'sgd') return el('circle', { r, ...st }, parent);
  if (kind === 'momentum') { const s = r * Math.sqrt(Math.PI); return el('rect', { width: s, height: s, x: -s / 2, y: -s / 2, ...st }, parent); }
  if (kind === 'rmsprop') { const s = r * Math.sqrt(Math.PI / 2); return el('path', { d: `M0,${-s * 1.0}L${s},0L0,${s}L${-s},0Z`, ...st }, parent); }
  const d = trianglePoints(0, 0, r).map(([x, y], k) => `${k ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join('') + 'Z';
  return el('path', { d, ...st }, parent);
}
const place = (node, x, y) => node.setAttribute('transform', `translate(${x.toFixed(2)},${y.toFixed(2)})`);

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let eta = GD_DEFAULT;
  let beta = 0.9;
  let shown = Infinity;
  let raceK = 0;
  let token = 0;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A contour map of a long narrow valley over two weights w₁ and w₂, tilted 30 degrees, with the path each optimiser takes from the start towards the lowest point; in the last step, training-loss curves of four optimisers on a small network.');

  const P = (v) => Number(v.toFixed(2));
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const X = linearScale(W1, [PX0, PX1]);
  const Y = linearScale(W2, [PY0, PY1]);

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l45-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  // ---- The valley: axes, contours, the valley's own axes, start and bottom.
  const valG = el('g', {}, svg);
  axes(valG, X, Y, { xLabel: 'weight w₁', yLabel: 'weight w₂' });
  const conG = el('g', { 'clip-path': 'url(#l45-plot)' }, valG);
  LEVELS.forEach((c, idx) => {
    const a = Math.sqrt(2 * c / CURV[0]);
    const b = Math.sqrt(2 * c / CURV[1]);
    let d = '';
    for (let k = 0; k <= 160; k++) {
      const t = (2 * Math.PI * k) / 160;
      const [w1, w2] = VALLEY.toW([a * Math.cos(t), b * Math.sin(t)]);
      d += `${k ? 'L' : 'M'}${P(X(w1))},${P(Y(w2))}`;
    }
    el('path', { d: d + 'Z', fill: 'none', stroke: INK, 'stroke-width': 1.1, opacity: (0.75 - idx * 0.11).toFixed(2) }, conG);
  });
  // The valley's u and v directions through the bottom.
  const axis = (dir, len) => {
    const [a, b] = VALLEY.toW(dir.map((x) => x * len));
    return { x1: P(X(-a)), y1: P(Y(-b)), x2: P(X(a)), y2: P(Y(b)) };
  };
  el('line', { ...axis([1, 0], 9), stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '1 4' }, conG);
  el('line', { ...axis([0, 1], 1.4), stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '1 4' }, conG);
  {
    const [a, b] = VALLEY.toW([1.9, 0]);
    el('text', { class: 'stage-text', x: P(X(a)), y: P(Y(b)) + 18, 'text-anchor': 'middle', text: 'u: along', ...HALO }, valG).style.fill = MUTED;
    const [c, d] = VALLEY.toW([0, 1.5]);
    el('text', { class: 'stage-text', x: P(X(c)) - 4, y: P(Y(d)) - 6, 'text-anchor': 'end', text: 'v: across', ...HALO }, valG).style.fill = MUTED;
  }
  el('rect', { x: P(X(0)) - 4.5, y: P(Y(0)) - 4.5, width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, valG);
  const [sx, sy] = [X(VALLEY.start[0]), Y(VALLEY.start[1])];
  el('circle', { cx: P(sx), cy: P(sy), r: 4, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, valG);
  el('text', { class: 'stage-text', x: P(sx) + 2, y: P(sy) - 10, text: 'start', ...HALO }, valG).style.fill = STRONG;

  // ---- Paths: one per optimiser, each with a dot per step and a ball at its end.
  const pathG = el('g', { 'clip-path': 'url(#l45-plot)' }, svg);
  const tracks = {};
  for (const kind of KINDS) {
    const g = el('g', {}, pathG);
    const dots = el('g', {}, g);
    const line = el('path', { fill: 'none', stroke: STYLE[kind].stroke, 'stroke-width': STYLE[kind].width, 'stroke-dasharray': STYLE[kind].dash || null, 'stroke-linejoin': 'round' }, g);
    tracks[kind] = { g, dots, line, ball: marker(g, kind) };
  }
  // Points far outside the plot (a diverging run) are not drawn.
  const FAR = 40;
  const drawTrack = (kind, run, upto) => {
    const t = tracks[kind];
    const n = Math.min(upto, run.path.length - 1);
    let d = '';
    let last = 0;
    for (let k = 0; k <= n; k++) {
      const [a, b] = run.path[k];
      if (Math.abs(a) > FAR || Math.abs(b) > FAR) break;
      d += `${k ? 'L' : 'M'}${P(X(a))},${P(Y(b))}`;
      last = k;
    }
    t.line.setAttribute('d', d);
    t.dots.replaceChildren();
    if (run.path.length <= 160) {
      for (let k = 1; k < last; k++) el('circle', { cx: P(X(run.path[k][0])), cy: P(Y(run.path[k][1])), r: 2, fill: STYLE[kind].stroke, opacity: 0.55 }, t.dots);
    }
    const [a, b] = run.path[last];
    place(t.ball, X(a), Y(b));
    show(t.ball, X(a) >= PX0 - 1 && X(a) <= PX1 + 1 && Y(b) >= PY1 - 1 && Y(b) <= PY0 + 1);
  };
  const divNote = el('text', { class: 'stage-text', x: (PX0 + PX1) / 2, y: PY1 + 20, 'text-anchor': 'middle', text: 'Diverged: the steps grow', ...HALO }, svg);
  divNote.style.fill = STRONG;

  // ---- Step 6: training-loss curves.
  const netG = el('g', {}, svg);
  const NX = linearScale([0, NET_EPOCHS], [PX0, PX1 - 10]);
  const NY = linearScale([0, LOSS_MAX], [PY0, PY1]);
  axes(netG, NX, NY, { ticks: 4, xLabel: 'epoch', yLabel: 'training loss' });
  const curves = {};
  for (const kind of KINDS) curves[kind] = el('path', { fill: 'none', stroke: STYLE[kind].stroke, 'stroke-width': STYLE[kind].width, 'stroke-dasharray': STYLE[kind].dash || null, 'stroke-linejoin': 'round' }, netG);
  const ends = {};
  for (const kind of KINDS) ends[kind] = marker(netG, kind, 4.5);

  // ---- Legend (steps 5 and 6).
  const legG = el('g', {}, svg);
  KINDS.forEach((kind, i) => {
    const y = 34 + 20 * i;
    el('line', { x1: RX0 + 4, x2: RX0 + 34, y1: y, y2: y, stroke: STYLE[kind].stroke, 'stroke-width': 2, 'stroke-dasharray': STYLE[kind].dash || null }, legG);
    place(marker(legG, kind, 4.5), RX0 + 19, y);
    el('text', { class: 'stage-text', x: RX0 + 42, y: y + 4, text: LABEL[kind] }, legG).style.fill = STRONG;
  });

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4, 5].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const etaGroup = make('div');
  const etaLabel = make('label');
  const etaInput = make('input', { type: 'range', min: 0, max: GD_ETAS.length - 1, step: 1, value: GD_ETAS.indexOf(GD_DEFAULT) });
  etaInput.style.width = '140px';
  const etaOut = make('output');
  etaOut.setAttribute('aria-hidden', 'true');
  etaLabel.append('η', make('span', { className: 'vh', textContent: ', the learning rate' }), ' ', etaInput, ' ', etaOut);
  etaGroup.append(etaLabel);
  const betaGroup = make('div');
  betaGroup.setAttribute('role', 'group');
  betaGroup.setAttribute('aria-label', 'momentum β');
  const betaBtns = BETAS.map((b) => make('button', { type: 'button', textContent: `β = ${b}` }));
  betaGroup.append(...betaBtns);
  const trainGroup = make('div');
  controls.append(etaGroup, betaGroup, trainGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // The run shown in steps 1 to 4.
  const single = () => {
    if (step === 0) return { kind: 'sgd', run: gdInfo(eta).run };
    if (step === 1) return { kind: 'momentum', run: momInfo(beta).run };
    if (step === 2) return { kind: 'rmsprop', run: kindInfo('rmsprop').run };
    return { kind: 'adam', run: kindInfo('adam').run };
  };
  const runs = raceRuns();
  const raceEnd = raceLength(runs);

  function info() {
    if (step === 0) return gdInfo(eta);
    if (step === 1) return momInfo(beta);
    if (step === 2) return kindInfo('rmsprop');
    if (step === 3) return kindInfo('adam');
    if (step === 4) return raceInfo(runs, raceK, loop.activity());
    return netInfo();
  }

  function render() {
    const onValley = step <= 4;
    show(valG, onValley);
    show(pathG, onValley);
    show(netG, step === 5);
    show(legG, step >= 4);
    for (const kind of KINDS) show(tracks[kind].g, false);
    show(divNote, false);
    if (step <= 3) {
      const { kind, run } = single();
      show(tracks[kind].g, true);
      drawTrack(kind, run, shown);
      show(divNote, run.diverged);
    }
    if (step === 4) {
      for (const kind of KINDS) {
        show(tracks[kind].g, true);
        drawTrack(kind, runs[kind], Math.min(raceK, runs[kind].reached));
      }
    }
    if (step === 5) {
      for (const kind of KINDS) {
        const r = netReady(kind) ? netRun(kind) : null;
        show(curves[kind], !!r);
        show(ends[kind], !!r);
        if (r) {
          curves[kind].setAttribute('d', r.losses.map((v, e) => `${e ? 'L' : 'M'}${P(NX(e))},${P(NY(Math.min(v, LOSS_MAX)))}`).join(''));
          place(ends[kind], NX(NET_EPOCHS), NY(r.final));
        }
      }
    }
    etaOut.textContent = String(eta);
    etaInput.setAttribute('aria-valuetext', `η ${eta}`);
    betaBtns.forEach((btn, k) => btn.setAttribute('aria-pressed', String(BETAS[k] === beta)));

    const texts = readout(step, info());
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
    const by = step >= 4 ? 120 : 24;
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
    const rows = readout(i, s);
    if (i <= 3) return `${rows.join(', ')}.`;
    if (i === 4) return raceK === 0 ? 'Press Train to start the race.' : `${rows.join(', ')}.`;
    return `After 100 epochs, training loss and test accuracy: ${rows.slice(1).join(', ')}.`;
  };

  // Steps 1 to 4: the path is drawn step by step, about a second in all.
  function reveal(animate) {
    const my = ++token;
    const { run } = single();
    const n = run.path.length - 1;
    if (!animate || reduced) {
      shown = Infinity;
      render();
      return Promise.resolve(true);
    }
    shown = 0;
    render();
    // Resolves false if a later step or setting took over, so nothing stale is announced.
    return tween(Math.min(1400, 400 + 10 * n), (t) => {
      if (my !== token) return;
      shown = Math.round(t * n);
      render();
    }, { reduced }).then(() => my === token);
  }

  // ---- Step 5: the race, run by the training loop, one step every other frame.
  let frame = 0;
  const loop = createTrainLoop({
    step: () => {
      frame++;
      if (frame % 2 === 0 && raceK < raceEnd) raceK++;
    },
    render,
    done: () => raceK >= raceEnd,
    onStop: () => {
      buttons.sync();
      render();
      if (raceK >= raceEnd) announce(`Finished. ${spoken(4)}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { raceK = 0; frame = 0; },
    reduced,
    work: (k) => { raceK = Math.min(raceEnd, raceK + k); },
  });

  // Pausing does not call render, so redraw the read-out after any Train/Pause/Reset click.
  trainGroup.addEventListener('click', () => { if (step === 4) render(); });

  // ---- Step 6's runs, computed when the step is first shown, in slices of
  // about 6 ms per animation frame.
  let job = null;
  const cancelNets = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const computeNets = () => {
    if (job !== null || KINDS.every(netReady)) return;
    const tick = () => {
      job = null;
      if (step !== 5) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const kind = KINDS.find((k) => !netReady(k));
        if (kind === undefined) break;
        if (netWork(kind, 25)) render();
      }
      if (KINDS.every(netReady)) announce(spoken(5));
      else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

  etaInput.addEventListener('input', () => {
    eta = GD_ETAS[Number(etaInput.value)];
    token++;
    shown = Infinity;
    render();
    announce(spoken(0));
  });
  betaBtns.forEach((btn, k) => btn.addEventListener('click', () => {
    beta = BETAS[k];
    reveal(true).then((ok) => { if (ok) announce(spoken(1)); });
  }));

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
      step = i;
      token++;
      shown = Infinity;
      if (i === 5) computeNets(); else cancelNets();
      showGroup(etaGroup, i === 0);
      showGroup(betaGroup, i === 1);
      showGroup(trainGroup, i === 4);
      controls.classList.toggle('has-train', i === 4);
      announce(spoken(i));
      if (i <= 3) return reveal(animate);
      render();
      return Promise.resolve();
    },
  };
}
