// Lesson L5.8, "Diffusion: from noise to data": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-diffusion.test.mjs); the
// DOM is only touched inside createStage. The schedule, the forward jump, the denoiser,
// its training and every reverse step come from gen.js, and every number in the page's
// text comes from pageValues or storyValues.

import { mulberry32, randn } from '../assets/maths.js';
import {
  spiral2D, LESSON_SPIRAL, ddpmSchedule, qSample, timeFeatures, denoiserInit, denoiserForward,
  denoiserFlatten, denoiserUnflatten, ddpmTrainer, ddpmStep, nearestDist, gmmInit, gmmEM, gmmLogLik, gmmSample,
} from '../assets/gen.js';
import { el } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- settings ---------------------------------------------------------------------------

export const T = 50;
export const N = LESSON_SPIRAL.n;
export const HIDDEN = [64, 64];
export const SEED_INIT = 311;
export const SEED_TRAIN = 312;
export const SEED_SAMPLE = 313;
export const SEED_FORWARD = 300;
export const SEED_FRESH = 307;
export const ETA = 3e-3;
export const BATCH = 128;
export const STEPS = 6000;
export const EVAL_EVERY = 100;
export const EVAL_N = 512;
export const EXAMPLE = 136;
export const T_START = 0;
export const T_EXAMPLE_MIN = 1;
// Generate's frames: each reverse step is drawn over GEN_FRAMES frames.
export const GEN_FRAMES = 4;
// The mixture from the generative-models lesson: K = 5, EM from seed 304 until a rise
// is below MIX_TOL or MIX_MAX iterations; its samples use seed 310, plus 1 per press.
export const MIX_K = 5;
export const MIX_SEED = 304;
export const MIX_TOL = 1e-6;
export const MIX_MAX = 500;
export const MIX_SAMPLE = 310;
// Presses summarised in the text.
export const PRESSES = 20;
// Points shown with their spread circles in step 2.
export const MARKED = [EXAMPLE, 40, 300];

const cache = new Map();
const cached = (key, make) => {
  if (!cache.has(key)) cache.set(key, make());
  return cache.get(key);
};

export const spiral = () => cached('x', () => spiral2D(LESSON_SPIRAL));
export const freshSpiral = () => cached('f', () => spiral2D({ ...LESSON_SPIRAL, seed: SEED_FRESH }));
export const schedule = () => cached('s', () => ddpmSchedule(T));

// One noise draw per data point (x then y), seed 300: the slider moves each point along
// its own straight line from x₀ towards its noise.
export const forwardNoise = () => cached('e', () => {
  const rng = mulberry32(SEED_FORWARD);
  return Array.from({ length: N }, () => [randn(rng), randn(rng)]);
});
export const forwardPoints = (t) => spiral().map((x, i) => qSample(x, t, forwardNoise()[i], schedule()));

// √ᾱₜ (what is left of x₀) and √(1 − ᾱₜ) (the noise's standard deviation).
export const signal = (t) => Math.sqrt(schedule().abar[t]);
export const noiseSd = (t) => Math.sqrt(1 - schedule().abar[t]);

export const startNet = () => denoiserInit({ hidden: HIDDEN, seed: SEED_INIT });
export const paramCount = () => startNet().params.length;

// The training run. advance(m) takes up to m more steps; done after STEPS.
export function makeLive(net = startNet()) {
  const trainer = ddpmTrainer({ net, X: spiral(), sched: schedule(), batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_TRAIN });
  const live = { trainer };
  live.advance = (m) => {
    const k = Math.max(0, Math.min(m, STEPS - trainer.state.iter));
    if (k > 0) trainer.step(k);
  };
  live.done = () => trainer.state.iter >= STEPS || trainer.state.diverged;
  live.reset = () => trainer.reset();
  return live;
}

// The loss of guessing 0 every time, on the trainer's fixed batch: a trainer whose
// network has every weight 0 reports exactly that as its first evaluation.
export const zeroGuessLoss = () => cached('z', () => {
  const n0 = startNet();
  const zero = denoiserUnflatten(n0, new Float64Array(n0.params.length));
  return ddpmTrainer({ net: zero, X: spiral(), sched: schedule(), batch: BATCH, hyper: { eta: ETA }, seed: SEED_TRAIN }).state.evalLoss[0];
});

// Step 3's example: data point EXAMPLE, its noise from the forward draw, and the
// network's guess at step t.
export function example(net, t) {
  const x0 = spiral()[EXAMPLE];
  const eps = forwardNoise()[EXAMPLE];
  const xt = qSample(x0, t, eps, schedule());
  const guess = denoiserForward(net, xt, timeFeatures(t, T));
  const err = (eps[0] - guess[0]) ** 2 + (eps[1] - guess[1]) ** 2;
  return { t, x0, eps, xt, guess, err };
}

// ---- generation ---------------------------------------------------------------------------

// The reverse walk from n points of N(0, I), one step t → t − 1 per call, drawing its
// random numbers in the same order as ddpmSample, so path[k] equals ddpmSample's path[k].
export function createSampler(net, seed, n = N) {
  const snap = denoiserUnflatten(net, denoiserFlatten(net));
  const rng = mulberry32(seed);
  const sched = schedule();
  let X = Array.from({ length: n }, () => [randn(rng), randn(rng)]);
  const s = { seed, path: [X], k: 0 };
  s.step = () => {
    if (s.k >= T) return;
    const t = T - s.k;
    X = X.map((x) => ddpmStep(snap, x, t, t > 1 ? [randn(rng), randn(rng)] : [0, 0], sched));
    s.path.push(X);
    s.k++;
  };
  s.done = () => s.k >= T;
  return s;
}

// Press number press (1, 2, …) of Generate, with seed 313 + press − 1: the sampler plus a frame counter f. Frame f draws the points
// at u = f / GEN_FRAMES reverse steps, between path[⌊u⌋] and the next.
export function makeGen(net, press, iter) {
  const g = { press, iter, seed: SEED_SAMPLE + press - 1, f: 0 };
  g.sampler = createSampler(net, g.seed);
  g.frame = () => {
    if (g.done()) return;
    g.f++;
    while (g.sampler.k < Math.min(T, Math.ceil(g.f / GEN_FRAMES))) g.sampler.step();
  };
  // Reduced motion: one reverse step per call, no frames to watch.
  g.quiet = () => {
    g.sampler.step();
    if (g.sampler.done()) g.f = T * GEN_FRAMES;
  };
  g.done = () => g.f >= T * GEN_FRAMES;
  g.u = () => Math.min(T, g.f / GEN_FRAMES);
  // The points as drawn now.
  g.points = () => {
    const u = g.u();
    const k0 = Math.min(Math.floor(u), g.sampler.k);
    const a = g.sampler.path[k0];
    const b = g.sampler.path[Math.min(k0 + 1, g.sampler.k)];
    const fr = u - k0;
    return a.map((p, i) => [p[0] + fr * (b[i][0] - p[0]), p[1] + fr * (b[i][1] - p[1])]);
  };
  // The step t the points have reached (T before the first step, 0 at the end).
  g.tNow = () => T - Math.floor(g.u());
  return g;
}

// precision: mean distance from each new point to its nearest data point.
// coverage: mean distance from each data point to its nearest new point.
export const quality = (S, X = spiral()) => ({ precision: nearestDist(S, X), coverage: nearestDist(X, S) });

// The K = 5 mixture, fitted in slices (mixWork) or at once (mixFit).
let MIX = null;
let MIXJOB = null;
export function mixWork(iters) {
  if (MIX) return true;
  const X = spiral();
  if (!MIXJOB) {
    const p = gmmInit(X, MIX_K, MIX_SEED);
    MIXJOB = { params: p, ll: [gmmLogLik(X, p)], iter: 0, done: false };
  }
  const j = MIXJOB;
  for (let i = 0; i < iters && !j.done; i++) {
    const r = gmmEM(X, j.params, 1);
    const L = r.logLik[1];
    const rise = L - j.ll.at(-1);
    j.params = r.params;
    j.ll.push(L);
    j.iter++;
    if (!Number.isFinite(L) || rise < MIX_TOL || j.iter >= MIX_MAX) j.done = true;
  }
  if (!j.done) return false;
  MIX = { params: j.params, logLik: j.ll, iter: j.iter };
  MIXJOB = null;
  return true;
}
export const mixReady = () => MIX !== null;
export function mixFit() {
  mixWork(Infinity);
  return MIX;
}
export const mixSamples = (press) => gmmSample(mixFit().params, N, MIX_SAMPLE + press - 1);

// ---- formatting ---------------------------------------------------------------------------

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
export const sig2 = (v) => num(Number(v.toPrecision(2)), Math.max(0, 1 - Math.floor(Math.log10(Math.abs(v)))));
export const dist = (v) => num(v, 2);
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
// ᾱₜ: three decimals, or two significant figures below 0.01.
export const abarText = (v) => (v >= 0.01 ? num(v, 3) : sig2(v));
const vec = (v) => `(${num(v[0], 2)}, ${num(v[1], 2)})`;

// ---- read-outs ----------------------------------------------------------------------------

export function liveInfo(live, running) {
  const s = live.trainer.state;
  return { iter: s.iter, loss: s.evalLoss.at(-1), evals: s.evalLoss.length, running, done: live.done(), diverged: s.diverged };
}

export function genInfo(g, mixQ) {
  if (!g) return { gen: null };
  const out = { gen: g, press: g.press, seed: g.seed, iter: g.iter, tNow: g.tNow(), done: g.done(), noise: quality(g.sampler.path[0]) };
  if (g.done()) out.samples = quality(g.sampler.path[T]);
  out.mix = mixQ;
  return out;
}

export function readout(step, s) {
  if (step === 0) {
    return [`t = ${s.t} of ${T} · ᾱₜ = ${abarText(schedule().abar[s.t])}`, `signal √ᾱₜ = ${num(signal(s.t), 2)} · noise SD ${num(noiseSd(s.t), 2)}`, `${N} points, one noise draw each`];
  }
  if (step === 1) {
    return [`t = ${s.t} · centre = ${num(signal(s.t), 2)} × x₀`, `spread: SD ${num(noiseSd(s.t), 2)} on each axis`, 'circles: 1 and 2 SD around each centre'];
  }
  if (step === 2) {
    const e = s.ex;
    return [`example point, t = ${e.t} · ${s.iter === 0 ? 'untrained network' : `after ${count(s.iter)} steps`}`, `noise ε ${vec(e.eps)} · guess ε̂ ${vec(e.guess)}`, `squared error ‖ε − ε̂‖² = ${num(e.err, 2)}`];
  }
  if (step === 3) {
    return [`step ${count(s.iter)} of ${count(STEPS)} · ${trainStatus(s)}`, `loss on ${EVAL_N} fixed rows: ${num(s.loss, 3)}`, `guessing 0 every time: ${num(zeroGuessLoss(), 3)}`];
  }
  if (step === 4) {
    if (!s.gen) return ['press Generate', `the network: ${s.netIter === 0 ? 'untrained (train it in step 4)' : `after ${count(s.netIter)} training steps`}`];
    const head = `press ${s.press} · seed ${s.seed} · t = ${s.tNow}${s.done ? ' · done' : ''}`;
    const net = s.iter >= STEPS ? 'network trained for all 6,000 steps' : s.iter === 0 ? 'network untrained: train it in step 4' : `network after ${count(s.iter)} of ${count(STEPS)} steps`;
    if (!s.done) return [head, 'walking back from noise…', net];
    return [head, `samples → data ${dist(s.samples.precision)} · data → samples ${dist(s.samples.coverage)}`, net];
  }
  if (step === 5) return ['1 add noise · 2 learn to guess it · 3 remove it', 'the same three steps at any scale'];
  return [];
}

// ---- values written into the page -----------------------------------------------------------

export function pageValues() {
  const sc = schedule();
  return {
    'abar-T': abarText(sc.abar[T]),
    'sqrt-abar-T': num(Math.sqrt(sc.abar[T]), 3),
    'beta-1': String(sc.beta[1]),
    'beta-T': String(Number(sc.beta[T].toPrecision(12))),
    'n-params': count(paramCount()),
    'zero-loss': num(zeroGuessLoss(), 2),
  };
}

let STORY = null;
// The page's quoted run: training to the end, then press 1 of Generate.
export function storyRun(net = startNet()) {
  const live = makeLive(net);
  live.advance(STEPS);
  const s = live.trainer.state;
  const g = makeGen(s.net, 1, s.iter);
  while (!g.done()) g.quiet();
  return { live, state: s, gen: g };
}
export function storyFrom(run) {
  const v = {};
  const ev = run.state.evalLoss;
  v['loss-start'] = num(ev[0], 2);
  v['loss-100'] = num(ev[1], 2);
  v['loss-end'] = num(ev.at(-1), 2);
  const noise = quality(run.gen.sampler.path[0]);
  const samp = quality(run.gen.sampler.path[T]);
  const mix = quality(mixSamples(1));
  const fresh = quality(freshSpiral());
  for (const [k, q] of [['noise', noise], ['diff', samp], ['mix', mix], ['fresh', fresh]]) {
    v[`${k}-prec`] = dist(q.precision);
    v[`${k}-cov`] = dist(q.coverage);
  }
  return v;
}
export function storyValues() {
  if (!STORY) STORY = storyFrom(storyRun());
  return STORY;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(1));
const RY = [292, 314, 336];

// The left scatter: a square of side PS showing −R … R on both axes.
const PX0 = 24;
const PY0 = 8;
const PS = 262;
const R = 3.2;
const px = (u) => PX0 + ((u + R) / (2 * R)) * PS;
const py = (v) => PY0 + ((R - v) / (2 * R)) * PS;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let t = T_START;
  let live = null;
  let gen = null;
  let presses = 0;

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const n = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    n.style.fill = fill;
    return n;
  };
  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'dif-clip' }, defs);
  el('rect', { x: PX0, y: PY0, width: PS, height: PS }, clip);
  const mk = (id, fill) => {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill }, m);
  };
  mk('dif-arrow-c', CYAN);
  mk('dif-arrow-s', STRONG);
  mk('dif-arrow-m', MUTED);

  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };

  // ---- the left scatter (steps 1 to 3 and 5).
  const plotG = el('g', {}, svg);
  el('rect', { x: PX0, y: PY0, width: PS, height: PS, fill: 'none', stroke: LINE }, plotG);
  el('line', { x1: PX0, x2: PX0 + PS, y1: P(py(0)), y2: P(py(0)), stroke: LINE, 'stroke-dasharray': '2 4' }, plotG);
  el('line', { x1: P(px(0)), x2: P(px(0)), y1: PY0, y2: PY0 + PS, stroke: LINE, 'stroke-dasharray': '2 4' }, plotG);
  const inner = el('g', { 'clip-path': 'url(#dif-clip)' }, plotG);
  // The data, faint and hollow, behind the samples and the example.
  const dataG = el('g', {}, inner);
  for (const p of spiral()) el('circle', { cx: P(px(p[0])), cy: P(py(p[1])), r: 2.2, fill: 'none', stroke: MUTED, 'stroke-opacity': 0.45 }, dataG);
  const trail = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-opacity': 0.12, 'stroke-width': 0.6 }, inner);
  const pts = Array.from({ length: N }, () => el('circle', { r: 2, fill: CYAN }, inner));
  // Step 2's centres and circles.
  const circG = el('g', {}, inner);
  const marks = MARKED.map(() => ({
    from: el('circle', { r: 4, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, circG),
    line: el('line', { stroke: STRONG, 'stroke-width': 1.25, 'marker-end': 'url(#dif-arrow-s)' }, circG),
    c1: el('circle', { fill: 'none', stroke: VIOLET, 'stroke-width': 1.5 }, circG),
    c2: el('circle', { fill: 'none', stroke: VIOLET, 'stroke-width': 1.25, 'stroke-dasharray': '4 3' }, circG),
    centre: el('path', { d: '', stroke: VIOLET, 'stroke-width': 2 }, circG),
  }));
  // Step 3's example.
  const exG = el('g', {}, inner);
  const exFrom = el('rect', { width: 9, height: 9, fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, exG);
  const exTo = el('circle', { r: 5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, exG);
  const exLine = el('line', { stroke: MUTED, 'stroke-dasharray': '3 3' }, exG);
  const plotLegend = el('g', {}, svg);
  const legendRows = [label(plotLegend, PX0 + 6, PY0 + 16, ''), label(plotLegend, PX0 + 6, PY0 + 32, '')];

  // ---- steps 1 and 2: √ᾱₜ and √(1 − ᾱₜ) against t.
  const curveG = el('g', {}, svg);
  const CX0 = 360;
  const CX1 = 580;
  const CY0 = 238;
  const CY1 = 62;
  const cx = (u) => CX0 + (u / T) * (CX1 - CX0);
  const cy = (v) => CY0 + v * (CY1 - CY0);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, curveG);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1 - 6, stroke: LINE }, curveG);
  for (const v of [0, 0.5, 1]) label(curveG, CX0 - 6, P(cy(v)) + 4, String(v), 'end', MUTED, 11);
  for (const u of [0, 25, 50]) label(curveG, P(cx(u)), CY0 + 16, String(u), 'middle', MUTED, 11);
  label(curveG, (CX0 + CX1) / 2, CY0 + 32, 'step t', 'middle', MUTED, 11);
  const ts = Array.from({ length: T + 1 }, (_, i) => i);
  el('path', { d: ts.map((u, i) => `${i ? 'L' : 'M'}${P(cx(u))},${P(cy(signal(u)))}`).join(''), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, curveG);
  el('path', { d: ts.map((u, i) => `${i ? 'L' : 'M'}${P(cx(u))},${P(cy(noiseSd(u)))}`).join(''), fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, curveG);
  label(curveG, CX0 + 8, 22, 'signal √ᾱₜ (solid)', 'start', CYAN, 12);
  label(curveG, CX0 + 8, 40, 'noise SD √(1 − ᾱₜ) (dashed)', 'start', VIOLET, 12);
  const tLine = el('line', { y1: CY0, y2: CY1, stroke: MUTED, 'stroke-dasharray': '2 3' }, curveG);
  const dotS = el('circle', { r: 4.5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, curveG);
  const dotN = el('rect', { width: 8, height: 8, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, curveG);

  // ---- step 3: the noise plane, ε and ε̂ as arrows from the origin.
  const nzG = el('g', {}, svg);
  const NC = [470, 140];
  const NU = 40;
  const NR = 2.75;
  el('rect', { x: NC[0] - NR * NU, y: NC[1] - NR * NU, width: 2 * NR * NU, height: 2 * NR * NU, fill: 'none', stroke: LINE }, nzG);
  el('line', { x1: NC[0] - NR * NU, x2: NC[0] + NR * NU, y1: NC[1], y2: NC[1], stroke: LINE, 'stroke-dasharray': '2 4' }, nzG);
  el('line', { x1: NC[0], x2: NC[0], y1: NC[1] - NR * NU, y2: NC[1] + NR * NU, stroke: LINE, 'stroke-dasharray': '2 4' }, nzG);
  label(nzG, NC[0] - NR * NU, 14, 'the noise and the guess', 'start', STRONG);
  const nclip = el('clipPath', { id: 'dif-nclip' }, defs);
  el('rect', { x: NC[0] - NR * NU, y: NC[1] - NR * NU, width: 2 * NR * NU, height: 2 * NR * NU }, nclip);
  const nzIn = el('g', { 'clip-path': 'url(#dif-nclip)' }, nzG);
  const nzErr = el('line', { stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '2 3' }, nzIn);
  const nzEps = el('line', { x1: NC[0], y1: NC[1], stroke: STRONG, 'stroke-width': 2, 'stroke-dasharray': '6 3', 'marker-end': 'url(#dif-arrow-s)' }, nzIn);
  const nzHat = el('line', { x1: NC[0], y1: NC[1], stroke: CYAN, 'stroke-width': 2.5, 'marker-end': 'url(#dif-arrow-c)' }, nzIn);
  const nzLab = [label(nzG, 0, 0, 'ε (dashed)', 'start', STRONG, 12), label(nzG, 0, 0, 'ε̂ (solid)', 'start', CYAN, 12)];

  // ---- step 4: the loss curve.
  const lossG = el('g', {}, svg);
  const LX0 = 70;
  const LX1 = 580;
  const LY0 = 240;
  const LY1 = 30;
  const LMAX = 3;
  const lx = (i) => LX0 + (i / STEPS) * (LX1 - LX0);
  const ly = (v) => LY0 + (Math.min(v, LMAX) / LMAX) * (LY1 - LY0);
  el('line', { x1: LX0, x2: LX1, y1: LY0, y2: LY0, stroke: LINE }, lossG);
  el('line', { x1: LX0, x2: LX0, y1: LY0, y2: LY1 - 6, stroke: LINE }, lossG);
  for (const v of [0, 1, 2, 3]) label(lossG, LX0 - 8, P(ly(v)) + 4, String(v), 'end', MUTED, 11);
  for (const i of [0, 2000, 4000, 6000]) label(lossG, P(lx(i)), LY0 + 16, count(i), 'middle', MUTED, 11);
  label(lossG, (LX0 + LX1) / 2, LY0 + 32, 'training step', 'middle', MUTED, 11);
  label(lossG, LX0, LY1 - 14, `loss: mean ‖ε − ε̂‖² on ${EVAL_N} fixed rows`, 'start', STRONG);
  const zy = P(ly(zeroGuessLoss()));
  el('line', { x1: LX0, x2: LX1, y1: zy, y2: zy, stroke: MUTED, 'stroke-dasharray': '5 4' }, lossG);
  label(lossG, LX1, zy - 6, 'guess 0 every time', 'end', MUTED, 11);
  const lossPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, lossG);
  const lossDot = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, lossG);

  // ---- step 5: precision and coverage bars.
  const qG = el('g', {}, svg);
  const QB0 = 432;
  const QW = 140;
  const QMAX = 0.25;
  label(qG, 312, 18, 'mean distance (smaller is closer)', 'start', STRONG);
  label(qG, QB0, 36, 'top: new → data', 'start', MUTED, 11);
  label(qG, QB0, 50, 'bottom: data → new', 'start', MUTED, 11);
  const qNames = ['pure noise', 'mixture, K = 5', 'diffusion', 'fresh spiral'];
  const qRows = qNames.map((name, i) => {
    const y = 66 + i * 50;
    label(qG, QB0 - 8, y + 21, name, 'end', i === 2 ? CYAN : STRONG, 12);
    const mkBar = (yy, dashed) => {
      el('rect', { x: QB0, y: yy, width: QW, height: 16, fill: 'none', stroke: LINE }, qG);
      const r = el('rect', { x: QB0, y: yy, width: 0, height: 16, fill: i === 2 ? CYAN : MUTED, 'fill-opacity': i === 2 ? 0.55 : 0.3, stroke: i === 2 ? CYAN : MUTED, ...(dashed ? { 'stroke-dasharray': '3 2' } : {}) }, qG);
      const v = label(qG, QB0 + 4, yy + 12, '', 'start', STRONG, 11);
      return { r, v };
    };
    return [mkBar(y, false), mkBar(y + 18, true)];
  });

  // ---- step 6: the three steps.
  const sumG = el('g', {}, svg);
  const boxes = [
    ['1 add noise', 'a fixed recipe:', 'no learning'],
    ['2 learn to guess it', 'a network predicts', 'the noise in xₜ'],
    ['3 remove it, T times', 'start from pure noise', 'and step back to t = 0'],
  ];
  boxes.forEach((b, i) => {
    const x = 20 + i * 196;
    el('rect', { x, y: 60, width: 172, height: 120, rx: 6, fill: 'none', stroke: i === 1 ? CYAN : LINE, 'stroke-width': i === 1 ? 2 : 1 }, sumG);
    label(sumG, x + 12, 92, b[0], 'start', STRONG, 14);
    label(sumG, x + 12, 124, b[1], 'start', MUTED, 12);
    label(sumG, x + 12, 144, b[2], 'start', MUTED, 12);
    if (i < 2) el('line', { x1: x + 176, x2: x + 192, y1: 120, y2: 120, stroke: MUTED, 'stroke-width': 1.5, 'marker-end': 'url(#dif-arrow-m)' }, sumG);
  });
  label(sumG, 20, 220, 'Picture-making diffusion models: far bigger networks,', 'start', STRONG, 13);
  label(sumG, 20, 240, 'images instead of points, often a text description as input.', 'start', STRONG, 13);

  const rowsT = RY.map((y, i) => label(svg, 16, y, '', 'start', i === 0 ? STRONG : MUTED, 13));

  // ---- controls: the t slider (steps 1 to 3), Train (step 4), Generate (step 5).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const tGroup = make('div');
  const tLab = make('label');
  const tIn = make('input', { type: 'range', min: '0', max: String(T), step: '1', value: String(t) });
  const tOut = make('output');
  tOut.setAttribute('aria-hidden', 'true');
  tLab.append('noise step t', ' ', tIn, ' ', tOut);
  tGroup.append(tLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the noise-guessing network');
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  const genGroup = make('div');
  genGroup.setAttribute('role', 'group');
  genGroup.setAttribute('aria-label', 'generate new points');
  Object.assign(genGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  const genWrap = make('div', { className: 'train-buttons' });
  const genBtn = make('button', { type: 'button', textContent: 'Generate' });
  genWrap.append(genBtn);
  genGroup.append(genWrap);
  controls.append(tGroup, trainGroup, genGroup);

  // The t used by step 3 is at least 1: at t = 0 there is no noise to guess.
  const tExample = () => Math.max(T_EXAMPLE_MIN, t);

  let mixJob = null;
  const mixQ = (press) => (mixReady() ? quality(mixSamples(press)) : null);

  function info() {
    if (step === 0 || step === 1) return { t };
    if (step === 2) {
      const lv = getLive();
      return { ex: example(lv.trainer.state.net, tExample()), iter: lv.trainer.state.iter };
    }
    if (step === 3) return liveInfo(getLive(), trainLoop.activity());
    if (step === 4) {
      const g = genInfo(gen, gen ? mixQ(gen.press) : null);
      g.netIter = getLive().trainer.state.iter;
      return g;
    }
    return {};
  }

  const drawPoints = (P2) => {
    for (let i = 0; i < N; i++) {
      pts[i].setAttribute('cx', P(px(P2[i][0])));
      pts[i].setAttribute('cy', P(py(P2[i][1])));
    }
  };

  const setBar = (b, v) => {
    if (v === null || v === undefined) { b.r.setAttribute('width', 0); b.v.textContent = '…'; b.v.setAttribute('x', QB0 + 4); return; }
    const w = Math.min(v / QMAX, 1) * QW;
    b.r.setAttribute('width', P(w));
    b.v.textContent = dist(v);
    b.v.setAttribute('x', P(QB0 + Math.min(w, QW - 30) + 4));
  };

  function render() {
    const s = info();
    show(plotG, step <= 2 || step === 4);
    show(plotLegend, step <= 2 || step === 4);
    show(curveG, step <= 1);
    show(nzG, step === 2);
    show(lossG, step === 3);
    show(qG, step === 4);
    show(sumG, step === 5);
    show(circG, step === 1);
    show(exG, step === 2);
    show(trail, step === 4);
    const ptsOn = step <= 1 || (step === 4 && gen);
    for (const c of pts) show(c, ptsOn);
    tOut.textContent = String(step === 2 ? tExample() : t);
    tIn.setAttribute('aria-valuetext', `t = ${step === 2 ? tExample() : t} of ${T}`);

    if (step <= 1) {
      drawPoints(forwardPoints(t));
      legendRows[0].textContent = `cyan: the ${N} points at step t = ${t}`;
      legendRows[1].textContent = step === 1 ? 'violet: centre √ᾱₜ·x₀ and 1, 2 SD' : 'grey hollow: the data, t = 0';
      tLine.setAttribute('x1', P(cx(t)));
      tLine.setAttribute('x2', P(cx(t)));
      dotS.setAttribute('cx', P(cx(t)));
      dotS.setAttribute('cy', P(cy(signal(t))));
      dotN.setAttribute('x', P(cx(t)) - 4);
      dotN.setAttribute('y', P(cy(noiseSd(t))) - 4);
    }
    if (step === 1) {
      const a = signal(t);
      const sd = noiseSd(t) * (PS / (2 * R));
      MARKED.forEach((idx, j) => {
        const m = marks[j];
        const x0 = spiral()[idx];
        const c = [a * x0[0], a * x0[1]];
        m.from.setAttribute('cx', P(px(x0[0])));
        m.from.setAttribute('cy', P(py(x0[1])));
        m.line.setAttribute('x1', P(px(x0[0])));
        m.line.setAttribute('y1', P(py(x0[1])));
        m.line.setAttribute('x2', P(px(c[0])));
        m.line.setAttribute('y2', P(py(c[1])));
        show(m.line, t > 0);
        m.c1.setAttribute('cx', P(px(c[0])));
        m.c1.setAttribute('cy', P(py(c[1])));
        m.c1.setAttribute('r', P(sd));
        m.c2.setAttribute('cx', P(px(c[0])));
        m.c2.setAttribute('cy', P(py(c[1])));
        m.c2.setAttribute('r', P(2 * sd));
        const X = P(px(c[0]));
        const Y = P(py(c[1]));
        m.centre.setAttribute('d', `M${X - 5},${Y - 5}L${X + 5},${Y + 5}M${X - 5},${Y + 5}L${X + 5},${Y - 5}`);
      });
    }
    if (step === 2) {
      const e = s.ex;
      exFrom.setAttribute('x', P(px(e.x0[0])) - 4.5);
      exFrom.setAttribute('y', P(py(e.x0[1])) - 4.5);
      exTo.setAttribute('cx', P(px(e.xt[0])));
      exTo.setAttribute('cy', P(py(e.xt[1])));
      exLine.setAttribute('x1', P(px(e.x0[0])));
      exLine.setAttribute('y1', P(py(e.x0[1])));
      exLine.setAttribute('x2', P(px(e.xt[0])));
      exLine.setAttribute('y2', P(py(e.xt[1])));
      legendRows[0].textContent = 'square: data point x₀';
      legendRows[1].textContent = `cyan dot: xₜ at t = ${e.t}`;
      const tip = (v) => [P(NC[0] + v[0] * NU), P(NC[1] - v[1] * NU)];
      const [ex, ey] = tip(e.eps);
      const [hx, hy] = tip(e.guess);
      nzEps.setAttribute('x2', ex);
      nzEps.setAttribute('y2', ey);
      nzHat.setAttribute('x2', hx);
      nzHat.setAttribute('y2', hy);
      nzErr.setAttribute('x1', ex);
      nzErr.setAttribute('y1', ey);
      nzErr.setAttribute('x2', hx);
      nzErr.setAttribute('y2', hy);
      nzLab[0].setAttribute('x', NC[0] - NR * NU + 6);
      nzLab[0].setAttribute('y', NC[1] + NR * NU - 24);
      nzLab[1].setAttribute('x', NC[0] - NR * NU + 6);
      nzLab[1].setAttribute('y', NC[1] + NR * NU - 8);
    }
    if (step === 3) {
      const ev = getLive().trainer.state.evalLoss;
      lossPath.setAttribute('d', ev.map((v, i) => `${i ? 'L' : 'M'}${P(lx(i * EVAL_EVERY))},${P(ly(v))}`).join(''));
      lossDot.setAttribute('cx', P(lx((ev.length - 1) * EVAL_EVERY)));
      lossDot.setAttribute('cy', P(ly(ev.at(-1))));
    }
    if (step === 4) {
      legendRows[0].textContent = gen ? `cyan: ${N} new points, t = ${gen.tNow()}` : 'grey hollow: the 400 data points';
      legendRows[1].textContent = gen ? 'grey: the data · lines: their paths' : 'press Generate';
      if (gen) {
        const now = gen.points();
        drawPoints(now);
        const k0 = Math.min(Math.floor(gen.u()), gen.sampler.k);
        const path = gen.sampler.path;
        let d = '';
        for (let i = 0; i < N; i++) {
          d += `M${P(px(path[0][i][0]))},${P(py(path[0][i][1]))}`;
          for (let k = 1; k <= k0; k++) d += `L${P(px(path[k][i][0]))},${P(py(path[k][i][1]))}`;
          d += `L${P(px(now[i][0]))},${P(py(now[i][1]))}`;
        }
        trail.setAttribute('d', d);
      } else trail.setAttribute('d', '');
      const fresh = cached('fq', () => quality(freshSpiral()));
      const rows = [s.gen ? s.noise : null, s.gen ? s.mix : null, s.gen && s.done ? s.samples : null, fresh];
      rows.forEach((q, i) => {
        setBar(qRows[i][0], q ? q.precision : null);
        setBar(qRows[i][1], q ? q.coverage : null);
      });
    }

    const texts = readout(step, s);
    rowsT.forEach((n, i) => { n.textContent = texts[i] || ''; });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `A scatter of the ${N} spiral points after ${t} noise steps, and two curves against t: the signal √ᾱₜ falls from 1 towards 0 and the noise standard deviation rises from 0 towards 1. ${rest}.`;
    if (step === 1) return `The noisy points at step ${t}, with three data points marked: an arrow from each to its shrunk centre, and circles of 1 and 2 standard deviations around the centre. ${rest}.`;
    if (step === 2) return `One data point and its noisy version at step ${tExample()}. Right: the true noise and the network's guess as two arrows from the origin, joined by the error. ${rest}.`;
    if (step === 3) return `The training loss against training step, with a dashed line for always guessing 0. ${rest}.`;
    if (step === 4) {
      const bars = s.gen && s.done && s.mix ? ` Bars, new to data then data to new: noise ${dist(s.noise.precision)}, ${dist(s.noise.coverage)}; mixture ${dist(s.mix.precision)}, ${dist(s.mix.coverage)}; diffusion ${dist(s.samples.precision)}, ${dist(s.samples.coverage)}.` : '';
      return `The spiral data in grey, and ${N} new points walking back from pure noise with faint trails.${bars} ${rest}.`;
    }
    return `Three boxes: add noise, learn to guess it, remove it step by step. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  // ---- step 4's training.
  const trainLoop = createTrainLoop({
    step: (n) => getLive().advance(n),
    render,
    done: () => getLive().done(),
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done()) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, trainLoop, {
    onReset: () => getLive().reset(),
    reduced,
    work: (m) => getLive().advance(m),
  });
  trainGroup.addEventListener('click', () => { if (step === 3) { render(); announce(spoken()); } });

  // ---- step 5's Generate: one reverse step every GEN_FRAMES frames, drawn in between.
  const genLoop = createTrainLoop({
    step: () => gen.frame(),
    render,
    done: () => !gen || gen.done(),
    onStop: () => {
      syncGen();
      render();
      if (gen && gen.done()) announce(`Finished. ${spoken()}`);
    },
  });
  const syncGen = () => {
    genBtn.textContent = genLoop.running() ? 'Pause' : gen && !gen.done() ? 'Continue' : 'Generate';
  };
  const stopGen = () => { genLoop.pause(); syncGen(); };
  genBtn.addEventListener('click', () => {
    if (genLoop.running()) { stopGen(); render(); return; }
    if (!gen || gen.done()) {
      presses++;
      const st = getLive().trainer.state;
      gen = makeGen(st.net, presses, st.iter);
    }
    if (reduced) genLoop.start({ quietStep: () => gen.quiet() });
    else genLoop.start();
    syncGen();
    render();
    announce(spoken());
  });

  // The mixture is fitted the first time step 5 opens, in slices.
  const fitMix = () => {
    if (mixJob !== null || mixReady()) return;
    const tick = () => {
      mixJob = null;
      const t0 = performance.now();
      while (performance.now() - t0 < 6 && !mixWork(5));
      if (mixReady()) render(); else mixJob = requestAnimationFrame(tick);
    };
    mixJob = requestAnimationFrame(tick);
  };

  tIn.addEventListener('input', () => {
    t = Math.min(T, Math.max(0, Number(tIn.value) || 0));
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
      stopGen();
      step = i;
      if (i === 4) fitMix();
      tGroup.style.display = i <= 2 ? 'contents' : 'none';
      trainGroup.style.display = i === 3 ? 'flex' : 'none';
      genGroup.style.display = i === 4 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 3 || i === 4);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
