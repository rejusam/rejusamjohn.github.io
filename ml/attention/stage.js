// Lesson L5.4, "Attention": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-attention.test.mjs); the
// DOM is only touched inside createStage. Every forward pass is attnForward from
// attn.js, every training run is attnTrainer, and every number in the page's text
// comes from pageValues or storyValues.

import { echoData, rnnInit, rnnFlatten } from '../assets/rnn.js';
import { attnInit, attnForward, attnTrainer, attnFlatten, attnUnflatten, attnParamCount, scaledDot } from '../assets/attn.js';
import { softmax } from '../assets/text.js';
import { el, heatmap, draggable, clamp } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- steps 2 and 3: four hand-placed keys and values in two dimensions ----------

// Key i and value i belong together, like a label and what is stored under it.
export const KEYS = [[2, 0], [0, 2], [-2, 0], [0, -2]];
export const VALUES = [[2, 2], [-2, 2], [-2, -2], [2, -2]];
export const Q_START = [1.5, 0.5];
export const Q_RANGE = 3;
export const DK_TOY = 2;

// Scores q · k / √d_k and their softmax: the weights drawn as bars.
export function queryWeights(q) {
  const scores = KEYS.map((k) => scaledDot(q, k));
  return { scores, weights: softmax(scores) };
}

// The weighted average of the values.
export function attendOutput(weights) {
  const o = [0, 0];
  weights.forEach((w, i) => { o[0] += w * VALUES[i][0]; o[1] += w * VALUES[i][1]; });
  return o;
}

// A dragged query is kept on a 0.1 grid, inside the plotted square.
export const snapQuery = (a, b) => [a, b].map((v) => Math.round(clamp(v, -Q_RANGE, Q_RANGE) * 10) / 10);

// ---- the echo task and the model -----------------------------------------------------

export const SEED_TRAIN = 211;
export const SEED_TEST = 212;
export const SEED_INIT = 271;
export const SEED_SHUFFLE = 272;
export const N_TRAIN = 200;
export const N_TEST = 200;
export const T = 12;
export const D = 16;
export const ETA = 0.02;
export const BATCH = 10;
export const EPOCHS = 10;
export const ITERS = (EPOCHS * N_TRAIN) / BATCH;
export const BATCHES_PER_EPOCH = N_TRAIN / BATCH;
export const K_LIVE = 8;
export const K_CHOICES = [1, 2, 4, 8];
export const EXAMPLE = 0;
export const CFG = { vocab: 2, len: T, d: D, heads: 1, causal: true, ln: false, mlp: 0, out: 'logit', seed: SEED_INIT };

const DATA = new Map();
const cached = (key, make) => {
  if (!DATA.has(key)) DATA.set(key, make());
  return DATA.get(key);
};
export const trainSet = (k) => cached(`tr${k}`, () => echoData({ n: N_TRAIN, T, k, seed: SEED_TRAIN }));
export const testSet = (k) => cached(`te${k}`, () => echoData({ n: N_TEST, T, k, seed: SEED_TEST }));
export const startModel = () => attnInit(CFG);
export const maskedCount = (k) => N_TEST * (T - k);
// The L4 recurrent network's size, for the comparison on the page (8 hidden units).
export const rnnParamCount = () => rnnFlatten(rnnInit({ nh: 8, seed: 213 })).length;
// Allowed (query, key) pairs in one stream: causal T(T + 1)/2 of all T².
export const causalPairs = (len = T) => (len * (len + 1)) / 2;

// σ(z) > 0.5 exactly when z > 0; p̂ = 0.5 predicts 0, as in the recurrent lesson.
export const predictBit = (z) => (z > 0 ? 1 : 0);

export function maskedAccuracy(model, Dset) {
  let ok = 0;
  let n = 0;
  for (let i = 0; i < Dset.seqs.length; i++) {
    const { logits } = attnForward(model, Dset.seqs[i]);
    for (let t = 0; t < T; t++) {
      if (!Dset.mask[i][t]) continue;
      n++;
      if (predictBit(logits[t]) === Dset.targets[i][t]) ok++;
    }
  }
  return { ok, n };
}

// The attention map of one stream: row t holds the weights over s ≤ t, and null for
// s > t (left out of the softmax, drawn hatched).
export function attentionMap(model, xs) {
  const A = attnForward(model, xs).A[0][0];
  return A.map((row, t) => row.map((w, s) => (s <= t ? w : null)));
}

// Over every row with a target (t ≥ k, zero-based) of every stream: in how many is the
// largest weight at s = t − k (ties go to the earliest s), and the mean weight there.
export function bandStats(model, Dset, k) {
  let hit = 0;
  let n = 0;
  let w = 0;
  for (const xs of Dset.seqs) {
    const A = attnForward(model, xs).A[0][0];
    for (let t = k; t < T; t++) {
      let best = 0;
      for (let s = 1; s <= t; s++) if (A[t][s] > A[t][best]) best = s;
      n++;
      if (best === t - k) hit++;
      w += A[t][t - k];
    }
  }
  return { hit, n, mean: w / n };
}

// One stream's map: of its rows with a target, how many have their largest weight at
// s = t − k (ties to the earliest s). The text alternative of the live map.
export function streamBand(model, xs, k) {
  const A = attnForward(model, xs).A[0][0];
  let hit = 0;
  for (let t = k; t < T; t++) {
    let best = 0;
    for (let s = 1; s <= t; s++) if (A[t][s] > A[t][best]) best = s;
    if (best === t - k) hit++;
  }
  return { hit, n: T - k };
}

// The same model with every position embedding set to 0.
export function withoutPositions(model) {
  const m = attnUnflatten(model, attnFlatten(model));
  m.P.fill(0);
  return m;
}

export const makeTrainer = (k, model = startModel()) =>
  attnTrainer({ model, data: trainSet(k), batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });

// Step 5's live run at k = 8: the trainer plus the masked test count after every batch.
export function makeLive(k = K_LIVE) {
  const te = testSet(k);
  const tr = makeTrainer(k);
  const live = { k, trainer: tr, n: maskedCount(k), testOk: [maskedAccuracy(tr.state.model, te).ok] };
  live.advance = (m) => {
    for (let i = 0; i < m && tr.state.iter < ITERS && !tr.state.diverged; i++) {
      tr.step(1);
      live.testOk.push(maskedAccuracy(tr.state.model, te).ok);
    }
  };
  live.reset = () => {
    tr.reset();
    live.testOk = [live.testOk[0]];
  };
  live.done = () => tr.state.iter >= ITERS || tr.state.diverged;
  return live;
}

// Pacing of the live run (frames only; the run itself is unchanged): during the first
// SLOW_EPOCHS epochs one batch every SLOW_FRAMES frames, so the band can be watched
// forming, then one batch per frame.
export const SLOW_FRAMES = 6;
export const SLOW_EPOCHS = 2;
export function paceFrame(iter, waited) {
  if (iter >= SLOW_EPOCHS * BATCHES_PER_EPOCH) return { take: 1, waited: 0 };
  if (waited <= 0) return { take: 1, waited: SLOW_FRAMES - 1 };
  return { take: 0, waited: waited - 1 };
}

// ---- step 6: one run per k, computed on demand, in slices -----------------------------

const RUNS = new Map();
const JOBS = new Map();

export function runWork(k, steps) {
  if (RUNS.has(k)) return true;
  if (!JOBS.has(k)) {
    const t = makeTrainer(k);
    JOBS.set(k, { t, testOk: [maskedAccuracy(t.state.model, testSet(k)).ok] });
  }
  const job = JOBS.get(k);
  const { t } = job;
  for (let i = 0; i < steps && t.state.iter < ITERS && !t.state.diverged; i++) {
    const e = t.state.epoch;
    t.step(1);
    if (t.state.epoch > e) job.testOk.push(maskedAccuracy(t.state.model, testSet(k)).ok);
  }
  if (t.state.iter < ITERS && !t.state.diverged) return false;
  const s = t.state;
  RUNS.set(k, { k, losses: s.epochLoss.slice(), testOk: job.testOk.slice(), n: maskedCount(k), model: s.model, diverged: s.diverged });
  JOBS.delete(k);
  return true;
}
export const runReady = (k) => RUNS.has(k);
export function runResult(k) {
  runWork(k, Infinity);
  return RUNS.get(k);
}
const readyRuns = () => (K_CHOICES.every(runReady) ? Object.fromEntries(K_CHOICES.map((k) => [k, RUNS.get(k)])) : null);

// ---- small helpers and formatting --------------------------------------------------------

export const firstAt = (ok, n, level) => ok.findIndex((v) => v / n >= level);
export function allRightFrom(ok, n) {
  let i = ok.length;
  while (i > 0 && ok[i - 1] === n) i--;
  return i < ok.length ? i : -1;
}
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
export function sig2(v) {
  const d = Math.max(0, 1 - Math.floor(Math.log10(Math.abs(v))));
  return num(Number(v.toPrecision(2)), d);
}
export function pctCount(ok, n) {
  const t = Math.round((ok * 1000) / n);
  return `${t % 10 === 0 ? String(t / 10) : (t / 10).toFixed(1)}%`;
}
export const lossText = (v) => (v >= 0.01 ? num(v, 3) : sig2(v));
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const vec = (v) => `(${num(v[0], 1)}, ${num(v[1], 1)})`;
const vec2 = (v) => `(${num(v[0], 2)}, ${num(v[1], 2)})`;

// ---- read-outs ---------------------------------------------------------------------------

export function liveInfo(live, running) {
  const s = live.trainer.state;
  const band = streamBand(s.model, testSet(live.k).seqs[EXAMPLE], live.k);
  return { iter: s.iter, epoch: s.epoch, ok: live.testOk.at(-1), n: live.n, loss: s.epochLoss.at(-1), running, diverged: s.diverged, done: live.done(), model: s.model, k: live.k, band };
}

export function readout(step, s) {
  if (step === 0) return [`test stream ${EXAMPLE + 1}, k = ${K_LIVE}`, `RNN: x₁ goes through ${K_LIVE} updates`, `attention: step ${K_LIVE + 1} looks at step 1 directly`];
  if (step === 1) {
    const top = s.weights.indexOf(Math.max(...s.weights));
    return [`query q = ${vec(s.q)} · scores q·k/√2`, `weights add up to ${num(s.weights.reduce((a, b) => a + b, 0), 2)}`, `largest: k${'₁₂₃₄'[top]}, weight ${num(s.weights[top], 2)}`];
  }
  if (step === 2) return [`query q = ${vec(s.q)}`, `output = Σ weight × value = ${vec2(s.out)}`, 'always inside the square of the values'];
  if (step === 3) return ['starting model, not trained · stream 1', `row t: weights over positions 1 to t`, `largest at k = ${K_LIVE} back: ${pctCount(s.band.hit, s.band.n)} of ${count(s.band.n)} rows`];
  if (step === 4) {
    return [`batch ${s.iter} of ${ITERS} · ${trainStatus(s)}`, `test accuracy ${pctCount(s.ok, s.n)} · loss ℒ ${lossText(s.loss)}`, `stream 1: largest ${s.k} back in ${s.band.hit} of ${s.band.n} rows`];
  }
  if (step === 5) {
    if (!s) return ['computing the four runs…'];
    return [`k = ${s.k} · ${EPOCHS} epochs from the same start`, `test: ${pctCount(s.ok, s.n)} (${count(s.ok)} of ${count(s.n)} masked steps)`, `largest weight at ${s.k} back: ${pctCount(s.band.hit, s.band.n)} of rows`];
  }
  return [];
}

// ---- values written into the page ------------------------------------------------------------

export function pageValues() {
  const w = queryWeights(Q_START);
  const o = attendOutput(w.weights);
  return {
    'n-params': count(attnParamCount(startModel())),
    'rnn-params': String(rnnParamCount()),
    'masked-k8': count(maskedCount(8)),
    'causal-pairs': String(causalPairs()),
    'all-pairs': String(T * T),
    'q0-w1': num(w.weights[0], 2),
    'q0-w2': num(w.weights[1], 2),
    'q0-s1': num(w.scores[0], 2),
    'q0-out': vec2(o),
  };
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const live = makeLive(K_LIVE);
  while (!live.done()) live.advance(ITERS);
  const ok = live.testOk;
  const n = live.n;
  v['start-acc'] = pctCount(ok[0], n);
  const f = firstAt(ok, n, 0.95);
  v['first-95'] = String(f);
  v['first-95-acc'] = pctCount(ok[f], n);
  v['all-right'] = String(allRightFrom(ok, n));
  v['end-acc'] = pctCount(ok.at(-1), n);
  const te = testSet(K_LIVE);
  const b0 = bandStats(startModel(), te, K_LIVE);
  v['start-band'] = pctCount(b0.hit, b0.n);
  const b = bandStats(live.trainer.state.model, te, K_LIVE);
  v['band-share'] = pctCount(b.hit, b.n);
  v['band-mean'] = num(b.mean, 2);
  const np = maskedAccuracy(withoutPositions(live.trainer.state.model), te);
  v['nopos-acc'] = pctCount(np.ok, np.n);
  const runs = Object.fromEntries(K_CHOICES.map((k) => [k, runResult(k)]));
  for (const k of K_CHOICES) v[`acc-k${k}`] = pctCount(runs[k].testOk[EPOCHS], runs[k].n);
  // The last epoch, over every k, from which every masked test step is right.
  v['all-k-epoch'] = String(Math.max(...K_CHOICES.map((k) => allRightFrom(runs[k].testOk, runs[k].n))));
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
const SUB = '₁₂₃₄';

// Step 1's columns.
const CX0 = 96;
const CW = 41;
const colX = (t) => CX0 + (t - 1) * CW;

// The two toy planes: 36 px per unit, 6 units across.
const UNIT = 36;
const LC = [160, 144];
const RC = [452, 144];

// The attention map (steps 4 to 6).
const HX = 70;
const HY = 52;
const CELL = 18;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let q = Q_START.slice();
  let kSel = K_CHOICES.indexOf(K_LIVE);
  let live = null;

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
  const mk = (id, fill) => {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill }, m);
  };
  mk('att-arrow', MUTED);
  mk('att-arrow-c', CYAN);
  mk('att-arrow-s', STRONG);
  const arrow = (parent, x1, y1, x2, y2, extra = {}) => el('line', { x1, y1, x2, y2, stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#att-arrow)', ...extra }, parent);

  const bitCell = (parent, x, y, w = 28, h = 24, size = 13) => {
    const r = el('rect', { x, y, width: w, height: h, rx: 3, fill: CYAN, 'fill-opacity': 0, stroke: LINE }, parent);
    const t = label(parent, x + w / 2, y + h / 2 + size * 0.38, '', 'middle', STRONG, size);
    return {
      text: t,
      set(v) {
        r.setAttribute('fill-opacity', v === 1 ? 0.3 : 0);
        t.textContent = v === undefined ? '' : v === null ? '–' : String(v);
        t.style.fill = v === null ? MUTED : STRONG;
      },
    };
  };

  // ---- step 1: the stream, its k = 8 targets, and the two routes from x₁ to step 9.
  const probG = el('g', {}, svg);
  {
    const D8 = testSet(K_LIVE);
    const xs = D8.seqs[EXAMPLE];
    for (let t = 1; t <= T; t++) label(probG, colX(t) + 14, 18, String(t), 'middle');
    label(probG, 16, 18, 't');
    label(probG, 16, 43, 'x', 'start', STRONG, 13);
    label(probG, 16, 71, 'target', 'start', STRONG, 13);
    for (let t = 1; t <= T; t++) {
      bitCell(probG, colX(t), 26).set(xs[t - 1]);
      bitCell(probG, colX(t), 54).set(D8.mask[EXAMPLE][t - 1] ? D8.targets[EXAMPLE][t - 1] : null);
    }
    const t9 = K_LIVE + 1;
    for (const [cx, cy] of [[colX(1), 26], [colX(t9), 54]]) el('rect', { x: cx - 2, y: cy - 2, width: 32, height: 28, rx: 4, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, probG);
    // Recurrent route: one hidden state per step, the bit passed along 8 links.
    label(probG, 16, 122, 'recurrent', 'start', STRONG, 13);
    label(probG, 16, 138, 'network');
    const yR = 128;
    for (let t = 1; t <= T; t++) {
      const on = t <= t9;
      el('circle', { cx: colX(t) + 14, cy: yR, r: 7, fill: BG, stroke: on ? CYAN : MUTED, 'stroke-width': on ? 2 : 1 }, probG);
      if (t < T) arrow(probG, colX(t) + 22, yR, colX(t + 1) + 5, yR, t < t9 ? { stroke: CYAN, 'stroke-width': 2, 'marker-end': 'url(#att-arrow-c)' } : {});
    }
    label(probG, colX(1) + 14, yR + 26, `${K_LIVE} updates between reading x₁ and step ${t9}`, 'start');
    // Attention route: step 9 reads position 1 in one step.
    label(probG, 16, 214, 'attention', 'start', STRONG, 13);
    const yA = 220;
    for (let t = 1; t <= T; t++) el('circle', { cx: colX(t) + 14, cy: yA, r: 7, fill: BG, stroke: t === 1 || t === t9 ? CYAN : MUTED, 'stroke-width': t === 1 || t === t9 ? 2 : 1 }, probG);
    const x1 = colX(t9) + 14;
    const x0 = colX(1) + 14;
    el('path', { d: `M${x1},${yA - 9} C${x1 - 40},${yA - 64} ${x0 + 40},${yA - 64} ${x0 + 2},${yA - 10}`, fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '6 3', 'marker-end': 'url(#att-arrow-c)' }, probG);
    label(probG, colX(1) + 14, yA + 28, `step ${t9} looks straight back at position 1`, 'start');
  }

  // ---- steps 2 and 3: the query plane (left), and bars or the values plane (right).
  const toyG = el('g', {}, svg);
  const lx = (u) => LC[0] + u * UNIT;
  const ly = (u) => LC[1] - u * UNIT;
  const rx = (u) => RC[0] + u * UNIT;
  const ry = (u) => RC[1] - u * UNIT;
  // A square plane of half-width h units around (cx, cy).
  const plane = (parent, cx, cy, title, h = Q_RANGE) => {
    const r = h * UNIT;
    el('rect', { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, fill: 'none', stroke: LINE }, parent);
    el('line', { x1: cx - r, x2: cx + r, y1: cy, y2: cy, stroke: LINE, 'stroke-dasharray': '2 4' }, parent);
    el('line', { x1: cx, x2: cx, y1: cy - r, y2: cy + r, stroke: LINE, 'stroke-dasharray': '2 4' }, parent);
    label(parent, cx - r, Math.min(cy - r - 8, 28), title, 'start', STRONG);
  };
  plane(toyG, LC[0], LC[1], 'keys and the query');
  KEYS.forEach((k, i) => {
    arrow(toyG, lx(0), ly(0), lx(k[0]), ly(k[1]), { 'stroke-width': 1.5 });
    const ox = k[0] === 0 ? 14 : k[0] > 0 ? -2 : 2;
    const oy = k[1] === 0 ? -8 : k[1] > 0 ? 4 : 14;
    label(toyG, lx(k[0]) + ox, ly(k[1]) + oy, `k${SUB[i]}`, k[0] < 0 ? 'start' : k[0] > 0 ? 'end' : 'start', STRONG, 13);
  });
  const qLine = el('line', { x1: lx(0), y1: ly(0), x2: 0, y2: 0, stroke: CYAN, 'stroke-width': 2.5, 'marker-end': 'url(#att-arrow-c)' }, toyG);
  const qLab = label(toyG, 0, 0, 'q', 'start', CYAN, 14);
  const handle = el('circle', { class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'button' }, toyG);
  handle.style.cursor = 'grab';
  const handleRing = el('circle', { r: 9, fill: 'none', stroke: CYAN, 'stroke-width': 1.5, 'stroke-dasharray': '3 2' }, toyG);
  handleRing.style.pointerEvents = 'none';
  label(toyG, LC[0], LC[1] + 3 * UNIT + 16, 'drag the tip of q', 'middle');

  // Step 2's bars: one row per key, length = weight.
  const barG = el('g', {}, toyG);
  const BX = 380;
  const BWID = 190;
  label(barG, 330, 34, 'score  →  weight (softmax)', 'start', STRONG);
  const bars = KEYS.map((_, i) => {
    const y = 56 + i * 52;
    const name = label(barG, 330, y + 17, `k${SUB[i]}`, 'start', STRONG, 13);
    el('rect', { x: BX, y, width: BWID, height: 24, fill: 'none', stroke: LINE }, barG);
    const rect = el('rect', { x: BX, y, height: 24, width: 0, fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN }, barG);
    const val = label(barG, BX + 6, y + 17, '', 'start', STRONG, 12);
    const sc = label(barG, BX, y + 40, '', 'start', MUTED, 11);
    return { name, rect, val, sc };
  });

  // Step 3's values plane: each value labelled with its weight, and the output point.
  const valG = el('g', {}, toyG);
  plane(valG, RC[0], RC[1], 'values and the output', 2.5);
  const valMarks = VALUES.map((v, i) => {
    const dot = el('rect', { x: rx(v[0]) - 6, y: ry(v[1]) - 6, width: 12, height: 12, fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, valG);
    // Outside the plane, beside its corner.
    const t = label(valG, rx(v[0]) + (v[0] > 0 ? 24 : -24), ry(v[1]) + 4, '', v[0] > 0 ? 'start' : 'end', STRONG, 11);
    return { dot, t, i };
  });
  const outLines = VALUES.map(() => el('line', { stroke: CYAN, 'stroke-opacity': 0.5 }, valG));
  const outDot = el('path', { d: '', fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, valG);
  el('path', { d: `M${RC[0] - 34},${LC[1] + 3 * UNIT + 12}l5,-5l5,5l-5,5Z`, fill: CYAN }, valG);
  label(valG, RC[0] - 20, LC[1] + 3 * UNIT + 16, 'output', 'start', MUTED, 12);

  // ---- steps 4 to 6: the attention map of test stream 1, its bits above.
  const mapG = el('g', {}, svg);
  label(mapG, HX - 8, 22, 'position', 'end');
  label(mapG, HX - 8, 40, 'x', 'end', STRONG, 12);
  const headBits = [];
  // Labels that grow on a phone, where the picture is drawn at about 0.6 of its size.
  const colNums = [];
  const small = [];
  for (let s = 0; s < T; s++) {
    colNums.push(label(mapG, HX + (s + 0.5) * CELL, 22, String(s + 1), 'middle', MUTED, 11));
    headBits.push(bitCell(mapG, HX + s * CELL + 1, 28, CELL - 2, 18, 11));
    small.push(headBits[s].text);
  }
  for (let t = 0; t < T; t++) small.push(label(mapG, HX - 8, HY + (t + 0.5) * CELL + 4, String(t + 1), 'end', MUTED, 11));
  const yt = HY + 6 * CELL;
  const ytl = label(mapG, 22, yt, 'query position t', 'middle');
  ytl.setAttribute('transform', `rotate(-90 22 ${yt})`);
  const hm = heatmap(mapG, T, T, { x: HX, y: HY, cell: CELL, title: 'attention weights: row t is the query position, column s the key position' });
  // Target and right/wrong columns (steps 5 and 6).
  const TX = HX + T * CELL + 8;
  const tgtG = el('g', {}, mapG);
  label(tgtG, TX + CELL / 2, 40, 'y', 'middle', STRONG, 12);
  const tgtCells = [];
  const tgtMarks = [];
  for (let t = 0; t < T; t++) {
    tgtCells.push(bitCell(tgtG, TX + 1, HY + t * CELL + 1, CELL - 2, CELL - 2, 11));
    small.push(tgtCells[t].text);
    tgtMarks.push(label(tgtG, TX + CELL + 10, HY + (t + 0.5) * CELL + 5, '', 'middle', STRONG, 13));
  }
  // Step 4's legend.
  const legG = el('g', {}, mapG);
  {
    const x = 340;
    label(legG, x, 70, 'xₜ = E[bit] + P[t]', 'start', STRONG, 13);
    label(legG, x, 92, 'q, k, v: learned weights × xₜ', 'start', STRONG, 13);
    el('rect', { x, y: 116, width: 16, height: 16, fill: CYAN, 'fill-opacity': 0.9, stroke: LINE }, legG);
    label(legG, x + 24, 129, 'large weight');
    el('rect', { x, y: 140, width: 16, height: 16, fill: CYAN, 'fill-opacity': 0.12, stroke: LINE }, legG);
    label(legG, x + 24, 153, 'small weight');
    el('rect', { x, y: 164, width: 16, height: 16, fill: 'none', stroke: MUTED }, legG);
    el('path', { d: `M${x + 3} ${177}L${x + 13} ${167}`, stroke: MUTED }, legG);
    label(legG, x + 24, 177, 'not allowed: s after t');
    el('circle', { cx: x + 8, cy: 196, r: 3, fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, legG);
    label(legG, x + 24, 201, 'largest weight in the row');
  }
  // Step 5's accuracy curve.
  const curG = el('g', {}, mapG);
  const AX0 = 376;
  const AX1 = 584;
  const AY0 = 240;
  const AY1 = 80;
  const ax = (i) => AX0 + (i / ITERS) * (AX1 - AX0);
  const ay = (a) => AY0 + ((a - 0.4) / 0.6) * (AY1 - AY0);
  for (const [a, txt] of [[1, '100%'], [0.5, '50%']]) {
    el('line', { x1: AX0, x2: AX1, y1: P(ay(a)), y2: P(ay(a)), stroke: LINE, 'stroke-dasharray': a === 0.5 ? '5 4' : '2 4' }, curG);
    label(curG, AX0 - 6, P(ay(a)) + 4, txt, 'end', MUTED, 11);
  }
  label(curG, AX1, P(ay(0.5)) - 6, 'chance', 'end', MUTED, 11);
  el('line', { x1: AX0, x2: AX0, y1: AY0, y2: AY1 - 6, stroke: LINE }, curG);
  el('line', { x1: AX0, x2: AX1, y1: AY0, y2: AY0, stroke: LINE }, curG);
  for (const i of [0, 100, 200]) label(curG, P(ax(i)), AY0 + 16, String(i), 'middle', MUTED, 11);
  label(curG, (AX0 + AX1) / 2, AY0 + 32, 'batch', 'middle', MUTED, 11);
  label(curG, AX0 - 30, AY1 - 18, 'masked test accuracy', 'start', STRONG);
  const curPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, curG);
  const curDot = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, curG);
  // Step 6's bars, one per k.
  const kG = el('g', {}, mapG);
  const KB0 = 410;
  const KBW = 150;
  label(kG, 344, 34, `test accuracy, ${EPOCHS} epochs`, 'start', STRONG);
  const kRows = K_CHOICES.map((k, i) => {
    const y = 56 + i * 44;
    const name = label(kG, KB0 - 10, y + 19, `k = ${k}`, 'end', STRONG, 13);
    const rect = el('rect', { x: KB0, y, height: 28, width: 0, fill: CYAN, 'fill-opacity': 0.35, stroke: CYAN }, kG);
    const val = label(kG, KB0 + 6, y + 19, '', 'start', STRONG, 12);
    return { name, rect, val };
  });
  el('line', { x1: KB0 + KBW / 2, x2: KB0 + KBW / 2, y1: 50, y2: 232, stroke: MUTED, 'stroke-dasharray': '5 4' }, kG);
  label(kG, KB0 + KBW / 2, 248, 'chance 50%', 'middle', MUTED, 11);
  el('line', { x1: KB0, x2: KB0, y1: 50, y2: 232, stroke: LINE }, kG);
  const kWait = label(kG, 480, 150, 'computing…', 'middle');

  const rowsT = RY.map((y, i) => label(svg, 16, y, '', 'start', i === 0 ? STRONG : MUTED, 13));

  // ---- controls: Train / Pause / Reset (step 5) and the k slider (step 6).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const kGroup = make('div');
  const kLab = make('label');
  const kIn = make('input', { type: 'range', min: '0', max: String(K_CHOICES.length - 1), step: '1', value: String(kSel) });
  const kOut = make('output');
  kOut.setAttribute('aria-hidden', 'true');
  kLab.append('steps back, k', ' ', kIn, ' ', kOut);
  kGroup.append(kLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the attention model');
  const trainTag = make('span', { textContent: `model, k = ${K_LIVE}` });
  trainTag.setAttribute('aria-hidden', 'true');
  Object.assign(trainTag.style, { color: MUTED, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', marginRight: '6px' });
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  trainGroup.append(trainTag);
  controls.append(trainGroup, kGroup);

  const getLive = () => {
    if (!live) live = makeLive(K_LIVE);
    return live;
  };

  let START_BAND = null;
  const startBand = () => {
    if (!START_BAND) START_BAND = bandStats(startModel(), testSet(K_LIVE), K_LIVE);
    return START_BAND;
  };
  const BAND = new Map();
  const bandOf = (k, model) => {
    if (!BAND.has(k)) BAND.set(k, bandStats(model, testSet(k), k));
    return BAND.get(k);
  };

  function info() {
    if (step === 1 || step === 2) {
      const w = queryWeights(q);
      return { q, ...w, out: attendOutput(w.weights) };
    }
    if (step === 3) return { band: startBand() };
    if (step === 4) return liveInfo(getLive(), loop.activity());
    if (step === 5) {
      const runs = readyRuns();
      if (!runs) return null;
      const k = K_CHOICES[kSel];
      const r = runs[k];
      return { k, ok: r.testOk[EPOCHS], n: r.n, runs, model: r.model, band: bandOf(k, r.model) };
    }
    return null;
  }

  const drawMap = (model, k, marks) => {
    const Dk = testSet(k);
    const xs = Dk.seqs[EXAMPLE];
    xs.forEach((b, s) => headBits[s].set(b));
    hm.update(attentionMap(model, xs));
    show(tgtG, marks);
    if (!marks) return;
    const { logits } = attnForward(model, xs);
    for (let t = 0; t < T; t++) {
      const m = Dk.mask[EXAMPLE][t];
      tgtCells[t].set(m ? Dk.targets[EXAMPLE][t] : null);
      const right = predictBit(logits[t]) === Dk.targets[EXAMPLE][t];
      tgtMarks[t].textContent = m ? (right ? '✓' : '✗') : '';
      tgtMarks[t].style.fill = right ? CYAN : VIOLET;
    }
  };

  function render() {
    show(probG, step === 0);
    show(toyG, step === 1 || step === 2);
    show(barG, step === 1);
    show(valG, step === 2);
    show(mapG, step >= 3);
    show(legG, step === 3);
    show(curG, step === 4);
    show(kG, step === 5);
    const narrow = svg.clientWidth > 0 && svg.clientWidth < 500;
    for (const n of small) n.style.fontSize = narrow ? '14px' : '';
    colNums.forEach((n, i) => { n.style.fontSize = narrow ? '14px' : ''; n.style.display = narrow && i % 2 === 1 ? 'none' : ''; });
    kOut.textContent = String(K_CHOICES[kSel]);
    kIn.setAttribute('aria-valuetext', `k = ${K_CHOICES[kSel]}`);
    const s = info();

    if (step === 1 || step === 2) {
      qLine.setAttribute('x2', P(lx(q[0])));
      qLine.setAttribute('y2', P(ly(q[1])));
      qLab.setAttribute('x', P(lx(q[0]) + 10));
      qLab.setAttribute('y', P(ly(q[1]) - 8));
      for (const n of [handle, handleRing]) { n.setAttribute('cx', P(lx(q[0]))); n.setAttribute('cy', P(ly(q[1]))); }
      handle.setAttribute('aria-label', `Query q = ${vec(q)}. Arrow keys move it.`);
    }
    if (step === 1) {
      bars.forEach((b, i) => {
        b.rect.setAttribute('width', P(s.weights[i] * BWID));
        b.val.textContent = num(s.weights[i], 2);
        b.val.setAttribute('x', P(BX + Math.min(s.weights[i] * BWID, BWID - 40) + 6));
        b.sc.textContent = `score ${num(s.scores[i], 2)}`;
      });
    }
    if (step === 2) {
      valMarks.forEach((m) => { m.t.textContent = `v${SUB[m.i]} ${num(s.weights[m.i], 2)}`; });
      const ox = rx(s.out[0]);
      const oy = ry(s.out[1]);
      outLines.forEach((ln, i) => {
        ln.setAttribute('x1', P(ox)); ln.setAttribute('y1', P(oy));
        ln.setAttribute('x2', P(rx(VALUES[i][0]))); ln.setAttribute('y2', P(ry(VALUES[i][1])));
        ln.setAttribute('stroke-width', P(0.5 + 3 * s.weights[i]));
      });
      outDot.setAttribute('d', `M${P(ox)},${P(oy - 8)}L${P(ox + 8)},${P(oy)}L${P(ox)},${P(oy + 8)}L${P(ox - 8)},${P(oy)}Z`);
    }
    if (step === 3) drawMap(startModel(), K_LIVE, false);
    if (step === 4) {
      const lv = getLive();
      drawMap(lv.trainer.state.model, K_LIVE, true);
      curPath.setAttribute('d', lv.testOk.map((v, i) => `${i ? 'L' : 'M'}${P(ax(i))},${P(ay(v / lv.n))}`).join(''));
      curDot.setAttribute('cx', P(ax(lv.testOk.length - 1)));
      curDot.setAttribute('cy', P(ay(lv.testOk.at(-1) / lv.n)));
    }
    if (step === 5) {
      show(kWait, !s);
      if (s) drawMap(s.model, s.k, true);
      else { hm.update(Array.from({ length: T }, (_, t) => Array.from({ length: T }, (_, c) => (c <= t ? 0 : null)))); show(tgtG, false); }
      kRows.forEach((row, i) => {
        const k = K_CHOICES[i];
        const r = s ? s.runs[k] : null;
        const a = r ? r.testOk[EPOCHS] / r.n : 0;
        row.rect.setAttribute('width', P(a * KBW));
        row.val.setAttribute('x', P(KB0 + Math.min(a * KBW, KBW - 46) + 6));
        row.val.textContent = r ? pctCount(r.testOk[EPOCHS], r.n) : '';
        const sel = i === kSel;
        row.rect.setAttribute('fill-opacity', sel ? 0.6 : 0.25);
        row.rect.setAttribute('stroke-width', sel ? 2.5 : 1);
        row.name.textContent = sel ? `▸ k = ${k}` : `k = ${k}`;
        row.name.style.fill = sel ? STRONG : MUTED;
      });
    }

    const texts = readout(step, s);
    rowsT.forEach((t, i) => { t.textContent = texts[i] || ''; });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Test stream 1, 12 bits, with its targets for k = ${K_LIVE}: the target at step t is the bit at step t − ${K_LIVE}. Below, a recurrent network passes the first bit along ${K_LIVE} links to reach step ${K_LIVE + 1}, while attention draws one direct arrow from step ${K_LIVE + 1} back to position 1. ${rest}.`;
    if (step === 1) return `Four keys in a plane and a query arrow. Bars show each key's score and its softmax weight: ${s.weights.map((w, i) => `key ${i + 1}: ${num(w, 2)}`).join(', ')}. ${rest}.`;
    if (step === 2) return `The four values at the corners of a square, each labelled with its weight, and the output point, the weighted average, at ${vec2(s.out)}. ${rest}.`;
    if (step === 3 || step === 4 || step === 5) {
      const what = step === 3 ? 'the starting model' : step === 4 ? 'the model as it trains' : `the model trained for k = ${s ? s.k : K_CHOICES[kSel]}`;
      const bars = step === 5 && s ? ` Bars of masked test accuracy: ${K_CHOICES.map((k) => `k = ${k}: ${pctCount(s.runs[k].testOk[EPOCHS], s.runs[k].n)}`).join(', ')}.` : '';
      return `A 12 by 12 attention map of test stream 1 for ${what}: row t is the query position, column s the key position; cells after the diagonal are not allowed and are hatched; a dot marks each row's largest weight.${bars} ${rest}.`;
    }
    return rest;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // ---- the query handle (steps 2 and 3).
  draggable(handle, {
    bounds: { x0: lx(-Q_RANGE), x1: lx(Q_RANGE), y0: ly(Q_RANGE), y1: ly(-Q_RANGE) },
    onDrag: (px, py) => {
      q = snapQuery((px - LC[0]) / UNIT, (LC[1] - py) / UNIT);
      render();
      say(spoken());
    },
  });

  // ---- step 5's live run.
  let waited = 0;
  const loop = createTrainLoop({
    step: () => {
      const r = paceFrame(getLive().trainer.state.iter, waited);
      waited = r.waited;
      if (r.take) getLive().advance(r.take);
    },
    render,
    done: () => getLive().done(),
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done()) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { waited = 0; getLive().reset(); },
    reduced,
    work: (m) => getLive().advance(m),
  });
  trainGroup.addEventListener('click', () => { if (step === 4) { render(); announce(spoken()); } });

  // ---- step 6's four runs, computed when first needed, in slices of about 6 ms per frame.
  let job = null;
  const cancelRuns = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const computeRuns = () => {
    if (job !== null || K_CHOICES.every(runReady)) return;
    const tick = () => {
      job = null;
      if (step !== 5) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const k = K_CHOICES.find((x) => !runReady(x));
        if (k === undefined) break;
        runWork(k, 2);
      }
      if (K_CHOICES.every(runReady)) { render(); announce(spoken()); } else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

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
    show(i) {
      buttons.stop();
      step = i;
      if (i === 5) computeRuns(); else cancelRuns();
      kGroup.style.display = i === 5 ? 'contents' : 'none';
      trainGroup.style.display = i === 4 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 4);
      const onToy = i === 1 || i === 2;
      handle.setAttribute('tabindex', onToy ? 0 : -1);
      if (!onToy && doc.activeElement === handle) handle.blur();
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
