// Lesson L5.5, "Transformers": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-transformers.test.mjs);
// the DOM is only touched inside createStage. Every forward pass is attnForward from
// attn.js, every training step is attnTrainer, and every number in the page's text
// comes from pageValues or storyValues.

import { mulberry32 } from '../assets/maths.js';
import { attnInit, attnForward, attnTrainer, attnFlatten, attnUnflatten, attnParamCount, layerNorm } from '../assets/attn.js';
import { softmax } from '../assets/text.js';
import { el, heatmap } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const SEED_TRAIN = 281;
export const SEED_TEST = 282;
export const SEED_INIT = 283;
export const SEED_SHUFFLE = 284;
export const SEED_PERM = 285;
export const N_TRAIN = 2000;
export const N_TEST = 500;
export const LEN = 8;
export const VOCAB = 10;
export const CFG = { vocab: VOCAB, len: LEN, d: 32, heads: 2, causal: false, ln: true, mlp: 64, out: 'softmax', seed: SEED_INIT };
export const ETA = 0.001;
export const BATCH = 40;
export const EPOCHS = 5;
export const STEPS = (EPOCHS * N_TRAIN) / BATCH;
// The test set is checked before training and every EVAL_EVERY steps.
export const EVAL_EVERY = 25;
// The sequence drawn in the maps: the first test sequence.
export const LITERAL_EXAMPLE = [4, 1, 9, 1, 7, 1, 3, 0];

// n sequences of LEN uniform digits ⌊10u⌋, from one mulberry32(seed) stream; the
// target is the same digits in reverse order.
export function reversalData(n, seed) {
  const rng = mulberry32(seed);
  const seqs = [];
  const targets = [];
  for (let i = 0; i < n; i++) {
    const s = Array.from({ length: LEN }, () => Math.floor(rng() * VOCAB));
    seqs.push(s);
    targets.push([...s].reverse());
  }
  return { seqs, targets };
}

const CACHE = new Map();
const cached = (key, make) => {
  if (!CACHE.has(key)) CACHE.set(key, make());
  return CACHE.get(key);
};
export const trainSet = () => cached('train', () => reversalData(N_TRAIN, SEED_TRAIN));
export const testSet = () => cached('test', () => reversalData(N_TEST, SEED_TEST));
export const example = () => testSet().seqs[0];
export const startModel = () => attnInit(CFG);

// ---- small helpers ----------------------------------------------------------------

const argmax = (v) => {
  let b = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
  return b;
};

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
// k of n as a percentage to one decimal; never "100%" unless k = n.
export function pct(k, n) {
  if (k === n) return '100%';
  const s = ((100 * k) / n).toFixed(1);
  return s === '100.0' ? '99.9%' : `${s}%`;
}
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

// The most likely digit at one position and its probability.
function top(logits) {
  const p = softmax(logits);
  const digit = argmax(p);
  return { digit, p: p[digit] };
}

// ---- step 1: the block, box by box -------------------------------------------------

export const BLOCK_PARTS = [
  { key: 'in', name: 'digits + positions', desc: 'Each digit’s embedding (32 numbers) plus the embedding of its position, 1 to 8.' },
  { key: 'ln1', name: 'layer norm', desc: 'Rescales each position’s 32 numbers to mean 0 and variance 1, then a learned scale and shift.' },
  { key: 'att', name: 'attention, 2 heads', desc: 'Each position takes a weighted average over all 8 positions, once per head.' },
  { key: 'add1', name: 'add (shortcut)', desc: 'The attention result is added to its input, so the input flows on unchanged.' },
  { key: 'ln2', name: 'layer norm', desc: 'The same rescaling again, with its own learned scale and shift.' },
  { key: 'mlp', name: 'small network', desc: 'The same 32 → 64 → 32 ReLU network, applied to each position on its own.' },
  { key: 'add2', name: 'add (shortcut)', desc: 'The network’s result is added to its input.' },
  { key: 'lnf', name: 'final layer norm', desc: 'One more layer norm before the output.' },
  { key: 'out', name: 'output', desc: 'At each position, 10 scores turned into probabilities for the digits 0 to 9.' },
];

// ---- step 2: attention ignores order --------------------------------------------------

// A copy of the model with every position embedding set to 0.
export function withoutPositions(model) {
  const v = attnFlatten(model);
  const m = attnUnflatten(model, v);
  m.P.fill(0);
  return m;
}

// The i-th shuffle (i = 0, 1, …): the (i + 1)-th Fisher–Yates shuffle of 0..7 drawn from
// one mulberry32(SEED_PERM) stream. perm[t] is the old position of the digit now at t.
export function permutation(i) {
  const rng = mulberry32(SEED_PERM);
  let p = null;
  for (let r = 0; r <= i; r++) {
    p = Array.from({ length: LEN }, (_, k) => k);
    for (let k = LEN - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [p[k], p[j]] = [p[j], p[k]];
    }
  }
  return p;
}

// Largest gap between the outputs for the shuffled input and the outputs for the
// original input moved by the same shuffle.
function movedGap(a, b, perm) {
  let w = 0;
  for (let t = 0; t < LEN; t++) for (let k = 0; k < VOCAB; k++) w = Math.max(w, Math.abs(b[t][k] - a[perm[t]][k]));
  return w;
}

// The untrained block on the example and on its i-th shuffle, without and with positions.
export function permDemo(i) {
  const perm = permutation(i);
  const x = example();
  const px = perm.map((j) => x[j]);
  const noPos = withoutPositions(startModel());
  const a = attnForward(noPos, x).logits;
  const b = attnForward(noPos, px).logits;
  const wa = attnForward(startModel(), x).logits;
  const wb = attnForward(startModel(), px).logits;
  return {
    perm, x, px,
    out: a.map(top), pout: b.map(top),
    diffNoPos: movedGap(a, b, perm),
    diffPos: movedGap(wa, wb, perm),
    // With positions: positions whose top digit is not the moved one.
    posChanged: wb.map((r, t) => (top(r).digit !== top(wa[perm[t]]).digit ? 1 : 0)).reduce((s, v) => s + v, 0),
  };
}

// ---- step 4: layer norm on one position ------------------------------------------------

// The vector entering the block at position pos of the example (untrained model), and
// the first layer norm's output; means and population (1/d) variances of both.
export function lnDemo(pos) {
  const m = startModel();
  const c = attnForward(m, example()).cache;
  const d = CFG.d;
  const x = Array.from(c.x0.subarray(pos * d, (pos + 1) * d));
  const y = layerNorm(x, m.blocks[0].g1, m.blocks[0].b1).y;
  const stats = (v) => {
    const mu = v.reduce((s, z) => s + z, 0) / v.length;
    return [mu, v.reduce((s, z) => s + (z - mu) ** 2, 0) / v.length];
  };
  const [mean, vr] = stats(x);
  const [meanY, varY] = stats(y);
  return { pos, digit: example()[pos], x, y, mean, var: vr, meanY, varY };
}

// ---- step 6: the parameter count, part by part ---------------------------------------------

export function paramParts(cfg) {
  const { vocab: V, len: T, d, heads: H, mlp: M } = cfg;
  const D = H * (d / H);
  return [
    { key: 'tok', name: 'digit embeddings E', formula: `${V} × ${d}`, count: V * d },
    { key: 'pos', name: 'position embeddings P', formula: `${T} × ${d}`, count: T * d },
    { key: 'ln1', name: 'layer norm 1 (γ, β)', formula: `2 × ${d}`, count: 2 * d },
    { key: 'qkv', name: 'Q, K, V for 2 heads', formula: `3 × ${d} × ${D}`, count: 3 * d * D },
    { key: 'wo', name: 'W_O (join the heads)', formula: `${D} × ${d}`, count: D * d },
    { key: 'ln2', name: 'layer norm 2 (γ, β)', formula: `2 × ${d}`, count: 2 * d },
    { key: 'mlp', name: 'small network', formula: `${d}×${M} + ${M} + ${M}×${d} + ${d}`, count: 2 * d * M + M + d },
    { key: 'lnf', name: 'final layer norm', formula: `2 × ${d}`, count: 2 * d },
    { key: 'out', name: 'output layer', formula: `${V} × ${d} + ${V}`, count: V * d + V },
  ];
}

// ---- training --------------------------------------------------------------------------

export function makeTrainer(model = startModel()) {
  return attnTrainer({ model, data: trainSet(), batch: BATCH, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
}

// Test-set check: right answers per output position, and per head how many of the
// 8 × N_TEST attention rows put their largest weight on the mirror position 7 − t
// (positions counted from 0), and the mean weight on it.
export function evaluate(model, set = testSet()) {
  const right = new Array(LEN).fill(0);
  const anti = [0, 0];
  const mirror = [0, 0];
  for (let i = 0; i < set.seqs.length; i++) {
    const f = attnForward(model, set.seqs[i]);
    for (let t = 0; t < LEN; t++) {
      if (argmax(f.logits[t]) === set.targets[i][t]) right[t]++;
      for (let h = 0; h < 2; h++) {
        const row = f.A[0][h][t];
        if (argmax(row) === LEN - 1 - t) anti[h]++;
        mirror[h] += row[LEN - 1 - t];
      }
    }
  }
  const rows = LEN * set.seqs.length;
  return { n: set.seqs.length, right, anti, mirror: mirror.map((v) => v / rows), rows };
}

// The live run: advance(k) takes at most k steps, never past STEPS, and checks the
// test set every EVAL_EVERY steps (history).
export function makeLive(model = startModel()) {
  const tr = makeTrainer(model);
  const live = { trainer: tr, history: [] };
  const record = () => {
    const e = evaluate(tr.state.model);
    live.history.push({ iter: tr.state.iter, right: e.right, anti: e.anti });
  };
  record();
  live.advance = (k) => {
    for (let i = 0; i < k && tr.state.iter < STEPS && !tr.state.diverged; i++) {
      tr.step(1);
      if (tr.state.iter % EVAL_EVERY === 0) record();
    }
  };
  live.reset = () => { tr.reset(); live.history = []; record(); };
  live.done = () => tr.state.iter >= STEPS || tr.state.diverged;
  return live;
}

// ---- values written into the page ---------------------------------------------------

// The example's most repeated digit, its positions (from 1) and what reversal needs there.
export function repeatedDigit() {
  const x = example();
  const counts = Array.from({ length: VOCAB }, (_, k) => x.filter((v) => v === k).length);
  const digit = argmax(counts);
  const at = x.map((v, t) => (v === digit ? t : -1)).filter((t) => t >= 0);
  return { digit, at, need: at.map((t) => x[LEN - 1 - t]) };
}
const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);

export function pageValues() {
  const rep = repeatedDigit();
  const total = paramParts(CFG).reduce((s, p) => s + p.count, 0);
  const ln = lnDemo(0);
  return {
    'n-params': count(total),
    'n-train': count(N_TRAIN),
    'n-test': String(N_TEST),
    'steps': String(STEPS),
    'example': example().join(' '),
    'example-rev': [...example()].reverse().join(' '),
    'ln-mean': num(ln.mean, 2),
    'ln-var': num(ln.var, 2),
    'ln-chance': num(Math.log(VOCAB), 2),
    'n-rows': count(LEN * N_TEST),
    'chance-row': pct(1, LEN),
    'rep-digit': String(rep.digit),
    'rep-at': listText(rep.at.map((t) => t + 1)),
    'rep-need': listText(rep.need),
  };
}

// The values quoted from the seeded run, from its start model and the finished live
// run (end model, training-set losses, test-set history).
export function runValues(start, live) {
  const st = live.trainer.state;
  const e0 = evaluate(start);
  const e1 = evaluate(st.model);
  const v = {};
  v['acc-start'] = pct(e0.right.reduce((s, r) => s + r, 0), e0.rows);
  v['right-min-end'] = String(Math.min(...e1.right));
  for (const h of [0, 1]) {
    v[`anti-${h + 1}-start`] = pct(e0.anti[h], e0.rows);
    v[`anti-${h + 1}-end`] = count(e1.anti[h]);
    v[`mirror-${h + 1}-end`] = num(e1.mirror[h], 2);
  }
  // The first test check at which every position of every test sequence is right.
  const all = live.history.find((h) => h.right.every((r) => r === N_TEST));
  v['all-right-at'] = all ? String(all.iter) : 'never';
  v['loss-start'] = num(st.epochLoss[0], 2);
  v['loss-end'] = num(st.epochLoss[EPOCHS], 2);
  return v;
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const start = startModel();
  const live = makeLive(start);
  while (!live.done()) live.advance(STEPS);
  STORY = runValues(start, live);
  return STORY;
}

// ---- read-outs --------------------------------------------------------------------

// What a read-out needs for a step. ui: {part, perm, pos}.
export function stageInfo(step, live, running, ui = {}) {
  const s = live ? live.trainer.state : null;
  const base = live
    ? { step, iter: s.iter, running, diverged: s.diverged, done: live.done(), h: live.history.at(-1), n: N_TEST, ui }
    : { step, ui };
  if (step === 0) return { ...base, part: BLOCK_PARTS[ui.part ?? 0] };
  if (step === 1) return { ...base, demo: permDemo(ui.perm ?? 0) };
  if (step === 3) return { ...base, ln: lnDemo(ui.pos ?? 0) };
  if (step === 5) return { ...base, parts: paramParts(CFG) };
  return base;
}

// Splits text into at most two lines of up to `width` characters, at spaces.
export function wrap2(text, width = 52) {
  if (text.length <= width) return [text, ''];
  const cut = text.lastIndexOf(' ', width);
  return [text.slice(0, cut), text.slice(cut + 1)];
}

export function readout(step, s) {
  if (step === 0) return [s.part.name, ...wrap2(s.part.desc)];
  if (step === 1) {
    const d = s.demo;
    return [
      `shuffled input: ${d.px.join(' ')}`,
      `no positions: outputs just move (gap ${d.diffNoPos === 0 ? '0' : '< 10⁻¹²'})`,
      `with positions: ${d.posChanged} of 8 top digits change (gap ${num(d.diffPos, 2)})`,
    ];
  }
  if (step === 5) {
    const total = s.parts.reduce((a, p) => a + p.count, 0);
    return [`${count(total)} numbers, every one learned`, `the largest part: ${[...s.parts].sort((a, b) => b.count - a.count)[0].name}`, 'real models: many such blocks, far wider'];
  }
  if (step === 3) {
    const l = s.ln;
    return [
      `position ${l.pos + 1} (digit ${l.digit}), untrained model`,
      `before: mean ${num(l.mean, 3)} · variance ${num(l.var, 3)}`,
      `after layer norm: mean ${num(l.meanY, 3)} · variance ${num(l.varY, 3)}`,
    ];
  }
  const where = s.iter === 0 ? 'random start' : `${s.iter} of ${STEPS} steps`;
  const mir = `rows on the mirror: head 1 ${pct(s.h.anti[0], LEN * s.n)} · head 2 ${pct(s.h.anti[1], LEN * s.n)}`;
  if (step === 2) return [`model: ${where}`, '32 numbers → 2 heads of 16 → joined (32) → W_O → 32', mir];
  if (step === 4) {
    const lo = Math.min(...s.h.right);
    return [
      `step ${s.iter} of ${STEPS} · ${trainStatus({ running: s.running, iter: s.iter, done: s.done, diverged: s.diverged })}`,
      `test set at step ${s.h.iter}: lowest position ${pct(lo, s.n)}`,
      mir,
    ];
  }
  return [];
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [292, 314, 336];

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let part = 0;
  let perm = 0;
  let pos = 0;
  let live = null;
  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'group');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  // Titles and read-outs (size 13 and up) take the larger phone size from ml.css
  // (class stage-text); the dense diagram labels keep their own size.
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: size >= 13 ? 'stage-text' : 'stage-label', x, y, 'text-anchor': anchor, text, ...HALO }, parent);
    t.style.fill = fill;
    if (size < 13) t.style.fontSize = `${size}px`;
    return t;
  };
  const arrow = (parent, pts, dashed = false) => {
    const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x},${y}`).join('');
    el('path', { d, fill: 'none', stroke: MUTED, 'stroke-width': 1.5, 'stroke-dasharray': dashed ? '4 3' : '' }, parent);
    const [x1, y1] = pts.at(-2);
    const [x2, y2] = pts.at(-1);
    const a = Math.atan2(y2 - y1, x2 - x1);
    const h = (s) => [x2 - 7 * Math.cos(a + s * 0.45), y2 - 7 * Math.sin(a + s * 0.45)].map(P);
    el('path', { d: `M${h(1).join(',')}L${x2},${y2}L${h(-1).join(',')}`, fill: 'none', stroke: MUTED, 'stroke-width': 1.5 }, parent);
  };

  // ---- step 1: the block diagram. The residual stream runs along y = SY; each
  // sublayer leaves it, works on a layer-normed copy and is added back at a ⊕.
  const diaG = el('g', {}, svg);
  const SY = 176;
  const BOX = {
    in: [16, SY - 22, 78, 44], ln1: [112, 58, 46, 40], att: [170, 52, 98, 52], add1: [286, SY, 11],
    ln2: [318, 58, 46, 40], mlp: [376, 52, 92, 52], add2: [486, SY, 11], lnf: [506, SY - 20, 34, 40], out: [548, SY - 22, 46, 44],
  };
  const TEXT = {
    in: ['digits', '+ positions'], ln1: ['LN'], att: ['attention', '2 heads'], add1: ['+'], ln2: ['LN'],
    mlp: ['small', 'network'], add2: ['+'], lnf: ['LN'], out: ['output', '0–9'],
  };
  label(diaG, 16, 24, 'one transformer block (pre-LN, as built here)', 'start', STRONG, 13);
  el('line', { x1: 94, x2: 506, y1: SY, y2: SY, stroke: MUTED, 'stroke-width': 1.5 }, diaG);
  arrow(diaG, [[540, SY], [548, SY]]);
  label(diaG, 300, SY + 40, 'the shortcut path: 32 numbers per position flow straight through', 'middle', MUTED, 11);
  // Branch out to each sublayer and back in.
  arrow(diaG, [[104, SY], [104, 78], [112, 78]]);
  arrow(diaG, [[158, 78], [170, 78]]);
  arrow(diaG, [[268, 78], [286, 78], [286, SY - 11]]);
  arrow(diaG, [[306, SY], [306, 78], [318, 78]]);
  arrow(diaG, [[364, 78], [376, 78]]);
  arrow(diaG, [[468, 78], [486, 78], [486, SY - 11]]);
  const boxes = BLOCK_PARTS.map((p, i) => {
    const g = el('g', { tabindex: 0, role: 'button', 'aria-label': `${p.name}: ${p.desc}`, style: 'cursor: pointer' }, diaG);
    const b = BOX[p.key];
    let shape;
    let cx;
    let cy;
    if (b.length === 3) {
      shape = el('circle', { cx: b[0], cy: b[1], r: b[2], fill: SURFACE, stroke: MUTED, 'stroke-width': 1.5 }, g);
      [cx, cy] = b;
    } else {
      shape = el('rect', { x: b[0], y: b[1], width: b[2], height: b[3], rx: 6, fill: SURFACE, stroke: MUTED, 'stroke-width': 1.5 }, g);
      cx = b[0] + b[2] / 2;
      cy = b[1] + b[3] / 2;
    }
    const lines = TEXT[p.key];
    lines.forEach((t, k) => label(g, cx, P(cy + 4 + (k - (lines.length - 1) / 2) * 15), t, 'middle', STRONG, lines.length > 1 ? 11 : 13));
    const pick = () => {
      if (part === i) return;
      part = i;
      render();
      announce(spoken());
    };
    g.addEventListener('pointerenter', pick);
    g.addEventListener('focus', pick);
    g.addEventListener('click', pick);
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    return { g, shape };
  });
  label(diaG, 16, 250, 'Real models stack many of these blocks; this lesson uses one.', 'start', MUTED, 11);

  // ---- step 2: shuffle the input, without positions.
  const pmG = el('g', {}, svg);
  const PX0 = 168;
  const PW = 52;
  label(pmG, 16, 18, 'the untrained block with every position embedding set to 0', 'start', STRONG, 13);
  const pmRows = [['input', 56], ['output', 190], ['shuffled input', 122], ['output, shuffled', 236]];
  pmRows.forEach(([t, y]) => label(pmG, 16, y + 5, t, 'start', MUTED, 12));
  for (let t = 0; t < LEN; t++) label(pmG, PX0 + t * PW + PW / 2, 37, `pos. ${t + 1}`, 'middle', MUTED, 10);
  label(pmG, 16, 168, 'the block’s most likely output digit and its probability', 'start', MUTED, 10);
  const pmLinks = el('g', {}, pmG);
  const pmCells = pmRows.map(([, y], r) => Array.from({ length: LEN }, (_, t) => {
    const x = PX0 + t * PW;
    el('rect', { x: x + 3, y: y - 14, width: PW - 6, height: r % 2 ? 34 : 26, rx: 4, fill: 'none', stroke: r % 2 ? LINE : MUTED }, pmG);
    const big = label(pmG, x + PW / 2, y + 4, '', 'middle', STRONG, 14);
    const small = r % 2 ? label(pmG, x + PW / 2, y + 17, '', 'middle', MUTED, 10) : null;
    return { big, small };
  }));
  const fromT = Array.from({ length: LEN }, (_, t) => label(pmG, PX0 + t * PW + PW / 2, 150, '', 'middle', VIOLET, 10));

  // ---- the attention maps (steps 3 and 5).
  const mapG = el('g', {}, svg);
  const mapTitles = [label(mapG, 0, 0, 'head 1', 'start', STRONG, 13), label(mapG, 0, 0, 'head 2', 'start', STRONG, 13)];
  const MAP = { 2: { cell: 14, xs: [334, 470], y: 64 }, 4: { cell: 18, xs: [44, 224], y: 52 } };
  const maps = { 2: [], 4: [] };
  const mapGroups = {};
  for (const k of [2, 4]) {
    mapGroups[k] = el('g', {}, mapG);
    for (const h of [0, 1]) {
      maps[k].push(heatmap(mapGroups[k], LEN, LEN, {
        x: MAP[k].xs[h], y: MAP[k].y, cell: MAP[k].cell,
        rowLabels: Array.from({ length: LEN }, (_, t) => String(t + 1)),
        colLabels: example().map(String),
        title: `Head ${h + 1}: attention weights for the example sequence. Row: output position; column: input position, labelled with its digit.`,
      }));
    }
  }
  // Step 3: how the heads are wired.
  const hdG = el('g', {}, mapG);
  const hbox = (x, y, w, h, lines, stroke = MUTED) => {
    el('rect', { x, y, width: w, height: h, rx: 5, fill: SURFACE, stroke, 'stroke-width': 1.5 }, hdG);
    lines.forEach((t, k) => label(hdG, x + w / 2, y + h / 2 + 4 + (k - (lines.length - 1) / 2) * 14, t, 'middle', STRONG, 11));
  };
  hbox(16, 108, 62, 40, ['LN(x)', '32']);
  hbox(100, 54, 148, 44, ['head 1: own Q, K, V', '16 numbers out'], CYAN);
  hbox(100, 158, 148, 44, ['head 2: own Q, K, V', '16 numbers out'], VIOLET);
  hbox(266, 108, 50, 40, ['join', '32']);
  arrow(hdG, [[78, 128], [88, 128], [88, 76], [100, 76]]);
  arrow(hdG, [[88, 128], [88, 180], [100, 180]]);
  arrow(hdG, [[248, 76], [258, 76], [258, 128], [266, 128]]);
  arrow(hdG, [[248, 180], [258, 180], [258, 128]]);
  label(hdG, 291, 168, '× W_O', 'middle', MUTED, 11);
  label(hdG, 291, 182, '→ 32', 'middle', MUTED, 11);
  // Step 5: the example's output, and accuracy by position.
  const exG = el('g', {}, mapG);
  label(exG, 44, 218, 'row: output position · column: input digit', 'start', MUTED, 10);
  label(exG, 44, 232, 'dot: the largest weight in the row', 'start', MUTED, 10);
  const exIn = label(exG, 44, 256, '', 'start', INK, 12);
  exIn.style.whiteSpace = 'pre';
  const accG = el('g', {}, mapG);
  const AX = 440;
  const AW = 18;
  const AB = 214;
  const AH = 150;
  label(accG, AX - 34, 28, 'test accuracy by position', 'start', STRONG, 12);
  el('line', { x1: AX, x2: AX + 8 * AW, y1: AB, y2: AB, stroke: LINE }, accG);
  for (const [v, t] of [[1, '100%'], [0.1, '10%']]) {
    el('line', { x1: AX, x2: AX + 8 * AW, y1: P(AB - v * AH), y2: P(AB - v * AH), stroke: MUTED, 'stroke-dasharray': '3 4' }, accG);
    label(accG, AX - 4, P(AB - v * AH) + 4, t, 'end', MUTED, 10);
  }
  label(accG, AX + 4 * AW, AB + 30, 'position · 10% is guessing', 'middle', MUTED, 10);
  const accBars = Array.from({ length: LEN }, (_, t) => {
    const x = AX + t * AW + 2;
    label(accG, x + (AW - 4) / 2, AB + 14, String(t + 1), 'middle', MUTED, 10);
    return el('rect', { x, y: AB, width: AW - 4, height: 0, fill: CYAN, 'fill-opacity': 0.7 }, accG);
  });

  // ---- step 4: one position's 32 numbers, before and after layer norm.
  const lnG = el('g', {}, svg);
  const BX = 64;
  const BW = 16;
  const charts = [{ y0: 92, t: 'x: digit embedding + position embedding' }, { y0: 210, t: 'LN(x): mean 0, variance 1 (γ = 1, β = 0 at the start)' }].map((c) => {
    label(lnG, 16, c.y0 - 52, c.t, 'start', STRONG, 12);
    el('line', { x1: BX, x2: BX + 32 * BW, y1: c.y0, y2: c.y0, stroke: LINE }, lnG);
    for (const v of [-2, 2]) {
      el('line', { x1: BX, x2: BX + 32 * BW, y1: c.y0 - v * 16, y2: c.y0 - v * 16, stroke: LINE, 'stroke-dasharray': '2 4' }, lnG);
      label(lnG, BX - 6, c.y0 - v * 16 + 4, num(v, 0), 'end', MUTED, 10);
    }
    label(lnG, BX - 6, c.y0 + 4, '0', 'end', MUTED, 10);
    const mean = el('line', { x1: BX, x2: BX + 32 * BW, stroke: VIOLET, 'stroke-width': 1.5, 'stroke-dasharray': '6 3' }, lnG);
    const bars = Array.from({ length: 32 }, (_, j) => el('rect', { x: BX + j * BW + 2, width: BW - 4, fill: CYAN, 'fill-opacity': 0.7 }, lnG));
    return { ...c, mean, bars };
  });
  label(lnG, BX + 32 * BW, 22, 'dashed: the mean', 'end', VIOLET, 11);

  // ---- step 6: the parameter count by part.
  const pcG = el('g', {}, svg);
  const PARTS = paramParts(CFG);
  const TOTAL = PARTS.reduce((s, p) => s + p.count, 0);
  const MAXC = Math.max(...PARTS.map((p) => p.count));
  label(pcG, 16, 22, 'part', 'start', MUTED, 11);
  label(pcG, 196, 22, 'size', 'start', MUTED, 11);
  label(pcG, 420, 22, 'numbers', 'end', MUTED, 11);
  PARTS.forEach((p, k) => {
    const y = 44 + k * 23;
    label(pcG, 16, y, p.name, 'start', STRONG, 12);
    label(pcG, 196, y, p.formula, 'start', INK, 11);
    label(pcG, 420, y, count(p.count), 'end', STRONG, 12);
    el('rect', { x: 432, y: y - 10, width: P((p.count / MAXC) * 150), height: 11, fill: CYAN, 'fill-opacity': 0.6 }, pcG);
  });
  el('line', { x1: 16, x2: 420, y1: 44 + PARTS.length * 23 - 14, y2: 44 + PARTS.length * 23 - 14, stroke: LINE }, pcG);
  label(pcG, 16, 44 + PARTS.length * 23 + 2, 'total', 'start', STRONG, 12);
  label(pcG, 420, 44 + PARTS.length * 23 + 2, count(TOTAL), 'end', STRONG, 12);

  // ---- read-out rows.
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- controls: Shuffle (step 2), Train (steps 3 and 5), position (step 4).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const shGroup = make('div');
  const shBtn = make('button', { type: 'button', textContent: 'Shuffle the input' });
  shGroup.append(shBtn);
  const posGroup = make('div');
  const posLab = make('label');
  const posIn = make('input', { type: 'range', min: '1', max: String(LEN), step: '1', value: '1' });
  const posOut = make('output');
  posOut.setAttribute('aria-hidden', 'true');
  posLab.append('position', ' ', posIn, ' ', posOut);
  posGroup.append(posLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the transformer');
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  controls.append(shGroup, posGroup, trainGroup);

  const info = () => stageInfo(step, step === 2 || step === 4 ? getLive() : null, loop.activity(), { part, perm, pos });

  function drawMaps(k, model) {
    const f = attnForward(model, example());
    for (const h of [0, 1]) maps[k][h].update(f.A[0][h]);
    MAP[k].xs.forEach((x, h) => {
      mapTitles[h].setAttribute('x', x);
      mapTitles[h].setAttribute('y', MAP[k].y - 26);
    });
    return f;
  }

  function render() {
    show(diaG, step === 0);
    show(pmG, step === 1);
    show(mapG, step === 2 || step === 4);
    show(mapGroups[2], step === 2);
    show(hdG, step === 2);
    show(mapGroups[4], step === 4);
    show(exG, step === 4);
    show(accG, step === 4);
    show(lnG, step === 3);
    show(pcG, step === 5);
    posOut.textContent = String(pos + 1);
    posIn.setAttribute('aria-valuetext', `position ${pos + 1}, digit ${example()[pos]}`);
    const s = info();

    if (step === 0) {
      boxes.forEach((b, i) => {
        b.shape.setAttribute('stroke', i === part ? CYAN : MUTED);
        b.shape.setAttribute('stroke-width', i === part ? 2.5 : 1.5);
      });
    }

    if (step === 1) {
      const d = s.demo;
      const vals = [d.x, d.out, d.px, d.pout];
      pmCells.forEach((row, r) => row.forEach((c, t) => {
        const v = vals[r][t];
        if (r % 2) {
          c.big.textContent = String(v.digit);
          c.small.textContent = num(v.p, 3);
        } else c.big.textContent = String(v);
      }));
      d.perm.forEach((j, t) => { fromT[t].textContent = `from ${j + 1}`; });
      pmLinks.replaceChildren();
      d.perm.forEach((j, t) => {
        const x1 = PX0 + j * PW + PW / 2;
        const x2 = PX0 + t * PW + PW / 2;
        el('path', { d: `M${x1},${56 + 12}L${x2},${122 - 14}`, stroke: VIOLET, 'stroke-width': 1, 'stroke-dasharray': '3 3', fill: 'none' }, pmLinks);
      });
    }

    if (step === 2 || step === 4) {
      const model = getLive().trainer.state.model;
      const f = drawMaps(step, model);
      if (step === 4) {
        const outDigits = f.logits.map((l) => argmax(l));
        exIn.textContent = `example ${example().join(' ')}  →  model says ${outDigits.join(' ')}`;
        s.h.right.forEach((r, t) => {
          const hgt = P((r / s.n) * AH);
          accBars[t].setAttribute('y', P(AB - hgt));
          accBars[t].setAttribute('height', hgt);
        });
      }
    }

    if (step === 3) {
      const l = s.ln;
      [[charts[0], l.x, l.mean], [charts[1], l.y, l.meanY]].forEach(([c, v, mu]) => {
        v.forEach((z, j) => {
          const zz = Math.max(-3.2, Math.min(3.2, z)) * 16;
          c.bars[j].setAttribute('y', P(zz >= 0 ? c.y0 - zz : c.y0));
          c.bars[j].setAttribute('height', P(Math.abs(zz)));
        });
        c.mean.setAttribute('y1', P(c.y0 - mu * 16));
        c.mean.setAttribute('y2', P(c.y0 - mu * 16));
      });
    }

    const texts = readout(step, s);
    rowsT.forEach((t, k) => {
      t.textContent = texts[k] || '';
    });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `A diagram of one transformer block: digits plus positions, then layer norm, attention with 2 heads and an add; then layer norm, a small network and an add; then a final layer norm and the output. ${rest}.`;
    if (step === 1) return `The input ${s.demo.x.join(' ')} and the shuffled input ${s.demo.px.join(' ')}, with the untrained block's top output digit and probability at each position: ${s.demo.out.map((o) => `${o.digit} ${num(o.p, 3)}`).join(', ')}; and for the shuffle: ${s.demo.pout.map((o) => `${o.digit} ${num(o.p, 3)}`).join(', ')}. ${rest}.`;
    if (step === 2 || step === 4) return `Two 8 by 8 attention maps, one per head, for the example ${example().join(' ')}. ${rest}.`;
    if (step === 3) return `Bar charts of the 32 numbers at position ${pos + 1} before and after layer norm. ${rest}.`;
    return `A table of the model's parts and how many numbers each holds: ${PARTS.map((p) => `${p.name} ${p.count}`).join(', ')}; total ${count(TOTAL)}. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  const loop = createTrainLoop({
    step: (k) => getLive().advance(k),
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
    work: (k) => getLive().advance(k),
  });
  trainGroup.addEventListener('click', () => { if (step === 2 || step === 4) { render(); announce(spoken()); } });

  shBtn.addEventListener('click', () => {
    perm++;
    render();
    announce(spoken());
  });
  posIn.addEventListener('input', () => {
    pos = Math.min(LEN - 1, Math.max(0, (Number(posIn.value) || 1) - 1));
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
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      showGroup(shGroup, i === 1);
      showGroup(posGroup, i === 3);
      trainGroup.style.display = i === 2 || i === 4 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 4);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
