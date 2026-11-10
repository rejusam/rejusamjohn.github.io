// Lesson L4.6, "Going deeper: vanishing gradients": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-deep-networks.test.mjs);
// the DOM is only touched inside createStage. Every bar is a per-layer norm of the
// real gradient from batchGrad, every curve is a real training run from nn.js, and
// every number in the page's text comes from storyValues (checked by the tests).

import { circlesData, mlpInit, batchGrad, trainer, accuracyNet, predictProb, mlpForward, ACT } from '../assets/nn.js';
import { linearScale, el, axes, tween, networkDiagram } from '../assets/plot.js';
import { createTrainLoop, trainButtons } from '../assets/train-loop.js';

// ---- the network and the data ---------------------------------------------

export const SEED_DATA = 181;
export const SEED_INIT = 182;
export const SEED_SHUFFLE = 183;
export const N_TRAIN = 200;
export const DEPTH = 10;
export const WIDTH = 8;
export const SIZES = [2, ...Array(DEPTH).fill(WIDTH), 1];
export const N_LAYERS = SIZES.length - 1;

export const ACTS = ['sigmoid', 'tanh', 'relu'];
export const ACT_LABEL = { sigmoid: 'sigmoid', tanh: 'tanh', relu: 'ReLU' };
export const INITS = ['small', 'matched', 'large'];
export const SMALL_SD = 0.1;
export const LARGE_SD = 2;

// 'matched' is Xavier (variance 1/n_in) for sigmoid and tanh, He (2/n_in) for ReLU.
export function initFor(act, key) {
  if (key === 'small') return SMALL_SD;
  if (key === 'large') return LARGE_SD;
  return act === 'relu' ? 'he' : 'xavier';
}
export function initLabel(act, key) {
  const i = initFor(act, key);
  if (i === 'he') return 'He';
  if (i === 'xavier') return 'Xavier';
  return `sd ${i}`;
}

const TR = circlesData({ n: N_TRAIN, seed: SEED_DATA });
export const trainSet = () => circlesData({ n: N_TRAIN, seed: SEED_DATA });
export const startNet = (act, key) => mlpInit(SIZES, { hidden: act, init: initFor(act, key), seed: SEED_INIT });
export const paramCount = (sizes) => sizes.slice(1).reduce((s, n, l) => s + n * (sizes[l] + 1), 0);

const ALL = Array.from({ length: N_TRAIN }, (_, i) => i);
const frob = (M) => Math.sqrt(M.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));

// ‖∂ℒ/∂W⁽ℓ⁾‖ for ℓ = 1..11: the Frobenius norm of the gradient of the mean loss
// over all 200 training points, at the start. Computed once per setting, on demand.
const NORMS = new Map();
export function gradNorms(act, key) {
  const id = `${act}|${key}`;
  if (NORMS.has(id)) return NORMS.get(id);
  const norms = batchGrad(startNet(act, key), TR.X, TR.y, ALL).dW.map(frob);
  const first = norms[0];
  const last = norms[N_LAYERS - 1];
  const r = { act, key, norms, first, last, ratio: first / last, largest: Math.max(...norms), shrink: Math.pow(last / first, 1 / (N_LAYERS - 1)) };
  NORMS.set(id, r);
  return r;
}

// The mean of g′(z) over the 8 units and 200 training points, per hidden layer, at the start.
export function meanSlopes(act, key) {
  const net = startNet(act, key);
  const dg = ACT[act].df;
  const m = new Array(DEPTH).fill(0);
  for (const x of TR.X) {
    const { zs } = mlpForward(net, x);
    for (let l = 0; l < DEPTH; l++) m[l] += zs[l].reduce((t, z) => t + dg(z), 0) / WIDTH;
  }
  return m.map((v) => v / N_TRAIN);
}

// The sigmoid/Xavier network at the start: the spread of its output logit over
// the 200 training points (largest minus smallest) and the probability it gives.
export function sigmoidStart() {
  const net = startNet('sigmoid', 'matched');
  const zs = TR.X.map((x) => mlpForward(net, x).out[0]);
  const ps = TR.X.map((x) => predictProb(net, x));
  return { spread: Math.max(...zs) - Math.min(...zs), probMin: Math.min(...ps), probMax: Math.max(...ps) };
}

// The highest training loss after epoch 10, and its epoch (the first if tied).
export function bump(losses, from = 10) {
  let e = from + 1;
  for (let k = from + 1; k < losses.length; k++) if (losses[k] > losses[e]) e = k;
  return { epoch: e, loss: losses[e] };
}

// The bar chart's log scale: 10⁻¹⁴ to 10². Bars beyond it are cut off and flagged.
export const LOG_RANGE = [-14, 2];
export function barSpan(v) {
  const e = Math.log10(v);
  if (!(e <= LOG_RANGE[1])) return { e: LOG_RANGE[1], top: true, bottom: false };
  if (e < LOG_RANGE[0]) return { e: LOG_RANGE[0], top: false, bottom: true };
  return { e, top: false, bottom: false };
}

// ---- the two training runs ----------------------------------------------------

export const ETA = 0.05;
export const BATCH = 20;
export const EPOCHS = 100;
export const ITERS = (EPOCHS * N_TRAIN) / BATCH;
export const RUN_ACTS = ['sigmoid', 'relu'];
export const RUN_LABEL = { sigmoid: 'sigmoid, Xavier', relu: 'ReLU, He' };

export const makeTrainer = (act) => trainer({ net: startNet(act, 'matched'), X: TR.X, y: TR.y, batch: BATCH, kind: 'sgd', hyper: { eta: ETA }, seed: SEED_SHUFFLE });

// How far each layer's weights moved: ‖W_end − W_start‖ / ‖W_start‖ (Frobenius).
export function moved(start, end) {
  return start.W.map((M, l) => frob(end.W[l].map((row, j) => row.map((w, i) => w - M[j][i]))) / frob(M));
}

const RUNS = new Map();
const JOBS = new Map();

// k more steps of this run; true once its result is ready. The page calls this
// in small slices; the slicing does not change the result.
export function runWork(act, k) {
  if (RUNS.has(act)) return true;
  if (!JOBS.has(act)) JOBS.set(act, makeTrainer(act));
  const t = JOBS.get(act);
  t.step(Math.max(0, Math.min(k, ITERS - t.state.iter)));
  if (t.state.iter < ITERS && !t.state.diverged) return false;
  const net = t.state.net;
  const ps = TR.X.map((x) => predictProb(net, x));
  const losses = t.state.epochLoss.slice();
  RUNS.set(act, {
    act, losses, final: losses[EPOCHS], diverged: t.state.diverged,
    moved: moved(startNet(act, 'matched'), net),
    probMin: Math.min(...ps), probMax: Math.max(...ps),
    acc: accuracyNet(net, TR.X, TR.y),
  });
  JOBS.delete(act);
  return true;
}
export const runReady = (act) => RUNS.has(act);
export function runResult(act) {
  runWork(act, Infinity);
  return RUNS.get(act);
}

// ---- formatting and read-outs -----------------------------------------------

export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
const SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '-': '⁻' };
export const sup = (n) => String(n).split('').map((c) => SUP[c]).join('');
// Two significant figures: plain from 0.01 up to 1000, otherwise as m×10ⁿ.
export function sci(v) {
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1000 && Number(v.toPrecision(2)) < 1000) {
    const s = String(Number(v.toPrecision(2)));
    return s.startsWith('-') ? '−' + s.slice(1) : s;
  }
  let e = Math.floor(Math.log10(Math.abs(v)));
  let m = (v / Math.pow(10, e)).toFixed(1);
  if (Math.abs(Number(m)) >= 10) { e += 1; m = (v / Math.pow(10, e)).toFixed(1); }
  if (e === 0) return m;
  return `${m}×10${sup(e)}`;
}

export const diagramInfo = () => ({ params: paramCount(SIZES) });
export const barsInfo = (act, key) => ({ ...gradNorms(act, key), label: `${ACT_LABEL[act]}, ${initLabel(act, key)}` });
export const liveInfo = (sig, relu, running = false) => ({ sig, relu, running, epoch: Math.min(sig.epoch, relu.epoch) });
export const afterInfo = () => ({ runs: Object.fromEntries(RUN_ACTS.map((a) => [a, RUNS.get(a) || null])) });

export function readout(step, s) {
  if (step === 0) return ['2 inputs, 1 output', `${DEPTH} hidden layers of ${WIDTH}`, `${N_LAYERS} layers of weights`, `${s.params} weights and biases`];
  if (step === 1 || step === 2) {
    const rows = [`${s.label}, start`, `layer 1: ${sci(s.first)}`, `layer ${N_LAYERS}: ${sci(s.last)}`, `ratio 1/${N_LAYERS}: ${sci(s.ratio)}`];
    if (step === 2) rows.push(`÷${num(s.shrink, 1)} per layer back`);
    return rows;
  }
  if (step === 3) {
    const cut = s.norms.some((v) => barSpan(v).top);
    return [s.label, `layer 1: ${sci(s.first)}`, `layer ${N_LAYERS}: ${sci(s.last)}`, `ratio 1/${N_LAYERS}: ${sci(s.ratio)}`, `largest: ${sci(s.largest)}`, ...(cut ? ['bars cut off at 10²'] : [])];
  }
  if (step === 4) {
    const rows = [`epoch ${s.epoch} of ${EPOCHS}`, 'training loss ℒ',
      `${RUN_LABEL.sigmoid}: ${num(s.sig.epochLoss.at(-1), 4)}`,
      `${RUN_LABEL.relu}: ${num(s.relu.epochLoss.at(-1), 4)}`];
    if (s.running && s.epoch < EPOCHS) rows.push('training…');
    else if (s.epoch === 0) rows.push('press Train');
    else if (s.epoch < EPOCHS) rows.push('not finished: press Train');
    return rows;
  }
  if (step === 5) {
    const head = ['weights moved by the end', `layer: 1 · ${N_LAYERS}`];
    return [...head, ...RUN_ACTS.map((a) => {
      const r = s.runs[a];
      const name = a === 'relu' ? 'ReLU' : 'sigmoid';
      return r ? `${name}: ${small(r.moved[0])} · ${small(r.moved[N_LAYERS - 1])}` : `${name}: computing…`;
    })];
  }
  return [];
}
// Two decimals, or m×10ⁿ below 0.01.
const small = (v) => (v < 0.01 ? sci(v) : num(v, 2));

// Filled into the page at run time (constants).
export function pageValues() {
  return {
    params: String(paramCount(SIZES)),
    eta: String(ETA),
  };
}

// From the computed gradients and runs; written in the page and checked by the tests.
let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const sx = gradNorms('sigmoid', 'matched');
  v['sx-first'] = sci(sx.first);
  v['sx-last'] = sci(sx.last);
  v['sx-ratio'] = sci(sx.ratio);
  v['sx-shrink'] = num(sx.shrink, 1);
  v['sx-million'] = String(Math.round(1 / sx.ratio / 1e6));
  v['sx-step1'] = String(Math.round(sx.norms[1] / sx.norms[0]));
  const slopes = meanSlopes('sigmoid', 'matched');
  v['slope-min'] = num(Math.min(...slopes), 2);
  v['slope-max'] = num(Math.max(...slopes), 2);
  const he = gradNorms('relu', 'matched');
  v['he-ratio'] = num(he.ratio, 2);
  v['tanh-ratio'] = num(gradNorms('tanh', 'matched').ratio, 2);
  v['big-largest'] = sci(gradNorms('relu', 'large').largest);
  v['he-largest'] = sci(he.largest);
  v['tanh-big-ratio'] = num(gradNorms('tanh', 'large').ratio, 0);
  v['sig-big-ratio'] = sci(gradNorms('sigmoid', 'large').ratio);
  const s = runResult('sigmoid');
  const r = runResult('relu');
  const st = sigmoidStart();
  v['sig-spread'] = sci(st.spread);
  v['sig-p0'] = num(st.probMax, 3);
  v['sig-start'] = num(s.losses[0], 3);
  v['sig-end'] = num(s.final, 3);
  v['sig-prob'] = num(s.probMax, 3);
  v['relu-start'] = num(r.losses[0], 3);
  v['relu-10'] = num(r.losses[10], 3);
  v['relu-end'] = num(r.final, 4);
  const b = bump(r.losses);
  v['relu-bump'] = num(b.loss, 3);
  v['relu-bump-epoch'] = String(b.epoch);
  v['relu-acc'] = pct(r.acc);
  v['sig-moved-1'] = sci(s.moved[0]);
  v['sig-moved-11'] = num(s.moved[N_LAYERS - 1], 2);
  v['relu-moved-1'] = num(r.moved[0], 2);
  v['relu-moved-11'] = num(r.moved[N_LAYERS - 1], 2);
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

// The plot area (left) and the read-out panel (right).
const PX0 = 84;
const PX1 = 372;
const PY0 = 290;
const PY1 = 40;
const RX0 = 388;
const ROW = 20;
const P = (v) => Number(v.toFixed(2));

// A log-scale chart frame: grid every 2 decades, labels every 4.
function logFrame(parent, Y, [lo, hi], yLabel, px1 = PX1) {
  const g = el('g', { class: 'axes' }, parent);
  const font = "'JetBrains Mono', monospace";
  for (let e = lo; e <= hi; e += 2) {
    const y = P(Y(e));
    el('line', { x1: PX0, x2: px1, y1: y, y2: y, stroke: LINE, 'stroke-width': 1, opacity: 0.6, 'stroke-dasharray': '2 4' }, g);
    if ((e - lo) % 4 === 0 || e === hi) {
      el('text', { x: PX0 - 8, y: y + 4, 'text-anchor': 'end', fill: MUTED, 'font-family': font, 'font-size': 11, text: e === 0 ? '1' : `10${sup(e)}` }, g);
    }
  }
  el('line', { x1: PX0, x2: px1, y1: PY0, y2: PY0, stroke: LINE }, g);
  el('line', { x1: PX0, x2: PX0, y1: PY0, y2: PY1, stroke: LINE }, g);
  const xs = linearScale([0.5, N_LAYERS + 0.5], [PX0, px1]);
  for (let l = 1; l <= N_LAYERS; l++) el('text', { x: P(xs(l)), y: PY0 + 16, 'text-anchor': 'middle', fill: MUTED, 'font-family': font, 'font-size': 11, text: String(l) }, g);
  el('text', { x: (PX0 + px1) / 2, y: PY0 + 36, 'text-anchor': 'middle', fill: MUTED, 'font-family': font, 'font-size': 12, text: 'layer ℓ' }, g);
  const cx = PX0 - 70;
  const cy = (PY0 + PY1) / 2;
  el('text', { x: cx, y: cy, transform: `rotate(-90 ${cx} ${cy})`, 'text-anchor': 'middle', fill: MUTED, 'font-family': font, 'font-size': 12, text: yLabel }, g);
  return { g, xs };
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let act = 'sigmoid';
  let key = 'matched';
  let token = 0;
  // The bars' current heights (log10), tweened between settings.
  let shownE = null;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A network with ten hidden layers; bar charts, on a log scale, of the gradient size for each of its eleven layers of weights; and training-loss curves of a deep sigmoid network and a deep ReLU network.');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };

  // ---- Step 1: the network diagram (structure only).
  const netG = el('g', {}, svg);
  const labels = ['x', ...Array.from({ length: DEPTH }, (_, i) => String(i + 1)), 'out'];
  networkDiagram(netG, SIZES, { box: { x: 30, y: 62, w: 330, h: 228 }, r: 6, labels });
  el('text', { class: 'stage-text', x: 195, y: 30, 'text-anchor': 'middle', text: 'hidden layers 1 to 10' }, netG).style.fill = MUTED;

  // ---- Steps 2 to 4: gradient size per layer.
  const GY = linearScale(LOG_RANGE, [PY0, PY1]);
  const barG = el('g', {}, svg);
  const { xs: BX } = logFrame(barG, GY, LOG_RANGE, '‖∂ℒ/∂W‖, log scale');
  const BW = 16;
  const bars = [];
  const caps = [];
  for (let l = 1; l <= N_LAYERS; l++) {
    bars.push(el('rect', { x: P(BX(l) - BW / 2), width: BW, y: PY0, height: 0, fill: CYAN, 'fill-opacity': 0.85 }, barG));
    caps.push(el('path', { d: `M${P(BX(l) - 6)},${PY1 - 3}L${P(BX(l))},${PY1 - 11}L${P(BX(l) + 6)},${PY1 - 3}Z`, fill: STRONG }, barG));
  }
  const cutNote = el('text', { class: 'stage-text', x: (PX0 + PX1) / 2, y: PY1 - 16, 'text-anchor': 'middle', text: 'cut off at 10²', ...HALO }, barG);
  cutNote.style.fill = STRONG;

  // Step 3: σ′(z) inset under the read-out.
  const insG = el('g', {}, svg);
  const IX = linearScale([-6, 6], [RX0 + 30, 590]);
  const IY = linearScale([0, 0.3], [310, 190]);
  el('line', { x1: IX(-6), x2: IX(6), y1: IY(0), y2: IY(0), stroke: LINE }, insG);
  el('line', { x1: IX(0), x2: IX(0), y1: IY(0), y2: IY(0.3), stroke: LINE }, insG);
  el('line', { x1: IX(-6), x2: IX(6), y1: P(IY(0.25)), y2: P(IY(0.25)), stroke: MUTED, 'stroke-dasharray': '3 3' }, insG);
  {
    let d = '';
    for (let k = 0; k <= 120; k++) {
      const z = -6 + (12 * k) / 120;
      d += `${k ? 'L' : 'M'}${P(IX(z))},${P(IY(ACT.sigmoid.df(z)))}`;
    }
    el('path', { d, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, insG);
  }
  el('text', { class: 'stage-text', x: IX(-6), y: P(IY(0.25)) - 8, text: '1/4', ...HALO }, insG).style.fill = STRONG;
  el('text', { class: 'stage-text', x: IX(6), y: P(IY(0.25)) - 8, 'text-anchor': 'end', text: 'σ′(z)', ...HALO }, insG).style.fill = CYAN;
  el('text', { class: 'stage-text', x: IX(0), y: IY(0) + 16, 'text-anchor': 'middle', text: 'z = 0' }, insG).style.fill = MUTED;

  // ---- Step 5: training-loss curves.
  const lossG = el('g', {}, svg);
  const LX = linearScale([0, EPOCHS], [PX0, PX1]);
  const LY = linearScale([0, 1.3], [PY0, PY1]);
  axes(lossG, LX, LY, { ticks: 4, xLabel: 'epoch', yLabel: 'training loss' });
  const STYLE = { sigmoid: { stroke: VIOLET, dash: '6 4' }, relu: { stroke: CYAN, dash: null } };
  const curves = {};
  const ends = {};
  for (const a of RUN_ACTS) {
    curves[a] = el('path', { fill: 'none', stroke: STYLE[a].stroke, 'stroke-width': 2, 'stroke-dasharray': STYLE[a].dash, 'stroke-linejoin': 'round' }, lossG);
    ends[a] = a === 'relu'
      ? el('circle', { r: 4.5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, lossG)
      : el('rect', { width: 8, height: 8, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, lossG);
  }

  // ---- Step 6: how far each layer's weights moved.
  const MR = [-8, 0];
  const MY = linearScale(MR, [PY0, PY1]);
  const movG = el('g', {}, svg);
  const { xs: MX } = logFrame(movG, MY, MR, 'weights moved, log scale', 340);
  const movBars = {};
  for (const a of RUN_ACTS) {
    movBars[a] = [];
    for (let l = 1; l <= N_LAYERS; l++) {
      const x = a === 'sigmoid' ? MX(l) - 9 : MX(l) + 0.5;
      movBars[a].push(a === 'sigmoid'
        ? el('rect', { x: P(x + 0.75), width: 7, y: PY0, height: 0, fill: 'none', stroke: VIOLET, 'stroke-width': 1.5, 'stroke-dasharray': '3 2' }, movG)
        : el('rect', { x: P(x), width: 8, y: PY0, height: 0, fill: CYAN, 'fill-opacity': 0.85 }, movG));
    }
  }

  // ---- Legend for steps 5 and 6.
  const legG = el('g', {}, svg);
  RUN_ACTS.forEach((a, i) => {
    const y = 228 + 24 * i;
    el('line', { x1: RX0 + 4, x2: RX0 + 30, y1: y, y2: y, stroke: STYLE[a].stroke, 'stroke-width': 2.5, 'stroke-dasharray': STYLE[a].dash }, legG);
    el('text', { class: 'stage-text', x: RX0 + 38, y: y + 4, text: RUN_LABEL[a] }, legG).style.fill = STRONG;
  });

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4, 5].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  // The two button groups each get a small visible label so they read as two rows of choices.
  const groupBox = (g, text) => {
    Object.assign(g.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
    const tag = make('span', { textContent: text });
    tag.setAttribute('aria-hidden', 'true');
    Object.assign(tag.style, { color: MUTED, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', marginRight: '2px' });
    g.append(tag);
  };
  const actGroup = make('div');
  actGroup.setAttribute('role', 'group');
  actGroup.setAttribute('aria-label', 'activation');
  const actBtns = ACTS.map((a) => make('button', { type: 'button', textContent: ACT_LABEL[a] }));
  groupBox(actGroup, 'activation');
  actGroup.append(...actBtns);
  const initGroup = make('div');
  initGroup.setAttribute('role', 'group');
  initGroup.setAttribute('aria-label', 'starting weights');
  const INIT_TEXT = { small: 'sd 0.1', matched: 'Xavier / He', large: 'sd 2' };
  const initBtns = INITS.map((k) => make('button', { type: 'button', textContent: INIT_TEXT[k] }));
  groupBox(initGroup, 'start');
  initGroup.append(...initBtns);
  const trainGroup = make('div');
  controls.append(actGroup, initGroup, trainGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // The live runs of step 5, made when the step is first shown.
  let live = null;
  const liveRuns = () => {
    if (!live) live = { sigmoid: makeTrainer('sigmoid'), relu: makeTrainer('relu') };
    return live;
  };

  const barSetting = () => (step === 3 ? [act, key] : ['sigmoid', 'matched']);

  function info() {
    if (step === 0) return diagramInfo();
    if (step <= 3) { const [a, k] = barSetting(); return barsInfo(a, k); }
    if (step === 4) { const L = liveRuns(); return liveInfo(L.sigmoid.state, L.relu.state, loop.running()); }
    return afterInfo();
  }

  function drawBars(target) {
    const es = shownE || target.map((v) => barSpan(v).e);
    let cut = false;
    es.forEach((e, l) => {
      const y = GY(e);
      bars[l].setAttribute('y', P(y));
      bars[l].setAttribute('height', P(Math.max(0, PY0 - y)));
      const top = barSpan(target[l]).top && shownE === null;
      show(caps[l], top);
      cut = cut || top;
    });
    show(cutNote, cut);
  }

  function render() {
    show(netG, step === 0);
    show(barG, step >= 1 && step <= 3);
    show(insG, step === 2);
    show(lossG, step === 4);
    show(movG, step === 5);
    show(legG, step >= 4);
    const s = info();
    if (step >= 1 && step <= 3) drawBars(s.norms);
    if (step === 4) {
      for (const a of RUN_ACTS) {
        const Ls = (a === 'relu' ? s.relu : s.sig).epochLoss;
        curves[a].setAttribute('d', Ls.map((v, e) => `${e ? 'L' : 'M'}${P(LX(e))},${P(LY(Math.min(v, 1.3)))}`).join(''));
        const e = Ls.length - 1;
        const x = LX(e);
        const y = LY(Math.min(Ls[e], 1.3));
        if (a === 'relu') { ends[a].setAttribute('cx', P(x)); ends[a].setAttribute('cy', P(y)); } else { ends[a].setAttribute('x', P(x - 4)); ends[a].setAttribute('y', P(y - 4)); }
      }
    }
    if (step === 5) {
      for (const a of RUN_ACTS) {
        const r = s.runs[a];
        movBars[a].forEach((b, l) => {
          const y = r ? MY(Math.max(MR[0], Math.min(MR[1], Math.log10(r.moved[l])))) : PY0;
          b.setAttribute('y', P(y));
          b.setAttribute('height', P(PY0 - y));
        });
      }
    }
    actBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(ACTS[k] === act)));
    initBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(INITS[k] === key)));

    const texts = readout(step, s);
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
    const by = 24;
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
    if (i === 4) {
      if (s.epoch === 0) return `Press Train to train both networks. Training loss at the start: ${rows[2]}, ${rows[3]}.`;
      return `Epoch ${s.epoch} of ${EPOCHS}. Training loss: ${rows[2]}, ${rows[3]}.`;
    }
    return `${rows.join(', ')}.`;
  };

  // The bars grow from their last heights to the new ones, about half a second.
  function growBars(animate) {
    const my = ++token;
    const [a, k] = barSetting();
    const target = gradNorms(a, k).norms.map((v) => barSpan(v).e);
    const from = bars.map((b) => {
      const h = Number(b.getAttribute('height')) || 0;
      return LOG_RANGE[0] + (h / (PY0 - PY1)) * (LOG_RANGE[1] - LOG_RANGE[0]);
    });
    if (!animate || reduced) {
      shownE = null;
      render();
      return Promise.resolve();
    }
    return tween(500, (t) => {
      if (my !== token) return;
      shownE = target.map((e, l) => from[l] + (e - from[l]) * t);
      render();
    }, { reduced }).then(() => {
      if (my !== token) return;
      shownE = null;
      render();
    });
  }

  // ---- Step 5: both runs, step for step, a few steps each per frame so the
  // curves can be watched (about four seconds in all).
  const PER_FRAME = 4;
  const liveDone = () => { const L = liveRuns(); return L.sigmoid.state.iter >= ITERS && L.relu.state.iter >= ITERS; };
  const advance = (k) => {
    const L = liveRuns();
    for (const a of RUN_ACTS) {
      const t = L[a];
      t.step(Math.max(0, Math.min(k, ITERS - t.state.iter)));
    }
  };
  const loop = createTrainLoop({
    step: (n) => advance(Math.max(1, Math.min(PER_FRAME, Math.floor(n / 2)))),
    render,
    done: liveDone,
    onStop: () => {
      buttons.sync();
      render();
      if (liveDone()) announce(`Finished. ${spoken(4)}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { const L = liveRuns(); L.sigmoid.reset(); L.relu.reset(); },
    reduced,
    runToEnd: () => advance(ITERS),
    render: () => { render(); announce(spoken(4)); },
  });

  // Pausing does not call render, so redraw the read-out after any Train/Pause/Reset click.
  trainGroup.addEventListener('click', () => { if (step === 4) render(); });

  // ---- Step 6's runs, computed when the step is first shown, in slices of
  // about 6 ms per animation frame.
  let job = null;
  const cancelRuns = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const computeRuns = () => {
    if (job !== null || RUN_ACTS.every(runReady)) return;
    const tick = () => {
      job = null;
      if (step !== 5) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const a = RUN_ACTS.find((x) => !runReady(x));
        if (a === undefined) break;
        if (runWork(a, 5)) render();
      }
      if (RUN_ACTS.every(runReady)) { render(); announce(spoken(5)); } else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

  actBtns.forEach((btn, k) => btn.addEventListener('click', () => {
    act = ACTS[k];
    growBars(true).then(() => announce(spoken(3)));
  }));
  initBtns.forEach((btn, k) => btn.addEventListener('click', () => {
    key = INITS[k];
    growBars(true).then(() => announce(spoken(3)));
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
      const was = step;
      step = i;
      token++;
      shownE = null;
      if (i === 5) computeRuns(); else cancelRuns();
      actGroup.style.display = i === 3 ? 'flex' : 'none';
      initGroup.style.display = i === 3 ? 'flex' : 'none';
      showGroup(trainGroup, i === 4);
      controls.classList.toggle('has-train', i === 4);
      if (i >= 1 && i <= 3) {
        if (was < 1 || was > 3) bars.forEach((b) => b.setAttribute('height', 0));
        return growBars(animate).then(() => announce(spoken(i)));
      }
      render();
      announce(spoken(i));
      return Promise.resolve();
    },
  };
}
