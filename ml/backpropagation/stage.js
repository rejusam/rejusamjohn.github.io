// Lesson L4.3, "Backpropagation": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-backpropagation.test.mjs);
// the DOM is only touched inside createStage. Every number on the diagram is
// computed here from the hand-chosen weights and the one example; the numbers
// in the page's text come from the same helpers (pageValues, storyValues).

import { sigmoid } from '../assets/classify.js';
import { xorData, bceLogit, flatten, unflatten, trainer, accuracyNet } from '../assets/nn.js';
import { linearScale, el, axes, networkDiagram, fmt } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- the network and the one example

export const X_EX = [1.0, -0.5];
export const Y_EX = 1;
// Hand-chosen weights, one decimal each. W[l][j][i] maps unit i of layer l to
// unit j of layer l + 1, as in nn.js.
const START_W = [[[0.5, -0.4], [-0.3, 0.8]], [[0.7, -0.6]]];
const START_B = [[0.1, -0.2], [0.2]];
export const startNet = () => ({
  sizes: [2, 2, 1],
  hidden: 'sigmoid',
  W: START_W.map((M) => M.map((r) => r.slice())),
  b: START_B.map((v) => v.slice()),
});

export const NUDGE = 0.01;
export const ETA = 3;

// The six weights, in the order flatten() lists them.
export const LINKS = [
  { id: 'x1-h1', l: 0, j: 0, i: 0, short: 'x₁ → h₁', name: 'weight from x1 to h1' },
  { id: 'x2-h1', l: 0, j: 0, i: 1, short: 'x₂ → h₁', name: 'weight from x2 to h1' },
  { id: 'x1-h2', l: 0, j: 1, i: 0, short: 'x₁ → h₂', name: 'weight from x1 to h2' },
  { id: 'x2-h2', l: 0, j: 1, i: 1, short: 'x₂ → h₂', name: 'weight from x2 to h2' },
  { id: 'h1-o', l: 1, j: 0, i: 0, short: 'h₁ → out', name: 'weight from h1 to output' },
  { id: 'h2-o', l: 1, j: 0, i: 1, short: 'h₂ → out', name: 'weight from h2 to output' },
];
export const DEFAULT_LINK = 4;

// Forward pass, written out for the 2–2–1 network: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾,
// a⁽¹⁾ = σ(z⁽¹⁾), z⁽²⁾ = w⁽²⁾·a⁽¹⁾ + b⁽²⁾, p̂ = σ(z⁽²⁾), ℒ = BCE on the logit.
export function forwardPass(net, x, y) {
  const [W1, W2] = net.W;
  const z1 = W1.map((row, j) => {
    let s = net.b[0][j];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    return s;
  });
  const a1 = z1.map(sigmoid);
  let z2 = net.b[1][0];
  for (let i = 0; i < a1.length; i++) z2 += W2[0][i] * a1[i];
  return { x: x.slice(), z1, a1, z2, p: sigmoid(z2), loss: bceLogit(y, z2) };
}

// Backward pass: δ⁽²⁾ = p̂ − y, then δ⁽¹⁾ = (W⁽²⁾ᵀ δ⁽²⁾) ⊙ σ′(z⁽¹⁾).
// Each weight's gradient is δ at its output end times a at its input end;
// each bias's gradient is the δ of its unit.
export function backwardPass(net, x, y) {
  const f = forwardPass(net, x, y);
  const delta2 = f.p - y;
  const delta1 = f.z1.map((z, j) => {
    const s = sigmoid(z);
    return net.W[1][0][j] * delta2 * s * (1 - s);
  });
  return {
    f,
    delta2,
    delta1,
    gW: [delta1.map((d) => x.map((xi) => d * xi)), [f.a1.map((a) => delta2 * a)]],
    gb: [delta1.slice(), [delta2]],
  };
}

// The gradients in the order of flatten(net): W⁽¹⁾ rows, b⁽¹⁾, w⁽²⁾, b⁽²⁾.
export const flatGrad = (g) => [...g.gW[0][0], ...g.gW[0][1], ...g.gb[0], ...g.gW[1][0], ...g.gb[1]];

export const linkGrad = (net, k) => backwardPass(net, X_EX, Y_EX).gW[k.l][k.j][k.i];
export const linkWeight = (net, k) => net.W[k.l][k.j][k.i];

// A copy of net with one weight set to v.
export function withWeight(net, k, v) {
  const W = net.W.map((M) => M.map((r) => r.slice()));
  W[k.l][k.j][k.i] = v;
  return { ...net, sizes: net.sizes.slice(), W, b: net.b.map((c) => c.slice()) };
}

const exLoss = (net) => forwardPass(net, X_EX, Y_EX).loss;

// Nudge one weight by +0.01 and by −0.01: the real change in the loss, and the
// linear estimate gradient × nudge.
export function nudgeTable(net, k) {
  const base = exLoss(net);
  const g = linkGrad(net, k);
  return [NUDGE, -NUDGE].map((h) => ({
    h,
    real: exLoss(withWeight(net, k, linkWeight(net, k) + h)) - base,
    est: g * h,
  }));
}

// One step of gradient descent on every weight and bias, for one example.
export function oneStep(net, x, y, eta) {
  const g = flatGrad(backwardPass(net, x, y));
  const next = unflatten(net, flatten(net).map((p, i) => p - eta * g[i]));
  return { net: next, before: forwardPass(net, x, y).loss, after: forwardPass(next, x, y).loss };
}

// ---- the XOR run (step 6)

export const SEED_XOR = 161;
export const SEED_SHUFFLE = 162;
export const N_XOR = 20;
export const XOR_STEPS = 200;
// Steps per frame at most, so 200 steps take a few seconds to watch.
export const PACE = 2;
export const LOSS_MAX = 0.8;

export const xorSet = () => xorData({ n: N_XOR, seed: SEED_XOR });
const XOR = xorSet();

// Full batch: every step uses all 20 points, so the shuffle seed has no effect
// on the result beyond the order of a sum.
export const makeRun = () => trainer({ net: startNet(), X: XOR.X, y: XOR.y, batch: N_XOR, kind: 'sgd', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
export const finished = (t) => t.state.iter >= XOR_STEPS || t.state.diverged;
export function advance(t, k) {
  const left = XOR_STEPS - t.state.iter;
  if (left > 0) t.step(Math.min(k, left));
  return t.state;
}
export const frameStep = (t, n) => advance(t, Math.min(n, PACE));

export function runInfo(t, running = false) {
  const e = t.state.epochLoss;
  return { iter: t.state.iter, loss: e[e.length - 1], acc: accuracyNet(t.state.net, XOR.X, XOR.y), done: finished(t), running };
}

// The run's state in words comes from the shared trainStatus (train-loop.js).
export { trainStatus };

// ---- formatting

// Three significant figures with a true minus sign, and never "−0.00".
export function sig3(v) {
  if (v === 0) return '0.00';
  const s = v.toPrecision(3);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
const signed3 = (v) => (v > 0 ? '+' : '') + sig3(v);
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
const nudgeLabel = (h) => (h > 0 ? '+' : '−') + String(Math.abs(h));

// ---- read-out rows (at most 5, each at most 26 characters)

export function readout(step, { net, link, nudge = 0, run }) {
  if (step === 5) {
    const r = oneStep(startNet(), X_EX, Y_EX, ETA);
    const rows = [`one step on x, η = ${ETA}:`, `ℒ ${sig3(r.before)} → ${sig3(r.after)}`,
      `XOR train: step ${run.iter} of ${XOR_STEPS}`, `loss ${sig3(run.loss)} · acc ${pct(run.acc)}`];
    rows.push(trainStatus(run));
    return rows;
  }
  const shown = step === 4 && nudge ? withWeight(net, link, linkWeight(net, link) + nudge * NUDGE) : net;
  const g = backwardPass(shown, X_EX, Y_EX);
  const f = g.f;
  if (step === 0) {
    return ['forward pass', `b⁽¹⁾ = (${fmt(net.b[0][0])}, ${fmt(net.b[0][1])})`, `b⁽²⁾ = ${fmt(net.b[1][0])}`, `p̂ = σ(z⁽²⁾) = ${sig3(f.p)}`];
  }
  if (step === 1) return [`loss for y = ${Y_EX}`, 'ℒ = −ln p̂', `= −ln ${sig3(f.p)}`, `= ${sig3(f.loss)}`];
  if (step === 2) return ['output error', 'δ⁽²⁾ = p̂ − y', `= ${sig3(f.p)} − ${Y_EX}`, `= ${sig3(g.delta2)}`];
  const d = link.l === 1 ? g.delta2 : g.delta1[link.j];
  const a = link.l === 1 ? f.a1[link.i] : X_EX[link.i];
  const head = `${link.short}: w = ${fmt(linkWeight(shown, link))}`;
  if (step === 3) return [head, '∂ℒ/∂w = δ × a', `= ${sig3(d)} × ${sig3(a)}`, `= ${sig3(d * a)}`];
  const rows = nudgeTable(net, link);
  const out = [head];
  for (const r of rows) {
    const mark = nudge && Math.sign(nudge) === Math.sign(r.h) ? '▸' : ' ';
    out.push(`${mark}${nudgeLabel(r.h)} real Δℒ ${signed3(r.real)}`, `  grad × ${NUDGE} ${signed3(r.est)}`);
  }
  return out;
}

// Filled into the page at run time (cheap to compute).
export function pageValues() {
  const net = startNet();
  const g = backwardPass(net, X_EX, Y_EX);
  const f = g.f;
  const k = LINKS[DEFAULT_LINK];
  const nt = nudgeTable(net, k);
  const step = oneStep(net, X_EX, Y_EX, ETA);
  return {
    'z1': sig3(f.z1[0]),
    'z1b': sig3(f.z1[1]),
    'a1': sig3(f.a1[0]),
    'a1b': sig3(f.a1[1]),
    'z2': sig3(f.z2),
    'p': sig3(f.p),
    'loss': sig3(f.loss),
    'delta2': sig3(g.delta2),
    'delta1': sig3(g.delta1[0]),
    'delta1b': sig3(g.delta1[1]),
    'grad-h1o': sig3(linkGrad(net, k)),
    'grad-x1h1': sig3(linkGrad(net, LINKS[0])),
    'nudge-real': sig3(nt[0].real),
    'nudge-est': sig3(nt[0].est),
    'eta': String(ETA),
    'step-loss': sig3(step.after),
    'n-xor': String(N_XOR),
    'steps': String(XOR_STEPS),
  };
}

// From the seeded XOR run. Written in the page and checked by the tests; the
// page does not train on load, so these are not recomputed in the browser.
export function storyValues() {
  const t = makeRun();
  const first = runInfo(t);
  let firstAll = null;
  while (!finished(t)) {
    advance(t, 1);
    if (firstAll === null && accuracyNet(t.state.net, XOR.X, XOR.y) === 1) firstAll = t.state.iter;
  }
  const end = runInfo(t);
  return {
    'xor-loss0': sig3(first.loss),
    'xor-acc0': pct(first.acc),
    'xor-loss': sig3(end.loss),
    'xor-acc': pct(end.acc),
    'xor-first': String(firstAll),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';

// Diagram geometry (600 × 340 view box).
const NX = [70, 290, 470];
const NY = [[112, 252], [112, 252], [182]];
const R = 19;
const ROW = 20;
// Where each link's label sits, and its text-anchor.
// The diagonal labels run along their own link: x₁ → h₂ just above it,
// x₂ → h₁ just below it, so neither sits in the wedge between the two.
const DIAG = Math.atan2(NY[1][1] - NY[0][0], NX[1] - NX[0]);
const along = (x0, y0, x1, y1, t, off) => {
  const len = Math.hypot(x1 - x0, y1 - y0);
  const ux = (x1 - x0) / len;
  const uy = (y1 - y0) / len;
  // Unit normal pointing to the lower side of the link's direction of travel.
  return [Number((x0 + t * (x1 - x0) - uy * off).toFixed(2)), Number((y0 + t * (y1 - y0) + ux * off).toFixed(2))];
};
const DEG = (DIAG * 180) / Math.PI;
const LABEL_AT = {
  'x1-h1': [165, 102, 'middle', 0],
  'x1-h2': [...along(NX[0], NY[0][0], NX[1], NY[1][1], 0.2, -9), 'start', DEG],
  'x2-h1': [...along(NX[0], NY[0][1], NX[1], NY[1][0], 0.2, 15), 'start', -DEG],
  'x2-h2': [165, 272, 'middle', 0],
  'h1-o': [390, 140, 'start', 0],
  'h2-o': [390, 236, 'start', 0],
};

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const net = startNet();
  let step = 0;
  let sel = DEFAULT_LINK;
  let nudge = 0;
  const run = makeRun();
  let fadeToken = 0;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A network with two inputs, two hidden units and one output, each link labelled with its weight and each unit with its values.');

  const P = (v) => Number(v.toFixed(2));
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const halo = { stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
  const text = (parent, x, y, anchor = 'middle', cls = 'stage-text', extra = {}) =>
    el('text', { class: cls, x, y, 'text-anchor': anchor, ...halo, ...extra }, parent);

  // ---- The worked diagram (steps 1 to 5).
  const diaG = el('g', {}, svg);
  const pos = (l, j) => [NX[l], NY[l][j]];
  const ends = (k) => [pos(k.l, k.i), pos(k.l + 1, k.j)];
  const selHalo = el('line', { stroke: CYAN, 'stroke-opacity': 0.35, 'stroke-width': 10, 'stroke-linecap': 'round' }, diaG);
  const linkEls = LINKS.map((k) => {
    const [[x1, y1], [x2, y2]] = ends(k);
    return el('line', { class: 'net-link', x1, y1, x2, y2 }, diaG);
  });
  const nodeG = el('g', {}, diaG);
  const names = [['x₁', 'x₂'], ['h₁', 'h₂'], ['out']];
  names.forEach((col, l) => col.forEach((nm, j) => {
    const [cx, cy] = pos(l, j);
    el('circle', { class: 'net-node', cx, cy, r: R }, nodeG);
    el('text', { class: 'stage-text', x: cx, y: cy + 5, 'text-anchor': 'middle', fill: STRONG, text: nm }, nodeG);
  }));

  // Link labels: the weight, or the gradient on the chosen link (steps 4 and 5).
  const labG = el('g', {}, diaG);
  const linkLabels = LINKS.map((k) => {
    const [x, y, anchor, deg] = LABEL_AT[k.id];
    return text(labG, x, y, anchor, 'stage-text', deg ? { transform: `rotate(${Number(deg.toFixed(2))} ${x} ${y})` } : {});
  });

  // Node labels, grouped by column so the passes can fill in order.
  const colG = [0, 1, 2].map(() => el('g', {}, diaG));
  const inLab = [text(colG[0], NX[0], NY[0][0] - R - 10), text(colG[0], NX[0], NY[0][1] + R + 20)];
  // Hidden: top unit's lines above it, bottom unit's below it.
  const hidY = [[NY[1][0] - R - 44, NY[1][0] - R - 26], [NY[1][1] + R + 20, NY[1][1] + R + 38]];
  const hidLab = [0, 1].map((j) => hidY[j].map((y) => text(colG[1], NX[1], y)));
  const hidDeltaG = el('g', {}, diaG);
  const hidDelta = [NY[1][0] - R - 8, NY[1][1] + R + 56].map((y) => text(hidDeltaG, NX[1], y, 'middle', 'stage-text', { fill: VIOLET }));
  const OX = NX[2] + R + 10;
  const outLab = [text(colG[2], OX, NY[2][0] - 14, 'start'), text(colG[2], OX, NY[2][0] + 5, 'start')];
  const outDeltaG = el('g', {}, diaG);
  const outDelta = text(outDeltaG, OX, NY[2][0] + 24, 'start', 'stage-text', { fill: VIOLET });
  const lossG = el('g', {}, diaG);
  const lossLab = [text(lossG, OX, 274, 'start'), text(lossG, OX, 293, 'start', 'stage-text', { fill: STRONG })];

  // Focusable hit areas over the links (steps 4 and 5): hover, focus or click
  // chooses a weight; the select in the controls does the same on touch.
  const hitG = el('g', {}, diaG);
  const hits = LINKS.map((k, n) => {
    const [[x1, y1], [x2, y2]] = ends(k);
    const h = el('line', { x1, y1, x2, y2, stroke: 'transparent', 'stroke-width': 16, 'pointer-events': 'stroke', tabindex: 0, role: 'button', 'aria-label': k.name, style: 'cursor: pointer' }, hitG);
    const pick = () => choose(n);
    h.addEventListener('pointerenter', pick);
    h.addEventListener('focus', pick);
    h.addEventListener('click', pick);
    h.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    return h;
  });

  // ---- The XOR run (step 6): loss curve left, the same network's weights right.
  const LX = linearScale([0, XOR_STEPS], [64, 320]);
  const LY = linearScale([0, LOSS_MAX], [290, 50]);
  const runG = el('g', {}, svg);
  axes(runG, LX, LY, { xLabel: 'step', ticks: 4 });
  el('text', { class: 'stage-text', x: 70, y: 36, text: 'mean loss on the 20 XOR points' }, runG);
  const lossPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, runG);
  const small = networkDiagram(runG, [2, 2, 1], { box: { x: 410, y: 58, w: 160, h: 96 }, r: 14, labels: ['inputs', 'hidden', 'output'] });

  // ---- Read-out box, top right.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.94, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4].map(() => el('text', { class: 'value', style: 'white-space: pre' }, box));

  // ---- Controls: weight picker and nudge (steps 4 and 5), Train (step 6).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const pickGroup = make('div');
  const pickLabel = make('label');
  const select = make('select');
  LINKS.forEach((k, n) => {
    const o = make('option', { value: String(n), textContent: k.short });
    o.setAttribute('aria-label', k.name);
    select.append(o);
  });
  select.value = String(sel);
  pickLabel.append('weight', ' ', select);
  pickGroup.append(pickLabel);
  const nudgeGroup = make('div');
  const nudgeBtns = [1, -1].map((s) => {
    const b = make('button', { type: 'button', textContent: `${s > 0 ? '+' : '−'}${NUDGE}` });
    b.setAttribute('aria-label', `nudge the weight by ${s > 0 ? 'plus' : 'minus'} ${NUDGE}`);
    b.addEventListener('click', () => {
      nudge = nudge === s ? 0 : s;
      render();
      announce(spoken(step));
    });
    nudgeGroup.append(b);
    return b;
  });
  const trainGroup = make('div');
  controls.append(pickGroup, nudgeGroup, trainGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  select.addEventListener('change', () => choose(Number(select.value)));

  function choose(n) {
    if (step !== 3 && step !== 4) return;
    if (n === sel) return;
    sel = n;
    nudge = 0;
    select.value = String(n);
    render();
    announce(spoken(step));
  }

  const shownNet = () => (step === 4 && nudge ? withWeight(net, LINKS[sel], linkWeight(net, LINKS[sel]) + nudge * NUDGE) : net);

  function render() {
    const onDia = step <= 4;
    show(diaG, onDia);
    show(runG, step === 5);
    if (onDia) {
      const cur = shownNet();
      const g = backwardPass(cur, X_EX, Y_EX);
      const f = g.f;
      const k = LINKS[sel];
      const picking = step === 3 || step === 4;
      LINKS.forEach((lk, n) => {
        const w = linkWeight(cur, lk);
        linkEls[n].setAttribute('class', w < 0 ? 'net-link w-neg' : 'net-link w-pos');
        linkEls[n].setAttribute('stroke-width', P(Math.min(5, Math.max(1, 3 * Math.abs(w)))));
        const isSel = picking && n === sel;
        const lab = linkLabels[n];
        lab.textContent = isSel ? `∂ℒ/∂w = ${sig3(g.gW[lk.l][lk.j][lk.i])}` : fmt(w);
        lab.setAttribute('fill', isSel ? CYAN : MUTED);
        hits[n].setAttribute('aria-pressed', String(isSel));
      });
      show(selHalo, picking);
      const [[x1, y1], [x2, y2]] = ends(k);
      Object.entries({ x1, y1, x2, y2 }).forEach(([a, v]) => selHalo.setAttribute(a, v));
      show(hitG, picking);
      inLab[0].textContent = `x₁ = ${sig3(X_EX[0])}`;
      inLab[1].textContent = `x₂ = ${sig3(X_EX[1])}`;
      [0, 1].forEach((j) => {
        hidLab[j][0].textContent = `z = ${sig3(f.z1[j])}`;
        hidLab[j][1].textContent = `a = ${sig3(f.a1[j])}`;
        hidDelta[j].textContent = `δ = ${sig3(g.delta1[j])}`;
      });
      outLab[0].textContent = `z = ${sig3(f.z2)}`;
      outLab[1].textContent = `p̂ = ${sig3(f.p)}`;
      outDelta.textContent = `δ = ${sig3(g.delta2)}`;
      lossLab[0].textContent = `y = ${Y_EX}`;
      lossLab[1].textContent = `ℒ = ${sig3(f.loss)}`;
      show(lossG, step >= 1);
      show(outDeltaG, step >= 2);
      show(hidDeltaG, step >= 3);
    } else {
      const e = run.state.epochLoss;
      lossPath.setAttribute('d', e.map((v, i) => `${i ? 'L' : 'M'}${P(LX(i))},${P(LY(Math.min(v, LOSS_MAX)))}`).join(''));
      small.update({ weights: run.state.net.W });
    }
    nudgeBtns.forEach((b, n) => b.setAttribute('aria-pressed', String(nudge === (n ? -1 : 1))));

    const texts = readout(step, { net, link: LINKS[sel], nudge, run: runInfo(run, loop.activity()) });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let wd = 230;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.length && lens.every((len) => len > 0)) wd = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = 598 - wd;
    const by = step === 5 ? 186 : 4;
    boxRect.setAttribute('x', P(bx));
    boxRect.setAttribute('y', by);
    boxRect.setAttribute('width', P(wd));
    boxRect.setAttribute('height', 14 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', P(bx + 12));
      r.setAttribute('y', by + 7 + ROW * i + ROW / 2 + 5);
    });
  }

  // Fade groups in one after another (the passes filling the diagram).
  function fadeIn(groups) {
    const token = ++fadeToken;
    groups.forEach((g) => { g.style.transition = 'none'; g.style.opacity = '0'; });
    return new Promise((resolve) => {
      groups.forEach((g, n) => {
        setTimeout(() => {
          if (token !== fadeToken) return;
          g.style.transition = 'opacity 300ms ease-out';
          g.style.opacity = '1';
        }, 40 + 260 * n);
      });
      setTimeout(resolve, 40 + 260 * groups.length + 300);
    });
  }
  const clearFades = () => {
    fadeToken++;
    for (const g of [labG, ...colG, lossG, outDeltaG, hidDeltaG]) { g.style.transition = 'none'; g.style.opacity = '1'; }
  };

  const spoken = (i) => {
    const cur = shownNet();
    const g = backwardPass(cur, X_EX, Y_EX);
    const f = g.f;
    const k = LINKS[sel];
    if (i === 0) return `Forward pass for x = (1.0, −0.5). Hidden unit 1: z = ${sig3(f.z1[0])}, a = ${sig3(f.a1[0])}. Hidden unit 2: z = ${sig3(f.z1[1])}, a = ${sig3(f.a1[1])}. Output: z = ${sig3(f.z2)}, p̂ = ${sig3(f.p)}.`;
    if (i === 1) return `With y = 1 the loss is minus the log of ${sig3(f.p)}, which is ${sig3(f.loss)}.`;
    if (i === 2) return `Output error δ = p̂ − y = ${sig3(g.delta2)}.`;
    const gr = g.gW[k.l][k.j][k.i];
    if (i === 3) return `Hidden errors δ = ${sig3(g.delta1[0])} and ${sig3(g.delta1[1])}. Chosen: ${k.name}, gradient ${sig3(gr)}.`;
    if (i === 4) {
      const rows = nudgeTable(net, k);
      const now = nudge ? ` Nudged by ${nudge > 0 ? 'plus' : 'minus'} ${NUDGE}: loss ${sig3(f.loss)}.` : '';
      return `${k.name}. Nudge plus ${NUDGE}: real change ${sig3(rows[0].real)}, gradient times ${NUDGE} ${sig3(rows[0].est)}. Nudge minus ${NUDGE}: real change ${sig3(rows[1].real)}, estimate ${sig3(rows[1].est)}.${now}`;
    }
    const s = runInfo(run);
    return `Press Train to run ${XOR_STEPS} steps of gradient descent on the XOR points. Step ${s.iter}: mean loss ${sig3(s.loss)}, accuracy ${pct(s.acc)}.`;
  };

  // ---- The training loop (step 6 only).
  const loop = createTrainLoop({
    step: (n) => frameStep(run, n),
    render: () => render(),
    done: () => finished(run),
    onStop: () => {
      buttons.sync();
      render();
      if (finished(run)) announce(`Finished ${XOR_STEPS} steps. ${spoken(5)}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { run.reset(); },
    reduced,
    work: (k) => { advance(run, k); },
  });

  // Pausing does not call render, so redraw the read-out after any Train/Pause/Reset click.
  trainGroup.addEventListener('click', () => { if (step === 5) render(); });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(() => render(), () => {});

  return {
    count: 6,
    show(i, animate) {
      buttons.stop();
      step = i;
      nudge = 0;
      showGroup(pickGroup, i === 3 || i === 4);
      showGroup(nudgeGroup, i === 4);
      showGroup(trainGroup, i === 5);
      controls.classList.toggle('has-train', i === 5);
      clearFades();
      render();
      announce(spoken(i));
      if (!animate || reduced) return Promise.resolve();
      if (i === 0) return fadeIn([colG[0], labG, colG[1], colG[2]]);
      if (i === 1) return fadeIn([lossG]);
      if (i === 2) return fadeIn([outDeltaG]);
      if (i === 3) return fadeIn([hidDeltaG]);
      return Promise.resolve();
    },
  };
}
