// Lesson 11, "k-nearest neighbours": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-k-nearest-neighbours.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mean, sd } from '../assets/maths.js';
import { blobs2, knnPredict, accuracy } from '../assets/classify.js';
import { linearScale, clamp, el, axes, tween, classMarker, regionMap, draggable } from '../assets/plot.js';

const BLOBS = { mu0: [2, 2], mu1: [4, 3.4], sd: 1.1 };
export const SEED_TRAIN = 41;
export const SEED_TEST = 42;
export const KS = Array.from({ length: 13 }, (_, i) => 2 * i + 1);
export const K_VOTE = 3;
export const SCALE = 100;
// The new case's starting position (feature 1, feature 2).
export const Q0 = [3.5, 2.5];
export const MODES = ['raw', 'x100', 'std'];
// Region map cells: square, 4.5 user units a side.
export const GRID = [60, 56];
const MODE_NAME = { raw: 'original units', x100: 'feature 2 × 100', std: 'standardised' };

export const trainData = () => blobs2({ n0: 20, n1: 20, ...BLOBS, seed: SEED_TRAIN });
export const testData = () => blobs2({ n0: 150, n1: 150, ...BLOBS, seed: SEED_TEST });

const TR = trainData();
const TE = testData();

// How each mode measures distance: feature 2 times 1 or 100, then, for the
// standardised modes, z = (value − training mean) / training SD (population SD,
// dividing by n). 'std' standardises the ×100 copy; 'std-raw' the original.
// c is the overall stretch of each feature, used to draw the neighbourhood.
const tCache = new Map();
export function transform(mode) {
  if (tCache.has(mode)) return tCache.get(mode);
  const f2 = mode === 'x100' || mode === 'std' ? SCALE : 1;
  const scaled = (x) => [x[0], x[1] * f2];
  let out;
  if (mode === 'std' || mode === 'std-raw') {
    const S = TR.X.map(scaled);
    const mu = [0, 1].map((j) => mean(S.map((r) => r[j])));
    const sig = [0, 1].map((j) => sd(S.map((r) => r[j]), { sample: false }));
    out = {
      c: [1 / sig[0], f2 / sig[1]],
      apply: (x) => {
        const s = scaled(x);
        return [(s[0] - mu[0]) / sig[0], (s[1] - mu[1]) / sig[1]];
      },
    };
  } else {
    out = { c: [1, f2], apply: scaled };
  }
  out.train = TR.X.map(out.apply);
  tCache.set(mode, out);
  return out;
}

// The k nearest training points to q (in the mode's units) and their vote.
// radii: the neighbourhood drawn on the original axes, an ellipse through the k-th.
export function neighbours(q, k, mode = 'raw') {
  const t = transform(mode);
  const tq = t.apply(q);
  const r = knnPredict(t.train, TR.y, tq, k);
  const far = t.train[r.idx[k - 1]];
  const dist = Math.hypot(far[0] - tq[0], far[1] - tq[1]);
  return { ...r, dist, radii: [dist / t.c[0], dist / t.c[1]] };
}

const pCache = new Map();
export function predictTest(k, mode = 'raw') {
  const key = `${k}:${mode}`;
  if (!pCache.has(key)) {
    const t = transform(mode);
    pCache.set(key, TE.X.map((x) => knnPredict(t.train, TR.y, t.apply(x), k).cls));
  }
  return pCache.get(key);
}

// Training accuracy counts each training point in its own vote.
let kRows = null;
export function kTable() {
  if (!kRows) {
    kRows = KS.map((k) => ({
      k,
      train: accuracy(TR.y, TR.X.map((x) => knnPredict(TR.X, TR.y, x, k).cls)),
      test: accuracy(TE.y, predictTest(k)),
    }));
  }
  return kRows;
}

// Highest test accuracy; a tie would go to the smaller k.
export function bestK() {
  return kTable().reduce((a, b) => (b.test > a.test ? b : a)).k;
}

export function unitsTable(k) {
  const raw = predictTest(k, 'raw');
  const big = predictTest(k, 'x100');
  const z = predictTest(k, 'std');
  const zr = predictTest(k, 'std-raw');
  return {
    raw: accuracy(TE.y, raw),
    x100: accuracy(TE.y, big),
    std: accuracy(TE.y, z),
    changed: raw.filter((v, i) => v !== big[i]).length,
    stdSame: z.every((v, i) => v === zr[i]),
  };
}

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
const clsWord = (c) => `class ${c}`;

export function readout(step, { k = K_VOTE, mode = 'raw', q = Q0 } = {}) {
  const rows = kTable();
  const row = (kk) => rows.find((r) => r.k === kk);
  if (step === 0) return [`${TR.X.length} training points`, `circles (class 0): ${TR.y.filter((v) => v === 0).length}`, `triangles (class 1): ${TR.y.filter((v) => v === 1).length}`];
  if (step === 1) {
    const n = neighbours(q, K_VOTE);
    return [`k = ${K_VOTE}`, `circles: ${K_VOTE - n.votes1}`, `triangles: ${n.votes1}`, `prediction: ${clsWord(n.cls)}`];
  }
  if (step === 2 || step === 3) {
    const kk = step === 2 ? 1 : k;
    const r = row(kk);
    return [`k = ${kk}`, `train accuracy: ${pct(r.train)}`, `test accuracy: ${pct(r.test)}`];
  }
  const b = bestK();
  if (step === 4) {
    const r = row(b);
    return [`best k = ${b}`, `test accuracy: ${pct(r.test)}`, `train accuracy: ${pct(r.train)}`];
  }
  const u = unitsTable(b);
  const acc = { raw: u.raw, x100: u.x100, std: u.std }[mode];
  return [MODE_NAME[mode], `k = ${b}`, `test accuracy: ${pct(acc)}`, `new case: ${clsWord(neighbours(q, b, mode).cls)}`];
}

export function pageValues() {
  const rows = kTable();
  const b = bestK();
  const br = rows.find((r) => r.k === b);
  const u = unitsTable(b);
  const q3 = neighbours(Q0, K_VOTE);
  const qa = neighbours(Q0, b, 'raw');
  const qb = neighbours(Q0, b, 'x100');
  const qc = neighbours(Q0, b, 'std');
  return {
    'q-x1': String(Q0[0]),
    'q-x2': String(Q0[1]),
    'q3-votes0': String(K_VOTE - q3.votes1),
    'q3-votes1': String(q3.votes1),
    'q3-cls': String(q3.cls),
    'k1-test': pct(rows[0].test),
    'k3-train': pct(rows[1].train),
    'k25-train': pct(rows[rows.length - 1].train),
    'k25-test': pct(rows[rows.length - 1].test),
    'best-k': String(b),
    'best-test': pct(br.test),
    'best-train': pct(br.train),
    'u-raw': pct(u.raw),
    'u-x100': pct(u.x100),
    'u-std': pct(u.std),
    'u-changed': String(u.changed),
    'n-test': String(TE.X.length),
    'qb-raw-votes1': String(qa.votes1),
    'qb-raw-cls': String(qa.cls),
    'qb-x100-votes1': String(qb.votes1),
    'qb-x100-cls': String(qb.cls),
    'qb-std-cls': String(qc.cls),
    'qb-shared': String(qa.idx.filter((i) => qb.idx.includes(i)).length),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The scatter: 36 user units per feature unit on both axes, so a circle of
// equal distance looks round.
const PX0 = 56;
const PX1 = 326;
const PY0 = 286;
const PY1 = 34;
const X_DOM = [-0.5, 7];
const Y_DOM = [0, 7];
const RX0 = 352;
const BOX_Y = 8;
const ROW = 20;
// The accuracy chart.
const CX0 = 404;
const CX1 = 586;
const CY0 = 286;
const CY1 = 150;
const ACC_DOM = [0.75, 1];

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const rows = kTable();
  const best = bestK();
  let k = 5;
  let mode = 'raw';
  let q = Q0.slice();

  const X = linearScale(X_DOM, [PX0, PX1]);
  const Y = linearScale(Y_DOM, [PY0, PY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Forty training points on two features: class 0 as cyan circles, class 1 as violet triangles. A new case can be dragged; lines join it to its nearest neighbours. Shaded regions show which class k-NN predicts, with a line along the boundary. A chart on the right shows training and test accuracy for each k.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l11-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  const regionG = el('g', {}, svg);
  const region = regionMap(regionG, X, Y, () => 0, { nx: GRID[0], ny: GRID[1] });
  const axesRaw = axes(svg, X, Y, { xLabel: 'feature 1', yLabel: 'feature 2' });
  const axesBig = axes(svg, X, linearScale([0, 700], [PY0, PY1]), { xLabel: 'feature 1', yLabel: 'feature 2 × 100' });

  // Neighbourhood: the ellipse through the k-th neighbour, then the links.
  const hoodG = el('g', { 'clip-path': 'url(#l11-plot)' }, svg);
  const hood = el('ellipse', { fill: STRONG, 'fill-opacity': 0.06, stroke: STRONG, 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }, hoodG);
  const linksG = el('g', {}, hoodG);

  const pointsG = el('g', {}, svg);
  TR.X.forEach((x, i) => classMarker(pointsG, X(x[0]), Y(x[1]), TR.y[i], { r: 5 }));
  const ringsG = el('g', {}, svg);

  // The new case: a hollow diamond on a larger, focusable hit area.
  const queryG = el('g', {}, svg);
  const focusRing = el('circle', { r: 14, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, queryG);
  const diamond = el('path', { fill: BG, stroke: STRONG, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, queryG);
  const handle = el('circle', {
    class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'button',
    'aria-roledescription': 'movable point',
  }, queryG);
  handle.style.outline = 'none';
  let focused = false;

  // Key, above the scatter.
  const keyG = el('g', {}, svg);
  classMarker(keyG, PX0 + 6, 16, 0, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 16, y: 20, text: 'class 0' }, keyG);
  classMarker(keyG, PX0 + 100, 16, 1, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 110, y: 20, text: 'class 1' }, keyG);
  const keyNew = el('g', {}, keyG);
  el('path', { d: diamondD(PX0 + 198, 16, 6), fill: BG, stroke: STRONG, 'stroke-width': 1.8 }, keyNew);
  el('text', { class: 'stage-text', x: PX0 + 209, y: 20, text: 'new case' }, keyNew);

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // Accuracy chart: train solid with dots, test dashed with squares.
  const chartG = el('g', {}, svg);
  const KX = linearScale([1, 25], [CX0 + 8, CX1 - 4]);
  const AY = linearScale(ACC_DOM, [CY0, CY1]);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, chartG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, chartG);
  for (const a of [0.75, 0.85, 0.95]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: AY(a), y2: AY(a), stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: AY(a) + 4, 'text-anchor': 'end', text: `${Math.round(a * 100)}%` }, chartG);
  }
  for (const kk of [1, 5, 9, 13, 17, 21, 25]) {
    el('line', { x1: KX(kk), x2: KX(kk), y1: CY0, y2: CY0 + 4, stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: KX(kk), y: CY0 + 16, 'text-anchor': 'middle', text: String(kk) }, chartG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'k' }, chartG);
  const kMarker = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, chartG);
  const bestLabel = el('text', { class: 'stage-text', y: CY1 - 4, 'text-anchor': 'middle', text: `best k = ${best}` }, chartG);
  bestLabel.style.fill = STRONG;
  const line = (key, attrs) => el('path', {
    d: rows.map((r, i) => `${i ? 'L' : 'M'}${KX(r.k).toFixed(2)},${AY(r[key]).toFixed(2)}`).join(''),
    fill: 'none', 'stroke-width': 2, ...attrs,
  }, chartG);
  line('train', { stroke: INK });
  line('test', { stroke: CYAN, 'stroke-dasharray': '5 3' });
  rows.forEach((r) => {
    el('circle', { cx: KX(r.k), cy: AY(r.train), r: 3, fill: INK }, chartG);
    el('rect', { x: KX(r.k) - 3, y: AY(r.test) - 3, width: 6, height: 6, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, chartG);
  });
  const ckey = el('g', {}, chartG);
  el('line', { x1: CX0 + 10, x2: CX0 + 26, y1: CY1 - 22, y2: CY1 - 22, stroke: INK, 'stroke-width': 2 }, ckey);
  el('text', { class: 'stage-text', x: CX0 + 30, y: CY1 - 18, text: 'train' }, ckey).style.fill = INK;
  el('line', { x1: CX0 + 84, x2: CX0 + 100, y1: CY1 - 22, y2: CY1 - 22, stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, ckey);
  el('text', { class: 'stage-text', x: CX0 + 104, y: CY1 - 18, text: 'test' }, ckey).style.fill = CYAN;

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const kGroup = make('div');
  const kLabel = make('label');
  const kInput = make('input', { type: 'range', min: 1, max: 25, step: 2, value: k });
  const kOut = make('output');
  kOut.setAttribute('aria-hidden', 'true');
  kLabel.append('k', make('span', { className: 'vh', textContent: ', the number of neighbours' }), ' ', kInput, ' ', kOut);
  kGroup.append(kLabel);

  const modeGroup = make('div');
  const modeBtns = MODES.map((m) => make('button', { type: 'button', textContent: { raw: 'Original', x100: 'Feature 2 × 100', std: 'Standardised' }[m] }));
  modeGroup.append(...modeBtns);
  controls.append(kGroup, modeGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { region: 0, query: 0, chart: 0 };
  let s = { ...LAYERS };
  const target = (i) => ({
    region: i >= 2 ? 1 : 0,
    query: i === 1 || i === 5 ? 1 : 0,
    chart: i === 3 || i === 4 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const curK = () => (step === 1 ? K_VOTE : step === 2 ? 1 : step === 3 ? k : best);
  const curMode = () => (step === 5 ? mode : 'raw');

  // The region is redrawn only when its k or mode changes, at most once a frame.
  let regionKey = '';
  let pending = false;
  function drawRegion() {
    const kk = curK();
    const m = curMode();
    const key = `${kk}:${m}`;
    if (key === regionKey) return;
    regionKey = key;
    const t = transform(m);
    region.update((x, y) => knnPredict(t.train, TR.y, t.apply([x, y]), kk).cls);
  }
  function queueRegion() {
    if (pending) return;
    pending = true;
    const run = () => { pending = false; drawRegion(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  }

  function render() {
    queueRegion();
    op(regionG, s.region);
    const big = curMode() === 'x100';
    op(axesRaw, big ? 0 : 1);
    op(axesBig, big ? 1 : 0);

    const kk = curK();
    const n = neighbours(q, kk, curMode());
    const qx = X(q[0]);
    const qy = Y(q[1]);
    hood.setAttribute('cx', qx.toFixed(2));
    hood.setAttribute('cy', qy.toFixed(2));
    // Capped so a very flat ellipse stays a finite shape; the clip trims it.
    hood.setAttribute('rx', Math.min(4000, n.radii[0] * (X(1) - X(0))).toFixed(2));
    hood.setAttribute('ry', Math.min(4000, n.radii[1] * (Y(0) - Y(1))).toFixed(2));
    linksG.replaceChildren();
    ringsG.replaceChildren();
    for (const i of n.idx) {
      const [a, b] = TR.X[i];
      el('line', { x1: qx, y1: qy, x2: X(a), y2: Y(b), stroke: MUTED, 'stroke-width': 1 }, linksG);
      el('circle', { cx: X(a), cy: Y(b), r: 9, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, ringsG);
    }
    diamond.setAttribute('d', diamondD(qx, qy, 8));
    handle.setAttribute('cx', qx.toFixed(2));
    handle.setAttribute('cy', qy.toFixed(2));
    focusRing.setAttribute('cx', qx.toFixed(2));
    focusRing.setAttribute('cy', qy.toFixed(2));
    focusRing.style.display = focused ? '' : 'none';
    op(hoodG, s.query);
    op(ringsG, s.query);
    op(queryG, s.query);
    op(keyNew, s.query);
    handle.setAttribute('aria-label', `New case at feature 1 = ${q[0].toFixed(1)}, feature 2 = ${q[1].toFixed(1)}. Arrow keys move it.`);

    op(chartG, s.chart);
    kMarker.setAttribute('x1', KX(kk).toFixed(2));
    kMarker.setAttribute('x2', KX(kk).toFixed(2));
    bestLabel.setAttribute('x', KX(best).toFixed(2));
    bestLabel.style.display = step === 4 ? '' : 'none';

    const texts = readout(step, { k, mode, q });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let w = 200;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0 + 4, 598 - w);
    boxRect.setAttribute('x', bx);
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', w.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', bx + 12);
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    kOut.textContent = String(k);
    kInput.setAttribute('aria-valuetext', `k = ${k}`);
    modeBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(MODES[i] === mode)));
  }

  function animateTo(to, ms) {
    const my = ++token;
    const start = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * kt;
      s = next;
      render();
    }, { reduced });
  }

  const shape = (c) => (c === 1 ? 'triangles' : 'circles');
  const spokenQuery = (kk, m) => {
    const n = neighbours(q, kk, m);
    return `New case at feature 1 = ${q[0].toFixed(1)}, feature 2 = ${q[1].toFixed(1)}. Of its ${kk} nearest neighbours, ${kk - n.votes1} are circles, class 0, and ${n.votes1} are triangles, class 1. Predicted class ${n.cls}, the ${shape(n.cls) === 'circles' ? 'circle' : 'triangle'}.`;
  };
  const spoken = (i) => {
    if (i === 0) return 'Forty training points: twenty circles, class 0, and twenty triangles, class 1.';
    if (i === 1) return spokenQuery(K_VOTE, 'raw');
    const r = rows.find((row) => row.k === curK());
    if (i <= 3) return `k = ${r.k}. Training accuracy ${pct(r.train)}, test accuracy ${pct(r.test)}.`;
    if (i === 4) return `Best k on the test points: ${best}, with test accuracy ${pct(r.test)}.`;
    const u = unitsTable(best);
    const acc = { raw: u.raw, x100: u.x100, std: u.std }[mode];
    return `Distances in ${MODE_NAME[mode]}. k = ${best}, test accuracy ${pct(acc)}. ${spokenQuery(best, mode)}`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  const setQuery = (a, b) => {
    q = [clamp(a, X_DOM[0], X_DOM[1]), clamp(b, Y_DOM[0], Y_DOM[1])];
    render();
    say(spoken(step));
  };
  draggable(handle, {
    bounds: { x0: PX0, x1: PX1, y0: PY0, y1: PY1 },
    onDrag: (px, py) => setQuery(X.invert(px), Y.invert(py)),
  });
  handle.addEventListener('focus', () => {
    let kb = true;
    try { kb = handle.matches(':focus-visible'); } catch { /* keep the ring */ }
    focused = kb;
    render();
  });
  handle.addEventListener('blur', () => { focused = false; render(); });

  kInput.addEventListener('input', () => {
    const v = Math.round(Number(kInput.value));
    k = clamp(v % 2 ? v : v + 1, 1, 25);
    render();
    say(spoken(3));
  });
  modeBtns.forEach((b, i) => b.addEventListener('click', () => {
    mode = MODES[i];
    render();
    announce(spoken(5));
  }));

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) p.hidden = false;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      showGroup(kGroup, i === 3);
      showGroup(modeGroup, i === 5);
      const live = i === 1 || i === 5;
      handle.setAttribute('tabindex', live ? 0 : -1);
      if (!live && doc.activeElement === handle) handle.blur();
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      return animateTo(to, 500);
    },
  };
}

function diamondD(cx, cy, r) {
  const P = (v) => Number(v.toFixed(2));
  return `M${P(cx)},${P(cy - r)}L${P(cx + r)},${P(cy)}L${P(cx)},${P(cy + r)}L${P(cx - r)},${P(cy)}Z`;
}
