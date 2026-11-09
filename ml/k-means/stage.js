// L3.8, "k-means clustering": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-k-means.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, randn } from '../assets/maths.js';
import { kmeans } from '../assets/unsupervised.js';
import { linearScale, clamp, el, axes, tween, classMarker, regionMap } from '../assets/plot.js';

export const SEED = 101;
export const BLOBS = [[2, 2], [6, 2.5], [4, 6]];
export const SD = 0.8;
export const N_PER = 20;
// Start A and start B: three data points each, by index. Start B is the first
// triple in index order whose final inertia is more than 20% above the best.
export const START_A = [0, 1, 2];
export const START_B = [0, 1, 3];
export const N_STARTS = 10;
export const KS = [1, 2, 3, 4, 5, 6, 7, 8];
export const X_DOM = [0, 7.5];
export const Y_DOM = [0, 7];
// Region map cells: square, 4.5 user units a side.
export const GRID = [60, 56];

// Three round blobs of 20, blob by blob; each point draws x, then y.
let DATA = null;
export function data() {
  if (!DATA) {
    const rng = mulberry32(SEED);
    DATA = [];
    for (const [cx, cy] of BLOBS) {
      for (let i = 0; i < N_PER; i++) {
        const x = cx + SD * randn(rng);
        const y = cy + SD * randn(rng);
        DATA.push([x, y]);
      }
    }
  }
  return DATA.map((p) => p.slice());
}
const X = data();

// Cluster k is drawn as marker class k: circle, triangle, square.
export const clusterClass = (k) => k;

const runs = {};
export function run(which) {
  if (!runs[which]) {
    const idx = which === 'B' ? START_B : START_A;
    runs[which] = kmeans(X, idx.map((i) => X[i]));
  }
  return runs[which];
}

// K distinct data points, drawn by index from mulberry32(200 + K); a repeat is drawn again.
const startCache = new Map();
export function randomStarts(K) {
  if (!startCache.has(K)) {
    const rng = mulberry32(200 + K);
    const all = [];
    for (let s = 0; s < N_STARTS; s++) {
      const idx = [];
      while (idx.length < K) {
        const i = Math.floor(rng() * X.length);
        if (!idx.includes(i)) idx.push(i);
      }
      all.push(idx);
    }
    startCache.set(K, all);
  }
  return startCache.get(K).map((s) => s.slice());
}

// For each K, run all ten starts and keep the lowest inertia (the first, on a tie).
let ELBOW = null;
export function elbow() {
  if (!ELBOW) {
    ELBOW = KS.map((K) => {
      const results = randomStarts(K).map((idx) => kmeans(X, idx.map((i) => X[i])));
      const inertias = results.map((r) => r.inertia);
      const best = Math.min(...inertias);
      return { K, inertias, best, bestRun: results[inertias.indexOf(best)] };
    });
  }
  return ELBOW;
}

const f1 = (v) => v.toFixed(1);
const sizes = (labels, K = 3) => Array.from({ length: K }, (_, k) => labels.filter((l) => l === k).length);
const sameAs = (v, b) => Math.abs(v - b) < 1e-9;

export function pageValues() {
  const A = run('A');
  const B = run('B');
  const rows = elbow();
  const k3 = rows[2];
  const s1 = sizes(A.states[1].labels);
  const others = k3.inertias.filter((v) => !sameAs(v, k3.best));
  const pv = {
    'n': String(X.length),
    'a-first': f1(A.states[1].inertia),
    'a-sizes': `${s1[0]}, ${s1[1]} and ${s1[2]}`,
    'a-final': f1(A.inertia),
    'a-rounds': String(A.iters),
    'b-final': f1(B.inertia),
    'ab-ratio': (B.inertia / A.inertia).toFixed(1),
    'starts-hit': String(k3.inertias.length - others.length),
    'starts-other': others.map(f1).join(', '),
    'drop23': f1(rows[1].best - rows[2].best),
    'drop34': f1(rows[2].best - rows[3].best),
  };
  rows.forEach((r) => { pv[`e${r.K}`] = f1(r.best); });
  return pv;
}

// One run's state as the stage shows it: index s into run('A').states.
const phaseWord = (st) => (st.phase === 'assign' ? 'assign' : 'update');
export function readout(step, { state = 2, start = 'A', K = 3 } = {}) {
  const A = run('A');
  if (step === 0) return [`${X.length} points`, 'no labels'];
  if (step === 1) return ['K = 3 centres', 'inertia: not yet'];
  if (step === 2) {
    const st = A.states[1];
    return ['round 1 · assign', `inertia: ${f1(st.inertia)}`, `sizes: ${sizes(st.labels).join(', ')}`];
  }
  if (step === 3) {
    const s = clamp(state, 1, A.states.length - 1);
    const st = A.states[s];
    const last = s === A.states.length - 1;
    return [`round ${Math.ceil(s / 2)} · ${phaseWord(st)}`, `inertia: ${f1(st.inertia)}`, last ? 'no label changed: done' : `next: ${st.phase === 'assign' ? 'update' : 'assign'}`];
  }
  if (step === 4) {
    const r = run(start);
    return [`start ${start}`, `inertia: ${f1(r.inertia)}`, `best of 10: ${f1(elbow()[2].best)}`];
  }
  const row = elbow()[clamp(K, 1, 8) - 1];
  return [`K = ${row.K}`, `best of 10: ${f1(row.best)}`];
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The scatter: 36 user units per feature unit on both axes, so distances look true.
const PX0 = 56;
const PX1 = 326;
const PY0 = 286;
const PY1 = 34;
const RX0 = 352;
const BOX_Y = 8;
const ROW = 20;
// The chart on the right.
const CX0 = 404;
const CX1 = 586;
const CY0 = 286;
const CY1 = 150;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const A = run('A');
  const rows = elbow();
  const k3 = rows[2];
  const LAST = A.states.length - 1;
  let step = 0;
  let state = 2;
  let start = 'A';
  let K = 3;
  let token = 0;
  // Centres as drawn; they glide on an update step unless motion is reduced.
  let shown = A.states[0].centres.map((c) => c.slice());

  const PX = linearScale(X_DOM, [PX0, PX1]);
  const PY = linearScale(Y_DOM, [PY0, PY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Sixty points on two features with no labels. k-means places K centres, gives each point to its nearest centre, shown by marker shape (cluster 1 circle, cluster 2 triangle, cluster 3 square), and shades the plot by nearest centre with a line along each border. A chart on the right shows the inertia.');

  const regionG = el('g', {}, svg);
  const region = regionMap(regionG, PX, PY, () => 0, { nx: GRID[0], ny: GRID[1] });
  axes(svg, PX, PY, { xLabel: 'feature 1', yLabel: 'feature 2' });
  const trailG = el('g', {}, svg);
  const pointsG = el('g', {}, svg);
  const startG = el('g', {}, svg);
  const centreG = el('g', {}, svg);

  // Key, above the scatter.
  const keyG = el('g', {}, svg);
  const keyU = el('g', {}, keyG);
  classMarker(keyU, PX0 + 6, 16, null, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 16, y: 20, text: 'point, no label' }, keyU);
  const keyC = el('g', {}, keyG);
  el('text', { class: 'stage-text', x: PX0, y: 20, text: 'cluster' }, keyC);
  [0, 1, 2].forEach((k) => {
    classMarker(keyC, PX0 + 78 + 42 * k, 16, k, { r: 5 });
    el('text', { class: 'stage-text', x: PX0 + 88 + 42 * k, y: 20, text: String(k + 1) }, keyC);
  });
  const keyBig = el('g', {}, keyG);
  classMarker(keyBig, PX0 + 186, 16, 0, { r: 7 }).classList.add('km-centre');
  el('text', { class: 'stage-text', x: PX0 + 200, y: 20, text: 'centre' }, keyBig);
  const keyCentre = el('g', {}, keyG);
  plus(keyCentre, PX0 + 186, 16, 6);
  el('text', { class: 'stage-text', x: PX0 + 198, y: 20, text: 'centre' }, keyCentre);

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2].map(() => el('text', { class: 'value' }, box));

  // Chart 1: inertia after each step of start A.
  const traceG = el('g', {}, svg);
  const TX = linearScale([1, LAST], [CX0 + 10, CX1 - 6]);
  const TY = linearScale([0, 500], [CY0, CY1]);
  frame(traceG, TX, TY, [0, 250, 500], A.states.slice(1).map((_, i) => i + 1).filter((s) => s % 2), 'step', CX0);
  const traceLine = el('path', { fill: 'none', stroke: INK, 'stroke-width': 1.6 }, traceG);
  const traceDots = el('g', {}, traceG);
  const tkey = el('g', {}, traceG);
  el('circle', { cx: CX0 + 16, cy: CY1 - 22, r: 3.5, fill: INK }, tkey);
  el('text', { class: 'stage-text', x: CX0 + 24, y: CY1 - 18, text: 'assign' }, tkey);
  el('rect', { x: CX0 + 96, y: CY1 - 25.5, width: 7, height: 7, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, tkey);
  el('text', { class: 'stage-text', x: CX0 + 108, y: CY1 - 18, text: 'update' }, tkey);

  // Chart 2: the final inertia of the ten random starts for K = 3, stacked.
  const stripG = el('g', {}, svg);
  const SX = linearScale([0, 300], [CX0 + 6, CX1 - 6]);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, stripG);
  for (const v of [0, 100, 200, 300]) {
    el('line', { x1: SX(v), x2: SX(v), y1: CY0, y2: CY0 + 4, stroke: LINE }, stripG);
    el('text', { class: 'stage-text', x: SX(v), y: CY0 + 16, 'text-anchor': 'middle', text: String(v) }, stripG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'final inertia' }, stripG);
  el('text', { class: 'stage-text', x: 598, y: CY1 - 22, 'text-anchor': 'end', text: '10 random starts, K = 3' }, stripG);
  const stacks = new Map();
  k3.inertias.forEach((v) => {
    const key = f1(v);
    const h = stacks.get(key) || 0;
    stacks.set(key, h + 1);
    el('circle', { cx: SX(v), cy: CY0 - 8 - 9 * h, r: 3.5, fill: CYAN, stroke: BG }, stripG);
  });
  const stripMark = el('line', { y1: CY1 - 6, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, stripG);
  const stripLabel = el('text', { class: 'stage-text', y: CY1 - 2, 'text-anchor': 'start' }, stripG);
  stripLabel.style.fill = STRONG;

  // Chart 3: best inertia for each K.
  const elbowG = el('g', {}, svg);
  const EX = linearScale([1, 8], [CX0 + 10, CX1 - 6]);
  const EY = linearScale([0, 450], [CY0, CY1]);
  frame(elbowG, EX, EY, [0, 200, 400], KS, 'K', CX0);
  el('path', { d: rows.map((r, i) => `${i ? 'L' : 'M'}${EX(r.K).toFixed(2)},${EY(r.best).toFixed(2)}`).join(''), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, elbowG);
  rows.forEach((r) => el('circle', { cx: EX(r.K), cy: EY(r.best), r: 3.5, fill: CYAN, stroke: BG }, elbowG));
  el('text', { class: 'stage-text', x: 598, y: CY1 - 22, 'text-anchor': 'end', text: 'inertia, best of 10 starts' }, elbowG);
  const elbowMark = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, elbowG);

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const stepGroup = make('div');
  const nextBtn = make('button', { type: 'button' });
  const resetBtn = make('button', { type: 'button', textContent: 'Start again' });
  stepGroup.append(nextBtn, resetBtn);
  const startGroup = make('div');
  const startBtns = ['A', 'B'].map((w) => make('button', { type: 'button', textContent: `Start ${w}` }));
  startGroup.append(...startBtns);
  const kGroup = make('div');
  const kLabel = make('label');
  const kInput = make('input', { type: 'range', min: 1, max: 8, step: 1, value: K });
  const kOut = make('output');
  kOut.setAttribute('aria-hidden', 'true');
  kLabel.append('K', make('span', { className: 'vh', textContent: ', the number of clusters' }), ' ', kInput, ' ', kOut);
  kGroup.append(kLabel);
  controls.append(stepGroup, startGroup, kGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // What the scatter shows at the current step.
  function scene() {
    if (step === 0) return { labels: null, centres: [], kind: 'none' };
    if (step === 1) return { labels: null, centres: A.states[0].centres, kind: 'shape' };
    if (step === 2) return { labels: A.states[1].labels, centres: A.states[1].centres, kind: 'shape', region: true };
    if (step === 3) return { labels: A.states[state].labels, centres: A.states[state].centres, kind: 'shape', region: true, trail: state };
    if (step === 4) {
      const r = run(start);
      return { labels: r.labels, centres: r.centres, kind: 'shape', region: true, trail: r.states.length - 1, runOf: r, starts: start === 'B' ? START_B : START_A };
    }
    const r = rows[K - 1].bestRun;
    return { labels: null, centres: r.centres, kind: 'plus', region: true, plain: true };
  }

  // Region: nearest of the drawn centres, at most once a frame.
  let regionKey = '';
  let pending = false;
  function drawRegion() {
    const sc = scene();
    const cs = sc.kind === 'shape' ? shown : sc.centres;
    const key = `${sc.plain ? 'p' : 'c'}:${cs.map((c) => c.map((v) => v.toFixed(4)).join(',')).join(';')}`;
    if (key === regionKey) return;
    regionKey = key;
    region.update((x, y) => {
      let best = 0;
      let bd = Infinity;
      cs.forEach((c, k) => {
        const d = (x - c[0]) ** 2 + (y - c[1]) ** 2;
        if (d < bd) { bd = d; best = k; }
      });
      return best;
    });
  }
  function queueRegion() {
    if (pending) return;
    pending = true;
    const go = () => { pending = false; drawRegion(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(go);
    else go();
  }

  let pointsKey = '';
  function render() {
    const sc = scene();
    regionG.style.display = sc.region ? '' : 'none';
    regionG.classList.toggle('km-plain', !!sc.plain);
    if (sc.region) queueRegion();

    const lk = sc.labels ? sc.labels.join('') : 'u';
    if (lk !== pointsKey) {
      pointsKey = lk;
      pointsG.replaceChildren();
      X.forEach((p, i) => classMarker(pointsG, PX(p[0]), PY(p[1]), sc.labels ? clusterClass(sc.labels[i]) : null, { r: 4.5 }));
    }

    // Trails: where each centre has been, as a dashed path.
    trailG.replaceChildren();
    if (sc.trail !== undefined) {
      const states = (sc.runOf || A).states.slice(0, sc.trail + 1);
      for (let k = 0; k < 3; k++) {
        const pts = states.map((s) => s.centres[k]);
        if (sc.kind === 'shape' && sc.runOf === undefined) pts[pts.length - 1] = shown[k];
        el('path', { d: pts.map((c, i) => `${i ? 'L' : 'M'}${PX(c[0]).toFixed(2)},${PY(c[1]).toFixed(2)}`).join(''), fill: 'none', stroke: STRONG, 'stroke-width': 1.3, 'stroke-dasharray': '4 3', 'stroke-opacity': 0.8 }, trailG);
      }
    }
    // Step 5: ring the three starting points.
    startG.replaceChildren();
    if (sc.starts) {
      for (const i of sc.starts) el('circle', { cx: PX(X[i][0]), cy: PY(X[i][1]), r: 9, fill: 'none', stroke: STRONG, 'stroke-width': 1.5, 'stroke-dasharray': '2 2' }, startG);
    }

    centreG.replaceChildren();
    if (sc.kind === 'shape') {
      shown.forEach((c, k) => {
        const m = classMarker(centreG, PX(c[0]), PY(c[1]), clusterClass(k), { r: 9 });
        m.classList.add('km-centre');
      });
    } else if (sc.kind === 'plus') {
      sc.centres.forEach((c) => plus(centreG, PX(c[0]), PY(c[1]), 8));
    }

    keyU.style.display = sc.labels ? 'none' : '';
    keyC.style.display = sc.labels ? '' : 'none';
    keyCentre.style.display = sc.kind === 'plus' ? '' : 'none';
    keyBig.style.display = step === 1 ? '' : 'none';

    // Charts.
    traceG.style.display = step === 2 || step === 3 ? '' : 'none';
    const upto = step === 2 ? 1 : state;
    const tr = A.states.slice(1, upto + 1);
    traceLine.setAttribute('d', tr.map((s, i) => `${i ? 'L' : 'M'}${TX(i + 1).toFixed(2)},${TY(s.inertia).toFixed(2)}`).join(''));
    traceDots.replaceChildren();
    tr.forEach((s, i) => {
      if (s.phase === 'assign') el('circle', { cx: TX(i + 1), cy: TY(s.inertia), r: 3.5, fill: INK }, traceDots);
      else el('rect', { x: TX(i + 1) - 3.5, y: TY(s.inertia) - 3.5, width: 7, height: 7, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, traceDots);
    });
    stripG.style.display = step === 4 ? '' : 'none';
    const cur = run(start).inertia;
    stripMark.setAttribute('x1', SX(Math.min(cur, 300)).toFixed(2));
    stripMark.setAttribute('x2', SX(Math.min(cur, 300)).toFixed(2));
    stripLabel.textContent = `start ${start}`;
    stripLabel.setAttribute('x', (SX(Math.min(cur, 300)) + (cur > 200 ? -52 : 4)).toFixed(2));
    elbowG.style.display = step === 5 ? '' : 'none';
    elbowMark.setAttribute('x1', EX(K).toFixed(2));
    elbowMark.setAttribute('x2', EX(K).toFixed(2));

    const texts = readout(step, { state, start, K });
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

    const nx = A.states[Math.min(state + 1, LAST)];
    nextBtn.textContent = state >= LAST ? 'Done' : nx.phase === 'assign' ? 'Next: assign' : 'Next: update';
    nextBtn.disabled = state >= LAST;
    resetBtn.disabled = state === 2;
    startBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(['A', 'B'][i] === start)));
    kOut.textContent = String(K);
    kInput.setAttribute('aria-valuetext', `K = ${K}`);
  }

  // Move the drawn centres to `to`, gliding unless motion is reduced.
  function glide(to) {
    const my = ++token;
    const from = shown.map((c) => c.slice());
    return tween(400, (t) => {
      if (my !== token) return;
      shown = from.map((c, k) => [c[0] + (to[k][0] - c[0]) * t, c[1] + (to[k][1] - c[1]) * t]);
      render();
    }, { reduced });
  }
  const jump = (to) => {
    token++;
    shown = to.map((c) => c.slice());
    render();
  };

  const shapeOf = ['circles', 'triangles', 'squares'];
  const spokenSizes = (labels) => sizes(labels).map((n, k) => `cluster ${k + 1}, ${shapeOf[k]}: ${n}`).join('; ');
  const spoken = (i) => {
    if (i === 0) return `${X.length} points on two features, no labels.`;
    if (i === 1) return 'Three starting centres, all three in the lower-left group of points.';
    if (i === 2) return `Assign step: each point goes to its nearest centre. ${spokenSizes(A.states[1].labels)}. Inertia ${f1(A.states[1].inertia)}.`;
    if (i === 3) {
      const st = A.states[state];
      const what = st.phase === 'assign' ? 'Assign step: each point goes to its nearest centre.' : 'Update step: each centre moves to the mean of its points.';
      const end = state >= LAST ? ' No label changed, so k-means has stopped.' : '';
      return `Round ${Math.ceil(state / 2)}. ${what} Inertia ${f1(st.inertia)}. ${spokenSizes(st.labels)}.${end}`;
    }
    if (i === 4) {
      const r = run(start);
      return `Start ${start}: the run stops with inertia ${f1(r.inertia)}. ${spokenSizes(r.labels)}. The best of 10 random starts reaches ${f1(k3.best)}.`;
    }
    return `K = ${K}. Best inertia of 10 starts: ${f1(rows[K - 1].best)}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 400);
    };
  })();

  nextBtn.addEventListener('click', () => {
    if (state >= LAST) return;
    state++;
    announce(spoken(3));
    glide(A.states[state].centres);
  });
  resetBtn.addEventListener('click', () => {
    state = 2;
    jump(A.states[state].centres);
    announce(spoken(3));
  });
  startBtns.forEach((b, i) => b.addEventListener('click', () => {
    start = ['A', 'B'][i];
    jump(run(start).centres);
    announce(spoken(4));
  }));
  kInput.addEventListener('input', () => {
    K = clamp(Math.round(Number(kInput.value)), 1, 8);
    render();
    say(spoken(5));
  });

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
    show(i) {
      step = i;
      showGroup(stepGroup, i === 3);
      showGroup(startGroup, i === 4);
      showGroup(kGroup, i === 5);
      announce(spoken(i));
      const sc = scene();
      jump(sc.kind === 'shape' ? sc.centres : []);
      return Promise.resolve();
    },
  };
}

// Axes for a small chart: y ticks on the left, x ticks below, x label under them.
function frame(g, xs, ys, yTicks, xTicks, xLabel, x0) {
  const [y0, y1] = ys.range;
  const [, x1] = xs.range;
  el('line', { x1: x0, x2: x0, y1: y0, y2: y1, stroke: LINE }, g);
  el('line', { x1: x0, x2: x1 + 6, y1: y0, y2: y0, stroke: LINE }, g);
  for (const v of yTicks) {
    el('line', { x1: x0 - 4, x2: x0, y1: ys(v), y2: ys(v), stroke: LINE }, g);
    el('text', { class: 'stage-text', x: x0 - 7, y: ys(v) + 4, 'text-anchor': 'end', text: String(v) }, g);
  }
  for (const v of xTicks) {
    el('line', { x1: xs(v), x2: xs(v), y1: y0, y2: y0 + 4, stroke: LINE }, g);
    el('text', { class: 'stage-text', x: xs(v), y: y0 + 16, 'text-anchor': 'middle', text: String(v) }, g);
  }
  el('text', { class: 'stage-text', x: (x0 + x1) / 2, y: y0 + 34, 'text-anchor': 'middle', text: xLabel }, g);
}

// A centre for any K: a plus sign with a dark halo.
function plus(parent, cx, cy, r) {
  const d = `M${cx - r},${cy}H${cx + r}M${cx},${cy - r}V${cy + r}`;
  el('path', { d, stroke: BG, 'stroke-width': 5, 'stroke-linecap': 'round' }, parent);
  el('path', { d, stroke: STRONG, 'stroke-width': 2.2, 'stroke-linecap': 'round' }, parent);
}
