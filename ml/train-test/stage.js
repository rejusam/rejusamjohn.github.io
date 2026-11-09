// Lesson 9, "Train, validate, test": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, mse, mean, polyFit, polyEval, trainTestSplit } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, tween } from '../assets/plot.js';
import { sineData, truth } from '../assets/sine-data.js';

export const SEED = 3;
export const N = 40;
export const TEST_FRACTION = 0.3;
// The first split shuffles with mulberry32(SPLIT_SEED); "New split" adds 1.
export const SPLIT_SEED = 3;
export const DEGREE = 3;
export const K = 5;
export const CANDIDATES = [1, 2, 3, 4, 5, 6];
// Step 4 shows this many splits at once, starting from the first.
export const STRIP_N = 20;
// "About the data" quotes a tally over this many splits.
export const MANY = 200;

export const lessonData = () => sineData({ n: N, seed: SEED });

export const splitFor = (seed) => trainTestSplit(N, TEST_FRACTION, mulberry32(seed));

const pick = (arr, idx) => idx.map((i) => arr[i]);

export function fitOn(d, idx, degree) {
  const r = polyFit(pick(d.xs, idx), pick(d.ys, idx), degree);
  if (!r.ok) throw new Error('polynomial fit failed');
  return r.coef;
}

export const mseOf = (coef, d, idx) => mse(pick(d.ys, idx), pick(d.xs, idx).map((x) => polyEval(coef, x)));

// Fit on the training points; score on both sets.
export function score(d, sp, degree = DEGREE) {
  const coef = fitOn(d, sp.train, degree);
  return { coef, train: mseOf(coef, d, sp.train), test: mseOf(coef, d, sp.test) };
}

// "Join the dots": straight lines between neighbouring points, flat past the ends.
export function joinDots(xs, ys) {
  const o = xs.map((x, i) => [x, ys[i]]).sort((p, q) => p[0] - q[0]);
  return (x) => {
    if (x <= o[0][0]) return o[0][1];
    if (x >= o[o.length - 1][0]) return o[o.length - 1][1];
    let k = 1;
    while (o[k][0] < x) k++;
    const [x0, y0] = o[k - 1];
    const [x1, y1] = o[k];
    if (x === x1) return y1;
    return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
  };
}

export function stepOneScores(d) {
  const f = joinDots(d.xs, d.ys);
  return { dots: mse(d.ys, d.xs.map(f)), truth: mse(d.ys, d.xs.map(truth)) };
}

// Over `count` splits from the first: how often training MSE < test MSE.
export function splitStats(d, count, start = SPLIT_SEED) {
  let below = 0;
  let st = 0;
  let se = 0;
  const rows = [];
  for (let seed = start; seed < start + count; seed++) {
    const s = score(d, splitFor(seed));
    rows.push({ seed, train: s.train, test: s.test });
    if (s.train < s.test) below++;
    st += s.train;
    se += s.test;
  }
  return { count, below, meanTrain: st / count, meanTest: se / count, rows };
}

// k folds: consecutive runs of the (already shuffled) training indices.
export function folds(trainIdx, k = K) {
  const n = trainIdx.length;
  const out = [];
  for (let j = 0; j < k; j++) out.push(trainIdx.slice(Math.round((j * n) / k), Math.round(((j + 1) * n) / k)));
  return out;
}

// k-fold cross-validation over the candidate degrees, on the training points
// only; then the chosen degree is refitted on all of them and scored on the
// test points once.
export function cvTable(d, sp, k = K) {
  const fs = folds(sp.train, k);
  const scores = CANDIDATES.map((degree) => {
    const each = fs.map((v) => {
      const tr = sp.train.filter((i) => !v.includes(i));
      return mseOf(fitOn(d, tr, degree), d, v);
    });
    return { degree, cv: mean(each) };
  });
  let best = scores[0];
  for (const r of scores) if (r.cv < best.cv) best = r;
  const final = score(d, sp, best.degree);
  return { scores, best: best.degree, train: final.train, test: final.test, folds: fs };
}

// Fixed decimals with a true minus sign.
export function fmt(v, dp = 3) {
  const s = Math.abs(v).toFixed(dp);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
}

export const splitNo = (seed) => seed - SPLIT_SEED + 1;

// The read-out rows for each step. state: { seed }.
export function readout(step, { seed = SPLIT_SEED } = {}) {
  const d = lessonData();
  if (step === 0) {
    const t = stepOneScores(d);
    return [`${N} points`, `join the dots: MSE ${fmt(t.dots)}`, `true curve: MSE ${fmt(t.truth)}`];
  }
  const sp = splitFor(seed);
  if (step === 1) return [`split ${splitNo(seed)}`, `training: ${sp.train.length} points`, `test: ${sp.test.length} points`];
  const s = score(d, sp);
  if (step === 2) return [`degree-${DEGREE} curve`, `train MSE: ${fmt(s.train)}`, `test MSE: ${fmt(s.test)}`];
  if (step === 3) return [`split ${splitNo(seed)}`, `train MSE: ${fmt(s.train)}`, `test MSE: ${fmt(s.test)}`];
  const cv = cvTable(d, sp);
  const low = cv.scores.find((r) => r.degree === cv.best).cv;
  return [`${K}-fold, ${sp.train.length} points`, `lowest: degree ${cv.best}, ${fmt(low)}`, `test MSE: ${fmt(cv.test)}`];
}

// Values quoted in the page text. Those that depend on the split are
// refreshed whenever the split changes.
export function pageValues(seed = SPLIT_SEED) {
  const d = lessonData();
  const sp = splitFor(seed);
  const s = score(d, sp);
  const t = stepOneScores(d);
  const strip = splitStats(d, STRIP_N);
  const many = splitStats(d, MANY);
  const cv = cvTable(d, sp);
  const tests = strip.rows.map((r) => r.test);
  return {
    'truth-mse': fmt(t.truth),
    'split-no': String(splitNo(seed)),
    'train-mse': fmt(s.train),
    'test-mse': fmt(s.test),
    'strip-n': String(STRIP_N),
    'strip-below': String(strip.below),
    'strip-lo': fmt(Math.min(...tests)),
    'strip-hi': fmt(Math.max(...tests)),
    'many-n': String(MANY),
    'many-below': String(many.below),
    'many-train': fmt(many.meanTrain),
    'many-test': fmt(many.meanTest),
    'cv-best': String(cv.best),
    'cv-test': fmt(cv.test),
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

const PX0 = 56;
const PX1 = 352;
const PY0 = 290;
const PY1 = 20;
const Y_DOM = [-1.5, 1.5];
const RX0 = 384;
const RX1 = 590;
const BOX_Y = 8;
const ROW = 20;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const d = lessonData();
  let seed = SPLIT_SEED;
  let sp = splitFor(seed);
  let sc = score(d, sp);
  let cv = cvTable(d, sp);
  const strip = splitStats(d, STRIP_N);
  const seen = new Map(strip.rows.map((r) => [r.seed, r]));
  const dots = joinDots(d.xs, d.ys);

  const X = linearScale([-1, 1], [PX0, PX1]);
  const Y = linearScale(Y_DOM, [PY0, PY1]);
  const GRID = Array.from({ length: 241 }, (_, i) => -1 + i / 120);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Forty points around a wave, split into training and test points, with a fitted curve and its scores. A read-out on the right changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l9-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  // --- The scatter (steps 1 to 4).
  const plotG = el('g', {}, svg);
  axes(plotG, X, Y, { xLabel: 'x', yLabel: 'y' });
  const curvesG = el('g', { 'clip-path': 'url(#l9-plot)' }, plotG);
  const truthPath = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 1.6, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round', d: pathFrom(GRID.map((x) => [x, truth(x)]), X, Y) }, curvesG);
  const order = d.xs.map((x, i) => [x, d.ys[i]]).sort((p, q) => p[0] - q[0]);
  const dotsPath = el('path', { fill: 'none', stroke: MUTED, 'stroke-width': 1.4, 'stroke-linejoin': 'round', d: pathFrom(order.map(([x]) => [x, dots(x)]), X, Y) }, curvesG);
  const fitPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, curvesG);
  const sticksG = el('g', {}, plotG);
  const sticks = d.xs.map(() => el('line', { stroke: VIOLET, 'stroke-width': 1.3, 'stroke-dasharray': '3 2' }, sticksG));
  const ptsG = el('g', {}, plotG);
  const circles = d.xs.map((x, i) => el('circle', { cx: X(x), cy: Y(d.ys[i]), r: 4.5, fill: CYAN, stroke: BG, 'stroke-width': 1.2 }, ptsG));
  const squares = d.xs.map((x, i) => el('rect', { x: X(x) - 4.5, y: Y(d.ys[i]) - 4.5, width: 9, height: 9, fill: BG, stroke: VIOLET, 'stroke-width': 2 }, ptsG));

  // Key, top-left of the plot.
  const keyG = el('g', {}, plotG);
  el('rect', { x: PX0 + 6, y: PY1 + 2, width: 128, height: 44, rx: 4, fill: SURFACE, 'fill-opacity': 0.9 }, keyG);
  const keyA = el('g', {}, keyG);
  const keyB = el('g', {}, keyG);
  const keyAShape = el('circle', { cx: PX0 + 20, cy: PY1 + 16, r: 4.5, fill: CYAN }, keyA);
  const keyAText = el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 20, text: 'training' }, keyA);
  el('rect', { x: PX0 + 15.5, y: PY1 + 29.5, width: 9, height: 9, fill: BG, stroke: VIOLET, 'stroke-width': 2 }, keyB);
  const keyBText = el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 38, text: 'test' }, keyB);
  keyBText.style.fill = VIOLET;
  // Step 1's key: the two curves.
  const key0 = el('g', {}, keyG);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 16, y2: PY1 + 16, stroke: MUTED, 'stroke-width': 1.6 }, key0);
  el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 20, text: 'join the dots' }, key0);
  el('line', { x1: PX0 + 12, x2: PX0 + 28, y1: PY1 + 34, y2: PY1 + 34, stroke: STRONG, 'stroke-width': 1.6, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round' }, key0);
  el('text', { class: 'stage-text', x: PX0 + 32, y: PY1 + 38, text: 'true curve' }, key0);

  // --- Step 5: the folds diagram, drawn where the scatter was.
  const foldG = el('g', {}, svg);
  el('text', { class: 'stage-text', x: PX0 - 40, y: 22, text: `${K}-fold cross-validation` }, foldG);
  const CELL = 8;
  const GAP = 1;
  const FGAP = 7;
  const FX0 = PX0 - 16;
  const cellRows = [];
  for (let r = 0; r < K; r++) {
    const y = 52 + r * 36;
    el('text', { class: 'stage-text', x: FX0 - 26, y: y + 12, 'text-anchor': 'start', text: String(r + 1) }, foldG);
    cellRows.push([]);
  }
  const testRow = el('g', {}, foldG);
  const testCells = [];
  const testLabel = el('text', { class: 'stage-text', x: FX0, y: 268, text: '' }, testRow);
  const foldKey = el('g', {}, foldG);
  el('rect', { x: FX0, y: 296, width: 10, height: 10, fill: CYAN, 'fill-opacity': 0.55, stroke: CYAN }, foldKey);
  el('text', { class: 'stage-text', x: FX0 + 16, y: 305, text: 'fit' }, foldKey);
  el('rect', { x: FX0 + 62, y: 296, width: 10, height: 10, fill: BG, stroke: VIOLET, 'stroke-width': 2 }, foldKey);
  const vk = el('text', { class: 'stage-text', x: FX0 + 78, y: 305, text: 'validate' }, foldKey);
  vk.style.fill = VIOLET;
  el('rect', { x: FX0 + 160, y: 296, width: 10, height: 10, fill: LINE, stroke: MUTED, 'stroke-dasharray': '2 2' }, foldKey);
  el('text', { class: 'stage-text', x: FX0 + 176, y: 305, text: 'test, locked' }, foldKey);

  function drawFolds() {
    for (const row of cellRows) row.splice(0).forEach((c) => c.remove());
    testCells.splice(0).forEach((c) => c.remove());
    cv.folds.forEach((_, r) => {
      const y = 52 + r * 36;
      let x = FX0;
      cv.folds.forEach((f, j) => {
        for (let q = 0; q < f.length; q++) {
          const v = j === r;
          cellRows[r].push(el('rect', {
            x, y, width: CELL, height: 16,
            fill: v ? BG : CYAN, 'fill-opacity': v ? 1 : 0.55,
            stroke: v ? VIOLET : CYAN, 'stroke-width': v ? 1.6 : 1,
          }, foldG));
          x += CELL + GAP;
        }
        x += FGAP;
      });
    });
    let x = FX0;
    for (let q = 0; q < sp.test.length; q++) {
      testCells.push(el('rect', { x, y: 238, width: CELL, height: 16, fill: LINE, stroke: MUTED, 'stroke-dasharray': '2 2' }, testRow));
      x += CELL + GAP;
    }
    testLabel.textContent = `test: ${sp.test.length} points, used once`;
  }
  el('text', { class: 'stage-text', x: FX0 - 26, y: 44, text: 'round' }, foldG);

  // --- Right panel: read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rows = [0, 1, 2].map(() => el('text', { class: 'value' }, box));

  // Step 3: two bars, train v test.
  const barsG = el('g', {}, svg);
  const BY0 = 290;
  const BY1 = 130;
  el('line', { x1: RX0 + 20, x2: RX1 - 20, y1: BY0, y2: BY0, stroke: LINE }, barsG);
  const BW = 40;
  const BX = [RX0 + 40, RX0 + 120];
  const bars = BX.map((bx, k) => el('rect', { x: bx, width: BW, fill: k ? VIOLET : CYAN, 'fill-opacity': 0.25, stroke: k ? VIOLET : CYAN, 'stroke-width': 1.6, 'stroke-dasharray': k ? '4 3' : null }, barsG));
  const barVals = BX.map((bx) => el('text', { class: 'stage-text', x: bx + BW / 2, 'text-anchor': 'middle' }, barsG));
  BX.forEach((bx, k) => {
    const t = el('text', { class: 'stage-text', x: bx + BW / 2, y: BY0 + 18, 'text-anchor': 'middle', text: k ? 'test' : 'train' }, barsG);
    if (k) t.style.fill = VIOLET;
  });

  // Step 4: every split's two scores on one MSE axis.
  const stripG = el('g', {}, svg);
  const SX0 = RX0 + 44;
  const SX1 = RX1 - 8;
  const SY_TRAIN = 160;
  const SY_TEST = 214;
  const SY_AXIS = 250;
  el('text', { class: 'stage-text', x: RX0, y: 132, text: `the first ${STRIP_N} splits` }, stripG);
  const stLabA = el('text', { class: 'stage-text', x: RX0, y: SY_TRAIN + 4, text: 'train' }, stripG);
  stLabA.style.fill = CYAN;
  const stLabB = el('text', { class: 'stage-text', x: RX0, y: SY_TEST + 4, text: 'test' }, stripG);
  stLabB.style.fill = VIOLET;
  el('line', { x1: SX0, x2: SX1, y1: SY_AXIS, y2: SY_AXIS, stroke: LINE }, stripG);
  const stripMax = Math.max(...strip.rows.map((r) => r.test), ...strip.rows.map((r) => r.train)) * 1.08;
  let S = linearScale([0, stripMax], [SX0, SX1]);
  const stripAxisG = el('g', {}, stripG);
  const stripMarks = el('g', {}, stripG);
  const ringA = el('circle', { r: 9, fill: 'none', stroke: STRONG, 'stroke-width': 2 }, stripG);
  const ringB = el('circle', { r: 10, fill: 'none', stroke: STRONG, 'stroke-width': 2 }, stripG);
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY_AXIS + 34, 'text-anchor': 'middle', text: 'MSE' }, stripG);

  function drawStrip() {
    const cur = seen.get(seed);
    const hi = Math.max(stripMax, cur.test * 1.08, cur.train * 1.08);
    S = linearScale([0, hi], [SX0, SX1]);
    stripAxisG.replaceChildren();
    for (const t of S.ticks(3)) {
      el('line', { x1: S(t), x2: S(t), y1: SY_AXIS, y2: SY_AXIS + 5, stroke: LINE }, stripAxisG);
      el('text', { class: 'stage-text', x: S(t), y: SY_AXIS + 18, 'text-anchor': 'middle', text: String(t) }, stripAxisG);
    }
    stripMarks.replaceChildren();
    const list = [...seen.values()].filter((r) => r.seed < SPLIT_SEED + STRIP_N || r.seed === seed);
    for (const r of list) {
      el('circle', { cx: S(r.train), cy: SY_TRAIN, r: 4, fill: CYAN, 'fill-opacity': 0.7 }, stripMarks);
      el('rect', { x: S(r.test) - 4, y: SY_TEST - 4, width: 8, height: 8, fill: BG, stroke: VIOLET, 'stroke-width': 1.6, 'stroke-opacity': 0.85 }, stripMarks);
    }
    set(ringA, { cx: S(cur.train), cy: SY_TRAIN });
    set(ringB, { cx: S(cur.test), cy: SY_TEST });
  }

  // Step 5: validation score for each candidate degree.
  const cvG = el('g', {}, svg);
  const CY0 = 290;
  const CY1 = 140;
  el('line', { x1: RX0 + 10, x2: RX1, y1: CY0, y2: CY0, stroke: LINE }, cvG);
  el('text', { class: 'stage-text', x: RX0 + 10, y: 116, text: 'validation MSE' }, cvG);
  const CW = 22;
  const CX = CANDIDATES.map((_, k) => RX0 + 18 + k * 32);
  const cvBars = CANDIDATES.map(() => el('rect', { width: CW, fill: CYAN, 'fill-opacity': 0.25, stroke: CYAN, 'stroke-width': 1.4 }, cvG));
  const cvVals = CANDIDATES.map(() => el('text', { class: 'stage-text cv-val', 'text-anchor': 'middle', stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' }, cvG));
  CANDIDATES.forEach((deg, k) => el('text', { class: 'stage-text', x: CX[k] + CW / 2, y: CY0 + 16, 'text-anchor': 'middle', text: String(deg) }, cvG));
  el('text', { class: 'stage-text', x: (RX0 + RX1) / 2 + 6, y: CY0 + 34, 'text-anchor': 'middle', text: 'degree' }, cvG);
  const cvRing = el('rect', { fill: 'none', stroke: STRONG, 'stroke-width': 2, rx: 3 }, cvG);

  function drawCv() {
    const top = Math.max(...cv.scores.map((r) => r.cv));
    const B = linearScale([0, top], [CY0, CY1]);
    cv.scores.forEach((r, k) => {
      const y = B(r.cv);
      set(cvBars[k], { x: CX[k], y, height: CY0 - y });
      const best = r.degree === cv.best;
      cvBars[k].setAttribute('fill-opacity', best ? 0.6 : 0.2);
      cvVals[k].textContent = best ? fmt(r.cv) : '';
      set(cvVals[k], { x: CX[k] + CW / 2, y: y - 32 });
    });
    const k = CANDIDATES.indexOf(cv.best);
    const y = B(cv.scores[k].cv);
    set(cvRing, { x: CX[k] - 4, y: y - 22, width: CW + 8, height: CY0 - y + 24 });
  }

  // --- Controls (step 4).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const newBtn = make('button', { type: 'button', textContent: 'New split' });
  const firstBtn = make('button', { type: 'button', textContent: 'First split' });
  const splitGroup = make('div');
  splitGroup.append(newBtn, firstBtn);
  controls.append(splitGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { plot: 0, split: 0, step0: 0, fit: 0, sticks: 0, bars: 0, strip: 0, folds: 0, cv: 0 };
  let s = { ...LAYERS, plot: 1, step0: 1 };
  function target(i) {
    if (i === 0) return { ...LAYERS, plot: 1, step0: 1 };
    if (i === 1) return { ...LAYERS, plot: 1, split: 1 };
    if (i === 2) return { ...LAYERS, plot: 1, split: 1, fit: 1, sticks: 1, bars: 1 };
    if (i === 3) return { ...LAYERS, plot: 1, split: 1, fit: 1, strip: 1 };
    return { ...LAYERS, folds: 1, cv: 1 };
  }

  function set(node, attrs) {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  }
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };

  function render() {
    op(plotG, s.plot);
    op(truthPath, s.step0);
    op(dotsPath, s.step0);
    op(key0, s.step0);
    op(keyA, 1 - s.step0);
    op(keyB, s.split);
    keyAShape.setAttribute('fill', CYAN);
    keyAText.textContent = s.split > 0.5 || s.step0 < 0.5 ? 'training' : 'data';

    const isTest = new Set(sp.test);
    d.xs.forEach((x, i) => {
      const t = isTest.has(i) ? s.split : 0;
      op(circles[i], 1 - t);
      op(squares[i], t);
    });

    set(fitPath, { d: pathFrom(GRID.map((x) => [x, polyEval(sc.coef, x)]), X, Y) });
    op(fitPath, s.fit);
    d.xs.forEach((x, i) => {
      const on = isTest.has(i);
      set(sticks[i], { x1: X(x), x2: X(x), y1: Y(d.ys[i]), y2: Y(clamp(polyEval(sc.coef, x), ...Y_DOM)) });
      sticks[i].style.display = on ? '' : 'none';
    });
    op(sticksG, s.sticks);

    // Bars.
    const top = Math.max(sc.train, sc.test) || 1;
    const B = linearScale([0, top], [BY0, BY1]);
    [sc.train, sc.test].forEach((v, k) => {
      const y = B(v);
      set(bars[k], { y, height: BY0 - y });
      barVals[k].textContent = fmt(v);
      set(barVals[k], { y: y - 7 });
    });
    op(barsG, s.bars);

    op(stripG, s.strip);
    if (s.strip > 0.001) drawStrip();
    op(foldG, s.folds);
    op(cvG, s.cv);
    if (s.cv > 0.001) drawCv();

    // Read-out box.
    const texts = readout(step, { seed });
    rows.forEach((r, k) => { r.textContent = texts[k] || ''; });
    let w = 196;
    try {
      const lens = rows.map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0 + 4, 598 - w);
    set(boxRect, { x: bx, y: BOX_Y, width: w, height: 16 + ROW * texts.length });
    rows.forEach((r, k) => set(r, { x: bx + 12, y: BOX_Y + 8 + ROW * k + ROW / 2 + 5 }));
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

  // Sentences on the page that quote values; refreshed when the split changes.
  function fillPage() {
    const filled = pageValues(seed);
    for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
      const val = filled[span.dataset.value];
      if (val !== undefined) span.textContent = val;
    }
    for (const p of doc.querySelectorAll('main.lesson [data-fill]')) {
      const spans = [...p.querySelectorAll('[data-value]')];
      let ok = spans.every((sp2) => sp2.dataset.value in filled);
      const when = p.dataset.when;
      if (when === 'train-higher') ok = ok && sc.train > sc.test;
      if (when === 'train-lower') ok = ok && sc.train <= sc.test;
      p.hidden = !ok;
    }
  }

  const spoken = (i) => {
    if (i === 0) {
      const t = stepOneScores(d);
      return `${N} points. Joining the dots scores an MSE of ${fmt(t.dots)} on them; the true curve scores ${fmt(t.truth)}.`;
    }
    if (i === 1) return `Split ${splitNo(seed)}: ${sp.train.length} training points, ${sp.test.length} test points.`;
    if (i === 2 || i === 3) {
      const more = sc.train > sc.test ? ' This time the training MSE is the higher one.' : '';
      return `Split ${splitNo(seed)}. Training MSE ${fmt(sc.train)}, test MSE ${fmt(sc.test)}.${more}`;
    }
    return `${K}-fold cross-validation on the training points picks degree ${cv.best}. Its test MSE, measured once, is ${fmt(cv.test)}.`;
  };

  function setSplit(next) {
    seed = next;
    sp = splitFor(seed);
    sc = score(d, sp);
    cv = cvTable(d, sp);
    if (!seen.has(seed)) seen.set(seed, { seed, train: sc.train, test: sc.test });
    drawFolds();
    fillPage();
    render();
    announce(spoken(step));
  }
  newBtn.addEventListener('click', () => setSplit(seed + 1));
  firstBtn.addEventListener('click', () => setSplit(SPLIT_SEED));

  drawFolds();
  fillPage();
  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 5,
    show(i, animate) {
      step = i;
      showGroup(splitGroup, i === 3);
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
