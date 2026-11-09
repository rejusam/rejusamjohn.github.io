// Lesson 3, "Centre and spread": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, randn, mean, median } from '../assets/maths.js';
import { linearScale, clamp, el, fmt as tickFmt, tween, draggable } from '../assets/plot.js';

export const SEED = 21;
export const N = 9;
export const LO = 0;
export const HI = 120;

// Nine made-up commute times in minutes: round(50 + 8z), z ~ N(0, 1).
export function lessonData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => Math.round(50 + 8 * randn(rng)));
}

// A dragged value: kept within 0–120 and rounded to a whole minute.
export const clampValue = (v) => Math.round(clamp(v, LO, HI));

// Centre and spread of the values. Variances are given both ways:
// popVar divides by n, sampleVar by n − 1.
export function stats(values) {
  const n = values.length;
  const m = mean(values);
  let ss = 0;
  let below = 0;
  let above = 0;
  for (const x of values) {
    ss += (x - m) ** 2;
    if (x < m) below += m - x;
    else above += x - m;
  }
  const popVar = ss / n;
  const sampleVar = ss / (n - 1);
  return {
    n,
    sum: values.reduce((s, x) => s + x, 0),
    mean: m,
    median: median(values),
    ss,
    popVar,
    sampleVar,
    popSd: Math.sqrt(popVar),
    sampleSd: Math.sqrt(sampleVar),
    below,
    above,
  };
}

// Index of the largest value (the first one, if two tie).
export const largestIndex = (values) => values.reduce((best, x, i) => (x > values[best] ? i : best), 0);

// A copy of the values with the largest one moved to `to`.
export function moveLargest(values, to) {
  const out = values.slice();
  out[largestIndex(values)] = clampValue(to);
  return out;
}

// z-score with the population SD, as in Lesson 2. Undefined (null) with no spread.
export const zscore = (x, st) => (st.popSd === 0 ? null : (x - st.mean) / st.popSd);

// Indices in sorted order (ties keep their original order).
export const order = (values) => values.map((_, i) => i).sort((i, j) => values[i] - values[j] || i - j);

// Stacking levels for dots at screen positions px, so that no two dots
// closer than `gap` share a level. Each dot takes the lowest free level.
export function levels(px, gap) {
  const idx = px.map((_, i) => i).sort((i, j) => px[i] - px[j] || i - j);
  const out = new Array(px.length).fill(0);
  const placed = [];
  for (const i of idx) {
    let lv = 0;
    while (placed.some((j) => out[j] === lv && Math.abs(px[j] - px[i]) < gap)) lv++;
    out[i] = lv;
    placed.push(i);
  }
  return out;
}

// Fixed decimals with a true minus sign.
export function fmt(v, dp) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

const f1 = (v) => fmt(v, 1);
const whole = (v) => fmt(v, 0);

// The read-out rows for each step, as shown on the stage.
// `sel` is the index of the chosen value; `start` is the starting data.
export function readout(step, values, sel, start = lessonData()) {
  const st = stats(values);
  const n = st.n;
  if (step === 0) {
    return [
      `mean: ${f1(st.mean)} min`,
      `below the mean: ${f1(st.below)} min in total`,
      `above the mean: ${f1(st.above)} min in total`,
    ];
  }
  if (step === 1) {
    const mid = (n + 1) / 2;
    return [
      `sorted: ${order(values).map((i) => whole(values[i])).join(' ')}`,
      `median: ${whole(st.median)} min (${ordinal(mid)} of ${n})`,
    ];
  }
  if (step === 2) {
    const s0 = stats(start);
    return [
      `mean: ${f1(s0.mean)} → ${f1(st.mean)} min`,
      `median: ${whole(s0.median)} → ${whole(st.median)} min`,
    ];
  }
  if (step === 3) {
    return [
      `squared distances, added up: ${f1(st.ss)} min²`,
      `their average (÷ ${n}): ${f1(st.popVar)} min²`,
    ];
  }
  if (step === 4) {
    return [
      `population variance (÷ ${n}): ${f1(st.popVar)} min²`,
      `sample variance (÷ ${n - 1}): ${f1(st.sampleVar)} min²`,
    ];
  }
  const z = zscore(values[sel], st);
  return [
    `population SD (÷ ${n}): ${f1(st.popSd)} min`,
    `sample SD (÷ ${n - 1}): ${f1(st.sampleSd)} min`,
    `person ${sel + 1}, ${whole(values[sel])} min: ${z === null ? 'z not defined (SD is 0)' : `z = ${fmt(z, 2)}`}`,
  ];
}

function ordinal(k) {
  const tail = k % 100 >= 11 && k % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[k % 10] || 'th');
  return `${k}${tail}`;
}

// Values quoted in the page text, all from the starting data.
export function pageValues(values) {
  const st = stats(values);
  const moved = stats(moveLargest(values, HI));
  return {
    n: String(st.n),
    sum: whole(st.sum),
    'mean-start': f1(st.mean),
    'median-start': whole(st.median),
    sorted: order(values).map((i) => whole(values[i])).join(', '),
    'max-start': whole(Math.max(...values)),
    hi: String(HI),
    'mean-moved': f1(moved.mean),
    'median-moved': whole(moved.median),
    ss: f1(st.ss),
    'pop-var': f1(st.popVar),
    'sample-var': f1(st.sampleVar),
    'pop-sd': f1(st.popSd),
    'sample-sd': f1(st.sampleSd),
  };
}

// ---------------------------------------------------------------- the stage

const PLANK = 252;
const AXIS = 290;
const TOP = 112;
const RULER = 152;
const R = 6;
const GAP = 13;
const BOX = { x: 40, y: 10, row: 20 };
const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
// A dark outline behind text, so a line passing under it doesn't cut through.
const HALO = { stroke: '#1e293b', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const start = lessonData();
  let values = start.slice();
  let sel = largestIndex(values);

  const X = linearScale([LO, HI], [40, 576]);
  const unit = X(1) - X(0);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Nine commute times as dots along a line from 0 to 120 minutes, with the mean, the median and the spread drawn on top.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l3-plot' }, defs);
  el('rect', { x: X(LO), y: TOP - 16, width: X(HI) - X(LO), height: PLANK - TOP + 16 }, clip);

  // Axis: minutes from 0 to 120.
  const axisG = el('g', { class: 'axes' }, svg);
  el('line', { x1: X(LO), x2: X(HI), y1: AXIS, y2: AXIS, stroke: '#334155' }, axisG);
  for (const t of X.ticks(6)) {
    el('line', { x1: X(t), x2: X(t), y1: AXIS, y2: AXIS + 5, stroke: '#334155' }, axisG);
    el('text', { x: X(t), y: AXIS + 18, 'text-anchor': 'middle', text: tickFmt(t) }, axisG);
  }
  el('text', { x: (X(LO) + X(HI)) / 2, y: AXIS + 38, 'text-anchor': 'middle', text: 'commute time in minutes' }, axisG);

  // The line the dots sit on, and the plank and its pivot for step 1.
  el('line', { x1: X(LO), x2: X(HI), y1: PLANK, y2: PLANK, stroke: '#334155' }, svg);
  const plank = el('line', { x1: X(LO), x2: X(HI), y1: PLANK, y2: PLANK, stroke: INK, 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
  const pivot = el('path', { fill: INK }, svg);

  // Squares on each distance from the mean (step 4), and the average squares.
  const clipped = el('g', { 'clip-path': 'url(#l3-plot)' }, svg);
  const squares = el('g', {}, clipped);
  const sq = values.map(() => el('rect', { fill: CYAN, 'fill-opacity': 0.06, stroke: MUTED, 'stroke-width': 1 }, squares));
  const avg = el('rect', { fill: 'none', stroke: STRONG, 'stroke-width': 2.5 }, clipped);
  const avg8 = el('rect', { class: 'series-b' }, clipped);

  // The z ruler (step 6): ticks one population SD apart.
  const ruler = el('g', {}, svg);
  const rulerLine = el('line', { y1: RULER, y2: RULER, stroke: INK, 'stroke-width': 1.5 }, ruler);
  const rulerTicks = [-3, -2, -1, 0, 1, 2, 3].map((k) => ({
    k,
    line: el('line', { y1: RULER - 5, y2: RULER + 5, stroke: INK, 'stroke-width': 1.5 }, ruler),
    text: el('text', { class: 'stage-text', y: RULER - 10, 'text-anchor': 'middle', text: tickFmt(k), ...HALO }, ruler),
  }));
  const rulerCap = el('text', { class: 'stage-text', y: RULER + 4, 'text-anchor': 'end', 'font-style': 'italic', text: 'z' }, ruler);
  const zGuide = el('line', { stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, ruler);

  // Mean (cyan, solid) and median (violet, dashed), with faint start lines for step 3.
  const ghostMean = el('line', { y1: TOP, y2: PLANK, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '1 4', 'stroke-linecap': 'round' }, svg);
  const ghostMedian = el('line', { y1: TOP, y2: PLANK, stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '1 4', 'stroke-linecap': 'round' }, svg);
  const meanLine = el('line', { class: 'series-a', y1: TOP, y2: PLANK }, svg);
  const medianLine = el('line', { class: 'series-b', y1: TOP, y2: PLANK }, svg);
  const meanLabel = el('text', { class: 'value', ...HALO }, svg);
  const medianLabel = el('text', { class: 'value', ...HALO }, svg);
  meanLabel.style.fill = CYAN;
  medianLabel.style.fill = VIOLET;

  // Dots, a ring for the chosen one, and larger invisible handles to drag.
  const ring = el('circle', { r: R + 4, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, svg);
  const dotsG = el('g', {}, svg);
  const dots = values.map(() => el('circle', { class: 'dot', r: R, stroke: BG, 'stroke-width': 1.5 }, dotsG));
  const handlesG = el('g', {}, svg);
  const handles = values.map((_, i) => el('circle', {
    class: 'handle', r: 16, fill: 'transparent', tabindex: 0, role: 'slider',
    'aria-label': `Person ${i + 1}’s commute time`,
    'aria-valuemin': LO, 'aria-valuemax': HI,
  }, handlesG));

  // Read-out box, top left, above everything the plot draws.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: '#1e293b', 'fill-opacity': 0.92, stroke: '#334155' }, box);
  const rows = [0, 1, 2].map(() => ({ sample: el('line', {}, box), text: el('text', { class: 'value' }, box), off: 12 }));
  const SAMPLES = {
    mean: { class: 'series-a' },
    median: { class: 'series-b' },
    avg: { stroke: STRONG, 'stroke-width': 2.5 },
    avg8: { class: 'series-b' },
  };
  const ROW_SAMPLES = [
    ['mean', null, null],
    [null, 'median'],
    ['mean', 'median'],
    [null, 'avg'],
    ['avg', 'avg8'],
    [null, null, null],
  ];

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const outlierBtn = make('button', { type: 'button', textContent: `Move the largest to ${HI}` });
  const resetBtn = make('button', { type: 'button', textContent: 'Reset the data' });
  const outlierGroup = make('div');
  outlierGroup.append(outlierBtn);
  controls.append(outlierGroup, resetBtn);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Layer opacities animate between steps.
  let step = 0;
  let token = 0;
  let valueToken = 0;
  const LAYERS = { plank: 0, mean: 0, median: 0, ghost: 0, squares: 0, avg: 0, avg8: 0, ruler: 0 };
  let s = { ...LAYERS, plank: 1, mean: 1 };

  function target(i) {
    if (i === 0) return { ...LAYERS, plank: 1, mean: 1 };
    if (i === 1) return { ...LAYERS, median: 1 };
    if (i === 2) return { ...LAYERS, mean: 1, median: 1, ghost: 1 };
    if (i === 3) return { ...LAYERS, mean: 1, squares: 1, avg: 1 };
    if (i === 4) return { ...LAYERS, mean: 1, squares: 0.35, avg: 1, avg8: 1 };
    return { ...LAYERS, mean: 1, ruler: 1 };
  }

  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };

  function setRow(k, text, key) {
    const r = rows[k];
    r.text.textContent = text || '';
    r.off = key ? 40 : 12;
    r.sample.style.display = key ? '' : 'none';
    for (const a of ['class', 'stroke', 'stroke-width']) r.sample.removeAttribute(a);
    if (key) for (const [a, v] of Object.entries(SAMPLES[key])) r.sample.setAttribute(a, v);
  }

  // Places the mean and median labels on opposite sides of their lines,
  // and inside the stage.
  function placeLabels(st) {
    const measure = (t) => {
      try { return t.getComputedTextLength() || 70; } catch { return 70; }
    };
    meanLabel.textContent = `mean ${f1(st.mean)}`;
    medianLabel.textContent = `median ${whole(st.median)}`;
    const both = s.mean > 0.001 && s.median > 0.001;
    const items = [
      { t: meanLabel, x: X(st.mean), right: !both || st.mean >= st.median },
      { t: medianLabel, x: X(st.median), right: !both || st.median > st.mean },
    ];
    const sides = items.map((it) => {
      let right = it.right;
      const w = measure(it.t);
      if (right && it.x + 6 + w > 596) right = false;
      if (!right && it.x - 6 - w < 4) right = true;
      it.t.setAttribute('x', (it.x + (right ? 6 : -6)).toFixed(1));
      it.t.setAttribute('text-anchor', right ? 'start' : 'end');
      return right;
    });
    meanLabel.setAttribute('y', TOP - 4);
    // Both pushed to the same side: drop the median label a line lower.
    medianLabel.setAttribute('y', both && sides[0] === sides[1] ? TOP + 12 : TOP - 4);
  }

  function render() {
    const st = stats(values);
    const px = values.map((v) => X(v));
    const lv = levels(px, GAP);
    const cy = lv.map((l) => PLANK - R - 2 - l * GAP);
    const ord = order(values);
    const mx = X(st.mean);

    op(plank, s.plank);
    pivot.setAttribute('d', `M${mx.toFixed(2)},${PLANK + 2} l-10,16 h20 Z`);
    op(pivot, s.plank);

    // Squares, anchored on the line, from each value across to the mean.
    sq.forEach((r, i) => {
      const w = Math.abs(px[i] - mx);
      r.setAttribute('x', Math.min(px[i], mx).toFixed(2));
      r.setAttribute('y', (PLANK - w).toFixed(2));
      r.setAttribute('width', w.toFixed(2));
      r.setAttribute('height', w.toFixed(2));
    });
    op(squares, s.squares);
    const setSquare = (node, side) => {
      node.setAttribute('x', mx.toFixed(2));
      node.setAttribute('y', (PLANK - side).toFixed(2));
      node.setAttribute('width', side.toFixed(2));
      node.setAttribute('height', side.toFixed(2));
    };
    setSquare(avg, st.popSd * unit);
    setSquare(avg8, st.sampleSd * unit);
    op(avg, s.avg);
    op(avg8, s.avg8);

    // Mean and median lines.
    const vline = (node, x) => { node.setAttribute('x1', x.toFixed(2)); node.setAttribute('x2', x.toFixed(2)); };
    vline(meanLine, mx);
    vline(medianLine, X(st.median));
    const s0 = stats(start);
    vline(ghostMean, X(s0.mean));
    vline(ghostMedian, X(s0.median));
    op(meanLine, s.mean);
    op(meanLabel, s.mean);
    op(medianLine, s.median);
    op(medianLabel, s.median);
    op(ghostMean, s.ghost * 0.9);
    op(ghostMedian, s.ghost * 0.9);
    placeLabels(st);

    // The z ruler, with a guide from the chosen dot.
    const sd = st.popSd;
    if (sd > 0) {
      const ks = rulerTicks.filter(({ k }) => st.mean + k * sd >= LO && st.mean + k * sd <= HI).map(({ k }) => k);
      const lo = X(Math.max(LO, st.mean - 3 * sd));
      const hi = X(Math.min(HI, st.mean + 3 * sd));
      rulerLine.setAttribute('x1', lo.toFixed(2));
      rulerLine.setAttribute('x2', hi.toFixed(2));
      rulerTicks.forEach(({ k, line, text }) => {
        const on = ks.includes(k);
        const x = X(st.mean + k * sd);
        line.style.display = on ? '' : 'none';
        text.style.display = on ? '' : 'none';
        line.setAttribute('x1', x.toFixed(2));
        line.setAttribute('x2', x.toFixed(2));
        text.setAttribute('x', x.toFixed(2));
      });
      rulerCap.setAttribute('x', (lo - 10).toFixed(2));
      rulerCap.style.display = '';
      rulerLine.style.display = '';
    } else {
      rulerTicks.forEach(({ line, text }) => { line.style.display = 'none'; text.style.display = 'none'; });
      rulerLine.style.display = 'none';
      rulerCap.style.display = 'none';
    }
    zGuide.setAttribute('x1', px[sel].toFixed(2));
    zGuide.setAttribute('x2', px[sel].toFixed(2));
    zGuide.setAttribute('y1', RULER);
    zGuide.setAttribute('y2', (cy[sel] - R - 4).toFixed(2));
    zGuide.style.display = sd > 0 ? '' : 'none';
    op(ruler, s.ruler);

    // Dots and their handles.
    dots.forEach((d, i) => {
      d.setAttribute('cx', px[i].toFixed(2));
      d.setAttribute('cy', cy[i].toFixed(2));
    });
    handles.forEach((h, i) => {
      h.setAttribute('cx', px[i].toFixed(2));
      h.setAttribute('cy', cy[i].toFixed(2));
      h.setAttribute('aria-valuenow', whole(values[i]));
      h.setAttribute('aria-valuetext', `${whole(values[i])} minutes`);
    });

    // The ring: the middle value (step 2), the largest (step 3), the chosen one (step 6).
    const ringed = step === 1 ? ord[(values.length - 1) / 2] : step === 2 ? largestIndex(values) : step === 5 ? sel : -1;
    ring.style.display = ringed < 0 ? 'none' : '';
    if (ringed >= 0) {
      ring.setAttribute('cx', px[ringed].toFixed(2));
      ring.setAttribute('cy', cy[ringed].toFixed(2));
    }

    // Read-out box.
    const texts = readout(step, values, sel, start);
    texts.forEach((t, k) => setRow(k, t, ROW_SAMPLES[step][k]));
    const nRows = texts.length;
    rows.forEach((r, k) => {
      const on = k < nRows;
      r.text.style.display = on ? '' : 'none';
      if (!on) r.sample.style.display = 'none';
    });
    let w = 300;
    try {
      const lens = rows.slice(0, nRows).map((r) => r.text.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...rows.slice(0, nRows).map((r, k) => r.off + lens[k])) + 12;
    } catch {
      // Keep the fixed width.
    }
    const h = 16 + BOX.row * nRows;
    boxRect.setAttribute('x', BOX.x);
    boxRect.setAttribute('y', BOX.y);
    boxRect.setAttribute('width', w.toFixed(1));
    boxRect.setAttribute('height', h);
    rows.forEach((r, k) => {
      const yc = BOX.y + 8 + BOX.row * k + BOX.row / 2;
      r.text.setAttribute('x', (BOX.x + r.off).toFixed(1));
      r.text.setAttribute('y', yc + 5);
      r.sample.setAttribute('x1', BOX.x + 12);
      r.sample.setAttribute('x2', BOX.x + 32);
      r.sample.setAttribute('y1', yc);
      r.sample.setAttribute('y2', yc);
    });
  }

  function animateTo(to, ms) {
    const my = ++token;
    const from = { ...s };
    return tween(ms, (k) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = from[key] + (to[key] - from[key]) * k;
      s = next;
      render();
    }, { reduced });
  }

  // --- Words for the live region: the box's numbers, as sentences.
  const spoken = (i) => {
    const st = stats(values);
    const n = st.n;
    if (i === 0) return `Mean ${f1(st.mean)} minutes, the balance point. The distances below it add up to ${f1(st.below)} minutes, and those above it to ${f1(st.above)}.`;
    if (i === 1) return `In order: ${order(values).map((k) => whole(values[k])).join(', ')}. The median is the middle value, ${whole(st.median)}.`;
    if (i === 2) {
      const s0 = stats(start);
      return `Mean ${f1(st.mean)} minutes, from ${f1(s0.mean)} at the start. Median ${whole(st.median)} minutes, from ${whole(s0.median)}.`;
    }
    if (i === 3) return `Squared distances add up to ${f1(st.ss)} square minutes. Their average, dividing by ${n}, is ${f1(st.popVar)}.`;
    if (i === 4) return `Population variance, dividing by ${n}: ${f1(st.popVar)} square minutes. Sample variance, dividing by ${n - 1}: ${f1(st.sampleVar)}.`;
    const z = zscore(values[sel], st);
    return `Population SD ${f1(st.popSd)} minutes, sample SD ${f1(st.sampleSd)}. Person ${sel + 1}, ${whole(values[sel])} minutes: ` +
      (z === null ? 'z-score not defined, because the SD is 0.' : `z-score ${fmt(z, 2)}.`);
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction.
  const setValue = (i, v) => {
    valueToken++;
    const next = clampValue(v);
    sel = i;
    if (next !== values[i]) {
      values = values.slice();
      values[i] = next;
    }
    render();
    say(`Person ${i + 1}: ${whole(values[i])} minutes. ${spoken(step)}`);
  };

  handles.forEach((hdl, i) => {
    // Registered before draggable, so these keys are handled here only.
    hdl.addEventListener('keydown', (e) => {
      const v = values[i];
      const map = {
        ArrowLeft: v - 1, ArrowDown: v - 1, ArrowRight: v + 1, ArrowUp: v + 1,
        PageDown: v - 10, PageUp: v + 10, Home: LO, End: HI,
      };
      if (!(e.key in map)) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      setValue(i, map[e.key]);
    });
    hdl.addEventListener('focus', () => {
      if (sel === i) return;
      sel = i;
      render();
      if (step === 5) say(spoken(5));
    });
    draggable(hdl, {
      bounds: { x0: X(LO), x1: X(HI), y0: 0, y1: 340 },
      onDrag: (x) => setValue(i, X.invert(x)),
    });
  });

  outlierBtn.addEventListener('click', () => {
    const i = largestIndex(values);
    sel = i;
    const from = values[i];
    const my = ++valueToken;
    tween(reduced ? 0 : 700, (k) => {
      if (my !== valueToken) return;
      values = values.slice();
      values[i] = from + (HI - from) * k;
      render();
    }, { reduced }).then(() => {
      if (my !== valueToken) return;
      values = values.slice();
      values[i] = HI;
      render();
      announce(`Person ${i + 1} moved to ${HI} minutes. ${spoken(2)}`);
    });
  });

  resetBtn.addEventListener('click', () => {
    valueToken++;
    values = start.slice();
    sel = largestIndex(values);
    render();
    announce(`Data reset. ${spoken(step)}`);
  });

  // Sentences on the page that quote values from the starting data (hidden until filled).
  const filled = pageValues(start);
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const v = filled[span.dataset.value];
    if (v !== undefined) span.textContent = v;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) p.hidden = false;
  }

  render();
  // Text widths change once the mono font arrives.
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      showGroup(outlierGroup, i === 2);
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      return animateTo(to, 600);
    },
  };
}
