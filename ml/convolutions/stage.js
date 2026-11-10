// Lesson L4.7, "Convolutions": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l4-convolutions.test.mjs);
// the DOM is only touched inside createStage. Every map is conv2d from conv.js on the
// image below, every product and sum is read off the same window, and every number in
// the page's text comes from storyValues (checked by the tests). Nothing is trained here.

import { mulberry32, randn } from '../assets/maths.js';
import { conv2d, convOutSize } from '../assets/conv.js';
import { el, tween, pixelGrid, pixelFill } from '../assets/plot.js';

// ---- the image --------------------------------------------------------------

export const N = 12;
export const F = 3;
export const SEED_NOISE = 191;
export const NOISE_SD = 0.1;

// 12 × 12, background 0. A 4 × 4 square of 1s in rows 2–5, columns 2–5, and a
// diagonal line of 0.5s from row 6, column 9 down to row 10, column 5 (r + c = 15).
const inSquare = (r, c) => r >= 2 && r <= 5 && c >= 2 && c <= 5;
export function cleanImage() {
  return Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => {
    if (inSquare(r, c)) return 1;
    if (r >= 6 && r <= 10 && r + c === 15) return 0.5;
    return 0;
  }));
}

// The noisy copy: one randn per pixel in row-major order from mulberry32(191);
// pixel = clean + 0.1 · randn, clipped to [0, 1].
export function noisyImage() {
  const rng = mulberry32(SEED_NOISE);
  return cleanImage().map((row) => row.map((v) => Math.min(1, Math.max(0, v + NOISE_SD * randn(rng)))));
}

const CLEAN = cleanImage();
const NOISY = noisyImage();

// Move every pixel d columns to the right; columns that come in from the left are 0.
export function shiftRight(img, d) {
  return img.map((row) => row.map((_, c) => (c >= d ? row[c - d] : 0)));
}

// ---- filters ------------------------------------------------------------------

export const FILTERS = {
  vertical: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
  horizontal: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
  blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
};
export const FILTER_LABEL = { vertical: 'vertical edge', horizontal: 'horizontal edge', blur: 'blur', custom: 'your own' };
// Suggested on the page for the editable filter: 1 from top right to bottom left, −0.5 elsewhere.
export const TRY_LINE = [[-0.5, -0.5, 1], [-0.5, 1, -0.5], [1, -0.5, -0.5]];

export const W_MIN = -2;
export const W_MAX = 2;
export const W_STEP = 0.5;

// One typed weight: a number in [−2, 2] on the 0.5 grid. Anything that is not a
// number becomes 0; the typographic minus is accepted.
export function parseWeight(raw) {
  const v = Number(String(raw ?? '').trim().replace(/−/g, '-'));
  if (Number.isNaN(v) || String(raw ?? '').trim() === '') return 0;
  const c = Math.min(W_MAX, Math.max(W_MIN, v));
  // Ties go away from 0 on both sides: 0.25 → 0.5 and −0.25 → −0.5.
  const snapped = Math.sign(c) * Math.round(Math.abs(c) / W_STEP) * W_STEP;
  return snapped === 0 ? 0 : snapped;
}
export const parseFilter = (raw) => raw.map((row) => row.map(parseWeight));

export const STRIDES = [1, 2];
export const PADS = [0, 1];
export const SHIFTS = [0, 1, 2];

export const mapFor = (img, W, { stride = 1, pad = 0 } = {}) => conv2d(img, W, { stride, pad });

// The nine cells under the window for map position (i, j): image row r, column c,
// pixel x (0 in the padding), weight w, product x·w; and their sum.
export function windowCells(img, W, i, j, { stride = 1, pad = 0 } = {}) {
  const cells = [];
  let sum = 0;
  for (let a = 0; a < F; a++) {
    for (let b = 0; b < F; b++) {
      const r = i * stride + a - pad;
      const c = j * stride + b - pad;
      const inside = r >= 0 && r < img.length && c >= 0 && c < img.length;
      const x = inside ? img[r][c] : 0;
      const w = W[a][b];
      const prod = x * w;
      sum += prod;
      cells.push({ a, b, r, c, x, w, prod, pad: !inside });
    }
  }
  return { cells, sum };
}

export function sizeInfo(stride, pad) {
  return {
    formula: Math.floor((N + 2 * pad - F) / stride) + 1,
    actual: mapFor(CLEAN, FILTERS.vertical, { stride, pad }).length,
  };
}

// Stride 1, padding 0: does the window at (i, j) cover any pixel of the square?
const touchesSquare = (i, j) => i <= 5 && i + 2 >= 2 && j <= 5 && j + 2 >= 2;

// The largest |value| among windows that hold no pixel of the square.
export function awayFromSquare(map) {
  let m = 0;
  map.forEach((row, i) => row.forEach((v, j) => { if (!touchesSquare(i, j)) m = Math.max(m, Math.abs(v)); }));
  return m;
}
// The largest value among windows on the square, and among the rest (the line's).
export function lineVsSquare(map) {
  let square = -Infinity;
  let line = -Infinity;
  map.forEach((row, i) => row.forEach((v, j) => {
    if (touchesSquare(i, j)) square = Math.max(square, v);
    else line = Math.max(line, v);
  }));
  return { square, line };
}

export const sharedCount = () => F * F;
// A dense layer from every pixel to every value of the 10 × 10 map (weights only).
export const denseCount = () => N * N * convOutSize(N, F) ** 2;

// Map positions holding the largest (or smallest) value, row-major.
export function extremes(map, pick = 'max') {
  const vals = map.flat();
  const best = pick === 'max' ? Math.max(...vals) : Math.min(...vals);
  const cells = [];
  map.forEach((row, i) => row.forEach((v, j) => { if (v === best) cells.push([i, j]); }));
  return { best, cells };
}

const span = (xs) => {
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  return lo === hi ? `${lo}` : `${lo}–${hi}`;
};
// "rows 2–3, columns 0–1" for a block of cells; singular when one row or column.
export function placeText(cells) {
  const rs = cells.map((c) => c[0]);
  const cs = cells.map((c) => c[1]);
  const rw = Math.min(...rs) === Math.max(...rs) ? 'row' : 'rows';
  const cw = Math.min(...cs) === Math.max(...cs) ? 'column' : 'columns';
  return `${rw} ${span(rs)}, ${cw} ${span(cs)}`;
}

// Up to two decimals, trailing zeros dropped, with a true minus sign.
export function fmtV(v) {
  const r = Math.round(v * 100) / 100;
  if (r === 0) return '0';
  const s = String(Math.abs(r));
  return r < 0 ? `−${s}` : s;
}
// A map cell carries a minus sign whenever its value reads as negative at two decimals.
export const minusShown = (v) => v < 0 && fmtV(v) !== '0';
const fmtW = (w) => (Math.abs(w - 1 / 9) < 1e-12 ? '1/9' : fmtV(w));
const fmtCount = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

// ---- read-out ------------------------------------------------------------------

// The state the stage is in, with defaults, and what it draws from.
function resolve(st) {
  const step = st.step;
  const stride = step === 4 ? st.stride || 1 : 1;
  const pad = step === 4 ? st.pad || 0 : 0;
  const filter = step === 3 ? st.filter || 'vertical' : 'vertical';
  const W = filter === 'custom' ? parseFilter(st.weights || FILTERS.vertical) : FILTERS[filter];
  let img = step === 3 && st.image === 'noisy' ? NOISY : CLEAN;
  const shift = step === 5 ? st.shift || 0 : 0;
  if (shift) img = shiftRight(img, shift);
  const map = mapFor(img, W, { stride, pad });
  return { step, stride, pad, filter, W, img, shift, map, m: map.length };
}

export function readout(st) {
  const s = resolve(st);
  const [a, b] = st.cursor || [0, 0];
  if (s.step === 0) {
    const r = Math.min(a, N - 1);
    const c = Math.min(b, N - 1);
    return [`pixel row ${r}, column ${c}: ${fmtV(s.img[r][c])}`, `${N} × ${N} = ${N * N} numbers · 0 dark, 1 bright`];
  }
  const i = Math.min(a, s.m - 1);
  const j = Math.min(b, s.m - 1);
  const v = s.map[i][j];
  const hi = extremes(s.map, 'max');
  const lo = extremes(s.map, 'min');
  if (s.step === 1) {
    const w = windowCells(s.img, s.W, i, j);
    return [`window for map row ${i}, column ${j}`, `nine products, sum = ${fmtV(w.sum)}`];
  }
  if (s.step === 2) {
    const total = s.m * s.m;
    const filled = st.filled === undefined ? total : st.filled;
    if (filled < total) return [`sliding… ${filled} of ${total} positions`];
    return [`map row ${i}, column ${j}: ${fmtV(v)}`, `largest ${fmtV(hi.best)} · smallest ${fmtV(lo.best)}`];
  }
  if (s.step === 3) {
    const which = st.image === 'noisy' ? 'noisy' : 'clean';
    return [`${FILTER_LABEL[s.filter]} filter, ${which} image`, `map row ${i}, column ${j}: ${fmtV(v)}`, `largest ${fmtV(hi.best)} · smallest ${fmtV(lo.best)}`];
  }
  if (s.step === 4) {
    const f = Math.floor((N + 2 * s.pad - F) / s.stride) + 1;
    return [
      `s = ${s.stride}, p = ${s.pad}: ⌊(${N} + 2·${s.pad} − ${F})/${s.stride}⌋ + 1 = ${f}`,
      `feature map: ${s.m} × ${s.m}`,
      `map row ${i}, column ${j}: ${fmtV(v)}`,
    ];
  }
  return [
    `image shifted right by ${s.shift}`,
    `largest ${fmtV(hi.best)} at ${placeText(hi.cells)}`,
    `${sharedCount()} weights, shared by all ${s.m * s.m} positions`,
  ];
}

// ---- story values ---------------------------------------------------------------

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const vm = mapFor(CLEAN, FILTERS.vertical);
  const vhi = extremes(vm, 'max');
  const vlo = extremes(vm, 'min');
  v['out-10'] = String(convOutSize(N, F, 1, 0));
  v['v-max'] = fmtV(vhi.best);
  v['v-max-at'] = placeText(vhi.cells);
  v['v-min'] = fmtV(vlo.best);
  v['v-min-at'] = placeText(vlo.cells);
  v['v-line'] = fmtV(awayFromSquare(vm));
  const hm = mapFor(CLEAN, FILTERS.horizontal);
  const hhi = extremes(hm, 'max');
  v['h-max'] = fmtV(hhi.best);
  v['h-max-at'] = placeText(hhi.cells);
  const hlo = extremes(hm, 'min');
  v['h-min'] = fmtV(hlo.best);
  v['h-min-at'] = placeText(hlo.cells);
  const bm = mapFor(CLEAN, FILTERS.blur);
  v['blur-in'] = fmtV(bm[2][2]);
  v['blur-line'] = fmtV(awayFromSquare(bm));
  const nm = mapFor(NOISY, FILTERS.vertical);
  const nhi = extremes(nm, 'max');
  v['n-max'] = fmtV(nhi.best);
  v['n-max-at'] = placeText(nhi.cells);
  const cf = vm.flat();
  v['n-off'] = fmtV(Math.max(...nm.flat().filter((_, k) => cf[k] === 0).map(Math.abs)));
  const lv = lineVsSquare(mapFor(CLEAN, TRY_LINE));
  v['try-line'] = fmtV(lv.line);
  v['try-square'] = fmtV(lv.square);
  for (const s of STRIDES) for (const p of PADS) v[`size-s${s}p${p}`] = String(sizeInfo(s, p).formula);
  v['dense'] = fmtCount(denseCount());
  v['shared'] = String(sharedCount());
  const sm = extremes(mapFor(shiftRight(CLEAN, 2), FILTERS.vertical), 'max');
  v['shift-at'] = placeText(sm.cells);
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

// Image on the left, filter in the middle, map (or products, or a zoom) on the right.
const C = 15;
const IX = 30;
const IY = 44;
const FC = 46;
const FX = 234;
const FY = IY + (N * C - 3 * FC) / 2;
const RX = 386;
const RY = IY;
const PW = 64;
const PH = 40;
const ZC = 38;
const TY = 286;
const P = (v) => Number(v.toFixed(2));

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let pix = [3, 3];
  let pos = [2, 0];
  let filter = 'vertical';
  let image = 'clean';
  let weights = FILTERS.vertical.map((row) => row.slice());
  let stride = 1;
  let pad = 0;
  let shift = 0;
  let filled = null;
  let token = 0;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A 12 by 12 greyscale image, a 3 by 3 filter of weights, and the feature map the filter gives as it slides over the image. The current window and its numbers are listed in a table after the picture.');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start') => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': 12, text, ...HALO }, parent);
    t.style.fill = MUTED;
    return t;
  };

  const state = () => ({ step, cursor: step === 0 ? pix : pos, filled: filled === null ? undefined : filled, filter, weights, image, stride, pad, shift });
  const current = () => resolve(state());

  // ---- the image, with its padding ring and the cursor / window.
  const imgG = el('g', {}, svg);
  const imgLabel = label(imgG, IX, 18, '');
  const ringG = el('g', {}, imgG);
  for (let r = -1; r <= N; r++) {
    for (let c = -1; c <= N; c++) {
      if (r >= 0 && r < N && c >= 0 && c < N) continue;
      el('rect', { x: IX + c * C + 1, y: IY + r * C + 1, width: C - 2, height: C - 2, fill: 'none', stroke: MUTED, 'stroke-width': 0.75, 'stroke-dasharray': '2 2' }, ringG);
    }
  }
  const imgGrid = pixelGrid(imgG, CLEAN, { x: IX, y: IY, cell: C, mode: 'grey' });
  const zoomBox = el('rect', { fill: 'none', stroke: MUTED, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, imgG);
  const pixBox = el('rect', { width: C, height: C, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, imgG);
  const winBox = el('rect', { width: 3 * C, height: 3 * C, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, imgG);

  // ---- the filter: weights printed in each cell; colour and the printed sign agree.
  const filG = el('g', {}, svg);
  label(filG, FX + 1.5 * FC, 18, 'filter W', 'middle');
  const filRects = [];
  const filTexts = [];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      filRects.push(el('rect', { x: FX + b * FC, y: FY + a * FC, width: FC, height: FC, stroke: LINE }, filG));
      const t = el('text', { class: 'stage-text', x: FX + (b + 0.5) * FC, y: FY + (a + 0.5) * FC + 4.5, 'text-anchor': 'middle', 'font-size': 13, ...HALO }, filG);
      t.style.fill = STRONG;
      filTexts.push(t);
    }
  }
  el('rect', { x: FX, y: FY, width: 3 * FC, height: 3 * FC, fill: 'none', stroke: MUTED }, filG);

  // ---- step 1: the numbers around the cursor, 5 × 5.
  const zoomG = el('g', {}, svg);
  label(zoomG, RX, 18, 'the numbers there');
  const zoomRects = [];
  const zoomTexts = [];
  for (let a = 0; a < 5; a++) {
    for (let b = 0; b < 5; b++) {
      zoomRects.push(el('rect', { x: RX + b * ZC, y: RY + a * ZC, width: ZC, height: ZC, stroke: LINE }, zoomG));
      const t = el('text', { class: 'stage-text', x: RX + (b + 0.5) * ZC, y: RY + (a + 0.5) * ZC + 4.5, 'text-anchor': 'middle', 'font-size': 12, ...HALO }, zoomG);
      t.style.fill = STRONG;
      zoomTexts.push(t);
    }
  }
  const zoomCur = el('rect', { width: ZC, height: ZC, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, zoomG);

  // ---- step 2: pixel × weight for the nine cells, and the sum.
  const prodG = el('g', {}, svg);
  label(prodG, RX, 18, 'pixel × weight');
  const prodTop = [];
  const prodBot = [];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      el('rect', { x: RX + b * PW, y: RY + a * PH, width: PW, height: PH, fill: SURFACE, stroke: LINE }, prodG);
      const t1 = el('text', { class: 'stage-text', x: RX + (b + 0.5) * PW, y: RY + a * PH + 16, 'text-anchor': 'middle', 'font-size': 12 }, prodG);
      t1.style.fill = MUTED;
      const t2 = el('text', { class: 'stage-text', x: RX + (b + 0.5) * PW, y: RY + a * PH + 34, 'text-anchor': 'middle', 'font-size': 12 }, prodG);
      t2.style.fill = STRONG;
      prodTop.push(t1);
      prodBot.push(t2);
    }
  }
  const sumT = el('text', { class: 'stage-text', x: RX, y: RY + 3 * PH + 30, 'font-size': 14 }, prodG);
  sumT.style.fill = CYAN;
  const sumT2 = label(prodG, RX, RY + 3 * PH + 54, '');

  // ---- steps 3 to 6: the feature map (signed: cyan positive, violet negative with a minus).
  const mapG = el('g', {}, svg);
  const mapLabel = label(mapG, RX, 18, '');
  const mapCells = el('g', { class: 'pixels' }, mapG);
  let mapM = 0;
  let mapRects = [];
  let mapMarks = [];
  let mapDots = [];
  let mapFrame = null;
  const mapCur = el('rect', { width: C, height: C, fill: 'none', stroke: CYAN, 'stroke-width': 2.5 }, mapG);
  function buildMap(m) {
    if (m === mapM) return;
    mapM = m;
    mapCells.replaceChildren();
    mapRects = [];
    mapMarks = [];
    mapDots = [];
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) mapRects.push(el('rect', { x: RX + j * C, y: RY + i * C, width: C, height: C }, mapCells));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        mapDots.push(el('rect', { x: RX + j * C + C / 2 - 1.5, y: RY + i * C + C / 2 - 1.5, width: 3, height: 3, fill: MUTED, stroke: 'none' }, mapCells));
        mapMarks.push(el('text', { class: 'px-sign', x: RX + (j + 0.5) * C, y: RY + (i + 0.5) * C, 'text-anchor': 'middle', text: '−' }, mapCells));
      }
    }
    mapFrame = el('rect', { class: 'px-frame', x: RX, y: RY, width: m * C, height: m * C }, mapCells);
  }

  // ---- read-out lines under the pictures.
  const rowsT = [0, 1, 2].map((k) => {
    const t = el('text', { class: 'stage-text', x: IX, y: TY + 22 * k, 'font-size': 13 }, svg);
    t.style.fill = k === 0 ? STRONG : MUTED;
    return t;
  });

  // ---- the current window as a table, for screen readers.
  const box = svg.parentNode;
  // A div carries the hiding class: a table will not shrink below its content's width.
  const tableWrap = doc.createElement('div');
  tableWrap.className = 'vh';
  const table = doc.createElement('table');
  const caption = doc.createElement('caption');
  const tbody = doc.createElement('tbody');
  table.append(caption, tbody);
  tableWrap.append(table);
  box.appendChild(tableWrap);

  // ---- controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const groupBox = (text, aria) => {
    const g = make('div');
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', aria);
    Object.assign(g.style, { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
    if (text) {
      const tag = make('span', { textContent: text });
      tag.setAttribute('aria-hidden', 'true');
      Object.assign(tag.style, { color: MUTED, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', marginRight: '2px' });
      g.append(tag);
    }
    return g;
  };

  // Row and column of the cursor (step 1) or the window (steps 2 and 3).
  const posGroup = groupBox('', 'window position');
  const slider = (name) => {
    const lab = make('label');
    const input = make('input', { type: 'range', min: '0', max: '11', step: '1', value: '0' });
    const out = make('output', { textContent: '0' });
    lab.append(make('span', { textContent: name }), input, out);
    posGroup.append(lab);
    return { input, out };
  };
  const rowS = slider('row');
  const colS = slider('column');
  const replay = make('button', { type: 'button', textContent: 'Slide again' });
  posGroup.append(replay);

  const filtGroup = groupBox('', 'filter and image');
  const select = (name, options) => {
    const lab = make('label');
    const sel = make('select');
    for (const [value, text] of options) sel.append(make('option', { value, textContent: text }));
    lab.append(make('span', { textContent: name }), sel);
    filtGroup.append(lab);
    return sel;
  };
  const filterSel = select('filter', Object.entries(FILTER_LABEL));
  const imageSel = select('image', [['clean', 'clean'], ['noisy', 'noisy']]);

  const sizeGroup = groupBox('', 'stride and padding');
  const btnRow = (text, aria, values, fmt) => {
    const g = groupBox(text, aria);
    const btns = values.map((v) => make('button', { type: 'button', textContent: fmt(v) }));
    g.append(...btns);
    sizeGroup.append(g);
    return btns;
  };
  const strideBtns = btnRow('stride', 'stride', STRIDES, (v) => `s = ${v}`);
  const padBtns = btnRow('padding', 'padding', PADS, (v) => `p = ${v}`);

  const shiftGroup = groupBox('shift right', 'shift the image right');
  const shiftBtns = SHIFTS.map((d) => make('button', { type: 'button', textContent: String(d) }));
  shiftGroup.append(...shiftBtns);

  controls.append(posGroup, filtGroup, sizeGroup, shiftGroup);

  // The editable filter sits under the picture, outside the controls bar.
  const editor = make('div');
  Object.assign(editor.style, { display: 'none', alignItems: 'center', flexWrap: 'wrap', gap: '8px 16px', marginTop: '10px', fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', color: MUTED });
  const edTitle = make('span', { textContent: 'your filter (−2 to 2, steps of 0.5)' });
  const edGrid = make('div');
  edGrid.setAttribute('role', 'group');
  edGrid.setAttribute('aria-label', 'your filter weights');
  Object.assign(edGrid.style, { display: 'grid', gridTemplateColumns: 'repeat(3, 4.25rem)', gap: '4px' });
  const inputs = [];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      const inp = make('input', { type: 'number', min: String(W_MIN), max: String(W_MAX), step: String(W_STEP), value: String(weights[a][b]) });
      inp.setAttribute('aria-label', `filter row ${a}, column ${b}`);
      Object.assign(inp.style, { width: '100%', height: '34px', boxSizing: 'border-box', padding: '0 4px', background: BG, color: STRONG, border: `1px solid ${LINE}`, borderRadius: '4px', font: 'inherit', fontSize: '14px', textAlign: 'center' });
      inputs.push(inp);
      edGrid.append(inp);
    }
  }
  const edBtns = make('div');
  Object.assign(edBtns.style, { display: 'flex', flexWrap: 'wrap', gap: '6px' });
  const tryLine = make('button', { type: 'button', textContent: 'Try the line filter' });
  const backV = make('button', { type: 'button', textContent: 'Back to vertical edge' });
  for (const btn of [tryLine, backV]) Object.assign(btn.style, { height: '34px', padding: '0 10px', border: `1px solid ${LINE}`, borderRadius: '4px', background: 'transparent', color: STRONG, font: 'inherit', cursor: 'pointer' });
  edBtns.append(tryLine, backV);
  editor.append(edTitle, edGrid, edBtns);
  svg.insertAdjacentElement('afterend', editor);

  // ---- drawing.
  function render() {
    const s = current();
    const sh = s.shift;
    show(imgG, true);
    show(ringG, step === 4 && pad === 1);
    show(filG, step >= 1);
    show(zoomG, step === 0);
    show(prodG, step === 1);
    show(mapG, step >= 2);
    show(pixBox, step === 0);
    show(zoomBox, step === 0);
    show(winBox, step >= 1);
    imgGrid.update(s.img);
    imgLabel.textContent = step === 5 && sh ? `X shifted right ${sh}` : step === 3 && image === 'noisy' ? 'image X, noisy copy' : `image X, ${N} × ${N}`;

    if (step === 0) {
      const [r, c] = pix;
      pixBox.setAttribute('x', IX + c * C);
      pixBox.setAttribute('y', IY + r * C);
      const r0 = Math.max(0, Math.min(N - 5, r - 2));
      const c0 = Math.max(0, Math.min(N - 5, c - 2));
      zoomBox.setAttribute('x', IX + c0 * C);
      zoomBox.setAttribute('y', IY + r0 * C);
      zoomBox.setAttribute('width', 5 * C);
      zoomBox.setAttribute('height', 5 * C);
      for (let a = 0; a < 5; a++) {
        for (let b = 0; b < 5; b++) {
          const v = s.img[r0 + a][c0 + b];
          const k = a * 5 + b;
          const f = pixelFill(v, 'grey');
          zoomRects[k].setAttribute('fill', `rgba(226, 232, 240, ${P(f.opacity)})`);
          zoomTexts[k].textContent = fmtV(v);
          // Dark digits on bright cells, light digits on dark ones.
          zoomTexts[k].style.fill = v > 0.55 ? BG : STRONG;
          zoomTexts[k].setAttribute('stroke-width', v > 0.55 ? 0 : 3);
        }
      }
      zoomCur.setAttribute('x', RX + (c - c0) * ZC);
      zoomCur.setAttribute('y', RY + (r - r0) * ZC);
    }

    if (step >= 1) {
      const scale = Math.max(...s.W.flat().map(Math.abs));
      s.W.flat().forEach((w, k) => {
        const f = pixelFill(w, { signed: scale });
        filRects[k].setAttribute('fill', w < 0 ? '#a78bfa' : CYAN);
        filRects[k].setAttribute('fill-opacity', P(0.15 + 0.55 * f.opacity));
        filTexts[k].textContent = fmtW(w);
      });
      const [i, j] = clampPos(s.m);
      winBox.setAttribute('x', IX + (j * s.stride - s.pad) * C);
      winBox.setAttribute('y', IY + (i * s.stride - s.pad) * C);
    }

    if (step === 1) {
      const [i, j] = clampPos(s.m);
      const w = windowCells(s.img, s.W, i, j);
      w.cells.forEach((k, n) => {
        prodTop[n].textContent = `${fmtV(k.x)}×${fmtV(k.w)}`;
        prodBot[n].textContent = `= ${fmtV(k.prod)}`;
      });
      sumT.textContent = `sum = ${fmtV(w.sum)}`;
      sumT2.textContent = `→ map row ${i}, column ${j}`;
    }

    if (step >= 2) {
      buildMap(s.m);
      mapLabel.textContent = `feature map, ${s.m} × ${s.m}`;
      const total = s.m * s.m;
      const upto = step === 2 && filled !== null ? filled : total;
      const scale = Math.max(...s.map.flat().map(Math.abs));
      for (let i = 0; i < s.m; i++) {
        for (let j = 0; j < s.m; j++) {
          const k = i * s.m + j;
          const on = k < upto;
          const v = s.map[i][j];
          const f = pixelFill(v, { signed: scale });
          mapRects[k].setAttribute('class', f.cls);
          mapRects[k].setAttribute('opacity', on ? P(f.opacity) : 0);
          mapDots[k].setAttribute('visibility', on ? 'hidden' : 'visible');
          mapMarks[k].setAttribute('visibility', on && minusShown(v) ? 'visible' : 'hidden');
        }
      }
      const [i, j] = clampPos(s.m);
      mapCur.setAttribute('x', RX + j * C);
      mapCur.setAttribute('y', RY + i * C);
    }

    const rows = readout(state());
    rowsT.forEach((t, k) => { t.textContent = rows[k] || ''; });
    fillTable(s);

    strideBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(STRIDES[k] === stride)));
    padBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(PADS[k] === pad)));
    shiftBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(SHIFTS[k] === shift)));
    syncSliders(s);
  }

  function clampPos(m) {
    pos = [Math.min(pos[0], m - 1), Math.min(pos[1], m - 1)];
    return pos;
  }

  function fillTable(s) {
    tbody.replaceChildren();
    const row = (cells, head = false) => {
      const tr = make('tr');
      for (const c of cells) tr.append(make(head ? 'th' : 'td', { textContent: c }));
      tbody.append(tr);
    };
    if (step === 0) {
      const [r, c] = pix;
      caption.textContent = `Pixel row ${r}, column ${c} has value ${fmtV(s.img[r][c])}. Its row of the image, column by column:`;
      row(Array.from({ length: N }, (_, k) => `column ${k}`), true);
      row(s.img[r].map(fmtV));
      return;
    }
    const [i, j] = clampPos(s.m);
    const w = windowCells(s.img, s.W, i, j, { stride: s.stride, pad: s.pad });
    caption.textContent = `Window for map row ${i}, column ${j}: nine pixels times nine weights, then the sum.`;
    row(['image row', 'image column', 'pixel', 'weight', 'product'], true);
    for (const k of w.cells) row([String(k.r), String(k.c), k.pad ? '0 (padding)' : fmtV(k.x), fmtW(k.w), fmtV(k.prod)]);
    row(['', '', '', 'sum', fmtV(w.sum)]);
  }

  function syncSliders(s) {
    const max = step === 0 ? N - 1 : s.m - 1;
    const [a, b] = step === 0 ? pix : pos;
    for (const [sl, v] of [[rowS, a], [colS, b]]) {
      sl.input.max = String(max);
      sl.input.value = String(v);
      sl.out.textContent = String(v);
    }
  }

  const spoken = () => `${readout(state()).join('. ')}.`;

  // ---- interaction.
  const setCursor = (a, b) => {
    if (step === 0) pix = [a, b];
    else pos = [a, b];
    render();
  };
  for (const sl of [rowS, colS]) {
    sl.input.addEventListener('input', () => {
      if (filled !== null) stopSweep();
      const a = Number(rowS.input.value) || 0;
      const b = Number(colS.input.value) || 0;
      setCursor(a, b);
      announce(spoken());
    });
  }

  // Pointer: hover (mouse) or tap moves the cursor or the window.
  const toUser = (e) => {
    const m = svg.getScreenCTM();
    if (!m) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(m.inverse());
  };
  const pick = (e) => {
    const u = toUser(e);
    if (!u) return false;
    const s = current();
    const ci = Math.floor((u.y - IY) / C);
    const cj = Math.floor((u.x - IX) / C);
    if (step === 0) {
      if (ci < 0 || ci >= N || cj < 0 || cj >= N) return false;
      setCursor(ci, cj);
      return true;
    }
    if (step >= 2) {
      const mi = Math.floor((u.y - RY) / C);
      const mj = Math.floor((u.x - RX) / C);
      if (mi >= 0 && mi < s.m && mj >= 0 && mj < s.m) {
        if (filled !== null) stopSweep();
        setCursor(mi, mj);
        return true;
      }
    }
    // Over the image: the window whose centre is nearest that pixel.
    if (ci < -1 || ci > N || cj < -1 || cj > N) return false;
    const near = (k) => Math.max(0, Math.min(s.m - 1, Math.round((k + s.pad - 1) / s.stride)));
    if (filled !== null) stopSweep();
    setCursor(near(ci), near(cj));
    return true;
  };
  svg.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') pick(e); });
  svg.addEventListener('pointerdown', (e) => { if (pick(e)) announce(spoken()); });

  filterSel.addEventListener('change', () => {
    filter = filterSel.value in FILTER_LABEL ? filterSel.value : 'vertical';
    syncEditor();
    render();
    announce(spoken());
  });
  imageSel.addEventListener('change', () => {
    image = imageSel.value === 'noisy' ? 'noisy' : 'clean';
    render();
    announce(spoken());
  });
  strideBtns.forEach((b, k) => b.addEventListener('click', () => { stride = STRIDES[k]; render(); announce(spoken()); }));
  padBtns.forEach((b, k) => b.addEventListener('click', () => { pad = PADS[k]; render(); announce(spoken()); }));
  shiftBtns.forEach((b, k) => b.addEventListener('click', () => { shift = SHIFTS[k]; render(); announce(spoken()); }));

  // Typed weights: parsed, clamped and snapped in code; the box shows the value used once you leave it.
  const readInputs = () => {
    weights = [0, 1, 2].map((a) => [0, 1, 2].map((b) => parseWeight(inputs[a * 3 + b].value)));
  };
  const writeInputs = () => { inputs.forEach((inp, k) => { inp.value = String(weights[Math.floor(k / 3)][k % 3]); }); };
  inputs.forEach((inp) => {
    inp.addEventListener('input', () => { readInputs(); render(); });
    inp.addEventListener('change', () => { readInputs(); writeInputs(); render(); announce(spoken()); });
  });
  tryLine.addEventListener('click', () => { weights = TRY_LINE.map((r) => r.slice()); writeInputs(); render(); announce(spoken()); });
  backV.addEventListener('click', () => { weights = FILTERS.vertical.map((r) => r.slice()); writeInputs(); render(); announce(spoken()); });
  const syncEditor = () => { editor.style.display = step === 3 && filter === 'custom' ? 'flex' : 'none'; };

  // ---- step 3: the window slides over every position and the map fills in.
  function stopSweep() {
    token++;
    filled = null;
  }
  function sweep(animate) {
    const my = ++token;
    const m = convOutSize(N, F);
    const total = m * m;
    if (!animate || reduced) {
      filled = null;
      render();
      return Promise.resolve();
    }
    filled = 0;
    pos = [0, 0];
    render();
    return tween(3000, (t) => {
      if (my !== token) return;
      const k = Math.min(total, Math.floor(t * total) + 1);
      filled = k;
      pos = [Math.floor((k - 1) / m), (k - 1) % m];
      render();
    }, { reduced: false }).then(() => {
      if (my !== token) return;
      filled = null;
      render();
    });
  }
  replay.addEventListener('click', () => { sweep(true).then(() => announce(spoken())); });

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      token++;
      filled = null;
      posGroup.style.display = 'flex';
      replay.style.display = i === 2 ? '' : 'none';
      filtGroup.style.display = i === 3 ? 'flex' : 'none';
      sizeGroup.style.display = i === 4 ? 'flex' : 'none';
      shiftGroup.style.display = i === 5 ? 'flex' : 'none';
      syncEditor();
      if (i === 1) pos = [2, 0];
      if (i === 2) return sweep(animate).then(() => announce(spoken()));
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
