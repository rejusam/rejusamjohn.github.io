// Lesson 5, "Slopes and derivatives": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here.

import { linearScale, clamp, el, pathFrom, axes, fmt as tickFmt, tween } from '../assets/plot.js';

// The curve f(x) = A·x² + B·x + C, drawn on [X_LO, X_HI].
export const A = 0.5;
export const B = -2;
export const C = 3;
export const X_LO = -1;
export const X_HI = 6;
export const Y_LO = -1;
export const Y_HI = 10;
// Where the x slider starts (steps 2, 4 and 5).
export const X_START = 0.5;
// Step 3: the secant from x = X_SECANT to x + h, with h on [H_MIN, H_MAX].
// h never reaches 0, where the secant slope would divide by zero.
export const X_SECANT = 3;
export const H_MIN = 0.01;
export const H_MAX = 2;
export const H_START = 2;

export const f = (x) => A * x * x + B * x + C;
// By the power rule: f′(x) = 2A·x + B, which is x − 2 here.
export const fPrime = (x) => 2 * A * x + B;

// The slope of the straight line through (x, f(x)) and (x + h, f(x + h)).
export const secantSlope = (x, h) => (f(x + h) - f(x)) / h;

// The bottom of the bowl, where f′(x) = 0 (A > 0, so it is a minimum).
export function vertex() {
  const x = -B / (2 * A);
  return { x, y: f(x) };
}

// Step 1: a straight line y = a + b·x and two triangles on it.
export const LINE = { a: 0.5, b: 1.5 };
export const BIG = [[2, 3.5], [6, 9.5]];
export const SMALL = [[0, 0.5], [1, 2]];
export const slopeBetween = ([x1, y1], [x2, y2]) => (y2 - y1) / (x2 - x1);

// Which way is downhill for a given slope. Slopes that round to 0.00 count as flat.
export function downhill(slope, tol = 0.005) {
  if (Math.abs(slope) < tol) return 'flat';
  return slope < 0 ? 'right' : 'left';
}

// The downhill arrow drawn along the curve from x: it starts a little way
// from the point and stops at or before the bottom, so it only ever runs
// downhill. Null when the slope is flat or the bottom is too close to draw it.
export const ARROW_GAP = 0.12;
export const ARROW_MAX = 0.95;
export const ARROW_MIN = 0.15;
export function downhillArrow(x, n = 21) {
  const d = downhill(fPrime(x));
  if (d === 'flat') return null;
  const dir = d === 'right' ? 1 : -1;
  const len = Math.min(ARROW_MAX, Math.abs(vertex().x - x) - ARROW_GAP);
  if (len < ARROW_MIN) return null;
  const pts = Array.from({ length: n }, (_, j) => {
    const t = x + dir * (ARROW_GAP + (j / (n - 1)) * len);
    return [t, f(t)];
  });
  return { dir, pts };
}

// Fixed decimals with a true minus sign.
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

const DIRECTION_ROW = { right: 'negative: go right', left: 'positive: go left', flat: 'zero: flat here' };

// The read-out rows for each step, as shown on the stage.
export function readout(step, { x, h }) {
  if (step === 0) {
    const [[x1, y1], [x2, y2]] = BIG;
    const [[s1, t1], [s2, t2]] = SMALL;
    return [
      `rise = ${tickFmt(y2)} − ${tickFmt(y1)} = ${tickFmt(y2 - y1)}`,
      `run = ${tickFmt(x2)} − ${tickFmt(x1)} = ${tickFmt(x2 - x1)}`,
      `slope = ${tickFmt(y2 - y1)}/${tickFmt(x2 - x1)} = ${tickFmt(slopeBetween(...BIG))}`,
      `small: ${tickFmt(t2 - t1)}/${tickFmt(s2 - s1)} = ${tickFmt(slopeBetween(...SMALL))}`,
    ];
  }
  if (step === 1) return [`x = ${fmt(x)}`, `f(x) = ${fmt(f(x))}`, `slope = ${fmt(fPrime(x))}`];
  if (step === 2) {
    const sec = secantSlope(X_SECANT, h);
    const tan = fPrime(X_SECANT);
    return [`x = ${tickFmt(X_SECANT)}, h = ${fmt(h, 3)}`, `secant: ${fmt(sec, 3)}`, `tangent: ${fmt(tan, 3)}`, `gap: ${fmt(sec - tan, 3)}`];
  }
  if (step === 3) return [`x = ${fmt(x)}`, `slope = ${fmt(fPrime(x))}`, DIRECTION_ROW[downhill(fPrime(x))]];
  const d = downhill(fPrime(x));
  return [`x = ${fmt(x)}`, `f(x) = ${fmt(f(x))}`, `slope = ${fmt(fPrime(x))}`, d === 'flat' ? 'the lowest point' : DIRECTION_ROW[d]];
}

// Words for numbers in the live region.
const say2 = (v, dp = 2) => {
  const s = fmt(v, dp);
  return s.startsWith('−') ? `minus ${s.slice(1)}` : s;
};

// ---------------------------------------------------------------- the stage

const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE_C = '#334155';
const HALO = { stroke: '#1e293b', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

const PX0 = 56;
const PX1 = 330;
const PY0 = 290;
const PY1 = 20;
const SX0 = 386;
const SX1 = 588;
const SY0 = 290;
const SY1 = 132;
const BOX = { x: 362, y: 8, row: 20 };
const R = 5;
// Half the length of the short tangent drawn at the point, in units of x.
const TAN_HALF = 1.1;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const v = vertex();
  let x = X_START;
  let h = H_START;

  const X = linearScale([X_LO, X_HI], [PX0, PX1]);
  const Y = linearScale([Y_LO, Y_HI], [PY0, PY1]);
  const g = (t) => LINE.a + LINE.b * t;

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'A plot of a straight line that bends into a U-shaped curve, with a point, its slope line, and a small chart of the slope beside it.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l5-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);
  const marker = el('marker', { id: 'l5-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 z', fill: STRONG }, marker);

  axes(svg, X, Y, { xLabel: 'x', yLabel: 'y' });

  // --- Main plot.
  const plotG = el('g', { 'clip-path': 'url(#l5-plot)' }, svg);
  const xsFine = Array.from({ length: 141 }, (_, j) => X_LO + (j / 140) * (X_HI - X_LO));
  const curve = el('path', { fill: 'none', stroke: INK, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, plotG);

  // Step 1: two rise-over-run triangles on the straight line.
  const triG = el('g', {}, svg);
  const triangle = (pts, labels) => {
    const [[x1, y1], [x2, y2]] = pts;
    el('path', { d: `M${X(x1)},${Y(y1)} L${X(x2)},${Y(y1)} L${X(x2)},${Y(y2)}`, fill: 'none', stroke: CYAN, 'stroke-width': 1.8, 'stroke-dasharray': '5 3' }, triG);
    el('text', { class: 'stage-text', x: (X(x1) + X(x2)) / 2, y: Y(y1) + 15, 'text-anchor': 'middle', text: `${labels[0]} ${tickFmt(x2 - x1)}`, ...HALO }, triG);
    const inside = labels[2];
    el('text', {
      class: 'stage-text', x: X(x2) + (inside ? -6 : 6), y: (Y(y1) + Y(y2)) / 2 + 4,
      'text-anchor': inside ? 'end' : 'start', text: `${labels[1]} ${tickFmt(y2 - y1)}`, ...HALO,
    }, triG);
    for (const [px, py] of pts) el('circle', { cx: X(px), cy: Y(py), r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, triG);
  };
  triangle(BIG, ['run', 'rise', true]);
  triangle(SMALL, ['run', 'rise', false]);

  // Steps 2, 4, 5: the point and its short tangent.
  const tanG = el('g', {}, svg);
  const tanLine = el('line', { stroke: CYAN, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, tanG);

  // Step 4: an arrow along the curve, pointing downhill.
  const downG = el('g', {}, svg);
  const arrow = el('path', { fill: 'none', stroke: STRONG, 'stroke-width': 2.2, 'marker-end': 'url(#l5-arrow)' }, downG);
  const arrowLabel = el('text', { class: 'stage-text', 'text-anchor': 'middle', text: 'downhill', ...HALO }, downG);
  arrowLabel.style.fill = STRONG;

  // Step 3: the secant through x and x + h, and the tangent it closes in on.
  const secG = el('g', {}, svg);
  const secClip = el('g', { 'clip-path': 'url(#l5-plot)' }, secG);
  const tanFull = el('line', { stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, secClip);
  const secLine = el('line', { stroke: CYAN, 'stroke-width': 2.2 }, secClip);
  const runLeg = el('line', { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, secG);
  const riseLeg = el('line', { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, secG);
  const hLabel = el('text', { class: 'stage-text', 'text-anchor': 'middle', text: 'h', ...HALO }, secG);
  const qDot = el('rect', { width: 10, height: 10, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, secG);
  const pLabel = el('text', { class: 'stage-text', 'text-anchor': 'end', text: 'x', ...HALO }, secG);
  const qLabel = el('text', { class: 'stage-text', 'text-anchor': 'start', text: 'x + h', ...HALO }, secG);
  const secName = el('text', { class: 'stage-text', 'text-anchor': 'end', text: 'secant', ...HALO }, secG);
  const tanName = el('text', { class: 'stage-text', 'text-anchor': 'end', text: 'tangent', ...HALO }, secG);
  secName.style.fill = CYAN;
  tanName.style.fill = VIOLET;

  // Step 5: the bottom of the bowl.
  const bottomG = el('g', {}, svg);
  el('text', {
    class: 'stage-text', x: X(v.x), y: Y(v.y) + 22, 'text-anchor': 'middle',
    text: `lowest point (${tickFmt(v.x)}, ${tickFmt(v.y)})`, ...HALO,
  }, bottomG).style.fill = STRONG;

  const pDot = el('circle', { r: R + 0.5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, svg);

  // --- Side panel: the slope at each x (steps 1, 2, 4, 5).
  const SXs = linearScale([X_LO, X_HI], [SX0, SX1]);
  const SYs = linearScale([-3, 4], [SY0, SY1]);
  const slopeG = el('g', {}, svg);
  const bands = el('g', {}, slopeG);
  el('rect', { x: SX0, y: SY1, width: SXs(v.x) - SX0, height: SY0 - SY1, fill: CYAN, 'fill-opacity': 0.06 }, bands);
  el('line', { x1: SXs(v.x), x2: SXs(v.x), y1: SY1, y2: SY0, stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '2 3' }, bands);
  el('text', { class: 'stage-text', x: SX0 + 6, y: SYs(3.1), text: 'slope < 0' }, bands);
  el('text', { class: 'stage-text', x: SX0 + 6, y: SYs(3.1) + 16, text: 'go right →' }, bands).style.fill = STRONG;
  el('text', { class: 'stage-text', x: SXs(4.4), y: SYs(-1.6), 'text-anchor': 'middle', text: 'slope > 0' }, bands);
  el('text', { class: 'stage-text', x: SXs(4.4), y: SYs(-1.6) + 16, 'text-anchor': 'middle', text: '← go left' }, bands).style.fill = STRONG;
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE_C }, slopeG);
  el('line', { x1: SX0, x2: SX1, y1: SYs(0), y2: SYs(0), stroke: MUTED, 'stroke-width': 1 }, slopeG);
  for (const t of [-2, 0, 2, 4]) {
    el('line', { x1: SX0 - 4, x2: SX0, y1: SYs(t), y2: SYs(t), stroke: LINE_C }, slopeG);
    el('text', { class: 'stage-text', x: SX0 - 8, y: SYs(t) + 4, 'text-anchor': 'end', text: tickFmt(t) }, slopeG);
  }
  for (const t of [0, 2, 4, 6]) {
    el('line', { x1: SXs(t), x2: SXs(t), y1: SY0, y2: SY0 + 4, stroke: LINE_C }, slopeG);
    el('text', { class: 'stage-text', x: SXs(t), y: SY0 + 17, 'text-anchor': 'middle', text: tickFmt(t) }, slopeG);
  }
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 35, 'text-anchor': 'middle', text: 'x' }, slopeG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'slope at each x' }, slopeG);
  const slopePath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2.2 }, slopeG);
  const slopeDot = el('circle', { r: 5, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, slopeG);
  const zeroLabel = el('text', { class: 'stage-text', x: SXs(v.x) + 8, y: SYs(0) + 16, text: 'slope 0', ...HALO }, slopeG);
  zeroLabel.style.fill = STRONG;

  // --- Side panel, step 3: the secant slope for each h.
  const HX = linearScale([0, H_MAX], [SX0, SX1]);
  const tan0 = fPrime(X_SECANT);
  const HY = linearScale([0, 3], [SY0, SY1]);
  const hG = el('g', {}, svg);
  el('line', { x1: SX0, x2: SX1, y1: SY0, y2: SY0, stroke: LINE_C }, hG);
  el('line', { x1: SX0, x2: SX0, y1: SY0, y2: SY1, stroke: LINE_C }, hG);
  for (const t of [0, 1, 2, 3]) {
    el('line', { x1: SX0 - 4, x2: SX0, y1: HY(t), y2: HY(t), stroke: LINE_C }, hG);
    el('text', { class: 'stage-text', x: SX0 - 8, y: HY(t) + 4, 'text-anchor': 'end', text: tickFmt(t) }, hG);
  }
  for (const t of [0, 0.5, 1, 1.5, 2]) {
    el('line', { x1: HX(t), x2: HX(t), y1: SY0, y2: SY0 + 4, stroke: LINE_C }, hG);
    if (Number.isInteger(t)) el('text', { class: 'stage-text', x: HX(t), y: SY0 + 17, 'text-anchor': 'middle', text: tickFmt(t) }, hG);
  }
  el('text', { class: 'stage-text', x: (SX0 + SX1) / 2, y: SY0 + 35, 'text-anchor': 'middle', text: 'h' }, hG);
  el('text', { class: 'value', x: SX0, y: SY1 - 10, text: 'secant slope for each h' }, hG);
  el('line', { x1: SX0, x2: SX1, y1: HY(tan0), y2: HY(tan0), stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, hG);
  el('text', { class: 'stage-text', x: SX1, y: HY(tan0) + 16, 'text-anchor': 'end', text: 'tangent slope' }, hG).style.fill = VIOLET;
  const hs = Array.from({ length: 81 }, (_, j) => H_MIN + (j / 80) * (H_MAX - H_MIN));
  el('path', { d: pathFrom(hs.map((t) => [t, secantSlope(X_SECANT, t)]), HX, HY), fill: 'none', stroke: CYAN, 'stroke-width': 2.2 }, hG);
  // h = 0 itself is left out: an open circle marks the value the secant closes in on.
  el('circle', { cx: HX(0), cy: HY(tan0), r: 4.5, fill: BG, stroke: CYAN, 'stroke-width': 1.8 }, hG);
  const hDot = el('rect', { width: 10, height: 10, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, hG);

  // --- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: '#1e293b', 'fill-opacity': 0.92, stroke: LINE_C }, box);
  const rows = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, hidden, min, max, step, value) => {
    const label = make('label');
    const input = make('input', { type: 'range', min, max, step, value });
    input.style.width = '130px';
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    label.append(name, make('span', { className: 'vh', textContent: hidden }), ' ', input, ' ', out);
    return { label, input, out };
  };
  const xs = slider('x', ', where the slope is measured', X_LO, X_HI, 0.01, X_START);
  const xGroup = make('div');
  xGroup.append(xs.label);
  const hs_ = slider('h', ', the gap between the two points', H_MIN, H_MAX, 0.01, H_START);
  const shrinkBtn = make('button', { type: 'button', textContent: 'Shrink h' });
  const resetBtn = make('button', { type: 'button', textContent: 'Reset h' });
  const hGroup = make('div');
  hGroup.append(hs_.label, shrinkBtn, resetBtn);
  controls.append(xGroup, hGroup);
  const showGroup = (grp, on) => { grp.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { u: 1, tri: 0, tan: 0, sec: 0, slope: 0, bands: 0, hchart: 0, down: 0, bottom: 0 };
  const target = (i) => {
    if (i === 0) return { ...LAYERS, u: 0, tri: 1, slope: 1 };
    if (i === 1) return { ...LAYERS, tan: 1, slope: 1 };
    if (i === 2) return { ...LAYERS, sec: 1, hchart: 1 };
    if (i === 3) return { ...LAYERS, tan: 1, slope: 1, bands: 1, down: 1 };
    return { ...LAYERS, tan: 1, slope: 1, bottom: 1 };
  };
  let s = target(0);

  const op = (node, val) => {
    node.setAttribute('opacity', val.toFixed(3));
    node.style.display = val > 0.001 ? '' : 'none';
  };
  const set = (node, attrs) => {
    for (const [k, val] of Object.entries(attrs)) node.setAttribute(k, typeof val === 'number' ? val.toFixed(2) : val);
  };

  function render() {
    const u = s.u;
    const shape = (t) => (1 - u) * g(t) + u * f(t);
    curve.setAttribute('d', pathFrom(xsFine.map((t) => [t, shape(t)]), X, Y));
    op(triG, s.tri);

    // The point: on the secant step it sits at X_SECANT.
    const px = step === 2 ? X_SECANT : x;
    set(pDot, { cx: X(px), cy: Y(f(px)) });
    op(pDot, Math.max(s.tan, s.sec));

    // Short tangent at x.
    const m = fPrime(x);
    const x1 = Math.max(X_LO, x - TAN_HALF);
    const x2 = Math.min(X_HI, x + TAN_HALF);
    set(tanLine, { x1: X(x1), y1: Y(f(x) + m * (x1 - x)), x2: X(x2), y2: Y(f(x) + m * (x2 - x)) });
    op(tanG, s.tan);

    // Downhill arrow along the curve, drawn just above it.
    const d = downhill(m);
    const arc = downhillArrow(x);
    if (d === 'flat') {
      arrow.setAttribute('d', '');
      arrowLabel.textContent = 'flat';
      arrowLabel.setAttribute('text-anchor', 'middle');
      set(arrowLabel, { x: X(x), y: Y(f(x)) + 24 });
    } else if (!arc) {
      // Too close to the bottom to draw; the box still gives the direction.
      arrow.setAttribute('d', '');
      arrowLabel.textContent = '';
    } else {
      const pts = arc.pts.map(([t, y]) => [t, y + 0.45]);
      arrow.setAttribute('d', pathFrom(pts, X, Y));
      const end = pts[pts.length - 1];
      arrowLabel.textContent = 'downhill';
      // Just beyond the arrowhead, on the side it points to.
      arrowLabel.setAttribute('text-anchor', arc.dir > 0 ? 'start' : 'end');
      set(arrowLabel, { x: X(end[0]) + arc.dir * 10, y: Y(end[1]) - 9 });
    }
    op(downG, s.down);

    // Secant (step 3).
    const p = [X_SECANT, f(X_SECANT)];
    const q = [X_SECANT + h, f(X_SECANT + h)];
    const sec = secantSlope(X_SECANT, h);
    set(secLine, { x1: X(X_LO), y1: Y(p[1] + sec * (X_LO - p[0])), x2: X(X_HI), y2: Y(p[1] + sec * (X_HI - p[0])) });
    set(tanFull, { x1: X(X_LO), y1: Y(p[1] + tan0 * (X_LO - p[0])), x2: X(X_HI), y2: Y(p[1] + tan0 * (X_HI - p[0])) });
    set(runLeg, { x1: X(p[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(p[1]) });
    set(riseLeg, { x1: X(q[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(q[1]) });
    set(hLabel, { x: (X(p[0]) + X(q[0])) / 2, y: Y(p[1]) + 15 });
    hLabel.style.display = X(q[0]) - X(p[0]) > 14 ? '' : 'none';
    set(qDot, { x: X(q[0]) - 5, y: Y(q[1]) - 5 });
    set(pLabel, { x: X(p[0]) - 8, y: Y(p[1]) - 8 });
    set(qLabel, { x: X(q[0]) + 9, y: Y(q[1]) + 4 });
    qLabel.style.display = h >= 0.6 ? '' : 'none';
    // Names near the right edge: the secant is always above the tangent there
    // (its slope is larger by h/2), so one name goes above and one below.
    const yAt = (slope, t) => p[1] + slope * (t - p[0]);
    const nameX = X_HI - 0.1;
    set(secName, { x: X(nameX), y: Y(yAt(sec, nameX)) - 8 });
    set(tanName, { x: X(nameX), y: Y(yAt(tan0, nameX)) + 16 });
    op(secG, s.sec);
    set(hDot, { x: HX(h) - 5, y: HY(sec) - 5 });
    op(hG, s.hchart);

    // Slope chart: flat for the straight line, x − 2 for the curve.
    const slopeAt = (t) => (1 - u) * LINE.b + u * fPrime(t);
    slopePath.setAttribute('d', pathFrom(xsFine.map((t) => [t, slopeAt(t)]), SXs, SYs));
    set(slopeDot, { cx: SXs(x), cy: SYs(clamp(fPrime(x), -3, 4)) });
    op(slopeDot, s.tan);
    op(bands, s.bands);
    op(zeroLabel, s.bottom);
    op(slopeG, s.slope);
    op(bottomG, s.bottom);

    // Read-out box.
    const texts = readout(step, { x, h });
    rows.forEach((r, k) => {
      r.textContent = texts[k] || '';
      r.style.display = k < texts.length ? '' : 'none';
    });
    let w = 196;
    try {
      const lens = rows.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(BOX.x, 598 - w);
    set(boxRect, { x: bx, y: BOX.y, width: w, height: 16 + BOX.row * texts.length });
    rows.forEach((r, k) => set(r, { x: bx + 12, y: BOX.y + 8 + BOX.row * k + BOX.row / 2 + 5 }));

    xs.out.textContent = fmt(x);
    xs.input.setAttribute('aria-valuetext', `x = ${say2(x)}, slope ${say2(fPrime(x))}`);
    hs_.out.textContent = fmt(h);
    hs_.input.setAttribute('aria-valuetext', `h = ${say2(h)}, secant slope ${say2(secantSlope(X_SECANT, h), 3)}`);
  }

  function animateTo(to, ms) {
    const my = ++token;
    const from = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = from[key] + (to[key] - from[key]) * kt;
      s = next;
      render();
    }, { reduced });
  }

  // --- Words for the live region.
  const DIR_WORDS = { right: 'negative, so downhill is to the right', left: 'positive, so downhill is to the left', flat: 'flat here, at the bottom of the bowl' };
  const spoken = (i) => {
    if (i === 0) return `A straight line. Rise ${tickFmt(BIG[1][1] - BIG[0][1])} over run ${tickFmt(BIG[1][0] - BIG[0][0])} gives slope ${tickFmt(slopeBetween(...BIG))}. The small triangle gives the same slope.`;
    if (i === 1) return `At x ${say2(x)}, the curve is at ${say2(f(x))} and its slope is ${say2(fPrime(x))}.`;
    if (i === 2) {
      const sec = secantSlope(X_SECANT, h);
      return `h ${say2(h, 3)}. Secant slope ${say2(sec, 3)}. Tangent slope ${say2(tan0, 3)}. Gap ${say2(sec - tan0, 3)}.`;
    }
    return `At x ${say2(x)}, the slope is ${say2(fPrime(x))}: ${DIR_WORDS[downhill(fPrime(x))]}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction: x.
  let xToken = 0;
  const setX = (val, { fromSlider = false } = {}) => {
    x = clamp(val, X_LO, X_HI);
    if (!fromSlider) xs.input.value = String(x);
    render();
  };
  xs.input.addEventListener('input', () => {
    xToken++;
    setX(Number(xs.input.value), { fromSlider: true });
    say(spoken(step));
  });
  const animateX = (to) => {
    const my = ++xToken;
    const from = x;
    return tween(reduced ? 0 : 900, (kt) => {
      if (my !== xToken) return;
      setX(from + (to - from) * kt);
    }, { reduced }).then(() => {
      if (my !== xToken) return;
      setX(to);
    });
  };

  // --- Interaction: h. Shrinking runs on a log scale so the small values get time on screen.
  let hToken = 0;
  const setH = (val, { fromSlider = false } = {}) => {
    h = clamp(val, H_MIN, H_MAX);
    if (!fromSlider) hs_.input.value = String(h);
    render();
  };
  hs_.input.addEventListener('input', () => {
    hToken++;
    setH(Number(hs_.input.value), { fromSlider: true });
    say(spoken(2));
  });
  const animateH = (to) => {
    const my = ++hToken;
    const a = Math.log(h);
    const b = Math.log(to);
    tween(reduced ? 0 : 2200, (kt) => {
      if (my !== hToken) return;
      setH(Math.exp(a + (b - a) * kt));
    }, { reduced }).then(() => {
      if (my !== hToken) return;
      setH(to);
      announce(spoken(2));
    });
  };
  shrinkBtn.addEventListener('click', () => animateH(H_MIN));
  resetBtn.addEventListener('click', () => animateH(H_START));

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 5,
    show(i, animate) {
      step = i;
      showGroup(xGroup, i === 1 || i === 3 || i === 4);
      showGroup(hGroup, i === 2);
      const to = target(i);
      let moved = Promise.resolve();
      // Step 5 is about the bottom, so x goes there; the slider can move it away again.
      // Step 4 is about direction, so a flat start (from step 5) goes back to the start value.
      if (i === 3 && downhill(fPrime(x)) === 'flat') {
        if (animate) moved = animateX(X_START);
        else {
          xToken++;
          setX(X_START);
        }
      }
      if (i === 4 && animate) moved = animateX(v.x);
      else if (i === 4) {
        xToken++;
        setX(v.x);
      }
      const done = () => announce(spoken(i));
      if (!animate) {
        token++;
        s = { ...to };
        render();
        done();
        return Promise.resolve();
      }
      return Promise.all([animateTo(to, 700), moved]).then(done);
    },
  };
}
