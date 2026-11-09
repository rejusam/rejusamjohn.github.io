// Lesson L3.4, "Support vector machines": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-support-vector-machines.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, uniform, dot } from '../assets/maths.js';
import { blobs2 } from '../assets/classify.js';
import { svmFit } from '../assets/svm.js';
import { linearScale, clamp, el, axes, tween, classMarker, draggable } from '../assets/plot.js';

export const SEED_SEP = 71;
export const SEED_OVER = 77;
export const SEED_CIRCLE = 73;
// Steps 3 and 4 solve with C = 1. On the separable points every α stays far
// below 1, so this is exactly the hard-margin answer (tested against C = 10⁶).
export const C_HARD = 1;
// The C slider: a 1-2-5 log scale from 0.01 to 100.
export const C_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
export const C_START = 6;
// Where the tilt and shift sliders start (degrees, data units).
export const START = { tilt: 120, shift: 0.5 };
// The two movable points in step 4: A is not a support vector, B is.
export const POINT_A = 20;
export const POINT_B = 15;
// The main plot's data window: 25 user units per data unit on both axes.
export const X_DOM = [-5, 5.8];
export const Y_DOM = [-4.8, 5.28];

const pm = (d) => ({ X: d.X, cls: d.y, y: d.y.map((c) => (c === 1 ? 1 : -1)) });
const BLOB = { n0: 12, n1: 12, mu0: [-2, -1.5], mu1: [2, 1.5] };
const SEP = pm(blobs2({ ...BLOB, sd: 0.8, seed: SEED_SEP }));
const OVER = pm(blobs2({ ...BLOB, sd: 1.4, seed: SEED_OVER }));
const copy = (d) => ({ X: d.X.map((r) => r.slice()), cls: d.cls.slice(), y: d.y.slice() });
export const sepData = () => copy(SEP);
export const overData = () => copy(OVER);

// 20 points inside radius 1 (class 0, y = −1), then 20 between radius 1.6 and 2.4
// (class 1, y = +1). For each point: r uniform, then the angle uniform on [0, 2π).
function makeCircles() {
  const rng = mulberry32(SEED_CIRCLE);
  const X = [];
  const cls = [];
  for (let i = 0; i < 40; i++) {
    const inner = i < 20;
    const r = inner ? uniform(rng, 0, 1) : uniform(rng, 1.6, 2.4);
    const t = uniform(rng, 0, 2 * Math.PI);
    X.push([r * Math.cos(t), r * Math.sin(t)]);
    cls.push(inner ? 0 : 1);
  }
  return { X, cls, y: cls.map((c) => (c === 1 ? 1 : -1)) };
}
const CIRC = makeCircles();
export const circleData = () => copy(CIRC);
export const lift = ([a, b]) => [a, b, a * a + b * b];

let FS = null;
const FO = new Map();
let FL = null;
let FC = null;
export const fitSep = () => (FS ??= svmFit(SEP.X, SEP.y, { C: C_HARD }));
export function fitOver(C) {
  if (!FO.has(C)) FO.set(C, svmFit(OVER.X, OVER.y, { C }));
  return FO.get(C);
}
export const fitLifted = () => (FL ??= svmFit(CIRC.X.map(lift), CIRC.y, { C: C_HARD }));
export const fitCircleLine = () => (FC ??= svmFit(CIRC.X, CIRC.y, { C: C_HARD }));

const margins = (f, X, y) => X.map((x, i) => y[i] * (dot(f.w, x) + f.b));
const TOL = 1e-6;

// Counts for a fit: margin violations (y f < 1: inside the street or on the
// wrong side), wrong side (y f < 0) and the total hinge Σ max(0, 1 − y f).
export function stats(f, X, y) {
  const m = margins(f, X, y);
  return {
    margin: f.margin,
    sv: f.sv.length,
    violations: m.filter((v) => v < 1 - TOL).length,
    wrong: m.filter((v) => v < 0).length,
    hinge: m.reduce((s, v) => s + Math.max(0, 1 - v), 0),
  };
}
export const overStats = (C) => stats(fitOver(C), OVER.X, OVER.y);
// The smallest C on the slider from which the answer no longer changes.
export function flatFrom() {
  const top = fitOver(C_GRID.at(-1));
  const same = (f) => [0, 1].every((j) => Math.abs(f.w[j] - top.w[j]) < TOL) && Math.abs(f.b - top.b) < TOL;
  let k = C_GRID.length - 1;
  while (k > 0 && same(fitOver(C_GRID[k - 1]))) k--;
  return C_GRID[k];
}
export const circleLineAcc = () => margins(fitCircleLine(), CIRC.X, CIRC.y).filter((v) => v > 0).length / CIRC.X.length;
export const liftedAcc = () => margins(fitLifted(), CIRC.X.map(lift), CIRC.y).filter((v) => v > 0).length / CIRC.X.length;

// A hand-placed line: tilt φ in degrees from the x₁ axis, shift c. Its unit
// normal is n = (sin φ, −cos φ) and the line is n·x = c. The side each class
// is on is whichever way round puts fewer points on the wrong side.
export function candidate(tilt, shift) {
  const rad = (tilt * Math.PI) / 180;
  const n = [Math.sin(rad), -Math.cos(rad)];
  const p = SEP.X.map((x) => dot(n, x) - shift);
  const wrongFor = (s) => p.filter((v, i) => s * SEP.y[i] * v <= 0).length;
  const s = wrongFor(-1) < wrongFor(1) ? -1 : 1;
  const wrong = wrongFor(s);
  let d0 = Infinity;
  let d1 = Infinity;
  let i0 = -1;
  let i1 = -1;
  p.forEach((v, i) => {
    const d = Math.abs(v);
    if (SEP.y[i] === -1 && d < d0) { d0 = d; i0 = i; }
    if (SEP.y[i] === 1 && d < d1) { d1 = d; i1 = i; }
  });
  const ok = wrong === 0;
  return { n, c: shift, s, wrong, d0, d1, i0, i1, width: ok ? 2 * Math.min(d0, d1) : null, centred: ok ? d0 + d1 : null };
}

// The widest street at one tilt: the gap between the classes along the normal.
export function tiltWidth(tilt) {
  const rad = (tilt * Math.PI) / 180;
  const n = [Math.sin(rad), -Math.cos(rad)];
  let best = -Infinity;
  for (const s of [1, -1]) {
    let lo = Infinity;
    let hi = -Infinity;
    SEP.X.forEach((x, i) => {
      const v = s * dot(n, x);
      if (SEP.y[i] === 1) lo = Math.min(lo, v); else hi = Math.max(hi, v);
    });
    best = Math.max(best, lo - hi);
  }
  return best;
}
export function bestTilt(stepDeg) {
  let out = { tilt: 0, width: -Infinity };
  for (let k = 0; k * stepDeg < 180; k++) {
    const w = tiltWidth(k * stepDeg);
    if (w > out.width) out = { tilt: k * stepDeg, width: w };
  }
  return out;
}

// Step 4: refit with some points moved, { index: [x₁, x₂] }.
let lastKey = '';
let lastFit = null;
export function dragFit(moved) {
  const key = JSON.stringify(moved);
  if (key !== lastKey) {
    const X = SEP.X.map((x, i) => (moved[i] ? moved[i].slice() : x.slice()));
    lastFit = svmFit(X, SEP.y, { C: C_HARD });
    lastFit.X = X;
    lastKey = key;
  }
  return lastFit;
}
export function lineMoved(g) {
  const f = fitSep();
  return Math.abs(g.w[0] - f.w[0]) > TOL || Math.abs(g.w[1] - f.w[1]) > TOL || Math.abs(g.b - f.b) > TOL;
}

// The lifted plane w₁x₁ + w₂x₂ + w₃(x₁² + x₂²) + b = k, back in the plane of
// x₁, x₂, is the circle |x − c|² = (k − b)/w₃ + |c|², c = −(w₁, w₂)/(2w₃).
export function circleBoundary() {
  const [w1, w2, w3] = fitLifted().w;
  const b = fitLifted().b;
  const cx = -w1 / (2 * w3);
  const cy = -w2 / (2 * w3);
  const rad = (k) => {
    const v = (k - b) / w3 + cx * cx + cy * cy;
    return v > 0 ? Math.sqrt(v) : null;
  };
  return { cx, cy, r: rad(0), rIn: rad(-1), rOut: rad(1) };
}
// The side view looks along the plane: across is u, the position along
// (w₁, w₂), up is x₁² + x₂². The plane depends only on (u, z), so it is a line.
const w12 = () => Math.hypot(fitLifted().w[0], fitLifted().w[1]);
export const sideView = (x) => {
  const [w1, w2] = fitLifted().w;
  return [(w1 * x[0] + w2 * x[1]) / w12(), x[0] * x[0] + x[1] * x[1]];
};
export const sideF = (u, z) => w12() * u + fitLifted().w[2] * z + fitLifted().b;

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
// Fixed decimals with a true minus sign, and never "−0.00".
export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
const width = (m) => (m === null ? 'none' : num(m, 2));
export const cText = (C) => String(C);

export function readout(step, { tilt = START.tilt, shift = START.shift, fit = null, cIdx = C_START, lifted = false } = {}) {
  if (step === 0 || step === 1) {
    const c = candidate(tilt, shift);
    const rows = [`tilt ${Math.round(tilt)}°, shift ${num(shift, 1)}`];
    if (c.width === null) return [...rows, 'separates: no', `wrong side: ${c.wrong}`];
    if (step === 0) return [...rows, 'separates: yes', `street width: ${num(c.width, 2)}`];
    return [...rows, `nearest circle: ${num(c.d0, 2)}`, `nearest triangle: ${num(c.d1, 2)}`, `street width: ${num(c.width, 2)}`, `centred: ${num(c.centred, 2)}`];
  }
  if (step === 2) {
    const f = fitSep();
    return [`street width: ${width(f.margin)}`, `w₁ = ${num(f.w[0], 2)}`, `w₂ = ${num(f.w[1], 2)}`, `b = ${num(f.b, 2)}`, `support vectors: ${f.sv.length}`];
  }
  if (step === 3) {
    const g = fit || fitSep();
    const s = stats(g, g.X || SEP.X, SEP.y);
    return [`street width: ${width(g.margin)}`, `support vectors: ${s.sv}`, `margin violations: ${s.violations}`, `line moved: ${lineMoved(g) ? 'yes' : 'no'}`];
  }
  if (step === 4) {
    const C = C_GRID[clamp(cIdx, 0, C_GRID.length - 1)];
    const s = overStats(C);
    return [`C = ${cText(C)}`, `street width: ${width(s.margin)}`, `support vectors: ${s.sv}`, `margin violations: ${s.violations}`, `wrong side: ${s.wrong}`, `total hinge: ${num(s.hinge, 2)}`];
  }
  if (!lifted) return ['straight line', `right: ${pct(circleLineAcc())}`];
  return ['flat plane, 3 features', `right: ${pct(liftedAcc())}`, `street width: ${width(fitLifted().margin)}`];
}

export function pageValues() {
  const f = fitSep();
  const c = candidate(START.tilt, START.shift);
  const lo = overStats(C_GRID[0]);
  const hi = overStats(C_GRID.at(-1));
  const one = overStats(1);
  const cb = circleBoundary();
  const lin = circleLineAcc();
  return {
    'n-sep': String(SEP.X.length),
    'start-tilt': String(START.tilt),
    'start-shift': num(START.shift, 1),
    'start-width': num(c.width, 2),
    'start-d0': num(c.d0, 2),
    'start-d1': num(c.d1, 2),
    'start-centred': num(c.centred, 2),
    'sep-width': num(f.margin, 2),
    'sep-sv': String(f.sv.length),
    'sep-sv0': String(f.sv.filter((k) => SEP.y[k] === -1).length),
    'sep-sv1': String(f.sv.filter((k) => SEP.y[k] === 1).length),
    'sep-nonsv': String(SEP.X.length - f.sv.length),
    'sep-w1': num(f.w[0], 2),
    'sep-w2': num(f.w[1], 2),
    'sep-b': num(f.b, 2),
    'sep-tilt': num(((Math.atan2(-f.w[0], f.w[1]) * 180) / Math.PI + 360) % 180, 1),
    'c-hard': String(C_HARD),
    'alpha-max': num(Math.max(...f.alpha), 2),
    'c-lo': cText(C_GRID[0]),
    'c-lo-width': num(lo.margin, 2),
    'c-lo-viol': String(lo.violations),
    'c-lo-sv': String(lo.sv),
    'c-hi': cText(C_GRID.at(-1)),
    'c-hi-width': num(hi.margin, 2),
    'c-hi-viol': String(hi.violations),
    'c-hi-wrong': String(hi.wrong),
    'c-flat': cText(flatFrom()),
    'c1-width': num(one.margin, 2),
    'c1-viol': String(one.violations),
    'n-circ': String(CIRC.X.length),
    'circ-line-acc': pct(lin),
    'circ-line-right': String(Math.round(lin * CIRC.X.length)),
    'lift-acc': pct(liftedAcc()),
    'lift-width': num(fitLifted().margin, 2),
    'lift-w3': num(fitLifted().w[2], 2),
    'lift-r': num(cb.r, 2),
    'lift-sv': String(fitLifted().sv.length),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const SURFACE = '#1e293b';
const LINE = '#334155';

const PX0 = 56;
const PX1 = 326;
const PY0 = 286;
const PY1 = 34;
const RX0 = 352;
const BOX_Y = 8;
const ROW = 20;
// Step 6: two panels, 35 user units per data unit.
const LX0 = 56;
const LX1 = 266;
const QX0 = 372;
const QX1 = 582;
const QY0 = 256;
const QY1 = 46;

const P = (v) => Number(v.toFixed(2));

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const fit0 = fitSep();
  let tilt = START.tilt;
  let shift = START.shift;
  let cIdx = C_START;
  let lifted = false;
  const home = { [POINT_A]: SEP.X[POINT_A].slice(), [POINT_B]: SEP.X[POINT_B].slice() };
  let moved = { [POINT_A]: home[POINT_A].slice(), [POINT_B]: home[POINT_B].slice() };
  let dfit = fit0;

  const X = linearScale(X_DOM, [PX0, PX1]);
  const Y = linearScale(Y_DOM, [PY0, PY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Points of two classes on two features: class 0 (y = −1) as cyan circles, class 1 (y = +1) as violet triangles. A solid line separates them, with a dashed street either side; support vectors are ringed. In the last step, points inside and outside a ring, with a side view where the height is x₁² + x₂².');

  const defs = el('defs', {}, svg);
  const clipM = el('clipPath', { id: 'l34-main' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clipM);
  const clipL = el('clipPath', { id: 'l34-left' }, defs);
  el('rect', { x: LX0, y: QY1, width: LX1 - LX0, height: QY0 - QY1 }, clipL);
  const clipR = el('clipPath', { id: 'l34-right' }, defs);
  el('rect', { x: QX0, y: QY1, width: QX1 - QX0, height: QY0 - QY1 }, clipR);

  // ---- Main plot (steps 1 to 5).
  const mainG = el('g', {}, svg);
  axes(mainG, X, Y, { xLabel: 'x₁', yLabel: 'x₂' });
  const streetG = el('g', { 'clip-path': 'url(#l34-main)' }, mainG);
  const band = el('path', { fill: STRONG, 'fill-opacity': 0.07 }, streetG);
  const edgeA = el('line', { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, streetG);
  const edgeB = el('line', { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, streetG);
  const ghost = el('line', { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '1 4', 'stroke-linecap': 'round' }, streetG);
  const boundary = el('line', { stroke: STRONG, 'stroke-width': 2.2 }, streetG);
  const perpG = el('g', {}, streetG);
  const ringsG = el('g', {}, mainG);
  const pointsG = el('g', {}, mainG);
  const handlesG = el('g', {}, mainG);

  const makeHandle = (idx, name) => {
    const g = el('g', {}, handlesG);
    const mark = el('circle', { r: 12, fill: 'none', stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '2 2' }, g);
    const ring = el('circle', { r: 14, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, g);
    const label = el('text', { class: 'stage-text', text: name }, g);
    label.style.fill = STRONG;
    const hit = el('circle', { class: 'handle', r: 15, fill: 'transparent', tabindex: -1, role: 'button', 'aria-roledescription': 'movable point' }, g);
    hit.style.outline = 'none';
    const h = { idx, name, g, mark, ring, label, hit, focused: false };
    draggable(hit, {
      bounds: { x0: PX0, x1: PX1, y0: PY0, y1: PY1 },
      onDrag: (px, py) => queueDrag(idx, [X.invert(px), Y.invert(py)]),
    });
    hit.addEventListener('focus', () => {
      let kb = true;
      try { kb = hit.matches(':focus-visible'); } catch { /* keep the ring */ }
      h.focused = kb;
      render();
    });
    hit.addEventListener('blur', () => { h.focused = false; render(); });
    return h;
  };
  const handles = [makeHandle(POINT_A, 'A'), makeHandle(POINT_B, 'B')];

  // Key above the plot.
  const keyG = el('g', {}, svg);
  classMarker(keyG, PX0 + 6, 16, 0, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 16, y: 20, text: 'class 0, y = −1' }, keyG);
  classMarker(keyG, PX0 + 166, 16, 1, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 176, y: 20, text: 'class 1, y = +1' }, keyG);

  // Line key under the read-out.
  const lineKey = el('g', {}, svg);
  const keyRow = (y, attrs, text) => {
    const g = el('g', {}, lineKey);
    el('line', { x1: RX0 + 8, x2: RX0 + 32, y1: y, y2: y, ...attrs }, g);
    el('text', { class: 'stage-text', x: RX0 + 38, y: y + 4, text }, g);
    return g;
  };
  const kBound = keyRow(196, { stroke: STRONG, 'stroke-width': 2.2 }, 'boundary');
  const kEdge = keyRow(216, { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, 'edges of the street');
  const kGhost = keyRow(236, { stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '1 4', 'stroke-linecap': 'round' }, 'boundary before moving');
  const kRing = el('g', {}, lineKey);
  el('circle', { cx: RX0 + 20, cy: 256, r: 7, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, kRing);
  el('text', { class: 'stage-text', x: RX0 + 38, y: 260, text: 'support vector' }, kRing);

  // ---- Step 6 panels.
  const sixG = el('g', {}, svg);
  const LXs = linearScale([-3, 3], [LX0, LX1]);
  const LYs = linearScale([-3, 3], [QY0, QY1]);
  const UX = linearScale([-3, 3], [QX0, QX1]);
  const ZY = linearScale([0, 6], [QY0, QY1]);
  axes(sixG, LXs, LYs, { xLabel: 'x₁', yLabel: 'x₂' });
  const leftLines = el('g', { 'clip-path': 'url(#l34-left)' }, sixG);
  const linFit = fitCircleLine();
  const straight = el('line', { stroke: STRONG, 'stroke-width': 2.2 }, leftLines);
  const circG = el('g', {}, leftLines);
  const cb = circleBoundary();
  const ux = LXs(1) - LXs(0);
  for (const [r, dash, sw] of [[cb.rIn, '6 4', 1.4], [cb.rOut, '6 4', 1.4], [cb.r, null, 2.2]]) {
    if (r === null) continue;
    el('circle', { cx: P(LXs(cb.cx)), cy: P(LYs(cb.cy)), r: P(r * ux), fill: 'none', stroke: dash ? MUTED : STRONG, 'stroke-width': sw, ...(dash ? { 'stroke-dasharray': dash } : {}) }, circG);
  }
  const leftPts = el('g', {}, sixG);
  CIRC.X.forEach((x, i) => classMarker(leftPts, LXs(x[0]), LYs(x[1]), CIRC.cls[i], { r: 4 }));
  const caption = (x, text) => el('text', { class: 'stage-text', x, y: 40, text }, sixG);
  caption(LX0, 'the data, x₁ and x₂');
  const rightG = el('g', {}, sixG);
  const sideAxes = axes(rightG, UX, ZY, { xLabel: 'across, seen side-on', yLabel: 'x₁² + x₂²' });
  sideAxes.setAttribute('aria-hidden', 'true');
  el('text', { class: 'stage-text', x: QX0, y: 40, text: 'side view' }, rightG);
  const sideLines = el('g', { 'clip-path': 'url(#l34-right)' }, rightG);
  const fl = fitLifted();
  const sideLine = (k) => {
    // ‖(w₁, w₂)‖ u + w₃ z + b = k.
    const z = (u) => (k - fl.b - w12() * u) / fl.w[2];
    return { x1: P(UX(-3)), y1: P(ZY(z(-3))), x2: P(UX(3)), y2: P(ZY(z(3))) };
  };
  el('line', { ...sideLine(-1), stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, sideLines);
  el('line', { ...sideLine(1), stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, sideLines);
  el('line', { ...sideLine(0), stroke: STRONG, 'stroke-width': 2.2 }, sideLines);
  CIRC.X.forEach((x, i) => {
    const [u, z] = sideView(x);
    classMarker(rightG, UX(u), ZY(z), CIRC.cls[i], { r: 4 });
  });
  const sixVals = el('text', { class: 'value', x: LX0, y: 326 }, sixG);
  const hint = el('text', { class: 'stage-text', x: (QX0 + QX1) / 2, y: (QY0 + QY1) / 2, 'text-anchor': 'middle', text: 'Press "Add x₁² + x₂²"' }, sixG);

  // ---- Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4, 5].map(() => el('text', { class: 'value' }, box));

  // ---- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, hint, min, max, stepv, value) => {
    const label = make('label');
    const input = make('input', { type: 'range', min, max, step: stepv, value });
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    label.append(name, make('span', { className: 'vh', textContent: hint }), ' ', input, ' ', out);
    return { label, input, out };
  };
  const lineGroup = make('div');
  const tS = slider('tilt', ', the angle of the line in degrees', 0, 179, 1, tilt);
  const sS = slider('shift', ', which slides the line sideways', -3, 3, 0.1, shift);
  lineGroup.append(tS.label, sS.label);
  const resetGroup = make('div');
  const resetBtn = make('button', { type: 'button', textContent: 'Put A and B back' });
  resetGroup.append(resetBtn);
  const cGroup = make('div');
  const cS = slider('C', ', the penalty for margin violations', 0, C_GRID.length - 1, 1, cIdx);
  cGroup.append(cS.label);
  const liftGroup = make('div');
  const liftBtn = make('button', { type: 'button', textContent: 'Add x₁² + x₂²' });
  liftGroup.append(liftBtn);
  controls.append(lineGroup, resetGroup, cGroup, liftGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // ---- State.
  let step = 0;
  let token = 0;
  let s = { main: 1, six: 0, side: 0 };
  const target = (i) => ({ main: i < 5 ? 1 : 0, six: i === 5 ? 1 : 0, side: i === 5 && lifted ? 1 : 0 });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };

  // A line n·x = c drawn across the plot, as segment attributes.
  const FAR = 40;
  const seg = (n, c) => {
    const p = [n[0] * c, n[1] * c];
    const d = [-n[1], n[0]];
    return { x1: P(X(p[0] - FAR * d[0])), y1: P(Y(p[1] - FAR * d[1])), x2: P(X(p[0] + FAR * d[0])), y2: P(Y(p[1] + FAR * d[1])) };
  };
  const setLine = (node, n, c) => {
    for (const [k, v] of Object.entries(seg(n, c))) node.setAttribute(k, v);
  };
  const hideStreet = () => {
    for (const node of [band, edgeA, edgeB, boundary]) node.style.display = 'none';
  };
  // Boundary n·x = c with the street c ± h either side; h = null hides the street.
  const drawStreet = (n, c, h) => {
    boundary.style.display = '';
    setLine(boundary, n, c);
    const on = h !== null;
    for (const node of [band, edgeA, edgeB]) node.style.display = on ? '' : 'none';
    if (!on) return;
    setLine(edgeA, n, c - h);
    setLine(edgeB, n, c + h);
    const a = seg(n, c - h);
    const b = seg(n, c + h);
    band.setAttribute('d', `M${a.x1},${a.y1}L${a.x2},${a.y2}L${b.x2},${b.y2}L${b.x1},${b.y1}Z`);
  };
  const fromFit = (f) => {
    const wn = Math.hypot(f.w[0], f.w[1]);
    return wn > 0 ? { n: [f.w[0] / wn, f.w[1] / wn], c: -f.b / wn, h: 1 / wn } : null;
  };

  const curData = () => (step === 4 ? OVER : step === 3 ? { X: dfit.X || SEP.X, cls: SEP.cls, y: SEP.y } : SEP);
  const curFit = () => (step === 4 ? fitOver(C_GRID[cIdx]) : step === 3 ? dfit : fit0);

  function render() {
    op(mainG, s.main);
    op(keyG, 1);
    op(lineKey, s.main);
    op(sixG, s.six);
    op(rightG, s.side);
    op(circG, s.side);
    op(straight, 1 - s.side);
    op(hint, 1 - s.side);

    const d = curData();
    pointsG.replaceChildren();
    ringsG.replaceChildren();
    perpG.replaceChildren();
    d.X.forEach((x, i) => {
      if (step === 3 && (i === POINT_A || i === POINT_B)) return;
      classMarker(pointsG, X(x[0]), Y(x[1]), d.cls[i], { r: 5 });
    });
    ghost.style.display = 'none';
    if (step <= 1) {
      const c = candidate(tilt, shift);
      drawStreet(c.n, c.c, c.width === null ? null : c.width / 2);
      if (step === 1 && c.width !== null) {
        for (const i of [c.i0, c.i1]) {
          const x = SEP.X[i];
          const t = dot(c.n, x) - c.c;
          const fx = x[0] - t * c.n[0];
          const fy = x[1] - t * c.n[1];
          el('line', { x1: P(X(x[0])), y1: P(Y(x[1])), x2: P(X(fx)), y2: P(Y(fy)), stroke: STRONG, 'stroke-width': 1.8, 'stroke-dasharray': '2 3' }, perpG);
          // The distance, beside the middle of the segment, to its left.
          const d = [-c.n[1], c.n[0]];
          const mx = X((x[0] + fx) / 2 - 0.6 * d[0]);
          const my = Y((x[1] + fy) / 2 - 0.6 * d[1]);
          const lab = el('text', { class: 'stage-text', x: P(mx), y: P(my + 4), 'text-anchor': 'middle', text: num(Math.abs(t), 2) }, perpG);
          lab.style.fill = STRONG;
          lab.setAttribute('paint-order', 'stroke');
          lab.setAttribute('stroke', SURFACE);
          lab.setAttribute('stroke-width', 3);
        }
      }
    } else {
      const f = curFit();
      const g = fromFit(f);
      if (g) drawStreet(g.n, g.c, g.h); else hideStreet();
      f.sv.forEach((i) => {
        const x = d.X[i];
        el('circle', { cx: P(X(x[0])), cy: P(Y(x[1])), r: 9.5, fill: 'none', stroke: STRONG, 'stroke-width': 1.5 }, ringsG);
      });
      if (step === 3 && lineMoved(f)) {
        const g0 = fromFit(fit0);
        ghost.style.display = '';
        setLine(ghost, g0.n, g0.c);
      }
    }
    op(handlesG, step === 3 ? s.main : 0);
    for (const h of handles) {
      const x = moved[h.idx];
      const px = X(x[0]);
      const py = Y(x[1]);
      // Redraw the point itself in its class shape, under the hit area.
      const old = h.g.querySelector('.mk');
      if (old) old.remove();
      const mk = classMarker(h.g, px, py, SEP.cls[h.idx], { r: 5 });
      h.g.insertBefore(mk, h.label);
      h.mark.setAttribute('cx', P(px));
      h.mark.setAttribute('cy', P(py));
      h.ring.setAttribute('cx', P(px));
      h.ring.setAttribute('cy', P(py));
      h.ring.style.display = h.focused ? '' : 'none';
      h.label.setAttribute('x', P(px + 12));
      h.label.setAttribute('y', P(py - 11));
      h.hit.setAttribute('cx', P(px));
      h.hit.setAttribute('cy', P(py));
      h.hit.setAttribute('aria-label', `Point ${h.name}, a ${SEP.cls[h.idx] ? 'triangle' : 'circle'}, at x₁ = ${num(x[0], 1)}, x₂ = ${num(x[1], 1)}. Arrow keys move it.`);
    }
    kBound.style.display = '';
    kEdge.style.display = '';
    kGhost.style.display = step === 3 ? '' : 'none';
    kRing.style.display = step >= 2 ? '' : 'none';

    // Step 6 line: the linear fit on the circle data.
    const g6 = fromFit(linFit);
    if (g6) {
      const p = [g6.n[0] * g6.c, g6.n[1] * g6.c];
      const dd = [-g6.n[1], g6.n[0]];
      straight.setAttribute('x1', P(LXs(p[0] - 20 * dd[0])));
      straight.setAttribute('y1', P(LYs(p[1] - 20 * dd[1])));
      straight.setAttribute('x2', P(LXs(p[0] + 20 * dd[0])));
      straight.setAttribute('y2', P(LYs(p[1] + 20 * dd[1])));
    }

    const texts = readout(step, { tilt, shift, fit: dfit, cIdx, lifted });
    if (step === 5) {
      sixVals.textContent = texts.join(' · ');
      box.style.display = 'none';
    } else {
      box.style.display = '';
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
    }

    tS.out.textContent = `${Math.round(tilt)}°`;
    sS.out.textContent = num(shift, 1);
    cS.out.textContent = cText(C_GRID[cIdx]);
    tS.input.setAttribute('aria-valuetext', `tilt ${Math.round(tilt)} degrees`);
    sS.input.setAttribute('aria-valuetext', `shift ${num(shift, 1)}`);
    cS.input.setAttribute('aria-valuetext', `C = ${cText(C_GRID[cIdx])}`);
    liftBtn.setAttribute('aria-pressed', String(lifted));
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

  // Dragging: keep the newest position, refit and redraw at most once a frame.
  let pending = false;
  function queueDrag(idx, x) {
    moved = { ...moved, [idx]: [clamp(x[0], X_DOM[0], X_DOM[1]), clamp(x[1], Y_DOM[0], Y_DOM[1])] };
    // Move the hit area now, so arrow keys pressed before the next frame add up.
    const h = handles.find((k) => k.idx === idx);
    h.hit.setAttribute('cx', P(X(moved[idx][0])));
    h.hit.setAttribute('cy', P(Y(moved[idx][1])));
    if (pending) return;
    pending = true;
    const run = () => {
      pending = false;
      dfit = dragFit(moved);
      render();
      say(spoken(3));
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  }

  const spoken = (i) => {
    if (i <= 1) {
      const c = candidate(tilt, shift);
      const head = `Line at tilt ${Math.round(tilt)} degrees, shift ${num(shift, 1)}.`;
      if (c.width === null) return `${head} It does not separate the classes: ${c.wrong} ${c.wrong === 1 ? 'point is' : 'points are'} on the wrong side.`;
      if (i === 0) return `${head} It separates the circles from the triangles. The empty street around it is ${num(c.width, 2)} wide.`;
      return `${head} Nearest circle ${num(c.d0, 2)} away, nearest triangle ${num(c.d1, 2)} away. Street width ${num(c.width, 2)}, twice the smaller distance. Centred at this tilt it would be ${num(c.centred, 2)}.`;
    }
    if (i === 2) {
      const n1 = fit0.sv.filter((k) => SEP.y[k] === 1).length;
      const n0 = fit0.sv.length - n1;
      return `The widest street: width ${num(fit0.margin, 2)}. ${fit0.sv.length} support vectors, ringed, sit on its edges: ${n0} ${n0 === 1 ? 'circle' : 'circles'} and ${n1} ${n1 === 1 ? 'triangle' : 'triangles'}.`;
    }
    if (i === 3) {
      const st = stats(dfit, dfit.X || SEP.X, SEP.y);
      return `Street width ${dfit.margin === null ? 'none' : num(dfit.margin, 2)}, ${st.sv} support vectors, ${st.violations} margin violations. The line ${lineMoved(dfit) ? 'has moved' : 'has not moved'}.`;
    }
    if (i === 4) {
      const st = overStats(C_GRID[cIdx]);
      return `C = ${cText(C_GRID[cIdx])}. Street width ${num(st.margin, 2)}, ${st.sv} support vectors, ${st.violations} margin violations, ${st.wrong} on the wrong side.`;
    }
    if (!lifted) return `Forty points: twenty circles inside radius 1 and twenty triangles in a ring around them. The best straight line gets ${pct(circleLineAcc())} right.`;
    return `With the third feature x₁² + x₂², the side view shows the circles low and the triangles high, split by a flat plane, drawn as a line. Back on the flat picture the boundary is a circle of radius ${num(cb.r, 2)}. All ${CIRC.X.length} points are on the right side.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  tS.input.addEventListener('input', () => { tilt = clamp(Math.round(Number(tS.input.value)), 0, 179); render(); say(spoken(step)); });
  sS.input.addEventListener('input', () => { shift = clamp(Math.round(Number(sS.input.value) * 10) / 10, -3, 3); render(); say(spoken(step)); });
  cS.input.addEventListener('input', () => { cIdx = clamp(Math.round(Number(cS.input.value)), 0, C_GRID.length - 1); render(); say(spoken(4)); });
  resetBtn.addEventListener('click', () => {
    moved = { [POINT_A]: home[POINT_A].slice(), [POINT_B]: home[POINT_B].slice() };
    dfit = dragFit(moved);
    render();
    announce(spoken(3));
  });
  liftBtn.addEventListener('click', () => {
    lifted = !lifted;
    liftBtn.textContent = lifted ? 'Back to two features' : 'Add x₁² + x₂²';
    animateTo(target(5), 500);
    announce(spoken(5));
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  dfit = dragFit(moved);
  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      showGroup(lineGroup, i <= 1);
      showGroup(resetGroup, i === 3);
      showGroup(cGroup, i === 4);
      showGroup(liftGroup, i === 5);
      for (const h of handles) {
        h.hit.setAttribute('tabindex', i === 3 ? 0 : -1);
        if (i !== 3 && doc.activeElement === h.hit) h.hit.blur();
      }
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
