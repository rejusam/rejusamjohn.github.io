// Lesson L3.9, "Principal component analysis": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-pca.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, randn, standardise } from '../assets/maths.js';
import { cov2, pca2, projVar, perpMSE } from '../assets/unsupervised.js';
import { linearScale, clamp, el, axes, tween, classMarker } from '../assets/plot.js';

export const SEED = 111;
export const N = 50;
// The slider: whole degrees, 0° to 179°.
export const ANGLES = Array.from({ length: 180 }, (_, i) => i);
export const UNITS = ['cm', 'mm', 'std'];
export const UNITS_ALL = ['cm', 'mm', 'std', 'std-mm'];
// The person picked out on the plot: the first row.
export const P1 = 0;

// Height ~ N(170, 8²); arm span = 0.9 (height − 170) + 170 + N(0, 4²), in cm.
export function makeData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  const X = [];
  for (let i = 0; i < n; i++) {
    const h = 170 + 8 * randn(rng);
    const a = 0.9 * (h - 170) + 170 + 4 * randn(rng);
    X.push([h, a]);
  }
  return X;
}

const CM = makeData();
const MM = CM.map(([h, a]) => [h, 10 * a]);
// z-scores with the population SD (divide by n), as in standardise().
const zOf = (X) => {
  const a = standardise(X.map((r) => r[0])).z;
  const b = standardise(X.map((r) => r[1])).z;
  return a.map((v, i) => [v, b[i]]);
};
const DATA = { cm: CM, mm: MM, std: zOf(CM), 'std-mm': zOf(MM) };
const FITS = Object.fromEntries(UNITS_ALL.map((u) => [u, pca2(DATA[u])]));

export const data = (units = 'cm') => DATA[units];
export const fit = (units = 'cm') => FITS[units];

const rad = (d) => (d * Math.PI) / 180;
// Direction of a vector in degrees, folded into [0, 180): a line has no arrow.
export function angleDeg([a, b]) {
  let d = (Math.atan2(b, a) * 180) / Math.PI;
  if (d < 0) d += 180;
  if (d >= 180) d -= 180;
  return d;
}
export const pc1Angle = (units = 'cm') => angleDeg(FITS[units].vecs[0]);

// Spread along and across the line at angle d (degrees), both sample variances (n − 1).
export const along = (d) => projVar(CM, rad(d));
export const across = (d) => perpMSE(CM, rad(d)) * CM.length / (CM.length - 1);
export const total = () => FITS.cm.S[0][0] + FITS.cm.S[1][1];

// The whole degree on the slider with the most variance along the line.
export function gridBest() {
  let best = ANGLES[0];
  for (const d of ANGLES) if (along(d) > along(best)) best = d;
  return best;
}

// One number per person: the PC1 score of the centred point.
export function scores() {
  const { mean, vecs } = FITS.cm;
  const v = vecs[0];
  return CM.map(([h, a]) => (h - mean[0]) * v[0] + (a - mean[1]) * v[1]);
}
// Each person rebuilt from their score: mean + score × v1.
export function rebuilt() {
  const { mean, vecs } = FITS.cm;
  const v = vecs[0];
  return scores().map((t) => [mean[0] + t * v[0], mean[1] + t * v[1]]);
}
// Mean (divide by n) of the squared distances from each point to its rebuilt point.
export const reconMSE = () => perpMSE(CM, rad(pc1Angle('cm')));

// Sample correlation of height and arm span (the divisor cancels).
export function corr() {
  const S = FITS.cm.S;
  return S[0][1] / Math.sqrt(S[0][0] * S[1][1]);
}

const f1 = (v) => {
  const s = v.toFixed(1);
  return s === '-0.0' ? '0.0' : s.replace('-', '−');
};
const f2 = (v) => {
  const s = v.toFixed(2);
  return s === '-0.00' ? '0.00' : s.replace('-', '−');
};
// A share as a percentage with one decimal place.
export const pct = (v) => `${(v * 100).toFixed(1)}%`;
const deg = (d) => `${d.toFixed(1)}°`;
const UNIT_NAME = { cm: 'both in cm', mm: 'arm span in mm', std: 'standardised' };

export function readout(step, { theta = 0, centred = true, units = 'cm', rebuilt: rb = true } = {}) {
  const p = FITS.cm;
  const x = CM[P1];
  if (step === 0) {
    if (!centred) return ['raw (cm)', `mean height ${f1(p.mean[0])}`, `mean arm span ${f1(p.mean[1])}`, `ringed: (${f1(x[0])}, ${f1(x[1])})`];
    return ['centred (cm)', 'mean height 0.0', 'mean arm span 0.0', `ringed: (${f1(x[0] - p.mean[0])}, ${f1(x[1] - p.mean[1])})`];
  }
  if (step === 1 || step === 2) {
    const d = clamp(theta, 0, 179.999);
    return [`angle ${deg(d)}`, `along: ${f1(along(d))} cm²`, `across: ${f1(across(d))} cm²`, `total: ${f1(total())} cm²`];
  }
  if (step === 3) {
    return [`γ₁ = ${f1(p.gammas[0])} cm²`, `γ₂ = ${f1(p.gammas[1])} cm²`, `γ₁ + γ₂ = ${f1(total())} cm²`, `PC1 share: ${pct(p.explained[0])}`, `PC2 share: ${pct(p.explained[1])}`];
  }
  if (step === 4) {
    const t = scores()[P1];
    const r = rebuilt()[P1];
    if (!rb) return ['original points', `ringed: (${f1(x[0])}, ${f1(x[1])})`, `its score: ${f1(t)}`];
    return ['rebuilt from PC1', `ringed: (${f1(r[0])}, ${f1(r[1])})`, `its score: ${f1(t)}`, `mean sq. error ${f1(reconMSE())} cm²`];
  }
  const q = FITS[units];
  const a = pc1Angle(units);
  return [UNIT_NAME[units], `PC1 at ${deg(a)}`, `${deg(90 - a)} off arm span`, `PC1 share: ${pct(q.explained[0])}`];
}

export const PAGE_PIN = {
  'mean-h': '169.7', 'mean-a': '169.5', 'var-h': '40.0', 'var-a': '45.8', total: '85.8',
  'pc1-angle': '47.7', 'grid-best': '48', gamma1: '73.9', gamma2: '11.9', e1: '86.1%', e2: '13.9%',
  mse: '11.7', 'mm-angle': '86.1', 'mm-off': '3.9', 'mm-e1': '99.6%', r: '0.72', 'std-e1': '86.0%',
};

export function pageValues() {
  const p = FITS.cm;
  const m = FITS.mm;
  const x = CM[P1];
  const t = scores()[P1];
  const r = rebuilt()[P1];
  return {
    n: String(CM.length),
    'n-1': String(CM.length - 1),
    'mean-h': f1(p.mean[0]),
    'mean-a': f1(p.mean[1]),
    'p1-h': f1(x[0]),
    'p1-a': f1(x[1]),
    'p1-ch': f1(x[0] - p.mean[0]),
    'p1-ca': f1(x[1] - p.mean[1]),
    'var-h': f1(p.S[0][0]),
    'var-a': f1(p.S[1][1]),
    'cov': f1(p.S[0][1]),
    total: f1(total()),
    'pc1-angle': f1(pc1Angle('cm')),
    'grid-best': String(gridBest()),
    'v1-h': f2(p.vecs[0][0]),
    'v1-a': f2(p.vecs[0][1]),
    'v2-h': f2(p.vecs[1][0]),
    'v2-a': f2(p.vecs[1][1]),
    gamma1: f1(p.gammas[0]),
    gamma2: f1(p.gammas[1]),
    e1: pct(p.explained[0]),
    e2: pct(p.explained[1]),
    'p1-t': f1(t),
    'p1-rh': f1(r[0]),
    'p1-ra': f1(r[1]),
    'p1-lost': f1(Math.hypot(x[0] - r[0], x[1] - r[1])),
    mse: f1(reconMSE()),
    'mm-var-a': Math.round(m.S[1][1]).toString(),
    'mm-angle': f1(pc1Angle('mm')),
    'mm-off': f1(90 - pc1Angle('mm')),
    'mm-e1': pct(m.explained[0]),
    r: f2(corr()),
    'std-angle': f1(pc1Angle('std')),
    'std-e1': pct(FITS.std.explained[0]),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The scatter is square, with the same scale on both axes, so angles on
// screen are the true angles.
const PX0 = 60;
const PX1 = 320;
const PY0 = 292;
const PY1 = 32;
const DOM = { cm: 20, mm: 200, std: 3 };
// The right-hand panel.
const RX0 = 352;
const BOX_Y = 8;
const ROW = 20;
const CX0 = 380;
const CX1 = 586;
const CY0 = 300;
const CY1 = 178;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const p = FITS.cm;
  const a1 = pc1Angle('cm');
  let theta = 0;
  let centred = true;
  let rb = true;
  let units = 'cm';

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Fifty people, each a grey dot: height across, arm span up, drawn to the same scale. A line through the middle of the cloud can be turned; the side panel shows the spread along and across it, and later the two principal directions and the people rebuilt from one number each.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'pca-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);
  const head = (id, colour) => {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0L10,5L0,10Z', fill: colour }, m);
  };
  head('pca-head-1', CYAN);
  head('pca-head-2', VIOLET);

  const scale = (u) => {
    const d = DOM[u];
    return { X: linearScale([-d, d], [PX0, PX1]), Y: linearScale([-d, d], [PY0, PY1]) };
  };
  const SC = { cm: scale('cm'), mm: scale('mm'), std: scale('std') };
  const raw = {
    X: linearScale([p.mean[0] - DOM.cm, p.mean[0] + DOM.cm], [PX0, PX1]),
    Y: linearScale([p.mean[1] - DOM.cm, p.mean[1] + DOM.cm], [PY0, PY1]),
  };
  const axesRaw = axes(svg, raw.X, raw.Y, { xLabel: 'height (cm)', yLabel: 'arm span (cm)' });
  const axesCm = axes(svg, SC.cm.X, SC.cm.Y, { xLabel: 'height (cm), centred', yLabel: 'arm span (cm), centred' });
  const axesMm = axes(svg, SC.mm.X, SC.mm.Y, { xLabel: 'height (cm), centred', yLabel: 'arm span (mm), centred' });
  const axesStd = axes(svg, SC.std.X, SC.std.Y, { xLabel: 'height (z)', yLabel: 'arm span (z)' });

  // Cross hairs through the mean: dashed while raw, the zero lines once centred.
  const midX = (PX0 + PX1) / 2;
  const midY = (PY0 + PY1) / 2;
  const crossG = el('g', {}, svg);
  el('line', { x1: PX0, x2: PX1, y1: midY, y2: midY, stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '4 4' }, crossG);
  el('line', { x1: midX, x2: midX, y1: PY0, y2: PY1, stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '4 4' }, crossG);
  const crossLabel = el('text', { class: 'stage-text', x: midX + 6, y: PY1 + 12 }, crossG);

  const plotG = el('g', { 'clip-path': 'url(#pca-plot)' }, svg);
  const lostG = el('g', {}, plotG);
  const theLine = el('line', { stroke: STRONG, 'stroke-width': 1.6 }, plotG);
  const footG = el('g', {}, plotG);
  const pointsG = el('g', {}, plotG);
  const marks = CM.map(() => classMarker(pointsG, 0, 0, null, { r: 4 }));
  const ring = el('circle', { r: 9, fill: 'none', stroke: STRONG, 'stroke-width': 1.6 }, plotG);
  const arrowsG = el('g', {}, svg);
  const arrow1 = el('line', { stroke: CYAN, 'stroke-width': 2.6, 'marker-end': 'url(#pca-head-1)' }, arrowsG);
  const arrow2 = el('line', { stroke: VIOLET, 'stroke-width': 2.6, 'stroke-dasharray': '6 3', 'marker-end': 'url(#pca-head-2)' }, arrowsG);
  const lab1 = el('text', { class: 'stage-text', text: 'PC1' }, arrowsG);
  lab1.style.fill = CYAN;
  const lab2 = el('text', { class: 'stage-text', text: 'PC2' }, arrowsG);
  lab2.style.fill = VIOLET;
  const foots = CM.map(() => el('circle', { r: 2.4, fill: CYAN }, footG));
  const losts = CM.map(() => el('line', { stroke: MUTED, 'stroke-width': 1 }, lostG));

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3, 4].map(() => el('text', { class: 'value' }, box));

  // Chart of along and across against the angle.
  const chartG = el('g', {}, svg);
  const AX = linearScale([0, 180], [CX0, CX1]);
  const VY = linearScale([0, 90], [CY0, CY1]);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, chartG);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, chartG);
  for (const v of [40, 80]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: VY(v), y2: VY(v), stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: VY(v) + 4, 'text-anchor': 'end', text: String(v) }, chartG);
  }
  for (const d of [0, 45, 90, 135, 180]) {
    el('line', { x1: AX(d), x2: AX(d), y1: CY0, y2: CY0 + 4, stroke: LINE }, chartG);
    el('text', { class: 'stage-text', x: AX(d), y: CY0 + 16, 'text-anchor': 'middle', text: `${d}°` }, chartG);
  }
  const curve = (fn) => ANGLES.concat([180]).map((d, i) => `${i ? 'L' : 'M'}${AX(d).toFixed(2)},${VY(fn(d % 180)).toFixed(2)}`).join('');
  el('path', { d: curve(() => total()), fill: 'none', stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '1 3' }, chartG);
  el('path', { d: curve(along), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, chartG);
  el('path', { d: curve(across), fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, chartG);
  const tmark = el('line', { y1: CY1 - 6, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, chartG);
  const ckey = el('g', {}, chartG);
  const keyRow = (x, colour, dash, text) => {
    el('line', { x1: x, x2: x + 16, y1: CY1 - 22, y2: CY1 - 22, stroke: colour, 'stroke-width': 2, 'stroke-dasharray': dash }, ckey);
    el('text', { class: 'stage-text', x: x + 20, y: CY1 - 18, text }, ckey).style.fill = colour;
  };
  keyRow(CX0, CYAN, null, 'along');
  keyRow(CX0 + 74, VIOLET, '5 3', 'across');
  keyRow(CX0 + 156, MUTED, '1 3', 'total');

  // Shares: two bars, PC2's hatched by a dash pattern on its outline.
  const shareG = el('g', {}, svg);
  const BW = CX1 - CX0 - 60;
  const bar = (k, y, colour, dash) => {
    el('text', { class: 'stage-text', x: CX0, y: y + 13, text: `PC${k + 1}` }, shareG).style.fill = colour;
    el('rect', { x: CX0 + 40, y, width: BW, height: 18, fill: 'none', stroke: LINE }, shareG);
    el('rect', { x: CX0 + 40, y, width: BW * p.explained[k], height: 18, fill: k ? 'none' : colour, stroke: colour, 'stroke-width': 1.6, 'stroke-dasharray': dash }, shareG);
  };
  bar(0, CY1 + 20, CYAN, null);
  bar(1, CY1 + 60, VIOLET, '4 2');
  el('text', { class: 'stage-text', x: CX0 + 40, y: CY1 + 104, text: 'share of total variance' }, shareG);

  // One number per person: the scores on a line.
  const stripG = el('g', {}, svg);
  const t = scores();
  const tMax = Math.ceil(Math.max(...t.map(Math.abs)) / 5) * 5;
  const TX = linearScale([-tMax, tMax], [CX0 + 6, CX1 - 6]);
  const SY = CY1 + 60;
  el('line', { x1: CX0, x2: CX1, y1: SY, y2: SY, stroke: LINE }, stripG);
  for (const v of [-tMax, 0, tMax]) {
    el('line', { x1: TX(v), x2: TX(v), y1: SY, y2: SY + 5, stroke: LINE }, stripG);
    el('text', { class: 'stage-text', x: TX(v), y: SY + 18, 'text-anchor': 'middle', text: v < 0 ? `−${-v}` : String(v) }, stripG);
  }
  t.forEach((v) => classMarker(stripG, TX(v), SY - 8, null, { r: 3.5 }));
  el('circle', { cx: TX(t[P1]), cy: SY - 8, r: 7, fill: 'none', stroke: STRONG, 'stroke-width': 1.4 }, stripG);
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: SY + 38, 'text-anchor': 'middle', text: 'PC1 score, one per person' }, stripG);

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const viewGroup = make('div');
  const viewBtns = [make('button', { type: 'button', textContent: 'Raw' }), make('button', { type: 'button', textContent: 'Centred' })];
  viewGroup.append(...viewBtns);

  const angGroup = make('div');
  const angLabel = make('label');
  const angInput = make('input', { type: 'range', min: 0, max: 179, step: 1, value: 0 });
  const angOut = make('output');
  angOut.setAttribute('aria-hidden', 'true');
  angLabel.append('angle', make('span', { className: 'vh', textContent: ' of the line, in degrees' }), ' ', angInput, ' ', angOut);
  const snapBtn = make('button', { type: 'button', textContent: 'Snap to PC1' });
  angGroup.append(angLabel, snapBtn);

  const rbGroup = make('div');
  const rbBtns = [make('button', { type: 'button', textContent: 'Original' }), make('button', { type: 'button', textContent: 'Rebuilt' })];
  rbGroup.append(...rbBtns);

  const unitGroup = make('div');
  const unitBtns = UNITS.map((u) => make('button', { type: 'button', textContent: { cm: 'Both cm', mm: 'Arm span mm', std: 'Standardised' }[u] }));
  unitGroup.append(...unitBtns);
  controls.append(viewGroup, angGroup, rbGroup, unitGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Layers fade; c (raw to centred) and r (points to rebuilt) move.
  let step = 0;
  let token = 0;
  let s = { c: 1, r: 0, line: 0, foot: 0, arrows: 0, chart: 0, share: 0, strip: 0, lost: 0 };
  const target = (i) => ({
    c: i === 0 && !centred ? 0 : 1,
    r: i === 4 && rb ? 1 : 0,
    line: i >= 1 && i <= 5 && i !== 3 ? 1 : 0,
    foot: i === 1 || i === 2 ? 1 : 0,
    arrows: i === 3 || i === 5 ? 1 : 0,
    chart: i === 1 || i === 2 ? 1 : 0,
    share: i === 3 ? 1 : 0,
    strip: i === 4 ? 1 : 0,
    lost: i === 1 || i === 2 || i === 4 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const curUnits = () => (step === 5 ? units : 'cm');
  const lineAngle = () => (step === 1 || step === 2 ? theta : step === 5 ? pc1Angle(units) : a1);

  function render() {
    const u = curUnits();
    const { X, Y } = SC[u];
    const D = DATA[u];
    const m = FITS[u].mean;
    const R = rebuilt();
    const P = (v) => v.toFixed(2);

    op(axesRaw, step === 0 ? 1 - s.c : 0);
    op(axesCm, u === 'cm' ? (step === 0 ? s.c : 1) : 0);
    op(axesMm, u === 'mm' ? 1 : 0);
    op(axesStd, u === 'std' ? 1 : 0);
    crossLabel.textContent = s.c < 0.5 && step === 0 ? 'mean' : '0';
    // In the mm view PC1 runs close to the vertical line; put the label on its left.
    const leftLabel = u === 'mm';
    crossLabel.setAttribute('x', leftLabel ? midX - 6 : midX + 6);
    crossLabel.setAttribute('text-anchor', leftLabel ? 'end' : 'start');

    D.forEach((x, i) => {
      // Rebuilt points slide along their own perpendicular.
      const cx = x[0] - m[0] + s.r * (R[i][0] - x[0]);
      const cy = x[1] - m[1] + s.r * (R[i][1] - x[1]);
      marks[i].setAttribute('cx', P(X(cx)));
      marks[i].setAttribute('cy', P(Y(cy)));
      if (i === P1) {
        ring.setAttribute('cx', P(X(cx)));
        ring.setAttribute('cy', P(Y(cy)));
      }
    });

    const d = lineAngle();
    const c = Math.cos(rad(d));
    const sn = Math.sin(rad(d));
    const big = 4 * DOM[u];
    theLine.setAttribute('x1', P(X(-big * c)));
    theLine.setAttribute('y1', P(Y(-big * sn)));
    theLine.setAttribute('x2', P(X(big * c)));
    theLine.setAttribute('y2', P(Y(big * sn)));
    op(theLine, s.line);

    // Feet of the perpendiculars from the centred points (cm only).
    if (u === 'cm') {
      CM.forEach((x, i) => {
        const hx = x[0] - p.mean[0];
        const hy = x[1] - p.mean[1];
        const tt = hx * c + hy * sn;
        foots[i].setAttribute('cx', P(X(tt * c)));
        foots[i].setAttribute('cy', P(Y(tt * sn)));
        losts[i].setAttribute('x1', P(X(hx)));
        losts[i].setAttribute('y1', P(Y(hy)));
        losts[i].setAttribute('x2', P(X(tt * c)));
        losts[i].setAttribute('y2', P(Y(tt * sn)));
      });
    }
    op(footG, s.foot);
    op(lostG, s.lost);

    // Arrows two standard deviations long along each direction.
    const q = FITS[u];
    const v1 = q.vecs[0];
    const v2 = q.vecs[1];
    const l1 = 2 * Math.sqrt(q.gammas[0]);
    const l2 = 2 * Math.sqrt(q.gammas[1]);
    arrow1.setAttribute('x1', P(X(0)));
    arrow1.setAttribute('y1', P(Y(0)));
    arrow1.setAttribute('x2', P(X(l1 * v1[0])));
    arrow1.setAttribute('y2', P(Y(l1 * v1[1])));
    arrow2.setAttribute('x1', P(X(0)));
    arrow2.setAttribute('y1', P(Y(0)));
    arrow2.setAttribute('x2', P(X(l2 * v2[0])));
    arrow2.setAttribute('y2', P(Y(l2 * v2[1])));
    lab1.setAttribute('x', P(X(l1 * v1[0]) + 8));
    lab1.setAttribute('y', P(Y(l1 * v1[1]) + 4));
    // PC2 is short when PC1 dominates; keep its label clear of the arrow head.
    const px2 = X(l2 * v2[0]);
    const py2 = Y(l2 * v2[1]);
    lab2.setAttribute('x', P(px2 + 8));
    lab2.setAttribute('y', P(py2 + 14));
    arrow2.style.display = step === 5 ? 'none' : '';
    lab2.style.display = step === 5 ? 'none' : '';
    op(arrowsG, s.arrows);

    op(crossG, step === 0 || step === 5 ? 1 : 0.6);
    op(ring, step === 0 || step === 4 ? 1 : 0);

    op(chartG, s.chart);
    tmark.setAttribute('x1', P(AX(theta)));
    tmark.setAttribute('x2', P(AX(theta)));
    op(shareG, s.share);
    op(stripG, s.strip);

    const texts = readout(step, { theta, centred, units, rebuilt: rb });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let w = 210;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(RX0, 598 - w);
    boxRect.setAttribute('x', bx);
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', w.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', bx + 12);
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    angInput.value = String(Math.round(theta));
    angOut.textContent = `${Math.round(theta)}°`;
    angInput.setAttribute('aria-valuetext', `${theta.toFixed(1)} degrees`);
    viewBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === (centred ? 1 : 0))));
    rbBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === (rb ? 1 : 0))));
    unitBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(UNITS[i] === units)));
  }

  function animateTo(to, ms, extra = null) {
    const my = ++token;
    const start = { ...s };
    const t0 = theta;
    return tween(ms, (k) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * k;
      s = next;
      if (extra !== null) theta = t0 + (extra - t0) * k;
      render();
    }, { reduced });
  }

  const spoken = (i) => {
    const rows = readout(i, { theta, centred, units, rebuilt: rb });
    const lead = [
      'Fifty people, height across and arm span up.',
      'A line through the centre at the chosen angle, with each point dropped onto it.',
      'The line on PC1, the direction with the most spread.',
      'PC1, solid cyan arrow, and PC2, dashed violet arrow, at right angles.',
      'Each person kept as one number, their PC1 score, and rebuilt on the PC1 line.',
      'PC1 in different units.',
    ][i];
    return `${lead} ${rows.join('. ')}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  viewBtns.forEach((b, i) => b.addEventListener('click', () => {
    centred = i === 1;
    animateTo(target(step), 600).then(() => announce(spoken(step)));
  }));
  angInput.addEventListener('input', () => {
    token++;
    theta = clamp(Math.round(Number(angInput.value)), 0, 179);
    render();
    say(spoken(step));
  });
  snapBtn.addEventListener('click', () => {
    animateTo(target(step), 700, a1).then(() => announce(spoken(step)));
  });
  rbBtns.forEach((b, i) => b.addEventListener('click', () => {
    rb = i === 1;
    animateTo(target(step), 700).then(() => announce(spoken(step)));
  }));
  unitBtns.forEach((b, i) => b.addEventListener('click', () => {
    units = UNITS[i];
    render();
    announce(spoken(step));
  }));

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const para of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...para.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) para.hidden = false;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i, animate) {
      const from = step;
      step = i;
      showGroup(viewGroup, i === 0);
      showGroup(angGroup, i === 1 || i === 2);
      snapBtn.style.display = i === 2 ? '' : 'none';
      showGroup(rbGroup, i === 4);
      showGroup(unitGroup, i === 5);
      // Step 2 starts flat; step 3 turns the line onto PC1.
      let goal = null;
      if (i === 1 && from !== 2) theta = 0;
      if (i === 2) goal = a1;
      if (i === 0) centred = true;
      if (i === 4) rb = true;
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        if (goal !== null) theta = goal;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      // Step 1 opens on the raw axes, then centres; step 5 opens on the
      // original points, then rebuilds.
      if (i === 0) s = { ...s, c: 0 };
      if (i === 4) s = { ...s, r: 0 };
      render();
      return animateTo(to, i === 0 || i === 4 ? 1200 : 700, goal);
    },
  };
}

