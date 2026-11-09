// Lesson 2, "Data as numbers": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { standardise, euclidean } from '../assets/maths.js';
import { linearScale, clamp, el, axes, tween } from '../assets/plot.js';

// Six made-up fruit. The values are invented but realistic: treated as a
// sphere, each one has a density between about 0.8 and 1 g/cm³.
export const FRUIT = [
  { kind: 'apple', mass: 140, diameter: 7.0 },
  { kind: 'orange', mass: 230, diameter: 8.0 },
  { kind: 'nectarine', mass: 136, diameter: 6.4 },
  { kind: 'plum', mass: 62, diameter: 5.0 },
  { kind: 'apple', mass: 152, diameter: 7.1 },
  { kind: 'apricot', mass: 40, diameter: 4.4 },
];

// Each row as a vector x = (x1, x2) = (mass in g, diameter in cm).
export const vectors = (rows = FRUIT) => rows.map((f) => [f.mass, f.diameter]);

// Standardise each feature (column) on its own, with the population SD.
export function standardiseRows(pts) {
  const cols = [0, 1].map((k) => standardise(pts.map((p) => p[k])));
  return {
    z: pts.map((_, i) => cols.map((c) => c.z[i])),
    mu: cols.map((c) => c.mu),
    sigma: cols.map((c) => c.sigma),
  };
}

export const distance = euclidean;

// Every pair i < j with its Euclidean distance.
export function pairDistances(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) out.push({ i, j, d: distance(pts[i], pts[j]) });
  }
  return out;
}

// The closest pair (the first one found if two tie).
export function nearestPair(pts) {
  return pairDistances(pts).reduce((best, p) => (p.d < best.d ? p : best));
}

// Pairs are named by their row numbers, smaller first: "1-3".
export const pairKey = ({ i, j }) => `${Math.min(i, j) + 1}-${Math.max(i, j) + 1}`;

// Fixed decimals with a true minus sign.
export function fmt(v, dp) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

export const vecText = (p) => `(${fmt(p[0], 0)}, ${fmt(p[1], 1)})`;
export const fruitName = (i) => `fruit ${i + 1} (${FRUIT[i].kind})`;

export const START = { a: 0, b: 3 };

// ---------------------------------------------------------------- the stage

const BOX = { row: 20 };
const TABLE = { x0: 40, x1: 372, head: 64, top: 98, row: 32 };
const COLS = { num: 56, kind: 80, mass: 262, diameter: 360 };
const PANEL = 398;
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const raw = vectors();
  const { z, mu, sigma } = standardiseRows(raw);
  const n = raw.length;
  const nearRaw = nearestPair(raw);
  const nearZ = nearestPair(z);

  // Raw axes start at zero, so a vector can be drawn as an arrow from the origin.
  const X = linearScale([0, 250], [64, 580]);
  const Y = linearScale([0, 9], [290, 22]);
  // Standardised axes use the same length per unit across and up, so the
  // length of a line on screen matches the distance.
  const unit = (290 - 22) / 3.6;
  const half = (580 - 64) / unit / 2;
  const ZX = linearScale([-half, half], [64, 580]);
  const ZY = linearScale([-1.8, 1.8], [290, 22]);

  const rawPx = raw.map(([m, d]) => [X(m), Y(d)]);
  const zPx = z.map(([a, b]) => [ZX(a), ZY(b)]);
  const rowY = (i) => TABLE.top + TABLE.row * i;
  const tablePx = raw.map((_, i) => [TABLE.x1 + 14, rowY(i) - 4]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Six fruit, shown first as a table of mass and diameter and then as points on two axes.');

  const defs = el('defs', {}, svg);
  const marker = el('marker', {
    id: 'l2-head', viewBox: '0 0 10 10', refX: 9, refY: 5,
    markerWidth: 8, markerHeight: 8, orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse',
  }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: CYAN }, marker);

  // --- The table (steps 1 and 2).
  const table = el('g', {}, svg);
  const featBand = el('g', {}, table);
  el('rect', {
    x: 182, y: TABLE.head - 18, width: TABLE.x1 - 182, height: rowY(n - 1) + 24 - (TABLE.head - 18),
    rx: 6, fill: CYAN, 'fill-opacity': 0.07, stroke: CYAN, 'stroke-opacity': 0.5,
  }, featBand);
  el('text', { class: 'stage-text', x: (182 + TABLE.x1) / 2, y: TABLE.head - 28, 'text-anchor': 'middle', fill: CYAN, text: 'features' }, featBand);
  el('text', { class: 'stage-text', x: COLS.kind, y: TABLE.head - 28, fill: MUTED, text: 'label' }, featBand);
  const exX = 24;
  const exY = (rowY(0) + rowY(n - 1)) / 2;
  el('text', {
    class: 'stage-text', x: exX, y: exY, 'text-anchor': 'middle', fill: MUTED,
    transform: `rotate(-90 ${exX} ${exY})`, text: 'examples',
  }, featBand);
  el('line', { x1: 34, x2: 34, y1: rowY(0) - 18, y2: rowY(n - 1) + 8, stroke: MUTED }, featBand);

  const rowHi = el('rect', {
    x: TABLE.x0, width: TABLE.x1 - TABLE.x0, height: 28, rx: 5,
    fill: CYAN, 'fill-opacity': 0.12, stroke: CYAN, 'stroke-width': 1.5,
  }, table);

  const cell = (x, y, text, anchor, fill = INK) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, fill, text }, table);
  cell(COLS.num, TABLE.head, '#', 'middle', MUTED);
  cell(COLS.kind, TABLE.head, 'kind', 'start', MUTED);
  cell(COLS.mass, TABLE.head, 'mass', 'end', MUTED);
  cell(COLS.diameter, TABLE.head, 'diameter', 'end', MUTED);
  cell(COLS.mass, TABLE.head + 16, '(g)', 'end', MUTED);
  cell(COLS.diameter, TABLE.head + 16, '(cm)', 'end', MUTED);
  el('line', { x1: TABLE.x0, x2: TABLE.x1, y1: TABLE.head + 26, y2: TABLE.head + 26, stroke: '#334155' }, table);
  const rowCells = FRUIT.map((f, i) => [
    cell(COLS.num, rowY(i) + 14, String(i + 1), 'middle', MUTED),
    cell(COLS.kind, rowY(i) + 14, f.kind, 'start'),
    cell(COLS.mass, rowY(i) + 14, fmt(f.mass, 0), 'end'),
    cell(COLS.diameter, rowY(i) + 14, fmt(f.diameter, 1), 'end'),
  ]);

  // The side panel beside the table.
  const panel = el('g', {}, svg);
  const panelRows = [0, 1, 2, 3].map((k) => el('text', { class: 'value', x: PANEL, y: 0 }, panel));

  // --- The plots (steps 3 to 5).
  const rawAxes = axes(el('g', {}, svg), X, Y, { xLabel: 'mass in g (x₁)', yLabel: 'diameter in cm (x₂)' });
  const zAxes = axes(el('g', {}, svg), ZX, ZY, { xLabel: 'mass, standardised (z₁)', yLabel: 'diameter, standardised (z₂)' });
  const zZero = el('g', {}, svg);
  const dash = { stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '2 4' };
  el('line', { x1: ZX(0), x2: ZX(0), y1: ZY(-1.8), y2: ZY(1.8), ...dash }, zZero);
  el('line', { x1: ZX(-half), x2: ZX(half), y1: ZY(0), y2: ZY(0), ...dash }, zZero);

  const guide = { stroke: MUTED, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' };
  const guideV = el('line', guide, svg);
  const guideH = el('line', guide, svg);
  const arrow = el('line', { class: 'series-a', 'marker-end': 'url(#l2-head)' }, svg);
  const arrowLabel = el('text', { class: 'value', 'font-weight': 700, 'text-anchor': 'middle', text: 'x' }, svg);

  const legH = el('line', guide, svg);
  const legV = el('line', guide, svg);
  const pairLine = el('line', { stroke: INK, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);

  const rings = [0, 1].map(() => el('circle', { r: 9, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, svg));
  const pts = el('g', {}, svg);
  const dots = raw.map(() => el('circle', { class: 'dot', r: 5, stroke: '#0f172a', 'stroke-width': 1.5 }, pts));
  const halo = { stroke: '#1e293b', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
  const labels = raw.map((_, i) => el('text', { class: 'stage-text', fill: INK, 'text-anchor': 'middle', ...halo, text: String(i + 1) }, svg));

  // Read-out box, bottom right of the plots, where no point goes.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: '#1e293b', 'fill-opacity': 0.92, stroke: '#334155' }, box);
  const rows = [0, 1, 2].map(() => ({ sample: el('line', {}, box), text: el('text', { class: 'value' }, box), off: 12 }));

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const picker = (name, hint, value) => {
    const label = make('label');
    const vh = make('span', { className: 'vh', textContent: hint });
    const select = make('select');
    FRUIT.forEach((f, i) => select.append(make('option', { value: String(i), textContent: `${i + 1} · ${f.kind}` })));
    select.value = String(value);
    label.append(name, vh, ' ', select);
    return { label, select };
  };
  const fruitGroup = make('div');
  const pf = picker('Fruit', ' shown as a vector', START.a);
  fruitGroup.append(pf.label);
  const pairGroup = make('div');
  const pa = picker('a', ' (first fruit)', START.a);
  const pb = picker('b', ' (second fruit)', START.b);
  pairGroup.append(pa.label, pb.label);
  const viewGroup = make('div');
  const viewButtons = [['raw', 'Raw units'], ['z', 'Standardised']].map(([v, text]) => {
    const b = make('button', { type: 'button', textContent: text });
    b.dataset.view = v;
    viewGroup.append(b);
    return b;
  });
  controls.append(fruitGroup, pairGroup, viewGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Numbers that animate are lerped between two states.
  let a = START.a;
  let b = START.b;
  let view = 'z';
  let step = 0;
  let token = 0;
  let s = { tableOp: 1, featOp: 1, rowOp: 0, morph: 0, zT: 0, arrowOp: 0, pairOp: 0 };

  function target(i) {
    const plot = { tableOp: 0, featOp: 0, rowOp: 0, morph: 1, zT: 0, arrowOp: 0, pairOp: 0 };
    if (i === 0) return { ...plot, tableOp: 1, featOp: 1, morph: 0 };
    if (i === 1) return { ...plot, tableOp: 1, rowOp: 1, morph: 0 };
    if (i === 2) return { ...plot, arrowOp: 1 };
    if (i === 3) return { ...plot, pairOp: 1 };
    return { ...plot, pairOp: 1, zT: view === 'z' ? 1 : 0 };
  }

  const near = () => (view === 'z' ? nearZ : nearRaw);
  const pair = () => (step === 4 ? [near().i, near().j] : [a, b]);

  function setRow(k, text, sample, off = sample ? 40 : 12) {
    const r = rows[k];
    r.text.textContent = text || '';
    r.off = off;
    r.sample.style.display = sample ? '' : 'none';
    if (sample) {
      for (const key of ['class', 'stroke', 'stroke-width', 'stroke-dasharray', 'marker-end']) r.sample.removeAttribute(key);
      for (const [key, v] of Object.entries(sample)) r.sample.setAttribute(key, v);
    }
  }

  const lerp = (p, q, t) => p + (q - p) * t;

  function render() {
    const positions = raw.map((_, i) => {
      const x = lerp(lerp(tablePx[i][0], rawPx[i][0], s.morph), zPx[i][0], s.zT);
      const y = lerp(lerp(tablePx[i][1], rawPx[i][1], s.morph), zPx[i][1], s.zT);
      return [x, y];
    });

    // Table, its captions and the highlighted row.
    table.setAttribute('opacity', s.tableOp.toFixed(3));
    table.style.display = s.tableOp > 0.001 ? '' : 'none';
    featBand.setAttribute('opacity', s.featOp.toFixed(3));
    rowHi.setAttribute('y', rowY(a) - 6);
    rowHi.setAttribute('opacity', s.rowOp.toFixed(3));
    rowCells.forEach((cells, i) => {
      const on = s.rowOp > 0.5 && i === a;
      cells.slice(1).forEach((c) => c.setAttribute('fill', on ? '#f1f5f9' : INK));
      cells.forEach((c) => c.setAttribute('font-weight', on ? 600 : 400));
    });

    // The side panel.
    let lines = [];
    if (step === 0) {
      lines = [`${n} rows = ${n} examples`, '2 features (columns)', 'kind is the label'];
    } else if (step === 1) {
      lines = [`fruit ${a + 1} · ${FRUIT[a].kind}`, null, 'x₁ = mass in g', 'x₂ = diameter in cm'];
    }
    panelRows.forEach((t, k) => {
      t.replaceChildren();
      t.removeAttribute('style');
      t.setAttribute('y', 112 + 30 * k);
      if (step === 1 && k === 1) {
        t.style.fontSize = '18px';
        el('tspan', { 'font-weight': 700, text: 'x' }, t);
        el('tspan', { text: ` = ${vecText(raw[a])}` }, t);
      } else {
        t.textContent = lines[k] || '';
      }
    });
    panel.setAttribute('opacity', s.tableOp.toFixed(3));
    panel.style.display = s.tableOp > 0.001 ? '' : 'none';

    // Axes.
    const rawOp = s.morph * (1 - s.zT);
    rawAxes.setAttribute('opacity', rawOp.toFixed(3));
    zAxes.setAttribute('opacity', (s.morph * s.zT).toFixed(3));
    zZero.setAttribute('opacity', (0.6 * s.morph * s.zT).toFixed(3));

    // Points fly from the end of their table row to the plot.
    dots.forEach((d, i) => {
      d.setAttribute('cx', positions[i][0].toFixed(2));
      d.setAttribute('cy', positions[i][1].toFixed(2));
      d.setAttribute('opacity', clamp(s.morph * 2, 0, 1).toFixed(3));
    });
    // Each number sits on the side away from its nearest neighbour on screen,
    // and, for a fruit in the chosen pair, also away from its partner.
    const [pi, pj] = pair();
    const away = (i, q) => {
      const dx = positions[i][0] - q[0];
      const dy = positions[i][1] - q[1];
      const len = Math.hypot(dx, dy);
      return len > 0 ? [dx / len, dy / len] : [0, 0];
    };
    labels.forEach((t, i) => {
      let best = null;
      let bd = Infinity;
      positions.forEach((q, j) => {
        if (j === i) return;
        const dd = Math.hypot(q[0] - positions[i][0], q[1] - positions[i][1]);
        if (dd < bd) { bd = dd; best = q; }
      });
      let [ux, uy] = away(i, best);
      const inPair = step >= 3 && pi !== pj && (i === pi || i === pj);
      if (inPair) {
        const [vx, vy] = away(i, positions[i === pi ? pj : pi]);
        ux += vx;
        uy += vy;
        // b's dashed side runs from b to the corner level with a.
        if (i === pj) {
          const [cx, cy] = away(i, [positions[pj][0], positions[pi][1]]);
          ux += cx;
          uy += cy;
        }
      }
      const len = Math.hypot(ux, uy);
      if (len < 0.3) { ux = 0; uy = -1; } else { ux /= len; uy /= len; }
      const r = inPair || (step === 2 && i === a) ? 18 : 14;
      t.setAttribute('x', (positions[i][0] + r * ux).toFixed(2));
      t.setAttribute('y', (positions[i][1] + r * uy + 4).toFixed(2));
      t.setAttribute('opacity', clamp(s.morph * 2 - 1, 0, 1).toFixed(3));
    });

    // Step 3: the arrow from the origin to the chosen fruit, with guides to both axes.
    const [px, py] = positions[a];
    const ox = X(0);
    const oy = Y(0);
    const alen = Math.hypot(px - ox, py - oy);
    const k = alen > 0 ? (alen - 7) / alen : 0;
    arrow.setAttribute('x1', ox);
    arrow.setAttribute('y1', oy);
    arrow.setAttribute('x2', (ox + (px - ox) * k).toFixed(2));
    arrow.setAttribute('y2', (oy + (py - oy) * k).toFixed(2));
    arrow.setAttribute('opacity', s.arrowOp.toFixed(3));
    // The "x" label sits above the middle of the arrow, on its upper-left side.
    const nx = -(py - oy) / (alen || 1);
    const ny = (px - ox) / (alen || 1);
    const sgn = ny < 0 ? 1 : -1;
    arrowLabel.setAttribute('x', ((ox + px) / 2 + sgn * 16 * nx).toFixed(2));
    arrowLabel.setAttribute('y', ((oy + py) / 2 + sgn * 16 * ny + 4).toFixed(2));
    arrowLabel.setAttribute('opacity', s.arrowOp.toFixed(3));
    guideV.setAttribute('x1', px);
    guideV.setAttribute('x2', px);
    guideV.setAttribute('y1', py);
    guideV.setAttribute('y2', oy);
    guideH.setAttribute('x1', px);
    guideH.setAttribute('x2', ox);
    guideH.setAttribute('y1', py);
    guideH.setAttribute('y2', py);
    guideV.setAttribute('opacity', s.arrowOp.toFixed(3));
    guideH.setAttribute('opacity', s.arrowOp.toFixed(3));

    // Steps 4 and 5: the line between two fruit, and its two gaps.
    const [ax, ay] = positions[pi];
    const [bx, by] = positions[pj];
    pairLine.setAttribute('x1', ax);
    pairLine.setAttribute('y1', ay);
    pairLine.setAttribute('x2', bx);
    pairLine.setAttribute('y2', by);
    pairLine.setAttribute('opacity', s.pairOp.toFixed(3));
    legH.setAttribute('x1', ax);
    legH.setAttribute('y1', ay);
    legH.setAttribute('x2', bx);
    legH.setAttribute('y2', ay);
    legV.setAttribute('x1', bx);
    legV.setAttribute('y1', ay);
    legV.setAttribute('x2', bx);
    legV.setAttribute('y2', by);
    legH.setAttribute('opacity', s.pairOp.toFixed(3));
    legV.setAttribute('opacity', s.pairOp.toFixed(3));

    const ringed = step === 2 ? [a] : step >= 3 ? [pi, pj] : [];
    rings.forEach((r, k2) => {
      const i = ringed[k2];
      r.style.display = i === undefined ? 'none' : '';
      if (i === undefined) return;
      r.setAttribute('cx', positions[i][0].toFixed(2));
      r.setAttribute('cy', positions[i][1].toFixed(2));
      r.setAttribute('opacity', s.morph.toFixed(3));
    });

    // Read-out box.
    const lineSample = { stroke: INK, 'stroke-width': 2 };
    let rowsOn = 0;
    if (step === 2) {
      setRow(0, '', { class: 'series-a', 'marker-end': 'url(#l2-head)' });
      el('tspan', { 'font-weight': 700, text: 'x' }, rows[0].text);
      el('tspan', { text: ` = ${vecText(raw[a])}` }, rows[0].text);
      setRow(1, fruitName(a), null, 40);
      rowsOn = 2;
    } else if (step === 3) {
      const dm = Math.abs(raw[a][0] - raw[b][0]);
      const dd = Math.abs(raw[a][1] - raw[b][1]);
      setRow(0, `a = fruit ${a + 1}, b = fruit ${b + 1}`);
      setRow(1, `gaps: ${fmt(dm, 0)} g and ${fmt(dd, 1)} cm`, guide);
      setRow(2, `distance: ${fmt(distance(raw[a], raw[b]), 2)}`, lineSample);
      rows[0].off = 40;
      rowsOn = 3;
    } else if (step === 4) {
      const onZ = view === 'z';
      setRow(0, 'nearest pair', null, 40);
      setRow(1, `raw units: ${nearRaw.i + 1} & ${nearRaw.j + 1}, ${fmt(nearRaw.d, 2)}`, onZ ? null : lineSample, 40);
      setRow(2, `standardised: ${nearZ.i + 1} & ${nearZ.j + 1}, ${fmt(nearZ.d, 2)}`, onZ ? lineSample : null, 40);
      rowsOn = 3;
    }
    rows.forEach((r, k2) => {
      const on = k2 < rowsOn;
      r.text.style.display = on ? '' : 'none';
      if (!on) r.sample.style.display = 'none';
    });
    let w = 250;
    try {
      const lens = rows.slice(0, rowsOn).map((r) => r.text.getComputedTextLength());
      if (lens.length && lens.every((len) => len > 0)) w = Math.max(...rows.slice(0, rowsOn).map((r, k2) => r.off + lens[k2])) + 12;
    } catch {
      // Keep the fixed width.
    }
    const h = 16 + BOX.row * rowsOn;
    const bx0 = 580 - 8 - w;
    const by0 = 290 - 8 - h;
    boxRect.setAttribute('x', bx0.toFixed(1));
    boxRect.setAttribute('y', by0);
    boxRect.setAttribute('width', w.toFixed(1));
    boxRect.setAttribute('height', h);
    rows.forEach((r, k2) => {
      const cy = by0 + 8 + BOX.row * k2 + BOX.row / 2;
      r.text.setAttribute('x', (bx0 + r.off).toFixed(1));
      r.text.setAttribute('y', cy + 5);
      r.sample.setAttribute('x1', (bx0 + 12).toFixed(1));
      r.sample.setAttribute('x2', (bx0 + 32).toFixed(1));
      r.sample.setAttribute('y1', cy);
      r.sample.setAttribute('y2', cy);
    });
    box.setAttribute('opacity', (rowsOn ? s.morph : 0).toFixed(3));
    box.style.display = rowsOn ? '' : 'none';

    viewButtons.forEach((btn) => btn.setAttribute('aria-pressed', String(btn.dataset.view === view)));
  }

  function animateTo(to, ms, from = s) {
    const my = ++token;
    const start = { ...from };
    return tween(ms, (t) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * t;
      s = next;
      render();
    }, { reduced });
  }

  function jumpTo(to) {
    token++;
    s = { ...to };
    render();
  }

  // --- Words for the live region.
  const pairText = (i, j) => {
    if (i === j) return `${fruitName(i)} with itself: distance 0.`;
    const dm = Math.abs(raw[i][0] - raw[j][0]);
    const dd = Math.abs(raw[i][1] - raw[j][1]);
    return `${fruitName(i)} and ${fruitName(j)}: gaps of ${fmt(dm, 0)} g and ${fmt(dd, 1)} cm, distance ${fmt(distance(raw[i], raw[j]), 2)}.`;
  };
  const nearText = () =>
    `In raw units the nearest pair is ${fruitName(nearRaw.i)} and ${fruitName(nearRaw.j)}, distance ${fmt(nearRaw.d, 2)}. ` +
    `Standardised, it is ${fruitName(nearZ.i)} and ${fruitName(nearZ.j)}, distance ${fmt(nearZ.d, 2)}. ` +
    `Showing ${view === 'z' ? 'standardised' : 'raw'} units.`;
  const summaries = [
    () => `A table of ${n} fruit. Each row is one example; the mass and diameter columns are the features, and kind is the label.`,
    () => `Fruit ${a + 1}, ${FRUIT[a].kind}, as a vector: x = ${vecText(raw[a])}.`,
    () => `${fruitName(a)} drawn as an arrow from the origin to the point ${vecText(raw[a])}.`,
    () => pairText(a, b),
    nearText,
  ];

  // --- Interaction.
  const setA = (i) => {
    a = i;
    pf.select.value = String(i);
    pa.select.value = String(i);
    render();
    if (step === 1) announce(summaries[1]());
    else if (step === 2) announce(summaries[2]());
    else announce(pairText(a, b));
  };
  pf.select.addEventListener('change', () => setA(Number(pf.select.value)));
  pa.select.addEventListener('change', () => setA(Number(pa.select.value)));
  pb.select.addEventListener('change', () => {
    b = Number(pb.select.value);
    render();
    announce(pairText(a, b));
  });
  viewButtons.forEach((btn) => btn.addEventListener('click', () => {
    if (view === btn.dataset.view) return;
    view = btn.dataset.view;
    announce(nearText());
    if (step === 4) animateTo(target(4), 700);
    else render();
  }));

  // Sentences on the page that quote values from the data (hidden until filled).
  const values = {
    'raw-pair': `${fruitName(nearRaw.i)} and ${fruitName(nearRaw.j)}`,
    'z-pair': `${fruitName(nearZ.i)} and ${fruitName(nearZ.j)}`,
    'mu-mass': fmt(mu[0], 1),
    'sigma-mass': fmt(sigma[0], 1),
    'mu-diameter': fmt(mu[1], 2),
    'sigma-diameter': fmt(sigma[1], 2),
  };
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const v = values[span.dataset.value];
    if (v !== undefined) span.textContent = v;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in values)) p.hidden = false;
  }

  render();
  // Text widths change once the mono font arrives.
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 5,
    show(i, animate) {
      const prev = step;
      step = i;
      showGroup(fruitGroup, i === 1 || i === 2);
      showGroup(pairGroup, i === 3);
      showGroup(viewGroup, i === 4);
      // Arriving at the last step from earlier shows the standardised view.
      if (i === 4 && prev !== 4) view = 'z';
      const to = target(i);
      announce(summaries[i]());
      if (!animate) {
        jumpTo(to);
        return Promise.resolve();
      }
      const crossing = (prev <= 1) !== (i <= 1);
      return animateTo(to, crossing || i === 4 ? 900 : 500);
    },
  };
}
