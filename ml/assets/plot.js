// SVG plot helpers for the ML Explorer. No dependencies.
// DOM and window are only touched inside function bodies so this module
// can be imported in Node for tests.

const SVG_NS = 'http://www.w3.org/2000/svg';

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function niceStep(raw) {
  const exp = Math.floor(Math.log10(raw));
  const f = raw / Math.pow(10, exp);
  const nice = f <= 1 + 1e-9 ? 1 : f <= 2 + 1e-9 ? 2 : f <= 5 + 1e-9 ? 5 : 10;
  return nice * Math.pow(10, exp);
}

export function linearScale([d0, d1], [r0, r1]) {
  const span = d1 - d0;
  const s = (x) => r0 + ((x - d0) / span) * (r1 - r0);
  s.invert = (px) => d0 + ((px - r0) / (r1 - r0)) * span;
  s.domain = [d0, d1];
  s.range = [r0, r1];
  s.ticks = (n = 5) => {
    const lo = Math.min(d0, d1);
    const hi = Math.max(d0, d1);
    if (!(hi > lo) || n < 1) return [lo];
    const step = niceStep((hi - lo) / n);
    const first = Math.ceil(lo / step - 1e-9);
    const last = Math.floor(hi / step + 1e-9);
    const out = [];
    for (let k = first; k <= last; k++) {
      out.push(Number((k * step).toPrecision(12)));
    }
    return out;
  };
  return s;
}

export function el(tag, attrs, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null) continue;
    if (k === 'text') node.textContent = String(v);
    else node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

// Tick label: up to six significant figures, with a true minus sign (U+2212).
export function fmt(v) {
  const s = String(Number(v.toPrecision(6)));
  return s.startsWith('-') ? '\u2212' + s.slice(1) : s;
}

export function axes(parent, xs, ys, { xLabel, yLabel, ticks = 5 } = {}) {
  const g = el('g', { class: 'axes' }, parent);
  const [x0, x1] = xs.range;
  const [y0, y1] = ys.range;
  const line = '#334155';
  const muted = '#94a3b8';
  const font = "'JetBrains Mono', monospace";
  el('line', { x1: x0, x2: x1, y1: y0, y2: y0, stroke: line }, g);
  el('line', { x1: x0, x2: x0, y1: y0, y2: y1, stroke: line }, g);
  for (const t of xs.ticks(ticks)) {
    const px = xs(t);
    el('line', { x1: px, x2: px, y1: y0, y2: y0 + 5, stroke: line }, g);
    el('text', { x: px, y: y0 + 18, 'text-anchor': 'middle', fill: muted, 'font-family': font, 'font-size': 11, text: fmt(t) }, g);
  }
  for (const t of ys.ticks(ticks)) {
    const py = ys(t);
    el('line', { x1: x0 - 5, x2: x0, y1: py, y2: py, stroke: line }, g);
    el('text', { x: x0 - 9, y: py + 4, 'text-anchor': 'end', fill: muted, 'font-family': font, 'font-size': 11, text: fmt(t) }, g);
  }
  if (xLabel) {
    el('text', { x: (x0 + x1) / 2, y: y0 + 38, 'text-anchor': 'middle', fill: muted, 'font-family': font, 'font-size': 12, text: xLabel }, g);
  }
  if (yLabel) {
    const cx = x0 - 44;
    const cy = (y0 + y1) / 2;
    el('text', { x: cx, y: cy, transform: `rotate(-90 ${cx} ${cy})`, 'text-anchor': 'middle', fill: muted, 'font-family': font, 'font-size': 12, text: yLabel }, g);
  }
  return g;
}

export function pathFrom(points, xs, ys) {
  let d = '';
  let pen = false;
  for (const [x, y] of points) {
    const px = xs(x);
    const py = ys(y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${Number(px.toFixed(2))},${Number(py.toFixed(2))}`;
    pen = true;
  }
  return d;
}

export function prefersReducedMotion() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function tween(ms, onFrame, { reduced = false } = {}) {
  if (reduced || !(ms > 0)) {
    onFrame(1);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const start = performance.now();
    const step = (now) => {
      const t = clamp((now - start) / ms, 0, 1);
      onFrame(easeInOutCubic(t));
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
}

// node is an SVG element positioned by the caller; onDrag receives the
// pointer position in the SVG's user units, clamped to bounds.
export function draggable(node, { onDrag, bounds }) {
  const { x0, x1, y0, y1 } = bounds;
  const toUser = (e) => {
    const svg = node.ownerSVGElement || node;
    const ctm = svg.getScreenCTM();
    if (!ctm) return [e.clientX, e.clientY];
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(ctm.inverse());
    return [p.x, p.y];
  };
  const move = (px, py) => onDrag(clamp(px, Math.min(x0, x1), Math.max(x0, x1)), clamp(py, Math.min(y0, y1), Math.max(y0, y1)));
  let active = false;
  node.style.touchAction = 'none';
  node.addEventListener('pointerdown', (e) => {
    active = true;
    node.setPointerCapture(e.pointerId);
    e.preventDefault();
    move(...toUser(e));
  });
  node.addEventListener('pointermove', (e) => {
    if (active) move(...toUser(e));
  });
  const end = (e) => {
    active = false;
    if (node.hasPointerCapture && node.hasPointerCapture(e.pointerId)) node.releasePointerCapture(e.pointerId);
  };
  node.addEventListener('pointerup', end);
  node.addEventListener('pointercancel', end);
  node.addEventListener('keydown', (e) => {
    const dx = (Math.abs(x1 - x0)) * 0.01;
    const dy = (Math.abs(y1 - y0)) * 0.01;
    const cx = Number(node.getAttribute('cx') ?? node.getAttribute('x') ?? 0);
    const cy = Number(node.getAttribute('cy') ?? node.getAttribute('y') ?? 0);
    let nx = cx;
    let ny = cy;
    if (e.key === 'ArrowLeft') nx -= dx;
    else if (e.key === 'ArrowRight') nx += dx;
    else if (e.key === 'ArrowUp') ny -= dy;
    else if (e.key === 'ArrowDown') ny += dy;
    else return;
    e.preventDefault();
    move(nx, ny);
  });
}

// ---------- Classification plots ----------

// Equilateral triangle centred (centroid) on (cx, cy) with the area of a circle of radius r.
export function trianglePoints(cx, cy, r) {
  const s = r * Math.sqrt((4 * Math.PI) / Math.sqrt(3));
  const h = (Math.sqrt(3) / 2) * s;
  return [[cx, cy - (2 * h) / 3], [cx - s / 2, cy + h / 3], [cx + s / 2, cy + h / 3]];
}

// Class 0 circle, class 1 triangle, class 2 square (all the same area), null unlabelled.
export function classMarker(parent, cx, cy, cls, { r = 5 } = {}) {
  if (cls === 1) {
    const d = trianglePoints(cx, cy, r).map(([x, y], k) => `${k ? 'L' : 'M'}${Number(x.toFixed(2))},${Number(y.toFixed(2))}`).join('') + 'Z';
    return el('path', { class: 'mk mk-1', d }, parent);
  }
  if (cls === 2) {
    const s = r * Math.sqrt(Math.PI);
    return el('rect', { class: 'mk mk-2', x: cx - s / 2, y: cy - s / 2, width: s, height: s }, parent);
  }
  return el('circle', { class: cls === 0 ? 'mk mk-0' : 'mk mk-u', cx, cy, r }, parent);
}

// nx × ny cells over the plot, row j = 0 at the top. (x, y) is the cell centre in data units.
export function regionCells(nx, ny, xs, ys) {
  const [dx0, dx1] = xs.domain;
  const [dy0, dy1] = ys.domain;
  const [px0, px1] = xs.range;
  const top = Math.min(...ys.range);
  const w = Math.abs(px1 - px0) / nx;
  const h = Math.abs(ys.range[1] - ys.range[0]) / ny;
  const left = Math.min(px0, px1);
  const yHi = ys(dy0) < ys(dy1) ? dy0 : dy1;
  const yLo = yHi === dy0 ? dy1 : dy0;
  const cells = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      cells.push({
        i, j,
        x: dx0 + ((i + 0.5) / nx) * (dx1 - dx0),
        y: yHi + ((j + 0.5) / ny) * (yLo - yHi),
        px: left + i * w,
        py: top + j * h,
        w, h,
      });
    }
  }
  return cells;
}

// Unit edges between neighbouring cells of different class, in grid units.
// grid[j][i]; a vertical edge sits at x = i + 1, a horizontal one at y = j + 1.
export function boundarySegments(grid) {
  const segs = [];
  const ny = grid.length;
  for (let j = 0; j < ny; j++) {
    const row = grid[j];
    for (let i = 0; i + 1 < row.length; i++) {
      if (row[i] !== row[i + 1]) segs.push([[i + 1, j], [i + 1, j + 1]]);
    }
  }
  for (let j = 0; j + 1 < ny; j++) {
    for (let i = 0; i < grid[j].length; i++) {
      if (grid[j][i] !== grid[j + 1][i]) segs.push([[i, j + 1], [i + 1, j + 1]]);
    }
  }
  return segs;
}

// In prob mode v is p̂ for class 1: the side is p̂ ≥ 0.5, the tint grows with |2p̂ − 1|.
export function regionFill(v, mode = 'class') {
  if (mode === 'prob') return { cls: v >= 0.5 ? 1 : 0, strength: Math.abs(2 * v - 1) };
  return { cls: v, strength: 1 };
}

// Tint per class, as a percentage of the class colour over the plot surface.
// Kept low so muted 11px text stays at 4.5:1 over any region. Matches ml.css.
const TINT = [10, 14, 12];

export function regionMap(parent, xs, ys, f, { nx = 60, ny = 40, mode = 'class' } = {}) {
  const g = el('g', { class: 'region' }, parent);
  const cells = regionCells(nx, ny, xs, ys);
  // A little overlap hides hairline seams between opaque cells.
  const rects = cells.map((c) => el('rect', { x: c.px, y: c.py, width: c.w + 0.5, height: c.h + 0.5 }, g));
  const line = el('path', { class: 'boundary' }, g);
  const update = (fn) => {
    const grid = Array.from({ length: ny }, () => new Array(nx));
    cells.forEach((c, k) => {
      const { cls, strength } = regionFill(fn(c.x, c.y), mode);
      grid[c.j][c.i] = cls;
      const r = rects[k];
      r.setAttribute('class', `r-${cls}`);
      if (mode === 'prob') {
        const pct = Math.round(TINT[cls] * strength);
        r.style.fill = `color-mix(in srgb, var(--mk-${cls}) ${pct}%, var(--region-base))`;
      }
    });
    const { w, h } = cells[0];
    const left = cells[0].px;
    const top = cells[0].py;
    const P = (v) => Number(v.toFixed(2));
    line.setAttribute('d', boundarySegments(grid).map(([[a, b], [c, d]]) => `M${P(left + a * w)},${P(top + b * h)}L${P(left + c * w)},${P(top + d * h)}`).join(''));
  };
  update(f);
  return { update, g };
}

// Leaves get x = 0, 1, 2, ... in left-to-right order; a split sits midway over its two children.
export function treeLayout(root) {
  const nodes = [];
  const links = [];
  let next = 0;
  let depth = 0;
  const walk = (node, d) => {
    const k = nodes.length;
    const p = { node, depth: d, x: 0 };
    nodes.push(p);
    depth = Math.max(depth, d);
    if (node.leaf) {
      p.x = next++;
    } else {
      const a = walk(node.left, d + 1);
      const b = walk(node.right, d + 1);
      links.push([k, a], [k, b]);
      p.x = (nodes[a].x + nodes[b].x) / 2;
    }
    return k;
  };
  walk(root, 0);
  return { nodes, links, leaves: next, depth };
}

