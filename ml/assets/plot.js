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
