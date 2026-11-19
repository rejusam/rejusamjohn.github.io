// The landing page's live figure: gradient descent learning the slope w of
// the line y = w·x through eight fixed points. The maths at the top is pure
// and tested in Node; initHubDemo draws it. Every number shown is computed
// here, in the browser, from DATA.

// Eight points, x centred on 0. y is 1.2·x plus a fixed nudge per point, so
// the best slope is close to 1.2 but not exactly 1.2.
export const DATA = (() => {
  const x = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
  const nudge = [0.4, -0.6, 0.3, 0.5, -0.4, 0.2, -0.7, 0.3];
  return { x, y: x.map((xi, i) => 1.2 * xi + nudge[i]) };
})();

// Mean squared error of the line y = w·x.
export function lossAt(w, { x, y } = DATA) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (y[i] - w * x[i]) ** 2;
  return s / x.length;
}

// dL/dw = -(2/n) Σ x (y - w x).
export function gradAt(w, { x, y } = DATA) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * (y[i] - w * x[i]);
  return (-2 * s) / x.length;
}

// The least-squares slope for a line through the origin: Σxy / Σx².
export function bestW({ x, y } = DATA) {
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < x.length; i++) {
    sxy += x[i] * y[i];
    sxx += x[i] * x[i];
  }
  return sxy / sxx;
}

// Plain gradient descent from w0: w ← w − η·dL/dw. Returns every step,
// step 0 being the start.
export function descend({ w0 = -1, eta = 0.03, steps = 18, data = DATA } = {}) {
  const out = [];
  let w = w0;
  for (let k = 0; k <= steps; k++) {
    out.push({ k, w, loss: lossAt(w, data), grad: gradAt(w, data) });
    w -= eta * gradAt(w, data);
  }
  return out;
}

// ---------- Drawing ----------

const NS = 'http://www.w3.org/2000/svg';
const W_MIN = -1.5;
const W_MAX = 3;

function el(name, attrs = {}, parent = null) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (parent) parent.appendChild(n);
  return n;
}

const fix = (v, d = 2) => (Object.is(Math.round(v * 10 ** d), -0) ? 0 : v).toFixed(d).replace('-', '−');

export function initHubDemo(figure, { reduced = false } = {}) {
  const svg = figure && figure.querySelector('svg');
  if (!svg) return;
  const read = figure.querySelector('.hub-demo-read');
  const replay = figure.querySelector('.hub-demo-replay');

  const run = descend();
  const wStar = bestW();
  const lossMax = lossAt(W_MIN);

  // Left panel: the loss curve. Right panel: the points and the line.
  const L = { x0: 34, x1: 300, y0: 22, y1: 224 };
  const R = { x0: 340, x1: 548, y0: 22, y1: 224 };
  const lx = (w) => L.x0 + ((w - W_MIN) / (W_MAX - W_MIN)) * (L.x1 - L.x0);
  const ly = (loss) => L.y1 - (loss / lossMax) * (L.y1 - L.y0);
  const span = 4.5;
  const rx = (x) => (R.x0 + R.x1) / 2 + (x / span) * ((R.x1 - R.x0) / 2);
  const ry = (y) => (R.y0 + R.y1) / 2 - (y / span) * ((R.y1 - R.y0) / 2);

  svg.replaceChildren();
  el('title', {}, svg).textContent = 'Gradient descent finding the slope of a line';

  // Loss curve.
  const gL = el('g', {}, svg);
  el('line', { x1: L.x0, y1: L.y1, x2: L.x1, y2: L.y1, class: 'hd-axis' }, gL);
  el('line', { x1: lx(wStar), y1: L.y1, x2: lx(wStar), y2: ly(lossAt(wStar)), class: 'hd-guide' }, gL);
  let d = '';
  for (let i = 0; i <= 90; i++) {
    const w = W_MIN + ((W_MAX - W_MIN) * i) / 90;
    d += `${i ? 'L' : 'M'}${lx(w).toFixed(1)} ${ly(lossAt(w)).toFixed(1)}`;
  }
  el('path', { d, class: 'hd-curve' }, gL);
  el('text', { x: L.x0, y: L.y0 - 6, class: 'hd-label' }, gL).textContent = 'loss';
  el('text', { x: L.x1, y: L.y1 + 18, class: 'hd-label', 'text-anchor': 'end' }, gL).textContent = 'slope w';
  el('text', { x: lx(wStar), y: L.y1 + 18, class: 'hd-label', 'text-anchor': 'middle' }, gL).textContent = `best ${fix(wStar)}`;
  const trail = el('g', {}, gL);
  const ball = el('circle', { r: 7, class: 'hd-ball' }, gL);

  // Points and line.
  const gR = el('g', {}, svg);
  el('line', { x1: R.x0, y1: ry(0), x2: R.x1, y2: ry(0), class: 'hd-axis' }, gR);
  el('line', { x1: rx(0), y1: R.y0, x2: rx(0), y2: R.y1, class: 'hd-axis' }, gR);
  const resid = el('g', {}, gR);
  const line = el('line', { class: 'hd-fit' }, gR);
  DATA.x.forEach((x, i) => el('circle', { cx: rx(x), cy: ry(DATA.y[i]), r: 4.5, class: 'hd-pt' }, gR));
  el('text', { x: R.x0, y: R.y0 - 6, class: 'hd-label' }, gR).textContent = 'the line y = w·x';

  const residLines = DATA.x.map(() => el('line', { class: 'hd-resid' }, resid));

  function draw(k) {
    const s = run[k];
    ball.setAttribute('cx', lx(s.w).toFixed(1));
    ball.setAttribute('cy', ly(s.loss).toFixed(1));
    trail.replaceChildren();
    for (let j = 0; j < k; j++) {
      el('circle', { cx: lx(run[j].w).toFixed(1), cy: ly(run[j].loss).toFixed(1), r: 3.5, class: 'hd-step' }, trail);
    }
    const xa = -span;
    const xb = span;
    line.setAttribute('x1', rx(xa));
    line.setAttribute('y1', ry(Math.max(-span, Math.min(span, s.w * xa))));
    line.setAttribute('x2', rx(xb));
    line.setAttribute('y2', ry(Math.max(-span, Math.min(span, s.w * xb))));
    // Keep the line inside the panel: clip its ends to |y| ≤ span.
    if (Math.abs(s.w * xb) > span) {
      const xe = span / Math.abs(s.w);
      line.setAttribute('x1', rx(-xe));
      line.setAttribute('y1', ry(-Math.sign(s.w) * span));
      line.setAttribute('x2', rx(xe));
      line.setAttribute('y2', ry(Math.sign(s.w) * span));
    }
    DATA.x.forEach((x, i) => {
      const r = residLines[i];
      r.setAttribute('x1', rx(x));
      r.setAttribute('x2', rx(x));
      r.setAttribute('y1', ry(DATA.y[i]));
      r.setAttribute('y2', ry(Math.max(-span, Math.min(span, s.w * x))));
    });
    if (read) {
      read.querySelector('[data-k]').textContent = String(s.k);
      read.querySelector('[data-w]').textContent = fix(s.w);
      read.querySelector('[data-loss]').textContent = fix(s.loss);
    }
  }

  let timer = null;
  function play() {
    clearTimeout(timer);
    if (reduced) {
      draw(run.length - 1);
      return;
    }
    let k = 0;
    const tick = () => {
      draw(k);
      if (k < run.length - 1) {
        k += 1;
        timer = setTimeout(tick, k < 4 ? 520 : 300);
      }
    };
    tick();
  }

  if (replay) replay.addEventListener('click', play);

  // Start when the figure first comes into view.
  draw(0);
  figure.classList.add('is-live');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        play();
      }
    }, { threshold: 0.4 });
    io.observe(figure);
  } else {
    play();
  }
}
