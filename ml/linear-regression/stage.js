// Lesson 6, "Linear regression": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data. Lesson 7 imports regressionData.

import { mulberry32, uniform, randn, olsFit, mean } from '../assets/maths.js';
import { linearScale, clamp, el, axes, pathFrom, fmt as tickFmt, tween, draggable } from '../assets/plot.js';

export const SEED = 11;
export const N = 25;
export const Y_LO = 0;
export const Y_HI = 24;
// The two handles sit at these x values and move up and down only.
export const HX = [1, 9];

// 25 made-up points. For each point in turn: x ~ Uniform(0, 10) is drawn
// first, then y = 4 + 1.5x + 2 × a standard normal draw.
export function regressionData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = uniform(rng, 0, 10);
    xs.push(x);
    ys.push(4 + 1.5 * x + 2 * randn(rng));
  }
  return { xs, ys };
}

// The least-squares line ŷ = a + b·x, straight from olsFit.
export function fitLine(xs, ys) {
  const r = olsFit(xs, ys);
  if (!r.ok) throw new Error('lesson data has no spread in x');
  return { a: r.intercept, b: r.slope };
}

export const predict = ({ a, b }, x) => a + b * x;

// The line through the two handles, (HX[0], y1) and (HX[1], y2).
export function lineFromHandles(y1, y2) {
  const b = (y2 - y1) / (HX[1] - HX[0]);
  return { a: y1 - b * HX[0], b };
}

// The reader's starting line: deliberately poor, it slopes the wrong way.
export const START_HANDLES = [15, 9];
export const START = lineFromHandles(...START_HANDLES);

export const residualsOf = (xs, ys, line) => ys.map((y, i) => y - predict(line, xs[i]));

// SSE = Σ eᵢ², the sum of squared residuals.
export const sse = (xs, ys, line) => residualsOf(xs, ys, line).reduce((s, e) => s + e * e, 0);

// SST = Σ (yᵢ − ȳ)², the SSE of the flat line at the mean of y.
export function sst(ys) {
  const m = mean(ys);
  return ys.reduce((s, y) => s + (y - m) ** 2, 0);
}

export const rSquared = (xs, ys, line) => 1 - sse(xs, ys, line) / sst(ys);

// Pearson correlation r = Sxy / √(Sxx · Syy).
export function correlation(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  return sxy / Math.sqrt(sxx * syy);
}

// One frame of "solve": t = 0 gives the start line, and t ≥ 1 gives the
// least-squares line itself, not a rounded copy of it.
export function solveFrame(from, to, t) {
  if (t >= 1) return { a: to.a, b: to.b };
  if (t <= 0) return { a: from.a, b: from.b };
  return { a: from.a + (to.a - from.a) * t, b: from.b + (to.b - from.b) * t };
}

// Step 5's slider: x and x + 1 both stay inside the data.
export const X0_LO = (xs) => Math.ceil(Math.min(...xs) * 100) / 100;
export const X0_HI = (xs) => Math.floor((Math.max(...xs) - 1) * 100) / 100;
export const X0_START = 3;

// Fixed decimals with a true minus sign.
export function fmt(v, dp = 2) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

// The read-out rows for each step, as shown on the stage.
// state: { xs, ys, user (the reader's line), line (least squares), x0, from }.
export function readout(step, { xs, ys, user, line, x0, from }) {
  const n = xs.length;
  if (step === 0) return [`n = ${n}`, `mean of x: ${fmt(mean(xs))}`, `mean of y: ${fmt(mean(ys))}`];
  if (step === 1) return ['your line', `a = ${fmt(user.a)}`, `b = ${fmt(user.b)}`, `SSE = ${fmt(sse(xs, ys, user))}`];
  if (step === 2) {
    const s = sse(xs, ys, user);
    return [`SSE = Σe² = ${fmt(s)}`, `MSE = SSE/${n} = ${fmt(s / n)}`];
  }
  if (step === 3) {
    return [`your SSE: ${fmt(sse(xs, ys, user))}`, `least-sq. SSE: ${fmt(sse(xs, ys, line))}`, `intercept: ${fmt(line.a)}`, `slope: ${fmt(line.b)}`];
  }
  if (step === 4) {
    return [`ŷ at ${fmt(x0)}: ${fmt(predict(line, x0))}`, `ŷ at ${fmt(x0 + 1)}: ${fmt(predict(line, x0 + 1))}`, `change: ${fmt(line.b)} (slope)`];
  }
  const s = sse(xs, ys, line);
  const t = sst(ys);
  return [from === 'mean' ? 'from the mean' : 'from the line', `SST = ${fmt(t)}`, `SSE = ${fmt(s)}`, `R² = ${fmt(1 - s / t)}`];
}

// Values quoted in the page text, all from the lesson's data.
export function pageValues(xs, ys) {
  const line = fitLine(xs, ys);
  const s = sse(xs, ys, line);
  const t = sst(ys);
  const r2 = 1 - s / t;
  return {
    n: String(xs.length),
    'a-hat': fmt(line.a),
    'b-hat': fmt(line.b),
    sse: fmt(s),
    mse: fmt(s / xs.length),
    sst: fmt(t),
    r2: fmt(r2),
    'r2-pct': String(Math.round(r2 * 100)),
    r: fmt(correlation(xs, ys)),
    'start-sse': fmt(sse(xs, ys, START)),
    'x-min': fmt(Math.min(...xs)),
    'x-max': fmt(Math.max(...xs)),
    xbar: fmt(mean(xs)),
    ybar: fmt(mean(ys)),
  };
}

// Words for numbers in the live region.
const say2 = (v) => {
  const s = fmt(v);
  return s.startsWith('−') ? `minus ${s.slice(1)}` : s;
};

// ---------------------------------------------------------------- the stage

const INK = '#e2e8f0';
const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';
const HALO = { stroke: SURFACE, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };

// Scatter plot on the left; the side panel on the right holds the read-out
// box and, in steps 4 and 6, a pair of bars.
export const PX0 = 56;
export const PX1 = 372;
// The x axis runs past the data on both sides, so step 5 can show the line
// continuing (dashed) where no data back it up.
export const X_DOMAIN = [-2, 12];

// Width in pixels of the dashed parts of the line in step 5, left and right.
export function extrapolationPx(xs) {
  const px = (x) => PX0 + ((x - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * (PX1 - PX0);
  return [px(Math.min(...xs)) - px(X_DOMAIN[0]), px(X_DOMAIN[1]) - px(Math.max(...xs))];
}
const PY0 = 290;
const PY1 = 20;
const SX0 = 438;
const SX1 = 578;
const SY0 = 290;
const SY1 = 162;
const BOX = { x: 398, y: 8, row: 20 };
const R = 4.5;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const { xs, ys } = regressionData();
  const ols = fitLine(xs, ys);
  const n = xs.length;
  const xbar = mean(xs);
  const ybar = mean(ys);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const SST = sst(ys);
  const SSE = sse(xs, ys, ols);
  let hy = START_HANDLES.slice();
  let user = lineFromHandles(...hy);
  let cur = { ...user };
  let x0 = X0_START;
  let from = 'line';

  const X = linearScale(X_DOMAIN, [PX0, PX1]);
  const Y = linearScale([Y_LO, Y_HI], [PY0, PY1]);
  const unit = PY0 - Y(1);
  const lineEnds = (l) => ({ x1: X(X_DOMAIN[0]), x2: X(X_DOMAIN[1]), y1: Y(predict(l, X_DOMAIN[0])), y2: Y(predict(l, X_DOMAIN[1])) });

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Scatter plot of 25 points with a straight line through them, and a read-out box that changes with each step.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l6-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);
  const clipSq = el('clipPath', { id: 'l6-sq' }, defs);
  el('rect', { x: PX0, y: PY1, width: 392 - PX0, height: PY0 - PY1 }, clipSq);

  // Step 5: the range the data covers.
  const band = el('rect', { x: X(xMin), width: X(xMax) - X(xMin), y: PY1, height: PY0 - PY1, fill: LINE, opacity: 0 }, svg);
  const bandLabel = el('text', { class: 'stage-text', x: (X(xMin) + X(xMax)) / 2, y: PY0 - 8, 'text-anchor': 'middle', text: 'the range of the data' }, svg);

  axes(svg, X, Y, { xLabel: 'x', yLabel: 'y' });

  // --- Main plot layers.
  const candG = el('g', { 'clip-path': 'url(#l6-plot)' }, svg);
  const CAND = [0.6, 1.4, 2.4];
  for (const b of CAND) {
    const l = { a: ybar - b * xbar, b };
    el('line', { ...lineEnds(l), stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '5 5' }, candG);
  }
  const squaresG = el('g', { 'clip-path': 'url(#l6-sq)' }, svg);
  const sq = xs.map(() => el('rect', { fill: CYAN, 'fill-opacity': 0.06, stroke: CYAN, 'stroke-width': 1 }, squaresG));
  const meanG = el('g', {}, svg);
  el('line', { x1: PX0, x2: PX1, y1: Y(ybar), y2: Y(ybar), stroke: VIOLET, 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }, meanG);
  const sticksG = el('g', {}, svg);
  const sticks = xs.map(() => el('line', { 'stroke-linecap': 'round', 'stroke-width': 1.5 }, sticksG));
  const lineG = el('g', { 'clip-path': 'url(#l6-plot)' }, svg);
  const ghost = el('line', { stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '6 5' }, lineG);
  // In step 5 the line is solid over the data and dashed outside it.
  const extL = el('line', { stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '4 4' }, lineG);
  const extR = el('line', { stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '4 4' }, lineG);
  const main = el('line', { stroke: CYAN, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, lineG);
  // A small key in the top-left corner of the plot (steps 4 to 6).
  const keyG = el('g', {}, svg);
  el('rect', { x: PX0 + 6, y: PY1 + 2, width: 132, height: 44, rx: 4, fill: SURFACE, 'fill-opacity': 0.9 }, keyG);
  el('line', { x1: PX0 + 14, x2: PX0 + 36, y1: PY1 + 16, y2: PY1 + 16, stroke: CYAN, 'stroke-width': 2.4 }, keyG);
  const mainLabel = el('text', { class: 'stage-text', x: PX0 + 42, y: PY1 + 20, text: 'least squares' }, keyG);
  mainLabel.style.fill = CYAN;
  const ghostKey = el('g', {}, keyG);
  el('line', { x1: PX0 + 14, x2: PX0 + 36, y1: PY1 + 34, y2: PY1 + 34, stroke: MUTED, 'stroke-width': 1.6, 'stroke-dasharray': '6 5' }, ghostKey);
  el('text', { class: 'stage-text', x: PX0 + 42, y: PY1 + 38, text: 'your line' }, ghostKey);
  const meanKey = el('g', {}, keyG);
  el('line', { x1: PX0 + 14, x2: PX0 + 36, y1: PY1 + 34, y2: PY1 + 34, stroke: VIOLET, 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }, meanKey);
  const meanKeyText = el('text', { class: 'stage-text', x: PX0 + 42, y: PY1 + 38, text: 'mean of y' }, meanKey);
  meanKeyText.style.fill = VIOLET;

  // Step 5: one step of 1 in x, and where the line crosses x = 0.
  const triG = el('g', {}, svg);
  const triRun = el('line', { stroke: STRONG, 'stroke-width': 1.6 }, triG);
  const triRise = el('line', { stroke: STRONG, 'stroke-width': 1.6 }, triG);
  const triDot0 = el('circle', { r: 4, fill: STRONG, stroke: BG, 'stroke-width': 1.5 }, triG);
  const triDot1 = el('circle', { r: 4, fill: STRONG, stroke: BG, 'stroke-width': 1.5 }, triG);
  const triRunLabel = el('text', { class: 'stage-text', 'text-anchor': 'middle', text: '+1', ...HALO }, triG);
  const triRiseLabel = el('text', { class: 'stage-text', ...HALO }, triG);
  const icpt = el('rect', { width: 9, height: 9, fill: VIOLET, stroke: BG, 'stroke-width': 1.5 }, triG);
  const icptLabel = el('text', { class: 'stage-text', text: 'intercept', ...HALO }, triG);
  icptLabel.style.fill = VIOLET;

  const dotsG = el('g', {}, svg);
  xs.forEach((x, i) => el('circle', { class: 'dot', cx: X(x), cy: Y(ys[i]), r: R, stroke: BG, 'stroke-width': 1.2 }, dotsG));

  // --- The two handles (steps 2 and 3).
  const handlesG = el('g', {}, svg);
  const focusRing = el('circle', { r: 13, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, handlesG);
  const knobs = HX.map(() => el('circle', { r: 7, fill: BG, stroke: STRONG, 'stroke-width': 2.5 }, handlesG));
  const NAMES = ['Left handle', 'Right handle'];
  const handles = HX.map((hx, k) => {
    const h = el('circle', {
      class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'slider',
      'aria-label': `${NAMES[k]}: the height of your line at x = ${hx}`,
      'aria-valuemin': Y_LO, 'aria-valuemax': Y_HI,
    }, handlesG);
    h.style.outline = 'none';
    return h;
  });
  let focusedK = -1;

  // --- Side panel: read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rows = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Side panel: two bars (step 4: yours v least squares; step 6: SST v SSE).
  const barsG = el('g', {}, svg);
  el('line', { x1: SX0 - 8, x2: SX1, y1: SY0, y2: SY0, stroke: LINE }, barsG);
  const BAR_W = 34;
  const BX = [SX0 + 6, SX0 + 100];
  const bars = BX.map(() => el('rect', { width: BAR_W, 'fill-opacity': 0.25, 'stroke-width': 1.6 }, barsG));
  const barVals = BX.map((bx) => el('text', { class: 'stage-text', x: bx + BAR_W / 2, 'text-anchor': 'middle' }, barsG));
  const barNames = BX.map((bx) => el('text', { class: 'stage-text', x: bx + BAR_W / 2, y: SY0 + 17, 'text-anchor': 'middle' }, barsG));
  const barCap = el('text', { class: 'value', x: SX0 - 8, y: SY1 - 30 }, barsG);

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const resetBtn = make('button', { type: 'button', textContent: 'Reset my line' });
  const dragGroup = make('div');
  dragGroup.append(resetBtn);
  const solveBtn = make('button', { type: 'button', textContent: 'Solve again' });
  const solveGroup = make('div');
  solveGroup.append(solveBtn);
  const xGroup = make('div');
  const xLabelEl = make('label');
  const xInput = make('input', { type: 'range', min: X0_LO(xs), max: X0_HI(xs), step: 0.01, value: X0_START });
  xInput.style.width = '140px';
  const xOut = make('output');
  xOut.setAttribute('aria-hidden', 'true');
  xLabelEl.append('x', make('span', { className: 'vh', textContent: ', where the step of 1 starts' }), ' ', xInput, ' ', xOut);
  xGroup.append(xLabelEl);
  const fromGroup = make('div');
  const fromLineBtn = make('button', { type: 'button', textContent: 'Misses from the line' });
  const fromMeanBtn = make('button', { type: 'button', textContent: 'Misses from the mean' });
  fromGroup.append(fromLineBtn, fromMeanBtn);
  controls.append(dragGroup, solveGroup, xGroup, fromGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Layer opacities fade between steps.
  let step = 0;
  let token = 0;
  let solveToken = 0;
  const LAYERS = { cands: 0, line: 0, ghost: 0, sticks: 0, squares: 0, handles: 0, band: 0, tri: 0, mean: 0, bars: 0 };
  let s = { ...LAYERS, cands: 1 };

  function target(i) {
    if (i === 0) return { ...LAYERS, cands: 1 };
    if (i === 1) return { ...LAYERS, line: 1, sticks: 1, handles: 1 };
    if (i === 2) return { ...LAYERS, line: 1, sticks: 1, squares: 1, handles: 1 };
    if (i === 3) return { ...LAYERS, line: 1, ghost: 1, squares: 1, bars: 1 };
    if (i === 4) return { ...LAYERS, line: 1, band: 1, tri: 1 };
    return { ...LAYERS, line: 1, sticks: 1, mean: 1, bars: 1 };
  }

  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const set = (node, attrs) => {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, typeof v === 'number' ? v.toFixed(2) : v);
  };
  const lineAttrs = (l, x1 = X_DOMAIN[0], x2 = X_DOMAIN[1]) => ({ x1: X(x1), x2: X(x2), y1: Y(predict(l, x1)), y2: Y(predict(l, x2)) });

  function barPair(vals, names, styles) {
    const top = Math.max(...vals) || 1;
    const B = linearScale([0, top], [SY0, SY1]);
    bars.forEach((b, k) => {
      const yv = B(clamp(vals[k], 0, top));
      set(b, { x: BX[k], y: yv, height: SY0 - yv });
      for (const [a, v] of Object.entries(styles[k])) b.setAttribute(a, v);
      if (!styles[k]['stroke-dasharray']) b.removeAttribute('stroke-dasharray');
      barVals[k].textContent = fmt(vals[k]);
      set(barVals[k], { y: yv - 6 });
      barNames[k].textContent = names[k];
    });
  }
  const CYAN_BAR = { fill: CYAN, stroke: CYAN };
  const MUTED_BAR = { fill: MUTED, stroke: MUTED, 'stroke-dasharray': '4 3' };
  const VIOLET_BAR = { fill: VIOLET, stroke: VIOLET, 'stroke-dasharray': '4 3' };

  function render() {
    const l = step >= 4 ? ols : step === 3 ? cur : user;
    const es = residualsOf(xs, ys, l);

    op(candG, s.cands);

    // The main line, and the reader's line as a ghost in step 4.
    set(main, step === 4 ? lineAttrs(l, xMin, xMax) : lineAttrs(l));
    set(extL, lineAttrs(l, X_DOMAIN[0], xMin));
    set(extR, lineAttrs(l, xMax, X_DOMAIN[1]));
    op(extL, step === 4 ? s.line : 0);
    op(extR, step === 4 ? s.line : 0);
    op(main, s.line);
    set(ghost, lineAttrs(user));
    op(ghost, s.ghost);
    op(keyG, step >= 3 ? s.line : 0);
    ghostKey.style.display = step === 3 ? '' : 'none';
    meanKey.style.display = step === 5 ? '' : 'none';
    keyG.firstChild.setAttribute('height', step === 4 ? 26 : 44);

    // Sticks: to the line, or to the mean in step 6.
    const toMean = step === 5 && from === 'mean';
    sticks.forEach((st, i) => {
      const base = toMean ? ybar : predict(l, xs[i]);
      set(st, { x1: X(xs[i]), x2: X(xs[i]), y1: Y(ys[i]), y2: Y(clamp(base, Y_LO, Y_HI)) });
      st.setAttribute('stroke', toMean ? VIOLET : MUTED);
      if (toMean) st.setAttribute('stroke-dasharray', '3 3');
      else st.removeAttribute('stroke-dasharray');
    });
    op(sticksG, s.sticks);

    // A square on each stick: its side is the miss, so its area is the miss squared.
    sq.forEach((r, i) => {
      const side = Math.abs(es[i]) * unit;
      const px = X(xs[i]);
      const right = px + side <= 388;
      set(r, {
        x: right ? px : px - side,
        y: Math.min(Y(ys[i]), Y(predict(l, xs[i]))),
        width: side,
        height: side,
      });
    });
    op(squaresG, s.squares);

    op(meanG, s.mean);

    // Handles.
    HX.forEach((hx, k) => {
      set(knobs[k], { cx: X(hx), cy: Y(hy[k]) });
      set(handles[k], { cx: X(hx), cy: Y(hy[k]) });
      handles[k].setAttribute('aria-valuenow', fmt(hy[k]));
      handles[k].setAttribute('aria-valuetext', `y ${say2(hy[k])}. Sum of squared errors ${say2(sse(xs, ys, user))}`);
    });
    if (focusedK >= 0) set(focusRing, { cx: X(HX[focusedK]), cy: Y(hy[focusedK]) });
    focusRing.style.display = focusedK >= 0 && (step === 1 || step === 2) ? '' : 'none';
    op(handlesG, s.handles);

    // Step 5: the data range, the step of 1 and the intercept.
    op(band, s.band * 0.45);
    op(bandLabel, s.band);
    const y0 = predict(ols, x0);
    const y1 = predict(ols, x0 + 1);
    set(triRun, { x1: X(x0), x2: X(x0 + 1), y1: Y(y0), y2: Y(y0) });
    set(triRise, { x1: X(x0 + 1), x2: X(x0 + 1), y1: Y(y0), y2: Y(y1) });
    set(triDot0, { cx: X(x0), cy: Y(y0) });
    set(triDot1, { cx: X(x0 + 1), cy: Y(y1) });
    set(triRunLabel, { x: X(x0 + 0.5), y: Y(y0) + 16 });
    triRiseLabel.textContent = `+${fmt(ols.b)}`;
    set(triRiseLabel, { x: X(x0 + 1) + 6, y: Y((y0 + y1) / 2) + 4 });
    set(icpt, { x: X(0) - 4.5, y: Y(ols.a) - 4.5 });
    set(icptLabel, { x: X(0) + 8, y: Y(ols.a) + 16 });
    op(triG, s.tri);

    // Bars.
    if (step === 3) {
      barPair([sse(xs, ys, user), SSE], ['yours', 'least sq.'], [MUTED_BAR, CYAN_BAR]);
      barCap.textContent = 'SSE';
    } else {
      barPair([SST, SSE], ['SST', 'SSE'], [VIOLET_BAR, CYAN_BAR]);
      barCap.textContent = 'sums of squares';
    }
    op(barsG, s.bars);

    // Read-out box.
    const texts = readout(step, { xs, ys, user, line: ols, x0, from });
    rows.forEach((r, k) => {
      r.textContent = texts[k] || '';
      r.style.display = k < texts.length ? '' : 'none';
    });
    let w = 176;
    try {
      const lens = rows.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const bx = Math.min(BOX.x, 598 - w);
    set(boxRect, { x: bx, y: BOX.y, width: w, height: 16 + BOX.row * texts.length });
    rows.forEach((r, k) => set(r, { x: bx + 12, y: BOX.y + 8 + BOX.row * k + BOX.row / 2 + 5 }));

    xOut.textContent = fmt(x0);
    xInput.setAttribute('aria-valuetext', `x = ${fmt(x0)}`);
    fromLineBtn.setAttribute('aria-pressed', String(from === 'line'));
    fromMeanBtn.setAttribute('aria-pressed', String(from === 'mean'));
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

  // --- Words for the live region.
  const spoken = (i) => {
    if (i === 0) return `${n} points. Which straight line describes them best?`;
    if (i === 1) return `Your line: a ${say2(user.a)}, b ${say2(user.b)}. Sum of squared errors ${say2(sse(xs, ys, user))}.`;
    if (i === 2) return `The squares add up to ${say2(sse(xs, ys, user))}. Divided by ${n}, the MSE is ${say2(sse(xs, ys, user) / n)}.`;
    if (i === 3) {
      return `Your line's sum of squared errors is ${say2(sse(xs, ys, user))}. The least-squares line's is ${say2(SSE)}, ` +
        `with intercept ${say2(ols.a)} and slope ${say2(ols.b)}.`;
    }
    if (i === 4) {
      return `At x ${say2(x0)} the line predicts ${say2(predict(ols, x0))}; at x ${say2(x0 + 1)}, ${say2(predict(ols, x0 + 1))}. ` +
        `The difference is the slope, ${say2(ols.b)}.`;
    }
    return `Misses measured from the ${from}. SST ${say2(SST)}, SSE ${say2(SSE)}, R squared ${say2(1 - SSE / SST)}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // --- Interaction: the handles (steps 2 and 3).
  const setHandle = (k, y) => {
    hy = hy.slice();
    hy[k] = clamp(y, Y_LO, Y_HI);
    user = lineFromHandles(...hy);
    cur = { ...user };
    render();
    say(spoken(step));
  };
  handles.forEach((h, k) => {
    h.addEventListener('keydown', (e) => {
      const y = hy[k];
      const map = {
        ArrowUp: y + 0.1, ArrowRight: y + 0.1, ArrowDown: y - 0.1, ArrowLeft: y - 0.1,
        PageUp: y + 1, PageDown: y - 1, Home: Y_LO, End: Y_HI,
      };
      if (!(e.key in map)) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      setHandle(k, map[e.key]);
    });
    h.addEventListener('focus', () => {
      let kb = true;
      try { kb = h.matches(':focus-visible'); } catch { /* keep the ring */ }
      focusedK = kb ? k : -1;
      render();
    });
    h.addEventListener('blur', () => { focusedK = -1; render(); });
    draggable(h, {
      bounds: { x0: X(HX[k]), x1: X(HX[k]), y0: Y(Y_HI), y1: Y(Y_LO) },
      onDrag: (_, py) => setHandle(k, Y.invert(py)),
    });
  });
  resetBtn.addEventListener('click', () => {
    hy = START_HANDLES.slice();
    user = lineFromHandles(...hy);
    cur = { ...user };
    render();
    announce(`Line reset. ${spoken(step)}`);
  });

  // --- "Solve" (step 4): the line moves from yours to the least-squares line.
  function solve(animate) {
    const my = ++solveToken;
    const start = { ...user };
    const finish = () => {
      if (my !== solveToken) return;
      cur = solveFrame(start, ols, 1);
      render();
    };
    if (!animate || reduced) {
      finish();
      return Promise.resolve();
    }
    return tween(1100, (kt) => {
      if (my !== solveToken) return;
      cur = solveFrame(start, ols, kt);
      render();
    }, { reduced }).then(finish);
  }
  solveBtn.addEventListener('click', () => {
    solve(true).then(() => announce(spoken(3)));
  });

  // --- Step 5 slider and step 6 toggle.
  xInput.addEventListener('input', () => {
    x0 = clamp(Number(xInput.value), X0_LO(xs), X0_HI(xs));
    render();
    say(spoken(4));
  });
  const setFrom = (v) => {
    from = v;
    render();
    announce(spoken(5));
  };
  fromLineBtn.addEventListener('click', () => setFrom('line'));
  fromMeanBtn.addEventListener('click', () => setFrom('mean'));

  // Sentences on the page that quote values from the data (hidden until filled).
  const filled = pageValues(xs, ys);
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
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
      showGroup(dragGroup, i === 1 || i === 2);
      showGroup(solveGroup, i === 3);
      showGroup(xGroup, i === 4);
      showGroup(fromGroup, i === 5);
      const live = i === 1 || i === 2;
      handles.forEach((h) => {
        h.setAttribute('tabindex', live ? 0 : -1);
        h.style.display = live ? '' : 'none';
        if (!live && doc.activeElement === h) h.blur();
      });
      solveToken++;
      cur = { ...user };
      announce(spoken(i));
      const to = target(i);
      const fade = () => {
        if (!animate) {
          token++;
          s = { ...to };
          render();
          return Promise.resolve();
        }
        return animateTo(to, 500);
      };
      return fade().then(() => (i === 3 ? solve(animate) : undefined));
    },
  };
}
