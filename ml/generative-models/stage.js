// Lesson L5.7, "Generative models": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-generative-models.test.mjs);
// the DOM is only touched inside createStage. Every fit is gauss1Fit or gmmEM from
// gen.js, every sample is drawn with a stated seed, and every number in the page's
// text comes from pageValues or storyValues.

import { mulberry32, randn } from '../assets/maths.js';
import { blobs3, spiral2D, LESSON_SPIRAL, BLOBS, gauss1Fit, gauss2Fit, gmmInit, gmmEM, gmmSample, gmmLogLik, nearestDist } from '../assets/gen.js';
import { el, ellipse, classMarker } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const N_1D = 300;
export const SEED_1D = 301;
export const SEED_1D_HELD = 306;
// The 1-D rule: with probability 0.4 a value from N(−1.5, 0.5²), otherwise from N(1.5, 0.75²).
export const RULE_1D = [{ w: 0.4, mu: -1.5, sd: 0.5 }, { w: 0.6, mu: 1.5, sd: 0.75 }];
export const N_BLOBS = 300;
export const SEED_TRAIN = 302;
export const SEED_HELD = 303;
export const SEED_INIT = 304;
export const SEED_FRESH = 307;
export const SEED_SAMPLE_1D = 308;
export const SEED_SAMPLE_BLOBS = 309;
export const SEED_SAMPLE_SPIRAL = 310;
export const K_TRUE = 3;
export const K_MAX = 5;
export const K_SPIRAL = 5;
// EM stops when one iteration raises the mean log-likelihood by less than TOL,
// or after MAX_ITERS iterations. No ridge is added to the covariances.
export const TOL = 1e-6;
export const MAX_ITERS = 500;
// The live run takes one EM iteration every SLOW_FRAMES frames, so it can be watched.
export const SLOW_FRAMES = 8;
// Ellipses are drawn at Mahalanobis distance 2 from the mean.
export const ELLIPSE_K = 2;
// Sample presses summarised in the text.
export const PRESSES = 20;
// Iterations of step 4's run in which the curve nearly stalls (story-tested).
export const STALL = [8, 14];
export const SHADE_MIN = 0.25;
export const HIST = { lo: -5, hi: 5, w: 0.5 };

const DATA = new Map();
const cached = (key, make) => {
  if (!DATA.has(key)) DATA.set(key, make());
  return DATA.get(key);
};

export function values1D(n, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const u = rng();
    const c = u < RULE_1D[0].w ? RULE_1D[0] : RULE_1D[1];
    return c.mu + c.sd * randn(rng);
  });
}
export const data1D = () => cached('d1', () => values1D(N_1D, SEED_1D));
export const held1D = () => cached('h1', () => values1D(N_1D, SEED_1D_HELD));
export const fit1D = () => cached('f1', () => gauss1Fit(data1D()));
export const blobsTrain = () => cached('bt', () => blobs3({ n: N_BLOBS, seed: SEED_TRAIN }));
export const blobsHeld = () => cached('bh', () => blobs3({ n: N_BLOBS, seed: SEED_HELD }));
export const spiralData = () => cached('sp', () => spiral2D(LESSON_SPIRAL));
export const freshSpiral = () => cached('fs', () => spiral2D({ ...LESSON_SPIRAL, seed: SEED_FRESH }));
// The three Gaussians that made the blobs, as a mixture with weights 1/3.
export const ruleMixture = () => ({ pi: [1 / 3, 1 / 3, 1 / 3], mu: BLOBS.map((b) => b.mu), S: BLOBS.map((b) => b.S) });

// ---- one dimension ---------------------------------------------------------------

export const gaussPdf1 = (x, mu, s2) => Math.exp(-((x - mu) ** 2) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);
export const ruleDensity1D = (x) => RULE_1D.reduce((s, c) => s + c.w * gaussPdf1(x, c.mu, c.sd * c.sd), 0);
// Mean log-likelihood per value (natural log).
export function logLik1(xs, { mu, s2 }) {
  let s = 0;
  for (const x of xs) s += -0.5 * Math.log(2 * Math.PI * s2) - ((x - mu) ** 2) / (2 * s2);
  return s / xs.length;
}
export const logLikRule1D = (xs) => xs.reduce((s, x) => s + Math.log(ruleDensity1D(x)), 0) / xs.length;

// Densities per bin: count / (n · width), so the bars have area 1 when every value
// is inside [lo, hi). Values outside are not drawn.
export function histogram(xs, { lo, hi, w } = HIST) {
  const nb = Math.round((hi - lo) / w);
  const c = new Array(nb).fill(0);
  for (const x of xs) {
    const b = Math.floor((x - lo) / w);
    if (b >= 0 && b < nb) c[b]++;
  }
  return { lo, w, density: c.map((v) => v / (xs.length * w)) };
}

export function samples1D(seed) {
  const { mu, s2 } = fit1D();
  const rng = mulberry32(seed);
  return Array.from({ length: N_1D }, () => mu + Math.sqrt(s2) * randn(rng));
}

// ---- EM, one iteration at a time ---------------------------------------------------

// A run of EM from gmmInit(X, K, seed). step(k) takes up to k iterations; done is set
// once a rise is below TOL, or at MAX_ITERS, or if a value stops being finite.
export function createFit(X, K, seed = SEED_INIT) {
  if (!(K >= 1 && K <= X.length)) throw new Error(`K = ${K} needs 1 ≤ K ≤ ${X.length}`);
  const init = gmmInit(X, K, seed);
  const fit = {};
  fit.reset = () => {
    fit.params = init;
    fit.logLik = [gmmLogLik(X, init)];
    fit.iter = 0;
    fit.done = false;
    fit.diverged = !Number.isFinite(fit.logLik[0]);
    if (fit.diverged) fit.done = true;
    return fit;
  };
  fit.step = (k = 1) => {
    for (let i = 0; i < k && !fit.done; i++) {
      const r = gmmEM(X, fit.params, 1);
      const L = r.logLik[1];
      if (!Number.isFinite(L)) { fit.diverged = true; fit.done = true; return fit; }
      const rise = L - fit.logLik.at(-1);
      fit.params = r.params;
      fit.logLik.push(L);
      fit.iter++;
      if (rise < TOL || fit.iter >= MAX_ITERS) fit.done = true;
    }
    return fit;
  };
  return fit.reset();
}

// The page's finished runs: 'k1' … 'k5' on the training blobs, 'spiral' with K = 5.
const FITS = new Map();
const JOBS = new Map();
const FIT_SPEC = (name) => (name === 'spiral' ? [spiralData(), K_SPIRAL] : [blobsTrain(), Number(name.slice(1))]);
// Up to `iters` more iterations of that run; true once it is finished. The page calls
// this in small slices; slicing does not change the result.
export function fitWork(name, iters) {
  if (FITS.has(name)) return true;
  if (!JOBS.has(name)) JOBS.set(name, createFit(...FIT_SPEC(name)));
  const f = JOBS.get(name).step(iters);
  if (!f.done) return false;
  FITS.set(name, { params: f.params, logLik: f.logLik.slice(), iter: f.iter, diverged: f.diverged });
  JOBS.delete(name);
  return true;
}
export const fitReady = (name) => FITS.has(name);
export function fitResult(name) {
  fitWork(name, Infinity);
  return FITS.get(name);
}

// ---- responsibilities -----------------------------------------------------------------

function logGauss2(x, mu, S) {
  const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  const dx = x[0] - mu[0];
  const dy = x[1] - mu[1];
  const q = (S[1][1] * dx * dx - (S[0][1] + S[1][0]) * dx * dy + S[0][0] * dy * dy) / det;
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q;
}

// R[i][k] = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j), by log-sum-exp; hard[i] is
// the k with the largest R[i][k]; logLik is the mean log-likelihood per point.
export function responsibilities(X, { pi, mu, S }) {
  const K = pi.length;
  const R = [];
  const hard = [];
  let total = 0;
  for (const x of X) {
    const l = pi.map((p, k) => Math.log(p) + logGauss2(x, mu[k], S[k]));
    const m = Math.max(...l);
    const lse = m + Math.log(l.reduce((s, v) => s + Math.exp(v - m), 0));
    total += lse;
    const r = l.map((v) => Math.exp(v - lse));
    R.push(r);
    let best = 0;
    for (let k = 1; k < K; k++) if (r[k] > r[best]) best = k;
    hard.push(best);
  }
  return { R, hard, logLik: total / X.length };
}

// Opacity of a point from its largest responsibility: SHADE_MIN when shared evenly, 1 when sure.
export function shade(maxR, K) {
  if (K <= 1) return 1;
  const t = (maxR - 1 / K) / (1 - 1 / K);
  return SHADE_MIN + (1 - SHADE_MIN) * Math.min(1, Math.max(0, t));
}

// ---- K, samples and quality -------------------------------------------------------------

export function kTable() {
  return Array.from({ length: K_MAX }, (_, i) => {
    const r = fitResult(`k${i + 1}`);
    return { K: i + 1, train: r.logLik.at(-1), held: gmmLogLik(blobsHeld(), r.params), iters: r.iter };
  });
}

export const blobSamples = (seed) => gmmSample(fitResult(`k${K_TRUE}`).params, N_BLOBS, seed);
export const spiralSamples = (seed) => gmmSample(fitResult('spiral').params, spiralData().length, seed);

// precision: mean distance from each new point to its nearest data point.
// coverage: mean distance from each data point to its nearest new point.
export const quality = (samples, X) => ({ precision: nearestDist(samples, X), coverage: nearestDist(X, samples) });

// Step 4's pacing: take one iteration, then wait SLOW_FRAMES − 1 frames.
export function paceFrame(waited) {
  if (waited <= 0) return { take: 1, waited: SLOW_FRAMES - 1 };
  return { take: 0, waited: waited - 1 };
}

// ---- formatting --------------------------------------------------------------------------

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
export const ll = (v) => num(v, 2);
export const dist = (v) => num(v, 2);
const pct = (v) => `${Math.round(v * 100)}%`;

// ---- read-outs ------------------------------------------------------------------------------

export function liveInfo(fit, running) {
  return { iter: fit.iter, logLik: fit.logLik.at(-1), pi: fit.params.pi, running, done: fit.done, diverged: fit.diverged };
}

export function infoFor(step, { seed1 = SEED_SAMPLE_1D, seed6 = 0, K = K_TRUE } = {}) {
  if (step === 0) return {};
  if (step === 1) return { seed: seed1 };
  if (step === 2) return {};
  if (step === 3) return liveInfo(createFit(blobsTrain(), K_TRUE), false);
  if (step === 4) return { K, row: kTable()[K - 1] };
  const sp = spiralData();
  return {
    press: seed6,
    blobs: quality(blobSamples(SEED_SAMPLE_BLOBS + seed6), blobsTrain()),
    spiral: quality(spiralSamples(SEED_SAMPLE_SPIRAL + seed6), sp),
    held: quality(blobsHeld(), blobsTrain()),
    fresh: quality(freshSpiral(), sp),
  };
}

export function readout(step, s) {
  if (step === 0) return [`left: the ${N_BLOBS} training points`, `right: ${N_BLOBS} new points sampled from a fitted mixture`, 'none of the right-hand points is in the data'];
  if (step === 1) {
    const f = fit1D();
    return [
      `one Gaussian: μ̂ = ${num(f.mu, 2)}, σ̂² = ${num(f.s2, 2)}`,
      `held-out mean log-likelihood: ${ll(logLik1(held1D(), f))} (the rule: ${ll(logLikRule1D(held1D()))})`,
      s.seed === null ? 'press Sample to draw from the fitted Gaussian' : `${N_1D} values sampled from it, seed ${s.seed} (dashed)`,
    ];
  }
  if (step === 2) {
    return [
      `one Gaussian (solid): mean log-likelihood ${ll(fitResult('k1').logLik.at(-1))}`,
      `the three that made the data (dashed): ${ll(gmmLogLik(blobsTrain(), ruleMixture()))}`,
      `each ellipse holds about ${pct(1 - Math.exp(-ELLIPSE_K * ELLIPSE_K / 2))} of its Gaussian's points`,
    ];
  }
  if (step === 3) {
    const status = trainStatus(s);
    return [`iteration ${s.iter} · ${status}`, `mean log-likelihood: ${ll(s.logLik)}`, `weights π: ${s.pi.map((p) => num(p, 2)).join(', ')}`];
  }
  if (step === 4) {
    if (!s) return ['computing the five fits…'];
    return [`K = ${s.K} · ${s.row.iters} EM iteration${s.row.iters === 1 ? '' : 's'}`, `training points: ${ll(s.row.train)}`, `held-out points: ${ll(s.row.held)}`];
  }
  if (step === 5) {
    if (!s) return ['computing the spiral fit…'];
    return [
      `draw ${s.press + 1} · mean distance to the nearest point`,
      `blobs: new→data ${dist(s.blobs.precision)}, data→new ${dist(s.blobs.coverage)} (held-out ${dist(s.held.precision)}, ${dist(s.held.coverage)})`,
      `spiral: new→data ${dist(s.spiral.precision)}, data→new ${dist(s.spiral.coverage)} (fresh ${dist(s.fresh.precision)}, ${dist(s.fresh.coverage)})`,
    ];
  }
  return [];
}

// ---- values written into the page ----------------------------------------------------------

export function pageValues() {
  const f = fit1D();
  return {
    'n-1d': String(N_1D),
    'mu-1d': num(f.mu, 2),
    's2-1d': num(f.s2, 2),
    'n-blobs': String(N_BLOBS),
    'n-spiral': String(spiralData().length),
    'ellipse-share': pct(1 - Math.exp(-ELLIPSE_K * ELLIPSE_K / 2)),
  };
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const v = {};
  const f = fit1D();
  const one = logLik1(held1D(), f);
  const rule = logLikRule1D(held1D());
  v['ll1-one'] = ll(one);
  v['ll1-rule'] = ll(rule);
  v['ll1-gap'] = num(rule - one, 2);
  v['rule-train'] = ll(gmmLogLik(blobsTrain(), ruleMixture()));
  v['rule-held'] = ll(gmmLogLik(blobsHeld(), ruleMixture()));
  const tab = kTable();
  for (const row of tab) {
    v[`train-k${row.K}`] = ll(row.train);
    v[`held-k${row.K}`] = ll(row.held);
  }
  const k3 = fitResult(`k${K_TRUE}`);
  v['iters-k3'] = String(k3.iter);
  v['start-k3'] = ll(k3.logLik[0]);
  v['end-k3'] = ll(k3.logLik.at(-1));
  const sp = spiralData();
  const fr = quality(freshSpiral(), sp);
  v['fresh-prec'] = dist(fr.precision);
  v['fresh-cov'] = dist(fr.coverage);
  const spq = Array.from({ length: PRESSES }, (_, i) => quality(spiralSamples(SEED_SAMPLE_SPIRAL + i), sp));
  v['spiral-prec-lo'] = dist(Math.min(...spq.map((q) => q.precision)));
  v['spiral-prec-hi'] = dist(Math.max(...spq.map((q) => q.precision)));
  v['spiral-cov-lo'] = dist(Math.min(...spq.map((q) => q.coverage)));
  v['spiral-cov-hi'] = dist(Math.max(...spq.map((q) => q.coverage)));
  const tr = blobsTrain();
  const hq = quality(blobsHeld(), tr);
  v['held-prec'] = dist(hq.precision);
  v['held-cov'] = dist(hq.coverage);
  const bq = Array.from({ length: PRESSES }, (_, i) => quality(blobSamples(SEED_SAMPLE_BLOBS + i), tr));
  v['blob-prec-lo'] = dist(Math.min(...bq.map((q) => q.precision)));
  v['blob-prec-hi'] = dist(Math.max(...bq.map((q) => q.precision)));
  v['blob-cov-lo'] = dist(Math.min(...bq.map((q) => q.coverage)));
  v['blob-cov-hi'] = dist(Math.max(...bq.map((q) => q.coverage)));
  STORY = v;
  return v;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [292, 314, 336];
// Component k: ellipse colour and dash, and the marker shape (classMarker 0, 1, 2).
const COMP = [{ stroke: CYAN, dash: '' }, { stroke: VIOLET, dash: '6 4' }, { stroke: INK, dash: '2 3' }];

// A square panel for 2-D points: equal units on both axes, so ellipses keep their shape.
function panel(x0, y0, ppu, [ax, bx], [ay, by]) {
  const xs = (v) => x0 + (v - ax) * ppu;
  const ys = (v) => y0 + (by - v) * ppu;
  return { xs, ys, x0, y0, w: (bx - ax) * ppu, h: (by - ay) * ppu };
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let seed1 = null;
  let press6 = 0;
  let kSel = K_TRUE;
  let live = null;

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  // A panel's frame, and a group clipped to it for everything drawn inside.
  const defs = el('defs', {}, svg);
  let clips = 0;
  const frame = (parent, p) => {
    el('rect', { x: p.x0, y: p.y0, width: p.w, height: p.h, fill: 'none', stroke: LINE }, parent);
    const cid = `gm-clip-${++clips}`;
    el('rect', { x: p.x0, y: p.y0, width: p.w, height: p.h }, el('clipPath', { id: cid }, defs));
    return el('g', { 'clip-path': `url(#${cid})` }, parent);
  };
  const hollow = (parent, p, X, r = 2.6) => {
    const g = el('g', {}, parent);
    for (const [x, y] of X) el('circle', { cx: P(p.xs(x)), cy: P(p.ys(y)), r, fill: 'none', stroke: MUTED, 'stroke-width': 1 }, g);
    return g;
  };
  // Filled cyan dots, redrawn in place.
  const dots = (parent, n, r = 2.6) => {
    const g = el('g', {}, parent);
    const cs = Array.from({ length: n }, () => el('circle', { class: 'mk mk-0', r, cx: -20, cy: -20 }, g));
    return {
      g,
      set(p, X) { X.forEach(([x, y], i) => { cs[i].setAttribute('cx', P(p.xs(x))); cs[i].setAttribute('cy', P(p.ys(y))); }); },
    };
  };
  const ellipses = (parent, p, K, style) => Array.from({ length: K }, (_, k) => {
    const e = ellipse(parent, p.xs, p.ys, { k: ELLIPSE_K });
    const st = style(k);
    e.path.style.stroke = st.stroke;
    if (st.dash) e.path.style.strokeDasharray = st.dash;
    if (st.width) e.path.style.strokeWidth = st.width;
    return e;
  });
  const BLOB_X = [-4.5, 4];
  const BLOB_Y = [-3.5, 3.5];

  // ---- step 1: the data and new points from a fitted mixture, side by side.
  const teaseG = el('g', {}, svg);
  const tL = panel(20, 40, 30, BLOB_X, BLOB_Y);
  const tR = panel(320, 40, 30, BLOB_X, BLOB_Y);
  const tLin = frame(teaseG, tL);
  const tRin = frame(teaseG, tR);
  label(teaseG, tL.x0, 28, 'the data', 'start', STRONG, 13);
  label(teaseG, tR.x0, 28, 'new points from a model of it', 'start', STRONG, 13);
  hollow(tLin, tL, blobsTrain());
  const teaseDots = dots(tRin, N_BLOBS);
  let teaseDrawn = false;

  // ---- step 2: the 1-D values, one Gaussian, and samples from it.
  const oneG = el('g', {}, svg);
  const HX0 = 60;
  const HX1 = 580;
  const HY0 = 240;
  const HY1 = 40;
  const DMAX = 0.4;
  const hx = (v) => HX0 + ((v - HIST.lo) / (HIST.hi - HIST.lo)) * (HX1 - HX0);
  const hy = (d) => HY0 - (Math.min(d, DMAX) / DMAX) * (HY0 - HY1);
  el('line', { x1: HX0, x2: HX1, y1: HY0, y2: HY0, stroke: LINE }, oneG);
  el('line', { x1: HX0, x2: HX0, y1: HY0, y2: HY1, stroke: LINE }, oneG);
  for (let t = -4; t <= 4; t += 2) label(oneG, P(hx(t)), HY0 + 16, num(t, 0), 'middle');
  label(oneG, (HX0 + HX1) / 2, HY0 + 32, 'value', 'middle');
  for (const d of [0.1, 0.2, 0.3, 0.4]) {
    el('line', { x1: HX0, x2: HX1, y1: P(hy(d)), y2: P(hy(d)), stroke: LINE, 'stroke-dasharray': '2 4' }, oneG);
    label(oneG, HX0 - 6, P(hy(d)) + 4, num(d, 1), 'end');
  }
  const yl = label(oneG, 18, (HY0 + HY1) / 2, 'density', 'middle');
  yl.setAttribute('transform', `rotate(-90 18 ${(HY0 + HY1) / 2})`);
  const h1 = histogram(data1D());
  h1.density.forEach((d, b) => {
    if (d === 0) return;
    const x = hx(HIST.lo + b * HIST.w);
    el('rect', { x: P(x), y: P(hy(d)), width: P(hx(HIST.lo + (b + 1) * HIST.w) - x), height: P(HY0 - hy(d)), fill: CYAN, 'fill-opacity': 0.25, stroke: CYAN, 'stroke-width': 1 }, oneG);
  });
  const f1 = fit1D();
  const curve = [];
  for (let i = 0; i <= 200; i++) {
    const x = HIST.lo + (i / 200) * (HIST.hi - HIST.lo);
    curve.push(`${i ? 'L' : 'M'}${P(hx(x))},${P(hy(gaussPdf1(x, f1.mu, f1.s2)))}`);
  }
  el('path', { d: curve.join(''), fill: 'none', stroke: STRONG, 'stroke-width': 2.5 }, oneG);
  const sampPath = el('path', { d: '', fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, oneG);
  el('rect', { x: 72, y: 34, width: 18, height: 10, fill: CYAN, 'fill-opacity': 0.25, stroke: CYAN }, oneG);
  label(oneG, 96, 43, `the ${N_1D} values`, 'start', STRONG);
  el('line', { x1: 72, x2: 90, y1: 58, y2: 58, stroke: STRONG, 'stroke-width': 2.5 }, oneG);
  label(oneG, 96, 62, 'one Gaussian, fitted', 'start', STRONG);
  const sampKey = el('g', {}, oneG);
  el('line', { x1: 72, x2: 90, y1: 78, y2: 78, stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, sampKey);
  label(sampKey, 96, 82, 'values sampled from it', 'start', STRONG);

  // ---- step 3: the blobs, one Gaussian, and the three Gaussians that made them.
  const mixG = el('g', {}, svg);
  const mP = panel(20, 40, 30, BLOB_X, BLOB_Y);
  const mIn = frame(mixG, mP);
  hollow(mIn, mP, blobsTrain());
  const g1 = gauss2Fit(blobsTrain());
  const oneE = ellipse(mIn, mP.xs, mP.ys, { k: ELLIPSE_K });
  oneE.update(g1.mu, g1.S);
  oneE.path.style.stroke = STRONG;
  oneE.path.style.strokeWidth = '2.5';
  const rule = ruleMixture();
  ellipses(mIn, mP, 3, () => ({ stroke: CYAN, dash: '6 4' })).forEach((e, k) => e.update(rule.mu[k], rule.S[k]));
  label(mixG, mP.x0, 28, `the ${N_BLOBS} training points`, 'start', STRONG, 13);
  el('line', { x1: 320, x2: 344, y1: 70, y2: 70, stroke: STRONG, 'stroke-width': 2.5 }, mixG);
  label(mixG, 352, 74, 'one Gaussian, fitted', 'start', STRONG);
  el('line', { x1: 320, x2: 344, y1: 96, y2: 96, stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, mixG);
  label(mixG, 352, 100, 'the three Gaussians that', 'start', STRONG);
  label(mixG, 352, 116, 'made the data, π = 1/3 each', 'start', STRONG);
  label(mixG, 320, 160, 'p(x) = π₁N₁(x) + π₂N₂(x) + π₃N₃(x)', 'start', INK, 13);
  label(mixG, 320, 184, 'π is a weight here, not 3.14…', 'start');

  // ---- step 4: EM live. Points shaped by their likeliest component, faded when shared.
  const emG = el('g', {}, svg);
  const eP = panel(20, 40, 30, BLOB_X, BLOB_Y);
  const eIn = frame(emG, eP);
  label(emG, eP.x0, 28, `K = ${K_TRUE}`, 'start', STRONG, 13);
  const emPts = el('g', {}, eIn);
  const X = blobsTrain();
  let emMarks = [];
  let emShapes = [];
  const emEll = ellipses(eIn, eP, K_TRUE, (k) => ({ ...COMP[k], width: 2.5 }));
  const CX0 = 340;
  const CX1 = 584;
  const CY0 = 240;
  const CY1 = 50;
  const LMIN = -4.5;
  const LMAX = -2.9;
  const CITER = 30;
  const cx = (i) => CX0 + (i / CITER) * (CX1 - CX0);
  const cy = (v) => CY0 - ((Math.max(v, LMIN) - LMIN) / (LMAX - LMIN)) * (CY0 - CY1);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, emG);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, emG);
  for (const v of [-4, -3.5, -3]) {
    el('line', { x1: CX0, x2: CX1, y1: P(cy(v)), y2: P(cy(v)), stroke: LINE, 'stroke-dasharray': '2 4' }, emG);
    label(emG, CX0 - 6, P(cy(v)) + 4, num(v, 1), 'end');
  }
  for (const i of [0, 10, 20, 30]) label(emG, P(cx(i)), CY0 + 16, String(i), 'middle');
  label(emG, (CX0 + CX1) / 2, CY0 + 32, 'EM iteration', 'middle');
  label(emG, CX0, 28, 'mean log-likelihood', 'start', STRONG, 13);
  const emPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, emG);
  const emDot = el('circle', { r: 4, fill: CYAN, stroke: BG, 'stroke-width': 1.5 }, emG);

  // ---- step 5: K from 1 to 5. The chosen fit's ellipses, and both scores per K.
  const kG = el('g', {}, svg);
  const kP = panel(20, 40, 30, BLOB_X, BLOB_Y);
  const kIn2 = frame(kG, kP);
  hollow(kIn2, kP, blobsTrain());
  const kTitle = label(kG, kP.x0, 28, '', 'start', STRONG, 13);
  const kEll = Array.from({ length: K_MAX }, () => ellipses(kIn2, kP, 1, () => ({ stroke: CYAN, width: 2 }))[0]);
  const KX0 = 350;
  const KX1 = 570;
  const KY0 = 240;
  const KY1 = 60;
  const KMIN = -4;
  const KMAX = -2.9;
  const kx = (K) => KX0 + ((K - 1) / (K_MAX - 1)) * (KX1 - KX0);
  const ky = (v) => KY0 - ((v - KMIN) / (KMAX - KMIN)) * (KY0 - KY1);
  el('line', { x1: KX0 - 10, x2: KX1 + 10, y1: KY0, y2: KY0, stroke: LINE }, kG);
  el('line', { x1: KX0 - 10, x2: KX0 - 10, y1: KY0, y2: KY1, stroke: LINE }, kG);
  for (const v of [-4, -3.5, -3]) {
    el('line', { x1: KX0 - 10, x2: KX1 + 10, y1: P(ky(v)), y2: P(ky(v)), stroke: LINE, 'stroke-dasharray': '2 4' }, kG);
    label(kG, KX0 - 16, P(ky(v)) + 4, num(v, 1), 'end');
  }
  for (let K = 1; K <= K_MAX; K++) label(kG, P(kx(K)), KY0 + 16, String(K), 'middle');
  label(kG, (KX0 + KX1) / 2, KY0 + 32, 'components K', 'middle');
  const kSelLine = el('line', { y1: KY0, y2: KY1, stroke: MUTED, 'stroke-dasharray': '3 3' }, kG);
  const kTrainPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2 }, kG);
  const kHeldPath = el('path', { d: '', fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, kG);
  const kTrainMk = el('g', {}, kG);
  const kHeldMk = el('g', {}, kG);
  el('line', { x1: 430, x2: 454, y1: 192, y2: 192, stroke: CYAN, 'stroke-width': 2 }, kG);
  classMarker(kG, 442, 192, 0, { r: 4 });
  label(kG, 462, 196, 'training', 'start', STRONG);
  el('line', { x1: 430, x2: 454, y1: 214, y2: 214, stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, kG);
  classMarker(kG, 442, 214, 1, { r: 4.5 });
  label(kG, 462, 218, 'held-out', 'start', STRONG);
  label(kG, KX0 - 10, 28, 'mean log-likelihood', 'start', STRONG, 13);
  const kWait = label(kG, 460, 130, 'computing…', 'middle');
  let kDrawn = false;

  // ---- step 6: samples from the blobs' fit (K = 3) and the spiral's fit (K = 5).
  const smpG = el('g', {}, svg);
  const sL = panel(20, 40, 30, BLOB_X, BLOB_Y);
  const sR = panel(352, 40, 36, [-3, 3], [-3, 3]);
  const sLin = frame(smpG, sL);
  const sRin = frame(smpG, sR);
  label(smpG, sL.x0, 28, `blobs, K = ${K_TRUE}`, 'start', STRONG, 13);
  label(smpG, sR.x0, 28, `spiral, K = ${K_SPIRAL}`, 'start', STRONG, 13);
  hollow(sLin, sL, blobsTrain(), 2.2);
  hollow(sRin, sR, spiralData(), 2.2);
  const sLe = ellipses(sLin, sL, K_TRUE, () => ({ stroke: MUTED, width: 1.5, dash: '4 3' }));
  const sRe = ellipses(sRin, sR, K_SPIRAL, () => ({ stroke: MUTED, width: 1.5, dash: '4 3' }));
  const sLd = dots(sLin, N_BLOBS, 2.4);
  const sRd = dots(sRin, spiralData().length, 2.4);
  const sWait = label(smpG, 456, 150, 'computing…', 'middle');
  let sFitDrawn = false;

  // ---- read-out rows, under every picture.
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- controls: Sample (steps 2 and 6), Train / Pause / Reset (step 4), the K slider (step 5).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const sampGroup = make('div');
  const sampBtn = make('button', { type: 'button', textContent: 'Sample' });
  sampGroup.append(sampBtn);
  const kGroup = make('div');
  const kLab = make('label');
  const kIn = make('input', { type: 'range', min: '1', max: String(K_MAX), step: '1', value: String(kSel) });
  const kOut = make('output');
  kOut.setAttribute('aria-hidden', 'true');
  kLab.append('components K', ' ', kIn, ' ', kOut);
  kGroup.append(kLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'run EM');
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  controls.append(sampGroup, trainGroup, kGroup);

  const getLive = () => {
    if (!live) live = createFit(X, K_TRUE);
    return live;
  };
  const allK = () => Array.from({ length: K_MAX }, (_, i) => `k${i + 1}`);
  const kReady = () => allK().every(fitReady);
  const sReady = () => fitReady('spiral') && fitReady(`k${K_TRUE}`);

  function info() {
    if (step === 1) return { seed: seed1 };
    if (step === 3) return liveInfo(getLive(), loop.activity());
    if (step === 4) return kReady() ? { K: kSel, row: kTable()[kSel - 1] } : null;
    if (step === 5) return sReady() ? infoFor(5, { seed6: press6 }) : null;
    return {};
  }

  function drawEm() {
    const f = getLive();
    const r = responsibilities(X, f.params);
    if (emMarks.length === 0) {
      emMarks = X.map(([x, y], i) => classMarker(emPts, P(eP.xs(x)), P(eP.ys(y)), r.hard[i], { r: 3.4 }));
      emShapes = r.hard.slice();
    }
    X.forEach(([x, y], i) => {
      if (emShapes[i] !== r.hard[i]) {
        const m = classMarker(emPts, P(eP.xs(x)), P(eP.ys(y)), r.hard[i], { r: 3.4 });
        emMarks[i].replaceWith(m);
        emMarks[i] = m;
        emShapes[i] = r.hard[i];
      }
      emMarks[i].setAttribute('opacity', P(shade(Math.max(...r.R[i]), K_TRUE)));
    });
    f.params.mu.forEach((mu, k) => emEll[k].update(mu, f.params.S[k]));
    emPath.setAttribute('d', f.logLik.map((v, i) => `${i ? 'L' : 'M'}${P(cx(i))},${P(cy(v))}`).join(''));
    emDot.setAttribute('cx', P(cx(f.logLik.length - 1)));
    emDot.setAttribute('cy', P(cy(f.logLik.at(-1))));
  }

  function drawK() {
    show(kWait, !kReady());
    if (!kReady()) return;
    const tab = kTable();
    if (!kDrawn) {
      kDrawn = true;
      kTrainPath.setAttribute('d', tab.map((r, i) => `${i ? 'L' : 'M'}${P(kx(r.K))},${P(ky(r.train))}`).join(''));
      kHeldPath.setAttribute('d', tab.map((r, i) => `${i ? 'L' : 'M'}${P(kx(r.K))},${P(ky(r.held))}`).join(''));
      for (const r of tab) {
        classMarker(kTrainMk, P(kx(r.K)), P(ky(r.train)), 0, { r: 4 });
        classMarker(kHeldMk, P(kx(r.K)), P(ky(r.held)), 1, { r: 4.5 });
      }
    }
    const p = fitResult(`k${kSel}`).params;
    kEll.forEach((e, k) => {
      show(e.path, k < kSel);
      if (k < kSel) e.update(p.mu[k], p.S[k]);
    });
    kSelLine.setAttribute('x1', P(kx(kSel)));
    kSelLine.setAttribute('x2', P(kx(kSel)));
  }

  function drawSamples() {
    show(sWait, !sReady());
    for (const n of [sLd.g, sRd.g, ...sLe.map((e) => e.path), ...sRe.map((e) => e.path)]) show(n, sReady());
    if (!sReady()) return;
    if (!sFitDrawn) {
      sFitDrawn = true;
      const pb = fitResult(`k${K_TRUE}`).params;
      sLe.forEach((e, k) => e.update(pb.mu[k], pb.S[k]));
      const ps = fitResult('spiral').params;
      sRe.forEach((e, k) => e.update(ps.mu[k], ps.S[k]));
    }
    sLd.set(sL, blobSamples(SEED_SAMPLE_BLOBS + press6));
    sRd.set(sR, spiralSamples(SEED_SAMPLE_SPIRAL + press6));
  }

  function render() {
    show(teaseG, step === 0);
    show(oneG, step === 1);
    show(mixG, step === 2);
    show(emG, step === 3);
    show(kG, step === 4);
    show(smpG, step === 5);
    kIn.value = String(kSel);
    kOut.textContent = String(kSel);
    kIn.setAttribute('aria-valuetext', `K = ${kSel}`);
    const s = info();

    if (step === 0 && !teaseDrawn) {
      teaseDrawn = true;
      teaseDots.set(tR, blobSamples(SEED_SAMPLE_BLOBS));
    }
    if (step === 1) {
      show(sampKey, seed1 !== null);
      if (seed1 === null) sampPath.setAttribute('d', '');
      else {
        const h = histogram(samples1D(seed1));
        let d = '';
        h.density.forEach((v, b) => {
          const xa = P(hx(HIST.lo + b * HIST.w));
          const xb = P(hx(HIST.lo + (b + 1) * HIST.w));
          d += `${b ? 'L' : 'M'}${xa},${P(hy(v))}L${xb},${P(hy(v))}`;
        });
        sampPath.setAttribute('d', d);
      }
    }
    if (step === 3) drawEm();
    if (step === 4) {
      kTitle.textContent = `K = ${kSel}`;
      drawK();
    }
    if (step === 5) drawSamples();

    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(texts));
  }

  function ariaFor(texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Two scatter plots of three groups of points. Left, the ${N_BLOBS} training points as hollow circles; right, ${N_BLOBS} filled points sampled from a mixture of three Gaussians fitted to them. ${rest}.`;
    if (step === 1) return `A histogram of ${N_1D} values with two humps, and one Gaussian curve fitted by maximum likelihood, highest between the humps. ${rest}.`;
    if (step === 2) return `The ${N_BLOBS} training points in three groups, with one large ellipse for a single fitted Gaussian and three dashed ellipses for the Gaussians that made the data. ${rest}.`;
    if (step === 3) return `EM fitting a mixture of three Gaussians: the points, each drawn as a circle, triangle or square for its likeliest component and faded when shared, three ellipses, and the mean log-likelihood after each of the ${getLive().iter} iterations so far. ${rest}.`;
    if (step === 4) return `The training points with the ellipses of the K = ${kSel} fit, and a chart of mean log-likelihood against K from 1 to ${K_MAX}: a solid line with circles for the training points and a dashed line with triangles for the held-out points. ${rest}.`;
    return `Two scatter plots. Left, the blobs as hollow circles with ${N_BLOBS} filled points sampled from the K = ${K_TRUE} fit; right, the spiral as hollow circles with ${spiralData().length} filled points sampled from a K = ${K_SPIRAL} mixture, whose five broad dashed ellipses overlap and spread the new points over the whole disc, between the arms as well as on them. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  // ---- step 4's live run: one EM iteration every SLOW_FRAMES frames.
  let waited = 0;
  const loop = createTrainLoop({
    step: () => {
      const r = paceFrame(waited);
      waited = r.waited;
      if (r.take) getLive().step(r.take);
    },
    render,
    done: () => getLive().done,
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { waited = 0; getLive().reset(); },
    reduced,
    work: (k) => getLive().step(k),
  });
  trainGroup.addEventListener('click', () => { if (step === 3) { render(); announce(spoken()); } });

  // ---- the fits for steps 5 and 6, computed when first needed, about 6 ms per frame.
  let job = null;
  const cancelJob = () => {
    if (job !== null) cancelAnimationFrame(job);
    job = null;
  };
  const compute = (names, onStep) => {
    if (job !== null || names.every(fitReady)) return;
    const tick = () => {
      job = null;
      if (step !== onStep) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const name = names.find((n) => !fitReady(n));
        if (name === undefined) break;
        fitWork(name, 5);
      }
      if (names.every(fitReady)) { render(); announce(spoken()); } else job = requestAnimationFrame(tick);
    };
    job = requestAnimationFrame(tick);
  };

  sampBtn.addEventListener('click', () => {
    if (step === 1) seed1 = seed1 === null ? SEED_SAMPLE_1D : seed1 + 1;
    if (step === 5) {
      if (!sReady()) return;
      press6 += 1;
    }
    render();
    announce(spoken());
  });
  kIn.addEventListener('input', () => {
    kSel = Math.min(K_MAX, Math.max(1, Math.round(Number(kIn.value)) || K_TRUE));
    render();
    announce(spoken());
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i) {
      buttons.stop();
      step = i;
      cancelJob();
      if (i === 4) compute(allK(), 4);
      if (i === 5) compute([`k${K_TRUE}`, 'spiral'], 5);
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      showGroup(sampGroup, i === 1 || i === 5);
      showGroup(kGroup, i === 4);
      trainGroup.style.display = i === 3 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 3);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
