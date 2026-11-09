import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, uniform, olsFit } from '../assets/maths.js';
import { sigmoid, logisticFit, logLoss, confusion, blobs2 } from '../assets/classify.js';
import * as L from '../logistic-regression/stage.js';

const html = readFileSync(new URL('../logistic-regression/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);

test('logistic data reproducible', () => {
  const d = L.data1D();
  assert.deepEqual(d, L.data1D());
  // The generator, written out: x ~ U(0, 10), then y = 1 when rng() < σ(−5 + x).
  const rng = mulberry32(51);
  const x = [];
  const y = [];
  for (let i = 0; i < 30; i++) {
    const xi = uniform(rng, 0, 10);
    x.push(xi);
    y.push(rng() < sigmoid(-5 + xi) ? 1 : 0);
  }
  assert.deepEqual(d, { x, y });
  assert.equal(L.SEED, 51);
  assert.equal(L.SEED_2D, 52);
  assert.deepEqual(L.data2D(), blobs2({ n0: 40, n1: 40, mu0: [2, 2], mu1: [4, 3.5], sd: 1, seed: 52 }));
  assert.equal(d.y.filter((v) => v === 1).length, 13);
  close(d.x[0], 1.4702514582313597);
});

test('logistic 1D data not separable and fit converges', () => {
  const { x, y } = L.data1D();
  const x0 = x.filter((_, i) => y[i] === 0);
  const x1 = x.filter((_, i) => y[i] === 1);
  // Separable on one feature means one class lies wholly to one side of the other.
  assert.ok(Math.max(...x0) > Math.min(...x1));
  assert.ok(Math.max(...x1) > Math.min(...x0));
  const f = L.fit1D();
  assert.equal(f.ok, true);
  assert.ok(f.gradNorm < 1e-8);
  assert.deepEqual(f, logisticFit(x.map((v) => [v]), y));
  // The 2D data: a finite stationary point, which separable data never has.
  const f2 = L.fit2D();
  assert.equal(f2.ok, true);
  assert.ok(f2.gradNorm < 1e-8);
  assert.ok(L.acc2D() < 1);
});

test('OLS line leaves [0,1] somewhere in [0,10]', () => {
  const { x, y } = L.data1D();
  const o = L.olsLine();
  const ref = olsFit(x, y);
  close(o.intercept, ref.intercept);
  close(o.slope, ref.slope);
  // A straight line takes its extremes at the ends of the interval.
  assert.ok(o.at0 < 0);
  assert.ok(o.at10 > 1);
  close(o.at0, ref.intercept);
  close(o.at10, ref.intercept + 10 * ref.slope);
  close(o.intercept + o.slope * o.cross0, 0);
  close(o.intercept + o.slope * o.cross1, 1);
  assert.ok(o.cross0 > 0 && o.cross1 < 10);
});

test('sigmoid model and per-point penalties', () => {
  const { y } = L.data1D();
  for (const [b, w] of [[-2, 0.4], [-10, 2.5], [4, -1], [0, 0], [-5.4, 1.07]]) {
    const p = L.probs(b, w);
    const pen = L.penalties(b, w);
    pen.forEach((v, i) => {
      assert.ok(Number.isFinite(v) && v >= 0);
      close(v, logLoss([y[i]], [p[i]]));
    });
    close(L.meanLoss(b, w), logLoss(y, p));
    close(L.meanLoss(b, w), pen.reduce((s, v) => s + v, 0) / pen.length);
  }
  // b = w = 0 gives p̂ = 0.5 for everyone: mean loss ln 2.
  close(L.meanLoss(0, 0), Math.log(2));
  // The log-odds of p̂ is the linear score.
  const p = L.probs(-2, 0.4);
  L.data1D().x.forEach((xi, i) => close(Math.log(p[i] / (1 - p[i])), -2 + 0.4 * xi, 1e-9));
  // Confident and wrong costs most: the largest penalty at the fit is a wrong prediction made with p̂ far from 0.5.
  const f = L.fit1D();
  const worst = L.worstPoint(f.w0, f.w[0]);
  const pf = L.probs(f.w0, f.w[0]);
  assert.notEqual(pf[worst.i] >= 0.5 ? 1 : 0, y[worst.i]);
  assert.equal(worst.y, 0);
  assert.ok(Math.abs(pf[worst.i] - 0.5) > 0.25);
});

test('gradient descent falls to the Newton answer and never rises', () => {
  const path = L.gdPath();
  assert.equal(path.length, L.GD_STEPS + 1);
  const f = L.fit1D();
  const min = L.meanLoss(f.w0, f.w[0]);
  for (let t = 1; t < path.length; t++) assert.ok(path[t].loss <= path[t - 1].loss + 1e-15, String(t));
  for (const s of path) assert.ok(s.loss >= min - 1e-12);
  const last = path[path.length - 1];
  assert.ok(last.loss - min < 1e-6);
  close(last.w0, f.w0, 0.01);
  close(last.w[0], f.w[0], 0.01);
  // η is below 2 / (largest curvature anywhere), since p(1 − p) ≤ 1/4.
  const { x } = L.data1D();
  const m1 = x.reduce((s, v) => s + v, 0) / x.length;
  const m2 = x.reduce((s, v) => s + v * v, 0) / x.length;
  const a = 0.25, c = 0.25 * m2, bb = 0.25 * m1;
  const top = (a + c) / 2 + Math.hypot((a - c) / 2, bb);
  assert.ok(L.ETA < 2 / top);
  const k = L.gdWithin(1e-3);
  assert.ok(path[k].loss - min < 1e-3 && path[k - 1].loss - min >= 1e-3);
});

test('threshold 0.8 vs 0.5: recall does not rise', () => {
  const grid = Array.from({ length: 99 }, (_, i) => (i + 1) / 100);
  let prev = null;
  for (const t of grid) {
    const c = L.confusionAt(t);
    if (prev) {
      assert.ok(c.recall <= prev.recall);
      assert.ok(c.tp + c.fp <= prev.tp + prev.fp);
    }
    prev = c;
  }
  const a = L.confusionAt(0.5);
  const b = L.confusionAt(0.8);
  assert.ok(b.recall < a.recall);
  assert.ok(b.precision > a.precision);
  // The slider never reaches a threshold where nothing is predicted 1.
  for (const t of L.THRESHOLDS) assert.ok(L.confusionAt(t).precision !== null, String(t));
  // Predicting 1 for p̂ ≥ t is the same as hours ≥ the cut-off.
  const { x } = L.data1D();
  for (const t of L.THRESHOLDS) {
    const cut = L.cutoffHours(t);
    const p = L.fitProbs();
    x.forEach((xi, i) => {
      assert.ok(Math.abs(xi - cut) > 1e-9);
      assert.equal(p[i] >= t, xi >= cut);
    });
  }
});

test('quiz numbers equal confusion() at those thresholds', () => {
  const { y } = L.data1D();
  const p = L.fitProbs();
  const at = (t) => confusion(y, p.map((q) => (q >= t ? 1 : 0)));
  const a = at(0.5);
  const b = at(0.8);
  const form = html.slice(html.indexOf('id="q2"'), html.indexOf('</form>', html.indexOf('id="q2"')));
  const right = form.slice(form.indexOf('data-for="trade"'));
  const nums = [L.pct(a.precision), L.pct(b.precision), L.pct(a.recall), L.pct(b.recall)];
  assert.deepEqual(nums, ['76.9%', '100%', '76.9%', '53.8%']);
  const why = right.slice(0, right.indexOf('</p>'));
  for (const n of nums) assert.ok(why.includes(n), n);
  assert.ok(why.indexOf(nums[0]) < why.indexOf(nums[1]));
  assert.ok(why.indexOf(nums[2]) < why.indexOf(nums[3]));
  assert.match(form, /data-answer="trade"/);
  // The counts behind them.
  assert.ok(why.includes(`${b.tp} of ${b.tp + b.fp}`));
  assert.ok(why.includes(`${b.tp} of ${b.tp + b.fn}`));
});

test('logistic read-out rows and page', () => {
  const states = [{}, { b: -10, w: 2.5 }, { b: 4, w: -1 }, { b: 0, w: 0 }, { t: 0.05 }, { t: 0.95 }, { gd: 0 }, { gd: L.GD_STEPS }];
  for (let step = 0; step < 6; step++) {
    for (const st of states) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Logistic regression · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /seed 51/);
  assert.match(html, /seed 52/);
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/naive-bayes\/">/);
  assert.match(html, /href="\/ml\/gradient-descent\/"/);
  assert.match(html, /href="\/ml\/what-is-a-model\/"/);
  assert.match(html, /href="\/ml\/k-nearest-neighbours\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  // Every filled value is already written in the page, and equals the computed one.
  const pv = L.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 15);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('logistic page values match the computations', () => {
  const pv = L.pageValues();
  const f = L.fit1D();
  const o = L.olsLine();
  assert.equal(pv['n'], '30');
  assert.equal(pv['n-pass'], '13');
  assert.equal(pv['n-fail'], '17');
  assert.equal(pv['ols-0'], L.num(o.at0, 2));
  assert.equal(pv['ols-0'], '−0.19');
  assert.equal(pv['ols-10'], '1.17');
  assert.equal(pv['fit-b'], '−5.41');
  assert.equal(pv['fit-w'], '1.07');
  close(Number(pv['fit-mid']), -f.w0 / f.w[0], 0.005);
  assert.equal(pv['fit-loss'], L.meanLoss(f.w0, f.w[0]).toFixed(3));
  assert.equal(pv['newton-iters'], String(f.iters));
  assert.equal(pv['gd-eta'], String(L.ETA));
  assert.equal(pv['gd-steps'], String(L.GD_STEPS));
  assert.equal(pv['gd-within'], String(L.gdWithin(1e-3)));
  const c = L.confusionAt(0.5);
  assert.equal(pv['cm-tp'], String(c.tp));
  assert.equal(pv['cm-fp'], String(c.fp));
  assert.equal(pv['cm-tn'], String(c.tn));
  assert.equal(pv['cm-fn'], String(c.fn));
  assert.equal(pv['acc-50'], L.pct(c.accuracy));
  const f2 = L.fit2D();
  assert.equal(pv['w1'], L.num(f2.w[0], 2));
  assert.equal(pv['w2'], L.num(f2.w[1], 2));
  assert.equal(pv['b2'], L.num(f2.w0, 2));
  assert.equal(pv['acc-2d'], L.pct(L.acc2D()));
  assert.equal(L.num(-0.004, 2), '0.00');
  assert.equal(L.pct(1), '100%');
});

test('logistic wording constants are computed', () => {
  const pv = L.pageValues();
  assert.equal(pv['pen-half'], (-Math.log(0.5)).toFixed(2));
  assert.equal(pv['pen-half'], '0.69');
  assert.equal(pv['pen-hundredth'], (-Math.log(0.01)).toFixed(1));
  assert.equal(pv['pen-hundredth'], '4.6');
  assert.equal(pv['pen-max'], String(L.PEN_MAX));
  assert.equal(pv['grid'], `${L.GRID[0]} × ${L.GRID[1]}`);
  // The precision dip: consecutive slider values, precision falls, and the one
  // student who leaves the predicted passes really passed (TP down by 1, FP unchanged).
  const [t0, t1] = L.precisionDip();
  assert.ok(Math.abs(t1 - t0 - 0.05) < 1e-12);
  assert.ok(L.THRESHOLDS.includes(t0) && L.THRESHOLDS.includes(t1));
  const a = L.confusionAt(t0);
  const b = L.confusionAt(t1);
  assert.ok(b.precision < a.precision);
  assert.equal(a.tp - b.tp, 1);
  assert.equal(a.fp, b.fp);
  assert.equal(pv['dip-p0'], L.pct(a.precision));
  assert.equal(pv['dip-p1'], L.pct(b.precision));
  assert.equal(pv['dip-t0'], '0.20');
  assert.equal(pv['dip-t1'], '0.25');
  assert.equal(pv['gd-eta'], '0.25');
  // No hand-typed copies of these values outside their spans.
  assert.doesNotMatch(html, /costs 0\.69/);
  assert.doesNotMatch(html, /grid of 60 × 56/);
  assert.doesNotMatch(html, /η = 0\.25/);
});
