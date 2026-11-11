import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { blobs3, spiral2D, LESSON_SPIRAL, BLOBS, gmmInit, gmmEM, gmmLogLik, gmmSample, gauss1Fit, gauss2Fit, nearestDist } from '../assets/gen.js';
import * as S from '../generative-models/stage.js';

const html = readFileSync(new URL('../generative-models/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../generative-models/stage.js', import.meta.url), 'utf8');

// The 1-D values written out again from their rule: per value one uniform draw u;
// u < 0.4 gives N(−1.5, 0.5²), otherwise N(1.5, 0.75²), drawn with randn.
function rewrite1D(n, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const u = rng();
    return u < 0.4 ? -1.5 + 0.5 * randn(rng) : 1.5 + 0.75 * randn(rng);
  });
}

// EM run straight from gen.js, stopping by the page's rule.
function directFit(X, K, init) {
  let p = init;
  const L = [gmmLogLik(X, p)];
  for (let i = 0; i < S.MAX_ITERS; i++) {
    const r = gmmEM(X, p, 1);
    p = r.params;
    L.push(r.logLik[1]);
    if (L.at(-1) - L.at(-2) < S.TOL) break;
  }
  return { params: p, logLik: L };
}

// Every number of the starting parameters times (1 ± 1e-12), the sign alternating.
function nudged(init, eps) {
  let i = 0;
  const f = (v) => v * (1 + (i++ % 2 ? -eps : eps));
  const S2 = init.S.map((M) => {
    const a = f(M[0][0]), b = f(M[0][1]), c = f(M[1][1]);
    return [[a, b], [b, c]];
  });
  const pi = init.pi.map(f);
  const tot = pi.reduce((s, v) => s + v, 0);
  return { pi: pi.map((v) => v / tot), mu: init.mu.map(([x, y]) => [f(x), f(y)]), S: S2 };
}

const isUp = (L) => L.every((v, i) => i === 0 || v >= L[i - 1]);

test('data reproducible', () => {
  assert.deepEqual(S.data1D(), rewrite1D(300, 301));
  assert.deepEqual(S.held1D(), rewrite1D(300, 306));
  assert.deepEqual(S.blobsTrain(), blobs3({ n: 300, seed: 302 }));
  assert.deepEqual(S.blobsHeld(), blobs3({ n: 300, seed: 303 }));
  assert.deepEqual(S.spiralData(), spiral2D({ n: 400, seed: 305, turns: 1, noise: 0.1 }));
  assert.deepEqual(S.spiralData(), spiral2D(LESSON_SPIRAL));
  assert.deepEqual(S.freshSpiral(), spiral2D({ n: 400, seed: 307, turns: 1, noise: 0.1 }));
  assert.equal(S.SEED_INIT, 304);
  assert.equal(S.K_TRUE, 3);
  assert.equal(S.K_MAX, 5);
  assert.equal(S.K_SPIRAL, 5);
  assert.equal(S.TOL, 1e-6);
  assert.equal(S.MAX_ITERS, 500);
  // One literal value each.
  assert.equal(S.data1D()[0].toFixed(6), rewrite1D(1, 301)[0].toFixed(6));
  assert.equal(S.num(S.fit1D().mu, 2), '0.36');
  assert.equal(S.num(S.fit1D().s2, 2), '2.39');
  assert.equal(S.blobsTrain().length, 300);
  assert.equal(S.spiralData().length, 400);
});

test('1-D fit is maximum likelihood with the population variance', () => {
  const xs = S.data1D();
  const f = S.fit1D();
  const m = xs.reduce((s, x) => s + x, 0) / xs.length;
  const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
  assert.ok(Math.abs(f.mu - m) < 1e-12);
  assert.ok(Math.abs(f.s2 - v) < 1e-12);
  assert.deepEqual(f, gauss1Fit(xs));
  // Moving μ or σ² either way lowers the log-likelihood of the training values.
  const ll = (mu, s2) => S.logLik1(xs, { mu, s2 });
  for (const d of [1e-3, -1e-3]) {
    assert.ok(ll(f.mu + d, f.s2) < ll(f.mu, f.s2));
    assert.ok(ll(f.mu, f.s2 * (1 + d)) < ll(f.mu, f.s2));
  }
  // The rule's density integrates to 1 (trapezoid on a fine grid).
  let area = 0;
  for (let x = -10; x < 10; x += 0.001) area += 0.001 * (S.ruleDensity1D(x) + S.ruleDensity1D(x + 0.001)) / 2;
  assert.ok(Math.abs(area - 1) < 1e-6);
  // The histogram has area 1 (every value is inside its range).
  const h = S.histogram(xs);
  assert.ok(Math.abs(h.density.reduce((s, d) => s + d * h.w, 0) - 1) < 1e-12);
  assert.ok(Math.min(...xs) > h.lo && Math.max(...xs) < h.lo + h.w * h.density.length);
});

test('1-D story: one Gaussian misses the two humps', () => {
  const f = S.fit1D();
  const held = S.held1D();
  const one = S.logLik1(held, f);
  const rule = S.logLikRule1D(held);
  assert.ok(rule > one + 0.2, `${rule} ${one}`);
  const sv = S.storyValues();
  assert.equal(sv['ll1-one'], S.ll(one));
  assert.equal(sv['ll1-rule'], S.ll(rule));
  assert.equal(sv['ll1-gap'], S.num(rule - one, 2));
  // The curve is highest at μ̂, but the bin there holds less than half the density
  // the curve gives it, while the two humps' bins rise above the curve.
  const h = S.histogram(S.data1D());
  const peak = S.gaussPdf1(f.mu, f.mu, f.s2);
  const binAt = (x) => h.density[Math.floor((x - h.lo) / h.w)];
  assert.ok(binAt(f.mu) < peak / 2, `${binAt(f.mu)} ${peak}`);
  assert.ok(binAt(-1.5) > S.gaussPdf1(-1.5, f.mu, f.s2));
  assert.ok(binAt(1.5) > S.gaussPdf1(1.5, f.mu, f.s2));
});

test('EM curve on the page never decreases', () => {
  // Every EM run the page shows: the blobs for K = 1 to 5 (step 4 is K = 3) and the spiral.
  for (let K = 1; K <= S.K_MAX; K++) {
    const r = S.fitResult(`k${K}`);
    assert.ok(isUp(r.logLik), `K ${K}`);
    assert.ok(!r.diverged);
    assert.ok(r.logLik.every(Number.isFinite));
    assert.ok(r.iter < S.MAX_ITERS, `K ${K} stopped by the tolerance`);
  }
  const sp = S.fitResult('spiral');
  assert.ok(isUp(sp.logLik));
  assert.ok(!sp.diverged && sp.iter < S.MAX_ITERS);
  // The live run gives the same curve, one iteration at a time, as gmmEM from gen.js.
  const live = S.createFit(S.blobsTrain(), 3);
  while (!live.done) live.step(1);
  const d = directFit(S.blobsTrain(), 3, gmmInit(S.blobsTrain(), 3, 304));
  assert.deepEqual(live.logLik, d.logLik);
  assert.deepEqual(live.params, d.params);
  assert.deepEqual(live.logLik, S.fitResult('k3').logLik);
  // gmmEM in one call of that many iterations gives the same parameters.
  assert.deepEqual(gmmEM(S.blobsTrain(), gmmInit(S.blobsTrain(), 3, 304), live.iter).params, live.params);
  // Reset goes back to the start.
  live.reset();
  assert.equal(live.iter, 0);
  assert.equal(live.logLik.length, 1);
});

test('no component collapses on any page run', () => {
  const det = (M) => M[0][0] * M[1][1] - M[0][1] * M[1][0];
  for (const name of ['k1', 'k2', 'k3', 'k4', 'k5', 'spiral']) {
    const { params } = S.fitResult(name);
    for (const M of params.S) assert.ok(det(M) > 1e-3 && M[0][0] > 0, name);
    for (const p of params.pi) assert.ok(p > 0.02, name);
    assert.ok(Math.abs(params.pi.reduce((s, v) => s + v, 0) - 1) < 1e-12);
  }
  // K never exceeds the number of points (gmmInit would loop for ever).
  assert.ok(S.K_MAX <= S.blobsTrain().length && S.K_SPIRAL <= S.spiralData().length);
  assert.throws(() => S.createFit([[0, 0], [1, 1]], 3));
});

test('responsibilities match gen.js', () => {
  const X = S.blobsTrain();
  const p = S.fitResult('k3').params;
  const r = S.responsibilities(X, p);
  for (const row of r.R) assert.ok(Math.abs(row.reduce((s, v) => s + v, 0) - 1) < 1e-12);
  assert.ok(Math.abs(r.logLik - gmmLogLik(X, p)) < 1e-12);
  r.hard.forEach((k, i) => assert.equal(r.R[i][k], Math.max(...r.R[i])));
  // Shading: a point shared evenly is faint, a sure point is full.
  assert.equal(S.shade(1 / 3, 3), S.SHADE_MIN);
  assert.equal(S.shade(1, 3), 1);
});

test('held-out log-likelihoods equal gmmLogLik', () => {
  const tab = S.kTable();
  assert.equal(tab.length, 5);
  for (const row of tab) {
    const d = directFit(S.blobsTrain(), row.K, gmmInit(S.blobsTrain(), row.K, 304));
    assert.equal(row.train, d.logLik.at(-1));
    assert.equal(row.train, gmmLogLik(S.blobsTrain(), d.params));
    assert.equal(row.held, gmmLogLik(blobs3({ n: 300, seed: 303 }), d.params));
  }
  // K = 1 is the single Gaussian fit.
  const g = gauss2Fit(S.blobsTrain());
  assert.ok(Math.abs(tab[0].train - gmmLogLik(S.blobsTrain(), { pi: [1], mu: [g.mu], S: [g.S] })) < 1e-12);
  const rule = { pi: [1 / 3, 1 / 3, 1 / 3], mu: BLOBS.map((b) => b.mu), S: BLOBS.map((b) => b.S) };
  assert.equal(S.ruleMixture().pi.length, 3);
  assert.equal(S.storyValues()['rule-train'], S.ll(gmmLogLik(S.blobsTrain(), rule)));
  assert.equal(S.storyValues()['rule-held'], S.ll(gmmLogLik(S.blobsHeld(), rule)));
});

test('story statements on K', () => {
  const tab = S.kTable();
  const sv = S.storyValues();
  // Training score rises with every extra component here.
  for (let i = 1; i < tab.length; i++) assert.ok(tab[i].train > tab[i - 1].train, `K ${i + 1}`);
  // Held-out is best at the true K, by more than the rounding shown.
  const best = tab.reduce((a, b) => (b.held > a.held ? b : a));
  assert.equal(best.K, 3);
  for (const row of tab) if (row.K !== 3) assert.notEqual(S.ll(row.held), S.ll(best.held));
  // K = 4 and 5 fit the training points better than K = 3 but score lower on held-out.
  assert.ok(tab[3].train > tab[2].train && tab[3].held < tab[2].held);
  assert.ok(tab[4].train > tab[2].train && tab[4].held < tab[2].held);
  // The rule that made the data scores a little higher on held-out than the K = 3 fit,
  // and lower on the training points.
  const rule = S.ruleMixture();
  assert.ok(gmmLogLik(S.blobsHeld(), rule) > tab[2].held);
  assert.ok(gmmLogLik(S.blobsTrain(), rule) < tab[2].train);
  for (let K = 1; K <= 5; K++) {
    assert.equal(sv[`train-k${K}`], S.ll(tab[K - 1].train));
    assert.equal(sv[`held-k${K}`], S.ll(tab[K - 1].held));
  }
  // The win is narrow: K = 4 and 5 are each about 0.02 lower, and another held-out set
  // can put K = 4 first.
  assert.equal(sv['held-gap'], S.num(tab[2].held - tab[3].held, 2));
  assert.equal(sv['held-gap'], S.num(tab[2].held - tab[4].held, 2));
  assert.equal(sv['held-gap'], '0.02');
  assert.ok(tab[2].held - tab[3].held < 0.05 && tab[2].held - tab[4].held < 0.05);
  let k4first = 0;
  for (let seed = 400; seed < 450; seed++) {
    const H = blobs3({ n: 300, seed });
    const sc = [1, 2, 3, 4, 5].map((K) => gmmLogLik(H, S.fitResult(`k${K}`).params));
    if (sc.indexOf(Math.max(...sc)) === 3) k4first++;
  }
  assert.ok(k4first >= 1, 'another held-out set puts K = 4 first');
  // Step 4: the curve rises fast, nearly stalls, then climbs again before it settles.
  const L = S.fitResult('k3').logLik;
  const rise = (i) => L[i] - L[i - 1];
  const stall = S.STALL;
  for (let i = stall[0]; i <= stall[1]; i++) assert.ok(rise(i) < 0.005, `stall ${i}`);
  assert.ok(L[stall[0] - 1] - L[0] > 0.9);
  let later = 0;
  for (let i = stall[1] + 1; i < L.length; i++) later = Math.max(later, rise(i));
  assert.ok(later > 0.03);
  assert.equal(sv['iters-k3'], String(S.fitResult('k3').iter));
  assert.equal(sv['end-k3'], S.ll(L.at(-1)));
  assert.equal(sv['start-k3'], S.ll(L[0]));
});

test('spiral nearestDist values equal the helper', () => {
  const sp = S.spiralData();
  const fresh = S.freshSpiral();
  const fit = S.fitResult('spiral').params;
  const q = S.quality(S.spiralSamples(S.SEED_SAMPLE_SPIRAL), sp);
  const smp = gmmSample(fit, 400, 310);
  assert.equal(q.precision, nearestDist(smp, sp));
  assert.equal(q.coverage, nearestDist(sp, smp));
  const qf = S.quality(fresh, sp);
  assert.equal(qf.precision, nearestDist(fresh, sp));
  assert.equal(qf.coverage, nearestDist(sp, fresh));
  const sv = S.storyValues();
  assert.equal(sv['fresh-prec'], S.dist(qf.precision));
  assert.equal(sv['fresh-cov'], S.dist(qf.coverage));
  // Over the first 20 presses: the mixture's new points sit about three times as far
  // from the data as fresh points from the rule, and cover the data less well.
  const presses = Array.from({ length: S.PRESSES }, (_, i) => S.quality(S.spiralSamples(310 + i), sp));
  const ps = presses.map((p) => p.precision);
  const cs = presses.map((p) => p.coverage);
  for (const p of presses) {
    assert.ok(p.precision / qf.precision > 2.5 && p.precision / qf.precision < 3.5);
    assert.ok(p.coverage / qf.coverage > 1.5 && p.coverage / qf.coverage < 2);
  }
  // "spread over the whole disc, between the arms": on every draw over a tenth of the new
  // points are more than 0.3 from any data point; no fresh point from the rule is.
  const far = (A) => A.filter((a) => Math.min(...sp.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1]))) > 0.3).length / A.length;
  assert.equal(far(fresh), 0);
  for (let i = 0; i < S.PRESSES; i++) assert.ok(far(S.spiralSamples(310 + i)) > 0.1, String(i));
  assert.equal(sv['spiral-prec-lo'], S.dist(Math.min(...ps)));
  assert.equal(sv['spiral-prec-hi'], S.dist(Math.max(...ps)));
  assert.equal(sv['spiral-cov-lo'], S.dist(Math.min(...cs)));
  assert.equal(sv['spiral-cov-hi'], S.dist(Math.max(...cs)));
  // The blobs: new points sit about as close to the data as held-out points from the rule.
  const tr = S.blobsTrain();
  const qh = S.quality(S.blobsHeld(), tr);
  assert.equal(sv['held-prec'], S.dist(qh.precision));
  assert.equal(sv['held-cov'], S.dist(qh.coverage));
  const bp = Array.from({ length: S.PRESSES }, (_, i) => S.quality(S.blobSamples(309 + i), tr));
  for (const p of bp) {
    assert.ok(Math.abs(p.precision / qh.precision - 1) < 0.15);
    assert.ok(Math.abs(p.coverage / qh.coverage - 1) < 0.2);
  }
  assert.equal(sv['blob-prec-lo'], S.dist(Math.min(...bp.map((p) => p.precision))));
  assert.equal(sv['blob-prec-hi'], S.dist(Math.max(...bp.map((p) => p.precision))));
  assert.equal(sv['blob-cov-lo'], S.dist(Math.min(...bp.map((p) => p.coverage))));
  assert.equal(sv['blob-cov-hi'], S.dist(Math.max(...bp.map((p) => p.coverage))));
});

test('sampling reproducible', () => {
  // Every distance comparison on the page is at equal sizes (300 vs 300, 400 vs 400).
  assert.equal(S.blobSamples(309).length, S.blobsTrain().length);
  assert.equal(S.blobsHeld().length, S.blobsTrain().length);
  assert.equal(S.spiralSamples(310).length, S.spiralData().length);
  assert.equal(S.freshSpiral().length, S.spiralData().length);
  assert.equal(S.blobsTrain().length, 300);
  assert.equal(S.spiralData().length, 400);
  assert.deepEqual(S.blobSamples(309), S.blobSamples(309));
  assert.deepEqual(S.blobSamples(309), gmmSample(S.fitResult('k3').params, 300, 309));
  assert.notDeepEqual(S.blobSamples(309), S.blobSamples(310));
  assert.deepEqual(S.spiralSamples(310), gmmSample(S.fitResult('spiral').params, 400, 310));
  const f = S.fit1D();
  const rng = mulberry32(308);
  const want = Array.from({ length: 300 }, () => f.mu + Math.sqrt(f.s2) * randn(rng));
  assert.deepEqual(S.samples1D(308), want);
  // A sample is new: none of the drawn points is a training point.
  const keys = new Set(S.blobsTrain().map((p) => p.join()));
  assert.ok(S.blobSamples(309).every((p) => !keys.has(p.join())));
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const sv = S.storyValues();
  const tr = S.blobsTrain();
  const he = S.blobsHeld();
  for (const eps of [1e-12, -1e-12]) {
    for (let K = 1; K <= 5; K++) {
      const d = directFit(tr, K, nudged(gmmInit(tr, K, 304), eps));
      assert.equal(S.ll(d.logLik.at(-1)), sv[`train-k${K}`], `train K ${K}`);
      assert.equal(S.ll(gmmLogLik(he, d.params)), sv[`held-k${K}`], `held K ${K}`);
      if (K === 4) assert.equal(S.num(gmmLogLik(he, directFit(tr, 3, nudged(gmmInit(tr, 3, 304), eps)).params) - gmmLogLik(he, d.params), 2), sv['held-gap']);
      if (K === 3) {
        assert.equal(String(d.logLik.length - 1), sv['iters-k3']);
        assert.equal(S.ll(d.logLik[0]), sv['start-k3']);
      }
    }
    const sp = S.spiralData();
    const d = directFit(sp, 5, nudged(gmmInit(sp, 5, 304), eps));
    const ps = [], cs = [];
    for (let i = 0; i < S.PRESSES; i++) {
      const smp = gmmSample(d.params, 400, 310 + i);
      ps.push(nearestDist(smp, sp));
      cs.push(nearestDist(sp, smp));
    }
    assert.equal(S.dist(Math.min(...ps)), sv['spiral-prec-lo']);
    assert.equal(S.dist(Math.max(...ps)), sv['spiral-prec-hi']);
    assert.equal(S.dist(Math.min(...cs)), sv['spiral-cov-lo']);
    assert.equal(S.dist(Math.max(...cs)), sv['spiral-cov-hi']);
    const bd = directFit(tr, 3, nudged(gmmInit(tr, 3, 304), eps));
    const bp = [], bc = [];
    for (let i = 0; i < S.PRESSES; i++) {
      const smp = gmmSample(bd.params, 300, 309 + i);
      bp.push(nearestDist(smp, tr));
      bc.push(nearestDist(tr, smp));
    }
    assert.equal(S.dist(Math.min(...bp)), sv['blob-prec-lo']);
    assert.equal(S.dist(Math.max(...bp)), sv['blob-prec-hi']);
    assert.equal(S.dist(Math.min(...bc)), sv['blob-cov-lo']);
    assert.equal(S.dist(Math.max(...bc)), sv['blob-cov-hi']);
  }
});

test('read-out rows', () => {
  for (let step = 0; step < 6; step++) {
    const rows = S.readout(step, S.infoFor(step));
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) assert.ok(!/NaN|undefined|Infinity/.test(r), r);
  }
  // Step 4's read-out before and after the run.
  const live = S.createFit(S.blobsTrain(), 3);
  const before = S.readout(3, S.liveInfo(live, false));
  assert.match(before[0], /iteration 0/);
  while (!live.done) live.step(1);
  const after = S.readout(3, S.liveInfo(live, false));
  assert.match(after[0], new RegExp(`iteration ${S.fitResult('k3').iter}`));
  assert.match(after[0], /finished/);
  assert.match(after[1], new RegExp(S.ll(S.fitResult('k3').logLik.at(-1))));
});

test('pacing: one EM iteration every SLOW_FRAMES frames', () => {
  let waited = 0;
  let taken = 0;
  for (let f = 0; f < S.SLOW_FRAMES * 3; f++) {
    const r = S.paceFrame(waited);
    waited = r.waited;
    taken += r.take;
  }
  assert.equal(taken, 3);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Generative models · ML Explorer<\/title>/);
  assert.match(html, /<h1>Generative models<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/generative-models\/">/);
  for (const seed of [301, 302, 303, 304, 305, 306, 307, 308, 309, 310]) assert.match(html, new RegExp(`\\b${seed}\\b`), String(seed));
  assert.match(html, /not 3\.14/);
  assert.match(html, /population variance/);
  assert.match(html, /local maximum/);
  assert.match(html, /far larger scale/);
  assert.match(html, /K = 5/);
  assert.match(html, /enough components/);
  assert.match(html, /href="\/ml\/k-means\/"/);
  assert.match(html, /href="\/ml\/logistic-regression\/"/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /href="\/ml\/llms\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/diffusion\/">Next lesson: Diffusion: from noise to data →<\/a>/);
  assert.match(html, /Dempster, Laird &amp; Rubin/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understand|thinks|knows|reasons|creative|intelligen|conscious|hallucinat/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
  assert.doesNotMatch(src, /localStorage|sessionStorage|fetch\(/);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What can a generative model do that a classifier such as logistic regression cannot\?/);
  assert.match(html, /Why judge K on held-out data\?/);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const usedPv = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(usedPv.has(key), `unused page value ${key}`);
  const sv = S.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  const usedSv = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(usedSv.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
