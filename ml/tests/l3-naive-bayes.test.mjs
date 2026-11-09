import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { blobs2, gaussianNB, nbPosterior, accuracy } from '../assets/classify.js';
import { regionCells, linearScale } from '../assets/plot.js';
import * as L from '../naive-bayes/stage.js';

const html = readFileSync(new URL('../naive-bayes/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);

test('nb data reproducible, class-wise SDs, one stream per set', () => {
  const a = L.trainData();
  assert.deepEqual(a, L.trainData());
  assert.equal(a.X.length, 100);
  assert.equal(a.y.filter((v) => v === 1).length, 40);
  assert.deepEqual(a.y.slice(0, 60), new Array(60).fill(0));
  assert.equal(L.testData().X.length, 1000);
  assert.equal(L.testData().y.filter((v) => v === 1).length, 400);
  assert.equal(L.corrTrain().X.length, 100);
  assert.equal(L.corrTest().X.length, 1000);
  // With one SD for both classes the generator is exactly blobs2.
  const same = L.classData({ n0: 30, n1: 20, mu0: [2, 2], mu1: [4.5, 3.5], sd0: 1, sd1: 1, rho: 0.85, seed: 62 });
  assert.deepEqual(same, blobs2({ n0: 30, n1: 20, mu0: [2, 2], mu1: [4.5, 3.5], sd: 1, rho: 0.85, seed: 62 }));
  // Seeds as stated on the page.
  assert.deepEqual([L.SEED_TRAIN, L.SEED_CORR, L.SEED_TEST, L.SEED_CORR_TEST], [61, 62, 63, 64]);
  for (const s of [61, 62, 63, 64]) assert.match(html, new RegExp(`seed ${s}\\b`));
  // Pin first and last rows.
  close(a.X[0][0], 1.7159167809362208);
  close(a.X[99][1], L.trainData().X[99][1]);
  const pinned = [a.X[0][1], a.X[60][0], a.X[99][0]].map((v) => Number(v.toFixed(10)));
  assert.deepEqual(pinned, L.PINNED);
});

test('nb fit uses MLE variance (divide by the class count) and counted priors', () => {
  const { X, y } = L.trainData();
  const m = L.model();
  assert.deepEqual(m.prior, [0.6, 0.4]);
  for (const c of [0, 1]) {
    const rows = X.filter((_, i) => y[i] === c);
    for (const j of [0, 1]) {
      const mu = rows.reduce((s, r) => s + r[j], 0) / rows.length;
      const v = rows.reduce((s, r) => s + (r[j] - mu) ** 2, 0) / rows.length;
      close(m.mu[c][j], mu);
      close(m.vari[c][j], v);
    }
  }
});

test('nb posteriors sum to 1 across the grid', () => {
  const xs = linearScale(L.X_DOM, [0, 1]);
  const ys = linearScale(L.Y_DOM, [1, 0]);
  const cells = regionCells(L.GRID[0], L.GRID[1], xs, ys);
  for (const m of [L.model(), L.corrModel(), L.withPrior(L.model(), 0.1), L.withPrior(L.model(), 0.9)]) {
    for (const c of cells) {
      const p = nbPosterior(m, [c.x, c.y]);
      assert.ok(p.every((v) => v >= 0 && v <= 1));
      close(p[0] + p[1], 1);
    }
    // Far from both classes: still finite and summing to 1.
    const far = nbPosterior(m, [60, -50]);
    close(far[0] + far[1], 1);
  }
});

test('boundary in 1D is where the weighted densities cross', () => {
  for (const p1 of L.PRIORS) {
    const m1 = L.oneD(L.withPrior(L.model(), p1));
    const roots = L.boundaries1D(p1);
    assert.equal(roots.length, 2);
    const diff = (x) => { const w = L.weighted(m1, x); return w[1] - w[0]; };
    for (const r of roots) {
      // Bisection on the weighted-density difference, in a bracket around the root.
      let lo = r - 0.5, hi = r + 0.5;
      assert.ok(Math.sign(diff(lo)) !== Math.sign(diff(hi)));
      for (let k = 0; k < 200; k++) {
        const mid = (lo + hi) / 2;
        if (Math.sign(diff(mid)) === Math.sign(diff(lo))) lo = mid; else hi = mid;
      }
      close((lo + hi) / 2, r, 1e-9);
      close(nbPosterior(m1, [r])[1], 0.5, 1e-9);
    }
    // The first crossing is between the class means; the second is right of the plot.
    const [b, far] = roots;
    assert.ok(b > L.model().mu[0][0] && b < L.model().mu[1][0], String(b));
    assert.ok(far > L.X_DOM[1], String(far));
    // Class 1 wins strictly between the two crossings, class 0 outside.
    assert.ok(nbPosterior(m1, [(b + far) / 2])[1] > 0.5);
    assert.ok(nbPosterior(m1, [b - 0.01])[1] < 0.5);
    assert.ok(nbPosterior(m1, [far + 0.01])[1] < 0.5);
  }
});

test('raising the prior of class 1 grows its region', () => {
  const counts = L.PRIORS.map((p) => L.regionCount(p));
  for (let i = 1; i < counts.length; i++) assert.ok(counts[i] >= counts[i - 1], `${counts}`);
  assert.ok(counts[counts.length - 1] > counts[0]);
  // In 1D the boundary moves left (towards class 0) as the prior of class 1 rises.
  const bs = L.PRIORS.map((p) => L.boundaries1D(p)[0]);
  for (let i = 1; i < bs.length; i++) assert.ok(bs[i] < bs[i - 1]);
  assert.equal(L.PRIORS[0], 0.1);
  assert.equal(L.PRIORS[L.PRIORS.length - 1], 0.9);
  assert.ok(L.PRIORS.includes(0.4) && L.PRIORS.includes(0.8));
});

test('correlated data: mean max-posterior exceeds test accuracy', () => {
  const c = L.calib('corr');
  const m = L.calib('main');
  assert.ok(c.conf > c.acc);
  // The gap is several points on correlated data and under one point on the first data.
  assert.ok(c.conf - c.acc > 0.04, String(c.conf - c.acc));
  assert.ok(Math.abs(m.conf - m.acc) < 0.01, String(m.conf - m.acc));
  // Yet the labels are still mostly right, a few points below the rule built from the true shapes.
  assert.ok(c.acc > 0.85);
  assert.ok(c.bayes > c.acc && c.bayes - c.acc < 0.05);
  // Direct recount.
  const te = L.corrTest();
  const cm = gaussianNB(L.corrTrain().X, L.corrTrain().y);
  const yhat = te.X.map((x) => (nbPosterior(cm, x)[1] >= 0.5 ? 1 : 0));
  assert.equal(accuracy(te.y, yhat), c.acc);
  close(te.X.reduce((s, x) => s + Math.max(...nbPosterior(cm, x)), 0) / te.X.length, c.conf);
  // The correlation really is there in the training data, in both classes.
  for (const cls of [0, 1]) assert.ok(L.sampleCorr(L.corrTrain(), cls) > 0.7);
});

test('Bayes rule with counts gives the same answer as counting', () => {
  const b = L.binCounts();
  const { X, y } = L.trainData();
  const inBin = X.map((x, i) => [x[0], y[i]]).filter(([v]) => v >= L.BIN[0] && v < L.BIN[1]);
  assert.equal(b.n0, inBin.filter((r) => r[1] === 0).length);
  assert.equal(b.n1, inBin.filter((r) => r[1] === 1).length);
  const direct = b.n1 / (b.n0 + b.n1);
  const bayes = ((b.n1 / 40) * 0.4) / ((b.n0 + b.n1) / 100);
  close(direct, bayes);
  close(b.post1, direct);
  assert.deepEqual([b.n0, b.n1], [11, 5]);
});

test('nb read-out rows and page', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { prior1: 0.1 }, { prior1: 0.9 }, { x1: -1 }, { x1: 7.5 }, { q: [-1, -1.5] }, { q: [7, 5.5] }, { data: 'main' }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 30, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Naive Bayes · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/support-vector-machines\/">/);
  assert.match(html, /href="\/ml\/logistic-regression\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 10);
  for (const key of keys) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('nb page values match the computations', () => {
  const pv = L.pageValues();
  const c = L.calib('corr');
  const m = L.calib('main');
  assert.equal(pv['corr-acc'], L.pct(c.acc));
  assert.equal(pv['corr-conf'], L.pct(c.conf));
  assert.equal(pv['corr-bayes'], L.pct(c.bayes));
  assert.equal(pv['main-acc'], L.pct(m.acc));
  assert.equal(pv['main-conf'], L.pct(m.conf));
  assert.equal(pv['b1'], L.boundaries1D(0.4)[0].toFixed(2));
  assert.equal(pv['b1-double'], L.boundaries1D(0.8)[0].toFixed(2));
  assert.equal(pv['b1-far'], L.boundaries1D(0.4)[1].toFixed(1));
  assert.equal(pv['bin-n0'], '11');
  assert.equal(pv['bin-n1'], '5');
  // Pinned, so a change to the data cannot silently change the story.
  assert.deepEqual(L.PAGE_PIN, Object.fromEntries(L.PIN_KEYS.map((k) => [k, pv[k]])));
  // Quiz 2: doubling the prior moves the boundary left, and the page says so with spans.
  assert.ok(Number(pv['b1-double']) < Number(pv['b1']));
  assert.match(html, /data-for="left">[^<]*<span data-value="b1"><\/span>[^<]*<span data-value="b1-double"><\/span>/);
});

test('nb wording claims hold', () => {
  const m = L.model();
  // "class 1 is narrower in both features" (smaller fitted variance), so the boundary curves.
  for (const j of [0, 1]) assert.ok(m.vari[1][j] < m.vari[0][j]);
  // The page states MLE variance and the fixed facts.
  assert.match(html, /variances divide by its own count, n<sub>c<\/sub>/);
  assert.match(html, /the variance divides by <span class="tex" data-tex="n_c"/);
  assert.match(html, new RegExp(`${L.GRID[0]} × ${L.GRID[1]}`));
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.match(html, /Section 4\.4\.4, "Naive Bayes"/);
  assert.doesNotMatch(html, /Taylor/);
  assert.deepEqual(L.Q0, [3.5, 2.5]);
  assert.match(html, /\(3\.5, 2\.5\)/);
});

test('nb fixed facts in the text', () => {
  assert.equal(L.X1_0, 3.5);
  assert.match(html, /At x₁ = 3\.5 the class 0 curve is higher/);
  // "No training case lies beyond 6", and the second crossing is far right of every case.
  const xs1 = L.trainData().X.map((x) => x[0]);
  assert.ok(Math.max(...xs1) < 6);
  assert.ok(L.boundaries1D(0.4)[1] > 8);
  // At x₁ = 3.5 the weighted class 0 curve is higher.
  const w = L.weighted(L.oneD(L.model()), 3.5);
  assert.ok(w[0] > w[1]);
  // "Naive Bayes is ... points behind it, less than its confidence is off by."
  const c = L.calib('corr');
  assert.ok(c.bayes - c.acc < c.conf - c.acc);
  // "within a point" on the independent data.
  const m = L.calib('main');
  assert.ok(Math.abs(m.conf - m.acc) < 0.01);
  // The weighted curves' areas are the priors (trapezoid rule on a wide range).
  const m1 = L.oneD(L.model());
  for (const cls of [0, 1]) {
    let a = 0;
    const h = 0.001;
    for (let x = -10; x < 15; x += h) a += (L.weighted(m1, x)[cls] + L.weighted(m1, x + h)[cls]) * h / 2;
    assert.ok(Math.abs(a - m1.prior[cls]) < 1e-6, String(a));
  }
  // The new case's starting posterior is below 0.5 (predicted class 0).
  assert.ok(nbPosterior(L.model(), L.Q0)[1] < 0.5);
});

test('nb review fixes: two crossings and the taller class 1 curve', () => {
  // Two crossings for every prior on the slider.
  for (const p1 of L.PRIORS) assert.equal(L.boundaries1D(p1).length, 2);
  // Class 1's weighted peak (at its mean) is above class 0's at the counted prior.
  const m1 = L.oneD(L.model());
  const peak0 = L.weighted(m1, m1.mu[0][0])[0];
  const peak1 = L.weighted(m1, m1.mu[1][0])[1];
  assert.ok(peak1 > peak0, `${peak1} vs ${peak0}`);
  assert.match(html, /narrower and, even after multiplying by its smaller prior, taller/);
  assert.match(html, /can cross at two points, and here they do/);
  assert.match(html, /\\prod_\{j=1\}\^\{d\}/);
  assert.doesNotMatch(html, /\\prod_\{j=1\}\^\{p\}/);
  assert.match(html, /\(seed 5\)/);
});
