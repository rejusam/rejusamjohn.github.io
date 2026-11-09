import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { boost, stumpFit } from '../assets/classify.js';
import { sineData } from '../assets/sine-data.js';
import { mse, centralDiff, dot } from '../assets/maths.js';
import * as L from '../gradient-boosting/stage.js';

const html = readFileSync(new URL('../gradient-boosting/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);
const s2Of = () => L.run(1).boost.stumps[1];
const hOf = (s, x) => (x < s.threshold ? s.left : s.right);

test('gb data reproducible from the shared sine generator', () => {
  const tr = L.trainData();
  const te = L.testData();
  assert.deepEqual(tr, sineData({ n: 40, sd: 0.25, seed: 91 }));
  assert.deepEqual(te, sineData({ n: 300, sd: 0.25, seed: 92 }));
  assert.equal(tr.xs.length, 40);
  assert.equal(te.xs.length, 300);
  assert.deepEqual([L.SEED_TRAIN, L.SEED_TEST, L.N_TRAIN, L.N_TEST], [91, 92, 40, 300]);
  for (const s of [91, 92]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.deepEqual(L.ETAS, [0.05, 0.1, 0.3, 1]);
  assert.equal(L.M_MAX, 300);
});

test('train MSE never rises for each η on the slider', () => {
  const { xs, ys } = L.trainData();
  const n = xs.length;
  for (const eta of L.ETAS) {
    const r = L.run(eta);
    assert.equal(r.train.length, L.M_MAX + 1);
    assert.equal(r.boost.stumps.length, L.M_MAX);
    const F = ys.map(() => r.boost.f0);
    for (let m = 1; m <= L.M_MAX; m++) {
      // Never rises (the general fact), and on this data it falls every round.
      assert.ok(r.train[m] <= r.train[m - 1], `η ${eta}, round ${m}`);
      assert.ok(r.train[m] < r.train[m - 1], `η ${eta}, round ${m} not a strict fall`);
      // The reason: h is a least-squares projection of r, so r·h = ‖h‖², and the
      // sum of squares falls by exactly η(2 − η)‖h‖².
      const s = r.boost.stumps[m - 1];
      const res = ys.map((y, i) => y - F[i]);
      const h = xs.map((x) => hOf(s, x));
      close(dot(res, h), dot(h, h), 1e-9);
      close((r.train[m - 1] - r.train[m]) * n, eta * (2 - eta) * dot(h, h), 1e-9);
      xs.forEach((x, i) => { F[i] += eta * h[i]; });
    }
  }
});

test('test MSE has an interior minimum for η = 0.1', () => {
  const r = L.run(0.1);
  const te = L.testData();
  const best = L.bestM(0.1);
  assert.ok(best > 0 && best < L.M_MAX, String(best));
  assert.ok(r.test[best] < r.test[0] && r.test[best] < r.test[L.M_MAX]);
  // The cached test curve equals a fresh prediction.
  for (const m of [0, 1, 37, best, L.M_MAX]) close(r.test[m], mse(te.ys, te.xs.map((x) => r.boost.predict(x, m))), 1e-12);
  // And the training curve equals boost's own.
  const { xs, ys } = L.trainData();
  assert.deepEqual(r.train, boost(xs, ys, { M: L.M_MAX, eta: 0.1 }).trainMSE);
});

test('smaller η reaches its minimum later', () => {
  const bests = L.ETAS.map((e) => L.bestM(e));
  for (let k = 1; k < bests.length; k++) assert.ok(bests[k] < bests[k - 1], bests.join(', '));
  for (const b of bests) assert.ok(b > 0 && b < L.M_MAX);
});

test('runs are computed once per η; the rounds slider does not refit', () => {
  for (const eta of L.ETAS) assert.equal(L.run(eta), L.run(eta));
});

test('residuals are the negative gradient of ½(y − F)²', () => {
  const { xs, ys } = L.trainData();
  for (const [eta, m] of [[1, 1], [0.1, L.STEP6_M], [0.3, 50]]) {
    const r = L.residuals(eta, m);
    const b = L.run(eta).boost;
    xs.forEach((x, i) => {
      const loss = (F) => 0.5 * (ys[i] - F) ** 2;
      close(r[i], ys[i] - b.predict(x, m));
      close(-centralDiff(loss, b.predict(x, m)), r[i], 1e-8);
    });
  }
});

test('the first stump is the one-split fit to y, and is F₁ at η = 1', () => {
  const { xs, ys } = L.trainData();
  const s = stumpFit(xs, ys);
  const b = L.run(1).boost;
  close(b.stumps[0].threshold, s.threshold);
  for (const x of xs) close(b.predict(x, 1), hOf(s, x), 1e-12);
  close(L.run(1).train[1], mse(ys, xs.map((x) => hOf(s, x))));
  // F₀ is the mean, and its train MSE is the population variance of y.
  const ybar = ys.reduce((a, v) => a + v, 0) / ys.length;
  close(b.f0, ybar);
  close(L.run(1).train[0], ys.reduce((a, v) => a + (v - ybar) ** 2, 0) / ys.length);
});

test('gb page structure and contract', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Gradient boosting · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/k-means\/">/);
  assert.match(html, /href="\/ml\/random-forests\/"/);
  assert.match(html, /href="\/ml\/gradient-descent\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  for (const step of [0, 1, 2, 3, 4, 5]) {
    for (const st of [{}, { m: 0 }, { m: 300 }, { eta: 0 }, { eta: 3, m: 1 }, { eta: 3, m: 300 }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 30, r);
      }
    }
  }
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 10);
  for (const key of keys) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('gb page values match the computations', () => {
  const pv = L.pageValues();
  const r1 = L.run(1);
  const r = L.run(0.1);
  const best = L.bestM(0.1);
  assert.equal(pv['mse0'], L.fmtMSE(r1.train[0]));
  assert.equal(pv['mse1'], L.fmtMSE(r1.train[1]));
  assert.equal(pv['mse2'], L.fmtMSE(r1.train[2]));
  assert.equal(pv['split1'], L.fmt(r1.boost.stumps[0].threshold));
  assert.equal(pv['split2'], L.fmt(r1.boost.stumps[1].threshold));
  assert.equal(pv['best-m'], String(best));
  assert.equal(pv['best-test'], L.fmtMSE(r.test[best]));
  assert.equal(pv['end-test'], L.fmtMSE(r.test[L.M_MAX]));
  assert.equal(pv['end-train'], L.fmtMSE(r.train[L.M_MAX]));
  L.ETAS.forEach((e, k) => {
    assert.equal(pv[`best-m-${k}`], String(L.bestM(e)));
    assert.equal(pv[`best-test-${k}`], L.fmtMSE(L.run(e).test[L.bestM(e)]));
  });
  // Pinned, so a change to the data cannot silently change the story.
  assert.deepEqual(L.PAGE_PIN, Object.fromEntries(L.PIN_KEYS.map((k) => [k, pv[k]])));
});

test('gb wording claims hold', () => {
  const r1 = L.run(1);
  // Step 1: the stump's left level is below the mean and its right level above.
  const s = r1.boost.stumps[0];
  assert.ok(s.left < 0 && s.right > 0);
  // Step 2: on the far right (beyond the second split), most points sit below the stump.
  const res1 = L.residuals(1, 1);
  const { xs } = L.trainData();
  const far = xs.map((x, i) => i).filter((i) => xs[i] >= s2Of().threshold);
  assert.ok(far.filter((i) => res1[i] < 0).length > far.length / 2);
  // Step 3: the second stump lowers the right-hand end (x above its split).
  const s2 = r1.boost.stumps[1];
  assert.ok(s2.right < 0 && s2.left > 0 && s2.threshold > 0);
  // Step 3: the second stump's train MSE is lower than the first's.
  assert.ok(r1.train[2] < r1.train[1]);
  // Step 5: η = 1 has the highest lowest-test-MSE of the four.
  const lows = L.ETAS.map((e) => L.run(e).test[L.bestM(e)]);
  assert.equal(Math.max(...lows), lows[3]);
  // "The three smaller values of η all get down to about the same test MSE."
  const pv = L.pageValues();
  assert.equal(pv['best-test-0'], pv['best-test-1']);
  assert.equal(pv['best-test-2'], pv['best-test-1']);
  assert.notEqual(pv['best-test-3'], pv['best-test-1']);
  // Step 5: after its best round, η = 0.1's test MSE has risen by M = 300, while train still falls.
  const r = L.run(0.1);
  assert.ok(r.test[L.M_MAX] > r.test[L.bestM(0.1)]);
  // Fixed facts in the text.
  assert.match(html, /0 &lt; η &lt; 2|0 < η < 2/);
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.match(html, /Section 8\.2\.3, "Boosting"/);
  assert.match(html, /Hastie, Tibshirani &amp; Friedman,/);
  assert.match(html, /Section 10\.10, "Numerical Optimization via Gradient Boosting"/);
  assert.match(html, /10\.12\.1, "Shrinkage"/);
  assert.doesNotMatch(html, /Taylor/);
  assert.match(html, new RegExp(`after ${L.STEP6_M} rounds`));
  assert.match(html, new RegExp(`M = ${L.M_START}\\b`));
});
