import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn, eig2Sym } from '../assets/maths.js';
import { cov2, pca2, projVar, perpMSE } from '../assets/unsupervised.js';
import * as L from '../pca/stage.js';

const html = readFileSync(new URL('../pca/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-9) => assert.ok(Math.abs(u - v) <= tol, `${u} vs ${v}`);
const rad = (d) => (d * Math.PI) / 180;

test('pca data reproducible', () => {
  const X = L.makeData();
  assert.deepEqual(X, L.makeData());
  assert.equal(X.length, 50);
  assert.equal(L.SEED, 111);
  // The generator, written out again.
  const rng = mulberry32(111);
  for (const [h, a] of X) {
    const hh = 170 + 8 * randn(rng);
    const aa = 0.9 * (hh - 170) + 170 + 4 * randn(rng);
    close(h, hh, 0);
    close(a, aa, 0);
  }
  // Pin the first row, so a change to the generator cannot pass unnoticed.
  close(X[0][0], 170.37204819306274, 1e-12);
  close(X[0][1], 161.82959887617258, 1e-12);
  // mm: the same people, arm span times 10.
  L.data('mm').forEach((r, i) => { close(r[0], X[i][0], 0); close(r[1], 10 * X[i][1], 1e-9); });
});

test('slider maximum matches the eigenvector', () => {
  const X = L.data('cm');
  const p = pca2(X);
  const best = L.gridBest();
  assert.deepEqual(L.ANGLES, Array.from({ length: 180 }, (_, i) => i));
  // A brute-force argmax over the slider's 1° grid.
  let arg = 0;
  for (const d of L.ANGLES) if (projVar(X, rad(d)) > projVar(X, rad(arg))) arg = d;
  assert.equal(best, arg);
  const a1 = L.angleDeg(p.vecs[0]);
  assert.ok(Math.abs(best - a1) <= 1, `${best} vs ${a1}`);
  // No angle at all beats the eigenvector, and its variance is gamma 1.
  close(projVar(X, rad(a1)), p.gammas[0], 1e-9);
  for (let d = 0; d < 180; d += 0.1) assert.ok(projVar(X, rad(d)) <= p.gammas[0] + 1e-9);
  // S v = gamma v for both directions, at right angles.
  p.vecs.forEach((v, k) => {
    close(p.S[0][0] * v[0] + p.S[0][1] * v[1], p.gammas[k] * v[0], 1e-9);
    close(p.S[1][0] * v[0] + p.S[1][1] * v[1], p.gammas[k] * v[1], 1e-9);
  });
  close(p.vecs[0][0] * p.vecs[1][0] + p.vecs[0][1] * p.vecs[1][1], 0, 1e-12);
  // The least variance is along PC2.
  close(projVar(X, rad(L.angleDeg(p.vecs[1]))), p.gammas[1], 1e-9);
});

test('explained shares sum to 1', () => {
  for (const u of L.UNITS_ALL) {
    const p = L.fit(u);
    close(p.explained[0] + p.explained[1], 1, 1e-12);
    assert.ok(p.gammas[0] >= p.gammas[1]);
    // gamma 1 + gamma 2 = trace of S = total sample variance.
    close(p.gammas[0] + p.gammas[1], p.S[0][0] + p.S[1][1], 1e-9 * p.S[1][1]);
  }
  const p = L.fit('cm');
  assert.deepEqual(p.gammas, eig2Sym(cov2(L.data('cm'))));
});

test('along + across = total at every angle; mean squared distance has the n/(n-1) factor', () => {
  const X = L.data('cm');
  const n = X.length;
  const S = cov2(X);
  const tot = S[0][0] + S[1][1];
  close(L.total(), tot, 1e-12);
  for (let d = 0; d < 180; d += 0.5) {
    close(L.along(d) + L.across(d), tot, 1e-9);
    close(L.across(d), perpMSE(X, rad(d)) * n / (n - 1), 1e-12);
  }
  // At 0° the line is the height axis.
  close(L.along(0), S[0][0], 1e-9);
  close(L.across(0), S[1][1], 1e-9);
  // Rebuilding from PC1: mean squared error = (n - 1)/n × gamma 2.
  const p = pca2(X);
  close(L.reconMSE(), (n - 1) / n * p.gammas[1], 1e-9);
  const R = L.rebuilt();
  const direct = X.reduce((s, x, i) => s + (x[0] - R[i][0]) ** 2 + (x[1] - R[i][1]) ** 2, 0) / n;
  close(L.reconMSE(), direct, 1e-9);
  // Each rebuilt point lies on the PC1 line through the mean.
  R.forEach((r, i) => {
    const t = L.scores()[i];
    close(r[0], p.mean[0] + t * p.vecs[0][0], 1e-9);
    close(r[1], p.mean[1] + t * p.vecs[0][1], 1e-9);
  });
  // The scores' sample variance is gamma 1.
  const t = L.scores();
  close(t.reduce((s, v) => s + v * v, 0) / (n - 1), p.gammas[0], 1e-9);
  close(t.reduce((s, v) => s + v, 0), 0, 1e-9);
});

test('mm version: PC1 swings to the arm-span axis; standardised PCA is the same in cm and mm', () => {
  const cm = L.pc1Angle('cm');
  const mm = L.pc1Angle('mm');
  // In cm the two spreads are alike and PC1 runs near the diagonal.
  assert.ok(cm > 40 && cm < 55, String(cm));
  // In mm it lies within 5° of the arm-span axis (90°). See the report: 1° is not reachable with ×10.
  assert.ok(Math.abs(90 - mm) < 5, String(mm));
  assert.ok(L.fit('mm').explained[0] > 0.99);
  // Standardised: the same direction and shares from cm or mm.
  const a = L.fit('std');
  const b = L.fit('std-mm');
  close(a.vecs[0][0], b.vecs[0][0], 1e-12);
  close(a.vecs[0][1], b.vecs[0][1], 1e-12);
  close(a.explained[0], b.explained[0], 1e-12);
  close(L.pc1Angle('std'), 45, 1e-9);
  close(L.pc1Angle('std-mm'), 45, 1e-9);
  // With two standardised features the shares are (1 ± r)/2, whatever the divisor.
  const r = L.corr();
  close(a.explained[0], (1 + r) / 2, 1e-12);
  // The z-scores use the population SD (divide by n).
  const Z = L.data('std');
  for (const j of [0, 1]) {
    const c = Z.map((z) => z[j]);
    close(c.reduce((s, v) => s + v, 0), 0, 1e-9);
    close(c.reduce((s, v) => s + v * v, 0) / c.length, 1, 1e-12);
  }
  // Covariance of those z-scores is n/(n-1) times the correlation matrix.
  const S = cov2(Z);
  close(S[0][1], (50 / 49) * r, 1e-12);
});

test('pca read-out rows', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { theta: 0 }, { theta: 179 }, { theta: 47.66 }, { centred: false }, { units: 'mm' }, { units: 'std' }, { rebuilt: false }]) {
      const rows = L.readout(step, st);
      assert.ok(rows.length >= 2 && rows.length <= 6);
      for (const r of rows) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 26, r);
      }
    }
  }
});

test('pca page contract', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Principal component analysis · ML Explorer<\/title>/);
  assert.match(html, /Level 3 · Classic ML/);
  assert.match(html, /seed 111/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/neurons\/">Next lesson: The artificial neuron →<\/a>/);
  assert.match(html, /href="\/ml\/k-means\/"/);
  assert.match(html, /href="\/ml\/loss-surface\/"/);
  assert.match(html, /href="\/ml\/centre-and-spread\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.doesNotMatch(html, /Taylor/);
  assert.match(html, /Deisenroth, Faisal &amp; Ong/);
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 15);
  for (const key of keys) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('pca page values match the computations', () => {
  const pv = L.pageValues();
  for (const [k, v] of Object.entries(L.PAGE_PIN)) assert.equal(pv[k], v, k);
  // The rounded numbers on the page add up as the text says.
  const f = (k) => Number(pv[k]);
  close(f('var-h') + f('var-a'), f('total'), 1e-9);
  close(f('gamma1') + f('gamma2'), f('total'), 1e-9);
  // "The best whole degree on the slider is 48°", next to the exact 47.7°.
  assert.equal(pv['grid-best'], String(L.gridBest()));
  // The quiz explanation quotes the mm angle from a span.
  assert.match(html, /data-for="arm">[^<]*<span data-value="mm-angle"><\/span>/);
  // Wording claims: height's spread is a little smaller than arm span's in cm.
  const S = L.fit('cm').S;
  assert.ok(S[0][0] < S[1][1]);
  // The highlighted person sits well off the line: their lost part is above the average.
  const p = L.fit('cm');
  const X = L.data('cm');
  const R = L.rebuilt();
  const d2 = (X[0][0] - R[0][0]) ** 2 + (X[0][1] - R[0][1]) ** 2;
  assert.ok(d2 > L.reconMSE());
  assert.ok(p.explained[0] > 0.8);
  // "a little taller than average, with arms well short of average"
  const c0 = [X[0][0] - p.mean[0], X[0][1] - p.mean[1]];
  assert.ok(c0[0] > 0 && c0[0] < 2 && c0[1] < -5);
  // "the slider's best whole degree is 48°" and the exact angle rounds differently.
  assert.equal(L.gridBest(), 48);

  // Misconception 1: the regression line of arm span on height is a different line.
  const regAngle = (Math.atan(S[0][1] / S[0][0]) * 180) / Math.PI;
  assert.ok(Math.abs(regAngle - L.pc1Angle('cm')) > 5, String(regAngle));
  // PC1 loses the least of any line through the centre (checked on a fine grid).
  for (let d = 0; d < 180; d += 0.1) assert.ok(L.across(d) >= L.across(L.pc1Angle('cm')) - 1e-9);
});
