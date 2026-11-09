import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { blobs2, knnPredict, accuracy } from '../assets/classify.js';
import * as L from '../k-nearest-neighbours/stage.js';

const html = readFileSync(new URL('../k-nearest-neighbours/index.html', import.meta.url), 'utf8');

test('knn data reproducible', () => {
  const a = L.trainData();
  const b = L.trainData();
  assert.deepEqual(a, b);
  assert.deepEqual(a, blobs2({ n0: 20, n1: 20, mu0: [2, 2], mu1: [4, 3.4], sd: 1.1, seed: 41 }));
  assert.deepEqual(L.testData(), blobs2({ n0: 150, n1: 150, mu0: [2, 2], mu1: [4, 3.4], sd: 1.1, seed: 42 }));
  assert.equal(a.X.length, 40);
  assert.equal(a.y.filter((v) => v === 1).length, 20);
  assert.equal(L.testData().X.length, 300);
  // Pin the first rows, so a change to the generator cannot pass unnoticed.
  const close = (u, v) => assert.ok(Math.abs(u - v) < 1e-12, `${u} vs ${v}`);
  close(a.X[0][0], 1.3761373064504236);
  close(a.X[0][1], 0.9855457097636011);
  close(a.X[39][0], 4.7783297921738015);
  close(a.X[39][1], 3.4314353087871896);
  assert.notDeepEqual(L.testData().X[0], a.X[0]);
});

test('k=1 training accuracy is 1', () => {
  const rows = L.kTable();
  assert.equal(rows[0].k, 1);
  assert.equal(rows[0].train, 1);
  // Every other k on the slider is below 100% on the training points.
  for (const r of rows.slice(1)) assert.ok(r.train < 1, String(r.k));
  // Each training point is its own nearest neighbour (no exact duplicates).
  const { X, y } = L.trainData();
  X.forEach((x, i) => assert.equal(knnPredict(X, y, x, 1).idx[0], i));
});

test('best test k is neither 1 nor 25', () => {
  const rows = L.kTable();
  assert.deepEqual(rows.map((r) => r.k), L.KS);
  assert.deepEqual(L.KS, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]);
  const best = L.bestK();
  assert.notEqual(best, 1);
  assert.notEqual(best, 25);
  const top = rows.find((r) => r.k === best).test;
  // A unique peak, and k = 1 has the lowest test accuracy on the slider (quiz 1).
  assert.equal(rows.filter((r) => r.test === top).length, 1);
  for (const r of rows.slice(1)) assert.ok(r.test > rows[0].test, String(r.k));
  assert.ok(rows[rows.length - 1].test < top);
  // The table agrees with a direct count.
  const tr = L.trainData();
  const te = L.testData();
  const yhat = te.X.map((x) => knnPredict(tr.X, tr.y, x, best).cls);
  assert.equal(accuracy(te.y, yhat), top);
});

test('rescaling feature 2 by 100 changes at least one test prediction, standardising both copies gives identical predictions', () => {
  const k = L.bestK();
  const raw = L.predictTest(k, 'raw');
  const big = L.predictTest(k, 'x100');
  const changed = raw.filter((v, i) => v !== big[i]).length;
  assert.ok(changed >= 1);
  const zRaw = L.predictTest(k, 'std-raw');
  const zBig = L.predictTest(k, 'std');
  assert.deepEqual(zRaw, zBig);
  const u = L.unitsTable(k);
  assert.equal(u.changed, changed);
  assert.equal(u.stdSame, true);
  assert.ok(u.x100 < u.raw);
  // Quiz 2: standardising is not a promise of higher accuracy; here raw units did slightly better.
  assert.ok(u.std < u.raw);
  assert.ok(u.std > u.x100);
  // Every k on the slider: standardising both copies agrees.
  for (const kk of L.KS) assert.deepEqual(L.predictTest(kk, 'std-raw'), L.predictTest(kk, 'std'));
});

test('transforms: ×100 and standardising, with training mean and population SD', () => {
  const t = L.transform('x100');
  assert.deepEqual(t.c, [1, 100]);
  assert.deepEqual(t.apply([2, 3]), [2, 300]);
  const s = L.transform('std');
  const Z = L.trainData().X.map(s.apply);
  for (const j of [0, 1]) {
    const col = Z.map((r) => r[j]);
    const m = col.reduce((a, b) => a + b, 0) / col.length;
    const v = col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length;
    assert.ok(Math.abs(m) < 1e-12);
    assert.ok(Math.abs(v - 1) < 1e-12);
  }
});

test('the starting query: k = 3 vote and the units story', () => {
  const q = L.Q0;
  const n3 = L.neighbours(q, 3, 'raw');
  assert.equal(n3.votes1, 1);
  assert.equal(n3.cls, 0);
  const k = L.bestK();
  const a = L.neighbours(q, k, 'raw');
  const b = L.neighbours(q, k, 'x100');
  const c = L.neighbours(q, k, 'std');
  assert.equal(a.cls, 1);
  assert.equal(b.cls, 0);
  assert.equal(c.cls, 1);
  const shared = a.idx.filter((i) => b.idx.includes(i)).length;
  assert.ok(shared < k / 2);
  // The drawn neighbourhood: every neighbour lies inside the ellipse, the k-th on it.
  for (const n of [a, b, c]) {
    const [rx, ry] = n.radii;
    const tr = L.trainData().X;
    const inside = (p) => ((p[0] - q[0]) / rx) ** 2 + ((p[1] - q[1]) / ry) ** 2;
    for (const i of n.idx) assert.ok(inside(tr[i]) <= 1 + 1e-9);
    assert.ok(Math.abs(inside(tr[n.idx[k - 1]]) - 1) < 1e-9);
  }
});

test('knn read-out rows and page', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { k: 1 }, { k: 25 }, { mode: 'x100' }, { mode: 'std' }, { q: [-0.5, 0] }, { q: [7, 7] }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>k-nearest neighbours · ML Explorer<\/title>/);
  assert.match(html, /Level 3 · Classic ML/);
  assert.match(html, /seed 41/);
  assert.match(html, /seed 42/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/logistic-regression\/">/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /href="\/ml\/train-test\/"/);
  assert.match(html, /href="\/ml\/data-as-numbers\/"/);
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

test('knn page values match the computations', () => {
  const pv = L.pageValues();
  const rows = L.kTable();
  const best = L.bestK();
  const u = L.unitsTable(best);
  assert.equal(pv['best-k'], String(best));
  assert.equal(pv['best-k'], '11');
  assert.equal(pv['k1-test'], L.pct(rows[0].test));
  assert.equal(pv['k1-test'], '78.3%');
  assert.equal(pv['best-test'], '85.7%');
  assert.equal(pv['u-raw'], '85.7%');
  assert.equal(pv['u-x100'], L.pct(u.x100));
  assert.equal(pv['u-std'], L.pct(u.std));
  assert.equal(pv['u-std'], '84.0%');
  // Quiz 2 explanation: the numbers come from spans, and "a few cases" is 5 of 300.
  assert.match(html, /data-for="always">[^<]*<span data-value="u-raw"><\/span> against <span data-value="u-std"><\/span>/);
  assert.equal(Math.round((u.raw - u.std) * L.testData().X.length), 5);
  assert.equal(pv['u-changed'], String(u.changed));
  assert.equal(pv['q3-votes0'], '2');
  assert.equal(pv['q3-votes1'], '1');
  assert.equal(L.pct(1), '100%');
  assert.equal(L.pct(0.8566666), '85.7%');
});

test('knn wording claims hold', () => {
  const rows = L.kTable();
  const best = L.bestK();
  const top = rows.find((r) => r.k === best);
  // "several other values of k score within one percentage point of it"
  assert.ok(rows.filter((r) => r.k !== best && top.test - r.test <= 0.01).length >= 2);
  // "Its training accuracy ... is not the highest"
  assert.ok(top.train < Math.max(...rows.map((r) => r.train)));
  // "a little below": standardised versus original units, under two points.
  const u = L.unitsTable(best);
  assert.ok(u.raw - u.std > 0 && u.raw - u.std < 0.02);
  // "the two features already had spreads of a similar size" (population SDs).
  const X = L.trainData().X;
  const sdOf = (j) => {
    const c = X.map((r) => r[j]);
    const m = c.reduce((a, b) => a + b, 0) / c.length;
    return Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length);
  };
  const ratio = sdOf(0) / sdOf(1);
  assert.ok(ratio > 1 / 1.5 && ratio < 1.5, String(ratio));
  // The page's fixed facts.
  assert.match(html, new RegExp(`${L.GRID[0]} × ${L.GRID[1]}`));
  assert.match(html, /\(3\.5, 2\.5\)/);
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /it is guaranteed/);
  assert.deepEqual(L.Q0, [3.5, 2.5]);
});
