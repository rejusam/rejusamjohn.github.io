import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { kmeans, inertia } from '../assets/unsupervised.js';
import { classMarker } from '../assets/plot.js';
import * as L from '../k-means/stage.js';

const html = readFileSync(new URL('../k-means/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);

test('k-means data reproducible', () => {
  const X = L.data();
  assert.deepEqual(X, L.data());
  assert.equal(X.length, 60);
  // Rebuild from the stated recipe: blob by blob, x then y for each point.
  const rng = mulberry32(101);
  const want = [];
  for (const [cx, cy] of [[2, 2], [6, 2.5], [4, 6]]) {
    for (let i = 0; i < 20; i++) {
      const x = cx + 0.8 * randn(rng);
      const y = cy + 0.8 * randn(rng);
      want.push([x, y]);
    }
  }
  assert.deepEqual(X, want);
  close(X[0][0], 2.148505281383813);
  close(X[0][1], 1.447480475736374);
  assert.equal(L.SEED, 101);
  // Every point fits inside the plot.
  for (const [x, y] of X) {
    assert.ok(x > L.X_DOM[0] && x < L.X_DOM[1]);
    assert.ok(y > L.Y_DOM[0] && y < L.Y_DOM[1]);
  }
});

test('good and bad starts both converge and the bad one is ≥20% worse', () => {
  const X = L.data();
  const n = X.length;
  // Search every triple of data points in lexicographic order.
  let best = Infinity;
  let firstBad = null;
  const finals = [];
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      for (let c = b + 1; c < n; c++) {
        const r = kmeans(X, [X[a], X[b], X[c]]);
        assert.ok(r.converged);
        finals.push([[a, b, c], r.inertia]);
        if (r.inertia < best) best = r.inertia;
      }
    }
  }
  firstBad = finals.find(([, v]) => v > 1.2 * best)[0];
  assert.deepEqual(L.START_B, firstBad);
  assert.deepEqual(L.START_A, [0, 1, 2]);
  const A = L.run('A');
  const B = L.run('B');
  assert.ok(A.converged && B.converged);
  close(A.inertia, best, 1e-9);
  assert.ok(B.inertia > 1.2 * A.inertia);
  // Inertia never rises, state by state, in both runs.
  for (const r of [A, B]) {
    const v = r.states.slice(1).map((s) => s.inertia);
    for (let i = 1; i < v.length; i++) assert.ok(v[i] <= v[i - 1] + 1e-12, `${i}`);
    for (const s of r.states.slice(1)) close(s.inertia, inertia(X, s.centres, s.labels), 1e-9);
    // No cluster is ever empty, so the empty-cluster rule only appears in The maths.
    for (const s of r.states.slice(1)) for (let k = 0; k < 3; k++) assert.ok(s.labels.includes(k));
  }
  // Both starts sit in the lower-left group (points 0 to 19) and differ in one point.
  for (const i of [...L.START_A, ...L.START_B]) assert.ok(i < 20);
  assert.equal(L.START_A.filter((i) => !L.START_B.includes(i)).length, 1);
  // Start A finds the three groups the points were made from.
  const blob = (i) => Math.floor(i / 20);
  const partition = (labels) => [0, 1, 2].map((k) => labels.map((l, i) => (l === k ? i : -1)).filter((i) => i >= 0).join(','));
  const groups = [0, 1, 2].map((g) => X.map((_, i) => (blob(i) === g ? i : -1)).filter((i) => i >= 0).join(','));
  assert.deepEqual(partition(A.labels).sort(), groups.slice().sort());
  // Start B: two centres share the lower-left group, one covers the other two groups.
  const bl = B.labels;
  const lowLeft = new Set(bl.slice(0, 20));
  assert.equal(lowLeft.size, 2);
  const rest = new Set(bl.slice(20));
  assert.equal(rest.size, 1);
  assert.ok(!lowLeft.has([...rest][0]));
  // The page's step numbers.
  assert.equal(A.iters, 5);
  assert.equal(A.states.length, 10);
  assert.equal(B.iters, 3);
});

test('inertia vs K non-increasing using best-of-starts', () => {
  const rows = L.elbow();
  assert.deepEqual(rows.map((r) => r.K), [1, 2, 3, 4, 5, 6, 7, 8]);
  const X = L.data();
  for (const r of rows) {
    assert.equal(r.inertias.length, 10);
    assert.equal(r.best, Math.min(...r.inertias));
    // Each start is K distinct data points from mulberry32(200 + K).
    const rng = mulberry32(200 + r.K);
    for (let s = 0; s < 10; s++) {
      const idx = [];
      while (idx.length < r.K) {
        const i = Math.floor(rng() * 60);
        if (!idx.includes(i)) idx.push(i);
      }
      assert.deepEqual(L.randomStarts(r.K)[s], idx);
      close(r.inertias[s], kmeans(X, idx.map((i) => X[i])).inertia, 1e-12);
    }
  }
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].best < rows[i - 1].best, String(rows[i].K));
  // K = 1: the sum of squared distances to the overall mean.
  const m = [0, 1].map((j) => X.reduce((s, p) => s + p[j], 0) / X.length);
  close(rows[0].best, X.reduce((s, p) => s + (p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2, 0), 1e-9);
  // K = 3: best of 10 equals start A; 9 of the 10 starts reach it, one stops higher.
  close(rows[2].best, L.run('A').inertia, 1e-9);
  const hit = rows[2].inertias.filter((v) => Math.abs(v - rows[2].best) < 1e-9).length;
  assert.equal(hit, 9);
  // Quiz 2: no cluster is empty in the best K = 8 run.
  const r8 = rows[7].bestRun;
  for (let k = 0; k < 8; k++) assert.ok(r8.labels.includes(k), String(k));
  // The bend: the drop from 2 to 3 is much larger than from 3 to 4.
  assert.ok(rows[1].best - rows[2].best > 5 * (rows[2].best - rows[3].best));
});

test('shapes for three clusters distinct (class names differ)', () => {
  const made = [];
  const node = (tag) => ({
    tag, attrs: {}, children: [],
    setAttribute(k, v) { this.attrs[k] = v; },
    appendChild(c) { this.children.push(c); return c; },
  });
  const had = globalThis.document;
  globalThis.document = { createElementNS: (_, tag) => { const n = node(tag); made.push(n); return n; } };
  try {
    const parent = node('g');
    const marks = [0, 1, 2].map((k) => classMarker(parent, 10, 10, L.clusterClass(k)));
    const cls = marks.map((m) => m.attrs.class);
    assert.equal(new Set(cls).size, 3);
    assert.equal(new Set(marks.map((m) => m.tag)).size, 3);
    assert.deepEqual(cls, ['mk mk-0', 'mk mk-1', 'mk mk-2']);
    const u = classMarker(parent, 10, 10, null);
    assert.equal(u.attrs.class, 'mk mk-u');
  } finally {
    if (had === undefined) delete globalThis.document;
    else globalThis.document = had;
  }
  // Start A's final clusters use all three shapes.
  assert.deepEqual([...new Set(L.run('A').labels)].sort(), [0, 1, 2]);
});

test('k-means page values match the computations and the page contract', () => {
  const pv = L.pageValues();
  const A = L.run('A');
  const B = L.run('B');
  const rows = L.elbow();
  assert.equal(pv['a-first'], A.states[1].inertia.toFixed(1));
  assert.equal(pv['a-first'], '461.9');
  assert.equal(pv['a-final'], '79.5');
  assert.equal(pv['b-final'], '223.2');
  assert.equal(pv['b-final'], B.inertia.toFixed(1));
  assert.equal(pv['ab-ratio'], (B.inertia / A.inertia).toFixed(1));
  assert.equal(pv['ab-ratio'], '2.8');
  assert.equal(pv['a-rounds'], '5');
  assert.equal(pv['a-sizes'], '8, 18 and 34');
  assert.equal(pv['starts-hit'], '9');
  assert.equal(pv['starts-other'], '240.1');
  assert.equal(pv['e1'], rows[0].best.toFixed(1));
  assert.equal(pv['e1'], '407.0');
  assert.equal(pv['e2'], '231.4');
  assert.equal(pv['e3'], '79.5');
  assert.equal(pv['e4'], '65.7');
  assert.equal(pv['e8'], '33.8');
  assert.equal(pv['drop23'], (rows[1].best - rows[2].best).toFixed(1));
  assert.equal(pv['drop34'], (rows[2].best - rows[3].best).toFixed(1));
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 10);
  for (const key of keys) assert.ok(key in pv, key);
  // Quiz 2's explanation uses the computed values.
  assert.match(html, /data-for="no">[^]*?<span data-value="e8"><\/span>[^]*?<span data-value="e3"><\/span>/);
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { state: 1 }, { state: 9 }, { start: 'B' }, { K: 1 }, { K: 8 }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>k-means clustering · ML Explorer<\/title>/);
  assert.match(html, /Level 3 · Classic ML/);
  assert.match(html, /seed 101/);
  assert.match(html, /mulberry32\(200 \+ K\)/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/pca\/">/);
  assert.match(html, /href="\/ml\/k-nearest-neighbours\/"/);
  assert.match(html, /<link rel="stylesheet" href="\/ml\/k-means\/lesson.css">/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.match(html, new RegExp(`${L.GRID[0]} × ${L.GRID[1]}`));
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.match(html, /Section 12\.4\.1, "K-Means Clustering"/);
  assert.doesNotMatch(html, /Taylor/);
  // Exact wording rulings.
  assert.match(html, /k-means\+\+/);
  assert.match(html, /fixed point/);
  assert.match(html, /not an average/);
  assert.match(html, /a hint, not a rule/);
  // Only the best (lowest found) inertia is claimed to fall as K grows.
  assert.match(html, /<p class="lede">With more clusters the best inertia you can reach is always lower, so inertia alone cannot choose K\./);
  assert.match(html, /the best inertia keeps falling as K grows, and the elbow is only a hint/);
  assert.match(html, /value="no"> No: the best inertia keeps falling as K grows, whatever the data, so it cannot choose K on its own/);
  assert.doesNotMatch(html, /[Ii]nertia always falls|always means a lower inertia|inertia falls as K grows/);
  // Single runs can rise with K: a K = 3 start stops above the K = 2 best.
  assert.ok(Math.max(...rows[2].inertias) > rows[1].best);
  assert.match(html, /keeps its centre/);
  // Outside the quoted misconceptions, never claim the best answer is guaranteed.
  const prose = html.replace(/<ul class="misconceptions">[^]*?<\/ul>/, '');
  assert.doesNotMatch(prose, /global minimum is guaranteed|always finds the best/i);
  assert.match(html, /"k-means always finds the best grouping\." It finds a fixed point/);
});
