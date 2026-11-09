import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { growTree, predictTree, forest, accuracy } from '../assets/classify.js';
import { treeData } from '../assets/tree-data.js';
import { linearScale, regionCells } from '../assets/plot.js';
import * as L from '../random-forests/stage.js';

const html = readFileSync(new URL('../random-forests/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);
const row = (rows, B) => rows.find((r) => r.B === B);

test('forest data and forests are the stated ones', () => {
  assert.deepEqual(L.trainData(), treeData({ n: 80, seed: 99 }));
  assert.deepEqual(L.testData(), treeData({ n: 400, seed: 100 }));
  assert.deepEqual(L.BS, [1, 3, 5, 11, 25, 51, 101]);
  const { X, y } = L.trainData();
  const bag = forest(X, y, { B: 101, seed: L.SEED_FOREST });
  const rf = forest(X, y, { B: 101, mtry: 1, seed: L.SEED_FOREST });
  assert.deepEqual(L.forestOf('bag').samples, bag.samples);
  assert.deepEqual(L.forestOf('rf').samples, rf.samples);
  // Same first draw, so tree 1 of each sees the same rows.
  assert.deepEqual(bag.samples[0], rf.samples[0]);
});

test('test accuracy tables match a direct count', () => {
  const { X, y } = L.testData();
  for (const kind of ['bag', 'rf']) {
    const f = L.forestOf(kind);
    for (const r of L.testTable(kind)) {
      close(r.test, accuracy(y, X.map((x) => f.predict(x, r.B))));
    }
  }
  const full = growTree(L.trainData().X, L.trainData().y);
  close(L.fullTreeTest(), accuracy(y, X.map((x) => predictTree(full, x))));
});

test('forest test accuracy at B=101 ≥ single full tree', () => {
  const full = L.fullTreeTest();
  assert.ok(row(L.testTable('rf'), 101).test >= full);
  assert.ok(row(L.testTable('bag'), 101).test >= full);
});

test('OOB accuracy within 0.1 of test accuracy at B=101', () => {
  for (const kind of ['bag', 'rf']) {
    const o = row(L.oobTable(kind), 101);
    const t = row(L.testTable(kind), 101);
    assert.ok(Math.abs(o.oob - t.test) <= 0.1);
    assert.equal(o.covered, 80);
  }
});

test('OOB table: votes only from trees that left the row out', () => {
  const f = L.forestOf('rf');
  const { X, y } = L.trainData();
  for (const r of L.oobTable('rf')) {
    let covered = 0;
    let right = 0;
    X.forEach((x, i) => {
      const ts = f.trees.slice(0, r.B).filter((_, t) => !f.samples[t].includes(i));
      if (!ts.length) return;
      covered++;
      const p = ts.filter((t) => predictTree(t, x) === 1).length / ts.length > 0.5 ? 1 : 0;
      if (p === y[i]) right++;
    });
    assert.equal(r.covered, covered);
    close(r.oob, right / covered);
    assert.equal(r.excluded, 80 - covered);
  }
});

test('distinct fraction shown equals the computed formula', () => {
  const s = L.bootstrapStats();
  const n = 80;
  close(s.expected, 1 - (1 - 1 / n) ** n);
  close(s.pOut, (1 - 1 / n) ** n);
  close(s.limit, 1 - Math.exp(-1));
  const idx = L.forestOf('bag').samples[0];
  assert.equal(s.distinct, new Set(idx).size);
  close(s.share, s.distinct / n);
  const all = L.forestOf('bag').samples;
  close(s.meanShare, all.reduce((a, smp) => a + new Set(smp).size, 0) / all.length / n);
  // Times drawn: 0, 1, 2, 3, 4 or more, summing to n rows and n draws.
  assert.equal(s.drawCounts.reduce((a, b) => a + b, 0), n);
  const times = new Array(n).fill(0);
  for (const i of idx) times[i]++;
  for (let k = 0; k < 5; k++) assert.equal(s.drawCounts[k], times.filter((t) => (k < 4 ? t === k : t >= 4)).length);
  assert.equal(s.drawCounts[0], n - s.distinct);
  const pv = L.pageValues();
  assert.equal(pv.expected, s.expected.toFixed(3));
  assert.equal(pv['p-out'], s.pOut.toFixed(3));
  assert.equal(pv['p-out-pct'], L.pct(s.pOut));
  assert.equal(pv['s-distinct'], String(s.distinct));
  assert.equal(pv['s-out'], String(n - s.distinct));
  assert.equal(pv['s-share'], L.pct(s.share));
  assert.equal(pv['mean-share'], L.pct(s.meanShare));
  assert.equal(pv.expected, '0.634');
  assert.equal(pv['p-out'], '0.366');
});

test('region maps cached per B', () => {
  for (const kind of ['bag', 'rf']) {
    for (const B of L.BS) {
      const a = L.voteGrid(kind, B);
      assert.equal(L.voteGrid(kind, B), a);
      assert.equal(a.length, L.GRID_N * L.GRID_N);
    }
  }
  assert.notEqual(L.voteGrid('rf', 25), L.voteGrid('rf', 51));
  for (let k = 0; k < 3; k++) assert.equal(L.panelGrid(k), L.panelGrid(k));
});

test('vote grids equal the forest prediction at each cell centre', () => {
  const X = linearScale([0, 10], [48, 258]);
  const Y = linearScale([0, 10], [268, 58]);
  const cells = regionCells(L.GRID_N, L.GRID_N, X, Y);
  const c = L.gridCentres(L.GRID_N);
  cells.forEach((cell, k) => { close(c[k][0], cell.x); close(c[k][1], cell.y); });
  for (const kind of ['bag', 'rf']) {
    const f = L.forestOf(kind);
    for (const B of [1, 11, 101]) {
      const g = L.voteGrid(kind, B);
      c.forEach((p, k) => assert.equal(g[k], f.predict(p, B)));
      // The lookup the stage uses finds the same cell.
      cells.forEach((cell, k) => assert.equal(L.cellIndex(cell.x, cell.y, L.GRID_N), k));
    }
  }
  const pc = L.gridCentres(L.PANEL_N);
  for (let t = 0; t < 3; t++) {
    const g = L.panelGrid(t);
    pc.forEach((p, k) => assert.equal(g[k], predictTree(L.forestOf('bag').trees[t], p)));
  }
});

test('diversity numbers match a direct count', () => {
  const { X, y } = L.testData();
  for (const kind of ['bag', 'rf']) {
    const P = L.forestOf(kind).trees.map((t) => X.map((x) => predictTree(t, x)));
    const accs = P.map((p) => accuracy(y, p));
    close(L.diversity(kind).meanTree, accs.reduce((a, b) => a + b, 0) / accs.length);
    let d = 0;
    let pairs = 0;
    for (let a = 0; a < P.length; a++) {
      for (let b = a + 1; b < P.length; b++) {
        d += P[a].filter((v, i) => v !== P[b][i]).length / X.length;
        pairs++;
      }
    }
    close(L.diversity(kind).disagree, d / pairs, 1e-12);
  }
});

test('the computed story the page tells', () => {
  const bag = L.testTable('bag');
  const rf = L.testTable('rf');
  const dB = L.diversity('bag');
  const dR = L.diversity('rf');
  // Step 4: at B = 101 the forest's trees are weaker one by one, disagree more, and vote better.
  assert.ok(dR.meanTree < dB.meanTree);
  assert.ok(dR.disagree > dB.disagree);
  assert.ok(row(rf, 101).test > row(bag, 101).test);
  // Step 3: one bootstrap tree is below the full tree; the vote of 101 is above it.
  assert.ok(row(bag, 1).test < L.fullTreeTest());
  // Step 1: each of the three bootstrap trees scores below the full tree on the test cases.
  for (const a of L.panelTests()) assert.ok(a < L.fullTreeTest());
  // Step 5: the first B in the list where every row has an out-of-bag vote.
  const o = L.oobTable('rf');
  assert.equal(L.firstFullB('rf'), o.find((r) => r.covered === 80).B);
  assert.ok(o[0].covered < 80);
  // The chart's range holds every value it plots.
  for (const v of [...bag, ...rf].map((r) => r.test).concat(o.map((r) => r.oob), [L.fullTreeTest()])) {
    assert.ok(v >= L.ACC_DOM[0] && v <= L.ACC_DOM[1], String(v));
  }
});

test('page values match the computations', () => {
  const pv = L.pageValues();
  const bag = L.testTable('bag');
  const rf = L.testTable('rf');
  const o = L.oobTable('rf');
  assert.equal(pv['full-test'], L.pct(L.fullTreeTest()));
  assert.equal(pv['bag-1'], L.pct(row(bag, 1).test));
  assert.equal(pv['bag-101'], L.pct(row(bag, 101).test));
  assert.equal(pv['rf-101'], L.pct(row(rf, 101).test));
  assert.equal(pv['oob-101'], L.pct(row(o, 101).oob));
  assert.equal(pv['oob-1'], L.pct(o[0].oob));
  assert.equal(pv['oob-1-rows'], String(o[0].covered));
  assert.equal(pv['oob-1-excl'], String(o[0].excluded));
  assert.equal(pv['oob-full-b'], String(L.firstFullB('rf')));
  assert.equal(pv['bag-tree'], L.pct(L.diversity('bag').meanTree));
  assert.equal(pv['rf-tree'], L.pct(L.diversity('rf').meanTree));
  assert.equal(pv['bag-dis'], L.pct(L.diversity('bag').disagree));
  assert.equal(pv['rf-dis'], L.pct(L.diversity('rf').disagree));
  const t = L.panelTests();
  assert.equal(pv['t1-test'], L.pct(t[0]));
  assert.equal(pv['t3-test'], L.pct(t[2]));
  // Pinned so a change to the data or the forest code is noticed.
  assert.equal(pv['full-test'], L.pct(0.7825));
  assert.equal(pv['rf-101'], '83.8%');
  assert.equal(pv['bag-101'], '81.8%');
  assert.equal(pv['oob-101'], '82.5%');
  assert.equal(pv['s-distinct'], '46');
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('random forests read-out rows and page', () => {
  for (let step = 0; step < 5; step++) {
    for (let bi = 0; bi < L.BS.length; bi++) {
      for (const r of L.readout(step, { bi })) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 5);
  assert.match(html, /<title>Bagging and random forests · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /seed 99/);
  assert.match(html, /seed 100/);
  assert.match(html, new RegExp(`seed ${L.SEED_FOREST}\\b`));
  assert.match(html, /<link rel="stylesheet" href="\/ml\/assets\/ml\.css">\s*<link rel="stylesheet" href="\/ml\/random-forests\/lesson\.css">/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/gradient-boosting\/">/);
  assert.match(html, /href="\/ml\/decision-trees\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 15);
  for (const key of keys) assert.ok(key in pv, key);
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /delve|unlock|harness the power|it's important to note|dive in|in today's world/i);
  // Fixed facts stated in words.
  assert.match(html, /B = 1, 3, 5, 11, 25, 51 or 101/);
  assert.match(html, /The 80 training cases use seed 99 and the 400 test cases seed 100/);
});

test('wording claims hold', () => {
  const pv = L.pageValues();
  // Step 3: the map settles: fewer cells change from 51 to 101 trees than from 1 to 3.
  const ch = L.mapChanges('bag');
  assert.ok(ch.at(-1) < ch[0]);
  const c = (a, b) => L.voteGrid('bag', a).filter((v, j) => v !== L.voteGrid('bag', b)[j]).length;
  assert.equal(pv['bag-chg-first'], String(c(1, 3)));
  assert.equal(pv['bag-chg-last'], String(c(51, 101)));
  assert.equal(pv.cells, '1,764');
  // Step 4: first questions.
  const count = (kind, f) => L.forestOf(kind).trees.filter((t) => !t.leaf && t.feature === f).length;
  assert.equal(pv['bag-root-x2'], String(count('bag', 1)));
  assert.equal(pv['rf-root-x1'], String(count('rf', 0)));
  assert.equal(pv['rf-root-x2'], String(count('rf', 1)));
  assert.ok(count('bag', 1) > count('rf', 1));
  assert.ok(L.forestOf('bag').trees.every((t) => !t.leaf));
  // Numbers typed in the text are the lesson's constants.
  assert.match(html, /Of the 101 bagged trees/);
  assert.match(html, /Across all 101 samples/);
  assert.equal(L.forestOf('bag').trees.length, 101);
  assert.match(html, /1 − \(1 − 1\/80\)⁸⁰/);
  assert.equal(L.N_TRAIN, 80);
  assert.match(html, /42 × 42 cells \(32 × 32 in the small maps\)/);
  assert.deepEqual([L.GRID_N, L.PANEL_N], [42, 32]);
  assert.match(html, /seed 10:/);
  assert.equal(L.SEED_FOREST, 10);
  // Step 1: the panels show trees 1 to 3; step 3: B = 1 is tree 1.
  assert.equal(L.testTable('bag')[0].test, L.panelTests()[0]);
  // Step 5: every row has an OOB vote at B = 101.
  assert.equal(L.oobTable('rf').at(-1).excluded, 0);
});
