import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { gini, bestSplit, growTree, predictTree, treeStats, accuracy } from '../assets/classify.js';
import { treeData, treeRule } from '../assets/tree-data.js';
import * as L from '../decision-trees/stage.js';

const html = readFileSync(new URL('../decision-trees/index.html', import.meta.url), 'utf8');
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);

test('tree data reproducible, ~10% labels flipped', () => {
  const a = treeData({ n: 80, seed: 99 });
  assert.deepEqual(a, treeData({ n: 80, seed: 99 }));
  assert.deepEqual(L.trainData(), a);
  assert.deepEqual(L.testData(), treeData({ n: 400, seed: 100 }));
  // The stated draw order: x₁, x₂, then the flip draw, for each point in turn.
  const rng = mulberry32(99);
  for (let i = 0; i < 80; i++) {
    const x = [10 * rng(), 10 * rng()];
    const flip = rng() < 0.1;
    assert.deepEqual(a.X[i], x);
    assert.equal(a.y[i], flip ? 1 - treeRule(x) : treeRule(x));
    assert.equal(a.flipped.includes(i), flip);
  }
  // The reported count equals the actual number of labels that disagree with the rule.
  for (const d of [a, L.testData()]) {
    const actual = d.X.map((x, i) => i).filter((i) => d.y[i] !== treeRule(d.X[i]));
    assert.deepEqual(d.flipped, actual);
    for (const x of d.X) for (const v of x) assert.ok(v >= 0 && v < 10);
  }
  assert.equal(a.flipped.length, 8);
  assert.equal(L.testData().flipped.length, 40);
  // Roughly 10% over a large sample.
  const big = treeData({ n: 20000, seed: 5 });
  close(big.flipped.length / 20000, 0.1, 0.01);
  // flip = 0 gives the rule exactly.
  const clean = treeData({ n: 50, seed: 3, flip: 0 });
  assert.deepEqual(clean.flipped, []);
  clean.X.forEach((x, i) => assert.equal(clean.y[i], treeRule(x)));
  // The rule itself.
  assert.equal(treeRule([6, 4]), 1);
  assert.equal(treeRule([6, 2]), 0);
  assert.equal(treeRule([2, 8]), 1);
  assert.equal(treeRule([4, 8]), 0);
  assert.equal(treeRule([2, 6]), 0);
  // Class counts on the page.
  assert.deepEqual(L.classCounts(a.y), [43, 37]);
  // Pin the first row.
  close(a.X[0][0], 2.604658124037087);
  close(a.X[0][1], 8.048227655235678);
  close(a.X[79][0], 6.536268191412091);
  close(a.X[79][1], 7.455419816542417);
});

test('weighted Gini of a split and the curve match a direct count', () => {
  const { X, y } = L.trainData();
  for (const f of [0, 1]) {
    for (const t of [-1, 0, 0.5, 2.5, 5, 7.25, 9.99, 10, 11]) {
      const s = L.splitScore(f, t);
      const left = [0, 0], right = [0, 0];
      X.forEach((x, i) => (x[f] < t ? left : right)[y[i]]++);
      assert.deepEqual(s.left, left);
      assert.deepEqual(s.right, right);
      const nl = left[0] + left[1], nr = right[0] + right[1];
      close(s.weighted, (nl * gini(left) + nr * gini(right)) / 80);
      close(s.gLeft, gini(left));
    }
    // The step curve covers [0, 10] and agrees with splitScore inside each step.
    const steps = L.giniSteps(f);
    assert.equal(steps[0].t0, 0);
    assert.equal(steps[steps.length - 1].t1, 10);
    for (let k = 1; k < steps.length; k++) assert.equal(steps[k].t0, steps[k - 1].t1);
    for (const st of steps) close(st.g, L.splitScore(f, (st.t0 + st.t1) / 2).weighted);
    // The ends are the parent's impurity: one side is empty.
    close(steps[0].g, gini([43, 37]));
    close(steps[steps.length - 1].g, gini([43, 37]));
  }
  close(L.parentGini(), 2 * (37 / 80) * (43 / 80));
  // Two classes: 1 − p₀² − p₁² = 2p(1 − p); pure is 0, half and half is 0.5.
  for (const c of [[3, 5], [10, 0], [4, 4], [1, 9]]) {
    const p = c[1] / (c[0] + c[1]);
    close(gini(c), 2 * p * (1 - p));
  }
  assert.equal(gini([7, 0]), 0);
  close(gini([4, 4]), 0.5);
});

test('best root split matches bestSplit', () => {
  const { X, y } = L.trainData();
  const idx = X.map((_, i) => i);
  const rc = L.rootChoice();
  const all = bestSplit(X, y, idx);
  assert.equal(rc.best.feature, all.feature);
  assert.equal(rc.best.threshold, all.threshold);
  close(rc.best.impurity, all.impurity);
  // Brute force over every midpoint of both features agrees.
  let bf = null;
  for (const f of [0, 1]) {
    const v = [...new Set(X.map((x) => x[f]))].sort((a, b) => a - b);
    for (let k = 0; k + 1 < v.length; k++) {
      const t = (v[k] + v[k + 1]) / 2;
      const g = L.splitScore(f, t).weighted;
      if (!bf || g < bf.g - 1e-12) bf = { f, t, g };
    }
  }
  assert.equal(bf.f, all.feature);
  assert.equal(bf.t, all.threshold);
  close(bf.g, all.impurity);
  // The story: x₂ wins, x₁'s best is close behind.
  assert.equal(rc.best.feature, 1);
  assert.equal(rc.byFeature[1].threshold, rc.best.threshold);
  assert.ok(rc.byFeature[0].impurity > rc.byFeature[1].impurity);
  assert.equal(L.fmtT(rc.byFeature[0].threshold), '4.96');
  assert.equal(L.fmtT(rc.byFeature[1].threshold), '3.04');
  assert.equal(L.fmtG(rc.byFeature[0].impurity), '0.414');
  assert.equal(L.fmtG(rc.byFeature[1].impurity), '0.400');
  assert.equal(L.fmtG(L.parentGini()), '0.497');
  // The root of the grown tree is this split.
  const t = L.treeAt(1);
  assert.equal(t.feature, 1);
  assert.equal(t.threshold, rc.best.threshold);
  // The lowest stretch on x₁ is a single interval, it holds the step-1 guess 5 and the midpoint.
  const m = L.minInterval(0);
  close(m.g, rc.byFeature[0].impurity);
  assert.ok(m.a < L.T_GUESS && L.T_GUESS <= m.b);
  close(m.mid, rc.byFeature[0].threshold);
  assert.equal(L.giniSteps(0).filter((s) => Math.abs(s.g - m.g) < 1e-12).length, 1);
  // The step-1 question x₁ < 5 makes the same split as x₁ < 4.96.
  close(L.splitScore(0, L.T_GUESS).weighted, rc.byFeature[0].impurity);
});

test('full tree train accuracy 1, test accuracy below the best depth\'s', () => {
  const full = L.fullTree();
  const { X, y } = L.trainData();
  const te = L.testData();
  assert.equal(accuracy(y, X.map((x) => predictTree(full, x))), 1);
  const fullTest = accuracy(te.y, te.X.map((x) => predictTree(full, x)));
  const best = L.depthTable().find((r) => r.d === L.bestDepth());
  assert.ok(fullTest < best.test);
  // The full tree is depth 6, so the slider's 6, 7 and 8 are the same tree; every leaf is pure.
  const st = treeStats(full);
  assert.equal(st.depth, 6);
  assert.equal(st.leaves, 18);
  const rows = L.depthTable();
  for (const r of rows.filter((r) => r.d >= 6)) {
    assert.equal(r.train, 1);
    assert.equal(r.test, fullTest);
    assert.deepEqual(L.treeAt(r.d), full);
  }
  for (const r of rows.filter((r) => r.d < 6)) assert.ok(r.train < 1);
  const pure = (n) => (n.leaf ? Math.min(...n.counts) === 0 : pure(n.left) && pure(n.right));
  assert.ok(pure(full));
  // The rule that made the data scores 90% on both sets (8 of 80, 40 of 400 flipped).
  assert.equal(L.ruleAccuracy().train, 0.9);
  assert.equal(L.ruleAccuracy().test, 0.9);
  assert.ok(fullTest < L.ruleAccuracy().test);
});

test('best test depth is between 2 and 6', () => {
  const rows = L.depthTable();
  assert.deepEqual(rows.map((r) => r.d), [1, 2, 3, 4, 5, 6, 7, 8]);
  const b = L.bestDepth();
  assert.ok(b >= 2 && b <= 6);
  assert.equal(b, 2);
  const top = rows.find((r) => r.d === b).test;
  assert.equal(rows.filter((r) => r.test === top).length, 1);
  // After the peak, test accuracy never rises again on the slider.
  for (let d = b; d < 8; d++) assert.ok(rows[d].test <= rows[d - 1].test, String(d));
  // Training accuracy never falls as the tree deepens.
  for (let d = 1; d < 8; d++) assert.ok(rows[d].train >= rows[d - 1].train);
  // Each row agrees with a tree grown directly.
  const { X, y } = L.trainData();
  const te = L.testData();
  for (const r of rows) {
    const t = growTree(X, y, { maxDepth: r.d });
    close(r.train, accuracy(y, X.map((x) => predictTree(t, x))));
    close(r.test, accuracy(te.y, te.X.map((x) => predictTree(t, x))));
    assert.equal(r.leaves, treeStats(t).leaves);
  }
});

test('leaf boxes partition the square and predict like the tree', () => {
  for (const d of [1, 2, 3, 4, 6]) {
    const t = L.treeAt(d);
    const boxes = L.leafBoxes(t);
    assert.equal(boxes.length, treeStats(t).leaves);
    close(boxes.reduce((s, b) => s + (b.x1 - b.x0) * (b.y1 - b.y0), 0), 100, 1e-9);
    const rng = mulberry32(7);
    for (let k = 0; k < 400; k++) {
      const p = [10 * rng(), 10 * rng()];
      const hit = boxes.filter((b) => p[0] >= b.x0 && p[0] < b.x1 && p[1] >= b.y0 && p[1] < b.y1);
      assert.equal(hit.length, 1);
      assert.equal(hit[0].cls, predictTree(t, p));
      assert.equal(L.leafOf(t, p), hit[0].node);
    }
    // Class edges lie between boxes of different class, on a shared side.
    for (const [x0, y0, x1, y1] of L.classEdges(boxes)) {
      assert.ok(x0 === x1 || y0 === y1);
      const mid = [(x0 + x1) / 2, (y0 + y1) / 2];
      const e = 1e-6;
      const a = x0 === x1 ? [mid[0] - e, mid[1]] : [mid[0], mid[1] - e];
      const b = x0 === x1 ? [mid[0] + e, mid[1]] : [mid[0], mid[1] + e];
      assert.notEqual(predictTree(t, a), predictTree(t, b));
    }
  }
  // Depth 1: one straight boundary across the whole square at x₂ = 3.04.
  const e1 = L.classEdges(L.leafBoxes(L.treeAt(1)));
  assert.equal(e1.length, 1);
  assert.deepEqual([e1[0][0], e1[0][2]], [0, 10]);
});

test('the drawn diagram never overlaps and its text fits', () => {
  const area = L.TREE_AREA;
  const trees = [L.stump(0, L.T_GUESS), ...L.DEPTHS.map((d) => L.treeAt(d))];
  for (const t of trees) {
    const lay = L.treeBoxes(t, area);
    // At most three levels drawn; deeper subtrees become stubs.
    assert.ok(lay.nodes.every((b) => b.level <= L.DISPLAY_DEPTH));
    const hidden = treeStats(t).depth > L.DISPLAY_DEPTH;
    assert.equal(lay.nodes.some((b) => b.kind === 'stub'), hidden);
    for (const b of lay.nodes) {
      assert.ok(b.w > 0 && b.h > 0);
      assert.ok(b.x >= area.x - 1e-9 && b.x + b.w <= area.x + area.w + 1e-9);
      assert.ok(b.y >= area.y - 1e-9 && b.y + b.h <= area.y + area.h + 1e-9);
      for (const line of b.lines) assert.ok(line.length * L.CHAR_W + 4 <= b.w, `${line} in ${b.w}`);
    }
    for (const a of lay.nodes) {
      for (const b of lay.nodes) {
        if (a === b) continue;
        const apart = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
        assert.ok(apart, `${a.lines} / ${b.lines}`);
      }
    }
    // Every stub counts the splits it hides.
    for (const b of lay.nodes.filter((n) => n.kind === 'stub')) {
      const splits = (n) => (n.leaf ? 0 : 1 + splits(n.left) + splits(n.right));
      assert.equal(b.hidden, splits(b.node));
      assert.ok(b.hidden > 0);
    }
  }
});

test('a point\'s path and the greedy example', () => {
  const t = L.treeAt(L.bestDepth());
  const p = L.pathOf(t, L.Q0);
  assert.equal(p.steps.length, 2);
  assert.deepEqual(p.steps.map((s) => s.yes), [false, true]);
  assert.equal(p.leaf, L.leafOf(t, L.Q0));
  assert.equal(p.leaf.cls, predictTree(t, L.Q0));
  // The start point sits in the rule's small corner, which depth 2 misses and depth 3 catches.
  assert.equal(treeRule(L.Q0), 1);
  assert.equal(predictTree(t, L.Q0), 0);
  assert.equal(predictTree(L.treeAt(3), L.Q0), 1);
  const rows = L.depthTable();
  assert.ok(rows[2].test < rows[1].test);
  // Depth 2 has a leaf tie (1 and 1), which goes to class 0.
  const leaves = L.leafBoxes(t).map((b) => b.node);
  const tie = leaves.filter((n) => n.counts[0] === n.counts[1]);
  assert.equal(tie.length, 1);
  assert.equal(tie[0].cls, 0);
  // Greedy is not globally optimal: the best depth-2 tree starts with the root greedy turned down.
  const g = L.depth2Search();
  assert.ok(g.best.impurity < g.greedy - 1e-9);
  assert.equal(g.best.feature, 0);
  close(g.best.threshold, L.rootChoice().byFeature[0].threshold);
  // Recomputed independently with bestSplit on each side of every root.
  const { X, y } = L.trainData();
  const idx = X.map((_, i) => i);
  const leafImp = (rows) => (rows.length * gini([0, 1].map((c) => rows.filter((i) => y[i] === c).length))) / 80;
  const side = (rows) => {
    const s = bestSplit(X, y, rows);
    return s ? leafImp(s.left) + leafImp(s.right) : leafImp(rows);
  };
  let best = Infinity;
  for (const f of [0, 1]) {
    const v = [...new Set(X.map((x) => x[f]))].sort((a, b) => a - b);
    for (let k = 0; k + 1 < v.length; k++) {
      const th = (v[k] + v[k + 1]) / 2;
      best = Math.min(best, side(idx.filter((i) => X[i][f] < th)) + side(idx.filter((i) => X[i][f] >= th)));
    }
  }
  close(g.best.impurity, best);
  const gl = L.leafBoxes(t).reduce((s, b) => s + (b.node.n * gini(b.node.counts)) / 80, 0);
  close(g.greedy, gl);
  assert.equal(L.fmtG(g.greedy), '0.266');
  assert.equal(L.fmtG(g.best.impurity), '0.221');
});

test('trees read-out rows and page', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { t: 0 }, { t: 10 }, { t: 4.2 }, { depth: 1 }, { depth: 8 }, { q: [0, 0] }, { q: [10, 10] }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Decision trees · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /seed 99/);
  assert.match(html, /seed 100/);
  assert.match(html, /<link rel="stylesheet" href="\/ml\/assets\/ml\.css">\s*<link rel="stylesheet" href="\/ml\/decision-trees\/lesson\.css">/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/random-forests\/">/);
  assert.match(html, /href="\/ml\/support-vector-machines\/"/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /href="\/ml\/k-nearest-neighbours\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 15);
  for (const key of keys) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('trees page values match the computations', () => {
  const pv = L.pageValues();
  const rows = L.depthTable();
  const b = L.bestDepth();
  const rc = L.rootChoice();
  assert.equal(pv['n-train'], '80');
  assert.equal(pv['n-test'], '400');
  assert.equal(pv['n0'], '43');
  assert.equal(pv['n1'], '37');
  assert.equal(pv['flipped'], String(L.trainData().flipped.length));
  assert.equal(pv['flipped'], '8');
  assert.equal(pv['test-flipped'], '40');
  assert.equal(pv['rule-test'], '90.0%');
  const s5 = L.splitScore(0, L.T_GUESS);
  assert.equal(pv['g-left0'], String(s5.left[0]));
  assert.equal(pv['g-left1'], String(s5.left[1]));
  assert.equal(pv['g-right0'], String(s5.right[0]));
  assert.equal(pv['g-right1'], String(s5.right[1]));
  assert.deepEqual([s5.left, s5.right], [[28, 9], [15, 28]]);
  assert.equal(pv['g-acc'], '70.0%');
  assert.equal(pv['g-gl'], L.fmtG(s5.gLeft));
  assert.equal(pv['parent-g'], '0.497');
  assert.equal(pv['x1-t'], '4.96');
  assert.equal(pv['x1-g'], '0.414');
  assert.equal(pv['x2-t'], '3.04');
  assert.equal(pv['x2-g'], '0.400');
  assert.equal(pv['x1-a'], L.minInterval(0).a.toFixed(3));
  assert.equal(pv['x1-b'], L.minInterval(0).b.toFixed(3));
  // "just above 5": the step ends above 5 but below 5.01.
  assert.ok(L.minInterval(0).b > 5 && L.minInterval(0).b < 5.01);
  assert.equal(pv['x1-b'], '5.001');
  assert.equal(pv['best-d'], String(b));
  assert.equal(pv['best-test'], L.pct(rows[b - 1].test));
  assert.equal(pv['best-test'], '82.5%');
  assert.equal(pv['best-train'], L.pct(rows[b - 1].train));
  assert.equal(pv['full-d'], '6');
  assert.equal(pv['full-leaves'], '18');
  assert.equal(pv['full-test'], L.pct(rows[7].test));
  assert.equal(pv['d3-test'], L.pct(rows[2].test));
  assert.equal(pv['d1-test'], L.pct(rows[0].test));
  assert.equal(pv['greedy-g'], '0.266');
  assert.equal(pv['best2-g'], '0.221');
  assert.equal(pv['best2-t'], L.fmtT(L.depth2Search().best.threshold));
  // Words the page states about these numbers.
  assert.equal(rc.byFeature[0].feature, 0);
  assert.match(html, /x₁ &lt; 5|x₁ < 5/);
  assert.match(html, /\(2, 8\)/);
  // Step 1: "yes gives class 0, no gives class 1".
  const st = L.stump(0, L.T_GUESS);
  assert.deepEqual([st.left.cls, st.right.cls], [0, 1]);
  assert.match(html, /yes gives class 0, no gives class 1/);
  assert.equal(L.readout(0)[0], 'x₁ < 5.00?');
  // Step 5 maths: no two training cases share a value of x₁ or of x₂.
  const tx = L.trainData().X;
  for (const f of [0, 1]) assert.equal(new Set(tx.map((x) => x[f])).size, tx.length);
  assert.match(html, /no two training cases share a value of x₁ or of x₂/);
  assert.doesNotMatch(html, /not globally optimal/);
  assert.deepEqual(L.Q0, [2, 8]);
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.doesNotMatch(html, /Taylor/);
  assert.match(html, /not always the best overall/);
  assert.equal(L.pct(1), '100%');
});

test('rescaling a feature by a positive number leaves the tree\'s splits unchanged', () => {
  const { X, y } = L.trainData();
  const big = X.map((x) => [x[0], x[1] * 100]);
  const leafSets = (t, data) => {
    const m = new Map();
    data.forEach((x, i) => {
      const n = (function walk(nd) { return nd.leaf ? nd : walk(x[nd.feature] < nd.threshold ? nd.left : nd.right); })(t);
      if (!m.has(n)) m.set(n, []);
      m.get(n).push(i);
    });
    return [...m.values()].map((v) => v.join(',')).sort();
  };
  for (const d of [1, 2, 3, Infinity]) {
    const a = growTree(X, y, { maxDepth: d });
    const b = growTree(big, y, { maxDepth: d });
    assert.deepEqual(leafSets(a, X), leafSets(b, big));
  }
});

test('trees wording claims hold', () => {
  // Step 4: "At max depth 2 one leaf holds 1 circle and 1 triangle."
  const ties = L.leafBoxes(L.treeAt(2)).map((b) => b.node).filter((n) => n.counts[0] === n.counts[1]);
  assert.deepEqual(ties.map((n) => n.counts), [[1, 1]]);
  assert.match(html, /At max depth 2 one leaf holds 1 circle and 1 triangle/);
  assert.equal(L.bestDepth(), 2);
  // Quiz 2: the x₁ split makes groups of more equal size than the x₂ split.
  const rc = L.rootChoice();
  const gap = (s) => Math.abs(s.left.length - s.right.length);
  assert.ok(gap(rc.byFeature[0]) < gap(rc.byFeature[1]));
  // About the data: fixed facts stated in words.
  assert.match(html, /The 80 training cases use seed 99 and the 400 test cases seed 100; 8 training labels and 40 test labels were flipped\./);
  assert.equal(L.trainData().flipped.length, 8);
  assert.equal(L.testData().flipped.length, 40);
  assert.match(html, /slider from 1 to 8/);
  assert.deepEqual([L.DEPTHS[0], L.DEPTHS.at(-1)], [1, 8]);
  assert.match(html, /draws three levels below the root/);
  assert.equal(L.DISPLAY_DEPTH, 3);
  // Step 6: the path goes right, then left.
  assert.match(html, /No, so go right\. Is x₁ less than <span data-value="d2-r-t"><\/span>\? Yes, so go left\./);
  const t2 = L.treeAt(2);
  assert.equal(t2.feature, 1);
  assert.equal(t2.right.feature, 0);
  // Step 3 / quiz 2: x₂ is asked first.
  assert.match(html, /the tree's first question is "is x₂ less than/);
  assert.doesNotMatch(html, /delve|unlock|harness the power|it's important to note|dive in|in today's world/i);
});
