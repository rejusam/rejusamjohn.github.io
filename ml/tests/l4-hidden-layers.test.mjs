import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { logisticFit, sigmoid } from '../assets/classify.js';
import { mlpInit, mlpForward, trainer, accuracyNet, predictProb } from '../assets/nn.js';
import * as L from '../hidden-layers/stage.js';
import { bestLinear } from '../neurons/stage.js';

const html = readFileSync(new URL('../hidden-layers/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_X0 = -0.18646246620992768;

// The run the page trains, written out from nn.js directly.
const direct = (H) => {
  const tr = L.trainSet();
  return trainer({
    net: mlpInit([2, H, 1], { hidden: 'tanh', init: 'xavier', seed: 155 }),
    X: tr.X, y: tr.y, batch: 20, kind: 'adam', hyper: { eta: 0.03 }, seed: 154,
  });
};

test('data reproducible', () => {
  assert.equal(L.SEED_TRAIN, 151);
  assert.equal(L.SEED_TEST, 152);
  assert.equal(L.SEED_INIT, 155);
  assert.equal(L.SEED_SHUFFLE, 154);
  // Two rings, written out: class 0 at radius 1, class 1 at radius 2.2, both
  // plus 0.25 × noise, each at a uniform random angle; class 0 rows first.
  const rings = (n, seed) => {
    const rng = mulberry32(seed);
    const X = [];
    const y = [];
    for (let i = 0; i < n; i++) {
      const cls = i < n / 2 ? 0 : 1;
      const th = 2 * Math.PI * rng();
      const r = (cls ? 2.2 : 1) + 0.25 * randn(rng);
      X.push([r * Math.cos(th), r * Math.sin(th)]);
      y.push(cls);
    }
    return { X, y };
  };
  assert.deepEqual(L.trainSet(), rings(200, 151));
  assert.deepEqual(L.testSet(), rings(400, 152));
  assert.deepEqual(L.trainSet(), L.trainSet());
  assert.equal(L.trainSet().X[0][0], LITERAL_TRAIN_X0);
  // Every point fits inside the plot.
  for (const { X } of [L.trainSet(), L.testSet()]) {
    for (const x of X) assert.ok(Math.abs(x[0]) < L.DOM[1] && Math.abs(x[1]) < L.DOM[1]);
  }
});

test('single neuron test accuracy ≤ 0.65 on circles', () => {
  const tr = L.trainSet();
  const te = L.testSet();
  const f = L.fitOne();
  assert.equal(f.ok, true);
  assert.deepEqual(f, logisticFit(tr.X, tr.y));
  const test = L.oneAccuracy(te.X, te.y);
  assert.ok(test <= 0.65, String(test));
  assert.equal(L.pct(test), '49.8%');
  assert.equal(L.pct(L.oneAccuracy(tr.X, tr.y)), '57.0%');
  // The best of all straight lines (checked against the previous lesson's search)
  // still gets well under 80%: it can only cut a slice off the outer ring.
  const best = L.bestLine(tr.X, tr.y);
  assert.equal(best, bestLinear(tr.X, tr.y).acc);
  assert.ok(best < 0.8 && best >= L.oneAccuracy(tr.X, tr.y));
  assert.equal(L.storyValues()['best-line'], L.pct(best));
  // The same neuron as a [2, 1] network gives the same predictions.
  const net = { sizes: [2, 1], hidden: 'tanh', W: [[f.w]], b: [[f.w0]] };
  te.X.forEach((x) => {
    assert.ok(Math.abs(predictProb(net, x) - sigmoid(f.w0 + f.w[0] * x[0] + f.w[1] * x[1])) < 1e-15);
  });
});

test('H = 4 reaches test accuracy ≥ 0.95 within 150 epochs', () => {
  const te = L.testSet();
  const t = L.makeRun(4);
  let first = null;
  while (!L.finished(t)) {
    L.advance(t, L.ITERS_PER_EPOCH);
    if (first === null && accuracyNet(t.state.net, te.X, te.y) >= 0.95) first = t.state.epoch;
  }
  assert.equal(t.state.epoch, L.EPOCHS);
  assert.equal(t.state.iter, L.MAX_ITER);
  const acc = accuracyNet(t.state.net, te.X, te.y);
  assert.ok(acc >= 0.95, String(acc));
  assert.ok(first !== null && first <= L.EPOCHS);
  const sv = L.storyValues();
  assert.equal(sv['h4-test'], L.pct(acc));
  // At epoch ends, 95% first shows at the end of epoch 19. Checked after every
  // step it is reached part-way through epoch 19, while the read-out still shows
  // 18 finished epochs; the page says so.
  assert.equal(first, 19);
  let firstIter = null;
  const u = L.makeRun(4);
  while (!L.finished(u)) {
    L.advance(u, 1);
    if (firstIter === null && accuracyNet(u.state.net, te.X, te.y) >= 0.95) firstIter = u.state.iter;
  }
  assert.ok(firstIter % L.ITERS_PER_EPOCH !== 0);
  const during = Math.ceil(firstIter / L.ITERS_PER_EPOCH);
  assert.equal(sv['h4-first95'], String(during));
  assert.equal(sv['h4-first95'], '19');
  assert.equal(Math.floor(firstIter / L.ITERS_PER_EPOCH), 18);
  assert.equal(L.first95Epoch(L.makeRun(4)), during);
  assert.match(html, /part-way through epoch <span data-story="h4-first95">19<\/span>, while the read-out still shows 18 finished epochs/);
  // The loss falls a long way: the end loss is under a tenth of the start.
  const el = t.state.epochLoss;
  assert.ok(el[L.EPOCHS] < el[0] / 10, `${el[0]} → ${el[L.EPOCHS]}`);
  assert.equal(sv['h4-loss0'], L.num(el[0], 2));
  assert.equal(sv['h4-loss'], L.num(el[L.EPOCHS], 3));
});

test('H = 1 stays ≤ 0.8', () => {
  const te = L.testSet();
  const t = L.makeRun(1);
  let best = 0;
  while (!L.finished(t)) {
    L.advance(t, L.ITERS_PER_EPOCH);
    best = Math.max(best, accuracyNet(t.state.net, te.X, te.y));
  }
  // Not just at the end: at no epoch does it reach 0.8.
  assert.ok(best <= 0.8, String(best));
  assert.ok(L.resultsRow(1).test <= 0.8);
  // H = 2 also falls short; 3, 4 and 8 all pass 95%.
  assert.ok(L.resultsRow(2).test < 0.9);
  for (const H of [3, 4, 8]) assert.ok(L.resultsRow(H).test >= 0.95, String(H));
});

test('H = 2 hidden view: the output neuron on the hidden activations is the network', () => {
  const tr = L.trainSet();
  const net = L.trainedNet(2);
  const pts = L.hiddenPoints(net, tr.X);
  pts.forEach((a, i) => {
    const { as } = mlpForward(net, tr.X[i]);
    assert.deepEqual(a, as[1]);
    for (const v of a) assert.ok(v > -1 && v < 1);
  });
  // Identity: a line in (a₁, a₂) gets exactly the network's accuracy.
  const acc = L.outputAccuracyHidden(net, tr.X, tr.y);
  assert.equal(acc, accuracyNet(net, tr.X, tr.y));
  assert.equal(L.storyValues()['h2-hidden'], L.pct(acc));
  // There, one line does far better than one line on (x₁, x₂).
  assert.ok(acc - L.oneAccuracy(tr.X, tr.y) > 0.25);
  // But not perfectly: two units cannot close a ring.
  assert.ok(acc < 0.9);
});

test('results table equals the trainer\'s numbers', () => {
  const tr = L.trainSet();
  const te = L.testSet();
  assert.deepEqual(L.H_CHOICES, [1, 2, 3, 4, 8]);
  const sv = L.storyValues();
  for (const H of L.H_CHOICES) {
    const t = direct(H);
    t.step(L.MAX_ITER);
    assert.equal(t.state.epoch, 150);
    const row = L.resultsRow(H);
    assert.equal(row.train, accuracyNet(t.state.net, tr.X, tr.y));
    assert.equal(row.test, accuracyNet(t.state.net, te.X, te.y));
    // The live run the stage keeps for H gives the same row when it reaches the end.
    const live = L.makeRun(H);
    L.advance(live, 7);
    L.advance(live, 10000);
    assert.deepEqual(L.rowOf(live), row);
    assert.equal(sv[`h${H}-test`], L.pct(row.test));
    assert.equal(sv[`h${H}-train`], L.pct(row.train));
  }
  assert.equal(sv['h1-test'], '63.2%');
  assert.equal(sv['h2-test'], '79.5%');
  assert.equal(sv['h4-test'], '97.5%');
  assert.equal(sv['h4-train'], '99.5%');
  // Table rows read as the page shows them.
  const rows = L.tableRows(new Map([[4, L.rowOf(direct(4))]]));
  assert.equal(rows.length, 5);
  assert.deepEqual(rows[0], ['1', '–', '–']);
});

test('training is deterministic across two runs', () => {
  const a = L.makeRun(4);
  const b = L.makeRun(4);
  // Different chunk sizes, as frames of different lengths would give.
  while (!L.finished(a)) L.advance(a, 3);
  while (!L.finished(b)) L.advance(b, 17);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
  assert.deepEqual(a.state.net, b.state.net);
  // advance never runs past 150 epochs.
  L.advance(a, 50);
  assert.equal(a.state.iter, L.MAX_ITER);
  // Reset gives the seeded start again.
  a.reset();
  assert.equal(a.state.iter, 0);
  assert.deepEqual(a.state.net, L.makeRun(4).state.net);
  // The page paces training: at most PACE steps a frame however many the loop offers.
  const c = L.makeRun(4);
  L.frameStep(c, 200);
  assert.equal(c.state.iter, L.PACE);
  assert.equal(L.ITERS_PER_EPOCH, Math.ceil(200 / 20));
  // "The first 20 epochs run at one step a frame; later epochs a few steps a frame."
  for (let it = 0; it < 20 * L.ITERS_PER_EPOCH; it++) assert.equal(L.paceAt(it), 1);
  for (let it = 20 * L.ITERS_PER_EPOCH; it < L.MAX_ITER; it++) assert.ok(L.paceAt(it) > 1 && L.paceAt(it) <= 6);
  assert.match(html, /The first 20 epochs run at one step a frame/);
});

test('outside the data the boundary is whatever the weights give', () => {
  const net = L.trainedNet(4);
  // Three times further out than any point, the network is confident anyway.
  const far = L.farRing(net, L.FAR_R, 360);
  for (const p of far) assert.ok(p > 0.9, String(p));
  let mx = 0;
  for (const { X } of [L.trainSet(), L.testSet()]) for (const x of X) mx = Math.max(mx, Math.hypot(x[0], x[1]));
  assert.ok(L.FAR_R > 2 * mx);
  const sv = L.storyValues();
  assert.equal(sv['far-min'], L.num(Math.min(...far), 2));
  // The boundary is not a circle: its distance from the centre varies with direction.
  const { min, max } = L.boundaryRadii(net, 360);
  assert.ok(max - min > 0.2, `${min} ${max}`);
  assert.equal(sv['r-min'], L.num(min, 1));
  assert.equal(sv['r-max'], L.num(max, 1));
  assert.equal(L.pageValues()['max-r'], L.num(mx, 1));
  assert.equal(L.maxRadius(), mx);
  // The four hidden lines end up close around the inner ring, inside the gap to the outer one.
  const d = L.lineDistances(net);
  assert.equal(d.length, 4);
  for (const v of d) assert.ok(v > 0.9 && v < 1.6, String(v));
  assert.equal(sv['line-min'], L.num(Math.min(...d), 1));
  assert.equal(sv['line-max'], L.num(Math.max(...d), 1));
  // The fast shading function is the network's p̂.
  for (const x of L.testSet().X.slice(0, 50)) assert.ok(Math.abs(L.fastProb(net)(x[0], x[1]) - predictProb(net, x)) < 1e-14);
  // A crossing really is a crossing.
  const r = L.boundaryRadii(net, 4).radii[0];
  assert.ok(predictProb(net, [r - 1e-6, 0]) < 0.5 && predictProb(net, [r + 1e-6, 0]) > 0.5);
});

test('a linear activation would collapse to one line', () => {
  // W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2) for any x.
  const net = mlpInit([2, 4, 1], { seed: 9 });
  const W1 = net.W[0];
  const b1 = [0.3, -0.2, 0.5, 0.1];
  const W2 = net.W[1][0];
  const b2 = -0.4;
  const Wc = [0, 1].map((i) => W2.reduce((s, w, j) => s + w * W1[j][i], 0));
  const bc = W2.reduce((s, w, j) => s + w * b1[j], 0) + b2;
  for (const x of [[1, 2], [-0.5, 0.7], [3, -3]]) {
    const h = W1.map((row, j) => row[0] * x[0] + row[1] * x[1] + b1[j]);
    const out = h.reduce((s, v, j) => s + W2[j] * v, b2);
    assert.ok(Math.abs(out - (Wc[0] * x[0] + Wc[1] * x[1] + bc)) < 1e-12);
  }
});

test('contour follows the p̂ = 0.5 level', () => {
  // p̂ = 0.5 on the circle of radius 1.5 for this radial function.
  const nx = 40;
  const ny = 40;
  const at = (i, j) => [-3 + (6 * (i + 0.5)) / nx, 3 - (6 * (j + 0.5)) / ny];
  const vals = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const [x, y] = at(i, j);
    vals.push(sigmoid(2 * (Math.hypot(x, y) - 1.5)));
  }
  const segs = L.contour(vals, nx, ny, 0.5);
  assert.ok(segs.length > 40);
  for (const seg of segs) {
    for (const [gx, gy] of seg) {
      const r = Math.hypot(-3 + (6 * gx) / nx, 3 - (6 * gy) / ny);
      assert.ok(Math.abs(r - 1.5) < 0.05, String(r));
    }
  }
  assert.deepEqual(L.contour(new Array(16).fill(0.2), 4, 4, 0.5), []);
});

test('hidden-layers read-out rows and page', () => {
  const t = L.makeRun(4);
  const states = [L.info(t, 4), (L.advance(t, 333), L.info(t, 4)), L.info(L.makeRun(2), 2), L.info(L.makeRun(1), 1)];
  for (let step = 0; step < 6; step++) {
    for (const st of states) {
      const rows = L.readout(step, st);
      assert.ok(rows.length <= 5);
      for (const r of rows) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 22, r);
      }
    }
  }
  // Step 6 says when its run is not finished, and stops saying so at the end.
  const fresh = L.makeRun(4);
  assert.ok(L.readout(5, L.info(fresh, 4)).includes('run not finished:'));
  L.advance(fresh, L.MAX_ITER);
  assert.ok(!L.readout(5, L.info(fresh, 4)).some((r) => /not finished/.test(r)));
  assert.match(html, /The numbers in this text are for the end of its 150-epoch run/);
  assert.match(html, /Play steps through the lesson, and Train trains the network/);
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Hidden layers: bending the boundary · ML Explorer<\/title>/);
  assert.match(html, /<h1>Hidden layers: bending the boundary<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  for (const s of [151, 152, 154, 155]) assert.match(html, new RegExp(`seed ${s}`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /approximate any continuous function on a bounded region as closely as you like/);
  assert.match(html, /href="\/ml\/neurons\/"/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/backpropagation\/">Next lesson: Backpropagation →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always better|can learn anything/i);
  // Values filled at run time are already in the page and equal the computed ones.
  const pv = L.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 6);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  // Numbers from the seeded training runs are written in the page and equal the runs'.
  const sv = L.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 10);
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
