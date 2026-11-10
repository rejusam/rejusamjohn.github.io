import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { cnnInit, cnnForward, cnnBackward, cnnFlatten, cnnUnflatten, cnnParamCount, denseParamCount, maxPool } from '../assets/conv.js';
import { mlpInit, trainer, accuracyNet, flatten, optInit, optStep, bceLogit } from '../assets/nn.js';
import * as S from '../cnns/stage.js';

const html = readFileSync(new URL('../cnns/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_00 = 0.11488865102349131;

// The shape images written out again: class i % 2; centre row, then centre column,
// each 1 + floor(u · 6); one randn per pixel, row-major; 1 on the shape, 0 off it,
// plus 0.15 · randn, clipped to [0, 1].
function shapes(n, seed) {
  const rng = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const cr = 1 + Math.floor(rng() * 6);
    const cc = 1 + Math.floor(rng() * 6);
    const img = [];
    for (let r = 0; r < 8; r++) {
      const row = [];
      for (let c = 0; c < 8; c++) {
        const dr = Math.abs(r - cr);
        const dc = Math.abs(c - cc);
        const on = cls === 0 ? dr + dc <= 1 : Math.max(dr, dc) === 1;
        row.push(Math.min(1, Math.max(0, (on ? 1 : 0) + 0.15 * randn(rng))));
      }
      img.push(row);
    }
    X.push(img);
    y.push(cls);
  }
  return { X, y };
}

const meanLoss = (net, X, y) => X.reduce((s, x, i) => s + bceLogit(y[i], cnnForward(net, x).z), 0) / X.length;
const acc = (net, X, y) => X.reduce((s, x, i) => s + ((cnnForward(net, x).z > 0 ? 1 : 0) === y[i] ? 1 : 0), 0) / X.length;

// Adam, η 0.02, batch 20, shuffle seed 204, written out without cnnTrainer.
function directRun(net0, X, y, epochs, { eta = 0.02, batch = 20, seed = 204 } = {}) {
  let params = cnnFlatten(net0);
  let net = cnnUnflatten(net0, params);
  let opt = optInit('adam', params.length);
  const rng = mulberry32(seed);
  const losses = [meanLoss(net, X, y)];
  for (let e = 0; e < epochs; e++) {
    const order = Array.from({ length: X.length }, (_, i) => i);
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let c = 0; c < X.length; c += batch) {
      const idx = order.slice(c, c + batch);
      const g = new Array(params.length).fill(0);
      for (const k of idx) cnnFlatten(cnnBackward(net, X[k], y[k]).grads).forEach((v, q) => { g[q] += v; });
      const r = optStep('adam', { eta }, opt, params, g.map((v) => v / idx.length));
      params = r.params;
      opt = r.state;
      net = cnnUnflatten(net, params);
    }
    losses.push(meanLoss(net, X, y));
  }
  return { net, losses };
}

const nudged = (net, eps) => {
  const v = cnnFlatten(net);
  v[0] *= 1 + eps;
  return cnnUnflatten(net, v);
};

// The page's end-of-run results, computed in odd-sized slices, shared below.
let RUNS = null;
function pageRuns() {
  if (RUNS) return RUNS;
  for (const k of S.RUN_KINDS) while (!S.runWork(k, 37));
  RUNS = { cnn: S.runResult('cnn'), dense: S.runResult('dense') };
  return RUNS;
}

test('data reproducible', () => {
  assert.equal(S.SEED_TRAIN, 201);
  assert.equal(S.SEED_TEST, 202);
  assert.equal(S.SEED_INIT, 203);
  assert.equal(S.SEED_SHUFFLE, 204);
  assert.equal(S.SEED_DENSE, 205);
  assert.equal(S.ETA, 0.02);
  assert.equal(S.BATCH, 20);
  assert.equal(S.EPOCHS, 30);
  assert.equal(S.ITERS, 300);
  assert.deepEqual(S.trainSet(), shapes(200, 201));
  assert.deepEqual(S.testSet(), shapes(400, 202));
  assert.equal(S.trainSet().X[0][0][0], LITERAL_TRAIN_00);
  assert.equal(S.trainSet().X[0][0][0], 0.11488865102349131);
  assert.deepEqual(S.startNet(), cnnInit({ seed: 203 }));
  assert.deepEqual(S.denseStart(), mlpInit([64, 18, 1], { hidden: 'relu', init: 'he', seed: 205 }));
  assert.deepEqual(S.flat(S.trainSet().X[3]), S.trainSet().X[3].flat());
});

test('cnnTrainer: batches, epochs and the starting loss', () => {
  const tr = shapes(200, 201);
  const net = cnnInit({ seed: 203 });
  const t = S.cnnTrainer({ net, X: tr.X, y: tr.y, batch: 20, kind: 'adam', hyper: { eta: 0.02 }, seed: 204 });
  // epochLoss[0] is the mean loss before any step.
  assert.deepEqual(t.state.epochLoss, [meanLoss(net, tr.X, tr.y)]);
  assert.equal(t.state.iter, 0);
  assert.equal(t.state.epoch, 0);
  // One step: the mean of the per-image gradients over the first shuffled batch, then Adam.
  const rng = mulberry32(204);
  const order = Array.from({ length: 200 }, (_, i) => i);
  for (let i = 199; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const idx = order.slice(0, 20);
  const per = idx.map((k) => cnnFlatten(cnnBackward(net, tr.X[k], tr.y[k]).grads));
  const mean = per[0].map((_, q) => per.reduce((s, g) => s + g[q], 0) / 20);
  const g = S.batchGradCnn(net, tr.X, tr.y, idx);
  g.grad.forEach((v, q) => assert.ok(Math.abs(v - mean[q]) < 1e-15, String(q)));
  assert.ok(Math.abs(g.loss - idx.reduce((s, k) => s + cnnBackward(net, tr.X[k], tr.y[k]).loss, 0) / 20) < 1e-15);
  const want = optStep('adam', { eta: 0.02 }, optInit('adam', 39), cnnFlatten(net), g.grad).params;
  t.step(1);
  assert.deepEqual(cnnFlatten(t.state.net), want);
  assert.equal(t.state.iter, 1);
  assert.equal(t.state.batchLoss.length, 1);
  assert.equal(t.state.batchLoss[0], g.loss);
  // The input net is not changed.
  assert.deepEqual(net, cnnInit({ seed: 203 }));
  // Ten steps make one epoch of 200 images in batches of 20.
  t.step(9);
  assert.equal(t.state.epoch, 1);
  assert.equal(t.state.epochLoss.length, 2);
  assert.equal(t.state.epochLoss[1], meanLoss(t.state.net, tr.X, tr.y));
  // A last partial batch is kept: batch 30 makes 7 steps per epoch (6 × 30 + 20).
  const t30 = S.cnnTrainer({ net, X: tr.X, y: tr.y, batch: 30, kind: 'adam', hyper: { eta: 0.02 }, seed: 204 });
  t30.step(6);
  assert.equal(t30.state.epoch, 0);
  t30.step(1);
  assert.equal(t30.state.epoch, 1);
  // Reset gives the same start and the same path again.
  const a = t.state.epochLoss.slice();
  t.reset();
  assert.deepEqual(t.state.epochLoss, [a[0]]);
  t.step(10);
  assert.deepEqual(t.state.epochLoss, a);
});

test('training deterministic', () => {
  const tr = shapes(200, 201);
  const one = S.makeTrainer();
  const two = S.makeTrainer();
  one.step(300);
  two.step(150);
  two.step(150);
  assert.deepEqual(one.state.epochLoss, two.state.epochLoss);
  assert.deepEqual(cnnFlatten(one.state.net), cnnFlatten(two.state.net));
  assert.equal(one.state.diverged, false);
  // And it is the run written out by hand.
  const d = directRun(cnnInit({ seed: 203 }), tr.X, tr.y, 30);
  assert.deepEqual(one.state.epochLoss, d.losses);
  assert.deepEqual(cnnFlatten(one.state.net), cnnFlatten(d.net));
  // The page's end-of-run result is the same run.
  const runs = pageRuns();
  assert.deepEqual(runs.cnn.losses, d.losses);
  assert.deepEqual(cnnFlatten(runs.cnn.net), cnnFlatten(d.net));
  // The live run with test accuracy, stepped one at a time, matches too.
  const live = S.makeLive();
  while (live.trainer.state.iter < 300) live.advance(1);
  assert.deepEqual(live.trainer.state.epochLoss, d.losses);
  assert.deepEqual(live.testAcc, runs.cnn.testAcc);
  assert.equal(live.testAcc.length, 31);
  // The dense run is nn.js's trainer with the same optimiser, η, batch, epochs and shuffle seed.
  const flatX = tr.X.map((m) => m.flat());
  const td = trainer({ net: mlpInit([64, 18, 1], { hidden: 'relu', init: 'he', seed: 205 }), X: flatX, y: tr.y, batch: 20, kind: 'adam', hyper: { eta: 0.02 }, seed: 204 });
  td.step(300);
  assert.deepEqual(runs.dense.losses, td.state.epochLoss);
  const te = shapes(400, 202);
  assert.equal(runs.dense.testAcc, accuracyNet(td.state.net, te.X.map((m) => m.flat()), te.y));
  assert.equal(runs.dense.trainAcc, accuracyNet(td.state.net, flatX, tr.y));
});

test('CNN reaches test accuracy ≥ 0.95', () => {
  const runs = pageRuns();
  const te = shapes(400, 202);
  const tr = shapes(200, 201);
  const c = runs.cnn;
  assert.equal(c.testAcc.at(-1), acc(c.net, te.X, te.y));
  assert.ok(c.testAcc.at(-1) >= 0.95);
  assert.equal(c.testAcc.at(-1), 1);
  assert.equal(c.trainAcc, acc(c.net, tr.X, tr.y));
  assert.equal(c.trainAcc, 1);
  // At the start it is close to a coin toss.
  assert.equal(c.testAcc[0], acc(cnnInit({ seed: 203 }), te.X, te.y));
  assert.ok(c.testAcc[0] > 0.4 && c.testAcc[0] < 0.6);
  // The loss falls every epoch.
  for (let e = 1; e <= 30; e++) assert.ok(c.losses[e] < c.losses[e - 1], String(e));
  // Test accuracy first reaches 95% at this epoch and stays there.
  const first = c.testAcc.findIndex((a) => a >= 0.95);
  assert.equal(S.firstAbove(c.testAcc, 0.95), first);
  for (let e = first; e <= 30; e++) assert.ok(c.testAcc[e] >= 0.95, String(e));
  const sv = S.storyValues();
  assert.equal(sv['cnn-start-loss'], S.num(c.losses[0], 3));
  assert.equal(sv['cnn-end-loss'], S.num(c.losses[30], 3));
  assert.equal(sv['cnn-start-acc'], S.pct(c.testAcc[0]));
  assert.equal(sv['cnn-test'], S.pct(c.testAcc[30]));
  assert.equal(sv['cnn-test'], '100%');
  assert.equal(sv['first-95'], String(first));
  assert.equal(sv['first-95-acc'], S.pct(c.testAcc[first]));
  assert.equal(sv['before-95-acc'], S.pct(c.testAcc[first - 1]));
});

test('the dense network fits the training images but does worse on the test images', () => {
  const runs = pageRuns();
  const d = runs.dense;
  assert.equal(d.trainAcc, 1);
  assert.ok(d.testAcc < runs.cnn.testAcc.at(-1));
  assert.ok(d.testAcc > 0.85 && d.testAcc < 0.95, String(d.testAcc));
  assert.ok(d.losses[30] < runs.cnn.losses[30]);
  const sv = S.storyValues();
  assert.equal(sv['dense-test'], S.pct(d.testAcc));
  assert.equal(sv['dense-train'], S.pct(d.trainAcc));
  assert.equal(sv['dense-wrong'], String(Math.round((1 - d.testAcc) * 400)));
  assert.equal(sv['dense-end-loss'], S.num(d.losses[30], 3));
  assert.equal(sv['cnn-end-loss'], S.num(runs.cnn.losses[30], 3));
});

test('param counts on the page equal the helpers', () => {
  assert.equal(cnnParamCount(cnnInit({ seed: 203 })), 39);
  assert.equal(denseParamCount(64, 18), 1189);
  assert.equal(flatten(mlpInit([64, 18, 1], { hidden: 'relu', init: 'he', seed: 205 })).length, 1189);
  const pv = S.pageValues();
  assert.equal(pv['cnn-params'], String(cnnParamCount(S.startNet())));
  assert.equal(pv['dense-params'], S.count(denseParamCount(64, 18)));
  assert.equal(pv['dense-params'], '1,189');
  assert.equal(pv['dense-ratio'], String(Math.round(1189 / 39)));
  // The parts written out on the page add up to the totals.
  assert.equal(2 * 9 + 2 + 18 + 1, 39);
  assert.equal(64 * 18 + 18 + 18 + 1, 1189);
  assert.equal(pv['conv-w'], '18');
  assert.equal(pv['dense-in'], '1,152');
  const step6 = S.readout(5, S.compareInfo(pageRuns()));
  assert.ok(step6.some((r) => r.includes('39 ')) && step6.some((r) => r.includes('1,189 ')), step6.join('|'));
});

test('pooled-map change under a 1-pixel shift is smaller than the raw-map change', () => {
  const tr = shapes(200, 201);
  const net = cnnInit({ seed: 203 });
  const img = tr.X[S.EXAMPLE];
  assert.equal(S.EXAMPLE, 0);
  const sh = img.map((row) => row.map((_, c) => (c >= 1 ? row[c - 1] : 0)));
  assert.deepEqual(S.shiftRight(img, 1), sh);
  const a = cnnForward(net, img);
  const b = cnnForward(net, sh);
  const change = (p, q) => {
    const P = p.flat(2);
    const Q = q.flat(2);
    return P.reduce((s, v, i) => s + Math.abs(v - Q[i]), 0) / P.reduce((s, v) => s + Math.abs(v), 0);
  };
  const info = S.shiftInfo(net, img);
  assert.ok(Math.abs(info.raw - change(a.relu, b.relu)) < 1e-12);
  assert.ok(Math.abs(info.pooled - change(a.pooled, b.pooled)) < 1e-12);
  assert.ok(info.pooled < info.raw);
  assert.ok(info.pooled < info.raw / 1.5);
  // The same holds for each of the 12 examples in step 1, with the starting filters.
  for (let k = 0; k < 12; k++) {
    const p = S.shiftInfo(net, tr.X[k]);
    assert.ok(p.pooled < p.raw, String(k));
  }
  const sv = S.storyValues();
  assert.equal(sv['shift-raw'], S.pctWhole(info.raw));
  assert.equal(sv['shift-pooled'], S.pctWhole(info.pooled));
  // The pooled maps' maxima: how many of the 18 pooled values are unchanged.
  assert.equal(sv['pool-same'], String(info.same));
  let same = 0;
  a.pooled.flat(2).forEach((v, i) => { if (v === b.pooled.flat(2)[i]) same++; });
  assert.equal(info.same, same);
  // Each unchanged pooled value is above 0, and its max is the same pixel pattern
  // one column to the right, inside the same 2 × 2 block.
  for (let k = 0; k < 2; k++) for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    if (a.pooled[k][r][c] !== b.pooled[k][r][c]) continue;
    assert.ok(a.pooled[k][r][c] > 0);
    const [i, j] = a.args[k][r][c];
    const [i2, j2] = b.args[k][r][c];
    assert.deepEqual([i2, j2], [i, j + 1]);
    assert.equal(j >> 1, (j + 1) >> 1);
  }
});

test('max-pool passes the gradient back only to the maximum', () => {
  const tr = shapes(200, 201);
  const net = cnnInit({ seed: 203 });
  const img = tr.X[S.EXAMPLE];
  const fw = cnnForward(net, img);
  // Every pooled value is the largest of its 2 × 2 window of ReLU values.
  assert.deepEqual(fw.relu.map((m) => maxPool(m, 2).out), fw.pooled);
  // Rebuild the filter gradient from the max positions alone: for each pooled cell,
  // δ times its dense weight times the 3 × 3 patch under the window's max (if that
  // map value is above 0). The other three cells of each window add nothing.
  for (const y of [0, 1]) {
    const g = cnnBackward(net, img, y).grads;
    const dz = 1 / (1 + Math.exp(-fw.z)) - y;
    for (let k = 0; k < 2; k++) {
      const want = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      let used = 0;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        const [i, j] = fw.args[k][r][c];
        if (!(fw.maps[k][i][j] > 0)) continue;
        used++;
        const d = dz * net.dw[k * 9 + r * 3 + c];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) want[a][b] += d * img[i + a][j + b];
      }
      assert.ok(used > 0);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) assert.ok(Math.abs(g.filters[k][a][b] - want[a][b]) < 1e-12, `${k} ${a} ${b}`);
    }
  }
  // Nudging a ReLU value that is not its window's max does not change the output.
  const non = S.nonMaxCells(fw);
  assert.equal(non.length, 2 * 36 - 2 * 9);
  for (const [k, i, j] of non) {
    const relu = fw.relu.map((m) => m.map((row) => row.slice()));
    const [ai, aj] = fw.args[k][i >> 1][j >> 1];
    const room = relu[k][ai][aj] - relu[k][i][j];
    if (room <= 1e-9) continue;
    relu[k][i][j] += room / 2;
    assert.deepEqual(relu.map((m) => maxPool(m, 2).out), fw.pooled);
  }
});

test('what the filters became', () => {
  const runs = pageRuns();
  const [f1, f2] = runs.cnn.net.filters;
  // Filter 1: top and bottom rows positive, middle row negative.
  for (const c of [0, 1, 2]) {
    assert.ok(f1[0][c] > 0 && f1[2][c] > 0 && f1[1][c] < 0, String(c));
    assert.ok(f2[c][0] > 0 && f2[c][2] > 0 && f2[c][1] < 0, String(c));
  }
  // The centre weight is the most negative of each.
  assert.equal(Math.min(...f1.flat()), f1[1][1]);
  assert.equal(Math.min(...f2.flat()), f2[1][1]);
  const sv = S.storyValues();
  assert.equal(sv['f1-centre'], S.num(f1[1][1], 2));
  assert.equal(sv['f2-centre'], S.num(f2[1][1], 2));
  // On the shown cross and square: the largest ReLU value of each map.
  const m = S.filterMaps(runs.cnn.net);
  const tr = shapes(200, 201);
  const fc = cnnForward(runs.cnn.net, tr.X[S.CROSS]);
  const fs = cnnForward(runs.cnn.net, tr.X[S.SQUARE]);
  assert.equal(tr.y[S.CROSS], 0);
  assert.equal(tr.y[S.SQUARE], 1);
  // On this cross every value of both maps is 0 after ReLU (each map before ReLU is
  // below −0.1 everywhere), so its 18 features are 0 and its logit is the dense bias.
  for (const k of [0, 1]) {
    assert.equal(m.crossMax[k], 0);
    assert.ok(Math.max(...fc.maps[k].flat()) < -0.1, String(k));
  }
  assert.equal(fc.z, runs.cnn.net.db);
  // All 18 output weights are positive, so any filter response pushes towards "square".
  assert.ok(runs.cnn.net.dw.every((w) => w > 0));
  assert.equal(sv['dw-positive'], '18');
  // Filter 1's side weights in its middle row, and its bias.
  assert.ok(f1[1][0] < 0 && f1[1][2] < 0 && runs.cnn.net.fb[0] < 0);
  assert.equal(sv['f1-side-l'], S.num(f1[1][0], 2));
  assert.equal(sv['f1-side-r'], S.num(f1[1][2], 2));
  assert.equal(sv['f1-bias'], S.num(runs.cnn.net.fb[0], 2));
  for (const k of [0, 1]) {
    assert.equal(m.crossMax[k], Math.max(...fc.relu[k].flat()));
    assert.equal(m.squareMax[k], Math.max(...fs.relu[k].flat()));
    assert.ok(m.squareMax[k] > 3 * m.crossMax[k], String(k));
  }
  for (const k of [0, 1]) {
    assert.equal(sv[`f${k + 1}-cross`], S.num(m.crossMax[k], 2));
    assert.equal(sv[`f${k + 1}-square`], S.num(m.squareMax[k], 2));
  }
  // The square's strongest response is where the filter sits on its whole 3 × 3 outline.
  assert.equal(sv['sq-prob'], S.num(1 / (1 + Math.exp(-fs.z)), 3));
  assert.equal(sv['cr-prob'], S.num(1 / (1 + Math.exp(-fc.z)), 3));
  assert.ok(fs.z > 0 && fc.z < 0);
  for (const k of [0, 1]) {
    const at = m.squareAt[k];
    // The square's centre is at the window's centre: (row + 1, column + 1).
    assert.deepEqual(at, S.shapeCentre(S.SQUARE).map((v) => v - 1));
  }
});

test('the quoted numbers do not hang on rounding', () => {
  // Another JavaScript engine may round Math.exp in the last place. Nudging the
  // first filter weight by 1e-12 stands in for that: the quoted numbers must not move.
  const tr = shapes(200, 201);
  const te = shapes(400, 202);
  const runs = pageRuns();
  const sv = S.storyValues();
  for (const eps of [1e-12, -1e-12]) {
    const t = S.cnnTrainer({ net: nudged(cnnInit({ seed: 203 }), eps), X: tr.X, y: tr.y, batch: 20, kind: 'adam', hyper: { eta: 0.02 }, seed: 204 });
    const accs = [acc(t.state.net, te.X, te.y)];
    for (let e = 1; e <= 30; e++) { t.step(10); accs.push(acc(t.state.net, te.X, te.y)); }
    const L = t.state.epochLoss;
    assert.equal(S.num(L[0], 3), sv['cnn-start-loss']);
    assert.equal(S.num(L[30], 3), sv['cnn-end-loss']);
    assert.equal(S.pct(accs[0]), sv['cnn-start-acc']);
    assert.equal(S.pct(accs[30]), sv['cnn-test']);
    assert.equal(String(S.firstAbove(accs, 0.95)), sv['first-95']);
    assert.equal(S.pct(accs[Number(sv['first-95'])]), sv['first-95-acc']);
    assert.equal(S.pct(accs[Number(sv['first-95']) - 1]), sv['before-95-acc']);
    // Every epoch's read-out (loss to 3 places, test accuracy) is unchanged.
    L.forEach((v, e) => assert.equal(S.num(v, 3), S.num(runs.cnn.losses[e], 3), `loss ${e}`));
    accs.forEach((v, e) => assert.equal(v, runs.cnn.testAcc[e], `acc ${e}`));
    const net = t.state.net;
    assert.equal(S.num(net.filters[0][1][1], 2), sv['f1-centre']);
    assert.equal(S.num(net.filters[1][1][1], 2), sv['f2-centre']);
    assert.equal(S.num(net.filters[0][1][0], 2), sv['f1-side-l']);
    assert.equal(S.num(net.filters[0][1][2], 2), sv['f1-side-r']);
    assert.equal(S.num(net.fb[0], 2), sv['f1-bias']);
    assert.equal(String(net.dw.filter((w) => w > 0).length), sv['dw-positive']);
    const m = S.filterMaps(net);
    for (const k of [0, 1]) {
      assert.equal(S.num(m.crossMax[k], 2), sv[`f${k + 1}-cross`]);
      assert.equal(S.num(m.squareMax[k], 2), sv[`f${k + 1}-square`]);
    }
    assert.equal(S.num(1 / (1 + Math.exp(-cnnForward(net, tr.X[S.SQUARE]).z)), 3), sv['sq-prob']);
    assert.equal(S.num(1 / (1 + Math.exp(-cnnForward(net, tr.X[S.CROSS]).z)), 3), sv['cr-prob']);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      for (const k of [0, 1]) assert.equal(S.num(net.filters[k][a][b], 2), S.num(runs.cnn.net.filters[k][a][b], 2));
    }
    // The dense network: nudge its first weight the same way.
    const dn = mlpInit([64, 18, 1], { hidden: 'relu', init: 'he', seed: 205 });
    dn.W[0][0][0] *= 1 + eps;
    const td = trainer({ net: dn, X: tr.X.map((m2) => m2.flat()), y: tr.y, batch: 20, kind: 'adam', hyper: { eta: 0.02 }, seed: 204 });
    td.step(300);
    assert.equal(S.pct(accuracyNet(td.state.net, te.X.map((m2) => m2.flat()), te.y)), sv['dense-test']);
    assert.equal(S.pct(accuracyNet(td.state.net, tr.X.map((m2) => m2.flat()), tr.y)), sv['dense-train']);
    assert.equal(S.num(td.state.epochLoss[30], 3), sv['dense-end-loss']);
    // The starting network's numbers (steps 2 and 3).
    const sn = nudged(cnnInit({ seed: 203 }), eps);
    const info = S.shiftInfo(sn, tr.X[S.EXAMPLE]);
    assert.equal(S.pctWhole(info.raw), sv['shift-raw']);
    assert.equal(S.pctWhole(info.pooled), sv['shift-pooled']);
    assert.equal(String(info.same), sv['pool-same']);
    for (let k = 0; k < 12; k++) {
      const p = S.shiftInfo(sn, tr.X[k]);
      assert.ok(p.pooled < p.raw, String(k));
    }
    const z = cnnForward(sn, tr.X[S.EXAMPLE]).z;
    assert.equal(S.num(1 / (1 + Math.exp(-z)), 2), sv['start-p']);
  }
});

test('read-out rows', () => {
  const runs = pageRuns();
  const live = S.makeLive();
  const infos = [
    [0, S.gridInfo()],
    ...[0, 1, 5, 11].map((k) => [1, S.layerInfo(k)]),
    ...[0, 1, 5, 11].map((k) => [2, S.poolInfo(k)]),
    [3, S.liveInfo(live, 0, false)],
  ];
  live.advance(37);
  infos.push([3, S.liveInfo(live, 0, true)], [3, S.liveInfo(live, 4, false)]);
  while (live.trainer.state.iter < 300) live.advance(50);
  infos.push([3, S.liveInfo(live, 0, false)]);
  infos.push([4, S.filtersInfo(runs)], [5, S.compareInfo(runs)]);
  infos.push([4, S.filtersInfo(null)], [5, S.compareInfo(null)]);
  for (const [step, s] of infos) {
    const rows = S.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 48, r);
    }
  }
  const sv = S.storyValues();
  const fresh = S.makeLive();
  const start = S.readout(3, S.liveInfo(fresh, 0, false));
  assert.ok(start.some((r) => /press Train/.test(r)), start.join('|'));
  assert.ok(start.some((r) => r.includes(sv['cnn-start-loss'])), start.join('|'));
  assert.ok(start.some((r) => r.includes(sv['cnn-start-acc'])), start.join('|'));
  fresh.advance(37);
  const mid = S.readout(3, S.liveInfo(fresh, 0, false));
  assert.ok(mid.some((r) => /not finished: press Train/.test(r)), mid.join('|'));
  const run = S.readout(3, S.liveInfo(fresh, 0, true));
  assert.ok(run.some((r) => /training…/.test(r)) && !run.some((r) => /press Train/.test(r)), run.join('|'));
  const end = S.readout(3, S.liveInfo(live, 0, false));
  assert.ok(end.some((r) => r.includes('epoch 30 of 30')), end.join('|'));
  assert.ok(end.some((r) => r.includes(sv['cnn-end-loss'])) && end.some((r) => r.includes(sv['cnn-test'])), end.join('|'));
  assert.ok(!end.some((r) => /press Train|training…/.test(r)), end.join('|'));
  const p = S.readout(2, S.poolInfo(S.EXAMPLE));
  assert.ok(p.some((r) => r.includes(sv['shift-raw'])) && p.some((r) => r.includes(sv['shift-pooled'])), p.join('|'));
  const l = S.readout(1, S.layerInfo(S.EXAMPLE));
  assert.ok(l.some((r) => r.includes(`p(square) = ${sv['start-p']}`)), l.join('|'));
  const f = S.readout(4, S.filtersInfo(runs));
  assert.ok(f.some((r) => r.includes(sv['f1-square'])) && f.some((r) => r.includes(sv['f2-cross'])), f.join('|'));
  const c = S.readout(5, S.compareInfo(runs));
  assert.ok(c.some((r) => r.includes(sv['dense-test'])) && c.some((r) => r.includes(sv['cnn-test'])), c.join('|'));
  assert.ok(S.readout(4, S.filtersInfo(null)).some((r) => /computing/.test(r)));
  assert.ok(S.readout(5, S.compareInfo(null)).some((r) => /computing/.test(r)));
});

test('runs computed in slices equal the runs computed at once', async () => {
  const F = await import('../cnns/stage.js?sliced');
  for (const k of F.RUN_KINDS) {
    assert.equal(F.runReady(k), false);
    let slices = 0;
    while (!F.runWork(k, 11)) slices++;
    assert.ok(slices >= 20);
  }
  const a = F.runResult('cnn');
  const b = pageRuns().cnn;
  assert.deepEqual(a.losses, b.losses);
  assert.deepEqual(a.testAcc, b.testAcc);
  assert.deepEqual(F.runResult('dense').losses, pageRuns().dense.losses);
  // Nothing is computed until asked: a fresh copy has no results.
  const G = await import('../cnns/stage.js?fresh');
  assert.equal(G.runReady('cnn'), false);
  assert.equal(G.runReady('dense'), false);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Convolutional networks · ML Explorer<\/title>/);
  assert.match(html, /<h1>Convolutional networks<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/cnns\/">/);
  for (const s of [201, 202, 203, 204, 205]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /cross-correlation \(no flip\)/);
  assert.match(html, /Play steps through the lesson, and Train trains/);
  assert.match(html, /trained the same way/);
  assert.match(html, /0\.15/);
  assert.match(html, /clipped to \[0, 1\]/);
  assert.match(html, /href="\/ml\/convolutions\/"/);
  assert.match(html, /href="\/ml\/optimisers\/"/);
  assert.match(html, /href="\/ml\/train-test\/"/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/rnns\/">Next lesson: Recurrent networks →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /often stays|One reason for it|\+ c = c/);
  assert.match(html, /data-tex="\\hat\{p\} = \\sigma\(z\)"|\\hat\{p\} = \\sigma\(z\)/);
  assert.match(html, /A likely reason, not tested here/);
  assert.doesNotMatch(html, /understands|learns like|it decided|always better|can learn anything/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 4);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const usedPv = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(usedPv.has(key), `unused page value ${key}`);
  const sv = S.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 15);
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('the cross is labelled by the network, not named by it', () => {
  assert.match(html, /the network labels this image a cross because neither filter responds/);
  assert.doesNotMatch(html, /is called a cross/);
});
