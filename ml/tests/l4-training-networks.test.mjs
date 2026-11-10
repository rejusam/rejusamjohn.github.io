import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { mlpInit, mlpForward, trainer, accuracyNet, bceLogit, batchGrad, flatten, optInit, optStep, unflatten } from '../assets/nn.js';
import * as T from '../training-networks/stage.js';

const html = readFileSync(new URL('../training-networks/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_X0 = 0.11783946381590707;

const meanLoss = (net, D) => D.X.reduce((s, x, i) => s + bceLogit(D.y[i], mlpForward(net, x).out[0]), 0) / D.X.length;

// The big network's run, written out from nn.js directly, with the test loss
// (from mlpForward) after every 10th epoch; null at the others.
function directBig(lambda) {
  const sm = T.smallSet();
  const te = T.testSet();
  const net = mlpInit([2, 64, 64, 1], { hidden: 'tanh', init: 'xavier', seed: 173 });
  const t = trainer({ net, X: sm.X, y: sm.y, batch: 20, kind: 'adam', hyper: { eta: 0.01 }, seed: 174, lambda });
  const testL = [meanLoss(net, te)];
  for (let e = 1; e <= 400; e++) {
    t.step(3);
    testL.push(e % 10 === 0 ? meanLoss(t.state.net, te) : null);
  }
  return { t, testL };
}

let BIG0 = null;
let BIGW = null;
const big0 = () => (BIG0 ??= directBig(0));
const bigW = () => (BIGW ??= directBig(T.LAMBDA));

test('data reproducible', () => {
  assert.equal(T.SEED_TRAIN, 171);
  assert.equal(T.SEED_TEST, 172);
  assert.equal(T.SEED_INIT, 173);
  assert.equal(T.SEED_SHUFFLE, 174);
  assert.equal(T.SEED_SMALL, 175);
  // Two spiral arms, written out: point i of arm c (i = 0, …, n/2 − 1) at
  // t = i/(n/2), radius 0.3 + 2.7t, angle 3πt + πc, plus 0.2 × noise on each coordinate.
  const spiral = (n, seed) => {
    const rng = mulberry32(seed);
    const X = [];
    const y = [];
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < n / 2; i++) {
        const t = i / (n / 2);
        const r = 0.3 + 2.7 * t;
        const th = 3 * Math.PI * t + Math.PI * c;
        X.push([r * Math.cos(th) + 0.2 * randn(rng), r * Math.sin(th) + 0.2 * randn(rng)]);
        y.push(c);
      }
    }
    return { X, y };
  };
  assert.deepEqual(T.trainSet(), spiral(160, 171));
  assert.deepEqual(T.testSet(), spiral(400, 172));
  assert.deepEqual(T.smallSet(), spiral(60, 175));
  assert.deepEqual(T.trainSet(), T.trainSet());
  assert.equal(T.trainSet().X[0][0], LITERAL_TRAIN_X0);
  // Every point fits inside the plot.
  for (const { X } of [T.trainSet(), T.testSet(), T.smallSet()]) {
    for (const x of X) assert.ok(Math.abs(x[0]) < T.DOM[1] && Math.abs(x[1]) < T.DOM[1]);
  }
  // The networks as stated: tanh hidden layers, Xavier start, seed 173.
  assert.deepEqual(T.startNet(), mlpInit([2, 16, 16, 1], { hidden: 'tanh', init: 'xavier', seed: 173 }));
  assert.deepEqual(T.bigStartNet(), mlpInit([2, 64, 64, 1], { hidden: 'tanh', init: 'xavier', seed: 173 }));
  // The batch sizes all divide the training set, so every epoch has equal batches.
  assert.deepEqual(T.BATCHES, [1, 8, 160]);
  for (const b of [...T.BATCHES, T.BATCH]) assert.equal(T.N_TRAIN % b, 0);
  assert.equal(T.N_SMALL % T.BIG_BATCH, 0);
  assert.equal(T.ITERS_PER_EPOCH, 10);
  // Weights and biases: 2·16 + 16 + 16·16 + 16 + 16 + 1, and the same for 64 units.
  assert.equal(T.paramCount(T.SIZES), 337);
  assert.equal(T.paramCount(T.BIG_SIZES), 2 * 64 + 64 + 64 * 64 + 64 + 64 + 1);
  assert.equal(T.pageValues()['big-params'], '4417');
  assert.equal(T.BIG_ITERS, 3);
});

test('fast logit equals mlpForward', () => {
  const nets = [T.startNet(), T.bigStartNet()];
  const t = T.makeRun();
  T.advance(t, 200);
  nets.push(t.state.net);
  const te = T.testSet();
  for (const net of nets) {
    const f = T.makeLogit(net);
    for (const x of te.X.slice(0, 50)) assert.ok(Math.abs(f(x[0], x[1]) - mlpForward(net, x).out[0]) < 1e-12);
    assert.ok(Math.abs(T.meanLossFast(net, te) - meanLoss(net, te)) < 1e-12);
  }
});

test('batch size 1 curve is noisier than full batch', () => {
  // Each run sees the same 1600 examples (10 epochs); the curve is the loss on
  // each step's batch, measured just before the step, at x = examples seen.
  assert.equal(T.NOISE_EXAMPLES, 1600);
  const runs = T.BATCHES.map((b) => T.noiseRun(b));
  runs.forEach((r, k) => {
    const b = T.BATCHES[k];
    assert.equal(r.steps, 1600 / b);
    assert.equal(r.ys.length, 1600 / b);
    assert.equal(r.xs.at(-1), 1600);
    assert.deepEqual(r.xs, r.ys.map((_, i) => (i + 1) * b));
    // The same run as nn.js's trainer, Adam η = 0.01, shuffle seed 174.
    const d = trainer({ net: T.startNet(), X: T.trainSet().X, y: T.trainSet().y, batch: b, kind: 'adam', hyper: { eta: 0.01 }, seed: 174 });
    d.step(1600 / b);
    assert.deepEqual(r.ys, d.state.batchLoss);
    assert.equal(r.end, d.state.epochLoss[10]);
    let s = 0;
    for (let i = 1; i < r.ys.length; i++) s += Math.abs(r.ys[i] - r.ys[i - 1]);
    assert.equal(r.noise, s / (r.ys.length - 1));
    for (const v of r.ys) assert.ok(v > 0 && v < T.NOISE_MAX, String(v));
  });
  // Story: mean |Δ loss| between consecutive steps falls as the batch grows.
  assert.ok(runs[0].noise > runs[1].noise && runs[1].noise > runs[2].noise);
  assert.ok(runs[0].noise > 10 * runs[2].noise);
  // The full batch's batch loss is the training loss at the weights before each step.
  const full = runs[2];
  const d = trainer({ net: T.startNet(), X: T.trainSet().X, y: T.trainSet().y, batch: 160, kind: 'adam', hyper: { eta: 0.01 }, seed: 174 });
  d.step(10);
  full.ys.forEach((v, i) => assert.ok(Math.abs(v - d.state.epochLoss[i]) < 1e-12));
  const sv = T.storyValues();
  assert.deepEqual([sv['noise-1'], sv['noise-8'], sv['noise-160']], runs.map((r) => T.sig3(r.noise)));
  assert.deepEqual([sv['end-1'], sv['end-8'], sv['end-160']], runs.map((r) => T.num(r.end, 3)));
  // "After 10 epochs the full batch is not lower than batch size 8."
  assert.ok(runs[2].end > runs[1].end);
});

test('on average the batch gradients equal the full gradient', () => {
  // At fixed weights, the mean of an epoch's equal-size batch gradients is the full-data gradient.
  const tr = T.trainSet();
  const net = T.startNet();
  const order = T.epochOrder(0);
  assert.deepEqual([...order].sort((a, b) => a - b), Array.from({ length: 160 }, (_, i) => i));
  const full = flatten({ ...net, ...(({ dW, db }) => ({ W: dW, b: db }))(batchGrad(net, tr.X, tr.y, order)) });
  for (const b of [1, 8, 16]) {
    const sum = new Array(full.length).fill(0);
    for (let k = 0; k < 160; k += b) {
      const g = batchGrad(net, tr.X, tr.y, order.slice(k, k + b));
      flatten({ ...net, W: g.dW, b: g.db }).forEach((v, i) => { sum[i] += v; });
    }
    sum.forEach((v, i) => assert.ok(Math.abs(v / (160 / b) - full[i]) < 1e-12, `${b} ${i}`));
  }
});

test('the batch shown is the batch the trainer used', () => {
  const tr = T.trainSet();
  const t = T.makeRun();
  // Replay the first 25 steps (two and a half epochs) by hand with epochOrder.
  let params = flatten(T.startNet());
  let opt = optInit('adam', params.length);
  let net = T.startNet();
  for (let k = 1; k <= 25; k++) {
    const idx = T.batchAt(k);
    assert.equal(idx.length, 16);
    const g = batchGrad(net, tr.X, tr.y, idx);
    const r = optStep('adam', { eta: 0.01 }, opt, params, flatten({ ...net, W: g.dW, b: g.db }));
    params = r.params;
    opt = r.state;
    net = unflatten(net, params);
    T.advance(t, 1);
    assert.deepEqual(flatten(t.state.net), params, String(k));
  }
  // Steps 1 to 10 cover every point once: one epoch.
  const seen = new Set();
  for (let k = 1; k <= 10; k++) for (const i of T.batchAt(k)) seen.add(i);
  assert.equal(seen.size, 160);
    assert.deepEqual(T.epochInfo(25), { epoch: 2, batch: 5 });
  assert.deepEqual(T.epochInfo(30), { epoch: 3, batch: 10 });
  assert.deepEqual(T.epochInfo(31), { epoch: 3, batch: 1 });
  assert.deepEqual(T.epochInfo(0), { epoch: 0, batch: 0 });
});

test('spiral net reaches train accuracy ≥ 0.95', () => {
  const tr = T.trainSet();
  const te = T.testSet();
  assert.equal(T.EPOCHS, 300);
  assert.equal(T.ETA, 0.01);
  assert.equal(T.BATCH, 16);
  const t = T.makeRun();
  while (!T.finished(t)) T.advance(t, 37);
  assert.equal(t.state.iter, 3000);
  assert.equal(t.state.epoch, 300);
  const acc = accuracyNet(t.state.net, tr.X, tr.y);
  assert.ok(acc >= 0.95, String(acc));
  const d = trainer({ net: T.startNet(), X: tr.X, y: tr.y, batch: 16, kind: 'adam', hyper: { eta: 0.01 }, seed: 174 });
  d.step(3000);
  assert.deepEqual(d.state.epochLoss, t.state.epochLoss);
  const sv = T.storyValues();
  assert.equal(sv['sp-train'], T.pct(acc));
  assert.equal(sv['sp-test'], T.pct(accuracyNet(t.state.net, te.X, te.y)));
  assert.equal(sv['sp-loss0'], T.num(t.state.epochLoss[0], 2));
  assert.equal(sv['sp-loss'], T.num(t.state.epochLoss[300], 3));
  for (const v of t.state.epochLoss) assert.ok(v > 0 && v < T.LOSS_MAX);
  // The read-out at the end shows the prose numbers.
  const rows = T.readout(2, T.spiralInfo(t));
  assert.ok(rows.includes(`train: ${sv['sp-train']}`));
  assert.ok(rows.includes(`test: ${sv['sp-test']}`));
  assert.ok(rows.includes(`train loss: ${sv['sp-loss']}`));
});

test('big net: test loss argmin is before the last epoch and test loss at the end is ≥ 10% above its min', () => {
  const { t, testL } = big0();
  const b = T.makeBig(0);
  while (!T.bigFinished(b)) T.bigAdvance(b, 7);
  assert.equal(b.t.state.epoch, 400);
  assert.equal(b.test.length, 401);
  testL.forEach((v, i) => { if (v !== null) assert.ok(Math.abs(v - b.test[i]) < 1e-12, String(i)); });
  assert.deepEqual(b.t.state.epochLoss, t.state.epochLoss);
  const best = T.earlyStop(b.test);
  assert.ok(best.epoch < 400);
  assert.ok(b.test[400] >= 1.1 * best.value, `${b.test[400]} ${best.value}`);
  // The training loss carries on down past the best test epoch.
  const tl = b.t.state.epochLoss;
  assert.ok(tl[400] < tl[best.epoch]);
  assert.ok(tl[400] < 0.1 * tl[best.epoch]);
  // The test loss wobbles on the way down: it rises between epochs before its lowest point.
  let ups = 0;
  for (let e = 1; e <= best.epoch; e++) if (b.test[e] > b.test[e - 1]) ups++;
  assert.ok(ups > 0);
  const sv = T.storyValues();
  assert.equal(sv['big-best-epoch'], String(best.epoch));
  assert.equal(sv['big-best'], T.num(best.value, 3));
  assert.equal(sv['big-end'], T.num(b.test[400], 3));
  assert.equal(sv['big-rise'], `${Math.round((b.test[400] / best.value - 1) * 100)}%`);
  assert.equal(sv['big-train-best'], T.num(tl[best.epoch], 3));
  assert.equal(sv['big-train-end'], T.num(tl[400], 3));
  assert.equal(sv['big-ups'], String(ups));
  assert.equal(sv['big-train-acc'], T.pct(accuracyNet(b.t.state.net, T.smallSet().X, T.smallSet().y)));
  for (const v of [...b.test, ...tl]) assert.ok(v > 0 && v < T.LOSS_MAX, String(v));
  // The read-out at the end agrees.
  const rows = T.readout(3, T.bigInfo(b));
  assert.ok(rows.includes(`test loss: ${sv['big-end']}`), rows.join('|'));
  assert.ok(rows.includes(`best test: ${sv['big-best']} @ ${sv['big-best-epoch']}`), rows.join('|'));
});

test('weight decay lowers final test loss', () => {
  assert.equal(T.LAMBDA, 0.001);
  const w = bigW();
  const b = T.makeBig(T.LAMBDA);
  while (!T.bigFinished(b)) T.bigAdvance(b, 11);
  w.testL.forEach((v, i) => { if (v !== null) assert.ok(Math.abs(v - b.test[i]) < 1e-12, String(i)); });
  assert.deepEqual(b.t.state.epochLoss, w.t.state.epochLoss);
  const a = T.makeBig(0);
  while (!T.bigFinished(a)) T.bigAdvance(a, 50);
  const plain = a.test;
  assert.ok(b.test[400] < plain[400]);
  // "Flatter": it ends much closer to its own lowest value than the run without weight decay.
  const bestW = T.earlyStop(b.test);
  const best0 = T.earlyStop(plain);
  const riseW = b.test[400] / bestW.value - 1;
  const rise0 = plain[400] / best0.value - 1;
  assert.ok(riseW < 0.1 && riseW < rise0 / 3, `${riseW} ${rise0}`);
  // Its lowest test loss is lower than the lowest without, too.
  assert.ok(bestW.value < best0.value);
  // The curves show the data loss, without the penalty: the trainer's epoch loss.
  const tl = b.t.state.epochLoss;
  assert.ok(Math.abs(tl[400] - meanLoss(b.t.state.net, T.smallSet())) < 1e-12);
  const sv = T.storyValues();
  assert.equal(sv['wd-end'], T.num(b.test[400], 3));
  assert.equal(sv['wd-rise'], `${Math.round(riseW * 100)}%`);
  assert.equal(sv['wd-best'], T.num(bestW.value, 3));
  assert.equal(sv['wd-train-end'], T.num(tl[400], 3));
  assert.equal(sv['wd-train-acc'], T.pct(accuracyNet(b.t.state.net, T.smallSet().X, T.smallSet().y)));
  assert.equal(sv['big-test-acc'], T.pct(accuracyNet(big0().t.state.net, T.testSet().X, T.testSet().y)));
  assert.equal(sv['wd-test-acc'], T.pct(accuracyNet(b.t.state.net, T.testSet().X, T.testSet().y)));
  for (const v of [...b.test, ...tl]) assert.ok(v > 0 && v < T.LOSS_MAX, String(v));
  // The read-out shows both final test losses once both runs are done.
  const rows = T.readout(4, T.wdInfo(b, { 0: a, [T.LAMBDA]: b }));
  assert.ok(rows.includes(`λ = 0: ${sv['big-end']}`), rows.join('|'));
  assert.ok(rows.includes(`λ = 0.001: ${sv['wd-end']}`), rows.join('|'));
});

test('runs are deterministic, paced and reset', () => {
  const a = T.makeRun();
  const b = T.makeRun();
  T.advance(a, 400);
  for (let k = 0; k < 40; k++) T.advance(b, 10);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
  a.reset();
  assert.equal(a.state.iter, 0);
  assert.deepEqual(a.state.net, T.startNet());
  // Never past the end.
  T.advance(b, 1e6);
  assert.equal(b.state.iter, T.MAX_ITER);
  // Step 2 is slow: one optimiser step every SLOW_FRAMES frames.
  const c = T.makeRun();
  const pace = T.slowPacer();
  for (let f = 0; f < T.SLOW_FRAMES * 3; f++) pace(c, 50);
  assert.equal(c.state.iter, 3);
  const d = T.makeRun();
  T.frameStep(d, 100);
  assert.ok(d.state.iter >= 1 && d.state.iter <= T.paceAt(0));
  // The big run resets to its seeded start and repeats exactly.
  const g = T.makeBig(0);
  T.bigAdvance(g, 30);
  const first = g.test.slice();
  g.reset();
  assert.equal(g.t.state.iter, 0);
  assert.equal(g.test.length, 1);
  T.bigAdvance(g, 30);
  assert.deepEqual(g.test, first);
  assert.equal(first.length, 11);
});

test('training-networks read-out rows and page', () => {
  const t = T.makeRun();
  const g = T.makeBig(0);
  const w = T.makeBig(T.LAMBDA);
  for (let k = 0; k < 3; k++) {
    for (const b of T.BATCHES) {
      const rows = T.readout(0, T.noiseInfo(b));
      assert.ok(rows.length >= 1 && rows.length <= 6);
      for (const r of rows) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 26, r);
      }
    }
    for (const step of [1, 2]) {
      const rows = T.readout(step, T.spiralInfo(t));
      for (const r of rows) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 26, r);
      }
      assert.ok(rows.length <= 6);
    }
    for (const rows of [T.readout(3, T.bigInfo(g)), T.readout(4, T.wdInfo(w, { 0: g, [T.LAMBDA]: w })), T.readout(4, T.wdInfo(g, { 0: g }))]) {
      assert.ok(rows.length <= 6);
      for (const r of rows) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 26, r);
      }
    }
    T.advance(t, 1500);
    T.bigAdvance(g, 600);
    T.bigAdvance(w, 600);
  }
  // Steps whose prose gives end-of-run numbers say when the run is not finished.
  const fresh = T.makeRun();
  assert.ok(T.readout(2, T.spiralInfo(fresh)).some((r) => /not finished/.test(r)));
  assert.ok(!T.readout(2, T.spiralInfo(t)).some((r) => /not finished/.test(r)));
  const freshBig = T.makeBig(0);
  assert.ok(T.readout(3, T.bigInfo(freshBig)).some((r) => /not finished/.test(r)));
  assert.ok(!T.readout(3, T.bigInfo(g)).some((r) => /not finished/.test(r)));
  assert.ok(T.readout(4, T.wdInfo(freshBig, { 0: freshBig })).some((r) => /not finished/.test(r)));
  // Step 2's read-out counts steps and epochs.
  const e = T.makeRun();
  T.advance(e, 25);
  const rows = T.readout(1, T.spiralInfo(e));
  assert.ok(rows.includes('steps: 25'), rows.join('|'));
  assert.ok(rows.includes('epochs done: 2'), rows.join('|'));
  assert.ok(rows.includes('batch 5 of 10 done'), rows.join('|'));
  // Step 1's read-out numbers are the noise runs' numbers.
  const sv = T.storyValues();
  assert.ok(T.readout(0, T.noiseInfo(1)).includes(`mean |change|: ${sv['noise-1']}`));

  // The page.
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Training a network · ML Explorer<\/title>/);
  assert.match(html, /<h1>Training a network<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  for (const s of [171, 172, 173, 174, 175]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /without the penalty/);
  assert.match(html, /Play steps through the lesson, and Train trains the network/);
  assert.match(html, /the end-of-run numbers here are for the full 300-epoch run/);
  assert.match(html, /the end-of-run numbers here are for the full 400-epoch run/);
  assert.match(html, /on average across the epoch's batches it equals the full gradient/);
  assert.match(html, /href="\/ml\/train-test\/"/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  assert.match(html, /href="\/ml\/backpropagation\/"/);
  assert.match(html, /href="\/ml\/gradient-descent\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/optimisers\/">Next lesson: Optimisers: momentum and Adam →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always better|can learn anything/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  // Values filled at run time are already in the page and equal the computed ones.
  const pv = T.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 8);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 15);
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
