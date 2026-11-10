import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { echoData } from '../assets/rnn.js';
import { attnInit, attnForward, attnBackward, attnFlatten, attnUnflatten, attnParamCount } from '../assets/attn.js';
import { optInit, optStep } from '../assets/nn.js';
import * as S from '../attention/stage.js';

const html = readFileSync(new URL('../attention/index.html', import.meta.url), 'utf8');
const LITERAL_TEST_0 = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1];

// The echo task written out again: one uniform draw per bit, 1 below 0.5.
function echo(n, k, seed, T = 12) {
  const rng = mulberry32(seed);
  const seqs = [];
  const targets = [];
  const mask = [];
  for (let i = 0; i < n; i++) {
    const xs = [];
    for (let t = 0; t < T; t++) xs.push(rng() < 0.5 ? 1 : 0);
    seqs.push(xs);
    targets.push(xs.map((_, t) => (t >= k ? xs[t - k] : 0)));
    mask.push(xs.map((_, t) => (t >= k ? 1 : 0)));
  }
  return { seqs, targets, mask };
}

// Softmax written out again (max-subtracted).
const softmaxDirect = (z) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

// Masked accuracy written out again: logit > 0 predicts 1.
function countRight(model, D) {
  let ok = 0;
  let n = 0;
  D.seqs.forEach((xs, i) => {
    attnForward(model, xs).logits.forEach((z, t) => {
      if (!D.mask[i][t]) return;
      n++;
      if ((z > 0 ? 1 : 0) === D.targets[i][t]) ok++;
    });
  });
  return { ok, n };
}

// Adam, η 0.02, batches of 10, shuffle seed 272, written out with attnBackward and no trainer.
function directRun(model0, k, epochs, { perBatch = false } = {}) {
  const tr = echo(200, k, 211);
  const te = echo(200, k, 212);
  let params = attnFlatten(model0);
  let model = attnUnflatten(model0, params);
  let opt = optInit('adam', params.length);
  const rng = mulberry32(272);
  const batch = [countRight(model, te).ok];
  const epoch = [countRight(model, te).ok];
  for (let e = 0; e < epochs; e++) {
    const order = Array.from({ length: 200 }, (_, i) => i);
    for (let i = 199; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let c = 0; c < 200; c += 10) {
      const g = new Float64Array(params.length);
      for (const q of order.slice(c, c + 10)) {
        const b = attnBackward(model, tr.seqs[q], tr.targets[q], tr.mask[q]);
        for (let z = 0; z < g.length; z++) g[z] += b.grads.theta[z] / 10;
      }
      const r = optStep('adam', { eta: 0.02 }, opt, params, Array.from(g));
      params = r.params;
      opt = r.state;
      model = attnUnflatten(model, params);
      if (perBatch) batch.push(countRight(model, te).ok);
    }
    epoch.push(countRight(model, te).ok);
  }
  return { model, batch, epoch, te };
}

// Every starting weight times (1 ± 1e-12), the sign alternating.
const nudged = (model, eps) => attnUnflatten(model, attnFlatten(model).map((v, i) => v * (1 + (i % 2 ? -eps : eps))));

let RUNS = null;
function pageRuns() {
  if (RUNS) return RUNS;
  for (const k of S.K_CHOICES) while (!S.runWork(k, 37));
  RUNS = Object.fromEntries(S.K_CHOICES.map((k) => [k, S.runResult(k)]));
  return RUNS;
}
let LIVE = null;
function liveRun() {
  if (LIVE) return LIVE;
  const live = S.makeLive();
  while (!live.done()) live.advance(1);
  LIVE = live;
  return live;
}

test('data reproducible', () => {
  assert.equal(S.SEED_TRAIN, 211);
  assert.equal(S.SEED_TEST, 212);
  assert.equal(S.SEED_INIT, 271);
  assert.equal(S.SEED_SHUFFLE, 272);
  assert.equal(S.N_TRAIN, 200);
  assert.equal(S.N_TEST, 200);
  assert.equal(S.T, 12);
  assert.equal(S.D, 16);
  assert.equal(S.ETA, 0.02);
  assert.equal(S.BATCH, 10);
  assert.equal(S.EPOCHS, 10);
  assert.equal(S.ITERS, 200);
  assert.equal(S.K_LIVE, 8);
  assert.deepEqual(S.K_CHOICES, [1, 2, 4, 8]);
  for (const k of S.K_CHOICES) {
    assert.deepEqual(S.trainSet(k), echo(200, k, 211));
    assert.deepEqual(S.testSet(k), echo(200, k, 212));
    assert.deepEqual(S.testSet(k), echoData({ n: 200, T: 12, k, seed: 212 }));
  }
  assert.deepEqual(S.testSet(8).seqs[0], LITERAL_TEST_0);
  assert.deepEqual(S.CFG, { vocab: 2, len: 12, d: 16, heads: 1, causal: true, ln: false, mlp: 0, out: 'logit', seed: 271 });
  assert.deepEqual(attnFlatten(S.startModel()), attnFlatten(attnInit(S.CFG)));
  assert.equal(attnParamCount(S.startModel()), 2 * 16 + 12 * 16 + 4 * 16 * 16 + 16 + 1);
  assert.equal(S.rnnParamCount(), 8 + 64 + 8 + 8 + 1);
  assert.equal(S.causalPairs(), 78);
  assert.equal(S.maskedCount(8), 800);
});

test('keys and values reproducible', () => {
  assert.deepEqual(S.KEYS, [[2, 0], [0, 2], [-2, 0], [0, -2]]);
  assert.deepEqual(S.VALUES, [[2, 2], [-2, 2], [-2, -2], [2, -2]]);
  assert.deepEqual(S.Q_START, [1.5, 0.5]);
  assert.equal(S.DK_TOY, 2);
  assert.match(html, /k<\/strong>₁ = \(2, 0\), <strong>k<\/strong>₂ = \(0, 2\), <strong>k<\/strong>₃ = \(−2, 0\), <strong>k<\/strong>₄ = \(0, −2\)/);
  assert.match(html, /v<\/strong>₁ = \(2, 2\), <strong>v<\/strong>₂ = \(−2, 2\), <strong>v<\/strong>₃ = \(−2, −2\), <strong>v<\/strong>₄ = \(2, −2\)/);
  assert.match(html, /starts at \(1\.5, 0\.5\)/);
  // The snap keeps the query on a 0.1 grid inside the square.
  assert.deepEqual(S.snapQuery(1.56, -0.04), [1.6, -0]);
  assert.deepEqual(S.snapQuery(9, -9), [3, -3]);
});

test('attention weights on the page equal softmax of the scores', () => {
  const qs = [S.Q_START, [0, 0], [-3, 3], [3, 3], [0.1, -2.7], ...Array.from({ length: 30 }, (_, i) => S.snapQuery(3 * Math.cos(i), 2.5 * Math.sin(2 * i)))];
  for (const q of qs) {
    const { scores, weights } = S.queryWeights(q);
    const direct = S.KEYS.map((k) => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(2));
    scores.forEach((s, i) => assert.ok(Math.abs(s - direct[i]) < 1e-12));
    const w = softmaxDirect(direct);
    weights.forEach((v, i) => assert.ok(Math.abs(v - w[i]) < 1e-12));
    assert.ok(Math.abs(weights.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    const o = S.attendOutput(weights);
    const od = [0, 1].map((j) => w.reduce((s, a, i) => s + a * S.VALUES[i][j], 0));
    assert.ok(Math.abs(o[0] - od[0]) < 1e-12 && Math.abs(o[1] - od[1]) < 1e-12);
    // Inside the square of the values.
    assert.ok(Math.abs(o[0]) < 2 && Math.abs(o[1]) < 2);
  }
  // The starting query, as quoted.
  const pv = S.pageValues();
  const { scores, weights } = S.queryWeights(S.Q_START);
  assert.equal(pv['q0-s1'], (3 / Math.sqrt(2)).toFixed(2));
  assert.equal(pv['q0-w1'], weights[0].toFixed(2));
  assert.equal(pv['q0-w2'], weights[1].toFixed(2));
  assert.ok(weights[0] === Math.max(...weights) && scores[0] === Math.max(...scores));
  assert.equal(pv['q0-out'], '(1.22, 1.78)');
  // The model's map: each row is the softmax of q_t · k_s / √16 over s ≤ t, and null after t.
  for (const model of [S.startModel(), liveRun().trainer.state.model]) {
    const xs = S.testSet(8).seqs[0];
    const { cache } = attnForward(model, xs);
    const { Q, K } = cache.blocks[0];
    const map = S.attentionMap(model, xs);
    for (let t = 0; t < 12; t++) {
      const sc = [];
      for (let s = 0; s <= t; s++) {
        let v = 0;
        for (let j = 0; j < 16; j++) v += Q[t * 16 + j] * K[s * 16 + j];
        sc.push(v / 4);
      }
      const w = softmaxDirect(sc);
      for (let s = 0; s < 12; s++) {
        if (s > t) assert.equal(map[t][s], null);
        else assert.ok(Math.abs(map[t][s] - w[s]) < 1e-12);
      }
      // The causal weights the model uses are exactly 0.
      for (let s = t + 1; s < 12; s++) assert.equal(attnForward(model, xs).A[0][0][t][s], 0);
    }
  }
});

test('masked accuracy: only masked steps count, and logit 0 predicts 0', () => {
  assert.equal(S.predictBit(0), 0);
  assert.equal(S.predictBit(1e-9), 1);
  const r = S.maskedAccuracy(S.startModel(), S.testSet(8));
  assert.deepEqual(r, countRight(S.startModel(), S.testSet(8)));
  assert.equal(r.n, 800);
});

test('training deterministic', () => {
  const one = S.makeTrainer(8);
  const two = S.makeTrainer(8);
  one.step(200);
  two.step(77);
  two.step(123);
  assert.deepEqual(one.state.epochLoss, two.state.epochLoss);
  assert.deepEqual(attnFlatten(one.state.model), attnFlatten(two.state.model));
  assert.equal(one.state.diverged, false);
  // The live run (one batch at a time) and step 6's k = 8 run are the same run.
  const live = liveRun();
  assert.equal(live.testOk.length, 201);
  assert.deepEqual(attnFlatten(live.trainer.state.model), attnFlatten(one.state.model));
  const runs = pageRuns();
  assert.deepEqual(attnFlatten(runs[8].model), attnFlatten(one.state.model));
  assert.deepEqual(runs[8].testOk, live.testOk.filter((_, i) => i % 20 === 0));
  // A run written out with attnBackward and Adam, without the trainer, gives the same
  // counts (its gradient sums in another order, so weights agree to rounding only).
  for (const k of S.K_CHOICES) {
    const d = directRun(S.startModel(), k, 10, { perBatch: k === 8 });
    assert.deepEqual(d.epoch, runs[k].testOk, `k ${k}`);
    if (k === 8) assert.deepEqual(d.batch, live.testOk);
    const a = attnFlatten(d.model);
    const b = attnFlatten(runs[k].model);
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= 1e-6 * (1 + Math.abs(v)), `k ${k} param ${i}`));
  }
  // Reset repeats the live run.
  const again = S.makeLive();
  again.advance(13);
  again.reset();
  assert.deepEqual(again.testOk, [live.testOk[0]]);
  while (!again.done()) again.advance(7);
  assert.deepEqual(again.testOk, live.testOk);
});

test('k = 8 reaches masked test accuracy ≥ 0.95 within 10 epochs', () => {
  const live = liveRun();
  const ok = live.testOk;
  const n = 800;
  assert.equal(live.n, n);
  assert.ok(ok.at(-1) / n >= 0.95);
  assert.equal(ok.at(-1), n);
  assert.ok(ok[0] / n > 0.45 && ok[0] / n < 0.55, String(ok[0]));
  const first = S.firstAt(ok, n, 0.95);
  for (let i = first; i < ok.length; i++) assert.ok(ok[i] / n >= 0.95, String(i));
  const allRight = S.allRightFrom(ok, n);
  assert.ok(allRight > first);
  for (let i = allRight; i < ok.length; i++) assert.equal(ok[i], n);
  const sv = S.storyValues();
  assert.equal(sv['start-acc'], S.pctCount(ok[0], n));
  assert.equal(sv['start-acc'], '49%');
  assert.equal(sv['first-95'], String(first));
  assert.equal(sv['first-95-acc'], S.pctCount(ok[first], n));
  assert.equal(sv['all-right'], String(allRight));
  assert.equal(sv['end-acc'], '100%');
  // "The first two epochs are shown slowly": the band and the rise happen inside them.
  assert.ok(allRight <= S.SLOW_EPOCHS * S.BATCHES_PER_EPOCH);
});

test("the trained map's largest weight in each row with a target is at lag k", () => {
  const sv = S.storyValues();
  const live = liveRun();
  const te = S.testSet(8);
  const b = S.bandStats(live.trainer.state.model, te, 8);
  assert.equal(b.n, 800);
  assert.equal(b.hit, 800);
  assert.equal(sv['band-share'], '100%');
  assert.equal(sv['band-mean'], b.mean.toFixed(2));
  assert.ok(b.mean > 0.9);
  // Written out again: argmax of each target row of every test stream.
  let hit = 0;
  for (const xs of te.seqs) {
    const A = attnForward(live.trainer.state.model, xs).A[0][0];
    for (let t = 8; t < 12; t++) {
      const row = A[t].slice(0, t + 1);
      if (row.indexOf(Math.max(...row)) === t - 8) hit++;
    }
  }
  assert.equal(hit, 800);
  // The starting model: spread out on test stream 1 (no row with a target above 0.5),
  // and rarely largest at lag 8.
  const start = S.attentionMap(S.startModel(), te.seqs[0]);
  for (let t = 8; t < 12; t++) assert.ok(Math.max(...start[t].filter((v) => v !== null)) < 0.5, String(t));
  assert.equal(start[0][0], 1);
  const b0 = S.bandStats(S.startModel(), te, 8);
  assert.equal(sv['start-band'], S.pctCount(b0.hit, b0.n));
  assert.ok(b0.hit / b0.n < 0.3);
  // For every k the band is at lag k (the page: "the dots sit k cells left of the diagonal").
  const runs = pageRuns();
  for (const k of S.K_CHOICES) {
    const bk = S.bandStats(runs[k].model, S.testSet(k), k);
    assert.equal(bk.hit, bk.n, `k ${k}`);
  }
});

test('every k in {1,2,4,8} reaches ≥ 0.95', () => {
  const runs = pageRuns();
  const sv = S.storyValues();
  let last = 0;
  for (const k of S.K_CHOICES) {
    const r = runs[k];
    assert.equal(r.n, 200 * (12 - k));
    assert.equal(r.diverged, false);
    assert.ok(r.testOk[10] / r.n >= 0.95, `k ${k}`);
    assert.equal(r.testOk[10], r.n);
    assert.equal(sv[`acc-k${k}`], '100%');
    const from = S.allRightFrom(r.testOk, r.n);
    for (let e = from; e <= 10; e++) assert.equal(r.testOk[e], r.n);
    last = Math.max(last, from);
  }
  assert.equal(sv['all-k-epoch'], String(last));
  // Without position embeddings the trained k = 8 model is near chance.
  const np = S.maskedAccuracy(S.withoutPositions(runs[8].model), S.testSet(8));
  assert.equal(sv['nopos-acc'], S.pctCount(np.ok, np.n));
  assert.ok(np.ok / np.n < 0.6);
  // withoutPositions leaves the model it was given unchanged.
  assert.ok(runs[8].model.P.some((v) => v !== 0));
});

test('step 1: the recurrent network fell short at k = 8 within its budget', async () => {
  const R = await import('../rnns/stage.js');
  const r8 = R.runResult(8);
  const r2 = R.runResult(2);
  assert.equal(r2.testOk[10], r2.n);
  assert.ok(r8.testOk[10] / r8.n < 0.95);
  assert.match(html, /got every k = 2 test step right but fell short at k = 8/);
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const sv = S.storyValues();
  const runs = pageRuns();
  const live = liveRun();
  for (const eps of [1e-12, -1e-12]) {
    const m0 = nudged(S.startModel(), eps);
    // Every batch's test count of the k = 8 live run, and every shown loss.
    const lv = S.makeTrainer(8, m0);
    const te = S.testSet(8);
    const per = [S.maskedAccuracy(lv.state.model, te).ok];
    for (let i = 0; i < 200; i++) { lv.step(1); per.push(S.maskedAccuracy(lv.state.model, te).ok); }
    assert.deepEqual(per, live.testOk);
    lv.state.epochLoss.forEach((v, e) => assert.equal(S.lossText(v), S.lossText(live.trainer.state.epochLoss[e]), `loss ${e}`));
    const b = S.bandStats(lv.state.model, te, 8);
    assert.equal(S.pctCount(b.hit, b.n), sv['band-share']);
    assert.equal(b.mean.toFixed(2), sv['band-mean']);
    const np = S.maskedAccuracy(S.withoutPositions(lv.state.model), te);
    assert.equal(S.pctCount(np.ok, np.n), sv['nopos-acc']);
    const b0 = S.bandStats(m0, te, 8);
    assert.equal(S.pctCount(b0.hit, b0.n), sv['start-band']);
    for (const k of S.K_CHOICES) {
      if (k === 8) continue;
      const t = S.makeTrainer(k, m0);
      const ep = [S.maskedAccuracy(t.state.model, S.testSet(k)).ok];
      for (let e = 0; e < 10; e++) { t.step(20); ep.push(S.maskedAccuracy(t.state.model, S.testSet(k)).ok); }
      assert.deepEqual(ep, runs[k].testOk, `k ${k}`);
      const bk = S.bandStats(t.state.model, S.testSet(k), k);
      assert.equal(bk.hit, bk.n, `k ${k}`);
    }
  }
});

test('live pacing: first two epochs slow, then one batch per frame, same run', () => {
  const live = S.makeLive();
  let waited = 0;
  let frames = 0;
  const takenAt = [];
  while (!live.done()) {
    const r = S.paceFrame(live.trainer.state.iter, waited);
    waited = r.waited;
    if (r.take) { live.advance(r.take); takenAt.push(frames); }
    frames++;
  }
  assert.equal(takenAt[0], 0);
  for (let i = 1; i < 40; i++) assert.equal(takenAt[i] - takenAt[i - 1], 6, String(i));
  for (let i = 41; i < 200; i++) assert.equal(takenAt[i] - takenAt[i - 1], 1, String(i));
  assert.deepEqual(live.testOk, liveRun().testOk);
});

test('runs computed in slices equal the runs computed at once', async () => {
  const F = await import('../attention/stage.js?sliced');
  for (const k of F.K_CHOICES) {
    assert.equal(F.runReady(k), false);
    let slices = 0;
    while (!F.runWork(k, 3)) slices++;
    assert.ok(slices >= 60);
  }
  for (const k of F.K_CHOICES) {
    assert.deepEqual(F.runResult(k).testOk, pageRuns()[k].testOk);
    assert.deepEqual(F.runResult(k).losses, pageRuns()[k].losses);
  }
  const G = await import('../attention/stage.js?fresh');
  for (const k of G.K_CHOICES) assert.equal(G.runReady(k), false);
});

test('read-out rows', () => {
  const runs = pageRuns();
  const fresh = S.makeLive();
  const toy = (q) => { const w = S.queryWeights(q); return { q, ...w, out: S.attendOutput(w.weights) }; };
  const infos = [[0, null], [1, toy(S.Q_START)], [1, toy([-3, -3])], [2, toy(S.Q_START)], [2, toy([0, 0])],
    [3, { band: S.bandStats(S.startModel(), S.testSet(8), 8) }], [4, S.liveInfo(fresh, false)]];
  fresh.advance(9);
  infos.push([4, S.liveInfo(fresh, true)], [4, S.liveInfo(fresh, 'computing')], [4, S.liveInfo(liveRun(), false)], [5, null]);
  for (const k of S.K_CHOICES) {
    const r = runs[k];
    infos.push([5, { k, ok: r.testOk[10], n: r.n, band: S.bandStats(r.model, S.testSet(k), k) }]);
  }
  for (const [step, s] of infos) {
    const rows = S.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 48, r);
    }
  }
  const sv = S.storyValues();
  assert.ok(S.readout(1, toy(S.Q_START)).some((r) => r.includes('weights add up to 1.00')));
  assert.ok(S.readout(3, infos[5][1]).some((r) => r.includes(sv['start-band'])));
  const start = S.readout(4, S.liveInfo(S.makeLive(), false));
  assert.ok(start.some((r) => /press Train/.test(r)) && start.some((r) => r.includes(sv['start-acc'])));
  assert.ok(S.readout(4, S.liveInfo(fresh, false)).some((r) => /not finished: press Train/.test(r)));
  assert.ok(S.readout(4, S.liveInfo(fresh, 'computing')).some((r) => /computing…/.test(r)));
  const end = S.readout(4, S.liveInfo(liveRun(), false));
  assert.ok(end.some((r) => r.includes('batch 200 of 200 · finished')) && end.some((r) => r.includes(sv['end-acc'])));
  assert.ok(S.readout(5, null).some((r) => /computing/.test(r)));
  // Step 5's text alternative says where the weight goes on the shown stream.
  const xs = S.testSet(8).seqs[S.EXAMPLE];
  const one = S.bandStats(liveRun().trainer.state.model, { seqs: [xs] }, 8);
  assert.deepEqual(S.streamBand(liveRun().trainer.state.model, xs, 8), { hit: one.hit, n: one.n });
  assert.ok(end.some((r) => r.includes('stream 1: largest 8 back in 4 of 4 rows')), end.join('|'));
  const b0 = S.streamBand(S.startModel(), xs, 8);
  assert.ok(start.some((r) => r.includes(`in ${b0.hit} of 4 rows`)), start.join('|'));
  assert.ok(end.some((r) => r.includes(S.lossText(liveRun().trainer.state.epochLoss[10]))), end.join('|'));
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Attention · ML Explorer<\/title>/);
  assert.match(html, /<h1>Attention<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/attention\/">/);
  for (const s of [211, 212, 271, 272]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /Play steps through the lesson, and Train trains/);
  assert.match(html, /far larger scale/);
  assert.match(html, /position embeddings/);
  assert.match(html, /square of the sequence length/);
  assert.match(html, /variance <span class="tex" data-tex="d_k"><\/span>|variance 1/);
  assert.match(html, /independent, each with mean 0 and variance 1/);
  assert.match(html, /The independence is a simplification/);
  assert.match(html, /\\operatorname\{Attention\}\(\\mathbf\{Q\}, \\mathbf\{K\}, \\mathbf\{V\}\)/);
  assert.match(html, /soft lookup/);
  // Wording fixed in review.
  assert.match(html, /Softmax, from <a href="\/ml\/embeddings\/">embeddings<\/a>/);
  assert.doesNotMatch(html, /not a contest of size/);
  // Only k = 1, 2, 4 and 8 are run: no phrase may claim the whole range 1 to 8.
  assert.doesNotMatch(html.replace(/<[^>]*>/g, ''), /from 1 to 8|every k from|1–8|k = 1 to 8/);
  assert.match(html, /k = 1, 2, 4 and 8 all end at/);
  assert.match(html, /so this is not a like-for-like contest/);
  assert.match(html, /in the rows with a target, the dots sit k cells left of the diagonal/);
  assert.match(html, /Suppose all the <span class="tex" data-tex="2 d_k">/);
  assert.match(html, /roughly unit variance/);
  assert.match(html, /all four keys here have the same length/);
  assert.match(html, /exactly 0/);
  for (const href of ['/ml/rnns/', '/ml/language-models/', '/ml/optimisers/', '/ml/logistic-regression/']) assert.match(html, new RegExp(`href="${href}"`));
  assert.match(html, /<a class="next-lesson" href="\/ml\/transformers\/">Next lesson: Transformers →<\/a>/);
  assert.match(html, /1706\.03762/);
  assert.match(html, /1409\.0473/);
  assert.match(html, /Chapter 7, "Transformers and Pretraining"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understand|thinks|\bknows\b|reasons|creative|intelligen|conscious|hallucinat|decides|chooses/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(html, /GPT|BERT|Llama|Gemini/);
  // No number from the recurrent lesson is quoted; its claim is pinned by the test above.
  assert.doesNotMatch(html, /88\.9|711/);
  // The parameter count written out in step 4's maths is the model's.
  assert.match(html, /data-tex="2 \\times 16 \+ 12 \\times 16 \+ 4 \\times 16 \\times 16 \+ 16 \+ 1 = 1265"/);
  assert.equal(attnParamCount(S.startModel()), 1265);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What do the attention weights in one row add up to\?/);
  assert.match(html, /Why can attention handle k = 8 as easily as k = 2 here\?/);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const usedPv = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(usedPv.has(key), `unused page value ${key}`);
  const sv = S.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
