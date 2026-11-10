import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { echoData, rnnInit, rnnForward, bptt, rnnFlatten, rnnUnflatten } from '../assets/rnn.js';
import { optInit, optStep } from '../assets/nn.js';
import * as S from '../rnns/stage.js';

const html = readFileSync(new URL('../rnns/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_0 = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1];

// The echo task written out again: one uniform draw per bit, 1 below 0.5;
// the target at zero-based step t is the bit at t − k, where t ≥ k.
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

// Masked accuracy written out again: z > 0 predicts 1, otherwise 0 (p̂ = 0.5 gives 0).
function countRight(net, D) {
  let ok = 0;
  let n = 0;
  D.seqs.forEach((xs, i) => {
    const { zs } = rnnForward(net, xs);
    zs.forEach((z, t) => {
      if (!D.mask[i][t]) return;
      n++;
      if ((z > 0 ? 1 : 0) === D.targets[i][t]) ok++;
    });
  });
  return { ok, n };
}

const meanLoss = (net, D) => D.seqs.reduce((s, xs, i) => s + bptt(net, xs, D.targets[i], D.mask[i]).loss, 0) / D.seqs.length;

// Adam, η 0.02, batches of 10 sequences, shuffle seed 214, written out without the trainer.
// Returns the epoch losses, the test counts after every batch and after every epoch.
function directRun(net0, k, epochs, { eta = 0.02, batch = 10, seed = 214 } = {}) {
  const tr = echo(200, k, 211);
  const te = echo(200, k, 212);
  let params = rnnFlatten(net0);
  let net = rnnUnflatten(net0, params);
  let opt = optInit('adam', params.length);
  const rng = mulberry32(seed);
  const losses = [meanLoss(net, tr)];
  const perBatch = [countRight(net, te).ok];
  const perEpoch = [countRight(net, te).ok];
  for (let e = 0; e < epochs; e++) {
    const order = Array.from({ length: 200 }, (_, i) => i);
    for (let i = 199; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let c = 0; c < 200; c += batch) {
      const idx = order.slice(c, c + batch);
      const g = new Array(params.length).fill(0);
      for (const q of idx) rnnFlatten(bptt(net, tr.seqs[q], tr.targets[q], tr.mask[q]).grads).forEach((v, z) => { g[z] += v; });
      const r = optStep('adam', { eta }, opt, params, g.map((v) => v / idx.length));
      params = r.params;
      opt = r.state;
      net = rnnUnflatten(net, params);
      perBatch.push(countRight(net, te).ok);
    }
    losses.push(meanLoss(net, tr));
    perEpoch.push(countRight(net, te).ok);
  }
  return { net, losses, perBatch, perEpoch, tr, te };
}

// Every starting weight times (1 ± 1e-12), the sign alternating: another engine's
// last-place rounding of Math.tanh or Math.exp, in miniature.
const nudged = (net, eps) => rnnUnflatten(net, rnnFlatten(net).map((v, i) => v * (1 + (i % 2 ? -eps : eps))));

const DIRECT = new Map();
const direct = (k) => {
  if (!DIRECT.has(k)) DIRECT.set(k, directRun(rnnInit({ nh: 8, seed: 213 }), k, 10));
  return DIRECT.get(k);
};

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

const geoMean = (rows, j) => Math.exp(rows.reduce((s, r) => s + Math.log(r[j]), 0) / rows.length);

test('data reproducible', () => {
  assert.equal(S.SEED_TRAIN, 211);
  assert.equal(S.SEED_TEST, 212);
  assert.equal(S.SEED_INIT, 213);
  assert.equal(S.SEED_SHUFFLE, 214);
  assert.equal(S.N_TRAIN, 200);
  assert.equal(S.N_TEST, 200);
  assert.equal(S.T, 12);
  assert.equal(S.NH, 8);
  assert.equal(S.ETA, 0.02);
  assert.equal(S.BATCH, 10);
  assert.equal(S.EPOCHS, 10);
  assert.equal(S.ITERS, 200);
  assert.equal(S.K_LIVE, 2);
  assert.deepEqual(S.K_CHOICES, [1, 2, 4, 8]);
  for (const k of S.K_CHOICES) {
    assert.deepEqual(S.trainSet(k), echo(200, k, 211));
    assert.deepEqual(S.testSet(k), echo(200, k, 212));
    assert.deepEqual(S.trainSet(k), echoData({ n: 200, T: 12, k, seed: 211 }));
  }
  assert.deepEqual(S.trainSet(2).seqs[0], LITERAL_TRAIN_0);
  assert.deepEqual(S.startNet(), rnnInit({ nh: 8, seed: 213 }));
  assert.equal(rnnFlatten(S.startNet()).length, 8 + 64 + 8 + 8 + 1);
});

test('targets correct', () => {
  for (const k of S.K_CHOICES) {
    for (const D of [S.trainSet(k), S.testSet(k)]) {
      D.seqs.forEach((xs, i) => {
        assert.equal(xs.length, 12);
        for (let t = 0; t < 12; t++) {
          assert.ok(xs[t] === 0 || xs[t] === 1);
          assert.equal(D.mask[i][t], t >= k ? 1 : 0);
          if (t >= k) assert.equal(D.targets[i][t], xs[t - k], `k ${k} seq ${i} t ${t}`);
        }
      });
    }
    // The same 200 bit streams for every k: only the targets change.
    assert.deepEqual(S.trainSet(k).seqs, S.trainSet(1).seqs);
    assert.deepEqual(S.testSet(k).seqs, S.testSet(1).seqs);
    assert.equal(S.maskedCount(k), 200 * (12 - k));
  }
  // Bits are close to half ones (the generator is fair).
  const ones = S.testSet(1).seqs.flat().reduce((s, b) => s + b, 0);
  assert.ok(ones > 1100 && ones < 1300, String(ones));
});

test('masked accuracy: only masked steps count, and p̂ = 0.5 predicts 0', () => {
  const net = S.startNet();
  for (const k of [2, 8]) {
    const te = S.testSet(k);
    const a = S.maskedAccuracy(net, te);
    assert.deepEqual(a, countRight(net, te));
    assert.equal(a.n, 200 * (12 - k));
  }
  // An all-zero network has z = 0 everywhere, so p̂ = 0.5 and every prediction is 0.
  const zero = rnnUnflatten(net, rnnFlatten(net).map(() => 0));
  const te = S.testSet(2);
  let zeros = 0;
  te.targets.forEach((ys, i) => ys.forEach((y, t) => { if (te.mask[i][t] && y === 0) zeros++; }));
  assert.equal(S.maskedAccuracy(zero, te).ok, zeros);
  assert.equal(S.predictBit(0), 0);
  assert.equal(S.predictBit(1e-9), 1);
  assert.equal(S.predictBit(-1e-9), 0);
});

test('rnnTrainer: batches, epochs and the starting loss', () => {
  const tr = echo(200, 2, 211);
  const net = rnnInit({ nh: 8, seed: 213 });
  const t = S.rnnTrainer({ net, data: tr, batch: 10, kind: 'adam', hyper: { eta: 0.02 }, seed: 214 });
  assert.deepEqual(t.state.epochLoss, [meanLoss(net, tr)]);
  assert.equal(t.state.iter, 0);
  assert.equal(t.state.epoch, 0);
  const rng = mulberry32(214);
  const order = Array.from({ length: 200 }, (_, i) => i);
  for (let i = 199; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const idx = order.slice(0, 10);
  const per = idx.map((q) => rnnFlatten(bptt(net, tr.seqs[q], tr.targets[q], tr.mask[q]).grads));
  const mean = per[0].map((_, q) => per.reduce((s, g) => s + g[q], 0) / 10);
  const g = S.batchGradRnn(net, tr, idx);
  g.grad.forEach((v, q) => assert.ok(Math.abs(v - mean[q]) < 1e-15, String(q)));
  const lossMean = idx.reduce((s, q) => s + bptt(net, tr.seqs[q], tr.targets[q], tr.mask[q]).loss, 0) / 10;
  assert.ok(Math.abs(g.loss - lossMean) < 1e-15);
  const want = optStep('adam', { eta: 0.02 }, optInit('adam', 89), rnnFlatten(net), g.grad).params;
  t.step(1);
  assert.deepEqual(rnnFlatten(t.state.net), want);
  assert.equal(t.state.iter, 1);
  assert.equal(t.state.batchLoss[0], g.loss);
  assert.deepEqual(net, rnnInit({ nh: 8, seed: 213 }));
  // Twenty batches of 10 make one epoch of 200 sequences.
  t.step(19);
  assert.equal(t.state.epoch, 1);
  assert.equal(t.state.epochLoss.length, 2);
  assert.equal(t.state.epochLoss[1], meanLoss(t.state.net, tr));
  // A last partial batch is kept: batch 30 makes 7 steps per epoch (6 × 30 + 20).
  const t30 = S.rnnTrainer({ net, data: tr, batch: 30, kind: 'adam', hyper: { eta: 0.02 }, seed: 214 });
  t30.step(6);
  assert.equal(t30.state.epoch, 0);
  t30.step(1);
  assert.equal(t30.state.epoch, 1);
  // Reset gives the same start and the same path again.
  const a = t.state.epochLoss.slice();
  t.reset();
  assert.deepEqual(t.state.epochLoss, [a[0]]);
  t.step(20);
  assert.deepEqual(t.state.epochLoss, a);
});

test('training deterministic', () => {
  const one = S.makeTrainer(2);
  const two = S.makeTrainer(2);
  one.step(200);
  two.step(77);
  two.step(123);
  assert.deepEqual(one.state.epochLoss, two.state.epochLoss);
  assert.deepEqual(rnnFlatten(one.state.net), rnnFlatten(two.state.net));
  assert.equal(one.state.diverged, false);
  const d = direct(2);
  assert.deepEqual(one.state.epochLoss, d.losses);
  assert.deepEqual(rnnFlatten(one.state.net), rnnFlatten(d.net));
  // The live run, one batch at a time, is the same run, with the test count after every batch.
  const live = liveRun();
  assert.deepEqual(live.trainer.state.epochLoss, d.losses);
  assert.deepEqual(live.testOk, d.perBatch);
  assert.equal(live.testOk.length, 201);
  // Each k's budget run is the same recipe.
  const runs = pageRuns();
  for (const k of S.K_CHOICES) {
    const dk = direct(k);
    assert.deepEqual(runs[k].losses, dk.losses, `k ${k}`);
    assert.deepEqual(runs[k].testOk, dk.perEpoch, `k ${k}`);
    assert.deepEqual(rnnFlatten(runs[k].net), rnnFlatten(dk.net), `k ${k}`);
    assert.equal(runs[k].n, 200 * (12 - k));
    assert.deepEqual(runs[k].trainAcc, countRight(dk.net, dk.tr));
  }
  // Step 3's live run and step 4's k = 2 run end in the same place.
  assert.deepEqual(rnnFlatten(runs[2].net), rnnFlatten(live.trainer.state.net));
  // Reset of the live run repeats it.
  const again = S.makeLive();
  again.advance(13);
  again.reset();
  assert.deepEqual(again.testOk, [d.perBatch[0]]);
  while (!again.done()) again.advance(7);
  assert.deepEqual(again.testOk, d.perBatch);
});

test('k = 2 reaches masked test accuracy ≥ 0.95', () => {
  const live = liveRun();
  const ok = live.testOk;
  const n = 2000;
  assert.equal(live.n, n);
  assert.ok(ok.at(-1) / n >= 0.95);
  assert.equal(ok.at(-1), n);
  // Starts near a coin toss.
  assert.ok(ok[0] / n > 0.5 && ok[0] / n < 0.65, String(ok[0]));
  // First batch at or above 95%, and every batch from then on at or above 95%.
  const first = ok.findIndex((v) => v / n >= 0.95);
  assert.equal(S.firstAt(ok, n, 0.95), first);
  for (let i = first; i < ok.length; i++) assert.ok(ok[i] / n >= 0.95, String(i));
  // From this batch on every masked test step is right.
  const allRight = S.allRightFrom(ok, n);
  assert.ok(allRight > first);
  for (let i = allRight; i < ok.length; i++) assert.equal(ok[i], n, String(i));
  assert.ok(ok[allRight - 1] < n);
  assert.ok(allRight <= 20);
  // The training loss falls every epoch.
  const L = live.trainer.state.epochLoss;
  for (let e = 1; e <= 10; e++) assert.ok(L[e] < L[e - 1], String(e));
  const sv = S.storyValues();
  assert.equal(sv['start-acc'], S.pctCount(ok[0], n));
  assert.equal(sv['first-95'], String(first));
  assert.equal(sv['first-95-acc'], S.pctCount(ok[first], n));
  assert.equal(sv['before-95-acc'], S.pctCount(ok[first - 1], n));
  assert.equal(sv['all-right'], String(allRight));
  assert.equal(sv['end-acc'], '100%');
  assert.equal(sv['start-loss'], S.lossText(L[0]));
  assert.equal(sv['end-loss'], S.lossText(L[10]));
});

test('accuracy at k = 8 lower than at k = 2 under the same budget', () => {
  const runs = pageRuns();
  const acc = (k, e = 10) => runs[k].testOk[e] / runs[k].n;
  assert.ok(acc(8) < acc(2));
  assert.ok(acc(8) < 0.95);
  assert.ok(acc(8) > 0.5);
  // k = 1 and k = 2: every masked test step right after one epoch, and at every later epoch.
  for (const k of [1, 2]) for (let e = 1; e <= 10; e++) assert.equal(acc(k, e), 1, `k ${k} e ${e}`);
  // k = 4: perfect first at epoch 4, and from then on.
  const p4 = runs[4].testOk.findIndex((v) => v === runs[4].n);
  assert.equal(p4, 4);
  for (let e = 4; e <= 10; e++) assert.equal(acc(4, e), 1);
  // k = 8: below 95% at every epoch of the budget, and starts at chance.
  for (let e = 0; e <= 10; e++) assert.ok(acc(8, e) < 0.95, String(e));
  assert.equal(acc(8, 0), 0.5);
  // Its training accuracy is below 95% too, so it is not only a test-set problem.
  assert.ok(runs[8].trainAcc.ok / runs[8].trainAcc.n < 0.95);
  // The ordering: further back, never better, here.
  assert.ok(acc(1) >= acc(2) && acc(2) >= acc(4) && acc(4) > acc(8));
  const sv = S.storyValues();
  for (const k of S.K_CHOICES) assert.equal(sv[`acc-k${k}`], S.pctCount(runs[k].testOk[10], runs[k].n));
  assert.equal(sv['k4-perfect'], String(p4));
  assert.equal(sv['k8-ok'], String(runs[8].testOk[10]));
  assert.equal(sv['k8-n'], String(runs[8].n));
  assert.equal(sv['k8-train'], S.pctCount(runs[8].trainAcc.ok, runs[8].trainAcc.n));
  assert.equal(sv['k8-start'], S.pctCount(runs[8].testOk[0], runs[8].n));
  assert.equal(sv['k8-best'], S.pctCount(Math.max(...runs[8].testOk), runs[8].n));
});

test('lag norm plot equals bptt lagNorms', () => {
  const te = echo(200, 2, 212);
  const net = rnnInit({ nh: 8, seed: 213 });
  const info = S.lagInfo(net, te);
  assert.equal(info.lines.length, S.LAG_SHOWN);
  info.lines.forEach((line, i) => assert.deepEqual(line, bptt(net, te.seqs[i], te.targets[i], te.mask[i]).lagNorms));
  const all = te.seqs.map((xs, i) => bptt(net, xs, te.targets[i], te.mask[i]).lagNorms);
  for (let j = 0; j < 12; j++) assert.ok(Math.abs(info.typical[j] - geoMean(all, j)) < 1e-15 * geoMean(all, j) * 10, String(j));
  for (const v of [...info.typical, ...info.lines.flat()]) assert.ok(Number.isFinite(v) && v > 0);
  // The plotted network is the starting network on the k = 2 targets.
  assert.deepEqual(S.lagPlot(), S.lagInfo(S.startNet(), S.testSet(2)));
  // The typical size falls at every step back, for the starting network.
  for (let j = 1; j < 12; j++) assert.ok(info.typical[j] < info.typical[j - 1], String(j));
  // lagNorms[0] is |p̂_T − y_T| ‖w_y‖: the target only scales each line, so the
  // ratios do not depend on k.
  const te8 = echo(200, 8, 212);
  for (let i = 0; i < 5; i++) {
    const a = bptt(net, te.seqs[i], te.targets[i], te.mask[i]).lagNorms;
    const b = bptt(net, te8.seqs[i], te8.targets[i], te8.mask[i]).lagNorms;
    const r = a[0] / b[0];
    a.forEach((v, j) => assert.ok(Math.abs(v / b[j] - r) < 1e-9 * r, `${i} ${j}`));
  }
  const p = 1 / (1 + Math.exp(-rnnForward(net, te.seqs[0]).zs[11]));
  assert.ok(Math.abs(info.lines[0][0] - Math.abs(p - te.targets[0][11]) * Math.hypot(...net.Wy)) < 1e-12);
  // The log axis covers every plotted value.
  const [lo, hi] = S.lagRange(info);
  for (const v of [...info.typical, ...info.lines.flat()]) assert.ok(Math.log10(v) >= lo && Math.log10(v) <= hi);
  const sv = S.storyValues();
  const t = info.typical;
  assert.equal(sv['lag0'], S.sig2(t[0]));
  assert.equal(sv['lag11'], S.sig2(t[11]));
  assert.equal(sv['lag-ratio'], String(Math.round(t[0] / t[11])));
  assert.equal(sv['lag2-ratio'], String(Math.round(t[0] / t[2])));
  assert.equal(sv['lag8-ratio'], String(Math.round(t[0] / t[8])));
  assert.ok(Math.round(t[0] / t[8]) > 3 * Math.round(t[0] / t[2]));
  // After the k = 8 run the sizes no longer fall step by step: 11 steps back is larger than at the last step.
  const trained = S.lagInfo(pageRuns()[8].net, S.testSet(8)).typical;
  assert.ok(trained[11] > trained[0]);
  assert.equal(sv['k8-lag0'], S.sig2(trained[0]));
  assert.equal(sv['k8-lag11'], S.sig2(trained[11]));
});

test('spectral norm of W_h', () => {
  const W = S.startNet().Wh;
  const { norm, v } = S.spectralNorm(W);
  const mul = (x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...mul(v)) - norm) < 1e-9);
  const rng = mulberry32(99);
  for (let i = 0; i < 2000; i++) {
    const x = Array.from({ length: 8 }, () => rng() - 0.5);
    assert.ok(Math.hypot(...mul(x)) <= norm * Math.hypot(...x) * (1 + 1e-9));
  }
  // A diagonal matrix's norm is its largest |entry|.
  assert.ok(Math.abs(S.spectralNorm([[0.5, 0], [0, -2]]).norm - 2) < 1e-12);
  assert.ok(norm > 1);
  assert.equal(S.storyValues()['wh-norm'], S.num(norm, 2));
});

test('bits the hidden state must carry at once', () => {
  // Written out again: after reading x_t (t = 1..12) the bits still needed now or
  // later are the x_j whose target step s = j + k satisfies t ≤ s ≤ min(12, t + k).
  const most = (k) => {
    let m = 0;
    for (let t = 1; t <= 12; t++) {
      let c = 0;
      for (let s = Math.max(k + 1, t); s <= Math.min(12, t + k); s++) c++;
      m = Math.max(m, c);
    }
    return m;
  };
  for (const k of S.K_CHOICES) assert.equal(S.bitsAtOnce(k), most(k), String(k));
  assert.deepEqual(S.K_CHOICES.map((k) => S.bitsAtOnce(k)), [2, 3, 5, 4]);
  // k = 8 needs fewer at once than k = 4, which reached 100% within the budget.
  assert.ok(S.bitsAtOnce(8) < S.bitsAtOnce(4));
  assert.equal(pageRuns()[4].testOk[10], pageRuns()[4].n);
  // k = 8's only targets are x_1..x_4, at steps 9 to 12 (zero-based 8..11).
  const D = S.testSet(8);
  D.mask.forEach((m, i) => {
    assert.deepEqual(m.map((v, t) => (v ? t : -1)).filter((t) => t >= 0), [8, 9, 10, 11]);
    assert.deepEqual([8, 9, 10, 11].map((t) => D.targets[i][t]), D.seqs[i].slice(0, 4));
  });
  const pv = S.pageValues();
  for (const k of S.K_CHOICES) assert.equal(pv[`held-k${k}`], String(S.bitsAtOnce(k)));
});

test('live pacing: first epoch slow, then one batch per frame, same run', () => {
  assert.equal(S.BATCHES_PER_EPOCH, 20);
  assert.equal(S.SLOW_FRAMES, 8);
  const live = S.makeLive();
  let waited = 0;
  let frames = 0;
  const takenAt = [];
  while (!live.done()) {
    const r = S.paceFrame(live.trainer.state.iter, waited);
    waited = r.waited;
    assert.ok(r.take === 0 || r.take === 1);
    if (r.take) { live.advance(r.take); takenAt.push(frames); }
    frames++;
  }
  // The first batch on the first frame; then 8 frames per batch for the first epoch.
  assert.equal(takenAt[0], 0);
  for (let i = 1; i < 20; i++) assert.equal(takenAt[i] - takenAt[i - 1], 8, String(i));
  for (let i = 21; i < 200; i++) assert.equal(takenAt[i] - takenAt[i - 1], 1, String(i));
  // The rise (to batch 18 and beyond) takes about 2.5 s at 60 frames a second.
  assert.ok(takenAt[18] / 60 > 2 && takenAt[19] / 60 < 3);
  // Pacing changes nothing about the run.
  assert.deepEqual(live.testOk, liveRun().testOk);
  assert.deepEqual(live.trainer.state.epochLoss, liveRun().trainer.state.epochLoss);
});

test('the quoted numbers do not hang on rounding', () => {
  const sv = S.storyValues();
  const runs = pageRuns();
  for (const eps of [1e-12, -1e-12]) {
    const net0 = nudged(rnnInit({ nh: 8, seed: 213 }), eps);
    const d2 = directRun(net0, 2, 10);
    // Every batch's test count in the live run, and every epoch's loss as shown.
    assert.deepEqual(d2.perBatch, liveRun().testOk);
    d2.losses.forEach((v, e) => assert.equal(S.lossText(v), S.lossText(liveRun().trainer.state.epochLoss[e]), `loss ${e}`));
    for (const k of S.K_CHOICES) {
      const dk = k === 2 ? d2 : directRun(net0, k, 10);
      assert.deepEqual(dk.perEpoch, runs[k].testOk, `k ${k}`);
      assert.deepEqual(countRight(dk.net, dk.tr), runs[k].trainAcc, `k ${k}`);
      dk.losses.forEach((v, e) => assert.equal(S.lossText(v), S.lossText(runs[k].losses[e]), `k ${k} loss ${e}`));
      if (k === 8) {
        const tr8 = S.lagInfo(dk.net, S.testSet(8)).typical;
        assert.equal(S.sig2(tr8[0]), sv['k8-lag0']);
        assert.equal(S.sig2(tr8[11]), sv['k8-lag11']);
      }
    }
    const info = S.lagInfo(net0, S.testSet(2));
    assert.equal(S.sig2(info.typical[0]), sv['lag0']);
    assert.equal(S.sig2(info.typical[11]), sv['lag11']);
    assert.equal(String(Math.round(info.typical[0] / info.typical[11])), sv['lag-ratio']);
    assert.equal(String(Math.round(info.typical[0] / info.typical[2])), sv['lag2-ratio']);
    assert.equal(String(Math.round(info.typical[0] / info.typical[8])), sv['lag8-ratio']);
    assert.equal(S.num(S.spectralNorm(net0.Wh).norm, 2), sv['wh-norm']);
    // Step 2's starting-network read-outs.
    const D = S.testSet(2);
    const { zs } = rnnForward(net0, D.seqs[S.EXAMPLE]);
    const p = zs.map((z) => 1 / (1 + Math.exp(-z)));
    assert.equal(S.num(p.at(-1), 2), sv['loop-p-last']);
  }
});

test('read-out rows', () => {
  const runs = pageRuns();
  const fresh = S.makeLive();
  const infos = [
    [0, S.seqInfo(2)],
    ...[0, 1, 6, 12].map((t) => [1, S.loopInfo(t)]),
    [2, S.liveInfo(fresh, false)],
  ];
  fresh.advance(9);
  infos.push([2, S.liveInfo(fresh, true)], [2, S.liveInfo(fresh, false)]);
  infos.push([2, S.liveInfo(liveRun(), false)]);
  for (const k of S.K_CHOICES) infos.push([3, S.budgetInfo(k, runs)], [3, S.budgetInfo(k, null)]);
  infos.push([4, S.lagPlot()], [5, null]);
  for (const [step, s] of infos) {
    const rows = S.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 48, r);
    }
  }
  const sv = S.storyValues();
  const start = S.readout(2, S.liveInfo(S.makeLive(), false));
  assert.ok(start.some((r) => /press Train/.test(r)) && !start.some((r) => /not finished/.test(r)), start.join('|'));
  assert.ok(start.some((r) => r.includes(sv['start-acc'])), start.join('|'));
  assert.ok(start.some((r) => r.includes(sv['start-loss'])), start.join('|'));
  const mid = S.readout(2, S.liveInfo(fresh, false));
  assert.ok(mid.some((r) => /not finished: press Train/.test(r)), mid.join('|'));
  const run = S.readout(2, S.liveInfo(fresh, true));
  assert.ok(run.some((r) => /training…/.test(r)) && !run.some((r) => /press Train/.test(r)), run.join('|'));
  const end = S.readout(2, S.liveInfo(liveRun(), false));
  assert.ok(end.some((r) => r.includes('batch 200 of 200')), end.join('|'));
  assert.ok(end.some((r) => r.includes(sv['end-loss'])) && end.some((r) => r.includes(sv['end-acc'])), end.join('|'));
  assert.ok(!end.some((r) => /press Train|training…/.test(r)), end.join('|'));
  const b8 = S.readout(3, S.budgetInfo(8, runs));
  assert.ok(b8.some((r) => r.includes(sv['acc-k8'])) && b8.some((r) => r.includes(`${sv['k8-ok']} of ${sv['k8-n']}`)), b8.join('|'));
  assert.ok(b8.some((r) => /test/.test(r)) && b8.some((r) => /train/.test(r)), b8.join('|'));
  assert.ok(S.readout(3, S.budgetInfo(8, null)).some((r) => /computing/.test(r)));
  const lag = S.readout(4, S.lagPlot());
  assert.ok(lag.some((r) => r.includes(sv['lag0'])) && lag.some((r) => r.includes(sv['lag11'])), lag.join('|'));
  assert.ok(lag.some((r) => /starting network/.test(r)), lag.join('|'));
  const loop = S.readout(1, S.loopInfo(12));
  assert.ok(loop.some((r) => r.includes(`p̂ = ${sv['loop-p-last']}`)), loop.join('|'));
  // loopInfo is the starting network's forward pass on the example.
  const D = S.testSet(2);
  const fw = rnnForward(S.startNet(), D.seqs[S.EXAMPLE]);
  assert.deepEqual(S.loopInfo(12).hs, fw.hs);
  assert.equal(S.loopInfo(0).t, 0);
});

test('runs computed in slices equal the runs computed at once', async () => {
  const F = await import('../rnns/stage.js?sliced');
  for (const k of F.K_CHOICES) {
    assert.equal(F.runReady(k), false);
    let slices = 0;
    while (!F.runWork(k, 11)) slices++;
    assert.ok(slices >= 15);
  }
  for (const k of F.K_CHOICES) {
    assert.deepEqual(F.runResult(k).losses, pageRuns()[k].losses);
    assert.deepEqual(F.runResult(k).testOk, pageRuns()[k].testOk);
  }
  // Nothing is computed until asked: a fresh copy has no results.
  const G = await import('../rnns/stage.js?fresh');
  for (const k of G.K_CHOICES) assert.equal(G.runReady(k), false);
  assert.equal(G.lagReady(), false);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Recurrent networks · ML Explorer<\/title>/);
  assert.match(html, /<h1>Recurrent networks<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/rnns\/">/);
  for (const s of [211, 212, 213, 214]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /Play steps through the lesson, and Train trains/);
  assert.match(html, /chance/);
  assert.match(html, /LSTM/);
  assert.match(html, /GRU/);
  assert.match(html, /attention/i);
  assert.match(html, /Level 5/);
  assert.match(html, /\\operatorname\{diag\}|\\mathrm\{diag\}/);
  assert.match(html, /href="\/ml\/cnns\/"/);
  assert.match(html, /href="\/ml\/deep-networks\/"/);
  assert.match(html, /href="\/ml\/backpropagation\/"/);
  assert.match(html, /href="\/ml\/optimisers\/"/);
  assert.match(html, /href="\/ml\/train-test\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/">Back to the path →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|it decided|always better|can learn anything|solves? (the )?(long|memory)|solved/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(html, /Next lesson: /);
  assert.doesNotMatch(html, /last 8 bits|next three targets|only needs the last 2/);
  // Quiz q2's distractor is refuted by the page's own numbers: k = 4 holds more bits at
  // once than k = 8, with the same 8 hidden numbers, and got every masked test step right.
  assert.match(html, /value="more"> Holding 4 bits at once is more than its 8 hidden numbers can manage/);
  assert.equal(S.bitsAtOnce(8), 4);
  assert.ok(S.bitsAtOnce(4) > S.bitsAtOnce(8));
  assert.equal(S.NH, 8);
  assert.equal(S.storyValues()['acc-k4'], '100%');
  assert.match(html, /data-for="more">The same network, with the same 8 hidden numbers, managed k = 4/);
  assert.doesNotMatch(html, /It has to hold more bits at once|Not because there is more to hold/);
  assert.match(html, /Not mainly because there is more to hold/);
  assert.match(html, /epoch 17/);
  assert.match(html, /measured on the k = 8 test targets/);
  assert.match(html, /above the 50% of chance/);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What does the hidden state carry\?/);
  assert.match(html, /Why is k = 8 harder to learn than k = 2\?/);
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
  assert.ok(story.length >= 15);
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
