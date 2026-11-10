import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { mlpInit, mlpForward, batchGrad, trainer, predictProb, accuracyNet, ACT } from '../assets/nn.js';
import * as D from '../deep-networks/stage.js';

const html = readFileSync(new URL('../deep-networks/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_X0 = -0.9511941714623376;

// The circles written out again: angle uniform on [0, 2π), radius r₀ + 0.25 z.
function circles(n, seed) {
  const rng = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const c = i < n / 2 ? 0 : 1;
    const th = 2 * Math.PI * rng();
    const r = (c ? 2.2 : 1) + 0.25 * randn(rng);
    X.push([r * Math.cos(th), r * Math.sin(th)]);
    y.push(c);
  }
  return { X, y };
}

const bceMean = (ps, y) => ps.reduce((t, p, i) => t - (y[i] ? Math.log(p) : Math.log(1 - p)), 0) / ps.length;
const ALL = Array.from({ length: 200 }, (_, i) => i);
const fro = (M) => Math.sqrt(M.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));
// Per-layer Frobenius norms of the batch-mean gradient over all 200 points.
function direct(act, init, nudge = 0) {
  const tr = circles(200, 181);
  const net = mlpInit([2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1], { hidden: act, init, seed: 182 });
  net.W[0][0][0] *= 1 + nudge;
  return batchGrad(net, tr.X, tr.y, ALL).dW.map(fro);
}

test('data reproducible', () => {
  assert.equal(D.SEED_DATA, 181);
  assert.equal(D.SEED_INIT, 182);
  assert.equal(D.SEED_SHUFFLE, 183);
  assert.deepEqual(D.SIZES, [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1]);
  assert.equal(D.N_LAYERS, 11);
  assert.deepEqual(D.trainSet(), circles(200, 181));
  assert.equal(D.trainSet().X[0][0], LITERAL_TRAIN_X0);
  assert.equal(D.trainSet().X[0][0], -0.9511941714623376);
  assert.equal(D.initFor('sigmoid', 'small'), 0.1);
  assert.equal(D.initFor('sigmoid', 'matched'), 'xavier');
  assert.equal(D.initFor('tanh', 'matched'), 'xavier');
  assert.equal(D.initFor('relu', 'matched'), 'he');
  assert.equal(D.initFor('relu', 'large'), 2);
  assert.deepEqual(D.startNet('relu', 'matched'), mlpInit(D.SIZES, { hidden: 'relu', init: 'he', seed: 182 }));
  // Xavier here is variance 1/n_in; He is 2/n_in. Same seed: the same draws, scaled.
  const x = D.startNet('sigmoid', 'matched');
  const h = D.startNet('relu', 'matched');
  const s = D.startNet('tanh', 'large');
  for (let l = 0; l < 11; l++) {
    const nIn = D.SIZES[l];
    for (let j = 0; j < D.SIZES[l + 1]; j++) {
      for (let i = 0; i < nIn; i++) {
        const z = s.W[l][j][i] / 2;
        assert.ok(Math.abs(x.W[l][j][i] - z * Math.sqrt(1 / nIn)) < 1e-12);
        assert.ok(Math.abs(h.W[l][j][i] - z * Math.sqrt(2 / nIn)) < 1e-12);
      }
    }
  }
  assert.equal(D.paramCount(D.SIZES), 681);
  assert.equal(D.pageValues().params, '681');
});

test('bars equal per-layer norms from batchGrad', () => {
  for (const act of D.ACTS) {
    for (const key of D.INITS) {
      const g = D.gradNorms(act, key);
      const d = direct(act, D.initFor(act, key));
      assert.equal(g.norms.length, 11);
      g.norms.forEach((v, l) => assert.ok(Math.abs(v - d[l]) <= 1e-12 * d[l], `${act} ${key} ${l}`));
      assert.equal(g.first, g.norms[0]);
      assert.equal(g.last, g.norms[10]);
      assert.equal(g.ratio, g.norms[0] / g.norms[10]);
      assert.equal(g.largest, Math.max(...g.norms));
      for (const v of g.norms) assert.ok(Number.isFinite(v) && v > 0);
    }
  }
});

test('sigmoid Xavier first-layer gradient ≤ last-layer / 100', () => {
  const g = D.gradNorms('sigmoid', 'matched');
  assert.ok(g.first <= g.last / 100, `${g.first} ${g.last}`);
  // Every layer back is smaller than the one after it.
  for (let l = 0; l < 10; l++) assert.ok(g.norms[l] < g.norms[l + 1], String(l));
  assert.ok(Math.abs(g.shrink - Math.pow(g.last / g.first, 1 / 10)) < 1e-12);
  const sv = D.storyValues();
  assert.equal(sv['sx-first'], D.sci(g.first));
  assert.equal(sv['sx-last'], D.sci(g.last));
  assert.equal(sv['sx-ratio'], D.sci(g.ratio));
  assert.equal(sv['sx-shrink'], D.num(g.shrink, 1));
  assert.equal(sv['sx-million'], String(Math.round(1 / g.ratio / 1e6)));
  // The step from layer 2 back to layer 1 is the largest of the ten.
  const steps = g.norms.slice(1).map((v, l) => v / g.norms[l]);
  assert.equal(Math.max(...steps), steps[0]);
  assert.equal(sv['sx-step1'], String(Math.round(steps[0])));
  assert.ok(steps[0] > 20 && steps[0] < 25);
  // The mean slope σ′ over the training points, per hidden layer, is near 1/4.
  const slopes = D.meanSlopes('sigmoid', 'matched');
  assert.equal(slopes.length, 10);
  const tr = circles(200, 181);
  const net = mlpInit(D.SIZES, { hidden: 'sigmoid', init: 'xavier', seed: 182 });
  slopes.forEach((m, l) => {
    let s = 0;
    for (const x of tr.X) {
      let a = x;
      for (let k = 0; k <= l; k++) {
        const z = net.W[k].map((row, j) => row.reduce((t, w, i) => t + w * a[i], net.b[k][j]));
        if (k === l) s += z.reduce((t, v) => t + ACT.sigmoid.df(v), 0) / 8;
        a = z.map(ACT.sigmoid.f);
      }
    }
    assert.ok(Math.abs(m - s / 200) < 1e-12, String(l));
    assert.ok(m > 0.2 && m <= 0.25);
  });
  assert.equal(sv['slope-min'], D.num(Math.min(...slopes), 2));
  assert.equal(sv['slope-max'], D.num(Math.max(...slopes), 2));
});

test('σ′ is at most 1/4, with equality at 0', () => {
  assert.equal(ACT.sigmoid.df(0), 0.25);
  for (let z = -20; z <= 20; z += 0.01) assert.ok(ACT.sigmoid.df(z) <= 0.25);
  for (const z of [1e-3, -1e-3, 1, -3]) assert.ok(ACT.sigmoid.df(z) < 0.25);
});

test('ReLU He first/last ratio within [0.1, 10]', () => {
  const g = D.gradNorms('relu', 'matched');
  assert.ok(g.ratio >= 0.1 && g.ratio <= 10, String(g.ratio));
  const t = D.gradNorms('tanh', 'matched');
  assert.ok(t.ratio >= 0.1 && t.ratio <= 10, String(t.ratio));
  const sv = D.storyValues();
  assert.equal(sv['he-ratio'], D.num(g.ratio, 2));
  assert.equal(sv['tanh-ratio'], D.num(t.ratio, 2));
  // Every He bar is between 0.1 and 10.
  for (const v of g.norms) assert.ok(v > 0.1 && v < 10, String(v));
});

test('sd 2.0 explodes; sd 0.1 makes every bar tiny', () => {
  const he = D.gradNorms('relu', 'matched');
  const big = D.gradNorms('relu', 'large');
  assert.ok(big.largest >= 1e3 * he.largest, `${big.largest} ${he.largest}`);
  // Clipped at the chart top, with a label.
  const tops = big.norms.map((v) => D.barSpan(v).top);
  assert.ok(tops.every(Boolean));
  assert.ok(D.readout(3, D.barsInfo('relu', 'large')).some((r) => /cut off at 10²/.test(r)));
  assert.ok(!D.readout(3, D.barsInfo('relu', 'matched')).some((r) => /cut off/.test(r)));
  // With sd 0.1, ReLU and tanh: every bar below 10⁻⁶; the sigmoid's first below 10⁻¹².
  for (const act of ['relu', 'tanh']) for (const v of D.gradNorms(act, 'small').norms) assert.ok(v < 1e-6, act);
  assert.ok(D.gradNorms('sigmoid', 'small').first < 1e-12);
  // No bar of any switch setting falls below the chart floor.
  for (const act of D.ACTS) for (const key of D.INITS) for (const v of D.gradNorms(act, key).norms) assert.equal(D.barSpan(v).bottom, false);
  const sv = D.storyValues();
  assert.equal(sv['big-largest'], D.sci(big.largest));
  assert.equal(sv['he-largest'], D.sci(he.largest));
  assert.equal(sv['tanh-big-ratio'], D.num(D.gradNorms('tanh', 'large').ratio, 0));
  assert.equal(sv['sig-big-ratio'], D.sci(D.gradNorms('sigmoid', 'large').ratio));
  assert.ok(D.gradNorms('tanh', 'large').ratio > 100);
  assert.ok(D.gradNorms('sigmoid', 'large').ratio < 1e-2 && D.gradNorms('sigmoid', 'large').ratio > D.gradNorms('sigmoid', 'matched').ratio);
});

test('sci and barSpan', () => {
  assert.equal(D.sci(1.234e-8), '1.2×10⁻⁸');
  assert.equal(D.sci(2.04e6), '2.0×10⁶');
  assert.equal(D.sci(0.14), '0.14');
  assert.equal(D.sci(0.6066), '0.61');
  assert.equal(D.sci(320.4), '320');
  assert.equal(D.sci(1234), '1.2×10³');
  assert.equal(D.sci(9.96e-3), '1.0×10⁻²');
  assert.equal(D.sci(3.3), '3.3');
  assert.deepEqual(D.LOG_RANGE, [-14, 2]);
  assert.deepEqual(D.barSpan(1e-3), { e: -3, top: false, bottom: false });
  assert.deepEqual(D.barSpan(1e5), { e: 2, top: true, bottom: false });
  assert.deepEqual(D.barSpan(1e-20), { e: -14, top: false, bottom: true });
});

// The page's runs, computed in odd-sized slices, shared below.
let RUNS = null;
function pageRuns() {
  if (RUNS) return RUNS;
  for (const act of D.RUN_ACTS) while (!D.runWork(act, 37));
  RUNS = Object.fromEntries(D.RUN_ACTS.map((a) => [a, D.runResult(a)]));
  return RUNS;
}

test('runs deterministic and equal the trainer', () => {
  const tr = circles(200, 181);
  const runs = pageRuns();
  for (const act of D.RUN_ACTS) {
    const net = mlpInit(D.SIZES, { hidden: act, init: act === 'relu' ? 'he' : 'xavier', seed: 182 });
    const t = trainer({ net, X: tr.X, y: tr.y, batch: 20, kind: 'sgd', hyper: { eta: 0.05 }, seed: 183 });
    t.step(1000);
    assert.equal(t.state.diverged, false);
    assert.deepEqual(runs[act].losses, t.state.epochLoss);
    assert.equal(runs[act].losses.length, 101);
    assert.equal(runs[act].acc, accuracyNet(t.state.net, tr.X, tr.y));
    // The live trainer is the same run.
    const live = D.makeTrainer(act);
    live.step(1000);
    assert.deepEqual(live.state.epochLoss, t.state.epochLoss);
  }
  assert.equal(D.ETA, 0.05);
  assert.equal(D.BATCH, 20);
  assert.equal(D.EPOCHS, 100);
});

test('ReLU He trains to lower loss than sigmoid Xavier after 100 epochs', () => {
  const runs = pageRuns();
  const s = runs.sigmoid;
  const r = runs.relu;
  assert.ok(r.final < s.final);
  // The sigmoid net makes no progress beyond a coin toss: its loss stays within
  // 0.001 of ln 2 from epoch 5 on, and it gives every point the same probability.
  for (let e = 5; e <= 100; e++) assert.ok(Math.abs(s.losses[e] - Math.LN2) < 1e-3, String(e));
  assert.equal(D.num(s.probMin, 3), D.num(s.probMax, 3));
  assert.ok(s.probMax < 0.5);
  assert.equal(s.acc, 0.5);
  // ReLU: its loss falls below 0.01 and every training point is right.
  assert.ok(r.final < 0.01);
  assert.equal(r.acc, 1);
  // Its curve is below the sigmoid's from epoch 10 on.
  for (let e = 10; e <= 100; e++) assert.ok(r.losses[e] < s.losses[e], String(e));
  const sv = D.storyValues();
  assert.equal(sv['sig-start'], D.num(s.losses[0], 3));
  assert.equal(sv['sig-end'], D.num(s.final, 3));
  assert.equal(sv['sig-end'], '0.693');
  assert.equal(sv['sig-prob'], D.num(s.probMax, 3));
  assert.equal(sv['relu-start'], D.num(r.losses[0], 3));
  assert.equal(sv['relu-10'], D.num(r.losses[10], 3));
  assert.equal(sv['relu-end'], D.num(r.final, 4));
  // The bumps after epoch 10: the largest is above epoch 10's loss.
  let be = 11;
  for (let e = 11; e <= 100; e++) if (r.losses[e] > r.losses[be]) be = e;
  assert.deepEqual(D.bump(r.losses), { epoch: be, loss: r.losses[be] });
  assert.ok(r.losses[be] > r.losses[10]);
  assert.equal(sv['relu-bump'], D.num(r.losses[be], 3));
  assert.equal(sv['relu-bump-epoch'], String(be));
  // At the start the sigmoid network gives every point the same prediction:
  // the output logit varies by under 10⁻⁶ across the 200 points.
  const st = D.sigmoidStart();
  const startNet = mlpInit(D.SIZES, { hidden: 'sigmoid', init: 'xavier', seed: 182 });
  const tr0 = circles(200, 181);
  const p0 = tr0.X.map((x) => predictProb(startNet, x));
  const z0 = p0.map((p) => Math.log(p / (1 - p)));
  assert.ok(st.spread < 1e-6 && st.spread > 0);
  assert.ok(Math.abs(st.spread - (Math.max(...z0) - Math.min(...z0))) < 1e-9);
  assert.equal(st.probMin, Math.min(...p0));
  assert.equal(D.num(st.probMin, 3), D.num(st.probMax, 3));
  assert.equal(sv['sig-p0'], D.num(st.probMax, 3));
  assert.equal(sv['sig-spread'], D.sci(st.spread));
  assert.equal(sv['sig-start'], D.num(bceMean(p0, tr0.y), 3));
  assert.equal(sv['relu-acc'], D.pct(r.acc));
  // The early layers barely move: relative change of W⁽¹⁾ and W⁽¹¹⁾.
  assert.ok(s.moved[0] < 1e-6 && s.moved[10] > 0.1);
  for (let l = 0; l < 10; l++) assert.ok(s.moved[l] <= s.moved[l + 1] * 1.01, String(l));
  for (const v of r.moved) assert.ok(v > 0.05, String(v));
  assert.equal(sv['sig-moved-1'], D.sci(s.moved[0]));
  assert.equal(sv['sig-moved-11'], D.num(s.moved[10], 2));
  assert.equal(sv['relu-moved-1'], D.num(r.moved[0], 2));
  assert.equal(sv['relu-moved-11'], D.num(r.moved[10], 2));
  const start = D.startNet('sigmoid', 'matched');
  const t = D.makeTrainer('sigmoid');
  t.step(1000);
  const rel = (l) => fro(t.state.net.W[l].map((row, j) => row.map((w, i) => w - start.W[l][j][i]))) / fro(start.W[l]);
  assert.ok(Math.abs(rel(0) - s.moved[0]) < 1e-15);
  const tr = circles(200, 181);
  const ps = tr.X.map((x) => predictProb(t.state.net, x));
  assert.equal(Math.min(...ps), s.probMin);
  assert.equal(Math.max(...ps), s.probMax);
});

test('the quoted numbers do not hang on rounding', () => {
  // Another JavaScript engine may round Math.exp in the last place. Nudging the
  // first weight by 1e-12 stands in for that: the quoted numbers must not move.
  for (const nudge of [1e-12, -1e-12]) {
    for (const [act, key] of [['sigmoid', 'matched'], ['relu', 'matched'], ['relu', 'large'], ['tanh', 'matched'], ['tanh', 'large'], ['sigmoid', 'large']]) {
      const d = direct(act, D.initFor(act, key), nudge);
      const g = D.gradNorms(act, key);
      d.forEach((v, l) => assert.equal(D.sci(v), D.sci(g.norms[l]), `${act} ${key} ${l}`));
      assert.equal(D.sci(d[0] / d[10]), D.sci(g.ratio));
      assert.equal(D.num(d[0] / d[10], 2), D.num(g.ratio, 2));
      assert.equal(D.num(d[0] / d[10], 0), D.num(g.ratio, 0));
      assert.equal(D.num(Math.pow(d[10] / d[0], 0.1), 1), D.num(g.shrink, 1));
    }
    {
      const net = mlpInit(D.SIZES, { hidden: 'sigmoid', init: 'xavier', seed: 182 });
      net.W[0][0][0] *= 1 + nudge;
      const tr1 = circles(200, 181);
      const z = tr1.X.map((x) => mlpForward(net, x).out[0]);
      const p = tr1.X.map((x) => predictProb(net, x));
      const st = D.sigmoidStart();
      assert.equal(D.sci(Math.max(...z) - Math.min(...z)), D.sci(st.spread));
      assert.equal(D.num(Math.max(...p), 3), D.num(st.probMax, 3));
    }
    const tr = circles(200, 181);
    const runs = pageRuns();
    for (const act of D.RUN_ACTS) {
      const net = D.startNet(act, 'matched');
      net.W[0][0][0] *= 1 + nudge;
      const t = trainer({ net, X: tr.X, y: tr.y, batch: 20, kind: 'sgd', hyper: { eta: 0.05 }, seed: 183 });
      t.step(1000);
      const L = t.state.epochLoss;
      assert.equal(D.num(L[0], 3), D.num(runs[act].losses[0], 3));
      assert.equal(D.num(L[10], 3), D.num(runs[act].losses[10], 3));
      assert.equal(D.num(L[100], 4), D.num(runs[act].final, 4));
      const b0 = D.bump(runs[act].losses);
      const b1 = D.bump(L);
      assert.equal(b1.epoch, b0.epoch);
      assert.equal(D.num(b1.loss, 3), D.num(b0.loss, 3));
      assert.equal(accuracyNet(t.state.net, tr.X, tr.y), runs[act].acc);
      const gap = Math.max(...L.map((v, i) => Math.abs(v - runs[act].losses[i])));
      assert.ok(gap < 1e-6, `${act} ${gap}`);
    }
  }
});

test('read-out rows', () => {
  pageRuns();
  const sig = D.makeTrainer('sigmoid');
  const relu = D.makeTrainer('relu');
  const infos = [
    [0, D.diagramInfo()],
    [1, D.barsInfo('sigmoid', 'matched')],
    [2, D.barsInfo('sigmoid', 'matched')],
    ...D.ACTS.flatMap((a) => D.INITS.map((k) => [3, D.barsInfo(a, k)])),
    [4, D.liveInfo(sig.state, relu.state)],
  ];
  sig.step(370); relu.step(370);
  infos.push([4, D.liveInfo(sig.state, relu.state)]);
  sig.step(630); relu.step(630);
  infos.push([4, D.liveInfo(sig.state, relu.state)]);
  infos.push([5, D.afterInfo()]);
  for (const [step, s] of infos) {
    const rows = D.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 6, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 26, r);
    }
  }
  const start = D.readout(4, D.liveInfo(D.makeTrainer('sigmoid').state, D.makeTrainer('relu').state));
  assert.ok(start.some((r) => /press Train/.test(r)), start.join('|'));
  const mid = D.readout(4, D.liveInfo(D.makeTrainer('sigmoid').step(370), D.makeTrainer('relu').step(370)));
  assert.ok(mid.some((r) => /not finished/.test(r)), mid.join('|'));
  const run = D.readout(4, D.liveInfo(D.makeTrainer('sigmoid').step(370), D.makeTrainer('relu').step(370), true));
  assert.ok(run.includes('training…') && !run.some((r) => /press Train/.test(r)), run.join('|'));
  const end = D.readout(4, D.liveInfo(sig.state, relu.state));
  const sv = D.storyValues();
  assert.ok(end.includes('epoch 100 of 100'), end.join('|'));
  assert.ok(end.some((r) => r.includes(sv['sig-end'])) && end.some((r) => r.includes(sv['relu-end'])), end.join('|'));
  const after = D.readout(5, D.afterInfo());
  assert.ok(after.some((r) => r.includes(sv['sig-moved-1'])), after.join('|'));
  const sx = D.readout(1, D.barsInfo('sigmoid', 'matched'));
  assert.ok(sx.some((r) => r.includes(sv['sx-first'])) && sx.some((r) => r.includes(sv['sx-ratio'])), sx.join('|'));
});

test('runs computed in slices equal the runs computed at once', async () => {
  const F = await import('../deep-networks/stage.js?sliced');
  for (const act of F.RUN_ACTS) {
    assert.equal(F.runReady(act), false);
    assert.ok(F.readout(5, F.afterInfo()).some((r) => /computing/.test(r)));
    let slices = 0;
    while (!F.runWork(act, 11)) slices++;
    assert.ok(slices >= 80);
    assert.deepEqual(F.runResult(act), pageRuns()[act]);
  }
  assert.ok(!F.readout(5, F.afterInfo()).some((r) => /computing/.test(r)));
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Going deeper: vanishing gradients · ML Explorer<\/title>/);
  assert.match(html, /<h1>Going deeper: vanishing gradients<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/deep-networks\/">/);
  for (const s of [181, 182, 183]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /variance 1\/n_in|1\/n<sub>in<\/sub>/);
  assert.match(html, /log scale/);
  assert.match(html, /Frobenius/);
  assert.match(html, /Play steps through the lesson, and Train trains/);
  assert.match(html, /residual connection/i);
  assert.match(html, /normalisation/);
  assert.match(html, /href="\/ml\/optimisers\/"/);
  assert.match(html, /href="\/ml\/backpropagation\/"/);
  assert.match(html, /href="\/ml\/logistic-regression\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/convolutions\/">Next lesson: Convolutions →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always vanish|always better|deeper is better|can learn anything/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  const pv = D.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 2);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const sv = D.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 20);
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
