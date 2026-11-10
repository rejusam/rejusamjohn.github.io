import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { circlesData, mlpInit, trainer, accuracyNet, optInit, optStep } from '../assets/nn.js';
import * as H from '../hidden-layers/stage.js';
import * as O from '../optimisers/stage.js';

const html = readFileSync(new URL('../optimisers/index.html', import.meta.url), 'utf8');
const LITERAL_TRAIN_X0 = -0.18646246620992768;

// The valley written out again: u along the valley, v across it, axes turned
// by 30° from the plot's w₁ and w₂.
const C30 = Math.cos(Math.PI / 6);
const S30 = Math.sin(Math.PI / 6);
const lossW = ([a, b]) => {
  const u = C30 * a + S30 * b;
  const v = -S30 * a + C30 * b;
  return 0.5 * (u * u + 25 * v * v);
};

// A run written out with optStep and finite differences for the gradient check.
function direct(kind, hyper, max = 2000, nudge = 0) {
  let p = [C30 * -4 - S30 * 2.5 + nudge, S30 * -4 + C30 * 2.5];
  let st = optInit(kind, 2);
  const path = [p];
  for (let k = 1; k <= max; k++) {
    const [a, b] = p;
    const u = C30 * a + S30 * b;
    const v = -S30 * a + C30 * b;
    const g = [C30 * u - S30 * 25 * v, S30 * u + C30 * 25 * v];
    const r = optStep(kind, hyper, st, p, g);
    p = r.params;
    st = r.state;
    path.push(p);
    const L = lossW(p);
    if (!Number.isFinite(L) || L > 1e6) return { path, reached: null, diverged: true };
    if (L < 1e-3) return { path, reached: k, diverged: false };
  }
  return { path, reached: null, diverged: false };
}

test('data reproducible', () => {
  assert.equal(O.ANGLE, 30);
  assert.deepEqual(O.START_UV, [-4, 2.5]);
  const s = O.VALLEY.start;
  assert.ok(Math.abs(s[0] - (C30 * -4 - S30 * 2.5)) < 1e-15);
  assert.ok(Math.abs(s[1] - (S30 * -4 + C30 * 2.5)) < 1e-15);
  assert.equal(O.pageValues()['w-start'], '(−4.71, 0.17)');
  assert.ok(Math.abs(O.VALLEY.loss(s) - 86.125) < 1e-12);
  assert.equal(O.pageValues().l0, '86.125');
  // The gradient is the true gradient of the loss.
  for (const w of [s, [1, -2], [0.3, 0.7]]) {
    const g = O.VALLEY.grad(w);
    for (let i = 0; i < 2; i++) {
      const up = w.slice(); const dn = w.slice();
      up[i] += 1e-6; dn[i] -= 1e-6;
      const num = (lossW(up) - lossW(dn)) / 2e-6;
      assert.ok(Math.abs(num - g[i]) < 1e-5 * Math.max(1, Math.abs(g[i])), `${w} ${i}`);
    }
    assert.ok(Math.abs(O.VALLEY.loss(w) - lossW(w)) < 1e-12);
  }
  // The network's data are L4.2's circles, as stated.
  assert.equal(O.SEED_TRAIN, 151);
  assert.equal(O.SEED_TEST, 152);
  assert.equal(O.SEED_INIT, 155);
  assert.equal(O.SEED_SHUFFLE, 154);
  const circles = (n, seed) => {
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
  };
  assert.deepEqual(O.trainSet(), circles(200, 151));
  assert.deepEqual(O.testSet(), circles(400, 152));
  assert.deepEqual(O.trainSet(), H.trainSet());
  assert.equal(O.trainSet().X[0][0], LITERAL_TRAIN_X0);
  assert.equal(O.trainSet().X[0][0], -0.18646246620992768);
  assert.deepEqual(O.startNet(), mlpInit([2, 4, 1], { hidden: 'tanh', init: 'xavier', seed: 155 }));
  assert.equal(O.BATCH, 20);
  assert.equal(O.NET_EPOCHS, 100);
});

test('valley runs equal the runs written out with optStep', () => {
  for (const [kind, eta] of [['sgd', 0.07], ['momentum', 0.02], ['rmsprop', 0.1], ['adam', 1], ['sgd', 0.085]]) {
    const r = O.valleyRun(kind, { eta });
    const d = direct(kind, { eta });
    assert.equal(r.reached, d.reached, kind);
    assert.equal(r.diverged, d.diverged, kind);
    assert.equal(r.path.length, d.path.length);
    r.path.forEach((p, k) => {
      assert.ok(Math.abs(p[0] - d.path[k][0]) < 1e-9 && Math.abs(p[1] - d.path[k][1]) < 1e-9, `${kind} ${k}`);
    });
  }
});

test('GD stability limit is 2/25 on this valley', () => {
  assert.equal(O.ETA_LIMIT, 2 / 25);
  assert.equal(O.ETA_LIMIT, 0.08);
  // The curvatures (Hessian eigenvalues) are 1 and 25, so κ = 25 and 2/γmax = 0.08.
  const h = 1e-4;
  const Hm = [[0, 0], [0, 0]];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const e = (k, d) => { const w = [0.2, -0.1]; w[k] += d; return w; };
      const gi = (w) => O.VALLEY.grad(w)[i];
      Hm[i][j] = (gi(e(j, h)) - gi(e(j, -h))) / (2 * h);
    }
  }
  const tr = Hm[0][0] + Hm[1][1];
  const det = Hm[0][0] * Hm[1][1] - Hm[0][1] * Hm[1][0];
  const disc = Math.sqrt(tr * tr / 4 - det);
  assert.ok(Math.abs(tr / 2 + disc - 25) < 1e-6);
  assert.ok(Math.abs(tr / 2 - disc - 1) < 1e-6);
  // Just under the limit it converges; just over it diverges, flagged, with finite points.
  const under = O.valleyRun('sgd', { eta: 0.0799 }, { max: 5000 });
  assert.equal(under.diverged, false);
  assert.ok(under.reached > 0);
  const over = O.valleyRun('sgd', { eta: 0.0801 }, { max: 5000 });
  assert.equal(over.diverged, true);
  assert.equal(over.reached, null);
  for (const p of over.path) assert.ok(p.every(Number.isFinite));
  for (const L of over.losses) assert.ok(Number.isFinite(L));
  // At exactly 2/25 the steep direction bounces for ever: never reached, never diverged.
  const at = O.valleyRun('sgd', { eta: 0.08 });
  assert.equal(at.diverged, false);
  assert.equal(at.reached, null);
  assert.equal(at.path.length, O.MAX_STEPS + 1);
  const minAt = Math.min(...at.losses);
  assert.ok(minAt > 78 && minAt < 78.2);
  assert.equal(O.storyValues()['gd-limit-loss'], O.num(minAt, 1));
  // The slider: every value below the limit reaches the target within 2000 steps;
  // every value above it is flagged and shown with the divergence wording.
  for (const eta of O.GD_ETAS) {
    const r = O.valleyRun('sgd', { eta });
    if (eta < 0.08) assert.ok(r.reached !== null && r.reached <= O.MAX_STEPS, String(eta));
    if (eta > 0.08) {
      assert.equal(r.diverged, true, String(eta));
      const rows = O.readout(0, O.gdInfo(eta));
      assert.ok(rows.includes('Diverged: the steps grow'), rows.join('|'));
      assert.ok(rows.includes('instead of shrinking'), rows.join('|'));
    }
  }
  assert.ok(O.GD_ETAS.includes(0.08) && O.GD_ETAS.includes(O.GD_DEFAULT));
  assert.ok(O.readout(0, O.gdInfo(0.08)).some((r) => /not below/.test(r)));
});

test('GD zig-zags across the valley above η = 1/25, and the tilt does not matter to it', () => {
  const across = (p) => -S30 * p[0] + C30 * p[1];
  const zig = O.valleyRun('sgd', { eta: O.GD_DEFAULT });
  for (let k = 1; k < zig.path.length; k++) {
    assert.ok(Math.sign(across(zig.path[k])) === -Math.sign(across(zig.path[k - 1])), String(k));
  }
  for (const eta of O.GD_ETAS.filter((e) => e <= 0.04)) {
    const r = O.valleyRun('sgd', { eta });
    for (const p of r.path) assert.ok(across(p) > -1e-12, String(eta)); // rounding only
  }
  const sv = O.storyValues();
  assert.equal(sv['gd-default'], String(zig.reached));
  // Lined up with the axes, plain GD takes exactly as many steps.
  for (const eta of [0.02, 0.05, 0.07]) {
    assert.equal(O.valleyRun('sgd', { eta }, { angle: 0 }).reached, O.valleyRun('sgd', { eta }).reached);
  }
  assert.equal(sv['gd-aligned'], String(O.valleyRun('sgd', { eta: O.GD_DEFAULT }, { angle: 0 }).reached));
});

test('momentum β 0.9 reaches ℒ < 1e-3 in fewer steps than GD at its best grid η', () => {
  const gd = O.tuneValley('sgd');
  const mom = O.tuneValley('momentum');
  assert.ok(mom.steps < gd.steps, `${mom.steps} ${gd.steps}`);
  // Step 2: η = 0.02 with β = 0, 0.5, 0.9; more momentum, fewer steps.
  assert.equal(O.MOM_ETA, 0.02);
  assert.deepEqual(O.BETAS, [0, 0.5, 0.9]);
  const steps = O.BETAS.map((beta) => O.momInfo(beta).reached);
  assert.ok(steps[0] > steps[1] && steps[1] > steps[2], steps.join(' '));
  // β = 0 is plain gradient descent.
  assert.equal(steps[0], O.valleyRun('sgd', { eta: 0.02 }).reached);
  const sv = O.storyValues();
  assert.deepEqual([sv['mom-0'], sv['mom-05'], sv['mom-09']], steps.map(String));
  // With β = 0.9 the path passes the bottom along the valley (u > 0) and swings back.
  const r = O.valleyRun('momentum', { eta: 0.02, beta: 0.9 });
  const us = r.path.map((p) => C30 * p[0] + S30 * p[1]);
  const first = us.findIndex((u) => u > 0);
  assert.ok(first > 0);
  assert.ok(us.slice(first).some((u) => u < 0));
  assert.equal(sv['mom-cross'], String(first));
  // It also swings across the valley: v changes sign several times on the way.
  const vs = r.path.map((p) => -S30 * p[0] + C30 * p[1]);
  let flips = 0;
  for (let i = 1; i < vs.length; i++) if (Math.sign(vs[i]) !== Math.sign(vs[i - 1])) flips++;
  assert.ok(flips >= 3);
  assert.equal(sv['mom-swings'], String(flips));
});

test('grid tuning picks the stated η for each optimiser', () => {
  assert.deepEqual(O.GRID, [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]);
  for (const kind of O.KINDS) {
    const t = O.tuneValley(kind);
    assert.equal(t.eta, O.VALLEY_ETA[kind], kind);
    // Fewest steps; ties go to the first η in the grid.
    for (const row of t.table) {
      if (row.reached !== null) assert.ok(row.reached >= t.steps, `${kind} ${row.eta}`);
    }
    assert.equal(t.table.length, O.GRID.length);
    assert.equal(t.table.find((r) => r.eta === t.eta).reached, t.steps);
  }
});

test('race numbers on the page equal the helper', () => {
  const runs = O.raceRuns();
  const sv = O.storyValues();
  for (const kind of O.KINDS) {
    assert.equal(runs[kind].reached, O.tuneValley(kind).steps);
    assert.equal(sv[`race-${kind}`], String(runs[kind].reached));
  }
  // The order the page reports: momentum first, RMSProp and Adam level, GD last.
  const k = Object.fromEntries(O.KINDS.map((x) => [x, runs[x].reached]));
  assert.ok(k.momentum < k.rmsprop);
  assert.equal(k.rmsprop, k.adam);
  assert.ok(k.adam < k.sgd);
  assert.deepEqual(O.raceOrder().map((r) => r.kind), ['momentum', 'rmsprop', 'adam', 'sgd']);
  // Off the grid the order changes: GD at the slider's 0.075 and Adam at η = 2.
  assert.equal(sv['gd-fine'], String(O.valleyRun('sgd', { eta: 0.075 }).reached));
  assert.ok(O.valleyRun('sgd', { eta: 0.075 }).reached < k.momentum);
  assert.ok(O.GD_ETAS.includes(0.075));
  // 0.075 is the fastest value on step 1's slider.
  const best = Math.min(...O.GD_ETAS.map((e) => O.valleyRun('sgd', { eta: e }).reached ?? Infinity));
  assert.equal(best, O.valleyRun('sgd', { eta: 0.075 }).reached);
  const a2 = O.valleyRun('adam', { eta: 2 }).reached;
  assert.equal(sv['adam-2'], String(a2));
  assert.ok(a2 < k.momentum);
  // Adam's best on the grid is its largest value.
  assert.equal(O.VALLEY_ETA.adam, O.GRID.at(-1));
  // The race read-out, part way and at the end.
  const mid = O.readout(4, O.raceInfo(runs, 30));
  assert.ok(mid.some((r) => /step 30/.test(r)), mid.join('|'));
  const end = O.readout(4, O.raceInfo(runs, O.raceLength(runs)));
  for (const kind of O.KINDS) assert.ok(end.some((r) => r.includes(`${O.LABEL[kind]}`) && r.includes(`at ${k[kind]}`)), end.join('|'));
  assert.equal(O.raceLength(runs), k.sgd);
  // Each ball stops where its run reached the target.
  const st = O.raceInfo(runs, 200);
  for (const kind of O.KINDS) assert.deepEqual(st.at[kind], runs[kind].path[k[kind]]);
});

test('RMSProp and Adam first steps, and RMSProp on the valley lined up with the axes', () => {
  const rms = O.valleyRun('rmsprop', { eta: O.VALLEY_ETA.rmsprop });
  const adam = O.valleyRun('adam', { eta: O.VALLEY_ETA.adam });
  const m1 = O.firstMove(rms);
  const a1 = O.firstMove(adam);
  // RMSProp: s starts at 0, so √s after one step is √0.001·|g| and each weight moves η/√0.001.
  for (const m of m1) assert.ok(Math.abs(m - 0.1 / Math.sqrt(0.001)) < 1e-6, String(m));
  // Adam's bias correction makes the first step η per weight (to within ε).
  for (const m of a1) assert.ok(Math.abs(m - 1) < 1e-6, String(m));
  const sv = O.storyValues();
  assert.equal(sv['rms-first'], '3.16');
  assert.equal(sv['adam-first'], '1.00');
  assert.equal(sv['rms-steps'], String(rms.reached));
  assert.equal(sv['adam-steps'], String(adam.reached));
  const aligned = O.valleyRun('rmsprop', { eta: O.VALLEY_ETA.rmsprop }, { angle: 0 });
  assert.ok(aligned.reached < rms.reached / 5);
  assert.equal(sv['rms-aligned'], String(aligned.reached));
  // Both read-outs quote the first step.
  assert.ok(O.readout(2, O.kindInfo('rmsprop')).includes('step 1: each weight ±3.16'));
  assert.ok(O.readout(3, O.kindInfo('adam')).includes('step 1: each weight ±1.00'));
});

// The page's own network runs, computed in odd-sized slices, shared below.
let NETS = null;
function pageNets() {
  if (NETS) return NETS;
  for (const kind of O.KINDS) while (!O.netWork(kind, 37));
  NETS = Object.fromEntries(O.KINDS.map((k) => [k, O.netRun(k)]));
  return NETS;
}

test('network curves deterministic', () => {
  const tr = O.trainSet();
  const te = O.testSet();
  const nets = pageNets();
  for (const kind of O.KINDS) {
    const d = trainer({ net: mlpInit([2, 4, 1], { hidden: 'tanh', init: 'xavier', seed: 155 }), X: tr.X, y: tr.y, batch: 20, kind, hyper: { eta: O.NET_ETA[kind] }, seed: 154 });
    d.step(1000);
    assert.deepEqual(nets[kind].losses, d.state.epochLoss, kind);
    assert.equal(nets[kind].losses.length, 101);
    assert.equal(nets[kind].test, accuracyNet(d.state.net, te.X, te.y));
    assert.equal(d.state.diverged, false);
    for (const v of nets[kind].losses) assert.ok(v > 0 && v < O.LOSS_MAX, `${kind} ${v}`);
  }
  // A second run gives the same numbers bit for bit.
  const again = trainer({ net: O.startNet(), X: tr.X, y: tr.y, batch: 20, kind: 'adam', hyper: { eta: O.NET_ETA.adam }, seed: 154 });
  again.step(1000);
  assert.deepEqual(again.state.epochLoss, nets.adam.losses);
});

test('network η from the grid and the story of step 6', () => {
  const nets = pageNets();
  for (const kind of O.KINDS) {
    const t = O.tuneNet(kind);
    assert.equal(t.eta, O.NET_ETA[kind], kind);
    assert.equal(t.loss, nets[kind].mean, kind);
    assert.equal(nets[kind].mean, nets[kind].losses.slice(1).reduce((a, v) => a + v, 0) / 100);
  }
  assert.equal(O.NET_ETA.sgd, O.GRID.at(-1));
  const sv = O.storyValues();
  for (const kind of O.KINDS) {
    assert.equal(sv[`net-${kind}`], O.num(nets[kind].final, 3));
    assert.equal(sv[`acc-${kind}`], O.pct(nets[kind].test));
  }
  // Order by final training loss: momentum, Adam, RMSProp, GD; test accuracy in the same order.
  const f = (k) => nets[k].final;
  assert.ok(f('momentum') < f('adam') && f('adam') < f('rmsprop') && f('rmsprop') < f('sgd'));
  const a = (k) => nets[k].test;
  assert.ok(a('momentum') > a('adam') && a('adam') > a('rmsprop') && a('rmsprop') > a('sgd'));
  // Every curve goes up between some epochs: they are jagged, not smooth.
  for (const kind of O.KINDS) {
    const L = nets[kind].losses;
    assert.ok(L.slice(1).some((v, i) => v > L[i]), kind);
  }
  // A large step can overshoot even with no batch noise: with full batches
  // RMSProp's loss still goes up between epochs; the other three do not.
  assert.ok(O.fullBatchUps('rmsprop') > 0);
  for (const kind of ['sgd', 'momentum', 'adam']) assert.equal(O.fullBatchUps(kind), 0, kind);
  assert.equal(sv['rms-full-ups'], String(O.fullBatchUps('rmsprop')));
  const rows = O.readout(5, O.netInfo());
  assert.equal(rows[0], 'train loss · test acc.');
  for (const kind of O.KINDS) assert.ok(rows.some((r) => r.includes(sv[`net-${kind}`]) && r.includes(sv[`acc-${kind}`])), rows.join('|'));
});

test('optimisers read-out rows and page', () => {
  const runs = O.raceRuns();
  const infos = [
    ...O.GD_ETAS.map((e) => [0, O.gdInfo(e)]),
    ...O.BETAS.map((b) => [1, O.momInfo(b)]),
    [2, O.kindInfo('rmsprop')], [3, O.kindInfo('adam')],
    [4, O.raceInfo(runs, 0)], [4, O.raceInfo(runs, 40)], [4, O.raceInfo(runs, 500)],
    [5, O.netInfo()],
  ];
  for (const [step, s] of infos) {
    const rows = O.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 6, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 26, r);
    }
  }
  assert.ok(O.readout(4, O.raceInfo(runs, 0)).some((r) => /press Train/.test(r)));
  assert.ok(O.readout(4, O.raceInfo(runs, 0)).includes('all four start together'));
  // The race's status row uses the shared wording.
  const end = O.raceLength(runs);
  assert.equal(O.readout(4, O.raceInfo(runs, 0)).at(-1), 'press Train');
  assert.equal(O.readout(4, O.raceInfo(runs, 40, 'training')).at(-1), 'training…');
  assert.equal(O.readout(4, O.raceInfo(runs, 40)).at(-1), 'not finished: press Train');
  assert.equal(O.readout(4, O.raceInfo(runs, 40, 'computing')).at(-1), 'computing…');
  assert.equal(O.readout(4, O.raceInfo(runs, end)).at(-1), 'finished');
  assert.equal(O.readout(4, O.raceInfo(runs, end, 'training')).at(-1), 'finished');
  for (const k of [0, 40, end]) assert.ok(O.readout(4, O.raceInfo(runs, k, 'training')).length <= 6);

  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Optimisers: momentum and Adam · ML Explorer<\/title>/);
  assert.match(html, /<h1>Optimisers: momentum and Adam<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  for (const s of [151, 152, 154, 155]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /Diverged: the steps grow instead of shrinking/);
  assert.match(html, /Play steps through the lesson, and Train starts the race/);
  assert.match(html, /rankings like these change from problem to problem/i);
  assert.match(html, /href="\/ml\/loss-surface\/"/);
  assert.match(html, /href="\/ml\/gradient-descent\/"/);
  assert.match(html, /href="\/ml\/hidden-layers\/"/);
  assert.match(html, /href="\/ml\/training-networks\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/deep-networks\/">Next lesson: Going deeper: vanishing gradients →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always better|always best|can learn anything|is the best optimiser/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  const pv = O.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 8);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const sv = O.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 20);
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('the story runs do not hang on rounding', () => {
  // Another JavaScript engine may round a Math function differently in the last
  // place. Nudging the start by 1e-12 stands in for that: the numbers the page
  // quotes must not move.
  const valley = [['sgd', { eta: 0.07 }], ['sgd', { eta: 0.075 }], ['sgd', { eta: 0.05 }],
    ['momentum', { eta: 0.02, beta: 0 }], ['momentum', { eta: 0.02, beta: 0.5 }], ['momentum', { eta: 0.02 }],
    ['rmsprop', { eta: 0.1 }], ['adam', { eta: 1 }], ['adam', { eta: 2 }]];
  for (const [kind, hyper] of valley) {
    for (const nudge of [1e-12, -1e-12]) {
      assert.equal(direct(kind, hyper, 2000, nudge).reached, O.valleyRun(kind, hyper).reached, `${kind} ${hyper.eta}`);
    }
  }
  const tr = O.trainSet();
  const nets = pageNets();
  for (const kind of O.KINDS) {
    const net = O.startNet();
    net.W[0][0][0] += 1e-12;
    const t = trainer({ net, X: tr.X, y: tr.y, batch: 20, kind, hyper: { eta: O.NET_ETA[kind] }, seed: 154 });
    t.step(1000);
    const gap = Math.max(...t.state.epochLoss.map((v, i) => Math.abs(v - nets[kind].losses[i])));
    assert.ok(gap < 1e-6, `${kind} ${gap}`);
    assert.equal(accuracyNet(t.state.net, O.testSet().X, O.testSet().y), nets[kind].test);
  }
});

test('network runs computed in slices equal the runs computed at once', async () => {
  const F = await import('../optimisers/stage.js?sliced');
  for (const kind of F.KINDS) {
    assert.equal(F.netReady(kind), false);
    const rows = F.readout(5, F.netInfo());
    assert.ok(rows.some((r) => /computing/.test(r)), rows.join('|'));
    let slices = 0;
    while (!F.netWork(kind, 11)) slices++;
    assert.ok(slices >= 80);
    assert.deepEqual(F.netRun(kind), pageNets()[kind]);
  }
  assert.ok(!F.readout(5, F.netInfo()).some((r) => /computing/.test(r)));
});
