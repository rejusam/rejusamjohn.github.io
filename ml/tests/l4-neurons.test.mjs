import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn, centralDiff } from '../assets/maths.js';
import { sigmoid, logisticFit } from '../assets/classify.js';
import { mlpForward } from '../assets/nn.js';
import * as L from '../neurons/stage.js';

const html = readFileSync(new URL('../neurons/index.html', import.meta.url), 'utf8');
const LITERAL_BLOB_X0 = -1.4042174750078877;
const close = (u, v, tol = 1e-12) => assert.ok(Math.abs(u - v) < tol, `${u} vs ${v}`);

test('data reproducible', () => {
  assert.equal(L.SEED_BLOBS, 143);
  assert.equal(L.SEED_XOR, 142);
  // The two clouds, written out: class 0 rows first, each point = centre + 0.8 × (z1, z2).
  let rng = mulberry32(143);
  const X = [];
  const y = [];
  for (const [mu, cls] of [[[-1, -1], 0], [[1.2, 1], 1]]) {
    for (let i = 0; i < 30; i++) {
      const z1 = randn(rng);
      const z2 = randn(rng);
      X.push([mu[0] + 0.8 * z1, mu[1] + 0.8 * (0 * z1 + 1 * z2)]);
      y.push(cls);
    }
  }
  assert.deepEqual(L.blobsData(), { X, y });
  assert.deepEqual(L.blobsData(), L.blobsData());
  // The XOR data, written out: a corner (±1, ±1) at random, plus 0.35 × noise on each axis.
  rng = mulberry32(142);
  const X2 = [];
  const y2 = [];
  for (let i = 0; i < 80; i++) {
    const c = Math.floor(rng() * 4);
    const sx = c % 2 ? 1 : -1;
    const sy = c >> 1 ? 1 : -1;
    const px = sx + 0.35 * randn(rng);
    const py = sy + 0.35 * randn(rng);
    X2.push([px, py]);
    y2.push(sx !== sy ? 1 : 0);
  }
  assert.deepEqual(L.xorSet(), { X: X2, y: y2 });
  close(L.blobsData().X[0][0], LITERAL_BLOB_X0);
  assert.equal(y2.filter((v) => v === 1).length, 42);
});

test('sigmoid neuron equals logistic regression', () => {
  const { X, y } = L.blobsData();
  const f = L.fit();
  assert.equal(f.ok, true);
  assert.ok(f.gradNorm < 1e-8);
  assert.deepEqual(f, logisticFit(X, y));
  const net = { sizes: [2, 1], hidden: 'tanh', W: [[f.w]], b: [[f.w0]] };
  X.forEach((x) => {
    const { a } = L.neuron(f.w, f.w0, x, 'sigmoid');
    const p = sigmoid(f.w0 + f.w[0] * x[0] + f.w[1] * x[1]);
    assert.ok(Math.abs(a - p) <= 1e-15, `${a} vs ${p}`);
    assert.ok(Math.abs(a - sigmoid(mlpForward(net, x).out[0])) <= 1e-15);
  });
  // The fitted weights sit inside the sliders' range, and the data are not perfectly split.
  for (const v of [f.w[0], f.w[1], f.w0]) assert.ok(v > L.RANGE.min && v < L.RANGE.max, String(v));
  assert.ok(L.accuracyLine(X, y, f.w, f.w0) < 1);
  assert.equal(L.pct(L.accuracyLine(X, y, f.w, f.w0)), '95.0%');
});

test('best linear accuracy on xor data is at most 0.8', () => {
  const { X, y } = L.xorSet();
  const best = L.bestLinear(X, y);
  assert.ok(best.acc <= 0.8, String(best.acc));
  assert.equal(best.acc, 0.775);
  // The grid of 360 angles finds the same answer as checking every way a line can split the points.
  assert.equal(L.bestLinearExact(X, y), best.acc);
  // The line it reports really scores that accuracy.
  const w = [Math.cos(best.theta), Math.sin(best.theta)];
  const acc = L.accuracyLine(X, y, w.map((v) => best.side * v), -best.side * best.offset);
  assert.equal(acc, best.acc);
  // No slider setting does better.
  const vals = [-4, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 4];
  for (const a of vals) for (const c of vals) for (const b of vals) assert.ok(L.accuracyLine(X, y, [a, c], b) <= best.acc);
  // The page's number is the helper's.
  assert.equal(L.pageValues()['xor-best'], L.pct(best.acc));
  assert.equal(L.pageValues()['xor-best'], '77.5%');
  // The blobs, by contrast, are split almost perfectly by one line.
  assert.ok(L.bestLinear(L.blobsData().X, L.blobsData().y).acc >= 0.95);
});

test('every activation derivative matches centralDiff away from kinks', () => {
  assert.deepEqual(L.ACT_KEYS, ['step', 'sigmoid', 'tanh', 'relu']);
  for (const key of L.ACT_KEYS) {
    const g = L.ACTS[key];
    for (let z = -4.05; z <= 4; z += 0.3) {
      // The step and ReLU have a kink at 0; every z here is at least 0.05 from it.
      assert.ok(Math.abs(z) > 0.04);
      close(g.df(z), centralDiff(g.f, z), 1e-8);
    }
  }
  // σ′ = σ(1 − σ), tanh′ = 1 − tanh², ReLU′ = 1 for z > 0 else 0.
  close(L.ACTS.sigmoid.df(0), 0.25, 1e-15);
  close(L.ACTS.tanh.df(0.7), 1 - Math.tanh(0.7) ** 2, 1e-15);
  assert.equal(L.ACTS.relu.df(2), 1);
  assert.equal(L.ACTS.relu.df(-2), 0);
  assert.equal(L.ACTS.relu.f(-3), 0);
  assert.equal(L.ACTS.step.f(0.1), 1);
});

test('one neuron draws one straight line: the class flips exactly at z = 0', () => {
  for (const [w, b] of [[[1, -0.5], 0.5], [[0.3, 2], -1], [[-4, 4], 4]]) {
    // Points on the line w·x + b = 0 give z = 0; a small step either side flips the class.
    for (const t of [-2, 0, 1.5]) {
      const n = Math.hypot(w[0], w[1]);
      const p = [(-b * w[0]) / (n * n) - t * w[1], (-b * w[1]) / (n * n) + t * w[0]];
      close(L.neuron(w, b, p, 'sigmoid').z, 0, 1e-12);
      const up = [p[0] + 1e-3 * w[0], p[1] + 1e-3 * w[1]];
      const dn = [p[0] - 1e-3 * w[0], p[1] - 1e-3 * w[1]];
      for (const g of L.ACT_KEYS) {
        assert.ok(L.neuron(w, b, up, g).z > 0);
        assert.ok(L.neuron(w, b, dn, g).z < 0);
      }
    }
  }
});

test('the hand-set two-neuron preview solves xor', () => {
  const { X, y } = L.xorSet();
  const acc = L.handAccuracy();
  // Two lines beat the best one line by a wide margin.
  assert.ok(acc >= 0.9, String(acc));
  assert.ok(acc - L.bestLinear(X, y).acc > 0.15);
  // The preview is a real step network run by mlpForward.
  X.forEach((x, i) => {
    const out = mlpForward(L.HAND, x).out[0];
    assert.equal(L.handPredict(x), out > 0 ? 1 : 0, String(i));
  });
  assert.equal(L.pageValues()['hand-acc'], L.pct(acc));
  // Four groups: points in each quadrant.
  const g = L.xorGroups();
  assert.equal(g.reduce((s, v) => s + v, 0), 80);
  for (const v of g) assert.ok(v >= 15);
});

test('neurons read-out rows and page', () => {
  const states = [{}, { w: [4, 4], b: 4 }, { w: [-4, -4], b: -4 }, { w: [0, 0], b: 0 }, { act: 'relu' }, { act: 'step' }, { act: 'tanh', w: [0, 0], b: -4 }];
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
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>The artificial neuron · ML Explorer<\/title>/);
  assert.match(html, /<h1>The artificial neuron<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /seed 143/);
  assert.match(html, /seed 142/);
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /href="\/ml\/logistic-regression\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/hidden-layers\/">Next lesson: Hidden layers: bending the boundary →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always better/i);
  // Every filled value is already written in the page, and equals the computed one.
  const pv = L.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 12);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('neurons page values match the computations', () => {
  const pv = L.pageValues();
  const f = L.fit();
  const { X, y } = L.blobsData();
  const p = X[L.HIGHLIGHT];
  const z = L.neuron(L.START.w, L.START.b, p, 'sigmoid').z;
  assert.equal(pv['fit-w1'], L.num(f.w[0], 2));
  assert.equal(pv['fit-w2'], L.num(f.w[1], 2));
  assert.equal(pv['fit-b'], L.num(f.w0, 2));
  assert.equal(pv['fit-acc'], L.pct(L.accuracyLine(X, y, f.w, f.w0)));
  assert.equal(pv['pt-z'], L.num(z, 2));
  for (const g of L.ACT_KEYS) assert.equal(pv[`pt-${g}`], L.num(L.ACTS[g].f(z), 2));
  // The start point's z lies inside the activation plot.
  assert.ok(Math.abs(z) < 4);
  assert.equal(pv['n-xor'], '80');
  assert.equal(pv['xor-ones'], '42');
  assert.equal(pv['xor-major'], '52.5%');
  assert.equal(pv['xor-wrong'], '18');
  assert.equal(Number(pv['xor-wrong']), 80 - Math.round(L.bestLinear(L.xorSet().X, L.xorSet().y).acc * 80));
  // The ringed point is the second point of class 1, as About the data says.
  assert.equal(L.HIGHLIGHT, 31);
  assert.equal(y[L.HIGHLIGHT], 1);
  assert.equal(y[L.HIGHLIGHT - 1], 1);
  assert.equal(y.indexOf(1), L.HIGHLIGHT - 1);
  // The quiz quotes the computed best-line accuracy.
  const quiz = html.slice(html.indexOf('id="q2"'), html.indexOf('</form>', html.indexOf('id="q2"')));
  assert.equal((quiz.match(/77\.5%/g) || []).length, 2);
  assert.equal(pv['xor-best'], '77.5%');
  assert.equal(pv['n-blobs'], String(X.length));
  assert.equal(L.num(-0.004, 2), '0.00');
  assert.equal(L.num(-1.5, 2), '−1.50');
  assert.equal(L.pct(1), '100%');
});
