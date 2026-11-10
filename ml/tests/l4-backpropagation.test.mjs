import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { sigmoid } from '../assets/classify.js';
import { mlpForward, mlpBackward, bceLogit, flatten, unflatten, gradCheck, trainer, accuracyNet } from '../assets/nn.js';
import * as B from '../backpropagation/stage.js';

const html = readFileSync(new URL('../backpropagation/index.html', import.meta.url), 'utf8');
const LITERAL_XOR_X0 = 1.0618960991702895;

const lossAt = (net) => bceLogit(B.Y_EX, mlpForward(net, B.X_EX).out[0]);

test('data reproducible', () => {
  assert.deepEqual(B.X_EX, [1.0, -0.5]);
  assert.equal(B.Y_EX, 1);
  assert.equal(B.SEED_XOR, 161);
  assert.equal(B.N_XOR, 20);
  // The hand-chosen weights, one decimal each, as stated on the page.
  const net = B.startNet();
  assert.deepEqual(net.W, [[[0.5, -0.4], [-0.3, 0.8]], [[0.7, -0.6]]]);
  assert.deepEqual(net.b, [[0.1, -0.2], [0.2]]);
  assert.equal(net.hidden, 'sigmoid');
  assert.deepEqual(net.sizes, [2, 2, 1]);
  for (const v of flatten(net)) assert.equal(Number(v.toFixed(1)), v);
  // XOR clumps, written out: a corner (±1, ±1) picked uniformly, plus 0.35 × noise
  // on each coordinate; class 1 where the corner's signs differ.
  const xor = (n, seed) => {
    const rng = mulberry32(seed);
    const X = [];
    const y = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(rng() * 4);
      const sx = c % 2 ? 1 : -1;
      const sy = c >> 1 ? 1 : -1;
      X.push([sx + 0.35 * randn(rng), sy + 0.35 * randn(rng)]);
      y.push(sx !== sy ? 1 : 0);
    }
    return { X, y };
  };
  assert.deepEqual(B.xorSet(), xor(20, 161));
  assert.deepEqual(B.xorSet(), B.xorSet());
  assert.equal(B.xorSet().X[0][0], LITERAL_XOR_X0);
});

test('forward pass values match nn.js and the diagram\'s 3 significant figures', () => {
  const net = B.startNet();
  const f = B.forwardPass(net, B.X_EX, B.Y_EX);
  const ref = mlpForward(net, B.X_EX);
  assert.deepEqual(f.z1, ref.zs[0]);
  assert.deepEqual(f.a1, ref.as[1]);
  assert.equal(f.z2, ref.out[0]);
  assert.equal(f.p, sigmoid(ref.out[0]));
  assert.equal(f.loss, lossAt(net));
  // The loss is binary cross-entropy: −ln p̂ when y = 1.
  assert.ok(Math.abs(f.loss + Math.log(f.p)) < 1e-15);
  // Hand check of hidden unit 1: z = 0.5 × 1 − 0.4 × (−0.5) + 0.1 = 0.8.
  assert.ok(Math.abs(f.z1[0] - 0.8) < 1e-15);
  assert.ok(Math.abs(f.z1[1] + 0.9) < 1e-15);
  assert.deepEqual(
    [f.z1[0], f.z1[1], f.a1[0], f.a1[1], f.z2, f.p, f.loss].map(B.sig3),
    ['0.800', '−0.900', '0.690', '0.289', '0.510', '0.625', '0.470'],
  );
  assert.equal(B.sig3(-0.0025842), '−0.00258');
  assert.equal(B.sig3(1), '1.00');
  assert.equal(B.sig3(-0), '0.00');
});

test('delta at the output is p̂ − y', () => {
  const net = B.startNet();
  const f = B.forwardPass(net, B.X_EX, B.Y_EX);
  const g = B.backwardPass(net, B.X_EX, B.Y_EX);
  assert.equal(g.delta2, f.p - B.Y_EX);
  assert.equal(B.sig3(g.delta2), '−0.375');
  // And it is ∂ℒ/∂z⁽²⁾: a central difference in the logit agrees.
  const h = 1e-6;
  const num = (bceLogit(1, f.z2 + h) - bceLogit(1, f.z2 - h)) / (2 * h);
  assert.ok(Math.abs(num - g.delta2) < 1e-9);
  // For y = 0 it is p̂ too.
  const g0 = B.backwardPass(net, B.X_EX, 0);
  assert.ok(Math.abs(g0.delta2 - f.p) < 1e-15);
  // The hidden deltas follow the backward rule δ⁽¹⁾ = (W⁽²⁾ᵀ δ⁽²⁾) ⊙ σ′(z⁽¹⁾).
  for (let j = 0; j < 2; j++) {
    const s = sigmoid(f.z1[j]);
    assert.ok(Math.abs(g.delta1[j] - net.W[1][0][j] * g.delta2 * s * (1 - s)) < 1e-15);
  }
  assert.deepEqual(g.delta1.map(B.sig3), ['−0.0562', '0.0463']);
});

test('every gradient on the diagram passes gradCheck < 1e-6', () => {
  const net = B.startNet();
  const g = B.backwardPass(net, B.X_EX, B.Y_EX);
  const grad = B.flatGrad(g);
  const params = flatten(net);
  assert.equal(grad.length, 9);
  const worst = gradCheck((v) => lossAt(unflatten(net, v)), params, grad);
  assert.ok(worst < 1e-6, String(worst));
  // The lesson's own sums give exactly what nn.js's mlpBackward gives.
  const ref = mlpBackward(net, B.X_EX, B.Y_EX);
  const refFlat = flatten({ ...net, W: ref.dW, b: ref.db });
  grad.forEach((v, i) => assert.ok(Math.abs(v - refFlat[i]) < 1e-15, String(i)));
  // Each link's gradient is (δ at its output end) × (a at its input end), and none is 0.
  const f = B.forwardPass(net, B.X_EX, B.Y_EX);
  assert.equal(B.LINKS.length, 6);
  for (const k of B.LINKS) {
    const d = k.l === 1 ? g.delta2 : g.delta1[k.j];
    const a = k.l === 1 ? f.a1[k.i] : B.X_EX[k.i];
    assert.equal(B.linkGrad(net, k), d * a);
    assert.ok(Math.abs(B.linkGrad(net, k) - ref.dW[k.l][k.j][k.i]) < 1e-15);
    assert.ok(Math.abs(B.linkGrad(net, k)) > 0.01, k.id);
  }
  assert.deepEqual(B.LINKS.map((k) => B.sig3(B.linkGrad(net, k))), ['−0.0562', '0.0281', '0.0463', '−0.0231', '−0.259', '−0.108']);
  // Links are in the flatten order of the weights, so an index means the same thing everywhere.
  B.LINKS.forEach((k, n) => assert.equal(flatten(B.withWeight(net, k, 99)).indexOf(99), [0, 1, 2, 3, 6, 7][n]));
  // Accessible names say which weight it is.
  assert.equal(B.LINKS[4].name, 'weight from h1 to output');
});

test('nudge table: real change and linear estimate agree to within 2% for ±0.01', () => {
  const net = B.startNet();
  assert.equal(B.NUDGE, 0.01);
  for (const k of B.LINKS) {
    const rows = B.nudgeTable(net, k);
    assert.deepEqual(rows.map((r) => r.h), [0.01, -0.01]);
    for (const r of rows) {
      // Computed both ways: the loss after a real nudge, and gradient × nudge.
      const real = lossAt(B.withWeight(net, k, net.W[k.l][k.j][k.i] + r.h)) - lossAt(net);
      assert.equal(r.real, real);
      assert.equal(r.est, B.linkGrad(net, k) * r.h);
      assert.ok(Math.abs(r.real - r.est) <= 0.02 * Math.abs(r.est), `${k.id} ${r.h}`);
    }
  }
  // The read-out shows them at 3 significant figures, as the page's example says.
  const rows = B.nudgeTable(net, B.LINKS[4]);
  assert.deepEqual(rows.map((r) => [B.sig3(r.real), B.sig3(r.est)]), [['−0.00258', '−0.00259'], ['0.00260', '0.00259']]);
  const pv = B.pageValues();
  assert.equal(pv['nudge-real'], '−0.00258');
  assert.equal(pv['nudge-est'], '−0.00259');
  // withWeight never changes the network it is given.
  B.withWeight(net, B.LINKS[0], 5);
  assert.deepEqual(net.W, B.startNet().W);
});

test('one step lowers the loss', () => {
  const net = B.startNet();
  assert.equal(B.ETA, 3);
  const r = B.oneStep(net, B.X_EX, B.Y_EX, B.ETA);
  const g = B.flatGrad(B.backwardPass(net, B.X_EX, B.Y_EX));
  const want = flatten(net).map((p, i) => p - B.ETA * g[i]);
  assert.deepEqual(flatten(r.net), want);
  assert.equal(r.before, lossAt(net));
  assert.equal(r.after, lossAt(r.net));
  assert.ok(r.after < r.before);
  // The same as one step of nn.js's plain gradient descent trainer on this one example.
  const t = trainer({ net, X: [B.X_EX], y: [B.Y_EX], batch: 1, kind: 'sgd', hyper: { eta: 3 }, seed: 1 });
  t.step(1);
  flatten(t.state.net).forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-15));
  const pv = B.pageValues();
  assert.equal(pv['loss'], '0.470');
  assert.equal(pv['step-loss'], B.sig3(r.after));
  assert.equal(pv['step-loss'], '0.0875');
});

test('200 steps on the xor set lower the loss', () => {
  const D = B.xorSet();
  const t = B.makeRun();
  assert.equal(B.XOR_STEPS, 200);
  while (!B.finished(t)) B.advance(t, 7);
  assert.equal(t.state.iter, 200);
  // Full batch: one step is one pass through the 20 points, so epochLoss has one entry a step.
  const e = t.state.epochLoss;
  assert.equal(e.length, 201);
  assert.ok(e[200] < e[0]);
  // "The loss falls at every one of the 200 steps."
  for (let i = 1; i <= 200; i++) assert.ok(e[i] < e[i - 1], String(i));
  // "Ends with every training point right", first reached at step 174.
  assert.equal(accuracyNet(t.state.net, D.X, D.y), 1);
  const sv = B.storyValues();
  assert.equal(sv['xor-loss0'], '0.766');
  assert.equal(sv['xor-loss'], '0.125');
  assert.equal(sv['xor-acc0'], '20.0%');
  assert.equal(sv['xor-acc'], '100%');
  assert.equal(sv['xor-first'], '174');
  const u = B.makeRun();
  let first = null;
  while (!B.finished(u)) {
    B.advance(u, 1);
    if (first === null && accuracyNet(u.state.net, D.X, D.y) === 1) first = u.state.iter;
  }
  assert.equal(String(first), sv['xor-first']);
  // The run is plain gradient descent from the same hand-chosen weights.
  const direct = trainer({ net: B.startNet(), X: D.X, y: D.y, batch: 20, kind: 'sgd', hyper: { eta: 3 }, seed: B.SEED_SHUFFLE });
  direct.step(200);
  assert.deepEqual(direct.state.epochLoss, e);
  // Every value fits the loss plot.
  for (const v of e) assert.ok(v > 0 && v < B.LOSS_MAX);
});

test('the run is deterministic, paced and resets', () => {
  const a = B.makeRun();
  const b = B.makeRun();
  while (!B.finished(a)) B.advance(a, 3);
  while (!B.finished(b)) B.advance(b, 50);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
  B.advance(a, 10);
  assert.equal(a.state.iter, B.XOR_STEPS);
  a.reset();
  assert.equal(a.state.iter, 0);
  assert.deepEqual(a.state.net.W, B.startNet().W);
  const c = B.makeRun();
  B.frameStep(c, 100);
  assert.equal(c.state.iter, B.PACE);
});

test('backpropagation read-out rows and page', () => {
  const net = B.startNet();
  const t = B.makeRun();
  const runs = [B.runInfo(t), (B.advance(t, 50), B.runInfo(t)), (B.advance(t, 500), B.runInfo(t))];
  for (let step = 0; step < 6; step++) {
    for (const link of B.LINKS) {
      for (const nudge of [0, 1, -1]) {
        for (const run of runs) {
          const rows = B.readout(step, { net, link, nudge, run });
          assert.ok(rows.length >= 1 && rows.length <= 5);
          for (const r of rows) {
            assert.ok(!/NaN|undefined|Infinity/.test(r), r);
            assert.ok(r.length <= 26, r);
          }
        }
      }
    }
  }
  // Step 6 says when its run is not finished, and stops saying so at the end.
  assert.ok(B.readout(5, { net, link: B.LINKS[4], nudge: 0, run: runs[0] }).some((r) => /not finished/.test(r)));
  assert.ok(!B.readout(5, { net, link: B.LINKS[4], nudge: 0, run: runs[2] }).some((r) => /not finished/.test(r)));
  assert.match(html, /the numbers in this paragraph are for the end of the 200-step run/);
  assert.match(html, /Play steps through the lesson, and Train trains the network/);
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Backpropagation · ML Explorer<\/title>/);
  assert.match(html, /<h1>Backpropagation<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /seed 161/);
  assert.match(html, /σ here is the sigmoid, not a standard deviation/);
  assert.match(html, /binary cross-entropy/);
  assert.match(html, /log-loss/);
  assert.match(html, /href="\/ml\/hidden-layers\/"/);
  assert.match(html, /href="\/ml\/gradient-descent\/"/);
  assert.match(html, /href="\/ml\/logistic-regression\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/training-networks\/">Next lesson: Training a network →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always better|can learn anything/i);
  // The weights written in step 1's maths are the network's.
  assert.ok(html.includes('\\begin{pmatrix} 0.5 &amp; -0.4 \\\\ -0.3 &amp; 0.8 \\end{pmatrix}'));
  assert.ok(html.includes('\\mathbf{W}^{(2)} = \\begin{pmatrix} 0.7 &amp; -0.6 \\end{pmatrix}, \\quad b^{(2)} = 0.2'));
  // The cost comparison is an approximate rule, not a constant.
  assert.match(html, /small multiple of/);
  // Values filled at run time are already in the page and equal the computed ones.
  const pv = B.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 10);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const sv = B.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 5);
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
