import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { attnInit, attnForward, attnFlatten, attnUnflatten, attnParamCount, layerNorm } from '../assets/attn.js';
import * as S from '../transformers/stage.js';

const html = readFileSync(new URL('../transformers/index.html', import.meta.url), 'utf8');

// The reversal task written out again: 8 uniform draws per sequence, digit ⌊10u⌋,
// target the same digits in reverse order.
function reversal(n, seed) {
  const rng = mulberry32(seed);
  const seqs = [];
  const targets = [];
  for (let i = 0; i < n; i++) {
    const s = [];
    for (let t = 0; t < 8; t++) s.push(Math.floor(rng() * 10));
    seqs.push(s);
    targets.push([...s].reverse());
  }
  return { seqs, targets };
}

const argmax = (v) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);

// Per-position accuracy and per-head mirror share written out again from attnForward.
function evalDirect(model, set) {
  const right = new Array(8).fill(0);
  const anti = [0, 0];
  for (let i = 0; i < set.seqs.length; i++) {
    const f = attnForward(model, set.seqs[i]);
    for (let t = 0; t < 8; t++) {
      if (argmax(f.logits[t]) === set.targets[i][t]) right[t]++;
      for (let h = 0; h < 2; h++) if (argmax(f.A[0][h][t]) === 7 - t) anti[h]++;
    }
  }
  return { right, anti };
}

// Every starting weight times (1 ± 1e-12), the sign alternating.
const nudged = (m, eps) => attnUnflatten(m, attnFlatten(m).map((v, i) => v * (1 + (i % 2 ? -eps : eps))));

let LIVE = null;
function liveRun() {
  if (LIVE) return LIVE;
  const live = S.makeLive();
  while (!live.done()) live.advance(7);
  LIVE = live;
  return live;
}

test('data reproducible', () => {
  assert.equal(S.SEED_TRAIN, 281);
  assert.equal(S.SEED_TEST, 282);
  assert.equal(S.SEED_INIT, 283);
  assert.equal(S.SEED_SHUFFLE, 284);
  assert.equal(S.N_TRAIN, 2000);
  assert.equal(S.N_TEST, 500);
  assert.deepEqual(S.CFG, { vocab: 10, len: 8, d: 32, heads: 2, causal: false, ln: true, mlp: 64, out: 'softmax', seed: 283 });
  assert.deepEqual(S.trainSet(), reversal(2000, 281));
  assert.deepEqual(S.testSet(), reversal(500, 282));
  assert.deepEqual(S.testSet().seqs[0], S.example());
  assert.deepEqual(S.example(), S.LITERAL_EXAMPLE);
  assert.deepEqual(S.startModel().theta, attnInit(S.CFG).theta);
  assert.equal(S.STEPS, (S.EPOCHS * S.N_TRAIN) / S.BATCH);
});

test('permutation demo equals attnForward', () => {
  const noPos = S.withoutPositions(S.startModel());
  assert.ok(noPos.P.every((v) => v === 0));
  assert.ok(S.startModel().P.some((v) => v !== 0));
  for (let i = 0; i < 6; i++) {
    const perm = S.permutation(i);
    assert.deepEqual([...perm].sort(), [0, 1, 2, 3, 4, 5, 6, 7]);
    const demo = S.permDemo(i);
    const x = S.example();
    const px = perm.map((j) => x[j]);
    assert.deepEqual(demo.x, x);
    assert.deepEqual(demo.px, px);
    const a = attnForward(noPos, x).logits;
    const b = attnForward(noPos, px).logits;
    // Without positions: output at t of the shuffled input is the output at perm[t].
    let worst = 0;
    for (let t = 0; t < 8; t++) for (let k = 0; k < 10; k++) worst = Math.max(worst, Math.abs(b[t][k] - a[perm[t]][k]));
    assert.ok(worst < 1e-12, `worst ${worst}`);
    assert.equal(demo.diffNoPos, worst);
    // The read-out: top digit and its probability, from the same logits.
    for (let t = 0; t < 8; t++) {
      assert.equal(demo.out[t].digit, argmax(a[t]));
      assert.equal(demo.pout[t].digit, argmax(b[t]));
      assert.equal(demo.pout[t].digit, demo.out[perm[t]].digit);
      assert.equal(S.num(demo.pout[t].p, 3), S.num(demo.out[perm[t]].p, 3));
    }
    // With positions the same shuffle does more than move the outputs about.
    const wa = attnForward(S.startModel(), x).logits;
    const wb = attnForward(S.startModel(), px).logits;
    let w2 = 0;
    for (let t = 0; t < 8; t++) for (let k = 0; k < 10; k++) w2 = Math.max(w2, Math.abs(wb[t][k] - wa[perm[t]][k]));
    assert.equal(demo.diffPos, w2);
    if (perm.some((j, t) => j !== t)) assert.ok(w2 > 0.01, `with positions ${w2}`);
  }
  // The read-out's "< 10⁻¹²" holds for many more shuffles than are drawn above.
  for (let i = 6; i < 40; i++) {
    const d = S.permDemo(i);
    assert.ok(d.diffNoPos < 1e-12, `shuffle ${i}: ${d.diffNoPos}`);
    assert.match(S.readout(1, S.stageInfo(1, null, false, { perm: i })).join('|'), /gap (0|< 10⁻¹²)\)/);
  }
  // With a causal mask the shuffle no longer just moves the outputs (the page says so).
  const causal = S.withoutPositions(attnInit({ ...S.CFG, causal: true }));
  const pm = S.permutation(0);
  const ca = attnForward(causal, S.example()).logits;
  const cb = attnForward(causal, pm.map((j) => S.example()[j])).logits;
  let cg = 0;
  for (let t = 0; t < 8; t++) for (let k = 0; k < 10; k++) cg = Math.max(cg, Math.abs(cb[t][k] - ca[pm[t]][k]));
  assert.ok(cg > 0.01, `causal gap ${cg}`);
  // The example's repeated digit: same output at its three positions without positions,
  // though reversal needs three different digits there.
  const rep = S.repeatedDigit();
  assert.deepEqual(rep, { digit: 1, at: [1, 3, 5], need: [3, 7, 9] });
  const ex = attnForward(noPos, S.example()).logits;
  for (const t of rep.at) for (let k = 0; k < 10; k++) assert.ok(Math.abs(ex[t][k] - ex[rep.at[0]][k]) < 1e-12);
  assert.equal(S.pageValues()['rep-at'], '2, 4 and 6');
  assert.equal(S.pageValues()['rep-need'], '3, 7 and 9');
  // Equal digits get equal outputs without positions: the reason reversal needs them.
  const x = [3, 1, 4, 1, 5, 9, 2, 6];
  const l = attnForward(noPos, x).logits;
  for (let k = 0; k < 10; k++) assert.ok(Math.abs(l[1][k] - l[3][k]) < 1e-12);
});

test('layer norm read-out uses population variance', () => {
  for (let pos = 0; pos < 8; pos++) {
    const r = S.lnDemo(pos);
    assert.equal(r.x.length, 32);
    const mu = r.x.reduce((s, v) => s + v, 0) / 32;
    const v = r.x.reduce((s, x) => s + (x - mu) ** 2, 0) / 32;
    assert.ok(Math.abs(r.mean - mu) < 1e-12);
    assert.ok(Math.abs(r.var - v) < 1e-12);
    const m = S.startModel();
    const ref = layerNorm(r.x, m.blocks[0].g1, m.blocks[0].b1).y;
    r.y.forEach((y, j) => assert.ok(Math.abs(y - ref[j]) < 1e-12));
    // γ = 1 and β = 0 at the start: mean 0 and variance var/(var + ε), 1.000 to 3 places.
    assert.ok(Math.abs(r.meanY) < 1e-12);
    assert.ok(Math.abs(r.varY - v / (v + 1e-5)) < 1e-12);
    assert.equal(S.num(r.varY, 3), '1.000');
    // The first position's vector is token embedding plus position embedding.
    const x0 = S.example()[pos];
    for (let j = 0; j < 32; j++) assert.equal(r.x[j], m.E[x0 * 32 + j] + m.P[pos * 32 + j]);
  }
});

test('param count on the page equals attnParamCount and the formula', () => {
  const m = S.startModel();
  const parts = S.paramParts(S.CFG);
  const total = parts.reduce((s, p) => s + p.count, 0);
  assert.equal(total, attnParamCount(m));
  // The formula, written out: |V|d + Td + 2d + 4dHd_k + 2d + 2dm + m + d + 2d + |V|d + |V|.
  const V = 10, T = 8, d = 32, H = 2, dk = 16, mm = 64;
  assert.equal(total, V * d + T * d + 2 * d + 4 * d * H * dk + 2 * d + 2 * d * mm + mm + d + 2 * d + V * d + V);
  assert.equal(total, 9386);
  assert.equal(S.pageValues()['n-params'], '9,386');
  // Each part equals the matching slice of the model.
  const b = m.blocks[0];
  const size = { tok: m.E.length, pos: m.P.length, ln1: b.g1.length + b.b1.length, qkv: b.Wq.length + b.Wk.length + b.Wv.length, wo: b.Wo.length, ln2: b.g2.length + b.b2.length, mlp: b.W1.length + b.c1.length + b.W2.length + b.c2.length, lnf: m.gf.length + m.bf.length, out: m.Wout.length + m.bout.length };
  for (const p of parts) assert.equal(p.count, size[p.key], p.key);
  assert.equal(Object.keys(size).length, parts.length);
  // "Most of them sit in the attention weights and the small network."
  assert.ok(size.qkv + size.wo + size.mlp > total / 2);
});

test('training deterministic', () => {
  const a = S.makeTrainer();
  const b = S.makeTrainer();
  a.step(30);
  b.step(30);
  assert.deepEqual(a.state.model.theta, b.state.model.theta);
  a.reset();
  a.step(30);
  assert.deepEqual(a.state.model.theta, b.state.model.theta);
  const lv = liveRun();
  assert.equal(lv.trainer.state.iter, S.STEPS);
  assert.equal(lv.trainer.state.epoch, S.EPOCHS);
  assert.equal(lv.trainer.state.diverged, false);
  lv.advance(5);
  assert.equal(lv.trainer.state.iter, S.STEPS);
  // Test-set checks were taken before training and every EVAL_EVERY steps.
  assert.deepEqual(lv.history.map((h) => h.iter), Array.from({ length: S.STEPS / S.EVAL_EVERY + 1 }, (_, i) => i * S.EVAL_EVERY));
  // A second live run stepped in other chunk sizes ends in the same state.
  const lv2 = S.makeLive();
  while (!lv2.done()) lv2.advance(3);
  assert.deepEqual(lv2.trainer.state.model.theta, lv.trainer.state.model.theta);
  assert.deepEqual(lv2.history, lv.history);
});

test('test accuracy per position ≥ 0.95 after the run', () => {
  const end = liveRun().trainer.state.model;
  const d = evalDirect(end, S.testSet());
  const e = S.evaluate(end);
  assert.deepEqual(e.right, d.right);
  for (const r of d.right) assert.ok(r / 500 >= 0.95, String(r));
  const sv = S.storyValues();
  assert.equal(sv['right-min-end'], String(Math.min(...d.right)));
  assert.equal(sv['right-min-end'], '500');
  // The first test check with every answer right.
  const first = liveRun().history.find((h) => h.right.every((r) => r === 500));
  assert.equal(sv['all-right-at'], String(first.iter));
  assert.ok(liveRun().history.filter((h) => h.iter < first.iter).every((h) => h.right.some((r) => r < 500)));
  // At the start the model is at chance, near 1 in 10.
  const d0 = evalDirect(S.startModel(), S.testSet());
  assert.ok(Math.max(...d0.right) / 500 < 0.2);
  assert.equal(sv['acc-start'], S.pct(d0.right.reduce((s, v) => s + v, 0), 4000));
  // The loss: start near ln 10, the loss of 1 in 10 for every digit; end near 0.
  assert.ok(Math.abs(liveRun().trainer.state.epochLoss[0] - Math.log(10)) < 0.5);
  assert.equal(sv['loss-start'], S.num(liveRun().trainer.state.epochLoss[0], 2));
  assert.equal(sv['loss-end'], S.num(liveRun().trainer.state.epochLoss[S.EPOCHS], 2));
  // pct never shows 100% unless every one is right.
  assert.equal(S.pct(3998, 4000), '99.9%');
  assert.equal(S.pct(4000, 4000), '100%');
  assert.equal(S.pct(707, 4000), '17.7%');
});

test('anti-diagonal share', () => {
  const end = liveRun().trainer.state.model;
  const d = evalDirect(end, S.testSet());
  const d0 = evalDirect(S.startModel(), S.testSet());
  const sv = S.storyValues();
  for (const h of [0, 1]) {
    assert.equal(sv[`anti-${h + 1}-end`], S.count(d.anti[h]));
    assert.equal(sv[`anti-${h + 1}-start`], S.pct(d0.anti[h], 4000));
    // Mean weight on the mirror position, written out again.
    let w = 0;
    for (const x of S.testSet().seqs) {
      const A = attnForward(end, x).A[0][h];
      for (let t = 0; t < 8; t++) w += A[t][7 - t];
    }
    assert.equal(sv[`mirror-${h + 1}-end`], S.num(w / 4000, 2));
  }
  // The page says both heads end up looking at the mirror position.
  for (const h of [0, 1]) assert.ok(d.anti[h] / 4000 >= 0.95, String(d.anti[h]));
  for (const h of [0, 1]) assert.ok(d0.anti[h] / 4000 < 0.4, String(d0.anti[h]));
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const sv = S.storyValues();
  for (const eps of [1e-12, -1e-12]) {
    const m0 = nudged(S.startModel(), eps);
    const lv = S.makeLive(m0);
    while (!lv.done()) lv.advance(50);
    assert.deepEqual(S.runValues(m0, lv), sv, `eps ${eps}`);
  }
});

test('read-out rows', () => {
  const fresh = S.makeLive();
  const all = [];
  for (const lv of [fresh, liveRun()]) {
    for (let k = 0; k < 6; k++) all.push(S.readout(k, S.stageInfo(k, lv, false, { perm: 1, pos: 0, part: 2 })));
  }
  for (const rows of all) for (const r of rows) assert.ok(!/NaN|undefined|Infinity/.test(r), r);
  assert.match(S.readout(4, S.stageInfo(4, fresh, false, {})).join('|'), /press Train/);
  assert.match(S.readout(4, S.stageInfo(4, fresh, 'computing', {})).join('|'), /computing…/);
  assert.match(S.readout(4, S.stageInfo(4, liveRun(), false, {})).join('|'), /finished/);
  assert.match(S.readout(5, S.stageInfo(5, fresh, false, {})).join('|'), /9,386/);
  // Every box of the diagram has a one-line description.
  assert.equal(S.BLOCK_PARTS.length, 9);
  for (const p of S.BLOCK_PARTS) assert.ok(p.name && p.desc && p.desc.length < 95, p.name);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Transformers · ML Explorer<\/title>/);
  assert.match(html, /<h1>Transformers<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /data-lesson="transformers"/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/transformers\/">/);
  for (const s of [281, 282, 283, 284, 285]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /far larger scale/);
  assert.match(html, /href="\/ml\/attention\/"/);
  assert.match(html, /href="\/ml\/deep-networks\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/llms\/">Next lesson: Large language models, in miniature →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /\b(understands?|thinks?|knows?|reasons|creative|intelligent|conscious)\b/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /Why does a transformer need position embeddings\?/);
  assert.match(html, /What does a residual connection add\?/);
  assert.match(html, /1706\.03762/);
  assert.match(html, /population/);
  assert.match(html, /with no mask \(as here\)/);
  assert.match(html, /brighter cell is a larger weight/);
  assert.doesNotMatch(html, /darker cell/);
  assert.match(html, /the original transformer, and the block here, add a position embedding/);
  assert.match(html, /base model/);
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
