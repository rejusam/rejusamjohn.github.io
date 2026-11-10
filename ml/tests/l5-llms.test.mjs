import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { ALPHABET, WORLD, checkFact } from '../assets/world.js';
import { ngramModel, nextProbs, crossEntropyBits, charTokenise, softmax, sampleIndex } from '../assets/text.js';
import { attnInit, attnForward, attnFlatten, attnUnflatten, attnParamCount, attnTrainer } from '../assets/attn.js';
import * as S from '../llms/stage.js';

const html = readFileSync(new URL('../llms/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../llms/stage.js', import.meta.url), 'utf8');

// The invented world's text written out again from its rule: per sentence one uniform
// draw r, then a person; r < 0.4 a "likes" sentence, r < 0.8 a "lives" sentence, else
// "<p> and <q> are friends." with q drawn until it differs from p.
function rewrite(n, seed) {
  const rng = mulberry32(seed);
  const pick = (xs) => xs[Math.floor(rng() * xs.length)];
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = rng();
    const p = pick(WORLD.people);
    if (r < 0.4) out.push(`${p} likes ${WORLD.likes[p]}.`);
    else if (r < 0.8) {
      const pl = WORLD.livesIn[p];
      out.push(`${p} lives ${WORLD.prep[pl]} the ${pl}.`);
    } else {
      let q = p;
      while (q === p) q = pick(WORLD.people);
      out.push(`${p} and ${q} are friends.`);
    }
  }
  return out.join(' ');
}

const ids = (s) => Array.from(s).map((c) => ALPHABET.indexOf(c));
const TRAIN = ids(rewrite(2000, 261));
const HELD = ids(rewrite(300, 262));

// log-softmax written out again.
const logSoftmax = (z) => {
  const m = Math.max(...z);
  const l = m + Math.log(z.reduce((s, v) => s + Math.exp(v - m), 0));
  return z.map((v) => v - l);
};

// Held-out bits written out again: windows of 32 starting at character 5 (0-based 4), each
// scoring its 32 next characters, so the positions 5 .. 5 + 32·195 − 1 are scored once each.
function directBits(model) {
  let sum = 0;
  let n = 0;
  for (let s = 4; s + 33 <= HELD.length; s += 32) {
    const { logits } = attnForward(model, HELD.slice(s, s + 32));
    for (let t = 0; t < 32; t++) { sum -= logSoftmax(logits[t])[HELD[s + t + 1]] / Math.LN2; n++; }
  }
  return { bits: sum / n, n };
}

// Probability that sampling at τ continues `prompt` with exactly `cont`, written out again
// one character at a time.
function directCont(model, prompt, cont, tau) {
  let xs = ids(prompt);
  let p = 1;
  for (const ch of cont) {
    const z = attnForward(model, xs).logits.at(-1);
    p *= softmax(z, tau)[ALPHABET.indexOf(ch)];
    xs = [...xs, ALPHABET.indexOf(ch)];
  }
  return p;
}

const nudged = (model, eps) => attnUnflatten(model, attnFlatten(model).map((v, i) => v * (1 + (i % 2 ? -eps : eps))));
const RUN = S.fullRun();
const SV = S.storyValues();

test('data reproducible', () => {
  assert.equal(S.trainText(), rewrite(2000, 261));
  assert.equal(S.heldText(), rewrite(300, 262));
  assert.equal(S.trainText().length, 42030);
  assert.equal(S.heldText().length, 6262);
  assert.deepEqual([S.TRAIN_SEED, S.HELD_SEED, S.SAMPLER_SEED, S.INIT_SEED, S.SAMPLE_SEED, S.FACT_SEED, S.CONT_SEED], [261, 262, 291, 292, 293, 294, 295]);
  assert.deepEqual([S.LEN, S.BATCH, S.ETA, S.RUN, S.EVAL_EVERY], [32, 8, 0.003, 2500, 100]);
  assert.deepEqual(S.CFG, { vocab: 28, len: 32, d: 32, heads: 2, causal: true, ln: true, mlp: 64, blocks: 1, out: 'softmax', seed: 292 });
  assert.deepEqual(S.TAUS, [0, 0.5, 1, 1.5]);
  // Held-out windows: 195 of 32, from character 5 on.
  const W = S.heldWindows();
  assert.equal(W.seqs.length, 195);
  W.seqs.forEach((xs, w) => {
    assert.deepEqual(xs, HELD.slice(4 + 32 * w, 36 + 32 * w));
    assert.deepEqual(W.targets[w], HELD.slice(5 + 32 * w, 37 + 32 * w));
  });
  assert.equal(S.scoredCount(), 6240);
  // The training batches: 8 random windows of 33 characters from one mulberry32(291) stream.
  const a = S.sampleBatch(mulberry32(291));
  const rng = mulberry32(291);
  for (let b = 0; b < 8; b++) {
    const s = Math.floor(rng() * (42030 - 32));
    assert.deepEqual(a.seqs[b], TRAIN.slice(s, s + 32));
    assert.deepEqual(a.targets[b], TRAIN.slice(s + 1, s + 33));
  }
  // One literal value: the first held-out window.
  assert.equal(W.seqs[0].map((i) => ALPHABET[i]).join(''), rewrite(300, 262).slice(4, 36));
  assert.equal(rewrite(300, 262).slice(0, 20), 'mia lives in the tow');
  assert.deepEqual(attnFlatten(S.startModel()), attnFlatten(attnInit(S.CFG)));
});

test('held-out bits: the same positions for the model and every n-gram line', () => {
  const m0 = S.startModel();
  const d = directBits(m0);
  assert.equal(d.n, 6240);
  assert.ok(Math.abs(S.heldBits(m0) - d.bits) < 1e-12);
  // The same number as the trainer's own held-out loss (nats), divided by ln 2.
  const tr = attnTrainer({ model: m0, data: { heldOut: S.heldWindows() }, batch: 8, kind: 'adam', hyper: { eta: 0.003 }, seed: 291, sampler: S.sampleBatch, evalEvery: 100 });
  tr.step(100);
  assert.ok(Math.abs(tr.state.evalLoss[0] / Math.LN2 - d.bits) < 1e-12);
  assert.ok(Math.abs(tr.state.evalLoss[1] / Math.LN2 - S.heldBits(tr.state.model)) < 1e-12);
  // n-gram lines: written out again over the same 6,240 positions.
  const text = rewrite(2000, 261);
  const held = rewrite(300, 262);
  for (const n of [1, 3, 5]) {
    const m = ngramModel(text, n);
    let sum = 0;
    for (let i = 5; i < 5 + 6240; i++) sum -= Math.log2(nextProbs(m, held.slice(i - n, i))[HELD[i]]);
    assert.ok(Math.abs(S.ngramBits(n) - sum / 6240) < 1e-12, `n ${n}`);
    // Rounded as the pages show them, the same as the previous lesson's numbers.
    assert.equal(S.ngramBits(n).toFixed(2), crossEntropyBits(m, held).toFixed(2), `n ${n}`);
  }
});

test('training deterministic', () => {
  const a = S.makeLive();
  const b = S.makeLive();
  // b in chunks of 7 units, a one unit at a time, to the same step.
  while (b.trainer.state.iter < 300 || !b.idle()) b.work(7);
  const end = b.trainer.state.iter;
  assert.ok(end >= 300 && end < 310);
  while (a.trainer.state.iter < end || !a.idle()) a.work(1);
  assert.equal(a.trainer.state.iter, end);
  assert.deepEqual(attnFlatten(a.trainer.state.model), attnFlatten(b.trainer.state.model));
  assert.deepEqual(a.curve, b.curve);
  assert.deepEqual(a.samples, b.samples);
  // The same as the trainer stepped on its own, and each curve point is heldBits of the
  // model at that step.
  const t = S.makeTrainer();
  const pts = [S.heldBits(t.state.model)];
  for (let k = 0; k < 3; k++) { t.step(100); pts.push(S.heldBits(t.state.model)); }
  t.step(end - 300);
  assert.deepEqual(attnFlatten(t.state.model), attnFlatten(a.trainer.state.model));
  assert.deepEqual(a.curve.map((c) => c.iter), [0, 100, 200, 300]);
  // Summed in slices of windows, so equal to the last bits only.
  a.curve.forEach((c, i) => assert.ok(Math.abs(c.bits - pts[i]) < 1e-12, String(c.iter)));
  // Reset gives the same run again.
  a.reset();
  assert.deepEqual(a.curve, []);
  while (a.trainer.state.iter < end || !a.idle()) a.work(1);
  assert.deepEqual(a.curve, b.curve);
  assert.deepEqual(a.samples, b.samples);
  // The full run: every 100 steps a point and a sample, 26 of each.
  assert.equal(RUN.trainer.state.iter, 2500);
  assert.equal(RUN.done(), true);
  assert.equal(RUN.trainer.state.diverged, false);
  assert.deepEqual(RUN.curve.map((c) => c.iter), Array.from({ length: 26 }, (_, i) => i * 100));
  assert.deepEqual(RUN.samples.map((c) => c.iter), Array.from({ length: 26 }, (_, i) => i * 100));
});

test('held-out bits per character after the run is below the n = 1 baseline', () => {
  const end = RUN.curve.at(-1).bits;
  const [n1, n3, n5] = [1, 3, 5].map(S.ngramBits);
  assert.ok(end < n1);
  // Reported as computed; no win over the n = 5 line is required. The page's sentences:
  assert.equal(SV['bits-start'], RUN.curve[0].bits.toFixed(2));
  assert.ok(RUN.curve[0].bits > Math.log2(28));        // the random start is worse than a uniform guess
  const firstBelow = (b) => RUN.curve.find((c) => c.bits < b).iter;
  assert.equal(SV['below-1-at'], S.count(firstBelow(n1)));
  const i1 = RUN.curve.findIndex((c) => c.bits < n1);
  assert.ok(RUN.curve.slice(i1).every((c) => c.bits < n1));
  assert.equal(SV['below-3-at'], S.count(firstBelow(n3)));
  // After it first drops below the 3-gram line it stays below it.
  const i3 = RUN.curve.findIndex((c) => c.bits < n3);
  assert.ok(RUN.curve.slice(i3).every((c) => c.bits < n3));
  assert.equal(SV['bits-end'], end.toFixed(3));
  assert.equal(SV['ngram-5-3'], n5.toFixed(3));
  // At the end just above the n = 5 line, after first dipping below it.
  assert.ok(end > n5 && end - n5 < 0.005);
  assert.equal((html.match(/ends just above the n = 5 counting line/g) || []).length, 2);
  assert.doesNotMatch(html, /ends level/);
  // Samples are claimed to reach words, not sentences.
  assert.doesNotMatch(html, /noise (to|into) sentences/);
  assert.equal(SV['dip-at'], S.count(RUN.curve.find((c) => c.bits < n5).iter));
  assert.ok(RUN.curve.find((c) => c.bits < n5).iter < 2500);
});

test('samples go from noise to the words of the invented world', () => {
  // Written out again: complete words (first and last dropped) found in the training text.
  const vocab = new Set(rewrite(2000, 261).split(/[ .]+/).filter(Boolean));
  const share = (t) => { const w = t.split(/[ .]+/).slice(1, -1).filter(Boolean); return w.filter((x) => vocab.has(x)).length / w.length; };
  const s = RUN.samples.map((x) => share(x.text));
  s.forEach((v, i) => assert.ok(Math.abs(v - S.worldWordShare(RUN.samples[i].text)) < 1e-15));
  assert.equal(s[0], 0);
  assert.equal(SV['words-start'], '0%');
  const from = (level) => { let i = s.length; while (i > 0 && s[i - 1] >= level) i--; return RUN.samples[i].iter; };
  assert.equal(SV['words-90-from'], S.count(from(0.9)));
  assert.equal(SV['words-all-from'], S.count(from(1)));
  assert.ok(s[s.findIndex((v, i) => RUN.samples[i].iter === from(0.9)) - 1] < 0.9);
  assert.equal(s.at(-1), 1);
});

test('true-fact probability rises across checkpoints', () => {
  const cps = S.checkpointFacts(RUN);
  assert.deepEqual(cps.map((c) => c.iter), S.CHECKPOINTS);
  assert.deepEqual(S.CHECKPOINTS, [500, 1000, 1500, 2000, 2500]);
  for (let i = 1; i < cps.length; i++) {
    assert.ok(cps[i].likes.t > cps[i - 1].likes.t, `likes ${cps[i].iter}`);
    assert.ok(cps[i].lives.t > cps[i - 1].lives.t, `lives ${cps[i].iter}`);
  }
  for (const c of cps) {
    assert.equal(SV[`food-${c.iter}`], c.likes.t.toFixed(2));
    assert.equal(SV[`place-${c.iter}`], c.lives.t.toFixed(2));
  }
  // Shape before facts: at step 500 most of the probability is on well-formed food sentences,
  // and the true one has less than a third of that.
  assert.equal(SV['form-first'], (cps[0].likes.t + cps[0].likes.f).toFixed(2));
  assert.ok(cps[0].likes.t + cps[0].likes.f > 0.7 && cps[0].likes.t < (cps[0].likes.t + cps[0].likes.f) / 3);
  // At the end the true food is likely, and a false place more likely than the true one.
  const end = cps.at(-1);
  assert.ok(end.likes.t > 0.8);
  assert.ok(end.lives.f > end.lives.t);
  assert.equal(SV['food-false-end'], end.likes.f.toFixed(2));
  assert.equal(SV['place-false-end'], end.lives.f.toFixed(2));
  // The place: the right "by" or "in" is likely, but between the two places that share it
  // the true one gets about half.
  const split = S.placeSplit(RUN.trainer.state.model);
  assert.ok(split.prep > 0.9);
  assert.ok(split.pair > 0.4 && split.pair < 0.6);
  assert.equal(SV['prep-end'], split.prep.toFixed(2));
  assert.equal(SV['pair-end'], split.pair.toFixed(2));
  // Temperature at the end: the true sentence gets less probability as τ rises from 0.5.
  const model = RUN.trainer.state.model;
  const byTau = S.factProbs(model, S.TAUS);
  for (const kind of ['likes', 'lives']) {
    assert.ok(byTau[1][kind].t > byTau[2][kind].t && byTau[2][kind].t > byTau[3][kind].t, kind);
    // ... and text that is not a fact sentence at all gets more.
    assert.ok(byTau[1][kind].o < byTau[2][kind].o && byTau[2][kind].o < byTau[3][kind].o, kind);
  }
  // Food: false sentences more likely at every step up from τ = 0.5.
  assert.ok(byTau[1].likes.f < byTau[2].likes.f && byTau[2].likes.f < byTau[3].likes.f);
  // Places: from τ = 1 to 1.5 the false share falls; the probability goes to broken text.
  assert.ok(byTau[3].lives.f < byTau[2].lives.f);
  S.TAUS.forEach((tau, i) => {
    if (i === 0) return;
    const r = byTau[i];
    assert.deepEqual(['ft', 'ff', 'fo', 'pt', 'pf', 'po'].map((k) => SV[`tau-${i}-${k}`]),
      [r.likes.t, r.likes.f, r.likes.o, r.lives.t, r.lives.f, r.lives.o].map((x) => Math.max(0, x).toFixed(2)), `τ ${tau}`);
  });
  // τ = 0: one sentence per prompt, always true or false; the counts out of 8.
  for (const r of byTau[0].per) for (const kind of ['likes', 'lives']) assert.equal(r[kind].t + r[kind].f, 1);
  assert.equal(SV['greedy-food'], String(byTau[0].per.filter((r) => r.likes.t === 1).length));
  assert.equal(SV['greedy-place'], String(byTau[0].per.filter((r) => r.lives.t === 1).length));
  assert.equal(Number(SV['greedy-food']) / 8, byTau[0].likes.t);
  assert.equal(Number(SV['greedy-place']) / 8, byTau[0].lives.t);
  // "a false one for the rest": the other greedy sentences are false, for foods and places.
  assert.equal(byTau[0].per.filter((r) => r.likes.f === 1).length, 8 - Number(SV['greedy-food']));
  assert.equal(byTau[0].per.filter((r) => r.lives.f === 1).length, 8 - Number(SV['greedy-place']));
});

test('fact probabilities are exact and match sampling', () => {
  const model = RUN.trainer.state.model;
  const [fp] = S.factProbs(model, [1]);
  // One prompt written out again.
  const p = WORLD.people[3];
  const t = directCont(model, `${p} likes `, `${WORLD.likes[p]}.`, 1);
  const f = WORLD.foods.filter((x) => x !== WORLD.likes[p]).reduce((s, x) => s + directCont(model, `${p} likes `, `${x}.`, 1), 0);
  assert.ok(Math.abs(fp.per[3].likes.t - t) < 1e-12);
  assert.ok(Math.abs(fp.per[3].likes.f - f) < 1e-12);
  for (const kind of ['likes', 'lives']) {
    assert.ok(Math.abs(fp[kind].t + fp[kind].f + fp[kind].o - 1) < 1e-12);
    assert.ok(fp[kind].o > -1e-12);
    const mean = fp.per.reduce((s, r) => s + r[kind].t, 0) / 8;
    assert.ok(Math.abs(fp[kind].t - mean) < 1e-12);
  }
  // At τ = 0 every probability is 0 or 1: the greedy sentence.
  const [g] = S.factProbs(model, [0]);
  for (const r of g.per) for (const kind of ['likes', 'lives']) assert.ok([0, 1].includes(r[kind].t) && [0, 1].includes(r[kind].f));
  // Sampling agrees: over 60 presses (960 sentences) the share of true sentences is within
  // four standard errors of the exact mean.
  let tr = 0;
  let n = 0;
  for (let k = 0; k < 60; k++) {
    for (const r of S.writeFacts(model, 1, 1000 + k)) { n++; if (r.verdict === 'true') tr++; }
  }
  const exact = (fp.likes.t + fp.lives.t) / 2;
  assert.ok(Math.abs(tr / n - exact) < 4 * Math.sqrt(exact * (1 - exact) / n), `${tr / n} vs ${exact}`);
});

test('fact counts on the page equal checkFact over the samples', () => {
  const model = RUN.trainer.state.model;
  for (const [tau, seed] of [[1, S.FACT_SEED], [1.5, S.FACT_SEED + 1], [0, S.FACT_SEED]]) {
    const list = S.writeFacts(model, tau, seed);
    assert.equal(list.length, 16);
    assert.deepEqual(list.map((r) => r.prompt), WORLD.people.flatMap((p) => [`${p} likes `, `${p} lives `]));
    const tally = { true: 0, false: 0, other: 0 };
    for (const r of list) {
      assert.equal(r.sentence, r.prompt + r.cont);
      assert.equal(r.verdict, checkFact(r.sentence));
      // The continuation stops at its first full stop, or after 16 characters.
      const stop = r.cont.indexOf('.');
      assert.ok(stop === r.cont.length - 1 || (stop === -1 && r.cont.length === S.FACT_MAX));
      assert.ok(Array.from(r.cont).every((c) => ALPHABET.includes(c)));
      tally[r.verdict]++;
    }
    assert.deepEqual(S.factCounts(list), tally);
    assert.deepEqual(S.writeFacts(model, tau, seed), list);
  }
  // Totals add press by press.
  const two = [...S.writeFacts(model, 1, 7), ...S.writeFacts(model, 1, 8)];
  const sum = S.factCounts(two);
  assert.equal(sum.true + sum.false + sum.other, 32);
});

test('param count equals attnParamCount', () => {
  const m = S.startModel();
  const parts = S.paramParts(m);
  assert.equal(parts.reduce((s, p) => s + p.n, 0), attnParamCount(m));
  assert.equal(attnParamCount(m), 11324);
  assert.deepEqual(parts.map((p) => p.n), [28 * 32, 32 * 32, 4 * 32 * 32, 64 * 32 + 64 + 32 * 64 + 32, 3 * 2 * 32, 28 * 32 + 28]);
  assert.equal(S.pageValues()['n-params'], '11,324');
  assert.equal(S.pageValues()['predictions'], S.count(2500 * 8 * 32));
  assert.match(html, /data-tex="28 \\times 32 \+ 32 \\times 32 \+ 4096 \+ 4192 \+ 192 \+ 924 = 11324"/);
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const keys = Object.keys(SV).filter((k) => !k.startsWith('ngram'));
  for (const eps of [1e-12, -1e-12]) {
    const v = S.storyValuesFor(nudged(S.startModel(), eps));
    for (const k of keys) assert.equal(v[k], SV[k], `${k} at ${eps}`);
  }
});

test('prompt sanitiser caps, lower-cases and marks unknowns', () => {
  const t = S.promptState('Mia LIKES 3 Figs!');
  assert.equal(t.text, 'mia likes 3 figs!');
  assert.equal(t.chips.map((c) => c.ch).join(''), t.text);
  assert.deepEqual(t.chips.filter((c) => c.unknown).map((c) => c.ch), ['3', '!']);
  assert.equal(t.unknown, 2);
  // The context is every known character, up to the last 32; the unknowns are skipped.
  assert.equal(t.context, 'mia likes  figs');
  assert.deepEqual(t.ids, charTokenise('mia likes  figs'));
  const p = S.nextChar(S.startModel(), t.ids, 1);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p.every(Number.isFinite));
  const long = S.promptState('A'.repeat(500));
  assert.equal(long.text, 'a'.repeat(80));
  assert.equal(long.ids.length, 32);
  assert.equal(long.chips.filter((c) => c.ctx).length, 32);
  assert.equal(S.MAX_INPUT, 80);
  const sc = S.promptState('<script>x</script>');
  assert.equal(sc.chips.map((c) => c.ch).join(''), '<script>x</script>');
  assert.deepEqual(sc.chips.filter((c) => c.unknown).map((c) => c.ch), ['<', '>', '<', '/', '>']);
  const em = S.promptState('a😀b');
  assert.deepEqual(em.chips.map((c) => c.ch), ['a', '😀', 'b']);
  assert.equal(em.unknown, 1);
  assert.equal(em.context, 'ab');
  // Nothing known: no context and no prediction.
  assert.equal(S.promptState('?? 1').context, ' ');
  assert.equal(S.promptState('??!').context, null);
  assert.equal(S.nextChar(S.startModel(), [], 1), null);
  // The smallest τ is the argmax.
  const z = attnForward(S.startModel(), t.ids).logits.at(-1);
  const g = S.nextChar(S.startModel(), t.ids, 0);
  assert.equal(g[z.indexOf(Math.max(...z))], 1);
  // Continue: one character drawn at τ from the given stream, from the same probabilities.
  const r1 = mulberry32(5);
  const r2 = mulberry32(5);
  assert.equal(S.continueOne(S.startModel(), t.ids, 1, r1), ALPHABET[sampleIndex(p, r2)]);
  // The stage writes typed text with textContent only, never as markup, and keeps nothing.
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
  assert.match(src, /maxLength = MAX_INPUT/);
  assert.doesNotMatch(src, /localStorage|sessionStorage|fetch\(|progress/);
});

test('samples: one per checkpoint, each from its own seed, alphabet only', () => {
  const m = RUN.trainer.state.model;
  const s = S.sampleAt(m, 2500);
  // Written out again: start ". ", then 60 draws at τ = 1, each from the last 32 characters.
  const rng = mulberry32(293 + 2500);
  let xs = ids('. ');
  for (let i = 0; i < 60; i++) {
    const z = attnForward(m, xs.slice(-32)).logits.at(-1);
    xs = [...xs, sampleIndex(softmax(z, 1), rng)];
  }
  assert.equal(s, xs.slice(2).map((i) => ALPHABET[i]).join(''));
  assert.equal(s.length, 60);
  RUN.samples.forEach((x) => {
    assert.equal(x.text.length, 60);
    assert.ok(Array.from(x.text).every((c) => ALPHABET.includes(c)));
  });
  assert.equal(RUN.samples.at(-1).text, s);
});

test('step 1: the n = 5 counting model cannot see the name', () => {
  const m5 = ngramModel(rewrite(2000, 261), 5);
  const p = nextProbs(m5, 'ikes ')[ALPHABET.indexOf('a')];
  assert.equal(S.pageValues()['p5-a'], p.toFixed(2));
  // Two people like apples, out of eight.
  assert.equal(WORLD.people.filter((x) => WORLD.likes[x] === 'apples').length, 2);
});

test('read-out rows', () => {
  const fresh = S.makeLive();
  const infos = [[0, S.stepOneInfo()], [1, S.liveInfo(fresh, false)], [1, S.liveInfo(RUN, false)], [1, S.liveInfo(fresh, 'computing')]];
  fresh.work(40);
  infos.push([1, S.liveInfo(fresh, true)]);
  for (const model of [S.startModel(), RUN.trainer.state.model]) {
    for (const raw of ['zoe likes ', '', '??', 'a'.repeat(80)]) {
      for (const tau of S.TAUS) infos.push([2, S.promptInfo(model, raw, tau, 0)]);
    }
    for (const tau of S.TAUS) infos.push([3, { iter: 0, tau, fp: S.factProbs(model, [tau])[0], counts: null }]);
  }
  infos.push([4, null], [5, null]);
  for (const [step, s] of infos) {
    const rows = S.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity|null/.test(r), r);
      assert.ok(r.length <= 50, r);
    }
  }
  assert.ok(S.readout(1, S.liveInfo(S.makeLive(), false)).some((r) => /press Train/.test(r)));
  assert.ok(S.readout(1, S.liveInfo(RUN, false)).some((r) => r.includes('step 2,500 of 2,500 · finished')));
  assert.ok(S.readout(1, S.liveInfo(RUN, false)).some((r) => r.includes(SV['bits-end'].slice(0, 4))));
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Large language models, in miniature · ML Explorer<\/title>/);
  assert.match(html, /<h1>Large language models, in miniature<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/llms\/">/);
  for (const s of [261, 262, 291, 292, 293, 294, 295]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /invented/);
  assert.match(html, /not claims about anyone/);
  assert.match(html, /far larger scale/);
  assert.match(html, /Instruction tuning/);
  assert.match(html, /Preference tuning/);
  assert.match(html, /Play steps through the lesson, and Train trains/);
  // "Hallucination" once, in quotes, and defined.
  assert.equal((html.match(/hallucinat/gi) || []).length, 1);
  assert.match(html, /"hallucination": fluent output that is false/);
  assert.doesNotMatch(html, /understand|thinks|\bknows\b|reasons|creative|intelligen|conscious|decides|chooses|\bbelieves\b/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(html, /GPT|BERT|Llama|Gemini|Google|Meta\b/);
  assert.doesNotMatch(html, /Taylor/);
  for (const href of ['/ml/language-models/', '/ml/attention/', '/ml/transformers/', '/ml/optimisers/', '/ml/tokens/']) assert.match(html, new RegExp(`href="${href}"`));
  assert.match(html, /<a class="next-lesson" href="\/ml\/generative-models\/">Next lesson: Generative models →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What is the model trained to do\?/);
  assert.match(html, /Why can a fluent sentence from the model be false\?/);
  // The maths: the causal objective, bits from nats, and the causal mask.
  assert.match(html, /\\log p\(x_\{t\+1\} \\mid x_1, \\dots, x_t\)/);
  assert.match(html, /\\ln 2/);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const usedPv = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(usedPv.has(key), `unused page value ${key}`);
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in SV, key);
    assert.equal(text, SV[key], key);
  }
  for (const key of Object.keys(SV)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(SV)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
