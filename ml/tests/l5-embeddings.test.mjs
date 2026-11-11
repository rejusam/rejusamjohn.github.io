import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { worldSentences, WORLD } from '../assets/world.js';
import { skipgramInit, skipgramGrad, skipgramLoss, cosine } from '../assets/embed.js';
import { optInit, optStep } from '../assets/nn.js';
import * as S from '../embeddings/stage.js';

const html = readFileSync(new URL('../embeddings/index.html', import.meta.url), 'utf8');

// The invented world's sentences written out again from world.js's stated rule:
// 0.4 "likes", 0.4 "lives", 0.2 "friends", names uniform, the friend different.
function sentences(n, seed) {
  const rng = mulberry32(seed);
  const pick = (xs) => xs[Math.floor(rng() * xs.length)];
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = rng();
    const p = pick(WORLD.people);
    if (r < 0.4) out.push(`${p} likes ${WORLD.likes[p]}.`);
    else if (r < 0.8) out.push(`${p} lives ${WORLD.prep[WORLD.livesIn[p]]} the ${WORLD.livesIn[p]}.`);
    else {
      let q = p;
      while (q === p) q = pick(WORLD.people);
      out.push(`${p} and ${q} are friends.`);
    }
  }
  return out;
}

// Words (full stop separate), sorted ids, and every (centre, neighbour) pair within 2.
function pairsOf(sents) {
  const toks = sents.map((s) => s.replace('.', ' .').split(' '));
  const words = [...new Set(toks.flat())].sort();
  const pairs = [];
  for (const t of toks) {
    for (let i = 0; i < t.length; i++) {
      for (let j = i - 2; j <= i + 2; j++) {
        if (j !== i && j >= 0 && j < t.length) pairs.push([words.indexOf(t[i]), words.indexOf(t[j])]);
      }
    }
  }
  return { words, pairs };
}

// Full-batch Adam written out with embed.js's gradient and nn.js's Adam, pairs in data order.
function directRun(model0, steps) {
  const { pairs } = S.data();
  const V = model0.V;
  const d = model0.d;
  const unflat = (v) => ({ V, d, Win: Array.from({ length: V }, (_, i) => v.slice(i * d, (i + 1) * d)), Wout: Array.from({ length: V }, (_, i) => v.slice(V * d + i * d, V * d + (i + 1) * d)) });
  let params = [...model0.Win.flat(), ...model0.Wout.flat()];
  let opt = optInit('adam', params.length);
  const losses = [skipgramLoss(model0, pairs)];
  let maxLen = 0;
  for (let s = 0; s < steps; s++) {
    const g = skipgramGrad(unflat(params), pairs);
    const r = optStep('adam', { eta: 0.05 }, opt, params, [...g.gIn.flat(), ...g.gOut.flat()]);
    params = r.params;
    opt = r.state;
    const m = unflat(params);
    losses.push(skipgramLoss(m, pairs));
    for (const v of m.Win) maxLen = Math.max(maxLen, Math.hypot(v[0], v[1]));
  }
  return { model: unflat(params), losses, maxLen };
}

const nudged = (m, eps) => {
  let k = 0;
  const f = (row) => row.map((v) => v * (1 + (k++ % 2 ? -eps : eps)));
  return { V: m.V, d: m.d, Win: m.Win.map(f), Wout: m.Wout.map(f) };
};

let LIVE = null;
function liveRun() {
  if (LIVE) return LIVE;
  const live = S.makeLive();
  while (!live.done()) live.advance(7);
  LIVE = live;
  return live;
}

test('data reproducible', () => {
  assert.equal(S.SEED_DATA, 251);
  assert.equal(S.SEED_INIT, 252);
  assert.equal(S.SEED_SHUFFLE, 253);
  assert.equal(S.N_SENTENCES, 1500);
  assert.equal(S.DIM, 2);
  assert.equal(S.WINDOW, 2);
  assert.equal(S.ETA, 0.05);
  assert.equal(S.STEPS, 200);
  const D = S.data();
  assert.deepEqual(D.sentences, sentences(1500, 251));
  assert.deepEqual(D.sentences, worldSentences({ n: 1500, seed: 251 }));
  assert.equal(D.sentences[0], 'zoe likes pears.');
  const P = pairsOf(D.sentences);
  assert.deepEqual(D.words, P.words);
  assert.deepEqual(D.pairs, P.pairs);
  assert.equal(D.words.length, 27);
  assert.equal(D.pairs.length, 22344);
  assert.equal(D.sentences[D.example], 'tom lives by the lake.');
  assert.equal(D.example, 1);
  assert.deepEqual(S.startModel(), skipgramInit({ V: 27, d: 2, seed: 252 }));
  // Every word is in exactly one group, and the groups are the world's lists.
  for (const g of ['name', 'food', 'place']) {
    const list = { name: WORLD.people, food: WORLD.foods, place: WORLD.places }[g];
    assert.deepEqual(D.words.filter((w) => S.groupOf(w) === g), [...list].sort());
  }
  assert.equal(D.words.filter((w) => S.groupOf(w) === 'other').length, 9);
});

test('one-hot distances all √2', () => {
  const V = S.data().words.length;
  const st = S.oneHotStats(V);
  assert.equal(st.pairs, (V * (V - 1)) / 2);
  assert.equal(st.pairs, 351);
  assert.equal(st.dMin, Math.SQRT2);
  assert.equal(st.dMax, Math.SQRT2);
  assert.equal(st.cMin, 0);
  assert.equal(st.cMax, 0);
  // By hand: e_i − e_j has one +1 and one −1, so its length is √(1 + 1).
  const a = S.oneHot(3, V);
  const b = S.oneHot(20, V);
  assert.equal(a.reduce((s, x) => s + x, 0), 1);
  assert.equal(Math.hypot(...a.map((x, k) => x - b[k])), Math.SQRT2);
  assert.equal(cosine(a, b), 0);
  assert.equal(S.pageValues()['onehot-dist'], '1.414');
});

test('example pairs: neighbours within 2, inside the sentence', () => {
  const e0 = S.examplePairs(0);
  assert.deepEqual(e0.tokens, ['tom', 'lives', 'by', 'the', 'lake', '.']);
  assert.deepEqual(e0.pairs, [['tom', 'lives'], ['tom', 'by']]);
  const e2 = S.examplePairs(2);
  assert.deepEqual(e2.pairs, [['by', 'tom'], ['by', 'lives'], ['by', 'the'], ['by', 'lake']]);
  const e5 = S.examplePairs(5);
  assert.deepEqual(e5.pairs, [['.', 'the'], ['.', 'lake']]);
  // Each sentence holds at most one food, so two foods never share a sentence.
  assert.equal(S.sentencesWithTwoFoods(), 0);
  // No sentence repeats a word, so a centre's up to four neighbours are all different words.
  for (const ids of S.data().ids) assert.equal(new Set(ids).size, ids.length);
  assert.equal(Math.max(...[0, 1, 2, 3, 4, 5].map((p) => S.examplePairs(p).pairs.length)), 4);
});

test('training deterministic', () => {
  const a = S.makeTrainer();
  const b = S.makeTrainer();
  a.step(25);
  b.step(25);
  assert.deepEqual(a.state.model, b.state.model);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
  // Full batch: one step is one pass over every pair, and the loss is logged after each.
  assert.equal(a.state.epoch, 25);
  assert.equal(a.state.epochLoss.length, 26);
  // Reset repeats the run; the live run stops at STEPS and is never diverged.
  a.reset();
  a.step(25);
  assert.deepEqual(a.state.model, b.state.model);
  const lv = liveRun();
  assert.equal(lv.trainer.state.iter, 200);
  assert.equal(lv.trainer.state.diverged, false);
  lv.advance(5);
  assert.equal(lv.trainer.state.iter, 200);
  // The shared trainer agrees with the run written out here (summation order aside).
  const d = directRun(S.startModel(), 200);
  const L = lv.trainer.state.epochLoss;
  d.losses.forEach((v, i) => assert.ok(Math.abs(v - L[i]) < 1e-9, `step ${i}`));
  assert.ok(d.maxLen < S.R_MAX ** 2, `longest vector ${d.maxLen}`);
});

test('after training, the nearest neighbour of every name is a name', () => {
  const end = liveRun().trainer.state.model;
  const start = S.startModel();
  for (const w of WORLD.people) assert.equal(S.groupOf(S.ranking(end, w)[0].word), 'name', w);
  const sv = S.storyValues();
  assert.equal(sv['end-name'], '8');
  // At the random start it was not so.
  assert.equal(S.groupShares(start).name.ok, 2);
  assert.equal(sv['start-name'], '2');
  // The loss: start above the same-guess-for-every-word value ln 27, end below it.
  assert.ok(liveRun().trainer.state.epochLoss[0] > Math.log(27));
  assert.ok(liveRun().trainer.state.epochLoss[200] < Math.log(27) - 0.5);
  assert.equal(sv['loss-start'], '3.34');
  assert.equal(sv['loss-end'], '2.51');
  assert.equal(S.pageValues()['ln-v'], '3.30');
});

test('foods and places group', () => {
  const end = liveRun().trainer.state.model;
  const sh = S.groupShares(end);
  assert.deepEqual(sh.food, { ok: 6, n: 6 });
  assert.deepEqual(sh.place, { ok: 4, n: 4 });
  const sv = S.storyValues();
  assert.equal(sv['end-food'], '6');
  assert.equal(sv['end-place'], '4');
  assert.equal(sv['start-food'], '0');
  assert.equal(sv['start-place'], '0');
  // Mean cosine inside each group is near 1; between groups it is below 0.
  assert.equal(sv['within-food'], '1.00');
  assert.equal(sv['within-place'], '1.00');
  assert.equal(sv['within-name'], '0.95');
  for (const k of ['name-place', 'name-food', 'food-place']) assert.ok(sv[k].startsWith('−'), k);
  assert.equal(sv['name-place'], '−0.89');
  // At the start, no group stands out.
  for (const g of ['name', 'food', 'place']) assert.ok(Math.abs(S.meanCos(S.startModel(), g, g)) < 0.2, g);
  // Two different foods, never in one sentence, end up with cosine 1.000 to 3 places.
  assert.equal(sv['apples-pears'], '1.000');
  assert.equal(sv['food-min'], '1.000');
  // In two dimensions "are", "and" and "friends" crowd in with the foods, and "the" with the
  // places: every food ranks its 5 fellow foods first, then those three words.
  assert.deepEqual(S.CROWD_FOOD, ['are', 'and', 'friends']);
  assert.equal(sv['food-first'], '5');
  for (const f of WORLD.foods) {
    const next = S.ranking(end, f).slice(5, 8).map((x) => x.word).sort();
    assert.deepEqual(next, ['and', 'are', 'friends'], f);
  }
  assert.equal(sv['crowd-food-min'], '1.000');
  assert.equal(sv['the-place-min'], '1.000');
  // The names end with much shorter vectors than the foods and places.
  const len = (w) => Math.hypot(...S.vecOf(end, w));
  const longestName = Math.max(...WORLD.people.map(len));
  const shortestOther = Math.min(...[...WORLD.foods, ...WORLD.places].map(len));
  assert.ok(longestName < 0.5 && shortestOther > 4, `${longestName} ${shortestOther}`);
  // The ranking is by cosine, best first, and leaves the word itself out.
  const r = S.ranking(end, 'sea');
  assert.equal(r.length, 26);
  for (let i = 1; i < r.length; i++) assert.ok(r[i - 1].cos >= r[i].cos);
  r.forEach((x) => assert.ok(Math.abs(Math.cos((x.angle * Math.PI) / 180) - x.cos) < 1e-9));
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const sv = S.storyValues();
  for (const eps of [1e-12, -1e-12]) {
    const m0 = nudged(S.startModel(), eps);
    const t = S.makeTrainer(m0);
    t.step(S.STEPS);
    const v = S.runValues(m0, t.state.model, t.state.epochLoss[0], t.state.epochLoss[S.STEPS]);
    assert.deepEqual(v, sv, `eps ${eps}`);
    // And the run written out without the trainer (another summation order).
    const d = directRun(m0, S.STEPS);
    assert.deepEqual(S.runValues(m0, d.model, d.losses[0], d.losses[S.STEPS]), sv, `direct eps ${eps}`);
  }
});

test('labels do not overlap on the trained picture', () => {
  const end = liveRun().trainer.state.model;
  const items = S.data().words.map((w) => {
    const [x, y] = S.plotPoint(S.vecOf(end, w));
    return { x: 180 + (x / S.R_MAX) * 132, y: 158 - (y / S.R_MAX) * 132, w: w.length * 11 * 0.62 };
  });
  const lab = S.placeLabels(items, { h: 13, bounds: [0, 0, 366, 286] });
  const boxes = lab.map((l, i) => ({ x0: l.x, x1: l.x + items[i].w, y0: l.y - 11, y1: l.y + 2 }));
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i];
      const b = boxes[j];
      assert.ok(!(a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0), `${i} ${j}`);
    }
  }
  // plotPoint keeps the direction and square-roots the length.
  const [px, py] = S.plotPoint([3, 4]);
  assert.ok(Math.abs(Math.hypot(px, py) - Math.sqrt(5)) < 1e-12);
  assert.ok(Math.abs(Math.atan2(py, px) - Math.atan2(4, 3)) < 1e-12);
});

test('read-out rows', () => {
  const V = 27;
  const stats = S.oneHotStats(V);
  const fresh = S.makeLive();
  const base = { V, stats, word: 'sea' };
  const mk = (lv, running = false) => {
    const m = lv.trainer.state.model;
    return { ...base, live: S.liveInfo(lv, running), model: m, rank: S.ranking(m, 'sea'), shares: S.groupShares(m) };
  };
  const all = [
    S.readout(0, base),
    S.readout(1, { ...base, ex: S.examplePairs(1), index: S.data().example, nPairs: S.data().pairs.length }),
    ...[2, 3, 4, 5].map((k) => S.readout(k, mk(fresh))),
    ...[2, 3, 4, 5].map((k) => S.readout(k, mk(liveRun()))),
  ];
  for (const rows of all) for (const r of rows) assert.ok(!/NaN|undefined|Infinity/.test(r), r);
  assert.match(S.readout(0, base).join('|'), /distance √2 = 1\.414, cosine 0/);
  assert.match(S.readout(1, { ...base, ex: S.examplePairs(1), index: 1, nPairs: 22344 }).join('|'), /22,344 pairs/);
  assert.match(S.readout(2, mk(fresh)).join('|'), /press Train/);
  assert.match(S.readout(2, mk(fresh, 'computing')).join('|'), /computing…/);
  assert.match(S.readout(2, mk(liveRun())).join('|'), /finished/);
  assert.match(S.readout(3, mk(fresh)).join('|'), /random start/);
  assert.match(S.readout(4, mk(liveRun())).join('|'), /names 8 of 8 · foods 6 of 6 · places 4 of 4/);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Embeddings: words as points · ML Explorer<\/title>/);
  assert.match(html, /<h1>Embeddings: words as points<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /data-lesson="embeddings"/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/embeddings\/">/);
  for (const s of [251, 252, 253]) assert.match(html, new RegExp(`seed ${s}\\b`));
  assert.match(html, /invented/);
  assert.match(html, /rules of (this|the) page/);
  assert.match(html, /far larger scale/);
  assert.match(html, /about 6 billion words/);
  assert.doesNotMatch(html, /1\.6 billion|That is all cosine measures/);
  assert.match(html, /square-root scale/);
  assert.match(html, /href="\/ml\/tokens\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/language-models\/">Next lesson: Predicting the next token →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /\b(understands?|thinks?|knows?|reasons|creative|intelligent|conscious)\b/i);
  assert.doesNotMatch(html, /\bking\b|\bqueen\b|analog/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /Why are one-hot vectors no good for similarity\?/);
  assert.match(html, /Two words have embeddings with cosine near 1\. What does that say\?/);
  assert.match(html, /\\mathcal\{V\}/);
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
