import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { ALPHABET, WORLD } from '../assets/world.js';
import { ngramModel, nextProbs, crossEntropyBits, unseenContextShare, generateNgram } from '../assets/text.js';
import * as S from '../language-models/stage.js';

const html = readFileSync(new URL('../language-models/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../language-models/stage.js', import.meta.url), 'utf8');

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

const TABLE = S.statsTable();
const SEEDS = Array.from({ length: 100 }, (_, i) => S.GEN_SEED + i);
// Complete words of a generated continuation (a run of letters followed by a space or a full stop).
const words = (g) => g.match(/[a-z]+(?=[ .])/g) || [];
const WORLD_WORDS = new Set(rewrite(2000, 261).split(/[ .]+/).filter(Boolean));
function worldShare(n, tau, which = 'full') {
  let ok = 0, all = 0;
  for (const s of SEEDS) {
    for (const w of words(S.generate(S.DEFAULT_PROMPT, n, tau, s, which))) { all++; if (WORLD_WORDS.has(w)) ok++; }
  }
  return ok / all;
}

test('data reproducible', () => {
  assert.equal(S.corpus('full'), rewrite(2000, 261));
  assert.equal(S.corpus('small'), rewrite(50, 264));
  assert.equal(S.heldText(), rewrite(300, 262));
  assert.equal(S.corpus('full').length, 42030);
  assert.equal(S.corpus('small').length, 1032);
  assert.equal(S.heldText().length, 6262);
  assert.equal(S.K, 0.01);
  assert.equal(S.N_MAX, 8);
  assert.deepEqual(S.TAUS, [0, 0.5, 1, 1.5]);
  assert.equal(S.GEN_SEED, 263);
  // One literal value: " li" is followed by k 830 times, v 782 and m 87.
  const c = S.countsFor(S.FIXED_PROMPT, 3);
  assert.equal(c.context, ' li');
  assert.equal(c.total, 1699);
  assert.equal(c.counts[ALPHABET.indexOf('k')], 830);
  assert.equal(c.counts[ALPHABET.indexOf('v')], 782);
  assert.equal(c.counts[ALPHABET.indexOf('m')], 87);
  // The same text and seed give the same generated text.
  assert.equal(S.generate(S.DEFAULT_PROMPT, 3, 1, 263), S.generate(S.DEFAULT_PROMPT, 3, 1, 263));
});

test('counting formula and probabilities sum to 1', () => {
  const c = S.countsFor(S.FIXED_PROMPT, 3);
  const p = S.predict(S.FIXED_PROMPT, 3).probs;
  for (let i = 0; i < ALPHABET.length; i++) {
    assert.ok(Math.abs(p[i] - (c.counts[i] + 0.01) / (1699 + 0.01 * 28)) < 1e-15);
  }
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
    for (const which of ['full', 'small']) {
      for (const prompt of ['', 'mia li', 'zoe lives by', 'qqqqqqqqq', 'Tom 3 LIKES', S.DEFAULT_PROMPT]) {
        const r = S.predict(prompt, n, which);
        assert.ok(Math.abs(r.probs.reduce((a, b) => a + b, 0) - 1) < 1e-12);
        for (const tau of S.TAUS) assert.ok(Math.abs(S.tempProbs(r.probs, tau).reduce((a, b) => a + b, 0) - 1) < 1e-12);
      }
    }
  }
  // An unseen context, or fewer than n known characters, is uniform.
  const u = S.predict('qqqqqqqqq', 3);
  assert.equal(u.seen, false);
  assert.ok(u.probs.every((v) => Math.abs(v - 1 / 28) < 1e-15));
  assert.equal(S.predict('ab', 3).context, null);
  const v = S.pageValues(TABLE);
  assert.equal(v['ctx-total'], '1,699');
  assert.equal(v['top-count'], '830');
  assert.equal(v['top-p'], (830.01 / 1699.28).toFixed(3));
  assert.equal(v['top-p'], '0.488');
  assert.equal(v['v-count'], '782');
  assert.equal(v['m-count'], '87');
  assert.equal(v['zero-p'], '0.0000059');
  assert.equal(v['uniform-bits'], '4.81');
  assert.equal(v['tie-count'], '199');
  assert.equal(v['gen-len'], '80');
  assert.equal(v['n-words'], String(WORLD_WORDS.size));
  assert.equal(v['n-words'], '26');
  // The worked example in The maths uses the same counts.
  assert.ok(html.includes(`(${c.counts[ALPHABET.indexOf('k')]} + 0.01) / (${c.total} + 0.01 \\times 28) \\approx ${v['top-p']}`));
  // Temperature: tau 1 is the model's own distribution; the tie after "he " is h and l, 199 each.
  const q = S.predict('the ', 3).probs;
  assert.deepEqual(S.tempProbs(q, 1).map((x) => x.toFixed(12)), q.map((x) => x.toFixed(12)));
  const he = S.countsFor('the ', 3);
  assert.equal(he.counts[ALPHABET.indexOf('h')], 199);
  assert.equal(he.counts[ALPHABET.indexOf('l')], 199);
  assert.equal(Math.max(...he.counts), 199);
  const g = S.tempProbs(q, 0);
  assert.equal(g[ALPHABET.indexOf('h')], 0.5);
  assert.equal(g[ALPHABET.indexOf('l')], 0.5);
});

test('held-out bits per n equal crossEntropyBits', () => {
  const held = rewrite(300, 262);
  for (const [which, text] of [['full', rewrite(2000, 261)], ['small', rewrite(50, 264)]]) {
    for (let n = 1; n <= 8; n++) {
      const m = ngramModel(text, n, { k: 0.01 });
      const row = TABLE[which][n - 1];
      assert.equal(row.n, n);
      assert.equal(row.held, crossEntropyBits(m, held));
      assert.equal(row.train, crossEntropyBits(m, text));
      assert.equal(row.unseen, unseenContextShare(m, held));
      assert.equal(row.contexts, m.counts.size);
      assert.deepEqual(S.stats(n, which), row);
    }
  }
  // The page's numbers are these, rounded.
  const v = S.pageValues(TABLE);
  assert.equal(v['full-bits-1'], TABLE.full[0].held.toFixed(2));
  assert.equal(v['full-bits-3'], TABLE.full[2].held.toFixed(2));
  assert.equal(v['full-bits-8'], TABLE.full[7].held.toFixed(2));
  assert.equal(v['small-bits-3'], TABLE.small[2].held.toFixed(2));
  assert.equal(v['small-bits-8'], TABLE.small[7].held.toFixed(2));
  assert.equal(v['small-train-8'], TABLE.small[7].train.toFixed(2));
  assert.equal(v['small-unseen-8'], `${(TABLE.small[7].unseen * 100).toFixed(1)}%`);
  assert.deepEqual(
    [v['full-bits-1'], v['full-bits-3'], v['full-bits-8'], v['small-bits-3'], v['small-bits-8'], v['small-train-8'], v['small-unseen-8']],
    ['1.99', '0.47', '0.31', '0.59', '1.32', '0.33', '15.3%'],
  );
  const rows = S.readout(4, { n: 8, which: 'small', table: TABLE, seed: 263 });
  assert.ok(rows.some((r) => r.includes('1.32')));
  assert.ok(rows.some((r) => r.includes('15.3%')));
  // No rounded number sits on a rounding edge (a last-bit difference cannot change it).
  for (const which of ['full', 'small']) {
    for (const row of TABLE[which]) {
      for (const x of [row.held * 100, row.train * 100, row.unseen * 1000]) assert.ok(Math.abs((x % 1) - 0.5) > 1e-6);
    }
  }
});

test('bits fall from n = 1 to n = 3', () => {
  const f = TABLE.full.map((r) => r.held);
  assert.ok(f[0] > f[1] && f[1] > f[2]);
  assert.ok(f[0] - f[2] > 1.5);          // a big fall from 1 to 3
  assert.ok(f[2] - f[7] < 0.2);          // then small gains up to 8
  assert.ok(Math.min(...f) === f[7]);    // n = 8 is the lowest with the full text
  // With the full text no held-out context is unseen, at any n, and held-out bits stay close to training bits.
  for (const r of TABLE.full) {
    assert.equal(r.unseen, 0);
    assert.ok(r.held - r.train < 0.02);
  }
  // Every n beats the uniform guess of log2 28 bits.
  for (const r of [...TABLE.full, ...TABLE.small]) assert.ok(r.held < Math.log2(28));
});

test('with the small text, unseen share rises with n and held-out bits turn up', () => {
  const s = TABLE.small;
  assert.equal(s[0].unseen, 0);
  assert.equal(s[1].unseen, 0);
  for (let i = 2; i < 8; i++) assert.ok(s[i].unseen > s[i - 1].unseen, `n = ${i + 1}`);
  // Lowest held-out bits at n = 3, then up at every step to n = 8.
  const h = s.map((r) => r.held);
  assert.equal(h.indexOf(Math.min(...h)), 2);
  for (let i = 3; i < 8; i++) assert.ok(h[i] > h[i - 1], `n = ${i + 1}`);
  // Training bits stay low (under 0.4 from n = 4) while held-out bits climb.
  for (let i = 3; i < 8; i++) assert.ok(s[i].train < 0.4);
  // The small text beats the full text at no n from 3 on.
  for (let i = 2; i < 8; i++) assert.ok(h[i] > TABLE.full[i].held);
  // Unseen contexts alone add unseen share × log2 28 bits per character.
  const v = S.pageValues(TABLE);
  assert.equal(v['small-unseen-cost-8'], (s[7].unseen * Math.log2(28)).toFixed(2));
  assert.equal(v['small-unseen-cost-8'], '0.74');
  assert.equal(v['full-contexts-8'], String(TABLE.full[7].contexts));
  assert.equal(v['full-contexts-8'], '731');
  assert.equal(v['possible-8'], '377,801,998,336');
});

test('generation at the smallest tau is the greedy path', () => {
  // Each generated character has the largest probability for its context (ties split at random).
  const m = ngramModel(rewrite(2000, 261), 3, { k: 0.01 });
  for (const seed of SEEDS.slice(0, 20)) {
    const g = S.generate(S.DEFAULT_PROMPT, 3, 0, seed);
    assert.equal(g.length, S.GEN_LEN);
    let text = S.DEFAULT_PROMPT;
    for (const ch of g) {
      const p = nextProbs(m, text);
      assert.equal(p[ALPHABET.indexOf(ch)], Math.max(...p));
      text += ch;
    }
    assert.equal(S.DEFAULT_PROMPT + g, generateNgram(m, { start: S.DEFAULT_PROMPT, length: S.GEN_LEN, tau: 0, seed }));
  }
  // Story: at tau = 0 a few words repeat; the share of different words grows with tau.
  const variety = (tau) => SEEDS.map((s) => { const w = words(S.generate(S.DEFAULT_PROMPT, 3, tau, s)); return new Set(w).size / w.length; });
  const v0 = variety(0);
  assert.ok(v0.every((x) => x <= 0.3));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const means = S.TAUS.map((t) => mean(variety(t)));
  for (let i = 1; i < means.length; i++) assert.ok(means[i] > means[i - 1], `tau ${S.TAUS[i]}`);
  // Story: at tau = 1.5 more of the words are not words of the world.
  const share = S.TAUS.map((t) => worldShare(3, t));
  assert.ok(share[0] >= 0.95 && share[1] >= 0.95 && share[2] >= 0.95);
  assert.ok(share[3] < 0.8);
});

test('longer contexts: from gibberish to words', () => {
  // Full text, tau 1: with n = 1 most complete words are not words of the world;
  // from n = 3 on, nearly all are.
  const share = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => worldShare(n, 1));
  assert.ok(share[0] < 0.2);
  assert.ok(share[1] > share[0] && share[2] > share[1]);
  for (let i = 2; i < 8; i++) assert.ok(share[i] >= 0.95, `n = ${i + 1}`);
  // Small text: from n = 3 on, the share of real words falls as n grows.
  const small = [3, 8].map((n) => worldShare(n, 1, 'small'));
  assert.ok(small[1] < small[0]);
  assert.ok(small[1] < 0.6);
  // A new seed per press.
  assert.notEqual(S.generate(S.DEFAULT_PROMPT, 3, 1, 263), S.generate(S.DEFAULT_PROMPT, 3, 1, 264));
});

test('prompt sanitiser caps, lower-cases and marks unknowns', () => {
  const t = S.promptChips('Mia LIKES 3 Figs!', 3);
  assert.equal(t.text, 'mia likes 3 figs!');
  assert.equal(t.chips.map((c) => c.ch).join(''), t.text);
  assert.deepEqual(t.chips.filter((c) => c.unknown).map((c) => c.ch), ['3', '!']);
  assert.equal(t.unknown, 2);
  // The context is the last 3 known characters; the unknown "!" is skipped.
  assert.equal(t.context, 'igs');
  assert.deepEqual(t.chips.filter((c) => c.ctx).map((c) => c.ch), ['i', 'g', 's']);
  const p = S.predict(t.text, 3).probs;
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  // Capped at 80 characters, after lower-casing.
  const long = S.promptChips('A'.repeat(500), 3);
  assert.equal(long.text, 'a'.repeat(80));
  assert.equal(S.MAX_INPUT, 80);
  // Markup is text: its symbols outside the alphabet are unknown.
  const sc = S.promptChips('<script>x</script>', 3);
  assert.equal(sc.chips.map((c) => c.ch).join(''), '<script>x</script>');
  assert.deepEqual(sc.chips.filter((c) => c.unknown).map((c) => c.ch), ['<', '>', '<', '/', '>']);
  // An emoji is one unknown character.
  const em = S.promptChips('a😀b', 1);
  assert.deepEqual(em.chips.map((c) => c.ch), ['a', '😀', 'b']);
  assert.equal(em.unknown, 1);
  assert.equal(em.context, 'b');
  // Generation from a prompt with unknowns stays finite and in the alphabet.
  const g = S.generate('Mia <b>LIVES</b> ', 3, 1, 263);
  assert.ok(Array.from(g).every((ch) => ALPHABET.includes(ch)));
  // Too few known characters: no context.
  assert.equal(S.promptChips('?? a', 3).context, null);
  // The stage writes typed text with textContent only, never as markup, and keeps nothing.
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
  assert.match(src, /maxLength = MAX_INPUT/);
  assert.doesNotMatch(src, /localStorage|sessionStorage|fetch\(|progress/);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Predicting the next token · ML Explorer<\/title>/);
  assert.match(html, /<h1>Predicting the next token<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/language-models\/">/);
  for (const seed of [261, 262, 263, 264]) assert.match(html, new RegExp(`seed ${seed}\\b`));
  assert.match(html, /invented/);
  assert.match(html, /rules of (this|the) page/);
  assert.match(html, /far larger scale/);
  assert.match(html, /href="\/ml\/embeddings\/"/);
  assert.match(html, /\\mathcal\{V\}/);
  assert.match(html, /\\tau/);
  assert.match(html, /\\log_2/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/attention\/">Next lesson: Attention →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understand|thinks|knows|reasons|creative|intelligen|conscious|hallucinat/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(html, /GPT|BERT|Llama|Gemini/);
  assert.match(html, /N-gram Language Models/);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What does temperature 0 do\?/);
  assert.match(html, /Why does held-out cross-entropy matter more than training cross-entropy\?/);
  const pv = S.pageValues(TABLE);
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 15);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const used = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(used.has(key), `unused page value ${key}`);
  for (const val of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(val), val);
  // Samples rule: no sentence in quotes in the prose that a model produced; the
  // only quoted text is the prompt and the world's own templates.
  assert.doesNotMatch(html, /sample (reads|says|was) "/i);
});
