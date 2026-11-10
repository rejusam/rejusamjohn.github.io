import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32 } from '../assets/maths.js';
import { ALPHABET, WORLD } from '../assets/world.js';
import * as S from '../tokens/stage.js';

const html = readFileSync(new URL('../tokens/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../tokens/stage.js', import.meta.url), 'utf8');

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

// Non-overlapping occurrences of the pair (a, b), left to right, in a token list.
function replacements(tokens, a, b) {
  let c = 0;
  for (let i = 0; i + 1 < tokens.length; i++) {
    if (tokens[i] === a && tokens[i + 1] === b) { c++; i++; }
  }
  return c;
}
function merge(tokens, a, b) {
  const out = [];
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === a && tokens[i + 1] === b) { out.push(a + b); i++; } else out.push(tokens[i]);
  }
  return out;
}

test('data reproducible', () => {
  const t = S.trainingText();
  assert.equal(t, rewrite(400, 241));
  assert.equal(t, S.trainingText());
  assert.ok(t.startsWith('eli likes plums. ana and noa are friends. '));
  assert.equal(Array.from(t).length, 8541);
  assert.equal(S.sampleText(), 'eli likes plums. ana and noa are friends.');
  const tk = S.tokeniser();
  assert.equal(tk.merges.length, S.MAX_MERGES);
  assert.equal(S.MAX_MERGES, 60);
  assert.deepEqual(tk.merges[0], ['l', 'i', 389]);
  assert.equal(tk.lengths[60], 3853);
});

test('token counts on the page equal the helper', () => {
  const t = S.trainingText();
  const { merges, lengths } = S.tokeniser();
  for (let r = 0; r <= S.MAX_MERGES; r++) {
    const toks = S.tokensAt(t, r);
    assert.equal(toks.length, lengths[r], `r = ${r}`);
    assert.equal(S.tokenCount(r), lengths[r]);
    assert.equal(toks.join(''), t);
    assert.equal(S.vocabSize(r), 28 + r);
    assert.equal(S.mergesAt(r).length, r);
    assert.deepEqual(S.mergesAt(r), merges.slice(0, r));
  }
  // Read-outs and page values come from the same helpers.
  const v = S.pageValues();
  assert.equal(v['n-chars'], '8,541');
  assert.equal(v['tokens-end'], S.count(lengths[60]));
  assert.equal(v['ratio-end'], (8541 / lengths[60]).toFixed(2));
  assert.equal(v['vocab-end'], '88');
  const rows = S.readout(2, { r: 60, running: false });
  assert.ok(rows.some((s) => s.includes('3,853')));
  assert.ok(rows.some((s) => s.includes('88')));
  const rows0 = S.readout(0, {});
  assert.ok(rows0.some((s) => s.includes('8,541')));
  // Every id is unique: no two merges build the same string.
  const strs = merges.map(([a, b]) => a + b);
  assert.equal(new Set(strs).size, strs.length);
  for (const s of strs) assert.ok(!ALPHABET.includes(s));
  assert.equal(S.tokenId('l', 60), 13);
  assert.equal(S.tokenId('li', 60), 28);
  assert.equal(S.tokenId('li', 0), null);
  assert.equal(S.tokenId(strs[59], 60), 87);
  assert.equal(S.tokenId('?', 60), null);
});

test('each merge lowers the token count by the number of replacements', () => {
  const { merges, lengths } = S.tokeniser();
  let toks = Array.from(S.trainingText());
  for (let r = 0; r < merges.length; r++) {
    const [a, b, c] = merges[r];
    const rep = replacements(toks, a, b);
    assert.equal(rep, c, `merge ${r + 1}`);
    toks = merge(toks, a, b);
    assert.equal(toks.length, lengths[r] - rep);
    assert.equal(lengths[r + 1], lengths[r] - c);
  }
  // Each later merge saves no more than the one before it on this text.
  for (let r = 1; r < merges.length; r++) assert.ok(merges[r][2] <= merges[r - 1][2]);
  assert.equal(merges[59][2], 17);
});

test('first merge is the most frequent within-word pair', () => {
  const words = S.trainingText().split(/[ .]+/).filter(Boolean);
  const count = new Map();
  const first = new Map();
  let pos = 0;
  for (const w of words) {
    for (let i = 0; i + 1 < w.length; i++) {
      const k = w[i] + w[i + 1];
      count.set(k, (count.get(k) || 0) + 1);
      if (!first.has(k)) first.set(k, pos);
      pos++;
    }
  }
  let best = null;
  for (const [k, c] of count) {
    if (!best || c > best[1] || (c === best[1] && first.get(k) < first.get(best[0]))) best = [k, c];
  }
  assert.deepEqual(best, ['li', 389]);
  const [a, b, c] = S.tokeniser().merges[0];
  assert.equal(a + b, best[0]);
  assert.equal(c, best[1]);
  // Within words only: no merge ever holds a space or a full stop.
  for (const [x, y] of S.tokeniser().merges) assert.ok(!/[ .]/.test(x + y));
});

test('prompt sanitiser caps, lower-cases and marks unknowns', () => {
  const t = S.typedTokens('Zoe LIKES 3 Figs!', 60);
  assert.equal(t.text, 'zoe likes 3 figs!');
  assert.equal(t.chips.map((c) => c.tok).join(''), t.text);
  const unk = t.chips.filter((c) => c.unknown);
  assert.deepEqual(unk.map((c) => c.tok), ['3', '!']);
  assert.ok(unk.every((c) => c.id === null));
  assert.equal(t.unknown, 2);
  assert.ok(t.chips.filter((c) => !c.unknown).every((c) => Number.isInteger(c.id) && c.id >= 0 && c.id < 88));
  // Long input is capped at 80 characters, after lower-casing.
  const long = S.typedTokens('A'.repeat(500), 0);
  assert.equal(long.text, 'a'.repeat(80));
  assert.equal(long.chips.length, 80);
  assert.equal(S.MAX_INPUT, 80);
  // Markup is text: every character of "<script>" outside the alphabet is unknown.
  const sc = S.typedTokens('<script>x</script>', 60);
  assert.equal(sc.chips.map((c) => c.tok).join(''), '<script>x</script>');
  assert.deepEqual(sc.chips.filter((c) => c.unknown).map((c) => c.tok), ['<', '>', '<', '/', '>']);
  // An emoji is one unknown character, and never merges.
  const em = S.typedTokens('a😀b', 60);
  assert.deepEqual(em.chips.map((c) => c.tok), ['a', '😀', 'b']);
  assert.equal(em.unknown, 1);
  // Round trip for typed text in the alphabet.
  for (const s of ['zoe and tom like kiwis.', 'the hills.', ' ', '', 'aaa bbb']) {
    for (const r of [0, 10, 60]) assert.equal(S.typedTokens(s, r).chips.map((c) => c.tok).join(''), s);
  }
  // Bytes: one byte per symbol of the alphabet, more for other characters.
  assert.deepEqual(S.bytesOf('a é😀').map((c) => c.bytes), [[97], [32], [195, 169], [240, 159, 152, 128]]);
  // The stage writes typed text with textContent only, never as markup.
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
  assert.match(src, /maxLength = MAX_INPUT|maxlength/);
  assert.doesNotMatch(src, /localStorage|sessionStorage|fetch\(|progress/);
});

test('the default sentence: an unseen word splits into known pieces', () => {
  assert.equal(S.DEFAULT_INPUT, 'zoe and tom like kiwis.');
  const toks = S.typedTokens(S.DEFAULT_INPUT, 60).chips.map((c) => c.tok);
  assert.deepEqual(toks, ['zoe', ' ', 'and', ' ', 'tom', ' ', 'lik', 'e', ' ', 'k', 'i', 'w', 'i', 's', '.']);
  assert.match(html, /<i>like<\/i> becomes <i>lik<\/i> \+ <i>e<\/i>/);
});

test('merge pacing and the loop', () => {
  assert.equal(S.MERGE_MS, 400);
  assert.equal(S.paceMerge(1000, -Infinity), true);
  assert.equal(S.paceMerge(1399, 1000), false);
  assert.equal(S.paceMerge(1400, 1000), true);
  assert.equal(S.oneHot(13, 28).length, 28);
  assert.equal(S.oneHot(13, 28).reduce((a, b) => a + b, 0), 1);
  assert.equal(S.oneHot(13, 28)[13], 1);
  const c = S.idCounts();
  assert.equal(c.length, 28);
  assert.equal(c.reduce((a, b) => a + b, 0), 8541);
  assert.equal(c[0], 1705);
  assert.equal(c[1], 400);
});

test('stated facts about this run', () => {
  const v = S.pageValues();
  assert.equal(v['absent'], 'c, j, q and x');
  assert.equal(v['first-pair'], 'l + i');
  assert.equal(v['first-count'], '389');
  assert.equal(v['friends-at'], '19');
  assert.equal(v['possible'], '63');
  assert.equal(v['unfinished'], 'figs');
  // After all possible merges, every word is one token.
  const t = S.trainingText();
  const all = S.allMerges();
  const end = all.lengths.at(-1);
  const words = t.split(/[ .]+/).filter(Boolean).length;
  const breaks = Array.from(t).filter((ch) => ch === ' ' || ch === '.').length;
  assert.equal(end, words + breaks);
  assert.equal(v['all-words'], S.count(end));
  // The words that hold the first pair, as step 3 lists them.
  const li = [...new Set(t.split(/[ .]+/).filter((w) => w.includes('li')))].sort();
  assert.deepEqual(li, ['eli', 'likes', 'limes', 'lives']);
  assert.match(html, /\(in likes, lives, limes and eli\)/);
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Tokens: text as numbers · ML Explorer<\/title>/);
  assert.match(html, /<h1>Tokens: text as numbers<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/tokens\/">/);
  assert.match(html, /seed 241\b/);
  assert.match(html, /invented/);
  assert.match(html, /rules of (this|the) page/);
  assert.match(html, /far larger scale/);
  assert.match(html, /href="\/ml\/data-as-numbers\/"/);
  assert.match(html, /\\mathcal\{V\}/);
  assert.match(html, /\\operatorname\*?\{arg\\,?max\}|\\arg\\max/);
  assert.match(html, /Play steps through the lesson/);
  assert.match(html, /Sennrich, Haddow &amp; Birch/);
  assert.match(html, /1508\.07909/);
  assert.match(html, /Words and Tokens/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/embeddings\/">Next lesson: Embeddings: words as points →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understand|thinks|knows|reasons|creative|intelligen|conscious|hallucinat/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(html, /GPT|BERT|Llama|Gemini/);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What does each BPE merge do\?/);
  assert.match(html, /Why not give every word its own token\?/);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 12);
  for (const [, key, text] of spans) {
    assert.ok(key in pv, key);
    assert.equal(text, pv[key], key);
  }
  const used = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(used.has(key), `unused page value ${key}`);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
