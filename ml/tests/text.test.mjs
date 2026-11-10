import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mulberry32} from '../assets/maths.js';
import {ALPHABET, worldText} from '../assets/world.js';
import {
  sanitise, charTokenise, bpeTrain, bpeEncode, bpeDecode, ngramModel, nextProbs,
  crossEntropyBits, unseenContextShare, softmax, sampleIndex, generateNgram,
} from '../assets/text.js';

const sum = xs => xs.reduce((s, x) => s + x, 0);

test('encode and context handle unknown characters', () => {
  const r = sanitise('Mia LIKES 3 figs!');
  assert.equal(r.text, 'mia likes 3 figs!');
  assert.deepEqual(r.unknown, [10, 16]);
  assert.deepEqual(charTokenise('a3 !'), [2, -1, 0, -1]);
  const model = ngramModel(worldText({n: 200, seed: 1}), 3, {});
  for (const ctx of ['mia likes 3 figs!', '3!?', '', 'a3b!c']) {
    assert.ok(Math.abs(sum(nextProbs(model, ctx)) - 1) < 1e-12, ctx);
  }
});

test('sanitise caps the length and counts emoji as one character', () => {
  const r = sanitise('x'.repeat(500));
  assert.equal(r.text.length, 80);
  assert.equal(sanitise('abcdef', 3).text, 'abc');
  const e = sanitise('a😀b');
  assert.equal(e.text, 'a😀b');
  assert.deepEqual(e.unknown, [1]);
  assert.deepEqual(charTokenise('a😀b'), [2, -1, 3]);
});

test('bpe by hand', () => {
  // 'aaab aab.' : words aaab and aab. Pairs inside words: aa 3 (aaab twice, aab once), ab 2.
  // Merge 1 is a+a (3). Tokens: aa a b | aa b | . = 7 tokens (9 characters before).
  // Then every pair (aa,a), (a,b), (aa,b) occurs once, so training stops.
  const t1 = bpeTrain('aaab aab.', 5);
  assert.deepEqual(t1.merges, [['a', 'a', 3]]);
  assert.deepEqual(t1.lengths, [9, 7]);
  assert.deepEqual(t1.vocab, [' ', '.', 'a', 'b', 'aa']);
  // 'abab abab.' : ab 4, ba 2. Merge 1 a+b (4): ab ab | ab ab | . = 6 tokens (10 before).
  // Merge 2 ab+ab (2): abab | abab | . = 4 tokens... with the space: abab ' ' abab '.' = 4.
  const t2 = bpeTrain('abab abab.', 5);
  assert.deepEqual(t2.merges, [['a', 'b', 4], ['ab', 'ab', 2]]);
  assert.deepEqual(t2.lengths, [10, 6, 4]);
  assert.deepEqual(bpeEncode('abab abab.', t2.merges), ['abab', ' ', 'abab', '.']);
  // Tie: 'abcd' has ab, bc, cd once each; nothing repeats, so no merge at all.
  assert.deepEqual(bpeTrain('abcd', 3).merges, []);
  // Tie broken by first occurrence: 'abxab cdxcd.' -> ab 2, xa 1, bx 1, cd 2, xc 1: ab first.
  const t3 = bpeTrain('abxab cdxcd.', 1);
  assert.deepEqual(t3.merges, [['a', 'b', 2]]);
  assert.deepEqual(bpeTrain('abxab cdxcd.', 2).merges, [['a', 'b', 2], ['c', 'd', 2]]);
  assert.equal(bpeTrain('abxab cdxcd.', 2).lengths[0], 12);
});

test('bpe round-trips', () => {
  const rng = mulberry32(5);
  const text = worldText({n: 60, seed: 2});
  const {merges} = bpeTrain(text, 40);
  assert.ok(merges.length > 10);
  for (let i = 0; i < 50; i++) {
    let s = '';
    const len = 1 + Math.floor(rng() * 40);
    for (let j = 0; j < len; j++) s += ALPHABET[Math.floor(rng() * ALPHABET.length)];
    assert.equal(bpeDecode(bpeEncode(s, merges)), s);
  }
  assert.equal(bpeDecode(bpeEncode(text, merges)), text);
  assert.equal(bpeEncode(text, merges).length, bpeTrain(text, 40).lengths.at(-1));
});

test('bpe never merges across a space or full stop', () => {
  const text = worldText({n: 80, seed: 4});
  const {vocab, lengths} = bpeTrain(text, 60);
  for (const v of vocab) {
    if (v.length > 1) assert.ok(!v.includes(' ') && !v.includes('.'), v);
  }
  for (let i = 1; i < lengths.length; i++) assert.ok(lengths[i] < lengths[i - 1]);
  assert.equal(lengths[0], text.length);
});

test('ngram probabilities sum to 1', () => {
  const text = worldText({n: 150, seed: 9});
  const rng = mulberry32(11);
  for (let n = 1; n <= 5; n++) {
    const model = ngramModel(text, n, {});
    for (let i = 0; i < 20; i++) {
      const at = Math.floor(rng() * (text.length - 8));
      const p = nextProbs(model, text.slice(at, at + 1 + Math.floor(rng() * 7)));
      assert.equal(p.length, ALPHABET.length);
      assert.ok(Math.abs(sum(p) - 1) < 1e-12);
    }
  }
});

test('ngram by hand', () => {
  const model = ngramModel('abab', 1, {k: 0});
  const p = nextProbs(model, 'a');
  assert.equal(p[ALPHABET.indexOf('b')], 1);
  assert.equal(sum(p), 1);
  // 'abab', n 1: contexts a->b, b->a, a->b; b -> a with probability 1 too.
  assert.equal(nextProbs(model, 'b')[ALPHABET.indexOf('a')], 1);
  // Add-k: counts a->b is 2; with k = 1: (2 + 1) / (2 + 28).
  const sm = ngramModel('abab', 1, {k: 1});
  assert.ok(Math.abs(nextProbs(sm, 'a')[ALPHABET.indexOf('b')] - 3 / 30) < 1e-15);
  // Unseen context: uniform.
  const u = nextProbs(model, 'z');
  for (const x of u) assert.ok(Math.abs(x - 1 / 28) < 1e-15);
});

test('cross-entropy and unseen share', () => {
  const empty = ngramModel('', 2, {k: 1});
  assert.ok(Math.abs(crossEntropyBits(empty, 'mia likes figs.') - Math.log2(28)) < 1e-12);
  const m = ngramModel('abab', 1, {k: 0});
  assert.equal(crossEntropyBits(m, 'abab'), 0);
  assert.equal(unseenContextShare(m, 'abab'), 0);
  // Positions 1..3 of 'abzb': contexts a, b, z; z never seen: 1 of 3.
  assert.ok(Math.abs(unseenContextShare(m, 'abzb') - 1 / 3) < 1e-15);
});

test('softmax is stable at +-800', () => {
  for (const z of [[800, -800], [-800, -800], [800, 800, 0], [1e4, -1e4, 3]]) {
    const p = softmax(z);
    assert.ok(p.every(Number.isFinite));
    assert.ok(Math.abs(sum(p) - 1) < 1e-12);
  }
  assert.equal(softmax([800, 0])[0], 1);
});

test('small tau tends to argmax', () => {
  assert.deepEqual(softmax([1, 3, 2], 1e-4), [0, 1, 0]);
  assert.deepEqual(softmax([3, 1, 3], 1e-4), [0.5, 0, 0.5]);
  const z = [1, 3, 2];
  const e = z.map(v => Math.exp(v)), s = sum(e);
  const p = softmax(z, 1);
  z.forEach((_, i) => assert.ok(Math.abs(p[i] - e[i] / s) < 1e-15));
  const hot = softmax(z, 100);
  for (const x of hot) assert.ok(Math.abs(x - 1 / 3) < 0.02);
});

test('sampleIndex matches the distribution', () => {
  const rng = mulberry32(21), p = [0.2, 0.5, 0.3], c = [0, 0, 0];
  for (let i = 0; i < 20000; i++) c[sampleIndex(p, rng)]++;
  p.forEach((q, i) => assert.ok(Math.abs(c[i] / 20000 - q) < 0.015));
  // Rounding shortfall: u above the total returns the last positive entry.
  assert.equal(sampleIndex([0.3, 0.3, 0], () => 0.99), 1);
});

test('generation is reproducible', () => {
  const model = ngramModel(worldText({n: 200, seed: 1}), 3, {});
  const g = seed => generateNgram(model, {start: 'mia ', length: 60, tau: 1, seed});
  assert.equal(g(1), g(1));
  assert.notEqual(g(1), g(2));
  assert.equal(g(1).length, 'mia '.length + 60);
  assert.ok(g(1).startsWith('mia '));
});

test('generation at tau near zero follows the greedy path', () => {
  const model = ngramModel('abab', 1, {k: 0});
  const g = generateNgram(model, {start: 'a', length: 5, tau: 1e-4, seed: 3});
  assert.equal(g, 'ababab');
});
