import test from 'node:test';
import assert from 'node:assert/strict';
import {wordTokens, skipgramPairs, skipgramInit, skipgramLoss, skipgramGrad,
  skipgramTrainer, cosine, nearest} from '../assets/embed.js';
import {gradCheck} from '../assets/nn.js';
import {worldSentences} from '../assets/world.js';

const flat = m => [...m.Win.flat(), ...m.Wout.flat()];
const unflat = (m, v) => {
  const {V, d} = m, rows = o => Array.from({length: V}, (_, i) => v.slice(o + i * d, o + (i + 1) * d));
  return {V, d, Win: rows(0), Wout: rows(V * d)};
};

test('pairs by hand', () => {
  const {words, ids} = wordTokens(['mia likes figs.']);
  assert.deepEqual(words, ['.', 'figs', 'likes', 'mia']);
  assert.deepEqual(ids, [[3, 2, 1, 0]]);
  assert.deepEqual(skipgramPairs([[0, 1, 2, 3]], 1),
    [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]);
  // never across sentences
  assert.deepEqual(skipgramPairs([[0, 1], [2, 3]], 2), [[0, 1], [1, 0], [2, 3], [3, 2]]);
});

test('skipgram passes a gradient check', () => {
  const V = 6, d = 3;
  const m = skipgramInit({V, d, seed: 221});
  const pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 0], [2, 2], [0, 5], [3, 1]];
  const g = skipgramGrad(m, pairs);
  const gv = [...g.gIn.flat(), ...g.gOut.flat()];
  const rel = gradCheck(p => skipgramLoss(unflat(m, p), pairs), flat(m), gv);
  assert.ok(rel < 1e-6, `rel ${rel}`);
  assert.ok(Math.abs(g.loss - skipgramLoss(m, pairs)) < 1e-12);
});

test('loss of a zero model is ln V', () => {
  const V = 7, z = () => Array.from({length: V}, () => [0, 0]);
  assert.ok(Math.abs(skipgramLoss({V, d: 2, Win: z(), Wout: z()}, [[0, 1], [3, 2]]) - Math.log(V)) < 1e-12);
});

const world = () => {
  const {words, ids} = wordTokens(worldSentences({n: 150, seed: 5}));
  return {words, pairs: skipgramPairs(ids)};
};

test('trainer is deterministic', () => {
  const {words, pairs} = world();
  const mk = () => skipgramTrainer({model: skipgramInit({V: words.length, seed: 3}), pairs,
    kind: 'adam', hyper: {eta: 0.05}, seed: 9});
  const a = mk(), b = mk();
  a.step(40); b.step(40);
  assert.deepEqual(a.state.model, b.state.model);
  assert.deepEqual(a.state.epochLoss, b.state.epochLoss);
});

test('reset repeats the run', () => {
  const {words, pairs} = world();
  const t = skipgramTrainer({model: skipgramInit({V: words.length, seed: 3}), pairs,
    kind: 'adam', hyper: {eta: 0.05}, seed: 9});
  const first = skipgramLoss(t.state.model, pairs);
  assert.equal(t.state.epochLoss[0], first);
  t.step(30);
  const snap = JSON.stringify(t.state);
  t.reset();
  assert.equal(t.state.iter, 0);
  assert.deepEqual(t.state.epochLoss, [first]);
  t.step(30);
  assert.equal(JSON.stringify(t.state), snap);
});

test('divergence keeps the last finite parameters', () => {
  const {words, pairs} = world();
  const t = skipgramTrainer({model: skipgramInit({V: words.length, seed: 3}), pairs,
    kind: 'sgd', hyper: {eta: 1e9}, seed: 9});
  t.step(50);
  assert.ok(t.state.diverged);
  assert.ok([...t.state.model.Win.flat(), ...t.state.model.Wout.flat()].every(Number.isFinite));
});

test('cosine basics', () => {
  assert.ok(Math.abs(cosine([1, 2], [2, 4]) - 1) < 1e-12);
  assert.ok(Math.abs(cosine([1, 2], [-1, -2]) + 1) < 1e-12);
  assert.equal(cosine([1, 0], [0, 3]), 0);
  assert.equal(cosine([0, 0], [1, 2]), 0);
  assert.equal(cosine([1, 2], [0, 0]), 0);
});

test('nearest sorts descending and honours exclude', () => {
  const m = {V: 4, d: 2, Win: [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], Wout: []};
  const words = ['a', 'b', 'c', 'd'];
  const r = nearest(m, 'a', words, {exclude: ['c']});
  assert.deepEqual(r.map(x => x.word), ['b', 'd']);
  assert.ok(r[0].cos > r[1].cos);
});

test('trainer lowers the loss on the world text', () => {
  const {words, ids} = wordTokens(worldSentences({n: 1500, seed: 222}));
  const pairs = skipgramPairs(ids);
  const t = skipgramTrainer({model: skipgramInit({V: words.length, d: 2, seed: 7}), pairs,
    kind: 'adam', hyper: {eta: 0.05}, seed: 8});
  while (t.state.epoch < 30 && !t.state.diverged) t.step(50);
  assert.ok(!t.state.diverged);
  const el = t.state.epochLoss;
  assert.ok(el[el.length - 1] < el[0], `${el[0]} -> ${el[el.length - 1]}`);
});
