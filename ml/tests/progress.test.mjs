import test from 'node:test';
import assert from 'node:assert/strict';
import { createProgress } from '../assets/progress.js';

function memoryStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    dump: () => m,
  };
}

test('progress store survives throwing storage', () => {
  const bad = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
  };
  const p = createProgress(bad);
  assert.doesNotThrow(() => p.set('x', { step: 2 }));
  assert.deepEqual(p.get('x'), { step: 0, done: false, quiz: {} });
  assert.equal(p.doneCount(), 0);
});

test('progress store survives missing storage', () => {
  const p = createProgress(undefined);
  assert.doesNotThrow(() => p.set('x', { done: true }));
  assert.deepEqual(p.get('x'), { step: 0, done: false, quiz: {} });
});

test('progress store resets on invalid JSON', () => {
  const s = { getItem: () => '{oops', setItem() {} };
  assert.equal(createProgress(s).doneCount(), 0);
});

test('progress store ignores a wrong version or shape', () => {
  const s = { getItem: () => JSON.stringify({ v: 2, lessons: { a: { done: true } } }), setItem() {} };
  assert.equal(createProgress(s).doneCount(), 0);
  const t = { getItem: () => JSON.stringify({ v: 1, lessons: { a: { step: 'x', done: 'yes', quiz: 3 } } }), setItem() {} };
  assert.deepEqual(createProgress(t).get('a'), { step: 0, done: false, quiz: {} });
});

test('progress round-trips', () => {
  const s = memoryStorage();
  const p = createProgress(s);
  p.set('a', { done: true });
  assert.equal(p.doneCount(), 1);
  p.set('b', { step: 3, quiz: { q1: 'left' } });
  p.set('b', { quiz: { q2: 'right' } });
  const again = createProgress(s);
  assert.deepEqual(again.get('a'), { step: 0, done: true, quiz: {} });
  assert.deepEqual(again.get('b'), { step: 3, done: false, quiz: { q1: 'left', q2: 'right' } });
  assert.equal(again.doneCount(), 1);
  const raw = JSON.parse(s.dump().get('rsj-ml:v1'));
  assert.equal(raw.v, 1);
  assert.ok(raw.lessons.a.done);
});
