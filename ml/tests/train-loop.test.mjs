import test from 'node:test';
import assert from 'node:assert/strict';
import { adaptSteps, createTrainLoop } from '../assets/train-loop.js';
import { xorData, mlpInit, trainer } from '../assets/nn.js';

// Fake frame scheduler: frames run only when the test calls tick().
function fakes() {
  let id = 0;
  const queue = new Map();
  const listeners = [];
  const doc = {
    hidden: false,
    addEventListener(type, fn) { if (type === 'visibilitychange') listeners.push(fn); },
  };
  let clock = 0;
  return {
    doc,
    raf: (fn) => { queue.set(++id, fn); return id; },
    caf: (i) => { queue.delete(i); },
    now: () => { clock += 1; return clock; },
    queued: () => queue.size,
    tick() {
      const [[i, fn]] = [...queue];
      queue.delete(i);
      fn(0);
    },
    fire() { listeners.forEach((fn) => fn()); },
    resetClock() { clock = 0; },
  };
}

test('adaptSteps keeps the frame under budget', () => {
  assert.equal(adaptSteps(100, 16), 50);
  assert.equal(adaptSteps(100, 2), 125);
  assert.equal(adaptSteps(100, 6), 100);
  assert.equal(adaptSteps(1, 1000), 1);
  assert.equal(adaptSteps(190, 0.1, 8, { max: 200 }), 200);
  assert.equal(adaptSteps(100, 1e6), 1);
});

test('start twice runs one loop', () => {
  const f = fakes();
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc });
  loop.start();
  loop.start();
  assert.equal(f.queued(), 1);
  assert.equal(loop.running(), true);
});

test('loop pauses when hidden', () => {
  const f = fakes();
  let stops = 0;
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc, onStop: () => stops++ });
  loop.start();
  f.doc.hidden = true;
  f.fire();
  assert.equal(loop.running(), false);
  assert.equal(f.queued(), 0);
  assert.equal(stops, 1);
});

test('loop stops when done', () => {
  const f = fakes();
  let frames = 0;
  let stops = 0;
  const loop = createTrainLoop({
    step() {}, render() { frames++; }, done: () => frames >= 3,
    raf: f.raf, caf: f.caf, now: f.now, doc: f.doc, onStop: () => stops++,
  });
  loop.start();
  while (f.queued()) f.tick();
  assert.equal(frames, 3);
  assert.equal(loop.running(), false);
  assert.equal(stops, 1);
});

test('pause cancels the queued frame', () => {
  const f = fakes();
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc });
  loop.start();
  loop.pause();
  assert.equal(f.queued(), 0);
  assert.equal(loop.running(), false);
});

test('reset then train repeats the same states', () => {
  const f = fakes();
  const { X, y } = xorData({ n: 40, seed: 1 });
  const t = trainer({ net: mlpInit([2, 3, 1], { seed: 2 }), X, y, batch: 10, kind: 'adam', hyper: { eta: 0.05 }, seed: 3 });
  let renders = 0;
  const run = () => {
    f.resetClock();
    loop.start();
    for (let i = 0; i < 5; i++) f.tick();
    loop.pause();
    return t.state.batchLoss.slice();
  };
  const loop = createTrainLoop({
    step: (k) => t.step(k), render: () => { renders++; },
    raf: f.raf, caf: f.caf, now: f.now, doc: f.doc,
  });
  const first = run();
  assert.ok(first.length >= 5);
  const before = renders;
  loop.reset(() => t.reset());
  assert.equal(renders, before + 1);
  assert.equal(t.state.batchLoss.length, 0);
  const second = run();
  assert.deepEqual(second, first);
});
