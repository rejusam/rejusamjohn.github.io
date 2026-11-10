import test from 'node:test';
import assert from 'node:assert/strict';
import { adaptSteps, createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';
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

test('start, pause, start leaves one queued frame', () => {
  const f = fakes();
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc });
  loop.start();
  loop.pause();
  loop.start();
  assert.equal(f.queued(), 1);
  assert.equal(loop.running(), true);
});

test('pause and reset do not call onStop', () => {
  const f = fakes();
  let stops = 0;
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc, onStop: () => stops++ });
  loop.start();
  loop.pause();
  loop.start();
  loop.reset();
  assert.equal(stops, 0);
  assert.equal(loop.running(), false);
});

test('hidden then visible does not resume', () => {
  const f = fakes();
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc });
  loop.start();
  f.doc.hidden = true;
  f.fire();
  f.doc.hidden = false;
  f.fire();
  assert.equal(loop.running(), false);
  assert.equal(f.queued(), 0);
});

test('a throwing step halts the loop and calls onStop', () => {
  const f = fakes();
  let stops = 0;
  const loop = createTrainLoop({
    step() { throw new Error('boom'); }, render() {},
    raf: f.raf, caf: f.caf, now: f.now, doc: f.doc, onStop: () => stops++,
  });
  loop.start();
  assert.throws(() => f.tick(), /boom/);
  assert.equal(loop.running(), false);
  assert.equal(f.queued(), 0);
  assert.equal(stops, 1);
  const g = fakes();
  let s2 = 0;
  const l2 = createTrainLoop({
    step() {}, render() { throw new Error('draw'); },
    raf: g.raf, caf: g.caf, now: g.now, doc: g.doc, onStop: () => s2++,
  });
  l2.start();
  assert.throws(() => g.tick(), /draw/);
  assert.equal(l2.running(), false);
  assert.equal(s2, 1);
});

// Minimal DOM stand-in: elements with the few members trainButtons touches.
function fakeDom() {
  const makeEl = (tag) => {
    const handlers = {};
    return {
      tag, className: '', type: '', textContent: '', children: [], attrs: {},
      setAttribute(k, v) { this.attrs[k] = v; },
      append(...kids) { this.children.push(...kids); },
      appendChild(k) { this.children.push(k); },
      addEventListener(type, fn) { handlers[type] = fn; },
      click() { handlers.click(); },
    };
  };
  const doc = { createElement: makeEl };
  const controls = makeEl('div');
  controls.ownerDocument = doc;
  return { controls };
}

test('trainButtons toggles label, stop() resyncs, reduced runs to end once', () => {
  const f = fakes();
  const { controls } = fakeDom();
  let buttons;
  const loop = createTrainLoop({
    step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc,
    onStop: () => buttons.sync(),
  });
  buttons = trainButtons(controls, loop);
  assert.equal(controls.children[0], buttons.el);
  assert.equal(buttons.el.className, 'train-buttons');
  const [train, reset] = buttons.el.children;
  assert.equal(train.textContent, 'Train');
  assert.equal(reset.textContent, 'Reset');
  assert.equal('aria-pressed' in train.attrs, false);
  train.click();
  assert.equal(loop.running(), true);
  assert.equal(train.textContent, 'Pause');
  train.click();
  assert.equal(loop.running(), false);
  assert.equal(train.textContent, 'Train');
  train.click();
  buttons.stop();
  assert.equal(loop.running(), false);
  assert.equal(train.textContent, 'Train');
  assert.equal(f.queued(), 0);
  train.click();
  f.doc.hidden = true;
  f.fire();
  assert.equal(train.textContent, 'Train');
  let resets = 0;
  const r = trainButtons(controls, loop, { onReset: () => resets++ });
  r.el.children[1].click();
  assert.equal(resets, 1);

  // Reduced motion: Train starts a quiet run with work(n), and Train again pauses it.
  const g = fakes();
  const dom = fakeDom();
  let left = 50;
  let draws = 0;
  const l2 = createTrainLoop({
    step() { throw new Error('paced step must not run under reduced motion'); },
    render: () => draws++, done: () => left === 0, raf: g.raf, caf: g.caf, now: g.now, doc: g.doc,
    onStop: () => b2.sync(),
  });
  const b2 = trainButtons(dom.controls, l2, { reduced: true, work: (k) => { left -= k; } });
  b2.el.children[0].click();
  assert.equal(l2.activity(), 'computing');
  assert.equal(b2.el.children[0].textContent, 'Pause');
  b2.el.children[0].click();
  assert.equal(l2.running(), false);
  assert.equal(left, 50);
  b2.el.children[0].click();
  while (g.queued()) g.tick();
  assert.equal(left, 0);
  assert.equal(draws, 0);
  assert.equal(l2.running(), false);
  assert.equal(b2.el.children[0].textContent, 'Train');
});

test('trainStatus: one wording for every live lesson', () => {
  assert.equal(trainStatus({}), 'press Train');
  assert.equal(trainStatus({ running: false, iter: 0 }), 'press Train');
  assert.equal(trainStatus({ running: 'training', iter: 0 }), 'training…');
  assert.equal(trainStatus({ running: true, iter: 40 }), 'training…');
  assert.equal(trainStatus({ running: 'computing', iter: 40 }), 'computing…');
  assert.equal(trainStatus({ running: false, iter: 40 }), 'not finished: press Train');
  assert.equal(trainStatus({ running: false, iter: 400, done: true }), 'finished');
  assert.equal(trainStatus({ running: 'training', iter: 400, done: true }), 'finished');
  assert.equal(trainStatus({ running: false, iter: 7, diverged: true }), 'diverged');
  assert.equal(trainStatus({ running: false, iter: 7, done: true, diverged: true }), 'diverged');
  for (const w of ['press Train', 'training…', 'computing…', 'not finished: press Train', 'finished', 'diverged']) assert.ok(w.length <= 26);
});

test('activity() reads false, training or computing', () => {
  const f = fakes();
  const loop = createTrainLoop({ step() {}, render() {}, raf: f.raf, caf: f.caf, now: f.now, doc: f.doc });
  assert.equal(loop.activity(), false);
  loop.start();
  assert.equal(loop.activity(), 'training');
  loop.pause();
  loop.start({ quietStep: () => {} });
  assert.equal(loop.activity(), 'computing');
  loop.pause();
  assert.equal(loop.activity(), false);
  loop.start();
  assert.equal(loop.activity(), 'training');
});

test('a quiet run slices its work, renders nothing and ends in the same state', () => {
  const { X, y } = xorData({ n: 40, seed: 1 });
  const make = () => trainer({ net: mlpInit([2, 3, 1], { seed: 2 }), X, y, batch: 10, kind: 'adam', hyper: { eta: 0.05 }, seed: 3 });
  const END = 300;
  const ref = make();
  ref.step(END);

  const f = fakes();
  const t = make();
  let renders = 0;
  let stops = 0;
  const loop = createTrainLoop({
    step: (k) => t.step(Math.min(k, 2)), render: () => { renders++; },
    done: () => t.state.iter >= END, budgetMs: 8,
    raf: f.raf, caf: f.caf, now: f.now, doc: f.doc, onStop: () => stops++,
  });
  loop.start({ quietStep: (k) => t.step(Math.min(k, END - t.state.iter)) });
  let frames = 0;
  while (f.queued()) { f.tick(); frames++; }
  // The fake clock moves 1 ms per read, so each frame does a handful of steps.
  assert.ok(frames > 10, String(frames));
  assert.equal(renders, 0);
  assert.equal(stops, 1);
  assert.equal(loop.running(), false);
  assert.equal(t.state.iter, END);
  assert.deepEqual(t.state.batchLoss, ref.state.batchLoss);
  assert.deepEqual(t.state.epochLoss, ref.state.epochLoss);
});

test('every live lesson uses the shared status and a quiet run under reduced motion', async () => {
  const { readFileSync } = await import('node:fs');
  for (const slug of ['hidden-layers', 'backpropagation', 'training-networks', 'optimisers', 'deep-networks', 'cnns', 'rnns']) {
    const src = readFileSync(new URL(`../${slug}/stage.js`, import.meta.url), 'utf8');
    assert.match(src, /import \{ createTrainLoop, trainButtons, trainStatus \} from '\.\.\/assets\/train-loop\.js';/, slug);
    // No lesson keeps its own status ladder.
    assert.doesNotMatch(src, /return 'training…'|\? 'training…'|push\('training…'\)/, slug);
    assert.doesNotMatch(src, /export function trainStatus/, slug);
    // The read-out is told whether the loop is training or computing.
    assert.match(src, /loop\.activity\(\)/, slug);
    assert.doesNotMatch(src, /runToEnd/, slug);
    assert.match(src, /trainButtons\(trainGroup, loop, \{\n(?:    .*\n){0,4}    work: \(k\) =>/, slug);
  }
});
