// Live training loop for the Level 4 lessons.
//
// createTrainLoop drives a model one animation frame at a time: each frame runs
// step(n) for as many steps as fit in about budgetMs, then render(). It never
// touches document, requestAnimationFrame or performance at import time, so Node
// tests can inject fakes. trainButtons builds the Train/Pause/Reset buttons.

// Steps for the next frame, given how long the last frame's steps took.
export function adaptSteps(prev, elapsedMs, budgetMs = 8, { min = 1, max = 200 } = {}) {
  let n = prev;
  if (elapsedMs > budgetMs) n = Math.floor(prev * (budgetMs / elapsedMs));
  else if (elapsedMs < budgetMs / 2) n = Math.ceil(prev * 1.25);
  return Math.min(max, Math.max(min, n));
}

// The run's state in words, the same in every live lesson: "diverged" if the run
// blew up; "finished" once it is done; "computing…" while a reduced-motion run
// works out the end state; "training…" while the loop runs; "press Train" before
// the first step; "not finished: press Train" when paused part-way. `running` is
// the loop's activity(): false, 'training' or 'computing' (true reads as training).
export function trainStatus({ running = false, iter = 0, done = false, diverged = false } = {}) {
  if (diverged) return 'diverged';
  if (done) return 'finished';
  if (running === 'computing') return 'computing…';
  if (running) return 'training…';
  return iter === 0 ? 'press Train' : 'not finished: press Train';
}

// onStop() is called when the loop stops by itself (done, or the tab is hidden),
// so the buttons can resync. Calls to pause() and reset() do not trigger it.
//
// start({ quietStep }) runs quietly, for reduced motion: each frame calls
// quietStep(1) until about budgetMs has passed or done() is true, with no render
// in between, so nothing animates and the page stays responsive. The lesson's
// onStop draws the end state. quietStep must give the same states as step, only
// without pacing.
export function createTrainLoop({
  step, render, done = () => false, budgetMs = 8, onStop = () => {},
  raf, caf, now, doc,
}) {
  const request = raf || ((fn) => requestAnimationFrame(fn));
  const cancel = caf || ((id) => cancelAnimationFrame(id));
  const clock = now || (() => performance.now());
  const d = doc || (typeof document === 'undefined' ? null : document);
  const first = 1;
  let n = first;
  let id = null;
  let on = false;
  let quiet = null;

  const halt = () => {
    on = false;
    quiet = null;
    if (id !== null) cancel(id);
    id = null;
  };

  const frame = () => {
    id = null;
    if (!on) return;
    if (done()) { halt(); onStop(); return; }
    const t0 = clock();
    let elapsed;
    if (quiet) {
      try {
        do quiet(1); while (!done() && clock() - t0 < budgetMs);
      } catch (err) {
        halt();
        onStop();
        throw err;
      }
      if (done()) { halt(); onStop(); return; }
      id = request(frame);
      return;
    }
    try {
      step(n);
      elapsed = clock() - t0;
      render();
    } catch (err) {
      halt();
      onStop();
      throw err;
    }
    n = adaptSteps(n, elapsed, budgetMs);
    if (done()) { halt(); onStop(); return; }
    id = request(frame);
  };

  const api = {
    start({ quietStep = null } = {}) {
      if (on) return;
      on = true;
      quiet = quietStep;
      id = request(frame);
    },
    pause() { halt(); },
    reset(onReset) {
      halt();
      n = first;
      if (onReset) onReset();
      render();
    },
    running() { return on; },
    // false when stopped, 'computing' during a quiet run, 'training' otherwise.
    activity() { return on ? (quiet ? 'computing' : 'training') : false; },
  };

  if (d && typeof d.addEventListener === 'function') {
    d.addEventListener('visibilitychange', () => {
      if (d.hidden && on) { halt(); onStop(); }
    });
  }
  return api;
}

// Train / Pause / Reset buttons, appended to `controls` inside a .train-buttons
// container. Pass `onStop: () => buttons.sync()` to createTrainLoop so the label
// returns to "Train" when the loop finishes or the tab is hidden. Lessons call
// buttons.stop() in their show(i), never a bare loop.pause(), so the label stays
// true. The Train label flips Train / Pause and carries the state (no aria-pressed).
// Under `reduced` Train starts a quiet run with work(n), which advances the model
// n steps with no pacing: the read-out says "computing…" and only the end state is
// drawn (the lesson re-renders after any click on its Train group, and in onStop).
export function trainButtons(controls, loop, { onReset, reduced = false, work = null } = {}) {
  const doc = controls.ownerDocument;
  const el = doc.createElement('div');
  el.className = 'train-buttons';
  const train = doc.createElement('button');
  train.type = 'button';
  const reset = doc.createElement('button');
  reset.type = 'button';
  reset.textContent = 'Reset';
  el.append(train, reset);
  controls.append(el);

  const sync = () => {
    const on = loop.running();
    train.textContent = on ? 'Pause' : 'Train';
  };
  train.addEventListener('click', () => {
    if (loop.running()) loop.pause();
    else if (reduced && work) loop.start({ quietStep: work });
    else loop.start();
    sync();
  });
  reset.addEventListener('click', () => {
    loop.reset(onReset);
    sync();
  });
  sync();
  return {
    el,
    sync,
    stop() {
      loop.pause();
      sync();
    },
  };
}
