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

// onStop() is called when the loop stops by itself (done, or the tab is hidden),
// so the buttons can resync. Calls to pause() and reset() do not trigger it.
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

  const halt = () => {
    on = false;
    if (id !== null) cancel(id);
    id = null;
  };

  const frame = () => {
    id = null;
    if (!on) return;
    if (done()) { halt(); onStop(); return; }
    const t0 = clock();
    step(n);
    const elapsed = clock() - t0;
    render();
    n = adaptSteps(n, elapsed, budgetMs);
    if (done()) { halt(); onStop(); return; }
    id = request(frame);
  };

  const api = {
    start() {
      if (on) return;
      on = true;
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
// returns to "Train" when the loop finishes or the tab is hidden. Under `reduced`
// Train runs to the end state (runToEnd) and renders once; nothing animates.
export function trainButtons(controls, loop, { onReset, reduced = false, runToEnd = () => {}, render = () => {} } = {}) {
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
    train.setAttribute('aria-pressed', on ? 'true' : 'false');
  };
  train.addEventListener('click', () => {
    if (reduced) {
      runToEnd();
      render();
    } else if (loop.running()) {
      loop.pause();
    } else {
      loop.start();
    }
    sync();
  });
  reset.addEventListener('click', () => {
    loop.reset(onReset);
    sync();
  });
  sync();
  return { el, sync };
}
