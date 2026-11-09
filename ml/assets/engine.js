// The step engine for Explorer lesson pages.
// The three small helpers at the top are pure and tested in Node; the DOM is
// only touched inside startLesson.

export function stepFromHash(hash, count) {
  const m = /^#step-([1-9]\d*)$/.exec(typeof hash === 'string' ? hash : '');
  if (!m) return 0;
  const n = Number(m[1]);
  return n <= count ? n - 1 : 0;
}

export function nextIndex(i, delta, count) {
  return Math.min(count - 1, Math.max(0, i + delta));
}

// Fields own every key. Buttons and summaries own Space (it activates
// them) but have no use for the left and right arrows, so those still step.
const FIELDS = new Set(['INPUT', 'SELECT', 'TEXTAREA']);
const ACTIVATORS = new Set(['BUTTON', 'SUMMARY']);
const ARROWS = new Set(['ArrowLeft', 'ArrowRight']);

export function shouldHandleKey(target, key) {
  if (!target) return true;
  if (target.isContentEditable) return false;
  const tag = String(target.tagName || '').toUpperCase();
  if (FIELDS.has(tag)) return false;
  if (ACTIVATORS.has(tag)) return ARROWS.has(key);
  return true;
}

const PLAY_MS = 2500;
const SVG_NS = 'http://www.w3.org/2000/svg';

function icon(d) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', '14');
  svg.setAttribute('height', '14');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', d);
  path.setAttribute('fill', 'currentColor');
  svg.appendChild(path);
  return svg;
}

const ICONS = {
  back: 'M11.5 2.5v11L3.5 8z',
  next: 'M4.5 2.5v11l8-5.5z',
  play: 'M4.5 2.5v11l8-5.5z',
  pause: 'M4 2.5h3v11H4zM9 2.5h3v11H9z',
};

function button(className, label, iconName, text) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  b.appendChild(icon(ICONS[iconName]));
  if (text) {
    const s = document.createElement('span');
    s.textContent = text;
    b.appendChild(s);
  } else {
    b.setAttribute('aria-label', label);
  }
  return b;
}

const textOf = (node) => (node ? node.textContent.replace(/\s+/g, ' ').trim() : '');

// Groups each step's parts so CSS can place the title and lede above the
// stage and the explanation beside it. Order inside each li is unchanged.
function shapeStep(li, index, count) {
  const main = document.createElement('div');
  main.className = 'step-main';
  const side = document.createElement('div');
  side.className = 'step-side';

  const counter = document.createElement('p');
  counter.className = 'step-count';
  counter.textContent = `Step ${index + 1} of ${count}`;
  main.appendChild(counter);

  for (const child of [...li.children]) {
    if (child.matches('.happening, .maths')) side.appendChild(child);
    else main.appendChild(child);
  }
  li.append(main, side);
}

function bindQuiz(main, saved, onAnswer) {
  const forms = [...main.querySelectorAll('section.quiz form.q')];
  const ids = forms.map((form, k) => form.id || form.dataset.q || `q${k + 1}`);

  forms.forEach((form, k) => {
    const id = ids[k];
    const answer = form.dataset.answer;
    const whys = [...form.querySelectorAll('.why')];
    whys.forEach((w) => { w.hidden = true; });

    const verdict = document.createElement('p');
    verdict.className = 'q-verdict';
    verdict.setAttribute('role', 'status');
    const firstWhy = whys[0];
    if (firstWhy) firstWhy.before(verdict);
    else form.appendChild(verdict);

    const render = (value) => {
      const right = value === answer;
      form.dataset.state = right ? 'right' : 'wrong';
      verdict.textContent = right ? 'Right.' : 'Not quite.';
      whys.forEach((w) => { w.hidden = w.dataset.for !== value; });
      form.querySelectorAll('input[type="radio"]').forEach((r) => {
        const label = r.closest('label');
        if (label) label.classList.toggle('is-answer', r.value === answer);
      });
    };

    form.addEventListener('submit', (e) => e.preventDefault());
    form.addEventListener('change', (e) => {
      const t = e.target;
      if (!(t instanceof HTMLInputElement) || t.type !== 'radio' || !t.checked) return;
      render(t.value);
      onAnswer(id, t.value);
    });

    const prior = saved[id];
    if (prior) {
      const radio = [...form.querySelectorAll('input[type="radio"]')].find((r) => r.value === prior);
      if (radio) {
        radio.checked = true;
        render(prior);
      }
    }
  });

  return ids;
}

/**
 * Runs one lesson page.
 * main: the main.lesson element. stageModuleUrl: where stage.js lives.
 * progress: a createProgress() store. reduced: prefers-reduced-motion.
 * onDone(): called once when the lesson becomes done.
 */
export async function startLesson(main, { progress, reduced = false, stageModuleUrl, onDone = () => {} }) {
  const slug = main.dataset.lesson;
  const steps = [...main.querySelectorAll('ol.steps > li.step')];
  const count = steps.length;
  const stageBox = main.querySelector('.stage');
  const svg = main.querySelector('#stage-svg');
  const controls = stageBox && stageBox.querySelector('.controls');
  if (!count) return null;
  if (!stageBox || !svg || !controls) {
    // No stage markup: keep every step readable.
    main.classList.add('stage-failed');
    return null;
  }

  steps.forEach((li, i) => shapeStep(li, i, count));

  // Controls: back, play, next | the lesson's own tools | step dots.
  const nav = document.createElement('div');
  nav.className = 'ml-nav';
  const back = button('ml-back', 'Previous step', 'back');
  const play = button('ml-play', '', 'play', 'Play');
  const next = button('ml-next', 'Next step', 'next');
  nav.append(back, play, next);

  const tools = document.createElement('div');
  tools.className = 'ml-tools';
  while (controls.firstChild) tools.appendChild(controls.firstChild);

  const dots = document.createElement('div');
  dots.className = 'ml-dots';
  dots.setAttribute('role', 'group');
  dots.setAttribute('aria-label', 'Steps');
  const dotButtons = steps.map((li, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ml-dot';
    b.setAttribute('aria-label', `Step ${i + 1}: ${textOf(li.querySelector('h2'))}`);
    dots.appendChild(b);
    return b;
  });

  controls.append(nav, tools, dots);

  // Two polite regions: one for the step text, one for what the stage
  // reports (a new value, a finished fit), so neither overwrites the other.
  const liveRegion = (className) => {
    const p = document.createElement('p');
    p.className = `${className} vh`;
    p.setAttribute('aria-live', 'polite');
    stageBox.appendChild(p);
    return p;
  };
  const live = liveRegion('ml-live');
  const stageLive = liveRegion('ml-live-stage');
  const announce = (text) => { stageLive.textContent = String(text); };

  // State.
  const saved = progress.get(slug);
  let current = -1;
  let timer = null;
  let reachedLast = saved.step >= count - 1;
  let done = saved.done;
  let answered = new Set(Object.keys(saved.quiz));
  let stage = null;

  const quizIds = bindQuiz(main, saved.quiz, (id, value) => {
    answered.add(id);
    progress.set(slug, { quiz: { [id]: value } });
    checkDone();
  });

  function checkDone() {
    if (done || !reachedLast) return;
    if (!quizIds.every((id) => answered.has(id))) return;
    done = true;
    progress.set(slug, { done: true });
    main.classList.add('is-done');
    onDone();
  }
  if (done) main.classList.add('is-done');

  function stopPlay() {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
    play.replaceChildren(icon(ICONS.play), Object.assign(document.createElement('span'), { textContent: 'Play' }));
    play.setAttribute('aria-pressed', 'false');
  }

  function writeStepHash() {
    const want = `#step-${current + 1}`;
    if (location.hash === want) return;
    try {
      history.replaceState(history.state, '', want);
    } catch {
      // Some sandboxed frames refuse; the lesson still works.
    }
  }

  function go(i, { animate = true, writeHash = true } = {}) {
    const target = nextIndex(i, 0, count);
    if (target === current) return;
    current = target;

    steps.forEach((li, k) => li.classList.toggle('is-current', k === current));
    dotButtons.forEach((b, k) => {
      b.classList.toggle('is-seen', k < current);
      if (k === current) b.setAttribute('aria-current', 'step');
      else b.removeAttribute('aria-current');
    });
    back.disabled = current === 0;
    next.disabled = current === count - 1;

    if (writeHash) writeStepHash();

    const li = steps[current];
    live.textContent = `Step ${current + 1} of ${count}: ${textOf(li.querySelector('h2'))}. ${textOf(li.querySelector('.lede'))}`;
    progress.set(slug, { step: current });

    if (current === count - 1) {
      reachedLast = true;
      checkDone();
    }

    if (stage) {
      try {
        const r = stage.show(current, animate && !reduced);
        if (r && typeof r.catch === 'function') r.catch((err) => console.error(err));
      } catch (err) {
        console.error(err);
      }
    }
  }

  const userGo = (i) => {
    stopPlay();
    go(i);
  };

  back.addEventListener('click', () => userGo(current - 1));
  next.addEventListener('click', () => userGo(current + 1));
  dotButtons.forEach((b, i) => b.addEventListener('click', () => userGo(i)));

  play.setAttribute('aria-pressed', 'false');
  play.addEventListener('click', () => {
    if (timer !== null) {
      stopPlay();
      return;
    }
    if (current >= count - 1) go(0);
    play.replaceChildren(icon(ICONS.pause), Object.assign(document.createElement('span'), { textContent: 'Pause' }));
    play.setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      if (current >= count - 1) {
        stopPlay();
        return;
      }
      go(current + 1);
      if (current >= count - 1) stopPlay();
    }, PLAY_MS);
  });

  // Space advances only while the stage is on screen, so it still scrolls
  // the page when the reader is down in the quiz or the reading list.
  let stageVisible = true;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      for (const e of entries) stageVisible = e.isIntersecting;
    }).observe(stageBox);
  }

  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    if (!shouldHandleKey(e.target, e.key)) return;
    if (e.target instanceof Element && e.target.closest('[role="dialog"], .ml-path')) return;
    if (e.key === 'ArrowRight') userGo(current + 1);
    else if (e.key === 'ArrowLeft') userGo(current - 1);
    else if ((e.key === ' ' || e.key === 'Spacebar') && !e.shiftKey && stageVisible) {
      e.preventDefault();
      userGo(current + 1);
    }
  });

  window.addEventListener('hashchange', () => {
    if (/^#step-\d+$/.test(location.hash)) {
      stopPlay();
      go(stepFromHash(location.hash, count));
      // An out-of-range #step-N falls back to step 1; show that in the URL.
      writeStepHash();
    }
  });

  // First step: the link's #step-N, else where the reader left off.
  const hasHash = /^#step-\d+$/.test(location.hash);
  const first = hasHash ? stepFromHash(location.hash, count) : nextIndex(saved.step, 0, count);
  go(first, { animate: false, writeHash: hasHash || first > 0 });
  main.classList.add('is-live');

  // The stage module is loaded after the text is in place, so a slow or
  // failing visual never hides the lesson.
  try {
    const mod = await import(stageModuleUrl);
    const made = mod.createStage(svg, { controls: tools, reduced, announce });
    if (!made || made.count !== count) {
      throw new Error(`stage.js reports ${made && made.count} steps; the page has ${count}`);
    }
    stage = made;
    await stage.show(current, false);
    main.classList.add('stage-ready');
  } catch (err) {
    console.error(err);
    stopPlay();
    main.classList.add('stage-failed');
  }

  if (hasHash) {
    const target = steps[current].querySelector('.step-main');
    if (target) target.scrollIntoView({ block: 'start' });
  }

  return { go: userGo, get index() { return current; } };
}
