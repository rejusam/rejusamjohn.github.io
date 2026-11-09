// Entry point for every ML Explorer page:
// <script type="module" src="/ml/assets/ml.js"></script>

import { PATH, TOTAL_LIVE, lessonHref } from './path.js';
import { createProgress } from './progress.js';
import { initCoffee } from './coffee.js';
import { startLesson } from './engine.js';
import { prefersReducedMotion } from './plot.js';

const root = document.documentElement;
// js-flag.js normally sets this first; kept here in case a page lacks it.
root.classList.add('js');

const progress = createProgress();
const lessonMain = document.querySelector('main.lesson[data-lesson]');
const here = lessonMain ? lessonMain.dataset.lesson : null;

function make(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text != null) n.textContent = text;
  return n;
}

// --- The path sidebar ---

let drawerClose = null;

function renderPath(nav) {
  const cap = make('p', 'ml-cap', 'The path');
  const levels = make('ol', 'ml-levels');

  for (const level of PATH) {
    const li = make('li', 'ml-level');
    if (level.soon) li.classList.add('is-soon');
    if (level.lessons.some((l) => l.slug === here)) li.classList.add('is-here');

    const body = make('div', 'ml-level-body');
    body.appendChild(make('p', 'ml-lt', level.name));

    if (level.soon) {
      body.appendChild(make('p', 'ml-topics', level.topics || ''));
      body.appendChild(make('p', 'ml-soon', 'Coming next'));
    } else {
      const ul = make('ul', 'ml-lessons');
      for (const lesson of level.lessons) {
        const item = make('li');
        const a = make('a', null, lesson.title);
        a.href = lessonHref(lesson.slug);
        if (lesson.slug === here) a.setAttribute('aria-current', 'page');
        if (progress.get(lesson.slug).done) {
          a.classList.add('is-done');
          a.appendChild(make('span', 'vh', ' (done)'));
        }
        item.appendChild(a);
        ul.appendChild(item);
      }
      body.appendChild(ul);
    }

    li.append(make('span', 'ml-lv', level.level), body);
    levels.appendChild(li);
  }

  nav.replaceChildren(...(drawerClose ? [drawerClose] : []), cap, levels);
}

// --- Top bar progress ---

function renderProgress(top) {
  let box = top.querySelector('.ml-progress');
  if (!box) {
    box = make('div', 'ml-progress');
    const label = make('span', 'ml-progress-label', 'Progress');
    const bar = make('span', 'ml-bar');
    bar.setAttribute('aria-hidden', 'true');
    bar.appendChild(make('i'));
    const count = make('span', 'ml-progress-count');
    box.append(label, bar, count);
    top.appendChild(box);
  }
  const done = Math.min(progress.doneCount(), TOTAL_LIVE);
  box.querySelector('.ml-bar i').style.width = `${(done / TOTAL_LIVE) * 100}%`;
  const count = box.querySelector('.ml-progress-count');
  count.textContent = `${done} / ${TOTAL_LIVE}`;
  count.setAttribute('aria-label', `${done} of ${TOTAL_LIVE} lessons done`);
}

// --- The "Path" drawer on narrow screens ---

function initDrawer(top, nav) {
  if (!nav.id) nav.id = 'ml-path';
  const toggle = make('button', 'ml-path-toggle', 'Path');
  toggle.type = 'button';
  toggle.setAttribute('aria-controls', nav.id);
  toggle.setAttribute('aria-expanded', 'false');
  const brand = top.querySelector('.ml-brand');
  if (brand) brand.after(toggle);
  else top.prepend(toggle);

  // The drawer's own close button; the page behind it is inert while open.
  drawerClose = make('button', 'ml-path-close', 'Close');
  drawerClose.type = 'button';
  nav.prepend(drawerClose);
  const behind = () => [top, document.querySelector('main'), document.querySelector('.coffee-float')].filter(Boolean);

  const scrim = make('div', 'ml-scrim');
  scrim.hidden = true;
  document.body.appendChild(scrim);

  const narrow = window.matchMedia('(max-width: 760px)');

  const setOpen = (open, { restoreFocus = true } = {}) => {
    root.classList.toggle('path-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    scrim.hidden = !open;
    for (const n of behind()) n.inert = open;
    if (open) {
      const first = nav.querySelector('a');
      if (first) first.focus();
    } else if (restoreFocus) {
      toggle.focus();
    }
  };

  toggle.addEventListener('click', () => setOpen(!root.classList.contains('path-open')));
  scrim.addEventListener('click', () => setOpen(false));
  drawerClose.addEventListener('click', () => setOpen(false));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && root.classList.contains('path-open')) setOpen(false);
  });
  narrow.addEventListener('change', () => {
    if (!narrow.matches && root.classList.contains('path-open')) setOpen(false, { restoreFocus: false });
  });
}

// --- Lift the coffee cup above the sticky step bar on phones ---

function trackControlsHeight() {
  const controls = document.querySelector('main.lesson .stage .controls');
  if (!controls || !('ResizeObserver' in window)) return;
  const narrow = window.matchMedia('(max-width: 760px)');
  const update = () => {
    const h = narrow.matches && !lessonMain.classList.contains('stage-failed') ? controls.offsetHeight : 0;
    root.style.setProperty('--controls-h', `${h}px`);
  };
  new ResizeObserver(update).observe(controls);
  narrow.addEventListener('change', update);
  update();
}

// --- Start ---

const top = document.querySelector('header.ml-top');
const nav = document.querySelector('nav.ml-path');

if (nav) renderPath(nav);
if (top) renderProgress(top);
if (top && nav) initDrawer(top, nav);
initCoffee(document);

if (lessonMain) {
  // The engine builds the controls synchronously, before it waits for
  // stage.js, so the bar height can be measured straight after the call.
  startLesson(lessonMain, {
    progress,
    reduced: prefersReducedMotion(),
    stageModuleUrl: new URL(`${lessonHref(here)}stage.js`, location.origin).href,
    onDone() {
      if (top) renderProgress(top);
      if (nav) renderPath(nav);
    },
  }).catch((err) => {
    console.error(err);
    lessonMain.classList.add('stage-failed');
  });
  trackControlsHeight();
}
