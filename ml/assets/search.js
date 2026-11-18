// Lesson search for the top bar: "/" or Ctrl/Cmd+K opens it on every
// Explorer page. searchLessons is pure and tested in Node; initSearch builds
// the dialog. Every string is set with textContent, never as HTML.

import { PATH, lessonHref } from './path.js';

// Lower case, no accents, single spaces.
export function fold(s) {
  return String(s)
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

// Every live lesson, in path order, with its level.
export function lessonIndex(path = PATH) {
  const out = [];
  for (const level of path) {
    if (level.soon) continue;
    for (const lesson of level.lessons) {
      out.push({
        slug: lesson.slug,
        title: lesson.title,
        level: level.level,
        levelName: level.name,
        order: out.length,
      });
    }
  }
  return out;
}

// How well one lesson matches the folded query words. 0 means no match.
// Every query word must match the start of a word in the title or the level
// name (or the level code, such as "l3"). Title matches beat level matches.
function score(item, words) {
  const title = fold(item.title).split(' ');
  const level = [...fold(item.levelName).split(' '), fold(item.level)];
  let total = 0;
  for (const w of words) {
    if (title.some((t) => t === w)) total += 4;
    else if (title.some((t) => t.startsWith(w))) total += 3;
    else if (level.some((t) => t.startsWith(w))) total += 1;
    else return 0;
  }
  if (fold(item.title).startsWith(words.join(' '))) total += 2;
  return total;
}

// The lessons matching the query, best first, path order on ties.
// An empty query lists every lesson in path order.
export function searchLessons(query, index = lessonIndex()) {
  const words = fold(query).split(' ').filter(Boolean);
  if (!words.length) return index.slice();
  return index
    .map((item) => ({ item, s: score(item, words) }))
    .filter((r) => r.s > 0)
    .sort((a, b) => b.s - a.s || a.item.order - b.item.order)
    .map((r) => r.item);
}

// Whether a keydown should open the search: "/" outside a field, or
// Ctrl/Cmd+K anywhere.
export function opensSearch(e) {
  const k = String(e.key || '').toLowerCase();
  if ((e.metaKey || e.ctrlKey) && !e.altKey && k === 'k') return true;
  if (e.metaKey || e.ctrlKey || e.altKey || k !== '/') return false;
  const t = e.target;
  if (!t) return true;
  if (t.isContentEditable) return false;
  return !['INPUT', 'SELECT', 'TEXTAREA'].includes(String(t.tagName || '').toUpperCase());
}

function make(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text != null) n.textContent = text;
  return n;
}

// Builds the search button and dialog. top: the header.ml-top element.
// isDone(slug) marks finished lessons; here is the current lesson's slug.
export function initSearch(top, { isDone = () => false, here = null } = {}) {
  if (!top || typeof HTMLDialogElement === 'undefined') return;

  const mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
  const open = make('button', 'ml-search-open');
  open.type = 'button';
  open.setAttribute('aria-haspopup', 'dialog');
  open.setAttribute('aria-label', 'Search lessons');
  const glass = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  glass.setAttribute('viewBox', '0 0 16 16');
  glass.setAttribute('width', '14');
  glass.setAttribute('height', '14');
  glass.setAttribute('aria-hidden', 'true');
  glass.setAttribute('focusable', 'false');
  const lens = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lens.setAttribute('d', 'M7 2a5 5 0 1 0 3.1 8.9l3.5 3.5 1.1-1.1-3.5-3.5A5 5 0 0 0 7 2zm0 1.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7z');
  lens.setAttribute('fill', 'currentColor');
  glass.appendChild(lens);
  open.append(glass, make('span', 'ml-search-label', 'Search'));
  const hint = make('kbd', 'ml-search-kbd', mac ? '⌘K' : 'Ctrl K');
  hint.setAttribute('aria-hidden', 'true');
  open.append(hint);
  const site = top.querySelector('.ml-site');
  if (site) site.after(open);
  else top.appendChild(open);

  const dialog = make('dialog', 'ml-search');
  dialog.setAttribute('aria-label', 'Search lessons');
  const form = make('form', 'ml-search-form');
  form.setAttribute('role', 'search');
  const label = make('label', 'vh', 'Search lessons');
  label.htmlFor = 'ml-search-q';
  const input = make('input', 'ml-search-q');
  input.id = 'ml-search-q';
  input.type = 'search';
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.placeholder = 'Search lessons, e.g. attention or trees';
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-controls', 'ml-search-list');
  input.setAttribute('aria-autocomplete', 'list');
  const close = make('button', 'ml-search-close', 'Esc');
  close.type = 'button';
  close.setAttribute('aria-label', 'Close search');
  form.append(label, input, close);

  const list = make('ul', 'ml-search-list');
  list.id = 'ml-search-list';
  list.setAttribute('role', 'listbox');
  list.setAttribute('aria-label', 'Lessons');
  const empty = make('p', 'ml-search-empty', 'No lesson matches that. Try one word, such as "trees".');
  empty.hidden = true;
  const count = make('p', 'vh');
  count.setAttribute('aria-live', 'polite');
  dialog.append(form, list, empty, count);
  document.body.appendChild(dialog);

  const index = lessonIndex();
  let results = [];
  let active = 0;

  const optionId = (i) => `ml-search-opt-${i}`;

  function paint() {
    const items = [...list.children];
    items.forEach((li, i) => li.setAttribute('aria-selected', String(i === active)));
    if (results.length) {
      input.setAttribute('aria-activedescendant', optionId(active));
      const el = items[active];
      if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ block: 'nearest' });
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  function render() {
    results = searchLessons(input.value, index);
    active = 0;
    list.replaceChildren(
      ...results.map((r, i) => {
        const li = make('li', 'ml-search-item');
        li.id = optionId(i);
        li.setAttribute('role', 'option');
        const row = make('span', 'ml-search-row');
        row.append(make('span', 'ml-search-lv', r.level), make('span', 'ml-search-title', r.title));
        if (r.slug === here) row.append(make('span', 'ml-search-tag', 'This lesson'));
        else if (isDone(r.slug)) row.append(make('span', 'ml-search-tag', 'Done'));
        li.appendChild(row);
        li.addEventListener('click', () => choose(r));
        li.addEventListener('mousemove', () => {
          if (active !== i) {
            active = i;
            paint();
          }
        });
        return li;
      }),
    );
    empty.hidden = results.length > 0;
    input.setAttribute('aria-expanded', String(results.length > 0));
    count.textContent = results.length === 1 ? '1 lesson' : `${results.length} lessons`;
    paint();
  }

  // The current lesson just closes the search, so its state is kept.
  function choose(r) {
    if (r.slug === here) dialog.close();
    else location.assign(lessonHref(r.slug));
  }

  function show() {
    if (dialog.open) return;
    input.value = '';
    render();
    dialog.showModal();
    input.focus();
  }

  open.addEventListener('click', show);
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });
  dialog.addEventListener('close', () => open.focus());
  input.addEventListener('input', render);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const r = results[active];
    if (r) choose(r);
  });
  input.addEventListener('keydown', (e) => {
    if (!results.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      active = (active + 1) % results.length;
      paint();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      active = (active - 1 + results.length) % results.length;
      paint();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || dialog.open) return;
    if (!opensSearch(e)) return;
    if (e.target instanceof Element && e.target.closest('dialog')) return;
    e.preventDefault();
    show();
  });
}
