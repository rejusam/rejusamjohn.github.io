// Lesson progress for the ML Explorer, kept in localStorage under one key.
// Every read and write is wrapped: the pages must work when storage is
// missing, full, blocked or holding something unexpected.

import { PATH } from './path.js';

const KEY = 'rsj-ml:v1';

// Only lessons on the current path count towards progress, so a renamed or
// retired lesson left in storage never inflates the total.
const LIVE = new Set(PATH.flatMap((level) => level.lessons.map((l) => l.slug)));

const blank = () => ({ v: 1, lessons: {} });

function cleanLesson(raw) {
  const out = { step: 0, done: false, quiz: {} };
  if (!raw || typeof raw !== 'object') return out;
  if (Number.isInteger(raw.step) && raw.step >= 0) out.step = raw.step;
  if (raw.done === true) out.done = true;
  if (raw.quiz && typeof raw.quiz === 'object' && !Array.isArray(raw.quiz)) {
    for (const [k, v] of Object.entries(raw.quiz)) {
      if (typeof v === 'string') out.quiz[k] = v;
    }
  }
  return out;
}

// Reading globalThis.localStorage itself throws in some privacy modes, so the
// default is fetched inside a try.
function defaultStorage() {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

export function createProgress(storage = defaultStorage()) {
  function load() {
    try {
      const text = storage && storage.getItem(KEY);
      if (!text) return blank();
      const data = JSON.parse(text);
      if (!data || data.v !== 1 || !data.lessons || typeof data.lessons !== 'object' || Array.isArray(data.lessons)) return blank();
      return data;
    } catch {
      return blank();
    }
  }

  function save(data) {
    try {
      if (storage) storage.setItem(KEY, JSON.stringify(data));
    } catch {
      // Storage is full or blocked; progress simply is not kept.
    }
  }

  return {
    get(slug) {
      return cleanLesson(load().lessons[slug]);
    },
    set(slug, patch = {}) {
      try {
        const state = load();
        const cur = cleanLesson(state.lessons[slug]);
        const next = cleanLesson({
          ...cur,
          ...patch,
          quiz: { ...cur.quiz, ...(patch.quiz || {}) },
        });
        state.lessons[slug] = next;
        save(state);
      } catch {
        // Never throw from here.
      }
    },
    doneCount() {
      try {
        return Object.entries(load().lessons)
          .filter(([slug, l]) => LIVE.has(slug) && cleanLesson(l).done).length;
      } catch {
        return 0;
      }
    },
  };
}
