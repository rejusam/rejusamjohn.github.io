// Lesson L6.4, "Memory": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-agent-memory.test.mjs); the
// DOM is only touched inside createStage. Every trace comes from runAgent in agent.js, every
// word count from countWords, and every number in the page's text from pageValues.
// All text goes in with textContent.

import { DEFAULT_CONFIG, TASKS, runAgent, countWords } from '../assets/agent.js';
import { createTraceView } from '../assets/trace-view.js';
import { el } from '../assets/plot.js';

// ---- the task and its runs -------------------------------------------------------------

export const TASK = 'pair-shop';
// The notepad run needs 10 steps, so this lesson uses a step limit of 10 throughout.
export const STEP_LIMIT = 10;
export const SMALL_BUDGET = 12;
export const BUDGET_MIN = 6;
export const BUDGET_MAX = 60;
export const BUDGET_STEP = 2;
export const BUDGETS = Array.from({ length: (BUDGET_MAX - BUDGET_MIN) / BUDGET_STEP + 1 }, (_, k) => BUDGET_MIN + BUDGET_STEP * k);
export const STEP_COUNT = 5;

export const memoryRun = (budget, notepad) => runAgent(TASK, { stepLimit: STEP_LIMIT, budget, tools: { notepad } });
export const fullRun = () => memoryRun(Infinity, false);
export const taskWords = () => countWords(TASKS[TASK].text);

// The smallest slider budget at which the run is done.
export const minBudgetDone = (notepad) => BUDGETS.find((b) => memoryRun(b, notepad).stop === 'done');

// The first step whose tool call repeats an earlier one (not only the one just before it).
export function againStep(r) {
  const seen = new Set();
  for (const m of r.messages) {
    if (m.role !== 'call') continue;
    if (seen.has(m.text)) return m.step;
    seen.add(m.text);
  }
  return null;
}

// ---- the window of one model call -------------------------------------------------------

// Each row shows the window of the model call in its step (the task row: step 1's call).
export function viewOf(messages, i) {
  const step = Math.max(1, messages[i].step);
  const planIdx = messages.findIndex((m) => m.role === 'plan' && m.step === step);
  return { step, planIdx, seen: messages[planIdx].seen };
}

// Rows that call was not given: older than its plan line and outside its window.
export function unseenFlags(messages, planIdx) {
  const seen = new Set(messages[planIdx].seen);
  return messages.map((_, k) => k < planIdx && !seen.has(k));
}

// The messages the call was given, newest first, with their word counts, and the next older
// message: the first one that did not fit.
export function windowParts(messages, planIdx) {
  const seen = new Set(messages[planIdx].seen);
  const kept = [];
  for (let k = planIdx - 1; k > 0 && seen.has(k); k--) kept.push({ i: k, role: messages[k].role, words: countWords(messages[k].text) });
  const total = kept.reduce((s, k) => s + k.words, 0);
  const n = planIdx - 1 - kept.length;
  const next = n > 0 ? { i: n, role: messages[n].role, words: countWords(messages[n].text) } : null;
  return { step: messages[planIdx].step, kept, total, next };
}

// "4 + 1 + 7 = 12": the kept word counts, newest first, and their total.
export const sumText = (w) => `${w.kept.map((k) => k.words).join(' + ')} = ${w.total}`;

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// The longest text line in the picture, in characters; it fits at phone width.
export const LINE_MAX = 46;

export function windowLines(w, budget) {
  const finite = Number.isFinite(budget);
  const n = w.kept.length;
  let kept;
  if (finite) kept = `kept: ${n ? plural(n, 'newer message') : 'no newer messages'}, ${w.total} of ${budget} words`;
  else kept = n ? `kept: all ${plural(n, 'newer message')}, ${w.total} words` : 'kept: nothing newer yet, 0 words';
  let sum = n > 1 ? sumText(w) : null;
  if (sum && sum.length > LINE_MAX) sum = null;
  let next = null;
  let left = null;
  if (finite && w.next) {
    next = `next older: ${plural(w.next.words, 'word')}; ${w.total} + ${w.next.words} = ${w.total + w.next.words} > ${budget}`;
    left = 'so it and every older message are left out';
  } else if (finite) next = 'nothing older is left out';
  return { kept, sum, next, left };
}

// ---- the notes ---------------------------------------------------------------------------

// The notes written up to row i: one per notepad result row.
export function notesUpTo(messages, i) {
  const out = [];
  for (let k = 0; k <= i && k < messages.length; k++) {
    const m = messages[k];
    if (m.role === 'result' && m.tool === 'notepad' && m.input && m.input.action === 'write') out.push(m.input.text);
  }
  return out;
}

// ---- step 5: three runs side by side -----------------------------------------------------

export const COMPARE_RUNS = [
  { id: 'full', label: 'budget no limit, notepad off', budget: Infinity, notepad: false },
  { id: 'small', label: `budget ${SMALL_BUDGET} words, notepad off`, budget: SMALL_BUDGET, notepad: false },
  { id: 'note', label: `budget ${SMALL_BUDGET} words, notepad on`, budget: SMALL_BUDGET, notepad: true },
].map((c) => ({ ...c, run: () => memoryRun(c.budget, c.notepad) }));

export function outcomeText(r) {
  if (r.stop === 'done') return `done after ${plural(r.steps, 'step')}: ${r.answer}`;
  return `stopped by the ${r.stop} after ${plural(r.steps, 'step')}, no answer`;
}

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const full = fullRun();
  const small = memoryRun(SMALL_BUDGET, false);
  const note = memoryRun(SMALL_BUDGET, true);
  const circling = BUDGETS.filter((b) => memoryRun(b, false).stop === 'step limit');
  const anaCall = small.messages.find((m) => m.role === 'call');
  const applesCall = small.messages.filter((m) => m.role === 'call')[1];
  const anaResult = small.messages.find((m) => m.role === 'result');
  const again = againStep(small);
  const w = windowParts(small.messages, small.messages.findIndex((m) => m.role === 'plan' && m.step === again));
  const callTexts = small.messages.filter((m) => m.role === 'call').map((m) => m.text);
  return {
    'task': TASKS[TASK].text,
    'task-words': String(taskWords()),
    'limit': String(STEP_LIMIT),
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
    'full-steps': String(full.steps),
    'full-answer': full.answer,
    'small': String(SMALL_BUDGET),
    'b-min': String(BUDGET_MIN),
    'b-max': String(BUDGET_MAX),
    'min-done': String(minBudgetDone(false)),
    'circle-from': String(circling[0]),
    'circle-to': String(circling.at(-1)),
    'tiny-steps': String(memoryRun(BUDGET_MIN, false).steps),
    'ana-call': anaCall.text,
    'apples-call': applesCall.text,
    'ana-result': anaResult.text,
    'ana-words': String(countWords(anaResult.text)),
    'again-step': String(again),
    'again-sum': sumText(w),
    'again-over': `${w.total} + ${w.next.words} = ${w.total + w.next.words}`,
    'small-steps': String(small.steps),
    'ana-lookups': String(callTexts.filter((t) => t === anaCall.text).length),
    'apples-lookups': String(callTexts.filter((t) => t === applesCall.text).length),
    'note-steps': String(note.steps),
    'note-count': String(note.state.notes.length),
    'note-extra': String(note.steps - full.steps),
    'note-min': String(minBudgetDone(true)),
    'note-calc-step': String(note.messages.find((m) => m.role === 'call' && m.tool === 'calculator').step),
  };
}

// ---- the stage ------------------------------------------------------------------------

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const W = 440;
const BAR_X = 10;
const BAR_W = 420;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let budget = SMALL_BUDGET;
  let noLimit = false;
  let notepad = true;
  let compareIdx = 0;

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  const budgetNow = () => {
    if (step === 0) return Infinity;
    if (step === 1) return noLimit ? Infinity : budget;
    if (step === 4) return COMPARE_RUNS[compareIdx].budget;
    return SMALL_BUDGET;
  };
  const current = () => {
    if (step === 0) return fullRun();
    if (step === 1) return memoryRun(budgetNow(), false);
    if (step === 2) return memoryRun(SMALL_BUDGET, false);
    if (step === 3) return memoryRun(SMALL_BUDGET, notepad);
    return COMPARE_RUNS[compareIdx].run();
  };
  let result = current();

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  const host = doc.createElement('div');
  host.className = 'agent-trace';
  host.style.marginTop = '12px';
  svg.after(host);
  // onSelect fires for every selection, the view's own included. It greys the rows the selected
  // row's model call was not given (markUnseen never selects) and redraws the picture.
  const tv = createTraceView(host, {
    tools: controls, label: 'Trace: every message in and out', reduced, announce,
    onSelect: (i) => {
      if (result && i >= 0 && i < result.messages.length) tv.markUnseen(unseenFlags(result.messages, viewOf(result.messages, i).planIdx));
      draw();
    },
  });

  // ---- controls, one group per step that has one ----
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const groups = [];
  const group = (forStep, label) => {
    const g = make('div');
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', label);
    Object.assign(g.style, { display: 'none', flexWrap: 'wrap', alignItems: 'center', gap: '6px 14px' });
    controls.append(g);
    groups.push([forStep, g]);
    return g;
  };

  const rerun = (say) => {
    tv.stop();
    result = current();
    tv.show(result, { animate: false });
    tv.select(result.messages.length - 1);
    announce(say);
  };

  // Step 2: the budget, in words, or no limit.
  const budgetIn = make('input', { type: 'range', min: String(BUDGET_MIN), max: String(BUDGET_MAX), step: String(BUDGET_STEP), value: String(budget) });
  const budgetOut = make('output');
  budgetOut.setAttribute('aria-hidden', 'true');
  const budgetLab = make('label');
  budgetLab.append('budget (words)', ' ', budgetIn, ' ', budgetOut);
  const noLimitIn = make('input', { type: 'checkbox', checked: noLimit });
  noLimitIn.style.accentColor = CYAN;
  const noLimitLab = make('label');
  noLimitLab.append(noLimitIn, ' ', 'no limit');
  group(1, 'context budget').append(budgetLab, noLimitLab);
  const budgetSay = () => `Budget ${noLimit ? 'no limit' : `${budget} words`}: ${outcomeText(memoryRun(budgetNow(), false))}.`;
  budgetIn.addEventListener('input', () => {
    budget = Number(budgetIn.value);
    syncControls();
    rerun(budgetSay());
  });
  noLimitIn.addEventListener('change', () => {
    noLimit = noLimitIn.checked;
    syncControls();
    rerun(budgetSay());
  });

  // Step 4: the notepad, a switch.
  const noteIn = make('input', { type: 'checkbox', checked: notepad });
  noteIn.setAttribute('role', 'switch');
  noteIn.style.accentColor = CYAN;
  const noteLab = make('label');
  noteLab.append(noteIn, ' ', 'notepad');
  group(3, 'notepad').append(noteLab);
  noteIn.addEventListener('change', () => {
    notepad = noteIn.checked;
    rerun(`Notepad ${notepad ? 'on' : 'off'}: ${outcomeText(memoryRun(SMALL_BUDGET, notepad))}.`);
  });

  // Step 5: which run the trace shows.
  const compareSel = make('select');
  COMPARE_RUNS.forEach((c, k) => compareSel.append(make('option', { value: String(k), textContent: c.label })));
  const compareLab = make('label');
  compareLab.append('run', ' ', compareSel);
  group(4, 'run to show').append(compareLab);
  compareSel.addEventListener('change', () => {
    compareIdx = Number(compareSel.value);
    const c = COMPARE_RUNS[compareIdx];
    rerun(`${c.label}: ${outcomeText(c.run())}.`);
  });

  function syncControls() {
    budgetIn.value = String(budget);
    budgetIn.disabled = noLimit;
    budgetOut.textContent = noLimit ? 'no limit' : String(budget);
    budgetIn.setAttribute('aria-valuetext', `${budget} words`);
    noLimitIn.checked = noLimit;
    noteIn.checked = notepad;
    compareSel.value = String(compareIdx);
  }

  // Sizes are in viewBox units. At phone width ml.css sets class stage-text to 15 units,
  // so every line here is at most LINE_MAX characters.
  const label = (x, y, text, { anchor = 'start', fill = MUTED, size = 13, weight = 400 } = {}) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, svg);

  const frame = (h) => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
  };

  const selectedIdx = () => {
    const i = tv.selected;
    return result && i >= 0 && i < result.messages.length ? i : result.messages.length - 1;
  };

  const runLine = () => `${plural(result.steps, 'step')}, limit ${STEP_LIMIT}, stopped: ${result.stop}`;

  // Steps 1 to 4: the window of the selected row's model call, as a bar of word counts
  // (newest message on the left), then the notes on step 4.
  function drawWindow() {
    const b = budgetNow();
    const finite = Number.isFinite(b);
    const i = selectedIdx();
    const v = viewOf(result.messages, i);
    const w = windowParts(result.messages, v.planIdx);
    const lines = windowLines(w, b);
    const notes = step === 3 ? notesUpTo(result.messages, i) : null;
    const textLines = [lines.kept, lines.sum, lines.next, lines.left].filter(Boolean);
    const h = 132 + textLines.length * 22 + (notes ? 30 + Math.max(1, notes.length) * 20 : 0) + 30;
    frame(h);
    label(10, 20, `window of the model call in step ${v.step}`, { fill: STRONG, weight: 700 });
    label(10, 42, finite ? `budget: ${b} words` : 'budget: no limit');
    label(10, 62, `task: always kept, ${taskWords()} words, not counted`);
    // The bar: kept messages solid, the first that did not fit dashed, the budget a dashed line.
    const y0 = 86;
    const bh = 28;
    const unit = finite ? Math.max(b, w.total + (w.next ? w.next.words : 0)) : Math.max(w.total, 1);
    const px = BAR_W / unit;
    if (finite) el('rect', { x: BAR_X, y: y0, width: b * px, height: bh, fill: 'none', stroke: LINE, 'stroke-width': 1 }, svg);
    let x = BAR_X;
    for (const k of w.kept) {
      const wd = k.words * px;
      el('rect', { x, y: y0, width: wd, height: bh, fill: CYAN, 'fill-opacity': 0.18, stroke: CYAN, 'stroke-width': 1.25 }, svg);
      if (wd >= 16) label(x + wd / 2, y0 + 19, String(k.words), { anchor: 'middle', fill: STRONG, size: 12 });
      x += wd;
    }
    if (finite && w.next) {
      const wd = w.next.words * px;
      el('rect', { x, y: y0, width: wd, height: bh, fill: 'none', stroke: MUTED, 'stroke-width': 1.25, 'stroke-dasharray': '4 3' }, svg);
      if (wd >= 16) label(x + wd / 2, y0 + 19, String(w.next.words), { anchor: 'middle', size: 12 });
    }
    if (finite) {
      const bx = BAR_X + b * px;
      el('line', { x1: bx, x2: bx, y1: y0 - 8, y2: y0 + bh + 8, stroke: STRONG, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, svg);
    }
    label(10, y0 + bh + 18, 'newest', { size: 12 });
    label(BAR_W + BAR_X, y0 + bh + 18, 'older', { anchor: 'end', size: 12 });
    let y = y0 + bh + 18;
    for (const t of textLines) { y += 22; label(10, y, t, { fill: t === lines.kept ? STRONG : MUTED }); }
    if (notes) {
      y += 30;
      label(10, y, `notes, kept outside the window: ${notes.length}`, { fill: STRONG, weight: 700 });
      if (!notes.length) { y += 20; label(10, y, 'none yet'); }
      for (const t of notes) { y += 20; label(10, y, t); }
    }
    label(10, h - 10, runLine(), { fill: STRONG });
    const noteText = notes ? ` Notes so far, kept outside the window: ${notes.length ? notes.join(' ') : 'none'}.` : '';
    return `The window of the model call in step ${v.step}. ${finite ? `Budget ${b} words.` : 'No budget limit.'} The task is always kept (${taskWords()} words, not counted). ${textLines.join('. ')}.${noteText} This run: ${runLine()}.`;
  }

  // Step 5: the three runs, the one in the trace drawn with a thick outline and bold text.
  function drawCompare() {
    frame(232);
    label(10, 20, `the pair task, step limit ${STEP_LIMIT}`, { fill: STRONG, weight: 700 });
    const parts = [];
    COMPARE_RUNS.forEach((c, k) => {
      const on = k === compareIdx;
      const y = 34 + k * 64;
      el('rect', { x: 6, y, width: 428, height: 54, rx: 6, fill: CYAN, 'fill-opacity': on ? 0.12 : 0, stroke: on ? CYAN : LINE, 'stroke-width': on ? 2.5 : 1.25 }, svg);
      const out = outcomeText(c.run());
      label(16, y + 22, c.label, { fill: on ? STRONG : MUTED, weight: on ? 700 : 400 });
      label(16, y + 43, out, { fill: on ? STRONG : MUTED });
      parts.push(`${c.label}: ${out}${on ? ' (shown in the trace)' : ''}`);
    });
    return `Three runs of the pair task, step limit ${STEP_LIMIT}. ${parts.join('. ')}`;
  }

  function draw() {
    svg.setAttribute('aria-label', step === 4 ? drawCompare() : drawWindow());
  }

  syncControls();

  return {
    count: STEP_COUNT,
    show(i, animate = false) {
      tv.stop();
      step = i;
      for (const [s, g] of groups) g.style.display = s === i ? 'flex' : 'none';
      syncControls();
      result = current();
      tv.show(result, { animate });
      // Opened without a run: step 3 selects the plan line of the step that looks ana up again,
      // the others the stop row.
      if (!(animate && !reduced)) {
        const k = i === 2 ? result.messages.findIndex((m) => m.role === 'plan' && m.step === againStep(result)) : result.messages.length - 1;
        tv.select(k);
      }
      draw();
    },
  };
}
