// Lesson L6.5, "Mistakes and limits": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-agent-mistakes.test.mjs); the
// DOM is only touched inside createStage. Every trace comes from runAgent in agent.js, except
// the one written-in tool-error case, whose rows are built from the calculator's real output.
// Every number in the page's text comes from pageValues. All text goes in with textContent.

import { DEFAULT_CONFIG, TASKS, TOOLS, runAgent, callText, fmtMoney, PRICES } from '../assets/agent.js';
import { WORLD } from '../assets/world.js';
import { createTraceView } from '../assets/trace-view.js';
import { el } from '../assets/plot.js';

// ---- tasks and runs --------------------------------------------------------------------

export const TASK_NF = 'kai-food';
export const TASK_SUM = 'sum';
export const TASK_SEND = 'tell-zoe';
export const STEP_COUNT = 6;
export const TOOL_NAMES = Object.keys(TOOLS);

export const NOT_FOUND_OPTIONS = ['say so', 'try again', 'guess'];
export const notFoundRun = (option) => runAgent(TASK_NF, { onNotFound: option });
export const defaultRun = (id) => runAgent(id);

// approve: undefined (no answer yet), true or false. ask: the "ask before changes" switch.
export function sendRun(approve, ask) {
  return runAgent(TASK_SEND, { askBeforeChanges: ask }, approve === undefined ? {} : { approve });
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);

export function outcomeText(r) {
  if (r.stop === 'done') return `done after ${plural(r.steps, 'step')}: ${r.answer}`;
  return `stopped: ${r.stop} after ${plural(r.steps, 'step')}, no answer`;
}

// The longest text line in the picture, in characters; it fits at phone width.
export const LINE_MAX = 46;

// outcomeText split into picture lines of at most LINE_MAX characters.
export function outcomeLines(r, prefix = '') {
  const one = prefix + outcomeText(r);
  if (one.length <= LINE_MAX) return [one];
  if (r.stop === 'done') return [`${prefix}done after ${plural(r.steps, 'step')}:`, r.answer];
  return [`${prefix}stopped: ${r.stop}`, `after ${plural(r.steps, 'step')}, no answer`];
}

// ---- a tool error: a written-in case ----------------------------------------------------

// The stand-in never writes a call the calculator cannot read, so this case is written in:
// the call row shows the malformed input itself, and the result row is the calculator's
// real output for it. The stop follows the loop's own rule (stopAfter, as in runAgent).
export const BAD_EXPR = '2+';

// The loop's rule after a tool runs: a failed result that is not lookup's "not found" stops the run.
export function stopAfter(tool, r) {
  const notFound = tool === 'lookup' && r.text.startsWith('not found: ');
  return !r.ok && !notFound ? 'tool error' : null;
}

export const WHY_CALL = 'No rule wrote this call: it is written into the page as an example of a broken call a language model could write. The stand-in never writes it.';
export const WHY_RESULT = "The calculator's real reply to this input: the loop ran the tool. No rule applies.";
export const WHY_STOP = 'No rule applies: the loop on this page stops when a tool fails, apart from lookup\'s "not found". Here the calculator could not read its input.';

export function toolErrorCase() {
  const input = { expr: BAD_EXPR };
  const r = TOOLS.calculator.run(input, {});
  const stop = stopAfter('calculator', r);
  return {
    writtenIn: true,
    messages: [
      { role: 'task', text: TASKS[TASK_SUM].text, step: 0 },
      { role: 'call', text: callText('calculator', input), step: 1, tool: 'calculator', input, why: WHY_CALL },
      { role: 'result', text: r.text, step: 1, tool: 'calculator', input, why: WHY_RESULT },
      { role: 'stop', text: stop, step: 1, why: WHY_STOP },
    ],
    stop, answer: null, steps: 1, state: { notes: [], sent: [] },
  };
}

// ---- only the tools a task needs ----------------------------------------------------------

// "Needs" here means: the task's default run writes a call for the tool, or asks to.
export function needsTable() {
  return Object.keys(TASKS).map((id) => {
    const r = defaultRun(id);
    const called = TOOL_NAMES.filter((n) => r.messages.some((m) => m.role === 'call' && m.tool === n));
    const asked = TOOL_NAMES.filter((n) => r.messages.some((m) => m.role === 'approval' && m.tool === n));
    const needs = TOOL_NAMES.filter((n) => called.includes(n) || asked.includes(n));
    return { id, called, asked, needs };
  });
}

export const unusedTools = () => TOOL_NAMES.filter((n) => !needsTable().some((row) => row.needs.includes(n)));

export function onlyNeededRun(id) {
  const row = needsTable().find((x) => x.id === id);
  return runAgent(id, { tools: Object.fromEntries(TOOL_NAMES.map((n) => [n, row.needs.includes(n)])) });
}

export function rowLine(row) {
  const parts = [];
  if (row.called.length) parts.push(row.called.join(', '));
  if (row.asked.length) parts.push(`${row.asked.join(', ')} after a yes`);
  return `${row.id}: ${parts.join('; ')}`;
}

// ---- ask before changes -----------------------------------------------------------------

// The sent list. The send tool on this page only records the message; nothing leaves the page.
export function sentLines(r) {
  const s = r.state.sent;
  if (!s.length) return ['sent: nothing'];
  return [`sent: ${plural(s.length, 'message')}`, ...s.map((m) => `to ${m.to}: "${m.text}"`)];
}

// ---- every run on the page, and what each safeguard stopped -------------------------------

export const PAGE_RUNS = [
  { id: 'say', label: 'kai, "say so"', run: () => notFoundRun('say so') },
  { id: 'again', label: 'kai, "try again"', run: () => notFoundRun('try again') },
  { id: 'guess', label: 'kai, "guess"', run: () => notFoundRun('guess') },
  { id: 'error', label: 'the calculator given "2+"', run: toolErrorCase },
  { id: 'wait', label: 'zoe, no answer yet', run: () => sendRun(undefined, true) },
  { id: 'yes', label: 'zoe, approved', run: () => sendRun(true, true) },
  { id: 'no', label: 'zoe, declined', run: () => sendRun(false, true) },
  { id: 'off', label: 'zoe, ask before changes off', run: () => sendRun(undefined, false) },
];

const runOf = (id) => PAGE_RUNS.find((p) => p.id === id).run();

// The stop reasons of every run the page shows (step 4's default runs included).
export function pageStops() {
  const all = [...PAGE_RUNS.map((p) => p.run()), ...Object.keys(TASKS).map(defaultRun)];
  return [...new Set(all.map((r) => r.stop))].sort();
}

export function safeguards() {
  const say = runOf('say');
  const again = runOf('again');
  const err = runOf('error');
  const wait = runOf('wait');
  const no = runOf('no');
  const nothing = (r) => (r.state.sent.length ? `${plural(r.state.sent.length, 'message')} sent` : 'nothing sent');
  return [
    { id: 'say-so', runIds: ['say'], runs: [say], lines: ['the instruction "say so"', `${say.stop}: "${say.answer}" (no guess)`] },
    { id: 'repeat', runIds: ['again'], runs: [again], lines: ['repeat check', `stopped: ${again.stop} after ${plural(again.steps, 'step')}`] },
    { id: 'tool-error', runIds: ['error'], runs: [err], lines: ['stop on a tool error', `stopped: ${err.stop} after ${plural(err.steps, 'step')}`] },
    { id: 'ask', runIds: ['wait', 'no'], runs: [wait, no], lines: ['ask before changes', `stopped: ${wait.stop}, ${nothing(wait)}`, `after a no: ${no.stop}, ${nothing(no)}`] },
  ];
}

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const say = notFoundRun('say so');
  const again = notFoundRun('try again');
  const guess = notFoundRun('guess');
  const sum = defaultRun(TASK_SUM);
  const err = toolErrorCase();
  const wait = sendRun(undefined, true);
  const yes = sendRun(true, true);
  const no = sendRun(false, true);
  const off = sendRun(undefined, false);
  const ask = wait.messages.find((m) => m.role === 'approval');
  return {
    'kai-task': TASKS[TASK_NF].text,
    'kai-call': say.messages.find((m) => m.role === 'call').text,
    'kai-result': say.messages.find((m) => m.role === 'result').text,
    'sayso-answer': say.answer,
    'sayso-steps': String(say.steps),
    'again-steps': String(again.steps),
    'again-runs': String(again.messages.filter((m) => m.role === 'result').length),
    'guess-answer': guess.answer,
    'first-food': WORLD.foods[0],
    'people-count': String(WORLD.people.length),
    'sum-task': TASKS[TASK_SUM].text,
    'bad-call': err.messages.find((m) => m.role === 'call').text,
    'bad-result': err.messages.find((m) => m.role === 'result').text,
    'good-call': sum.messages.find((m) => m.role === 'call').text,
    'good-result': sum.messages.find((m) => m.role === 'result').text,
    'task-count': String(Object.keys(TASKS).length),
    'tool-count': String(TOOL_NAMES.length),
    'unused-tools': listText(unusedTools()),
    'send-kind': TOOLS.send.kind,
    'zoe-task': TASKS[TASK_SEND].text,
    'zoe-ask': callText(ask.tool, ask.input),
    'wait-steps': String(wait.steps),
    'yes-steps': String(yes.steps),
    'no-steps': String(no.steps),
    'off-steps': String(off.steps),
    'yes-answer': yes.answer,
    'yes-sent': String(yes.state.sent.length),
    'pears-price': fmtMoney(PRICES.pears),
    'stop-count': String(pageStops().length),
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
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

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let nfIdx = 0;
  let taskIdx = Object.keys(TASKS).indexOf(TASK_SEND);
  let approve;
  let ask = true;
  let runIdx = 0;

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  const table = needsTable();
  const current = () => {
    if (step === 0) return notFoundRun('say so');
    if (step === 1) return notFoundRun(NOT_FOUND_OPTIONS[nfIdx]);
    if (step === 2) return toolErrorCase();
    if (step === 3) return defaultRun(table[taskIdx].id);
    if (step === 4) return sendRun(approve, ask);
    return PAGE_RUNS[runIdx].run();
  };
  let result = current();

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  const host = doc.createElement('div');
  host.className = 'agent-trace';
  host.style.marginTop = '12px';
  svg.after(host);
  // onSelect fires for every selection, the view's own included; draw() never selects, so there is no loop.
  const tv = createTraceView(host, { tools: controls, label: 'Trace: every message in and out', reduced, announce, onSelect: () => draw() });

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
  const selectOf = (labelText, options, forStep, groupLabel, onChange) => {
    const sel = make('select');
    options.forEach((t, k) => sel.append(make('option', { value: String(k), textContent: t })));
    const lab = make('label');
    lab.append(labelText, ' ', sel);
    group(forStep, groupLabel).append(lab);
    sel.addEventListener('change', () => onChange(Number(sel.value)));
    return sel;
  };

  const rerun = (say) => {
    tv.stop();
    result = current();
    tv.show(result, { animate: false });
    tv.select(result.messages.length - 1);
    announce(say);
  };

  // Step 2: what to do when lookup finds nothing.
  const nfSel = selectOf('when lookup finds nothing', NOT_FOUND_OPTIONS, 1, 'instruction for not found', (k) => {
    nfIdx = k;
    rerun(`${NOT_FOUND_OPTIONS[k]}: ${outcomeText(notFoundRun(NOT_FOUND_OPTIONS[k]))}.`);
  });

  // Step 4: which task's default run the trace shows.
  const taskSel = selectOf('task', table.map((row) => row.id), 3, 'task to show', (k) => {
    taskIdx = k;
    rerun(`${rowLine(table[k])}. ${outcomeText(defaultRun(table[k].id))}.`);
  });

  // Step 5: the answer to the ask, and the switch.
  const yesBtn = make('button', { type: 'button', textContent: 'Approve' });
  const noBtn = make('button', { type: 'button', textContent: 'Decline' });
  const resetBtn = make('button', { type: 'button', textContent: 'No answer yet' });
  const askIn = make('input', { type: 'checkbox', checked: ask });
  askIn.setAttribute('role', 'switch');
  askIn.style.accentColor = CYAN;
  const askLab = make('label');
  askLab.append(askIn, ' ', 'ask before changes');
  const sendGroup = group(4, 'answer to the ask');
  sendGroup.append(yesBtn, noBtn, resetBtn, askLab);
  for (const b of [yesBtn, noBtn, resetBtn]) {
    Object.assign(b.style, { font: 'inherit', fontSize: '0.875rem', padding: '4px 12px', borderRadius: '6px', border: `1px solid ${LINE}`, background: 'transparent', color: STRONG, cursor: 'pointer' });
  }
  const sendSay = () => `${ask ? (approve === undefined ? 'No answer yet' : approve ? 'Approved' : 'Declined') : 'Ask before changes off'}: ${outcomeText(sendRun(approve, ask))}. ${sentLines(sendRun(approve, ask)).join('; ')}.`;
  const answer = (a) => { approve = a; syncControls(); rerun(sendSay()); };
  yesBtn.addEventListener('click', () => answer(true));
  noBtn.addEventListener('click', () => answer(false));
  resetBtn.addEventListener('click', () => answer(undefined));
  askIn.addEventListener('change', () => { ask = askIn.checked; syncControls(); rerun(sendSay()); });

  // Step 6: which run of the page the trace shows.
  const runSel = selectOf('run', PAGE_RUNS.map((p) => p.label), 5, 'run to show', (k) => {
    runIdx = k;
    rerun(`${PAGE_RUNS[k].label}: ${outcomeText(PAGE_RUNS[k].run())}.`);
  });

  function syncControls() {
    nfSel.value = String(nfIdx);
    taskSel.value = String(taskIdx);
    runSel.value = String(runIdx);
    askIn.checked = ask;
    yesBtn.disabled = !ask;
    noBtn.disabled = !ask;
    resetBtn.disabled = !ask;
    yesBtn.setAttribute('aria-pressed', String(ask && approve === true));
    noBtn.setAttribute('aria-pressed', String(ask && approve === false));
    resetBtn.setAttribute('aria-pressed', String(ask && approve === undefined));
    for (const b of [yesBtn, noBtn, resetBtn]) {
      const on = b.getAttribute('aria-pressed') === 'true';
      b.style.borderColor = on ? CYAN : LINE;
      b.style.fontWeight = on ? '700' : '400';
      b.style.opacity = b.disabled ? '0.5' : '1';
      b.style.cursor = b.disabled ? 'default' : 'pointer';
    }
  }

  // Sizes are in viewBox units. At phone width ml.css sets class stage-text to 15 units,
  // so every line here is at most LINE_MAX characters.
  const label = (x, y, text, { anchor = 'start', fill = MUTED, size = 13, weight = 400 } = {}) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, svg);

  const frame = (h) => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
    const m = el('marker', { id: 'am-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg));
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, m);
  };
  const arrow = (x, y1, y2) => el('path', { d: `M${x},${y1} L${x},${y2 - 2}`, fill: 'none', stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#am-arrow)' }, svg);

  // A box; the one in use is drawn with a thick outline and bold text, not by colour alone.
  // A dashed box is a case this page does not run.
  const box = (x, y, w, h, on, { dashed = false } = {}) => el('rect', {
    x, y, width: w, height: h, rx: 6,
    fill: CYAN, 'fill-opacity': on ? 0.14 : 0,
    stroke: on ? CYAN : LINE, 'stroke-width': on ? 2.5 : 1.25,
    'stroke-dasharray': dashed ? '5 4' : undefined,
  }, svg);

  // A box holding one or two lines of text.
  const textBox = (y, lines, on, opts = {}) => {
    const h = 14 + lines.length * 20;
    box(6, y, 428, h, on, opts);
    lines.forEach((t, k) => label(16, y + 24 + k * 20, t, { fill: on || k === 0 ? STRONG : MUTED, weight: on && k === 0 ? 700 : 400 }));
    return y + h;
  };

  // Step 1: four ways to fail, and where the page shows each.
  const WAYS = [
    { lines: ['a made-up answer', 'step 2, the instruction "guess"'] },
    { lines: ['going in circles', 'step 2, the instruction "try again"'] },
    { lines: ['a tool error', 'step 3, a written-in call'] },
    { lines: ['the wrong tool, or one that does not exist', 'named only: the stand-in never writes one'], dashed: true },
  ];
  function drawWays() {
    const out = outcomeLines(result, 'this run: ');
    frame(310 + out.length * 20);
    label(10, 20, 'four ways a run can go wrong', { fill: STRONG, weight: 700 });
    let y = 34;
    for (const w of WAYS) y = textBox(y, w.lines, false, { dashed: w.dashed }) + 10;
    out.forEach((t, k) => label(10, 320 + k * 20, t, { fill: STRONG }));
    return `Four ways a run can go wrong: ${WAYS.map((w) => w.lines.join(': ')).join('; ')}. The last is dashed: this page names it but does not run it. This run: ${outcomeText(result)}.`;
  }

  // Step 2: the three instructions side by side; the chosen one bold.
  function drawNotFound() {
    const outs = NOT_FOUND_OPTIONS.map((o) => outcomeLines(notFoundRun(o)));
    frame(34 + outs.reduce((n, l) => n + 24 + (l.length + 1) * 20, 0) + 34);
    label(10, 20, 'lookup("kai") gives "not found: kai"', { fill: STRONG, weight: 700 });
    const parts = [];
    let y = 34;
    NOT_FOUND_OPTIONS.forEach((o, k) => {
      y = textBox(y, [`"${o}"`, ...outs[k]], k === nfIdx) + 10;
      parts.push(`"${o}": ${outcomeText(notFoundRun(o))}${k === nfIdx ? ' (shown in the trace)' : ''}`);
    });
    label(10, y + 14, `kai is not one of the ${WORLD.people.length} people in the world`);
    return `When lookup("kai") gives "not found: kai". ${parts.join('. ')}. kai is not one of the ${WORLD.people.length} people in the invented world.`;
  }

  // Step 3: the written-in call, the calculator's result, the loop's stop.
  function drawToolError() {
    const c = toolErrorCase();
    const sum = defaultRun(TASK_SUM);
    const own = sum.messages.find((m) => m.role === 'call').text;
    const ownResult = sum.messages.find((m) => m.role === 'result').text;
    frame(290);
    label(10, 20, 'written-in example, not from the stand-in', { fill: STRONG, weight: 700 });
    const rows = [
      ['tool call', c.messages[1].text],
      ['result', c.messages[2].text],
      ['the loop stops', `stopped: ${c.stop}`],
    ];
    let y = 34;
    rows.forEach(([a, b], k) => {
      const end = textBox(y, [a, b], k === 2);
      if (k < 2) arrow(220, end, end + 18);
      y = end + 18;
    });
    label(10, 262, "the stand-in's own call for this task:");
    label(10, 282, `${own} gives ${ownResult}`);
    return `A written-in example, not from the stand-in: the tool call ${c.messages[1].text}, the result ${c.messages[2].text}, and the loop stops with ${c.stop}. The stand-in's own call for this task is ${own}, which gives ${ownResult}.`;
  }

  // Step 4: the tools each task's default run calls; the task in the trace bold.
  function drawNeeds() {
    const h = 34 + table.length * 44 + 60;
    frame(h);
    label(10, 20, 'tools each default run calls or asks for', { fill: STRONG, weight: 700 });
    let y = 34;
    table.forEach((row, k) => { y = textBox(y, [rowLine(row)], k === taskIdx) + 10; });
    const unused = unusedTools();
    label(10, y + 14, `called by no task: ${listText(unused)}`);
    label(10, y + 36, `send ${TOOLS.send.kind} outside the run`);
    return `The tools each task's default run calls or asks to use. ${table.map(rowLine).join('. ')}. Called by no task: ${listText(unused)}. Send ${TOOLS.send.kind} outside the run; calculator and lookup only read, and notepad only writes the run's own notes. Shown in the trace: ${table[taskIdx].id}.`;
  }

  // Step 5: the ask, the answer, and the sent list.
  function drawSend() {
    const r = result;
    const out = outcomeLines(r);
    const sl = sentLines(r);
    frame(210 + (out.length + sl.length) * 20);
    const askRow = sendRun(undefined, true).messages.find((m) => m.role === 'approval');
    label(10, 20, `send ${TOOLS.send.kind}`, { fill: STRONG, weight: 700 });
    textBox(34, ['ask first:', callText(askRow.tool, askRow.input)], ask && approve === undefined, { dashed: !ask });
    const states = [['answer: yes', true], ['answer: no', false]];
    states.forEach(([t, a], k) => {
      const x = 6 + k * 219;
      const on = ask && approve === a;
      box(x, 102, 209, 34, on, { dashed: !ask });
      label(x + 104, 124, t, { anchor: 'middle', fill: on ? STRONG : MUTED, weight: on ? 700 : 400 });
    });
    label(10, 162, ask ? 'ask before changes: on' : 'ask before changes: off, no ask', { fill: STRONG });
    let y = 164;
    for (const t of out) { y += 20; label(10, y, t); }
    y += 8;
    sl.forEach((t, k) => { y += 20; label(10, y, t, { fill: STRONG, weight: k === 0 ? 700 : 400 }); });
    label(10, y + 24, 'recorded on this page only; nothing leaves it', { size: 12 });
    const state = !ask ? 'Ask before changes is off, so there is no ask.' : approve === undefined ? 'No answer yet.' : approve ? 'Approved.' : 'Declined.';
    return `Send ${TOOLS.send.kind}. The ask: ${callText(askRow.tool, askRow.input)}. ${state} ${outcomeText(r)}. ${sl.join('; ')}. The message is only recorded on this page; nothing leaves it.`;
  }

  // Step 6: what each safeguard stopped; the one for the run in the trace bold.
  function drawSummary() {
    const g = safeguards();
    const h = 34 + g.reduce((s, x) => s + 24 + x.lines.length * 20, 0) + 40;
    frame(h);
    label(10, 20, 'what each safeguard did on this page', { fill: STRONG, weight: 700 });
    const id = PAGE_RUNS[runIdx].id;
    let y = 34;
    let match = null;
    for (const x of g) {
      const on = x.runIds.includes(id);
      if (on) match = x;
      y = textBox(y, x.lines, on) + 10;
    }
    const tail = match ? `${PAGE_RUNS[runIdx].label}: ${match.lines[0]}` : `${PAGE_RUNS[runIdx].label}: no safeguard stopped it`;
    label(10, h - 12, tail, { fill: STRONG });
    return `What each safeguard did on this page. ${g.map((x) => x.lines.join(', ')).join('. ')}. The run in the trace, ${tail}.`;
  }

  function draw() {
    const fns = [drawWays, drawNotFound, drawToolError, drawNeeds, drawSend, drawSummary];
    svg.setAttribute('aria-label', fns[step]());
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
      if (!(animate && !reduced)) tv.select(result.messages.length - 1);
      draw();
    },
  };
}
