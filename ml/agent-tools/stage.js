// Lesson L6.2, "Tools": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-agent-tools.test.mjs); the
// DOM is only touched inside createStage. Every trace comes from runAgent in agent.js, every
// true value from the invented world, the price list and calc, and every number in the
// page's text from pageValues. All text goes in with textContent.

import { WORLD } from '../assets/world.js';
import { PRICES, TASKS, TOOLS, runAgent, calc, fmtMoney, fmtNumber, roundOneFigure } from '../assets/agent.js';
import { createTraceView, rulesFor } from '../assets/trace-view.js';
import { el } from '../assets/plot.js';

// ---- tasks, tools and runs --------------------------------------------------------

export const TASK_SUM = 'sum';
export const TASK_SHOP = 'eli-shop';

// The tools this page switches on and off; send comes in a later lesson.
export const LESSON_TOOLS = ['calculator', 'lookup', 'notepad'];

export const configFor = (t) => ({ tools: { calculator: !!t.calculator, lookup: !!t.lookup, notepad: !!t.notepad } });
export const runFor = (t) => runAgent(TASK_SHOP, configFor(t));

// Every on/off combination of the three tools, calculator first.
export const COMBOS = [];
for (const calculator of [true, false]) for (const lookup of [true, false]) for (const notepad of [false, true]) {
  COMBOS.push({ calculator, lookup, notepad });
}

export const START_TOGGLES = { calculator: true, lookup: true, notepad: false };

// The run each lesson step shows (step 5 starts from START_TOGGLES).
export function stepRuns() {
  return [
    runAgent(TASK_SUM),
    runAgent(TASK_SHOP),
    runAgent(TASK_SUM, { tools: { calculator: false } }),
    runAgent(TASK_SHOP, { tools: { lookup: false } }),
    runFor(START_TOGGLES),
  ];
}

// ---- true values, guesses and the result line ----------------------------------------

const exprOf = (terms) => terms.map(([a, b]) => `${fmtNumber(a)}*${fmtNumber(b)}`).join('+');

function trueTerms(taskId) {
  const g = TASKS[taskId].goal;
  if (g.kind === 'sum') return g.terms;
  return g.people.map((p) => [g.qty, PRICES[WORLD.likes[p]]]);
}

export const trueValue = (taskId) => calc(exprOf(trueTerms(taskId))).value;

const fmtFor = (taskId) => (TASKS[taskId].goal.kind === 'sum' ? fmtNumber : fmtMoney);

// The number in an answer: "The sum is 4.6." or "That costs $3.20."
export function answerValue(answer) {
  const m = String(answer).match(/(?:\$|is )(\d+(?:\.\d+)?)\.?$/);
  return m ? Number(m[1]) : NaN;
}

export const errorOf = (taskId, answer) => trueValue(taskId) - answerValue(answer);

export function verdict(taskId, answer) {
  const f = fmtFor(taskId);
  const got = answerValue(answer);
  if (f(got) === f(trueValue(taskId))) return 'right';
  return `wrong by ${f(Math.abs(trueValue(taskId) - got))}`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function resultLine(result, taskId = TASK_SHOP) {
  const v = verdict(taskId, result.answer);
  return `${result.answer} ${v === 'right' ? '✓' : '✗'} ${v} · ${plural(result.steps, 'step')}`;
}

// The guess rule for sums: round each number to one significant figure, then work the sum out exactly.
export function guessSum(terms) {
  const rounded = terms.map(([a, b]) => [roundOneFigure(a), roundOneFigure(b)]);
  const expr = exprOf(rounded);
  return { rounded, expr, value: calc(expr).value };
}

// ---- who made a row, and the call up close ---------------------------------------------

const WHO = { plan: 'model', call: 'model', answer: 'model', result: 'tool', stop: 'loop' };
export const whoMade = (msg) => WHO[msg.role] || null;

export function whoText(msg) {
  if (msg.role === 'task') return 'the task: given to the model';
  if (msg.role === 'plan') return 'a plan line: text the model wrote';
  if (msg.role === 'call') return 'a call: text the model wrote';
  if (msg.role === 'answer') return 'the answer: text the model wrote';
  if (msg.role === 'result') return `a result: what ${msg.tool} returned`;
  if (msg.role === 'stop') return 'the loop stopped the run';
  return '';
}

// The tool call and its result in the step of row i, or null if that step calls no tool.
export function callOfStep(result, i) {
  const m = result.messages[i];
  if (!m || m.role === 'task' || m.role === 'stop') return null;
  const call = result.messages.find((x) => x.role === 'call' && x.step === m.step);
  if (!call) return null;
  const res = result.messages.find((x) => x.role === 'result' && x.step === m.step) || null;
  return { call, result: res };
}

export const inputText = (input) => Object.keys(input).map((k) => `${k} = "${input[k]}"`).join(', ');

const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
const money = (text) => text.match(/\$\d+\.\d\d/)[0];

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const [sum, eli, sumGuess, noLookup] = stepRuns();
  const g = guessSum(TASKS[TASK_SUM].goal.terms);
  const sumCall = sum.messages.find((m) => m.role === 'call');
  const eliCalls = eli.messages.filter((m) => m.role === 'call');
  const goal = TASKS[TASK_SHOP].goal;
  const food = WORLD.likes[goal.people[0]];
  const flat = TASKS[TASK_SUM].goal.terms.flat();
  const flatR = g.rounded.flat();
  const notepadRun = runFor({ calculator: true, lookup: true, notepad: true });
  const noteCall = notepadRun.messages.find((m) => m.role === 'call' && m.tool === 'notepad');
  const noCalc = runFor({ calculator: false, lookup: true, notepad: false });
  const noCalcGuess = guessSum([[goal.qty, PRICES[food]]]);
  return {
    'sum-task': TASKS[TASK_SUM].text,
    'sum-call': sumCall.text,
    'sum-result': sum.messages.find((m) => m.role === 'result').text,
    'sum-agent': sum.answer,
    'sum-steps': String(sum.steps),
    'guess-expr': g.expr,
    'guess-value': fmtNumber(g.value),
    'guess-answer': sumGuess.answer,
    'guess-steps': String(sumGuess.steps),
    'sum-error': fmtNumber(errorOf(TASK_SUM, sumGuess.answer)),
    'rounded-list': listText(flat.map((x, k) => (x === flatR[k] ? `${fmtNumber(x)} stays ${fmtNumber(x)}` : `${fmtNumber(x)} becomes ${fmtNumber(flatR[k])}`))),
    'guess-rules': listText(rulesFor(sumGuess.messages[1]).map((r) => r.id)),
    'about-calculator': TOOLS.calculator.about,
    'about-lookup': TOOLS.lookup.about,
    'about-notepad': TOOLS.notepad.about,
    'eli-call-1': eliCalls[0].text,
    'eli-input-1': inputText(eliCalls[0].input),
    'eli-result-1': eli.messages.find((m) => m.role === 'result').text,
    'eli-calls': String(eliCalls.length),
    'eli-steps': String(eli.steps),
    'eli-qty': String(goal.qty),
    'eli-food': food,
    'eli-price': fmtMoney(PRICES[food]),
    'true-eli': fmtMoney(trueValue(TASK_SHOP)),
    'guess-food': WORLD.foods[0],
    'guess-price': fmtMoney(PRICES[Object.keys(PRICES)[0]]),
    'nolookup-call': noLookup.messages.find((m) => m.role === 'call').text,
    'nolookup-answer': money(noLookup.answer),
    'nolookup-error': fmtMoney(errorOf(TASK_SHOP, noLookup.answer)),
    'nolookup-steps': String(noLookup.steps),
    'nolookup-rules': listText(rulesFor(noLookup.messages[1]).map((r) => r.id)),
    'combos': `${LESSON_TOOLS.map(() => '2').join(' × ')} = ${COMBOS.length}`,
    'notepad-steps': String(notepadRun.steps),
    'notepad-extra': String(notepadRun.steps - eli.steps),
    'notepad-rule': noteCall.rule,
    'nocalc-expr': noCalcGuess.expr,
    'nocalc-answer': money(noCalc.answer),
    'price-list': Object.keys(PRICES).map((f) => `${f} ${fmtMoney(PRICES[f])}`).join(', '),
    'likes-list': WORLD.people.map((p) => `${p} ${WORLD.likes[p]}`).join(', '),
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
  const toggles = { ...START_TOGGLES };
  const results = stepRuns();

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  const host = doc.createElement('div');
  host.className = 'agent-trace';
  host.style.marginTop = '12px';
  svg.after(host);
  const tv = createTraceView(host, { tools: controls, label: 'Trace: every message in and out', reduced, announce, onSelect: () => draw() });
  // onSelect fires for every selection, the view's own included; draw() never selects, so there is no loop.

  // Step 5's checkboxes, one per tool, in the tools slot.
  const group = doc.createElement('div');
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', 'tools the agent may use');
  Object.assign(group.style, { display: 'none', flexWrap: 'wrap', alignItems: 'center', gap: '6px 14px' });
  const boxes = {};
  for (const name of LESSON_TOOLS) {
    const lab = doc.createElement('label');
    const input = doc.createElement('input');
    input.type = 'checkbox';
    input.checked = toggles[name];
    input.style.accentColor = CYAN;
    input.addEventListener('change', () => {
      toggles[name] = input.checked;
      tv.stop();
      results[4] = runFor(toggles);
      tv.show(results[4], { animate: false });
      draw();
      announce(`${name} ${input.checked ? 'on' : 'off'}. ${resultLine(results[4])}`);
    });
    lab.append(input, ' ', name);
    boxes[name] = input;
    group.append(lab);
  }
  controls.append(group);

  // Sizes are in viewBox units. At phone width ml.css sets class stage-text to 15 units,
  // so every line here is laid out to fit at 15 units too (about 9 units per character).
  const label = (x, y, text, { anchor = 'start', fill = MUTED, size = 13, weight = 400 } = {}) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, svg);

  const marker = () => {
    const m = el('marker', { id: 'at-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg));
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, m);
  };
  const arrow = (d) => el('path', { d, fill: 'none', stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#at-arrow)' }, svg);

  // A box; the one in use is drawn with a thick outline and bold text, not by colour alone.
  // An "off" box is dashed.
  const box = (x, y, w, h, text, on, { dashed = false, size = 13 } = {}) => {
    el('rect', {
      x, y, width: w, height: h, rx: 6,
      fill: CYAN, 'fill-opacity': on ? 0.14 : 0,
      stroke: on ? CYAN : LINE, 'stroke-width': on ? 2.5 : 1.25,
      'stroke-dasharray': dashed ? '5 4' : undefined,
    }, svg);
    if (text) label(x + w / 2, y + h / 2 + size * 0.36, text, { anchor: 'middle', fill: on ? STRONG : MUTED, size, weight: on ? 700 : 400 });
  };

  const frame = (h) => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
    marker();
  };

  const selected = () => {
    const r = results[step];
    const i = tv.selected;
    return i >= 0 && i < r.messages.length ? i : -1;
  };

  // Step 1: who made the selected row: the model (text), the tool (a program) or the loop.
  function drawWho(i) {
    frame(152);
    const r = results[step];
    const msg = i >= 0 ? r.messages[i] : null;
    const who = msg ? whoMade(msg) : null;
    const tool = r.messages.find((m) => m.role === 'call');
    label(65, 16, 'writes text', { anchor: 'middle', size: 12 });
    label(220, 16, 'runs the call', { anchor: 'middle', size: 12 });
    label(375, 16, 'a program', { anchor: 'middle', size: 12 });
    box(10, 26, 110, 36, 'model', who === 'model');
    box(165, 26, 110, 36, 'loop', who === 'loop');
    box(320, 26, 110, 36, tool ? tool.tool : 'tool', who === 'tool');
    arrow('M120,44 L163,44');
    arrow('M275,44 L318,44');
    arrow('M375,62 L375,86 L65,86 L65,64');
    label(220, 106, 'its output goes back as a message', { anchor: 'middle', size: 12 });
    const said = msg ? whoText(msg) : 'no row selected';
    label(10, 142, said, { fill: STRONG });
    return `The model writes text, the loop runs the call, the tool is an ordinary program, and its output goes back as a message. Selected row: ${said}.`;
  }

  // Step 2: the tool call of the selected row's step, up close.
  function drawCall(i) {
    frame(150);
    const r = results[step];
    const c = i >= 0 ? callOfStep(r, i) : null;
    if (!c) {
      label(10, 30, 'this row\'s step calls no tool', { fill: STRONG });
      label(10, 54, 'select a plan, call or result row');
      return 'The selected row\'s step calls no tool. Select a plan, call or result row to see its call up close.';
    }
    label(10, 20, 'the call, as the model wrote it', { size: 12 });
    label(10, 42, c.call.text, { fill: STRONG, weight: 700 });
    label(10, 64, `tool: ${c.call.tool}`, { fill: STRONG });
    label(10, 84, `input: ${inputText(c.call.input)}`, { fill: STRONG });
    label(10, 112, `the result, as ${c.call.tool} returned it`, { size: 12 });
    label(10, 134, c.result ? c.result.text : '', { fill: STRONG, weight: 700 });
    return `Step ${c.call.step}. The call ${c.call.text}: tool ${c.call.tool}, input ${inputText(c.call.input)}. The result: ${c.result ? c.result.text : 'none'}.`;
  }

  // Steps 3 and 4: the guess with a tool off, above the run with it on.
  function drawCompare(rows) {
    frame(14 + rows.length * 82);
    rows.forEach((row, k) => {
      const y = 6 + k * 82;
      el('rect', {
        x: 10, y, width: 420, height: 72, rx: 6,
        fill: row.on ? CYAN : 'none', 'fill-opacity': row.on ? 0.08 : 0,
        stroke: row.on ? CYAN : LINE, 'stroke-width': row.on ? 2 : 1.25, 'stroke-dasharray': row.on ? undefined : '5 4',
      }, svg);
      label(22, y + 22, row.title, { fill: STRONG, weight: 700 });
      label(22, y + 42, row.line1, { fill: STRONG });
      label(22, y + 62, row.line2, { fill: STRONG });
    });
    return rows.map((r) => `${r.title}: ${r.line1}; ${r.line2}.`).join(' ');
  }

  function drawSum() {
    const v = filled;
    const on = results[0];
    const off = results[2];
    return drawCompare([
      { title: 'calculator off: a guess', on: false, line1: `${v['guess-expr']} = ${v['guess-value']}`, line2: `✗ wrong by ${v['sum-error']}` },
      { title: 'calculator on', on: true, line1: `${on.messages.find((m) => m.role === 'call').text} = ${v['sum-result']}`, line2: `✓ ${verdict(TASK_SUM, on.answer)}` },
    ]) + ` ${off.answer}`;
  }

  function drawShop() {
    const v = filled;
    const on = results[1];
    const off = results[3];
    const onCalc = on.messages.find((m) => m.role === 'call' && m.tool === 'calculator');
    return drawCompare([
      { title: 'lookup off: guesses', on: false, line1: `${v['guess-food']} at ${v['guess-price']}; ${v['nolookup-call']}`, line2: `${money(off.answer)} ✗ ${verdict(TASK_SHOP, off.answer)}` },
      { title: 'lookup on', on: true, line1: `${v['eli-food']} at ${v['eli-price']}; ${onCalc.text}`, line2: `${money(on.answer)} ✓ ${verdict(TASK_SHOP, on.answer)}` },
    ]);
  }

  // Step 5: each tool on (solid) or off (dashed, and the word "off"), and the result line.
  function drawToggles() {
    frame(130);
    LESSON_TOOLS.forEach((name, k) => {
      const x = 10 + k * 142;
      box(x, 8, 136, 36, name, toggles[name], { dashed: !toggles[name] });
      label(x + 68, 62, toggles[name] ? 'on' : 'off', { anchor: 'middle', fill: toggles[name] ? STRONG : MUTED, size: 12 });
    });
    const r = results[4];
    const v = verdict(TASK_SHOP, r.answer);
    label(10, 94, `answer: ${money(r.answer)} ${v === 'right' ? '✓' : '✗'} ${v}`, { fill: STRONG, weight: 700 });
    label(10, 120, `${plural(r.steps, 'step')}, ${plural(r.messages.filter((m) => m.role === 'call').length, 'tool call')}`);
    const on = LESSON_TOOLS.filter((n) => toggles[n]);
    return `Tools on: ${on.length ? listText(on) : 'none'}. ${resultLine(r)}.`;
  }

  function draw() {
    const i = selected();
    let aria;
    if (step === 0) aria = drawWho(i);
    else if (step === 1) aria = drawCall(i);
    else if (step === 2) aria = drawSum();
    else if (step === 3) aria = drawShop();
    else aria = drawToggles();
    svg.setAttribute('aria-label', aria);
  }

  return {
    count: 5,
    show(i, animate = false) {
      tv.stop();
      step = i;
      group.style.display = i === 4 ? 'flex' : 'none';
      for (const name of LESSON_TOOLS) boxes[name].checked = toggles[name];
      tv.show(results[i], { animate });
      // Opened without a run: steps 1 and 2 select the first call, steps 3 and 4 the first plan line.
      if (i < 4 && !(animate && !reduced)) {
        tv.select(results[i].messages.findIndex((m) => m.role === (i < 2 ? 'call' : 'plan')));
      }
      draw();
    },
  };
}
